"""Helpers for vLLM OpenAPI server Prometheus /metrics (same port as the API, not /v1)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class PrometheusScrape:
    url: str
    fetched_at_utc: str
    status_code: int
    text: str
    text_content_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scrape": {
                "url": self.url,
                "fetched_at_utc": self.fetched_at_utc,
                "http_status": self.status_code,
                "content_type": self.text_content_type,
                "body_bytes": len(self.text.encode("utf-8")),
            },
            "raw_prometheus": self.text,
        }


def openai_v1_base_to_metrics_url(openai_v1_base: str) -> str:
    """`http://host:port/v1` -> `http://host:port/metrics`."""

    parts = urlsplit(openai_v1_base.rstrip("/"))
    path = parts.path
    if path.endswith("/v1"):
        path = path[: -len("/v1")] or "/"
    elif path == "/v1":
        path = "/"
    metrics_path = (path.rstrip("/") + "/metrics") if path not in ("", "/") else "/metrics"
    if not metrics_path.startswith("/"):
        metrics_path = "/" + metrics_path
    return urlunsplit((parts.scheme, parts.netloc, metrics_path, "", ""))


def fetch_prometheus_text(metrics_url: str, timeout_s: float = 30.0) -> PrometheusScrape:
    req = Request(metrics_url, method="GET", headers={"User-Agent": "vllm_bf16-harness/1.0"})
    with urlopen(req, timeout=timeout_s) as resp:
        status = getattr(resp, "status", 200) or 200
        data = resp.read()
        text = data.decode("utf-8", errors="replace")
        content_type = None
        if hasattr(resp, "getheader"):
            content_type = resp.getheader("Content-Type")
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return PrometheusScrape(
        url=metrics_url,
        fetched_at_utc=fetched_at,
        status_code=status,
        text=text,
        text_content_type=content_type,
    )


_RE_SAMPLE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|NaN|Inf|-Inf)$"
)


def parse_prometheus_samples(text: str) -> dict[str, float | int]:
    """Loose text-format parse: map full metric+label string -> last seen numeric value."""

    out: dict[str, float | int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _RE_SAMPLE.match(line)
        if not match:
            continue
        key, raw_value = match.group(1), match.group(2)
        if raw_value in ("NaN", "Inf", "-Inf"):
            continue
        value = float(raw_value)
        if value == int(value) and abs(value) < 1 << 53:
            out[key] = int(value)
        else:
            out[key] = value
    return out


def scrape_to_json_dict(scrape: PrometheusScrape, include_samples: bool) -> dict[str, Any]:
    data: dict[str, Any] = {
        "scrape": {
            "url": scrape.url,
            "fetched_at_utc": scrape.fetched_at_utc,
            "http_status": scrape.status_code,
            "content_type": scrape.text_content_type,
        },
    }
    if include_samples:
        data["samples"] = parse_prometheus_samples(scrape.text)
    data["raw_prometheus"] = scrape.text
    return data


def _matching_values(samples: dict[str, float | int], metric_name: str) -> list[float]:
    out: list[float] = []
    prefix = metric_name + "{"
    for key, value in samples.items():
        if key == metric_name or key.startswith(prefix):
            out.append(float(value))
    return out


def extract_kv_cache_usage_perc_max(samples: dict[str, float | int]) -> float | None:
    kv_usage = _matching_values(samples, "vllm:kv_cache_usage_perc")
    if not kv_usage:
        kv_usage = _matching_values(samples, "vllm:gpu_cache_usage_perc")
    if not kv_usage:
        return None
    return max(kv_usage)


def diff_prometheus_samples(
    after: dict[str, float | int], before: dict[str, float | int]
) -> dict[str, float | int]:
    """Return a per-key numeric delta for monotonically increasing scrape metrics."""

    out: dict[str, float | int] = {}
    for key, after_value in after.items():
        before_value = before.get(key, 0)
        delta = float(after_value) - float(before_value)
        if abs(delta) < 1e-12:
            delta = 0.0
        if delta == int(delta) and abs(delta) < 1 << 53:
            out[key] = int(delta)
        else:
            out[key] = delta
    return out


def summarize_vllm_samples(samples: dict[str, float | int]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    def add_hist_mean(out_key: str, metric_base: str) -> None:
        sum_values = _matching_values(samples, metric_base + "_sum")
        count_values = _matching_values(samples, metric_base + "_count")
        total_sum = sum(sum_values)
        total_count = sum(count_values)
        if total_count > 0:
            summary[out_key] = total_sum / total_count
            summary[out_key + "_count"] = int(total_count)

    add_hist_mean("ttft_mean_s", "vllm:time_to_first_token_seconds")
    add_hist_mean("e2e_latency_mean_s", "vllm:e2e_request_latency_seconds")
    add_hist_mean("queue_time_mean_s", "vllm:request_queue_time_seconds")
    add_hist_mean("inference_time_mean_s", "vllm:request_inference_time_seconds")
    add_hist_mean("prefill_time_mean_s", "vllm:request_prefill_time_seconds")
    add_hist_mean("decode_time_mean_s", "vllm:request_decode_time_seconds")
    add_hist_mean("inter_token_latency_mean_s", "vllm:inter_token_latency_seconds")

    kv_usage = extract_kv_cache_usage_perc_max(samples)
    if kv_usage is not None:
        summary["kv_cache_usage_perc_max"] = kv_usage

    prefix_hits = sum(_matching_values(samples, "vllm:prefix_cache_hits_total"))
    prefix_queries = sum(_matching_values(samples, "vllm:prefix_cache_queries_total"))
    if prefix_hits or prefix_queries:
        summary["prefix_cache_hits"] = int(prefix_hits)
        summary["prefix_cache_queries"] = int(prefix_queries)
        if prefix_queries > 0:
            summary["prefix_cache_hit_rate"] = prefix_hits / prefix_queries

    return summary
