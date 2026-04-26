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
    """`http://host:port/v1` -> `http://host:port/metrics`"""
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
        typ = None
        if hasattr(resp, "getheader"):
            typ = resp.getheader("Content-Type")
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return PrometheusScrape(
        url=metrics_url,
        fetched_at_utc=ts,
        status_code=status,
        text=text,
        text_content_type=typ,
    )


# Sample lines: "name{labels} value" or "name value"
_RE_SAMPLE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|NaN|Inf)$"
)


def parse_prometheus_samples(text: str) -> dict[str, float | int]:
    """Loose text-format parse: map full metric+label string -> last seen numeric value (float)."""
    out: dict[str, float | int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _RE_SAMPLE.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if v in ("NaN", "Inf", "-Inf"):
            continue
        f = float(v)
        if f == int(f) and abs(f) < 1 << 53:
            out[k] = int(f)
        else:
            out[k] = f
    return out


def scrape_to_json_dict(scrape: PrometheusScrape, include_samples: bool) -> dict[str, Any]:
    d: dict[str, Any] = {
        "scrape": {
            "url": scrape.url,
            "fetched_at_utc": scrape.fetched_at_utc,
            "http_status": scrape.status_code,
            "content_type": scrape.text_content_type,
        },
    }
    if include_samples:
        d["samples"] = parse_prometheus_samples(scrape.text)
    d["raw_prometheus"] = scrape.text
    return d
