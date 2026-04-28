"""Run a single-article or JSONL benchmark; benchmark.py is only imported, never modified."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import asyncio
from openai import AsyncOpenAI

from common.config import LABEL_MAP, MAX_TOKENS, PROMPT_PATH, TEMPERATURE, TOP_P
from common.data_utils import get_text, load_jsonl
from common.nvidia_smi_sampler import nvidia_smi_log_csv
from common.nvidia_smi_summary import summarize_nvidia_smi_csv
from common.parser import parse_label
from common.prometheus_utils import (
    fetch_prometheus_text,
    openai_v1_base_to_metrics_url,
    parse_prometheus_samples,
    summarize_vllm_samples,
)
from services.frontend.configs import FrontendConfig
from services.frontend.prom_delta import prom_delta_from_text

PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")
_bench_mods: dict[str, Any] = {}


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    k = int(round((p / 100.0) * (len(ys) - 1)))
    return ys[k]


def _latency_summary(metric_prefix: str, values: list[float]) -> dict[str, float | None]:
    return {
        f"{metric_prefix}_avg_s": mean(values) if values else None,
        f"{metric_prefix}_p50_s": _percentile(values, 50),
        f"{metric_prefix}_p95_s": _percentile(values, 95),
        f"{metric_prefix}_p99_s": _percentile(values, 99),
    }


def get_benchmark(kind: str) -> Any:
    if kind in _bench_mods:
        return _bench_mods[kind]
    root = Path(__file__).resolve().parent.parent.parent
    module_map = {
        "hf": root / "scripts" / "benchmark_hf.py",
        "vllm": root / "scripts" / "benchmark.py",
    }
    p = module_map[kind]
    spec = importlib.util.spec_from_file_location(f"vllm_bf16_bench_harness_{kind}", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load benchmark module for kind={kind}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    _bench_mods[kind] = m
    return m


def _label_name_from_row(row: dict[str, Any]) -> str | None:
    label_name = row.get("label_name")
    if isinstance(label_name, str) and label_name.strip():
        return label_name.strip()
    for key in ("label", "labels"):
        if key not in row:
            continue
        try:
            label_id = int(row[key])
        except (TypeError, ValueError):
            return None
        return LABEL_MAP.get(label_id)
    return None


def _normalize_benchmark_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    normalized: list[dict[str, Any]] = []
    has_labels = False
    for idx, row in enumerate(rows):
        text = get_text(row)
        label_name = _label_name_from_row(row)
        if label_name is not None:
            has_labels = True
        normalized.append(
            {
                "id": row.get("id", idx),
                "text": text,
                "label_name": label_name,
            }
        )
    return normalized, has_labels


def _scrape_prometheus(murl: str) -> str:
    return fetch_prometheus_text(murl, timeout_s=30.0).text


async def _scrape_prometheus_async(murl: str) -> str:
    return await asyncio.get_event_loop().run_in_executor(
        None, _scrape_prometheus, murl
    )


async def run_jsonl_bench(
    config: FrontendConfig,
    openai_v1_base: str,
    jsonl_path: Path,
    config_dir: Path,
    concurrency: int = 4,
    warmup: int = 10,
    limit: int | None = None,
) -> dict[str, Any]:
    benchmark_kind = "hf" if config.name.startswith("hf_") else "vllm"
    m = get_benchmark(benchmark_kind)
    out_json = config_dir / "bench.json"
    nvidia = config_dir / "nvidia_smi" / "smi.csv"
    (config_dir / "nvidia_smi").mkdir(parents=True, exist_ok=True)
    proms_raw, proms_j = (None, None)
    if config.has_prometheus:
        (config_dir / "proms").mkdir(parents=True, exist_ok=True)
        proms_raw = str((config_dir / "proms" / "prom.txt").resolve())
        proms_j = str((config_dir / "proms" / "prom.json").resolve())

    args = argparse.Namespace(
        config_name=config.name,
        model_id=config.openai_model_id,
        input=str(jsonl_path.resolve()),
        base_url=openai_v1_base,
        warmup=warmup,
        nvidia_smi_interval=1.0,
        prometheus_samples=bool(config.has_prometheus),
        prometheus_poll_interval=0.5,
    )
    source_rows = load_jsonl(jsonl_path, limit=limit)
    rows, has_labels = _normalize_benchmark_rows(source_rows)
    if not rows:
        return {
            "error": "empty JSONL (after limit)",
            "row_count": 0,
        }
    client = AsyncOpenAI(base_url=openai_v1_base, api_key="dummy")
    nvidia_smi_csv = str(nvidia.resolve())
    if benchmark_kind == "hf":
        result = await m.run_one_concurrency(
            client,
            args,
            rows,
            concurrency,
            out_json,
            nvidia_smi_csv,
            is_sweep=False,
        )
    else:
        result = await m.run_one_concurrency(
            client,
            args,
            rows,
            concurrency,
            out_json,
            proms_raw,
            proms_j,
            nvidia_smi_csv,
            is_sweep=False,
        )
    result["has_gold_labels"] = has_labels
    if not has_labels:
        result["accuracy_valid_only"] = None
        result["accuracy_overall_invalid_as_wrong"] = None
        result["label_mode"] = "prediction_only_unlabeled_jsonl"
    return result


async def _stream_one_text_request(
    client: AsyncOpenAI,
    config: FrontendConfig,
    sem: asyncio.Semaphore,
    article: str,
    input_name: str,
) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(article=article)
    queued_t0 = time.perf_counter()
    t_ttft: float | None = None
    raw = ""
    n_out = 0
    last_usage: object | None = None
    async with sem:
        dispatch_t0 = time.perf_counter()
        cparams: dict[str, Any] = {
            "model": config.openai_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "stream": True,
        }
        try:
            cparams["stream_options"] = {"include_usage": True}
            stream = await client.chat.completions.create(**cparams)  # type: ignore[call-overload, arg-type]
        except (TypeError, ValueError):
            cparams.pop("stream_options", None)
            stream = await client.chat.completions.create(**cparams)  # type: ignore[call-overload, arg-type]
        has_first = False
        async for chunk in stream:
            u = getattr(chunk, "usage", None)
            if u is not None:
                last_usage = u
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            c = d.content if d else None
            if c and not has_first:
                t_ttft = time.perf_counter() - dispatch_t0
                has_first = True
            if c:
                raw += c
        service_latency_s = time.perf_counter() - dispatch_t0
        if last_usage is not None and getattr(last_usage, "completion_tokens", None) is not None:
            n_out = int(getattr(last_usage, "completion_tokens", 0) or 0)
    latency_s = time.perf_counter() - queued_t0
    queue_wait_s = max(0.0, latency_s - service_latency_s)
    if not n_out and raw:
        n_out = max(1, len(raw) // 4)
    result = {
        "input_name": input_name,
        "pred_label": parse_label(raw),
        "raw_output": raw,
        "latency_s": latency_s,
        "service_latency_s": service_latency_s,
        "queue_wait_s": queue_wait_s,
        "ttft_s": t_ttft,
        "n_output_tokens": n_out,
    }
    if n_out and latency_s > 0:
        result["tps"] = n_out / latency_s
    return result


async def run_text_batch(
    config: FrontendConfig,
    openai_v1_base: str,
    text_inputs: list[dict[str, str]],
    config_dir: Path,
    concurrency: int = 4,
) -> dict[str, Any]:
    if not text_inputs:
        return {"error": "no text inputs provided", "n_inputs": 0}
    client = AsyncOpenAI(base_url=openai_v1_base, api_key="dummy", timeout=300.0)
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    (config_dir / "nvidia_smi").mkdir(parents=True, exist_ok=True)
    nvidia_path = str((config_dir / "nvidia_smi" / "text_batch.csv").resolve())
    t0 = time.perf_counter()
    with nvidia_smi_log_csv(nvidia_path, interval_s=1.0) as p_csv:
        items = await asyncio.gather(
            *[
                _stream_one_text_request(client, config, sem, item["text"], item["name"])
                for item in text_inputs
            ]
        )
    total_time = time.perf_counter() - t0
    try:
        smi_summary = summarize_nvidia_smi_csv(p_csv)
    except (OSError, FileNotFoundError) as exc:  # noqa: BLE001
        smi_summary = {"error": str(exc)}
    latencies = [float(item["latency_s"]) for item in items]
    service_latencies = [float(item["service_latency_s"]) for item in items]
    queue_waits = [float(item["queue_wait_s"]) for item in items]
    ttfts = [float(item["ttft_s"]) for item in items if item.get("ttft_s") is not None]
    tps_values = [float(item["tps"]) for item in items if item.get("tps") is not None]
    summary: dict[str, Any] = {
        "mode": "text_batch",
        "config_name": config.name,
        "model_id": config.openai_model_id,
        "n_inputs": len(items),
        "concurrency": max(1, int(concurrency)),
        "throughput_req_per_s": (len(items) / total_time) if total_time > 0 else 0.0,
        "latency_semantics": "end_to_end_client_observed_including_local_semaphore_wait",
        "measured_phase_wall_time_s": total_time,
        "ttft_avg_s": mean(ttfts) if ttfts else None,
        "ttft_p50_s": _percentile(ttfts, 50),
        "ttft_p95_s": _percentile(ttfts, 95),
        "tps_avg": mean(tps_values) if tps_values else None,
        "items": items,
        "nvidia_smi": {"csv": nvidia_path, "summary": smi_summary},
    }
    summary.update(_latency_summary("latency", latencies))
    summary.update(_latency_summary("service_latency", service_latencies))
    summary.update(_latency_summary("queue_wait", queue_waits))
    if len(items) == 1:
        summary.update(items[0])
    out_p = config_dir / "text_batch_result.json"
    out_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


async def run_single_text_bench(
    config: FrontendConfig,
    openai_v1_base: str,
    article: str,
    config_dir: Path,
) -> dict[str, Any]:
    return await run_text_batch(
        config,
        openai_v1_base,
        [{"name": "input.txt", "text": article}],
        config_dir,
        concurrency=1,
    )
