"""Run a single-article or JSONL benchmark; benchmark.py is only imported, never modified."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import asyncio
from openai import AsyncOpenAI

from common.config import MAX_TOKENS, PROMPT_PATH, TEMPERATURE, TOP_P
from common.data_utils import load_jsonl
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

BENCH_MODULE_NAME = "vllm_bf16_bench_harness"
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")
_bench_mod: Any = None


def get_benchmark() -> Any:
    global _bench_mod
    if _bench_mod is not None:
        return _bench_mod
    root = Path(__file__).resolve().parent.parent.parent
    p = root / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location(BENCH_MODULE_NAME, p)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load scripts/benchmark.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    _bench_mod = m
    return m


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
    m = get_benchmark()
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
    )
    rows = load_jsonl(jsonl_path, limit=limit)
    if not rows:
        return {
            "error": "empty JSONL (after limit)",
            "row_count": 0,
        }
    client = AsyncOpenAI(base_url=openai_v1_base, api_key="dummy")
    nvidia_smi_csv = str(nvidia.resolve())
    return await m.run_one_concurrency(
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


async def run_single_text_bench(
    config: FrontendConfig,
    openai_v1_base: str,
    article: str,
    config_dir: Path,
) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(article=article)
    client = AsyncOpenAI(base_url=openai_v1_base, api_key="dummy", timeout=300.0)
    murl = openai_v1_base_to_metrics_url(openai_v1_base)
    (config_dir / "nvidia_smi").mkdir(parents=True, exist_ok=True)
    nvidia_path = str((config_dir / "nvidia_smi" / "single.csv").resolve())
    result: dict[str, Any] = {
        "mode": "single_text",
        "model_id": config.openai_model_id,
        "config_name": config.name,
    }
    before_txt = ""
    if config.has_prometheus:
        before_txt = await _scrape_prometheus_async(murl)

    t_start = time.perf_counter()
    t_ttft: float | None = None
    raw: str = ""
    n_out = 0
    t_total = 0.0
    last_usage: object | None = None

    with nvidia_smi_log_csv(nvidia_path, interval_s=1.0) as p_csv:
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
                t_ttft = time.perf_counter() - t_start
                has_first = True
            if c:
                raw += c
        t_total = time.perf_counter() - t_start
        if last_usage is not None and getattr(last_usage, "completion_tokens", None) is not None:
            n_out = int(getattr(last_usage, "completion_tokens", 0) or 0)
    smi_summary = {}
    try:
        smi_summary = summarize_nvidia_smi_csv(p_csv)
    except (OSError, FileNotFoundError) as e:  # noqa: BLE001
        smi_summary = {"error": str(e)}
    if config.has_prometheus and before_txt:
        after_txt = await _scrape_prometheus_async(murl)
        pm = prom_delta_from_text(before_txt, after_txt)
        result["prometheus"] = {**pm, "metrics_url": murl}
        samples = parse_prometheus_samples(after_txt)
        if samples:
            result["prometheus"]["cumulative"] = summarize_vllm_samples(samples)
    if t_ttft is not None:
        result["ttft_s"] = t_ttft
    result["latency_s"] = t_total
    if t_total > 0 and n_out:
        result["tps"] = n_out / t_total
    if not n_out and raw:
        n_out = max(1, len(raw) // 4)
        result["n_output_tokens_estimated"] = True
    result["n_output_tokens"] = n_out
    result["raw_output"] = raw
    result["pred_label"] = parse_label(raw)
    result["nvidia_smi"] = {"csv": nvidia_path, "summary": smi_summary}
    out_p = config_dir / "single_result.json"
    out_p.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
