"""Benchmark script for HF baseline: captures TTFT from server response and logs nvidia-smi."""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from statistics import mean

from openai import AsyncOpenAI

from common.config import MAX_TOKENS, PROMPT_PATH, TEMPERATURE, TOP_P
from common.data_utils import load_jsonl
from common.parser import parse_label

PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = int(round((p / 100) * (len(xs) - 1)))
    return xs[k]


def _per_concurrency_path(base: Path, concurrency: int) -> Path:
    return base.parent / f"{base.stem}_c{concurrency}{base.suffix}"


def _strip_existing_c_suffix(path_str: str) -> str:
    path = Path(path_str)
    stem = path.stem
    if re.search(r"_c\d+$", stem):
        stem = re.sub(r"_c\d+$", "", stem)
    return str(path.parent / f"{stem}{path.suffix}")


def _parse_concurrencies(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _latency_summary(metric_prefix: str, values: list[float]) -> dict[str, float | None]:
    return {
        f"{metric_prefix}_avg_s": mean(values) if values else None,
        f"{metric_prefix}_p50_s": percentile(values, 50),
        f"{metric_prefix}_p95_s": percentile(values, 95),
        f"{metric_prefix}_p99_s": percentile(values, 99),
    }


def _warmup_request_count(total_rows: int, requested_warmup: int, concurrency: int) -> int:
    return min(total_rows, max(requested_warmup, concurrency))


async def run_one_concurrency(
    client: AsyncOpenAI,
    args,
    rows: list,
    concurrency: int,
    output_path: Path,
    nvidia_smi_csv: str | None,
    is_sweep: bool,
) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def one_request(row):
        prompt = PROMPT_TEMPLATE.format(article=row["text"])
        queued_t0 = time.perf_counter()
        async with sem:
            dispatch_t0 = time.perf_counter()
            resp = await client.chat.completions.create(
                model=args.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )
            service_latency_s = time.perf_counter() - dispatch_t0
        end_to_end_latency_s = time.perf_counter() - queued_t0
        queue_wait_s = max(0.0, end_to_end_latency_s - service_latency_s)
        raw_output = resp.choices[0].message.content or ""
        pred_label = parse_label(raw_output)
        ttft_s = (resp.model_extra or {}).get("ttft_s") if hasattr(resp, "model_extra") else None
        return {
            "id": row.get("id"),
            "gold_label": row.get("label_name"),
            "pred_label": pred_label,
            "raw_output": raw_output,
            "latency_s": end_to_end_latency_s,
            "service_latency_s": service_latency_s,
            "queue_wait_s": queue_wait_s,
            "ttft_s": ttft_s,
        }

    async def run_requests_phase():
        warmup_count = _warmup_request_count(len(rows), args.warmup, concurrency)
        warmup_rows = rows[:warmup_count]
        if warmup_rows:
            await asyncio.gather(*[one_request(row) for row in warmup_rows])
        measured_rows = rows[warmup_count:]
        phase_t0 = time.perf_counter()
        results = await asyncio.gather(*[one_request(row) for row in measured_rows])
        return warmup_rows, results, time.perf_counter() - phase_t0

    nvidia_meta: dict = {}
    if nvidia_smi_csv:
        from common.nvidia_smi_sampler import nvidia_smi_log_csv
        from common.nvidia_smi_summary import summarize_nvidia_smi_csv

        try:
            with nvidia_smi_log_csv(nvidia_smi_csv, interval_s=args.nvidia_smi_interval) as process_path:
                warmup_rows, results, total_time = await run_requests_phase()
            nvidia_summary = summarize_nvidia_smi_csv(process_path)
            nvidia_meta = {
                "csv": str(process_path),
                "interval_s": args.nvidia_smi_interval,
                "summary": nvidia_summary,
            }
        except OSError as exc:
            warmup_rows, results, total_time = await run_requests_phase()
            nvidia_meta = {"error": f"{type(exc).__name__}: {exc}", "csv": nvidia_smi_csv}
    else:
        warmup_rows, results, total_time = await run_requests_phase()

    latencies = [row["latency_s"] for row in results]
    service_latencies = [row["service_latency_s"] for row in results]
    queue_waits = [row["queue_wait_s"] for row in results]
    throughput = len(results) / total_time if total_time > 0 else 0.0

    valid = [row for row in results if row["pred_label"] is not None]
    n_valid = len(valid)
    n_invalid = len(results) - n_valid
    n_correct = sum(1 for row in valid if row["pred_label"] == row["gold_label"])

    accuracy_valid_only = (n_correct / n_valid) if n_valid > 0 else 0.0
    accuracy_overall = (n_correct / len(results)) if results else 0.0
    ttfts = [row["ttft_s"] for row in results if row.get("ttft_s") is not None]

    summary = {
        "config_name": args.config_name,
        "model_id": args.model_id,
        "input_file": args.input,
        "n_requests_measured": len(results),
        "warmup_requests": len(warmup_rows),
        "concurrency": concurrency,
        "throughput_req_per_s": throughput,
        "latency_semantics": "end_to_end_client_observed_including_local_semaphore_wait",
        "warmup_mode": "concurrent_at_target_concurrency",
        "measured_phase_wall_time_s": total_time,
        "ttft_avg_s": mean(ttfts) if ttfts else None,
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "n_valid_predictions": n_valid,
        "n_invalid_predictions": n_invalid,
        "accuracy_valid_only": accuracy_valid_only,
        "accuracy_overall_invalid_as_wrong": accuracy_overall,
    }
    summary.update(_latency_summary("latency", latencies))
    summary.update(_latency_summary("service_latency", service_latencies))
    summary.update(_latency_summary("queue_wait", queue_waits))
    if nvidia_meta:
        summary["nvidia_smi"] = nvidia_meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.pred_output is not None:
        if is_sweep:
            pred_base = Path(_strip_existing_c_suffix(args.pred_output))
            pred_path = _per_concurrency_path(pred_base, concurrency)
        else:
            pred_path = Path(args.pred_output)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--config-name", default="hf_baseline_bf16")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--concurrency-list", type=str, default="1,2,4,8,16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pred-output", default=None)
    parser.add_argument("--nvidia-smi-csv", default=None)
    parser.add_argument("--nvidia-smi-interval", type=float, default=1.0)
    args = parser.parse_args()

    concurrencies = [args.concurrency] if args.concurrency is not None else _parse_concurrencies(args.concurrency_list)
    if not concurrencies:
        raise SystemExit("empty concurrency list")

    client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")
    rows = load_jsonl(Path(args.input), limit=args.limit)
    out_base = Path(args.output)

    is_sweep = len(concurrencies) > 1
    for c in concurrencies:
        if is_sweep:
            output_path = _per_concurrency_path(out_base, c)
            nvidia_path = str(_per_concurrency_path(Path(args.nvidia_smi_csv), c)) if args.nvidia_smi_csv else None
        else:
            output_path = out_base
            nvidia_path = args.nvidia_smi_csv

        summary = await run_one_concurrency(client, args, rows, c, output_path, nvidia_path, is_sweep)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
