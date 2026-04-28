"""Benchmark script for HF baseline — captures TTFT from server response and logs nvidia-smi."""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from statistics import mean
from openai import AsyncOpenAI

from common.config import PROMPT_PATH, TEMPERATURE, TOP_P, MAX_TOKENS
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


def _strip_existing_c_suffix(p: str) -> str:
    path = Path(p)
    s = path.stem
    if re.search(r"_c\d+$", s):
        s = re.sub(r"_c\d+$", "", s)
    return str(path.parent / f"{s}{path.suffix}")


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
        async with sem:
            t0 = time.perf_counter()
            resp = await client.chat.completions.create(
                model=args.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )
            latency_s = time.perf_counter() - t0
            raw_output = resp.choices[0].message.content or ""
            pred_label = parse_label(raw_output)
            ttft_s = (resp.model_extra or {}).get("ttft_s") if hasattr(resp, "model_extra") else None
            return {
                "id": row["id"],
                "gold_label": row["label_name"],
                "pred_label": pred_label,
                "raw_output": raw_output,
                "latency_s": latency_s,
                "ttft_s": ttft_s,
            }

    async def run_requests_phase():
        warmup_rows = rows[: min(args.warmup, len(rows))]
        for row in warmup_rows:
            await one_request(row)
        measured_rows = rows[min(args.warmup, len(rows)) :]
        t0 = time.perf_counter()
        r = await asyncio.gather(*[one_request(r) for r in measured_rows])
        return warmup_rows, r, time.perf_counter() - t0

    nvidia_meta: dict = {}
    if nvidia_smi_csv:
        from common.nvidia_smi_sampler import nvidia_smi_log_csv
        from common.nvidia_smi_summary import summarize_nvidia_smi_csv

        try:
            with nvidia_smi_log_csv(nvidia_smi_csv, interval_s=args.nvidia_smi_interval) as p:
                warmup_rows, results, total_time = await run_requests_phase()
            nvidia_summary = summarize_nvidia_smi_csv(p)
            nvidia_meta = {
                "csv": str(p),
                "interval_s": args.nvidia_smi_interval,
                "summary": nvidia_summary,
            }
        except OSError as e:
            warmup_rows, results, total_time = await run_requests_phase()
            nvidia_meta = {"error": f"{type(e).__name__}: {e}", "csv": nvidia_smi_csv}
    else:
        warmup_rows, results, total_time = await run_requests_phase()

    latencies = [r["latency_s"] for r in results]
    throughput = len(results) / total_time if total_time > 0 else 0.0

    valid = [r for r in results if r["pred_label"] is not None]
    n_valid = len(valid)
    n_invalid = len(results) - n_valid
    n_correct = sum(1 for r in valid if r["pred_label"] == r["gold_label"])

    accuracy_valid_only = (n_correct / n_valid) if n_valid > 0 else 0.0
    accuracy_overall = (n_correct / len(results)) if results else 0.0

    ttfts = [r["ttft_s"] for r in results if r.get("ttft_s") is not None]

    summary = {
        "config_name": args.config_name,
        "model_id": args.model_id,
        "input_file": args.input,
        "n_requests_measured": len(results),
        "warmup_requests": len(warmup_rows),
        "concurrency": concurrency,
        "throughput_req_per_s": throughput,
        "latency_avg_s": mean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "ttft_avg_s": mean(ttfts) if ttfts else None,
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "n_valid_predictions": n_valid,
        "n_invalid_predictions": n_invalid,
        "accuracy_valid_only": accuracy_valid_only,
        "accuracy_overall_invalid_as_wrong": accuracy_overall,
    }
    if nvidia_meta:
        summary["nvidia_smi"] = nvidia_meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.pred_output is not None:
        if is_sweep:
            pbase = Path(_strip_existing_c_suffix(args.pred_output))
            pred_path = _per_concurrency_path(pbase, concurrency)
        else:
            pred_path = Path(args.pred_output)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return summary


def _parse_concurrencies(s: str) -> list[int]:
    return [int(p.strip()) for p in s.split(",") if p.strip()]


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
            o_path = _per_concurrency_path(out_base, c)
            nvidia_path = str(_per_concurrency_path(Path(args.nvidia_smi_csv), c)) if args.nvidia_smi_csv else None
        else:
            o_path = out_base
            nvidia_path = args.nvidia_smi_csv

        s = await run_one_concurrency(client, args, rows, c, o_path, nvidia_path, is_sweep)
        print(json.dumps(s, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
