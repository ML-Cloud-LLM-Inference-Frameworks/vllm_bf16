import argparse
import asyncio
import json
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


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Shared input JSONL file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--config-name", default="vllm_bf16")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", required=True, help="Path to save benchmark summary JSON")
    parser.add_argument("--pred-output", default=None, help="Optional path to save per-request predictions")
    parser.add_argument(
        "--prometheus-raw-output",
        default=None,
        help="After benchmark, GET vLLM /metrics (same port as --base-url) and write text here",
    )
    parser.add_argument(
        "--prometheus-json-output",
        default=None,
        help="After benchmark, write JSON with scrape info, raw_prometheus, optional samples",
    )
    parser.add_argument(
        "--prometheus-samples",
        action="store_true",
        help="When used with --prometheus-json-output, add parsed 'samples' dict (can be large)",
    )
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")
    rows = load_jsonl(Path(args.input), limit=args.limit)

    sem = asyncio.Semaphore(args.concurrency)

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
            return {
                "id": row["id"],
                "gold_label": row["label_name"],
                "pred_label": pred_label,
                "raw_output": raw_output,
                "latency_s": latency_s,
            }

    # Warmup
    warmup_rows = rows[: min(args.warmup, len(rows))]
    for row in warmup_rows:
        await one_request(row)

    measured_rows = rows[min(args.warmup, len(rows)) :]
    t_start = time.perf_counter()
    results = await asyncio.gather(*[one_request(r) for r in measured_rows])
    total_time = time.perf_counter() - t_start

    latencies = [r["latency_s"] for r in results]
    throughput = len(results) / total_time if total_time > 0 else 0.0

    valid = [r for r in results if r["pred_label"] is not None]
    n_valid = len(valid)
    n_invalid = len(results) - n_valid
    n_correct = sum(1 for r in valid if r["pred_label"] == r["gold_label"])

    accuracy_valid_only = (n_correct / n_valid) if n_valid > 0 else 0.0
    accuracy_overall = (n_correct / len(results)) if results else 0.0

    summary = {
        "config_name": args.config_name,
        "model_id": args.model_id,
        "input_file": args.input,
        "n_requests_measured": len(results),
        "warmup_requests": len(warmup_rows),
        "concurrency": args.concurrency,
        "throughput_req_per_s": throughput,
        "latency_avg_s": mean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "n_valid_predictions": n_valid,
        "n_invalid_predictions": n_invalid,
        "accuracy_valid_only": accuracy_valid_only,
        "accuracy_overall_invalid_as_wrong": accuracy_overall,
    }

    if args.prometheus_raw_output or args.prometheus_json_output:
        from common.prometheus_utils import (
            fetch_prometheus_text,
            openai_v1_base_to_metrics_url,
            scrape_to_json_dict,
        )

        murl = openai_v1_base_to_metrics_url(args.base_url)
        try:
            sc = fetch_prometheus_text(murl, timeout_s=30.0)
            pr: dict = {
                "metrics_url": murl,
                "fetched_at_utc": sc.fetched_at_utc,
                "http_status": sc.status_code,
            }
            if args.prometheus_raw_output:
                pr_path = Path(args.prometheus_raw_output)
                pr_path.parent.mkdir(parents=True, exist_ok=True)
                pr_path.write_text(sc.text, encoding="utf-8")
                pr["raw_output"] = str(args.prometheus_raw_output)
            if args.prometheus_json_output:
                pj = scrape_to_json_dict(sc, include_samples=args.prometheus_samples)
                pj_path = Path(args.prometheus_json_output)
                pj_path.parent.mkdir(parents=True, exist_ok=True)
                pj_path.write_text(json.dumps(pj, indent=2), encoding="utf-8")
                pr["json_output"] = str(args.prometheus_json_output)
            summary["prometheus"] = pr
        except Exception as e:  # noqa: BLE001
            summary["prometheus"] = {
                "metrics_url": murl,
                "error": f"{type(e).__name__}: {e}",
            }
    elif args.prometheus_samples:
        summary["prometheus_note"] = "ignored: --prometheus-samples needs --prometheus-json-output"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.pred_output is not None:
        pred_path = Path(args.pred_output)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())