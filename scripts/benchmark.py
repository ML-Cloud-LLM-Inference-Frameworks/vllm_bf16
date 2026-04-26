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
    """e.g. outputs/bench_foo.json -> outputs/bench_foo_c4.json"""
    return base.parent / f"{base.stem}_c{concurrency}{base.suffix}"


def _strip_existing_c_suffix(p: str) -> str:
    """.../name_c8.json -> .../name.json  (avoids _c4_c4 when re-running)"""
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
    prom_raw: str | None,
    prom_json: str | None,
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
            return {
                "id": row["id"],
                "gold_label": row["label_name"],
                "pred_label": pred_label,
                "raw_output": raw_output,
                "latency_s": latency_s,
            }

    async def run_requests_phase():
        """Warmup + measured batch. GPU sampling should wrap this if nvidia_smi is used."""
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
            nvidia_meta = {
                "error": f"{type(e).__name__}: {e}",
                "csv": nvidia_smi_csv,
            }
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
        "n_valid_predictions": n_valid,
        "n_invalid_predictions": n_invalid,
        "accuracy_valid_only": accuracy_valid_only,
        "accuracy_overall_invalid_as_wrong": accuracy_overall,
    }
    if nvidia_meta:
        summary["nvidia_smi"] = nvidia_meta

    if prom_raw or prom_json:
        from common.prometheus_utils import (
            fetch_prometheus_text,
            openai_v1_base_to_metrics_url,
            parse_prometheus_samples,
            scrape_to_json_dict,
            summarize_vllm_samples,
        )

        murl = openai_v1_base_to_metrics_url(args.base_url)
        try:
            sc = fetch_prometheus_text(murl, timeout_s=30.0)
            pr: dict = {
                "metrics_url": murl,
                "fetched_at_utc": sc.fetched_at_utc,
                "http_status": sc.status_code,
            }
            if prom_raw:
                pr_path = Path(prom_raw)
                pr_path.parent.mkdir(parents=True, exist_ok=True)
                pr_path.write_text(sc.text, encoding="utf-8")
                pr["raw_output"] = str(prom_raw)
            if prom_json:
                pj = scrape_to_json_dict(sc, include_samples=args.prometheus_samples)
                pj_path = Path(prom_json)
                pj_path.parent.mkdir(parents=True, exist_ok=True)
                pj_path.write_text(json.dumps(pj, indent=2), encoding="utf-8")
                pr["json_output"] = str(prom_json)
            if args.prometheus_samples:
                pr["derived"] = summarize_vllm_samples(parse_prometheus_samples(sc.text))
            summary["prometheus"] = pr
        except Exception as e:  # noqa: BLE001
            summary["prometheus"] = {
                "metrics_url": murl,
                "error": f"{type(e).__name__}: {e}",
            }
    elif args.prometheus_samples:
        summary["prometheus_note"] = "ignored: --prometheus-samples needs --prometheus-json-output"

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
    out = []
    for p in s.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return out


def _concurrency_suffixed_paths(
    out_base: Path,
    prom_raw_base: str | None,
    prom_json_base: str | None,
    c: int,
) -> tuple[Path, str | None, str | None]:
    op = _per_concurrency_path(out_base, c)
    pr, pj = prom_raw_base, prom_json_base
    if pr:
        prp = _per_concurrency_path(Path(pr), c)
        pr = str(prp)
    if pj:
        pjp = _per_concurrency_path(Path(pj), c)
        pj = str(pjp)
    return op, pr, pj


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Shared input JSONL file")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--config-name", default="vllm_bf16")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Single run only. If set, runs only this value (overrides --concurrency-list).",
    )
    parser.add_argument(
        "--concurrency-list",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated concurrencies when --concurrency is not set. Default: 1,2,4,8,16",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", required=True, help="Path to save benchmark JSON (or stem for multi-c)")
    parser.add_argument("--pred-output", default=None, help="Optional per-request JSONL (multi-c: _cN suffix)")
    parser.add_argument(
        "--prometheus-raw-output",
        default=None,
        help="After each run, GET vLLM /metrics; multi-c: _cN suffix on path; put under your experiment output dir if you use it",
    )
    parser.add_argument(
        "--prometheus-json-output",
        default=None,
        help="JSON scrape artifact; multi-c: _cN suffix",
    )
    parser.add_argument(
        "--prometheus-samples",
        action="store_true",
        help="When used with --prometheus-json-output, add parsed 'samples' dict (can be large)",
    )
    parser.add_argument(
        "--nvidia-smi-csv",
        default=None,
        help="During warmup+measured requests, run nvidia-smi -l in background and log CSV; sweep: _cN suffix",
    )
    parser.add_argument(
        "--nvidia-smi-interval",
        type=float,
        default=1.0,
        help="Seconds between nvidia-smi samples (rounded up to an integer >=1 for the driver -l arg)",
    )
    args = parser.parse_args()

    if args.concurrency is not None:
        concurrencies = [args.concurrency]
    else:
        concurrencies = _parse_concurrencies(args.concurrency_list)
    if not concurrencies:
        raise SystemExit("empty concurrency list")

    client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")
    rows = load_jsonl(Path(args.input), limit=args.limit)
    out_base = Path(args.output)

    is_sweep = len(concurrencies) > 1
    summaries: list[dict] = []
    for c in concurrencies:
        if is_sweep:
            o_path, pr, pj = _concurrency_suffixed_paths(
                out_base, args.prometheus_raw_output, args.prometheus_json_output, c
            )
            nvidia_path = None
            if args.nvidia_smi_csv:
                nvidia_path = str(_per_concurrency_path(Path(args.nvidia_smi_csv), c))
        else:
            o_path = out_base
            pr, pj = args.prometheus_raw_output, args.prometheus_json_output
            nvidia_path = args.nvidia_smi_csv

        s = await run_one_concurrency(
            client, args, rows, c, o_path, pr, pj, nvidia_path, is_sweep
        )
        summaries.append(s)
        print(json.dumps(s, indent=2), flush=True)

    if len(summaries) > 1:
        print(json.dumps({"concurrency_sweep": [x["concurrency"] for x in summaries], "runs": len(summaries)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
