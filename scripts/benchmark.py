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
DEFAULT_PROMETHEUS_POLL_INTERVAL_S = 0.5


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = int(round((p / 100) * (len(xs) - 1)))
    return xs[k]


def _per_concurrency_path(base: Path, concurrency: int) -> Path:
    """e.g. outputs/bench_foo.json -> outputs/bench_foo_c4.json"""

    return base.parent / f"{base.stem}_c{concurrency}{base.suffix}"


def _strip_existing_c_suffix(path_str: str) -> str:
    """.../name_c8.json -> .../name.json (avoids _c4_c4 when re-running)."""

    path = Path(path_str)
    stem = path.stem
    if re.search(r"_c\d+$", stem):
        stem = re.sub(r"_c\d+$", "", stem)
    return str(path.parent / f"{stem}{path.suffix}")


def _parse_concurrencies(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _concurrency_suffixed_paths(
    out_base: Path,
    prom_raw_base: str | None,
    prom_json_base: str | None,
    c: int,
) -> tuple[Path, str | None, str | None]:
    output_path = _per_concurrency_path(out_base, c)
    prom_raw = str(_per_concurrency_path(Path(prom_raw_base), c)) if prom_raw_base else None
    prom_json = str(_per_concurrency_path(Path(prom_json_base), c)) if prom_json_base else None
    return output_path, prom_raw, prom_json


def _latency_summary(metric_prefix: str, values: list[float]) -> dict[str, float | None]:
    return {
        f"{metric_prefix}_avg_s": mean(values) if values else None,
        f"{metric_prefix}_p50_s": percentile(values, 50),
        f"{metric_prefix}_p95_s": percentile(values, 95),
        f"{metric_prefix}_p99_s": percentile(values, 99),
    }


def _warmup_request_count(total_rows: int, requested_warmup: int, concurrency: int) -> int:
    # Warm enough requests to exercise the target concurrency level before the measured phase.
    return min(total_rows, max(requested_warmup, concurrency))


async def _fetch_prometheus_scrape_and_samples(metrics_url: str, timeout_s: float = 30.0):
    from common.prometheus_utils import fetch_prometheus_text, parse_prometheus_samples

    scrape = await asyncio.to_thread(fetch_prometheus_text, metrics_url, timeout_s)
    samples = parse_prometheus_samples(scrape.text)
    return scrape, samples


async def _poll_prometheus_during_run(
    metrics_url: str,
    interval_s: float,
    stop_event: asyncio.Event,
    state: dict,
) -> None:
    from common.prometheus_utils import extract_kv_cache_usage_perc_max, summarize_vllm_samples

    timeout_s = max(5.0, interval_s * 4)
    while not stop_event.is_set():
        state["attempts"] += 1
        try:
            _, samples = await _fetch_prometheus_scrape_and_samples(metrics_url, timeout_s=timeout_s)
            state["successful_scrapes"] += 1
            derived = summarize_vllm_samples(samples)
            kv_max = derived.get("kv_cache_usage_perc_max")
            if kv_max is not None:
                current_max = state.get("kv_cache_usage_perc_max")
                if current_max is None or kv_max > current_max:
                    state["kv_cache_usage_perc_max"] = kv_max
        except Exception as exc:  # noqa: BLE001
            if len(state["errors"]) < 3:
                state["errors"].append(f"{type(exc).__name__}: {exc}")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            continue


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
    prom_enabled = bool(prom_raw or prom_json or args.prometheus_samples)
    prom_metrics_url = None
    prom_pre_scrape = None
    prom_pre_samples = None
    prom_pre_error = None
    prom_poll_state = {
        "interval_s": args.prometheus_poll_interval,
        "attempts": 0,
        "successful_scrapes": 0,
        "errors": [],
        "kv_cache_usage_perc_max": None,
    }

    if prom_enabled:
        from common.prometheus_utils import openai_v1_base_to_metrics_url

        prom_metrics_url = openai_v1_base_to_metrics_url(args.base_url)
        try:
            prom_pre_scrape, prom_pre_samples = await _fetch_prometheus_scrape_and_samples(prom_metrics_url)
        except Exception as exc:  # noqa: BLE001
            prom_pre_error = f"{type(exc).__name__}: {exc}"

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
        return {
            "id": row.get("id"),
            "gold_label": row.get("label_name"),
            "pred_label": pred_label,
            "raw_output": raw_output,
            "latency_s": end_to_end_latency_s,
            "service_latency_s": service_latency_s,
            "queue_wait_s": queue_wait_s,
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

    poll_stop_event = asyncio.Event()
    poll_task = None
    if prom_enabled and prom_metrics_url is not None:
        poll_task = asyncio.create_task(
            _poll_prometheus_during_run(
                prom_metrics_url,
                args.prometheus_poll_interval,
                poll_stop_event,
                prom_poll_state,
            )
        )

    nvidia_meta: dict = {}
    try:
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
    finally:
        if poll_task is not None:
            poll_stop_event.set()
            await poll_task

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

    if prom_enabled and prom_metrics_url is not None:
        from common.prometheus_utils import (
            diff_prometheus_samples,
            scrape_to_json_dict,
            summarize_vllm_samples,
        )

        prom_summary: dict = {
            "metrics_url": prom_metrics_url,
            "polling": prom_poll_state,
        }
        if prom_pre_scrape is not None:
            prom_summary["pre_run_fetched_at_utc"] = prom_pre_scrape.fetched_at_utc
        if prom_pre_error is not None:
            prom_summary["pre_run_error"] = prom_pre_error

        try:
            prom_post_scrape, prom_post_samples = await _fetch_prometheus_scrape_and_samples(prom_metrics_url)
            prom_summary["fetched_at_utc"] = prom_post_scrape.fetched_at_utc
            prom_summary["http_status"] = prom_post_scrape.status_code

            samples_for_summary = prom_post_samples
            if prom_pre_samples is not None:
                samples_for_summary = diff_prometheus_samples(prom_post_samples, prom_pre_samples)
                prom_summary["derived_from_pre_run_delta"] = True

            derived = summarize_vllm_samples(samples_for_summary)
            polled_kv_peak = prom_poll_state.get("kv_cache_usage_perc_max")
            if polled_kv_peak is not None:
                derived["kv_cache_usage_perc_max"] = polled_kv_peak
            if derived:
                prom_summary["derived"] = derived

            if prom_raw:
                prom_raw_path = Path(prom_raw)
                prom_raw_path.parent.mkdir(parents=True, exist_ok=True)
                prom_raw_path.write_text(prom_post_scrape.text, encoding="utf-8")
                prom_summary["raw_output"] = str(prom_raw)
            if prom_json:
                json_payload = scrape_to_json_dict(prom_post_scrape, include_samples=args.prometheus_samples)
                json_payload["polling"] = prom_poll_state
                if prom_pre_scrape is not None:
                    json_payload["pre_run"] = {
                        "fetched_at_utc": prom_pre_scrape.fetched_at_utc,
                        "http_status": prom_pre_scrape.status_code,
                    }
                if prom_pre_error is not None:
                    json_payload["pre_run_error"] = prom_pre_error
                if prom_pre_samples is not None:
                    json_payload["derived_from_pre_run_delta"] = True
                if derived:
                    json_payload["derived"] = derived
                prom_json_path = Path(prom_json)
                prom_json_path.parent.mkdir(parents=True, exist_ok=True)
                prom_json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
                prom_summary["json_output"] = str(prom_json)
        except Exception as exc:  # noqa: BLE001
            prom_summary["error"] = f"{type(exc).__name__}: {exc}"

        summary["prometheus"] = prom_summary
    elif args.prometheus_samples:
        summary["prometheus_note"] = "ignored: --prometheus-samples needs --prometheus-json-output"

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
        "--prometheus-poll-interval",
        type=float,
        default=DEFAULT_PROMETHEUS_POLL_INTERVAL_S,
        help="Seconds between in-run Prometheus polls for gauge metrics such as KV cache usage.",
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

    concurrencies = [args.concurrency] if args.concurrency is not None else _parse_concurrencies(args.concurrency_list)
    if not concurrencies:
        raise SystemExit("empty concurrency list")

    client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")
    rows = load_jsonl(Path(args.input), limit=args.limit)
    out_base = Path(args.output)

    is_sweep = len(concurrencies) > 1
    summaries: list[dict] = []
    for c in concurrencies:
        if is_sweep:
            output_path, prom_raw, prom_json = _concurrency_suffixed_paths(
                out_base,
                args.prometheus_raw_output,
                args.prometheus_json_output,
                c,
            )
            nvidia_path = str(_per_concurrency_path(Path(args.nvidia_smi_csv), c)) if args.nvidia_smi_csv else None
        else:
            output_path = out_base
            prom_raw = args.prometheus_raw_output
            prom_json = args.prometheus_json_output
            nvidia_path = args.nvidia_smi_csv

        summary = await run_one_concurrency(
            client,
            args,
            rows,
            c,
            output_path,
            prom_raw,
            prom_json,
            nvidia_path,
            is_sweep,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    if len(summaries) > 1:
        print(
            json.dumps(
                {
                    "concurrency_sweep": [summary["concurrency"] for summary in summaries],
                    "runs": len(summaries),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
