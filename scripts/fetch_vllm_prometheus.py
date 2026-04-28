#!/usr/bin/env python3
"""
Fetch vLLM Prometheus text from the same host/port as the API (GET /metrics, not /v1).

Normally the `serve_vllm_*.sh bench` scripts capture these artifacts automatically.
This CLI is for ad-hoc scrapes such as `--metrics-url http://127.0.0.1:8000/metrics`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

# repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.prometheus_utils import (  # noqa: E402
    PrometheusScrape,
    fetch_prometheus_text,
    openai_v1_base_to_metrics_url,
    scrape_to_json_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openai-v1-base",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI base URL; used to infer /metrics on same port",
    )
    parser.add_argument(
        "--metrics-url",
        default=None,
        help="If set, overrides --openai-v1-base and is used directly (e.g. http://host:8000/metrics)",
    )
    parser.add_argument("--raw-output", default=None, help="Write text/plain body here")
    parser.add_argument(
        "--json-output",
        default=None,
        help="Write JSON: scrape metadata, optional samples, raw_prometheus",
    )
    parser.add_argument(
        "--no-raw-in-json",
        action="store_true",
        help="Omit 'raw_prometheus' from the JSON (keep samples + scrape only if samples on)",
    )
    parser.add_argument(
        "--samples-in-json",
        action="store_true",
        help="Add parsed 'samples' dict to the JSON (can be large)",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    metrics_url = (args.metrics_url or "").strip() or openai_v1_base_to_metrics_url(args.openai_v1_base)

    try:
        scrape: PrometheusScrape = fetch_prometheus_text(metrics_url, timeout_s=args.timeout)
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"scrape failed: {metrics_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not (200 <= scrape.status_code < 300):
        print(
            f"scrape: unexpected HTTP {scrape.status_code} for {metrics_url} ({len(scrape.text)} bytes body)",
            file=sys.stderr,
        )
        if scrape.text:
            print(scrape.text[:2000], file=sys.stderr)
        sys.exit(1)

    if args.raw_output:
        output_path = Path(args.raw_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(scrape.text, encoding="utf-8")

    if args.json_output:
        payload = scrape_to_json_dict(scrape, include_samples=args.samples_in_json)
        if args.no_raw_in_json:
            payload = {key: value for key, value in payload.items() if key != "raw_prometheus"}
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(scrape.to_dict()["scrape"], indent=2))


if __name__ == "__main__":
    main()
