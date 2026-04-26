#!/usr/bin/env python3
"""
Fetch vLLM Prometheus text from the same host/port as the API (GET /metrics, not /v1).
Artefacts from `serve_*. sh bench` live under `outputs/<EXPERIMENT_NAME>/proms/`. This
CLI is for ad-hoc scrapes, e.g. --metrics-url http://127.0.0.1:8000/metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.prometheus_utils import (  # noqa: E402
    fetch_prometheus_text,
    openai_v1_base_to_metrics_url,
    scrape_to_json_dict,
    PrometheusScrape,
)
from urllib.error import HTTPError, URLError  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--openai-v1-base",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI base URL; used to infer /metrics on same port",
    )
    p.add_argument(
        "--metrics-url",
        default=None,
        help="If set, overrides --openai-v1-base and is used directly (e.g. http://host:8000/metrics)",
    )
    p.add_argument("--raw-output", default=None, help="Write text/plain body here")
    p.add_argument(
        "--json-output",
        default=None,
        help="Write JSON: scrape metadata, optional samples, raw_prometheus",
    )
    p.add_argument(
        "--no-raw-in-json",
        action="store_true",
        help="Omit 'raw_prometheus' from the JSON (keep samples + scrape only if samples on)",
    )
    p.add_argument(
        "--samples-in-json",
        action="store_true",
        help="Add parsed 'samples' dict to the JSON (can be large)",
    )
    p.add_argument("--timeout", type=float, default=30.0)
    args = p.parse_args()
    murl = (args.metrics_url or "").strip() or openai_v1_base_to_metrics_url(args.openai_v1_base)

    try:
        sc: PrometheusScrape = fetch_prometheus_text(murl, timeout_s=args.timeout)
    except (HTTPError, URLError, TimeoutError) as e:
        print(f"scrape failed: {murl}: {e}", file=sys.stderr)
        sys.exit(1)

    if not (200 <= sc.status_code < 300):
        print(
            f"scrape: unexpected HTTP {sc.status_code} for {murl} ({len(sc.text)} bytes body)",
            file=sys.stderr,
        )
        if sc.text:
            print(sc.text[:2000], file=sys.stderr)
        sys.exit(1)

    if args.raw_output:
        pth = Path(args.raw_output)
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_text(sc.text, encoding="utf-8")

    if args.json_output:
        d = scrape_to_json_dict(sc, include_samples=args.samples_in_json)
        if args.no_raw_in_json:
            d = {k: v for k, v in d.items() if k != "raw_prometheus"}
        pth = Path(args.json_output)
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_text(json.dumps(d, indent=2), encoding="utf-8")

    print(json.dumps(sc.to_dict()["scrape"], indent=2))


if __name__ == "__main__":
    main()
