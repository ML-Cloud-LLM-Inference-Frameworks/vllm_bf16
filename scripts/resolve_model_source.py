#!/usr/bin/env python
"""Inspect the shared Mistral model source used by HF and vLLM BF16 stacks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import (
    DEFAULT_LOCAL_MISTRAL_PATH,
    get_hf_baseline_selection,
    get_vllm_bf16_selection,
    inspect_local_model_dir,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _local_file_signatures(raw_path: str | None) -> dict[str, Any] | None:
    if not raw_path:
        return None
    inspection = inspect_local_model_dir(raw_path)
    if not inspection["complete_for_local_serving"]:
        return None

    root = Path(str(inspection["resolved_path"]))
    files: dict[str, str] = {}
    inspection_tokenizer_files = list(inspection.get("tokenizer_files", []))
    for name in (
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        *inspection_tokenizer_files,
    ):
        candidate = root / name
        if candidate.is_file():
            files[name] = _sha256(candidate)
    return {
        "resolved_path": str(root),
        "files": files,
    }


def build_report() -> dict[str, Any]:
    default_local = inspect_local_model_dir(DEFAULT_LOCAL_MISTRAL_PATH)
    hf = get_hf_baseline_selection()
    vllm = get_vllm_bf16_selection()

    hf_signatures = _local_file_signatures(str(hf["selected_source"])) if hf["selected_is_local"] else None
    vllm_signatures = _local_file_signatures(str(vllm["selected_source"])) if vllm["selected_is_local"] else None

    return {
        "default_local_path": DEFAULT_LOCAL_MISTRAL_PATH,
        "default_local_inspection": default_local,
        "hf": hf,
        "vllm_bf16": vllm,
        "hf_vllm_same_source": hf["selected_source"] == vllm["selected_source"],
        "hf_vllm_same_client_model_id": hf["selected_client_model_id"] == vllm["selected_client_model_id"],
        "hf_local_signatures": hf_signatures,
        "vllm_local_signatures": vllm_signatures,
        "hf_vllm_same_local_signatures": hf_signatures == vllm_signatures if hf_signatures and vllm_signatures else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", choices=("hf", "vllm"), help="Return one field for a single stack.")
    parser.add_argument(
        "--field",
        choices=("source", "is-local", "reason", "client-model-id"),
        help="Field to print when --stack is set.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON report.")
    args = parser.parse_args()

    report = build_report()
    if args.stack:
        if not args.field:
            parser.error("--field is required when --stack is set")
        key = "hf" if args.stack == "hf" else "vllm_bf16"
        stack = report[key]
        if args.field == "source":
            print(stack["selected_source"])
        elif args.field == "is-local":
            print("1" if stack["selected_is_local"] else "0")
        elif args.field == "reason":
            print(stack["selected_reason"])
        else:
            print(stack["selected_client_model_id"])
        return

    if args.field:
        parser.error("--field requires --stack")

    indent = 2 if args.pretty else None
    print(json.dumps(report, indent=indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
