#!/usr/bin/env bash
# Hugging Face baseline with OpenAI streaming (used by the comparison UI for TTFT)
# Default: Hub model in common.config.MODEL_ID. Optional: export HF_BASELINE_MODEL_PATH=/path/to/local/snapshot
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec uvicorn services.hf_baseline.server_streaming:app --host "${VLLM_HOST:-127.0.0.1}" --port "${VLLM_PORT:-8000}"
