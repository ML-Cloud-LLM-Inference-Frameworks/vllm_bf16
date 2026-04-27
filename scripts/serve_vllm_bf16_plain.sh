#!/usr/bin/env bash
# vLLM BF16 without automatic prefix caching (alias for configs/vllm_bf16.yaml)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec vllm serve --config "$ROOT/configs/vllm_bf16.yaml"
