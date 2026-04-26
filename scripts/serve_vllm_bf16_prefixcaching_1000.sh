#!/usr/bin/env bash
# vLLM BF16 + prefix caching, then AG News 1000-request benchmark.
# Matches: common/config.py (DEFAULT_BASE_URL, MODEL_ID) + outputs/bench_vllm_bf16_prefixcaching_1000.json
#
# Prereq: conda env with vllm and openai deps, e.g.
#   conda activate llm-inference
#
# Usage (two steps — serve blocks until stopped):
#   ./scripts/serve_vllm_bf16_prefixcaching_1000.sh serve
#   ./scripts/serve_vllm_bf16_prefixcaching_1000.sh bench
#
# Repro (same as 2026-04-26 run, vllm 0.19.1, GPU L4):
#   VLLM_MODEL_PATH=/home/sgcjin/mistral_models/7B-Instruct-v0.3 ./scripts/serve_vllm_bf16_prefixcaching_1000.sh serve
#   (wait for Uvicorn / models) then
#   ./scripts/serve_vllm_bf16_prefixcaching_1000.sh bench
# Optional: log server with `nohup ... serve > /tmp/vllm_serve.log 2>&1 &`
#
# Flags: `vllm serve --help=ModelConfig` / `--help=CacheConfig` / `--help=Frontend`

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-llm-inference}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  . "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
# Hub id (used in benchmark --model-id; also --served-model-name when serving from a local directory)
VLLM_MODEL_HUB="${VLLM_MODEL_HUB:-mistralai/Mistral-7B-Instruct-v0.3}"
# Optional local weights. If unset, use default path when it exists; else serve from hub.
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-}"
if [ -z "$VLLM_MODEL_PATH" ] && [ -d /home/sgcjin/mistral_models/7B-Instruct-v0.3 ]; then
  VLLM_MODEL_PATH="/home/sgcjin/mistral_models/7B-Instruct-v0.3"
fi
BASE_URL="http://127.0.0.1:${PORT}/v1"
BENCH_IN="${BENCH_INPUT:-data/agnews_bench_1000.jsonl}"
BENCH_OUT="${BENCH_OUTPUT:-$ROOT/outputs/bench_vllm_bf16_prefixcaching_1000.json}"
CONFIG_NAME="${BENCH_CONFIG_NAME:-vllm_bf16_prefixcaching}"

case "${1:-}" in
  serve)
    if [ -n "$VLLM_MODEL_PATH" ] && [ -d "$VLLM_MODEL_PATH" ]; then
      exec vllm serve "$VLLM_MODEL_PATH" \
        --served-model-name "$VLLM_MODEL_HUB" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype bfloat16 \
        --enable-prefix-caching
    else
      exec vllm serve "$VLLM_MODEL_HUB" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype bfloat16 \
        --enable-prefix-caching
    fi
    ;;
  bench)
    python "$ROOT/scripts/benchmark.py" \
      --input "$BENCH_IN" \
      --base-url "$BASE_URL" \
      --model-id "$VLLM_MODEL_HUB" \
      --config-name "$CONFIG_NAME" \
      --output "$BENCH_OUT"
    ;;
  *)
    echo "Usage: $0 {serve|bench}" >&2
    echo "  serve  — vllm serve (bf16, prefix caching) on ${HOST}:${PORT}" >&2
    echo "  bench  — PYTHONPATH=. python scripts/benchmark.py -> ${BENCH_OUT}" >&2
    echo "  conda: CONDA_BASE=$CONDA_BASE CONDA_ENV=$CONDA_ENV" >&2
    echo "  model: VLLM_MODEL_HUB (API name) and optional VLLM_MODEL_PATH (local dir)" >&2
    echo "  other: VLLM_PORT, BENCH_INPUT, BENCH_OUTPUT, BENCH_CONFIG_NAME" >&2
    exit 1
    ;;
esac
