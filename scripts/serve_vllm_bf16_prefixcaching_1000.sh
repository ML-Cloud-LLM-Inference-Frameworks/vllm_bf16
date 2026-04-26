#!/usr/bin/env bash
# vLLM BF16 + prefix caching, then AG News 1000-request benchmark.
# Matches: common/config.py (DEFAULT_BASE_URL, MODEL_ID) + outputs/bench_*_c{1,2,4,8,16}.json, outputs/proms/
#
# Prereq: conda env with vllm and openai deps, e.g.
#   conda activate llm-inference
#
# Usage (serve blocks; bench = default concurrency sweep 1,2,4,8,16 + proms/):
#   ./scripts/serve_vllm_bf16_prefixcaching_1000.sh serve
#   ./scripts/serve_vllm_bf16_prefixcaching_1000.sh bench
#   (writes outputs/bench_vllm_bf16_prefixcaching_1000_c{1,2,4,8,16}.json, outputs/proms/prom_*_cN.{txt,json})
#   Single concurrency only: BENCH_CONCURRENCY=4 ./scripts/...sh bench
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
# vLLM exposes OpenMetrics at http://host:port/metrics; sweep uses outputs/proms/*_cN.suffix
mkdir -p "$ROOT/outputs/proms" 2>/dev/null || true
PROM_RAW="${PROMETHEUS_RAW_OUTPUT:-$ROOT/outputs/proms/prom_vllm_bf16_prefixcaching_1000.txt}"
PROM_JSON="${PROMETHEUS_JSON_OUTPUT:-$ROOT/outputs/proms/prom_vllm_bf16_prefixcaching_1000.json}"
# Set PROMETHEUS_SAMPLES=1 to add parsed "samples" into the JSON (larger file)
PROMETHEUS_SAMPLES="${PROMETHEUS_SAMPLES:-0}"
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
    SAMPLES=()
    if [ "${PROMETHEUS_SAMPLES}" = "1" ]; then
      SAMPLES=(--prometheus-samples)
    fi
    BENCH_EX=( )
    if [ -n "${BENCH_CONCURRENCY:-}" ]; then
      BENCH_EX=(--concurrency "${BENCH_CONCURRENCY}")
    fi
    CL_EX=( )
    if [ -n "${BENCH_CONCURRENCY_LIST:-}" ]; then
      CL_EX=(--concurrency-list "${BENCH_CONCURRENCY_LIST}")
    fi
    # Default: 1,2,4,8,16 in benchmark.py. BENCH_CONCURRENCY=4 forces a single run.
    python "$ROOT/scripts/benchmark.py" \
      --input "$BENCH_IN" \
      --base-url "$BASE_URL" \
      --model-id "$VLLM_MODEL_HUB" \
      --config-name "$CONFIG_NAME" \
      --output "$BENCH_OUT" \
      --prometheus-raw-output "$PROM_RAW" \
      --prometheus-json-output "$PROM_JSON" \
      "${BENCH_EX[@]}" \
      "${CL_EX[@]}" \
      "${SAMPLES[@]}"
    ;;
  *)
    echo "Usage: $0 {serve|bench}" >&2
    echo "  serve  — vllm serve (bf16, prefix caching) on ${HOST}:${PORT}" >&2
    echo "  bench  — sweep concurrency 1,2,4,8,16 (default) + proms/; ${PROM_RAW}* , ${PROM_JSON}*" >&2
    echo "         BENCH_CONCURRENCY=N for single N; BENCH_CONCURRENCY_LIST=1,4,8 for custom sweep" >&2
    echo "  conda: CONDA_BASE=$CONDA_BASE CONDA_ENV=$CONDA_ENV" >&2
    echo "  model: VLLM_MODEL_HUB and optional VLLM_MODEL_PATH (local dir)" >&2
    echo "  other: VLLM_PORT, BENCH_*, PROMETHEUS_*, set PROMETHEUS_SAMPLES=1 for sample dict in JSON" >&2
    exit 1
    ;;
esac
