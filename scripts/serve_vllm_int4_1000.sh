#!/usr/bin/env bash
# vLLM INT4 (AWQ) quantization — Optimization 3 (Jiaming Liu)
# Model: solidrust/Mistral-7B-Instruct-v0.3-AWQ
# All run artifacts: outputs/<EXPERIMENT_NAME>/{ bench_*.json, nvidia_smi/ }
#
# Usage:
#   ./scripts/serve_vllm_int4_1000.sh serve
#   ./scripts/serve_vllm_int4_1000.sh bench
#   Single concurrency: BENCH_CONCURRENCY=4 ./scripts/serve_vllm_int4_1000.sh bench
#
# Repro (2026-04-26, vllm 0.8.x, GPU L4):
#   ./scripts/serve_vllm_int4_1000.sh serve   # wait for Uvicorn
#   ./scripts/serve_vllm_int4_1000.sh bench

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- venv activation ---
VENV_PATH="${VENV_PATH:-$ROOT/.venv}"
if [ -f "$VENV_PATH/bin/activate" ]; then
  # shellcheck source=/dev/null
  . "$VENV_PATH/bin/activate"
else
  echo "Warning: venv not found at $VENV_PATH — skipping activation" >&2
fi

# --- Config ---
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MODEL="solidrust/Mistral-7B-Instruct-v0.3-AWQ"
SERVED_NAME="mistral-7b-int4"
GPU_MEM="${GPU_UTIL:-0.80}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

# --- Experiment output structure ---
EXPERIMENT_NAME="${EXPERIMENT_NAME:-vllm_int4}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/outputs/$EXPERIMENT_NAME}"
BENCH_IN="${BENCH_INPUT:-data/agnews_bench_1000.jsonl}"
BENCH_OUT="${BENCH_OUTPUT:-$EXPERIMENT_DIR/bench_1000.json}"

mkdir -p "$EXPERIMENT_DIR/nvidia_smi" 2>/dev/null || true

# --- nvidia-smi ---
NVIDIA_SMI_CSV_BASE="${NVIDIA_SMI_CSV_BASE:-$EXPERIMENT_DIR/nvidia_smi/smi_1000.csv}"
NVIDIA_SMI_INTERVAL="${NVIDIA_SMI_INTERVAL:-1.0}"

# --- Benchmark config ---
CONFIG_NAME="${BENCH_CONFIG_NAME:-$EXPERIMENT_NAME}"

case "${1:-}" in
  serve)
    exec vllm serve "$MODEL" \
      --served-model-name "$SERVED_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --quantization awq \
      --dtype float16 \
      --max-model-len 2048 \
      --gpu-memory-utilization "$GPU_MEM"
    ;;
  bench)
    BENCH_EX=( )
    if [ -n "${BENCH_CONCURRENCY:-}" ]; then
      BENCH_EX=(--concurrency "${BENCH_CONCURRENCY}")
    fi

    CL_EX=( )
    if [ -n "${BENCH_CONCURRENCY_LIST:-}" ]; then
      CL_EX=(--concurrency-list "${BENCH_CONCURRENCY_LIST}")
    fi

    NV_SMI=()
    if [ "${BENCH_NVIDIA_SMI:-1}" != "0" ]; then
      NV_SMI=(--nvidia-smi-csv "$NVIDIA_SMI_CSV_BASE" --nvidia-smi-interval "${NVIDIA_SMI_INTERVAL}")
    fi

    # Default: 1,2,4,8,16 in benchmark.py. BENCH_CONCURRENCY=4 forces a single run.
    python "$ROOT/scripts/benchmark.py" \
      --input "$BENCH_IN" \
      --base-url "$BASE_URL" \
      --model-id "$SERVED_NAME" \
      --config-name "$CONFIG_NAME" \
      --output "$BENCH_OUT" \
      "${BENCH_EX[@]}" \
      "${CL_EX[@]}" \
      "${NV_SMI[@]}"
    ;;
  *)
    echo "Usage: $0 {serve|bench}" >&2
    echo "  serve  — vllm serve (AWQ INT4) on ${HOST}:${PORT}" >&2
    echo "  bench  — sweep c 1,2,4,8,16 (default) + nvidia_smi/ CSV" >&2
    echo "         BENCH_NVIDIA_SMI=0 to skip GPU csv; BENCH_CONCURRENCY=N; BENCH_CONCURRENCY_LIST=1,4,8" >&2
    echo "  env:   VENV_PATH (default .venv), VLLM_PORT, GPU_UTIL" >&2
    echo "         EXPERIMENT_NAME (and EXPERIMENT_DIR=.../outputs/\$name), BENCH_*" >&2
    echo "         NVIDIA_SMI_CSV_BASE, NVIDIA_SMI_INTERVAL" >&2
    exit 1
    ;;
esac