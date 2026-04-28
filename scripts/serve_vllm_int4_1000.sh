#!/usr/bin/env bash
# vLLM INT4 (AWQ) quantization — Optimization 3 (Jiaming Liu)
# Model: solidrust/Mistral-7B-Instruct-v0.3-AWQ
# All run artifacts: outputs/<EXPERIMENT_NAME>/{ bench_*.json, proms/, nvidia_smi/ }
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

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-llm-inference}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  . "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MODEL="solidrust/Mistral-7B-Instruct-v0.3-AWQ"
SERVED_NAME="mistral-7b-int4"
GPU_MEM="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

# Experiment: all run outputs go under outputs/<name>/
EXPERIMENT_NAME="${EXPERIMENT_NAME:-vllm_int4}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/outputs/$EXPERIMENT_NAME}"
BENCH_IN="${BENCH_INPUT:-data/agnews_bench_1000.jsonl}"
BENCH_OUT="${BENCH_OUTPUT:-$EXPERIMENT_DIR/bench_1000.json}"
mkdir -p "$EXPERIMENT_DIR/proms" "$EXPERIMENT_DIR/nvidia_smi" 2>/dev/null || true

PROM_RAW="${PROMETHEUS_RAW_OUTPUT:-$EXPERIMENT_DIR/proms/prom_1000.txt}"
PROM_JSON="${PROMETHEUS_JSON_OUTPUT:-$EXPERIMENT_DIR/proms/prom_1000.json}"

# nvidia-smi: logs during each concurrency's warmup+measured phase (BENCH_NVIDIA_SMI=0 to disable)
NVIDIA_SMI_CSV_BASE="${NVIDIA_SMI_CSV_BASE:-$EXPERIMENT_DIR/nvidia_smi/smi_1000.csv}"
NVIDIA_SMI_INTERVAL="${NVIDIA_SMI_INTERVAL:-1.0}"

# Set PROMETHEUS_SAMPLES=1 to add parsed "samples" into the JSON (larger file)
PROMETHEUS_SAMPLES="${PROMETHEUS_SAMPLES:-0}"

# Records as config_name in benchmark JSON; default matches the experiment
CONFIG_NAME="${BENCH_CONFIG_NAME:-$EXPERIMENT_NAME}"

case "${1:-}" in
  serve)
    exec vllm serve "$MODEL" \
      --served-model-name "$SERVED_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --quantization awq_marlin \
      --dtype float16 \
      --scheduling-policy fcfs \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --enable-chunked-prefill \
      --no-enable-prefix-caching \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEM"
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
      --prometheus-raw-output "$PROM_RAW" \
      --prometheus-json-output "$PROM_JSON" \
      "${BENCH_EX[@]}" \
      "${CL_EX[@]}" \
      "${NV_SMI[@]}" \
      "${SAMPLES[@]}"
    ;;
  *)
    echo "Usage: $0 {serve|bench}" >&2
    echo "  serve  — vllm serve (AWQ INT4) on ${HOST}:${PORT}" >&2
    echo "  bench  — sweep c 1,2,4,8,16 (default) + proms/ + nvidia_smi/ CSV" >&2
    echo "         BENCH_NVIDIA_SMI=0 to skip GPU csv; BENCH_CONCURRENCY=N; BENCH_CONCURRENCY_LIST=1,4,8" >&2
    echo "  conda: CONDA_BASE=$CONDA_BASE CONDA_ENV=$CONDA_ENV" >&2
    echo "  model: MODEL=$MODEL SERVED_NAME=$SERVED_NAME" >&2
    echo "  other: EXPERIMENT_NAME (and EXPERIMENT_DIR=.../outputs/\$name), BENCH_*, VLLM_PORT" >&2
    echo "         PROMETHEUS_*, NVIDIA_SMI_CSV_BASE, NVIDIA_SMI_INTERVAL" >&2
    exit 1
    ;;
esac
