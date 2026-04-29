#!/usr/bin/env bash
# vLLM BF16 without prefix caching, then AG News 1000-request benchmark.
# All run artifacts: outputs/<EXPERIMENT_NAME>/{ bench_*.json, proms/, nvidia_smi/ } (defaults in this file)

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
VLLM_MODEL_HUB="${VLLM_MODEL_HUB:-mistralai/Mistral-7B-Instruct-v0.3}"
VLLM_MODEL_SOURCE="$(python "$ROOT/scripts/resolve_model_source.py" --stack vllm --field source)"
VLLM_MODEL_REASON="$(python "$ROOT/scripts/resolve_model_source.py" --stack vllm --field reason)"
GPU_MEM="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-vllm_bf16}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/outputs/$EXPERIMENT_NAME}"
BENCH_IN="${BENCH_INPUT:-data/agnews_bench_1000.jsonl}"
BENCH_OUT="${BENCH_OUTPUT:-$EXPERIMENT_DIR/bench_1000.json}"
mkdir -p "$EXPERIMENT_DIR/proms" "$EXPERIMENT_DIR/nvidia_smi" 2>/dev/null || true
PROM_RAW="${PROMETHEUS_RAW_OUTPUT:-$EXPERIMENT_DIR/proms/prom_1000.txt}"
PROM_JSON="${PROMETHEUS_JSON_OUTPUT:-$EXPERIMENT_DIR/proms/prom_1000.json}"
NVIDIA_SMI_CSV_BASE="${NVIDIA_SMI_CSV_BASE:-$EXPERIMENT_DIR/nvidia_smi/smi_1000.csv}"
NVIDIA_SMI_INTERVAL="${NVIDIA_SMI_INTERVAL:-1.0}"
PROMETHEUS_SAMPLES="${PROMETHEUS_SAMPLES:-1}"
CONFIG_NAME="${BENCH_CONFIG_NAME:-$EXPERIMENT_NAME}"

case "${1:-}" in
  serve)
    echo "[vllm_bf16] using source: $VLLM_MODEL_SOURCE ($VLLM_MODEL_REASON)" >&2
    exec vllm serve "$VLLM_MODEL_SOURCE" \
      --served-model-name "$VLLM_MODEL_HUB" \
      --host "$HOST" \
      --port "$PORT" \
      --dtype bfloat16 \
      --scheduling-policy fcfs \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --enable-chunked-prefill \
      --no-enable-prefix-caching \
      --gpu-memory-utilization "$GPU_MEM" \
      --max-model-len "$MAX_MODEL_LEN"
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
      "${NV_SMI[@]}" \
      "${SAMPLES[@]}"
    ;;
  *)
    echo "Usage: $0 {serve|bench}" >&2
    echo "  serve  - vllm serve (bf16, no prefix caching) on ${HOST}:${PORT}" >&2
    echo "  bench  - sweep c 1,2,4,8,16 (default) + proms/ + nvidia_smi/ CSV" >&2
    echo "           BENCH_NVIDIA_SMI=0 to skip GPU csv; BENCH_CONCURRENCY=N; BENCH_CONCURRENCY_LIST=1,4,8" >&2
    echo "  conda: CONDA_BASE=$CONDA_BASE CONDA_ENV=$CONDA_ENV" >&2
    echo "  model: VLLM_MODEL_PATH or SHARED_MISTRAL_MODEL_PATH; else validated /home/sgcjin/mistral_models/7B-Instruct-v0.3; else Hub" >&2
    echo "  other: EXPERIMENT_NAME, BENCH_*, VLLM_PORT, PROMETHEUS_* (PROMETHEUS_SAMPLES defaults to 1)" >&2
    exit 1
    ;;
esac
