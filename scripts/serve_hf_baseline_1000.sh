#!/usr/bin/env bash
# HF baseline (transformers BF16), then AG News 1000-request benchmark.
# Artifacts: outputs/hf_baseline_bf16/{ bench_*.json, nvidia_smi/ }
#
# Usage:
#   ./scripts/serve_hf_baseline_1000.sh serve   # blocks; start in one terminal
#   ./scripts/serve_hf_baseline_1000.sh bench   # run in another terminal
#
# Env overrides:
#   HF_BASELINE_MODEL_PATH  path to local model weights (default: local instruct checkpoint if present, else Hub)
#   HF_BASELINE_PORT        port for uvicorn (default: 8000)
#   BENCH_CONCURRENCY       single concurrency run (e.g. BENCH_CONCURRENCY=1)
#   BENCH_CONCURRENCY_LIST  comma-separated list (default: 1,2,4,8,16)
#   BENCH_NVIDIA_SMI        set to 0 to skip GPU logging
#   EXPERIMENT_NAME         output subfolder name under outputs/ (default: hf_baseline_bf16)

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

HOST="${HF_BASELINE_HOST:-0.0.0.0}"
PORT="${HF_BASELINE_PORT:-8000}"
MODEL_HUB="${HF_BASELINE_MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.3}"
HF_SOURCE="$(python "$ROOT/scripts/resolve_model_source.py" --stack hf --field source)"
HF_REASON="$(python "$ROOT/scripts/resolve_model_source.py" --stack hf --field reason)"
export HF_BASELINE_MODEL_PATH="$HF_SOURCE"
export HF_BASELINE_CONFIG_NAME="${HF_BASELINE_CONFIG_NAME:-hf_baseline_bf16}"

BASE_URL="http://127.0.0.1:${PORT}/v1"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-hf_baseline_bf16}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$ROOT/outputs/$EXPERIMENT_NAME}"
BENCH_IN="${BENCH_INPUT:-data/agnews_bench_1000.jsonl}"
BENCH_OUT="${BENCH_OUTPUT:-$EXPERIMENT_DIR/bench_1000.json}"
mkdir -p "$EXPERIMENT_DIR/nvidia_smi" 2>/dev/null || true
NVIDIA_SMI_CSV_BASE="${NVIDIA_SMI_CSV_BASE:-$EXPERIMENT_DIR/nvidia_smi/smi_1000.csv}"
NVIDIA_SMI_INTERVAL="${NVIDIA_SMI_INTERVAL:-1.0}"
CONFIG_NAME="${BENCH_CONFIG_NAME:-$EXPERIMENT_NAME}"

case "${1:-}" in
  serve)
    echo "[hf_baseline] using source: $HF_SOURCE ($HF_REASON)" >&2
    exec uvicorn services.hf_baseline.server_streaming:app \
      --host "$HOST" \
      --port "$PORT"
    ;;
  bench)
    BENCH_EX=()
    if [ -n "${BENCH_CONCURRENCY:-}" ]; then
      BENCH_EX=(--concurrency "${BENCH_CONCURRENCY}")
    fi
    CL_EX=()
    if [ -n "${BENCH_CONCURRENCY_LIST:-}" ]; then
      CL_EX=(--concurrency-list "${BENCH_CONCURRENCY_LIST}")
    fi
    NV_SMI=()
    if [ "${BENCH_NVIDIA_SMI:-1}" != "0" ]; then
      NV_SMI=(--nvidia-smi-csv "$NVIDIA_SMI_CSV_BASE" --nvidia-smi-interval "${NVIDIA_SMI_INTERVAL}")
    fi
    python3 "$ROOT/scripts/benchmark_hf.py" \
      --input "$BENCH_IN" \
      --base-url "$BASE_URL" \
      --model-id "$MODEL_HUB" \
      --config-name "$CONFIG_NAME" \
      --output "$BENCH_OUT" \
      "${BENCH_EX[@]}" \
      "${CL_EX[@]}" \
      "${NV_SMI[@]}"
    ;;
  *)
    echo "Usage: $0 {serve|bench}" >&2
    echo "  serve  — start uvicorn HF baseline on ${HOST}:${PORT} (blocks)" >&2
    echo "  bench  — sweep c 1,2,4,8,16 (default) + nvidia_smi/ CSV" >&2
    echo "           BENCH_NVIDIA_SMI=0 to skip GPU csv; BENCH_CONCURRENCY=N for single run" >&2
    echo "  model: HF_BASELINE_MODEL_PATH or SHARED_MISTRAL_MODEL_PATH; else validated /home/sgcjin/mistral_models/7B-Instruct-v0.3; else Hub" >&2
    echo "  other: EXPERIMENT_NAME, HF_BASELINE_PORT, BENCH_CONCURRENCY_LIST" >&2
    exit 1
    ;;
esac
