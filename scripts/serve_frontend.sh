#!/usr/bin/env bash
# Web UI: compare all four configs sequentially (port 8000 = model; 7860 = UI by default)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# HF baseline loads from Hub (common.config.MODEL_ID) unless you export HF_BASELINE_MODEL_PATH to a local dir
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-llm-inference}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  . "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
PORT="${FRONTEND_PORT:-7860}"
exec python -m uvicorn services.frontend.app:app --host "${FRONTEND_HOST:-0.0.0.0}" --port "$PORT"
