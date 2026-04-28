import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "configs"
PROMPT_PATH = PROJECT_ROOT / "prompt_template.txt"

DATASET_ID = "pietrolesci/agnews"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def resolve_hf_baseline_path() -> str:
    """
    Return a local directory, file inside a model tree, or Hub id for ``from_pretrained``.
    When ``HF_BASELINE_MODEL_PATH`` is unset, use ``MODEL_ID`` (Hub). If it is set to a missing
    explicit local path (e.g. ``/path/to/...``), fall back to ``MODEL_ID``. ``org/model`` with no
    matching folder under ``cwd`` is left as a Hub id.
    """
    raw = (os.environ.get("HF_BASELINE_MODEL_PATH") or "").strip()
    if not raw:
        return MODEL_ID
    exp = os.path.expanduser(os.path.expandvars(raw))
    p0 = Path(exp)
    explicit = p0.is_absolute() or exp.startswith((".", "~"))
    if not explicit and "/" in exp and ".." not in exp.split("/"):
        if not (Path.cwd() / exp).resolve().exists():
            return raw
    p = p0 if p0.is_absolute() else (Path.cwd() / p0).resolve()
    if p.is_file():
        p = p.parent
    if p.is_dir() and p.exists() and (p / "config.json").is_file():
        return str(p)
    if p.is_dir() and p.exists():
        return str(p)
    if explicit and not p.exists():
        print(
            f"[hf_baseline] HF_BASELINE_MODEL_PATH={raw!r} is not on disk; using Hub {MODEL_ID!r} instead.",
            file=sys.stderr,
        )
        return MODEL_ID
    return raw


LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

LABELS = list(LABEL_MAP.values())

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 8
