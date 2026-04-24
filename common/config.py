from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
PROMPT_PATH = PROJECT_ROOT / "prompt_template.txt"

DATASET_ID = "pietrolesci/agnews"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

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