import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "configs"
PROMPT_PATH = PROJECT_ROOT / "prompt_template.txt"

DATASET_ID = "pietrolesci/agnews"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_LOCAL_MISTRAL_PATH = os.environ.get(
    "DEFAULT_LOCAL_MISTRAL_PATH",
    "/home/sgcjin/mistral_models/7B-Instruct-v0.3",
)


def _looks_like_hub_id(raw: str) -> bool:
    if not raw:
        return False
    if raw.startswith("/"):
        return False
    if raw.startswith((".", "~")):
        return False
    if Path(raw).is_absolute():
        return False
    return "/" in raw and ".." not in raw.split("/")


def _resolve_path_candidate(raw: str) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(raw))
    p0 = Path(expanded)
    if p0.is_absolute():
        return p0
    return (Path.cwd() / p0).resolve()


def inspect_local_model_dir(raw_path: str | os.PathLike[str]) -> dict[str, object]:
    raw = str(raw_path)
    path = _resolve_path_candidate(raw)
    if path.is_file():
        path = path.parent

    tokenizer_files: list[str] = []
    if path.is_dir():
        for candidate in path.iterdir():
            if not candidate.is_file():
                continue
            name = candidate.name
            if name in {"tokenizer.json", "tokenizer.model", "tekken.json"} or name.startswith("tokenizer.model."):
                tokenizer_files.append(name)
    tokenizer_files = sorted(tokenizer_files)
    safetensors_files = sorted(p.name for p in path.glob("*.safetensors")) if path.is_dir() else []
    bin_files = sorted(p.name for p in path.glob("*.bin")) if path.is_dir() else []
    has_config = (path / "config.json").is_file()
    has_tokenizer_config = (path / "tokenizer_config.json").is_file()
    has_special_tokens_map = (path / "special_tokens_map.json").is_file()
    has_generation_config = (path / "generation_config.json").is_file()
    has_safetensors_index = (path / "model.safetensors.index.json").is_file()
    has_bin_index = (path / "pytorch_model.bin.index.json").is_file()
    safetensors_index_files: list[str] = []
    missing_safetensors_index_files: list[str] = []
    if has_safetensors_index:
        try:
            payload = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
            safetensors_index_files = sorted(set(payload.get("weight_map", {}).values()))
            missing_safetensors_index_files = [
                name for name in safetensors_index_files if not (path / name).is_file()
            ]
        except Exception:
            missing_safetensors_index_files = ["<unreadable-index>"]
    has_tokenizer_assets = bool(tokenizer_files) and has_tokenizer_config
    has_weight_assets = has_safetensors_index or has_bin_index or bool(safetensors_files) or bool(bin_files)
    complete = path.is_dir() and has_config and has_tokenizer_assets and has_weight_assets
    has_consolidated_safetensors = "consolidated.safetensors" in safetensors_files
    hf_shard_link_compatible = bool(missing_safetensors_index_files) and has_consolidated_safetensors

    missing_core = []
    if not has_config:
        missing_core.append("config.json")
    if not tokenizer_files:
        missing_core.append("tokenizer.json or tokenizer.model")
    if not has_tokenizer_config:
        missing_core.append("tokenizer_config.json")
    if not has_weight_assets:
        missing_core.append("model weights or index")

    return {
        "input_path": raw,
        "resolved_path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "has_config": has_config,
        "has_tokenizer_config": has_tokenizer_config,
        "has_special_tokens_map": has_special_tokens_map,
        "has_generation_config": has_generation_config,
        "tokenizer_files": tokenizer_files,
        "has_safetensors_index": has_safetensors_index,
        "has_bin_index": has_bin_index,
        "safetensors_index_files": safetensors_index_files,
        "missing_safetensors_index_files": missing_safetensors_index_files,
        "safetensors_files": safetensors_files,
        "bin_files": bin_files,
        "has_consolidated_safetensors": has_consolidated_safetensors,
        "hf_shard_link_compatible": hf_shard_link_compatible,
        "complete_for_local_serving": complete,
        "missing_core": missing_core,
    }


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_hf_local_model_path(raw_path: str | os.PathLike[str]) -> dict[str, object]:
    raw = str(raw_path).strip()
    if _looks_like_hub_id(raw):
        return {
            "prepared_path": raw,
            "prepared_is_temp": False,
            "reason": "hub id does not require local shard preparation",
            "inspection": None,
        }
    inspection = inspect_local_model_dir(raw_path)
    resolved = str(inspection["resolved_path"])
    if not inspection["hf_shard_link_compatible"]:
        return {
            "prepared_path": resolved,
            "prepared_is_temp": False,
            "reason": "local directory already directly loadable or not eligible for shard-link compatibility",
            "inspection": inspection,
        }

    source_root = Path(resolved)
    compat_root = Path(tempfile.mkdtemp(prefix="hf_model_compat_"))
    for candidate in source_root.iterdir():
        target = compat_root / candidate.name
        if candidate.is_file():
            _link_or_copy(candidate, target)
        elif candidate.is_dir():
            shutil.copytree(candidate, target, symlinks=True)

    consolidated = source_root / "consolidated.safetensors"
    for missing_name in inspection["missing_safetensors_index_files"]:
        target = compat_root / str(missing_name)
        if not target.exists():
            _link_or_copy(consolidated, target)

    return {
        "prepared_path": str(compat_root),
        "prepared_is_temp": True,
        "reason": "created a temporary HF-compatible shard view from consolidated.safetensors",
        "inspection": inspection,
    }


def select_preferred_model_source(
    primary_env_var: str,
    hub_id: str = MODEL_ID,
    default_local_path: str = DEFAULT_LOCAL_MISTRAL_PATH,
) -> dict[str, object]:
    shared_override = (os.environ.get("SHARED_MISTRAL_MODEL_PATH") or "").strip()
    primary_raw = (os.environ.get(primary_env_var) or "").strip()

    report: dict[str, object] = {
        "primary_env_var": primary_env_var,
        "primary_env_value": primary_raw or None,
        "shared_env_value": shared_override or None,
        "hub_id": hub_id,
        "default_local_path": default_local_path,
    }

    def maybe_select(raw: str, source_name: str) -> dict[str, object] | None:
        if not raw:
            return None
        if _looks_like_hub_id(raw):
            return {
                "selected_source": raw,
                "selected_is_local": False,
                "selected_reason": f"{source_name} uses hub id",
                "inspection": None,
                "selected_client_model_id": raw,
            }
        inspection = inspect_local_model_dir(raw)
        if inspection["complete_for_local_serving"]:
            return {
                "selected_source": inspection["resolved_path"],
                "selected_is_local": True,
                "selected_reason": f"{source_name} uses validated local checkpoint",
                "inspection": inspection,
                "selected_client_model_id": hub_id,
            }
        report[f"{source_name}_inspection"] = inspection
        return None

    for raw, source_name in (
        (primary_raw, primary_env_var),
        (shared_override, "SHARED_MISTRAL_MODEL_PATH"),
        (default_local_path, "default_local_path"),
    ):
        selected = maybe_select(raw, source_name)
        if selected is not None:
            report.update(selected)
            return report

    report.update(
        {
            "selected_source": hub_id,
            "selected_is_local": False,
            "selected_reason": "falling back to hub id",
            "inspection": None,
            "selected_client_model_id": hub_id,
        }
    )
    return report


def get_hf_baseline_selection() -> dict[str, object]:
    return select_preferred_model_source("HF_BASELINE_MODEL_PATH")


def get_vllm_bf16_selection() -> dict[str, object]:
    return select_preferred_model_source("VLLM_MODEL_PATH")


def resolve_hf_baseline_path() -> str:
    """
    Return a local directory, file inside a model tree, or Hub id for ``from_pretrained``.
    When ``HF_BASELINE_MODEL_PATH`` is unset, use ``MODEL_ID`` (Hub). If it is set to a missing
    explicit local path (e.g. ``/path/to/...``), fall back to ``MODEL_ID``. ``org/model`` with no
    matching folder under ``cwd`` is left as a Hub id.
    """
    return str(get_hf_baseline_selection()["selected_source"])


def resolve_vllm_bf16_path() -> str:
    return str(get_vllm_bf16_selection()["selected_source"])


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
