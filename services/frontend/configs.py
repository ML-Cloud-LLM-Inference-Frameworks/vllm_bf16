"""Launch commands and OpenAI model ids for the four comparison backends."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass

from common.config import CONFIG_DIR, MODEL_ID
from common.service_specs import get_orchestrator_cwd, get_service_specs

FRONTEND_CONFIG_ORDER: tuple[str, ...] = (
    "hf_baseline_bf16",
    "vllm_bf16",
    "vllm_bf16_apc",
    "vllm_awq_int4",
)

_DEFAULT_AWQ_MODEL = "solidrust/Mistral-7B-Instruct-v0.3-AWQ"
_DEFAULT_AWQ_SERVED_NAME = "mistral-7b-int4"


def _frontend_base_model_source() -> str:
    return (os.environ.get("FRONTEND_BASE_MODEL_SOURCE") or MODEL_ID).strip() or MODEL_ID


@dataclass(frozen=True, slots=True)
class FrontendConfig:
    name: str
    label: str
    description: str
    command: tuple[str, ...]
    env: dict[str, str]
    openai_model_id: str
    has_prometheus: bool
    available: bool
    unavailable_reason: str | None = None
    config_path: str | None = None


def _awq_model() -> str:
    return (os.environ.get("VLLM_AWQ_MODEL") or _DEFAULT_AWQ_MODEL).strip()


def _vllm_cli() -> str:
    return shutil.which("vllm") or "vllm"


def _vllm_config_path(relative: str) -> str:
    return str((CONFIG_DIR / relative).resolve())


def build_uvicorn_streaming_hf_command() -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "uvicorn",
        "services.hf_baseline.server_streaming:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    )


def get_frontend_configs() -> dict[str, FrontendConfig]:
    base = get_service_specs()
    out: dict[str, FrontendConfig] = {}
    frontend_base_model_source = _frontend_base_model_source()
    for key in FRONTEND_CONFIG_ORDER:
        spec = base[key]
        openai_id = spec.model_id
        command = spec.command
        env = dict(spec.env)
        available = spec.available
        reason = spec.unavailable_reason
        if key == "vllm_awq_int4":
            awq = _awq_model()
            openai_id = os.environ.get("VLLM_AWQ_SERVED_NAME", _DEFAULT_AWQ_SERVED_NAME).strip() or _DEFAULT_AWQ_SERVED_NAME
            p = _vllm_config_path("vllm_awq_int4.yaml")
            command = (_vllm_cli(), "serve", awq, "--config", p)
            available = True
            reason = None
        if key == "vllm_bf16":
            openai_id = MODEL_ID
            p = _vllm_config_path("vllm_bf16.yaml")
            command = (_vllm_cli(), "serve", frontend_base_model_source, "--served-model-name", MODEL_ID, "--config", p)
            env["VLLM_MODEL_PATH"] = frontend_base_model_source
        if key == "vllm_bf16_apc":
            openai_id = MODEL_ID
            p = _vllm_config_path("vllm_bf16_apc.yaml")
            command = (_vllm_cli(), "serve", frontend_base_model_source, "--served-model-name", MODEL_ID, "--config", p)
            env["VLLM_MODEL_PATH"] = frontend_base_model_source
        if key == "hf_baseline_bf16":
            openai_id = os.environ.get("HF_BASELINE_MODEL_ID", MODEL_ID)
            command = build_uvicorn_streaming_hf_command()
            env["HF_BASELINE_MODEL_PATH"] = frontend_base_model_source
        out[key] = FrontendConfig(
            name=spec.name,
            label=spec.label,
            description=spec.description,
            command=command,
            env=env,
            openai_model_id=openai_id,
            has_prometheus=not key.startswith("hf_"),
            available=available,
            unavailable_reason=reason,
            config_path=spec.config_path,
        )
    return out


def get_config(name: str) -> FrontendConfig:
    cfg = get_frontend_configs()
    if name not in cfg:
        raise KeyError(f"Unknown config {name!r}. Choose from: {', '.join(cfg)}")
    return cfg[name]


def merge_environ(base: dict[str, str] | None) -> dict[str, str]:
    out = dict(os.environ)
    if base:
        out.update(base)
    root = get_orchestrator_cwd()
    if out.get("PYTHONPATH"):
        out["PYTHONPATH"] = f"{root}{os.pathsep}{out['PYTHONPATH']}"
    else:
        out["PYTHONPATH"] = root
    return out
