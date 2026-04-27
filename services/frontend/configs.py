"""Launch commands and OpenAI model ids for the four comparison backends."""

from __future__ import annotations

import os
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


def _vllm_config_path(relative: str) -> str:
    return str((CONFIG_DIR / relative).resolve())


def build_uvicorn_streaming_hf_command() -> tuple[str, ...]:
    return (
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
    for key in FRONTEND_CONFIG_ORDER:
        spec = base[key]
        openai_id = spec.model_id
        command = spec.command
        available = spec.available
        reason = spec.unavailable_reason
        if key == "vllm_awq_int4":
            awq = _awq_model()
            openai_id = awq
            p = _vllm_config_path("vllm_awq_int4.yaml")
            command = ("vllm", "serve", awq, "--config", p)
            available = True
            reason = None
        if key == "vllm_bf16":
            openai_id = MODEL_ID
        if key == "vllm_bf16_apc":
            openai_id = MODEL_ID
        if key == "hf_baseline_bf16":
            openai_id = os.environ.get("HF_BASELINE_MODEL_ID", MODEL_ID)
            command = build_uvicorn_streaming_hf_command()
        out[key] = FrontendConfig(
            name=spec.name,
            label=spec.label,
            description=spec.description,
            command=command,
            env=dict(spec.env),
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
