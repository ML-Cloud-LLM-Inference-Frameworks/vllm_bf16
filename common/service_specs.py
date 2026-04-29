import os
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

from common.config import (
    CONFIG_DIR,
    MODEL_ID,
    PROJECT_ROOT,
    get_vllm_bf16_selection,
    resolve_hf_baseline_path,
)

BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
BACKEND_HEALTH_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/health"
BACKEND_BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/v1"
DEFAULT_AWQ_MODEL = "solidrust/Mistral-7B-Instruct-v0.3-AWQ"
DEFAULT_AWQ_SERVED_NAME = "mistral-7b-int4"


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    label: str
    description: str
    command: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    config_path: str | None = None
    model_id: str = MODEL_ID
    base_url: str = BACKEND_BASE_URL
    health_url: str = BACKEND_HEALTH_URL
    server_policy: dict[str, Any] = field(default_factory=dict)
    launch_notes: tuple[str, ...] = ()
    available: bool = True
    unavailable_reason: str | None = None

    def shell_command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.command)


def _vllm_policy(enable_prefix_caching: bool) -> dict[str, Any]:
    return {
        "framework": "vllm",
        "dtype": "bfloat16",
        "generation_config": "vllm",
        "scheduling_policy": "fcfs",
        "max_num_seqs": 16,
        "max_num_batched_tokens": 4096,
        "enable_chunked_prefill": True,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 2048,
        "enable_logging_iteration_details": True,
        "enable_prefix_caching": enable_prefix_caching,
    }


def _python_uvicorn_command() -> tuple[str, ...]:
    return (sys.executable, "-m", "uvicorn")


def _vllm_cli() -> str:
    return shutil.which("vllm") or "vllm"


def get_service_specs() -> dict[str, ServiceSpec]:
    hf_model_id = os.getenv("HF_BASELINE_MODEL_ID", MODEL_ID)
    hf_model_path = resolve_hf_baseline_path()
    vllm_selection = get_vllm_bf16_selection()
    vllm_model_source = str(vllm_selection["selected_source"])
    awq_model = os.getenv("VLLM_AWQ_MODEL", "").strip() or DEFAULT_AWQ_MODEL

    vllm_bf16_config = str((CONFIG_DIR / "vllm_bf16.yaml").resolve())
    vllm_bf16_apc_config = str((CONFIG_DIR / "vllm_bf16_apc.yaml").resolve())
    vllm_awq_int4_config = str((CONFIG_DIR / "vllm_awq_int4.yaml").resolve())

    return {
        "hf_baseline_bf16": ServiceSpec(
            name="hf_baseline_bf16",
            label="HF Baseline BF16",
            description="Transformers + PyTorch baseline with one-request-at-a-time generation on the GPU.",
            command=(
                *_python_uvicorn_command(),
                "services.hf_baseline.server_streaming:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(BACKEND_PORT),
            ),
            env={
                "HF_BASELINE_MODEL_ID": hf_model_id,
                "HF_BASELINE_MODEL_PATH": hf_model_path,
                "HF_BASELINE_CONFIG_NAME": "hf_baseline_bf16",
                "HF_BASELINE_DTYPE": "bfloat16",
            },
            model_id=hf_model_id,
            server_policy={
                "framework": "transformers",
                "dtype": "bfloat16",
                "server_side_batching": False,
                "scheduler": "none",
                "streaming_api": True,
            },
            launch_notes=(
                "This backend uses the Hugging Face streaming-compatible server in services/hf_baseline/server_streaming.py.",
                "Unset HF_BASELINE_MODEL_PATH to use the Hub model (common.config.MODEL_ID), or set it to a local snapshot directory.",
            ),
        ),
        "vllm_bf16": ServiceSpec(
            name="vllm_bf16",
            label="vLLM BF16",
            description="OpenAI-compatible vLLM server with BF16 weights and fixed scheduler controls.",
            command=(
                _vllm_cli(),
                "serve",
                vllm_model_source,
                "--served-model-name",
                MODEL_ID,
                "--config",
                vllm_bf16_config,
            ),
            config_path=vllm_bf16_config,
            model_id=MODEL_ID,
            server_policy=_vllm_policy(enable_prefix_caching=False),
        ),
        "vllm_bf16_apc": ServiceSpec(
            name="vllm_bf16_apc",
            label="vLLM BF16 + APC",
            description="Same vLLM BF16 policy, but with automatic prefix caching enabled.",
            command=(
                _vllm_cli(),
                "serve",
                vllm_model_source,
                "--served-model-name",
                MODEL_ID,
                "--config",
                vllm_bf16_apc_config,
            ),
            config_path=vllm_bf16_apc_config,
            model_id=MODEL_ID,
            server_policy=_vllm_policy(enable_prefix_caching=True),
        ),
        "vllm_awq_int4": ServiceSpec(
            name="vllm_awq_int4",
            label="vLLM AWQ INT4",
            description="vLLM with an AWQ-quantized checkpoint. Requires a quantized model path or HF repo on the VM.",
            command=(_vllm_cli(), "serve", awq_model or "SET_VLLM_AWQ_MODEL", "--config", vllm_awq_int4_config),
            config_path=vllm_awq_int4_config,
            model_id=os.getenv("VLLM_AWQ_SERVED_NAME", DEFAULT_AWQ_SERVED_NAME),
            server_policy={
                **_vllm_policy(enable_prefix_caching=False),
                "dtype": "half",
                "quantization": "awq_marlin",
            },
            available=True,
            unavailable_reason=None,
            launch_notes=(
                "By default this launches the public AWQ checkpoint solidrust/Mistral-7B-Instruct-v0.3-AWQ.",
                "Set VLLM_AWQ_MODEL on the VM to override the checkpoint path or Hugging Face repo.",
            ),
        ),
    }


def get_service_spec(name: str) -> ServiceSpec:
    specs = get_service_specs()
    if name not in specs:
        choices = ", ".join(sorted(specs))
        raise KeyError(f"Unknown service '{name}'. Available services: {choices}")
    return specs[name]


def get_orchestrator_cwd() -> str:
    return str(PROJECT_ROOT)
