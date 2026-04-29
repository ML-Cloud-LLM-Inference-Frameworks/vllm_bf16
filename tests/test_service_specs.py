import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from common.config import MODEL_ID
from common.service_specs import get_service_specs


def _create_complete_local_model(root: Path) -> None:
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (root / "generation_config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors.index.json").write_text("{}", encoding="utf-8")


class ServiceSpecsTest(unittest.TestCase):
    def test_hf_ui_backend_uses_streaming_server(self) -> None:
        spec = get_service_specs()["hf_baseline_bf16"]
        self.assertEqual(spec.command[0], "uvicorn")
        self.assertEqual(spec.command[1], "services.hf_baseline.server_streaming:app")

    def test_ui_vllm_prefix_caching_policies_match_intended_comparison(self) -> None:
        specs = get_service_specs()

        self.assertFalse(specs["vllm_bf16"].server_policy["enable_prefix_caching"])
        self.assertTrue(specs["vllm_bf16_apc"].server_policy["enable_prefix_caching"])
        self.assertFalse(specs["vllm_awq_int4"].server_policy["enable_prefix_caching"])

        bf16_cfg = Path(specs["vllm_bf16"].config_path)
        apc_cfg = Path(specs["vllm_bf16_apc"].config_path)
        int4_cfg = Path(specs["vllm_awq_int4"].config_path)

        self.assertIn("enable-prefix-caching: false", bf16_cfg.read_text(encoding="utf-8"))
        self.assertIn("enable-prefix-caching: true", apc_cfg.read_text(encoding="utf-8"))
        self.assertIn("enable-prefix-caching: false", int4_cfg.read_text(encoding="utf-8"))

    def test_ui_bf16_and_apc_share_the_same_validated_model_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _create_complete_local_model(root)
            env = {
                "SHARED_MISTRAL_MODEL_PATH": str(root),
                "HF_BASELINE_MODEL_PATH": "",
                "VLLM_MODEL_PATH": "",
            }
            with patch.dict(os.environ, env, clear=False):
                specs = get_service_specs()

            bf16 = specs["vllm_bf16"]
            apc = specs["vllm_bf16_apc"]
            self.assertEqual(bf16.command[0:2], ("vllm", "serve"))
            self.assertEqual(apc.command[0:2], ("vllm", "serve"))
            self.assertEqual(bf16.command[2], str(root))
            self.assertEqual(apc.command[2], str(root))
            self.assertEqual(bf16.command[3:5], ("--served-model-name", MODEL_ID))
            self.assertEqual(apc.command[3:5], ("--served-model-name", MODEL_ID))


if __name__ == "__main__":
    unittest.main()
