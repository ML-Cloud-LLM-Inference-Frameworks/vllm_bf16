from pathlib import Path
import unittest

from common.service_specs import get_service_specs


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


if __name__ == "__main__":
    unittest.main()
