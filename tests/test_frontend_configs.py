from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from common.config import MODEL_ID
from services.frontend.configs import get_frontend_configs
from services.frontend.orchestrator import _frontend_service_specs


class FrontendConfigsTest(unittest.TestCase):
    def test_ui_defaults_non_int4_backends_to_hub_source(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FRONTEND_BASE_MODEL_SOURCE": MODEL_ID,
                "HF_BASELINE_MODEL_PATH": "",
                "VLLM_MODEL_PATH": "",
            },
            clear=False,
        ):
            cfgs = get_frontend_configs()

        hf = cfgs["hf_baseline_bf16"]
        bf16 = cfgs["vllm_bf16"]
        apc = cfgs["vllm_bf16_apc"]
        int4 = cfgs["vllm_awq_int4"]

        self.assertEqual(hf.env["HF_BASELINE_MODEL_PATH"], MODEL_ID)
        self.assertEqual(bf16.command[2], MODEL_ID)
        self.assertEqual(apc.command[2], MODEL_ID)
        self.assertEqual(bf16.env["VLLM_MODEL_PATH"], MODEL_ID)
        self.assertEqual(apc.env["VLLM_MODEL_PATH"], MODEL_ID)
        self.assertNotEqual(int4.command[2], MODEL_ID)

    def test_ui_backend_manager_launch_specs_use_frontend_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FRONTEND_BASE_MODEL_SOURCE": MODEL_ID,
                "HF_BASELINE_MODEL_PATH": "/should/not/win",
                "VLLM_MODEL_PATH": "/should/not/win",
            },
            clear=False,
        ):
            specs = _frontend_service_specs()

        self.assertEqual(specs["hf_baseline_bf16"].env["HF_BASELINE_MODEL_PATH"], MODEL_ID)
        self.assertEqual(specs["vllm_bf16"].command[2], MODEL_ID)
        self.assertEqual(specs["vllm_bf16_apc"].command[2], MODEL_ID)
        self.assertNotEqual(specs["vllm_awq_int4"].command[2], MODEL_ID)


if __name__ == "__main__":
    unittest.main()
