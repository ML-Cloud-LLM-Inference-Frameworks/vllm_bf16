from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.config import inspect_local_model_dir, select_preferred_model_source


def _create_complete_local_model(root: Path) -> None:
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (root / "generation_config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors.index.json").write_text("{}", encoding="utf-8")


class ModelSourceResolutionTest(unittest.TestCase):
    def test_inspect_local_model_dir_detects_complete_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _create_complete_local_model(root)
            report = inspect_local_model_dir(str(root))
            self.assertTrue(report["complete_for_local_serving"])
            self.assertEqual(report["missing_core"], [])

    def test_invalid_local_override_falls_back_to_hub(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {"HF_BASELINE_MODEL_PATH": str(root)}, clear=False):
                report = select_preferred_model_source(
                    "HF_BASELINE_MODEL_PATH",
                    hub_id="mistralai/Mistral-7B-Instruct-v0.3",
                    default_local_path=str(root),
                )
            self.assertEqual(report["selected_source"], "mistralai/Mistral-7B-Instruct-v0.3")
            self.assertFalse(report["selected_is_local"])

    def test_shared_local_override_can_drive_both_hf_and_vllm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _create_complete_local_model(root)
            env = {
                "SHARED_MISTRAL_MODEL_PATH": str(root),
                "HF_BASELINE_MODEL_PATH": "",
                "VLLM_MODEL_PATH": "",
            }
            with patch.dict(os.environ, env, clear=False):
                hf = select_preferred_model_source("HF_BASELINE_MODEL_PATH", default_local_path="/missing")
                vllm = select_preferred_model_source("VLLM_MODEL_PATH", default_local_path="/missing")
            self.assertTrue(hf["selected_is_local"])
            self.assertEqual(hf["selected_source"], str(root))
            self.assertEqual(vllm["selected_source"], str(root))
            self.assertEqual(hf["selected_client_model_id"], vllm["selected_client_model_id"])


if __name__ == "__main__":
    unittest.main()
