from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.frontend.configs import FrontendConfig
from services.frontend.runner import run_jsonl_bench


class _FakeBenchModule:
    def __init__(self) -> None:
        self.observed_warmup: int | None = None

    async def run_one_concurrency(
        self,
        client,
        args,
        rows,
        concurrency,
        output_path,
        *extra,
        **kwargs,
    ):
        self.observed_warmup = int(args.warmup)
        return {
            "config_name": args.config_name,
            "model_id": args.model_id,
            "n_requests_measured": max(0, len(rows) - int(args.warmup)),
            "warmup_requests": int(args.warmup),
            "concurrency": concurrency,
            "throughput_req_per_s": 1.23,
            "latency_p50_s": 0.45,
            "latency_p95_s": 0.67,
            "ttft_avg_s": 0.12,
            "accuracy_valid_only": 1.0,
            "accuracy_overall_invalid_as_wrong": 1.0,
        }


class FrontendRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_small_jsonl_batches_reduce_warmup_so_requests_are_measured(self) -> None:
        fake = _FakeBenchModule()
        cfg = FrontendConfig(
            name="hf_baseline_bf16",
            label="HF Baseline BF16",
            description="test",
            command=("uvicorn",),
            env={},
            openai_model_id="mistralai/Mistral-7B-Instruct-v0.3",
            has_prometheus=False,
            available=True,
        )
        with tempfile.TemporaryDirectory() as td:
            config_dir = Path(td) / "out"
            config_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = Path(td) / "small.jsonl"
            jsonl_path.write_text(
                "\n".join(
                    [
                        json.dumps({"text": "hello", "label_name": "World"}),
                        json.dumps({"text": "world", "label_name": "World"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch("services.frontend.runner.get_benchmark", return_value=fake):
                result = await run_jsonl_bench(
                    cfg,
                    "http://127.0.0.1:8000/v1",
                    jsonl_path,
                    config_dir,
                    concurrency=2,
                    warmup=10,
                )

        self.assertEqual(fake.observed_warmup, 1)
        self.assertEqual(result["ui_requested_warmup"], 10)
        self.assertEqual(result["ui_effective_warmup"], 1)
        self.assertEqual(result["n_requests_measured"], 1)
        self.assertIn("Warmup reduced", result["ui_note"])


if __name__ == "__main__":
    unittest.main()
