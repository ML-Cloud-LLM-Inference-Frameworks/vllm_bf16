# LLM Inference Benchmark and Demo

This repo now contains two layers:

- Benchmark clients in `scripts/` that talk to an OpenAI-compatible backend at `http://127.0.0.1:8000/v1`
- A demo orchestrator in `services/api_app.py` that can start exactly one heavyweight backend at a time for the frontend demo

## Checked-in backend configs

The four experiment/demo configs are:

- `hf_baseline_bf16`
- `vllm_bf16`
- `vllm_bf16_apc`
- `vllm_awq_int4`

The checked-in vLLM YAML files live in `configs/`:

- `configs/vllm_bf16.yaml`
- `configs/vllm_bf16_apc.yaml`
- `configs/vllm_awq_int4.yaml`

These YAMLs freeze the server-side controls you want to document in the report:

- `generation-config: vllm`
- `scheduling-policy: fcfs`
- `max-num-seqs: 16`
- `max-num-batched-tokens: 4096`
- `enable-chunked-prefill: true`
- `gpu-memory-utilization: 0.9`
- `enable-logging-iteration-details: true`

Only the feature under study changes across the vLLM configs:

- `vllm_bf16`: BF16, prefix caching off
- `vllm_bf16_apc`: BF16, prefix caching on
- `vllm_awq_int4`: AWQ INT4, prefix caching off

`configs/vllm_awq_int4.yaml` intentionally uses `SET_VLLM_AWQ_MODEL` as a placeholder. Replace it on the VM with the AWQ checkpoint you actually plan to benchmark.

## VM launch commands

Activate the VM environment first:

```bash
conda activate llm-inference
```

Run the Hugging Face baseline:

```bash
export HF_BASELINE_MODEL_PATH=/home/hl3945/mistral-7b
uvicorn services.hf_baseline.server:app --host 0.0.0.0 --port 8000
```

Run the plain vLLM BF16 backend:

```bash
vllm serve --config configs/vllm_bf16.yaml
```

Run the vLLM BF16 + prefix caching backend:

```bash
vllm serve --config configs/vllm_bf16_apc.yaml
```

Run the AWQ INT4 backend after replacing the model placeholder:

```bash
export VLLM_AWQ_MODEL=/path/to/your-awq-checkpoint
vllm serve "$VLLM_AWQ_MODEL" --config configs/vllm_awq_int4.yaml
```

Useful checks:

```bash
curl -sf http://127.0.0.1:8000/health && echo ready
curl -s http://127.0.0.1:8000/metrics > logs/current_backend_metrics.prom
```

## Demo orchestrator

Run the orchestrator on a separate port so it can launch the active backend on `8000`:

```bash
uvicorn services.api_app:app --host 0.0.0.0 --port 8001
```

The orchestrator exposes:

- `GET /configs`
- `POST /service/select`
- `GET /service/status`
- `POST /service/stop`
- `POST /classify`

Suggested frontend flow:

1. Load `GET /configs`.
2. Let the user pick one config.
3. Call `POST /service/select` with `{"config_name":"vllm_bf16"}`.
4. Poll `GET /service/status` until `status == "ready"`.
5. Call `POST /classify` with `{"text":"..."}`.

`/classify` proxies to the currently active backend and returns the prediction plus measured end-to-end latency from the backend client.

## Benchmarking

For experiments, benchmark the backend directly on port `8000`, not the orchestrator:

```bash
python scripts/benchmark.py \
  --input data/agnews_bench_1000.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --config-name vllm_bf16 \
  --concurrency 4 \
  --warmup 10 \
  --output outputs/vllm_bf16_c4.json
```

## Notes

- The Hugging Face baseline service now reads `HF_BASELINE_MODEL_PATH`, `HF_BASELINE_MODEL_ID`, `HF_BASELINE_CONFIG_NAME`, and `HF_BASELINE_DTYPE` from the environment.
- The orchestrator uses `VLLM_AWQ_MODEL` to know which AWQ checkpoint to launch for `vllm_awq_int4`.
- The orchestrator keeps one backend alive at a time and writes backend logs into `logs/<config_name>.log`.
- The benchmark client already controls client-side concurrency in `scripts/benchmark.py`; the checked-in YAML files make the vLLM server-side scheduler policy explicit.
