import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.config import MAX_TOKENS, MODEL_ID, PROMPT_PATH, resolve_hf_baseline_path
from common.parser import parse_label

MODEL_PATH = resolve_hf_baseline_path()
SERVE_MODEL_ID = os.getenv("HF_BASELINE_MODEL_ID", MODEL_ID)
CONFIG_NAME = os.getenv("HF_BASELINE_CONFIG_NAME", "hf_baseline_bf16")
DTYPE_NAME = os.getenv("HF_BASELINE_DTYPE", "bfloat16").lower()
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")

app = FastAPI(title="HF Baseline Inference Service")


def resolve_dtype(dtype_name: str):
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if dtype_name not in mapping:
        choices = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported HF_BASELINE_DTYPE '{dtype_name}'. Choose from: {choices}")
    return mapping[dtype_name]


print("loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=resolve_dtype(DTYPE_NAME),
    device_map="auto",
)
model.eval()
print("model loaded.")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = SERVE_MODEL_ID
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = MAX_TOKENS


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatResponse(BaseModel):
    id: str = "chatcmpl-hf-baseline"
    object: str = "chat.completion"
    model: str
    choices: list[Choice]
    latency_s: float


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    config_name: str
    prediction: Optional[str]
    raw_output: str
    latency_s: float


def run_inference(prompt: str) -> tuple[str, float]:
    formatted = f"[INST] {prompt} [/INST]"
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0
    raw = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()
    return raw, latency


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    user_msg = next((m.content for m in req.messages if m.role == "user"), "")
    raw, latency = run_inference(user_msg)
    return ChatResponse(
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=raw),
                finish_reason="stop",
            )
        ],
        latency_s=round(latency, 4),
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    prompt = PROMPT_TEMPLATE.format(article=req.text)
    raw, latency = run_inference(prompt)
    return ClassifyResponse(
        config_name=CONFIG_NAME,
        prediction=parse_label(raw),
        raw_output=raw,
        latency_s=round(latency, 4),
    )
