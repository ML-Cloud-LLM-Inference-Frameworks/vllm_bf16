import os
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.config import MAX_TOKENS, MODEL_ID, PROMPT_PATH
from common.parser import parse_label

MODEL_PATH = os.getenv("HF_BASELINE_MODEL_PATH", "/home/hl3945/mistral-7b")
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
    dtype=resolve_dtype(DTYPE_NAME),
    device_map="auto",
)
model.eval()
print("model loaded.")


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = MODEL_ID
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
    ttft_s: float


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    config_name: str
    prediction: Optional[str]
    raw_output: str
    latency_s: float
    ttft_s: float


def run_inference(prompt: str) -> tuple[str, float, float]:
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(chat_text, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        "input_ids": input_ids,
        "streamer": streamer,
        "max_new_tokens": MAX_TOKENS,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "pad_token_id": tokenizer.eos_token_id,
    }

    t0 = time.perf_counter()
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    ttft: float | None = None
    chunks: list[str] = []
    for chunk in streamer:
        if ttft is None:
            ttft = time.perf_counter() - t0
        chunks.append(chunk)

    thread.join()
    total_latency = time.perf_counter() - t0
    raw = "".join(chunks).strip()
    return raw, total_latency, ttft if ttft is not None else total_latency

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    user_msg = next((m.content for m in req.messages if m.role == "user"), "")
    raw, latency, ttft = run_inference(user_msg)
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
        ttft_s=round(ttft, 4),
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    prompt = PROMPT_TEMPLATE.format(article=req.text)
    raw, latency, ttft = run_inference(prompt)
    return ClassifyResponse(
        config_name=CONFIG_NAME,
        prediction=parse_label(raw),
        raw_output=raw,
        latency_s=round(latency, 4),
        ttft_s=round(ttft, 4),
    )
