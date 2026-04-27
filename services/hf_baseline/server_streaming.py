"""HF + Transformers: OpenAI-compatible /v1/chat with optional SSE streaming (TTFT)."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.config import MAX_TOKENS, MODEL_ID, PROMPT_PATH, resolve_hf_baseline_path
from common.parser import parse_label

MODEL_PATH = resolve_hf_baseline_path()
SERVE_MODEL_ID = os.getenv("HF_BASELINE_MODEL_ID", MODEL_ID)
CONFIG_NAME = os.getenv("HF_BASELINE_CONFIG_NAME", "hf_baseline_bf16")
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")


def _resolve_dtype(n: str):
    m = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    return m.get(n.lower().strip() or "bfloat16", torch.bfloat16)


print("loading model (server_streaming)...")
_tok = AutoTokenizer.from_pretrained(MODEL_PATH)
_llm = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=_resolve_dtype(os.environ.get("HF_BASELINE_DTYPE", "bfloat16")),
    device_map="auto",
)
_llm.eval()
print("model loaded.")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = SERVE_MODEL_ID
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = MAX_TOKENS
    stream: bool = False
    stream_options: Optional[dict[str, Any]] = None  # type: ignore[valid-type]


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str = "chatcmpl-hf"
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


def _user_text(msgs: list[Message]) -> str:
    for m in msgs:
        if m.role == "user" and m.content:
            return m.content
    return ""


def _mistr_inst(user_block: str) -> str:
    t = (user_block or "").strip()
    if t.startswith("[INST]") or t.startswith("You are a "):
        return t
    return f"[INST] {user_block} [/INST]"


def _sync_generate(prompt: str) -> tuple[str, float]:
    s = _mistr_inst(prompt)
    inputs = _tok(s, return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = _llm.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=_tok.eos_token_id,
        )
    lat = time.perf_counter() - t0
    raw = _tok.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    return raw, lat


def _iter_streaming(body: str, model_name: str, cap: int):
    """Yield SSE `data: …` lines for one completion."""
    s = _mistr_inst(body)
    ins = _tok(s, return_tensors="pt").to("cuda")
    n_prompt = int(ins["input_ids"].shape[1])
    streamer = TextIteratorStreamer(_tok, skip_special_tokens=True)
    gen_kw: dict[str, Any] = {k: v for k, v in ins.items()}
    gen_kw.update(
        {
            "max_new_tokens": min(int(cap or MAX_TOKENS), MAX_TOKENS),
            "do_sample": False,
            "pad_token_id": int(_tok.eos_token_id) if _tok.eos_token_id is not None else None,
            "streamer": streamer,
        }
    )
    thr = threading.Thread(target=_llm.generate, kwargs=gen_kw, daemon=True)
    thr.start()
    cmid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    acc: list[str] = []
    for new in streamer:
        if not new:
            continue
        acc.append(new)
        ch = {
            "id": cmid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": new},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(ch, ensure_ascii=False)}\n\n"
    raw = "".join(acc)
    n_comp = max(0, int(len(_tok.encode(raw, add_special_tokens=False)) if raw else 0))
    fin = {
        "id": cmid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": n_prompt,
            "completion_tokens": n_comp,
            "total_tokens": n_prompt + n_comp,
        },
    }
    yield f"data: {json.dumps(fin, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


app = FastAPI(title="HF Baseline Streaming OpenAI API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    u = _user_text(req.messages)
    if not req.stream:
        raw, lat = _sync_generate(u)
        return ChatResponse(
            id="chatcmpl-hf",
            object="chat.completion",
            model=req.model,
            choices=[Choice(index=0, message=Message(role="assistant", content=raw), finish_reason="stop")],
            latency_s=round(float(lat), 4),
        )
    return StreamingResponse(
        _iter_streaming(u, req.model, int(req.max_tokens or MAX_TOKENS)),
        media_type="text/event-stream",
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    p = PROMPT_TEMPLATE.format(article=req.text)
    raw, lat = _sync_generate(p)
    return ClassifyResponse(
        config_name=CONFIG_NAME,
        prediction=parse_label(raw),
        raw_output=raw,
        latency_s=round(float(lat), 4),
    )
