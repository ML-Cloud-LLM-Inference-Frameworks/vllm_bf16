import sys
import time
import torch
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common.config import MODEL_ID, PROMPT_PATH, MAX_TOKENS
from common.parser import parse_label

MODEL_PATH = "/home/hl3945/mistral-7b"
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")

app = FastAPI(title="HF Baseline Inference Service")

print("loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto"
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
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
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
        choices=[Choice(
            index=0,
            message=Message(role="assistant", content=raw),
            finish_reason="stop"
        )],
        latency_s=round(latency, 4)
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    prompt = PROMPT_TEMPLATE.format(article=req.text)
    raw, latency = run_inference(prompt)
    return ClassifyResponse(
        config_name="hf_baseline",
        prediction=parse_label(raw),
        raw_output=raw,
        latency_s=round(latency, 4)
    )
