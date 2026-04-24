from fastapi import FastAPI
from pydantic import BaseModel

from common.backend_client import classify_article
from common.config import DEFAULT_BASE_URL, MODEL_ID

app = FastAPI(title="LLM News Classification Demo API")


class ClassifyRequest(BaseModel):
    text: str
    base_url: str = DEFAULT_BASE_URL
    model_id: str = MODEL_ID
    config_name: str = "vllm_bf16"


class ClassifyResponse(BaseModel):
    config_name: str
    prediction: str | None
    raw_output: str
    latency_s: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    result = classify_article(
        article=req.text,
        base_url=req.base_url,
        model_id=req.model_id,
    )
    return {
        "config_name": req.config_name,
        "prediction": result["prediction"],
        "raw_output": result["raw_output"],
        "latency_s": result["latency_s"],
    }