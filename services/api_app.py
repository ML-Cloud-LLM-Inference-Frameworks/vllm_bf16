import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common.backend_client import classify_article
from services.orchestrator import BackendServiceManager

app = FastAPI(title="LLM News Classification Demo Orchestrator")
manager = BackendServiceManager()


class ConfigInfo(BaseModel):
    name: str
    label: str
    description: str
    available: bool
    unavailable_reason: str | None
    config_path: str | None
    health_url: str
    base_url: str
    model_id: str
    server_policy: dict
    launch_command: str
    launch_notes: list[str]


class ServiceSelectionRequest(BaseModel):
    config_name: str


class ServiceStatusResponse(BaseModel):
    status: str
    message: str
    active_config_name: str | None
    desired_config_name: str | None
    pid: int | None
    started_at: str | None
    ready_at: str | None
    health_url: str | None
    base_url: str | None
    model_id: str | None
    log_path: str | None


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    config_name: str
    prediction: str | None
    raw_output: str
    latency_s: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/configs", response_model=list[ConfigInfo])
def list_configs():
    return manager.list_service_specs()


@app.get("/service/status", response_model=ServiceStatusResponse)
def service_status():
    return manager.status_snapshot()


@app.post("/service/select", response_model=ServiceStatusResponse)
async def service_select(req: ServiceSelectionRequest):
    try:
        return await manager.schedule_switch(req.config_name)
    except (RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/service/stop", response_model=ServiceStatusResponse)
async def service_stop():
    try:
        return await manager.schedule_stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    try:
        target = manager.get_ready_target()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = await asyncio.to_thread(
        classify_article,
        article=req.text,
        base_url=target.base_url,
        model_id=target.model_id,
    )
    return {
        "config_name": target.name,
        "prediction": result["prediction"],
        "raw_output": result["raw_output"],
        "latency_s": result["latency_s"],
    }


@app.on_event("shutdown")
async def shutdown_event():
    await manager.shutdown()
