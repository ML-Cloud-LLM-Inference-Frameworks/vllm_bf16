"""FastAPI: static UI, create job, single-shot SSE (no on-disk job artifacts)."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from services.frontend.configs import get_frontend_configs
from services.frontend.orchestrator import (
    Event,
    Job,
    backend_status_snapshot,
    create_job,
    run_job,
    shutdown_backend_manager,
)

app = FastAPI(title="4-config LLM comparison")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC = Path(__file__).resolve().parent / "static"
_PENDING: dict[str, Job] = {}


def _static_dir() -> Path:
    _STATIC.mkdir(parents=True, exist_ok=True)
    return _STATIC


@app.get("/")
def index():
    p = _static_dir() / "index.html"
    if p.is_file():
        return FileResponse(p)
    return JSONResponse(
        {
            "ok": True,
            "message": f"Add {_static_dir() / 'index.html'} for the UI, or call POST/GET /api/jobs/…",
        }
    )


if _STATIC.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=_STATIC),
        name="static",
    )


@app.get("/api/configs")
def api_configs() -> dict[str, Any]:
    from services.frontend.configs import get_frontend_configs

    out: dict[str, Any] = {}
    for k, c in get_frontend_configs().items():
        out[k] = {
            "id": c.name,
            "label": c.label,
            "description": c.description,
            "has_prometheus": c.has_prometheus,
            "openai_model_id": c.openai_model_id,
            "available": c.available,
            "unavailable_reason": c.unavailable_reason,
        }
    return out


def _parse_configs(raw: str) -> list[str] | None:
    t = (raw or "").strip()
    if not t or t in ("[]", "null"):
        return None
    d = json.loads(t)
    if not isinstance(d, list):
        raise ValueError("configs must be a JSON array of strings")
    return [str(x) for x in d]


def _save_upload(upload: UploadFile, upload_dir: Path) -> Path:
    safe_name = upload.filename.replace("/", "_").replace("\\", "_")[:120]
    dest = upload_dir / f"u_{uuid.uuid4().hex}_{safe_name}"
    with dest.open("wb") as outw:
        shutil.copyfileobj(upload.file, outw)
    return dest


@app.get("/api/backend-status")
def api_backend_status() -> dict[str, Any]:
    status = backend_status_snapshot()
    configs = get_frontend_configs()
    active_id = status.get("active_config_name")
    desired_id = status.get("desired_config_name")
    if active_id in configs:
        status["active_label"] = configs[active_id].label
    if desired_id in configs:
        status["desired_label"] = configs[desired_id].label
    return status


@app.post("/api/jobs")
async def api_create_job(
    mode: str = Form("text"),
    text: str = Form(""),
    configs: str = Form("[]"),
    concurrency: int = Form(4),
    limit: int = Form(0),
    files: list[UploadFile] = File([]),
) -> Any:
    try:
        cfg = _parse_configs(configs)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    lim: int | None
    if limit and int(limit) > 0:
        lim = int(limit)
    else:
        lim = None
    if mode not in ("text", "jsonl"):
        raise HTTPException(400, "mode must be text or jsonl")
    udir = Path(__file__).resolve().parent / "_uploads"
    udir.mkdir(parents=True, exist_ok=True)
    text_body = (text or "").strip() or None
    uploaded_files: list[tuple[str, Path]] = []
    for upload in files:
        if upload and upload.filename:
            display_name = upload.filename.replace("/", "_").replace("\\", "_")[:120]
            uploaded_files.append((display_name, _save_upload(upload, udir)))
    text_inputs: list[dict[str, str]] = []
    jsonl_path: Path | None = None
    if mode == "text":
        if text_body:
            text_inputs.append({"name": "pasted_text.txt", "text": text_body})
        for display_name, path in uploaded_files:
            if path.suffix.lower() != ".txt":
                raise HTTPException(400, "text mode only accepts .txt uploads; use JSONL mode for .jsonl files")
            text_inputs.append({"name": display_name, "text": path.read_text(encoding="utf-8")})
        if not text_inputs:
            raise HTTPException(400, "text mode: provide pasted text or upload one or more .txt files")
    else:
        if uploaded_files:
            if len(uploaded_files) != 1 or uploaded_files[0][1].suffix.lower() != ".jsonl":
                raise HTTPException(400, "jsonl mode: upload exactly one .jsonl file")
            jsonl_path = uploaded_files[0][1]
        elif text_body:
            pasted_path = udir / f"pasted_{uuid.uuid4().hex}.jsonl"
            pasted_path.write_text(text_body, encoding="utf-8")
            jsonl_path = pasted_path
        else:
            raise HTTPException(400, "jsonl mode: upload one .jsonl file or paste JSONL lines into the text box")
    j = create_job(
        "jsonl" if mode == "jsonl" else "text",
        text=None if mode == "jsonl" else text_body,
        text_inputs=text_inputs,
        jsonl_path=jsonl_path,
        config_ids=cfg,
        concurrency=concurrency,
        limit=lim,
    )
    _PENDING[j.id] = j
    return {
        "id": j.id,
        "config_ids": j.config_ids,
        "mode": j.mode,
    }


@app.get("/api/jobs/{job_id}/events")
async def api_events(job_id: str) -> Any:
    j = _PENDING.get(job_id)
    if not j:
        raise HTTPException(404, "No pending job: POST /api/jobs first (each run is one-shot)")

    del _PENDING[job_id]

    async def g():
        try:
            async for ev in run_job(j):
                yield ev.sse()
        except Exception as e:  # noqa: BLE001
            yield Event("error", {"message": f"stream error: {e!s}"}).sse()

    return StreamingResponse(
        g(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str) -> Any:
    if job_id in _PENDING:
        j = _PENDING[job_id]
        return {
            "id": j.id,
            "state": "pending",
            "mode": j.mode,
            "config_ids": j.config_ids,
        }
    return JSONResponse(
        {"message": "Jobs are not stored; connect to /api/jobs/{id}/events to stream results/"},
        status_code=404,
    )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await shutdown_backend_manager()
