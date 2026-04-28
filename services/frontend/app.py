"""FastAPI: static UI, create job, single-shot SSE (no on-disk job artifacts)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from services.frontend.orchestrator import Event, Job, create_job, run_job

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


@app.post("/api/jobs")
async def api_create_job(
    mode: str = Form("text"),
    text: str = Form(""),
    configs: str = Form("[]"),
    concurrency: int = Form(4),
    limit: int = Form(0),
    file: UploadFile | None = File(None),
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
    upload: Path | None = None
    if file and file.filename:
        dest = udir / f"u_{file.filename.replace('/', '_')[:120]}"
        with dest.open("wb") as outw:
            shutil.copyfileobj(file.file, outw)
        upload = dest
    j = create_job(
        "jsonl" if mode == "jsonl" else "text",
        text=None if mode == "jsonl" else text_body,
        jsonl_path=upload if mode == "jsonl" else None,
        config_ids=cfg,
        concurrency=concurrency,
        limit=lim,
    )
    if mode == "text" and (not (j.text and str(j.text).strip())) and upload and upload.is_file():
        if not str(upload).lower().endswith((".jsonl", ".json")):
            j.text = upload.read_text(encoding="utf-8")
    if mode == "text" and not (j.text and str(j.text).strip()):
        raise HTTPException(400, "text mode: provide the `text` form field or upload a plain-text file (not .jsonl)")
    if mode == "jsonl" and (not j.jsonl_path or not Path(j.jsonl_path).is_file()):
        raise HTTPException(400, "jsonl mode: upload a .jsonl file (e.g. data/agnews_bench_1000.jsonl)")
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
