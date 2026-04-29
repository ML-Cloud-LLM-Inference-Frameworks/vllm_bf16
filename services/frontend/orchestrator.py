"""Sequential multi-config UI runner backed by the persistent backend manager."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Sequence

from services.frontend import runner
from services.frontend.configs import FRONTEND_CONFIG_ORDER, FrontendConfig, get_frontend_configs
from services.orchestrator import BackendServiceManager

Mode = Literal["text", "jsonl"]
_BACKEND_MANAGER = BackendServiceManager()


class Phase(str, Enum):
    launching = "launching"
    ready_wait = "ready_wait"
    running = "running"
    done = "done"
    skipped = "skipped"


@dataclass
class Job:
    id: str
    mode: Mode
    text: str | None
    text_inputs: list[dict[str, str]]
    jsonl_path: Path | None
    concurrency: int
    limit: int | None
    config_ids: list[str]
    cancel_requested: bool = False
    results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class Event:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def sse(self) -> str:
        return f"data: {json.dumps({'type': self.type, 'payload': self.payload}, ensure_ascii=False)}\n\n"


def create_job(
    mode: Mode,
    text: str | None,
    text_inputs: Sequence[dict[str, str]] | None,
    jsonl_path: Path | None,
    config_ids: Sequence[str] | None,
    concurrency: int = 4,
    limit: int | None = None,
) -> Job:
    if not config_ids:
        cids = list(FRONTEND_CONFIG_ORDER)
    else:
        cids = [str(x) for x in config_ids]
    return Job(
        id=str(uuid.uuid4()),
        mode=mode,
        text=text,
        text_inputs=[{"name": str(item["name"]), "text": str(item["text"])} for item in (text_inputs or [])],
        jsonl_path=jsonl_path,
        concurrency=max(1, int(concurrency)),
        limit=limit,
        config_ids=cids,
    )


def cancel_job(j: Job) -> None:
    j.cancel_requested = True


def _get_cfg(name: str) -> FrontendConfig:
    c = get_frontend_configs()[name]
    if not c.available:
        raise KeyError(f"{c.name} unavailable: {c.unavailable_reason or 'n/a'}")
    return c


def get_backend_manager() -> BackendServiceManager:
    return _BACKEND_MANAGER


def backend_status_snapshot() -> dict[str, Any]:
    return _BACKEND_MANAGER.status_snapshot()


async def shutdown_backend_manager() -> None:
    await _BACKEND_MANAGER.shutdown()


def _log_tail(log_path: str | None, max_lines: int = 80) -> str:
    if not log_path:
        return ""
    p = Path(log_path)
    if not p.is_file():
        return ""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


async def _ensure_backend_ready(name: str, cfg: FrontendConfig) -> dict[str, Any]:
    status = await _BACKEND_MANAGER.schedule_switch(name)
    if status.get("status") == "ready" and status.get("active_config_name") == name:
        return status
    deadline = time.perf_counter() + 1200.0
    while time.perf_counter() < deadline:
        status = backend_status_snapshot()
        if status.get("status") == "ready" and status.get("active_config_name") == name:
            return status
        if status.get("status") == "error":
            raise RuntimeError(str(status.get("message") or f"{cfg.label} failed to start"))
        await asyncio.sleep(1.0)
    tail = _log_tail(status.get("log_path"), 200)
    raise TimeoutError(f"Timed out waiting for {cfg.label} to become ready.\n---\n{tail}")


async def run_job(job: Job) -> AsyncIterator[Event]:
    with tempfile.TemporaryDirectory(prefix="vllm4cmp_") as td:
        d = Path(td)
        jmeta: dict[str, Any] = {
            "id": job.id,
            "mode": job.mode,
            "concurrency": job.concurrency,
            "limit": job.limit,
            "config_ids": job.config_ids,
        }
        yield Event("job", {"id": job.id, "state": "running", **jmeta, "ephemeral": True, "work_dir": "memory"})
        yield Event("log", {"message": f"Temp work dir (deleted when run ends): {d}", "stage": "init"})

        input_path: Path | None = None
        if job.mode == "jsonl":
            if job.jsonl_path is None or not job.jsonl_path.is_file():
                raise FileNotFoundError("JSONL mode requires a .jsonl file on disk")
            dest = d / "input.jsonl"
            if dest.resolve() != job.jsonl_path.resolve():
                shutil.copy2(job.jsonl_path, dest)
            input_path = dest
        else:
            inputs_dir = d / "inputs"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            if not job.text_inputs and job.text and str(job.text).strip():
                job.text_inputs = [{"name": "pasted_text.txt", "text": job.text}]
            if not job.text_inputs:
                raise ValueError("text mode: provide article text or upload one or more .txt files")
            for idx, item in enumerate(job.text_inputs):
                clean_name = item["name"].replace("/", "_").replace("\\", "_")
                safe_name = f"{idx:03d}_{clean_name}"
                (inputs_dir / safe_name).write_text(item["text"], encoding="utf-8")

        for name in job.config_ids:
            if job.cancel_requested:
                job.errors[name] = "cancelled"
                yield Event("config", {"id": name, "phase": Phase.skipped.value})
                break
            try:
                cfg = _get_cfg(name)
            except (KeyError, OSError) as exc:  # noqa: BLE001
                job.errors[name] = str(exc)
                yield Event("error", {"id": name, "message": str(exc)})
                continue
            cdir = d / cfg.name
            cdir.mkdir(parents=True, exist_ok=True)
            try:
                pre_status = backend_status_snapshot()
                reused = pre_status.get("status") == "ready" and pre_status.get("active_config_name") == name
                if reused:
                    yield Event(
                        "log",
                        {
                            "message": f"[{name}] reusing the already-running backend on the VM.",
                            "config": name,
                            "label": cfg.label,
                            "stage": "reuse",
                        },
                    )
                else:
                    yield Event(
                        "log",
                        {
                            "message": f"-- [{cfg.label}] ({name}) - starting backend on the VM",
                            "config": name,
                            "stage": "start",
                        },
                    )
                    yield Event("config", {"id": name, "phase": Phase.launching.value, "label": cfg.label})
                yield Event("config", {"id": name, "phase": Phase.ready_wait.value, "label": cfg.label})
                if not reused:
                    ready_task = asyncio.create_task(_ensure_backend_ready(name, cfg))
                    t_ready0 = time.perf_counter()
                    while not ready_task.done():
                        await asyncio.wait({ready_task}, timeout=1.2)
                        if ready_task.done():
                            break
                        status = backend_status_snapshot()
                        elapsed = time.perf_counter() - t_ready0
                        yield Event(
                            "log",
                            {
                                "message": f"[{name}] waiting for backend readiness ({elapsed:.0f}s) - {status.get('message', 'starting')}",
                                "config": name,
                                "label": cfg.label,
                                "stage": "ready_wait",
                                "elapsed_s": round(elapsed, 1),
                                "log_tail": _log_tail(status.get("log_path"), 80),
                            },
                        )
                    ready_status = await ready_task
                else:
                    ready_status = backend_status_snapshot()
                openai_base = str(ready_status.get("base_url") or "http://127.0.0.1:8000/v1")
                yield Event(
                    "log",
                    {"message": f"[{name}] healthy - running inference...", "config": name, "label": cfg.label, "stage": "healthy"},
                )
                await asyncio.sleep(0.05)
                yield Event(
                    "config",
                    {
                        "id": name,
                        "phase": Phase.running.value,
                        "log_tail": _log_tail(ready_status.get("log_path"), 20),
                        "label": cfg.label,
                    },
                )
                if job.mode == "jsonl" and input_path and input_path.suffix == ".jsonl":
                    yield Event(
                        "log",
                        {
                            "message": f"[{name}] JSONL benchmark (can take a while)... concurrency {job.concurrency} warmup=10",
                            "config": name,
                            "stage": "run_jsonl",
                        },
                    )
                    bench_task = asyncio.create_task(
                        runner.run_jsonl_bench(
                            cfg,
                            openai_base,
                            input_path,
                            cdir,
                            concurrency=job.concurrency,
                            warmup=10,
                            limit=job.limit,
                        )
                    )
                    t_run = time.perf_counter()
                    while not bench_task.done():
                        await asyncio.wait({bench_task}, timeout=5.0)
                        if bench_task.done():
                            break
                        yield Event(
                            "log",
                            {
                                "message": f"[{name}] benchmark still running ({time.perf_counter() - t_run:.0f}s)...",
                                "config": name,
                                "label": cfg.label,
                                "stage": "jsonl_bench",
                                "elapsed_s": round(time.perf_counter() - t_run, 1),
                                "log_tail": _log_tail(ready_status.get("log_path"), 40),
                            },
                        )
                    res = await bench_task
                else:
                    yield Event(
                        "log",
                        {
                            "message": f"[{name}] running {len(job.text_inputs)} text input(s) with concurrency {job.concurrency}...",
                            "config": name,
                            "stage": "run_text",
                        },
                    )
                    res = await runner.run_text_batch(
                        cfg,
                        openai_base,
                        job.text_inputs,
                        cdir,
                        concurrency=job.concurrency,
                    )
                job.results[name] = res
                yield Event("config", {"id": name, "phase": Phase.done.value, "summary": res, "label": cfg.label})
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc!s}"
                status = backend_status_snapshot()
                tail = _log_tail(status.get("log_path"), 200)
                if tail:
                    err = f"{err}\n---\n{tail}"
                job.errors[name] = err
                yield Event("error", {"id": name, "message": err})
            await asyncio.sleep(0.15)
            yield Event(
                "log",
                {
                    "message": f"-- done [{name}], backend remains available for reuse on :8000",
                    "config": name,
                    "stage": "teardown",
                },
            )

        yield Event(
            "job",
            {
                "id": job.id,
                "state": "finished",
                "mode": job.mode,
                "ephemeral": True,
                "errors": job.errors,
                "config_ids": job.config_ids,
                "results": job.results,
            },
        )
