"""Sequential multi-config job runner with an async event iterator (in-memory; no job dir on disk)."""

from __future__ import annotations
import json
import asyncio
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Sequence

import httpx

from services.frontend import runner, server_lifecycle
from services.frontend.configs import FRONTEND_CONFIG_ORDER, FrontendConfig, get_frontend_configs
from services.frontend.server_lifecycle import DEFAULT_HEALTH_URL

Mode = Literal["text", "jsonl"]


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
    jsonl_path: Path | None,
    config_ids: Sequence[str] | None,
    concurrency: int = 4,
    limit: int | None = None,
) -> Job:
    cids: list[str]
    if not config_ids:
        cids = list(FRONTEND_CONFIG_ORDER)
    else:
        cids = [str(x) for x in config_ids]
    return Job(
        id=str(uuid.uuid4()),
        mode=mode,
        text=text,
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
            oneline = d / "input_oneline.txt"
            if job.text and str(job.text).strip():
                oneline.write_text(job.text, encoding="utf-8")
            if oneline.exists() and (not job.text or not str(job.text).strip()):
                job.text = oneline.read_text(encoding="utf-8")
            if not (job.text and str(job.text).strip()):
                raise ValueError("text mode: provide article text (app should set job.text or write input_oneline.txt)")
            oneline.write_text(job.text, encoding="utf-8")
            input_path = oneline

        openai_base = "http://127.0.0.1:8000/v1"
        for name in job.config_ids:
            if job.cancel_requested:
                job.errors[name] = "cancelled"
                yield Event("config", {"id": name, "phase": Phase.skipped.value})
                break
            try:
                cfg = _get_cfg(name)
            except (KeyError, OSError) as e:  # noqa: BLE001
                job.errors[name] = str(e)
                yield Event("error", {"id": name, "message": str(e)})
                continue
            cdir = d / cfg.name
            cdir.mkdir(parents=True, exist_ok=True)
            sp: server_lifecycle.LaunchedServer | None = None
            try:
                server_lifecycle.assert_port_free()
                yield Event("log", {"message": f"── [{cfg.label}] ({name}) — starting subprocess", "config": name, "stage": "start"})
                yield Event("config", {"id": name, "phase": Phase.launching.value, "label": cfg.label})
                sp = server_lifecycle.launch(cfg)
                yield Event(
                    "config",
                    {"id": name, "phase": Phase.ready_wait.value, "log_tail": sp.get_log_tail(40), "label": cfg.label},
                )
                t_ready0 = time.perf_counter()
                while time.perf_counter() - t_ready0 < 1200.0:
                    el = time.perf_counter() - t_ready0
                    yield Event(
                        "log",
                        {
                            "message": f"[{name}] waiting for /health ({el:.0f}s) — subprocess log (tail):",
                            "config": name,
                            "label": cfg.label,
                            "stage": "ready_wait",
                            "elapsed_s": round(el, 1),
                            "log_tail": sp.get_log_tail(100),
                        },
                    )
                    try:
                        async with httpx.AsyncClient() as ac:
                            r = await ac.get(DEFAULT_HEALTH_URL, timeout=4.0)
                    except (httpx.RequestError, OSError):  # noqa: BLE001
                        await asyncio.sleep(1.2)
                        continue
                    if r is not None and r.status_code < 500:
                        break
                    await asyncio.sleep(1.2)
                else:
                    tail = sp.get_log_tail(200)
                    err = f"ready timeout 1200s\n---\n{tail}"
                    job.errors[name] = err
                    yield Event("error", {"id": name, "message": "ready timeout", "detail": err})
                    if sp:
                        server_lifecycle.shutdown(sp)
                    sp = None
                if sp is None:
                    continue
                yield Event(
                    "log",
                    {"message": f"[{name}] healthy — running inference…", "config": name, "label": cfg.label, "stage": "healthy"},
                )
                await asyncio.sleep(0.05)
                yield Event(
                    "config",
                    {"id": name, "phase": Phase.running.value, "log_tail": sp.get_log_tail(20), "label": cfg.label},
                )
                if job.mode == "jsonl" and input_path and input_path.suffix == ".jsonl":
                    yield Event(
                        "log",
                        {
                            "message": f"[{name}] JSONL benchmark (can take a while)… concurrency {job.concurrency} warmup=10",
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
                                "message": f"[{name}] benchmark still running ({time.perf_counter() - t_run:.0f}s)…",
                                "config": name,
                                "label": cfg.label,
                                "stage": "jsonl_bench",
                                "elapsed_s": round(time.perf_counter() - t_run, 1),
                                "log_tail": sp.get_log_tail(40) if sp else "",
                            },
                        )
                    res = await bench_task
                else:
                    tw = d / "input_oneline.txt"
                    text = job.text or (tw.read_text(encoding="utf-8") if tw.exists() else "")
                    if not (text and text.strip()) and (input_path and (not str(input_path).endswith((".jsonl",)))) and input_path.is_file():
                        text = input_path.read_text(encoding="utf-8")
                    if not (text and text.strip()):
                        raise ValueError("no input text: paste text, upload a .txt, or .jsonl for article mode")
                    yield Event("log", {"message": f"[{name}] single article — streaming request…", "config": name, "stage": "run_text"})
                    res = await runner.run_single_text_bench(cfg, openai_base, text, cdir)
                job.results[name] = res
                yield Event("config", {"id": name, "phase": Phase.done.value, "summary": res, "label": cfg.label})
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e!s}"
                if sp:
                    err = f"{err}\n---\n{sp.get_log_tail(200)}"
                job.errors[name] = err
                yield Event("error", {"id": name, "message": err})
            finally:
                if sp is not None:
                    server_lifecycle.shutdown(sp)
            await asyncio.sleep(0.15)
            yield Event("log", {"message": f"── done [{name}], closed subprocess on :8000", "config": name, "stage": "teardown"})

        yield Event(
            "job",
            {
                "id": job.id,
                "state": "finished",
                "ephemeral": True,
                "errors": job.errors,
                "config_ids": job.config_ids,
                "results": job.results,
            },
        )
