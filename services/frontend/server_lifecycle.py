"""Start model servers as subprocesses, wait for /health, capture logs, terminate."""

from __future__ import annotations

import asyncio
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import httpx

from common.service_specs import get_orchestrator_cwd
from services.frontend.configs import FrontendConfig, merge_environ

BACKEND_PORT = 8000
HEALTH_PATH = "/health"
DEFAULT_HEALTH_URL = f"http://127.0.0.1:{BACKEND_PORT}{HEALTH_PATH}"


@dataclass
class LaunchedServer:
    """Holds a running backend subprocess and a bounded log line buffer."""

    process: Optional[subprocess.Popen[bytes] | subprocess.Popen[str]] = None
    log_lines: Deque[str] = field(default_factory=lambda: deque(maxlen=8000))
    _drain: Optional[threading.Thread] = None

    def get_log_tail(self, n: int = 80) -> str:
        lines = list(self.log_lines)
        return "\n".join(lines[-n:]) if lines else ""


def assert_port_free(port: int = BACKEND_PORT, host: str = "127.0.0.1") -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    err = s.connect_ex((host, port))
    s.close()
    if err == 0:
        raise RuntimeError(
            f"Port {port} is already in use. Stop the other process using this port before running a comparison job."
        )


def _start_log_thread(proc: subprocess.Popen[str], lines: Deque[str]) -> threading.Thread:
    if proc.stdout is None:
        raise RuntimeError("Popen was started without stdout=PIPE")
    assert proc.stdout is not None

    def _run() -> None:
        for raw in iter(proc.stdout.readline, ""):
            if raw == "":
                break
            lines.append(raw.rstrip())

    t = threading.Thread(target=_run, name="log-drain", daemon=True)
    t.start()
    return t


def launch(config: FrontendConfig) -> LaunchedServer:
    assert_port_free()
    ctx = get_orchestrator_cwd()
    env = merge_environ(config.env)
    proc = subprocess.Popen(
        list(config.command),
        cwd=ctx,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    ls = LaunchedServer(process=proc, log_lines=deque(maxlen=8000))
    ls._drain = _start_log_thread(proc, ls.log_lines)
    return ls


def shutdown(ls: LaunchedServer, wait_s: float = 5.0) -> None:
    p = ls.process
    if p is None or p.poll() is not None:
        ls.process = None
        return
    p.terminate()
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        if p.poll() is not None:
            break
        time.sleep(0.1)
    if p.poll() is None:
        p.kill()
    ls.process = None


def health_url(override: str | None = None) -> str:
    if override and override.strip():
        return override
    return DEFAULT_HEALTH_URL


async def wait_for_health(
    url: str = DEFAULT_HEALTH_URL,
    total_timeout_s: float = 900.0,
    interval_s: float = 1.0,
) -> None:
    u = url.rstrip("/")
    tdead = time.monotonic() + total_timeout_s
    while time.monotonic() < tdead:
        try:
            async with httpx.AsyncClient() as ac:
                r = await ac.get(u, timeout=5.0)
        except (httpx.RequestError, OSError):
            r = None
        else:
            if r is not None and r.status_code < 500:
                return
        await asyncio.sleep(interval_s)
    raise TimeoutError(f"Server did not become healthy at {u} within {total_timeout_s:.0f}s")
