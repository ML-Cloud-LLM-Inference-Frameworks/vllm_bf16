from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from common.config import LOG_DIR
from collections.abc import Callable

from common.service_specs import ServiceSpec, get_orchestrator_cwd, get_service_spec, get_service_specs


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _port_is_listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _health_url_host_port(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    host = parts.hostname or "127.0.0.1"
    port = int(parts.port or (443 if parts.scheme == "https" else 80))
    return host, port


class BackendServiceManager:
    def __init__(
        self,
        startup_timeout_s: int = 600,
        health_poll_interval_s: float = 1.0,
        specs_getter: Callable[[], dict[str, ServiceSpec]] | None = None,
    ):
        self.startup_timeout_s = startup_timeout_s
        self.health_poll_interval_s = health_poll_interval_s
        self._specs_getter = specs_getter
        self._lock = asyncio.Lock()
        self._status = "idle"
        self._message = "No backend service is running."
        self._active_service_name: str | None = None
        self._desired_service_name: str | None = None
        self._started_at: str | None = None
        self._ready_at: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle = None
        self._log_path: Path | None = None
        self._switch_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None

    def _get_specs(self) -> dict[str, ServiceSpec]:
        if self._specs_getter is not None:
            return self._specs_getter()
        return get_service_specs()

    def _get_spec(self, name: str) -> ServiceSpec:
        specs = self._get_specs()
        if name not in specs:
            choices = ", ".join(sorted(specs))
            raise KeyError(f"Unknown service '{name}'. Available services: {choices}")
        return specs[name]

    def list_service_specs(self) -> list[dict]:
        payload = []
        for spec in self._get_specs().values():
            payload.append(
                {
                    "name": spec.name,
                    "label": spec.label,
                    "description": spec.description,
                    "available": spec.available,
                    "unavailable_reason": spec.unavailable_reason,
                    "config_path": spec.config_path,
                    "health_url": spec.health_url,
                    "base_url": spec.base_url,
                    "model_id": spec.model_id,
                    "server_policy": spec.server_policy,
                    "launch_command": spec.shell_command(),
                    "launch_notes": list(spec.launch_notes),
                }
            )
        return payload

    def status_snapshot(self) -> dict:
        process = self._process
        spec = self._get_specs().get(self._active_service_name) if self._active_service_name else None
        return {
            "status": self._status,
            "message": self._message,
            "active_config_name": self._active_service_name,
            "desired_config_name": self._desired_service_name,
            "pid": process.pid if process is not None and process.poll() is None else None,
            "started_at": self._started_at,
            "ready_at": self._ready_at,
            "health_url": spec.health_url if spec else None,
            "base_url": spec.base_url if spec else None,
            "model_id": spec.model_id if spec else None,
            "log_path": str(self._log_path) if self._log_path else None,
        }

    async def schedule_switch(self, service_name: str) -> dict:
        spec = self._get_spec(service_name)
        if not spec.available:
            raise ValueError(spec.unavailable_reason or f"Service '{service_name}' is not available.")
        if self._switch_task and not self._switch_task.done():
            raise RuntimeError("A backend change is already in progress.")

        async with self._lock:
            if self._switch_task and not self._switch_task.done():
                raise RuntimeError("A backend change is already in progress.")
            if self._active_service_name == service_name and self._status == "ready":
                self._message = f"{spec.label} is already active."
                return self.status_snapshot()

            self._status = "starting"
            self._message = f"Starting {spec.label}."
            self._desired_service_name = service_name
            self._ready_at = None
            self._switch_task = asyncio.create_task(self._switch_to(service_name))
            return self.status_snapshot()

    async def schedule_stop(self) -> dict:
        if self._switch_task and not self._switch_task.done():
            raise RuntimeError("A backend change is already in progress.")

        async with self._lock:
            if self._switch_task and not self._switch_task.done():
                raise RuntimeError("A backend change is already in progress.")
            if self._process is None:
                self._status = "idle"
                self._message = "No backend service is running."
                self._active_service_name = None
                self._desired_service_name = None
                self._ready_at = None
                self._started_at = None
                return self.status_snapshot()

            self._status = "stopping"
            self._message = f"Stopping {self._active_service_name}."
            self._desired_service_name = None
            self._switch_task = asyncio.create_task(self._stop_current())
            return self.status_snapshot()

    async def shutdown(self) -> None:
        task = self._switch_task
        if task and not task.done():
            await task
        async with self._lock:
            await self._terminate_process_locked()
            self._status = "idle"
            self._message = "Orchestrator shut down."
            self._active_service_name = None
            self._desired_service_name = None
            self._ready_at = None
            self._started_at = None

    def get_ready_target(self) -> ServiceSpec:
        if self._status != "ready" or self._active_service_name is None:
            raise RuntimeError("No backend is ready. Select a configuration and wait for it to become ready.")
        spec = self._get_spec(self._active_service_name)
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError("The active backend process is no longer running.")
        return spec

    async def _switch_to(self, service_name: str) -> None:
        try:
            spec = self._get_spec(service_name)
            async with self._lock:
                await self._terminate_process_locked()
                await self._start_process_locked(spec)
        except Exception as exc:
            async with self._lock:
                await self._terminate_process_locked()
                self._status = "error"
                self._message = f"Failed to start {service_name}: {exc}"
                self._active_service_name = None
                self._desired_service_name = None
                self._ready_at = None
                self._started_at = None
        finally:
            self._switch_task = None

    async def _stop_current(self) -> None:
        try:
            async with self._lock:
                await self._terminate_process_locked()
                self._status = "idle"
                self._message = "Backend service stopped."
                self._active_service_name = None
                self._desired_service_name = None
                self._ready_at = None
                self._started_at = None
        finally:
            self._switch_task = None

    async def _start_process_locked(self, spec: ServiceSpec) -> None:
        host, port = _health_url_host_port(spec.health_url)
        if _port_is_listening(host, port):
            raise RuntimeError(
                f"Refusing to launch {spec.name}: {host}:{port} is already occupied by another process. "
                "Stop any manually started backend on the VM before using the UI."
            )
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path = LOG_DIR / f"{spec.name}.log"
        self._log_handle = self._log_path.open("ab")
        header = (
            f"\n=== {_utc_now()} launching {spec.name} ===\n"
            f"cwd: {get_orchestrator_cwd()}\n"
            f"command: {spec.shell_command()}\n\n"
        )
        self._log_handle.write(header.encode("utf-8"))
        self._log_handle.flush()

        env = os.environ.copy()
        env.update(spec.env)

        popen_kwargs = {
            "cwd": get_orchestrator_cwd(),
            "env": env,
            "stdout": self._log_handle,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(spec.command, **popen_kwargs)
        self._process = process
        self._active_service_name = spec.name
        self._desired_service_name = spec.name
        self._started_at = _utc_now()
        self._ready_at = None
        self._status = "starting"
        self._message = f"Waiting for {spec.label} to pass health checks."

        await self._wait_for_health_locked(spec, process)

        self._status = "ready"
        self._message = f"{spec.label} is ready."
        self._ready_at = _utc_now()
        self._monitor_task = asyncio.create_task(self._watch_process(spec.name, process))

    async def _wait_for_health_locked(self, spec: ServiceSpec, process: subprocess.Popen[bytes]) -> None:
        deadline = asyncio.get_running_loop().time() + self.startup_timeout_s
        async with httpx.AsyncClient(timeout=2.0) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Process exited before becoming ready with code {process.returncode}.")
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(
                        f"Timed out after {self.startup_timeout_s}s waiting for {spec.health_url}."
                    )
                try:
                    response = await client.get(spec.health_url)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(self.health_poll_interval_s)

    async def _watch_process(self, service_name: str, process: subprocess.Popen[bytes]) -> None:
        return_code = await asyncio.to_thread(process.wait)
        async with self._lock:
            if self._process is process and self._active_service_name == service_name:
                await self._close_log_locked()
                self._process = None
                self._active_service_name = None
                self._desired_service_name = None
                self._ready_at = None
                self._started_at = None
                self._status = "error"
                self._message = f"{service_name} exited unexpectedly with code {return_code}."

    async def _terminate_process_locked(self) -> None:
        monitor_task = self._monitor_task
        self._monitor_task = None
        if monitor_task and not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            if os.name == "nt":
                process.terminate()
                try:
                    await asyncio.to_thread(process.wait, 10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait)
            else:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.to_thread(process.wait, 10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    await asyncio.to_thread(process.wait)

        await self._close_log_locked()
        self._active_service_name = None

    async def _close_log_locked(self) -> None:
        if self._log_handle is not None:
            self._log_handle.flush()
            self._log_handle.close()
            self._log_handle = None
