"""Background nvidia-smi repeated CSV output (one row per query interval) into a file."""

from __future__ import annotations

import contextlib
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Generator

# nvidia-smi -q  names; [MiB] / [%] appear as column names in csv output
_DEFAULT_QUERY = (
    "timestamp,"
    "name,memory.total [MiB],memory.used [MiB],memory.free [MiB],"
    "utilization.gpu [%],utilization.memory [%]"
)

try:
    _SIG = signal.Signals.SIGTERM
except Exception:  # noqa: BLE001
    _SIG = signal.SIGTERM


@contextlib.contextmanager
def nvidia_smi_log_csv(
    out_path: str | Path,
    interval_s: float = 1.0,
    query: str = _DEFAULT_QUERY,
    executable: str = "nvidia-smi",
) -> Generator[Path, None, None]:
    """
    Spawns: nvidia-smi -l <s> in loop mode, writing CSV to ``out_path``, while
    the context body runs. Terminates the child on exit.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ex = shutil.which(executable) or executable
    t_int = max(1, int(round(float(interval_s)))) if interval_s >= 1.0 else 1
    cmd: list[str] = [
        ex,
        f"--query-gpu={query}",
        "--format=csv",
        "-l",
        str(t_int),
    ]
    f = open(path, "w", encoding="utf-8")
    p: subprocess.Popen[bytes] | None = None
    try:
        p = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        f.close()
        if path.exists() and path.stat().st_size == 0:
            try:
                path.unlink()
            except OSError:
                pass
        raise OSError("failed to start nvidia-smi") from e
    try:
        yield path
    finally:
        if p is not None and p.poll() is None:
            p.send_signal(_SIG)
            try:
                p.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                p.kill()
            except Exception:  # noqa: BLE001
                try:
                    p.kill()
                except OSError:
                    pass
        f.close()
