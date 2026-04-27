"""Helpers for summarizing repeated nvidia-smi CSV logs."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from statistics import mean

_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")


def _parse_number(value: str) -> float | None:
    match = _NUM_RE.search((value or "").strip())
    if not match:
        return None
    return float(match.group(0))


def summarize_nvidia_smi_csv(path: str | Path) -> dict:
    rows = list(csv.DictReader(Path(path).open("r", encoding="utf-8", newline="")))
    if not rows:
        return {"samples": 0}

    memory_used = []
    gpu_util = []
    memory_util = []
    timestamps = []

    for row in rows:
        ts = (row.get("timestamp") or "").strip()
        if ts:
            timestamps.append(ts)
        for bucket, target in (
            ("memory.used [MiB]", memory_used),
            ("utilization.gpu [%]", gpu_util),
            ("utilization.memory [%]", memory_util),
        ):
            parsed = _parse_number(row.get(bucket, ""))
            if parsed is not None:
                target.append(parsed)

    summary = {
        "samples": len(rows),
        "timestamp_first": timestamps[0] if timestamps else None,
        "timestamp_last": timestamps[-1] if timestamps else None,
    }
    if memory_used:
        summary["memory_used_mib_avg"] = mean(memory_used)
        summary["memory_used_mib_max"] = max(memory_used)
    if gpu_util:
        summary["gpu_util_percent_avg"] = mean(gpu_util)
        summary["gpu_util_percent_max"] = max(gpu_util)
    if memory_util:
        summary["memory_util_percent_avg"] = mean(memory_util)
        summary["memory_util_percent_max"] = max(memory_util)
    return summary
