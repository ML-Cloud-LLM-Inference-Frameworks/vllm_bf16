"""Delta metrics from two Prometheus /metrics text scrapes (before vs after a run)."""

from __future__ import annotations

from common.prometheus_utils import parse_prometheus_samples, summarize_vllm_samples


def _numeric_delta(before: dict[str, float | int], after: dict[str, float | int]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for k, av in after.items():
        if not isinstance(av, (int, float)):
            continue
        bv = before.get(k, 0) if k in before else 0.0
        if not isinstance(bv, (int, float)):
            continue
        d = float(av) - float(bv)
        if d != 0.0:
            out[k] = d
    return out


def prom_delta_from_text(before_text: str, after_text: str) -> dict:
    b = parse_prometheus_samples(before_text)
    a = parse_prometheus_samples(after_text)
    d = _numeric_delta(b, a)
    derived = summarize_vllm_samples(d) if d else {}
    return {
        "derived_from_delta": derived,
    }
