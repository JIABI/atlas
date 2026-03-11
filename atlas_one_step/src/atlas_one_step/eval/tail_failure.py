from __future__ import annotations

"""Tail-failure metrics based on per-sample error scores."""

from typing import Sequence


def _percentile(sorted_vals: list[float], q: float) -> float:
    idx = int((len(sorted_vals) - 1) * q)
    return float(sorted_vals[idx])


def compute(sample_errors: Sequence[float], worst_k: int = 10) -> dict[str, float]:
    if not sample_errors:
        raise ValueError("sample_errors must not be empty")
    vals = sorted(float(v) for v in sample_errors)
    k = min(worst_k, len(vals))
    worst = vals[-k:]
    return {
        "worst_k_score": float(sum(worst) / k),
        "percentile_95": _percentile(vals, 0.95),
        "percentile_99": _percentile(vals, 0.99),
        "rare_failure_rate": float(sum(1 for v in vals if v >= _percentile(vals, 0.99)) / len(vals)),
    }
