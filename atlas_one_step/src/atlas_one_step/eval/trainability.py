from __future__ import annotations

"""Training stability and convergence metrics."""

from typing import Sequence
import math


def compute(loss_curve: Sequence[float], threshold: float = 0.1) -> dict[str, float | bool]:
    if not loss_curve:
        raise ValueError("loss_curve must not be empty")
    nan = any(math.isnan(x) or math.isinf(x) for x in loss_curve)
    diverged = loss_curve[-1] > loss_curve[0] * 5.0
    converged = min(loss_curve) <= threshold
    time_to_threshold = next((i for i, v in enumerate(loss_curve) if v <= threshold), len(loss_curve))
    collapse_rate = float(sum(1 for v in loss_curve if v > loss_curve[0] * 2.0)) / len(loss_curve)
    return {
        "converged": bool(converged and not nan),
        "diverged": bool(diverged),
        "nan": bool(nan),
        "time_to_threshold": float(time_to_threshold),
        "collapse_rate": float(collapse_rate),
    }
