from __future__ import annotations

"""Pilot evaluation composition for small-scale regimes."""

from .trainability import compute as trainability
from .tail_failure import compute as tail


def compute(loss_curve: list[float], errors: list[float]) -> dict:
    return {"trainability": trainability(loss_curve), "tail": tail(errors)}
