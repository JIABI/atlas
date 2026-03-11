from __future__ import annotations

"""Mode-collapse proxy metrics."""

from typing import Sequence
import statistics


def compute(sample_scores: Sequence[float]) -> dict[str, float]:
    if not sample_scores:
        raise ValueError("sample_scores must not be empty")
    mean = statistics.fmean(sample_scores)
    std = statistics.pstdev(sample_scores)
    low_diversity = sum(1 for x in sample_scores if x < mean - std) / len(sample_scores)
    return {"collapse_proxy": float(low_diversity), "score_std": float(std)}
