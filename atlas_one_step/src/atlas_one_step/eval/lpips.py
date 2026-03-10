from __future__ import annotations

"""Lightweight LPIPS-like proxy metric computed from L1 feature distance."""


def compute(l1_feature_distances: list[float]) -> float:
    if not l1_feature_distances:
        raise ValueError("l1_feature_distances must not be empty")
    return float(sum(l1_feature_distances) / len(l1_feature_distances))
