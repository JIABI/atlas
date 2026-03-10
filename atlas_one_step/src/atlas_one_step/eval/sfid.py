from __future__ import annotations

"""sFID proxy metric using scaled FID."""

from .fid import compute as fid_compute


def compute(real_mean: float, real_var: float, fake_mean: float, fake_var: float, scale: float = 0.5) -> float:
    return float(scale * fid_compute(real_mean, real_var, fake_mean, fake_var))
