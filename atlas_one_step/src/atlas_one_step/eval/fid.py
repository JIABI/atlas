from __future__ import annotations

"""FID proxy from mean and variance statistics."""


def compute(real_mean: float, real_var: float, fake_mean: float, fake_var: float) -> float:
    if real_var < 0 or fake_var < 0:
        raise ValueError("variances must be non-negative")
    mean_term = (real_mean - fake_mean) ** 2
    var_term = real_var + fake_var - 2.0 * (real_var * fake_var) ** 0.5
    return float(mean_term + var_term)
