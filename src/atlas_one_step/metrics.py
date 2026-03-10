from __future__ import annotations

import numpy as np
import torch


def mse_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).flatten(1).mean(dim=1)


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-8))


def feature_fd(real: torch.Tensor, fake: torch.Tensor) -> float:
    # Lightweight Fréchet distance proxy on average-pooled features.
    real_f = torch.nn.functional.adaptive_avg_pool2d(real, (8, 8)).flatten(1).detach().cpu().numpy()
    fake_f = torch.nn.functional.adaptive_avg_pool2d(fake, (8, 8)).flatten(1).detach().cpu().numpy()
    mu_r, mu_f = real_f.mean(0), fake_f.mean(0)
    cov_r = np.cov(real_f, rowvar=False)
    cov_f = np.cov(fake_f, rowvar=False)
    diff = mu_r - mu_f
    # Numerically stable trace sqrt approximation via eigen decomposition.
    prod = cov_r @ cov_f
    eigvals = np.linalg.eigvals(prod)
    sqrt_trace = np.sum(np.sqrt(np.clip(np.real(eigvals), 0, None)))
    return float(diff @ diff + np.trace(cov_r) + np.trace(cov_f) - 2 * sqrt_trace)


def summarize_tail(errors: torch.Tensor, collapse_threshold: float, percentiles: list[int]) -> dict[str, float]:
    err_np = errors.detach().cpu().numpy()
    out = {
        'worst_k_score': float(np.sort(err_np)[-max(1, len(err_np)//10):].mean()),
        'rare_failure_rate': float((err_np > collapse_threshold).mean()),
    }
    for p in percentiles:
        out[f'percentile_{p}'] = float(np.percentile(err_np, p))
    return out
