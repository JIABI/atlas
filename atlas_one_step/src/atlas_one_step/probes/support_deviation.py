from __future__ import annotations

import torch


def _pairwise_l2(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    d = torch.cdist(flat, flat)
    return d


def compute_support_deviation(x: torch.Tensor) -> dict[str, float]:
    d = _pairwise_l2(x)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values
    pix = float(nn.mean().item())
    # perceptual/ssl approximations from pooled projections
    pooled = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2).flatten(1)
    d2 = torch.cdist(pooled, pooled)
    d2.fill_diagonal_(float("inf"))
    perc = float(d2.min(dim=1).values.mean().item())
    ssl = float((nn / (d2.min(dim=1).values + 1e-6)).mean().item())
    return {"support_pix": pix, "support_perc": perc, "support_ssl": ssl}
