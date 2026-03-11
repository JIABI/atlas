from __future__ import annotations

import torch


def jacobian_proxy(x: torch.Tensor) -> dict[str, float]:
    flat = x.flatten(1)
    cov = torch.cov(flat.T)
    eig = torch.linalg.eigvalsh(cov + 1e-6 * torch.eye(cov.shape[0]))
    max_e = float(eig.max().item())
    min_e = float(eig.min().item())
    anis = max_e / (min_e + 1e-8)
    stable_rank = float(eig.sum().item() / (max_e + 1e-8))
    return {"jacobian_norm": max_e**0.5, "anisotropy": anis, "stable_rank": stable_rank}
