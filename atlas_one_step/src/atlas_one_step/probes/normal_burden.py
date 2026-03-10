from __future__ import annotations

import torch


def compute_normal_burden(x: torch.Tensor) -> dict[str, float]:
    flat = x.flatten(1)
    centered = flat - flat.mean(0, keepdim=True)
    _, s, _ = torch.pca_lowrank(centered, q=min(8, centered.shape[0] - 1))
    energy = s**2
    tan = float((energy[: max(1, len(energy)//2)].sum() / (energy.sum() + 1e-8)).item())
    nor = float(1.0 - tan)
    return {"rho_tan": tan, "rho_nor": nor}
