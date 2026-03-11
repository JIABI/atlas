from __future__ import annotations

import torch


def early_pathology(x: torch.Tensor | None = None) -> dict[str, float]:
    if x is None:
        return {"grad_var": 0.0, "sharpness": 0.0}
    flat = x.flatten(1)
    grad_var = float(flat.var(dim=1).mean().item())
    sharpness = float((flat[:, 1:] - flat[:, :-1]).abs().mean().item())
    return {"grad_var": grad_var, "sharpness": sharpness}
