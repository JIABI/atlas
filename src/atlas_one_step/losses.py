from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    pred_weight: float = 1.0
    sem_weight: float = 0.5
    stab_weight: float = 1e-4


def prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def semantic_loss(mapped_pred: torch.Tensor, loss_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(mapped_pred, loss_target)


def stability_loss(phi_map, pred: torch.Tensor) -> torch.Tensor:
    reg = phi_map.regularization_loss() if hasattr(phi_map, 'regularization_loss') else torch.tensor(0.0, device=pred.device)
    smooth = 0.0
    if pred.ndim == 4 and pred.shape[-1] > 2 and pred.shape[-2] > 2:
        smooth = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).pow(2).mean() + (pred[:, :, :, 1:] - pred[:, :, :, :-1]).pow(2).mean()
    return reg + 1e-5 * smooth
