from __future__ import annotations

"""Base model utilities."""

import torch.nn as nn


class BaseModel(nn.Module):
    """Common interface for denoising/one-step predictors."""

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
