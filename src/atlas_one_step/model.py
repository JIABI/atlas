from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.proj(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = F.silu(self.conv2(h))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base_channels: int = 64, time_dim: int = 64) -> None:
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(in_ch, base_channels, 3, padding=1)
        self.block1 = ResBlock(base_channels, base_channels, time_dim)
        self.block2 = ResBlock(base_channels, base_channels, time_dim)
        self.block3 = ResBlock(base_channels, base_channels, time_dim)
        self.out = nn.Conv2d(base_channels, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time(t)
        h = self.in_conv(x)
        h = self.block1(h, t_emb)
        h = self.block2(h, t_emb)
        h = self.block3(h, t_emb)
        return self.out(h)


class IdentityPhi(nn.Module):
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return z

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(iter(self.parameters()), torch.tensor(0.0)).device)


class AffinePhi(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.scale * z + self.bias

    def regularization_loss(self) -> torch.Tensor:
        return 1e-4 * (self.scale.pow(2).mean() + self.bias.pow(2).mean())


class ShallowPhi(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, 1),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return z + 0.1 * self.net(z)

    def regularization_loss(self) -> torch.Tensor:
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.parameters():
            reg = reg + p.pow(2).mean()
        return 1e-4 * reg


def build_model(cfg: dict, channels: int) -> nn.Module:
    name = cfg.get('name', 'tiny_unet')
    if name != 'tiny_unet':
        raise ValueError(f'Unsupported model: {name}')
    return TinyUNet(channels, channels, base_channels=int(cfg.get('base_channels', 64)), time_dim=int(cfg.get('time_dim', 64)))


def build_phi_map(cfg: dict, channels: int) -> nn.Module:
    typ = cfg.get('type', 'identity')
    if typ == 'identity':
        return IdentityPhi()
    if typ == 'affine':
        return AffinePhi(channels)
    if typ == 'shallow':
        return ShallowPhi(channels, hidden_channels=int(cfg.get('hidden_channels', 16)))
    raise ValueError(f'Unsupported phi_map type: {typ}')
