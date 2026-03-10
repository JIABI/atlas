from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import Dataset


class UCF101Dataset(Dataset):
    """UCF101 pilot frame-stack dataset from pre-extracted tensors."""

    def __init__(self, root: str, split: str, resolution: int) -> None:
        split_root = Path(root) / split
        if not split_root.exists():
            raise FileNotFoundError(f"UCF101 split missing at {split_root}. Run tools/dataset_prep/prepare_ucf101.py")
        self.files = sorted(split_root.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt clips found in {split_root}")
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        clip = torch.load(self.files[idx])
        return clip
