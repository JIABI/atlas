from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import Dataset


class PDEBenchDarcyDataset(Dataset):
    """PDEBench Darcy pilot dataset loading tensor files saved as .pt."""

    def __init__(self, root: str, split: str, resolution: int) -> None:
        split_root = Path(root) / split
        if not split_root.exists():
            raise FileNotFoundError(f"PDEBench split missing at {split_root}. Run tools/dataset_prep/prepare_pdebench.py")
        self.files = sorted(split_root.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt PDEBench files in {split_root}")
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        x = torch.load(self.files[idx])
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x
