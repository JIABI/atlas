from __future__ import annotations

"""Dataset builders with explicit path validation."""

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

from .cifar10 import CIFAR10Dataset
from .imagenet64 import ImageNet64Dataset
from .lsun import LSUNDataset
from .pdebench import PDEBenchDarcyDataset
from .ucf101 import UCF101Dataset


class TensorDatasetWrapper(Dataset):
    """Wrap tensors into dict output format expected by runners."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.base[idx]
        if isinstance(item, tuple):
            x = item[0]
        else:
            x = item
        return {"x0": x.float()}


def build_dataset(name: str, root: str, resolution: int, split: str = "train") -> Dataset:
    if name == "cifar10":
        return CIFAR10Dataset(root=root, split=split, resolution=resolution)
    if name == "imagenet64":
        return ImageNet64Dataset(root=root, split=split, resolution=resolution)
    if name in {"lsun_bedroom256", "lsun_cat256"}:
        subset = "bedroom" if "bedroom" in name else "cat"
        return LSUNDataset(root=root, split=split, resolution=resolution, subset=subset)
    if name == "pdebench_darcy":
        return PDEBenchDarcyDataset(root=root, split=split, resolution=resolution)
    if name == "ucf101":
        return UCF101Dataset(root=root, split=split, resolution=resolution)
    raise ValueError(f"Unknown dataset: {name}")


def build_dataloader(batch_size: int = 8, resolution: int = 32, name: str = "cifar10", root: str = "data/cifar10") -> DataLoader:
    ds = build_dataset(name=name, root=root, resolution=resolution)
    return DataLoader(TensorDatasetWrapper(ds), batch_size=batch_size, shuffle=True, num_workers=0)
