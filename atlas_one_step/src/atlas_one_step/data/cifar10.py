from __future__ import annotations

from pathlib import Path
from torchvision.datasets import CIFAR10
from .transforms import image_transform


class CIFAR10Dataset(CIFAR10):
    """CIFAR-10 dataset wrapper with explicit root and split."""

    def __init__(self, root: str, split: str, resolution: int) -> None:
        Path(root).mkdir(parents=True, exist_ok=True)
        super().__init__(root=root, train=(split == "train"), transform=image_transform(resolution), download=True)
