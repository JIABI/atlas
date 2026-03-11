from __future__ import annotations

from pathlib import Path
from torchvision.datasets import ImageFolder
from .transforms import image_transform


class ImageNet64Dataset(ImageFolder):
    """ImageNet-64 wrapper expecting manually prepared ImageFolder layout."""

    def __init__(self, root: str, split: str, resolution: int) -> None:
        split_root = Path(root) / split
        if not split_root.exists():
            raise FileNotFoundError(
                f"ImageNet64 split path missing: {split_root}. Prepare data with tools/dataset_prep/prepare_imagenet64.py"
            )
        super().__init__(str(split_root), transform=image_transform(resolution))
