from __future__ import annotations

from pathlib import Path
from torchvision.datasets import ImageFolder
from .transforms import image_transform


class LSUNDataset(ImageFolder):
    """LSUN dataset wrapper requiring manual extraction into folder trees."""

    def __init__(self, root: str, split: str, resolution: int, subset: str) -> None:
        data_root = Path(root) / subset / split
        if not data_root.exists():
            raise FileNotFoundError(f"LSUN path not found: {data_root}. Run tools/dataset_prep/prepare_lsun.py")
        super().__init__(str(data_root), transform=image_transform(resolution))
