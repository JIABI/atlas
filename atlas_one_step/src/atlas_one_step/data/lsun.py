
from __future__ import annotations
from torch.utils.data import Dataset
import torch

class RandomImageDataset(Dataset):
    def __init__(self, n: int = 128, channels: int = 3, resolution: int = 32):
        self.n=n; self.channels=channels; self.res=resolution
    def __len__(self) -> int: return self.n
    def __getitem__(self, idx: int):
        x=torch.randn(self.channels,self.res,self.res)
        return {"x0":x,"label":idx%10}
