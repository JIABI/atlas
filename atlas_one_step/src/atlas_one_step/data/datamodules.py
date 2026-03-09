
from torch.utils.data import DataLoader
from .cifar10 import RandomImageDataset

def build_dataloader(batch_size:int=8,resolution:int=32):
    return DataLoader(RandomImageDataset(resolution=resolution), batch_size=batch_size)
