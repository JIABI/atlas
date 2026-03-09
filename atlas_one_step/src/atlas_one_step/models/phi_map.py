
import torch.nn as nn

def build_phi(kind:str='identity'):
    if kind=='identity':
        return nn.Identity()
    if kind=='affine':
        return nn.Sequential(nn.Conv2d(3,3,1), nn.Tanh())
    return nn.Sequential(nn.Conv2d(3,16,1), nn.ReLU(), nn.Conv2d(16,3,1), nn.Tanh())
