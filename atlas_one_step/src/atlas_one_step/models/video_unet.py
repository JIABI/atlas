import torch.nn as nn
class VideoUNet(nn.Module):
    def __init__(self): super().__init__(); self.id=nn.Identity()
    def forward(self,x,t=None): return self.id(x)
