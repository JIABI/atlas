import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self,c):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(c,c,3,1,1),nn.ReLU())
    def forward(self,x): return self.net(x)
