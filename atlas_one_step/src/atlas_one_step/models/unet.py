
import torch.nn as nn
class SimpleUNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,base_channels=32):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels,base_channels,3,1,1),nn.ReLU(),
            nn.Conv2d(base_channels,out_channels,3,1,1)
        )
    def forward(self,x,t=None):
        return self.net(x)
