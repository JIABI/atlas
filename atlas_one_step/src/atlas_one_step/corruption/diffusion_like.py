
import torch

def apply_corruption(x0: torch.Tensor, t: torch.Tensor):
    eps=torch.randn_like(x0)
    a=(1-t).view(-1,1,1,1)
    xt=a*x0 + (1-a)*eps
    return xt, eps
