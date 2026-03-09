
from dataclasses import dataclass
import torch

@dataclass
class StateTuple:
    x0: torch.Tensor
    xt: torch.Tensor
    eps: torch.Tensor
    t: torch.Tensor
