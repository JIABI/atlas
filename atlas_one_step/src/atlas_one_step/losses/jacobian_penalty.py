import torch
def jacobian_penalty(x): return torch.mean(torch.abs(x))
