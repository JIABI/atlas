import torch
def semantic_gap(pred,target):
    return torch.mean((pred-target)**2)
