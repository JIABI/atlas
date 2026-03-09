import torch
def semantic_loss(pred,target): return torch.mean(torch.abs(pred-target))
