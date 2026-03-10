import torch
def stability_loss(pred): return torch.mean(pred**2)
