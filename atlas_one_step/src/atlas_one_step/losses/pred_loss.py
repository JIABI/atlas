import torch
def pred_loss(pred,target): return torch.mean((pred-target)**2)
