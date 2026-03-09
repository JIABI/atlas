import torch
def time_embed(t,dim=16):
    return torch.stack([t]*dim,dim=-1)
