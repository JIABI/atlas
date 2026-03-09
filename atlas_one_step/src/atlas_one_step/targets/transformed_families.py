import torch
def transformed(y, kind="tanh"):
    return torch.tanh(y) if kind=="tanh" else y
