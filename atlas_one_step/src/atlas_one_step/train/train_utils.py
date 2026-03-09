import torch
def step(model,batch):
    return model(batch["x0"])
