import random, numpy as np, torch
def seed_everything(seed:int): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
