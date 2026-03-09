import torch
def compute_support_deviation(x): return {"support_pix": float(torch.mean(torch.abs(x)).item()), "support_perc": 0.0, "support_ssl": 0.0}
