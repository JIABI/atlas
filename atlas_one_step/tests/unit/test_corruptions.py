from atlas_one_step.corruption.diffusion_like import apply_corruption
import torch
def test_corruptions():
 x=torch.randn(2,3,4,4); xt,eps=apply_corruption(x,torch.rand(2)); assert xt.shape==x.shape and eps.shape==x.shape
