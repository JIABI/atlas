from atlas_one_step.targets.line_families import line_x0_u
import torch
def test_line_shape():
 x=torch.randn(2,3,4,4); y=line_x0_u(0.5,x,x,torch.randn_like(x),torch.rand(2)); assert y.shape==x.shape
