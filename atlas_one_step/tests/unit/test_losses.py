from atlas_one_step.losses.loss_factory import total_loss
import torch
def test_loss():
 x=torch.randn(2,3,4,4,requires_grad=True); l,_=total_loss(x,x); assert l.item()>=0
