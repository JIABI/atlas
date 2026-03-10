import importlib.util
import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch unavailable")
def test_loss():
    from atlas_one_step.losses.loss_factory import total_loss
    import torch

    x = torch.randn(2, 3, 4, 4, requires_grad=True)
    l, _ = total_loss(x, x)
    assert l.item() >= 0
