import importlib.util
import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch unavailable")
def test_corruptions():
    from atlas_one_step.corruption.diffusion_like import apply_corruption
    import torch

    x = torch.randn(2, 3, 4, 4)
    xt, eps = apply_corruption(x, torch.rand(2))
    assert xt.shape == x.shape and eps.shape == x.shape
