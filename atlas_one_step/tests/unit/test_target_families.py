import importlib.util
import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch unavailable")
def test_line_shape():
    from atlas_one_step.targets.line_families import line_x0_u
    import torch

    x = torch.randn(2, 3, 4, 4)
    y = line_x0_u(0.5, x, x, torch.randn_like(x), torch.rand(2))
    assert y.shape == x.shape
