import importlib.util
import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch unavailable")
def test_probes():
    from atlas_one_step.probes.probe_pipeline import compute_probes
    import torch

    d = compute_probes(torch.randn(2, 3, 4, 4))
    assert "pathology_score" in d
