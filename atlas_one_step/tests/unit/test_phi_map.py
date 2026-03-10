import importlib.util
import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch unavailable")
def test_phi():
    from atlas_one_step.models.phi_map import build_phi

    assert build_phi("identity") is not None
