import importlib.util
from pathlib import Path

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch unavailable")
def test_line_sweep_pipeline(tmp_path):
    from atlas_one_step.atlas.line_sweep import run_line_sweep

    out = tmp_path / "sweeps"
    run_line_sweep(out_dir=str(out), seeds=(0,), family="line_x0_u")
    assert (out / "line_x0_u_seed0.json").exists()
