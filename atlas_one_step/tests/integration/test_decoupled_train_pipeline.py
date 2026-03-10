import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch unavailable")
def test_decoupled_train_pipeline():
    from atlas_one_step.cli._common import load_cfg
    from atlas_one_step.train.decoupled_runner import DecoupledRunner

    cfg = load_cfg(["dataset=cifar10", "train=cheap_probe"])
    out = DecoupledRunner(cfg).run(exp_id="EXP-TEST")
    assert out["exp_id"] == "EXP-TEST"
