from atlas_one_step.utils.simple_config import compose_config
from pathlib import Path


def test_compose_config():
    cfg = compose_config(Path(__file__).resolve().parents[2], ["dataset=cifar10", "train=cheap_probe"])
    assert cfg.dataset.name == "cifar10"
    assert cfg.train.name == "cheap_probe"
