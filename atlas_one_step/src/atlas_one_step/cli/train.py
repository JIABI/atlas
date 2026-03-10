from __future__ import annotations

from ._common import load_cfg
from ..train.coupled_runner import CoupledRunner
from ..train.decoupled_runner import DecoupledRunner
from ..train.manual_target_runner import ManualTargetRunner
from ..train.unguided_decoupled_runner import UnguidedDecoupledRunner


def main() -> None:
    cfg = load_cfg()
    train_name = cfg.train["name"] if isinstance(cfg.train, dict) else cfg.train.name
    if "manual" in train_name:
        runner = ManualTargetRunner(cfg)
    elif "unguided" in train_name:
        runner = UnguidedDecoupledRunner(cfg)
    elif "decoupled" in train_name:
        runner = DecoupledRunner(cfg)
    else:
        runner = CoupledRunner(cfg)
    print(runner.run())


if __name__ == "__main__":
    main()
