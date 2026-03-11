from __future__ import annotations

from .line_sweep import run_line_sweep


def run_schedule_sweep(out_dir: str = "outputs/atlas/sweeps", seeds: tuple[int, ...] = (0, 1), family: str = "scheduled") -> None:
    run_line_sweep(out_dir=out_dir, seeds=seeds, family=family)
