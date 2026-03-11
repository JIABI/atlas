from __future__ import annotations

from .line_sweep import run_line_sweep


def run_simplex_sweep(out_dir: str = "outputs/atlas/sweeps", seeds: tuple[int, ...] = (0, 1), family: str = "simplex_x0_u_r") -> None:
    run_line_sweep(out_dir=out_dir, seeds=seeds, family=family)
