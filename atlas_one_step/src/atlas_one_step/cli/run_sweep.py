from __future__ import annotations

from ._common import load_cfg
from ..atlas.line_sweep import run_line_sweep
from ..atlas.simplex_sweep import run_simplex_sweep
from ..atlas.schedule_sweep import run_schedule_sweep


def main() -> None:
    cfg = load_cfg()
    name = cfg.atlas["name"] if isinstance(cfg.atlas, dict) else cfg.atlas.name
    family = cfg.target["name"] if isinstance(cfg.target, dict) else cfg.target.name
    seeds = cfg.atlas.get("seeds", [0]) if isinstance(cfg.atlas, dict) else cfg.atlas.seeds
    if "simplex" in name:
        run_simplex_sweep(seeds=tuple(seeds), family=family)
    elif "schedule" in name:
        run_schedule_sweep(seeds=tuple(seeds), family=family)
    else:
        run_line_sweep(seeds=tuple(seeds), family=family)


if __name__ == "__main__":
    main()
