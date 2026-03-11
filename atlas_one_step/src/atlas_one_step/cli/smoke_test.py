from __future__ import annotations

from ..atlas.line_sweep import run_line_sweep
from ..atlas.build_atlas import build_atlas


def main() -> None:
    run_line_sweep(out_dir="outputs/atlas/smoke", seeds=(0,), family="line_x0_u")
    n = build_atlas(in_dir="outputs/atlas/smoke", out="outputs/atlas/smoke_atlas.parquet")
    print(f"smoke test ok: {n} records")


if __name__ == "__main__":
    main()
