from __future__ import annotations

from ._common import load_cfg
from ..atlas.build_atlas import build_atlas


def main() -> None:
    cfg = load_cfg()
    out = cfg.atlas.get("out_dir", "outputs/atlas/build_atlas") if isinstance(cfg.atlas, dict) else cfg.atlas.out_dir
    n = build_atlas(in_dir="outputs/atlas/sweeps", out="outputs/atlas/atlas.parquet")
    print({"records": n, "atlas_dir": out})


if __name__ == "__main__":
    main()
