from __future__ import annotations

from pathlib import Path
from .atlas_io import read_records, to_parquet


def aggregate(in_dir: str, out_path: str) -> int:
    records = read_records(Path(in_dir))
    if not records:
        raise FileNotFoundError(f"No sweep records found in {in_dir}")
    to_parquet(records, Path(out_path))
    return len(records)
