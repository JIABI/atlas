from __future__ import annotations

"""I/O helpers for sweep records and atlas artifacts."""

import json
from pathlib import Path
from typing import Any


def write_record(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")


def read_records(folder: Path) -> list[dict[str, Any]]:
    if not folder.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]


def write_jsonl(records: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def to_parquet(records: list[dict[str, Any]], out: Path) -> None:
    """Write parquet when pandas/pyarrow are available and always emit JSONL fallback."""
    write_jsonl(records, out.with_suffix(".jsonl"))
    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(records).to_parquet(out, index=False)
    except Exception as exc:
        out.with_suffix(".parquet.error.txt").write_text(
            f"Parquet export unavailable: {exc}\nJSONL fallback: {out.with_suffix('.jsonl')}\n",
            encoding="utf-8",
        )
