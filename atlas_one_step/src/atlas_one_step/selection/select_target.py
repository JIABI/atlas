from __future__ import annotations

"""Target selection methods based on atlas records."""

from pathlib import Path
import json
from typing import Any


def select_target(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    return min(candidates, key=lambda c: float(c.get("pathology_score", 1e9)))


def select_from_atlas(atlas_path: Path, strategy: str = "pathology_min") -> dict[str, Any]:
    rows = [json.loads(line) for line in atlas_path.with_suffix(".jsonl").read_text().splitlines() if line.strip()]
    if strategy == "pathology_min":
        best = min(rows, key=lambda r: float(r.get("pathology", {}).get("pathology_score", 1e9)))
    else:
        best = min(rows, key=lambda r: float(r.get("quality", {}).get("fid", 1e9)))
    return {
        "target_family": best.get("target_family"),
        "target_lambda": best.get("target_lambda"),
        "seed": best.get("seed"),
        "pathology_score": best.get("pathology", {}).get("pathology_score"),
    }
