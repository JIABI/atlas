from __future__ import annotations

"""Surrogate fitting over atlas records."""

from pathlib import Path
import json
from typing import Any


def _flatten(rec: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    p = rec.get("pathology", {})
    q = rec.get("quality", {})
    out["pathology_score"] = float(p.get("pathology_score", 0.0))
    out["jacobian_norm"] = float(p.get("jacobian_norm", 0.0))
    out["rho_nor"] = float(p.get("rho_nor", 0.0))
    out["fid"] = float(q.get("fid", 0.0))
    return out


def _fit_linear(xs: list[list[float]], ys: list[float]) -> tuple[list[float], float]:
    n = len(xs)
    if n == 0:
        raise ValueError("No training rows for surrogate")
    # two-feature closed form with intercept for robustness
    x1 = [r[0] for r in xs]
    x2 = [r[1] for r in xs]
    y = ys
    mean = lambda arr: sum(arr) / len(arr)
    m1, m2, my = mean(x1), mean(x2), mean(y)
    c11 = sum((a - m1) ** 2 for a in x1) + 1e-8
    c22 = sum((b - m2) ** 2 for b in x2) + 1e-8
    c12 = sum((a - m1) * (b - m2) for a, b in zip(x1, x2))
    cy1 = sum((a - m1) * (yy - my) for a, yy in zip(x1, y))
    cy2 = sum((b - m2) * (yy - my) for b, yy in zip(x2, y))
    det = c11 * c22 - c12 * c12 + 1e-8
    w1 = (cy1 * c22 - cy2 * c12) / det
    w2 = (cy2 * c11 - cy1 * c12) / det
    b = my - w1 * m1 - w2 * m2
    pred = [b + w1 * a + w2 * b2 for a, b2 in zip(x1, x2)]
    sse = sum((yy - pp) ** 2 for yy, pp in zip(y, pred))
    sst = sum((yy - my) ** 2 for yy in y) + 1e-8
    r2 = 1.0 - sse / sst
    return [b, w1, w2], float(r2)


def fit_from_atlas(atlas_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = atlas_path.with_suffix(".jsonl")
    if not jsonl.exists():
        raise FileNotFoundError(f"Atlas JSONL fallback not found: {jsonl}")
    rows = [_flatten(json.loads(line)) for line in jsonl.read_text().splitlines() if line.strip()]
    xs = [[r["pathology_score"], r["jacobian_norm"]] for r in rows]
    ys = [r["fid"] for r in rows]
    coeffs, r2 = _fit_linear(xs, ys)
    model = {"intercept": coeffs[0], "w_pathology": coeffs[1], "w_jacobian": coeffs[2], "r2": r2}
    (out_dir / "model.json").write_text(json.dumps(model, indent=2))
    return model
