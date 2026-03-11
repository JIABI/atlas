from __future__ import annotations

"""Resolution scaling summary over per-resolution benchmark records."""

from typing import Sequence


def compute(records: Sequence[dict]) -> dict[int, float]:
    out: dict[int, float] = {}
    for rec in records:
        res = int(rec["resolution"])
        out.setdefault(res, 0.0)
        out[res] += float(rec.get("quality", {}).get("fid", 0.0))
    for k in list(out):
        n = sum(1 for r in records if int(r["resolution"]) == k)
        out[k] = out[k] / max(n, 1)
    return out
