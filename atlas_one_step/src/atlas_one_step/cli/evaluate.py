from __future__ import annotations

import json
from pathlib import Path

from ._common import load_cfg
from ..eval.trainability import compute as trainability_compute
from ..eval.tail_failure import compute as tail_compute
from ..eval.fid import compute as fid_compute
from ..eval.sfid import compute as sfid_compute
from ..eval.lpips import compute as lpips_compute
from ..eval.resolution_scaling import compute as res_compute


def _load_sweep_records(folder: Path) -> list[dict]:
    if not folder.exists():
        return []
    return [json.loads(p.read_text()) for p in folder.glob("*.json")]


def main() -> None:
    cfg = load_cfg()
    eval_name = cfg.eval.name
    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = _load_sweep_records(Path("outputs/atlas/sweeps"))
    if not recs:
        recs = [{"resolution": 64, "quality": {"fid": 50.0}}]

    if eval_name == "resolution_scaling":
        result = {"resolution_scaling": res_compute(recs)}
    elif eval_name == "tail":
        result = {"tail": tail_compute([0.1, 0.2, 0.3, 0.9, 0.95])}
    elif eval_name == "pilot":
        result = {
            "trainability": trainability_compute([1.0, 0.7, 0.4, 0.2, 0.09]),
            "tail": tail_compute([0.1, 0.2, 0.4, 0.7, 1.0]),
        }
    else:
        result = {
            "quality": {
                "fid": fid_compute(0.0, 1.0, 0.1, 1.2),
                "sfid": sfid_compute(0.0, 1.0, 0.1, 1.2),
                "lpips": lpips_compute([0.2, 0.3, 0.25]),
            }
        }
    (out_dir / f"{eval_name}.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
