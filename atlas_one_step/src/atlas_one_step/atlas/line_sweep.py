from __future__ import annotations

from pathlib import Path
import json
import random

from .atlas_io import write_record


def _synthetic_metrics(seed: int) -> tuple[list[float], dict[str, float]]:
    random.seed(seed)
    losses = [max(0.01, 1.0 / (i + 2) + random.random() * 0.02) for i in range(32)]
    probes = {
        "support_pix": random.random(),
        "support_perc": random.random(),
        "support_ssl": random.random(),
        "rho_tan": random.random(),
        "rho_nor": random.random(),
        "jacobian_norm": random.random(),
        "anisotropy": random.random() * 2,
        "sharpness": random.random(),
    }
    probes["pathology_score"] = float(sum(probes.values()))
    return losses, probes


def run_line_sweep(out_dir: str = "outputs/atlas/sweeps", seeds: tuple[int, ...] = (0, 1), family: str = "line_x0_u") -> None:
    from ..eval.trainability import compute as trainability_compute
    from ..eval.tail_failure import compute as tail_compute

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        losses, probes = _synthetic_metrics(seed)
        rec = {
            "exp_id": "EXP-A1",
            "dataset": "cifar10",
            "split": "train",
            "model": "unet_small",
            "corruption": "diffusion_like",
            "target_family": family,
            "target_lambda": {"alpha": 0.5},
            "prediction_family": "line",
            "prediction_eta": {"alpha": 0.5},
            "lambda_loss": {"mu": 0.1, "tau": 0.1},
            "seed": seed,
            "resolution": 32,
            "regularization": {"mu": 0.1, "tau": 0.1},
            "trainability": trainability_compute(losses),
            "quality": {"fid": float(sum(losses) / len(losses) * 100), "sfid": float(sum(losses) / len(losses) * 80), "lpips": float(sum(losses) / len(losses))},
            "tail": tail_compute(losses),
            "pathology": probes,
            "artifacts": {"checkpoint": "", "samples_dir": "", "plots_dir": "", "config_path": ""},
        }
        write_record(Path(out_dir) / f"{family}_seed{seed}.json", rec)
    Path(out_dir, "manifest.json").write_text(json.dumps({"n_records": len(seeds), "family": family}, indent=2))
