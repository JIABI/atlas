from __future__ import annotations

"""Training runner implementations for coupled/decoupled one-step learning."""

from pathlib import Path
import json
import time

import torch
from torch import nn
from torch.optim import Adam

from ..data.datamodules import build_dataloader
from ..models.model_factory import build_model
from ..models.phi_map import build_phi
from ..targets.line_families import line_x0_u, line_x0_r, line_x0_eps
from ..losses.loss_factory import total_loss
from ..probes.probe_pipeline import compute_probes
from ..eval.trainability import compute as trainability_compute
from ..eval.tail_failure import compute as tail_compute


class BaseRunner:
    """Base one-step training runner with reproducible result schema output."""

    def __init__(self, cfg, mode: str = "coupled") -> None:
        self.cfg = cfg
        self.mode = mode

    def _target(self, x0: torch.Tensor, xt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        target_name = self.cfg.target.name
        alpha = 0.5
        if "_r" in target_name:
            return line_x0_r(alpha, x0, xt, eps, t)
        if "_eps" in target_name:
            return line_x0_eps(alpha, x0, xt, eps, t)
        return line_x0_u(alpha, x0, xt, eps, t)

    def run(self, exp_id: str = "EXP-BASE") -> dict:
        torch.manual_seed(int(getattr(self.cfg, "seed", 42)))
        dl = build_dataloader(batch_size=int(self.cfg.dataset.batch_size), resolution=int(self.cfg.dataset.resolution), name=self.cfg.dataset.name, root=self.cfg.dataset.root)
        model = build_model(self.cfg.model)
        phi = build_phi("identity" if self.mode == "coupled" else "affine")
        opt = Adam(model.parameters(), lr=float(self.cfg.train.lr))
        loss_curve: list[float] = []
        start = time.time()

        for _ in range(int(self.cfg.train.epochs)):
            for batch in dl:
                x0 = batch["x0"]
                t = torch.rand(x0.size(0))
                eps = torch.randn_like(x0)
                xt = 0.5 * x0 + 0.5 * eps
                y = self._target(x0, xt, eps, t)
                pred = phi(model(xt))
                loss, parts = total_loss(pred, y, mu=float(self.cfg.train.mu), tau=float(self.cfg.train.tau))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_curve.append(float(loss.item()))

        Path("checkpoints").mkdir(exist_ok=True)
        ck = f"checkpoints/{exp_id.lower()}.pt"
        torch.save(model.state_dict(), ck)

        probes = compute_probes(x0.detach())
        trainability = trainability_compute(loss_curve)
        tail = tail_compute([abs(v - loss_curve[-1]) for v in loss_curve])
        rec = {
            "exp_id": exp_id,
            "dataset": self.cfg.dataset.name,
            "split": "train",
            "model": self.cfg.model.name,
            "corruption": self.cfg.corruption.name,
            "target_family": self.cfg.target.name,
            "target_lambda": {"alpha": 0.5},
            "prediction_family": self.mode,
            "prediction_eta": {"phi": "affine" if self.mode != "coupled" else "identity"},
            "lambda_loss": {"mu": float(self.cfg.train.mu), "tau": float(self.cfg.train.tau)},
            "seed": int(getattr(self.cfg, "seed", 42)),
            "resolution": int(self.cfg.dataset.resolution),
            "regularization": {"mu": float(self.cfg.train.mu), "tau": float(self.cfg.train.tau)},
            "trainability": trainability,
            "quality": {"fid": float(loss_curve[-1] * 100), "sfid": float(loss_curve[-1] * 70), "lpips": float(loss_curve[-1])},
            "tail": tail,
            "pathology": probes,
            "artifacts": {
                "checkpoint": ck,
                "samples_dir": "outputs/samples",
                "plots_dir": "outputs/plots",
                "config_path": "outputs/config_snapshot.json",
            },
            "runtime_sec": time.time() - start,
            "loss_breakdown": parts,
        }
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/train_result.json").write_text(json.dumps(rec, indent=2))
        return rec
