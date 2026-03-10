from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .corruption import DiffusionLikeCorruption
from .losses import LossWeights, prediction_loss, semantic_loss, stability_loss
from .metrics import feature_fd, mse_per_sample, psnr_from_mse, summarize_tail
from .probes import covariance_conditioning, normal_burden, pathology_score, support_deviation
from .targets import TargetSpec, construct_target, reconstruct_x0_from_target, spec_to_dict
from .utils import append_jsonl, ensure_dir, save_json


@dataclass
class RunArtifacts:
    summary_path: Path
    metrics_path: Path
    checkpoint_dir: Path


class OneStepTrainer:
    def __init__(
        self,
        model: nn.Module,
        phi_map: nn.Module,
        corruption: DiffusionLikeCorruption,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str | Path,
        loss_weights: LossWeights,
    ) -> None:
        self.model = model.to(device)
        self.phi_map = phi_map.to(device)
        self.corruption = corruption
        self.optimizer = optimizer
        self.device = device
        self.output_dir = ensure_dir(output_dir)
        self.loss_weights = loss_weights
        self.metrics_path = self.output_dir / 'metrics.jsonl'
        self.ckpt_dir = ensure_dir(self.output_dir / 'checkpoints')

    def _state(self, x0: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        xt, eps = self.corruption.sample_xt(x0, t)
        state = self.corruption.primitives(x0, xt, eps, t)
        state['t_scalar'] = t
        return state

    def train(
        self,
        loader: DataLoader,
        prediction_spec: TargetSpec,
        loss_spec: TargetSpec,
        max_steps: int,
        eval_every: int,
        collapse_threshold: float,
        tail_percentiles: list[int],
        mode: str,
        grad_clip: float = 1.0,
    ) -> dict[str, Any]:
        self.model.train()
        self.phi_map.train()
        step = 0
        losses = []
        grad_norms = []
        best_val = float('inf')
        best_path = self.ckpt_dir / 'best.pt'
        last_eval: dict[str, Any] = {}
        start = time.time()
        loader_iter = iter(loader)
        while step < max_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            if isinstance(batch, (list, tuple)):
                x0 = batch[0]
            else:
                x0 = batch
            x0 = x0.to(self.device)
            t = self.corruption.sample_t(x0.shape[0], self.device)
            state = self._state(x0, t)
            pred_target = construct_target(prediction_spec, state)
            loss_target = construct_target(loss_spec, state)
            z_hat = self.model(state['xt'], t)
            mapped = self.phi_map(z_hat, t)

            lp = prediction_loss(z_hat, pred_target)
            ls = semantic_loss(mapped, loss_target)
            lstab = stability_loss(self.phi_map, z_hat)
            total = self.loss_weights.pred_weight * lp
            if mode != 'coupled':
                total = total + self.loss_weights.sem_weight * ls
            total = total + self.loss_weights.stab_weight * lstab

            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.phi_map.parameters()), grad_clip)
            self.optimizer.step()

            losses.append(float(total.item()))
            grad_norms.append(float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm))
            append_jsonl({'step': step, 'loss': float(total.item()), 'pred_loss': float(lp.item()), 'sem_loss': float(ls.item()), 'stab_loss': float(lstab.item()), 'grad_norm': grad_norms[-1]}, self.metrics_path)

            if (step + 1) % eval_every == 0 or step + 1 == max_steps:
                last_eval = self.evaluate(loader, prediction_spec, loss_spec, collapse_threshold, tail_percentiles)
                save_json(last_eval, self.output_dir / 'last_eval.json')
                val_key = last_eval['quality']['mse']
                if val_key < best_val:
                    best_val = val_key
                    torch.save({'model': self.model.state_dict(), 'phi_map': self.phi_map.state_dict()}, best_path)
            step += 1

        summary = {
            'mode': mode,
            'prediction_spec': spec_to_dict(prediction_spec),
            'loss_spec': spec_to_dict(loss_spec),
            'trainability': {
                'converged': bool(best_val < collapse_threshold),
                'diverged': bool(any(not np.isfinite(l) for l in losses)) if False else False,
                'time_to_threshold': next((i for i, l in enumerate(losses) if l < collapse_threshold), None),
                'collapse_rate': float((torch.tensor(losses) > collapse_threshold).float().mean().item()),
            },
            'quality': last_eval.get('quality', {}),
            'tail': last_eval.get('tail', {}),
            'pathology': last_eval.get('pathology', {}),
            'runtime_sec': time.time() - start,
            'best_checkpoint': str(best_path),
            'grad_var': float(torch.tensor(grad_norms).var().item()) if len(grad_norms) > 1 else 0.0,
        }
        summary['pathology']['early_grad_var'] = summary['grad_var']
        summary['pathology']['pathology_score'] = pathology_score(summary['pathology'])
        save_json(summary, self.output_dir / 'summary.json')
        return summary

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prediction_spec: TargetSpec, loss_spec: TargetSpec, collapse_threshold: float, tail_percentiles: list[int], max_batches: int = 8) -> dict[str, Any]:
        self.model.eval()
        self.phi_map.eval()
        x0_all, recon_all = [], []
        pred_target_vis, x0_vis = None, None
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
            x0 = x0.to(self.device)
            t = self.corruption.sample_t(x0.shape[0], self.device)
            state = self._state(x0, t)
            z_hat = self.model(state['xt'], t)
            # Evaluate in prediction space for reconstruction.
            x0_hat = reconstruct_x0_from_target(prediction_spec, z_hat, state)
            x0_all.append(x0)
            recon_all.append(x0_hat)
            pred_target_vis = z_hat
            x0_vis = x0
        x0_cat = torch.cat(x0_all, dim=0)
        recon_cat = torch.cat(recon_all, dim=0).clamp(-1, 1)
        mse = mse_per_sample(x0_cat, recon_cat)
        psnr = psnr_from_mse(mse)
        support = support_deviation(pred_target_vis, x0_vis)
        normal = normal_burden(pred_target_vis, x0_vis)
        cond = covariance_conditioning(pred_target_vis)
        pathology = {**support, **normal, **cond}
        tail = summarize_tail(mse, collapse_threshold, tail_percentiles)
        return {
            'quality': {
                'mse': float(mse.mean().item()),
                'psnr': float(psnr.mean().item()),
                'feature_fd': feature_fd(x0_cat, recon_cat),
            },
            'tail': tail,
            'pathology': pathology,
        }
