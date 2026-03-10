from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd

from .targets import TargetSpec, sample_target_specs, spec_to_dict
from .utils import save_json


def semantic_gap(candidate: TargetSpec, loss_target: TargetSpec) -> float:
    # Lightweight analytic gap on parameter values.
    if candidate.family == loss_target.family:
        keys = set(candidate.params) | set(loss_target.params)
        return float(sum(abs(float(candidate.params.get(k, 0.0)) - float(loss_target.params.get(k, 0.0))) for k in keys if isinstance(candidate.params.get(k, 0.0), (int, float))))
    return 1.0 + 0.1 * candidate.complexity()


def rank_candidates(
    candidates: list[TargetSpec],
    surrogate_path: str | Path,
    loss_target: TargetSpec,
    semantic_gap_weight: float,
    complexity_weight: float,
) -> list[dict[str, Any]]:
    model = joblib.load(surrogate_path)
    clf = model['clf_diag']
    diag_cols = model['diag_cols']
    # Without a candidate atlas row, use heuristic pseudo-diagnostics from target complexity and family.
    # This keeps the selection pipeline runnable while remaining explicit.
    rows = []
    for spec in candidates:
        fam_score = {'line_x0_u': 0.45, 'line_x0_r': 0.35, 'line_x0_eps': 0.8, 'simplex': 0.5, 'scheduled': 0.4}.get(spec.family, 0.6)
        pseudo = {
            'pathology.support_pix': fam_score + 0.02 * spec.complexity(),
            'pathology.support_perc': fam_score,
            'pathology.support_ssl': fam_score,
            'pathology.rho_nor': min(1.0, fam_score),
            'pathology.conditioning': 1.0 + fam_score,
            'pathology.pathology_score': fam_score,
            'grad_var': fam_score,
        }
        X = pd.DataFrame([{c: pseudo.get(c, 0.0) for c in diag_cols}])
        proba = clf.predict_proba(X)[0]
        classes = list(clf.classes_)
        trainable_prob = float(proba[classes.index('trainable')]) if 'trainable' in classes else 0.0
        score = (1.0 - trainable_prob) + semantic_gap_weight * semantic_gap(spec, loss_target) + complexity_weight * spec.complexity()
        rows.append({'spec': spec_to_dict(spec), 'objective': score, 'trainable_prob': trainable_prob})
    rows = sorted(rows, key=lambda x: x['objective'])
    return rows


def select_target(
    surrogate_path: str | Path,
    output_path: str | Path,
    family: str,
    num_points: int,
    loss_target: TargetSpec,
    semantic_gap_weight: float = 0.5,
    complexity_weight: float = 0.05,
    schedule_basis_order: int = 3,
) -> dict[str, Any]:
    candidates = sample_target_specs(family, num_points, schedule_basis_order=schedule_basis_order)
    ranked = rank_candidates(candidates, surrogate_path, loss_target, semantic_gap_weight, complexity_weight)
    selected = {'selected': ranked[0], 'topk': ranked[:5]}
    save_json(selected, output_path)
    return selected
