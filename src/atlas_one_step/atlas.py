from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_summaries(sweep_dir: str | Path) -> pd.DataFrame:
    paths = sorted(Path(sweep_dir).rglob('summary.json'))
    if not paths:
        raise FileNotFoundError(f'No summary.json files found under {sweep_dir}')
    rows = []
    for p in paths:
        row = pd.read_json(p).to_dict()
        rows.append(row)
    return pd.json_normalize(rows)


def assign_phase_regions(df: pd.DataFrame, mse_threshold: float = 0.2, collapse_threshold: float = 0.2) -> pd.Series:
    region = []
    for _, row in df.iterrows():
        mse = row.get('quality.mse', 1.0)
        rare = row.get('tail.rare_failure_rate', 1.0)
        if mse < mse_threshold and rare < collapse_threshold:
            region.append('trainable')
        elif mse < mse_threshold * 1.5:
            region.append('rescue')
        else:
            region.append('failure')
    return pd.Series(region, index=df.index)


def _save_atlas_table(df: pd.DataFrame, output_dir: Path) -> Path:
    parquet_path = output_dir / 'atlas.parquet'
    csv_path = output_dir / 'atlas.csv'
    pkl_path = output_dir / 'atlas.pkl'
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        return pkl_path


def _load_atlas_table(atlas_path: str | Path) -> pd.DataFrame:
    atlas_path = Path(atlas_path)
    if atlas_path.suffix == '.parquet':
        return pd.read_parquet(atlas_path)
    if atlas_path.suffix == '.pkl':
        return pd.read_pickle(atlas_path)
    if atlas_path.suffix == '.csv':
        return pd.read_csv(atlas_path)
    raise ValueError(f'Unsupported atlas file format: {atlas_path}')


def build_atlas(sweep_dir: str | Path, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_summaries(sweep_dir)
    df['phase_region'] = assign_phase_regions(df)
    atlas_path = _save_atlas_table(df, output_dir)
    return atlas_path


def fit_surrogates(atlas_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_atlas_table(atlas_path)
    df['target_family'] = df['prediction_spec.family'].fillna(df['loss_spec.family'])
    # Label-only
    X_label = df[['target_family']].copy()
    y_cls = df['phase_region']
    y_reg = df['quality.mse']
    cat = ColumnTransformer([('fam', OneHotEncoder(handle_unknown='ignore'), ['target_family'])])
    clf_label = Pipeline([('cat', cat), ('rf', RandomForestClassifier(n_estimators=100, random_state=0))])
    reg_label = Pipeline([('cat', cat), ('rf', RandomForestRegressor(n_estimators=100, random_state=0))])
    clf_label.fit(X_label, y_cls)
    reg_label.fit(X_label, y_reg)

    # Diagnostics-only
    diag_cols = [
        'pathology.support_pix', 'pathology.support_perc', 'pathology.support_ssl',
        'pathology.rho_nor', 'pathology.conditioning', 'pathology.pathology_score', 'grad_var'
    ]
    diag_cols = [c for c in diag_cols if c in df.columns]
    X_diag = df[diag_cols].fillna(0.0)
    clf_diag = RandomForestClassifier(n_estimators=200, random_state=0)
    reg_diag = RandomForestRegressor(n_estimators=200, random_state=0)
    clf_diag.fit(X_diag, y_cls)
    reg_diag.fit(X_diag, y_reg)

    # Combined
    X_comb = pd.concat([X_label, X_diag], axis=1)
    comb_pre = ColumnTransformer([
        ('fam', OneHotEncoder(handle_unknown='ignore'), ['target_family']),
        ('diag', 'passthrough', diag_cols),
    ])
    clf_comb = Pipeline([('pre', comb_pre), ('rf', RandomForestClassifier(n_estimators=200, random_state=0))])
    reg_comb = Pipeline([('pre', comb_pre), ('rf', RandomForestRegressor(n_estimators=200, random_state=0))])
    clf_comb.fit(X_comb, y_cls)
    reg_comb.fit(X_comb, y_reg)

    metrics = {
        'label_only_acc': float(accuracy_score(y_cls, clf_label.predict(X_label))),
        'diag_only_acc': float(accuracy_score(y_cls, clf_diag.predict(X_diag))),
        'combined_acc': float(accuracy_score(y_cls, clf_comb.predict(X_comb))),
        'label_only_r2': float(r2_score(y_reg, reg_label.predict(X_label))),
        'diag_only_r2': float(r2_score(y_reg, reg_diag.predict(X_diag))),
        'combined_r2': float(r2_score(y_reg, reg_comb.predict(X_comb))),
        'diag_cols': diag_cols,
    }
    joblib.dump({'clf_diag': clf_diag, 'reg_diag': reg_diag, 'diag_cols': diag_cols}, output_dir / 'surrogate.joblib')
    pd.Series(metrics).to_json(output_dir / 'surrogate_metrics.json', indent=2)
    return metrics
