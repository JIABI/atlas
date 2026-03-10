from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd
import torch

from .atlas import build_atlas, fit_surrogates
from .config import load_yaml, save_yaml
from .corruption import DiffusionLikeCorruption
from .data import build_dataset_bundle
from .losses import LossWeights
from .model import build_model, build_phi_map
from .runners import OneStepTrainer
from .selection import select_target
from .targets import TargetSpec, sample_target_specs, spec_from_dict
from .utils import ensure_dir, infer_device, save_json, set_seed, setup_logger

logger = setup_logger()


def _common_from_config(cfg: dict):
    set_seed(int(cfg.get('seed', 0)))
    bundle = build_dataset_bundle(cfg['dataset'])
    device = infer_device()
    corr_cfg = dict(cfg['corruption']); corr_cfg.pop('name', None); corr = DiffusionLikeCorruption(**corr_cfg)
    model = build_model(cfg['model'], bundle.channels)
    phi = build_phi_map(cfg.get('phi_map', {'type': 'identity'}), bundle.channels)
    params = list(model.parameters()) + list(phi.parameters())
    opt = torch.optim.Adam(params, lr=float(cfg['train']['lr']))
    loss_weights = LossWeights(
        pred_weight=float(cfg['loss'].get('pred_weight', 1.0)),
        sem_weight=float(cfg['loss'].get('sem_weight', 0.5)),
        stab_weight=float(cfg['loss'].get('stab_weight', 1e-4)),
    )
    return bundle, device, corr, model, phi, opt, loss_weights


def run_sweep(cfg_path: str):
    cfg = load_yaml(cfg_path)
    bundle, device, corr, _, _, _, loss_weights = _common_from_config(cfg)
    output_root = ensure_dir(cfg['output_root'])
    sweep_dir = ensure_dir(output_root / 'sweeps')
    family = cfg['sweep']['family']
    specs = sample_target_specs(family, int(cfg['sweep']['num_points']), int(cfg.get('selection', {}).get('schedule_basis_order', 3)))
    max_steps = int(cfg['sweep']['max_steps_per_target'])
    for seed in cfg['sweep']['seeds']:
        for i, spec in enumerate(specs):
            set_seed(int(seed))
            model = build_model(cfg['model'], bundle.channels)
            phi = build_phi_map(cfg.get('phi_map', {'type': 'identity'}), bundle.channels)
            opt = torch.optim.Adam(list(model.parameters()) + list(phi.parameters()), lr=float(cfg['train']['lr']))
            run_dir = ensure_dir(sweep_dir / f'{spec.family}_{i:03d}_seed{seed}')
            trainer = OneStepTrainer(model, phi, corr, opt, device, run_dir, loss_weights)
            summary = trainer.train(
                bundle.loader,
                prediction_spec=spec,
                loss_spec=spec,
                max_steps=max_steps,
                eval_every=max(5, max_steps // 2),
                collapse_threshold=float(cfg['eval']['collapse_threshold']),
                tail_percentiles=list(cfg['eval']['tail_percentiles']),
                mode='coupled',
                grad_clip=float(cfg['train'].get('grad_clip', 1.0)),
            )
            summary['exp_id'] = f'sweep_{family}'
            summary['dataset'] = cfg['dataset']['name']
            save_json(summary, run_dir / 'summary.json')
            logger.info('finished %s seed=%s objective=%.4f', spec.family, seed, summary['pathology']['pathology_score'])


def build_atlas_cli(cfg_path: str, sweep_dir: str | None = None):
    cfg = load_yaml(cfg_path)
    output_root = ensure_dir(cfg['output_root'])
    sweep_dir = sweep_dir or str(output_root / 'sweeps')
    atlas_path = build_atlas(sweep_dir, output_root / 'atlas')
    logger.info('atlas saved to %s', atlas_path)


def fit_surrogate_cli(atlas_path: str):
    atlas_path = Path(atlas_path)
    metrics = fit_surrogates(atlas_path, atlas_path.parent)
    logger.info('surrogate metrics: %s', metrics)


def select_target_cli(atlas_path: str, surrogate_path: str, output_path: str, cfg_path: str):
    cfg = load_yaml(cfg_path)
    lt_cfg = cfg['loss_target']
    loss_target = TargetSpec(family=lt_cfg['family'], params={k: v for k, v in lt_cfg.items() if k != 'family'})
    out = select_target(
        surrogate_path=surrogate_path,
        output_path=output_path,
        family=cfg['selection']['candidate_family'],
        num_points=int(cfg.get('sweep', {}).get('num_points', 10)),
        loss_target=loss_target,
        semantic_gap_weight=float(cfg['selection'].get('semantic_gap_weight', 0.5)),
        complexity_weight=float(cfg['selection'].get('complexity_weight', 0.05)),
        schedule_basis_order=int(cfg['selection'].get('schedule_basis_order', 3)),
    )
    logger.info('selected target written to %s', output_path)
    return out


def train_cli(cfg_path: str, mode: str, selected_target: str | None = None):
    cfg = load_yaml(cfg_path)
    bundle, device, corr, model, phi, opt, loss_weights = _common_from_config(cfg)
    output_root = ensure_dir(cfg['output_root'])
    out_dir = ensure_dir(output_root / mode)
    trainer = OneStepTrainer(model, phi, corr, opt, device, out_dir, loss_weights)

    lt_cfg = cfg['loss_target']
    loss_spec = TargetSpec(family=lt_cfg['family'], params={k: v for k, v in lt_cfg.items() if k != 'family'})

    if mode == 'coupled':
        pred_spec = loss_spec
    elif mode == 'manual':
        mt_cfg = cfg['manual_prediction_target']
        pred_spec = TargetSpec(family=mt_cfg['family'], params={k: v for k, v in mt_cfg.items() if k != 'family'})
    elif mode == 'unguided':
        pred_spec = TargetSpec(family='scheduled', params={'ax': [0.8, -0.1, 0.0], 'bu': [0.15, 0.05, 0.0], 'cr': [0.05, 0.0, 0.0]})
    elif mode == 'atlas_guided':
        if selected_target is None:
            raise ValueError('selected_target path is required for atlas_guided mode')
        data = load_yaml(selected_target) if selected_target.endswith('.yaml') else __import__('json').load(open(selected_target, 'r', encoding='utf-8'))
        pred_spec = spec_from_dict(data['selected']['spec'])
    else:
        raise ValueError(f'Unsupported mode: {mode}')

    summary = trainer.train(
        bundle.loader,
        prediction_spec=pred_spec,
        loss_spec=loss_spec,
        max_steps=int(cfg['train']['max_steps']),
        eval_every=int(cfg['train']['eval_every']),
        collapse_threshold=float(cfg['eval']['collapse_threshold']),
        tail_percentiles=list(cfg['eval']['tail_percentiles']),
        mode='coupled' if mode == 'coupled' else 'decoupled',
        grad_clip=float(cfg['train'].get('grad_clip', 1.0)),
    )
    logger.info('train summary saved to %s', out_dir / 'summary.json')
    return summary


def evaluate_cli(cfg_path: str, checkpoint: str, mode: str = 'coupled'):
    cfg = load_yaml(cfg_path)
    bundle, device, corr, model, phi, opt, loss_weights = _common_from_config(cfg)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    phi.load_state_dict(ckpt['phi_map'])
    trainer = OneStepTrainer(model, phi, corr, opt, device, Path(checkpoint).parent.parent, loss_weights)

    lt_cfg = cfg['loss_target']
    loss_spec = TargetSpec(family=lt_cfg['family'], params={k: v for k, v in lt_cfg.items() if k != 'family'})
    pred_spec = loss_spec
    eval_metrics = trainer.evaluate(bundle.loader, pred_spec, loss_spec, float(cfg['eval']['collapse_threshold']), list(cfg['eval']['tail_percentiles']))
    save_json(eval_metrics, Path(checkpoint).parent.parent / 'evaluation.json')
    logger.info('evaluation complete')
    return eval_metrics


def smoke_test(cfg_path: str):
    cfg = load_yaml(cfg_path)
    root = ensure_dir(cfg['output_root'])
    run_sweep(cfg_path)
    build_atlas_cli(cfg_path)
    atlas_path = root / 'atlas' / 'atlas.parquet'
    fit_surrogate_cli(str(atlas_path))
    select_target_cli(str(atlas_path), str(root / 'atlas' / 'surrogate.joblib'), str(root / 'atlas' / 'selected_target.json'), cfg_path)
    train_cli(cfg_path, 'coupled')
    # create temporary train config for atlas-guided
    tmp_cfg = dict(cfg)
    tmp_cfg['loss_target'] = {'family': 'line_x0_u', 'alpha': 0.6}
    save_yaml(tmp_cfg, root / 'tmp_train.yaml')
    train_cli(str(root / 'tmp_train.yaml'), 'atlas_guided', selected_target=str(root / 'atlas' / 'selected_target.json'))
    logger.info('smoke test completed successfully')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('smoke-test')
    p.add_argument('--config', required=True)

    p = sub.add_parser('run-sweep')
    p.add_argument('--config', required=True)

    p = sub.add_parser('build-atlas')
    p.add_argument('--config', required=True)
    p.add_argument('--sweep-dir', default=None)

    p = sub.add_parser('fit-surrogate')
    p.add_argument('--atlas', required=True)

    p = sub.add_parser('select-target')
    p.add_argument('--atlas', required=True)
    p.add_argument('--surrogate', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True)

    p = sub.add_parser('train')
    p.add_argument('--config', required=True)
    p.add_argument('--mode', choices=['coupled', 'manual', 'unguided', 'atlas_guided'], required=True)
    p.add_argument('--selected-target', default=None)

    p = sub.add_parser('evaluate')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)

    args = parser.parse_args()
    if args.command == 'smoke-test':
        smoke_test(args.config)
    elif args.command == 'run-sweep':
        run_sweep(args.config)
    elif args.command == 'build-atlas':
        build_atlas_cli(args.config, args.sweep_dir)
    elif args.command == 'fit-surrogate':
        fit_surrogate_cli(args.atlas)
    elif args.command == 'select-target':
        select_target_cli(args.atlas, args.surrogate, args.output, args.config)
    elif args.command == 'train':
        train_cli(args.config, args.mode, args.selected_target)
    elif args.command == 'evaluate':
        evaluate_cli(args.config, args.checkpoint)


if __name__ == '__main__':
    main()
