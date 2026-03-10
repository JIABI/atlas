# ATLAS: A Target-Pathology Atlas for One-Step High-Dimensional Generation

ATLAS is a runnable research codebase for one-step generation under fixed corruption, target-family sweeps, pathology probes, atlas construction, diagnostics-guided selection, decoupled training, and paper artifact export.

## 1) Scientific objective
Under a fixed forward process, different targets induce different inverse problems. ATLAS quantifies those pathologies and uses them to guide prediction-target selection for one-step training.

## 2) Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 3) Dataset preparation
- **CIFAR-10**: auto-download via torchvision.
- **ImageNet-64 / LSUN / UCF101 / PDEBench**: manual licensed download and conversion using tools:
```bash
python tools/dataset_prep/prepare_imagenet64.py --root data/imagenet64
python tools/dataset_prep/prepare_lsun.py --root data/lsun
python tools/dataset_prep/prepare_pdebench.py --root data/pdebench/darcy
python tools/dataset_prep/prepare_ucf101.py --root data/ucf101
```

## 4) Smoke test
```bash
python -m atlas_one_step.cli.smoke_test
bash scripts/smoke/smoke_all.sh
```

## 5) Atlas construction
```bash
python -m atlas_one_step.cli.run_sweep atlas=sweep_line dataset=cifar10 target=line_x0_u
python -m atlas_one_step.cli.run_sweep atlas=sweep_simplex dataset=cifar10 target=simplex_x0_u_r
python -m atlas_one_step.cli.run_sweep atlas=sweep_schedule dataset=cifar10 target=scheduled
python -m atlas_one_step.cli.compute_probes
python -m atlas_one_step.cli.build_atlas atlas=build_atlas
python -m atlas_one_step.cli.fit_surrogate atlas=fit_surrogate
python -m atlas_one_step.cli.select_target selection=surrogate_guided
```

## 6) Training modes
- **Coupled**: prediction target and loss target are tied.
- **Unguided decoupled**: prediction family independent from semantic loss target, no atlas selection.
- **ATLAS-guided decoupled**: prediction family selected from atlas/surrogate diagnostics.

Run:
```bash
python -m atlas_one_step.cli.train train=coupled dataset=imagenet64
python -m atlas_one_step.cli.train train=decoupled selection=heuristic dataset=imagenet64
python -m atlas_one_step.cli.train train=decoupled selection=surrogate_guided dataset=imagenet64
```

## 7) Evaluation
```bash
python -m atlas_one_step.cli.evaluate eval=metrics
python -m atlas_one_step.cli.evaluate eval=tail
python -m atlas_one_step.cli.evaluate eval=resolution_scaling
python -m atlas_one_step.cli.evaluate eval=pilot
```

## 8) Paper artifacts
```bash
python -m atlas_one_step.cli.make_figures eval=paper
bash scripts/paper/make_all_figures.sh
bash scripts/paper/make_all_tables.sh
bash scripts/paper/export_paper_artifacts.sh
```

## 9) Result schema
Each sweep/train run writes a structured record with experiment id, target/prediction parameters, trainability, quality, tail, pathology, and artifact pointers (JSON + atlas JSONL/parquet fallback).

## 10) Troubleshooting
- Missing `torchvision`: install dependencies from `requirements.txt`.
- Restricted datasets: ensure path matches `configs/dataset/*.yaml`.
- Parquet export failure: JSONL fallback is always written to `outputs/atlas/atlas.jsonl`.

## 11) Runtime notes
- Smoke mode: CPU minutes.
- CIFAR sweeps: tens of minutes.
- Large-scale LSUN/ImageNet and ablations: multi-GPU hours to days.
