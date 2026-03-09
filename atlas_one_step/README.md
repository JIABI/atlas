
# ATLAS: A Target-Pathology Atlas for One-Step High-Dimensional Generation

## Overview
ATLAS is a production-oriented research codebase for exploring one-step inverse target families under fixed corruption processes and selecting robust decoupled training targets.

## Scientific goal
The core thesis is that one-step generation fails when induced inverse problems are pathological; this repo maps pathologies into an atlas and uses diagnostics-guided target selection.

## Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dataset prep
Use `tools/dataset_prep/*.py` scripts. Restricted datasets (ImageNet, LSUN, UCF101, PDEBench mirrors) require manual paths and licenses.

## Smoke tests
```bash
python -m atlas_one_step.cli.smoke_test
bash scripts/smoke/smoke_all.sh
```

## Atlas construction
```bash
python -m atlas_one_step.cli.run_sweep atlas=sweep_line dataset=cifar10 target=line_x0_u
python -m atlas_one_step.cli.compute_probes atlas_input=outputs/atlas/sweeps
python -m atlas_one_step.cli.build_atlas atlas=build_atlas
python -m atlas_one_step.cli.fit_surrogate atlas=fit_surrogate
```

## Main training
```bash
python -m atlas_one_step.cli.train train=coupled dataset=imagenet64
python -m atlas_one_step.cli.train train=decoupled selection=surrogate_guided dataset=imagenet64
```

## Evaluation
```bash
python -m atlas_one_step.cli.evaluate eval=metrics
python -m atlas_one_step.cli.make_figures eval=paper
```

## Coupled vs decoupled
Coupled uses a single prediction/semantic target. Decoupled predicts a selected target-side family then maps through `Phi_t` into fixed semantic loss-space.

## Atlas-guided selection
Use sweep diagnostics + probes to fit surrogate predictors; selection module proposes `eta` minimizing predicted pathology and maximizing trainability.

## Troubleshooting
- Ensure dataset paths in configs.
- Use `tools/checks/check_logs.py` for malformed outputs.
- If missing parquet deps, install `pyarrow`.

## Runtime notes
Smoke runs finish in minutes on CPU; full experiments require multi-GPU and can run for days.
