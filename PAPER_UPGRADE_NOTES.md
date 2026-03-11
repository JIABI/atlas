# Paper-Level Upgrade Notes

## What changed

This upgrade promotes the CIFAR-10 one-step target-geometry code from MVP to a stronger paper baseline while preserving the original CLI and `tiny_unet` debug path.

- Added `paper_unet` with residual GroupNorm blocks, sinusoidal time embeddings, multi-scale down/up paths, and optional bottleneck attention.
- Trainer now supports AdamW workflows, EMA for model and phi-map, optional AMP, periodic checkpointing (`last.pt`, `best.pt`, and step checkpoints), richer evaluation metrics, and sample grid export.
- Corruption now supports optional discrete time sampling via `num_time_samples` while preserving `t_min`/`t_max` scaling.
- Target sweep sampling now supports `alpha_min`, `alpha_max`, `custom_alphas`, and keeps default compatibility.
- Data loader threading now includes `num_workers`, `pin_memory`, and safe persistent worker handling.
- Losses now support configurable `loss_kind` (currently `mse`) through prediction and semantic terms.
- Pathology probes were expanded with support deviation aggregate, normal burden, covariance conditioning, relative shift, prediction sensitivity, and combined pathology score.
- Atlas summary loading is now robust plain JSON parsing, and atlas export gracefully falls back when parquet is unavailable.
- Paper configs were switched to `paper_unet` and updated with stronger defaults.

## Recommended entry points

- Run line-family sweep:
  - `python -m atlas_one_step.cli run-sweep --config configs/cifar10_line_paper.yaml`
- Build atlas from sweep outputs:
  - `python -m atlas_one_step.cli build-atlas --config configs/cifar10_line_paper.yaml`
- Train with paper train config:
  - `python -m atlas_one_step.cli train --config configs/cifar10_train_paper.yaml --mode coupled`

## Validation notes

- Existing command structure remains unchanged (`run-sweep`, `build-atlas`, `train`, etc.).
- `tiny_unet` remains supported for smoke/debug compatibility.
- Stronger configs use `paper_unet` and can still run on CPU-only environments (with AMP disabled automatically unless CUDA is available).
