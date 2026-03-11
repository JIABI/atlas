#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.run_sweep atlas=sweep_simplex dataset=cifar10 target=simplex_x0_u_r
