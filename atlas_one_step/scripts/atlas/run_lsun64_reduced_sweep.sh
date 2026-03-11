#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.run_sweep atlas=sweep_line dataset=lsun_bedroom256 target=line_x0_u dataset.resolution=64
