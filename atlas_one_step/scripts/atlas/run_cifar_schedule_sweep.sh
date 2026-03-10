#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.run_sweep atlas=sweep_schedule dataset=cifar10 target=scheduled
