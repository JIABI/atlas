#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.run_sweep target=line_x0_u
python -m atlas_one_step.cli.run_sweep target=scheduled atlas=sweep_schedule
