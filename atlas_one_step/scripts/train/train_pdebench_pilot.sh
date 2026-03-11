#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.train train=coupled dataset=pdebench_darcy
python -m atlas_one_step.cli.train train=decoupled selection=surrogate_guided dataset=pdebench_darcy
