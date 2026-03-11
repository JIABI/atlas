#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.select_target selection=heuristic
python -m atlas_one_step.cli.select_target selection=surrogate_guided
