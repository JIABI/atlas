#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.train train=decoupled selection=heuristic dataset=imagenet64
