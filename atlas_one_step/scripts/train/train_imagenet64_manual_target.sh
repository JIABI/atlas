#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.train train=full_train dataset=imagenet64 target=line_x0_r
