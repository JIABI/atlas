#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.train train=decoupled selection=surrogate_guided dataset=imagenet64
