#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.train train=coupled dataset=imagenet64
