#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.train train=ablation train.tau=0
python -m atlas_one_step.cli.train train=ablation train.tau=0.01
python -m atlas_one_step.cli.train train=ablation train.tau=0.1
