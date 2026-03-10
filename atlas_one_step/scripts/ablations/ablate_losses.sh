#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.train train=ablation train.mu=0 train.tau=0
python -m atlas_one_step.cli.train train=ablation train.mu=0.1 train.tau=0
python -m atlas_one_step.cli.train train=ablation train.mu=0 train.tau=0.1
python -m atlas_one_step.cli.train train=ablation train.mu=0.1 train.tau=0.1
