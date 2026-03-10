#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.train train=coupled dataset=lsun_cat256
python -m atlas_one_step.cli.train train=decoupled selection=surrogate_guided dataset=lsun_cat256
