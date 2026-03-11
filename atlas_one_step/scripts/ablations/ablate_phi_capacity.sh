#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.train train=decoupled model=unet_small
python -m atlas_one_step.cli.train train=decoupled model=unet_base
python -m atlas_one_step.cli.train train=decoupled model=unet_large
