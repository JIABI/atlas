#!/usr/bin/env bash
set -euo pipefail
python -m atlas_one_step.cli.evaluate eval=pilot dataset=ucf101
