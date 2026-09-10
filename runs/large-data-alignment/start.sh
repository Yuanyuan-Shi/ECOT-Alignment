#!/usr/bin/env bash
set -euo pipefail
cd /home/exx/Projects/ECOT-Alignment
source runs/round3/environment.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=online
exec python -u runs/large-data-alignment/launch.py
