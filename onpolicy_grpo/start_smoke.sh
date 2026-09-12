#!/usr/bin/env bash
set -euo pipefail
cd /home/exx/Projects/ECOT-Alignment
source runs/round3/environment.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
exec python -u -m onpolicy_grpo.smoke --config onpolicy_grpo/config_smoke_g4.json
