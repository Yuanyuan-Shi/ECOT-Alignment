#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source runs/round3/environment.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -u -m onpolicy_grpo.local_train --config onpolicy_grpo/config_workstation_g4_10.json
