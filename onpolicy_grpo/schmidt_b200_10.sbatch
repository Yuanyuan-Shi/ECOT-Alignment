#!/usr/bin/env bash
#SBATCH --job-name=ecot-grpo-10
#SBATCH --account=ssci-ys-shigroup
#SBATCH --partition=b200
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task=120
#SBATCH --mem=1000G
#SBATCH --time=12:00:00
#SBATCH --output=onpolicy_grpo/b200-grpo-10-%j.log
#SBATCH --signal=B:USR1@180
set -euo pipefail
module load cuda/12.8.1
: "${ECOT_REPO:?Set ECOT_REPO}"
: "${ECOT_GRPO_RUN_DIR:?Set ECOT_GRPO_RUN_DIR}"
: "${ECOT_BASE_CHECKPOINT:?Set ECOT_BASE_CHECKPOINT}"
: "${ECOT_PYTHON:?Set ECOT_PYTHON}"
cd "$ECOT_REPO"
export PYTHONPATH="$ECOT_REPO:$ECOT_SCRATCH/deps/LIBERO:$ECOT_REPO/artifacts/vq-bet:$ECOT_SCRATCH/deps/dlimp_openvla"
export HF_HOME="$ECOT_REPO/artifacts/hf-cache"
export PRISMATIC_DATA_ROOT="$ECOT_REPO/artifacts"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi -L
"$ECOT_PYTHON" -m pytest -q onpolicy_grpo/tests
srun --kill-on-bad-exit=1 "$ECOT_PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  -m onpolicy_grpo.cluster_train \
  --config onpolicy_grpo/config_b200_10.json \
  --run-dir "$ECOT_GRPO_RUN_DIR"
