#!/usr/bin/env bash
set -euo pipefail
cd /home/exx/Projects/ECOT-Alignment
source runs/round3/environment.sh
suffix="$1"
output="round3-move-eval-lambda${suffix}-90task-2trial-seed20261201"
python -u experiments/robot/libero/run_libero_eval.py \
  --model_family prismatic \
  --pretrained_checkpoint "runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambda${suffix}-seed20261102/checkpoints/step-000200-move-lora.pt" \
  --task_suite_name libero_90 --episodes_per_task 2 --max_tasks 90 --seed 20261201 \
  --center_crop False --use_wrist_image False --use_cot True \
  --num_open_loop_steps 10 --n_procs_per_gpu 6 \
  --enable_alignment_evaluator True --alignment_cosine_similarity True \
  --alignment_save_frames False \
  --alignment_output_dir "experiments/robot/libero/results/$output" \
  --output_dir "$output" --use_wandb False --run_id_note "$output"
