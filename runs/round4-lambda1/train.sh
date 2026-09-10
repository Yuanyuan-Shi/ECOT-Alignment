#!/usr/bin/env bash
set -euo pipefail
cd /home/exx/Projects/ECOT-Alignment
source runs/round3/environment.sh
name="$1"
shift
case "$name" in
  failure-weight3) failure_weight=3; balanced=False ;;
  balanced-sampling) failure_weight=1; balanced=True ;;
  *) echo "Unknown experiment: $name" >&2; exit 2 ;;
esac
python -u vla-scripts/finetune_move_alignment.py \
  --checkpoint artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt \
  --annotations analysis_results/annotations/round2_move_labels_effective_90episodes_2026-09-05.json \
  --audit_queries experiments/robot/libero/results/second-round-pretrained-libero90-seed20261101/alignment_queries.jsonl \
  --run_name "round4-lambda1-restart-${name}" \
  --wandb_project ecot-move-alignment --wandb_mode online \
  --seed 20261102 --validation_fraction 0.2 \
  --epochs 10 --batch_size 32 --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 --weight_decay 0 --warmup_steps 20 \
  --move_loss_type thresholded_squared_hinge \
  --failure_move_weight "$failure_weight" --balanced_sampling "$balanced" \
  --move_loss_weight 1.0 --move_alignment_threshold 0.5 \
  --lora_rank 16 --lora_alpha 16 --lora_dropout 0.05 \
  --save_every_epochs 100 --max_optimizer_steps 200 \
  --expected_query_count 1557 --expected_task_count 90 "$@"
