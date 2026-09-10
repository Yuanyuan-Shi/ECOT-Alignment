# ECoT-Lite Move-alignment feasibility experiment

Run all commands from `/home/exx/Projects/ECOT-Alignment`.

```bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate /home/exx/.conda/envs/openvla
export PYTHONPATH=/home/exx/Projects/ECOT-Alignment:/home/exx/Projects/OpenVLA/LIBERO:/tmp/vq-bet
export HF_HOME=/home/exx/Projects/ECOT-Alignment/artifacts/hf-cache
export PRISMATIC_DATA_ROOT=/home/exx/Projects/ECOT-Alignment/data
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
```

If `/tmp/vq-bet` is absent, restore the repository's VQ-VAE dependency first:

```bash
git clone https://github.com/jayLEE0301/vq_bet_official.git /tmp/vq-bet
```

## Training

The defaults reproduce the completed run: 10 epochs, 200 optimizer steps,
LoRA rank 16, and Move-loss weight 0.1.

```bash
python vla-scripts/finetune_move_alignment.py
```

## Three-trial paired evaluation

```bash
export MUJOCO_GL=egl
python experiments/robot/libero/run_libero_eval.py \
  --model_family prismatic \
  --pretrained_checkpoint artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt \
  --task_suite_name libero_90 \
  --task_ids 0,3,9,18,35,46,73,81,84 \
  --episodes_per_task 3 --seed 20261004 \
  --center_crop False --use_wrist_image False --use_cot True \
  --num_open_loop_steps 10 --n_procs_per_gpu 3 \
  --enable_alignment_evaluator True --alignment_cosine_similarity True \
  --alignment_save_frames False \
  --alignment_output_dir experiments/robot/libero/results/move-alignment-eval-pretrained-3trial-seed20261004 \
  --output_dir move-alignment-eval-pretrained-3trial-seed20261004 \
  --use_wandb False --run_id_note move-alignment-pretrained-3trial

python experiments/robot/libero/run_libero_eval.py \
  --model_family prismatic \
  --pretrained_checkpoint runs/ecot-move-alignment/ecot-lite-libero9-move-lora-r16-seed20260904/checkpoints/step-000200-move-lora.pt \
  --task_suite_name libero_90 \
  --task_ids 0,3,9,18,35,46,73,81,84 \
  --episodes_per_task 3 --seed 20261004 \
  --center_crop False --use_wrist_image False --use_cot True \
  --num_open_loop_steps 10 --n_procs_per_gpu 3 \
  --enable_alignment_evaluator True --alignment_cosine_similarity True \
  --alignment_save_frames False \
  --alignment_output_dir experiments/robot/libero/results/move-alignment-eval-finetuned-3trial-seed20261004 \
  --output_dir move-alignment-eval-finetuned-3trial-seed20261004 \
  --use_wandb False --run_id_note move-alignment-finetuned-3trial
```

## Comparison report

```bash
python experiments/robot/libero/move_alignment_comparison.py \
  --pretrained-root experiments/robot/libero/results/move-alignment-eval-pretrained-3trial-seed20261004 \
  --fine-tuned-root experiments/robot/libero/results/move-alignment-eval-finetuned-3trial-seed20261004 \
  --output-json analysis_results/move_alignment_comparison.json \
  --output-html analysis_results/move_alignment_comparison.html \
  --review-report analysis_results/index.html \
  --base-seed 20261004
```

## Round-2 Move-only human-review audit

This uses a new one-trial-per-task seed schedule and the untouched pretrained
checkpoint. It records frames and raw state transitions for all 90 tasks.

```bash
export MUJOCO_GL=egl
python experiments/robot/libero/run_libero_eval.py \
  --model_family prismatic \
  --pretrained_checkpoint artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt \
  --task_suite_name libero_90 \
  --episodes_per_task 1 --max_tasks 90 --seed 20261101 \
  --center_crop False --use_wrist_image False --use_cot True \
  --num_open_loop_steps 10 --n_procs_per_gpu 6 \
  --enable_alignment_evaluator True --alignment_cosine_similarity True \
  --alignment_save_frames True \
  --alignment_output_dir experiments/robot/libero/results/second-round-pretrained-libero90-seed20261101 \
  --output_dir second-round-pretrained-libero90-seed20261101 \
  --use_wandb False --run_id_note second-round-move-review

python experiments/robot/libero/move_only_review_report.py \
  --run-root experiments/robot/libero/results/second-round-pretrained-libero90-seed20261101 \
  --output-json analysis_results/round2_move_review.json \
  --review-report analysis_results/index.html
```

## LIBERO-90 Move-loss weight sweep (completed 2026-09-05)

The three runs use the frozen 90-episode/1,557-query artifact, the same
task-stratified split and seed, and 500 optimizer steps. Only the Move-loss
weight and run name differ.

```bash
for spec in 01:0.1 03:0.3 05:0.5; do
  suffix="${spec%%:*}"
  weight="${spec#*:}"
  python vla-scripts/finetune_move_alignment.py \
    --checkpoint artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt \
    --annotations analysis_results/annotations/round2_move_labels_effective_90episodes_2026-09-05.json \
    --audit_queries experiments/robot/libero/results/second-round-pretrained-libero90-seed20261101/alignment_queries.jsonl \
    --run_name "ecot-lite-libero90-move-lora-r16-lambda${suffix}-seed20261102" \
    --wandb_project ecot-move-alignment --wandb_mode online \
    --seed 20261102 --validation_fraction 0.2 \
    --epochs 10 --batch_size 8 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --weight_decay 0 \
    --move_loss_weight "$weight" --move_alignment_threshold 0.5 \
    --lora_rank 16 --lora_alpha 16 --lora_dropout 0.05 \
    --save_every_epochs 100 --max_optimizer_steps 500 \
    --expected_query_count 1557 --expected_task_count 90
done
```

### Paired 90-task evaluation

Each invocation uses two trials per task. The runner assigns seed
`20261201 + task_id * 2 + trial_index` and LIBERO initial-state index
`trial_index`, so all checkpoints receive identical paired conditions.

```bash
declare -A checkpoints=(
  [pretrained]="artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt"
  [lambda01]="runs/ecot-move-alignment/ecot-lite-libero90-move-lora-r16-lambda01-seed20261102/checkpoints/step-000500-move-lora.pt"
  [lambda03]="runs/ecot-move-alignment/ecot-lite-libero90-move-lora-r16-lambda03-seed20261102/checkpoints/step-000500-move-lora.pt"
  [lambda05]="runs/ecot-move-alignment/ecot-lite-libero90-move-lora-r16-lambda05-seed20261102/checkpoints/step-000500-move-lora.pt"
)
for name in pretrained lambda01 lambda03 lambda05; do
  output="move-sweep-eval-${name}-90task-2trial-seed20261201"
  python experiments/robot/libero/run_libero_eval.py \
    --model_family prismatic --pretrained_checkpoint "${checkpoints[$name]}" \
    --task_suite_name libero_90 --episodes_per_task 2 --max_tasks 90 --seed 20261201 \
    --center_crop False --use_wrist_image False --use_cot True \
    --num_open_loop_steps 10 --n_procs_per_gpu 6 \
    --enable_alignment_evaluator True --alignment_cosine_similarity True \
    --alignment_save_frames False \
    --alignment_output_dir "experiments/robot/libero/results/$output" \
    --output_dir "$output" --use_wandb False --run_id_note "$output"
done
```

### Sweep comparison and report insertion

```bash
python experiments/robot/libero/move_alignment_sweep_comparison.py \
  --model pretrained=experiments/robot/libero/results/move-sweep-eval-pretrained-90task-2trial-seed20261201 \
  --model lambda01=experiments/robot/libero/results/move-sweep-eval-lambda01-90task-2trial-seed20261201 \
  --model lambda03=experiments/robot/libero/results/move-sweep-eval-lambda03-90task-2trial-seed20261201 \
  --model lambda05=experiments/robot/libero/results/move-sweep-eval-lambda05-90task-2trial-seed20261201 \
  --display-name pretrained='Untouched pretrained' \
  --display-name lambda01='Move fine-tuned (lambda=0.1)' \
  --display-name lambda03='Move fine-tuned (lambda=0.3)' \
  --display-name lambda05='Move fine-tuned (lambda=0.5)' \
  --wandb-run lambda01=https://wandb.ai/yus047-/ecot-move-alignment/runs/skyk898d \
  --wandb-run lambda03=https://wandb.ai/yus047-/ecot-move-alignment/runs/1vcjl54d \
  --wandb-run lambda05=https://wandb.ai/yus047-/ecot-move-alignment/runs/oba3lh09 \
  --base-seed 20261201 --trials-per-task 2 \
  --output-json analysis_results/move_alignment_sweep_comparison.json \
  --output-html analysis_results/move_alignment_sweep_comparison.html \
  --review-report analysis_results/index.html
```

Completed paired results (2026-09-05):

| Checkpoint | Success | Mean Move cosine | Overall Move mismatch | Successful-episode pooled mismatch | Failed-episode pooled mismatch |
|---|---:|---:|---:|---:|---:|
| Pretrained | 156/180 (86.7%) | 0.682 | 624/3256 (19.2%) | 313/2296 (13.6%) | 311/960 (32.4%) |
| lambda 0.1 | 148/180 (82.2%) | 0.687 | 712/3459 (20.6%) | 259/2179 (11.9%) | 453/1280 (35.4%) |
| lambda 0.3 | 152/180 (84.4%) | 0.708 | 678/3556 (19.1%) | 309/2436 (12.7%) | 369/1120 (32.9%) |
| lambda 0.5 | 143/180 (79.4%) | 0.712 | 714/3710 (19.2%) | 233/2230 (10.4%) | 481/1480 (32.5%) |
