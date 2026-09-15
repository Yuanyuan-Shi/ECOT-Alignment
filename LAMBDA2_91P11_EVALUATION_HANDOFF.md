# MiniVLA Joint FT λ=2: exact 270-episode evaluation handoff

This is the complete handoff for the MiniVLA joint fine-tuning λ=2 checkpoint reported at **246/270 task successes (91.11%)**. A second run on the original workstation reproduced all 270 episode outcomes exactly.

## Immutable inputs

| Item | Value |
|---|---|
| Evaluator code commit | [`5403dc2828ed4a65c3ff366e75ec86fb839cd8e7`](https://github.com/Yuanyuan-Shi/ECOT-Alignment/tree/5403dc2828ed4a65c3ff366e75ec86fb839cd8e7) |
| Evaluator | `experiments/robot/libero/run_libero_eval.py` |
| Checkpoint | [`minivla_joint_ft_lambda2_seed20260910_step2000`](https://huggingface.co/yyshi0619/ECOT-Alignment-Checkpoints/tree/main/minivla_joint_ft_lambda2_seed20260910_step2000) |
| Checkpoint filename | `step-002000-move-lora.pt` |
| Checkpoint size | `5,554,852,935` bytes |
| Checkpoint SHA256 | `0b0ef328c8d4f5478e6da25fde7639c688fdd1ac3b041c60ea06476424b52b1d` |
| Case schedule | [`libero90_matched_evaluation_manifest.csv`](libero90_matched_evaluation_manifest.csv) |
| Case-schedule SHA256 | `8c2fcb9c3c5d92be38b9ed787f55d212b2a8aba19065f499585a775351ded564` |
| LIBERO commit | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |

The Hugging Face directory contains the checkpoint, `config.json`, and `dataset_statistics.json`. The native loader requires this local layout:

```text
<RUN_DIR>/
├── config.json
├── dataset_statistics.json
└── checkpoints/
    └── step-002000-move-lora.pt
```

Do not load a LoRA adapter on top of this checkpoint. It is the complete native model with the LoRA weights already merged.

## Exact launch script

Set `REPO`, `RUN_DIR`, and `REPORT_DIR` for the local machine. `RUN_DIR` must use the layout above. The working directory is significant because the action tokenizer resolves the VQ-VAE files from the MiniVLA reference directory.

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO=/absolute/path/to/ECOT-Alignment
PYTHON="$REPO/pi05_policy_training/data/minivla_venv/bin/python"
RUN_DIR=/absolute/path/to/minivla_joint_ft_lambda2_seed20260910_step2000
CHECKPOINT="$RUN_DIR/checkpoints/step-002000-move-lora.pt"
REPORT_DIR=/absolute/path/to/lambda2_270_evaluation
TASK_IDS="$($PYTHON -c 'print(",".join(map(str, range(90))))')"

echo "0b0ef328c8d4f5478e6da25fde7639c688fdd1ac3b041c60ea06476424b52b1d  $CHECKPOINT" | sha256sum -c -

export PYTHONPATH="$REPO:$REPO/pi05_policy_training/openpi/third_party/libero"
export LIBERO_CONFIG_PATH="$REPO/pi05_policy_training/configs/libero"
export PRISMATIC_DATA_ROOT="$REPO/pi05_policy_training/data/rlds"
export HF_HOME="$REPO/pi05_policy_training/data/minivla_reference/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3

cd "$REPO/pi05_policy_training/data/minivla_reference"

"$PYTHON" "$REPO/experiments/robot/libero/run_libero_eval.py" \
  --model_family prismatic \
  --pretrained_checkpoint "$CHECKPOINT" \
  --task_suite_name libero_90 \
  --task_ids "$TASK_IDS" \
  --num_trials_per_task 3 \
  --episodes_per_task 3 \
  --initial_state_offset 13 \
  --seed 7 \
  --center_crop False \
  --use_wrist_image False \
  --use_cot True \
  --num_open_loop_steps 10 \
  --n_procs_per_gpu 6 \
  --enable_alignment_evaluator True \
  --alignment_cosine_similarity True \
  --alignment_save_frames False \
  --use_wandb False \
  --alignment_output_dir "$REPORT_DIR" \
  --output_dir minivla-lambda2-90task-3trial-seed7-state13 \
  --run_id_note minivla-lambda2-90task-3trial-seed7-state13
```

`n_procs_per_gpu=6` only partitions the cases among six workers. It does not change the case assignment, model decoding, or result definition.

## Effective saved configuration

```json
{
  "model_family": "prismatic",
  "task_suite_name": "libero_90",
  "task_ids": "all integers 0 through 89",
  "num_trials_per_task": 3,
  "episodes_per_task": 3,
  "initial_state_offset": 13,
  "initial_state_indices": [13, 14, 15],
  "seed": 7,
  "seed_schedule": "7 + task_id * 3 + trial",
  "episode_seeds": "integers 7 through 276",
  "center_crop": false,
  "use_wrist_image": false,
  "obs_history": 1,
  "use_cot": true,
  "plans": false,
  "unnorm_key": "libero_lm_90",
  "num_open_loop_steps": 10,
  "predicted_action_horizon": 10,
  "executed_actions_before_replan": 10,
  "num_steps_wait": 10,
  "max_episode_steps": null,
  "effective_post_settle_horizon": 400,
  "load_in_4bit": false,
  "load_in_8bit": false,
  "enable_alignment_evaluator": true,
  "alignment_cosine_similarity": true,
  "alignment_save_frames": false,
  "motion_dead_band": 0.003,
  "gripper_open_fraction": 0.5,
  "decoding": "greedy; do_sample=False",
  "inference_precision": "bfloat16"
}
```

For every episode, the evaluator calls `env.seed(episode_seed)`, resets the environment, and then applies `task_suite.get_task_init_states(task_id)[initial_state_index]`. Verify each initial-state byte hash against the case manifest.

The MiniVLA VQ decoder returns a `10 × 7` action chunk. The evaluator takes `actions[0]`, executes all 10 rows before replanning, maps the gripper from `[0,1]` to `[-1,+1]`, binarizes it, and then inverts its sign for LIBERO. Changing any of these details changes task success.

## Mismatch definition used in the paper table

The paper table pools **all policy queries**. For directional MOVE commands, mismatch means commanded-translation cosine similarity below `0.5`. Gripper-only, rotation-only, missing, or empty XYZ MOVE commands are assigned expected translation `[0,0,0]` and included as implicit-stop mismatches. Do not remove these queries from the denominator.

The alternative refined eligible-query score is useful diagnostically but is not the table metric:

| Score | Result |
|---|---:|
| All-query mismatch used in the table | **360/4,463 = 8.07%** |
| Refined eligible-query mismatch | 228/4,331 = 5.26% |

## Reproduction checks

Both runs on the original workstation produced the same case-level outcomes:

| Result | Original run | Replication |
|---|---:|---:|
| Successful episodes | 246/270 | 246/270 |
| Task success | 91.11% | 91.11% |
| Policy queries | 4,463 | 4,463 |
| All-query mismatches | 360 | 360 |
| Case-outcome differences | — | 0 |

The six worker success totals must be `[39, 44, 41, 41, 44, 37]`, with 45 episodes per worker. If these differ, compare checkpoint SHA256, initial-state hashes, image preprocessing, `unnorm_key`, greedy decoding, action-chunk extraction, gripper processing, and the predict-10/execute-10 controller before comparing aggregate rates.

The environment used for the reproduced run included Python 3.10, PyTorch `2.8.0+cu128`, Transformers `4.40.1`, NumPy `1.26.4`, MuJoCo `3.2.3`, robosuite `1.4.1`, LIBERO `0.1.0`, PEFT `0.11.1`, and timm `0.9.10`.
