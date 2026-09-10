# Round 3 threshold-aware Move alignment — 2026-09-06

Run from `/home/exx/Projects/ECOT-Alignment`.

The persistent launcher is already running. Do not launch duplicate jobs.

```bash
source runs/round3/environment.sh
python -u runs/round3/launch.py
```

The launcher trains these three independent runs sequentially on the one GPU:

```bash
bash runs/round3/train.sh 01 0.1 32 1
bash runs/round3/train.sh 03 0.3 32 1
bash runs/round3/train.sh 05 0.5 32 1
```

`train.sh` contains the full command and all explicit training arguments. Each
starts from `artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt`.
Run directories are created exclusively; existing directories cause an error.

Checkpoints (created when each training run completes):

- `runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambda01-seed20261102/checkpoints/step-000200-move-lora.pt`
- `runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambda03-seed20261102/checkpoints/step-000200-move-lora.pt`
- `runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambda05-seed20261102/checkpoints/step-000200-move-lora.pt`

Each run directory also contains `adapter-final`, `training_config.json`,
`split_manifest.json`, `step_plan.json`, `metrics.jsonl`, `wandb_run_url.txt`,
and, when finished, `result.json`.

After training, the launcher runs:

```bash
bash runs/round3/evaluate.sh 01
bash runs/round3/evaluate.sh 03
bash runs/round3/evaluate.sh 05
```

Each evaluation uses all 90 tasks, two trials each, base seed `20261201`,
episode seed `20261201 + task_id*2 + trial_index`, LIBERO initial-state index
`trial_index`, 10 open-loop steps, and six rollout processes. These are the
unchanged Round 2 evaluation settings. The completed untouched-pretrained
180-episode evaluation is reused as baseline. Three new evaluations total
540 episodes.

Evaluation roots:

- `experiments/robot/libero/results/round3-move-eval-lambda01-90task-2trial-seed20261201`
- `experiments/robot/libero/results/round3-move-eval-lambda03-90task-2trial-seed20261201`
- `experiments/robot/libero/results/round3-move-eval-lambda05-90task-2trial-seed20261201`

The report refresh command (also invoked automatically by the launcher) is:

```bash
python -m experiments.robot.libero.round3_move_comparison
```

Review at:
`http://127.0.0.1:8000/analysis_results/index.html#round3-move-alignment-comparison`

Standalone results:
`analysis_results/round3_move_alignment_comparison.html` and `.json`.
Existing report sections are preserved exactly. Incomplete evaluations are
not summarized as final results. Completed reports validate all 180 paired
(task, trial, seed) entries against the baseline and the expected schedule.

## Configuration and optimizer-step calculation

- Loss: `mean((relu(0.5-cosine)/1.5)**2)`; total is performance loss plus lambda
  times this loss. Reasoning vectors, cumulative translation, epsilon,
  soft VQ decode, and zero-vector cosine convention are unchanged.
- A stop query has cosine zero, penalty `1/9`, and no directional gradient.
- Dataset: the exact Round 2 artifact contains **1,557**, not 1,577, queries
  from 90 episodes. Its task-stratified split is unchanged: 1,245 training,
  312 validation, seed 20261102. Both ID lists were checked against Round 2.
- Per-device batch 32, accumulation 1. Full-model forward/backward probe
  passed at 57.956 GB peak allocated GPU memory; fallback was unnecessary.
- `ceil(1245/32) = 39` updates per epoch; the final batch has 29 queries,
  preserving the prior loader's inclusion of all examples.
- `5*39 + 5 = 200` optimizer updates. This processes `5*1245 + 5*32 = 6385`
  training queries, about 5.129 dataset passes. Epoch 6 is explicitly logged
  as partial. The configured ten-epoch ceiling does not truncate 200 steps.
- Warmup: 20 optimizer steps (10%), linear from `5e-6` to `1e-4`, followed by
  constant LR. Round 2 had constant `1e-4` with no warmup.
- Unchanged AdamW, zero weight decay, gradient clipping 1.0, LoRA rank/alpha
  16, dropout 0.05, target modules, BF16, gradient checkpointing, seed,
  query split, targets, and pretrained checkpoint.
- All raw batch measurements are logged, with separate optimizer-window,
  query-weighted epoch, and validation aggregates. Local JSONL retains the
  unsmoothed measurements. Bins include both counts and percentages.
- Training cosine is from soft-decoded predicted action chunks. Evaluation
  cosine is from observed end-effector displacement, as in Round 2.

## Monitoring and verification

```bash
cat runs/round3/status.json
cat runs/round3/launcher.pid
tail -n 20 runs/round3/train-lambda01.log
source runs/round3/environment.sh
python -m pytest -q tests/test_move_alignment_training.py tests/test_round3_move_comparison.py
```

The launcher runs in a detached session and survives this chat turn. Its
status file and the report gain each W&B URL when that run initializes.
Training/evaluation errors are recorded per run; other independent runs
continue. It does not automatically overwrite or restart failed outputs.

Verification before launch: threshold-boundary values, stronger negative
penalties, finite cosine/action gradients, zero-vector/stop behavior, full
batch-32 model backward pass, exact Round 2 split, weighted metrics, paired
schedule validation, and report preservation. The existing test suite passed
(87 passed, 1 skipped); the two added report tests also passed afterward.

The missing temporary VQ-VAE dependency was restored at `/tmp/vq-bet` using
the same repository specified in the Round 2 reproduction commands. Probe
outputs are retained in a separate `lambda-probe32` run directory; they are
not experiment checkpoints or W&B training runs.

## W&B runs

- lambda 0.1: https://wandb.ai/yus047-/ecot-move-alignment/runs/36utml0u
- lambda 0.3: https://wandb.ai/yus047-/ecot-move-alignment/runs/geob5kq5
- lambda 0.5: https://wandb.ai/yus047-/ecot-move-alignment/runs/xla4x26y

## Lambda 1.0 extension — launched 2026-09-06

The extension uses the same loss, data, split, batch 32, 200 updates, 20-step
warmup, learning rate, LoRA configuration, seed, and untouched checkpoint.
Only lambda changes to 1.0. It trains first, then evaluates the same 180
paired episodes and adds its results to the existing Round 3 comparison.

Already launched as a detached process:

```bash
source runs/round3/environment.sh
python -u runs/round3/launch_lambda1.py
```

Underlying commands (do not duplicate the active job):

```bash
bash runs/round3/train.sh 10 1.0 32 1
bash runs/round3/evaluate.sh 10
```

Checkpoint on completion:
`runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambda10-seed20261102/checkpoints/step-000200-move-lora.pt`

Logs: `runs/round3/train-lambda10.log`, `runs/round3/eval-lambda10.log`, and
`runs/round3/launcher-lambda1.log`. The shared `status.json` includes all
previous completed runs plus lambda 1.0. Original status was preserved in
`status_before_lambda1.json`; previous checkpoints and rollout results remain
unchanged. The active extension PID is in `launcher-lambda1.pid` and
`launcher.pid`.
