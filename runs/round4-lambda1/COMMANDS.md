# Additional Round 4 lambda=1 pair

These are fresh independent 200-step runs from the untouched pretrained
checkpoint, retaining all Round 4 settings except lambda=1.0. The interrupted
old lambda=1 runs are superseded and are not resumed.

Already submitted controller:

```bash
source runs/round3/environment.sh
python -u runs/round4-lambda1/launch.py
```

It waits for `runs/round4/status.json` to record completion of the lambda=.5
pair's training and evaluations before releasing these GPU gates.

Underlying commands (do not duplicate active jobs):

```bash
bash runs/round4-lambda1/train.sh failure-weight3
bash runs/round4-lambda1/train.sh balanced-sampling
bash runs/round4-lambda1/evaluate.sh failure-weight3
bash runs/round4-lambda1/evaluate.sh balanced-sampling
```

Checkpoint paths on completion:

- `runs/ecot-move-alignment/round4-lambda1-restart-failure-weight3/checkpoints/step-000200-move-lora.pt`
- `runs/ecot-move-alignment/round4-lambda1-restart-balanced-sampling/checkpoints/step-000200-move-lora.pt`

Status: `runs/round4-lambda1/status.json`. PID: `launcher.pid` in this folder.
Logs: `train-<method>.log`, `eval-<method>.log`, and `launcher.log`.
The report builder reads both Round 4 status files and preserves all prior
results in the shared table. No completed checkpoints or rollout outputs are
overwritten. See `runs/round4/COMMANDS.md` for detailed sampler/loss settings.

## Overnight continuation

See `OVERNIGHT_HANDOFF_2026-09-06.md` for all four runs and reboot recovery.

## W&B runs

- failure-weight3: https://wandb.ai/yus047-/ecot-move-alignment/runs/umvh91x5
- balanced-sampling: https://wandb.ai/yus047-/ecot-move-alignment/runs/nobmut2c
