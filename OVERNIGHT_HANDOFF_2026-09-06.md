# Overnight failure-focused experiments — September 6, 2026

## Latest user decision (supersedes earlier lambda reversals)

Run **four experiments overnight**: failure-weight3 and balanced sampling,
each at lambda=0.5 and lambda=1.0. All are fresh independent runs from the
untouched pretrained ECoT-Lite MiniVLA checkpoint. Keep the exact Round 3
seed, split, optimizer, learning rate, warmup, LoRA, thresholded hinge loss,
batch 32, and 200 optimizer updates.

## Active jobs and sequence

1. Lambda=.5 failure-weight3: training, then paired evaluation.
2. Lambda=.5 balanced sampling: queued, then paired evaluation.
3. Fresh lambda=1 failure-weight3: waits for both .5 evaluations, then trains.
4. Fresh lambda=1 balanced sampling: trains after the weighted lambda=1 run.

Within each pair, both trainings happen first, followed by both evaluations.
Each evaluation is the same 90 tasks × 2 trials (180 episodes), base seed
20261201 and identical initial-state indices. Four evaluations = 720 episodes.

Detached controllers survive the chat turn:

- Lambda=.5: `runs/round4/launch.py`. Live PID: `runs/round4/launcher.pid`.
  Live status: `runs/round4/status.json`.
- Lambda=1: `runs/round4-lambda1/launch.py`.
  Live PID: `runs/round4-lambda1/launcher.pid`.
  Live status: `runs/round4-lambda1/status.json`.

Do not duplicate active jobs or manually touch their `start-*` GPU gate files.

## Reproduction and paths

- Lambda=.5 commands, checkpoint paths and W&B links: `runs/round4/COMMANDS.md`.
- Lambda=1 commands and checkpoint paths: `runs/round4-lambda1/COMMANDS.md`.
- Each run directory has `training_config.json`, `split_manifest.json`,
  `wandb_run_url.txt`, and unsmoothed `metrics.jsonl` when training begins.
- Native checkpoints are `checkpoints/step-000200-move-lora.pt` under:
  - `runs/ecot-move-alignment/round4-lambda05-restart-failure-weight3`
  - `runs/ecot-move-alignment/round4-lambda05-restart-balanced-sampling`
  - `runs/ecot-move-alignment/round4-lambda1-restart-failure-weight3`
  - `runs/ecot-move-alignment/round4-lambda1-restart-balanced-sampling`

## Report

http://127.0.0.1:8000/analysis_results/index.html#round3-move-alignment-comparison

All four appear in the shared comparison. Status and W&B links refresh at
phase transitions. Final metrics are inserted only after validating the exact
180-episode paired schedule. Prior pretrained and Round 3 results are retained.
The builder merges the three independent status files with serialized report
writes. Refresh the browser to see changes.

## Preserved superseded attempts

- Original lambda=.5 jobs were canceled before training when the user first
  changed lambda. Their status/logs are in `runs/round4/superseded-lambda05`.
- The first lambda=1 weighted job was interrupted at logged step 44, and its
  balanced counterpart before training, when the user reverted to .5.
  Their status/logs are in `runs/round4/superseded-lambda1`.
- The latest instruction requests both lambdas. The replacement lambda=1
  pair starts from pretrained; it does not resume the interrupted 44 steps.
- Old W&B runs are marked superseded. Their directories are retained.

## Verification and recovery

Weighted loss is `sum(w*h)/sum(w)`, only on Move, with successful weight 1
and failure weight 3. Balanced sampling has unit weights and exactly 16/16
queries per full batch; validation distribution stays natural. The unchanged
split has 925 successful/320 failed training queries and 232/80 validation.
Tests and small full-model finite-gradient probes passed. Existing completed
results and train/validation IDs were checked for preservation.

Controllers record failures and continue other independent jobs. They reject
existing output directories. If a controller stops unexpectedly, inspect its
logs, live processes, and status before recovery; do not blindly restart a
script that would duplicate or overwrite work. Current live status files take
precedence over the snapshots in this handoff.

## Saved before monitor-off: 2026-09-06T22:57:57.991441-07:00

Machine-readable snapshot: `OVERNIGHT_STATUS_SNAPSHOT_2026-09-06.json`.
Live status files and logs remain authoritative as work continues overnight.

- Round 4 lambda=0.5 — failure-weight3: training; last logged training step 121/200. W&B: https://wandb.ai/yus047-/ecot-move-alignment/runs/foxogq97.
- Round 4 lambda=0.5 — balanced-sampling: waiting for GPU slot; last logged training step 0/200. W&B: https://wandb.ai/yus047-/ecot-move-alignment/runs/jb9v5bbl.
- Round 4 lambda=1.0 — failure-weight3: waiting for GPU slot; last logged training step 0/200. W&B: https://wandb.ai/yus047-/ecot-move-alignment/runs/umvh91x5.
- Round 4 lambda=1.0 — balanced-sampling: waiting for GPU slot; last logged training step 0/200. W&B: https://wandb.ai/yus047-/ecot-move-alignment/runs/nobmut2c.

## Ethernet disconnection or reboot

All configurations, splits, raw training metrics, completed checkpoints,
rollout logs, and the HTML report are saved locally under this repository.
W&B retains local run files under `wandb/`; an internet outage can delay
online syncing. Check local logs/status rather than relying only on W&B.

The temporary VQ dependency has been copied to `artifacts/vq-bet`, and
`runs/round3/environment.sh` now points there, so rebooting does not remove
that dependency. Model caches are already under `artifacts/hf-cache`.

Turning off only the monitor leaves computation running. A computer restart
stops the queues; they are not configured to restart at boot.

**Checkpoint limitation:** these training scripts save the final adapter and
merged checkpoint after 200 optimizer updates. They do not save resumable
optimizer state during a run. If reboot interrupts training, that unfinished
run must restart from the untouched pretrained checkpoint for 200 updates;
the saved metrics are a record of progress, not a resumable model checkpoint.
Completed training should be reused for evaluation.

### Tomorrow's restart procedure

1. Open this repository and give the assistant the resume instruction below.
2. Inspect both live status files, each run's `result.json`, checkpoint files,
   and evaluation outputs. A saved PID may be stale after reboot; verify the
   command/process, not only the PID number.
3. Keep completed 200-step checkpoints and all completed paired evaluations.
4. Archive interrupted run directories and logs before restarting any missing
   training. The launchers deliberately reject existing outputs; do not blindly
   rerun a whole launcher or delete previous results.
5. Preserve an interrupted evaluation's outputs in a separate archive and rerun
   only that checkpoint's 180 paired episodes if its merge was incomplete.
6. Rebuild the report from the reconciled live status files. If the local HTTP
   server stopped, run `python -m http.server 8000 --bind 127.0.0.1` from the
   repository root.

Resume instruction:

> Read OVERNIGHT_HANDOFF_2026-09-06.md and the current status files in
> runs/round4 and runs/round4-lambda1. Continue the four failure-focused
> experiments at lambda=.5 and lambda=1 for both methods. Reuse completed
> checkpoints/evaluations. Preserve interrupted outputs before restarting only
> unfinished work. Restore detached queues and the local results server.
> Do not resume the superseded lambda=1 44-step run or overwrite prior results.

## Latest instruction: pause until 7 AM

At approximately 23:47 PDT Sept 6, user requested quiet overnight and resume
at 07:00 PDT Sept 7. Both controller process groups (88621 and 89260),
including all evaluation workers and waiting training jobs, are suspended
with SIGSTOP. Their in-memory progress is preserved while the computer stays
on. GPU utilization fell to 3%.

Local systemd user timer `ecot-resume-20260907-0700.timer` is scheduled for
2026-09-07 07:00:00 America/Los_Angeles. It runs
`runs/resume_paused_experiments.py`, which validates boot ID and process
start times before sending SIGCONT and refreshing the report.
Pause state: `runs/overnight_pause.json`. Check `resumed_at` to determine
whether the scheduled resumption occurred. This supersedes the instruction
to compute continuously overnight. No internet is needed for the timer.

Keep the computer on and awake for the scheduled resume. A reboot loses the
suspended in-memory processes; follow the recovery procedure above instead.
The timer is transient and is not a boot-time job-restoration mechanism.
