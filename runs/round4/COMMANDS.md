# Round 4 failure-focused Move alignment

Run from `/home/exx/Projects/ECOT-Alignment`.

Both jobs have been submitted. The detached controller waits for the active
Round 3 lambda=1 evaluation to finish, releases the two training jobs one at
a time, and then runs their paired evaluations. W&B initializes before each
training job's GPU gate, so its run is visible while waiting.

## Commands

Already-running controller (do not launch a duplicate):

```bash
source runs/round3/environment.sh
python -u runs/round4/launch.py
```

Underlying training commands, with all shared arguments expanded in `train.sh`:

```bash
bash runs/round4/train.sh failure-weight3
bash runs/round4/train.sh balanced-sampling
```

The controller supplies `--start_gate runs/round4/start-<name>` to serialize
GPU access. Do not manually create these gate files while another run owns
the GPU.

Automatic paired evaluations:

```bash
bash runs/round4/evaluate.sh failure-weight3
bash runs/round4/evaluate.sh balanced-sampling
```

Each evaluates 90 tasks, two trials per task, base seed 20261201, episode seed
`20261201 + task_id*2 + trial_index`, initial-state index `trial_index`, 10
open-loop actions, and six rollout processes. Settings are unchanged from
Round 3. Total new evaluation episodes: 360.

## Checkpoint paths (created at training completion)

- `runs/ecot-move-alignment/round4-lambda05-restart-failure-weight3/checkpoints/step-000200-move-lora.pt`
- `runs/ecot-move-alignment/round4-lambda05-restart-balanced-sampling/checkpoints/step-000200-move-lora.pt`

Both independently start from:
`artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt`.
Existing run directories are rejected, never overwritten.

## Exact configuration

Both use lambda 0.5; the existing normalized squared-hinge Move loss;
200 optimizer updates; physical/effective batch 32; accumulation 1; seed
20261102; 20 warmup updates to learning rate 1e-4 then constant LR; AdamW;
zero weight decay; clipping 1.0; LoRA rank/alpha 16, dropout .05, and the
same target modules, BF16, and gradient checkpointing as Round 3 lambda=.5.

The same 1,557-query artifact has 1,157 successful and 400 failed queries.
The identical query split has 1,245 training queries (925 successful/320
failed) and 312 validation queries (232 successful/80 failed). No validation
query participates in training sampling.

### Failure-weight3

Natural shuffled sampling is unchanged. Only the Move term is weighted:
`sum(w*h)/sum(w)`, w=1 for successful-episode queries and w=3 for failed.
Performance loss remains the unweighted model cross-entropy loss. Cosine,
bin, and mismatch metrics also remain unweighted.

39 optimizer updates per full epoch; the last natural batch has 29 queries.
200 updates = five full epochs plus five updates, or 6,385 query exposures.

### Balanced sampling

Unit Move weights, no additional failure weighting. Every batch contains
exactly 16 successful and 16 failed training queries. Each outcome pool is
shuffled independently and reused after exhaustion; this oversamples the
smaller failed pool. Batch order is shuffled deterministically. The sampler
also supports balanced physical microbatches of 8 (4+4) with accumulation 4.
The launched run uses physical batch 32.

Keep 39 optimizer updates per full epoch, using 1,248 draws instead of a
partial batch. 200 updates = 6,400 draws: exactly 3,200 successful and 3,200
failed queries. Epoch 6 is partial, as in Round 3. Validation uses the original
natural distribution for both experiments.

## Logging, report, and monitoring

Raw batch, optimizer-window, epoch, and validation logs retain all Round 3
metrics. Added successful/failed query counts make balanced sampling auditable.
Epoch summaries are query-weighted averages of the observed batch metrics;
for the weighted experiment, `move_loss` is the observed normalized weighted
batch objective. Local `metrics.jsonl` is unsmoothed.

```bash
cat runs/round4/status.json
tail -n 20 runs/round4/train-failure-weight3.log
tail -n 20 runs/round4/train-balanced-sampling.log
```

The shared report builder reads separate Round 3 and Round 4 status files
and serializes report writes. It adds completed runs to the same comparison
table, preserves prior rows, and validates all paired task/trial/seed entries.
The displayed columns retain success counts/rates, cosine/query counts,
mismatch counts/rates, and successful/failed episode means and pooled counts.

Report:
`http://127.0.0.1:8000/analysis_results/index.html#round3-move-alignment-comparison`

W&B URLs appear in `status.json` and each run's `wandb_run_url.txt`, including
while the runs wait for their GPU gate.

Verification: exact weighted normalization and finite gradients; unit-weight
regression; unchanged performance-loss gradient; deterministic 16/16 and 4/4
sampling; repeat sampling confined to training indices; report preservation;
full-model backward probes for both modes (physical batch 2, finite gradients).
Full suite: 95 passed, 1 skipped; the additional report test passed afterward.
Probe artifacts use separate `round4-probe-*` directories and are not the
training checkpoints. No probe runtime failures occurred. Active GPU use by
the preceding lambda=1 evaluation delays training intentionally.



## Current lambda revision

Both jobs use lambda=.5, per the latest instruction. The lambda=1 jobs were
stopped; their logs and registrations are in `superseded-lambda1`.
The restarted runs load the untouched pretrained checkpoint, with 200 fresh
updates. New run names use `round4-lambda05-restart-*` to preserve all earlier
artifacts. The paired evaluation settings and both experiment modes are unchanged.

## Current W&B runs

- failure-weight3: https://wandb.ai/yus047-/ecot-move-alignment/runs/foxogq97
- balanced-sampling: https://wandb.ai/yus047-/ecot-move-alignment/runs/jb9v5bbl

## Overnight continuation

See `OVERNIGHT_HANDOFF_2026-09-06.md` for all four runs and reboot recovery.
