# Approved large-data experiment — launched 2026-09-09

User authorized implementation and execution of 900-episode data collection, statistics, three 2,000-step fine-tunes (lambda0/lambda1/moveonly), paired final tests, and a separate existing-report tab. No early stopping. All models initialize from original artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt.

Entry point: `bash runs/large-data-alignment/start.sh`. Singleton file lock prevents duplicate queues. Current progress and worker PID: status.json. If the queue is restarted while its child remains running, it waits for that child before continuing, preserving collection. Completed 100-episode collection batches are verified and reused. An interrupted incomplete batch is archived and rerun. Training resumes from optimizer/adapter/RNG checkpoint every250 steps.

SUPERSEDED batch64 setup: Physical batch64 failed stress test with CUDA OOM on longest reasoning sequences. Effective batch64 is implemented as two microbatches32, weighting supervised CE by labeled-token count and alignment by eligible-query count (not naive mean of microbatch means). Stress test passed:54.71GiB peak allocated,62.88GiB reserved; two updates5.31sec. User informed. Plan metadata records this change.

Collection:90tasks*10initialstates indices3..12. Split720/180episodes deterministically per-task8/2 after collection. Test: same30task IDs as previous fast tests, indices13..15, separate seeds. All90 tasks verified at least50states and distinct collection/test state vectors. Dataset uses generated reasoning and actions, including failed behavior, not corrected expert targets. Preserve this explicit caveat.

Shared metric/loss: refined_move_alignment.refined_objective, H10, clip controller inputs[-1,1], scale translation .05m, net sum; cosine threshold.5; literalstop threshold.03m; pooled eligible mean; other zero-vector claims excluded. Stats use episode-bootstrap2000samples for failed-success differences. Query IDs are episode_id::policy_query_index (must be unique across multiple rollouts per task).

Trainer: vla-scripts/finetune_large_alignment.py, LR5e-5,warmup50,LoRAr16/dropout.05,weightdecay0,clipgrad1. No early stopping or validation-based checkpoint selection. Save resumable250/500/1000/1500/2000; validate every250. Verify merge logits and serialized model state at final export. Final evaluation uses2000step models plus original,90episodes/model.

Network: Ethernet and Wi-Fi default routes present (metrics100/600). No routing changes made. Models load entirely from cached HF artifacts with offline flags. W&B logs offline locally and queue syncs each training segment after completion, retrying at end; pending files retained if disconnected. No dependency on network for collection/training/testing.

Report: http://127.0.0.1:8000/analysis_results/index.html#large-data-alignment
Iframe standalone: analysis_results/large_data_alignment/index.html. Existing report kept intact; existing/new tab navigation added. Auto-refresh new report60sec. Source report.py.

Verification:16tests pass (14 existing alignment tests +2 new dataset denominator/pooled accumulated-gradient tests), compilation and HTTP tab checks pass. Further changes should test relevant portions. No agents used.

## Latest user change
User explicitly requested physical batch32 with no gradient accumulation and the same2000optimizersteps. Implemented batch_size32,gradient_accumulation_steps1. Run directories now large-data-{mode}-b32-s2000. LR/checkpoints/validation schedule unchanged. This is roughly5epochs on estimated12k–12.8ktrainingqueries. Collection uninterrupted; replacement queue adopts existing worker and reloads revised plan.

2026-09-10 status repair: collection finished900episodes/16322queries (train13093,val3229). Initial lambda0 validation failed before optimizer steps because generated reasoning contained an extra action-like token. Added terminal_action_labels to isolate final contiguous7tokens for alignment while preserving all supervised CE labels.17tests pass. Queue restarted162147; completed collection reused.


Latest authoritative saved state: [SAVED_PROGRESS_2026-09-10.md](SAVED_PROGRESS_2026-09-10.md). Remain paused. Proposed per-step stop loss is not implemented. All900episodes complete; no trained large-data checkpoint exists.

## Authorized restart 2026-09-10
User finalized stop loss as squared ACCUMULATED net magnitude (rho/0.03)^2, no hinge; evaluation and collected table keep3cm net threshold. Implemented explicit net_squared option only in large-data trainer, preserving old round defaults. loss_version=directional_hinge_pooled_net_stop_squared_v2; resume checks enforce matching loss.18tests passed. User authorized all3trainings and final evaluation. Sandbox process visibility previously hid paused hostPID162147; host check found it stopped holding queue lock with zombie child162486. Replaced stale launcher; current launchedPID166431. Status/file state is authoritative; use approved host process inspection when needed. No data collection repeated.


Latest authoritative state: [LATEST_PROGRESS.md](LATEST_PROGRESS.md). Lambda0 finished2000steps but export pending. Exact repeatability restored by math attention in export diagnostic; merged/unmerged equivalence still awaits verification after interrupted tool call. Lambda1/moveonly pending.
