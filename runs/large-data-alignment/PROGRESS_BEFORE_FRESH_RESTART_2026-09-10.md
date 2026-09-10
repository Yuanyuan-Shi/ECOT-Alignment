# Latest saved progress — 2026-09-10, after export diagnosis

This file supersedes SAVED_PROGRESS_2026-09-10.md where they differ.

## Current state
- Dataset complete:900episodes,16322queries. Training720episodes/13093queries; validation180episodes/3229queries. Original source success772/900=85.8%; training618/720; validation154/180.
- Lambda0 completed all2000optimizersteps. Saved adapter and optimizer/RNG checkpoint are intact: runs/ecot-move-alignment/large-data-lambda0-b32-s2000/resume.pt, resume-step002000.pt, adapter-step002000/. All intermediate250/500/1000/1500 checkpoints exist.
- Lambda0 validation at2000:266/3123=8.51745% mismatch, performance loss0.03068438. Stop43/43 mismatch; directional223/3080. This is performance-only training, so stop magnitude is not optimized.
- Lambda0 failed at FINAL EXPORT VERIFICATION, not training. No result.json or verified merged checkpoint yet. Lambda1 and mismatch-only have not started. Test rollouts have not started.
- status.json currently says failed, stage training lambda0; last attempted queue171129 failed11:02:27. Do not interpret stale PID as proof of a running process.
- User interrupted the latest diagnostic tool invocation, then requested saving progress. No additional training/restart was launched during this save. Verify host processes before any restart; sandbox ps cannot reliably see host processes. The aborted merge-diagnostic command may not have executed; no export-probe-merge.log output was available at save time.

## Final authorized experiment settings
User explicitly finalized ACCUMULATED stop loss (not previously suggested per-step loss):
 d_act=sum over10 of .05*clip(decoded_xyz,-1,1) metres; rho=||d_act||.
 directional loss=[max(0,.5-c)/1.5]^2.
 explicitstop loss=(rho/.03)^2, WITHOUT hinge or subtraction.
 pool one mean over all eligible direction/explicitstop queries.
 Evaluation UNCHANGED: direction cosine>=.5, stop rho<=.03m. Existing collected dataset table unchanged.
- Implemented stop_loss='net_squared' in large-data trainer; shared refined_objective retains threshold_hinge default for old rounds. loss_version=directional_hinge_pooled_net_stop_squared_v2. Resume enforces version.
- Three fresh models(original MiniVLA checkpoint):lambda0,lambda1,moveonly. Batch32 physical, no accumulation,2000steps, LR5e-5,warmup50,LoRAr16/dropout.05,weightdecay0. No early stopping. Validation250steps; save250/500/1000/1500/2000. Final test four models*90episodes.
- User asked solo mode/no routine approvals. No agents. Still comply with actual tool sandbox escalation requirements when necessary.

## Earlier token-extraction issue — fixed
Extra action-like special tokens inside generated reasoning caused8 rather than7 tokens. terminal_action_labels in experiments/robot/libero/move_alignment_training.py isolates final contiguous7action tokens for alignment while preserving complete supervised CE labels. Full lambda0 training now completed successfully after this fix.18tests passed including new stop loss tests.

## Export failure diagnosis — key new finding
Original before/after LoRA-merge check failed maxlogit difference2.15. A retry using float32 and disablingTF32 still failed. Diagnostic revealed repeated UNMERGED forward passes already differ; all modules in eval mode, CUDA RNG unchanged.
- export-probe.log: repeated vision feature maxdifference3.40199, projector.41697, first LLM layer.43398, logits1.80165.
- export-probe-math.log: wrapping forward in torch.nn.attention.sdpa_kernel(SDPBackend.MATH) gives EXACT repeated equality: vision/projector/firstlayer/logits maxdifference0.0.
Conclusion supported by experiment: automatic attention backend in full-precision vision/export check introduces variation; math attention removes it. NOT established that LoRA merging is incorrect. Do not claim reduced precision was root cause (weights originallyfloat32). Do not infer this proves previous bf16 training or rollout results invalid; those paths have not been audited for this issue.

## Latest edits / remaining recovery
- vla-scripts/finetune_large_alignment.py now uses float32, TF32off and sdpa_kernel(MATH) ONLY for before/repeat/after export verification. Training and standard evaluation paths unchanged.
- runs/large-data-alignment/diagnose_export.py loads saved lambda0 adapter, runs identical-input diagnostic and now also merges adapter then asserts merged/unmerged logits allclose(atol=.003,rtol=.003).
- Wrapper diagnose_export.sh uses cached experiment env. The last command intended to write export-probe-merge.log was interrupted during the tool call; final merge-equivalence NOT YET VERIFIED.
- Next: inspect any host diagnostic process, run/finish merge probe, confirm tiny merge error, then restart start.sh. Trainer resumes at2000, skips optimizer loop, recomputes final metrics and exports. Do not retrain lambda0 from scratch or weaken comparison to conceal a discrepancy.
- After verified export, queue syncs W&B and trains lambda1 thenmoveonly, then four final tests. Data collection batches are verified/reused automatically.
- Consider caching final summary results before export to avoid repeated full-dataset scoring on retries; not implemented.

## W&B and reporting
- Offline W&B local run directories retained; queue syncs after completed export, retries at end. Multiple failed/export diagnostic segments exist; keep metadata honest.
- New tab: http://127.0.0.1:8000/analysis_results/index.html?ui=large-data-v2#large-data-alignment
- Stop example page: http://127.0.0.1:8000/analysis_results/large_data_alignment/stop_examples.html
- report.py adds Train task success above Test task success: original85.8%(618/720), tuned N/A—fixed dataset; no training-state rollout evaluation performed.
- Local HTTP server was restored on hostPID166122 and both pages verified200. Check host if inaccessible again.
- Earlier user question: original paper90.8% is not this reproduction checkpoint. Our checkpoint100k published87%(783/900) is THIRD-PARTY self-report RHYu2233/ecot-libero90, corroborated Erwin2233/latent-ecot README, not independently verified. Official200k FullECoT checkpoint was not found on official project/HF listings. User accepted87vs85.8 variation.

## Key files
- plan.json,status.json,HANDOFF.md,records.jsonl,split.json,dataset_statistics.json in runs/large-data-alignment.
- Trainer vla-scripts/finetune_large_alignment.py.
- Shared loss experiments/robot/libero/refined_move_alignment.py.
- Diagnoses export-probe.log, export-probe-math.log; finalmerge probe pending.
- Existing implementation snapshots copied again at save time.
