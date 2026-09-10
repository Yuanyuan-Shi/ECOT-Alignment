# Saved progress — 2026-09-10

## Current instruction: remain paused
User requested pause, then stop-metric analysis and a proposed formula, then saving progress. Do not resume collection, training, evaluation, or implement the proposed loss until instructed. Previously paused process IDs162147/162486 are no longer present (verified at save time). No completed optimizer steps, resume checkpoints, or model results exist for the large-data round.

## Completed data and approved experiment plan
- All900 collection episodes are complete:16322queries. Split720training episodes/13093queries and180validation episodes/3229queries. The prior status stage showing collection batch5/9 was a restart verifying already-complete batches, not incomplete collection.
- Data: runs/large-data-alignment/records.jsonl, split.json, dataset_complete.json, dataset_statistics.json, episodes.json. Original collection folders large-data-collection-00 through08 contain images, reasoning, action tokens, decoded commands, and outcomes.
- Approved training: original MiniVLA checkpoint for EACH lambda0(performance only),lambda1(performance+alignment),moveonly(alignment only); physical batch32, no accumulation,2000optimizersteps,LR5e-5,warmup50,LoRAr16/dropout.05. No early stopping. Save250/500/1000/1500/2000; validate250steps. Four-model final tests: same30sampled tasks,3rollouts/task, initialstates13..15 distinct from collection3..12.
- Last training failure was initial validation before optimization: extra action-like token inside reasoning caused extraction to count8 rather than7 action tokens. Fixed with terminal_action_labels selecting final contiguous7tokens for alignment, preserving full supervised CE labels.17tests passed. Queue was restarted but user paused it while verifying existing collection batches; no new training completed.
- Network interruption protection: cached HF models, offline local W&B logs, later sync/retry. Ethernet+Wi-Fi routes were available at launch. Do not change network configuration unnecessarily.

## Stop analysis on generated900episodes
- Eligible directional queries15553; literalstop236; other excluded533.
- Original commanded criterion: H10 sum, per-step translation .05*clip(decoded_xyz,-1,1) metres; directional cosine>=.5, literalstop net norm<=.03m.
- Overall mismatch1470/15789=9.31%; directional1234/15553=7.93%; stop236/236=100%.
- Stop net translation norms4.4449–4.8303cm; mean4.8258cm.235/236queries use just two distinct action-token chunks (168 and67instances), third chunk1instance. Repeated decoder pattern warrants investigation; not established to be policy-intended physical motion.
- Non-stop directional net magnitude mean24.4149cm,median24.0467cm. Mean single-step directional magnitude2.5814cm.
- Hypothetical stop threshold5cm on net10step displacement:0/236mismatch.
- Hypothetical per-step3cm threshold:0/2360steps exceed;0/236queries have ANY exceeding step. Mean stop per-step magnitude.48645cm,max.57783cm.
- No threshold change was applied.

## Latest proposed stop loss — discussed ONLY, not implemented/approved
User suggested penalizing stop motion magnitude directly, without a3cm dead zone. Assistant proposed:
  directional: [max(0,tau-c)/(1+tau)]^2, tau=.5 (unchanged)
  literalstop: (1/H) sum_h ||a_xyz[h]/s||_2^2, H=10, actions in metres.
This is mean squared per-step translation, avoids cancellation. s is normalization scale, NOT threshold; .03m was an illustrative choice, not finalized. Every nonzero translation is penalized.
Proposed separate evaluation criterion: max_h ||a_xyz[h]||_2 <= epsilon_stop. Epsilon not selected;3cm/step is permissive relative to observed stop steps. Batch pooling and combined objective unchanged.
IMPORTANT: Implementation STILL uses old net-displacement stop hinge above. Do not claim proposed per-step magnitude loss is implemented. Clarify final scale and evaluation tolerance when user resumes this discussion.

## Links and files
- New tab: http://127.0.0.1:8000/analysis_results/index.html?ui=large-data-v2#large-data-alignment
- Standalone report: analysis_results/large_data_alignment/index.html
- Stop scenarios: http://127.0.0.1:8000/analysis_results/large_data_alignment/stop_examples.html ; accompanying stop_examples.json contains exact commands and reasoning. Three examples: task14episode2query25 successful bowl placement; task79episode5query28 failed book/caddy; task51episode4query39 failed butter/basket.
- UI has fixed visible top nav with Previous experiments / New:900-episode experiment. Existing results preserved.
- Trainer: vla-scripts/finetune_large_alignment.py; shared loss experiments/robot/libero/refined_move_alignment.py; token selector experiments/robot/libero/move_alignment_training.py.
- Queue: runs/large-data-alignment/start.sh and launch.py. Restart only on user authorization. Current state intentionally paused; prior PIDs stale.

## Earlier screenshot clarification
User's1.24%(15/1211) training mismatch screenshot belongs to older separate-group-mean run refined-stop-lr5e5-moveonly-seed20261102, selectedstep200,stopped300. All15errors directional;10stopqueries aligned.13/15cosines.44–.49, overall alignmentloss.00002193. This differs from corrected pooled run(selected225,19/1211mismatches including10stops). Near-zero squared-hinge loss need not imply0%threshold mismatch.
