# Refined stop/directional loss with validation-based early stopping

Authorized Sept9: launch lambda=1 and mismatch-only from original MiniVLA,
500-step default budget, lower LR and validation-based early stopping, fast paired
evaluation on a sampled 30 tasks ×3 trials =90 episodes/model. Original MiniVLA is
also evaluated on that schedule. No prior finetuned initialization is used.

ALREADY LAUNCHED. Check status.json and launcher.pid; do not duplicate the queue.
The previously stopped generated-training evaluation remains stopped.

Training commands (executed sequentially):
```
source runs/round3/environment.sh
python -u vla-scripts/finetune_refined_alignment.py --mode lambda1
python -u vla-scripts/finetune_refined_alignment.py --mode moveonly
```

Both start at artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt.
LoRA rank/alpha16, dropout.05, all prior target modules, AdamW weight decay0,
grad clipping1, physical batch32/accumulation1, BF16 forward, gradient checkpointing,
seed20261102. LR5e-5, 20-step warmup. Budget500 optimizer steps (13-epoch ceiling
in config; explicit step loop). Stop earlier after four validation checks with no
strict decrease in eligible-query mismatch rate. Validation every25 steps, plus
step0 baseline; best selected checkpoint may be step0 if no improvement. Ties
count as no improvement. Every checked adapter and validation scores are saved;
best weights restored before final training/validation scoring and merged export.
No test outcomes are used for selection. RNG states are preserved around validation
so validation frequency does not alter training dropout/shuffle behavior.

Refined loss = separate directional and stop means, summed:
- Directional: mean((relu(.5 - cos(reasoning_xyz,sum(predicted_xyz)))/1.5)^2).
- Literal MOVE: stop: mean((relu(norm(sum(.05*clip(predicted_xyz,-1,1))) - .03 - 1e-8)/.03)^2).
- Other zero-vector claims excluded from alignment; lambda1 still uses their
  supervised targets. No path-energy/cancellation penalty added, matching the
  agreed net-command criterion. Raw directional metric retained for continuity.
- Total lambda1 = supervised loss + refined alignment loss; mismatch-only =
  refined alignment loss alone. Constants/means differ from legacy Round3 because
  the objective is intentionally refined. No loss-gradient contribution from
  supervised loss in mismatch-only.

plan.json records exact train/validation IDs and sampled tasks before training.
Episode-level split stratified by source outcome and stop presence: 72 training
source episodes /1251 queries (1201 directional,10 literalstop,40 excluded);
18 validation source episodes /306 queries (297 directional,5 stop,4 excluded).
With one source episode/task, validation also holds out source tasks from this
adaptation split. Original MiniVLA pretraining already included LIBERO90.

Evaluation: fixed uniform 30-task sample seed20260909; 3 trials each, base seed
20261211; episode seed=base+task_id*3+trial; initial-state indices0,1,2.
Full exact arguments are saved as evaluate-{model}-command.json before execution.
Same standard evaluation implementation: no center crop/wrist, use_cot=True,
10 open-loop steps,6 rollout processes, save decoded action chunks/reasoning,
no auxiliary policy table inserted into old reports. 270 evaluation episodes total.

Report includes original, selected lambda1, selected mismatch-only; overall and
source-success/failure training mismatch, overall and rollout-success/failure test
mismatch, task success. Training scores use fixed reasoning/soft-decoded predictions
with dropout off, matching user-selected standard protocol. Test generates reasoning
and discrete codes. Directional or literal-stop queries eligible, others excluded.
New dataset split/test sample makes these distinct from old table denominators.
Report validates exact paired task/seed schedule and rejects incomplete results.

Tests:12 targeted tests passed (gradient exclusion, stop penalty/boundary/gradient,
zero/unknown exclusion, net cancellation behavior, split leakage/reproducibility,
early-stop patience/ties/reset). Scripts compiled. A 1e-8m numerical allowance
prevents float32 roundoff classifying exactly3cm as mismatch.

Live report: http://127.0.0.1:8000/analysis_results/refined_stop_generalization/index.html
State: runs/refined-stop-generalization/status.json
Raw logs: train-{lambda1,moveonly}.log, eval-{pretrained,lambda1,moveonly}.log
Run artifacts: runs/ecot-move-alignment/refined-stop-lr5e5-{lambda1,moveonly}-seed20261102
Detached queue survives chat turn, not reboot; errors saved in status.json.
