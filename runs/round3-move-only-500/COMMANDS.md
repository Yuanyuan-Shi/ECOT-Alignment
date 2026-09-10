# Mismatch-only Round 3 ablation — September 9, 2026

User requested a fresh experiment with the exact Round 3 setting, performance
loss excluded, 500 optimizer steps, followed by analysis of whether mismatch
can reach zero. The detached pipeline has already been launched; do not duplicate it.

Training: `bash runs/round3-move-only-500/train.sh`
Evaluation: `bash runs/round3-move-only-500/evaluate.sh`
Report: `source runs/round3/environment.sh` then
`python runs/round3-move-only-500/analyze.py`

The full arguments are recorded in the two shell scripts. Relative to Round 3
lambda=1: performance-loss weight changes from 1 to 0; max steps from 200 to
500; epoch ceiling from 10 to 13 to permit 500 steps. The 20-step warmup stays
unchanged. A final evaluation-mode pass over the complete training set is added
for diagnostics after all updates. Teacher forcing and targets remain unchanged.
Performance loss remains logged, but has no contribution to backward().
All other training configuration is unchanged, including natural sampling,
unit query weights, seed 20261102, batch 32, accumulation 1, AdamW 1e-4,
zero weight decay, clipping 1, LoRA r/alpha16/dropout .05 and pretrained weights.
The exact ordered train/validation ID lists were verified against Round 3 lambda1.

13 epochs provide 507 possible updates; the cap stops at 500, after 12 full
39-step epochs plus 32 batches: 15,964 training query presentations.
The final checkpoint is `step-000500-move-lora.pt` under the run directory
recorded in status.json. Evaluation reuses the exact 180 paired episode schedule,
seed 20261201, 10 open-loop steps, six processes, same simulator settings.

Fixed-dataset floors under unchanged zero-vector convention:
- Training: 50/1245 zero vectors, mismatch floor 4.016064%, loss floor .004462294.
- Validation: 9/312 zero vectors, mismatch floor 2.884615%, loss floor .003205128.
- Zero-vector cosine is always zero; hinge penalty 1/9; no directional gradient.

Seven objective tests passed, including exclusion of performance gradients and
preservation of the original combined objective. Scripts compiled and shell syntax
passed. Training, evaluation and report errors are saved in status.json/logs.
The pipeline continues after the chat turn, but does not survive a workstation reboot.

Report: http://127.0.0.1:8000/analysis_results/move_only_500/index.html
The report updates approximately once per minute; reload the browser to see updates.
It reports fixed-data soft-decoded training/validation metrics, then realized-motion
and decoded-policy rollout metrics separately, each with directional-only mismatch.
Existing Round 3/Round 4 tables and approved qualitative figures are preserved.
