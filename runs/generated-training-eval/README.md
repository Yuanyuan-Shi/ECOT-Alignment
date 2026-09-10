# Frozen-model evaluation on training images, with test-time generation

Authorized by user: try identical generation evaluation, show updated table.
NO training / gradient updates. Active queue: status.json, launcher.pid.
Six independent inference workers, two per model, all on GPU0; exact merged
checkpoints used for test, same BF16 conversion/get_prismatic_vla_action,
no center crop, no wrist, no fixed reasoning prefix beyond PLAN:, full generated
reasoning and discrete 7 VQ action codes per 10-step chunk. torch.inference_mode,
frozen parameters, eval mode. Images are the exact saved before PNGs used for
training. All 1,245 train IDs taken in saved order from Round3 lambda1 split.

Resume interrupted worker safely (do not duplicate live workers): source
runs/round3/environment.sh then python scripts/analysis/evaluate_generated_training.py
--model pretrained|lambda1|moveonly --rank 0|1 --world 2.
Already-written query IDs are skipped. Inspect logs for errors before restarting.

Original MiniVLA regeneration pilot: first 8 generated queries exactly match both
saved source reasoning and action-token sequences. All three models are still
regenerated in full for consistent provenance. Do not use partial counts as final.
Report watcher refreshes every45s until queue exits; final renderer validates IDs.

Outputs: analysis_results/move_only_500/generated_training/{model}-rank{rank}.jsonl
Each record saves image source, generated reasoning, discrete action codes, decoded
actions, parsed direction/stop, cosine, scaled command sum, outcome and timing.

Metric: direction cosine<.5, literal stop norm(sum(.05*clip(a_xyz,-1,1)))>.03m;
other zero-vector cases excluded. Train success/failure is the source episode label,
not a newly tested training-set task success. Existing 180-episode rollout test
rows retained from stop_and_execution_analysis.json. New reasoning means inclusion
and denominator can vary by model. Evaluate parsing coverage alongside mismatch.

Report http://127.0.0.1:8000/analysis_results/move_only_500/generated_training.html
Update: python scripts/analysis/render_generated_training.py
No simulator run or retraining. Matching decoding does not match state distributions;
generated reasoning-command self-consistency does not establish reasoning correctness.
