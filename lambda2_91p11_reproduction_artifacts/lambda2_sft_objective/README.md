# Exact λ=2 SFT training objective

This directory preserves the source and saved configuration used to train the checkpoint that produced 246/270 closed-loop successes.

- `finetune_large_alignment.py`: complete 2,000-step trainer and objective composition.
- `refined_move_alignment.py`: directional and literal-stop continuous penalties.
- `move_alignment_training.py`: teacher-forced action-logit extraction, soft VQ decode, action unnormalization, and loss composition helpers.
- `training_config.json`: exact λ=2 run configuration.
- `result.json`: final training/validation losses and exported-checkpoint verification.

The optimized loss was `L_performance + 2 * L_alignment`. Task success and the thresholded mismatch ratio were diagnostics, not loss terms. See the dedicated section in `LAMBDA2_91P11_EVALUATION_HANDOFF.md` for the complete equations and RL-consistency notes.
