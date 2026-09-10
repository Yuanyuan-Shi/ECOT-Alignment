# Round 3 lambda=2 and lambda=5 extension — September 7, 2026

Both jobs are submitted; do not duplicate the active queue.

```bash
source runs/round3/environment.sh
python -u runs/round3-high-lambda/launch.py
```

The wrappers use the existing Round 3 scripts directly:

```bash
bash runs/round3/train.sh 20 2.0 32 1
bash runs/round3/train.sh 50 5.0 32 1
bash runs/round3/evaluate.sh 20
bash runs/round3/evaluate.sh 50
```

Both independently load the untouched pretrained checkpoint. Both retain
Round 3's thresholded squared-hinge loss, natural sampling, unit query weights,
1,557-query artifact, original 1,245/312 split, seed 20261102, LoRA settings,
AdamW, LR 1e-4, 20 warmup steps then constant LR, batch 32, accumulation 1,
and 200 optimizer updates. Only lambda and identifying names/paths change.
The controller adds a GPU start gate so both W&B runs register immediately,
then trains sequentially on the single GPU. Both trainings precede evaluation.

Each evaluation uses the unchanged Round 3 90 tasks × 2 trials, seed 20261201,
10 open-loop steps and six rollout processes. All task/trial/seed entries are
validated against the saved pretrained baseline before metrics enter the table.

Checkpoint paths on completion:

- `runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambda20-seed20261102/checkpoints/step-000200-move-lora.pt`
- `runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambda50-seed20261102/checkpoints/step-000200-move-lora.pt`

Live state: `runs/round3-high-lambda/status.json`. PID: `launcher.pid` here.
Logs: `train-20.log`, `train-50.log`, `eval-20.log`, `eval-50.log`.
W&B links appear in status and each run's `wandb_run_url.txt`.

The shared HTML table places lambda=2, then lambda=5, after Round 3 lambda=1
and before Round 4. Pending rows remain in those positions until evaluation
finishes. Existing results and checkpoints are preserved.

http://127.0.0.1:8000/analysis_results/index.html#round3-move-alignment-comparison

## W&B runs

- lambda=2.0: https://wandb.ai/yus047-/ecot-move-alignment/runs/y8a06kkb
- lambda=5.0: https://wandb.ai/yus047-/ecot-move-alignment/runs/wv1c8aez
