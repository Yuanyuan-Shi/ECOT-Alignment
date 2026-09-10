# ECOT-Alignment repository snapshot

This repository preserves the upstream ecot-lite history and the local ECOT alignment implementation, tests, experiment configurations, progress notes, and generated analysis reports as of September 10, 2026.

The current large-data experiment uses 900 collected episodes, split into 720 training and 180 validation episodes. Each of lambda0, lambda1, and mismatch-only starts from the original MiniVLA checkpoint and trains for 2,000 optimizer steps with physical batch size 32. The fresh round supersedes the archived lambda0 run.

Experiment entry point: `runs/large-data-alignment/start.sh`. Configuration: `runs/large-data-alignment/plan.json`. Training implementation: `vla-scripts/finetune_large_alignment.py`. Report generator: `runs/large-data-alignment/report.py`.

W&B project: https://wandb.ai/yus047-/ecot-large-data-900-b32-2k-20260910

Model weights, optimizer snapshots, raw training-query JSONL files under `runs/`, rollout data, environment caches, credentials, and runtime logs are excluded from Git. These remain on the original machine. Scripts currently reference that machine's absolute paths and Conda environment; another machine needs equivalent dependencies and data, plus adjusted paths. Report links to excluded artifacts will only resolve on the original machine.

Generated reports are snapshots at commit time. Ongoing evaluation updates the local report; GitHub receives updates only when committed and pushed.
