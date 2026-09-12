# ECoT-Lite MiniVLA on-policy GRPO

This directory is isolated from the retained PPO pilot. It implements task-grouped
GRPO starting from the original MiniVLA checkpoint. The accelerated workstation
smoke uses four independently sampled episodes for each of the exact ten frozen
PPO-pilot tasks; the staged B200 production configuration retains group size eight.

The three roles are explicit: a trainable LoRA policy, one immutable rollout-policy
snapshot refreshed at the iteration boundary, and a separately loaded frozen original
reference policy. The workstation controller is hard-limited to one iteration.

Run unit tests:

```bash
source runs/round3/environment.sh
python -m pytest -q onpolicy_grpo/tests
```

Run or deterministically resume the one-iteration workstation smoke:

```bash
bash onpolicy_grpo/start_smoke.sh
```

Results are written only under `onpolicy_grpo/outputs/workstation-smoke-g4-seed7`.
The validated outcome is summarized in [SMOKE_RESULTS.md](SMOKE_RESULTS.md).
See `CLUSTER_COMMANDS.md` for the staged B200 configuration and safety constraints.

The bounded B200 pilot uses `config_b200_10.json` with ten on-policy iterations,
G=8, and one rollout rank per GPU. Its `cluster_train.py` controller writes a durable
completion marker after every validated iteration and resumes the same W&B run.
It retains only the latest checkpoint needed for resume and the final iteration-10
policy; validated intermediate policy copies and rollout snapshots are pruned.
