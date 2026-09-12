# B200 launch and resume commands

## Active bounded pilot

The ten-iteration, eight-B200 pilot was submitted on 2026-09-11 as Slurm job
`386060` under the `scavenger` QoS. Its persistent directory is
`/scratch/schmidt/ssci-ys-shigroup/runs/ecot-grpo-g8-10iter-seed7-20260911`, and
its W&B run is https://wandb.ai/yus047-/ecot-onpolicy-grpo-pilot-seed7/runs/efeagqy3.
At submission time Slurm estimated a 2026-09-12 16:30 PDT start because of queue
priority. The full 100/200-iteration experiment remains unlaunched.

The workstation controller refuses configurations with more than one iteration. The
100-iteration configuration is staged for cluster testing only and has not been run.

The B200 configuration replaces task 27 (`put the wine bottle on the wine rack`)
with task 84 (`pick up the red mug and place it to the right of the caddy`). In the
frozen pre-PPO selection data, task 27 succeeded 0/3 with 109 queries, whereas task
84 succeeded 2/3 with 62 queries and still had a high 33.9% alignment-mismatch rate.
This substitution was requested after task 27 became the workstation smoke-test
straggler. The completed workstation smoke itself retains the exact original subset.

On the B200 allocation, set the transferred original MiniVLA checkpoint and a new
persistent run directory:

```bash
export ECOT_REPO=/scratch/schmidt/ssci-ys-shigroup/ECOT-Alignment
export ECOT_BASE_CHECKPOINT="$ECOT_REPO/artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt"
export ECOT_GRPO_RUN_DIR=/scratch/schmidt/ssci-ys-shigroup/runs/ecot-grpo-seed7
export HF_HOME="$ECOT_REPO/artifacts/hf-cache"
export PYTHONPATH="$ECOT_REPO:/scratch/schmidt/ssci-ys-shigroup/deps/LIBERO:$ECOT_REPO/artifacts/vq-bet"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$ECOT_REPO"
```

First run the unit tests and a one-iteration B200 smoke using a copied config whose
`iterations` and `run_dir` are explicitly set to 1 and a fresh cluster-smoke path.
The workstation result does not substitute for this B200 validation.

```bash
python -m pytest -q onpolicy_grpo/tests
export ECOT_GRPO_SMOKE_DIR="${ECOT_GRPO_RUN_DIR}-smoke"
jq --arg run_dir "$ECOT_GRPO_SMOKE_DIR" \
  '.iterations = 1 | .run_dir = $run_dir | .wandb_run_name += "-smoke"' \
  onpolicy_grpo/config_cluster_100.json > onpolicy_grpo/config_cluster_smoke.runtime.json
python -u -m onpolicy_grpo.smoke \
  --config onpolicy_grpo/config_cluster_smoke.runtime.json \
  --run-dir "$ECOT_GRPO_SMOKE_DIR"
```

Resume the interrupted B200 smoke with the identical frozen config and run directory:

```bash
python -u -m onpolicy_grpo.smoke \
  --config onpolicy_grpo/config_cluster_smoke.runtime.json \
  --run-dir "$ECOT_GRPO_SMOKE_DIR"
```

The resume path is idempotent at a completed boundary and resumes after an existing
immutable rollout refresh. This exact controller path was exercised twice on the
workstation: the second invocation skipped the completed iteration without modifying
its status, checkpoint, rollouts, optimization data, or W&B run.

The production config is [config_cluster_100.json](config_cluster_100.json). A
cluster production launcher must support one iteration per durable commit marker,
refresh `rollout_policy` once at each boundary, and resume from the last completed
policy checkpoint. Do not point the workstation-only `smoke.py` controller at the
100-iteration config; it intentionally refuses it.

After the B200 smoke passes, launch through the cluster scheduler with the cluster
controller developed from this tested one-iteration path. Resume with the identical
frozen config and run directory. Never edit a prepared run in place.
