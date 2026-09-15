# λ=2 reference-run diagnostic artifacts

These files come from the original MiniVLA λ=2 evaluation that produced 246/270 successes and 360/4,463 all-query mismatches.

| File | Contents |
|---|---|
| `episode_summary.json` | All 270 case outcomes, state indices, seeds, query counts, and episode diagnostics |
| `first_query_fingerprints.csv` | First-query VQ tokens, complete decoded 10×7 action chunk, and hashes of reasoning and simulator state for all 270 cases |
| `task24_state14_seed80_trace/` | Lossless image, exact model inputs, token/logit trace, runtime flags, and source hashes for the requested diagnostic case |
| `alignment_queries.jsonl.gz` | All 4,463 original policy queries, including reasoning, VQ tokens, decoded actions, executed end-effector trajectories, and simulator state before/after each chunk |
| `all_query_scores.jsonl` | Paper-metric score for every policy query |
| `experiment_metadata.json` | Evaluator-emitted saved configuration; see the stale-field warning in the main handoff |
| `run_summary.json` | Original aggregate and per-task evaluator output |
| `native_eval_results.jsonl` | Native worker-merged task-success result records |
| `worker_logs/` | Six original evaluator logs, one for each local worker |
| `original_lambda_sweep_launcher.py` | Original combined train/evaluate orchestration script |
| `checkpoint_config.json` | Native checkpoint loader configuration |
| `dataset_statistics.json` | Action unnormalization statistics used by evaluation |
| `training_config.json` | Saved λ=2 fine-tuning configuration |
| `pip_freeze.txt` | Complete MiniVLA virtual-environment package inventory captured on 2026-09-15 |
| `torch_collect_env.txt` | PyTorch environment report captured on 2026-09-15 |
| `robosuite_macros.py` | Exact installed robosuite macro file |
| `SHA256SUMS` | Integrity hashes for every artifact in this directory |

No `robosuite/macros_private.py` existed in the evaluation virtual environment. The run therefore used the defaults in `robosuite_macros.py`, including `SIMULATION_TIMESTEP=0.002`, `ENABLE_NUMBA=True`, `CACHE_NUMBA=True`, `IMAGE_CONVENTION="opengl"`, and `MUJOCO_GPU_RENDERING=True`.

The full alignment file is gzip-compressed for GitHub. Decompress it with:

```bash
gzip -dc alignment_queries.jsonl.gz > alignment_queries.jsonl
```

The lossless initial camera inputs were not saved because the run used `alignment_save_frames=False`. Rollout MP4s are lossy and are not included. The first-query file contains the strongest remaining original-run fingerprints available without rerunning evaluation.

`torch_collect_env.txt` was collected after the completed evaluation, when CUDA was temporarily unavailable to that process. Use it for the CPU, OS, Python, PyTorch-build, and installed CUDA-library inventory. The actual evaluation GPU, driver, VBIOS, and runtime values are recorded in the main handoff from the successful run.

The editable-package line in `pip_freeze.txt` names the older local Git HEAD `18cf7c8`; evaluation imported the working-tree source through the explicit `PYTHONPATH`. The three evaluator files used at runtime match the hashes and pinned GitHub checkout documented in the main handoff. LIBERO and VQ-BET were clean at their documented commits.
