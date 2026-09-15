# MiniVLA Joint FT λ=2: exact 270-episode evaluation handoff

This is the complete handoff for the MiniVLA joint fine-tuning λ=2 checkpoint reported at **246/270 task successes (91.11%)**. A second run on the original workstation reproduced all 270 episode outcomes exactly.

## Immutable inputs

| Item | Value |
|---|---|
| Evaluator code commit | [`5403dc2828ed4a65c3ff366e75ec86fb839cd8e7`](https://github.com/Yuanyuan-Shi/ECOT-Alignment/tree/5403dc2828ed4a65c3ff366e75ec86fb839cd8e7) |
| Evaluator | `experiments/robot/libero/run_libero_eval.py` |
| Checkpoint | [`minivla_joint_ft_lambda2_seed20260910_step2000`](https://huggingface.co/yyshi0619/ECOT-Alignment-Checkpoints/tree/main/minivla_joint_ft_lambda2_seed20260910_step2000) |
| Checkpoint filename | `step-002000-move-lora.pt` |
| Checkpoint size | `5,554,852,935` bytes |
| Checkpoint SHA256 | `0b0ef328c8d4f5478e6da25fde7639c688fdd1ac3b041c60ea06476424b52b1d` |
| Checkpoint `config.json` SHA256 | `b214e101d5d4bbb2d5ea804e6fa762be938551a4a527ee1eda5e21b5a82c71e0` |
| Checkpoint `dataset_statistics.json` SHA256 | `cca56a3704ee47672153415a658daaa7199edf295638604ecc08d3ed1d5c28c4` |
| Case schedule | [`libero90_matched_evaluation_manifest.csv`](libero90_matched_evaluation_manifest.csv) |
| Case-schedule SHA256 | `8c2fcb9c3c5d92be38b9ed787f55d212b2a8aba19065f499585a775351ded564` |
| Reference episode outcomes | [`lambda2_91p11_episode_outcomes.csv`](lambda2_91p11_episode_outcomes.csv) |
| Episode-outcomes SHA256 | `5db476b16dc3947c4dbae1a364d7935116569de988db330b11c08dc342e1b6d5` |
| Original-run diagnostic artifacts | [`lambda2_91p11_reproduction_artifacts/`](lambda2_91p11_reproduction_artifacts/) |
| Artifact hash manifest | [`SHA256SUMS`](lambda2_91p11_reproduction_artifacts/SHA256SUMS) |
| Task 24/state 14/seed 80 trace | [`task24_state14_seed80_trace/`](lambda2_91p11_reproduction_artifacts/task24_state14_seed80_trace/) |
| Task 24/state 14/seed 80 query-1 trace | [`task24_state14_seed80_query1_trace/`](lambda2_91p11_reproduction_artifacts/task24_state14_seed80_query1_trace/) |
| Task 24 attention-kernel profile and vision/projector outputs | [`task24_state14_seed80_attention_profile/`](lambda2_91p11_reproduction_artifacts/task24_state14_seed80_attention_profile/) |
| Task 24 projector five-layer outputs and parameter hashes | [`task24_state14_seed80_projector_layers/`](lambda2_91p11_reproduction_artifacts/task24_state14_seed80_projector_layers/) |
| Projector BF16 reduction and CUDA GEMM-kernel probe | [`projector-bf16-kernels-20260915T110324/`](lambda2_91p11_reproduction_artifacts/projector-bf16-kernels-20260915T110324/) |
| LIBERO commit | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |

The Hugging Face directory contains the checkpoint, `config.json`, and `dataset_statistics.json`. The native loader requires this local layout:

```text
<RUN_DIR>/
├── config.json
├── dataset_statistics.json
└── checkpoints/
    └── step-002000-move-lora.pt
```

Do not load a LoRA adapter on top of this checkpoint. It is the complete native model with the LoRA weights already merged.

## VQ action-tokenizer assets

The evaluator resolves the original VQ-VAE through the working directory shown in the launch script. Use this exact relative layout under `pi05_policy_training/data/minivla_reference`:

```text
vq/pretrain_vq+mx-libero_lm_90+fach-9+ng-7+nemb-256+nlatent-512/
├── config.json
└── checkpoints/
    └── model.pt
```

| Asset | Size | SHA256 |
|---|---:|---|
| VQ `config.json` | 243 bytes | `cc6f6b25a7a2941d7b18f936b3f70eddd22151864f4bfdf38c30f3e4fef0af16` |
| VQ `checkpoints/model.pt` | 9,562,450 bytes | `cf36b5f1534f16ad4bd160d64023bb7707897277dbfc06f68a68b2e26ce9eac9` |
| VQ-BET source commit | — | `09d4851288ca5deaaa1ab367a208e520f8ee9a84` |

The VQ files are required even though the policy checkpoint is otherwise self-contained. A different VQ codebook or configuration changes the decoded continuous action chunks.

## Cached model, tokenizer, and configuration assets

The launch script points `HF_HOME` to `pi05_policy_training/data/minivla_reference/hf_cache` and enables offline mode. The language model cache is `Qwen/Qwen2.5-0.5B`, snapshot revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.

| Cached Qwen file | Size | SHA256 |
|---|---:|---|
| `config.json` | 681 bytes | `479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b` |
| `generation_config.json` | 138 bytes | `8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8` |
| `merges.txt` | 1,671,839 bytes | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `tokenizer.json` | 7,031,645 bytes | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `tokenizer_config.json` | 7,228 bytes | `c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9` |
| `vocab.json` | 2,776,833 bytes | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

The cached vision backbones used by the `dinosiglip` encoder are:

| Backbone and snapshot | Cached weights | Size | SHA256 |
|---|---|---:|---|
| `timm/vit_large_patch14_reg4_dinov2.lvd142m` at `f3c408e77602bb412aa65fb03dfa0d5f95cb3832` | `model.safetensors` | 1,217,515,128 bytes | `c893d72294d4c327e631ff92f428dbc14c4f93cb5581b6c5f9d89bb5d17def27` |
| `timm/ViT-SO400M-14-SigLIP` at `9179d15177ece40964c50492136eda2f3e0c9f61` | `open_clip_model.safetensors` | 3,509,517,656 bytes | `0a04a48b797e187a568335ab67e57a8958c32cee5a707a3feb6d4c97149dcd9c` |

The complete MiniVLA checkpoint already contains the learned model parameters. These cache fingerprints identify the exact tokenizer/config inputs and the backbone assets available to the native loader.

## Exact launch script

Set `REPO`, `RUN_DIR`, and `REPORT_DIR` for the local machine. `RUN_DIR` must use the layout above. The working directory is significant because the action tokenizer resolves the VQ-VAE files from the MiniVLA reference directory.

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO=/absolute/path/to/ECOT-Alignment
PYTHON="$REPO/pi05_policy_training/data/minivla_venv/bin/python"
RUN_DIR=/absolute/path/to/minivla_joint_ft_lambda2_seed20260910_step2000
CHECKPOINT="$RUN_DIR/checkpoints/step-002000-move-lora.pt"
REPORT_DIR=/absolute/path/to/lambda2_270_evaluation
TASK_IDS="$($PYTHON -c 'print(",".join(map(str, range(90))))')"

echo "0b0ef328c8d4f5478e6da25fde7639c688fdd1ac3b041c60ea06476424b52b1d  $CHECKPOINT" | sha256sum -c -

export PYTHONPATH="$REPO:$REPO/pi05_policy_training/openpi/third_party/libero"
export LIBERO_CONFIG_PATH="$REPO/pi05_policy_training/configs/libero"
export PRISMATIC_DATA_ROOT="$REPO/pi05_policy_training/data/rlds"
export HF_HOME="$REPO/pi05_policy_training/data/minivla_reference/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3

cd "$REPO/pi05_policy_training/data/minivla_reference"

"$PYTHON" "$REPO/experiments/robot/libero/run_libero_eval.py" \
  --model_family prismatic \
  --pretrained_checkpoint "$CHECKPOINT" \
  --task_suite_name libero_90 \
  --task_ids "$TASK_IDS" \
  --num_trials_per_task 3 \
  --episodes_per_task 3 \
  --initial_state_offset 13 \
  --seed 7 \
  --center_crop False \
  --use_wrist_image False \
  --use_cot True \
  --num_open_loop_steps 10 \
  --n_procs_per_gpu 6 \
  --enable_alignment_evaluator True \
  --alignment_cosine_similarity True \
  --alignment_save_frames False \
  --use_wandb False \
  --alignment_output_dir "$REPORT_DIR" \
  --output_dir minivla-lambda2-90task-3trial-seed7-state13 \
  --run_id_note minivla-lambda2-90task-3trial-seed7-state13
```

`n_procs_per_gpu=6` only partitions the cases among six workers. It does not change the case assignment, model decoding, or result definition.

## Effective saved configuration

```json
{
  "model_family": "prismatic",
  "task_suite_name": "libero_90",
  "task_ids": "all integers 0 through 89",
  "num_trials_per_task": 3,
  "episodes_per_task": 3,
  "initial_state_offset": 13,
  "initial_state_indices": [13, 14, 15],
  "seed": 7,
  "seed_schedule": "7 + task_id * 3 + trial",
  "episode_seeds": "integers 7 through 276",
  "center_crop": false,
  "use_wrist_image": false,
  "obs_history": 1,
  "use_cot": true,
  "plans": false,
  "unnorm_key": "libero_lm_90",
  "num_open_loop_steps": 10,
  "predicted_action_horizon": 10,
  "executed_actions_before_replan": 10,
  "num_steps_wait": 10,
  "max_episode_steps": null,
  "effective_post_settle_horizon": 400,
  "load_in_4bit": false,
  "load_in_8bit": false,
  "enable_alignment_evaluator": true,
  "alignment_cosine_similarity": true,
  "alignment_save_frames": false,
  "motion_dead_band": 0.003,
  "gripper_open_fraction": 0.5,
  "decoding": "greedy; do_sample=False",
  "inference_precision": "bfloat16"
}
```

For every episode, the evaluator calls `env.seed(episode_seed)`, resets the environment, and then applies `task_suite.get_task_init_states(task_id)[initial_state_index]`. Verify each initial-state byte hash against the case manifest.

The MiniVLA VQ decoder returns a `10 × 7` action chunk. The evaluator takes `actions[0]`, executes all 10 rows before replanning, maps the gripper from `[0,1]` to `[-1,+1]`, binarizes it, and then inverts its sign for LIBERO. Changing any of these details changes task success.

## Reference episode outcomes and successful state IDs

The reference outcome CSV contains one row for every one of the 270 cases. Each row records the case ID, task ID and name, trial, episode seed, initial-state index, initial-state SHA256, success label, policy-query count, all-query mismatch count, and refined eligible-query diagnostics. Use `episode_success` and `initial_state_sha256` for case-level comparison with another machine.

For the following 72 task IDs, all three initial-state indices `13, 14, 15` succeeded:

```text
0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
19, 20, 21, 22, 26, 28, 29, 30, 31, 33, 34, 35, 36, 37, 39, 40,
41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57,
58, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 73, 74, 76, 77, 79,
80, 81, 82, 84, 85, 87, 88, 89
```

The remaining tasks had these outcomes:

| Task ID | Successful initial-state indices | Failed initial-state indices |
|---:|---|---|
| 4 | 13, 15 | 14 |
| 6 | 14, 15 | 13 |
| 12 | 13, 15 | 14 |
| 23 | 15 | 13, 14 |
| 24 | 14, 15 | 13 |
| 25 | 14, 15 | 13 |
| 27 | 13, 15 | 14 |
| 32 | 15 | 13, 14 |
| 38 | 14, 15 | 13 |
| 51 | 13, 14 | 15 |
| 59 | 13, 14 | 15 |
| 65 | 14 | 13, 15 |
| 71 | 13, 15 | 14 |
| 72 | 13 | 14, 15 |
| 75 | 14 | 13, 15 |
| 78 | 13, 14 | 15 |
| 83 | 15 | 13, 14 |
| 86 | 13, 14 | 15 |

This table and the CSV fully specify the 246 successful episodes and 24 failed episodes. The `trial` values `0, 1, 2` correspond to initial-state indices `13, 14, 15` and seeds `7 + task_id × 3 + trial`.

## Execution details for cross-machine diagnosis

- The camera input is the fixed third-person `agentview_image`. The evaluator applies one vertical flip, resizes with Pillow LANCZOS to the model resolution, converts to RGB, and passes one image with no history beyond the current frame. Wrist images, center cropping, and proprioceptive policy input are disabled.
- Each episode calls `env.seed(episode_seed)`, then `env.reset()`, then `env.set_init_state()` with the recorded LIBERO state. It executes 10 dummy no-op actions before policy control. The LIBERO-90 loop permits 400 policy-control steps and stops immediately when the environment reports success.
- Inference uses greedy generation, `do_sample=False`, BF16 autocast, no 4-bit or 8-bit quantization, and no FlashAttention-specific CLI override in the native Prismatic loader.
- The policy emits seven VQ action tokens. The original VQ decoder maps them to a `10 × 7` continuous chunk, and the controller executes all 10 actions before the next policy query.
- The reference run made 4,463 policy queries. Per-episode query totals are in the outcome CSV; query-count differences show that closed-loop trajectories or stopping times have diverged.
- The six local workers processed 45 episodes each and reported success totals `[39, 44, 41, 41, 44, 37]`. Worker count changes scheduling and throughput, while task/state/seed assignment remains defined by the manifest.
- At evaluator startup, `set_seed_everywhere(cfg.seed)` seeds Python, NumPy, PyTorch, and every CUDA device; it also sets `cudnn.deterministic=True` and `cudnn.benchmark=False`. The separate commented `set_seed` call inside the model loader is redundant. PyTorch deterministic algorithms and a CUBLAS workspace configuration are not enabled. Greedy decoding removes sampling but does not guarantee bit-identical BF16 logits across GPU architectures. A token argmax change can select a different VQ code and cause a discontinuous action-chunk change.
- The original report's `experiment_metadata.json` contains legacy descriptive strings saying “one trial per task” and a one-trial seed schedule. Those strings are stale and were not used by execution. The effective config, case manifest, outcome CSV, and evaluator formula in this handoff are authoritative.

For the fastest diagnosis, compare the reference and new runs in this order for the same case: initial-state SHA256, first preprocessed image bytes, first generated token IDs, first decoded `10 × 7` action chunk, then later query tokens. If the initial image matches but the first tokens differ, focus on model/runtime numerics. If initial tokens match and later tokens diverge, compare MuJoCo/robosuite state evolution and CPU-side dependencies.

## Attached original-run diagnostics

The [diagnostic artifact directory](lambda2_91p11_reproduction_artifacts/) supplies the additional files needed for an AWS comparison:

- [`episode_summary.json`](lambda2_91p11_reproduction_artifacts/episode_summary.json) contains all 270 reference outcomes and policy-query counts.
- [`first_query_fingerprints.csv`](lambda2_91p11_reproduction_artifacts/first_query_fingerprints.csv) contains the original first-query VQ token IDs and complete decoded action chunk for every case, plus hashes of the generated reasoning and pre-action simulator state.
- [`alignment_queries.jsonl.gz`](lambda2_91p11_reproduction_artifacts/alignment_queries.jsonl.gz) contains all 4,463 policy queries, decoded actions, executed end-effector trajectories, reasoning, and simulator states. Its compressed SHA256 is `b9fa977167de81bf812ecbf513cc7de49eff3fa44e9dd0f43a60d08311036cea`; the decompressed JSONL SHA256 is `ebcae6aec77750e31e7ec91c596bddc4570a5281fb863748edff45d25aa83254`.
- [`all_query_scores.jsonl`](lambda2_91p11_reproduction_artifacts/all_query_scores.jsonl) contains the exact all-query and eligible-query mismatch decisions.
- [`experiment_metadata.json`](lambda2_91p11_reproduction_artifacts/experiment_metadata.json), [`run_summary.json`](lambda2_91p11_reproduction_artifacts/run_summary.json), and [`native_eval_results.jsonl`](lambda2_91p11_reproduction_artifacts/native_eval_results.jsonl) preserve the saved evaluator outputs.
- [`worker_logs/`](lambda2_91p11_reproduction_artifacts/worker_logs/) contains the six original worker logs.
- [`pip_freeze.txt`](lambda2_91p11_reproduction_artifacts/pip_freeze.txt), [`torch_collect_env.txt`](lambda2_91p11_reproduction_artifacts/torch_collect_env.txt), and [`robosuite_macros.py`](lambda2_91p11_reproduction_artifacts/robosuite_macros.py) record the environment and simulator macros. No `macros_private.py` existed, so the published default macro file was effective.
- [`checkpoint_config.json`](lambda2_91p11_reproduction_artifacts/checkpoint_config.json), [`dataset_statistics.json`](lambda2_91p11_reproduction_artifacts/dataset_statistics.json), and [`training_config.json`](lambda2_91p11_reproduction_artifacts/training_config.json) preserve the local configuration files.
- [`original_lambda_sweep_launcher.py`](lambda2_91p11_reproduction_artifacts/original_lambda_sweep_launcher.py) is the actual orchestration script that launched the original run. The cleaned portable command earlier in this handoff remains the recommended command for another machine.

The original run did not save lossless camera frames because `alignment_save_frames=False`. The available rollout videos are lossy and cannot serve as byte-level image fingerprints. To isolate the H100 gap without rerunning the workstation evaluation, first compare the AWS outcome file with `lambda2_91p11_episode_outcomes.csv`, then compare first-query token IDs with `first_query_fingerprints.csv` for the changed cases.

### Targeted task 24 trace

The requested [task 24, state 14, seed 80 trace](lambda2_91p11_reproduction_artifacts/task24_state14_seed80_trace/) was captured in a new diagnostic replay while preserving the original state 13 → 14 → 15 execution order. It reproduced the original task outcomes `failure, success, success`. State 14 reproduced the archived action tokens `[151804, 151894, 151751, 151802, 151817, 151751, 151780]` and decoded-action SHA256 `90d24c34086b2a4c4ce0cc637bb3dcf7cb0690749f86e37d6c8a7cf727b7ec41` exactly.

The trace supplies the lossless first-query RGB image, exact BF16 DINO and SigLIP tensors, prompt text and token IDs, complete generated token IDs, decoded 10×7 action chunk, top-two logits for every action-token position, live GPU/backend/determinism flags after model loading, generation configuration, and hashes of the imported evaluator, policy, and robosuite helper files.

One action-token position has a zero stored BF16 margin: the model selected token `151894`, while `151916` and `151894` both recorded logits of `24.125`. This is direct evidence that small rendering, preprocessing, or GPU-kernel numerical differences can flip a VQ token despite greedy decoding. Compare the AWS policy on the packaged `exact_model_inputs.pt`: matching tensors with different generated IDs isolate the gap to inference numerics; different DINO/SigLIP tensor hashes isolate it to rendering or preprocessing.

### Follow-up policy-query-1 trace

The follow-up [task 24, state 14, seed 80, policy-query-1 trace](lambda2_91p11_reproduction_artifacts/task24_state14_seed80_query1_trace/) replayed the same state order and again produced `failure, success, success`, with per-trial policy-query counts `40, 13, 11`. Query 1 exactly reproduced the original archived reasoning hash, pre-action simulator-state hash, tokens `[151723, 151869, 151763, 151796, 151785, 151673, 151895]`, and decoded action-chunk SHA256 `85d307fa3399cdcddef5726a95a0e1b7fbe8714154ba69c73dc22e13b8bb7cfa`.

At action-token position 4, the workstation BF16 score for reference token `151785` is `24.375`; the score for AWS math-SDPA token `151746` is `24.000`. They are the top two candidates, separated by only `0.375`. The trace includes the lossless query-1 image, exact input tensors, full generation, all seven top-two comparisons, both requested candidate scores, live runtime settings, and source hashes.

The requested capture coordinates and acceptance checks are recorded explicitly here:

| Field | Recorded value |
|---|---|
| Task / initial state / seed | `24 / 14 / 80` |
| Episode ID | `libero_90-task24-episode1-seed80` |
| Policy query index | `1` (the second policy query) |
| Global generation call | `41` when counted from zero: 40 calls in state 13, then query 0 in state 14 |
| Original trial order | state `13 → 14 → 15` |
| Replayed outcomes | `failure, success, success` |
| Replayed query counts | `40, 13, 11` |
| Exact action tokens | `[151723, 151869, 151763, 151796, 151785, 151673, 151895]` |
| Requested fifth-token comparison | reference `151785`: `24.375`; AWS math-SDPA `151746`: `24.000` |

The trace directory contains the lossless RGB image after the normal 10 settling steps, exact prompt IDs, exact BF16 DINO and SigLIP tensors, complete generated token IDs, decoded 10×7 action chunk, top-two scores at every action-token position, `state_before`, `state_after`, executed end-effector trajectory, runtime/source fingerprints, SHA256 manifest, and the capture script. It preserves the earlier first-query trace in its separate directory.

### Selected attention-kernel profile

The enabled-backend flags in the earlier traces were insufficient to identify the kernel that PyTorch actually dispatched. The requested [workstation profiler record](lambda2_91p11_reproduction_artifacts/task24_state14_seed80_attention_profile/) therefore replayed the same two state-14 queries under unchanged BF16, greedy-decoding, default-SDPA settings and captured the selected operator separately for the vision encoders, language prefill, and cached decoding.

| Query | Phase | Actual selected operator | Calls |
|---:|---|---|---:|
| 0 | DINO/SigLIP vision encoders | `aten::_scaled_dot_product_flash_attention` | 51 |
| 0 | Qwen language prefill | `aten::_scaled_dot_product_flash_attention` | 24 |
| 0 | Qwen cached decoding | `aten::_scaled_dot_product_flash_attention` | 6,288 |
| 1 | DINO/SigLIP vision encoders | `aten::_scaled_dot_product_flash_attention` | 51 |
| 1 | Qwen language prefill | `aten::_scaled_dot_product_flash_attention` | 24 |
| 1 | Qwen cached decoding | `aten::_scaled_dot_product_flash_attention` | 6,768 |

The RTX PRO 6000 workstation selected PyTorch **Flash SDPA in all three phases** for both queries. Neither `aten::_scaled_dot_product_efficient_attention`, `aten::_scaled_dot_product_cudnn_attention`, nor `aten::_scaled_dot_product_attention_math` appeared as the selected implementation. The JSON profiler records retain the individual selected-operator events and input shapes.

The same artifact directory includes exact CPU copies of the BF16 vision and projector outputs for direct AWS comparison. Query 0 produced vision output shape `1×256×2176` with raw-tensor SHA256 `8a0c8e7869866f94929a57666b3050386b250924bd6dbe50fa050d2e7896ef41` and projector output shape `1×256×896` with SHA256 `bd1bdef0f02c48773ec884bb25321f65b44a976fb89762bb25977331d5a72c33`. Query 1 produced corresponding hashes `707a20ea79285f9544e3f60ccf75df032b70c95766aafe9dd029133268bfccc6` and `bf8397e0a1af4308b80d9da62330e39c654464767af82a4e8183c418e6dc2e0a`.

### Projector-only five-layer replay

The requested [projector-layer diagnostic](lambda2_91p11_reproduction_artifacts/task24_state14_seed80_projector_layers/) fed each saved BF16 `vision_encoder_output` directly through the original loaded projector. It did not run the simulator, vision encoders, or language model. The live module was `prismatic.util.nn_utils.FusedMLPProjector` with the following exact structure:

```text
Linear(2176 -> 8704) -> GELU -> Linear(8704 -> 896) -> GELU -> Linear(896 -> 896)
```

The combined live-projector parameter hash is `fe13bd8f3745da1088fc1ac4d32bc6243f53073a1722daf3b212a239dfa6b768`. Its source file is `prismatic/util/nn_utils.py`, SHA256 `7aa0562eab374f0585bb53639b861f06b6c44b82384e938dda9cd32d844bc0d0`. The machine-readable report includes hashes for all six individual weight/bias tensors.

| Query | Replayed final-output hash | Archived final-output hash | Exact equality | Maximum absolute difference |
|---:|---|---|---|---:|
| 0 | `bd1bdef0f02c48773ec884bb25321f65b44a976fb89762bb25977331d5a72c33` | `bd1bdef0f02c48773ec884bb25321f65b44a976fb89762bb25977331d5a72c33` | Yes | `0.0` |
| 1 | `bf8397e0a1af4308b80d9da62330e39c654464767af82a4e8183c418e6dc2e0a` | `bf8397e0a1af4308b80d9da62330e39c654464767af82a4e8183c418e6dc2e0a` | Yes | `0.0` |

The two saved `.pt` artifacts contain the vision input, outputs after each of the five layers, and the archived final output. The result shows no unexpected difference inside the workstation projector replay.

### Projector BF16 reduction and CUDA GEMM kernels

The follow-up [BF16 CUDA-kernel probe](lambda2_91p11_reproduction_artifacts/projector-bf16-kernels-20260915T110324/) used only the saved projector inputs and the original checkpoint. The workstation default is:

```text
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
```

Under that default, both query outputs matched the archived projector tensors byte-for-byte. Disabling reduced-precision BF16 reduction changed both outputs, with maximum absolute difference `0.0625` in each query.

| Query | BF16 reduced reduction | Exact archive match | Maximum absolute difference |
|---:|---|---|---:|
| 0 | `True` (default) | Yes | `0.0` |
| 1 | `True` (default) | Yes | `0.0` |
| 0 | `False` | No | `0.0625` |
| 1 | `False` | No | `0.0625` |

With the default enabled, the profiler observed three CUTLASS BF16 GEMM kernels ending in `128x64_32x6_tn_align8`, `64x256_32x4_tn_align8`, and `32x32_128x2_tn_align8`. With the flag disabled, the `128x64_32x6` kernel changed to `64x64_64x6`; the other two remained unchanged. The artifact report contains the full profiler kernel names and event records. The reusable capture program is also available at repository root as [`projector_bf16_kernel_probe.py`](projector_bf16_kernel_probe.py).

## Mismatch definition used in the paper table

The paper table pools **all policy queries**. For directional MOVE commands, mismatch means commanded-translation cosine similarity below `0.5`. Gripper-only, rotation-only, missing, or empty XYZ MOVE commands are assigned expected translation `[0,0,0]` and included as implicit-stop mismatches. Do not remove these queries from the denominator.

The alternative refined eligible-query score is useful diagnostically but is not the table metric:

| Score | Result |
|---|---:|
| All-query mismatch used in the table | **360/4,463 = 8.07%** |
| Refined eligible-query mismatch | 228/4,331 = 5.26% |

## Reproduction checks

Both runs on the original workstation produced the same case-level outcomes:

| Result | Original run | Replication |
|---|---:|---:|
| Successful episodes | 246/270 | 246/270 |
| Task success | 91.11% | 91.11% |
| Policy queries | 4,463 | 4,463 |
| All-query mismatches | 360 | 360 |
| Case-outcome differences | — | 0 |

The six worker success totals must be `[39, 44, 41, 41, 44, 37]`, with 45 episodes per worker. If these differ, compare checkpoint SHA256, initial-state hashes, image preprocessing, `unnorm_key`, greedy decoding, action-chunk extraction, gripper processing, and the predict-10/execute-10 controller before comparing aggregate rates.

## Workstation, driver, and dependency fingerprint

| Component | Reproduced-run value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| GPU UUID | `GPU-59541196-bb6d-d1c9-22af-24ff7b0d2c92` |
| GPU memory | 97,887 MiB |
| VBIOS | `98.02.81.00.07` |
| NVIDIA driver | `575.64.05` |
| Driver CUDA compatibility | CUDA 12.9 |
| PyTorch | `2.8.0+cu128` (CUDA 12.8 build) |
| Local `nvcc` toolkit | CUDA 11.6, build `11.6.55` |
| Python | 3.10 |

The local `nvcc` version differs from the PyTorch CUDA build. Evaluation used the prebuilt PyTorch `cu128` runtime with driver `575.64.05`; it did not compile a custom CUDA extension.

| Image, model, and simulator dependency | Version |
|---|---:|
| Pillow | `12.3.0` |
| torchvision | `0.23.0+cu128` |
| timm | `0.9.10` |
| OpenCV (`opencv-python`) | `4.6.0.66` |
| TensorFlow | `2.15.0` |
| einops | `0.8.2` |
| NumPy | `1.26.4` |
| SciPy | `1.10.1` |
| Transformers | `4.40.1` |
| tokenizers | `0.19.1` |
| safetensors | `0.8.0` |
| huggingface-hub | `0.36.2` |
| MuJoCo | `3.2.3` |
| robosuite | `1.4.1` |
| LIBERO | `0.1.0` |
| PEFT | `0.11.1` |

The preprocessing behavior is fixed by the pinned evaluator commit and `center_crop=False`, `use_wrist_image=False`, and `obs_history=1` in the saved configuration. For an additional integrity check, the evaluator source fingerprints at that checkout are:

| Source file | SHA256 |
|---|---|
| `experiments/robot/libero/run_libero_eval.py` | `47502933b361790829030b033dc3a2ad74a8f3c2e19fe28b5d2e9b77c27a32ac` |
| `experiments/robot/libero/libero_utils.py` | `d7a479847350b662e089476a3f967aa84b154d981df960d9f2d1692e6b5d5148` |
| `experiments/robot/openvla_utils.py` | `859ef53e0745005fdcbd63354aa09468a8f3f6fe995277feb548015ca747bf53` |
