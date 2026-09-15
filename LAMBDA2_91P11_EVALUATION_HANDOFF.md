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
