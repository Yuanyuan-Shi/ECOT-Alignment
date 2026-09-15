# Targeted trace: task 24, state 14, seed 80, policy query 1

This trace captures the second policy query (`policy_query_index=1`) requested after the first-query comparison. It replayed task 24 in the original state order `13 → 14 → 15` and reproduced the original outcomes and per-trial query counts:

```text
outcomes:     failure, success, success
query counts: 40, 13, 11
```

| Artifact | Contents |
|---|---|
| `task24_state14_seed80_query001.png` | Lossless 224×224 RGB image at policy query 1 |
| `exact_model_inputs.pt` | Exact prompt IDs and BF16 DINO/SigLIP tensors used by query-1 inference |
| `inference_trace.json` | Complete generated IDs, action-token logits, requested candidate scores, runtime settings, generation config, and source hashes |
| `query001_record.json` | Reasoning, VQ tokens, decoded 10×7 action chunk, executed trajectory, and simulator state before/after the chunk |
| `trace_summary.json` | Compact integrity and outcome summary |
| `capture_script.py` | Standalone instrumentation used for this trace |

Load the exact tensors with:

```python
import torch

inputs = torch.load("exact_model_inputs.pt", map_location="cpu", weights_only=True)
prompt_ids = inputs["input_ids"]
dino = inputs["pixel_values"]["dino"]
siglip = inputs["pixel_values"]["siglip"]
```

The reproduced action tokens are:

```text
[151723, 151869, 151763, 151796, 151785, 151673, 151895]
```

The reasoning, pre-action simulator state, action tokens, and decoded action chunk all match the original archived query exactly. The decoded action-chunk SHA256 is `85d307fa3399cdcddef5726a95a0e1b7fbe8714154ba69c73dc22e13b8bb7cfa`.

At action-token position 4, the two specifically requested logits are:

| Candidate | Workstation BF16 logit |
|---|---:|
| Reference token `151785` | `24.375` |
| AWS math-SDPA token `151746` | `24.000` |
| Reference-minus-AWS margin | `0.375` |

These are also the top two candidates at that position. The small BF16 margin supports an inference-numerics explanation if AWS produces different tokens from the exact same packaged tensors.

Stable tensor hashes:

| Tensor | Shape and dtype | Raw-byte SHA256 |
|---|---|---|
| Prompt IDs | `1×50`, `torch.int64` | `c88d98990065f1c9087b5849136eab5d323e39d699cfbc282481acbd3372bc04` |
| DINO pixels | `1×3×224×224`, `torch.bfloat16` | `8aeab0e3f5b1cca9f2929a1ad67be2c51d0b0c5081b3dd535618a7cb129d8307` |
| SigLIP pixels | `1×3×224×224`, `torch.bfloat16` | `11c3c854df3dc1902c4098640cfba119ab1c41b4c607ddeddc0eec3fae5921bd` |

The live runtime/source fingerprint format is identical to the first-query trace. See `inference_trace.json` for SDPA/TF32/determinism flags, generation configuration, and imported-source SHA256 values.
