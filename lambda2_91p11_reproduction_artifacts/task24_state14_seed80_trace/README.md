# Targeted trace: task 24, state 14, seed 80

This trace replays task 24 in the original trial order: state 13/seed 79, state 14/seed 80, then state 15/seed 81. The outcomes reproduced the original `failure, success, success` sequence. Only the first policy query for state 14/seed 80 is packaged here.

| Artifact | Contents |
|---|---|
| `task24_state14_seed80_first_query.png` | Lossless 224×224 RGB PNG passed into the model preprocessing path after 10 settling steps |
| `exact_model_inputs.pt` | Exact prompt IDs and BF16 DINO/SigLIP tensors used by inference |
| `inference_trace.json` | Prompt text/IDs, complete generated IDs, action-token top-two logits, tensor hashes, runtime backend settings, generation config, and imported-source hashes |
| `first_query_record.json` | Generated reasoning, VQ IDs, decoded 10×7 action chunk, executed trajectory, and simulator states |
| `trace_summary.json` | Compact integrity and reproduction summary |
| `capture_script.py` | Standalone instrumentation used to create this trace without editing the evaluator |

Load the exact tensors with:

```python
import torch

inputs = torch.load("exact_model_inputs.pt", map_location="cpu", weights_only=True)
prompt_ids = inputs["input_ids"]
dino = inputs["pixel_values"]["dino"]
siglip = inputs["pixel_values"]["siglip"]
```

The original action tokens are:

```text
[151804, 151894, 151751, 151802, 151817, 151751, 151780]
```

The decoded action-chunk SHA256 is `90d24c34086b2a4c4ce0cc637bb3dcf7cb0690749f86e37d6c8a7cf727b7ec41`, matching the previously archived first-query fingerprint exactly.

The second action-token position is numerically unstable in BF16. The chosen token was `151894`, while tokens `151916` and `151894` both had a stored BF16 logit of `24.125`, giving a recorded margin of zero. A small backend-dependent change before BF16 rounding can change this argmax and select a different VQ action chunk.

Stable tensor hashes for direct comparison:

| Tensor | Shape and dtype | Raw-byte SHA256 |
|---|---|---|
| Prompt IDs | `1×50`, `torch.int64` | `c88d98990065f1c9087b5849136eab5d323e39d699cfbc282481acbd3372bc04` |
| DINO pixels | `1×3×224×224`, `torch.bfloat16` | `c838e89be2a48187462780690251e4180350bb33dd764a656d8b2ed7bde3337e` |
| SigLIP pixels | `1×3×224×224`, `torch.bfloat16` | `f9900824c9d883d256853f24ba8083904f092dfafd34f2cea01664b101011056` |

The live runtime used PyTorch `2.8.0+cu128`, SDPA attention, BF16, compute capability 12.0, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `cuda.matmul.allow_tf32=False`, and `cudnn.allow_tf32=True`. PyTorch deterministic algorithms were not globally enabled. See `inference_trace.json` for the full generation configuration and source-file hashes.
