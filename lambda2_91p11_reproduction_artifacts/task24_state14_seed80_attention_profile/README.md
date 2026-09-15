# Workstation attention-kernel profile: task 24, state 14, seed 80

This is the requested profiler record for policy queries 0 and 1 under the unchanged Lambda=2 evaluation setup: BF16 model execution, greedy decoding, and PyTorch's default SDPA dispatcher settings. The replay preserved the original state order `13 -> 14 -> 15`, outcomes `failure, success, success`, per-trial query counts `40, 13, 11`, and the archived action tokens for both profiled queries.

## Actual selected SDPA operators

The PyTorch profiler recorded the selected implementation operator, rather than only the enabled-backend flags:

| Query | Model phase | Selected operator | Calls |
|---:|---|---|---:|
| 0 | DINO/SigLIP vision encoders | `aten::_scaled_dot_product_flash_attention` | 51 |
| 0 | Qwen language prefill | `aten::_scaled_dot_product_flash_attention` | 24 |
| 0 | Qwen cached decoding | `aten::_scaled_dot_product_flash_attention` | 6,288 |
| 1 | DINO/SigLIP vision encoders | `aten::_scaled_dot_product_flash_attention` | 51 |
| 1 | Qwen language prefill | `aten::_scaled_dot_product_flash_attention` | 24 |
| 1 | Qwen cached decoding | `aten::_scaled_dot_product_flash_attention` | 6,768 |

Thus, the original RTX PRO 6000 workstation selected PyTorch **Flash SDPA for all three requested phases** in both traces. It did not select the memory-efficient, cuDNN, or math SDPA operator in either profiled generation. Vision calls are labeled by the vision-backbone hook. Within Qwen, the prefill call has query length 306; cached-decoding calls have query length 1 and a growing KV length.

## Saved vision and projector tensors

Each `.pt` file is a CPU copy of the exact BF16 outputs produced during the corresponding profiled query:

| Query | Tensor | Shape | Raw-tensor SHA256 |
|---:|---|---|---|
| 0 | Vision encoder output | `1 x 256 x 2176` | `8a0c8e7869866f94929a57666b3050386b250924bd6dbe50fa050d2e7896ef41` |
| 0 | Projector output | `1 x 256 x 896` | `bd1bdef0f02c48773ec884bb25321f65b44a976fb89762bb25977331d5a72c33` |
| 1 | Vision encoder output | `1 x 256 x 2176` | `707a20ea79285f9544e3f60ccf75df032b70c95766aafe9dd029133268bfccc6` |
| 1 | Projector output | `1 x 256 x 896` | `bf8397e0a1af4308b80d9da62330e39c654464767af82a4e8183c418e6dc2e0a` |

Load them with:

```python
import torch

outputs = torch.load("query0_vision_projector_outputs.pt", map_location="cpu", weights_only=True)
vision = outputs["vision_encoder_output"]
projector = outputs["projector_output"]
```

The two `query*_attention_profile.json` files retain every selected-operator event, its phase, input shapes, timings, tensor metadata, runtime flags, generation configuration, and source fingerprints. `workstation_kernel_probe.py` is the exact capture script. `SHA256SUMS` covers every artifact in this directory.

