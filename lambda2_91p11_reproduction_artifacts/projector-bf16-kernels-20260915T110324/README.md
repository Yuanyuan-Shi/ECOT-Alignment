# BF16 projector CUDA-kernel probe

This projector-only probe used the original Lambda=2 checkpoint and the two saved task-24/state-14/seed-80 vision tensors. It ran no simulator or full evaluation. The root-level `projector_bf16_kernel_probe.py` is the reusable launch script; the copy in this directory is the exact script saved by this run.

## Result

The workstation default was:

```text
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
```

With the default enabled, both query outputs matched their archived BF16 projector outputs exactly. Disabling the flag changed both outputs:

| Query | BF16 reduced reduction | Exact archive match | Maximum absolute difference | Output SHA256 |
|---:|---|---|---:|---|
| 0 | `True` (workstation default) | Yes | `0.0` | `bd1bdef0f02c48773ec884bb25321f65b44a976fb89762bb25977331d5a72c33` |
| 1 | `True` (workstation default) | Yes | `0.0` | `bf8397e0a1af4308b80d9da62330e39c654464767af82a4e8183c418e6dc2e0a` |
| 0 | `False` | No | `0.0625` | `b2277e7f7aae8244697381072bf4c79a57549b62c9f130e99aeccd338f7d8600` |
| 1 | `False` | No | `0.0625` | `a3643c78c9c23db0892aafc663c8aadfd1e39bd96c06b66286535654f335c5fc` |

## Actual CUDA matrix-multiplication kernels

With the workstation default (`True`), both queries selected these three kernels, one call each:

```text
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_tn_align8>(...)
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x256_32x4_tn_align8>(...)
void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(...)
```

With reduced-precision BF16 reduction disabled, the `128x64_32x6` kernel was replaced by:

```text
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8>(...)
```

The other two kernels were unchanged. `projector_bf16_kernel_report.json` preserves the full unmangled profiler names, CUDA events and parent chains, runtime flags, output comparisons, live parameter hashes, and source hashes.

## Exact command

```bash
REPO=/mnt1/ECOT-Alignment
"$REPO/pi05_policy_training/data/minivla_venv/bin/python" \
  "$REPO/projector_bf16_kernel_probe.py" \
  --repo "$REPO" \
  --checkpoint "$REPO/pi05_policy_training/runs/minivla_lambda_sweep_20260914/large-data-lambda2-b32-s2000-20260914/checkpoints/step-002000-move-lora.pt" \
  --inputs "$REPO/pi05_policy_training/reports/lambda2_task24_state14_seed80_projector_layers" \
  --output "$REPO/pi05_policy_training/reports/projector-bf16-kernels-20260915T110324"
```

