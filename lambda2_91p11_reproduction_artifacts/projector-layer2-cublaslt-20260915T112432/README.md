# Projector layer-2 cuBLASLt algorithm capture

This probe isolates `projector.projector[2]`, the BF16 `Linear(8704 -> 896, bias=True)` layer. It feeds the two saved layer-1 GELU outputs into the original checkpoint weights with the workstation's original setting:

```text
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
```

No simulator, vision encoder, language model, or full evaluation was run. Both resulting tensors match the archived layer-2 outputs byte-for-byte with maximum absolute difference `0.0`.

## Exact selected cuBLASLt configuration

The LD_PRELOAD interposer queried the opaque algorithm passed by PyTorch immediately before every `cublasLtMatmul` call. All seven `cublasLtMatmulAlgoConfigGetAttribute` calls returned success (`0`), and all eight GEMM calls used the same descriptor:

| Attribute | Value |
|---|---|
| Algorithm ID | `21` |
| Tile ID | `18` = `CUBLASLT_MATMUL_TILE_128x64` |
| Stages ID | `12` = `CUBLASLT_MATMUL_STAGES_32x6` |
| Split-K count | `6` |
| Reduction scheme | `1` = `CUBLASLT_REDUCTION_SCHEME_INPLACE` |
| CTA swizzle | `0` |
| Custom option | `0` |
| Workspace | `1,048,576` bytes; non-null |
| Compute type | `COMPUTE_32F` |
| Scale type | `R_32F` |
| Input/output type | `R_16BF` |
| Transpose | `OP_T` for A |
| Epilogue | `EPILOGUE_BIAS` |
| `CUBLAS_WORKSPACE_CONFIG` | Unset |

The exact serialized 64-byte `cublasLtMatmulAlgo_t` descriptor, represented as eight little-endian `uint64` words, is:

```text
0000001200000015
000000060000000c
0000000000000001
0000000000000000
0001d7e400000001
0000000e0000000e
00000044000e000e
0000000000000000
```

The selected CUDA kernel was:

```text
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_tn_align8>(...)
```

The cuBLASLt heuristic request allowed a maximum workspace of `1,048,576` bytes and returned algorithm `21`. The raw API/performance/heuristics trace is in `cublaslt_logger_messages.jsonl`; the independently intercepted attribute values and serialized descriptors are in `algo_descriptors.jsonl`.

## Library fingerprint

| Library | Size | SHA256 |
|---|---:|---|
| `libcublasLt.so.12` | `751,771,728` bytes | `10b5e6631cf8115c661eb895ed1533826308b58f7956466f53d236a40c9b622c` |
| `libcublas.so.12` | `116,388,640` bytes | `031ce6c2cbfbb9468f040527cab5c599069ce5609e73e28f87503881063eac21` |

## Capture implementation

`projector_layer2_cublaslt_probe.py` performs the isolated layer replay and profiler check. `cublaslt_algo_interposer.c` queries the actual descriptor passed by PyTorch. The included `.so` is the exact workstation build; rebuilding it is preferable on another machine.

```bash
REPO=/mnt1/ECOT-Alignment
CUDA_PY="$REPO/pi05_policy_training/data/minivla_venv/lib/python3.10/site-packages/nvidia"

gcc -shared -fPIC -O2 \
  -I"$CUDA_PY/cublas/include" \
  -I"$CUDA_PY/cuda_runtime/include" \
  -I/usr/local/cuda-11.6/targets/x86_64-linux/include \
  -o "$REPO/libcublaslt_algo_interposer.so" \
  "$REPO/cublaslt_algo_interposer.c" -ldl -pthread

env \
  LD_PRELOAD="$REPO/libcublaslt_algo_interposer.so" \
  ECOT_CUBLASLT_CAPTURE_FILE="$REPO/pi05_policy_training/reports/projector-layer2-cublaslt-20260915T112432/algo_descriptors.jsonl" \
  "$REPO/pi05_policy_training/data/minivla_venv/bin/python" \
  "$REPO/projector_layer2_cublaslt_probe.py" \
  --repo "$REPO" \
  --checkpoint "$REPO/pi05_policy_training/runs/minivla_lambda_sweep_20260914/large-data-lambda2-b32-s2000-20260914/checkpoints/step-002000-move-lora.pt" \
  --inputs "$REPO/pi05_policy_training/reports/lambda2_task24_state14_seed80_projector_layers" \
  --output "$REPO/pi05_policy_training/reports/projector-layer2-cublaslt-20260915T112432"
```

cuBLASLt documents its algorithm descriptor as reusable only with the same cuBLAS version. The H100 comparison should therefore report these attributes and hashes rather than blindly loading the workstation descriptor.

