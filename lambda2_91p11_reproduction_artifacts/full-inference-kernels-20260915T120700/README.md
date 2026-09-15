# Full inference kernel capture for task 24/state 14/seed 80

This capture runs the two previously saved model inputs directly through the original λ=2 checkpoint. It does not launch LIBERO or alter the input tensors. Both queries reproduced their archived complete generated-token sequences exactly.

| Query | Generated tokens | Exact reference match | Correlated CUDA matrix events |
|---:|---:|---|---:|
| 0 | 263 | Yes | 57,381/57,382 classified |
| 1 | 283 | Yes | 61,720/61,722 classified |

The raw Chrome traces contain all CPU and CUDA events. The correlated reports link actual CUDA GEMM, GEMV, CUTLASS, and attention kernels to model module, language prefill/cached-decode step, CPU operator, input shapes, dtypes, and strides. The descriptor log records 39,726 cuBLASLt calls with their model context, matrix layouts, compute/data types, algorithm ID, tile, stages, split-K, reduction scheme, swizzle, custom option, workspace, epilogue, and serialized 64-byte descriptor.

The descriptor pass observed six configurations. Cached language decoding used algorithm 13 for 39,168 cuBLASLt calls. Query 0 generation step 255, where AWS reported its first differing generated token, has 72 full cuBLASLt records in `cublaslt_descriptor_summary.json`; the corresponding standard cuBLAS GEMV and Flash Attention kernel launches are in `query0_correlated_matmuls.json.gz`.

`LD_PRELOAD` must pin the exact PyTorch cuBLAS libraries after the interceptor, as shown in `run_capture.sh`. Without those explicit libraries, TensorFlow's earlier-loaded cuBLAS instance can make `RTLD_NEXT` bind incorrectly and force fused GEMMs to fall back. The validated capture emitted no fallback warnings and preserved both token sequences exactly.

The two `query*_full_profiler_trace.json.gz` files are complete Chrome traces. `cublaslt_calls.jsonl.gz` and `query*_correlated_matmuls.json.gz` are gzip-compressed JSON/JSONL and can be read without extraction using Python's `gzip` module.
