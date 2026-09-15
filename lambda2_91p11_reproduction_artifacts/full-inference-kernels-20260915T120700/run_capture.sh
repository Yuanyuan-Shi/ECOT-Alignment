#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/mnt1/ECOT-Alignment}
PYTHON="$REPO/pi05_policy_training/data/minivla_venv/bin/python"
CHECKPOINT="$REPO/pi05_policy_training/runs/minivla_lambda_sweep_20260914/large-data-lambda2-b32-s2000-20260914/checkpoints/step-002000-move-lora.pt"
Q0="$REPO/pi05_policy_training/reports/lambda2_task24_state14_seed80_trace"
Q1="$REPO/pi05_policy_training/reports/lambda2_task24_state14_seed80_query1_trace"
OUT=${OUT:-"$REPO/pi05_policy_training/reports/full-inference-kernels-reproduction"}
CUBLAS_DIR="$REPO/pi05_policy_training/data/minivla_venv/lib/python3.10/site-packages/nvidia/cublas/lib"

gcc -shared -fPIC -O2 \
  -I "$REPO/pi05_policy_training/data/minivla_venv/lib/python3.10/site-packages/nvidia/cublas/include" \
  -I "$REPO/pi05_policy_training/data/minivla_venv/lib/python3.10/site-packages/nvidia/cuda_runtime/include" \
  -I /usr/local/cuda-11.6/targets/x86_64-linux/include \
  -o "$REPO/libcublaslt_algo_interposer.so" "$REPO/cublaslt_algo_interposer.c" -ldl -pthread

mkdir -p "$OUT"
cd "$REPO/pi05_policy_training/data/minivla_reference"
env LD_PRELOAD="$REPO/libcublaslt_algo_interposer.so:$CUBLAS_DIR/libcublasLt.so.12:$CUBLAS_DIR/libcublas.so.12" \
  "$PYTHON" "$REPO/full_inference_kernel_probe.py" \
  --repo "$REPO" --checkpoint "$CHECKPOINT" \
  --query0-input "$Q0/exact_model_inputs.pt" --query0-reference "$Q0/inference_trace.json" \
  --query1-input "$Q1/exact_model_inputs.pt" --query1-reference "$Q1/inference_trace.json" \
  --output "$OUT"

for query in 0 1; do
  "$PYTHON" "$REPO/postprocess_full_inference_trace.py" \
    --trace "$OUT/query${query}_full_profiler_trace.json.gz" \
    --output "$OUT/query${query}_correlated_matmuls.json"
done

