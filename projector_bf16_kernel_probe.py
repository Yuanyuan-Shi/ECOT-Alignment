#!/usr/bin/env python3
"""Profile CUDA projector kernels and BF16 reduced-precision reduction behavior."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


ARGS = parse_args()
REPO = ARGS.repo.resolve()
ROOT = REPO / "pi05_policy_training"
REFERENCE = ROOT / "data/minivla_reference"
os.environ.setdefault("HF_HOME", str(REFERENCE / "hf_cache"))
os.environ.setdefault("PRISMATIC_DATA_ROOT", str(ROOT / "data/rlds"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(REPO))

import torch

from prismatic.util.nn_utils import FusedMLPProjector
import prismatic.util.nn_utils as nn_utils


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_record(tensor: torch.Tensor) -> dict:
    cpu = tensor.detach().contiguous().cpu()
    return {"shape": list(cpu.shape), "dtype": str(cpu.dtype), "sha256_raw_bytes": tensor_sha256(cpu)}


def parameter_record(module: torch.nn.Module) -> dict:
    combined = hashlib.sha256()
    rows = []
    for name, tensor in module.named_parameters():
        cpu = tensor.detach().contiguous().cpu()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        combined.update(b"parameter\0" + name.encode() + b"\0" + raw)
        rows.append({"name": name, **tensor_record(cpu)})
    return {"combined_sha256": combined.hexdigest(), "parameters": rows}


def parent_names(event) -> list[str]:
    names = []
    parent = event.cpu_parent
    while parent is not None:
        names.append(parent.name)
        parent = parent.cpu_parent
    return names


def profiler_record(prof) -> dict:
    cuda_counts = Counter()
    matmul_counts = Counter()
    cuda_events = []
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        parents = parent_names(event)
        cuda_counts[event.name] += 1
        is_matmul = any(
            parent in {"aten::linear", "aten::addmm", "aten::mm", "aten::matmul"} for parent in parents
        ) or any(token in event.name.lower() for token in ("gemm", "matmul", "mma"))
        if is_matmul:
            matmul_counts[event.name] += 1
        cuda_events.append(
            {
                "kernel": event.name,
                "counted_as_matmul": is_matmul,
                "cpu_parent_chain": parents,
                "device_time_us": event.device_time_total,
            }
        )
    if not matmul_counts:
        raise RuntimeError("Profiler did not identify a CUDA matrix-multiplication kernel")
    return {
        "matrix_multiplication_cuda_kernels": [
            {"kernel": name, "count": count} for name, count in sorted(matmul_counts.items())
        ],
        "all_cuda_kernel_counts": [
            {"kernel": name, "count": count} for name, count in sorted(cuda_counts.items())
        ],
        "cuda_events": cuda_events,
    }


def main() -> None:
    checkpoint = ARGS.checkpoint.resolve()
    inputs = ARGS.inputs.resolve()
    output = ARGS.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    projector_state = payload["model"]["projector"]
    projector = FusedMLPProjector(fused_vision_dim=2176, llm_dim=896)
    projector.load_state_dict(projector_state, strict=True)
    del projector_state, payload
    projector.eval().to(device="cuda", dtype=torch.bfloat16)

    default_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    tested_flags = [default_reduction]
    if not tested_flags or tested_flags[-1] is not False:
        tested_flags.append(False)
    if True not in tested_flags:
        tested_flags.append(True)

    source = Path(inspect.getsourcefile(nn_utils)).resolve()
    report = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "inputs": str(inputs),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "execution_dtype": "torch.bfloat16",
        "default_allow_bf16_reduced_precision_reduction": default_reduction,
        "sdpa_enabled_flags": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
        "projector_class": f"{type(projector).__module__}.{type(projector).__qualname__}",
        "projector_repr": repr(projector),
        "projector_source_file": str(source),
        "projector_source_sha256": sha256_file(source),
        "live_parameters": parameter_record(projector),
        "runs": [],
    }

    try:
        with torch.inference_mode():
            for allow_reduction in tested_flags:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow_reduction
                for query_index in (0, 1):
                    input_path = inputs / f"query{query_index}_projector_layer_outputs.pt"
                    saved = torch.load(input_path, map_location="cpu", weights_only=True)
                    vision = saved["vision_encoder_output"].to("cuda")
                    archived = saved["archived_projector_output"].contiguous()

                    for _ in range(3):
                        projector(vision)
                    torch.cuda.synchronize()
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,
                        profile_memory=False,
                        with_stack=False,
                    ) as prof:
                        with torch.profiler.record_function(
                            f"PROJECTOR_QUERY_{query_index}_BF16_REDUCED_{allow_reduction}"
                        ):
                            projected = projector(vision)
                        torch.cuda.synchronize()

                    projected_cpu = projected.detach().contiguous().cpu()
                    exact_match = torch.equal(projected_cpu, archived)
                    max_abs_diff = float((projected_cpu.float() - archived.float()).abs().max())
                    run = {
                        "policy_query_index": query_index,
                        "allow_bf16_reduced_precision_reduction": allow_reduction,
                        "input_file": str(input_path),
                        "input_file_sha256": sha256_file(input_path),
                        "vision_encoder_output": tensor_record(saved["vision_encoder_output"]),
                        "projector_output": tensor_record(projected_cpu),
                        "archived_projector_output": tensor_record(archived),
                        "exact_match_to_archive": exact_match,
                        "max_abs_difference_float32": max_abs_diff,
                        "profiler": profiler_record(prof),
                    }
                    report["runs"].append(run)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = default_reduction

    report_path = output / "projector_bf16_kernel_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (output / "projector_bf16_kernel_probe.py").write_text(Path(__file__).read_text())
    summary = {
        "status": report["status"],
        "default_allow_bf16_reduced_precision_reduction": default_reduction,
        "runs": [
            {
                "query": run["policy_query_index"],
                "allow_bf16_reduced_precision_reduction": run["allow_bf16_reduced_precision_reduction"],
                "exact_match_to_archive": run["exact_match_to_archive"],
                "max_abs_difference_float32": run["max_abs_difference_float32"],
                "matrix_multiplication_cuda_kernels": run["profiler"]["matrix_multiplication_cuda_kernels"],
            }
            for run in report["runs"]
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
