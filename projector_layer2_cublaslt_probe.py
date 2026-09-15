#!/usr/bin/env python3
"""Capture cuBLASLt configuration for projector layer 2 (Linear 8704 -> 896)."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
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
os.environ.setdefault("PRISMATIC_DATA_ROOT", str(ROOT / "data/rlds"))
sys.path.insert(0, str(REPO))

import torch


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


def mapped_library(name: str) -> dict | None:
    maps = Path("/proc/self/maps").read_text().splitlines()
    candidates = []
    for line in maps:
        path = line.split()[-1]
        if name in path and path.startswith("/") and Path(path).is_file():
            candidates.append(Path(path).resolve())
    if not candidates:
        return None
    path = sorted(set(candidates))[0]
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def parent_names(event) -> list[str]:
    names = []
    parent = event.cpu_parent
    while parent is not None:
        names.append(parent.name)
        parent = parent.cpu_parent
    return names


def profile_record(prof) -> dict:
    counts = Counter()
    events = []
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        parents = parent_names(event)
        is_matmul = any(
            parent in {"aten::linear", "aten::addmm", "aten::mm", "aten::matmul"} for parent in parents
        ) or any(token in event.name.lower() for token in ("gemm", "matmul", "mma"))
        if is_matmul:
            counts[event.name] += 1
            events.append(
                {
                    "kernel": event.name,
                    "cpu_parent_chain": parents,
                    "device_time_us": event.device_time_total,
                }
            )
    if not counts:
        raise RuntimeError("No CUDA GEMM kernel was captured")
    return {
        "kernel_counts": [{"kernel": name, "count": count} for name, count in sorted(counts.items())],
        "events": events,
    }


def install_cublaslt_logger() -> tuple[ctypes.CDLL, object, list[dict], dict]:
    library_path = (
        ROOT
        / "data/minivla_venv/lib/python3.10/site-packages/nvidia/cublas/lib/libcublasLt.so.12"
    ).resolve()
    library = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    messages: list[dict] = []
    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)

    @callback_type
    def callback(level, function_name, message):
        messages.append(
            {
                "level": int(level),
                "function": function_name.decode("utf-8", errors="replace") if function_name else "",
                "message": message.decode("utf-8", errors="replace") if message else "",
            }
        )

    library.cublasLtLoggerSetCallback.argtypes = [callback_type]
    library.cublasLtLoggerSetCallback.restype = ctypes.c_int
    library.cublasLtLoggerSetLevel.argtypes = [ctypes.c_int]
    library.cublasLtLoggerSetLevel.restype = ctypes.c_int
    library.cublasLtLoggerSetMask.argtypes = [ctypes.c_int]
    library.cublasLtLoggerSetMask.restype = ctypes.c_int
    statuses = {
        "library_path": str(library_path),
        "library_sha256": sha256_file(library_path),
        "set_callback_status": int(library.cublasLtLoggerSetCallback(callback)),
        "set_level_5_status": int(library.cublasLtLoggerSetLevel(5)),
        "set_mask_31_status": int(library.cublasLtLoggerSetMask(31)),
    }
    return library, callback, messages, statuses


def main() -> None:
    checkpoint = ARGS.checkpoint.resolve()
    inputs = ARGS.inputs.resolve()
    output = ARGS.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = payload["model"]["projector"]
    layer = torch.nn.Linear(8704, 896, bias=True)
    layer.load_state_dict({"weight": state["projector.2.weight"], "bias": state["projector.2.bias"]})
    del state, payload
    layer.eval().to(device="cuda", dtype=torch.bfloat16)

    default_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    library, callback, logger_messages, logger_status = install_cublaslt_logger()
    del library

    report = {
        "status": "complete",
        "target": "projector.projector[2] Linear(8704 -> 896, bias=True)",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "execution_dtype": "torch.bfloat16",
        "original_default_allow_bf16_reduced_precision_reduction": default_reduction,
        "effective_allow_bf16_reduced_precision_reduction": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "logger_registration": logger_status,
        "layer": {
            "repr": repr(layer),
            "weight": tensor_record(layer.weight),
            "bias": tensor_record(layer.bias),
        },
        "queries": [],
    }

    try:
        with torch.inference_mode():
            for query_index in (0, 1):
                saved_path = inputs / f"query{query_index}_projector_layer_outputs.pt"
                saved = torch.load(saved_path, map_location="cpu", weights_only=True)
                layer1 = saved["layer_outputs"]["layer_1_GELU"].contiguous()
                archived = saved["layer_outputs"]["layer_2_Linear"].contiguous()
                layer1_cuda = layer1.to("cuda")

                message_start = len(logger_messages)
                for _ in range(3):
                    layer(layer1_cuda)
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                ) as prof:
                    with torch.profiler.record_function(f"PROJECTOR_LAYER2_QUERY_{query_index}"):
                        actual = layer(layer1_cuda)
                    torch.cuda.synchronize()
                actual_cpu = actual.detach().contiguous().cpu()
                exact = torch.equal(actual_cpu, archived)
                max_abs = float((actual_cpu.float() - archived.float()).abs().max())
                if not exact:
                    raise RuntimeError(f"Query {query_index} layer-2 output mismatch: max abs={max_abs}")
                report["queries"].append(
                    {
                        "policy_query_index": query_index,
                        "input_file": str(saved_path),
                        "input_file_sha256": sha256_file(saved_path),
                        "layer1_gelu_input": tensor_record(layer1),
                        "layer2_output": tensor_record(actual_cpu),
                        "archived_layer2_output": tensor_record(archived),
                        "exact_match_to_archive": exact,
                        "max_abs_difference_float32": max_abs,
                        "profiler": profile_record(prof),
                        "cublaslt_logger_message_range": [message_start, len(logger_messages)],
                    }
                )
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = default_reduction

    report["mapped_libraries_after_execution"] = {
        "libcublasLt.so": mapped_library("libcublasLt.so"),
        "libcublas.so": mapped_library("libcublas.so"),
    }
    report["cublaslt_logger_message_count"] = len(logger_messages)
    report_path = output / "projector_layer2_cublaslt_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    logger_path = output / "cublaslt_logger_messages.jsonl"
    logger_path.write_text("".join(json.dumps(row) + "\n" for row in logger_messages))
    (output / "projector_layer2_cublaslt_probe.py").write_text(Path(__file__).read_text())
    print(
        json.dumps(
            {
                "status": report["status"],
                "logger_registration": logger_status,
                "logger_message_count": len(logger_messages),
                "queries": report["queries"],
                "mapped_libraries": report["mapped_libraries_after_execution"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
