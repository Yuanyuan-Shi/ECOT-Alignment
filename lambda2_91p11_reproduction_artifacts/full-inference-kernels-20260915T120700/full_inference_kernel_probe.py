#!/usr/bin/env python3
"""Capture full MiniVLA inference GEMM algorithms and CUDA kernels for two saved queries."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query0-input", type=Path, required=True)
    parser.add_argument("--query1-input", type=Path, required=True)
    parser.add_argument("--query0-reference", type=Path, required=True)
    parser.add_argument("--query1-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-profiler", action="store_true", help="Capture cuBLAS descriptors without CUPTI")
    return parser.parse_args()


ARGS = arguments()
REPO = ARGS.repo.resolve()
ROOT = REPO / "pi05_policy_training"
REFERENCE_ROOT = ROOT / "data/minivla_reference"
os.environ.setdefault("PRISMATIC_DATA_ROOT", str(ROOT / "data/rlds"))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "configs/libero"))
os.environ.setdefault("HF_HOME", str(REFERENCE_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "openpi/third_party/libero"))

import torch

from prismatic.models.load import load_vla
from prismatic.models.vlms.prismatic import PrismaticVLM


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mapped_library(fragment: str) -> dict | None:
    candidates = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if fragment in path and path.startswith("/") and Path(path).is_file():
            candidates.add(Path(path).resolve())
    if not candidates:
        return None
    path = sorted(candidates)[0]
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def tensor_record(value: torch.Tensor) -> dict:
    cpu = value.detach().contiguous().cpu()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256_raw_bytes": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def first_difference(actual: list[int], expected: list[int]) -> dict | None:
    for index, (left, right) in enumerate(zip(actual, expected)):
        if left != right:
            return {"index": index, "actual": left, "expected": right}
    if len(actual) != len(expected):
        index = min(len(actual), len(expected))
        return {
            "index": index,
            "actual": actual[index] if index < len(actual) else None,
            "expected": expected[index] if index < len(expected) else None,
        }
    return None


def source_hashes(model) -> list[dict]:
    modules = [
        inspect.getmodule(type(model)),
        inspect.getmodule(type(model.vision_backbone)),
        inspect.getmodule(type(model.projector)),
        inspect.getmodule(type(model.llm_backbone.llm)),
    ]
    result = []
    for module in modules:
        source = inspect.getsourcefile(module) if module else None
        if source and Path(source).is_file():
            path = Path(source).resolve()
            row = {"module": module.__name__, "path": str(path), "sha256": sha256_file(path)}
            if row not in result:
                result.append(row)
    return result


def runtime_record(model) -> dict:
    props = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,uuid", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
    except OSError:
        smi = ""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu": props.name,
        "gpu_uuid": str(getattr(props, "uuid", "")),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_smi": smi,
        "execution_dtype": "torch.bfloat16",
        "decoding": "greedy (Transformers generate defaults; do_sample=False)",
        "llm_attention_implementation": getattr(model.llm_backbone.llm.config, "_attn_implementation", None),
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "generation_config": model.llm_backbone.llm.generation_config.to_dict(),
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
            "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        )},
        "source_files": source_hashes(model),
    }


class ContextHooks:
    """Label synchronous cuBLAS calls and profiler events by query, layer, and LLM step."""

    def __init__(self, model, library: ctypes.CDLL | None, query_index: int):
        self.model = model
        self.library = library
        self.query_index = query_index
        self.llm_call = -1
        self.phase = "vision"
        self.context_stack: list[str] = []
        self.profiler_stack = []
        self.handles = []

    def label(self, name: str) -> str:
        if self.phase == "language":
            stage = "prefill" if self.llm_call == 0 else "cached_decode"
            return f"query={self.query_index}|phase=language_{stage}|step={self.llm_call}|module={name}"
        return f"query={self.query_index}|phase={self.phase}|step=-1|module={name}"

    def set_context(self, value: str) -> None:
        if self.library is not None:
            self.library.ecot_set_context(value.encode())

    def root_llm_pre(self, _module, _args, _kwargs):
        self.llm_call += 1
        self.phase = "language"

    def module_pre(self, name):
        def hook(_module, _args, _kwargs):
            label = self.label(name)
            self.context_stack.append(label)
            self.set_context(label)
            marker = torch.profiler.record_function("MODEL_CONTEXT::" + label)
            marker.__enter__()
            self.profiler_stack.append(marker)
        return hook

    def module_post(self, _name):
        def hook(_module, _args, _kwargs, output):
            marker = self.profiler_stack.pop()
            marker.__exit__(None, None, None)
            self.context_stack.pop()
            self.set_context(self.context_stack[-1] if self.context_stack else self.label("outside_module"))
            return output
        return hook

    def install(self):
        llm = self.model.llm_backbone.llm
        self.handles.append(llm.register_forward_pre_hook(self.root_llm_pre, with_kwargs=True, prepend=True))
        selected = []
        for name, module in self.model.named_modules():
            cls = type(module).__name__.lower()
            if isinstance(module, torch.nn.Linear) or "attention" in cls or name in {"vision_backbone", "projector"}:
                selected.append((name or "model", module))
        # Parent hooks first; nested leaf hooks temporarily override their parent's context.
        for name, module in selected:
            self.handles.append(module.register_forward_pre_hook(self.module_pre(name), with_kwargs=True))
            self.handles.append(module.register_forward_hook(self.module_post(name), with_kwargs=True))
        self.set_context(self.label("generation_start"))

    def remove(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.set_context("capture_complete")


def profile_summary(prof) -> dict:
    counts = Counter()
    events = []
    tokens = ("gemm", "matmul", "cutlass", "cublas", "mma", "attention")
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if not any(token in event.name.lower() for token in tokens):
            continue
        parents = []
        parent = event.cpu_parent
        while parent is not None:
            parents.append(parent.name)
            parent = parent.cpu_parent
        context = next((name.split("MODEL_CONTEXT::", 1)[1] for name in parents if name.startswith("MODEL_CONTEXT::")), None)
        counts[event.name] += 1
        events.append({
            "kernel": event.name,
            "model_context": context,
            "device_time_us": event.device_time_total,
            "cpu_parent_chain": parents,
        })
    return {
        "cuda_kernel_counts": [{"kernel": key, "count": value} for key, value in sorted(counts.items())],
        "cuda_kernel_events": events,
    }


def gzip_trace(path: Path) -> Path:
    target = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(target, "wb", compresslevel=6) as sink:
        shutil.copyfileobj(source, sink, length=8 << 20)
    path.unlink()
    return target


def run_query(model, library, query_index: int, input_path: Path, reference_path: Path, output: Path) -> dict:
    saved = torch.load(input_path, map_location="cpu", weights_only=True)
    reference = json.loads(reference_path.read_text())
    input_ids = saved["input_ids"].to(model.device)
    pixels = saved["pixel_values"]
    if isinstance(pixels, dict):
        pixels = {key: value.to(model.device).bfloat16() for key, value in pixels.items()}
    else:
        pixels = pixels.to(model.device).bfloat16()

    hooks = ContextHooks(model, library, query_index)
    hooks.install()
    try:
        torch.cuda.synchronize()
        if ARGS.no_profiler:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sequence = super(PrismaticVLM, model).generate(
                    input_ids=input_ids, pixel_values=pixels, max_new_tokens=1024
                )
                torch.cuda.synchronize()
            prof = None
        else:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                sequence = super(PrismaticVLM, model).generate(
                    input_ids=input_ids,
                    pixel_values=pixels,
                    max_new_tokens=1024,
                )
                torch.cuda.synchronize()
    finally:
        hooks.remove()

    generated = sequence[0, input_ids.shape[1]:].detach().cpu().tolist()
    expected = reference["generated_token_ids"]
    compressed = None
    compact_path = None
    compact = None
    if prof is not None:
        trace = output / f"query{query_index}_full_profiler_trace.json"
        prof.export_chrome_trace(str(trace))
        compressed = gzip_trace(trace)
        compact = profile_summary(prof)
        compact_path = output / f"query{query_index}_kernel_events.json"
        compact_path.write_text(json.dumps(compact, indent=2) + "\n")
    report = {
        "policy_query_index": query_index,
        "input_file": str(input_path.resolve()),
        "input_file_sha256": sha256_file(input_path),
        "input_tensors": {"input_ids": tensor_record(saved["input_ids"]), "pixel_values": {k: tensor_record(v) for k, v in saved["pixel_values"].items()}},
        "reference_file": str(reference_path.resolve()),
        "reference_file_sha256": sha256_file(reference_path),
        "prompt_length": input_ids.shape[1],
        "complete_generated_token_ids": generated,
        "reference_generated_token_ids": expected,
        "exact_generated_token_match": generated == expected,
        "first_generated_token_difference": first_difference(generated, expected),
        "profiler_trace": None if compressed is None else {"filename": compressed.name, "size": compressed.stat().st_size, "sha256": sha256_file(compressed)},
        "kernel_events": None if compact_path is None else {"filename": compact_path.name, "sha256": sha256_file(compact_path), "count": len(compact["cuda_kernel_events"])},
    }
    (output / f"query{query_index}_inference_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; full inference kernel capture requires the workstation GPU")
    output = ARGS.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["ECOT_CUBLAS_CAPTURE_FILE"] = str(output / "cublas_calls.jsonl")
    os.environ["ECOT_CUBLASLT_CAPTURE_FILE"] = os.environ["ECOT_CUBLAS_CAPTURE_FILE"]
    Path(os.environ["ECOT_CUBLAS_CAPTURE_FILE"]).unlink(missing_ok=True)

    process_library = ctypes.CDLL(None)
    try:
        process_library.ecot_set_context.argtypes = [ctypes.c_char_p]
        process_library.ecot_set_context.restype = None
        interposer = process_library
    except AttributeError:
        interposer = None

    old_cwd = Path.cwd()
    os.chdir(REFERENCE_ROOT)
    try:
        token = Path("/home/james/.cache/huggingface/token").read_text().strip()
        model = load_vla(str(ARGS.checkpoint.resolve()), hf_token=token, load_for_training=False)
        for parameter in model.parameters():
            if parameter.dtype != torch.float32:
                raise RuntimeError(f"Loaded parameter was not FP32 before evaluator cast: {parameter.dtype}")
        model.vision_backbone.to(dtype=model.vision_backbone.half_precision_dtype)
        model.llm_backbone.to(dtype=model.llm_backbone.half_precision_dtype)
        model.to(dtype=model.llm_backbone.half_precision_dtype, device="cuda:0")
        model.eval()
        model.enable_cot(True)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        runtime = runtime_record(model)
        reports = [
            run_query(model, interposer, 0, ARGS.query0_input.resolve(), ARGS.query0_reference.resolve(), output),
            run_query(model, interposer, 1, ARGS.query1_input.resolve(), ARGS.query1_reference.resolve(), output),
        ]
    finally:
        os.chdir(old_cwd)

    runtime["mapped_libraries_after_inference"] = {
        "libcublasLt.so": mapped_library("libcublasLt.so"),
        "libcublas.so": mapped_library("libcublas.so"),
    }
    cublas_path = output / "cublas_calls.jsonl"
    summary = {
        "status": "complete",
        "checkpoint": str(ARGS.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(ARGS.checkpoint.resolve()),
        "runtime": runtime,
        "queries": reports,
        "cublas_capture": {
            "filename": cublas_path.name,
            "size": cublas_path.stat().st_size,
            "sha256": sha256_file(cublas_path),
            "calls": sum(1 for _ in cublas_path.open()),
        },
    }
    (output / "full_inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copy2(__file__, output / Path(__file__).name)
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "query_matches": [row["exact_generated_token_match"] for row in reports],
        "cublas_calls": summary["cublas_capture"]["calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
