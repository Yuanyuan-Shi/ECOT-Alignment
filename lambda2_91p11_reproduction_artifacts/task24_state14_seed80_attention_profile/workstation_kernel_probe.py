#!/usr/bin/env python3
"""Profile selected SDPA kernels for λ=2 task 24/state 14 queries 0 and 1."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/mnt1/ECOT-Alignment")
ROOT = REPO / "pi05_policy_training"
os.environ.setdefault("PYTHONPATH", f"{REPO}:{ROOT / 'openpi/third_party/libero'}")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "configs/libero"))
os.environ.setdefault("PRISMATIC_DATA_ROOT", str(ROOT / "data/rlds"))
os.environ.setdefault("HF_HOME", str(ROOT / "data/minivla_reference/hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "openpi/third_party/libero"))

import numpy as np
import torch
import torch.nn.functional as F

from experiments.robot import openvla_utils, robot_utils
from experiments.robot.libero import libero_utils, run_libero_eval
from prismatic.models.vlas import openvla as openvla_module
from prismatic.models.vlas.openvla import OpenVLA
from prismatic.models.vlms.prismatic import PrismaticVLM


CHECKPOINT = (
    ROOT
    / "runs/minivla_lambda_sweep_20260914/large-data-lambda2-b32-s2000-20260914"
    / "checkpoints/step-002000-move-lora.pt"
)
OUTPUT = ROOT / "reports/lambda2_task24_state14_seed80_attention_profile"
TARGETS = {
    40: {"policy_query_index": 0, "tokens": [151804, 151894, 151751, 151802, 151817, 151751, 151780]},
    41: {"policy_query_index": 1, "tokens": [151723, 151869, 151763, 151796, 151785, 151673, 151895]},
}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_description(tensor: torch.Tensor) -> dict:
    cpu = tensor.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    as_float = cpu.float()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256_raw_bytes": hashlib.sha256(raw).hexdigest(),
        "minimum": float(as_float.min()),
        "maximum": float(as_float.max()),
        "mean": float(as_float.mean()),
    }


def detach_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().contiguous().cpu()
    if isinstance(value, dict):
        return {key: detach_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_tree(item) for item in value)
    if isinstance(value, list):
        return [detach_tree(item) for item in value]
    return value


def describe_tree(value):
    if isinstance(value, torch.Tensor):
        return tensor_description(value)
    if isinstance(value, dict):
        return {key: describe_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe_tree(item) for item in value]
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def source_fingerprints() -> list[dict]:
    modules = [
        run_libero_eval,
        libero_utils,
        openvla_utils,
        robot_utils,
        openvla_module,
    ]
    import robosuite
    import robosuite.macros
    import robosuite.utils.robot_utils

    modules.extend([robosuite, robosuite.macros, robosuite.utils.robot_utils])
    rows = []
    for module in modules:
        source = inspect.getsourcefile(module)
        if source and Path(source).is_file():
            path = Path(source).resolve()
            rows.append({"module": module.__name__, "path": str(path), "sha256": sha256_file(path)})
    return rows


def runtime_description(model: OpenVLA) -> dict:
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    llm = model.llm_backbone.llm
    return {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_runtime_version": torch._C._cuda_getRuntimeVersion() if hasattr(torch._C, "_cuda_getRuntimeVersion") else None,
        "gpu_name": props.name,
        "gpu_uuid": str(props.uuid) if hasattr(props, "uuid") else None,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory": props.total_memory,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "llm_class": f"{type(llm).__module__}.{type(llm).__qualname__}",
        "llm_attention_implementation": getattr(llm.config, "_attn_implementation", None),
        "generation_config": llm.generation_config.to_dict(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "environment": {
            key: os.environ.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
                "TOKENIZERS_PARALLELISM",
                "TF_CPP_MIN_LOG_LEVEL",
                "HF_HOME",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            ]
        },
        "source_files": source_fingerprints(),
    }


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    capture = {"generation_call": 0, "saved_calls": []}

    selected_names = (
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_cudnn_attention",
        "_scaled_dot_product_attention_math",
    )

    def phase_for_parent(event) -> str:
        parent = event.cpu_parent
        while parent is not None:
            if parent.name.startswith("SDPA_PHASE::"):
                return parent.name.split("::", 1)[1]
            parent = parent.cpu_parent
        return "unclassified"

    def summarize_profile(prof, call_index: int) -> dict:
        rows = []
        counts = Counter()
        for event in prof.events():
            if any(name in event.name for name in selected_names):
                phase = phase_for_parent(event)
                if (
                    phase == "language_prefill"
                    and event.input_shapes
                    and len(event.input_shapes[0]) >= 3
                    and event.input_shapes[0][2] == 1
                ):
                    phase = "language_cached_decode"
                counts[(phase, event.name)] += 1
                rows.append(
                    {
                        "phase": phase,
                        "operator": event.name,
                        "input_shapes": event.input_shapes,
                        "cpu_time_us": event.cpu_time_total,
                        "cuda_time_us": event.device_time_total,
                    }
                )
        summary = {
            "global_generation_call_zero_based": call_index,
            "policy_query_index": TARGETS[call_index]["policy_query_index"],
            "selected_operator_counts": [
                {"phase": phase, "operator": operator, "count": count}
                for (phase, operator), count in sorted(counts.items())
            ],
            "selected_operator_events": rows,
            "phase_classification": {
                "vision_encoders": "captured by the vision_backbone forward hook",
                "language_prefill": "Qwen SDPA calls with query length greater than 1",
                "language_cached_decode": "Qwen SDPA calls with query length equal to 1 and growing KV length",
            },
        }
        if not rows:
            raise RuntimeError(f"Profiler found no selected SDPA operators for call {call_index}")
        return summary

    def traced_raw_generate(self: OpenVLA, input_ids, pixel_values):
        call_index = capture["generation_call"]
        capture["generation_call"] += 1
        if isinstance(pixel_values, torch.Tensor):
            model_pixels = pixel_values.to(self.device).bfloat16()
        elif isinstance(pixel_values, dict):
            model_pixels = {key: value.to(self.device).bfloat16() for key, value in pixel_values.items()}
        else:
            raise TypeError(type(pixel_values))

        if call_index not in TARGETS:
            return super(PrismaticVLM, self).generate(
                input_ids=input_ids.to(self.device), pixel_values=model_pixels, max_new_tokens=1024
            )

        phase = {"name": "outside_model"}
        captured_outputs = {}

        def vision_pre(_module, _args, _kwargs):
            phase["name"] = "vision_encoders"

        def vision_post(_module, _args, _kwargs, output):
            captured_outputs["vision_encoder_output"] = detach_tree(output)
            phase["name"] = "outside_model"
            return output

        def projector_post(_module, _args, _kwargs, output):
            captured_outputs["projector_output"] = detach_tree(output)
            return output

        def llm_pre(_module, args, kwargs):
            past = kwargs.get("past_key_values")
            if past is None:
                phase["name"] = "language_prefill"
                return
            try:
                cached_length = int(past.get_seq_length())
            except Exception:
                cached_length = 0
            phase["name"] = "language_cached_decode" if cached_length > 0 else "language_prefill"

        def llm_post(_module, _args, _kwargs, output):
            phase["name"] = "outside_model"
            return output

        handles = [
            self.vision_backbone.register_forward_pre_hook(vision_pre, with_kwargs=True),
            self.vision_backbone.register_forward_hook(vision_post, with_kwargs=True),
            self.projector.register_forward_hook(projector_post, with_kwargs=True),
            self.llm_backbone.llm.register_forward_pre_hook(llm_pre, with_kwargs=True),
            self.llm_backbone.llm.register_forward_hook(llm_post, with_kwargs=True),
        ]
        original_sdpa = F.scaled_dot_product_attention

        def marked_sdpa(*args, **kwargs):
            with torch.profiler.record_function(f"SDPA_PHASE::{phase['name']}"):
                return original_sdpa(*args, **kwargs)

        F.scaled_dot_product_attention = marked_sdpa
        try:
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                sequences = super(PrismaticVLM, self).generate(
                    input_ids=input_ids.to(self.device), pixel_values=model_pixels, max_new_tokens=1024
                )
                torch.cuda.synchronize()
        finally:
            F.scaled_dot_product_attention = original_sdpa
            for handle in handles:
                handle.remove()

        new_ids = sequences[0, input_ids.shape[1] :].detach().cpu().tolist()
        num_end_tokens = 2
        action_start = len(new_ids) - num_end_tokens - 7
        action_stop = len(new_ids) - num_end_tokens
        action_ids = new_ids[action_start:action_stop]
        expected = TARGETS[call_index]["tokens"]
        if action_ids != expected:
            raise RuntimeError(f"Profiled action tokens changed: {action_ids} != {expected}")
        if set(captured_outputs) != {"vision_encoder_output", "projector_output"}:
            raise RuntimeError(f"Missing captured vision/projector outputs: {sorted(captured_outputs)}")
        summary = summarize_profile(prof, call_index)
        summary["action_token_ids"] = action_ids
        outputs_name = f"query{TARGETS[call_index]['policy_query_index']}_vision_projector_outputs.pt"
        outputs_path = OUTPUT / outputs_name
        torch.save(captured_outputs, outputs_path)
        summary["vision_projector_outputs"] = {
            "filename": outputs_name,
            "sha256": sha256_file(outputs_path),
            "tensors": describe_tree(captured_outputs),
        }
        summary["runtime_after_model_load"] = runtime_description(self)
        name = f"query{TARGETS[call_index]['policy_query_index']}_attention_profile.json"
        (OUTPUT / name).write_text(json.dumps(summary, indent=2) + "\n")
        capture["saved_calls"].append(call_index)
        return sequences

    OpenVLA.raw_generate = traced_raw_generate
    cfg = run_libero_eval.GenerateConfig(
        model_family="prismatic",
        hf_token=Path("/home/james/.cache/huggingface/token"),
        pretrained_checkpoint=str(CHECKPOINT),
        task_suite_name="libero_90",
        task_ids="24",
        num_trials_per_task=3,
        episodes_per_task=3,
        initial_state_offset=13,
        seed=7,
        center_crop=False,
        use_wrist_image=False,
        use_cot=True,
        num_open_loop_steps=10,
        n_procs_per_gpu=1,
        enable_alignment_evaluator=True,
        alignment_cosine_similarity=True,
        alignment_save_frames=True,
        use_wandb=False,
        alignment_output_dir=str(OUTPUT / "evaluation"),
        output_dir="lambda2-task24-state14-seed80-attention-profile",
        run_id_note="lambda2-task24-state14-seed80-attention-profile",
    )
    cfg.rank = 0
    cfg.world_size = 1
    run_libero_eval._eval_libero_main(cfg)
    run_libero_eval.merge_rank_outputs(cfg.alignment_output_dir)
    if capture["saved_calls"] != [40, 41]:
        raise RuntimeError(f"Expected profiler calls [40, 41], got {capture['saved_calls']}")
    episodes = json.loads((OUTPUT / "evaluation/rank-0/episode_summary.json").read_text())
    observed_query_counts = [episode["num_policy_queries"] for episode in episodes]
    if observed_query_counts != [40, 13, 11]:
        raise RuntimeError(f"Unexpected per-trial query counts: {observed_query_counts}")
    summary = {
        "status": "complete",
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "profiled_generation_calls": capture["saved_calls"],
        "profiled_policy_query_indices": [0, 1],
        "per_trial_query_counts": observed_query_counts,
    }
    (OUTPUT / "profile_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
