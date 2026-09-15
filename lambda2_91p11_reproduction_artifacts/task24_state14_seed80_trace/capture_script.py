#!/usr/bin/env python3
"""Capture an inference-level trace for λ=2 task 24, state 14, seed 80."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import sys
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
sys.path.insert(0, str(ROOT / "openpi/third_party/libero"))

import numpy as np
import torch

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
OUTPUT = ROOT / "reports/lambda2_task24_state14_seed80_trace"
TARGET_GENERATION_CALL = 40  # task 24 state 13 made 40 queries before state 14 began.
EXPECTED_ACTION_TOKENS = [151804, 151894, 151751, 151802, 151817, 151751, 151780]


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
    capture = {"generation_call": 0, "saved": False}

    def traced_raw_generate(self: OpenVLA, input_ids, pixel_values):
        call_index = capture["generation_call"]
        capture["generation_call"] += 1
        if isinstance(pixel_values, torch.Tensor):
            model_pixels = pixel_values.to(self.device).bfloat16()
        elif isinstance(pixel_values, dict):
            model_pixels = {key: value.to(self.device).bfloat16() for key, value in pixel_values.items()}
        else:
            raise TypeError(type(pixel_values))

        should_capture = call_index == TARGET_GENERATION_CALL
        kwargs = {"max_new_tokens": 1024}
        if should_capture:
            kwargs.update(return_dict_in_generate=True, output_scores=True)

        generated = super(PrismaticVLM, self).generate(
            input_ids=input_ids.to(self.device),
            pixel_values=model_pixels,
            **kwargs,
        )
        if not should_capture:
            return generated

        sequences = generated.sequences
        new_ids = sequences[0, input_ids.shape[1] :].detach().cpu().tolist()
        num_end_tokens = 2
        action_start = len(new_ids) - num_end_tokens - 7
        action_stop = len(new_ids) - num_end_tokens
        action_ids = new_ids[action_start:action_stop]
        if action_ids != EXPECTED_ACTION_TOKENS:
            raise RuntimeError(f"Target trace action tokens changed: {action_ids} != {EXPECTED_ACTION_TOKENS}")

        top2 = []
        for position in range(action_start, action_stop):
            values, indices = torch.topk(generated.scores[position][0].float(), k=2)
            top2.append(
                {
                    "generated_position": position,
                    "chosen_token_id": new_ids[position],
                    "top_token_ids": indices.detach().cpu().tolist(),
                    "top_logits_float32": values.detach().cpu().tolist(),
                    "top_logit_margin": float(values[0] - values[1]),
                }
            )

        saved_pixels = (
            model_pixels.detach().cpu()
            if isinstance(model_pixels, torch.Tensor)
            else {key: value.detach().cpu() for key, value in model_pixels.items()}
        )
        torch.save(
            {"input_ids": input_ids.detach().cpu(), "pixel_values": saved_pixels},
            OUTPUT / "exact_model_inputs.pt",
        )
        pixel_descriptions = (
            tensor_description(model_pixels)
            if isinstance(model_pixels, torch.Tensor)
            else {key: tensor_description(value) for key, value in model_pixels.items()}
        )
        trace = {
            "case": {"task_id": 24, "trial": 1, "initial_state_index": 14, "episode_seed": 80},
            "global_generation_call_zero_based": call_index,
            "prompt_input_ids": input_ids.detach().cpu().tolist()[0],
            "prompt_text_from_ids": self.llm_backbone.tokenizer.decode(input_ids[0]),
            "complete_sequence_token_ids": sequences[0].detach().cpu().tolist(),
            "generated_token_ids": new_ids,
            "action_token_ids": action_ids,
            "action_token_top2_logits": top2,
            "input_tensors": {"input_ids": tensor_description(input_ids), "pixel_values": pixel_descriptions},
            "runtime_after_model_load": runtime_description(self),
        }
        (OUTPUT / "inference_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        capture["saved"] = True
        return sequences

    OpenVLA.raw_generate = traced_raw_generate
    cfg = run_libero_eval.GenerateConfig(
        model_family="prismatic",
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
        output_dir="lambda2-task24-state14-seed80-trace",
        run_id_note="lambda2-task24-state14-seed80-trace",
    )
    cfg.rank = 0
    cfg.world_size = 1
    run_libero_eval._eval_libero_main(cfg)
    run_libero_eval.merge_rank_outputs(cfg.alignment_output_dir)
    if not capture["saved"]:
        raise RuntimeError("Target generation call was not captured")

    frame = (
        OUTPUT
        / "evaluation/rank-0/assets/libero_90-task24-episode1-seed80-query000-before.png"
    )
    if not frame.is_file():
        raise FileNotFoundError(frame)
    shutil.copy2(frame, OUTPUT / "task24_state14_seed80_first_query.png")

    rows = [json.loads(line) for line in (OUTPUT / "evaluation/rank-0/alignment_queries.jsonl").open()]
    target = next(
        row
        for row in rows
        if row["episode_id"] == "libero_90-task24-episode1-seed80" and row["policy_query_index"] == 0
    )
    (OUTPUT / "first_query_record.json").write_text(json.dumps(target, indent=2) + "\n")
    summary = {
        "status": "complete",
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "camera_png_sha256": sha256_file(OUTPUT / "task24_state14_seed80_first_query.png"),
        "model_inputs_pt_sha256": sha256_file(OUTPUT / "exact_model_inputs.pt"),
        "inference_trace_sha256": sha256_file(OUTPUT / "inference_trace.json"),
        "first_query_record_sha256": sha256_file(OUTPUT / "first_query_record.json"),
        "action_token_ids": target["action_tokens"],
        "decoded_action_chunk_sha256": hashlib.sha256(canonical(target["decoded_action_chunk"]).encode()).hexdigest(),
        "episode_success": target["episode_success"],
        "capture_generation_call": TARGET_GENERATION_CALL,
    }
    (OUTPUT / "trace_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
