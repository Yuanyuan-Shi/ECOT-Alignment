#!/usr/bin/env python3
"""Replay saved vision outputs through the original Lambda=2 BF16 projector."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

REPO = Path("/mnt1/ECOT-Alignment")
ROOT = REPO / "pi05_policy_training"
REFERENCE = ROOT / "data/minivla_reference"
INPUT_ROOT = ROOT / "reports/lambda2_task24_state14_seed80_attention_profile"
OUTPUT = ROOT / "reports/lambda2_task24_state14_seed80_projector_layers"
CHECKPOINT = (
    ROOT
    / "runs/minivla_lambda_sweep_20260914/large-data-lambda2-b32-s2000-20260914"
    / "checkpoints/step-002000-move-lora.pt"
)

os.environ.setdefault("PYTHONPATH", f"{REPO}:{ROOT / 'openpi/third_party/libero'}")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "configs/libero"))
os.environ.setdefault("PRISMATIC_DATA_ROOT", str(ROOT / "data/rlds"))
os.environ.setdefault("HF_HOME", str(REFERENCE / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "openpi/third_party/libero"))

import torch

from experiments.robot.libero import run_libero_eval
from experiments.robot.robot_utils import get_model, set_seed_everywhere
import prismatic.util.nn_utils as nn_utils


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_description(tensor: torch.Tensor) -> dict:
    cpu = tensor.detach().contiguous().cpu()
    fp32 = cpu.float()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256_raw_bytes": tensor_sha256(cpu),
        "minimum": float(fp32.min()),
        "maximum": float(fp32.max()),
        "mean": float(fp32.mean()),
    }


def parameter_record(projector: torch.nn.Module) -> dict:
    records = []
    combined = hashlib.sha256()
    for kind, iterator in (("parameter", projector.named_parameters()), ("buffer", projector.named_buffers())):
        for name, tensor in iterator:
            cpu = tensor.detach().contiguous().cpu()
            raw = cpu.view(torch.uint8).numpy().tobytes()
            digest = hashlib.sha256(raw).hexdigest()
            combined.update(kind.encode() + b"\0" + name.encode() + b"\0" + raw)
            records.append(
                {
                    "kind": kind,
                    "name": name,
                    "shape": list(cpu.shape),
                    "dtype": str(cpu.dtype),
                    "sha256_raw_bytes": digest,
                }
            )
    return {"combined_sha256": combined.hexdigest(), "tensors": records}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    set_seed_everywhere(7)
    cfg = run_libero_eval.GenerateConfig(
        model_family="prismatic",
        hf_token=Path("/home/james/.cache/huggingface/token"),
        pretrained_checkpoint=str(CHECKPOINT),
        seed=7,
        use_cot=True,
    )
    model = get_model(cfg)
    model.eval()
    model.enable_cot(True)
    projector = model.projector
    layers = projector.projector
    if len(layers) != 5:
        raise RuntimeError(f"Expected five projector layers, found {len(layers)}")

    source_path = Path(inspect.getsourcefile(type(projector))).resolve()
    report = {
        "status": "complete",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "execution_dtype": "torch.bfloat16",
        "projector_class": f"{type(projector).__module__}.{type(projector).__qualname__}",
        "projector_repr": repr(projector),
        "layer_types": [f"{type(layer).__module__}.{type(layer).__qualname__}" for layer in layers],
        "source_file": str(source_path),
        "source_file_sha256": sha256_file(source_path),
        "live_parameters": parameter_record(projector),
        "queries": [],
    }

    with torch.inference_mode():
        for query_index in (0, 1):
            input_path = INPUT_ROOT / f"query{query_index}_vision_projector_outputs.pt"
            archived = torch.load(input_path, map_location="cpu", weights_only=True)
            vision_cpu = archived["vision_encoder_output"].contiguous()
            archived_final = archived["projector_output"].contiguous()
            value = vision_cpu.to("cuda")
            layer_outputs = []
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for layer in layers:
                    value = layer(value)
                    layer_outputs.append(value.detach().contiguous().cpu())
            final = layer_outputs[-1]
            exact_match = torch.equal(final, archived_final)
            max_abs_difference = float((final.float() - archived_final.float()).abs().max())
            if not exact_match:
                raise RuntimeError(
                    f"Query {query_index} projector replay differs from archive; max abs diff={max_abs_difference}"
                )

            saved = {
                "vision_encoder_output": vision_cpu,
                "layer_outputs": {f"layer_{idx}_{type(layer).__name__}": output for idx, (layer, output) in enumerate(zip(layers, layer_outputs))},
                "archived_projector_output": archived_final,
            }
            output_path = OUTPUT / f"query{query_index}_projector_layer_outputs.pt"
            torch.save(saved, output_path)
            report["queries"].append(
                {
                    "policy_query_index": query_index,
                    "source_outputs_file": str(input_path),
                    "source_outputs_file_sha256": sha256_file(input_path),
                    "vision_encoder_output": tensor_description(vision_cpu),
                    "layers": [
                        {
                            "index": idx,
                            "type": f"{type(layer).__module__}.{type(layer).__qualname__}",
                            "output": tensor_description(output),
                        }
                        for idx, (layer, output) in enumerate(zip(layers, layer_outputs))
                    ],
                    "archived_projector_output": tensor_description(archived_final),
                    "final_exact_match": exact_match,
                    "final_max_abs_difference_float32": max_abs_difference,
                    "saved_layer_outputs_file": output_path.name,
                    "saved_layer_outputs_file_sha256": sha256_file(output_path),
                }
            )

    report_path = OUTPUT / "projector_layer_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "queries": report["queries"]}, indent=2))


if __name__ == "__main__":
    main()
