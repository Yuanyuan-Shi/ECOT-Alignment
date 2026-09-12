"""MiniVLA policy, immutable rollout snapshot, and frozen original reference roles."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from prismatic.models import load_vla
from prismatic.models.vlms.prismatic import PrismaticVLM
from torch.nn.attention import SDPBackend, sdpa_kernel

from onpolicy_grpo.common import generated_token_mask


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_base(config: dict[str, Any]):
    seed_everything(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vla = load_vla(Path(config["base_checkpoint"]), load_for_training=True, image_sequence_len=1).to("cuda")
    vla.requires_grad_(False)
    vla.vision_backbone_requires_grad = False
    tokenizer = vla.action_tokenizer
    for module in (tokenizer.vq_vae.encoder, tokenizer.vq_vae.decoder, tokenizer.vq_vae.vq_layer):
        module.to("cuda").requires_grad_(False).eval()
    device = f"cuda:{torch.cuda.current_device()}"
    tokenizer.vq_vae.device = device
    tokenizer.vq_vae.vq_layer.device = torch.device(device)
    tokenizer.device = device
    return vla


def build_policy(config: dict[str, Any], checkpoint: str | Path | None = None, trainable: bool = True):
    vla = _load_base(config)
    rank = int(config["lora_rank"])
    vla.llm_backbone.llm = get_peft_model(vla.llm_backbone.llm, LoraConfig(
        r=rank, lora_alpha=rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM", init_lora_weights="gaussian",
    ))
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(vla.llm_backbone.llm, state["adapter"])
    vla.requires_grad_(False)
    if trainable:
        for name, parameter in vla.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
    vla.eval()
    return vla


def build_reference_policy(config: dict[str, Any]):
    reference = _load_base(config)
    reference.requires_grad_(False).eval()
    assert not any(parameter.requires_grad for parameter in reference.parameters())
    return reference


def save_policy(path: str | Path, policy, config: dict[str, Any], iteration: int, optimizer=None) -> None:
    state = {
        "adapter": {name: value.detach().cpu() for name, value in get_peft_model_state_dict(policy.llm_backbone.llm).items()},
        "iteration": iteration, "config": config, "has_value_head": False, "algorithm": "GRPO",
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def trainable_parameters(policy) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    assert parameters
    return parameters


def generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Map logical no-top-k (`null`) to HF's explicit disabling value (`0`)."""
    return {
        "max_new_tokens": int(config["max_new_tokens"]),
        "do_sample": bool(config["do_sample"]),
        "temperature": float(config["temperature"]),
        "top_p": float(config["top_p"]),
        "top_k": 0 if config["top_k"] is None else int(config["top_k"]),
        "repetition_penalty": 1.0,
        "return_dict_in_generate": True,
        "output_scores": True,
        "use_cache": True,
    }


@torch.no_grad()
def sample_query(policy, image: np.ndarray, instruction: str, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    tokenizer = policy.llm_backbone.tokenizer
    builder = policy.get_prompt_builder()
    builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
    prompt = torch.cat([tokenizer(builder.get_prompt(), return_tensors="pt").input_ids,
                        tokenizer(policy.base_prompt, return_tensors="pt").input_ids], dim=1).cuda()
    pixels = policy.vision_backbone.get_image_transform()(Image.fromarray(image))
    pixels = ({key: value[None].cuda().float() for key, value in pixels.items()}
              if isinstance(pixels, dict) else pixels[None].cuda().float())
    features = []
    handle = policy.projector.register_forward_hook(lambda _module, _inputs, output: features.append(output.detach().cpu()))
    generation = {"input_ids": prompt, "pixel_values": pixels, **generation_kwargs(config)}
    try:
        with sdpa_kernel(SDPBackend.MATH):
            generated = super(PrismaticVLM, policy).generate(**generation)
    finally:
        handle.remove()
    generated_ids = generated.sequences[0, prompt.shape[1]:]
    sampled_logps = torch.stack([torch.log_softmax(scores[0].float(), dim=-1)[token].cpu()
                                 for token, scores in zip(generated_ids, generated.scores)])
    text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=False)
    reasoning = policy.base_prompt + text.split("ACTION:")[0]
    begin, end = policy.action_tokenizer.action_token_begin_idx, policy.action_tokenizer.action_token_end_idx
    runs, active_ids, active_positions = [], [], []
    for position, token in enumerate(generated_ids.tolist()):
        if begin < token < end:
            active_ids.append(token); active_positions.append(position)
        elif active_ids:
            runs.append((active_ids, active_positions)); active_ids, active_positions = [], []
    if active_ids:
        runs.append((active_ids, active_positions))
    action_ids, action_positions = runs[-1] if runs else ([], [])
    valid_action = len(action_ids) == 7
    if valid_action:
        with sdpa_kernel(SDPBackend.MATH):
            actions = policy.action_tokenizer.decode_token_ids_to_actions(np.asarray(action_ids)).detach().float().cpu().numpy()[0]
        statistics = policy.get_action_stats("libero_lm_90")
        mask = np.asarray(statistics.get("mask", [True] * 7))
        actions = np.where(mask, .5 * (actions + 1) * (np.asarray(statistics["q99"]) - np.asarray(statistics["q01"])) + np.asarray(statistics["q01"]), actions)
    else:
        actions = np.zeros((int(config["action_horizon"]), 7), dtype=np.float64)
    token_mask = generated_token_mask(generated_ids.cpu(), set(tokenizer.all_special_ids), valid_action)
    record = {"prompt_ids": prompt[0].cpu(), "generated_ids": generated_ids.cpu(),
              "generated_token_mask": token_mask.cpu(), "projected_patches": features[0][0],
              "sampled_logps": sampled_logps, "reasoning": reasoning, "action_tokens": action_ids,
              "action_token_positions": action_positions, "valid_action_tokens": valid_action,
              "sampling": {"do_sample": True, "temperature": 1.0, "top_p": 1.0,
                           "top_k": None, "backend_top_k": 0}}
    return actions, record


def forward_logits(model, record: dict[str, Any]) -> torch.Tensor:
    llm = model.llm_backbone.llm
    ids = torch.cat([record["prompt_ids"], record["generated_ids"]]).cuda()[None]
    embeddings = llm.get_input_embeddings()(ids)
    patches = record["projected_patches"].cuda()[None]
    fused = torch.cat([embeddings[:, :1], patches, embeddings[:, 1:]], dim=1)
    with sdpa_kernel(SDPBackend.MATH):
        output = llm(inputs_embeds=fused, attention_mask=torch.ones(fused.shape[:2], device="cuda", dtype=torch.long),
                     use_cache=False, return_dict=True)
    start = len(record["prompt_ids"]) + patches.shape[1] - 1
    return output.logits[0, start:start + len(record["generated_ids"])]


def sampled_token_logps(logits: torch.Tensor, generated_ids: torch.Tensor) -> torch.Tensor:
    output = []
    for start in range(0, len(generated_ids), 32):
        logps = torch.log_softmax(logits[start:start + 32].float(), dim=-1)
        targets = generated_ids[start:start + 32].cuda()
        output.append(logps.gather(1, targets[:, None]).squeeze(1))
    return torch.cat(output)


def role_audit(model, role: str) -> dict[str, Any]:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    correct = (bool(trainable) and all("lora_" in name for name in trainable)) if role == "policy" else not trainable
    return {"role": role, "training": bool(model.training), "trainable_parameter_tensors": len(trainable),
            "role_parameters_correct": correct, "has_value_head": False}
