"""Pure, testable GRPO rewards, grouping, masking, and manifests."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.robot.libero.alignment_evaluator import parse_reasoning
from experiments.robot.libero.refined_move_alignment import literal_stop

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alignment(reasoning: str, actions: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Alignment of parsed reasoning to the decoded commanded H-step action chunk."""
    claims = parse_reasoning(reasoning)
    reason = np.asarray([claims.motion_axes.get(axis, 0) for axis in ("x", "y", "z")], dtype=np.float64)
    horizon = int(config["action_horizon"])
    commanded = float(config["action_step_m"]) * np.clip(np.asarray(actions, dtype=np.float64)[:horizon, :3], -1, 1).sum(axis=0)
    rho = float(np.linalg.norm(commanded))
    reason_norm = float(np.linalg.norm(reason))
    directional = reason_norm > 0
    stop = not directional and literal_stop(reasoning)
    applicable = directional or stop
    cosine = None
    cost = None
    mismatch = None
    if directional:
        cosine = float(np.dot(reason, commanded) / (reason_norm * rho + 1e-12))
        tau = float(config["tau"])
        cost = float((max(0.0, tau - cosine) / (1.0 + tau)) ** 2)
        mismatch = cosine < tau
    elif stop:
        delta = float(config["delta_stop_m"])
        cost = float(rho * rho / (rho * rho + delta * delta))
        mismatch = rho > delta
    return {
        "applicable": applicable,
        "directional": directional,
        "stop": stop,
        "cost": cost,
        "mismatch": mismatch,
        "cosine": cosine,
        "stop_displacement_m": rho if stop else None,
        "commanded_displacement_m": commanded.tolist(),
        "commanded_displacement_norm_m": rho,
    }


def episode_reward(success: bool, applicable_costs: list[float], reward_weight: float) -> float:
    if not applicable_costs:
        raise ValueError("episode has no applicable motion query")
    return float(bool(success)) - float(reward_weight) * float(np.mean(applicable_costs))


def group_advantages(rewards: list[float], epsilon: float, zero_threshold: float) -> dict[str, Any]:
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("a GRPO group must contain rewards")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    zero_variance = std <= float(zero_threshold)
    advantages = np.zeros_like(values) if zero_variance else (values - mean) / (std + float(epsilon))
    return {
        "reward_mean": mean,
        "reward_std": std,
        "zero_variance": zero_variance,
        "advantages": advantages.tolist(),
        "advantage_mean": float(advantages.mean()),
    }


def generated_token_mask(generated_ids: torch.Tensor, special_ids: set[int], valid_action: bool = True) -> torch.Tensor:
    """Mask only generated, non-special tokens from a query with a valid decoded action."""
    mask = torch.ones_like(generated_ids, dtype=torch.bool)
    for token in special_ids:
        mask &= generated_ids != int(token)
    if not valid_action:
        mask.zero_()
    return mask


def sampled_kl(delta: torch.Tensor) -> torch.Tensor:
    """Schulman sampled reverse-KL estimator, exp(delta)-delta-1."""
    return torch.expm1(delta) - delta


def clipped_grpo_terms(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    advantage: float | torch.Tensor,
    clip_epsilon: float,
    kl_beta: float,
) -> dict[str, torch.Tensor]:
    ratio = torch.exp(current_logps - old_logps)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    advantage_tensor = torch.as_tensor(advantage, dtype=current_logps.dtype, device=current_logps.device)
    surrogate = torch.minimum(ratio * advantage_tensor, clipped_ratio * advantage_tensor)
    delta = reference_logps - current_logps
    kl = sampled_kl(delta)
    objective = surrogate - kl_beta * kl
    return {"ratio": ratio, "clipped_ratio": clipped_ratio, "surrogate": surrogate, "kl": kl, "objective": objective}


def make_iteration_manifest(config: dict[str, Any], iteration: int = 1) -> dict[str, Any]:
    """Ten groups with independent with-replacement training-state draws."""
    if iteration < 1:
        raise ValueError("iteration numbers are one-based")
    tasks = list(config["task_ids"])
    if len(tasks) != 10 or len(set(tasks)) != 10:
        raise ValueError("the targeted smoke test requires ten unique tasks")
    task_order = tasks.copy()
    random.Random(int(config["seed"]) + iteration).shuffle(task_order)
    groups = []
    for group_index, task_id in enumerate(task_order):
        episodes = []
        for episode_index in range(int(config["group_size"])):
            seed_base = int(config["seed"]) * 100_000_000 + iteration * 1_000_000 + task_id * 1_000 + episode_index
            state_rng = random.Random(seed_base + 17)
            initial_state = state_rng.randrange(int(config["training_state_min"]), int(config["training_state_max_exclusive"]))
            episodes.append({
                "task_id": task_id,
                "group_index": group_index,
                "episode_index": episode_index,
                "initial_state": initial_state,
                "environment_seed": seed_base + 31,
                "sampling_seed": seed_base + 47,
            })
        groups.append({"task_id": task_id, "group_index": group_index, "episodes": episodes})
    return {
        "iteration": iteration,
        "seed": config["seed"],
        "task_order": task_order,
        "groups": groups,
        "sampling": {
            "do_sample": config["do_sample"],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
        },
    }


def validate_manifest(manifest: dict[str, Any], expected_tasks: list[int], group_size: int) -> None:
    groups = manifest["groups"]
    if len(groups) != 10 or sorted(group["task_id"] for group in groups) != sorted(expected_tasks):
        raise ValueError("manifest must contain every selected task exactly once")
    all_episodes = []
    for group in groups:
        episodes = group["episodes"]
        if len(episodes) != group_size or any(row["task_id"] != group["task_id"] for row in episodes):
            raise ValueError("task groups must contain exactly G same-task episodes")
        all_episodes.extend(episodes)
    if len(all_episodes) != 10 * group_size:
        raise ValueError("incorrect total episode count")
    if not manifest["sampling"]["do_sample"] or manifest["sampling"]["temperature"] != 1.0:
        raise ValueError("smoke rollout must sample at temperature 1.0")


def deterministic_resume_action(run_dir: str | Path, iteration: int) -> str:
    run_dir = Path(run_dir)
    if (run_dir / f"iteration-{iteration:04d}.complete.json").exists():
        return "complete"
    if (run_dir / f"iteration-{iteration:04d}.rollout-refreshed.json").exists():
        return "resume_after_refresh"
    return "refresh"


def finite_numbers(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return True
