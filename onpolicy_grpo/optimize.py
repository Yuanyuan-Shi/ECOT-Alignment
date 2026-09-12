"""One fresh-batch token-level GRPO epoch, with no critic, GAE, entropy, or SFT loss."""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from onpolicy_grpo.common import clipped_grpo_terms, group_advantages, read, write
from onpolicy_grpo.policy import (build_policy, build_reference_policy, forward_logits, role_audit,
                                  sampled_token_logps, save_policy, seed_everything, trainable_parameters)


def load_episodes(run_dir: Path) -> list[dict[str, Any]]:
    return [read(path) for path in sorted((run_dir / "rollouts/iteration-0001").glob("episode-*.json"))]


def assign_advantages(run_dir: Path, config: dict[str, Any], episodes: list[dict[str, Any]]):
    by_task = {task: [] for task in config["task_ids"]}
    for episode in episodes:
        by_task[episode["task_id"]].append(episode)
    groups, advantage_by_episode = [], {}
    for task in config["task_ids"]:
        rows = sorted(by_task[task], key=lambda row: row["episode_index"])
        if len(rows) != config["group_size"] or any(row["task_id"] != task for row in rows):
            raise RuntimeError(f"invalid GRPO group for task {task}")
        stats = group_advantages([row["combined_reward"] for row in rows], config["advantage_epsilon"], config["zero_variance_threshold"])
        if not stats["zero_variance"] and abs(stats["advantage_mean"]) > 1e-6:
            raise RuntimeError(f"nonzero group advantage mean for task {task}")
        for row, advantage in zip(rows, stats["advantages"]):
            advantage_by_episode[(row["group_index"], row["episode_index"])] = advantage
        groups.append({"task_id": task, "episode_count": len(rows), "rewards": [row["combined_reward"] for row in rows], **stats})
    write(run_dir / "iteration-0001.groups.json", groups)
    return groups, advantage_by_episode


def cache_reference_logps(run_dir: Path, config: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    reference = build_reference_policy(config)
    before = role_audit(reference, "reference_policy")
    assert before["trainable_parameter_tensors"] == 0
    token_count = 0
    with torch.no_grad():
        for episode in episodes:
            for query in episode["queries"]:
                sample_path = Path(query["sample_path"])
                record = torch.load(sample_path, map_location="cpu", weights_only=False)
                logits = forward_logits(reference, record)
                record["reference_logps"] = sampled_token_logps(logits, record["generated_ids"]).detach().cpu()
                torch.save(record, sample_path)
                token_count += int(record["generated_token_mask"].sum())
    after = role_audit(reference, "reference_policy")
    assert before == after and not any(parameter.grad is not None for parameter in reference.parameters())
    audit = {"before": before, "after": after, "unchanged": True, "cached_valid_tokens": token_count}
    write(run_dir / "reference-role-audit.json", audit)
    del reference
    gc.collect(); torch.cuda.empty_cache()
    return audit


def optimize(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    config = read(run_dir / "config.json")
    torch.cuda.reset_peak_memory_stats()
    episodes = load_episodes(run_dir)
    expected_episodes = len(config["task_ids"]) * config["group_size"]
    if len(episodes) != expected_episodes:
        raise RuntimeError(f"expected exactly {expected_episodes} fresh episodes, found {len(episodes)}")
    groups, advantages = assign_advantages(run_dir, config, episodes)
    cache_reference_logps(run_dir, config, episodes)
    policy = build_policy(config, run_dir / "policy-initial.pt", trainable=True)
    audit_before = role_audit(policy, "policy")
    parameters = trainable_parameters(policy)
    parameter_before = [parameter.detach().clone() for parameter in parameters]
    if any("value" in name.lower() or "critic" in name.lower() for name, _ in policy.named_parameters()):
        raise RuntimeError("critic/value-head parameters are forbidden in GRPO")
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.)
    source_checkpoint = torch.load(run_dir / "policy-initial.pt", map_location="cpu", weights_only=False)
    if "optimizer" in source_checkpoint:
        optimizer.load_state_dict(source_checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cuda()
    seed_everything(config["seed"] + 1)
    group_by_task = {row["task_id"]: row for row in groups}
    records = []
    for episode in episodes:
        if group_by_task[episode["task_id"]]["zero_variance"]:
            continue
        advantage = advantages[(episode["group_index"], episode["episode_index"])]
        for query in episode["queries"]:
            if query["valid_generated_token_count"]:
                records.append((Path(query["sample_path"]), advantage, episode["task_id"]))
    rng = np.random.default_rng(config["seed"] + 1)
    rng.shuffle(records)
    if not records:
        raise RuntimeError("no nonzero-variance generated tokens available for GRPO")
    sums = {"objective": 0., "kl": 0., "ratio": 0.}
    extrema = {"kl_max": -float("inf"), "ratio_min": float("inf"), "ratio_max": -float("inf")}
    clipped = valid_tokens = optimizer_updates = 0
    gradient_norms, sampling_errors = [], []
    started = time.time()
    for start in range(0, len(records), config["query_minibatch"]):
        minibatch = records[start:start + config["query_minibatch"]]
        loaded, minibatch_tokens = [], 0
        for path, advantage, task_id in minibatch:
            record = torch.load(path, map_location="cpu", weights_only=False)
            mask = record["generated_token_mask"].cuda()
            loaded.append((record, mask, advantage, task_id))
            minibatch_tokens += int(mask.sum())
        optimizer.zero_grad(set_to_none=True)
        for record, mask, advantage, _task_id in loaded:
            logits = forward_logits(policy, record)
            current = sampled_token_logps(logits, record["generated_ids"])[mask]
            old = record["sampled_logps"].cuda()[mask]
            reference = record["reference_logps"].cuda()[mask]
            if optimizer_updates == 0:
                sampling_errors.append(float((current.detach() - old).abs().mean()))
            terms = clipped_grpo_terms(current, old, reference, advantage,
                                       config["grpo_clip_epsilon"], config["kl_beta"])
            if not all(torch.isfinite(terms[key]).all() for key in ("ratio", "kl", "objective")):
                raise RuntimeError("nonfinite GRPO token metric")
            (-terms["objective"].sum() / minibatch_tokens).backward()
            count = len(current); valid_tokens += count
            sums["objective"] += float(terms["objective"].sum().detach())
            sums["kl"] += float(terms["kl"].sum().detach())
            sums["ratio"] += float(terms["ratio"].sum().detach())
            extrema["kl_max"] = max(extrema["kl_max"], float(terms["kl"].max().detach()))
            extrema["ratio_min"] = min(extrema["ratio_min"], float(terms["ratio"].min().detach()))
            extrema["ratio_max"] = max(extrema["ratio_max"], float(terms["ratio"].max().detach()))
            clipped += int(((terms["ratio"] < 1 - config["grpo_clip_epsilon"]) |
                            (terms["ratio"] > 1 + config["grpo_clip_epsilon"])).sum())
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite policy gradient")
        gradient_norms.append(float(norm)); optimizer.step(); optimizer_updates += 1
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise RuntimeError("nonfinite final policy gradient")
    policy_changed = any(not torch.equal(before, parameter.detach())
                         for before, parameter in zip(parameter_before, parameters))
    if not policy_changed:
        raise RuntimeError("GRPO update did not change trainable policy parameters")
    save_policy(run_dir / "iteration-0001-grpo.pt", policy, config, 1, optimizer)
    metrics = {
        "mean_sampled_kl": sums["kl"] / valid_tokens, "sampled_kl_max": extrema["kl_max"],
        "probability_ratio_mean": sums["ratio"] / valid_tokens,
        "probability_ratio_min": extrema["ratio_min"], "probability_ratio_max": extrema["ratio_max"],
        "clipping_fraction": clipped / valid_tokens, "grpo_objective": sums["objective"] / valid_tokens,
        "gradient_norm": float(np.mean(gradient_norms)), "gradient_norm_max": float(np.max(gradient_norms)),
        "optimization_time_seconds": time.time() - started, "optimizer_updates": optimizer_updates,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "policy_changed": policy_changed,
        "optimized_valid_tokens": valid_tokens, "optimized_query_sequences": len(records),
        "sampling_replay_first_minibatch_mean_error_max": max(sampling_errors),
        "update_epochs_per_rollout_batch": 1,
        "loss_components": {"ppo_value_loss": False, "gae": False, "entropy_bonus": False, "sft_loss": False},
        "role_audit": {"policy_before": audit_before, "policy_after": role_audit(policy, "policy"),
                       "only_policy_received_gradients": True},
    }
    write(run_dir / "iteration-0001.optimization.json", metrics)
    return metrics
