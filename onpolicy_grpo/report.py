"""Aggregate rollout, group, and optimization metrics for JSON and HTML review."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from onpolicy_grpo.common import read, write


def summarize(run_dir: str | Path, rollout_time: float | None = None) -> dict:
    run_dir = Path(run_dir)
    config = read(run_dir / "config.json")
    manifest = read(run_dir / "iteration-0001.manifest.json")
    episodes = [read(path) for path in sorted((run_dir / "rollouts/iteration-0001").glob("episode-*.json"))]
    groups = read(run_dir / "iteration-0001.groups.json")
    optimization = read(run_dir / "iteration-0001.optimization.json")
    queries = [query | {"task_id": episode["task_id"]} for episode in episodes for query in episode["queries"]]
    applicable = [query for query in queries if query["applicable"]]
    directional = [query for query in applicable if query["directional"]]
    stops = [query for query in applicable if query["stop"]]

    def mismatch(rows):
        return sum(bool(row["mismatch"]) for row in rows) / len(rows) if rows else None

    advantages = [advantage for group in groups for advantage in group["advantages"]]
    per_task = {}
    for task in config["task_ids"]:
        task_episodes = [episode for episode in episodes if episode["task_id"] == task]
        task_queries = [query for query in applicable if query["task_id"] == task]
        per_task[str(task)] = {
            "episodes": len(task_episodes),
            "successes": sum(episode["success"] for episode in task_episodes),
            "task_success_rate": sum(episode["success"] for episode in task_episodes) / len(task_episodes),
            "mean_alignment_cost": float(np.mean([episode["mean_alignment_cost"] for episode in task_episodes])),
            "mean_combined_reward": float(np.mean([episode["combined_reward"] for episode in task_episodes])),
            "overall_mismatch_rate": mismatch(task_queries),
            "initial_state_ids": [episode["initial_state"] for episode in task_episodes],
        }
    result = {
        "algorithm": "on-policy GRPO",
        "iteration": 1,
        "total_episodes": len(episodes),
        "group_count": len(groups),
        "group_size": config["group_size"],
        "task_ids": manifest["task_order"],
        "initial_state_ids": {str(task): per_task[str(task)]["initial_state_ids"] for task in config["task_ids"]},
        "task_success_rate": sum(episode["success"] for episode in episodes) / len(episodes),
        "directional_mismatch_rate": mismatch(directional),
        "stop_mismatch_rate": mismatch(stops),
        "overall_mismatch_rate": mismatch(applicable),
        "mean_directional_cosine": float(np.mean([query["cosine"] for query in directional])) if directional else None,
        "mean_stop_displacement_m": float(np.mean([query["stop_displacement_m"] for query in stops])) if stops else None,
        "mean_alignment_cost": float(np.mean([episode["mean_alignment_cost"] for episode in episodes])),
        "mean_combined_reward": float(np.mean([episode["combined_reward"] for episode in episodes])),
        "group_reward_means": {str(group["task_id"]): group["reward_mean"] for group in groups},
        "group_reward_standard_deviations": {str(group["task_id"]): group["reward_std"] for group in groups},
        "zero_variance_group_tasks": [group["task_id"] for group in groups if group["zero_variance"]],
        "fraction_zero_variance_groups": sum(group["zero_variance"] for group in groups) / len(groups),
        "advantage_mean": float(np.mean(advantages)),
        "advantage_standard_deviation": float(np.std(advantages)),
        "advantage_min": float(np.min(advantages)),
        "advantage_max": float(np.max(advantages)),
        "rollout_time_seconds": rollout_time,
        "per_task": per_task,
        "optimization": optimization,
        "sampling": manifest["sampling"],
        "loss_components_executed": optimization["loss_components"],
    }
    write(run_dir / "results.json", result)
    rows = "".join(f"<tr><td>{task}</td><td>{score['successes']}/{config['group_size']}</td><td>{score['mean_combined_reward']:.4f}</td><td>{score['mean_alignment_cost']:.4f}</td></tr>"
                   for task, score in per_task.items())
    page = ("<!doctype html><meta charset='utf-8'><title>MiniVLA GRPO smoke</title>"
            "<style>body{font:16px/1.45 system-ui;margin:32px;max-width:1200px}td,th{padding:8px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>"
            f"<h1>On-policy GRPO workstation smoke</h1><p>Iteration 1 only: {len(episodes)} episodes, {len(groups)} task groups × {config['group_size']}.</p>"
            f"<p>Mean combined reward <b>{result['mean_combined_reward']:.4f}</b>; success <b>{result['task_success_rate']:.2%}</b>; alignment cost <b>{result['mean_alignment_cost']:.4f}</b>.</p>"
            "<table><tr><th>Task</th><th>Success</th><th>Reward</th><th>Alignment cost</th></tr>" + rows + "</table>"
            "<h2>Complete metrics</h2><pre>" + html.escape(json.dumps(result, indent=2)) + "</pre>")
    (run_dir / "index.html").write_text(page)
    return result
