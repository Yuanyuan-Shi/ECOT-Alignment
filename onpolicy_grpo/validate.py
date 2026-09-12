"""Post-run integration validation for the complete 80-episode smoke iteration."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from onpolicy_grpo.common import read, sha256


def validate(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir).resolve()
    config = read(run_dir / "config.json")
    manifest = read(run_dir / "iteration-0001.manifest.json")
    result = read(run_dir / "results.json")
    groups = read(run_dir / "iteration-0001.groups.json")
    optimization = read(run_dir / "iteration-0001.optimization.json")
    episodes = [read(path) for path in sorted((run_dir / "rollouts/iteration-0001").glob("episode-*.json"))]
    checks = {}
    expected_episodes = len(config["task_ids"]) * config["group_size"]
    checks["exact_episode_count"] = len(episodes) == expected_episodes == result["total_episodes"]
    checks["exactly_10_groups_of_configured_size"] = len(groups) == 10 and all(group["episode_count"] == config["group_size"] for group in groups)
    checks["all_tasks_exactly_once"] = sorted(group["task_id"] for group in groups) == sorted(config["task_ids"])
    checks["same_task_within_groups"] = all(len({row["task_id"] for row in episodes if row["group_index"] == index}) == 1 for index in range(10))
    checks["independent_state_variation"] = all(len({row["initial_state"] for row in episodes if row["task_id"] == task}) > 1 for task in config["task_ids"])
    checks["sampling_distribution"] = manifest["sampling"] == {"do_sample": True, "temperature": 1., "top_p": 1., "top_k": None}
    checks["group_advantage_zero_mean"] = all(group["zero_variance"] or abs(group["advantage_mean"]) < 1e-6 for group in groups)
    # The exact query count excludes invalid-action sequences and zero-variance groups.
    expected_sequences = sum(query["valid_generated_token_count"] > 0 for episode in episodes for query in episode["queries"]
                             if not next(group for group in groups if group["task_id"] == episode["task_id"])["zero_variance"])
    checks["zero_variance_groups_excluded"] = optimization["optimized_query_sequences"] == expected_sequences
    checks["reference_unchanged"] = read(run_dir / "reference-role-audit.json")["unchanged"]
    rollout_audits = [read(path) for path in run_dir.glob("rollout-role-audit-worker-*.json")]
    checks["rollout_frozen"] = len(rollout_audits) == config["rollout_workers"] and all(audit["trainable_parameter_tensors"] == 0 for audit in rollout_audits)
    marker = read(run_dir / "iteration-0001.rollout-refreshed.json")
    checks["rollout_refreshed_exactly_once"] = marker["refresh_count"] == 1 and sha256(run_dir / "rollout-policy-iteration-0001.pt") == marker["snapshot_sha256"]
    checkpoint = torch.load(run_dir / "iteration-0001-grpo.pt", map_location="cpu", weights_only=False)
    checks["no_critic_or_value_head"] = checkpoint["algorithm"] == "GRPO" and not checkpoint["has_value_head"] and optimization["loss_components"] == {"ppo_value_loss": False, "gae": False, "entropy_bonus": False, "sft_loss": False}
    checks["only_policy_gradients"] = optimization["role_audit"]["only_policy_received_gradients"] and optimization["policy_changed"]
    numeric = [result["task_success_rate"], result["mean_alignment_cost"], result["mean_combined_reward"],
               result["advantage_mean"], result["advantage_standard_deviation"],
               optimization["mean_sampled_kl"], optimization["sampled_kl_max"],
               optimization["probability_ratio_mean"], optimization["probability_ratio_min"],
               optimization["probability_ratio_max"], optimization["clipping_fraction"],
               optimization["grpo_objective"], optimization["gradient_norm"]]
    checks["all_core_metrics_finite"] = all(math.isfinite(value) for value in numeric)
    checks["sampling_replay_agreement"] = optimization["sampling_replay_first_minibatch_mean_error_max"] < 1e-3
    checks["all_episodes_applicable"] = all(episode["applicable_query_count"] > 0 for episode in episodes)
    checks["fixed_instruction_per_task"] = all(len({row["instruction"] for row in episodes if row["task_id"] == task}) == 1 for task in config["task_ids"])
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    return {"passed": True, "checks": checks, "check_count": len(checks)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", required=True); args = parser.parse_args()
    from onpolicy_grpo.common import write
    outcome = validate(args.run_dir); write(Path(args.run_dir) / "integration-validation.json", outcome); print(outcome)


if __name__ == "__main__":
    main()
