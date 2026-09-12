"""Durable controller for exactly one configured-size workstation GRPO iteration."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

from onpolicy_grpo.common import deterministic_resume_action, read, sha256, write
from onpolicy_grpo.optimize import optimize
from onpolicy_grpo.policy import build_policy, role_audit, save_policy
from onpolicy_grpo.prepare import prepare
from onpolicy_grpo.report import summarize
from onpolicy_grpo.validate import validate


def initialize_and_refresh(run_dir: Path, config: dict) -> None:
    initial = run_dir / "policy-initial.pt"
    if not initial.exists():
        policy = build_policy(config, trainable=True)
        audit = role_audit(policy, "policy")
        save_policy(initial, policy, config, iteration=0)
        write(run_dir / "policy-initial.audit.json", audit | {"initialized_from": config["base_checkpoint"]})
        del policy
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    marker = run_dir / "iteration-0001.rollout-refreshed.json"
    snapshot = run_dir / "rollout-policy-iteration-0001.pt"
    if marker.exists():
        state = read(marker)
        if state["refresh_count"] != 1 or not snapshot.exists() or sha256(snapshot) != state["snapshot_sha256"]:
            raise RuntimeError("rollout-policy refresh marker or immutable snapshot is inconsistent")
        return
    temporary = snapshot.with_suffix(".pt.tmp")
    shutil.copy2(initial, temporary)
    temporary.replace(snapshot)
    write(marker, {"iteration": 1, "refresh_count": 1, "source": str(initial),
                   "snapshot": str(snapshot), "snapshot_sha256": sha256(snapshot),
                   "refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def collect(run_dir: Path, config: dict) -> float:
    rollout_dir = run_dir / "rollouts/iteration-0001"
    existing = list(rollout_dir.glob("episode-*.json")) if rollout_dir.exists() else []
    expected_episodes = len(config["task_ids"]) * config["group_size"]
    if len(existing) == expected_episodes:
        return read(run_dir / "rollout-summary.json")["rollout_time_seconds"] if (run_dir / "rollout-summary.json").exists() else 0.
    started = time.time()
    processes, handles = [], []
    for worker in range(config["rollout_workers"]):
        log = (run_dir / f"rollout-worker-{worker}.log").open("a")
        handles.append(log)
        command = [sys.executable, "-m", "onpolicy_grpo.rollout", "--run-dir", str(run_dir),
                   "--worker", str(worker), "--workers", str(config["rollout_workers"])]
        processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[1]))
    try:
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("a GRPO rollout worker failed")
            time.sleep(10)
        if any(process.returncode for process in processes):
            raise RuntimeError("a GRPO rollout worker failed")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for handle in handles:
            handle.close()
    count = len(list(rollout_dir.glob("episode-*.json")))
    if count != expected_episodes:
        raise RuntimeError(f"expected {expected_episodes} collected episodes, found {count}")
    elapsed = time.time() - started
    write(run_dir / "rollout-summary.json", {"rollout_time_seconds": elapsed, "accepted_episodes": count,
                                              "rejected_no_motion_episodes": len(list((run_dir / "errors/iteration-0001").glob("*.json")))})
    return elapsed


def log_wandb(run_dir: Path, config: dict, result: dict) -> dict:
    import wandb
    wandb_dir = run_dir / "wandb"; wandb_dir.mkdir(exist_ok=True)
    saved = read(run_dir / "wandb.json") if (run_dir / "wandb.json").exists() else {}
    run = wandb.init(project=config["wandb_project"], entity=config["wandb_entity"],
                     name=config["wandb_run_name"], config=config, dir=str(wandb_dir),
                     mode=config["wandb_mode"], id=saved.get("id"), resume="allow")
    write(run_dir / "wandb.json", {"id": run.id, "url": run.url, "project": config["wandb_project"],
                                    "run_name": config["wandb_run_name"]})
    ordered_names = ["train/mean_combined_reward", "train/task_success_rate", "train/mean_alignment_cost",
                     "train/overall_mismatch_rate", "train/mean_sampled_kl", "train/grpo_objective",
                     "train/clipping_fraction", "train/gradient_norm"]
    for name in ordered_names:
        wandb.define_metric(name, step_metric="iteration")
    optimization = result["optimization"]
    metrics = OrderedDict([
        ("train/mean_combined_reward", result["mean_combined_reward"]),
        ("train/task_success_rate", result["task_success_rate"]),
        ("train/mean_alignment_cost", result["mean_alignment_cost"]),
        ("train/overall_mismatch_rate", result["overall_mismatch_rate"]),
        ("train/mean_sampled_kl", optimization["mean_sampled_kl"]),
        ("train/grpo_objective", optimization["grpo_objective"]),
        ("train/clipping_fraction", optimization["clipping_fraction"]),
        ("train/gradient_norm", optimization["gradient_norm"]),
        ("iteration", 1), ("train/total_episodes", result["total_episodes"]),
        ("train/directional_mismatch_rate", result["directional_mismatch_rate"]),
        ("train/stop_mismatch_rate", result["stop_mismatch_rate"]),
        ("train/mean_directional_cosine", result["mean_directional_cosine"]),
        ("train/mean_stop_displacement_m", result["mean_stop_displacement_m"]),
        ("train/fraction_zero_variance_groups", result["fraction_zero_variance_groups"]),
        ("train/advantage_mean", result["advantage_mean"]),
        ("train/advantage_std", result["advantage_standard_deviation"]),
        ("train/advantage_min", result["advantage_min"]), ("train/advantage_max", result["advantage_max"]),
        ("train/sampled_kl_max", optimization["sampled_kl_max"]),
        ("train/probability_ratio_mean", optimization["probability_ratio_mean"]),
        ("train/probability_ratio_min", optimization["probability_ratio_min"]),
        ("train/probability_ratio_max", optimization["probability_ratio_max"]),
        ("train/learning_rate", config["learning_rate"]),
        ("time/rollout_seconds", result["rollout_time_seconds"]),
        ("time/optimization_seconds", optimization["optimization_time_seconds"]),
        ("system/peak_gpu_memory_bytes", optimization["peak_gpu_memory_bytes"]),
    ])
    for task, values in result["per_task"].items():
        for key in ("task_success_rate", "mean_alignment_cost", "mean_combined_reward", "overall_mismatch_rate"):
            metrics[f"task/{task}/{key}"] = values[key]
        metrics[f"task/{task}/reward_mean"] = result["group_reward_means"][task]
        metrics[f"task/{task}/reward_std"] = result["group_reward_standard_deviations"][task]
    run.log(metrics, step=1)
    run.summary["task_ids"] = result["task_ids"]
    run.summary["initial_state_ids"] = json.dumps(result["initial_state_ids"], sort_keys=True)
    run.summary["group_reward_means"] = result["group_reward_means"]
    run.summary["group_reward_standard_deviations"] = result["group_reward_standard_deviations"]
    run.summary["smoke_only_no_full_training"] = True
    identity = {"id": run.id, "url": run.url}
    run.finish()
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_smoke_g4.json")))
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    run_dir = prepare(args.config, args.run_dir)
    lock_handle = (run_dir / "run.lock").open("a")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if deterministic_resume_action(run_dir, 1) == "complete":
        print(f"Smoke iteration already complete: {run_dir}")
        return
    state = {"status": "running", "stage": "initialization", "pid": os.getpid(), "iteration": 1}
    write(run_dir / "status.json", state)
    config = read(run_dir / "config.json")
    if config["iterations"] != 1:
        raise RuntimeError("this controller refuses to launch more than one workstation iteration")
    try:
        initialize_and_refresh(run_dir, config)
        state["stage"] = "rollout collection"; write(run_dir / "status.json", state)
        rollout_time = collect(run_dir, config)
        state["stage"] = "GRPO optimization"; write(run_dir / "status.json", state)
        optimization = (read(run_dir / "iteration-0001.optimization.json")
                        if (run_dir / "iteration-0001-grpo.pt").exists() and (run_dir / "iteration-0001.optimization.json").exists()
                        else optimize(run_dir))
        result = summarize(run_dir, rollout_time)
        state["stage"] = "W&B logging"; write(run_dir / "status.json", state)
        wandb_identity = log_wandb(run_dir, config, result)
        validation = validate(run_dir)
        write(run_dir / "integration-validation.json", validation)
        state.update(status="complete", stage="complete", pid=None, completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     wandb=wandb_identity, checkpoint=str(run_dir / "iteration-0001-grpo.pt"))
        write(run_dir / "status.json", state)
        write(run_dir / "iteration-0001.complete.json", {"status": "complete", "results": str(run_dir / "results.json"),
                                                           "checkpoint": str(run_dir / "iteration-0001-grpo.pt"),
                                                           "validation": validation})
        print(json.dumps({"status": "complete", "run_dir": str(run_dir), "wandb": wandb_identity,
                          "mean_combined_reward": result["mean_combined_reward"]}, indent=2))
    except Exception:
        state.update(status="failed", stage="failed", pid=None, error=traceback.format_exc())
        write(run_dir / "status.json", state)
        raise


if __name__ == "__main__":
    main()
