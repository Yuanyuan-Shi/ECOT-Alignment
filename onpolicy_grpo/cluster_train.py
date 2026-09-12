"""Durable 8-GPU, multi-iteration GRPO controller for one B200 node."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import shutil
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist

from onpolicy_grpo.common import PACKAGE, make_iteration_manifest, read, sha256, validate_manifest, write


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def world() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def freeze_run(config_path: str | Path, run_dir_override: str | None) -> tuple[Path, dict]:
    source = Path(config_path).resolve()
    config = json.loads(os.path.expandvars(source.read_text()))
    run_dir = Path(run_dir_override or config["run_dir"]).resolve()
    config["run_dir"] = str(run_dir)
    config["base_checkpoint"] = str(Path(config["base_checkpoint"]).resolve())
    selection = read(PACKAGE / "selection_b200.json")
    if config["task_ids"] != selection["selected"]:
        raise ValueError("B200 task selection differs from selection_b200.json")
    if config["iterations"] != 10 or config["group_size"] != 8 or config["rollout_workers"] != 8:
        raise ValueError("this bounded pilot requires exactly 10 iterations, G=8, and 8 rollout workers")
    if world() != 8:
        raise ValueError(f"the B200 pilot requires torchrun world size 8, got {world()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / "config.json"
    if frozen.exists() and read(frozen) != config:
        raise RuntimeError("refusing to alter the frozen B200 pilot configuration")
    write(frozen, config)
    selection_target = run_dir / "selection.json"
    if not selection_target.exists():
        shutil.copy2(PACKAGE / "selection_b200.json", selection_target)
    write(run_dir / "provenance.json", {
        "config_source": str(source),
        "base_checkpoint_sha256": sha256(config["base_checkpoint"]),
        "algorithm": "on-policy GRPO",
        "bounded_pilot": True,
        "requested_gpus": 8,
        "ppo_artifacts_modified": False,
    })
    return run_dir, config


def prepare_iteration(root: Path, config: dict, iteration: int) -> Path:
    destination = root / "iterations" / f"iteration-{iteration:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    local = dict(config)
    local["iterations"] = 1
    local["run_dir"] = str(destination)
    local["seed"] = int(config["seed"]) + (iteration - 1) * 1_000_003
    frozen = destination / "config.json"
    if frozen.exists() and read(frozen) != local:
        raise RuntimeError(f"iteration {iteration} frozen configuration changed")
    write(frozen, local)
    manifest = make_iteration_manifest(config, iteration)
    validate_manifest(manifest, config["task_ids"], config["group_size"])
    manifest_path = destination / "iteration-0001.manifest.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise RuntimeError(f"iteration {iteration} manifest changed")
    write(manifest_path, manifest)
    return destination


def initialize_policy(root: Path, destination: Path, config: dict, iteration: int) -> None:
    from onpolicy_grpo.policy import build_policy, role_audit, save_policy

    initial = destination / "policy-initial.pt"
    if not initial.exists():
        if iteration == 1:
            policy = build_policy(config, trainable=True)
            save_policy(initial, policy, config, iteration=0)
            write(destination / "policy-initial.audit.json", role_audit(policy, "policy") | {
                "initialized_from": config["base_checkpoint"]
            })
            del policy
            cleanup()
        else:
            previous = root / "iterations" / f"iteration-{iteration - 1:04d}" / "iteration-0001-grpo.pt"
            if not previous.exists():
                raise RuntimeError(f"missing previous checkpoint for iteration {iteration}")
            temporary = initial.with_suffix(".pt.tmp")
            shutil.copy2(previous, temporary)
            temporary.replace(initial)
            # The durable input copy now makes the previous output checkpoint
            # redundant. Keep at most the latest resumable policy on disk.
            previous.unlink()
    marker = destination / "iteration-0001.rollout-refreshed.json"
    snapshot = destination / "rollout-policy-iteration-0001.pt"
    if marker.exists():
        saved = read(marker)
        if saved["refresh_count"] != 1 or not snapshot.exists() or sha256(snapshot) != saved["snapshot_sha256"]:
            raise RuntimeError(f"iteration {iteration} rollout refresh is inconsistent")
        return
    temporary = snapshot.with_suffix(".pt.tmp")
    shutil.copy2(initial, temporary)
    temporary.replace(snapshot)
    write(marker, {"iteration": iteration, "refresh_count": 1, "source": str(initial),
                   "snapshot": str(snapshot), "snapshot_sha256": sha256(snapshot),
                   "refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def collect(destination: Path, config: dict, iteration: int) -> float:
    from onpolicy_grpo.rollout import main as rollout

    rollout_dir = destination / "rollouts/iteration-0001"
    expected = len(config["task_ids"]) * config["group_size"]
    summary_path = destination / "rollout-summary.json"
    if len(list(rollout_dir.glob("episode-*.json"))) == expected:
        return read(summary_path)["rollout_time_seconds"] if summary_path.exists() else 0.0
    started = time.time()
    rollout(["--run-dir", str(destination), "--worker", str(rank()), "--workers", str(world())])
    cleanup()
    barrier()
    if rank() == 0:
        count = len(list(rollout_dir.glob("episode-*.json")))
        if count != expected:
            raise RuntimeError(f"iteration {iteration}: expected {expected} episodes, found {count}")
        write(summary_path, {
            "rollout_time_seconds": time.time() - started,
            "accepted_episodes": count,
            "rejected_no_motion_episodes": len(list((destination / "errors/iteration-0001").glob("*.json"))),
        })
    barrier()
    return read(summary_path)["rollout_time_seconds"]


def wandb_metrics(result: dict, iteration: int, global_optimizer_step: int) -> OrderedDict:
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
        ("iteration", iteration),
        ("train/global_optimizer_step", global_optimizer_step),
        ("train/iteration_episodes", result["total_episodes"]),
        ("train/total_episodes", iteration * result["total_episodes"]),
        ("task_ids", json.dumps(result["task_ids"])),
        ("initial_state_ids", json.dumps(result["initial_state_ids"], sort_keys=True)),
        ("train/directional_mismatch_rate", result["directional_mismatch_rate"]),
        ("train/stop_mismatch_rate", result["stop_mismatch_rate"]),
        ("train/mean_directional_cosine", result["mean_directional_cosine"]),
        ("train/mean_stop_displacement_m", result["mean_stop_displacement_m"]),
        ("train/fraction_zero_variance_groups", result["fraction_zero_variance_groups"]),
        ("train/advantage_mean", result["advantage_mean"]),
        ("train/advantage_std", result["advantage_standard_deviation"]),
        ("train/advantage_min", result["advantage_min"]),
        ("train/advantage_max", result["advantage_max"]),
        ("train/sampled_kl_max", optimization["sampled_kl_max"]),
        ("train/probability_ratio_mean", optimization["probability_ratio_mean"]),
        ("train/probability_ratio_min", optimization["probability_ratio_min"]),
        ("train/probability_ratio_max", optimization["probability_ratio_max"]),
        ("train/learning_rate", 1e-6),
        ("time/rollout_seconds", result["rollout_time_seconds"]),
        ("time/optimization_seconds", optimization["optimization_time_seconds"]),
        ("system/peak_gpu_memory_bytes", optimization["peak_gpu_memory_bytes"]),
    ])
    for task, values in result["per_task"].items():
        for key in ("task_success_rate", "mean_alignment_cost", "mean_combined_reward", "overall_mismatch_rate"):
            metrics[f"task/{task}/{key}"] = values[key]
        metrics[f"task/{task}/reward_mean"] = result["group_reward_means"][task]
        metrics[f"task/{task}/reward_std"] = result["group_reward_standard_deviations"][task]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE / "config_b200_10.json"))
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    if world() > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    root = None
    lock = None
    run = None
    try:
        if rank() == 0:
            root, config = freeze_run(args.config, args.run_dir)
            lock = (root / "run.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        barrier()
        root = Path(args.run_dir or os.path.expandvars(read(Path(args.config).resolve())["run_dir"])).resolve()
        config = read(root / "config.json")
        if rank() == 0:
            import wandb
            saved = read(root / "wandb.json") if (root / "wandb.json").exists() else {}
            (root / "wandb").mkdir(exist_ok=True)
            run = wandb.init(project=config["wandb_project"], entity=config["wandb_entity"],
                             name=config["wandb_run_name"], config=config, dir=str(root / "wandb"),
                             mode=config["wandb_mode"], id=saved.get("id"), resume="allow")
            write(root / "wandb.json", {"id": run.id, "url": run.url})
            print(f"WANDB_URL {run.url}", flush=True)
            for name in ("train/mean_combined_reward", "train/task_success_rate", "train/mean_alignment_cost",
                         "train/overall_mismatch_rate", "train/mean_sampled_kl", "train/grpo_objective",
                         "train/clipping_fraction", "train/gradient_norm"):
                wandb.define_metric(name, step_metric="iteration")
        barrier()

        global_optimizer_step = 0
        history = read(root / "history.json") if (root / "history.json").exists() else []
        # Only committed iterations contribute to resume state. A crash after
        # writing metrics but before the completion marker is safely replayed.
        history = [row for row in history
                   if (root / "iterations" / f"iteration-{int(row['iteration']):04d}" /
                       "iteration-0001.complete.json").exists()]
        if history:
            global_optimizer_step = int(history[-1]["global_optimizer_step"])
        for iteration in range(1, config["iterations"] + 1):
            destination = prepare_iteration(root, config, iteration) if rank() == 0 else None
            barrier()
            destination = root / "iterations" / f"iteration-{iteration:04d}"
            complete = destination / "iteration-0001.complete.json"
            if complete.exists():
                continue
            if rank() == 0:
                write(root / "status.json", {"status": "running", "stage": "refresh", "iteration": iteration})
                initialize_policy(root, destination, config, iteration)
            barrier()
            if rank() == 0:
                write(root / "status.json", {"status": "running", "stage": "rollouts", "iteration": iteration})
            rollout_time = collect(destination, config, iteration)
            if rank() == 0:
                from onpolicy_grpo.optimize import optimize
                from onpolicy_grpo.report import summarize
                from onpolicy_grpo.validate import validate

                write(root / "status.json", {"status": "running", "stage": "optimization", "iteration": iteration})
                checkpoint = destination / "iteration-0001-grpo.pt"
                optimization_path = destination / "iteration-0001.optimization.json"
                optimization = (read(optimization_path)
                                if checkpoint.exists() and optimization_path.exists()
                                else optimize(destination))
                result = summarize(destination, rollout_time)
                validation = validate(destination)
                write(destination / "integration-validation.json", validation)
                global_optimizer_step += int(optimization["optimizer_updates"])
                result["iteration"] = iteration
                write(destination / "results.json", result)
                record = {"iteration": iteration, "global_optimizer_step": global_optimizer_step,
                          "mean_combined_reward": result["mean_combined_reward"],
                          "task_success_rate": result["task_success_rate"],
                          "mean_alignment_cost": result["mean_alignment_cost"],
                          "grpo_objective": optimization["grpo_objective"],
                          "mean_sampled_kl": optimization["mean_sampled_kl"],
                          "checkpoint": str(destination / "iteration-0001-grpo.pt")}
                history = [row for row in history if row["iteration"] != iteration]
                history.append(record)
                history.sort(key=lambda row: row["iteration"])
                write(root / "history.json", history)
                run.log(wandb_metrics(result, iteration, global_optimizer_step), step=iteration)
                write(complete, {"status": "complete", "global_iteration": iteration,
                                 "global_optimizer_step": global_optimizer_step, "validation": validation})
                # These immutable inputs have served their rollout/replay audit.
                # The validated output remains until copied into the next
                # iteration, and iteration 10 remains as the final checkpoint.
                (destination / "policy-initial.pt").unlink(missing_ok=True)
                (destination / "rollout-policy-iteration-0001.pt").unlink(missing_ok=True)
                write(root / "status.json", {"status": "running", "stage": "iteration_complete",
                                               "iteration": iteration, "wandb": {"id": run.id, "url": run.url}})
            barrier()
            cleanup()
        if rank() == 0:
            write(root / "status.json", {"status": "complete", "stage": "complete", "iteration": 10,
                                           "wandb": {"id": run.id, "url": run.url},
                                           "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    except Exception:
        if rank() == 0 and root is not None:
            write(root / "status.json", {"status": "failed", "stage": "failed", "error": traceback.format_exc()})
        raise
    finally:
        if run is not None:
            run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
