"""Durable ten-iteration G=4 GRPO runner for the single-GPU workstation."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
import traceback
from pathlib import Path

from onpolicy_grpo.cluster_train import (cleanup, initialize_policy, prepare_iteration,
                                         wandb_metrics)
from onpolicy_grpo.common import PACKAGE, read, sha256, write


def freeze_local_run(config_path: str | Path, run_dir_override: str | None = None) -> tuple[Path, dict]:
    source = Path(config_path).resolve()
    config = json.loads(os.path.expandvars(source.read_text()))
    root = Path(run_dir_override or config["run_dir"]).resolve()
    config["run_dir"] = str(root)
    config["base_checkpoint"] = str(Path(config["base_checkpoint"]).resolve())
    selection = read(PACKAGE / "selection_b200.json")
    if config["task_ids"] != selection["selected"]:
        raise ValueError("workstation task selection differs from selection_b200.json")
    if config["iterations"] != 10 or config["group_size"] != 4 or config["rollout_workers"] != 4:
        raise ValueError("the overnight workstation run requires 10 iterations, G=4, and 4 workers")
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "config.json"
    if frozen.exists() and read(frozen) != config:
        raise RuntimeError("refusing to alter the frozen workstation training configuration")
    write(frozen, config)
    write(root / "provenance.json", {
        "config_source": str(source),
        "base_checkpoint_sha256": sha256(config["base_checkpoint"]),
        "algorithm": "on-policy GRPO",
        "bounded_iterations": 10,
        "group_size": 4,
        "ppo_artifacts_modified": False,
    })
    return root, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE / "config_workstation_g4_10.json"))
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    root, config = freeze_local_run(args.config, args.run_dir)
    lock = (root / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = None
    try:
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

        history = read(root / "history.json") if (root / "history.json").exists() else []
        history = [row for row in history
                   if (root / "iterations" / f"iteration-{int(row['iteration']):04d}" /
                       "iteration-0001.complete.json").exists()]
        global_optimizer_step = int(history[-1]["global_optimizer_step"]) if history else 0
        for iteration in range(1, config["iterations"] + 1):
            destination = prepare_iteration(root, config, iteration)
            complete = destination / "iteration-0001.complete.json"
            if complete.exists():
                continue
            write(root / "status.json", {"status": "running", "stage": "refresh",
                                           "iteration": iteration, "pid": os.getpid()})
            initialize_policy(root, destination, config, iteration)
            write(root / "status.json", {"status": "running", "stage": "rollouts",
                                           "iteration": iteration, "pid": os.getpid()})
            from onpolicy_grpo.smoke import collect
            rollout_time = collect(destination, read(destination / "config.json"))
            write(root / "status.json", {"status": "running", "stage": "optimization",
                                           "iteration": iteration, "pid": os.getpid()})
            from onpolicy_grpo.optimize import optimize
            from onpolicy_grpo.report import summarize
            from onpolicy_grpo.validate import validate

            checkpoint = destination / "iteration-0001-grpo.pt"
            optimization_path = destination / "iteration-0001.optimization.json"
            optimization = (read(optimization_path) if checkpoint.exists() and optimization_path.exists()
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
                      "checkpoint": str(checkpoint)}
            history = [row for row in history if row["iteration"] != iteration] + [record]
            history.sort(key=lambda row: row["iteration"])
            write(root / "history.json", history)
            run.log(wandb_metrics(result, iteration, global_optimizer_step), step=iteration)
            write(complete, {"status": "complete", "global_iteration": iteration,
                             "global_optimizer_step": global_optimizer_step, "validation": validation})
            (destination / "policy-initial.pt").unlink(missing_ok=True)
            (destination / "rollout-policy-iteration-0001.pt").unlink(missing_ok=True)
            write(root / "status.json", {"status": "running", "stage": "iteration_complete",
                                           "iteration": iteration, "pid": os.getpid(),
                                           "wandb": {"id": run.id, "url": run.url}})
            cleanup()
        write(root / "status.json", {"status": "complete", "stage": "complete", "iteration": 10,
                                       "pid": None, "wandb": {"id": run.id, "url": run.url},
                                       "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    except Exception:
        write(root / "status.json", {"status": "failed", "stage": "failed", "pid": None,
                                       "error": traceback.format_exc()})
        raise
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
