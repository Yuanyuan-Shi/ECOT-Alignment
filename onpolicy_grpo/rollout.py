"""Collect grouped, on-policy LIBERO episodes from one immutable rollout snapshot."""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
from onpolicy_grpo.common import alignment, episode_reward, read, write
from onpolicy_grpo.policy import build_policy, role_audit, sample_query, seed_everything


def collect_one(policy, suite, assignment, destination: Path, config, attempt: int):
    task_id = assignment["task_id"]
    environment_seed = assignment["environment_seed"] + attempt * 10_000_019
    sampling_seed = assignment["sampling_seed"] + attempt * 10_000_079
    state_rng = np.random.default_rng(environment_seed)
    initial_state = (int(state_rng.integers(config["training_state_min"], config["training_state_max_exclusive"]))
                     if attempt else assignment["initial_state"])
    seed_everything(sampling_seed)
    task = suite.get_task(task_id)
    environment, instruction = get_libero_env(task, "prismatic", resolution=224)
    try:
        environment.seed(environment_seed)
        environment.reset()
        states = suite.get_task_init_states(task_id)
        observation = environment.set_init_state(states[initial_state])
        initial_hash = hashlib.sha256(np.asarray(states[initial_state]).tobytes()).hexdigest()
        for _ in range(10):
            observation, _, _, _ = environment.step(get_libero_dummy_action("prismatic"))
        queries, success, environment_steps = [], False, 0
        started = time.time()
        prefix = destination / f"episode-{assignment['group_index']:02d}-{assignment['episode_index']:02d}"
        while environment_steps < config["max_environment_steps"] and not success:
            image = get_libero_image(observation, 224)
            actions, record = sample_query(policy, image, instruction, config)
            score = alignment(record["reasoning"], actions, config) if record["valid_action_tokens"] else {
                "applicable": False, "directional": False, "stop": False, "cost": None,
                "mismatch": None, "cosine": None, "stop_displacement_m": None,
                "commanded_displacement_m": [0., 0., 0.], "commanded_displacement_norm_m": 0.,
            }
            query_index = len(queries)
            tensor_path = prefix.with_name(prefix.name + f"-query-{query_index:03d}.pt")
            torch.save(record, tensor_path)
            row = dict(score, query_index=query_index, reasoning=record["reasoning"],
                       action_tokens=record["action_tokens"], valid_action_tokens=record["valid_action_tokens"],
                       decoded_action_chunk=actions.tolist(), token_count=len(record["generated_ids"]),
                       valid_generated_token_count=int(record["generated_token_mask"].sum()),
                       sample_path=str(tensor_path))
            if not record["valid_action_tokens"]:
                queries.append(row)
                break
            executed = 0
            for action in actions[:config["action_horizon"]]:
                command = invert_gripper_action(normalize_gripper_action(action.copy(), binarize=True))
                observation, _, done, _ = environment.step(command.tolist())
                environment_steps += 1
                executed += 1
                success = bool(done)
                if success or environment_steps >= config["max_environment_steps"]:
                    break
            row["executed_actions"] = executed
            queries.append(row)
        applicable = [query["cost"] for query in queries if query["applicable"]]
        return dict(assignment, instruction=instruction, initial_state=initial_state,
                    environment_seed=environment_seed, sampling_seed=sampling_seed, regeneration_attempt=attempt,
                    initial_state_sha256=initial_hash, success=success, environment_steps=environment_steps,
                    queries=queries, applicable_query_count=len(applicable),
                    mean_alignment_cost=float(np.mean(applicable)) if applicable else None,
                    combined_reward=episode_reward(success, applicable, config["lambda_reward"]) if applicable else None,
                    wall_seconds=time.time() - started, sampling=True,
                    rollout_checkpoint=str(config["rollout_checkpoint"]))
    finally:
        environment.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    config = read(run_dir / "config.json")
    manifest = read(run_dir / "iteration-0001.manifest.json")
    snapshot = run_dir / "rollout-policy-iteration-0001.pt"
    config["rollout_checkpoint"] = str(snapshot)
    policy = build_policy(config, snapshot, trainable=False)
    audit = role_audit(policy, "rollout_policy")
    assert audit["trainable_parameter_tensors"] == 0
    write(run_dir / f"rollout-role-audit-worker-{args.worker}.json", audit)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    assignments = [episode for group in manifest["groups"] for episode in group["episodes"]]
    rollout_dir = run_dir / "rollouts" / "iteration-0001"
    error_dir = run_dir / "errors" / "iteration-0001"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    for global_index, assignment in enumerate(assignments):
        if global_index % args.workers != args.worker:
            continue
        destination = rollout_dir / f"episode-{assignment['group_index']:02d}-{assignment['episode_index']:02d}.json"
        if destination.exists():
            continue
        for attempt in range(config["max_episode_regenerations"] + 1):
            result = collect_one(policy, suite, assignment, rollout_dir, config, attempt)
            if result["applicable_query_count"]:
                write(destination, result)
                print(f"accepted group={assignment['group_index']} task={assignment['task_id']} episode={assignment['episode_index']} success={result['success']} reward={result['combined_reward']:.5f}", flush=True)
                break
            error_path = error_dir / f"episode-{assignment['group_index']:02d}-{assignment['episode_index']:02d}-attempt-{attempt:02d}.json"
            write(error_path, result | {"error": "no applicable motion query; excluded and regenerated"})
            for query in result["queries"]:
                sample = Path(query["sample_path"])
                if sample.exists():
                    sample.replace(error_dir / (sample.stem + f"-attempt-{attempt:02d}.pt"))
            print(f"ERROR regenerate group={assignment['group_index']} task={assignment['task_id']} episode={assignment['episode_index']} attempt={attempt}", flush=True)
        else:
            raise RuntimeError(f"unable to collect applicable motion episode after retries: {assignment}")
    audit["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    audit["snapshot_unchanged_during_worker"] = True
    write(run_dir / f"rollout-role-audit-worker-{args.worker}.json", audit)


if __name__ == "__main__":
    main()
