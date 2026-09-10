"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import imageio.v2 as imageio
import numpy as np
import torch
import torch.multiprocessing as mp
import tqdm
from libero.libero import benchmark

try:
    import wandb
except ImportError:
    wandb = None

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.alignment_evaluator import (
    EvaluatorConfig,
    evaluate_alignment,
    parse_reasoning,
    to_jsonable,
)
from experiments.robot.libero.alignment_logging import AlignmentRecorder, merge_rank_outputs
from experiments.robot.libero.libero_utils import (
    capture_alignment_state,
    get_expanded_libero_env,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    project_world_trajectory,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    draw_bboxes,
    draw_gripper,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    make_reasoning_image,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.action_tokenizer import VQActionTokenizer


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    hf_token: str = Path(".hf_token")                       # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    obs_history: int = 1                             # Number of images to pass in from history
    use_wrist_image: bool = False                    # Use wrist images (doubles the number of input images)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite.
    #                                       Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    prefix: str = ''

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "prismatic"                 # Name of W&B project to log to (use default!)
    wandb_entity: Optional[str] = None               # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    use_cot: bool = True                                # Whether to use ECOT

    n_procs_per_gpu: int = 1                         # Number of processes to launch on each GPU
    subset_size: int = 1                             # Split tasks into subset_size chunks
    subset: int = 0                                  # Which chunk to evaluate
    num_open_loop_steps: int = 10                    # Number of actions to execute open-loop
    plans: bool = False
    save_reasoning: bool = False                     # Save reasoning images
    unnorm_key: str = "libero_lm_90"                 # Action un-normalization key

    output_dir: str = "out"                          # Output results file

    enable_alignment_evaluator: bool = False         # Emit reasoning--action alignment records
    alignment_output_dir: str = "experiments/robot/libero/results/alignment"
    motion_dead_band: float = 0.003                  # World-frame EE displacement threshold (meters)
    target_progress_scale: float = 0.05              # Progress corresponding to target score 1 (meters)
    relational_resolution_margin: float = 0.02      # Min extreme-instance separation (meters)
    gripper_open_fraction: float = 0.5               # Normalized joint-range aperture classified as open
    max_tasks: Optional[int] = None                  # Optional task cap, after subset selection
    task_ids: Optional[str] = None                   # Optional comma-separated exact task IDs
    initial_state_offset: int = 0                   # Reserve disjoint collection/test configurations
    episodes_per_task: Optional[int] = None          # Optional override for num_trials_per_task
    max_episode_steps: Optional[int] = None           # Optional post-settle action-step cap (smoke tests)
    alignment_log_text: bool = False                 # Print reasoning/action details for each query
    alignment_cosine_similarity: bool = False        # Analysis-only; signed-axis remains primary
    alignment_save_frames: bool = True               # Save before/middle/after third-person RGB frames

    expansion_half_len_factor: float = 0.0           # Factor by which to expand half lengths of initial state regions
    ood_only: bool = False                           # Whether to only sample states in the expanded part

    min_distractors: int = 0                         # Minimum number of distractors to add to scenes
    max_distractors: int = 0                         # Maximum number of distractors to add to scenes (recommended <=2)

    # fmt: on


def _save_alignment_frames(output_dir, rank, episode_id, query_index, before, middle, after):
    """Save locally portable RGB assets and return paths relative to the run root."""
    asset_dir = Path(output_dir) / f"rank-{rank}" / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{episode_id}-query{query_index:03d}"
    paths = {}
    for label, frame in (("before", before), ("intermediate", middle), ("after", after)):
        path = asset_dir / f"{stem}-{label}.png"
        imageio.imwrite(path, np.asarray(frame, dtype=np.uint8))
        paths[label] = str(Path(f"rank-{rank}") / "assets" / path.name)
    return paths


def _expected_motion_world_segment(claims, start, length=0.08):
    if not claims.motion_axes:
        return None
    vector = np.zeros(3, dtype=float)
    for axis, direction in claims.motion_axes.items():
        vector[{"x": 0, "y": 1, "z": 2}[axis]] = direction
    vector *= length / np.linalg.norm(vector)
    start = np.asarray(start, dtype=float)
    return np.stack((start, start + vector))


def _write_experiment_metadata(cfg, wall_clock_seconds):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    checkpoint = Path(cfg.pretrained_checkpoint).resolve()
    metadata = {
        "experiment_label": "Initial LIBERO-90 qualitative audit: one trial per task",
        "repository_commit": commit,
        "checkpoint_path": str(checkpoint),
        "base_seed": cfg.seed,
        "seed_schedule": "base_seed + task_index (for exactly one trial per task)",
        "reasoning_generation_enabled": cfg.use_cot,
        "wall_clock_seconds": wall_clock_seconds,
        "configuration": json.loads(json.dumps(to_jsonable(asdict(cfg)), default=str)),
        "observation": "fixed third-person agentview RGB",
        "action_representation": "7 VQ-VAE tokens decoded to a 10-step, 7D action chunk",
    }
    root = Path(cfg.alignment_output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _extract_explicit_reasoning(decoded_text: str) -> str:
    """Remove prompt and action tokens while retaining the tagged generated trace."""
    starts = [decoded_text.find(tag) for tag in ("PLAN:", "SUBTASK:", "MOVE:") if tag in decoded_text]
    if not starts:
        raise RuntimeError(
            "The selected checkpoint emitted actions without tagged reasoning. Alignment evaluation requires the full "
            "ECoT LIBERO checkpoint, or a reasoning-dropout checkpoint generated from the reasoning prefix (PLAN:), "
            "not ACTION:."
        )
    reasoning = decoded_text[min(starts) :]
    action_index = reasoning.find("ACTION:")
    if action_index >= 0:
        reasoning = reasoning[:action_index]
    reasoning = reasoning.strip(" \t\r\n;")
    if not re.search(r":\s*[^;\s]", reasoning):
        raise RuntimeError(
            "The checkpoint produced only a forced reasoning tag followed by actions, with no explicit reasoning "
            "content. This checkpoint is not evaluable for reasoning--action alignment."
        )
    claims = parse_reasoning(reasoning)
    if "no_recognized_reasoning_fields" in claims.parse_warnings:
        raise RuntimeError("Generated text did not contain evaluable ECoT reasoning fields before ACTION:.")
    return reasoning


def _eval_libero_main(cfg: GenerateConfig) -> None:
    if not hasattr(cfg, "rank"):
        cfg.rank = 0
    if not hasattr(cfg, "world_size"):
        cfg.world_size = 1

    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Load model (each process loads its own model)
    model = get_model(cfg)
    model.enable_cot(cfg.use_cot)

    if cfg.enable_alignment_evaluator:
        if cfg.model_family != "prismatic" or not isinstance(
            getattr(model, "action_tokenizer", None), VQActionTokenizer
        ):
            raise ValueError(
                "Alignment evaluation requires the prismatic MiniVLA checkpoint with a VQActionTokenizer; "
                "action-only OpenVLA / non-VQ checkpoints are not evaluable."
            )
        if not cfg.use_cot:
            raise ValueError(
                "Alignment evaluation requires explicit inference-time reasoning. Set --use_cot True and use the "
                "full ECoT LIBERO checkpoint, or a reasoning-dropout checkpoint queried from its reasoning prefix."
            )

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family in ["openvla", "prismatic"]:
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging — make per-rank log file so multiple processes don't overwrite
    run_id = f"{cfg.prefix}EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    run_id = f"{run_id}-r{cfg.rank}"  # append rank to keep logs separate
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"[rank {cfg.rank}] Logging to local log file: {local_log_filepath}")
    alignment_recorder = None
    alignment_config = None
    if cfg.enable_alignment_evaluator:
        rank_output_dir = os.path.join(cfg.alignment_output_dir, f"rank-{cfg.rank}")
        alignment_recorder = AlignmentRecorder(rank_output_dir)
        alignment_config = EvaluatorConfig(
            motion_deadband=cfg.motion_dead_band,
            target_progress_scale=cfg.target_progress_scale,
            relational_resolution_margin=cfg.relational_resolution_margin,
            include_cosine_similarity=cfg.alignment_cosine_similarity,
        )
        print(f"[rank {cfg.rank}] Alignment records: {alignment_recorder.records_path}")

    # Initialize Weights & Biases logging as well (each worker gets its own run name to avoid collisions)
    if cfg.use_wandb and wandb is None:
        raise RuntimeError("Weights & Biases logging was requested, but wandb could not be imported")
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"[rank {cfg.rank}] Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    if cfg.task_ids:
        task_ids = [int(value.strip()) for value in cfg.task_ids.split(",") if value.strip()]
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("--task_ids must contain unique comma-separated integers")
        invalid = [task_id for task_id in task_ids if not 0 <= task_id < num_tasks_in_suite]
        if invalid:
            raise ValueError(f"Task IDs outside [0, {num_tasks_in_suite}): {invalid}")
        print(f"[rank {cfg.rank}] Explicit task IDs: {task_ids}")
        log_file.write(f"Explicit task IDs: {task_ids}\n")
    else:
        task_ids = list(range(num_tasks_in_suite))
    if cfg.task_ids is None and cfg.subset_size is not None:
        chunk_size = len(task_ids) // cfg.subset_size
        task_ids = task_ids[cfg.subset * chunk_size : (cfg.subset + 1) * chunk_size]
        print(f"[rank {cfg.rank}] Sampled task IDs: {task_ids}")
        log_file.write(f"Sampled task IDs: {task_ids}\n")
    if cfg.max_tasks is not None:
        task_ids = task_ids[: cfg.max_tasks]

    my_task_ids = task_ids[cfg.rank :: cfg.world_size]

    print(f"[rank {cfg.rank}] → evaluating tasks: {my_task_ids}")
    for task_id in tqdm.tqdm(my_task_ids, desc=f"Rank {cfg.rank} tasks"):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        if cfg.expansion_half_len_factor > 0 or cfg.min_distractors > 0:
            print(f"Expansion Half Len Factor: {cfg.expansion_half_len_factor}")
            print(f"OOD Positions Only? {cfg.ood_only}")
            print(f"Num Distractors Added? Between {cfg.min_distractors} and {cfg.max_distractors}")

            # Wait to create the environment at the beginning of every episode if we are adding distractors.
            # This is so that distractors get randomized for each episode.
            if cfg.min_distractors == 0:
                env, task_description = get_expanded_libero_env(
                    task,
                    expansion_half_len_factor=cfg.expansion_half_len_factor,
                    ood_only=cfg.ood_only,
                    min_distractors=cfg.min_distractors,
                    max_distractors=cfg.max_distractors,
                    seed=cfg.seed,
                    distractor_seed=cfg.seed,
                    resolution=resize_size,
                )
        else:
            env, task_description = get_libero_env(task, cfg.model_family, resolution=resize_size)

        # Start episodes
        task_episodes, task_successes = 0, 0
        episode_count = cfg.episodes_per_task if cfg.episodes_per_task is not None else cfg.num_trials_per_task
        if cfg.initial_state_offset < 0 or cfg.initial_state_offset + episode_count > len(initial_states):
            raise ValueError("Requested initial-state indices are unavailable")
        for episode_idx in tqdm.tqdm(range(episode_count)):
            episode_seed = cfg.seed + task_id * max(1, episode_count) + episode_idx

            if cfg.min_distractors > 0:
                env, task_description = get_expanded_libero_env(
                    task,
                    expansion_half_len_factor=cfg.expansion_half_len_factor,
                    ood_only=cfg.ood_only,
                    min_distractors=cfg.min_distractors,
                    max_distractors=cfg.max_distractors,
                    seed=int(1e5) * cfg.seed + task_id * cfg.num_trials_per_task + episode_idx,
                    distractor_seed=int(1e5) * cfg.seed + task_id * cfg.num_trials_per_task + episode_idx,
                    resolution=resize_size,
                )

            print(f"\n[rank {cfg.rank}] Task: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            episode_start_time = time.perf_counter()
            env.seed(episode_seed)
            env.reset()

            # Set initial states
            if not (cfg.expansion_half_len_factor > 0 or cfg.min_distractors > 0):
                obs = env.set_init_state(initial_states[cfg.initial_state_offset + episode_idx])

            action_queue = deque(maxlen=cfg.num_open_loop_steps)
            reasoning = ""
            pending_alignment = None
            policy_query_index = 0
            episode_id = f"{cfg.task_suite_name}-task{task_id}-episode{episode_idx}-seed{episode_seed}"
            # Setup
            t = 0
            replay_images = []
            replay_images_reason = []
            replay_wrist_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            elif cfg.task_suite_name == "libero_generalize":
                max_steps = 400  # from libero_90
            if cfg.max_episode_steps is not None:
                max_steps = min(max_steps, cfg.max_episode_steps)

            print(f"[rank {cfg.rank}] Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            done = False
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, _reward, done, _info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # use_wrist_image
                    if cfg.use_wrist_image:
                        wrist_img = get_libero_image(obs, resize_size, key="robot0_eye_in_hand_image")
                        replay_wrist_images.append(wrist_img)

                    # buffering #obs_history images, optionally
                    image_history = replay_images[-cfg.obs_history :]
                    if len(image_history) < cfg.obs_history:
                        image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))

                    # same but for optional wrist images
                    if cfg.use_wrist_image:
                        wrist_image_history = replay_wrist_images[-cfg.obs_history :]
                        if len(wrist_image_history) < cfg.obs_history:
                            wrist_image_history.extend(
                                [replay_wrist_images[-1]] * (cfg.obs_history - len(wrist_image_history))
                            )
                        # interleaved images [... image_t, wrist_t ...]
                        image_history = [val for tup in zip(image_history, wrist_image_history) for val in tup]

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": image_history,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    if len(action_queue) == 0:
                        generation_info = {} if cfg.enable_alignment_evaluator else None
                        state_before = (
                            capture_alignment_state(env, obs, cfg.gripper_open_fraction)
                            if cfg.enable_alignment_evaluator
                            else None
                        )
                        generation_start_time = time.perf_counter()
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            if cfg.plans:
                                actions, reasoning = get_action(
                                    cfg,
                                    model,
                                    observation,
                                    task_description,
                                    processor=processor,
                                    task_id=task_id,
                                    info_dict=generation_info,
                                )
                            else:
                                actions, reasoning = get_action(
                                    cfg,
                                    model,
                                    observation,
                                    task_description,
                                    processor=processor,
                                    info_dict=generation_info,
                                )
                        generation_seconds = time.perf_counter() - generation_start_time
                        if cfg.model_family == "prismatic" and isinstance(model.action_tokenizer, VQActionTokenizer):
                            decoded_action_chunk = np.asarray(actions[0], dtype=float).copy()
                            actions = [decoded_action_chunk[i].copy() for i in range(decoded_action_chunk.shape[0])]
                        else:
                            decoded_action_chunk = np.asarray(actions, dtype=float).copy()
                            actions = [actions]

                        if cfg.enable_alignment_evaluator:
                            explicit_reasoning = _extract_explicit_reasoning(reasoning)
                            action_tokens = generation_info.get("action_token_ids")
                            if action_tokens is None or len(action_tokens) != 7:
                                raise RuntimeError(
                                    "MiniVLA inference did not expose exactly seven VQ action tokens for the current "
                                    "query."
                                )
                            pending_alignment = {
                                "reasoning_raw": explicit_reasoning,
                                "action_tokens": action_tokens,
                                "decoded_action_chunk": decoded_action_chunk.copy(),
                                "state_before": state_before,
                                "executed_chunk_length": 0,
                                "policy_query_index": policy_query_index,
                                "frame_before": img.copy(),
                                "frame_intermediate": None,
                                "executed_ee_trajectory": [state_before.ee_position.copy()],
                                "reasoning_generation_seconds": generation_seconds,
                                "simulator_execution_seconds": 0.0,
                            }
                            policy_query_index += 1
                            if cfg.alignment_log_text:
                                print(f"[alignment query {policy_query_index - 1}] {explicit_reasoning}")
                                print(f"[alignment action tokens] {action_tokens}")
                        action_queue.extend(actions[: cfg.num_open_loop_steps])

                    action = action_queue.popleft()

                    if cfg.save_reasoning:
                        img = get_libero_image(obs, resize_size=(640, 480))
                        reasoning_img, metadata = make_reasoning_image(reasoning)
                        draw_gripper(img, metadata["gripper"])
                        draw_bboxes(img, metadata["bboxes"])
                        reason_img = np.concatenate([img, reasoning_img], axis=1)

                        replay_images_reason.append(reason_img)

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family in ["openvla", "prismatic"]:
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    simulator_start_time = time.perf_counter()
                    obs, _, done, _ = env.step(action.tolist())
                    if pending_alignment is not None:
                        pending_alignment["simulator_execution_seconds"] += time.perf_counter() - simulator_start_time
                        pending_alignment["executed_chunk_length"] += 1
                        pending_alignment["executed_ee_trajectory"].append(np.asarray(obs["robot0_eef_pos"]).copy())
                        if pending_alignment["frame_intermediate"] is None and pending_alignment[
                            "executed_chunk_length"
                        ] >= max(1, min(5, len(pending_alignment["decoded_action_chunk"]) // 2)):
                            pending_alignment["frame_intermediate"] = get_libero_image(obs, resize_size)
                    t += 1

                    if pending_alignment is not None and (done or len(action_queue) == 0):
                        logging_start_time = time.perf_counter()
                        state_after = capture_alignment_state(env, obs, cfg.gripper_open_fraction)
                        claims = parse_reasoning(pending_alignment["reasoning_raw"])
                        alignment = evaluate_alignment(
                            claims,
                            pending_alignment["decoded_action_chunk"][: pending_alignment["executed_chunk_length"]],
                            pending_alignment["state_before"],
                            state_after,
                            config=alignment_config,
                        )
                        frame_after = get_libero_image(obs, resize_size)
                        frame_intermediate = pending_alignment["frame_intermediate"]
                        if frame_intermediate is None:
                            frame_intermediate = frame_after
                        frame_paths = None
                        if cfg.alignment_save_frames:
                            frame_paths = _save_alignment_frames(
                                cfg.alignment_output_dir,
                                cfg.rank,
                                episode_id,
                                pending_alignment["policy_query_index"],
                                pending_alignment["frame_before"],
                                frame_intermediate,
                                frame_after,
                            )
                        projected_trajectory = project_world_trajectory(
                            env, pending_alignment["executed_ee_trajectory"], frame_after.shape
                        )
                        expected_world = _expected_motion_world_segment(
                            claims, pending_alignment["executed_ee_trajectory"][0]
                        )
                        expected_pixels = (
                            project_world_trajectory(env, expected_world, frame_after.shape)
                            if expected_world is not None
                            else None
                        )
                        visualization_logging_seconds = time.perf_counter() - logging_start_time
                        alignment_recorder.append_query(
                            {
                                "episode_id": episode_id,
                                "task_suite": cfg.task_suite_name,
                                "task_id": task_id,
                                "task_instruction": task_description,
                                "seed": episode_seed,
                                "base_seed": cfg.seed,
                                "policy_query_index": pending_alignment["policy_query_index"],
                                "reasoning_raw": pending_alignment["reasoning_raw"],
                                "claims": to_jsonable(claims),
                                "action_tokens": pending_alignment["action_tokens"],
                                "decoded_action_chunk": pending_alignment["decoded_action_chunk"],
                                "executed_chunk_length": pending_alignment["executed_chunk_length"],
                                "executed_ee_trajectory": pending_alignment["executed_ee_trajectory"],
                                "executed_ee_trajectory_pixels": projected_trajectory,
                                "expected_motion_pixels": expected_pixels,
                                "frames": frame_paths,
                                "timing": {
                                    "reasoning_generation_seconds": pending_alignment[
                                        "reasoning_generation_seconds"
                                    ],
                                    "simulator_execution_seconds": pending_alignment[
                                        "simulator_execution_seconds"
                                    ],
                                    "visualization_logging_seconds": visualization_logging_seconds,
                                    "query_wall_seconds": pending_alignment["reasoning_generation_seconds"]
                                    + pending_alignment["simulator_execution_seconds"]
                                    + visualization_logging_seconds,
                                },
                                "state_before": to_jsonable(pending_alignment["state_before"]),
                                "state_after": to_jsonable(state_after),
                                "alignment": to_jsonable(alignment),
                                "episode_success": False,
                            }
                        )
                        pending_alignment = None
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    if cfg.enable_alignment_evaluator:
                        raise
                    break

            # A max-step or exception can truncate an in-flight chunk.
            if pending_alignment is not None:
                logging_start_time = time.perf_counter()
                state_after = capture_alignment_state(env, obs, cfg.gripper_open_fraction)
                claims = parse_reasoning(pending_alignment["reasoning_raw"])
                alignment = evaluate_alignment(
                    claims,
                    pending_alignment["decoded_action_chunk"][: pending_alignment["executed_chunk_length"]],
                    pending_alignment["state_before"],
                    state_after,
                    config=alignment_config,
                )
                frame_after = get_libero_image(obs, resize_size)
                frame_intermediate = pending_alignment["frame_intermediate"]
                if frame_intermediate is None:
                    frame_intermediate = frame_after
                frame_paths = None
                if cfg.alignment_save_frames:
                    frame_paths = _save_alignment_frames(
                        cfg.alignment_output_dir,
                        cfg.rank,
                        episode_id,
                        pending_alignment["policy_query_index"],
                        pending_alignment["frame_before"],
                        frame_intermediate,
                        frame_after,
                    )
                projected_trajectory = project_world_trajectory(
                    env, pending_alignment["executed_ee_trajectory"], frame_after.shape
                )
                expected_world = _expected_motion_world_segment(
                    claims, pending_alignment["executed_ee_trajectory"][0]
                )
                expected_pixels = (
                    project_world_trajectory(env, expected_world, frame_after.shape)
                    if expected_world is not None
                    else None
                )
                visualization_logging_seconds = time.perf_counter() - logging_start_time
                alignment_recorder.append_query(
                    {
                        "episode_id": episode_id,
                        "task_suite": cfg.task_suite_name,
                        "task_id": task_id,
                        "task_instruction": task_description,
                        "seed": episode_seed,
                        "base_seed": cfg.seed,
                        "policy_query_index": pending_alignment["policy_query_index"],
                        "reasoning_raw": pending_alignment["reasoning_raw"],
                        "claims": to_jsonable(claims),
                        "action_tokens": pending_alignment["action_tokens"],
                        "decoded_action_chunk": pending_alignment["decoded_action_chunk"],
                        "executed_chunk_length": pending_alignment["executed_chunk_length"],
                        "executed_ee_trajectory": pending_alignment["executed_ee_trajectory"],
                        "executed_ee_trajectory_pixels": projected_trajectory,
                        "expected_motion_pixels": expected_pixels,
                        "frames": frame_paths,
                        "timing": {
                            "reasoning_generation_seconds": pending_alignment["reasoning_generation_seconds"],
                            "simulator_execution_seconds": pending_alignment["simulator_execution_seconds"],
                            "visualization_logging_seconds": visualization_logging_seconds,
                            "query_wall_seconds": pending_alignment["reasoning_generation_seconds"]
                            + pending_alignment["simulator_execution_seconds"]
                            + visualization_logging_seconds,
                        },
                        "state_before": to_jsonable(pending_alignment["state_before"]),
                        "state_after": to_jsonable(state_after),
                        "alignment": to_jsonable(alignment),
                        "episode_success": False,
                    }
                )
                pending_alignment = None

            if alignment_recorder is not None:
                alignment_recorder.end_episode(
                    done,
                    {
                        "episode_id": episode_id,
                        "task_id": task_id,
                        "task_instruction": task_description,
                        "seed": episode_seed,
                        "base_seed": cfg.seed,
                        "initial_state_index": cfg.initial_state_offset + episode_idx,
                        "episode_wall_seconds": time.perf_counter() - episode_start_time,
                        "peak_gpu_memory_bytes": (
                            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
                        ),
                    },
                )

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images_reason if cfg.save_reasoning else replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
                output_dir=f"experiments/robot/libero/results/{cfg.output_dir}",
            )

            # Save the videos to wandb
            if cfg.use_wandb and (task_successes < 10 or task_episodes - task_successes < 10):
                group = "success" if done else "failure"
                idx = task_successes if done else task_episodes - task_successes
                wandb.log(
                    {f"{task_description}/{group}/{idx}": wandb.Video(np.array(replay_images).transpose(0, 3, 1, 2))}
                )

            # Log current results
            print(f"[rank {cfg.rank}] Success: {done}")
            print(f"[rank {cfg.rank}] # episodes completed so far: {total_episodes}")
            print(f"[rank {cfg.rank}] # successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"[rank {cfg.rank}] Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"[rank {cfg.rank}] Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

        results_path = os.path.join(f"experiments/robot/libero/results/{cfg.output_dir}", cfg.output_dir + ".jsonl")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        result = {
            "task_id": task_id,
            "task_description": task_description,
            "task_successes": task_successes,
            "task_episodes": task_episodes,
            "task_success_rate": float(task_successes) / float(task_episodes),
        }
        with open(results_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    if alignment_recorder is not None:
        alignment_summary = alignment_recorder.close()
        print(
            f"[rank {cfg.rank}] Alignment summary: "
            f"{alignment_summary['num_policy_queries']} queries, "
            f"aggregate count={alignment_summary['aggregate_score']['count']}"
        )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


def _mp_entry(rank, cfg, nprocs):
    cfg.rank = int(rank)
    cfg.world_size = int(nprocs)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    _eval_libero_main(cfg)


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    run_start_time = time.perf_counter()
    # Multi-process on single GPU
    if cfg.n_procs_per_gpu is not None and int(cfg.n_procs_per_gpu) > 1:
        nprocs = int(cfg.n_procs_per_gpu)
        mp.spawn(_mp_entry, args=(cfg, nprocs), nprocs=nprocs, join=True)
        if cfg.enable_alignment_evaluator:
            summary = merge_rank_outputs(cfg.alignment_output_dir)
            print(
                f"Merged {nprocs} alignment ranks: {summary['num_episodes']} episodes, "
                f"{summary['num_policy_queries']} queries"
            )
            _write_experiment_metadata(cfg, time.perf_counter() - run_start_time)
        return

    # Single-process
    cfg.rank = 0
    cfg.world_size = 1
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    _eval_libero_main(cfg)
    if cfg.enable_alignment_evaluator:
        merge_rank_outputs(cfg.alignment_output_dir)
        _write_experiment_metadata(cfg, time.perf_counter() - run_start_time)


if __name__ == "__main__":
    eval_libero()
