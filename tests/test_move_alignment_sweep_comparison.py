import json

import pytest

from experiments.robot.libero.move_alignment_sweep_comparison import render_html, validate_paired


def _write_run(root, seed_offset=0):
    root.mkdir()
    episodes = []
    queries = []
    for task in range(90):
        for trial in range(2):
            seed = 1000 + task * 2 + trial + seed_offset
            episode_id = f"libero_90-task{task}-episode{trial}-seed{seed}"
            episodes.append(
                {
                    "episode_id": episode_id,
                    "task_id": task,
                    "episode_success": True,
                    "task_instruction": f"task {task}",
                }
            )
            queries.append(
                {
                    "episode_id": episode_id,
                    "task_id": task,
                    "claims": {"motion_axes": {"x": 1}},
                    "state_before": {"ee_position": [0, 0, 0]},
                    "state_after": {"ee_position": [1, 0, 0]},
                }
            )
    (root / "episode_summary.json").write_text(json.dumps(episodes), encoding="utf-8")
    (root / "alignment_queries.jsonl").write_text(
        "".join(json.dumps(query) + "\n" for query in queries), encoding="utf-8"
    )


def test_validate_paired_accepts_identical_90_task_schedule(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_run(first)
    _write_run(second)
    validate_paired({"a": {"root": str(first)}, "b": {"root": str(second)}}, 2)


def test_validate_paired_rejects_different_seed_schedule(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _write_run(first)
    _write_run(second, seed_offset=1)
    with pytest.raises(ValueError, match="schedule does not match"):
        validate_paired({"a": {"root": str(first)}, "b": {"root": str(second)}}, 2)


def test_render_includes_outcome_split_and_wandb_link():
    outcome = {
        "episodes": 1,
        "queries": 2,
        "mismatches": 1,
        "pooled_rate": 0.5,
        "mean_episode_rate": 0.5,
    }
    model = {
        "name": "lambda=0.1",
        "wandb_url": "https://wandb.example/run",
        "successes": 1,
        "episodes": 2,
        "task_success_rate": 0.5,
        "average_move_cosine": 0.75,
        "queries": 4,
        "move_mismatches": 2,
        "move_misalignment_rate": 0.5,
        "outcome_move_misalignment": {"successful": outcome, "failed": outcome},
    }
    rendered = render_html(
        {"base_seed": 123, "model_order": ["model"], "models": {"model": model}}
    )
    assert "Successful-episode average mismatch" in rendered
    assert "Failed-episode average mismatch" in rendered
    assert "https://wandb.example/run" in rendered
