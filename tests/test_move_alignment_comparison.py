import json

import pytest

from experiments.robot.libero.move_alignment_comparison import move_reward, summarize


def record(axes, before, after):
    return {
        "claims": {"motion_axes": axes},
        "state_before": {"ee_position": before},
        "state_after": {"ee_position": after},
    }


def test_move_reward_uses_raw_measured_displacement():
    value = move_reward(record({"x": -1, "z": -1}, [0, 0, 0], [-2, 0, -2]))
    assert value == pytest.approx(1.0)


def test_move_reward_is_zero_for_stop():
    assert move_reward(record({}, [0, 0, 0], [1, 2, 3])) == 0.0


def test_move_reward_preserves_opposing_direction():
    value = move_reward(record({"z": -1}, [0, 0, 0], [0, 0, 2]))
    assert value == pytest.approx(-1.0)


def test_summarize_reports_mean_episode_mismatch_by_outcome(tmp_path):
    queries = []
    # The successful episode has one mismatch in two queries (50%).
    for index, after in enumerate(([1, 0, 0], [-1, 0, 0])):
        queries.append(
            {
                **record({"x": 1}, [0, 0, 0], after),
                "episode_id": "success",
                "task_id": 0,
            }
        )
    # The failed episode has one mismatch in one query (100%).
    queries.append(
        {
            **record({"x": 1}, [0, 0, 0], [-1, 0, 0]),
            "episode_id": "failure",
            "task_id": 0,
        }
    )
    (tmp_path / "alignment_queries.jsonl").write_text(
        "".join(json.dumps(query) + "\n" for query in queries), encoding="utf-8"
    )
    (tmp_path / "episode_summary.json").write_text(
        json.dumps(
            [
                {"episode_id": "success", "task_id": 0, "episode_success": True, "task_instruction": "task"},
                {"episode_id": "failure", "task_id": 0, "episode_success": False, "task_instruction": "task"},
            ]
        ),
        encoding="utf-8",
    )

    result = summarize(tmp_path, "model")

    assert result["outcome_move_misalignment"]["successful"]["mean_episode_rate"] == pytest.approx(0.5)
    assert result["outcome_move_misalignment"]["failed"]["mean_episode_rate"] == pytest.approx(1.0)
