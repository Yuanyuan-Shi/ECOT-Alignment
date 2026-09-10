import pytest

from experiments.robot.libero.move_only_review_report import _move_evidence, render_section


def record(axes, before, after):
    return {
        "claims": {"motion_axes": axes},
        "state_before": {"ee_position": before},
        "state_after": {"ee_position": after},
    }


def test_move_only_evidence_uses_raw_cosine_threshold():
    evidence = _move_evidence(record({"x": -1, "z": -1}, [0, 0, 0], [-2, 0, -2]))
    assert evidence["score"] == pytest.approx(1.0)
    assert evidence["label"] == "T"


def test_move_only_stop_is_zero_and_false():
    evidence = _move_evidence(record({}, [0, 0, 0], [1, 2, 3]))
    assert evidence["score"] == 0.0
    assert evidence["label"] == "F"


def test_rendered_workspace_has_only_move_label_column():
    payload = {
        "episode_count": 1,
        "query_count": 1,
        "base_seed": 10,
        "summary": [],
        "episodes": [
            {
                "episode_id": "episode",
                "task_id": 0,
                "instruction": "task",
                "seed": 10,
                "success": True,
                "queries": [
                    {
                        "id": "episode::q0",
                        "query": 0,
                        "reasoning_raw": "MOVE: move down",
                        "fields": {"MOVE": "move down"},
                        "claims": {"subtask": "subtask"},
                        "evidence": {
                            "move": [0, 0, -1], "delta_m": [0, 0, -0.1],
                            "move_norm": 1, "delta_norm_m": 0.1, "dot_m": 0.1,
                            "score": 1, "label": "T",
                        },
                        "frames": {"before": "before.png", "intermediate": "middle.png", "after": "after.png"},
                    }
                ],
            }
        ],
    }
    output = render_section(payload)
    assert "Move Alignment" in output
    assert "Subtask Alignment" not in output
    assert "Target Alignment" not in output
