import csv
import json

from experiments.robot.libero.alignment_logging import AlignmentRecorder, merge_rank_outputs


def record(episode_id="ep-0", success=False):
    return {
        "episode_id": episode_id,
        "task_suite": "libero_90",
        "task_id": 0,
        "task_instruction": "move to mug",
        "seed": 7,
        "policy_query_index": 0,
        "reasoning_raw": "MOVE: right",
        "claims": {"parse_warnings": []},
        "action_tokens": [1] * 7,
        "decoded_action_chunk": [[0.0] * 7] * 10,
        "executed_chunk_length": 10,
        "state_before": {},
        "state_after": {},
        "alignment": {
            "kinematic_score": 1.0,
            "target_score": None,
            "subtask_score": None,
            "aggregate_score": 1.0,
            "evaluated_claims": ["kinematic"],
            "diagnostics": {"failure_labels": []},
        },
        "episode_success": success,
    }


def test_jsonl_and_json_csv_summaries(tmp_path):
    recorder = AlignmentRecorder(str(tmp_path))
    recorder.append_query(record())
    recorder.end_episode(True)
    summary = recorder.close()

    query = json.loads((tmp_path / "alignment_queries.jsonl").read_text().strip())
    assert query["episode_success"] is True
    assert summary["kinematic_score"]["count"] == 1
    assert summary["target_score"]["count"] == 0
    assert summary["target_score"]["coverage"] == 0
    assert json.loads((tmp_path / "run_summary.json").read_text())["num_episodes"] == 1
    assert len(json.loads((tmp_path / "episode_summary.json").read_text())) == 1
    with (tmp_path / "run_summary.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 1
    with (tmp_path / "episode_summary.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_episode_without_policy_query_is_counted(tmp_path):
    recorder = AlignmentRecorder(str(tmp_path))
    recorder.end_episode(
        False,
        {"episode_id": "empty-episode", "task_id": 2, "task_instruction": "open microwave"},
    )
    summary = recorder.close()
    assert summary["num_episodes"] == 1
    assert summary["num_policy_queries"] == 0
    assert summary["per_task"]["2"]["num_episodes"] == 1


def test_merge_rank_outputs_writes_global_artifacts(tmp_path):
    for rank, success in enumerate((True, False)):
        recorder = AlignmentRecorder(str(tmp_path / f"rank-{rank}"))
        item = record(episode_id=f"ep-{rank}", success=success)
        item["task_id"] = rank
        item["task_instruction"] = f"task {rank}"
        recorder.append_query(item)
        recorder.end_episode(success)
        recorder.close()

    summary = merge_rank_outputs(str(tmp_path))
    assert summary["num_episodes"] == 2
    assert summary["num_policy_queries"] == 2
    assert summary["task_success_rate"] == 0.5
    assert len((tmp_path / "alignment_queries.jsonl").read_text().splitlines()) == 2
    assert len(json.loads((tmp_path / "episode_summary.json").read_text())) == 2
    assert json.loads((tmp_path / "run_summary.json").read_text())["num_successful_episodes"] == 1
