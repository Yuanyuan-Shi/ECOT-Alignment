import json

import numpy as np
from PIL import Image

from experiments.robot.libero.alignment_report import (
    EXPERIMENT_LABEL,
    _format_reasoning_for_figure,
    _incorrect_reasoning_intro_example,
    _intro_example,
    _rescore_record_kinematics,
    _visible_object_boxes,
    cluster_bootstrap_difference,
    compute_statistics,
    generate_report,
    is_mismatch,
    mismatch_overlap,
    select_examples,
    validate_local_assets,
)


def query(episode, task, success, *, k=1.0, target=0.2, subtask=1.0, index=0, warning=False):
    labels = []
    if k is not None and k < 1:
        labels.append("kinematic_no_motion")
    if target is not None and target < 0:
        labels.append("target_moved_away")
    if subtask is not None and subtask < 1:
        labels.append("subtask_not_completed")
    return {
        "episode_id": episode,
        "episode_success": success,
        "task_id": task,
        "task_instruction": f"task {task}",
        "seed": 100 + task,
        "policy_query_index": index,
        "reasoning_raw": "SUBTASK: move to the mug MOVE: move right",
        "claims": {
            "subtask": "move to the mug",
            "target_object": "mug",
            "motion_axes": {"y": 1},
            "parse_warnings": ["fixture_warning"] if warning else [],
        },
        "alignment": {
            "kinematic_score": k,
            "target_score": target,
            "subtask_score": subtask,
            "aggregate_score": None,
            "diagnostics": {
                "failure_labels": labels,
                "target_resolution": {"before": {"status": "resolved"}},
            },
        },
        "decoded_action_chunk": [[0] * 7] * 10,
        "executed_chunk_length": 10,
        "executed_ee_trajectory_pixels": [[5, 5], [10, 10]],
        "expected_motion_pixels": [[5, 5], [20, 5]],
        "frames": {
            "before": "rank-0/assets/b.png",
            "intermediate": "rank-0/assets/m.png",
            "after": "rank-0/assets/a.png",
        },
        "timing": {
            "reasoning_generation_seconds": 1,
            "simulator_execution_seconds": 2,
            "visualization_logging_seconds": 0.1,
        },
    }


def test_offline_kinematic_rescore_uses_weighted_set_denominator():
    record = query("e", 0, True)
    record["claims"]["motion_axes"] = {"y": 1, "z": 1}
    record["state_before"] = {"ee_position": [0, 0, 0]}
    record["state_after"] = {"ee_position": [0.01, 0.01, 0.01]}
    _rescore_record_kinematics(record)
    diagnostic = record["alignment"]["diagnostics"]["kinematic"]
    assert record["alignment"]["kinematic_score"] == 1 / 3
    assert diagnostic["counts"] == {"match": 2, "missing": 0, "opposite": 0, "extra": 1}
    assert diagnostic["contributions"] == {"match": 2, "missing": 0, "opposite": 0, "extra": -1}
    assert diagnostic["numerator"] == 1
    assert diagnostic["denominator"] == 3


def fixture_records():
    return [
        query("s0", 0, True, k=1, target=0.2, subtask=1),
        query("s1", 1, True, k=0, target=-0.1, subtask=0),
        query("f0", 2, False, k=-1, target=-0.2, subtask=0),
        query("f1", 3, False, k=1, target=0.1, subtask=1),
    ]


def test_intro_figure_format_keeps_complete_reasoning_sections():
    reasoning = (
        "PLAN: move to the mug VISIBLE OBJECTS: mug 1 [1, 2, 3, 4] "
        "SUBTASK REASONING: the mug is the target SUBTASK: move to the mug "
        "MOVE REASONING: align with it MOVE: move right GRIPPER POSITION: [2, 3]"
    )
    formatted = _format_reasoning_for_figure(reasoning, width=60)
    assert "VISIBLE OBJECTS:" in formatted
    assert "SUBTASK REASONING:" in formatted
    assert "MOVE REASONING:" in formatted
    assert "GRIPPER POSITION:" in formatted
    assert "VISIBLE\nOBJECTS" not in formatted


def test_visible_object_boxes_convert_released_yx_coordinate_order():
    boxes = _visible_object_boxes(
        "VISIBLE OBJECTS: chocolate pudding 1 [83, 38, 97, 57] SUBTASK: grasp the chocolate pudding"
    )
    assert boxes["chocolate pudding 1"] == [38.0, 83.0, 57.0, 97.0]


def test_intro_example_is_the_manually_reviewed_correct_reasoning_wrong_action_case():
    record = query("candidate", 73, True, k=1, target=0, subtask=0, index=13)
    record["alignment"]["diagnostics"]["failure_labels"] = [
        "subtask_not_completed",
    ]
    record["alignment"]["diagnostics"]["target"] = {"resolved_object": "black_book_1"}
    record["state_after"] = {
        "gripper_open": False,
        "objects": {
            "black_book_1": {
                "grasped": True,
                "contained_by": ["desk_caddy_1_front_contain_region"],
            }
        },
    }
    assert _intro_example([record]) is record


def test_incorrect_reasoning_intro_example_is_task_correct_bowl_action():
    record = query("bowl", 64, True, k=1, target=0.01, subtask=0.1, index=5)
    record["alignment"]["diagnostics"]["target"] = {
        "resolved_object": "akita_black_bowl_1"
    }
    record["alignment"]["diagnostics"]["wrong_object_interaction"] = [
        "akita_black_bowl_2"
    ]
    record["state_after"] = {
        "objects": {"akita_black_bowl_2": {"grasped": True}}
    }
    assert _incorrect_reasoning_intro_example([record]) is record


def test_missing_scores_are_excluded_from_denominators():
    records = [query("a", 0, True, k=None), query("b", 1, False, k=0)]
    stats = compute_statistics(records)["levels"]["kinematic"]
    assert stats["total_queries"] == 2
    assert stats["evaluable_queries"] == 1
    assert stats["mismatch_count"] == 1
    assert stats["query_mismatch_rate"] == 1


def test_query_and_episode_denominators_and_success_grouping():
    records = fixture_records()
    stats = compute_statistics(records)["levels"]["kinematic"]
    assert stats["mismatch_count"] == 2
    assert stats["evaluable_queries"] == 4
    assert stats["episodes_with_mismatch"] == 2
    assert stats["episodes_with_evaluable_claims"] == 4
    interval = cluster_bootstrap_difference(records, "kinematic", "query_mismatch_rate", replicates=50)
    assert interval is not None


def test_single_kinematic_threshold_and_no_extra_category():
    assert is_mismatch(query("a", 0, True, k=0.999), "kinematic")
    assert not is_mismatch(query("a", 0, True, k=1.0), "kinematic")
    levels = compute_statistics(fixture_records())["levels"]
    assert set(levels) == {"kinematic", "target", "subtask", "any"}


def test_multilabel_overlap_uses_three_primary_levels():
    overlap = mismatch_overlap([query("a", 0, True, k=0, target=-0.1, subtask=0)])
    assert overlap == {"Kinematic + Target + Subtask": 1}


def test_bootstrap_is_episode_clustered_and_reproducible():
    records = fixture_records()
    first = cluster_bootstrap_difference(records, "target", "episode_mismatch_prevalence", replicates=100, seed=7)
    second = cluster_bootstrap_difference(records, "target", "episode_mismatch_prevalence", replicates=100, seed=7)
    assert first == second


def test_example_selection_is_deterministic():
    records = fixture_records()
    assert [r["episode_id"] for r in select_examples(records)["figure1"]] == [
        r["episode_id"] for r in select_examples(list(reversed(records)))["figure1"]
    ]


def test_html_generation_from_fixture_and_local_asset_links(tmp_path):
    root = tmp_path / "run"
    assets = root / "rank-0" / "assets"
    assets.mkdir(parents=True)
    for name in ("a.png", "b.png", "m.png"):
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(assets / name)
    records = fixture_records()
    (root / "alignment_queries.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records))
    episodes = [
        {
            "episode_id": item["episode_id"],
            "task_id": item["task_id"],
            "task_instruction": item["task_instruction"],
            "episode_success": item["episode_success"],
            "episode_wall_seconds": 3.5,
            "peak_gpu_memory_bytes": 2**30,
        }
        for item in records
    ]
    (root / "episode_summary.json").write_text(json.dumps(episodes))
    (root / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_path": "/checkpoint.pt",
                "repository_commit": "abc",
                "base_seed": 100,
                "seed_schedule": "base + task",
                "reasoning_generation_enabled": True,
                "wall_clock_seconds": 10,
                "observation": "fixed third person",
                "action_representation": "7 tokens / 10x7",
                "configuration": {"n_procs_per_gpu": 1},
            }
        )
    )
    index = generate_report(root, tmp_path / "report", bootstrap_replicates=20)
    text = index.read_text()
    assert EXPERIMENT_LABEL in text
    assert "90 tasks × 1 trial per task" in text  # noqa: RUF001
    assert "Kinematic—strict" not in text
    assert "Kinematic—weak" not in text
    assert "Query taxonomy and human review" in text
    assert "Episode case timeline and proposed turning points" not in text
    assert "Approve proposal (A)" not in text
    assert 'id="qr-card"' not in text
    assert "query_review_ui.js" in text
    assert "Manual review complete for the nine retained scenarios" in text
    assert "Human-labeled query-level results" in text
    assert "Agent-proposed representative proportions" not in text
    assert "Action-judgment sanity check" not in text
    assert "Compact step-by-step review — 9 retained episodes" in text
    assert 'id="qr-compact-episodes"' in text
    assert ".compact-scene" in text
    for component in ("task", "plan", "subtask", "move", "gripper", "objects"):
        assert f".component-{component}" in text
    assert "Actual action (Σ executed chunk)" in (tmp_path / "report" / "query_review_ui.js").read_text()
    review_js = (tmp_path / "report" / "query_review_ui.js").read_text()
    assert "measured translation ΔEE" in review_js
    assert "commanded rotation vector Δr" in review_js
    assert "convention: direction threshold ±3 mm" in review_js
    assert "convention: per-component direction threshold ±0.3 radians" in review_js
    assert "convention: CLOSE 0, OPEN +1" in review_js
    assert '{axis: "x", name: "roll"}' in review_js
    assert '{axis: "y", name: "pitch"}' in review_js
    assert '{axis: "z", name: "yaw"}' in review_js
    assert "right-hand rule" in review_js
    assert "no commanded rotation" in review_js
    assert "open gripper" in review_js
    assert "mixed gripper commands" in review_js
    assert "action-badge" not in review_js
    assert "Move Alignment" in review_js
    assert "Target Alignment" not in review_js
    assert "Subtask Alignment" in review_js
    assert "Overall alignment:" not in review_js
    assert "function overallAlignment" in review_js
    assert "function sceneSpatialOverlay" in review_js
    assert "world (0,0,0)" in review_js
    assert "EE start" in review_js
    assert "EE end" in review_js
    assert "function revisedTargetMetric" in review_js
    assert "detail: `score ${Number(score).toFixed(2)}" in review_js
    assert "Target: ${position(evidence.target_before)}" in review_js
    assert "EE Before:" in review_js
    assert "d_before: ||EE Before − Target||" in review_js
    assert "EE After:" in review_js
    assert "d_after: ||EE After − Target||" in review_js
    assert "target after" not in review_js.lower()
    assert "moved closer to" in review_js
    assert "moved farther from" in review_js
    assert 'values.includes("F")' in review_js
    assert "Move Reasoning" not in review_js
    assert '<label>Move<select data-field="action">' not in review_js
    assert '<label>Move Alignment<select data-field="kinematic_alignment">' in review_js
    assert "Target Reasoning" not in review_js
    assert "Target Action" not in review_js
    assert "Subtask Reasoning" not in review_js
    assert "Subtask Action" not in review_js
    assert 'data-field="target_reasoning"' not in review_js
    assert 'data-field="subtask_action"' not in review_js
    assert 'data-case-domain="move"' not in review_js
    assert 'data-case-domain="target"' not in review_js
    assert 'data-case-domain="subtask"' not in review_js
    assert "Case color definitions" not in review_js
    assert "Initial suggestion:" in review_js
    assert "record.initial_labels || {}" in review_js
    assert "review.subtask_alignment || suggested.subtask_alignment" in review_js
    assert "subtaskReward > 0 ? \"T\" : \"F\"" in review_js
    assert "savedMoveAlignment || suggested.kinematic_alignment" in review_js
    assert "function moveCosine" in review_js
    assert "function parsedMoveVector" in review_js
    assert "const expected = parsedMoveVector(record)" in review_js
    assert "const actual = delta.map(Number)" in review_js
    assert "if (!expectedNorm) return {score: 0" in review_js
    assert 'cos convention = 0.000 because MOVE is [0,0,0]' in review_js
    assert "dot / (expectedNorm * actualNorm)" in review_js
    assert "Σ 10-step action ΔEE" in review_js
    assert "3 mm action dead-band" not in review_js
    assert "dot = ${dotTerms} = ${dotMillimetres.toFixed(2)}" in review_js
    assert "||MOVE|| = ${result.expectedNorm.toFixed(3)}; ||ΔEE||" in review_js
    assert "${dotMillimetres.toFixed(2)} / (${result.expectedNorm.toFixed(3)} × ${actualNormMillimetres.toFixed(2)})" in review_js
    assert "score >= 0.5" in review_js
    assert "≥ 0.5 aligned; < 0.5 not aligned" in review_js
    assert 'TASK_0_3_9_BACKUP_KEY = "ecotTask0Task3Task9LabelsBackupV48"' in review_js
    assert "task_ids: [0, 3, 9]" in review_js
    assert "subtask_checkpoint" in review_js
    assert '[["", "—"], ["T", "T"], ["F", "F"]]' in review_js
    assert "function buildSubtaskReview" in review_js
    assert "function subtaskCheckpoint" in review_js
    assert 'fields["SUBTASK REASONING"]' in review_js
    assert 'data-subtask-judgment' in review_js
    assert 'distributedScore / period.records.length' in review_js
    assert 'finalCompletionUnits' in review_js
    assert 'return steps.length - index' in review_js
    assert 'periods.slice(index + 1).some(later => later.key === period.key)' in review_js
    assert 'this subtask returns later' in review_js
    assert 'SUBTASK_CHECKPOINT_RULE_VERSION = 3' in review_js
    assert 'judgment === "complete" ? Math.max(1' in review_js
    assert 'completed_units: completedUnits' in review_js
    assert 'const displayRecord = period.records[period.records.length - 1]' in review_js
    assert 'At query ${record.query_index}, label changes' in review_js
    assert 'sceneRecord: period.records[period.records.length - 1]' in review_js
    assert 'scene is after query ${sceneRecord.query_index}' in review_js
    assert 'terminalBonus' in review_js
    assert 'completedUnits > 1 && period.records.length > 1' in review_js
    assert 'Previous query ${checkpoint.previousPeriodRecord.query_index} after-scene' in review_js
    assert 'Final query ${sceneRecord.query_index} after-scene' in review_js
    assert "0.3 rad per-component threshold" in review_js
    assert "radiansPerRawRotationUnit = 0.5" in review_js
    assert "const threshold = 0.003" in review_js
    assert "expectedNorm = Math.hypot(...expected)" in review_js
    assert "actualNorm = Math.hypot(...actual)" in review_js
    assert "mean raw gripper over" in review_js
    assert "convention: CLOSE 0, OPEN +1" in review_js
    assert "Achieved EE orientation was not logged" in review_js
    assert "raw first 7D" not in review_js
    review_bundle = json.loads((tmp_path / "report" / "query_review_queue.json").read_text())
    required_initial_fields = {
        "reasoning",
        "action",
        "kinematic_alignment",
        "target_alignment",
        "subtask_alignment",
        "explanation",
        "source",
    }
    assert all(
        required_initial_fields <= set(record["initial_labels"])
        for record in review_bundle["records"].values()
    )
    assert all(
        record["initial_labels"]["reasoning"] in {"T", "F"}
        and record["initial_labels"]["action"] in {"T", "F"}
        and record["initial_labels"]["kinematic_alignment"] in {"T", "Partial", "F", "N/A"}
        and record["initial_labels"]["target_alignment"] in {"T", "F", "N/A"}
        and record["initial_labels"]["subtask_alignment"] in {"T", "F", "N/A"}
        and bool(record["initial_labels"]["explanation"])
        for record in review_bundle["records"].values()
    )
    assert review_bundle["compact_episode_task_ids"] == [0, 3, 9, 18, 35, 46, 73, 81, 84]
    assert review_bundle["excluded_manual_review_task_ids"] == [64]
    assert set(review_bundle["compact_episode_ids"]) == {"s0", "f1"}
    assert all(
        review_bundle["records"][query_id]["executed_chunk_length"] == 10
        for query_id in review_bundle["records"]
        if review_bundle["records"][query_id]["episode_id"] in review_bundle["compact_episode_ids"]
    )
    visible = text.split("</main>", 1)[0]
    assert "Experiment configuration and checkpoint" not in visible
    assert "Runtime and projections" not in visible
    assert "Coverage and exclusions" not in visible
    assert "Publication figures" not in visible
    assert visible.rstrip().endswith("</table></div>")
    assert (tmp_path / "report" / "query_review_queue.json").exists()
    assert (tmp_path / "report" / "query_review_ui.js").exists()
    assert validate_local_assets(index) == []
