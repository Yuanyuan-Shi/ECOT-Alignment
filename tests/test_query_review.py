from copy import deepcopy

from experiments.robot.libero.query_review import (
    build_review_queues,
    derive_case,
    episode_milestones,
    local_state_change,
    propose_all,
    review_statistics,
)


def record(episode, task, query, *, success=True, score=1.0):
    return {
        "episode_id": episode,
        "episode_success": success,
        "task_id": task,
        "task_instruction": "put the mug on the plate",
        "seed": 100 + task,
        "policy_query_index": query,
        "reasoning_raw": "SUBTASK: move to the mug MOVE: move right",
        "claims": {
            "subtask": "move to the mug",
            "target_object": "mug",
            "motion_axes": {"y": 1},
            "parse_warnings": [],
        },
        "alignment": {
            "kinematic_score": score,
            "target_score": 0.1 if score == 1 else -0.1,
            "subtask_score": score,
            "evaluated_claims": ["kinematic", "target", "subtask"],
            "diagnostics": {
                "failure_labels": [] if score == 1 else ["kinematic_opposite", "target_moved_away"],
                "target": {"resolved_object": "mug_1", "progress": 0.01},
                "target_resolution": {"before": {"status": "resolved"}},
                "subtask": {"primitive": "approach", "observations": {"progress": 0.01}},
            },
        },
        "frames": {"before": "b.png", "intermediate": "m.png", "after": "a.png"},
        "state_before": {
            "ee_position": [0, 0, 0],
            "gripper_open": True,
            "gripper_openness": 1.0,
            "objects": {"mug_1": {"position": [0, 0.1, 0], "grasped": False, "contained_by": []}},
        },
        "state_after": {
            "ee_position": [0, 0.01, 0],
            "gripper_open": True,
            "gripper_openness": 1.0,
            "objects": {"mug_1": {"position": [0, 0.1, 0], "grasped": False, "contained_by": []}},
        },
        "decoded_action_chunk": [[0] * 7] * 10,
        "executed_chunk_length": 10,
    }


def test_derive_case_enforces_six_case_truth_table_and_unresolved_statuses():
    assert derive_case("Correct", "Correct", "Yes") == "Case 1"
    assert derive_case("Incorrect", "Correct", "No") == "Case 2"
    assert derive_case("Correct", "Incorrect", "No") == "Case 3"
    assert derive_case("Incorrect", "Incorrect", "Yes") == "Case 4"
    assert derive_case("Incorrect", "Incorrect", "No") == "Case 5"
    assert derive_case("Correct", "Alternative valid", "No") == "Case 6"
    assert derive_case("Correct", "Correct", "No") == "Ambiguous"
    assert derive_case("Unverifiable", "Correct", "Yes") == "Unverifiable"


def test_review_queues_are_deterministic_disjoint_and_cover_every_episode():
    records = [record(f"e{episode}", episode, query) for episode in range(10) for query in range(4)]
    proposals = propose_all(records)
    first = build_review_queues(records, proposals, representative_size=15, diagnostic_size=10, seed=7)
    second = build_review_queues(records, proposals, representative_size=15, diagnostic_size=10, seed=7)
    assert first == second
    representative = first["representative"]
    diagnostic = first["diagnostic"]
    assert len(representative) == 15
    assert len(diagnostic) == 10
    assert {item["episode_id"] for item in representative} == {f"e{index}" for index in range(10)}
    assert {item["id"] for item in representative}.isdisjoint(item["id"] for item in diagnostic)
    assert all(item["analysis_weight"] > 0 for item in representative)


def test_representative_statistics_exclude_diagnostic_queue_and_are_reproducible():
    items = [
        {
            "id": f"e{index}::q0",
            "episode_id": f"e{index}",
            "episode_success": index < 3,
            "analysis_weight": 1.0,
            "derived_case": "Case 1" if index % 2 == 0 else "Case 3",
        }
        for index in range(6)
    ]
    first = review_statistics(items, replicates=50, seed=3)
    second = review_statistics(items, replicates=50, seed=3)
    assert first == second
    assert first["rows"]["Case 1"]["raw_count"] == 3
    assert first["rows"]["Case 3"]["failed_denominator"] == 3
    assert first["unresolved_count"] == 0


def test_proposals_preserve_ambiguous_instead_of_forcing_a_case():
    item = record("e", 1, 0)
    item["claims"]["parse_warnings"] = ["contradictory movement"]
    proposal = propose_all([item])["e::q0"]
    assert proposal["reasoning"] == "Ambiguous"
    assert proposal["derived_case"] == "Ambiguous"
    assert proposal["confidence"] == "Low"


def test_alignment_contradiction_does_not_prove_task_relative_action_is_wrong():
    item = record("e", 1, 0, success=False, score=-1)
    proposal = propose_all([item])["e::q0"]
    assert proposal["reasoning"] == "Correct"
    assert proposal["faithfulness"] == "No"
    assert proposal["action"] == "Ambiguous"
    assert proposal["derived_case"] == "Ambiguous"


def test_final_success_is_independent_evidence_of_correct_action():
    item = record("e", 1, 0, success=True, score=-1)
    item["claims"]["target_object"] = "bowl"
    item["alignment"]["diagnostics"]["target_resolution"]["before"]["status"] = "resolved"
    proposal = propose_all([item])["e::q0"]
    assert proposal["reasoning"] == "Incorrect"
    assert proposal["faithfulness"] == "No"
    assert proposal["action"] == "Correct"
    assert proposal["derived_case"] == "Case 2"


def test_episode_milestones_and_recovery_are_query_ordered():
    # Task 73/query 13 is an explicitly frame/state-reviewed wrong action;
    # an automatic alignment contradiction alone is intentionally insufficient.
    records = [record("e", 73, 12), record("e", 73, 13, score=-1), record("e", 73, 14)]
    proposals = propose_all(records)
    result = episode_milestones(records, proposals)["e"]
    assert result["first_non_case1"] == 13
    assert result["first_wrong_action"] == 13
    assert result["recovery"] == 14


def test_local_state_change_reports_grasp_and_containment_transitions():
    item = record("e", 1, 0)
    changed = deepcopy(item)
    changed["state_after"]["objects"]["mug_1"]["grasped"] = True
    changed["state_after"]["objects"]["mug_1"]["contained_by"] = ["plate_1_region"]
    summary = local_state_change(changed)
    assert summary["ee_delta"] == [0.0, 0.01, 0.0]
    assert summary["changed_objects"][0]["changes"]["grasped"] == [False, True]
    assert summary["changed_objects"][0]["changes"]["contained_by"] == [[], ["plate_1_region"]]
