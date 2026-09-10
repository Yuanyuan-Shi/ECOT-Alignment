import json

import numpy as np
import pytest

from experiments.robot.libero.alignment_evaluator import (
    EvaluatorConfig,
    ObjectState,
    ReasoningClaims,
    StepState,
    evaluate_alignment,
    normalized_gripper_openness,
    parse_reasoning,
    resolve_object,
    to_jsonable,
)


def state(ee=(0, 0, 0), objects=None, gripper_open=None):
    return StepState(np.asarray(ee, dtype=float), gripper_open=gripper_open, objects=objects or {})


def obj(name, pos, **kwargs):
    return ObjectState(name, np.asarray(pos, dtype=float), **kwargs)


@pytest.mark.parametrize(
    ("movement", "expected"),
    [
        ("move left and upward", {"y": -1, "z": 1}),
        ("go right & down", {"y": 1, "z": -1}),
        ("move forward", {"x": -1}),
        ("move backward", {"x": 1}),
    ],
)
def test_movement_synonym_normalization(movement, expected):
    claims = parse_reasoning(f"PLAN: reach;\nMOVE: {movement};\nACTION:")
    assert claims.motion_axes == expected


def test_representative_released_libero_trace():
    claims = parse_reasoning(
        "PLAN: 1. move to the red coffee mug, 2. grasp it;\n"
        "OBJECTS: {'red coffee mug': [[1, 2], [3, 4]]};\n"
        "SUBTASK REASON: the mug is not grasped;\n"
        "SUBTASK: grasp the red coffee mug;\n"
        "MOVE REASON: align the gripper;\n"
        "MOVE: move up and right and back and close gripper;\n"
        "GRIPPER: [100, 120];\nACTION: tokens"
    )
    assert claims.subtask == "grasp the red coffee mug"
    assert claims.target_object == "red coffee mug"
    assert claims.motion_axes == {"x": 1, "y": 1, "z": 1}
    assert claims.gripper_command == "close"
    assert claims.raw_text.startswith("PLAN:")


def test_released_trace_with_space_separated_fields():
    claims = parse_reasoning(
        "PLAN: 1. move to the cabinet, 2. grasp the drawer "
        "VISIBLE OBJECTS: wooden cabinet 1 [77, 0, 176, 101] "
        "SUBTASK REASONING: the cabinet needs attention "
        "SUBTASK: move to the cabinet "
        "MOVE REASONING: align with the cabinet "
        "MOVE: move up and right GRIPPER POSITION: [36, 108]"
    )
    assert claims.subtask == "move to the cabinet"
    assert claims.target_object == "cabinet"
    assert claims.motion_axes == {"y": 1, "z": 1}


def test_rotation_direction_is_not_parsed_as_translation():
    claims = parse_reasoning("MOVE: move down and forward and rotate up")
    assert claims.motion_axes == {"x": -1, "z": -1}
    assert "contradictory_motion_z" not in claims.parse_warnings


@pytest.mark.parametrize(
    ("subtask", "target"),
    [
        ("move to the butter at the back", "butter at the back"),
        ("grasp the left mug", "left mug"),
        ("move the left mug to the tray", "left mug"),
    ],
)
def test_relational_target_claim_is_preserved(subtask, target):
    assert parse_reasoning(f"SUBTASK: {subtask}").target_object == target


def test_articulated_motion_suffix_is_not_part_of_target():
    claims = parse_reasoning("SUBTASK: move the top drawer downwards")
    assert claims.target_object == "top drawer"


def test_wooden_cabinet_drawer_alias_resolution():
    objects = {"wooden_cabinet_1_top_region": obj("wooden_cabinet_1_top_region", (0, 0, 0))}
    resolved, diagnostic = resolve_object("top drawer of the wooden cabinet", objects)
    assert resolved == "wooden_cabinet_1_top_region"
    assert diagnostic["canonical_query"] == "top region"


@pytest.mark.parametrize(
    ("movement", "delta"),
    [("left", (0, -0.02, 0)), ("right", (0, 0.02, 0)), ("up", (0, 0, 0.02)), ("down", (0, 0, -0.02))],
)
def test_directional_matches(movement, delta):
    claims = parse_reasoning(f"MOVE: move {movement}")
    result = evaluate_alignment(claims, np.zeros((10, 7)), state(), state(delta), config=EvaluatorConfig())
    assert result.kinematic_score == 1


def test_opposite_direction_motion():
    result = evaluate_alignment(
        parse_reasoning("MOVE: move left"), np.zeros((10, 7)), state(), state((0, 0.02, 0)), config=EvaluatorConfig()
    )
    assert result.kinematic_score == -1
    assert "kinematic_opposite" in result.diagnostics["failure_labels"]


def test_deadband_no_motion():
    result = evaluate_alignment(
        parse_reasoning("MOVE: move right"),
        np.zeros((10, 7)),
        state(),
        state((0.001, 0, 0)),
        config=EvaluatorConfig(motion_deadband=0.005),
    )
    assert result.kinematic_score == -1
    assert "kinematic_no_motion" in result.diagnostics["failure_labels"]


@pytest.mark.parametrize(
    ("delta", "expected_score", "expected_counts"),
    [
        ((0.01, 0.01, 0.01), 1 / 3, {"match": 2, "missing": 0, "opposite": 0, "extra": 1}),
        ((0, 0, 0.01), 0, {"match": 1, "missing": 1, "opposite": 0, "extra": 0}),
        ((0, 0.01, -0.01), 0, {"match": 1, "missing": 0, "opposite": 1, "extra": 0}),
        ((0.01, 0, 0), -1, {"match": 0, "missing": 2, "opposite": 0, "extra": 1}),
    ],
)
def test_set_kinematic_scoring_penalizes_missing_opposite_and_extra(delta, expected_score, expected_counts):
    result = evaluate_alignment(
        parse_reasoning("MOVE: move up and right"),
        np.zeros((10, 7)),
        state(),
        state(delta),
        config=EvaluatorConfig(motion_deadband=0.003),
    )
    assert result.kinematic_score == pytest.approx(expected_score)
    assert result.diagnostics["kinematic"]["counts"] == expected_counts


def test_reasoning_with_no_movement_claim():
    result = evaluate_alignment(
        parse_reasoning("PLAN: grasp mug; SUBTASK: grasp the mug"),
        np.zeros((10, 7)),
        state(objects={"mug_1": obj("mug_1", (1, 0, 0), grasped=False)}),
        state(objects={"mug_1": obj("mug_1", (1, 0, 0), grasped=True)}, gripper_open=False),
        config=EvaluatorConfig(),
    )
    assert result.kinematic_score is None


@pytest.mark.parametrize(("after_x", "sign"), [(0.8, 1), (-0.2, -1)])
def test_target_distance_decreasing_and_increasing(after_x, sign):
    objects = {"red_coffee_mug_1": obj("red_coffee_mug_1", (1, 0, 0))}
    result = evaluate_alignment(
        parse_reasoning("SUBTASK: move to the red mug"),
        np.zeros((3, 7)),
        state(objects=objects),
        state((after_x, 0, 0), objects=objects),
        config=EvaluatorConfig(target_progress_scale=1),
    )
    assert np.sign(result.target_score) == sign


def test_absent_target_claim():
    result = evaluate_alignment(
        parse_reasoning("MOVE: move right"), np.zeros((10, 7)), state(), state((0.1, 0, 0)), config=EvaluatorConfig()
    )
    assert result.target_score is None


def test_ambiguous_object_resolution():
    objects = {
        "akita_black_bowl_1": obj("akita_black_bowl_1", (0, 0, 0)),
        "akita_black_bowl_2": obj("akita_black_bowl_2", (1, 0, 0)),
    }
    resolved, diagnostic = resolve_object("black bowl", objects)
    assert resolved is None
    assert diagnostic["status"] == "ambiguous"


def test_explicit_libero_object_alias_resolution():
    objects = {"akita_black_bowl_1": obj("akita_black_bowl_1", (0, 0, 0))}
    resolved, diagnostic = resolve_object("black bowl", objects)
    assert resolved == "akita_black_bowl_1"
    assert diagnostic["canonical_query"] == "akita black bowl"


def test_relational_resolution_prefers_explicit_libero_region():
    objects = {
        "butter_1": obj("butter_1", (0, 0, 0), contained_by=["kitchen_table_butter_front_init_region"]),
        "butter_2": obj("butter_2", (0, 0, 0), contained_by=["kitchen_table_butter_back_init_region"]),
    }
    resolved, diagnostic = resolve_object("butter at the back", objects)
    assert resolved == "butter_2"
    assert diagnostic["method"] == "explicit_libero_region"


@pytest.mark.parametrize(
    ("claim", "expected"),
    [("left mug", "mug_1"), ("right mug", "mug_2"), ("front mug", "mug_3"), ("back mug", "mug_4")],
)
def test_relational_resolution_uses_world_position_with_margin(claim, expected):
    objects = {
        "mug_1": obj("mug_1", (0.0, -0.2, 0)),
        "mug_2": obj("mug_2", (0.0, 0.2, 0)),
        "mug_3": obj("mug_3", (0.2, 0.0, 0)),
        "mug_4": obj("mug_4", (-0.2, 0.0, 0)),
    }
    resolved, diagnostic = resolve_object(claim, objects, relational_margin=0.02)
    assert resolved == expected
    assert diagnostic["method"] == "relational_world_position"


def test_relational_resolution_respects_ambiguity_margin():
    objects = {
        "mug_1": obj("mug_1", (0, -0.005, 0)),
        "mug_2": obj("mug_2", (0, 0.005, 0)),
    }
    resolved, diagnostic = resolve_object("left mug", objects, relational_margin=0.02)
    assert resolved is None
    assert diagnostic["status"] == "ambiguous"


def test_middle_relational_resolution_requires_odd_candidate_count():
    objects = {
        "bowl_1": obj("bowl_1", (0, -0.2, 0)),
        "bowl_2": obj("bowl_2", (0, 0.0, 0)),
        "bowl_3": obj("bowl_3", (0, 0.2, 0)),
    }
    resolved, diagnostic = resolve_object("middle bowl", objects)
    assert resolved == "bowl_2"
    assert diagnostic["method"] == "relational_median_y"


@pytest.mark.parametrize(
    ("positions", "expected"),
    [([0.04, -0.04], 1.0), ([0.0, 0.0], 0.0), ([0.02, -0.02], 0.5)],
)
def test_gripper_openness_uses_joint_limits(positions, expected):
    assert normalized_gripper_openness(positions, [[0.0, 0.04], [-0.04, 0.0]]) == pytest.approx(expected)


def test_gripper_openness_unavailable_without_matching_limits():
    assert normalized_gripper_openness([0.02], []) is None


def test_grasp_success_and_approach_without_grasp():
    before = state(objects={"mug_1": obj("mug_1", (0.1, 0, 0), grasped=False)}, gripper_open=True)
    success = state(objects={"mug_1": obj("mug_1", (0.1, 0, 0.03), grasped=True)}, gripper_open=False)
    approach_only = state((0.08, 0, 0), objects={"mug_1": obj("mug_1", (0.1, 0, 0), grasped=False)}, gripper_open=False)
    claims = parse_reasoning("SUBTASK: grasp the mug")
    assert evaluate_alignment(claims, np.zeros((10, 7)), before, success, config=EvaluatorConfig()).subtask_score == 1
    assert (
        evaluate_alignment(claims, np.zeros((10, 7)), before, approach_only, config=EvaluatorConfig()).subtask_score == 0
    )


def test_unsupported_subtask():
    claims = parse_reasoning("SUBTASK: sing to the mug")
    result = evaluate_alignment(claims, np.zeros((10, 7)), state(), state(), config=EvaluatorConfig())
    assert result.subtask_score is None
    assert "unsupported_subtask" in result.diagnostics["failure_labels"]


@pytest.mark.parametrize(
    ("subtask", "before_kwargs", "after_kwargs"),
    [
        ("open the cabinet", {"open_state": False}, {"open_state": True}),
        ("close the cabinet", {"closed_state": False}, {"closed_state": True}),
    ],
)
def test_open_close_uses_libero_predicate_transition(subtask, before_kwargs, after_kwargs):
    result = evaluate_alignment(
        parse_reasoning(f"SUBTASK: {subtask}"),
        np.zeros((10, 7)),
        state(objects={"cabinet_1": obj("cabinet_1", (0, 0, 0), **before_kwargs)}),
        state(objects={"cabinet_1": obj("cabinet_1", (0, 0, 0), **after_kwargs)}),
        config=EvaluatorConfig(),
    )
    assert result.subtask_score == 1
    assert result.diagnostics["subtask"]["observations"]["predicate_after"] is True


def test_open_not_completed_uses_libero_predicate():
    result = evaluate_alignment(
        parse_reasoning("SUBTASK: open the cabinet"),
        np.zeros((10, 7)),
        state(objects={"cabinet_1": obj("cabinet_1", (0, 0, 0), open_state=False)}),
        state(objects={"cabinet_1": obj("cabinet_1", (0, 0, 0), open_state=False)}),
        config=EvaluatorConfig(),
    )
    assert result.subtask_score == 0
    assert "subtask_not_completed" in result.diagnostics["failure_labels"]


def test_open_already_satisfied_is_not_scored():
    result = evaluate_alignment(
        parse_reasoning("SUBTASK: open the cabinet"),
        np.zeros((10, 7)),
        state(objects={"cabinet_1": obj("cabinet_1", (0, 0, 0), open_state=True)}),
        state(objects={"cabinet_1": obj("cabinet_1", (0, 0, 0), open_state=True)}),
        config=EvaluatorConfig(),
    )
    assert result.subtask_score is None
    assert "unsupported_subtask" in result.diagnostics["failure_labels"]


def test_aggregation_with_one_two_and_three_components():
    before_objects = {"mug_1": obj("mug_1", (1, 0, 0), grasped=False)}
    after_objects = {"mug_1": obj("mug_1", (1, 0, 0), grasped=True)}
    before = state(objects=before_objects, gripper_open=True)
    after = state((0.1, 0, 0), objects=after_objects, gripper_open=False)

    one = evaluate_alignment(
        ReasoningClaims("", motion_axes={"x": 1}), np.zeros((1, 7)), before, after, config=EvaluatorConfig()
    )
    two = evaluate_alignment(
        ReasoningClaims("", target_object="mug", motion_axes={"x": 1}),
        np.zeros((1, 7)),
        before,
        after,
        config=EvaluatorConfig(target_progress_scale=1),
    )
    three = evaluate_alignment(
        ReasoningClaims("", subtask="grasp the mug", target_object="mug", motion_axes={"x": 1}),
        np.zeros((1, 7)),
        before,
        after,
        config=EvaluatorConfig(target_progress_scale=1),
    )
    assert one.aggregate_score == one.kinematic_score
    assert two.aggregate_score == pytest.approx((two.kinematic_score + two.target_score) / 2)
    assert three.aggregate_score == pytest.approx((three.kinematic_score + three.target_score + three.subtask_score) / 3)


def test_all_components_unavailable():
    result = evaluate_alignment(
        ReasoningClaims("PLAN: wait"), np.zeros((1, 7)), state(), state(), config=EvaluatorConfig()
    )
    assert result.aggregate_score is None
    assert "no_verifiable_claim" in result.diagnostics["failure_labels"]


def test_truncated_action_chunk_is_accepted():
    result = evaluate_alignment(
        ReasoningClaims("MOVE: right", motion_axes={"x": 1}),
        np.zeros((3, 7)),
        state(),
        state((0.1, 0, 0)),
        config=EvaluatorConfig(),
    )
    assert result.kinematic_score == 1


def test_stable_json_serialization():
    value = {"z": np.asarray([1, 2]), "a": ReasoningClaims("MOVE: right", motion_axes={"x": 1})}
    first = json.dumps(to_jsonable(value), sort_keys=True, allow_nan=False)
    second = json.dumps(to_jsonable(value), sort_keys=True, allow_nan=False)
    assert first == second
