"""Deterministic reasoning--action alignment metrics for LIBERO rollouts.

This module deliberately has no torch, robosuite, or LIBERO dependency so its
metrics can be unit tested on CPU-only machines.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


@dataclass
class ReasoningClaims:
    raw_text: str
    subtask: Optional[str] = None
    target_object: Optional[str] = None
    motion_axes: Dict[str, int] = field(default_factory=dict)
    gripper_command: Optional[str] = None
    parse_warnings: List[str] = field(default_factory=list)


@dataclass
class ObjectState:
    name: str
    position: np.ndarray
    joint_positions: List[float] = field(default_factory=list)
    grasped: Optional[bool] = None
    open_state: Optional[bool] = None
    closed_state: Optional[bool] = None
    contacts: List[str] = field(default_factory=list)
    contained_by: List[str] = field(default_factory=list)


@dataclass
class StepState:
    ee_position: np.ndarray
    gripper_positions: List[float] = field(default_factory=list)
    gripper_open: Optional[bool] = None
    gripper_openness: Optional[float] = None
    gripper_joint_limits: List[List[float]] = field(default_factory=list)
    objects: Dict[str, ObjectState] = field(default_factory=dict)


@dataclass
class EvaluatorConfig:
    motion_deadband: float = 0.003
    target_progress_scale: float = 0.05
    approach_progress_threshold: float = 0.005
    transport_distance_tolerance: float = 0.03
    lift_height_threshold: float = 0.02
    joint_motion_threshold: float = 0.01
    relational_resolution_margin: float = 0.02
    component_weights: Dict[str, float] = field(
        default_factory=lambda: {"kinematic": 1.0, "target": 1.0, "subtask": 1.0}
    )
    include_cosine_similarity: bool = False


@dataclass
class AlignmentResult:
    kinematic_score: Optional[float]
    target_score: Optional[float]
    subtask_score: Optional[float]
    aggregate_score: Optional[float]
    evaluated_claims: List[str]
    diagnostics: Dict[str, Any]


_FIELD_RE = re.compile(
    r"(?i)(?:^|[;\n]|\s)(PLAN|SUBTASK\s+(?:REASON|REASONING)|SUBTASK|"
    r"MOVE\s+(?:REASON|REASONING)|MOVE|VISIBLE\s+OBJECTS|OBJECTS|"
    r"GRIPPER(?:\s+POSITION)?|ACTION)\s*:\s*"
)

_AXIS_TERMS: Dict[str, Tuple[str, int]] = {
    # Released LIBERO annotations are camera-oriented: image left/right maps
    # to world -y/+y, while image forward/back maps to world -x/+x.
    "left": ("y", -1),
    "right": ("y", 1),
    "up": ("z", 1),
    "upward": ("z", 1),
    "above": ("z", 1),
    "down": ("z", -1),
    "downward": ("z", -1),
    "below": ("z", -1),
    "forward": ("x", -1),
    "front": ("x", -1),
    "back": ("x", 1),
    "backward": ("x", 1),
    "behind": ("x", 1),
}

_ROTATION_DIRECTION_RE = re.compile(
    r"\b(?:rotate|rotating)\s+(?:left|right|up|upward|down|downward|forward|front|back|backward|behind)\b",
    flags=re.I,
)

# Released language names that intentionally differ from LIBERO BDDL categories.
# This table is explicit so resolution never devolves into fuzzy best-match logic.
_LIBERO_OBJECT_ALIASES = {
    "black bowl": "akita black bowl",
    "red mug": "red coffee mug",
    "white mug": "porcelain mug",
    "yellow and white mug": "white yellow mug",
    "white and yellow mug": "white yellow mug",
    "frying pan": "chefmate 8 frypan",
    "salad dressing": "new salad dressing",
    "top drawer": "top region",
    "top drawer of cabinet": "top region",
    "top drawer of wooden cabinet": "top region",
    "bottom drawer": "bottom region",
    "bottom drawer of cabinet": "bottom region",
    "bottom drawer of wooden cabinet": "bottom region",
}


def _split_fields(text: str) -> Dict[str, List[str]]:
    matches = list(_FIELD_RE.finditer(text))
    fields: Dict[str, List[str]] = {}
    for index, match in enumerate(matches):
        key = re.sub(r"\s+", " ", match.group(1).upper()).strip()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end() : value_end].strip(" \t\r\n;")
        fields.setdefault(key, []).append(value)
    return fields


def _clean_target(value: str, *, strip_destination: bool = True) -> Optional[str]:
    value = value.strip(" .,;:()[]{}\"'")
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)
    if strip_destination:
        value = re.split(
            r"\s+(?:to|toward|towards|into|in|onto|on|above|below|under|over|near|away)\b",
            value,
            maxsplit=1,
            flags=re.I,
        )[0]
    value = re.sub(r"\s+(?:because|since|so that)\b.*$", "", value, flags=re.I).strip()
    return value or None


def _target_from_subtask(subtask: str) -> Optional[str]:
    normalized = subtask.strip()
    # Released annotations sometimes include a rationale followed by an arrow.
    if "→" in normalized:
        normalized = normalized.rsplit("→", 1)[-1].strip()
    approach = re.match(
        r"^(?:move|go|reach|approach)\s+(?:the\s+)?(?:robot|gripper|end[- ]effector)?\s*"
        r"(?:to|toward|towards)\s+(.+)$",
        normalized,
        flags=re.I,
    )
    if approach:
        return _clean_target(approach.group(1), strip_destination=False)
    grasp = re.match(r"^(?:grasp|grab|pick\s+up|lift)\s+(.+)$", normalized, flags=re.I)
    if grasp:
        return _clean_target(grasp.group(1), strip_destination=False)
    release = re.match(r"^(?:release|drop)\s+(.+)$", normalized, flags=re.I)
    if release:
        return _clean_target(release.group(1))
    transfer = re.match(r"^(?:move|transport|carry|place|put)\s+(.+)$", normalized, flags=re.I)
    if transfer:
        target = _clean_target(transfer.group(1))
        # In articulated-object traces, a final vertical direction describes
        # the requested motion rather than forming part of the object name.
        if target:
            target = re.sub(r"\s+(?:upwards?|downwards?)$", "", target, flags=re.I).strip()
        return target or None
    articulated = re.match(r"^(?:open|close)\s+(.+)$", normalized, flags=re.I)
    if articulated:
        return _clean_target(articulated.group(1), strip_destination=False)
    return None


def parse_reasoning(text: str) -> ReasoningClaims:
    """Parse only explicit claims in the released tagged ECoT format."""
    raw = text if isinstance(text, str) else str(text)
    fields = _split_fields(raw)
    warnings: List[str] = []
    if not fields:
        warnings.append("no_recognized_reasoning_fields")

    subtask_values = fields.get("SUBTASK", [])
    subtask = subtask_values[-1] or None if subtask_values else None
    if len({value.casefold() for value in subtask_values if value}) > 1:
        warnings.append("contradictory_subtask_claims")

    move_values = fields.get("MOVE", [])
    move = move_values[-1].rsplit("→", 1)[-1].strip() if move_values else ""
    axes: Dict[str, int] = {}
    seen: Dict[str, set] = {}
    for move_claim in move_values:
        move_claim = move_claim.rsplit("→", 1)[-1].strip()
        translation_claim = _ROTATION_DIRECTION_RE.sub("", move_claim)
        for term, (axis, direction) in _AXIS_TERMS.items():
            if re.search(rf"\b{re.escape(term)}\b", translation_claim, flags=re.I):
                seen.setdefault(axis, set()).add(direction)
    for axis, directions in seen.items():
        if len(directions) == 1:
            axes[axis] = next(iter(directions))
        else:
            warnings.append(f"contradictory_motion_{axis}")
    movement_words = re.findall(r"[a-z]+", move.casefold())
    known_words = set(_AXIS_TERMS) | {
        "move",
        "moving",
        "and",
        "the",
        "to",
        "toward",
        "towards",
        "slightly",
        "a",
        "bit",
        "none",
        "stay",
    }
    unknown_direction_words = [word for word in movement_words if word not in known_words]
    if move and not axes and not re.search(r"\b(?:open|close|grasp|release|none|stay)\b", move, re.I):
        warnings.append("unknown_movement_claim:" + "_".join(unknown_direction_words[:4]))

    command_candidates: List[str] = []
    # GRIPPER POSITION is normally pixel metadata; accept commands only when explicit.
    command_values = [value.rsplit("→", 1)[-1] for value in move_values] + fields.get("GRIPPER", [])
    for value in command_values:
        if re.search(r"\b(?:open|release)\b", value, flags=re.I):
            command_candidates.append("open")
        if re.search(r"\b(?:close|grasp|grab)\b", value, flags=re.I):
            command_candidates.append("close")
    commands = set(command_candidates)
    gripper_command = next(iter(commands)) if len(commands) == 1 else None
    if len(commands) > 1:
        warnings.append("contradictory_gripper_commands")

    target = _target_from_subtask(subtask) if subtask else None
    return ReasoningClaims(raw, subtask, target, axes, gripper_command, warnings)


def normalize_object_name(name: str) -> str:
    normalized = name.casefold().replace("&", " and ")
    normalized = re.sub(r"[_-]+", " ", normalized)
    normalized = re.sub(r"\b(?:the|a|an)\b", " ", normalized)
    normalized = re.sub(r"\s+\d+$", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def normalized_gripper_openness(
    positions: List[float], joint_limits: List[List[float]]
) -> Optional[float]:
    """Return mean parallel-jaw openness in [0, 1] from MuJoCo joint ranges."""
    if not positions or len(positions) != len(joint_limits):
        return None
    fractions = []
    for position, limits in zip(positions, joint_limits):
        if len(limits) != 2:
            return None
        low, high = (float(limits[0]), float(limits[1]))
        closed, opened = sorted((abs(low), abs(high)))
        span = opened - closed
        if span <= np.finfo(float).eps:
            return None
        fractions.append(float(np.clip((abs(float(position)) - closed) / span, 0.0, 1.0)))
    return float(np.mean(fractions))


_RELATION_AXES = {
    "left": (1, -1),
    "right": (1, 1),
    # Object-layout front/back follows LIBERO's named initialization regions,
    # which is distinct from the camera-oriented movement-label convention.
    "front": (0, 1),
    "back": (0, -1),
}


def _relation_and_base(query: str) -> Tuple[Optional[str], str]:
    relations = [term for term in ("left", "right", "front", "back", "middle", "center") if term in query.split()]
    if len(relations) != 1:
        return None, query
    relation = "middle" if relations[0] == "center" else relations[0]
    base = re.sub(rf"\b{relations[0]}\b", " ", query)
    base = re.sub(r"\b(?:at|on|in|of|from|side|most)\b", " ", base)
    return relation, " ".join(base.split())


def resolve_object(
    name: str, objects: Mapping[str, ObjectState], *, relational_margin: float = 0.02
) -> Tuple[Optional[str], Dict[str, Any]]:
    query = normalize_object_name(name)
    relation, base_query = _relation_and_base(query)
    canonical = _LIBERO_OBJECT_ALIASES.get(base_query, base_query)
    exact_matches = [key for key in objects if normalize_object_name(key) == canonical]
    matches = exact_matches or [key for key in objects if normalize_object_name(key).endswith(" " + canonical)]
    diagnostics: Dict[str, Any] = {
        "query": name,
        "normalized_query": query,
        "canonical_query": canonical,
        "relation": relation,
        "method": "exact" if exact_matches else "unique_canonical_suffix",
        "matches": sorted(matches),
    }
    if len(matches) == 1 and relation is None:
        diagnostics["status"] = "resolved"
        return matches[0], diagnostics
    if relation and matches:
        # Prefer explicit LIBERO initialization / compartment regions over a
        # geometric guess. This resolves names such as "butter at the back".
        region_matches = []
        for key in matches:
            relation_regions = [
                region
                for region in objects[key].contained_by
                if relation in normalize_object_name(region).split()
            ]
            if relation_regions:
                region_matches.append((key, sorted(relation_regions)))
        diagnostics["relation_region_matches"] = [
            {"object": key, "regions": regions} for key, regions in region_matches
        ]
        if len(region_matches) == 1:
            diagnostics.update(status="resolved", method="explicit_libero_region")
            return region_matches[0][0], diagnostics

        if relation == "middle":
            if len(matches) >= 3 and len(matches) % 2 == 1:
                ordered = sorted(matches, key=lambda key: float(np.asarray(objects[key].position)[1]))
                selected = ordered[len(ordered) // 2]
                diagnostics.update(
                    status="resolved",
                    method="relational_median_y",
                    ordered_candidates=ordered,
                )
                return selected, diagnostics
        elif len(matches) >= 2:
            axis, direction = _RELATION_AXES[relation]
            ordered = sorted(
                matches,
                key=lambda key: direction * float(np.asarray(objects[key].position)[axis]),
                reverse=True,
            )
            best, runner_up = ordered[:2]
            separation = direction * (
                float(np.asarray(objects[best].position)[axis])
                - float(np.asarray(objects[runner_up].position)[axis])
            )
            diagnostics.update(
                method="relational_world_position",
                relation_axis="xyz"[axis],
                candidate_coordinates={key: float(np.asarray(objects[key].position)[axis]) for key in ordered},
                required_separation=relational_margin,
                measured_separation=separation,
            )
            if separation >= relational_margin:
                diagnostics["status"] = "resolved"
                return best, diagnostics
    diagnostics["status"] = "ambiguous" if matches else "unresolved"
    return None, diagnostics


def _primitive(subtask: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not subtask:
        return None, None
    value = subtask.rsplit("→", 1)[-1].strip().casefold()
    receptacle = None
    if re.match(r"^(?:move|go|reach|approach)\b", value) and re.search(r"\b(?:to|toward|towards)\b", value):
        return "approach", None
    if re.match(r"^(?:grasp|grab|pick up|lift)\b", value):
        return "grasp", None
    if re.match(r"^(?:move|transport|carry)\b", value):
        return "transport", None
    if re.match(r"^(?:place|put)\b", value):
        match = re.search(r"\b(?:in|into|on|onto)\s+(?:the\s+)?(.+)$", value)
        receptacle = _clean_target(match.group(1), strip_destination=False) if match else None
        return "place", receptacle
    if re.match(r"^open\b", value):
        return "open", None
    if re.match(r"^close\b", value):
        return "close", None
    if re.match(r"^(?:release|drop)\b", value):
        return "release", None
    return None, None


def _aggregate(scores: Mapping[str, Optional[float]], weights: Mapping[str, float]) -> Optional[float]:
    available = [(name, score) for name, score in scores.items() if score is not None and weights.get(name, 1.0) > 0]
    if not available:
        return None
    denominator = sum(weights.get(name, 1.0) for name, _ in available)
    return float(sum(weights.get(name, 1.0) * float(score) for name, score in available) / denominator)


def evaluate_alignment(
    claims: ReasoningClaims,
    action_chunk: np.ndarray,
    state_before: StepState,
    state_after: StepState,
    *,
    config: EvaluatorConfig,
) -> AlignmentResult:
    """Evaluate claims independently against measured simulator state changes."""
    del action_chunk  # Logged for provenance; metrics intentionally use actual simulator state.
    diagnostics: Dict[str, Any] = {"failure_labels": [], "parse_warnings": list(claims.parse_warnings)}
    evaluated: List[str] = []
    delta = np.asarray(state_after.ee_position, dtype=float) - np.asarray(state_before.ee_position, dtype=float)

    kinematic_score: Optional[float] = None
    if claims.motion_axes:
        axis_index = {"x": 0, "y": 1, "z": 2}
        per_axis = {}
        categories: Dict[str, List[str]] = {
            "match": [], "missing": [], "opposite": [], "extra": []
        }
        for axis, expected in sorted(claims.motion_axes.items()):
            measured = float(delta[axis_index[axis]])
            signed = measured * expected
            category = (
                "match" if signed > config.motion_deadband
                else "opposite" if signed < -config.motion_deadband
                else "missing"
            )
            score = 1 if category == "match" else -1
            categories[category].append(axis)
            per_axis[axis] = {
                "expected_direction": expected,
                "measured_displacement": measured,
                "threshold": config.motion_deadband,
                "category": category,
                "score": score,
            }
        for axis, index in axis_index.items():
            if axis not in claims.motion_axes and abs(float(delta[index])) > config.motion_deadband:
                categories["extra"].append(axis)
        counts = {name: len(axes) for name, axes in categories.items()}
        contributions = {
            "match": counts["match"],
            "missing": -counts["missing"],
            "opposite": -counts["opposite"],
            "extra": -counts["extra"],
        }
        denominator = sum(counts.values())
        numerator = sum(contributions.values())
        kinematic_score = float(numerator / denominator)
        diagnostics["kinematic"] = {
            "delta_ee": delta.tolist(),
            "per_axis": per_axis,
            "categories": categories,
            "counts": counts,
            "contributions": contributions,
            "numerator": numerator,
            "denominator": denominator,
            "formula": "(match - missing - opposite - extra) / total",
        }
        evaluated.append("kinematic")
        if categories["opposite"]:
            diagnostics["failure_labels"].append("kinematic_opposite")
        if len(categories["missing"]) == len(claims.motion_axes):
            diagnostics["failure_labels"].append("kinematic_no_motion")
        if config.include_cosine_similarity:
            expected_vector = np.zeros(3)
            for axis, direction in claims.motion_axes.items():
                expected_vector[axis_index[axis]] = direction
            norm = np.linalg.norm(expected_vector) * np.linalg.norm(delta)
            diagnostics["kinematic"]["cosine_similarity"] = (
                float(np.dot(expected_vector, delta) / norm) if norm else None
            )

    target_score: Optional[float] = None
    target_key: Optional[str] = None
    if claims.target_object:
        before_key, before_resolution = resolve_object(
            claims.target_object,
            state_before.objects,
            relational_margin=config.relational_resolution_margin,
        )
        after_key, after_resolution = resolve_object(
            claims.target_object,
            state_after.objects,
            relational_margin=config.relational_resolution_margin,
        )
        diagnostics["target_resolution"] = {"before": before_resolution, "after": after_resolution}
        if before_key is not None and after_key is not None and before_key == after_key:
            target_key = before_key
            before_obj = state_before.objects[before_key]
            after_obj = state_after.objects[after_key]
            d_before = float(np.linalg.norm(np.asarray(state_before.ee_position) - np.asarray(before_obj.position)))
            d_after = float(np.linalg.norm(np.asarray(state_after.ee_position) - np.asarray(after_obj.position)))
            progress = d_before - d_after
            target_score = float(np.clip(progress / config.target_progress_scale, -1.0, 1.0))
            diagnostics["target"] = {
                "resolved_object": target_key,
                "distance_before": d_before,
                "distance_after": d_after,
                "progress": progress,
                "progress_scale": config.target_progress_scale,
            }
            evaluated.append("target")
            if progress < 0:
                diagnostics["failure_labels"].append("target_moved_away")
        else:
            diagnostics["failure_labels"].append("target_unresolved")

    subtask_score: Optional[float] = None
    primitive, receptacle_claim = _primitive(claims.subtask)
    if claims.subtask:
        diagnostics["subtask"] = {"raw": claims.subtask, "primitive": primitive}
        if primitive is None:
            diagnostics["failure_labels"].append("unsupported_subtask")
            diagnostics["subtask"]["unsupported_subtask"] = claims.subtask
        elif target_key is None and claims.target_object:
            diagnostics["subtask"]["reason"] = "target_unresolved"
        elif target_key is None:
            diagnostics["failure_labels"].append("unsupported_subtask")
            diagnostics["subtask"]["reason"] = "subtask_has_no_resolvable_target"
        else:
            before_obj, after_obj = state_before.objects[target_key], state_after.objects[target_key]
            observations: Dict[str, Any] = {}
            if primitive == "approach":
                before_d = float(np.linalg.norm(np.asarray(state_before.ee_position) - np.asarray(before_obj.position)))
                after_d = float(np.linalg.norm(np.asarray(state_after.ee_position) - np.asarray(after_obj.position)))
                progress = before_d - after_d
                subtask_score = float(np.clip(progress / config.approach_progress_threshold, 0.0, 1.0))
                observations.update(distance_before=before_d, distance_after=after_d, progress=progress)
            elif primitive == "grasp":
                lifted = float(after_obj.position[2] - before_obj.position[2]) >= config.lift_height_threshold
                closed = state_after.gripper_open is False
                subtask_score = float(after_obj.grasped is True or (lifted and closed))
                observations.update(grasped=after_obj.grasped, lifted=lifted, gripper_closed=closed)
            elif primitive == "transport":
                obj_delta = np.asarray(after_obj.position) - np.asarray(before_obj.position)
                consistent = float(np.linalg.norm(obj_delta - delta)) <= config.transport_distance_tolerance
                was_grasped = before_obj.grasped is True or after_obj.grasped is True
                subtask_score = float(was_grasped and consistent)
                observations.update(
                    was_grasped=was_grasped,
                    object_delta=obj_delta.tolist(),
                    ee_delta=delta.tolist(),
                    consistent=consistent,
                )
            elif primitive == "place":
                receptacle_key = None
                resolution = None
                if receptacle_claim:
                    receptacle_key, resolution = resolve_object(
                        receptacle_claim,
                        state_after.objects,
                        relational_margin=config.relational_resolution_margin,
                    )
                if receptacle_key is None:
                    diagnostics["failure_labels"].append("unsupported_subtask")
                    observations.update(receptacle=receptacle_claim, receptacle_resolution=resolution)
                else:
                    contained = receptacle_key in after_obj.contained_by or receptacle_key in after_obj.contacts
                    released = after_obj.grasped is False and state_after.gripper_open is True
                    subtask_score = float(contained and released)
                    observations.update(receptacle=receptacle_key, contained_or_contact=contained, released=released)
            elif primitive in {"open", "close"}:
                desired_attr = "open_state" if primitive == "open" else "closed_state"
                before_desired = getattr(before_obj, desired_attr)
                after_desired = getattr(after_obj, desired_attr)
                observations.update(
                    predicate=desired_attr,
                    predicate_before=before_desired,
                    predicate_after=after_desired,
                )
                if before_desired is True:
                    diagnostics["failure_labels"].append("unsupported_subtask")
                    observations["reason"] = "desired_state_already_satisfied_before_chunk"
                elif after_desired is not None:
                    subtask_score = float(after_desired)
                else:
                    diagnostics["failure_labels"].append("unsupported_subtask")
                    observations["reason"] = "libero_open_close_predicate_unavailable"
                if before_obj.joint_positions and len(before_obj.joint_positions) == len(after_obj.joint_positions):
                    observations["joint_delta"] = (
                        np.asarray(after_obj.joint_positions) - np.asarray(before_obj.joint_positions)
                    ).tolist()
            elif primitive == "release":
                subtask_score = float(state_after.gripper_open is True and after_obj.grasped is False)
                observations.update(gripper_open=state_after.gripper_open, grasped=after_obj.grasped)
            diagnostics["subtask"]["observations"] = observations
            if subtask_score is not None:
                evaluated.append("subtask")
                if subtask_score < 1.0:
                    diagnostics["failure_labels"].append("subtask_not_completed")

    if target_key:
        newly_grasped = [
            name
            for name, obj in state_after.objects.items()
            if obj.grasped is True
            and (name not in state_before.objects or state_before.objects[name].grasped is not True)
        ]
        if newly_grasped and target_key not in newly_grasped:
            diagnostics["failure_labels"].append("wrong_object_interaction")
            diagnostics["wrong_object_interaction"] = newly_grasped
    if "no_recognized_reasoning_fields" in claims.parse_warnings:
        diagnostics["failure_labels"].append("reasoning_parse_failure")
    scores = {"kinematic": kinematic_score, "target": target_score, "subtask": subtask_score}
    aggregate = _aggregate(scores, config.component_weights)
    if aggregate is None:
        diagnostics["failure_labels"].append("no_verifiable_claim")
    diagnostics["failure_labels"] = sorted(set(diagnostics["failure_labels"]))
    return AlignmentResult(kinematic_score, target_score, subtask_score, aggregate, evaluated, diagnostics)


def to_jsonable(value: Any) -> Any:
    """Stable conversion for dataclasses, NumPy values, and non-finite floats."""
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
