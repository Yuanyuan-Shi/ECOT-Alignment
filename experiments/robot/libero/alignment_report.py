# ruff: noqa: E501, RUF001
"""Offline statistics, figures, and local HTML for the LIBERO alignment audit."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

from experiments.robot.libero.query_review import (
    ACTION_VALUES,
    ALL_OUTCOMES,
    FAITHFULNESS_VALUES,
    PROPOSAL_VERSION,
    REASONING_VALUES,
    build_review_queues,
    episode_milestones,
    local_state_change,
    propose_all,
    review_statistics,
)
from experiments.robot.libero.alignment_evaluator import parse_reasoning
from experiments.robot.libero.move_alignment_comparison import render_html as render_move_comparison_html
from experiments.robot.libero.move_alignment_sweep_comparison import render_html as render_move_sweep_html
from experiments.robot.libero.query_review import (
    query_id as review_query_id,
)

plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]

EXPERIMENT_LABEL = "Initial LIBERO-90 qualitative audit: one trial per task"
LEVELS = ("kinematic", "target", "subtask", "any")
DISPLAY = {"kinematic": "Kinematic", "target": "Target", "subtask": "Subtask", "any": "Any level"}
COMPACT_REVIEW_TASK_IDS = (0, 3, 9, 18, 35, 46, 73, 81, 84)
EXCLUDED_MANUAL_REVIEW_TASK_IDS = (64,)
REPORT_KINEMATIC_DEADBAND = 0.003


def _rescore_record_kinematics(record: Dict[str, Any]) -> None:
    """Apply the reviewed set score to an existing rollout record in place."""
    claims = record.get("claims", {}).get("motion_axes", {})
    before = record.get("state_before", {}).get("ee_position")
    after = record.get("state_after", {}).get("ee_position")
    if not claims or before is None or after is None:
        return
    delta = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    axis_index = {"x": 0, "y": 1, "z": 2}
    categories = {"match": [], "missing": [], "opposite": [], "extra": []}
    per_axis = {}
    actual = {
        axis: int(np.sign(delta[index]))
        for axis, index in axis_index.items()
        if abs(float(delta[index])) > REPORT_KINEMATIC_DEADBAND
    }
    for axis, expected_value in sorted(claims.items()):
        if axis not in axis_index or int(expected_value) not in (-1, 1):
            continue
        expected = int(expected_value)
        observed = actual.get(axis)
        category = "missing" if observed is None else "match" if observed == expected else "opposite"
        categories[category].append(axis)
        per_axis[axis] = {
            "expected_direction": expected,
            "measured_displacement": float(delta[axis_index[axis]]),
            "threshold": REPORT_KINEMATIC_DEADBAND,
            "category": category,
            "score": 1 if category == "match" else -1,
        }
    if not per_axis:
        return
    for axis in actual:
        if axis not in per_axis:
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
    alignment = record["alignment"]
    alignment["kinematic_score"] = numerator / denominator
    diagnostics = alignment.setdefault("diagnostics", {})
    labels = [
        label
        for label in diagnostics.get("failure_labels", [])
        if label not in {"kinematic_opposite", "kinematic_no_motion"}
    ]
    if categories["opposite"]:
        labels.append("kinematic_opposite")
    if len(categories["missing"]) == len(per_axis):
        labels.append("kinematic_no_motion")
    diagnostics["failure_labels"] = labels
    diagnostics["kinematic"] = {
        "delta_ee": delta.tolist(),
        "per_axis": per_axis,
        "categories": categories,
        "counts": counts,
        "contributions": contributions,
        "numerator": numerator,
        "denominator": denominator,
        "formula": "(match - missing - opposite - extra) / total",
        "offline_rescore": True,
    }
    component_scores = [alignment.get(f"{level}_score") for level in ("kinematic", "target", "subtask")]
    available = [float(score) for score in component_scores if score is not None]
    alignment["aggregate_score"] = sum(available) / len(available) if available else None


def _fit_episode_projection(records: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Fit the fixed camera's 3D-to-pixel projective map from logged EE tracks."""
    pairs = [
        (world, pixel)
        for record in records
        for world, pixel in zip(
            record.get("executed_ee_trajectory", []),
            record.get("executed_ee_trajectory_pixels", []),
        )
    ]
    if len(pairs) < 6:
        return None
    world = np.asarray([item[0] for item in pairs], dtype=float)
    pixels = np.asarray([item[1] for item in pairs], dtype=float)
    if world.shape[1:] != (3,) or pixels.shape[1:] != (2,):
        return None
    design, values = [], []
    for (x, y, z), (u, v) in zip(world, pixels):
        point = [x, y, z, 1.0]
        design.append(point + [0.0] * 4 + [-u * x, -u * y, -u * z])
        values.append(u)
        design.append([0.0] * 4 + point + [-v * x, -v * y, -v * z])
        values.append(v)
    design_array = np.asarray(design, dtype=float)
    if np.linalg.matrix_rank(design_array) < 11:
        return None
    coefficients = np.linalg.lstsq(design_array, np.asarray(values), rcond=None)[0]
    matrix = np.concatenate([coefficients, [1.0]]).reshape(3, 4)
    homogeneous = np.column_stack([world, np.ones(len(world))]) @ matrix.T
    predicted = homogeneous[:, :2] / homogeneous[:, 2:3]
    residuals = np.linalg.norm(predicted - pixels, axis=1)
    return {
        "matrix": matrix,
        "median_residual_px": float(np.median(residuals)),
        "p90_residual_px": float(np.percentile(residuals, 90)),
    }


def _project_review_point(calibration: Optional[Mapping[str, Any]], point: Any) -> Optional[List[float]]:
    if calibration is None or point is None:
        return None
    position = np.asarray(point, dtype=float)
    if position.shape != (3,):
        return None
    homogeneous = np.asarray(calibration["matrix"], dtype=float) @ np.append(position, 1.0)
    if not np.isfinite(homogeneous).all() or abs(float(homogeneous[2])) < 1e-9:
        return None
    pixel = homogeneous[:2] / homogeneous[2]
    return pixel.tolist() if np.isfinite(pixel).all() else None


def _add_review_spatial_evidence(records: Sequence[Dict[str, Any]]) -> None:
    grouped = _groups(records)
    for episode_records in grouped.values():
        calibration = _fit_episode_projection(episode_records)
        if calibration is None or calibration["median_residual_px"] > 2.0:
            calibration = None
        for record in episode_records:
            before = record.get("state_before", {})
            after = record.get("state_after", {})
            target = record.get("alignment", {}).get("diagnostics", {}).get("target", {})
            target_name = target.get("resolved_object")
            before_target = before.get("objects", {}).get(target_name, {}) if target_name else {}
            after_target = after.get("objects", {}).get(target_name, {}) if target_name else {}
            trajectory_pixels = record.get("executed_ee_trajectory_pixels", [])
            record["_review_spatial_evidence"] = {
                "world_origin_pixel": _project_review_point(calibration, [0, 0, 0]),
                "ee_before_pixel": trajectory_pixels[0] if trajectory_pixels else None,
                "ee_after_pixel": trajectory_pixels[-1] if trajectory_pixels else None,
                "target_before_pixel": _project_review_point(calibration, before_target.get("position")),
                "projection_median_residual_px": calibration.get("median_residual_px") if calibration else None,
                "target_name": target_name,
                "ee_before": before.get("ee_position"),
                "ee_after": after.get("ee_position"),
                "target_before": before_target.get("position"),
                "target_after": after_target.get("position"),
                "distance_before": target.get("distance_before"),
                "distance_after": target.get("distance_after"),
                "progress": target.get("progress"),
                "progress_scale": target.get("progress_scale"),
            }


def _score(record: Mapping[str, Any], level: str) -> Optional[float]:
    if level == "any":
        return None
    value = record["alignment"].get(f"{level}_score")
    return None if value is None else float(value)


def is_evaluable(record: Mapping[str, Any], level: str) -> bool:
    if level == "any":
        return any(is_evaluable(record, item) for item in LEVELS[:3])
    return _score(record, level) is not None


def is_mismatch(record: Mapping[str, Any], level: str) -> bool:
    """Apply one primary mismatch definition per evaluator component."""
    if level == "kinematic":
        score = _score(record, level)
        return score is not None and score < 1.0
    if level == "target":
        score = _score(record, level)
        labels = record["alignment"].get("diagnostics", {}).get("failure_labels", [])
        return score is not None and (score < 0.0 or "wrong_object_interaction" in labels)
    if level == "subtask":
        score = _score(record, level)
        return score is not None and score < 1.0
    return any(is_mismatch(record, item) for item in LEVELS[:3] if is_evaluable(record, item))


def _groups(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["episode_id"])].append(record)
    return groups


def level_statistics(records: Sequence[Mapping[str, Any]], level: str) -> Dict[str, Any]:
    evaluable = [record for record in records if is_evaluable(record, level)]
    mismatches = [record for record in evaluable if is_mismatch(record, level)]
    scores = [_score(record, level) for record in evaluable] if level != "any" else []
    scores = [value for value in scores if value is not None]
    episode_groups = _groups(records)
    evaluable_episodes = []
    mismatch_episodes = []
    fractions = []
    for episode_id, group in episode_groups.items():
        episode_evaluable = [record for record in group if is_evaluable(record, level)]
        if not episode_evaluable:
            continue
        evaluable_episodes.append(episode_id)
        count = sum(is_mismatch(record, level) for record in episode_evaluable)
        fractions.append(count / len(episode_evaluable))
        if count:
            mismatch_episodes.append(episode_id)
    failure_labels = [record["alignment"].get("diagnostics", {}).get("failure_labels", []) for record in records]
    return {
        "total_queries": len(records),
        "evaluable_queries": len(evaluable),
        "coverage": len(evaluable) / len(records) if records else None,
        "mismatch_count": len(mismatches),
        "query_mismatch_rate": len(mismatches) / len(evaluable) if evaluable else None,
        "mean_score": mean(scores) if scores else None,
        "median_score": median(scores) if scores else None,
        "episodes_with_evaluable_claims": len(evaluable_episodes),
        "episodes_with_mismatch": len(mismatch_episodes),
        "episode_mismatch_prevalence": len(mismatch_episodes) / len(evaluable_episodes) if evaluable_episodes else None,
        "mean_episode_mismatch_fraction": mean(fractions) if fractions else None,
        "parser_failures": sum("reasoning_parse_failure" in labels for labels in failure_labels),
        "unresolved_or_ambiguous_targets": sum("target_unresolved" in labels for labels in failure_labels),
        "unsupported_subtasks": sum("unsupported_subtask" in labels for labels in failure_labels),
    }


def compute_statistics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labels = [
        label for record in records for label in record["alignment"].get("diagnostics", {}).get("failure_labels", [])
    ]
    parser_failures = sum(
        "reasoning_parse_failure" in record["alignment"]["diagnostics"].get("failure_labels", []) for record in records
    )
    warnings = sum(bool(record.get("claims", {}).get("parse_warnings")) for record in records)
    unresolved = sum(
        "target_unresolved" in record["alignment"]["diagnostics"].get("failure_labels", []) for record in records
    )
    unsupported = sum(
        "unsupported_subtask" in record["alignment"]["diagnostics"].get("failure_labels", []) for record in records
    )
    return {
        "levels": {level: level_statistics(records, level) for level in LEVELS},
        "parser_failures": parser_failures,
        "queries_with_parser_warnings": warnings,
        "unresolved_or_ambiguous_targets": unresolved,
        "unsupported_subtasks": unsupported,
        "diagnostic_labels": dict(sorted(Counter(labels).items())),
    }


def _mode_value(records: Sequence[Mapping[str, Any]], level: str, mode: str) -> Optional[float]:
    stats = level_statistics(records, level)
    return stats[mode]


def cluster_bootstrap_difference(
    records: Sequence[Mapping[str, Any]],
    level: str,
    mode: str,
    *,
    replicates: int = 2000,
    seed: int = 20260901,
) -> Optional[Dict[str, float]]:
    """Bootstrap failed-minus-successful rates by resampling whole episodes."""
    groups = _groups(records)
    by_status = {
        status: [group for group in groups.values() if bool(group[-1]["episode_success"]) is status]
        for status in (True, False)
    }
    if min(len(by_status[True]), len(by_status[False])) < 2:
        return None
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        sampled = {}
        for status in (True, False):
            source = by_status[status]
            indices = rng.integers(0, len(source), size=len(source))
            sampled[status] = [record for index in indices for record in source[int(index)]]
        success_value = _mode_value(sampled[True], level, mode)
        failed_value = _mode_value(sampled[False], level, mode)
        if success_value is not None and failed_value is not None:
            differences.append(failed_value - success_value)
    if not differences:
        return None
    observed_success = _mode_value([r for r in records if r["episode_success"]], level, mode)
    observed_failed = _mode_value([r for r in records if not r["episode_success"]], level, mode)
    return {
        "difference": float(observed_failed - observed_success),
        "low": float(np.percentile(differences, 2.5)),
        "high": float(np.percentile(differences, 97.5)),
    }


def cluster_bootstrap_interval(
    records: Sequence[Mapping[str, Any]],
    level: str,
    mode: str,
    *,
    replicates: int = 2000,
    seed: int = 20260901,
) -> Optional[tuple[float, float]]:
    groups = list(_groups(records).values())
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        indices = rng.integers(0, len(groups), size=len(groups))
        sample = [record for index in indices for record in groups[int(index)]]
        value = _mode_value(sample, level, mode)
        if value is not None:
            values.append(value)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else None


def comparison_table(records: Sequence[Mapping[str, Any]], mode: str, bootstrap_replicates: int) -> List[Dict[str, Any]]:
    success = [record for record in records if record["episode_success"]]
    failed = [record for record in records if not record["episode_success"]]
    rows = []
    for level in LEVELS:
        overall = level_statistics(records, level)
        success_stats = level_statistics(success, level)
        failed_stats = level_statistics(failed, level)
        success_value = success_stats[mode]
        failed_value = failed_stats[mode]
        numerator_key = "mismatch_count" if mode == "query_mismatch_rate" else "episodes_with_mismatch"
        denominator_key = "evaluable_queries" if mode == "query_mismatch_rate" else "episodes_with_evaluable_claims"
        interval = cluster_bootstrap_difference(records, level, mode, replicates=bootstrap_replicates)
        rows.append(
            {
                "level": DISPLAY[level],
                "coverage": overall["coverage"],
                "coverage_count": overall["evaluable_queries"],
                "coverage_denominator": overall["total_queries"],
                "overall": overall[mode],
                "overall_numerator": overall[
                    "mismatch_count" if mode == "query_mismatch_rate" else "episodes_with_mismatch"
                ]
                if mode != "mean_episode_mismatch_fraction"
                else None,
                "overall_denominator": overall[
                    "evaluable_queries" if mode == "query_mismatch_rate" else "episodes_with_evaluable_claims"
                ]
                if mode != "mean_episode_mismatch_fraction"
                else overall["episodes_with_evaluable_claims"],
                "successful": success_value,
                "successful_numerator": success_stats[numerator_key]
                if mode != "mean_episode_mismatch_fraction"
                else None,
                "successful_denominator": success_stats[denominator_key],
                "failed": failed_value,
                "failed_numerator": failed_stats[numerator_key] if mode != "mean_episode_mismatch_fraction" else None,
                "failed_denominator": failed_stats[denominator_key],
                "difference": interval["difference"]
                if interval
                else (failed_value - success_value if failed_value is not None and success_value is not None else None),
                "ci": [interval["low"], interval["high"]] if interval else None,
            }
        )
    return rows


def mismatch_overlap(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for record in records:
        active = [DISPLAY[level] for level in LEVELS[:3] if is_evaluable(record, level) and is_mismatch(record, level)]
        counts[" + ".join(active) if active else "No detected mismatch"] += 1
    return dict(sorted(counts.items()))


def _reliable(record: Mapping[str, Any]) -> bool:
    if record.get("claims", {}).get("parse_warnings"):
        return False
    if record.get("claims", {}).get("target_object"):
        status = record["alignment"].get("diagnostics", {}).get("target_resolution", {}).get("before", {}).get("status")
        if status != "resolved":
            return False
    return bool(record.get("frames")) and bool(record.get("executed_ee_trajectory_pixels"))


def select_examples(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    reliable = [record for record in records if _reliable(record)]

    def order(record):
        mismatch_count = sum(is_mismatch(record, level) for level in LEVELS[:3] if is_evaluable(record, level))
        aggregate = record["alignment"].get("aggregate_score")
        return (
            bool(record["episode_success"]),
            -mismatch_count,
            aggregate if aggregate is not None else 2,
            int(record["task_id"]),
            int(record["policy_query_index"]),
        )

    ordered = sorted(reliable, key=order)
    figure1 = [record for record in ordered if any(is_mismatch(record, level) for level in LEVELS[:3])][:6]
    timeline_episodes = {record["episode_id"] for record in figure1[:3]}
    explanatory = {}
    used_explanatory = set()
    for level in LEVELS[:3]:
        candidates = sorted(
            [record for record in ordered if is_mismatch(record, level) and _record_id(record) not in used_explanatory],
            key=lambda record: (
                sum(is_mismatch(record, other) for other in LEVELS[:3] if other != level),
                order(record),
            ),
        )[:4]
        explanatory[level] = candidates
        used_explanatory.update(_record_id(record) for record in candidates)
    episode_groups = _groups(records)
    failed_clean_episodes = {
        episode_id
        for episode_id, group in episode_groups.items()
        if not group[-1]["episode_success"] and not any(is_mismatch(record, "any") for record in group)
    }
    categories = {
        "figure1": figure1,
        **explanatory,
        "aligned": [record for record in ordered if is_evaluable(record, "any") and not is_mismatch(record, "any")][:4],
        "multi_level": [record for record in ordered if sum(is_mismatch(record, level) for level in LEVELS[:3]) >= 2][
            :6
        ],
        "successful_temporary": [
            record for record in ordered if record["episode_success"] and is_mismatch(record, "any")
        ][:6],
        "failed_no_detected": [record for record in ordered if record["episode_id"] in failed_clean_episodes][:6],
        "timelines": sorted(
            [record for record in records if record["episode_id"] in timeline_episodes],
            key=lambda record: (str(record["episode_id"]), int(record["policy_query_index"])),
        ),
    }
    return categories


def _percent(value: Optional[float]) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _number(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.3f}"


def _percentage_points(value: Optional[float]) -> str:
    return "Unavailable" if value is None else f"{value * 100:+.1f} pp"


def _percentage_point_interval(value: Optional[Sequence[float]]) -> str:
    if value is None:
        return "Unavailable: group too small"
    return f"{value[0] * 100:+.1f} to {value[1] * 100:+.1f} pp"


def _load_browser_manual_reviews(output_dir: Path) -> tuple[Dict[str, Any], Optional[Path]]:
    """Load the preserved final browser labels without depending on live localStorage."""
    snapshot = output_dir / "annotations" / "manual_labels_browser_snapshot_2026-09-04.json"
    if not snapshot.exists():
        return {}, None
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    reviews = payload.get("values", {}).get("ecotQueryTaxonomyReviewsV2", {})
    return (reviews if isinstance(reviews, dict) else {}), snapshot


def _move_cosine(record: Mapping[str, Any]) -> Optional[float]:
    claims = parse_reasoning(str(record.get("reasoning_raw", "")))
    expected = np.asarray(
        [claims.motion_axes.get(axis, 0) for axis in ("x", "y", "z")], dtype=float
    )
    before = record.get("state_before", {}).get("ee_position")
    after = record.get("state_after", {}).get("ee_position")
    if before is None or after is None:
        return None
    actual = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    expected_norm = float(np.linalg.norm(expected))
    actual_norm = float(np.linalg.norm(actual))
    if expected_norm == 0:
        return 0.0
    if actual_norm == 0:
        return None
    return float(np.dot(expected, actual) / (expected_norm * actual_norm))


def _normalized_subtask(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _plan_steps(reasoning: str) -> List[str]:
    match = re.search(r"\bPLAN\s*:\s*(.*?)(?=\s+(?:VISIBLE\s+OBJECTS|OBJECTS)\s*:|$)", reasoning, re.I | re.S)
    if not match:
        return []
    return [item.group(2).strip() for item in re.finditer(r"(?:^|,\s*)(\d+)\.\s*(.*?)(?=,\s*\d+\.|$)", match.group(1))]


def _default_subtask_rewards(
    episode_records: Sequence[Mapping[str, Any]], reviews: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Mirror the reviewed UI's period/checkpoint reward rule (version 3)."""
    periods: List[Dict[str, Any]] = []
    for record in episode_records:
        label = record.get("claims", {}).get("subtask") or "No stated subtask"
        key = _normalized_subtask(label)
        if not periods or periods[-1]["key"] != key:
            periods.append({"key": key, "label": label, "records": [record]})
        else:
            periods[-1]["records"].append(record)

    rewards: Dict[str, Dict[str, Any]] = {}
    success = bool(episode_records[0].get("episode_success")) if episode_records else False
    for index, period in enumerate(periods):
        next_period = periods[index + 1] if index + 1 < len(periods) else None
        checkpoint_record = next_period["records"][0] if next_period else period["records"][-1]
        kind = "transition" if next_period else "final"
        suggested_units = 1
        if kind == "final" and success:
            steps = _plan_steps(str(period["records"][-1].get("reasoning_raw", "")))
            try:
                current_index = [_normalized_subtask(step) for step in steps].index(period["key"])
                suggested_units = len(steps) - current_index
            except ValueError:
                pass
        returns_later = any(later["key"] == period["key"] for later in periods[index + 1 :])
        suggested_judgment = "incomplete" if returns_later or (kind == "final" and not success) else "complete"
        checkpoint_id = review_query_id(checkpoint_record)
        stored = reviews.get(checkpoint_id, {}).get("subtask_checkpoint", {})
        saved = stored if stored.get("rule_version") == 3 else {}
        judgment = saved.get("judgment") if saved.get("judgment") in {"complete", "incomplete"} else suggested_judgment
        if judgment == "complete":
            completed_units = max(1, int(saved.get("completed_units") or suggested_units))
        else:
            completed_units = 0
        checkpoint_score = completed_units if judgment == "complete" else -1
        distributed_score = 1 if kind == "final" and judgment == "complete" else checkpoint_score
        per_query = distributed_score / len(period["records"])
        terminal_bonus = completed_units - 1 if kind == "final" and judgment == "complete" else 0
        for position, record in enumerate(period["records"]):
            reward = per_query + (terminal_bonus if position == len(period["records"]) - 1 else 0)
            rewards[review_query_id(record)] = {
                "reward": reward,
                "checkpoint_judgment": judgment,
                "completed_subtasks": completed_units,
                "period_queries": len(period["records"]),
                "checkpoint_source": "saved" if saved else "accepted_default",
            }
    return rewards


def _materialize_manual_labels(
    records: Sequence[Mapping[str, Any]], output_dir: Path
) -> tuple[List[Dict[str, Any]], Optional[Path]]:
    """Freeze effective labels: browser overrides layered on accepted UI defaults."""
    reviews, snapshot = _load_browser_manual_reviews(output_dir)
    retained = [record for record in records if int(record["task_id"]) in COMPACT_REVIEW_TASK_IDS]
    subtask_rewards: Dict[str, Dict[str, Any]] = {}
    for episode_records in _groups(retained).values():
        ordered = sorted(episode_records, key=lambda row: int(row["policy_query_index"]))
        subtask_rewards.update(_default_subtask_rewards(ordered, reviews))

    effective: List[Dict[str, Any]] = []
    for record in sorted(retained, key=lambda row: (int(row["task_id"]), int(row["policy_query_index"]))):
        record_id = review_query_id(record)
        saved = reviews.get(record_id, {})
        cosine = _move_cosine(record)
        move_default = "N/A" if cosine is None else "T" if cosine >= 0.5 else "F"
        reward_info = subtask_rewards[record_id]
        subtask_default = "T" if reward_info["reward"] > 0 else "F"
        move = saved.get("kinematic_alignment") if saved.get("kinematic_alignment") in {"T", "F", "N/A"} else move_default
        subtask = saved.get("subtask_alignment") if saved.get("subtask_alignment") in {"T", "F", "N/A"} else subtask_default
        effective.append(
            {
                "id": record_id,
                "task_id": int(record["task_id"]),
                "episode_id": str(record["episode_id"]),
                "query": int(record["policy_query_index"]),
                "episode_success": bool(record["episode_success"]),
                "move_alignment": move,
                "move_cosine": cosine,
                "move_source": "saved" if move != move_default or "kinematic_alignment" in saved else "accepted_default",
                "subtask_alignment": subtask,
                "subtask_source": "saved" if subtask != subtask_default or "subtask_alignment" in saved else "accepted_default",
                **reward_info,
            }
        )

    if snapshot is None:
        return effective, None
    destination = output_dir / "annotations" / "manual_labels_effective_9episodes_2026-09-04.json"
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "captured_at": "2026-09-04",
                "source_browser_snapshot": snapshot.name,
                "retained_task_ids": list(COMPACT_REVIEW_TASK_IDS),
                "excluded_task_ids": list(EXCLUDED_MANUAL_REVIEW_TASK_IDS),
                "label_semantics": {
                    "move": "saved human label, otherwise accepted cosine default (T iff cosine >= 0.5)",
                    "subtask": "saved human label, otherwise accepted period/checkpoint reward default (T iff reward > 0)",
                    "any_level": "F if either evaluable retained level is F; otherwise T when at least one is T",
                },
                "records": effective,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return effective, destination


def _manual_query_comparison(rows: Sequence[Mapping[str, Any]], replicates: int, seed: int = 20260904) -> List[Dict[str, Any]]:
    def label(row: Mapping[str, Any], level: str) -> str:
        if level == "move":
            return str(row["move_alignment"])
        if level == "subtask":
            return str(row["subtask_alignment"])
        values = {str(row["move_alignment"]), str(row["subtask_alignment"])}
        return "F" if "F" in values else "T" if "T" in values else "N/A"

    def counts(group: Sequence[Mapping[str, Any]], level: str) -> tuple[int, int, Optional[float]]:
        values = [label(row, level) for row in group]
        evaluable = [value for value in values if value in {"T", "F"}]
        mismatches = sum(value == "F" for value in evaluable)
        return mismatches, len(evaluable), mismatches / len(evaluable) if evaluable else None

    episode_groups = list(_groups(rows).values())
    success_groups = [group for group in episode_groups if bool(group[0]["episode_success"])]
    failed_groups = [group for group in episode_groups if not bool(group[0]["episode_success"])]
    rng = np.random.default_rng(seed)
    result = []
    for level, display in (("move", "Move"), ("subtask", "Subtask"), ("any", "Any level")):
        overall_n, overall_d, overall_rate = counts(rows, level)
        successful_rows = [row for group in success_groups for row in group]
        failed_rows = [row for group in failed_groups for row in group]
        success_n, success_d, success_rate = counts(successful_rows, level)
        failed_n, failed_d, failed_rate = counts(failed_rows, level)
        difference = None if success_rate is None or failed_rate is None else failed_rate - success_rate
        samples = []
        if len(success_groups) >= 2 and len(failed_groups) >= 2:
            for _ in range(replicates):
                sampled_success = [success_groups[index] for index in rng.integers(0, len(success_groups), len(success_groups))]
                sampled_failed = [failed_groups[index] for index in rng.integers(0, len(failed_groups), len(failed_groups))]
                sample_success_rate = counts([row for group in sampled_success for row in group], level)[2]
                sample_failed_rate = counts([row for group in sampled_failed for row in group], level)[2]
                if sample_success_rate is not None and sample_failed_rate is not None:
                    samples.append(sample_failed_rate - sample_success_rate)
        ci = list(np.percentile(samples, [2.5, 97.5])) if samples else None
        result.append(
            {
                "level": display,
                "coverage_count": overall_d,
                "coverage_denominator": len(rows),
                "coverage": overall_d / len(rows) if rows else None,
                "overall_numerator": overall_n,
                "overall_denominator": overall_d,
                "overall": overall_rate,
                "successful_numerator": success_n,
                "successful_denominator": success_d,
                "successful_coverage_count": success_d,
                "successful_coverage_denominator": len(successful_rows),
                "successful_coverage": success_d / len(successful_rows) if successful_rows else None,
                "successful": success_rate,
                "failed_numerator": failed_n,
                "failed_denominator": failed_d,
                "failed_coverage_count": failed_d,
                "failed_coverage_denominator": len(failed_rows),
                "failed_coverage": failed_d / len(failed_rows) if failed_rows else None,
                "failed": failed_rate,
                "difference": difference,
                "ci": ci,
            }
        )
    return result


def _save_figure(fig, figures_dir: Path, stem: str) -> Dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{stem}.png"
    pdf = figures_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": f"figures/{png.name}", "pdf": f"figures/{pdf.name}"}


def generate_manual_teaser_candidates(
    records: Sequence[Mapping[str, Any]],
    manual_labels: Sequence[Mapping[str, Any]],
    run_root: Path,
    figures_dir: Path,
) -> List[Dict[str, Any]]:
    """Create annotated teaser candidates from strong human-retained Move mismatches."""
    axis_words = {
        ("x", -1): "FORWARD",
        ("x", 1): "BACK",
        ("y", -1): "LEFT",
        ("y", 1): "RIGHT",
        ("z", -1): "DOWN",
        ("z", 1): "UP",
    }

    def intuitive_motion_summary(expected: Sequence[int], delta_mm: np.ndarray) -> tuple[str, str, str]:
        stated = " + ".join(
            axis_words[(axis, int(np.sign(value)))]
            for axis, value in zip(("x", "y", "z"), expected)
            if value
        ) or "NO DIRECTIONAL MOVE"
        dominant_index = int(np.argmax(np.abs(delta_mm)))
        dominant_axis = ("x", "y", "z")[dominant_index]
        dominant_sign = int(np.sign(delta_mm[dominant_index]))
        actual = axis_words[(dominant_axis, dominant_sign)] if dominant_sign else "NO MOTION"
        actual_amount = abs(float(delta_mm[dominant_index]))
        opposing = [
            (abs(float(delta)), axis_words[(axis, int(np.sign(delta)))])
            for axis, wanted, delta in zip(("x", "y", "z"), expected, delta_mm)
            if wanted and delta and wanted * delta < 0
        ]
        if opposing:
            opposing_amount, opposing_word = max(opposing)
            if opposing_word == actual:
                contrast = f"opposing the stated {axis_words[(dominant_axis, int(np.sign(expected[dominant_index])))]}"
            else:
                contrast = f"its {opposing_word} component ({opposing_amount:.1f} mm) opposes the stated direction"
        else:
            contrast = "the overall vector still disagrees with the stated direction"
        summary = f"Reasoning says {stated}; robot actually moved mostly {actual} ({actual_amount:.1f} mm), {contrast}."
        return summary, stated, actual

    record_by_id = {review_query_id(record): record for record in records}
    eligible = sorted(
        (
            label
            for label in manual_labels
            if label.get("move_alignment") == "F"
            and label.get("move_cosine") is not None
            and float(label["move_cosine"]) < 0
            and label.get("id") in record_by_id
        ),
        key=lambda label: float(label["move_cosine"]),
    )
    preferred_queries = (("A", 3, 11), ("D", 81, 37))
    eligible_by_query = {
        (int(label["task_id"]), int(label["query"])): label
        for label in eligible
    }
    selected = [
        (letter, eligible_by_query[(task_id, query)])
        for letter, task_id, query in preferred_queries
        if (task_id, query) in eligible_by_query
    ]
    concise_explanations = {
        (3, 11): {
            "reason": "For finishing the current subtask “grasp the butter”, the robot should move down to reach and grasp the butter.",
            "download": "To grasp the butter, moving down is locally sensible; the action instead moves up.",
        },
        (81, 37): {
            "reason": "For finishing the current subtask “release the black book into the front compartment of the desk caddy”, the robot should move down, forward, and right to align the book with the front compartment.",
            "download": "To release the book into the caddy, moving down, forward, and right is locally sensible; the action instead moves back.",
        },
    }

    candidates = []
    for letter, label in selected:
        record = record_by_id[str(label["id"])]
        reasoning = str(record.get("reasoning_raw", ""))
        move_reasoning_match = re.search(
            r"\bMOVE\s+REASONING\s*:\s*(.*?)(?=\s+MOVE\s*:|$)", reasoning, re.I | re.S
        )
        move_reasoning = move_reasoning_match.group(1).strip() if move_reasoning_match else "Unavailable"
        move_match = re.search(r"\bMOVE\s*:\s*(.*?)(?=\s+GRIPPER(?:\s+POSITION)?\s*:|$)", reasoning, re.I | re.S)
        move_text = move_match.group(1).strip() if move_match else "No parsed MOVE"
        claims = parse_reasoning(reasoning)
        subtask = str(claims.subtask or "current subtask")
        expected = [int(claims.motion_axes.get(axis, 0)) for axis in ("x", "y", "z")]
        before_position = np.asarray(record["state_before"]["ee_position"], dtype=float)
        after_position = np.asarray(record["state_after"]["ee_position"], dtype=float)
        delta_mm = (after_position - before_position) * 1000
        motion_summary, stated_direction, actual_direction = intuitive_motion_summary(expected, delta_mm)
        before_path = run_root / str(record["frames"]["before"])
        before_image = Image.open(before_path).convert("RGB")

        fig, axis = plt.subplots(figsize=(7.2, 7.6))
        axis.imshow(before_image)
        axis.axis("off")
        axis.text(
            0.02, 0.98, f"REASONING → {stated_direction}", transform=axis.transAxes,
            ha="left", va="top", fontsize=11, weight="bold", color="white",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#16883f", "edgecolor": "white", "alpha": 0.94},
        )
        axis.text(
            0.02, 0.90, f"ACTION → {actual_direction}", transform=axis.transAxes,
            ha="left", va="top", fontsize=11, weight="bold", color="white",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#d62728", "edgecolor": "white", "alpha": 0.94},
        )

        expected_pixels = record.get("expected_motion_pixels") or []
        if len(expected_pixels) >= 2:
            start = np.asarray(expected_pixels[0], dtype=float)
            end = np.asarray(expected_pixels[-1], dtype=float)
            direction = end - start
            length = float(np.linalg.norm(direction))
            visible_end = end if length == 0 else start + direction / length * max(length, 55.0)
            axis.annotate(
                "",
                xy=visible_end,
                xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": "#16883f", "lw": 4, "mutation_scale": 18},
            )
        trajectory = record.get("executed_ee_trajectory_pixels") or []
        if len(trajectory) >= 2:
            x_values = [point[0] for point in trajectory]
            y_values = [point[1] for point in trajectory]
            axis.plot(x_values, y_values, color="#d62728", lw=3, marker="o", markersize=2, alpha=0.82)
            start = np.asarray(trajectory[0], dtype=float)
            end = np.asarray(trajectory[-1], dtype=float)
            direction = end - start
            length = float(np.linalg.norm(direction))
            if length == 0:
                fallback_vectors = {
                    "UP": (0.0, -55.0), "DOWN": (0.0, 55.0),
                    "LEFT": (-55.0, 0.0), "RIGHT": (55.0, 0.0),
                    "FORWARD": (-40.0, 40.0), "BACK": (40.0, -40.0),
                }
                visible_end = start + np.asarray(fallback_vectors.get(actual_direction, (40.0, -40.0)))
                axis.annotate(
                    "",
                    xy=visible_end,
                    xytext=start,
                    arrowprops={"arrowstyle": "-|>", "color": "#d62728", "lw": 4, "mutation_scale": 18},
                )
            else:
                visible_end = start + direction / length * max(length, 55.0)
                axis.annotate(
                    "",
                    xy=visible_end,
                    xytext=start,
                    arrowprops={"arrowstyle": "-|>", "color": "#d62728", "lw": 4, "mutation_scale": 18},
                )

        cosine = float(label["move_cosine"])
        query_key = (int(record["task_id"]), int(record["policy_query_index"]))
        explanation = concise_explanations[query_key]
        stem = f"teaser_candidate_{letter.lower()}_task{record['task_id']}_q{record['policy_query_index']}"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        files = _save_figure(fig, figures_dir, stem)
        candidates.append(
            {
                **files,
                "preview": files["png"],
                "letter": letter,
                "task_id": int(record["task_id"]),
                "query": int(record["policy_query_index"]),
                "success": bool(record["episode_success"]),
                "move": move_text,
                "move_reasoning": move_reasoning,
                "subtask": subtask,
                "motion_summary": motion_summary,
                "stated_direction": stated_direction,
                "actual_direction": actual_direction,
                "cosine": cosine,
                "delta_mm": delta_mm.tolist(),
                "reason_explanation": explanation["reason"],
            }
        )
    return candidates


def generate_quantitative_figures(records: Sequence[Mapping[str, Any]], output_dir: Path) -> Dict[str, Dict[str, str]]:
    figures = {}
    stats = compute_statistics(records)["levels"]
    levels = list(LEVELS)
    colors = ["#4472c4", "#ed7d31", "#70ad47", "#7f6000"]

    def bar_figure(field, denominator_field, stem, title):
        values = [stats[level][field] or 0 for level in levels]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar([DISPLAY[level] for level in levels], np.asarray(values) * 100, color=colors)
        for bar, level in zip(bars, levels):
            numerator = stats[level]["mismatch_count" if field == "query_mismatch_rate" else "episodes_with_mismatch"]
            denominator = stats[level][denominator_field]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{numerator}/{denominator}",
                ha="center",
                fontsize=9,
            )
        ax.set_ylabel("Mismatch (%)")
        ax.set_ylim(0, max(10, max(values, default=0) * 120))
        ax.set_title(title + "\n90 tasks × 1 trial per task")
        ax.grid(axis="y", alpha=0.25)
        return _save_figure(fig, output_dir, stem)

    figures["query_mismatch"] = bar_figure(
        "query_mismatch_rate", "evaluable_queries", "query_mismatch", "Query-level mismatch"
    )
    figures["episode_prevalence"] = bar_figure(
        "episode_mismatch_prevalence",
        "episodes_with_evaluable_claims",
        "episode_prevalence",
        "Episode-level mismatch prevalence",
    )

    groups = [
        ("Successful", [r for r in records if r["episode_success"]]),
        ("Failed", [r for r in records if not r["episode_success"]]),
    ]
    x = np.arange(len(levels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for index, (name, group) in enumerate(groups):
        values = [level_statistics(group, level)["query_mismatch_rate"] or 0 for level in levels]
        intervals = [cluster_bootstrap_interval(group, level, "query_mismatch_rate") for level in levels]
        errors = np.asarray(
            [
                [0, 0] if interval is None else [(value - interval[0]) * 100, (interval[1] - value) * 100]
                for value, interval in zip(values, intervals)
            ]
        ).T
        bars = ax.bar(
            x + (index - 0.5) * width,
            np.asarray(values) * 100,
            width,
            label=f"{name} episodes",
            yerr=errors,
            capsize=3,
        )
        for bar, level in zip(bars, levels):
            s = level_statistics(group, level)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{s['mismatch_count']}/{s['evaluable_queries']}",
                ha="center",
                fontsize=7,
                rotation=90,
            )
    ax.set_xticks(x, [DISPLAY[level] for level in levels])
    ax.set_ylabel("Query mismatch (%)")
    ax.set_title("Preliminary success-versus-failure comparison\n90 tasks × 1 trial per task")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    figures["success_failure"] = _save_figure(fig, output_dir, "success_failure")

    by_task = defaultdict(list)
    for record in records:
        by_task[int(record["task_id"])].append(record)
    matrix = np.full((90, 3), np.nan)
    for task_id in range(90):
        group = by_task.get(task_id, [])
        for column, level in enumerate(LEVELS[:3]):
            evaluable = [r for r in group if is_evaluable(r, level)]
            if evaluable:
                matrix[task_id, column] = mean(float(is_mismatch(r, level)) for r in evaluable)
    fig, ax = plt.subplots(figsize=(6.5, 15))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    for task_id in range(90):
        group = by_task.get(task_id, [])
        for column, level in enumerate(LEVELS[:3]):
            task_stats = level_statistics(group, level)
            label = (
                f"{task_stats['mismatch_count']}/{task_stats['evaluable_queries']}"
                if task_stats["evaluable_queries"]
                else "—"
            )
            ax.text(column, task_id, label, ha="center", va="center", fontsize=3.5)
    ax.set_xticks(range(3), [DISPLAY[level] for level in LEVELS[:3]])
    ax.set_yticks(range(90), [str(index) for index in range(90)], fontsize=5)
    ax.set_ylabel("LIBERO-90 task ID (one observed episode)")
    ax.set_title("Per-task observed mismatch fraction—not a task rate")
    fig.colorbar(image, ax=ax, label="Observed fraction of evaluable queries mismatched")
    figures["task_heatmap"] = _save_figure(fig, output_dir, "task_audit_heatmap")

    overlap = mismatch_overlap(records)
    fig, ax = plt.subplots(figsize=(9, 5))
    items = sorted(overlap.items(), key=lambda item: (-item[1], item[0]))
    bars = ax.barh([item[0] for item in items], [item[1] for item in items], color="#5b9bd5")
    for bar, (_, count) in zip(bars, items):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{count}/{len(records)}", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("Queries")
    ax.set_title("Mismatch co-occurrence\nKinematic × Target × Subtask; 90 tasks × 1 trial")
    figures["overlap"] = _save_figure(fig, output_dir, "mismatch_overlap")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    primary = list(LEVELS[:3])
    covered = [stats[level]["evaluable_queries"] for level in primary]
    excluded = [len(records) - value for value in covered]
    ax.bar([DISPLAY[level] for level in primary], covered, label="Evaluated", color="#70ad47")
    ax.bar([DISPLAY[level] for level in primary], excluded, bottom=covered, label="Excluded", color="#c9c9c9")
    for index, _level in enumerate(primary):
        ax.text(index, len(records) + 2, f"{covered[index]}/{len(records)}", ha="center")
    ax.set_ylabel("Queries")
    ax.set_title("Evaluator coverage and exclusions\n90 tasks × 1 trial per task")
    ax.legend()
    figures["coverage"] = _save_figure(fig, output_dir, "coverage_exclusions")
    return figures


def _resolve_frame(run_root: Path, record: Mapping[str, Any], label: str) -> Optional[Path]:
    frames = record.get("frames") or {}
    value = frames.get(label)
    if not value:
        return None
    path = run_root / value
    return path if path.exists() else None


def generate_example_figure(record: Mapping[str, Any], run_root: Path, destination: Path) -> bool:
    paths = [_resolve_frame(run_root, record, label) for label in ("before", "intermediate", "after")]
    if any(path is None for path in paths):
        return False
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for ax, path, label in zip(axes, paths, ("Before", "Intermediate", "After")):
        ax.imshow(Image.open(path))
        ax.set_title(label)
        ax.axis("off")
    trajectory = np.asarray(record.get("executed_ee_trajectory_pixels") or [], dtype=float)
    expected = np.asarray(record.get("expected_motion_pixels") or [], dtype=float)
    if len(trajectory) >= 2:
        axes[2].plot(trajectory[:, 0], trajectory[:, 1], color="red", linewidth=3, marker="o", markersize=2)
    if len(expected) == 2:
        delta = expected[1] - expected[0]
        axes[2].arrow(
            expected[0, 0],
            expected[0, 1],
            delta[0],
            delta[1],
            color="lime",
            width=1.5,
            head_width=8,
            length_includes_head=True,
        )
    scores = record["alignment"]
    labels = ", ".join(scores["diagnostics"].get("failure_labels", [])) or "none"
    claims = record["claims"]
    warnings = ", ".join(claims.get("parse_warnings", [])) or "none"
    exact_excerpt = record["reasoning_raw"][-500:]
    caption = (
        f"Instruction: {record['task_instruction']}\n"
        f"Exact reasoning excerpt: {exact_excerpt}\n"
        f"Claims — motion={claims.get('motion_axes')}; target={claims.get('target_object')}; "
        f"subtask={claims.get('subtask')}\n"
        f"Scores K/T/S: {scores.get('kinematic_score')} / {scores.get('target_score')} / "
        f"{scores.get('subtask_score')} · labels={labels} · warnings={warnings}"
    )
    fig.suptitle(
        f"Task {record['task_id']} · {record['episode_id']} · query {record['policy_query_index']} · "
        f"success={record['episode_success']}",
        fontsize=10,
    )
    fig.text(0.02, 0.02, textwrap.fill(caption, width=170, replace_whitespace=False), va="bottom", fontsize=7)
    fig.tight_layout(rect=(0, 0.27, 1, 0.93))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def _visible_object_boxes(reasoning: str) -> Dict[str, List[float]]:
    section = re.search(
        r"VISIBLE\s+OBJECTS\s*:\s*(.*?)(?:SUBTASK\s+REASON(?:ING)?\s*:|SUBTASK\s*:)",
        reasoning,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not section:
        return {}
    boxes = {}
    for name, index, coordinates in re.findall(
        r"(?:^|,\s*)([A-Za-z][A-Za-z0-9 _-]*?)\s+(\d+)\s*\[([^\]]+)\]",
        section.group(1),
    ):
        values = [float(value.strip()) for value in coordinates.split(",")]
        if len(values) == 4:
            # Released VISIBLE OBJECTS coordinates are [y_min, x_min, y_max, x_max],
            # while matplotlib rectangles expect [x_min, y_min, x_max, y_max].
            boxes[f"{name.strip().lower()} {index}"] = [values[1], values[0], values[3], values[2]]
    return boxes


def _intro_example(records: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    # This example was manually reviewed for both task-level reasoning correctness
    # and reasoning--action inconsistency. The primary metrics alone establish only
    # the latter and must not be used to infer that the reasoning itself is correct.
    for record in records:
        if (int(record["task_id"]), int(record["policy_query_index"])) != (73, 13):
            continue
        alignment = record["alignment"]
        labels = alignment.get("diagnostics", {}).get("failure_labels", [])
        if (
            alignment.get("kinematic_score") == 1.0
            and alignment.get("subtask_score") == 0.0
            and "subtask_not_completed" in labels
            and not record.get("claims", {}).get("parse_warnings")
        ):
            target_name = alignment.get("diagnostics", {}).get("target", {}).get("resolved_object")
            after_target = record.get("state_after", {}).get("objects", {}).get(target_name, {})
            if (
                after_target.get("grasped") is True
                and record.get("state_after", {}).get("gripper_open") is False
                and "desk_caddy_1_front_contain_region" in after_target.get("contained_by", [])
            ):
                return record
    return None


def _incorrect_reasoning_intro_example(
    records: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Return the manually reviewed wrong-reasoning / task-correct-action example."""
    for record in records:
        if (int(record["task_id"]), int(record["policy_query_index"])) != (64, 5):
            continue
        diagnostics = record["alignment"].get("diagnostics", {})
        target_name = diagnostics.get("target", {}).get("resolved_object")
        wrong_names = diagnostics.get("wrong_object_interaction", [])
        after_objects = record.get("state_after", {}).get("objects", {})
        if (
            target_name == "akita_black_bowl_1"
            and wrong_names == ["akita_black_bowl_2"]
            and after_objects.get("akita_black_bowl_2", {}).get("grasped") is True
            and not record.get("claims", {}).get("parse_warnings")
        ):
            return record
    return None


def _format_reasoning_for_figure(reasoning: str, width: int = 165) -> str:
    """Wrap the complete policy reasoning while keeping every tagged section visible."""
    sectioned = re.sub(
        r"\s+(?=(?:PLAN|VISIBLE\s+OBJECTS|(?<!VISIBLE\s)OBJECTS|SUBTASK\s+REASON(?:ING)?|SUBTASK|"
        r"MOVE\s+REASON(?:ING)?|MOVE|GRIPPER(?:\s+POSITION)?|ACTION)\s*:)",
        "\n",
        reasoning.strip(),
        flags=re.IGNORECASE,
    )
    return "\n".join(
        textwrap.fill(
            section.strip(),
            width=width,
            subsequent_indent="  ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        for section in sectioned.splitlines()
        if section.strip()
    )


def generate_incorrect_reasoning_intro_figure(
    record: Mapping[str, Any], run_root: Path, figures_dir: Path
) -> Optional[Dict[str, str]]:
    """Render Failure Mode 1: incorrect reasoning followed by a task-correct action."""
    before_path = _resolve_frame(run_root, record, "before")
    after_path = _resolve_frame(run_root, record, "after")
    trajectory = np.asarray(record.get("executed_ee_trajectory_pixels", []), dtype=float)
    if before_path is None or after_path is None or trajectory.ndim != 2 or len(trajectory) < 2:
        return None

    # Manually reviewed against simulator identities for task 64 / query 5.
    left_bowl_box = [52.0, 105.0, 97.0, 149.0]
    right_bowl_box = [109.0, 109.0, 148.0, 153.0]
    alignment = record["alignment"]
    full_reasoning = _format_reasoning_for_figure(record.get("reasoning_raw", ""))
    fig = plt.figure(figsize=(18, 12), facecolor="white")
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=[0.9, 1.15, 1.15],
        height_ratios=[0.52, 1, 0.16],
        hspace=0.12,
    )
    reasoning_axis = fig.add_subplot(grid[0, :])
    text_axis = fig.add_subplot(grid[1, 0])
    before_axis = fig.add_subplot(grid[1, 1])
    after_axis = fig.add_subplot(grid[1, 2])
    banner_axis = fig.add_subplot(grid[2, :])

    reasoning_axis.axis("off")
    reasoning_axis.text(
        0,
        0.98,
        "FULL POLICY REASONING — INCORRECT STACKING ORDER",
        fontsize=18,
        fontweight="bold",
        color="#9a6700",
        va="top",
    )
    reasoning_axis.text(
        0,
        0.82,
        full_reasoning,
        fontsize=11.5,
        family="monospace",
        linespacing=1.35,
        va="top",
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#fff4d6", "edgecolor": "#d69e00"},
    )

    text_axis.axis("off")
    text_axis.text(0, 0.98, "TASK VS. REASONING", fontsize=19, fontweight="bold", color="#9a6700", va="top")
    explanation = (
        "TASK: stack the RIGHT bowl on the LEFT bowl.\n\n"
        "REASONING: move the LEFT bowl onto the RIGHT bowl—the stacking relation is reversed.\n\n"
        "ACTION: the simulator reports the RIGHT bowl newly grasped, which is the task-correct source object."
    )
    text_axis.text(
        0,
        0.86,
        "\n\n".join(textwrap.fill(paragraph, 36) for paragraph in explanation.split("\n\n")),
        fontsize=13.5,
        linespacing=1.25,
        va="top",
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#fff4d6", "edgecolor": "#d69e00"},
    )
    text_axis.text(
        0,
        0.16,
        "Reasoning correctness: INCORRECT\n"
        "Action task consistency: CORRECT\n"
        "  (source-object selection)\n"
        "Reasoning–action relation: DISAGREE\n"
        f"Kinematic score: {alignment.get('kinematic_score'):.3f}",
        fontsize=12.5,
        fontweight="bold",
        va="bottom",
    )

    def rectangle(axis, box, color, label):
        x1, y1, x2, y2 = box
        axis.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=4, color=color))
        axis.text(
            0.03,
            0.95,
            label,
            color="white",
            fontsize=12.5,
            fontweight="bold",
            va="top",
            transform=axis.transAxes,
            bbox={"facecolor": color, "edgecolor": color, "pad": 3},
        )

    before_image = ImageEnhance.Brightness(Image.open(before_path).convert("RGB")).enhance(1.25)
    after_image = ImageEnhance.Brightness(Image.open(after_path).convert("RGB")).enhance(1.25)
    before_axis.imshow(before_image)
    rectangle(before_axis, left_bowl_box, "#d69e00", "INCORRECT REASONING TARGET: LEFT BOWL")
    before_axis.axis("off")

    after_axis.imshow(after_image)
    rectangle(after_axis, right_bowl_box, "#20a34a", "TASK-CORRECT ACTION: RIGHT BOWL GRASPED")
    after_axis.plot(trajectory[:, 0], trajectory[:, 1], color="#20c45a", lw=6)
    after_axis.scatter(trajectory[-1, 0], trajectory[-1, 1], s=130, color="#20c45a", marker="X")
    after_axis.axis("off")

    banner_axis.axis("off")
    banner_axis.text(
        0.5,
        0.62,
        "FAILURE MODE 1 — INCORRECT REASONING → TASK-CORRECT ACTION",
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#9a6700", "edgecolor": "#9a6700"},
    )
    banner_axis.text(
        0.5,
        0.08,
        f"Task {record['task_id']} · Query {record['policy_query_index']} · {record['task_instruction']} · "
        f"episode success={record['episode_success']}",
        ha="center",
        fontsize=13,
        color="#344054",
    )
    fig.suptitle("Task says RIGHT ON LEFT. Reasoning reverses it to LEFT ON RIGHT.", fontsize=25, fontweight="bold")
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / "figure1a_incorrect_reasoning_correct_action.png"
    pdf = figures_dir / "figure1a_incorrect_reasoning_correct_action.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": os_path(figures_dir.parent, png), "pdf": os_path(figures_dir.parent, pdf)}


def generate_intro_figure(
    record: Mapping[str, Any], run_root: Path, figures_dir: Path
) -> Optional[Dict[str, str]]:
    before_path = _resolve_frame(run_root, record, "before")
    after_path = _resolve_frame(run_root, record, "after")
    if before_path is None or after_path is None:
        return None
    diagnostics = record["alignment"].get("diagnostics", {})
    target_name = diagnostics.get("target", {}).get("resolved_object")
    boxes = _visible_object_boxes(record.get("reasoning_raw", ""))

    def box_for(simulator_name):
        return boxes.get(str(simulator_name).replace("_", " ").lower())

    target_box = box_for(target_name)
    receptacle_box = boxes.get("desk caddy 1")
    if target_box is None or receptacle_box is None:
        return None
    reasoning = record.get("reasoning_raw", "")
    full_reasoning = _format_reasoning_for_figure(reasoning)
    alignment = record["alignment"]
    fig = plt.figure(figsize=(18, 12), facecolor="white")
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=[0.9, 1.15, 1.15],
        height_ratios=[0.52, 1, 0.16],
        hspace=0.12,
    )
    reasoning_axis = fig.add_subplot(grid[0, :])
    text_axis = fig.add_subplot(grid[1, 0])
    before_axis = fig.add_subplot(grid[1, 1])
    after_axis = fig.add_subplot(grid[1, 2])
    banner_axis = fig.add_subplot(grid[2, :])
    reasoning_axis.axis("off")
    reasoning_axis.text(
        0,
        0.98,
        "FULL POLICY REASONING",
        fontsize=18,
        fontweight="bold",
        color="#237a3b",
        va="top",
    )
    reasoning_axis.text(
        0,
        0.82,
        full_reasoning,
        fontsize=11.5,
        family="monospace",
        linespacing=1.35,
        va="top",
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#eaf7ed", "edgecolor": "#39a852"},
    )
    text_axis.axis("off")
    text_axis.text(0, 0.98, "EXECUTED ACTION", fontsize=19, fontweight="bold", color="#b42318", va="top")
    action_text = (
        "By the end of the chunk, the book had entered the requested front-compartment region. The reasoning said "
        "RELEASE, but the gripper remained closed and the book remained grasped."
    )
    text_axis.text(
        0,
        0.86,
        textwrap.fill(action_text, 34),
        fontsize=15,
        va="top",
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#fff0ee", "edgecolor": "#d92d20"},
    )
    text_axis.text(
        0,
        0.34,
        f"Kinematic: ALIGNED ({alignment.get('kinematic_score'):.3f})\n"
        "Placement: CORRECT FRONT REGION\n"
        "Gripper after chunk: CLOSED\n"
        "Book after chunk: STILL GRASPED\n"
        f"Release subtask: INCOMPLETE ({alignment.get('subtask_score'):.3f})",
        fontsize=13.5,
        fontweight="bold",
        va="bottom",
    )

    def rectangle(axis, box, color, label):
        x1, y1, x2, y2 = box
        axis.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=4, color=color))
        axis.text(
            0.03,
            0.95,
            label,
            color="white",
            fontsize=13,
            fontweight="bold",
            va="top",
            transform=axis.transAxes,
            bbox={"facecolor": color, "edgecolor": color, "pad": 3},
        )

    def enhanced_frame(path: Path) -> Image.Image:
        image = Image.open(path).convert("RGB")
        image = ImageEnhance.Brightness(image).enhance(1.55)
        image = ImageEnhance.Contrast(image).enhance(1.18)
        return ImageEnhance.Sharpness(image).enhance(1.25)

    def label_nested_objects(axis, status_color: str, status: str) -> None:
        rectangle(axis, receptacle_box, status_color, status)
        book_x1, book_y1, book_x2, book_y2 = target_box
        axis.add_patch(
            plt.Rectangle(
                (book_x1, book_y1),
                book_x2 - book_x1,
                book_y2 - book_y1,
                facecolor="#ffd84d",
                edgecolor="#ffd84d",
                alpha=0.22,
                lw=4,
            )
        )
        axis.annotate(
            "BLACK BOOK",
            xy=((book_x1 + book_x2) / 2, (book_y1 + book_y2) / 2),
            xytext=(book_x2 + 12, book_y1 - 12),
            color="#1d2430",
            fontsize=11,
            fontweight="bold",
            arrowprops={"arrowstyle": "-|>", "color": "#ffd84d", "lw": 3},
            bbox={"facecolor": "#fff4ad", "edgecolor": "#ffd84d", "pad": 3},
        )
        caddy_x1, caddy_y1, caddy_x2, caddy_y2 = receptacle_box
        axis.text(
            caddy_x1 + 3,
            caddy_y2 + 9,
            "DESK CADDY",
            color="white",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "#176b87", "edgecolor": "#176b87", "pad": 3},
        )
        image_width, image_height = Image.open(before_path).size
        axis.set_xlim(max(0, caddy_x1 - 16), min(image_width, caddy_x2 + 30))
        axis.set_ylim(min(image_height, caddy_y2 + 28), max(0, caddy_y1 - 48))

    before_axis.imshow(enhanced_frame(before_path))
    label_nested_objects(before_axis, "#20a34a", "REASONING: RELEASE INTO FRONT REGION")
    before_axis.axis("off")

    after_axis.imshow(enhanced_frame(after_path))
    label_nested_objects(after_axis, "#d92d20", "IN REGION, BUT GRIPPER STILL CLOSED")
    x1, y1, x2, y2 = target_box
    after_axis.scatter((x1 + x2) / 2, (y1 + y2) / 2, s=160, color="#ef233c", marker="X")
    after_axis.axis("off")

    banner_axis.axis("off")
    banner_axis.text(
        0.5,
        0.62,
        "REASONING–ACTION ERROR — reasoning correctly says RELEASE; execution keeps the gripper CLOSED",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#b42318", "edgecolor": "#b42318"},
    )
    banner_axis.text(
        0.5,
        0.08,
        f"Task {record['task_id']} · Query {record['policy_query_index']} · {record['task_instruction']} · "
        f"subtask_not_completed · episode success={record['episode_success']}",
        ha="center",
        fontsize=13,
        color="#344054",
    )
    fig.suptitle("Reasoning says RELEASE. The gripper remains CLOSED.", fontsize=25, fontweight="bold")
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / "figure1b_correct_reasoning_incorrect_action.png"
    pdf = figures_dir / "figure1b_correct_reasoning_incorrect_action.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": os_path(figures_dir.parent, png), "pdf": os_path(figures_dir.parent, pdf)}


def _record_id(record: Mapping[str, Any]) -> str:
    return f"{record['episode_id']}::q{record['policy_query_index']}"


def _record_payload(
    record: Mapping[str, Any], run_root: Path, report_root: Path, overlay: Optional[Path]
) -> Dict[str, Any]:
    frame_links = {}
    for label in ("before", "intermediate", "after"):
        path = _resolve_frame(run_root, record, label)
        frame_links[label] = Path(os_path(report_root, path)).as_posix() if path else None
    alignment = record["alignment"]
    return {
        "id": _record_id(record),
        "task_id": record["task_id"],
        "task": record["task_instruction"],
        "seed": record["seed"],
        "episode_id": record["episode_id"],
        "query": record["policy_query_index"],
        "success": bool(record["episode_success"]),
        "reasoning": record["reasoning_raw"],
        "claims": record["claims"],
        "scores": {key: alignment.get(f"{key}_score") for key in LEVELS[:3]},
        "aggregate_score": alignment.get("aggregate_score"),
        "labels": alignment["diagnostics"].get("failure_labels", []),
        "warnings": record.get("claims", {}).get("parse_warnings", []),
        "mismatch_levels": [level for level in LEVELS[:3] if is_evaluable(record, level) and is_mismatch(record, level)],
        "frames": frame_links,
        "overlay": Path(os_path(report_root, overlay)).as_posix() if overlay else None,
        "executed_chunk_length": record.get("executed_chunk_length"),
        "decoded_chunk_length": len(record.get("decoded_action_chunk", [])),
    }


def _review_record_payload(
    record: Mapping[str, Any],
    assignment: Mapping[str, Any],
    proposal: Mapping[str, Any],
    run_root: Path,
    report_root: Path,
) -> Dict[str, Any]:
    frame_links = {}
    for label in ("before", "intermediate", "after"):
        path = _resolve_frame(run_root, record, label)
        frame_links[label] = Path(os_path(report_root, path)).as_posix() if path else None
    return {
        **assignment,
        "task_instruction": record["task_instruction"],
        "seed": record["seed"],
        "episode_success": bool(record["episode_success"]),
        "reasoning_raw": record["reasoning_raw"],
        "claims": record["claims"],
        "action_tokens": record.get("action_tokens", []),
        "decoded_action_chunk": record.get("decoded_action_chunk", []),
        "executed_chunk_length": record.get("executed_chunk_length"),
        "executed_ee_trajectory": record.get("executed_ee_trajectory", []),
        "executed_ee_trajectory_pixels": record.get("executed_ee_trajectory_pixels", []),
        "expected_motion_pixels": record.get("expected_motion_pixels", []),
        "frames": frame_links,
        "alignment": record.get("alignment", {}),
        "local_state_change": local_state_change(record),
        "spatial_evidence": record.get("_review_spatial_evidence", {}),
        "proposal": proposal,
        "initial_labels": _initial_compact_labels(record, proposal),
    }


def _initial_compact_labels(
    record: Mapping[str, Any], proposal: Mapping[str, Any]
) -> Dict[str, str]:
    """Return a complete, reviewable first pass for the five compact labels.

    These suggestions intentionally use only evidence already captured by the
    rollout: the ECoT trace, before/after state, scene-derived target evidence,
    and evaluator diagnostics. They are defaults rather than confirmations;
    browser-saved human values take precedence in the review UI.
    """
    alignment = record.get("alignment", {})
    diagnostics = alignment.get("diagnostics", {})
    failure_labels = set(diagnostics.get("failure_labels", []))

    proposal_reasoning = proposal.get("reasoning")
    proposal_evidence = str(proposal.get("evidence", ""))
    reasoning = "T" if proposal_reasoning == "Correct" else "F"
    # The conservative queue proposal treats every extra object descriptor as
    # a task mismatch (for example, "black book" versus "book"). For this
    # requested binary first pass, accept a matching object head noun and then
    # re-check the observable grasp prerequisite.
    descriptor_match = False
    if proposal_reasoning == "Incorrect" and proposal_evidence.startswith(
        "The stated target is not named by the task instruction."
    ):
        target_tokens = re.findall(r"[a-z0-9]+", str(record.get("claims", {}).get("target_object", "")).lower())
        task_tokens = set(re.findall(r"[a-z0-9]+", str(record.get("task_instruction", "")).lower()))
        descriptor_match = bool(target_tokens and target_tokens[-1] in task_tokens)
        if descriptor_match:
            primitive = diagnostics.get("subtask", {}).get("primitive")
            resolved_target = diagnostics.get("target", {}).get("resolved_object")
            target_before = record.get("state_before", {}).get("objects", {}).get(resolved_target, {})
            grasped = target_before.get("grasped")
            subtask_text = str(record.get("claims", {}).get("subtask", "")).lower()
            transport_without_grasp = subtask_text.startswith("move the ") and " to " in subtask_text and grasped is False
            contradicted = (primitive == "grasp" and grasped is True) or (
                primitive in {"transport", "place", "release"} and grasped is False
            ) or transport_without_grasp
            reasoning = "F" if contradicted else "T"

    kinematic_score = alignment.get("kinematic_score")
    if kinematic_score is None:
        move_alignment = "N/A"
    elif float(kinematic_score) >= 1.0:
        move_alignment = "T"
    elif float(kinematic_score) >= 0.0:
        move_alignment = "Partial"
    else:
        move_alignment = "F"

    target_score = alignment.get("target_score")
    if target_score is None:
        target_alignment = "N/A"
    elif float(target_score) > 0.0 and "wrong_object_interaction" not in failure_labels:
        target_alignment = "T"
    else:
        target_alignment = "F"

    subtask_score = alignment.get("subtask_score")
    if subtask_score is None:
        subtask_alignment = "N/A"
    elif float(subtask_score) >= 1.0:
        subtask_alignment = "T"
    else:
        subtask_alignment = "F"

    proposal_action = proposal.get("action")
    if proposal_action in {"Correct", "Alternative valid"}:
        move = "T"
        move_basis = "the existing proposal identifies a task-correct local step"
    elif proposal_action == "Incorrect":
        move = "F"
        move_basis = "the recorded state shows the local step is task-incorrect"
    elif "T" in {target_alignment, subtask_alignment}:
        move = "T"
        move_basis = "resolved-target progress or the observable subtask predicate supports the local step"
    else:
        move = "F"
        move_basis = "neither resolved-target progress nor the observable subtask predicate supports the local step"

    if proposal_reasoning == "Correct":
        reasoning_basis = "the stated rationale is grounded in the task and pre-action state"
    elif descriptor_match and reasoning == "T":
        reasoning_basis = "the added object descriptor denotes the task object and its observable prerequisite is consistent"
    elif descriptor_match:
        reasoning_basis = "the object name matches, but the pre-action grasp state contradicts the stated subtask"
    elif proposal_reasoning == "Incorrect":
        reasoning_basis = "the task or pre-action state contradicts the stated rationale"
    elif proposal_reasoning == "Unverifiable":
        reasoning_basis = "the trace has no evaluable move claim, so the required binary first pass is conservatively F"
    else:
        reasoning_basis = "target/subtask grounding is unresolved, so the required binary first pass is conservatively F"

    counts = diagnostics.get("kinematic", {}).get("counts", {})
    if move_alignment == "N/A":
        kinematic_basis = "no directional move claim was evaluable"
    else:
        kinematic_basis = (
            f"translation evidence has {int(counts.get('match', 0))} matching, "
            f"{int(counts.get('missing', 0))} missing, {int(counts.get('opposite', 0))} opposite, "
            f"and {int(counts.get('extra', 0))} extra axes"
        )

    spatial = record.get("_review_spatial_evidence", {})
    target_name = str(spatial.get("target_name") or "resolved target").replace("_", " ")
    progress = spatial.get("progress")
    if target_alignment == "N/A":
        target_basis = "target evidence is unresolved"
    elif "wrong_object_interaction" in failure_labels:
        target_basis = "the state records a wrong-object interaction"
    elif progress is not None:
        verb = "closer to" if float(progress) > 0 else "farther from or unchanged relative to"
        target_basis = f"the EE moved {abs(float(progress)) * 1000:.1f} mm {verb} {target_name}"
    else:
        target_basis = f"the target score is {float(target_score):+.2f} for {target_name}"

    if subtask_alignment == "N/A":
        subtask_basis = "the subtask predicate is not evaluable"
    elif subtask_alignment == "T":
        subtask_basis = "the observable subtask predicate is satisfied"
    else:
        subtask_basis = f"the observable subtask score is {float(subtask_score):.2f}, below completion"

    explanation = (
        f"Move Reasoning {reasoning}: {reasoning_basis}; Move {move}: {move_basis}. "
        f"Move Alignment {move_alignment}: {kinematic_basis}; Target {target_alignment}: {target_basis}; "
        f"Subtask {subtask_alignment}: {subtask_basis}."
    )
    return {
        "reasoning": reasoning,
        "action": move,
        "kinematic_alignment": move_alignment,
        "target_alignment": target_alignment,
        "subtask_alignment": subtask_alignment,
        "explanation": explanation,
        "source": "Initial model suggestion from existing ECoT, scene, state, and evaluator evidence; not human-confirmed.",
    }


def os_path(base: Path, target: Optional[Path]) -> str:
    if target is None:
        return ""
    import os

    return os.path.relpath(target, base)


def _runtime(records, episodes, metadata):
    episode_times = [
        float(row["episode_wall_seconds"]) for row in episodes if row.get("episode_wall_seconds") is not None
    ]
    queries_by_episode = Counter(str(record["episode_id"]) for record in records)
    timing_keys = ("reasoning_generation_seconds", "simulator_execution_seconds", "visualization_logging_seconds")
    timing = {key: sum(float(record.get("timing", {}).get(key, 0)) for record in records) for key in timing_keys}
    wall = metadata.get("wall_clock_seconds")
    per_episode_wall = float(wall) / len(episodes) if wall is not None and episodes else None
    projections = {
        str(count): per_episode_wall * count if per_episode_wall is not None else None
        for count in (300, 600, 4500, 13500)
    }
    peaks = [row.get("peak_gpu_memory_bytes") for row in episodes if row.get("peak_gpu_memory_bytes") is not None]
    return {
        "wall_clock_seconds": wall,
        "episode_median_seconds": median(episode_times) if episode_times else None,
        "episode_p90_seconds": float(np.percentile(episode_times, 90)) if episode_times else None,
        "queries_per_episode_median": median(queries_by_episode.values()) if queries_by_episode else None,
        "queries_per_episode_p90": float(np.percentile(list(queries_by_episode.values()), 90))
        if queries_by_episode
        else None,
        "timing_totals": timing,
        "peak_gpu_memory_bytes": max(peaks) if peaks else None,
        "projections_seconds": projections,
    }


def generate_report(
    run_root: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = 2000,
    smoke_root: Optional[Path] = None,
) -> Path:
    run_root = Path(run_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        json.loads(line)
        for line in (run_root / "alignment_queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        _rescore_record_kinematics(record)
    _add_review_spatial_evidence(records)
    episodes = json.loads((run_root / "episode_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_root / "experiment_metadata.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(metadata.get("checkpoint_path", ""))
    checkpoint_config_path = checkpoint_path.parent.parent / "config.json"
    checkpoint_config = (
        json.loads(checkpoint_config_path.read_text(encoding="utf-8")) if checkpoint_config_path.exists() else {}
    )
    checkpoint_id = (
        "RHYu2233/ecot-libero90"
        if "ecot-libero90" in checkpoint_path.as_posix()
        else metadata.get("checkpoint_id", "unreported")
    )
    stats = compute_statistics(records)
    review_proposals = propose_all(records)
    review_queues = build_review_queues(records, review_proposals)
    record_by_review_id = {review_query_id(record): record for record in records}
    review_assignments = [
        assignment
        for assignment in review_queues["representative"] + review_queues["diagnostic"]
        if int(record_by_review_id[assignment["id"]]["task_id"]) not in EXCLUDED_MANUAL_REVIEW_TASK_IDS
    ]
    review_records = {
        assignment["id"]: _review_record_payload(
            record_by_review_id[assignment["id"]],
            assignment,
            review_proposals[assignment["id"]],
            run_root,
            output_dir,
        )
        for assignment in review_assignments
    }
    compact_episode_ids = []
    for task_id in COMPACT_REVIEW_TASK_IDS:
        episode_records = sorted(
            (record for record in records if int(record["task_id"]) == task_id),
            key=lambda record: int(record["policy_query_index"]),
        )
        if not episode_records:
            continue
        episode_id = str(episode_records[0]["episode_id"])
        compact_episode_ids.append(episode_id)
        for position, record in enumerate(episode_records, 1):
            item_id = review_query_id(record)
            if item_id in review_records:
                continue
            assignment = {
                "id": item_id,
                "queue": "compact_episode",
                "position": position,
                "batch": len(compact_episode_ids),
                "inclusion_probability": None,
                "analysis_weight": None,
                "diagnostic_priority": None,
                "episode_id": episode_id,
                "task_id": task_id,
                "query_index": record["policy_query_index"],
            }
            review_records[item_id] = _review_record_payload(
                record, assignment, review_proposals[item_id], run_root, output_dir
            )
    representative_items = [
        {
            **assignment,
            **review_proposals[assignment["id"]],
            "episode_success": bool(record_by_review_id[assignment["id"]]["episode_success"]),
        }
        for assignment in review_queues["representative"]
        if int(record_by_review_id[assignment["id"]]["task_id"]) not in EXCLUDED_MANUAL_REVIEW_TASK_IDS
    ]
    taxonomy_stats = review_statistics(representative_items, replicates=bootstrap_replicates)
    milestones = episode_milestones(records, review_proposals)
    episodes_for_timeline = []
    for episode_id, episode_records in sorted(
        _groups(records).items(), key=lambda pair: int(pair[1][0]["task_id"])
    ):
        ordered_records = sorted(episode_records, key=lambda row: int(row["policy_query_index"]))
        if int(ordered_records[0]["task_id"]) in EXCLUDED_MANUAL_REVIEW_TASK_IDS:
            continue
        cases = [review_proposals[review_query_id(record)]["derived_case"] for record in ordered_records]
        episodes_for_timeline.append(
            {
                "episode_id": episode_id,
                "task_id": ordered_records[0]["task_id"],
                "task_instruction": ordered_records[0]["task_instruction"],
                "seed": ordered_records[0]["seed"],
                "success": bool(ordered_records[0]["episode_success"]),
                "queries": [
                    {
                        "query": record["policy_query_index"],
                        "id": review_query_id(record),
                        "case": review_proposals[review_query_id(record)]["derived_case"],
                        "confidence": review_proposals[review_query_id(record)]["confidence"],
                        "in_review_queue": review_query_id(record) in review_records,
                    }
                    for record in ordered_records
                ],
                "case_counts": dict(Counter(cases)),
                "cases_present": sorted(set(cases)),
                "milestones": milestones[episode_id],
            }
        )
    review_bundle = {
        "schema_version": 2,
        "proposal_version": PROPOSAL_VERSION,
        "experiment_label": EXPERIMENT_LABEL,
        "taxonomy": {
            "reasoning_values": REASONING_VALUES,
            "action_values": ACTION_VALUES,
            "faithfulness_values": FAITHFULNESS_VALUES,
            "outcomes": ALL_OUTCOMES,
        },
        "sampling": {
            key: value for key, value in review_queues.items() if key not in {"representative", "diagnostic"}
        },
        "queue_sizes": {
            "representative": sum(item["queue"] == "representative" for item in review_assignments),
            "diagnostic": sum(item["queue"] == "diagnostic" for item in review_assignments),
            "total": len(review_assignments),
        },
        "estimated_review_minutes_per_query": 2,
        "estimated_review_hours": len(review_assignments) * 2 / 60,
        "records": review_records,
        "representative_ids": [
            item["id"]
            for item in review_queues["representative"]
            if int(record_by_review_id[item["id"]]["task_id"]) not in EXCLUDED_MANUAL_REVIEW_TASK_IDS
        ],
        "diagnostic_ids": [
            item["id"]
            for item in review_queues["diagnostic"]
            if int(record_by_review_id[item["id"]]["task_id"]) not in EXCLUDED_MANUAL_REVIEW_TASK_IDS
        ],
        "compact_episode_ids": compact_episode_ids,
        "compact_episode_task_ids": list(COMPACT_REVIEW_TASK_IDS),
        "excluded_manual_review_task_ids": list(EXCLUDED_MANUAL_REVIEW_TASK_IDS),
        "compact_episode_selection_rule": (
            "Nine retained, manually labeled episodes spanning articulated fixtures, pick-and-place, "
            "stacking, caddy placement, successes, and failures: task IDs "
            + ", ".join(str(task_id) for task_id in COMPACT_REVIEW_TASK_IDS)
            + ". Task 64 was excluded because its generated plan reversed the requested bowl-stacking relation."
        ),
        "agent_proposed_representative_statistics": taxonomy_stats,
        "episode_timelines": episodes_for_timeline,
    }
    (output_dir / "query_review_queue.json").write_text(
        json.dumps(review_bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manual_labels, manual_labels_path = _materialize_manual_labels(records, output_dir)
    manual_table = _manual_query_comparison(manual_labels, bootstrap_replicates)
    review_bundle["manual_label_artifact"] = os_path(output_dir, manual_labels_path)
    review_bundle["manual_query_comparison"] = manual_table
    (output_dir / "query_review_queue.json").write_text(
        json.dumps(review_bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    figures = generate_quantitative_figures(records, output_dir / "figures")
    teaser_candidates = generate_manual_teaser_candidates(
        records, manual_labels, run_root, output_dir / "figures"
    )
    examples = select_examples(records)
    selected = {_record_id(record): record for group in examples.values() for record in group}
    overlay_dir = output_dir / "assets" / "examples"
    overlays = {}
    for record_id, record in selected.items():
        path = overlay_dir / (record_id.replace("::", "-").replace("/", "_") + ".png")
        overlays[record_id] = path if path.exists() or generate_example_figure(record, run_root, path) else None
    payloads = {
        record_id: _record_payload(record, run_root, output_dir, overlays[record_id])
        for record_id, record in selected.items()
    }
    category_ids = {name: [_record_id(record) for record in group] for name, group in examples.items()}
    tables = {
        mode: comparison_table(records, mode, bootstrap_replicates)
        for mode in ("query_mismatch_rate", "episode_mismatch_prevalence", "mean_episode_mismatch_fraction")
    }
    successes = len({record["episode_id"] for record in records if record["episode_success"]})
    failures = len(episodes) - successes
    manual_successes = len({row["episode_id"] for row in manual_labels if row["episode_success"]})
    manual_failures = len({row["episode_id"] for row in manual_labels if not row["episode_success"]})
    manual_episode_ids = {str(row["episode_id"]) for row in manual_labels}
    manual_episodes = [row for row in episodes if str(row["episode_id"]) in manual_episode_ids]
    manual_wall_seconds = sum(
        float(row.get("episode_wall_seconds") or 0.0) for row in manual_episodes
    )
    manual_any = next(item for item in manual_table if item["level"] == "Any level")
    runtime = _runtime(records, episodes, metadata)
    smoke_runtime = None
    if smoke_root is not None:
        smoke_root = Path(smoke_root).resolve()
        smoke_records = [
            json.loads(line)
            for line in (smoke_root / "alignment_queries.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        smoke_episodes = json.loads((smoke_root / "episode_summary.json").read_text(encoding="utf-8"))
        smoke_metadata = json.loads((smoke_root / "experiment_metadata.json").read_text(encoding="utf-8"))
        smoke_runtime = _runtime(smoke_records, smoke_episodes, smoke_metadata)
    tasks = sorted({(int(r["task_id"]), r["task_instruction"]) for r in records})
    seeds = sorted({int(r["seed"]) for r in records})
    per_task = []
    for task_id, instruction in tasks:
        group = [record for record in records if int(record["task_id"]) == task_id]
        per_task.append(
            {
                "task_id": task_id,
                "instruction": instruction,
                "seed": group[0]["seed"],
                "success": group[0]["episode_success"],
                "queries": len(group),
                **{level: level_statistics(group, level) for level in LEVELS},
            }
        )
    low_coverage = sorted(
        per_task,
        key=lambda task: (
            (task["target"]["coverage"] or 0) + (task["subtask"]["coverage"] or 0),
            task["task_id"],
        ),
    )[:10]
    low_coverage_text = ", ".join(str(task["task_id"]) for task in low_coverage)
    low_task_types = "; ".join(task["instruction"] for task in low_coverage[:5])
    data = {
        "records": payloads,
        "categories": category_ids,
        "tables": tables,
        "tasks": per_task,
        "review": review_bundle,
    }
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    figure_html = "".join(
        f'<figure><a href="{item["png"]}"><img src="{item["png"]}" alt="{html.escape(name)}"></a><figcaption>{html.escape(name.replace("_", " ").title())}: one-trial-per-task audit. <a href="{item["pdf"]}">PDF</a> · <a href="{item["png"]}">PNG</a></figcaption></figure>'
        for name, item in figures.items()
    )
    config = metadata.get("configuration", {})
    warning = "Confidence intervals are episode-clustered over the nine retained manual-review episodes. " + (
        "One outcome group has fewer than two episodes; its difference CI is intentionally unavailable."
        if min(manual_successes, manual_failures) < 2
        else "These descriptive comparisons do not support causal interpretation."
    )
    detail_rows = "".join(
        "<tr>"
        f"<td>{DISPLAY[level]}</td>"
        f"<td>{item['total_queries']}</td>"
        f"<td>{item['evaluable_queries']}/{item['total_queries']} ({_percent(item['coverage'])})</td>"
        f"<td>{item['mismatch_count']}/{item['evaluable_queries']} ({_percent(item['query_mismatch_rate'])})</td>"
        f"<td>{_number(item['mean_score'])}</td>"
        f"<td>{_number(item['median_score'])}</td>"
        f"<td>{item['episodes_with_evaluable_claims']}/{len(episodes)}</td>"
        f"<td>{item['episodes_with_mismatch']}/{item['episodes_with_evaluable_claims']} "
        f"({_percent(item['episode_mismatch_prevalence'])})</td>"
        f"<td>{_percent(item['mean_episode_mismatch_fraction'])} "
        f"(mean over {item['episodes_with_evaluable_claims']} episodes)</td>"
        f"<td>{item['parser_failures']}/{item['total_queries']}</td>"
        f"<td>{item['unresolved_or_ambiguous_targets']}/{item['total_queries']}</td>"
        f"<td>{item['unsupported_subtasks']}/{item['total_queries']}</td>"
        "</tr>"
        for level, item in stats["levels"].items()
    )
    primary_rows = "".join(
        "<tr>"
        f"<th scope=\"row\">{item['level']}</th>"
        f"<td><strong>{item['coverage_count']:,}/{item['coverage_denominator']:,}</strong>"
        f"<span>{_percent(item['coverage'])}</span></td>"
        f"<td><strong>{item['overall_numerator']:,}/{item['overall_denominator']:,}</strong>"
        f"<span>{_percent(item['overall'])}</span></td>"
        f"<td><strong>{item['successful_coverage_count']:,}/{item['successful_coverage_denominator']:,}</strong>"
        f"<span>{_percent(item['successful_coverage'])}</span></td>"
        f"<td><strong>{item['successful_numerator']:,}/{item['successful_denominator']:,}</strong>"
        f"<span>{_percent(item['successful'])}</span></td>"
        f"<td><strong>{item['failed_coverage_count']:,}/{item['failed_coverage_denominator']:,}</strong>"
        f"<span>{_percent(item['failed_coverage'])}</span></td>"
        f"<td><strong>{item['failed_numerator']:,}/{item['failed_denominator']:,}</strong>"
        f"<span>{_percent(item['failed'])}</span></td>"
        f"<td><strong>{_percentage_points(item['difference'])}</strong>"
        "<span>failed − successful</span></td>"
        f"<td><strong>{_percentage_point_interval(item['ci'])}</strong>"
        "<span>episode bootstrap</span></td>"
        "</tr>"
        for item in manual_table
    )
    move_comparison_path = output_dir / "move_alignment_comparison.json"
    move_comparison_html = (
        render_move_comparison_html(json.loads(move_comparison_path.read_text(encoding="utf-8")))
        if move_comparison_path.exists()
        else ""
    )
    round2_path = output_dir / "round2_move_review.json"
    if round2_path.exists():
        from experiments.robot.libero.move_only_review_report import render_section

        round2_html = render_section(json.loads(round2_path.read_text(encoding="utf-8")))
    else:
        round2_html = ""
    move_sweep_path = output_dir / "move_alignment_sweep_comparison.json"
    move_sweep_html = (
        render_move_sweep_html(json.loads(move_sweep_path.read_text(encoding="utf-8")))
        if move_sweep_path.exists()
        else ""
    )
    teaser_candidates_html = "".join(
        f'<figure class="teaser-candidate"><a href="{candidate["png"]}">'
        f'<img src="{candidate["preview"]}" alt="Task {candidate["task_id"]}, query '
        f'{candidate["query"]}: reasoning direction versus action direction"></a>'
        f'<figcaption><strong>Task {candidate["task_id"]}, query {candidate["query"]}:</strong> '
        f'{html.escape(candidate["motion_summary"])} '
        f'<br><strong>Why the reasoning is locally sensible:</strong> '
        f'{html.escape(candidate["reason_explanation"])} '
        f'<br><strong>Why the action is wrong:</strong> the action chunk moves mostly '
        f'<strong>{html.escape(candidate["actual_direction"])}</strong>, contradicting the proposed '
        f'<strong>{html.escape(candidate["stated_direction"])}</strong> direction. '
        f'<span class="teaser-download-links"><a href="{candidate["pdf"]}">PDF</a> · '
        f'<a href="{candidate["png"]}">PNG</a></span></figcaption></figure>'
        for candidate in teaser_candidates
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{EXPERIMENT_LABEL}</title>
<style>body{{font:16px/1.45 system-ui,sans-serif;margin:0;color:#1d2430;background:#f4f6f8}}main{{max-width:1440px;margin:auto;background:white;padding:28px}}h1,h2{{color:#17365d}}h2{{margin-top:34px}}.pilot{{background:#fff3cd;border-left:6px solid #e0a800;padding:16px;font-size:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}.card,figure{{border:1px solid #ccd5df;border-radius:8px;padding:12px;background:#fff}}figure img,.card img{{max-width:100%;height:auto}}.table-wrap{{overflow-x:auto;border:1px solid #b9c5d2;border-radius:8px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:11px;border-bottom:1px solid #d7dee7;text-align:left;vertical-align:top}}thead th{{background:#17365d;color:white}}tbody tr:nth-child(even){{background:#f4f7fa}}.primary{{min-width:1100px}}.primary th[scope=row]{{font-size:17px;background:#eaf0f7;color:#17365d}}.primary th[scope=row] span,.primary td strong,.primary td span{{display:block}}.primary th[scope=row] span{{font-size:12px;font-weight:400;color:#566474;max-width:210px}}.primary td strong,.primary td span{{white-space:nowrap}}.primary td span{{color:#566474;margin-top:2px}}.metric{{font-size:25px;font-weight:700}}.lead{{font-size:18px;max-width:1000px}}details.section{{margin:28px 0;border:1px solid #ccd5df;border-radius:8px;padding:0 16px 16px}}details.section>summary{{font-size:20px;font-weight:700;color:#17365d;padding:16px 0;cursor:pointer}}.filters,.review-toolbar{{position:sticky;top:0;background:#eef4fa;padding:10px;z-index:5;display:flex;gap:10px;flex-wrap:wrap;align-items:end}}button{{padding:7px 11px;cursor:pointer}}pre.reason{{white-space:pre-wrap;max-height:280px;overflow:auto;background:#f7f7f7;padding:10px}}.tag{{display:inline-block;padding:3px 7px;margin:2px;background:#e8edf3;border-radius:10px}}.bad{{background:#f8d7da}}.good{{background:#d4edda}}.review-shell{{border:2px solid #8aa6c1;border-radius:10px;padding:16px;background:#fbfdff}}.review-progress{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}}.review-progress>div{{background:#eef4fa;padding:12px;border-radius:7px}}.timeline{{display:flex;gap:4px;flex-wrap:wrap;padding:12px;background:#f3f5f7;border-radius:8px}}.timeline-dot{{width:28px;height:28px;border:0;border-radius:5px;color:white;font-size:11px;padding:0}}.case-1{{background:#27823b}}.case-2{{background:#2774ae}}.case-3{{background:#c83232}}.case-4{{background:#d17905}}.case-5{{background:#7a3e9d}}.case-6{{background:#008b8b}}.case-ambiguous{{background:#6c757d}}.case-unverifiable{{background:#24292f}}.review-frames{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.review-frame{{position:relative}}.review-frame img{{display:block;width:100%}}.review-frame svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}.judgments{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:10px;margin:12px 0}}.judgments label,.review-toolbar label{{display:flex;flex-direction:column;font-size:13px;font-weight:700}}select,textarea,input{{font:inherit;padding:5px}}textarea{{width:100%;min-height:80px;box-sizing:border-box}}.proposal{{background:#eef8f0;border-left:5px solid #27823b;padding:12px}}.review-actions{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}}.review-actions button:first-child{{background:#27823b;color:white}}.review-actions button:nth-child(2){{background:#2774ae;color:white}}.review-actions button:nth-child(3){{background:#6c757d;color:white}}kbd{{background:#20252b;color:white;padding:2px 5px;border-radius:3px}}#task-table th{{cursor:pointer}}.compact-episode{{border:1px solid #b9c5d2;border-radius:8px;margin:14px 0;background:white}}.compact-episode>summary{{padding:14px;font-weight:700;color:#17365d;cursor:pointer}}.compact-scene{{display:flex;gap:18px;align-items:flex-start;padding:14px;background:#eef4fa;border-top:1px solid #c8d5e2}}.compact-scene img{{width:280px;max-width:42%;height:auto;border:1px solid #9eacba;border-radius:6px}}.compact-scene strong{{font-size:17px}}.step-table{{min-width:1380px;font-size:13px}}.step-table th,.step-table td{{padding:8px}}.step-table .reason-cell{{min-width:310px;max-width:430px}}.step-table .action-cell{{min-width:260px}}.step-table .align-cell{{min-width:105px}}.step-table .human-cell{{min-width:270px}}.claim-line{{display:block;margin:4px 0;line-height:1.45}}.claim-key{{display:inline-block;font-size:10px;font-weight:800;padding:2px 5px;border-radius:4px;color:#263238;margin-right:5px;letter-spacing:.02em}}.component-task{{background:#cfe8f7;color:#18536f}}.component-plan{{background:#dce5fa;color:#334e87}}.component-subtask{{background:#ccebd8;color:#28613c}}.component-move{{background:#f8e4ae;color:#75530b}}.component-gripper{{background:#e6d5f4;color:#633780}}.component-objects{{background:#cdeceb;color:#246361}}.component-missing{{color:#8a949e;font-style:italic}}.action-title{{font-size:14px;margin-bottom:5px}}.action-badge{{display:inline-block;font:700 11px ui-monospace,monospace;padding:2px 5px;border-radius:4px;margin-left:4px}}.action-match{{background:#d9eee3;color:#26734a}}.action-differ{{background:#f8dddd;color:#b33b3b}}.action-extra{{background:#fff0c2;color:#795b12}}.action-na{{background:#e8edf3;color:#687684}}.action-vector{{font:12px ui-monospace,monospace;color:#66717d;white-space:nowrap}}.action-divider{{color:#a4adb6;margin:0 3px}}.action-raw{{font:11px ui-monospace,monospace;color:#98a1ab;margin:5px 0}}.metric-pass{{background:#d4edda}}.metric-fail{{background:#f8d7da}}.metric-na{{background:#e8edf3;color:#566474}}.compact-labels{{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}}.compact-labels label{{font-size:10px;font-weight:700}}.compact-labels select{{width:100%;font-size:12px;padding:3px}}.initial-rationale{{margin-top:7px;padding:7px;background:#fff8d9;border-left:3px solid #d69e00;font-size:11px;line-height:1.35}}.compact-status{{font-size:11px;color:#566474;margin-top:4px}}.compact-evidence img{{width:31%;max-width:180px;margin:3px}}@media(max-width:800px){{main{{padding:14px}}h1{{font-size:26px}}.review-frames,.judgments{{grid-template-columns:1fr}}.compact-scene{{display:block}}.compact-scene img{{max-width:100%;width:100%;margin-bottom:10px}}}}@media print{{.filters,.review-toolbar,.review-actions,button{{display:none}}body{{background:white}}main{{max-width:none}}.card{{break-inside:avoid}}details.section{{display:block}}details.section>*{{display:block}}}}</style></head><body><main>
<style>.metric-detail{{display:block;white-space:pre-line;margin-top:4px;line-height:1.35}}.compact-alignments{{display:grid;grid-template-columns:1fr;gap:5px;margin-top:9px;padding-top:8px;border-top:1px solid #ccd5df}}.compact-alignments label{{font-size:11px;font-weight:700}}.compact-alignments select{{width:100%;font-size:12px;padding:3px}}.compact-overall{{font-weight:700}}.scene-map{{position:relative;width:280px;max-width:42%;flex:0 0 auto}}.compact-scene .scene-map img{{display:block;width:100%;max-width:none;height:auto;margin:0}}.scene-map svg{{position:absolute;left:0;top:0;width:100%;height:auto;aspect-ratio:1;pointer-events:none}}.scene-map svg text{{font:700 8px system-ui,sans-serif;paint-order:stroke;stroke:white;stroke-width:2px;stroke-linejoin:round}}.origin-marker path{{stroke:#111;stroke-width:3}}.origin-marker text{{fill:#111}}.target-marker text:first-child{{fill:#ffd000;stroke:#8a5b00;stroke-width:1px;font-size:20px;text-anchor:middle;dominant-baseline:middle}}.target-marker text:last-child{{fill:#8a5b00}}.ee-before-marker circle{{fill:#1597d4;stroke:white;stroke-width:2}}.ee-before-marker text{{fill:#006a9d}}.ee-after-marker circle{{fill:#d62976;stroke:white;stroke-width:2}}.ee-after-marker text{{fill:#a71150}}.scene-legend{{display:flex;gap:7px;flex-wrap:wrap;font-size:10px;margin-top:4px}}.legend-target{{color:#8a5b00}}.legend-before{{color:#006a9d}}.legend-after{{color:#a71150}}@media(max-width:800px){{.scene-map{{width:100%;max-width:100%}}}}</style>
<style>main{{max-width:none;margin:0 12px;padding:20px;box-sizing:border-box}}</style>
<style>.metric-partial{{background:#fff3cd}}.step-table th:nth-child(5),.step-table td:nth-child(5){{min-width:280px}}.subtask-checkpoint{{margin-top:8px;padding-top:6px;border-top:1px solid #a8b5a8}}.subtask-checkpoint summary{{cursor:pointer;font-weight:700}}.subtask-checkpoint img{{display:block;width:100%;max-width:240px;margin:6px 0;border:1px solid #98a598;border-radius:4px}}.subtask-checkpoint span,.subtask-checkpoint label,.subtask-checkpoint small{{display:block;margin:5px 0;white-space:normal}}.subtask-checkpoint select,.subtask-checkpoint input{{box-sizing:border-box;width:100%}}</style>
<style>.domain-labels{{display:grid;grid-template-columns:1fr;gap:5px;padding:8px 0;border-bottom:1px solid #ccd5df}}.domain-labels label{{font-size:10px;font-weight:700;min-width:0}}.domain-labels select{{box-sizing:border-box;width:100%;font-size:12px;padding:3px}}.human-cell{{min-width:240px!important}}</style>
<style>.subtask-labels{{grid-template-columns:1fr}}</style>
<style>.teaser-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,600px));justify-content:center;gap:18px}}.teaser-candidate{{margin:0}}.teaser-candidate img{{display:block;width:100%;max-width:480px;height:auto;margin:0 auto}}.teaser-candidate figcaption{{margin-top:8px;line-height:1.4}}@media(max-width:900px){{.teaser-grid{{grid-template-columns:1fr}}}}</style>
<h1>{EXPERIMENT_LABEL}</h1><div class="pilot"><strong>Pilot warning:</strong> 90 tasks × 1 trial per task. This is an initial qualitative audit, not a complete statistical benchmark. Do not interpret per-task observations as stable rates.</div>
<h2>Executive summary — nine retained human-labeled episodes</h2><div class="grid"><div class="card"><div class="metric">{manual_successes}/{len(manual_episodes)}</div>successful episodes</div><div class="card"><div class="metric">{len(manual_labels)}</div>manually labeled policy queries</div><div class="card"><div class="metric">{_percent(manual_any["overall"])}</div>Move-or-Subtask query mismatch ({manual_any["overall_numerator"]}/{manual_any["overall_denominator"]})</div><div class="card"><div class="metric">{manual_wall_seconds:.1f}s</div>measured wall clock for retained episodes</div></div>
<h2>Figure 1 — Teaser candidates: stated MOVE → opposing execution</h2>
<p class="lead">Two human-labeled examples show locally sensible reasoning followed by an opposing action. Green is the direction proposed by the reasoning; red is the action chunk's measured direction. The page shows only the annotated pre-action scene; downloadable PNG/PDF files add a concise explanation box.</p>
<div class="teaser-grid">{teaser_candidates_html}</div>
<h2 id="query-review">Query taxonomy and human review</h2>
<div class="pilot"><strong>Manual review complete for the nine retained scenarios.</strong> The results below are descriptive for this deliberately selected set and are not LIBERO-90 prevalence estimates. Task 64 is excluded from manual review because its generated plan reverses the requested task relation.</div>
<div class="review-shell">
<h3>Review workspace</h3>
<p>Judge the current observation and local state transition—not final episode success alone. Edit Reasoning, Action, and Faithfulness; the case is derived automatically, so an inconsistent case cannot be selected. Agent proposals are always retained separately.</p>
<details class="section" open><summary>Compact step-by-step review — 9 retained episodes</summary>
<p>Each row pairs the exact ECoT output from one policy query with the <strong>10-step action chunk generated by that same query</strong>. Its action heading summarizes net translation, commanded rotation, and gripper command. Move Alignment is the cosine similarity between the ECoT MOVE translation vector and raw summed measured 10-step EE displacement; rotation phrases are excluded from the translation vector. Cosine ≥ 0.5 is <code>T</code>, cosine &lt; 0.5 is <code>F</code>, and unavailable displacement is <code>N/A</code>. The human-review column contains only Move Alignment and Subtask Alignment. Initial suggestions show their scores briefly; browser-saved human labels override these defaults. The Subtask column groups consecutive identical subtask labels into periods and distributes completion rewards as documented below. Every proposal remains editable and persists with the annotations. No rollout was repeated.</p>
<p><strong>Initial subtask proposal rule:</strong> if a subtask label appears again later in the episode, its earlier period is proposed as incomplete: checkpoint score −1, zero completed subtasks, distributed evenly across that earlier period. For example, Task 0's return to <code>move to the cabinet</code> makes queries 0–2 each −1/3. Human judgments saved under the current rule override proposals; checkpoint judgments from the superseded rule are reset.</p>
<p><strong>Checkpoint evidence alignment:</strong> the main image is the after-scene from the final query of the subtask period being judged. When a successful final query completes multiple subtasks and its period spans multiple queries, the preceding query from that same period is also shown for comparison. The label-change text and original ECoT subtask reasoning come from the following query. All relevant query numbers are shown.</p>
<p><strong>Scene markers:</strong> black cross = projected world origin; gold star = resolved simulator target; blue dot = chunk-start EE; magenta dot = chunk-end EE. Projection is fitted from the episode's logged world/pixel EE trajectory, and its median residual is displayed.</p>
<p><strong>Selection:</strong> {html.escape(review_bundle['compact_episode_selection_rule'])}</p>
<div class="review-toolbar"><div id="qr-compact-progress"></div><button id="qr-export">Export annotations</button><label>Import annotations<input id="qr-import" type="file" accept="application/json"></label></div>
<div id="qr-compact-episodes"></div></details>
</div>
<h2 id="human-labeled-results">Human-labeled query-level results</h2><p class="lead">These metrics use the completed labels for the nine retained scenarios. Browser-saved edits override accepted initial labels. Task 64 is excluded because its generated plan reversed the requested bowl-stacking relation. Target is omitted because its definition and labels have not yet been adjudicated. Results are separated by episode outcome; each cell shows <strong>count/denominator</strong> first and the percentage underneath.</p><div class="table-wrap"><table class="primary outcome-split"><thead><tr><th rowspan="2">Level</th><th colspan="2">Overall — 9 episodes, 192 queries</th><th colspan="2">Successful episodes — {manual_successes} episodes, 72 queries</th><th colspan="2">Failed episodes — {manual_failures} episodes, 120 queries</th><th colspan="2">Failure–success comparison</th></tr><tr><th>Coverage</th><th>Mismatch</th><th>Coverage</th><th>Mismatch</th><th>Coverage</th><th>Mismatch</th><th>Difference</th><th>Bootstrap 95% CI</th></tr></thead><tbody>{primary_rows}</tbody></table></div>
{move_comparison_html}
{round2_html}
{move_sweep_html}
<p>{html.escape(warning)} Difference means <strong>failed minus successful</strong>; positive values indicate more mismatches in failed episodes.</p>
<details class="section"><summary>Experiment configuration and checkpoint</summary><table><tr><th>Checkpoint</th><td>{html.escape(metadata.get("checkpoint_path", ""))}</td></tr><tr><th>Repository commit</th><td>{html.escape(str(metadata.get("repository_commit")))}</td></tr><tr><th>Benchmark</th><td>LIBERO-90, Standard condition, all 90 tasks</td></tr><tr><th>Policy</th><td>Full ECoT MiniVLA; explicit reasoning enabled={metadata.get("reasoning_generation_enabled")}</td></tr><tr><th>Observation</th><td>{html.escape(metadata.get("observation", ""))}</td></tr><tr><th>Action</th><td>{html.escape(metadata.get("action_representation", ""))}</td></tr><tr><th>Seeds</th><td>base {metadata.get("base_seed")}; {html.escape(metadata.get("seed_schedule", ""))}</td></tr><tr><th>Processes</th><td>{config.get("n_procs_per_gpu")}</td></tr></table>
<p><strong>Checkpoint provenance:</strong> {html.escape(str(checkpoint_id))}, a third-party Full-ECoT reproduction rather than an official paper-author checkpoint. Model configuration: base VLM <code>{html.escape(str(checkpoint_config.get("vla", {}).get("base_vlm", "unknown")))}</code>; reasoning mode <code>{html.escape(str(checkpoint_config.get("reasoning_mode", "unknown")))}</code>; action tokenizer <code>{html.escape(str(checkpoint_config.get("vla", {}).get("action_tokenizer", "unknown")))}</code>; image sequence length {checkpoint_config.get("vla", {}).get("image_sequence_len", "unknown")}; wrist image {checkpoint_config.get("vla", {}).get("use_wrist_image", "unknown")}.</p>
<p>References: <a href="https://arxiv.org/abs/2505.08243">ECoT-Lite paper</a> · <a href="https://github.com/globc/ecot-lite">evaluated repository</a>.</p></details>
<details class="section"><summary>Runtime and projections</summary><div id="runtime"></div><p>Projections use measured wall-clock throughput on this hardware and process count, not hard-coded estimates.</p><h3>Two-task smoke runtime</h3><div id="smoke-runtime"></div></details>
<details class="section"><summary>Episode-level and detailed statistics</summary><label>View <select id="mode"><option value="episode_mismatch_prevalence">Episode-level mismatch prevalence</option><option value="mean_episode_mismatch_fraction">Mean episode mismatch fraction</option><option value="query_mismatch_rate">Query-level mismatch rate</option></select></label><div id="main-table"></div><h3>Full component details</h3><div class="table-wrap"><table><tr><th>Level</th><th>Total queries</th><th>Evaluable / coverage</th><th>Query mismatch</th><th>Mean score</th><th>Median score</th><th>Episodes evaluable</th><th>Episodes with mismatch</th><th>Mean episode mismatch fraction</th><th>Parser failures</th><th>Unresolved / ambiguous targets</th><th>Unsupported subtasks</th></tr>{detail_rows}</table></div></details>
<h2>Coverage and exclusions</h2><p>Parser failures: {stats["parser_failures"]}/{len(records)}; queries with warnings: {stats["queries_with_parser_warnings"]}/{len(records)}; unresolved/ambiguous targets: {stats["unresolved_or_ambiguous_targets"]}/{len(records)}; unsupported subtasks: {stats["unsupported_subtasks"]}/{len(records)}. Missing scores are excluded from every denominator.</p>
<h2>Publication figures</h2><div class="grid">{figure_html}</div>
<details class="section"><summary>Mismatch overlap counts</summary><pre>{html.escape(json.dumps(mismatch_overlap(records), indent=2))}</pre></details>
<details class="section"><summary>Sortable per-task audit table</summary><p>Each row is one observed episode, not a per-task rate.</p><div id="task-table"></div></details>
<h2>Qualitative examples</h2><details class="section"><summary>Filters and review export tools</summary><div class="filters"><label>Task <select id="f-task"><option value="">All</option>{"".join(f'<option value="{i}">{i}</option>' for i, _ in tasks)}</select></label><label>Seed <select id="f-seed"><option value="">All</option>{"".join(f"<option>{s}</option>" for s in seeds)}</select></label><label>Outcome <select id="f-success"><option value="">All</option><option value="true">Success</option><option value="false">Failure</option></select></label><label>Mismatch <select id="f-level"><option value="">All</option><option value="kinematic">Kinematic</option><option value="target">Target</option><option value="subtask">Subtask</option></select></label><label>Score min <input id="f-min" type="number" min="-1" max="1" step="0.1"></label><label>max <input id="f-max" type="number" min="-1" max="1" step="0.1"></label><label><input id="f-warning" type="checkbox">warnings only</label><button id="export-text">Export review text</button><button id="export-html">Export review HTML</button></div></details>
<h3>Figure 1 candidates</h3><p>Candidate selection is deterministic; final selection requires manual review.</p><div id="gallery-figure1" class="grid"></div><h3>Three-level explanatory examples</h3><h4>Kinematic</h4><div id="gallery-kinematic" class="grid"></div><h4>Target</h4><div id="gallery-target" class="grid"></div><h4>Subtask</h4><div id="gallery-subtask" class="grid"></div><details class="section"><summary>Appendix example bank and full timelines</summary><h3>Aligned controls</h3><div id="gallery-aligned" class="grid"></div><h3>Multi-level mismatches</h3><div id="gallery-multi_level" class="grid"></div><h3>Successful episodes with temporary mismatches</h3><div id="gallery-successful_temporary" class="grid"></div><h3>Failed episodes without any detected mismatch</h3><div id="gallery-failed_no_detected" class="grid"></div><h3>Selected full episode timelines</h3><div id="gallery-timelines" class="grid"></div></details>
<h2>Manual-table definitions</h2><ul><li>Move: the saved human label, or the accepted initial cosine label when unchanged. Cosine ≥ 0.5 is <code>T</code>; otherwise <code>F</code>.</li><li>Subtask: the saved human label, or the accepted period/checkpoint completion label when unchanged. Positive reward is <code>T</code>; non-positive reward is <code>F</code>.</li><li>Any level: <code>F</code> if Move or Subtask is <code>F</code>; otherwise <code>T</code> when at least one retained level is evaluable.</li><li>Target is omitted pending definition and label review.</li><li>Confidence intervals: {bootstrap_replicates}-replicate episode-clustered bootstrap, seed 20260904; difference is failed minus successful.</li></ul>
<h2>Known limitations and recommended next experiment</h2><p>One trial cannot estimate task-specific variability. Generic repeated-object references remain unresolved rather than inferred. Visual arrows use MuJoCo camera calibration, but should still be manually inspected. The automatic evaluator does not establish reasoning correctness; the new taxonomy labels are conservative agent proposals until a person confirms them.</p><p>This audit observed <strong>{failures} failed episodes out of {len(episodes)}</strong>. Lowest target/subtask coverage was observed on task IDs {low_coverage_text}, including: {html.escape(low_task_types)}. Repeated-object, articulated-fixture, and multi-stage placement tasks should receive additional sampling.</p><ol><li><strong>Predefined representative quantitative sample:</strong> select 30 tasks before observing new outcomes, stratified by task primitive and current coverage, and run 10 additional Standard-condition trials each (300 episodes). Use this group for prevalence estimates.</li><li><strong>Failure-enriched diagnostic sample:</strong> run 5 additional Standard-condition trials on every failed task plus the ten lowest-coverage tasks, capped at 100 diagnostic episodes. Use it only for qualitative failure discovery, never overall prevalence.</li></ol><p>After the 300-episode Standard replication, add a separately labeled 300-episode Perturbation + Distractors robustness condition using the same predefined tasks and seeds. Keep it separate from Standard-condition prevalence.</p>
</main><script id="audit-data" type="application/json">{data_json}</script><script>const D=JSON.parse(document.getElementById('audit-data').textContent), decisions=JSON.parse(localStorage.getItem('ecotAuditReviews')||'{{}}');
const pct=x=>x==null?'—':(100*x).toFixed(1)+'%'; const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function table(){{let mode=document.getElementById('mode').value, rows=D.tables[mode], count=(n,d)=>n==null?`mean over ${{d}} episodes`:`${{n}}/${{d}}`;document.getElementById('main-table').innerHTML='<table><tr><th>Level</th><th>Coverage</th><th>Overall mismatch</th><th>Successful episodes</th><th>Failed episodes</th><th>Difference</th><th>95% CI</th></tr>'+rows.map(r=>`<tr><td>${{r.level}}</td><td>${{pct(r.coverage)}} (${{r.coverage_count}}/${{r.coverage_denominator}})</td><td>${{pct(r.overall)}} (${{count(r.overall_numerator,r.overall_denominator)}})</td><td>${{pct(r.successful)}} (${{count(r.successful_numerator,r.successful_denominator)}})</td><td>${{pct(r.failed)}} (${{count(r.failed_numerator,r.failed_denominator)}})</td><td>${{pct(r.difference)}} (failed − successful)</td><td>${{r.ci?r.ci.map(pct).join(' to '):'Unavailable: group too small'}}</td></tr>`).join('')+'</table>'}} document.getElementById('mode').onchange=table;table();
const runtime={json.dumps(runtime)},smoke={json.dumps(smoke_runtime)};document.getElementById('runtime').innerHTML=`<table><tr><th>Full-run wall</th><td>${{runtime.wall_clock_seconds.toFixed(1)}} s</td></tr><tr><th>Episode median / P90</th><td>${{runtime.episode_median_seconds.toFixed(1)}} / ${{runtime.episode_p90_seconds.toFixed(1)}} s</td></tr><tr><th>Queries/episode median / P90</th><td>${{runtime.queries_per_episode_median}} / ${{runtime.queries_per_episode_p90}}</td></tr><tr><th>Reasoning generation</th><td>${{runtime.timing_totals.reasoning_generation_seconds.toFixed(1)}} s</td></tr><tr><th>Simulator execution</th><td>${{runtime.timing_totals.simulator_execution_seconds.toFixed(1)}} s</td></tr><tr><th>Visualization/logging</th><td>${{runtime.timing_totals.visualization_logging_seconds.toFixed(1)}} s</td></tr><tr><th>Peak GPU allocated</th><td>${{(runtime.peak_gpu_memory_bytes/2**30).toFixed(2)}} GiB per worker</td></tr>${{Object.entries(runtime.projections_seconds).map(([n,s])=>`<tr><th>${{n}} episode projection</th><td>${{(s/3600).toFixed(2)}} h</td></tr>`).join('')}}</table>`;document.getElementById('smoke-runtime').innerHTML=smoke?`<table><tr><th>Wall</th><td>${{smoke.wall_clock_seconds.toFixed(1)}} s</td></tr><tr><th>Episode median / P90</th><td>${{smoke.episode_median_seconds.toFixed(1)}} / ${{smoke.episode_p90_seconds.toFixed(1)}} s</td></tr><tr><th>Queries/episode</th><td>${{smoke.queries_per_episode_median}}</td></tr><tr><th>Reasoning / simulator / logging</th><td>${{smoke.timing_totals.reasoning_generation_seconds.toFixed(1)}} / ${{smoke.timing_totals.simulator_execution_seconds.toFixed(1)}} / ${{smoke.timing_totals.visualization_logging_seconds.toFixed(1)}} s</td></tr><tr><th>Peak GPU allocated</th><td>${{(smoke.peak_gpu_memory_bytes/2**30).toFixed(2)}} GiB</td></tr></table>`:'Not supplied';
function card(r){{let id=esc(r.id), choice=decisions[r.id]||'';return `<article class="card" data-task="${{r.task_id}}" data-seed="${{r.seed}}" data-success="${{r.success}}" data-levels="${{r.mismatch_levels.join(',')}}" data-warning="${{r.warnings.length>0}}" data-score="${{r.aggregate_score??''}}"><h4>Task ${{r.task_id}} · query ${{r.query}} · ${{r.success?'success':'failure'}}</h4>${{r.overlay?`<img src="${{r.overlay}}">`:''}}<p>${{esc(r.task)}}</p><p>Claims: subtask=${{esc(r.claims.subtask)}}; target=${{esc(r.claims.target_object)}}; motion=${{esc(JSON.stringify(r.claims.motion_axes))}}</p><p>Scores K/T/S: ${{esc(JSON.stringify(r.scores))}}</p><p>${{r.labels.map(x=>`<span class="tag bad">${{esc(x)}}</span>`).join('')}} ${{r.warnings.map(x=>`<span class="tag">warning: ${{esc(x)}}</span>`).join('')}}</p><details><summary>Exact reasoning</summary><pre class="reason">${{esc(r.reasoning)}}</pre></details><p>ID: ${{id}} · chunk ${{r.executed_chunk_length}}/${{r.decoded_chunk_length}}</p><div>${{['Good Figure 1 Candidate','Appendix Only','Reject','Needs Investigation'].map(x=>`<button onclick="review('${{id}}','${{x}}')" class="${{choice==x?'good':''}}">${{x}}</button>`).join('')}}</div></article>`}}
function render(name,ids){{document.getElementById('gallery-'+name).innerHTML=ids.length?ids.map(id=>card(D.records[id])).join(''):'<p>No reliable example available in this 90-episode audit.</p>'}}['figure1','kinematic','target','subtask','aligned','multi_level','successful_temporary','failed_no_detected','timelines'].forEach(name=>render(name,D.categories[name]));
window.review=(id,value)=>{{decisions[id]=value;localStorage.setItem('ecotAuditReviews',JSON.stringify(decisions));location.reload()}};
function filters(){{let task=document.getElementById('f-task').value,seed=document.getElementById('f-seed').value,success=document.getElementById('f-success').value,level=document.getElementById('f-level').value,warn=document.getElementById('f-warning').checked,min=parseFloat(document.getElementById('f-min').value),max=parseFloat(document.getElementById('f-max').value);document.querySelectorAll('article.card').forEach(el=>{{let score=parseFloat(el.dataset.score),ok=(!task||el.dataset.task==task)&&(!seed||el.dataset.seed==seed)&&(!success||el.dataset.success==success)&&(!level||el.dataset.levels.split(',').includes(level))&&(!warn||el.dataset.warning=='true')&&(isNaN(min)||(!isNaN(score)&&score>=min))&&(isNaN(max)||(!isNaN(score)&&score<=max));el.style.display=ok?'':'none'}})}}document.querySelectorAll('.filters input,.filters select').forEach(x=>x.oninput=filters);
function exportReview(kind){{let text=Object.entries(decisions).map(([id,v])=>id+'\\t'+v).join('\\n'),blob=new Blob([kind==='html'?'<pre>'+esc(text)+'</pre>':text],{{type:kind==='html'?'text/html':'text/plain'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='ecot-review.'+(kind==='html'?'html':'txt');a.click()}}document.getElementById('export-text').onclick=()=>exportReview('text');document.getElementById('export-html').onclick=()=>exportReview('html');
document.getElementById('task-table').innerHTML='<table><tr><th>Task</th><th>Instruction</th><th>Seed</th><th>Outcome</th><th>Queries</th><th>Kinematic observed</th><th>Target observed</th><th>Subtask observed</th><th>Any observed</th></tr>'+D.tasks.map(t=>`<tr><td>${{t.task_id}}</td><td>${{esc(t.instruction)}}</td><td>${{t.seed}}</td><td>${{t.success?'Success':'Failure'}}</td><td>${{t.queries}}</td>${{['kinematic','target','subtask','any'].map(l=>`<td>${{t[l].mismatch_count}}/${{t[l].evaluable_queries}} (${{pct(t[l].query_mismatch_rate)}})</td>`).join('')}}</tr>`).join('')+'</table>';
document.querySelectorAll('#task-table th').forEach((th,col)=>th.onclick=()=>{{let table=th.closest('table'),rows=[...table.rows].slice(1),numeric=[0,2,4].includes(col);rows.sort((a,b)=>{{let x=a.cells[col].textContent,y=b.cells[col].textContent;return numeric?parseFloat(x)-parseFloat(y):x.localeCompare(y)}});rows.forEach(row=>table.tBodies[0].appendChild(row))}});
</script><script src="query_review_ui.js?v=51"></script></body></html>"""
    results_start = document.index('<h2 id="human-labeled-results">Human-labeled query-level results</h2>')
    if move_sweep_html:
        sweep_start = document.index('<section id="move-alignment-lambda-sweep-comparison">', results_start)
        results_end = document.index("</section>", sweep_start) + len("</section>")
    elif round2_html:
        round2_start = document.index('<section id="round2-move-review">', results_start)
        results_end = document.index(
            "</script>", document.index('<script src="round2_move_review_ui.js', round2_start)
        ) + len("</script>")
    elif move_comparison_html:
        comparison_start = document.index('<section id="move-alignment-finetune-comparison">', results_start)
        results_end = document.index("</section>", comparison_start) + len("</section>")
    else:
        results_end = document.index("</table></div>", results_start) + len("</table></div>")
    document = (
        document[:results_end]
        + f'\n</main><script id="audit-data" type="application/json">{data_json}</script>'
        + '<script>window.D=JSON.parse(document.getElementById("audit-data").textContent);</script>'
        + '<script src="query_review_ui.js?v=54"></script></body></html>'
    )
    index = output_dir / "index.html"
    index.write_text(document, encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("query_review_ui.js"), output_dir / "query_review_ui.js")
    if round2_html:
        shutil.copyfile(Path(__file__).with_name("round2_move_review_ui.js"), output_dir / "round2_move_review_ui.js")
    return index


def validate_local_assets(index: Path) -> List[str]:
    import re

    text = Path(index).read_text(encoding="utf-8")
    missing = []
    for value in re.findall(r'(?:src|href)="([^"#]+)"', text):
        if value.startswith(("http:", "https:", "data:", "javascript:", "$")):
            continue
        local_path = value.split("?", 1)[0]
        if not (Path(index).parent / local_path).exists():
            missing.append(value)
    return sorted(set(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--smoke-root", type=Path)
    args = parser.parse_args()
    index = generate_report(
        args.run_root,
        args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        smoke_root=args.smoke_root,
    )
    missing = validate_local_assets(index)
    if missing:
        raise RuntimeError(f"Missing local report assets: {missing[:10]}")
    print(index)


if __name__ == "__main__":
    main()
