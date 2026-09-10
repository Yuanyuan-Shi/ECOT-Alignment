"""Offline query taxonomy, review queues, and preliminary review statistics."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

REASONING_VALUES = ("Correct", "Incorrect", "Ambiguous", "Unverifiable")
ACTION_VALUES = ("Correct", "Incorrect", "Alternative valid", "Ambiguous", "Unverifiable")
FAITHFULNESS_VALUES = ("Yes", "No", "Partial", "Ambiguous", "Unverifiable")
RESOLVED_CASES = tuple(f"Case {index}" for index in range(1, 7))
ALL_OUTCOMES = (*RESOLVED_CASES, "Ambiguous", "Unverifiable")
PROPOSAL_VERSION = 2


def query_id(record: Mapping[str, Any]) -> str:
    return f"{record['episode_id']}::q{int(record['policy_query_index'])}"


def derive_case(reasoning: str, action: str, faithfulness: str) -> str:
    """Derive the only case compatible with the three editable judgments."""
    if "Unverifiable" in (reasoning, action, faithfulness):
        return "Unverifiable"
    mapping = {
        ("Correct", "Correct", "Yes"): "Case 1",
        ("Incorrect", "Correct", "No"): "Case 2",
        ("Correct", "Incorrect", "No"): "Case 3",
        ("Incorrect", "Incorrect", "Yes"): "Case 4",
        ("Incorrect", "Incorrect", "No"): "Case 5",
        ("Correct", "Alternative valid", "No"): "Case 6",
    }
    return mapping.get((reasoning, action, faithfulness), "Ambiguous")


def _primitive(record: Mapping[str, Any]) -> Optional[str]:
    return record.get("alignment", {}).get("diagnostics", {}).get("subtask", {}).get("primitive")


def _target_resolution(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("alignment", {}).get("diagnostics", {}).get("target_resolution", {}).get("before", {})


def _strong_faithfulness(record: Mapping[str, Any]) -> tuple[str, List[str]]:
    alignment = record.get("alignment", {})
    diagnostics = alignment.get("diagnostics", {})
    labels = set(diagnostics.get("failure_labels", []))
    evaluated = alignment.get("evaluated_claims", [])
    evidence = []
    if not evaluated:
        return "Unverifiable", ["No action-relevant claim was evaluable."]
    contradictions = labels & {
        "kinematic_opposite",
        "kinematic_no_motion",
        "target_moved_away",
        "wrong_object_interaction",
        "subtask_not_completed",
    }
    if contradictions:
        evidence.append("Alignment diagnostics: " + ", ".join(sorted(contradictions)) + ".")
        return "No", evidence
    values = {
        "kinematic": alignment.get("kinematic_score"),
        "target": alignment.get("target_score"),
        "subtask": alignment.get("subtask_score"),
    }
    if len(evaluated) >= 2 and all(
        values[name] is not None
        and (values[name] >= 1.0 if name in {"kinematic", "subtask"} else values[name] >= 0.0)
        for name in evaluated
    ):
        evidence.append("All available alignment claims agree with the executed state change.")
        return "Yes", evidence
    return "Partial", ["Available alignment components are incomplete or only partially supportive."]


def _target_is_task_relevant(record: Mapping[str, Any]) -> Optional[bool]:
    target = record.get("claims", {}).get("target_object")
    if not target:
        return None
    task_words = set(re.findall(r"[a-z0-9]+", record.get("task_instruction", "").lower()))
    target_words = set(re.findall(r"[a-z0-9]+", str(target).lower())) - {
        "the",
        "a",
        "an",
        "left",
        "right",
        "middle",
        "front",
        "back",
        "top",
        "bottom",
    }
    return bool(target_words and target_words <= task_words)


def _reasoning_judgment(record: Mapping[str, Any]) -> tuple[str, List[str]]:
    claims = record.get("claims", {})
    alignment = record.get("alignment", {})
    diagnostics = alignment.get("diagnostics", {})
    warnings = claims.get("parse_warnings", [])
    if not record.get("reasoning_raw") or not record.get("frames") or not record.get("state_before"):
        return "Unverifiable", ["Reasoning, frames, or pre-action state is absent."]
    if warnings:
        return "Ambiguous", ["Parser warnings require human inspection: " + ", ".join(warnings) + "."]
    resolution = _target_resolution(record)
    if claims.get("target_object") and resolution.get("status") != "resolved":
        return "Ambiguous", [f"Target resolution is {resolution.get('status', 'missing')}."]
    relevant = _target_is_task_relevant(record)
    if relevant is False:
        return "Incorrect", ["The stated target is not named by the task instruction."]
    primitive = _primitive(record)
    if primitive is None or "unsupported_subtask" in diagnostics.get("failure_labels", []):
        return "Ambiguous", ["The stated subtask primitive is not reliably supported by the state predicates."]
    target_name = diagnostics.get("target", {}).get("resolved_object")
    target_before = record.get("state_before", {}).get("objects", {}).get(target_name, {})
    if primitive in {"transport", "place", "release"} and target_before.get("grasped") is False:
        return "Incorrect", [
            f"The {primitive} prerequisite is contradicted: the target was not grasped before execution."
        ]
    if primitive == "grasp" and target_before.get("grasped") is True:
        return "Incorrect", ["The trace proposes grasping an object already grasped before execution."]
    if relevant is True:
        return "Correct", ["The stated target is task-relevant and the observable subtask prerequisites are consistent."]
    return "Ambiguous", ["The trace lacks enough grounded target evidence to establish reasoning correctness."]


def propose_judgment(
    record: Mapping[str, Any], *, is_last_successful_query: bool = False
) -> Dict[str, Any]:
    """Produce a conservative evidence-backed proposal, not a human ground-truth label."""
    # Two examples received explicit frame/state review while preparing the audit figures.
    key = (int(record["task_id"]), int(record["policy_query_index"]))
    if key == (64, 5):
        result = {
            "reasoning": "Incorrect",
            "action": "Correct",
            "faithfulness": "No",
            "confidence": "High",
            "evidence": (
                "Task requires stacking the right bowl on the left bowl, but the reasoning reverses the source "
                "and target. Simulator state shows the right bowl was grasped, a locally correct source-object "
                "selection, so the action "
                "does not faithfully realize the stated left-bowl intention."
            ),
        }
        result["derived_case"] = derive_case(result["reasoning"], result["action"], result["faithfulness"])
        return result
    if key == (73, 13):
        result = {
            "reasoning": "Correct",
            "action": "Incorrect",
            "faithfulness": "No",
            "confidence": "High",
            "evidence": (
                "The book is already in the front caddy region and release is the correct next step. After execution "
                "the gripper remains closed and the book remains grasped, so the action neither completes nor "
                "faithfully realizes release."
            ),
        }
        result["derived_case"] = derive_case(result["reasoning"], result["action"], result["faithfulness"])
        return result

    reasoning, reasoning_evidence = _reasoning_judgment(record)
    faithfulness, faithfulness_evidence = _strong_faithfulness(record)
    # Action correctness is task-relative, whereas automatic alignment is
    # reasoning-relative. Moving away from a wrong stated target, for example,
    # can be a correct action (Case 2). Never convert an alignment contradiction
    # directly into an incorrect-action judgment.
    if is_last_successful_query and faithfulness == "No" and reasoning == "Incorrect":
        action = "Correct"
    elif is_last_successful_query and faithfulness == "No" and reasoning == "Correct":
        action = "Alternative valid"
    elif reasoning == "Correct" and faithfulness == "Yes":
        action = "Correct"
    elif reasoning == "Incorrect" and faithfulness == "Yes":
        action = "Incorrect"
    elif reasoning == "Unverifiable" or faithfulness == "Unverifiable":
        action = "Unverifiable"
    else:
        action = "Ambiguous"

    evidence = reasoning_evidence + faithfulness_evidence
    if is_last_successful_query:
        evidence.append("This chunk immediately precedes recorded task success.")
    if action == "Incorrect":
        evidence.append("The action faithfully executes an action-relevant reasoning error.")
    elif action == "Alternative valid":
        evidence.append(
            "Despite misalignment, this different chunk immediately achieved task success; "
            "review as an alternative route."
        )
    elif action == "Correct":
        if is_last_successful_query:
            evidence.append("This chunk directly triggered the recorded task-success predicate.")
        else:
            evidence.append("The action faithfully realizes a locally supported, task-correct next step.")
    else:
        evidence.append(
            "Alignment evidence alone cannot decide whether this different action was task-correct; human review is required."
        )
    derived = derive_case(reasoning, action, faithfulness)
    confidence = "High" if derived in RESOLVED_CASES and len(evidence) >= 3 else "Medium"
    if derived in {"Ambiguous", "Unverifiable"}:
        confidence = "Low"
    return {
        "reasoning": reasoning,
        "action": action,
        "faithfulness": faithfulness,
        "derived_case": derived,
        "confidence": confidence,
        "evidence": " ".join(evidence),
    }


def propose_all(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    last_by_episode = {
        episode: max(int(record["policy_query_index"]) for record in group)
        for episode, group in _episode_groups(records).items()
    }
    return {
        query_id(record): propose_judgment(
            record,
            is_last_successful_query=bool(record.get("episode_success"))
            and int(record["policy_query_index"]) == last_by_episode[str(record["episode_id"])],
        )
        for record in records
    }


def _episode_groups(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["episode_id"])].append(record)
    for group in groups.values():
        group.sort(key=lambda row: int(row["policy_query_index"]))
    return groups


def _stable_tie(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _balanced_batches(items: Sequence[str], maximum: int = 25) -> List[List[str]]:
    if not items:
        return []
    count = math.ceil(len(items) / maximum)
    base, remainder = divmod(len(items), count)
    batches, offset = [], 0
    for index in range(count):
        size = base + (index < remainder)
        batches.append(list(items[offset : offset + size]))
        offset += size
    return batches


def build_review_queues(
    records: Sequence[Mapping[str, Any]],
    proposals: Mapping[str, Mapping[str, Any]],
    *,
    representative_size: int = 110,
    diagnostic_size: int = 70,
    seed: int = 20260902,
) -> Dict[str, Any]:
    """Create a recorded unequal-probability representative design and diagnostic queue."""
    groups = _episode_groups(records)
    if representative_size < len(groups):
        raise ValueError("Representative queue must be large enough to include every episode.")
    rng = random.Random(seed)
    representative = []
    for episode in sorted(groups):
        representative.append(query_id(rng.choice(groups[episode])))
    remaining_ids = sorted({query_id(record) for record in records} - set(representative))
    extra_count = min(representative_size - len(representative), len(remaining_ids))
    representative.extend(rng.sample(remaining_ids, extra_count))
    rng.shuffle(representative)
    representative_set = set(representative)
    total_remaining_after_episode_draw = len(records) - len(groups)
    record_by_id = {query_id(record): record for record in records}
    inclusion = {}
    for _episode, group in groups.items():
        n_episode = len(group)
        probability = 1.0 / n_episode
        if total_remaining_after_episode_draw:
            probability += (1.0 - probability) * extra_count / total_remaining_after_episode_draw
        for record in group:
            inclusion[query_id(record)] = probability

    first_non_case1 = set()
    for group in groups.values():
        for record in group:
            if proposals[query_id(record)]["derived_case"] != "Case 1":
                first_non_case1.add(query_id(record))
                break

    def diagnostic_score(item_id: str) -> tuple[float, str]:
        record = record_by_id[item_id]
        proposal = proposals[item_id]
        labels = set(record.get("alignment", {}).get("diagnostics", {}).get("failure_labels", []))
        score = 0.0
        if not record.get("episode_success"):
            score += 50
        if proposal["derived_case"] in {"Ambiguous", "Unverifiable"}:
            score += 42
        if proposal["confidence"] == "Low":
            score += 18
        if "wrong_object_interaction" in labels:
            score += 45
        if "kinematic_opposite" in labels:
            score += 25
        if "target_moved_away" in labels:
            score += 18
        if "subtask_not_completed" in labels:
            score += 12
        if item_id in first_non_case1:
            score += 30
        return (-score, _stable_tie(seed, item_id))

    diagnostic_candidates = sorted(
        (item_id for item_id in record_by_id if item_id not in representative_set), key=diagnostic_score
    )
    diagnostic = []

    def take(predicate, limit):
        for item_id in diagnostic_candidates:
            if len([chosen for chosen in diagnostic if predicate(chosen)]) >= limit:
                break
            if item_id not in diagnostic and predicate(item_id):
                diagnostic.append(item_id)

    def has_label(item_id, label):
        return label in record_by_id[item_id].get("alignment", {}).get("diagnostics", {}).get("failure_labels", [])

    take(lambda item_id: has_label(item_id, "wrong_object_interaction"), 8)
    take(
        lambda item_id: not record_by_id[item_id].get("episode_success")
        and proposals[item_id]["derived_case"] in {"Case 3", "Case 4", "Case 5"},
        20,
    )
    for rare_case in ("Case 2", "Case 4", "Case 5", "Case 6"):
        take(lambda item_id, rare_case=rare_case: proposals[item_id]["derived_case"] == rare_case, 4)
    take(
        lambda item_id: proposals[item_id]["derived_case"] == "Case 1"
        and proposals[item_id]["confidence"] == "High",
        8,
    )
    take(lambda item_id: proposals[item_id]["derived_case"] in {"Ambiguous", "Unverifiable"}, 25)
    for item_id in diagnostic_candidates:
        if len(diagnostic) >= diagnostic_size:
            break
        if item_id not in diagnostic:
            diagnostic.append(item_id)

    def annotate(queue: str, ids: Sequence[str]) -> List[Dict[str, Any]]:
        batches = _balanced_batches(ids)
        batch_lookup = {item_id: index + 1 for index, batch in enumerate(batches) for item_id in batch}
        output = []
        for position, item_id in enumerate(ids, 1):
            record = record_by_id[item_id]
            output.append(
                {
                    "id": item_id,
                    "queue": queue,
                    "position": position,
                    "batch": batch_lookup[item_id],
                    "inclusion_probability": inclusion[item_id] if queue == "representative" else None,
                    "analysis_weight": 1.0 / inclusion[item_id] if queue == "representative" else None,
                    "diagnostic_priority": None if queue == "representative" else -diagnostic_score(item_id)[0],
                    "episode_id": record["episode_id"],
                    "task_id": record["task_id"],
                    "query_index": record["policy_query_index"],
                }
            )
        return output

    return {
        "sampling_seed": seed,
        "representative_rule": (
            "One query sampled uniformly within each of 90 episodes, then 20 queries sampled uniformly without "
            "replacement from all remaining queries. Query prevalence uses inverse inclusion-probability weights."
        ),
        "diagnostic_rule": (
            "Additional non-overlapping queries ranked for failed episodes, unresolved/low-confidence proposals, "
            "first departures from Case 1, wrong-object interactions, and strong alignment contradictions. "
            "Not used for prevalence."
        ),
        "representative": annotate("representative", representative),
        "diagnostic": annotate("diagnostic", diagnostic),
    }


def _weighted_rate(items: Sequence[Mapping[str, Any]], outcome: str) -> Optional[float]:
    denominator = sum(float(item["analysis_weight"]) for item in items)
    return (
        sum(float(item["analysis_weight"]) for item in items if item["derived_case"] == outcome) / denominator
        if denominator
        else None
    )


def review_statistics(
    queue_items: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 2000,
    seed: int = 20260902,
) -> Dict[str, Any]:
    """Weighted proposal statistics with an episode-clustered percentile interval."""
    episodes: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in queue_items:
        episodes[str(item["episode_id"])].append(item)
    rng = np.random.default_rng(seed)
    episode_ids = sorted(episodes)
    rows = {}
    for outcome in ALL_OUTCOMES:
        estimate = _weighted_rate(queue_items, outcome)
        samples = []
        for _ in range(replicates):
            chosen = rng.choice(episode_ids, size=len(episode_ids), replace=True)
            sample = [item for episode in chosen for item in episodes[str(episode)]]
            value = _weighted_rate(sample, outcome)
            if value is not None:
                samples.append(value)
        success_items = [item for item in queue_items if item["episode_success"]]
        failed_items = [item for item in queue_items if not item["episode_success"]]
        episodes_with = sum(any(item["derived_case"] == outcome for item in group) for group in episodes.values())
        rows[outcome] = {
            "raw_count": sum(item["derived_case"] == outcome for item in queue_items),
            "raw_denominator": len(queue_items),
            "weighted_proportion": estimate,
            "ci": [float(value) for value in np.percentile(samples, [2.5, 97.5])] if samples else None,
            "successful_proportion": _weighted_rate(success_items, outcome),
            "successful_raw_count": sum(item["derived_case"] == outcome for item in success_items),
            "successful_denominator": len(success_items),
            "failed_proportion": _weighted_rate(failed_items, outcome),
            "failed_raw_count": sum(item["derived_case"] == outcome for item in failed_items),
            "failed_denominator": len(failed_items),
            "episodes_with_case": episodes_with,
            "episode_denominator": len(episodes),
        }
    unresolved_count = sum(item["derived_case"] in {"Ambiguous", "Unverifiable"} for item in queue_items)
    unresolved_outcomes = {"Ambiguous", "Unverifiable"}
    unresolved_estimate = sum(
        float(item["analysis_weight"]) for item in queue_items if item["derived_case"] in unresolved_outcomes
    ) / sum(float(item["analysis_weight"]) for item in queue_items)
    unresolved_samples = []
    unresolved_rng = np.random.default_rng(seed + 1)
    for _ in range(replicates):
        chosen = unresolved_rng.choice(episode_ids, size=len(episode_ids), replace=True)
        sample = [item for episode in chosen for item in episodes[str(episode)]]
        denominator = sum(float(item["analysis_weight"]) for item in sample)
        unresolved_samples.append(
            sum(float(item["analysis_weight"]) for item in sample if item["derived_case"] in unresolved_outcomes)
            / denominator
        )
    success_items = [item for item in queue_items if item["episode_success"]]
    failed_items = [item for item in queue_items if not item["episode_success"]]

    def unresolved_rate(items):
        denominator = sum(float(item["analysis_weight"]) for item in items)
        return (
            sum(float(item["analysis_weight"]) for item in items if item["derived_case"] in unresolved_outcomes)
            / denominator
            if denominator
            else None
        )

    rows["Unresolved"] = {
        "raw_count": unresolved_count,
        "raw_denominator": len(queue_items),
        "weighted_proportion": unresolved_estimate,
        "ci": [float(value) for value in np.percentile(unresolved_samples, [2.5, 97.5])],
        "successful_proportion": unresolved_rate(success_items),
        "successful_raw_count": sum(item["derived_case"] in unresolved_outcomes for item in success_items),
        "successful_denominator": len(success_items),
        "failed_proportion": unresolved_rate(failed_items),
        "failed_raw_count": sum(item["derived_case"] in unresolved_outcomes for item in failed_items),
        "failed_denominator": len(failed_items),
        "episodes_with_case": sum(
            any(item["derived_case"] in unresolved_outcomes for item in group) for group in episodes.values()
        ),
        "episode_denominator": len(episodes),
    }
    return {
        "rows": rows,
        "unresolved_count": unresolved_count,
        "unresolved_denominator": len(queue_items),
        "unresolved_rate": unresolved_count / len(queue_items) if queue_items else None,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def _first_index(proposed, predicate) -> Optional[int]:
    match = next((record for record, proposal in proposed if predicate(record, proposal)), None)
    return int(match["policy_query_index"]) if match is not None else None


def episode_milestones(
    records: Sequence[Mapping[str, Any]], proposals: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Optional[int]]]:
    output = {}
    for episode, group in _episode_groups(records).items():
        proposed = [(record, proposals[query_id(record)]) for record in group]
        first_non_case1 = _first_index(proposed, lambda _record, proposal: proposal["derived_case"] != "Case 1")
        first_reasoning_error = _first_index(proposed, lambda _record, proposal: proposal["reasoning"] == "Incorrect")
        first_wrong_action = _first_index(proposed, lambda _record, proposal: proposal["action"] == "Incorrect")
        decisive = None
        if not group[0].get("episode_success"):
            midpoint = len(group) // 2
            decisive = _first_index(
                proposed,
                lambda record, proposal, midpoint=midpoint: proposal["action"] == "Incorrect"
                and proposal["confidence"] in {"High", "Medium"}
                and int(record["policy_query_index"]) >= midpoint,
            )
            decisive = decisive if decisive is not None else first_wrong_action
        recovery = None
        if first_wrong_action is not None:
            recovery = _first_index(
                proposed,
                lambda record, proposal, first_wrong_action=first_wrong_action: int(record["policy_query_index"])
                > first_wrong_action
                and proposal["derived_case"] == "Case 1",
            )
        output[episode] = {
            "first_non_case1": first_non_case1,
            "first_reasoning_error": first_reasoning_error,
            "first_wrong_action": first_wrong_action,
            "proposed_decisive_failure": decisive,
            "recovery": recovery,
        }
    return output


def local_state_change(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a compact, JSON-safe summary for human query adjudication."""
    before = record.get("state_before", {})
    after = record.get("state_after", {})
    before_ee = np.asarray(before.get("ee_position", []), dtype=float)
    after_ee = np.asarray(after.get("ee_position", []), dtype=float)
    delta = (after_ee - before_ee).tolist() if before_ee.shape == (3,) and after_ee.shape == (3,) else None
    before_objects = before.get("objects", {})
    after_objects = after.get("objects", {})
    changed_objects = []
    for name in sorted(set(before_objects) & set(after_objects)):
        old, new = before_objects[name], after_objects[name]
        changes = {}
        if old.get("grasped") != new.get("grasped"):
            changes["grasped"] = [old.get("grasped"), new.get("grasped")]
        if old.get("contained_by") != new.get("contained_by"):
            changes["contained_by"] = [old.get("contained_by"), new.get("contained_by")]
        if old.get("open_state") != new.get("open_state"):
            changes["open_state"] = [old.get("open_state"), new.get("open_state")]
        if old.get("closed_state") != new.get("closed_state"):
            changes["closed_state"] = [old.get("closed_state"), new.get("closed_state")]
        old_position = np.asarray(old.get("position", []), dtype=float)
        new_position = np.asarray(new.get("position", []), dtype=float)
        if old_position.shape == (3,) and new_position.shape == (3,):
            displacement = float(np.linalg.norm(new_position - old_position))
            if displacement >= 0.005:
                changes["displacement_m"] = displacement
        if changes:
            changed_objects.append({"name": name, "changes": changes})
    diagnostics = record.get("alignment", {}).get("diagnostics", {})
    return {
        "ee_delta": delta,
        "gripper_open": [before.get("gripper_open"), after.get("gripper_open")],
        "gripper_openness": [before.get("gripper_openness"), after.get("gripper_openness")],
        "target": diagnostics.get("target"),
        "subtask": diagnostics.get("subtask"),
        "wrong_object_interaction": diagnostics.get("wrong_object_interaction", []),
        "changed_objects": changed_objects[:12],
        "executed_chunk_length": record.get("executed_chunk_length"),
        "decoded_chunk_length": len(record.get("decoded_action_chunk", [])),
    }
