"""Structured output and summaries for LIBERO alignment evaluation."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional

from experiments.robot.libero.alignment_evaluator import to_jsonable

COMPONENTS = ("kinematic_score", "target_score", "subtask_score", "aggregate_score")


def _metric(values: Iterable[Optional[float]], denominator: int) -> Dict[str, Any]:
    available = [float(value) for value in values if value is not None]
    return {
        "mean": mean(available) if available else None,
        "std": pstdev(available) if len(available) > 1 else (0.0 if available else None),
        "count": len(available),
        "coverage": len(available) / denominator if denominator else None,
    }


def _query_metrics(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = {
        component: _metric((record["alignment"].get(component) for record in records), len(records))
        for component in COMPONENTS
    }
    warnings = sum(bool(record.get("claims", {}).get("parse_warnings")) for record in records)
    unresolved = sum(
        "target_unresolved" in record["alignment"].get("diagnostics", {}).get("failure_labels", []) for record in records
    )
    target_attempts = sum(bool(record.get("claims", {}).get("target_object")) for record in records)
    failures = Counter(
        label for record in records for label in record["alignment"].get("diagnostics", {}).get("failure_labels", [])
    )
    metrics.update(
        {
            "num_policy_queries": len(records),
            "parse_warning_count": warnings,
            "parse_warning_rate": warnings / len(records) if records else None,
            "target_resolution_failure_count": unresolved,
            "target_resolution_attempt_count": target_attempts,
            "target_resolution_failure_rate": unresolved / target_attempts if target_attempts else None,
            "mismatch_counts": dict(sorted(failures.items())),
        }
    )
    return metrics


def summarize_records(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    episode_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        episode_groups[str(record["episode_id"])].append(record)
    successes = [bool(group[-1]["episode_success"]) for group in episode_groups.values() if group]
    summary = _query_metrics(records)
    summary.update(
        {
            "num_episodes": len(episode_groups),
            "num_successful_episodes": sum(successes),
            "task_success_rate": sum(successes) / len(successes) if successes else None,
        }
    )
    for status, expected in (("successful_episodes", True), ("failed_episodes", False)):
        subset = [record for record in records if bool(record["episode_success"]) is expected]
        summary[f"alignment_for_{status}"] = {
            component: _metric((record["alignment"].get(component) for record in subset), len(subset))
            for component in COMPONENTS
        }
        summary[f"alignment_for_{status}"]["num_policy_queries"] = len(subset)

    task_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        task_groups[str(record["task_id"])].append(record)
    per_task = {}
    for task_id, group in sorted(task_groups.items(), key=lambda pair: int(pair[0])):
        task_summary = _query_metrics(group)
        task_episodes = {record["episode_id"]: bool(record["episode_success"]) for record in group}
        task_summary.update(
            {
                "task_instruction": group[0]["task_instruction"],
                "num_episodes": len(task_episodes),
                "task_success_rate": sum(task_episodes.values()) / len(task_episodes),
            }
        )
        per_task[task_id] = task_summary
    summary["per_task"] = per_task
    return to_jsonable(summary)


def merge_rank_outputs(output_dir: str) -> Dict[str, Any]:
    """Merge completed per-rank JSONL and summaries into one global run."""
    root = Path(output_dir)
    record_paths = sorted(root.glob("rank-*/alignment_queries.jsonl"))
    if not record_paths:
        raise FileNotFoundError(f"No per-rank alignment records found beneath {root}")

    records: List[Dict[str, Any]] = []
    for path in record_paths:
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(key=lambda item: (int(item["task_id"]), str(item["episode_id"]), int(item["policy_query_index"])))

    episode_rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("rank-*/episode_summary.json")):
        episode_rows.extend(json.loads(path.read_text(encoding="utf-8")))
    episode_rows.sort(key=lambda item: (int(item["task_id"]), str(item["episode_id"])))

    summary = summarize_records(records)
    if episode_rows:
        summary["num_episodes"] = len(episode_rows)
        summary["num_successful_episodes"] = sum(bool(row["episode_success"]) for row in episode_rows)
        summary["task_success_rate"] = summary["num_successful_episodes"] / len(episode_rows)
        episodes_by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in episode_rows:
            episodes_by_task[str(row["task_id"])].append(row)
        for task_id, episodes in episodes_by_task.items():
            task_summary = summary["per_task"].setdefault(task_id, _query_metrics([]))
            task_summary.update(
                {
                    "task_instruction": episodes[0]["task_instruction"],
                    "num_episodes": len(episodes),
                    "task_success_rate": sum(bool(row["episode_success"]) for row in episodes) / len(episodes),
                }
            )

    with (root / "alignment_queries.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    (root / "episode_summary.json").write_text(
        json.dumps(episode_rows, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    AlignmentRecorder._write_csv(root / "episode_summary.csv", episode_rows)
    (root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    run_row: Dict[str, Any] = {}
    _flatten("", {key: value for key, value in summary.items() if key != "per_task"}, run_row)
    AlignmentRecorder._write_csv(root / "run_summary.csv", [run_row])
    return summary


def _flatten(prefix: str, value: Any, output: Dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif not isinstance(value, list):
        output[prefix] = value


class AlignmentRecorder:
    """Buffers queries until episode success is known, then writes valid JSONL."""

    def __init__(self, output_dir: str, run_name: str = "alignment") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.output_dir / f"{run_name}_queries.jsonl"
        self._handle = self.records_path.open("w", encoding="utf-8")
        self.records: List[Dict[str, Any]] = []
        self._pending: List[Dict[str, Any]] = []
        self.episodes: List[Dict[str, Any]] = []

    def append_query(self, record: Mapping[str, Any]) -> None:
        self._pending.append(to_jsonable(dict(record)))

    def end_episode(self, success: bool, metadata: Optional[Mapping[str, Any]] = None) -> None:
        episode = dict(metadata or (self._pending[0] if self._pending else {}))
        if "episode_id" not in episode:
            raise ValueError("Episode metadata with episode_id is required when an episode has no policy queries")
        episode_metadata = dict(episode)
        episode_metadata["episode_success"] = bool(success)
        self.episodes.append(to_jsonable(episode_metadata))
        for record in self._pending:
            record["episode_success"] = bool(success)
            self._handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            self.records.append(record)
        self._handle.flush()
        self._pending.clear()

    def close(self) -> Dict[str, Any]:
        if self._pending:
            raise RuntimeError("Cannot finalize alignment output with an unfinished episode")
        self._handle.close()
        summary = summarize_records(self.records)
        summary["num_episodes"] = len(self.episodes)
        summary["num_successful_episodes"] = sum(episode["episode_success"] for episode in self.episodes)
        summary["task_success_rate"] = summary["num_successful_episodes"] / len(self.episodes) if self.episodes else None
        episodes_by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        records_by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for episode in self.episodes:
            episodes_by_task[str(episode["task_id"])].append(episode)
        for record in self.records:
            records_by_task[str(record["task_id"])].append(record)
        for task_id, episodes in episodes_by_task.items():
            task_summary = summary["per_task"].setdefault(task_id, _query_metrics(records_by_task[task_id]))
            task_summary.update(
                {
                    "task_instruction": episodes[0]["task_instruction"],
                    "num_episodes": len(episodes),
                    "task_success_rate": sum(episode["episode_success"] for episode in episodes) / len(episodes),
                }
            )
        (self.output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        episode_rows = []
        groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for record in self.records:
            groups[record["episode_id"]].append(record)
        for episode in sorted(self.episodes, key=lambda item: str(item["episode_id"])):
            episode_id = episode["episode_id"]
            records = groups[episode_id]
            row = {
                **episode,
                "episode_id": episode_id,
                "task_id": episode["task_id"],
                "task_instruction": episode["task_instruction"],
                "episode_success": episode["episode_success"],
                **_query_metrics(records),
            }
            episode_rows.append(to_jsonable(row))
        (self.output_dir / "episode_summary.json").write_text(
            json.dumps(episode_rows, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        self._write_csv(self.output_dir / "episode_summary.csv", episode_rows)
        run_row: Dict[str, Any] = {}
        _flatten("", {key: value for key, value in summary.items() if key != "per_task"}, run_row)
        self._write_csv(self.output_dir / "run_summary.csv", [run_row])
        return summary

    @staticmethod
    def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
        flat_rows: List[Dict[str, Any]] = []
        for row in rows:
            flat: Dict[str, Any] = {}
            _flatten("", row, flat)
            flat_rows.append(flat)
        fields = sorted({key for row in flat_rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat_rows)
