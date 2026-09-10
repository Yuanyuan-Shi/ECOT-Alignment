"""Append a Move-only human-review workspace for a complete LIBERO-90 audit."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


THRESHOLD = 0.5
STORAGE_KEY = "ecotRound2MoveAlignmentReviewsV1"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reasoning_fields(raw: str) -> Dict[str, str]:
    names = (
        "PLAN|SUBTASK REASONING|SUBTASK|MOVE REASONING|MOVE|GRIPPER POSITION|"
        "GRIPPER|VISIBLE OBJECTS|OBJECTS"
    )
    matches = list(re.finditer(rf"(?:^|\s)({names}):\s*", raw, re.I))
    fields: Dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        fields[match.group(1).upper()] = raw[match.end() : end].strip()
    return fields


def _move_vector(record: Mapping[str, Any]) -> np.ndarray:
    axes = record.get("claims", {}).get("motion_axes", {})
    return np.asarray([axes.get(axis, 0) for axis in ("x", "y", "z")], dtype=float)


def _move_evidence(record: Mapping[str, Any], epsilon: float = 1e-8) -> Dict[str, Any]:
    move = _move_vector(record)
    before = np.asarray(record["state_before"]["ee_position"], dtype=float)
    after = np.asarray(record["state_after"]["ee_position"], dtype=float)
    delta = after - before
    move_norm = float(np.linalg.norm(move))
    delta_norm = float(np.linalg.norm(delta))
    dot = float(np.dot(move, delta))
    score = 0.0 if move_norm == 0 else dot / (move_norm * delta_norm + epsilon)
    return {
        "move": move.tolist(),
        "delta_m": delta.tolist(),
        "move_norm": move_norm,
        "delta_norm_m": delta_norm,
        "dot_m": dot,
        "score": score,
        "label": "T" if score >= THRESHOLD else "F",
    }


def _relative_frame(run_root: Path, report_dir: Path, record: Mapping[str, Any], name: str) -> str:
    frame = run_root / str(record.get("frames", {}).get(name, ""))
    return Path(os.path.relpath(frame, report_dir)).as_posix()


def build_payload(run_root: Path, report_dir: Path) -> Dict[str, Any]:
    queries = _read_jsonl(run_root / "alignment_queries.jsonl")
    episodes = json.loads((run_root / "episode_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_root / "experiment_metadata.json").read_text(encoding="utf-8"))
    task_ids = [int(episode["task_id"]) for episode in episodes]
    if len(episodes) != 90 or sorted(task_ids) != list(range(90)):
        raise ValueError(
            "Round-2 review requires exactly one complete episode for each LIBERO-90 task; "
            f"found {len(episodes)} episodes with task IDs {sorted(task_ids)}"
        )
    by_episode: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in queries:
        by_episode[str(record["episode_id"])].append(record)

    output_episodes = []
    for episode in sorted(episodes, key=lambda item: int(item["task_id"])):
        episode_id = str(episode["episode_id"])
        records = sorted(by_episode[episode_id], key=lambda item: int(item["policy_query_index"]))
        if not records:
            raise ValueError(f"Episode {episode_id} has no recorded policy queries")
        output_records = []
        for record in records:
            fields = _reasoning_fields(str(record.get("reasoning_raw", "")))
            output_records.append(
                {
                    "id": f"{episode_id}::q{record['policy_query_index']}",
                    "query": int(record["policy_query_index"]),
                    "reasoning_raw": str(record.get("reasoning_raw", "")),
                    "fields": fields,
                    "claims": record.get("claims", {}),
                    "evidence": _move_evidence(record),
                    "frames": {
                        name: _relative_frame(run_root, report_dir, record, name)
                        for name in ("before", "intermediate", "after")
                    },
                }
            )
        output_episodes.append(
            {
                "episode_id": episode_id,
                "task_id": int(episode["task_id"]),
                "instruction": str(episode["task_instruction"]),
                "seed": int(episode["seed"]),
                "success": bool(episode["episode_success"]),
                "queries": output_records,
            }
        )
    return {
        "schema_version": 1,
        "storage_key": STORAGE_KEY,
        "threshold": THRESHOLD,
        "run_root": str(run_root.resolve()),
        "checkpoint": metadata.get("checkpoint_path"),
        "base_seed": metadata.get("base_seed"),
        "episodes": output_episodes,
        "episode_count": len(output_episodes),
        "query_count": len(queries),
    }


def _vector(values: Sequence[float], precision: int = 0, plus: bool = False) -> str:
    parts = []
    for value in values:
        number = float(value)
        text = f"{number:.{precision}f}"
        parts.append(f"+{text}" if plus and number > 0 else text)
    return "[" + ", ".join(parts) + "]"


def summarize_labels(payload: Mapping[str, Any], replicates: int = 2000, seed: int = 20261101) -> List[Dict[str, Any]]:
    """Summarize score-defined mismatch categories with an episode bootstrap."""
    successful = [episode for episode in payload["episodes"] if episode["success"]]
    failed = [episode for episode in payload["episodes"] if not episode["success"]]
    categories = (
        ("Move mismatch", lambda score: score < THRESHOLD),
        ("↳ Partial mismatch", lambda score: 0 < score < THRESHOLD),
        ("↳ Opposite / nonpositive", lambda score: score <= 0),
    )

    def count(episodes: Sequence[Mapping[str, Any]], predicate) -> tuple[int, int, float]:
        scores = [float(query["evidence"]["score"]) for episode in episodes for query in episode["queries"]]
        numerator = sum(predicate(score) for score in scores)
        return numerator, len(scores), numerator / len(scores) if scores else float("nan")

    rng = np.random.default_rng(seed)
    rows = []
    for label, predicate in categories:
        overall_n, overall_d, overall_rate = count(payload["episodes"], predicate)
        success_n, success_d, success_rate = count(successful, predicate)
        failed_n, failed_d, failed_rate = count(failed, predicate)
        samples = []
        for _ in range(replicates):
            sampled_success = [successful[index] for index in rng.integers(0, len(successful), len(successful))]
            sampled_failed = [failed[index] for index in rng.integers(0, len(failed), len(failed))]
            samples.append(count(sampled_failed, predicate)[2] - count(sampled_success, predicate)[2])
        rows.append(
            {
                "category": label,
                "overall_numerator": overall_n,
                "overall_denominator": overall_d,
                "overall_rate": overall_rate,
                "successful_numerator": success_n,
                "successful_denominator": success_d,
                "successful_rate": success_rate,
                "failed_numerator": failed_n,
                "failed_denominator": failed_d,
                "failed_rate": failed_rate,
                "difference": failed_rate - success_rate,
                "bootstrap_ci": np.percentile(samples, [2.5, 97.5]).tolist(),
            }
        )
    return rows


def freeze_effective_labels(payload: Mapping[str, Any], destination: Path) -> None:
    records = [
        {
            "id": query["id"],
            "task_id": episode["task_id"],
            "episode_id": episode["episode_id"],
            "query": query["query"],
            "episode_success": episode["success"],
            "move_cosine": query["evidence"]["score"],
            "move_alignment": query["evidence"]["label"],
            "source": "accepted_initial_cosine",
        }
        for episode in payload["episodes"]
        for query in episode["queries"]
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "captured_at": "2026-09-05",
                "storage_key": STORAGE_KEY,
                "browser_overrides_present": False,
                "threshold": THRESHOLD,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _summary_table(payload: Mapping[str, Any]) -> str:
    successes = sum(bool(episode["success"]) for episode in payload["episodes"])
    failures = len(payload["episodes"]) - successes
    success_queries = sum(len(episode["queries"]) for episode in payload["episodes"] if episode["success"])
    failed_queries = payload["query_count"] - success_queries
    rows = "".join(
        "<tr>"
        f'<th scope="row">{html.escape(row["category"])}</th>'
        f'<td><strong>{row["overall_denominator"]}/{row["overall_denominator"]}</strong><span>100.0%</span></td>'
        f'<td><strong>{row["overall_numerator"]}/{row["overall_denominator"]}</strong><span>{_pct(row["overall_rate"])}</span></td>'
        f'<td><strong>{row["successful_denominator"]}/{row["successful_denominator"]}</strong><span>100.0%</span></td>'
        f'<td><strong>{row["successful_numerator"]}/{row["successful_denominator"]}</strong><span>{_pct(row["successful_rate"])}</span></td>'
        f'<td><strong>{row["failed_denominator"]}/{row["failed_denominator"]}</strong><span>100.0%</span></td>'
        f'<td><strong>{row["failed_numerator"]}/{row["failed_denominator"]}</strong><span>{_pct(row["failed_rate"])}</span></td>'
        f'<td><strong>{100 * row["difference"]:+.1f} pp</strong><span>failed − successful</span></td>'
        f'<td><strong>{100 * row["bootstrap_ci"][0]:+.1f} to {100 * row["bootstrap_ci"][1]:+.1f} pp</strong><span>episode bootstrap</span></td>'
        "</tr>"
        for row in payload["summary"]
    )
    return (
        '<h3>Current Move-label query-level results</h3>'
        '<p>The current labels are frozen cosine-based defaults; no Round-2 browser overrides were present. '
        'Partial mismatch is <code>0 &lt; cosine &lt; 0.5</code>; opposite/nonpositive is <code>cosine ≤ 0</code>. '
        'The two subcategories partition all Move mismatches.</p>'
        '<div class="table-wrap"><table class="primary outcome-split"><thead>'
        f'<tr><th rowspan="2">Category</th><th colspan="2">Overall — 90 episodes, {payload["query_count"]} queries</th>'
        f'<th colspan="2">Successful episodes — {successes} episodes, {success_queries} queries</th>'
        f'<th colspan="2">Failed episodes — {failures} episodes, {failed_queries} queries</th>'
        '<th colspan="2">Failure–success comparison</th></tr><tr><th>Coverage</th><th>Cases</th>'
        '<th>Coverage</th><th>Cases</th><th>Coverage</th><th>Cases</th><th>Difference</th><th>Bootstrap 95% CI</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _reasoning_html(record: Mapping[str, Any], instruction: str) -> str:
    fields = record["fields"]
    components = (
        ("TASK", instruction),
        ("PLAN", fields.get("PLAN")),
        ("SUBTASK", fields.get("SUBTASK") or record.get("claims", {}).get("subtask")),
        ("MOVE", fields.get("MOVE")),
        ("GRIPPER", fields.get("GRIPPER POSITION") or fields.get("GRIPPER")),
        ("OBJECTS", fields.get("VISIBLE OBJECTS") or fields.get("OBJECTS")),
    )
    lines = "".join(
        f'<span class="claim-line"><span class="claim-key component-{name.lower()}">{name}</span>'
        f'{html.escape(str(value or "not stated"))}</span>'
        for name, value in components
    )
    return lines + f'<details><summary>Exact ECoT trace</summary><pre class="reason">{html.escape(record["reasoning_raw"])}</pre></details>'


def _action_html(record: Mapping[str, Any]) -> str:
    delta_mm = np.asarray(record["evidence"]["delta_m"]) * 1000
    words = []
    for value, negative, positive in zip(delta_mm, ("forward", "left", "down"), ("back", "right", "up")):
        if value < -3:
            words.append(negative)
        elif value > 3:
            words.append(positive)
    direction = " + ".join(words) if words else "no net translation above 3 mm"
    return (
        f"<strong>{html.escape(direction)}</strong>"
        f'<span class="metric-detail">Σ executed chunk ΔEE = {_vector(delta_mm, 2, plus=True)} mm</span>'
        f'<details><summary>Before / intermediate / after frames</summary><div class="compact-evidence">'
        + "".join(f'<img loading="lazy" src="{record["frames"][name]}" alt="{name} frame">' for name in ("before", "intermediate", "after"))
        + "</div></details>"
    )


def _metric_html(record: Mapping[str, Any]) -> str:
    evidence = record["evidence"]
    move = evidence["move"]
    delta_mm = np.asarray(evidence["delta_m"]) * 1000
    dot_mm = float(evidence["dot_m"]) * 1000
    delta_norm_mm = float(evidence["delta_norm_m"]) * 1000
    denominator = float(evidence["move_norm"]) * delta_norm_mm
    if float(evidence["move_norm"]) == 0:
        computation = "cos convention = 0.000 because MOVE is [0, 0, 0]"
    else:
        computation = (
            "cos = MOVE·ΔEE / (||MOVE|| ||ΔEE||)\n"
            f"= {dot_mm:.2f} / ({evidence['move_norm']:.3f} × {delta_norm_mm:.2f})\n"
            f"= {dot_mm:.2f} / {denominator:.2f} = {evidence['score']:.3f}"
        )
    css = "metric-pass" if evidence["label"] == "T" else "metric-fail"
    title = "Aligned" if evidence["label"] == "T" else "Not aligned"
    detail = (
        f"cosine {evidence['score']:.3f}\nMOVE {_vector(move)}\n"
        f"Σ 10-step action ΔEE {_vector(delta_mm, 2, plus=True)} mm\n"
        f"||MOVE|| = {evidence['move_norm']:.3f}; ||ΔEE|| = {delta_norm_mm:.2f} mm\n"
        f"{computation}\n≥ 0.5 → T; < 0.5 → F"
    )
    return f'<td class="align-cell {css}"><strong>{title}</strong><span class="metric-detail">{html.escape(detail)}</span></td>'


def render_section(payload: Mapping[str, Any]) -> str:
    if "summary" not in payload:
        payload = {**payload, "summary": summarize_labels(payload)}
    episodes_html = []
    for episode_index, episode in enumerate(payload["episodes"]):
        rows = []
        for record in episode["queries"]:
            default = record["evidence"]["label"]
            human = (
                f'<label>Move Alignment<select data-round2-label data-id="{html.escape(record["id"])}" '
                f'data-default="{default}"><option value="T"{" selected" if default == "T" else ""}>T</option>'
                f'<option value="F"{" selected" if default == "F" else ""}>F</option></select></label>'
                f'<div><button data-round2-save="reviewed">Save label</button> '
                f'<button data-round2-save="uncertain">Uncertain</button></div>'
                f'<div class="initial-rationale"><strong>Initial suggestion:</strong> Move {default} '
                f'(cos {record["evidence"]["score"]:.3f}).</div><div class="compact-status">unreviewed</div>'
            )
            rows.append(
                f'<tr data-round2-row data-id="{html.escape(record["id"])}" data-task="{episode["task_id"]}" '
                f'data-outcome="{"success" if episode["success"] else "failure"}" data-default="{default}">'
                f'<th>{record["query"]}</th><td class="reason-cell">{_reasoning_html(record, episode["instruction"])}</td>'
                f'<td class="action-cell">{_action_html(record)}</td>{_metric_html(record)}'
                f'<td class="human-cell"><div class="domain-labels">{human}</div></td></tr>'
            )
        initial = episode["queries"][0]
        episodes_html.append(
            f'<details class="compact-episode" data-round2-episode data-task="{episode["task_id"]}" '
            f'data-outcome="{"success" if episode["success"] else "failure"}"{" open" if episode_index == 0 else ""}>'
            f'<summary>Task {episode["task_id"]} · {"SUCCESS" if episode["success"] else "FAILURE"} · '
            f'{len(episode["queries"])} policy queries — {html.escape(episode["instruction"])}</summary>'
            f'<div class="compact-scene"><img loading="lazy" src="{initial["frames"]["before"]}" alt="Task {episode["task_id"]} initial scene">'
            f'<div><strong>Task scene</strong><p>{html.escape(episode["instruction"])}</p>'
            f'<p>Task {episode["task_id"]} · seed {episode["seed"]} · episode {"SUCCESS" if episode["success"] else "FAILURE"}</p>'
            f'<p><small>Fixed third-person RGB immediately before policy query 0.</small></p></div></div>'
            f'<div class="table-wrap"><table class="step-table round2-table"><thead><tr><th>Step</th>'
            f'<th>ECoT Reasoning (6 components)</th><th>Actual action (Σ executed chunk)</th>'
            f'<th>Move Alignment</th><th>Human label</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></details>'
        )
    payload_json = html.escape(json.dumps({"storage_key": STORAGE_KEY}), quote=False)
    return f"""
<section id="round2-move-review">
<h2>Round 2 — pretrained MiniVLA Move-only review: all 90 LIBERO tasks</h2>
<div class="pilot"><strong>New review set:</strong> one new trial per task, {payload['episode_count']} episodes and {payload['query_count']} policy queries; base seed <code>{payload['base_seed']}</code>. Only Move Alignment is included. Initial labels are <code>T</code> when cosine ≥ {THRESHOLD} and <code>F</code> otherwise.</div>
<p>Cosine uses the ECoT MOVE vector and raw summed measured end-effector displacement over the executed action chunk. MOVE with no directional phrase is <code>[0,0,0]</code> and receives cosine 0 by convention. This workspace uses separate browser storage from the calibrated nine-episode review.</p>
{_summary_table(payload)}
<div class="review-toolbar round2-toolbar"><label>Task<select id="round2-task"><option value="">All 90</option>{''.join(f'<option>{i}</option>' for i in range(90))}</select></label><label>Outcome<select id="round2-outcome"><option value="">All</option><option value="success">Success</option><option value="failure">Failure</option></select></label><label>Status<select id="round2-status"><option value="">All</option><option value="unreviewed">Unreviewed</option><option value="reviewed">Reviewed</option><option value="uncertain">Uncertain</option></select></label><div id="round2-progress"></div><button id="round2-export">Export annotations</button><label>Import annotations<input id="round2-import" type="file" accept="application/json"></label></div>
<div id="round2-episodes">{''.join(episodes_html)}</div>
<script id="round2-config" type="application/json">{payload_json}</script>
</section>
<script src="round2_move_review_ui.js?v=1"></script>
""".strip()


def inject_report(report: Path, section: str) -> None:
    document = report.read_text(encoding="utf-8")
    marker = '<section id="round2-move-review">'
    if marker in document:
        start = document.index(marker)
        script_end = document.index('</script>', document.index('<script src="round2_move_review_ui.js', start)) + len('</script>')
        document = document[:start] + section + document[script_end:]
    else:
        document = document.replace("</main>", section + "\n</main>", 1)
    report.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path)
    parser.add_argument("--review-report", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args.run_root, args.review_report.parent)
    payload["summary"] = summarize_labels(payload)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    labels_output = args.labels_output or args.output_json.parent / "annotations" / "round2_move_labels_effective_90episodes_2026-09-05.json"
    freeze_effective_labels(payload, labels_output)
    inject_report(args.review_report, render_section(payload))
    shutil.copyfile(Path(__file__).with_name("round2_move_review_ui.js"), args.review_report.parent / "round2_move_review_ui.js")
    print(args.review_report)


if __name__ == "__main__":
    main()
