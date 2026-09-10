"""Summarize paired LIBERO Move-alignment evaluations and update the review HTML."""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


THRESHOLD = 0.5


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def move_reward(record: Dict[str, Any], epsilon: float = 1e-8) -> float:
    axes = record.get("claims", {}).get("motion_axes", {})
    reasoning = np.asarray([axes.get(axis, 0) for axis in ("x", "y", "z")], dtype=float)
    if np.linalg.norm(reasoning) == 0:
        return 0.0
    before = np.asarray(record["state_before"]["ee_position"], dtype=float)
    after = np.asarray(record["state_after"]["ee_position"], dtype=float)
    displacement = after - before
    return float(np.dot(reasoning, displacement) / (np.linalg.norm(reasoning) * np.linalg.norm(displacement) + epsilon))


def summarize(root: Path, name: str, *, reward_fn=move_reward) -> Dict[str, Any]:
    queries = _read_jsonl(root / "alignment_queries.jsonl")
    episodes = json.loads((root / "episode_summary.json").read_text(encoding="utf-8"))
    rewards = np.asarray([reward_fn(record) for record in queries], dtype=float)
    by_task_queries: Dict[int, list] = defaultdict(list)
    by_task_episodes: Dict[int, list] = defaultdict(list)
    for record, reward in zip(queries, rewards):
        by_task_queries[int(record["task_id"])].append(reward)
    for episode in episodes:
        by_task_episodes[int(episode["task_id"])].append(bool(episode["episode_success"]))
    rewards_by_episode: Dict[str, List[float]] = defaultdict(list)
    for record, reward in zip(queries, rewards):
        rewards_by_episode[str(record["episode_id"])].append(float(reward))

    outcome_misalignment = {}
    for key, success in (("successful", True), ("failed", False)):
        selected = [episode for episode in episodes if bool(episode["episode_success"]) is success]
        episode_rates = []
        pooled_rewards = []
        for episode in selected:
            episode_rewards = rewards_by_episode[str(episode["episode_id"])]
            episode_rates.append(float(np.mean(np.asarray(episode_rewards) < THRESHOLD)))
            pooled_rewards.extend(episode_rewards)
        pooled = np.asarray(pooled_rewards, dtype=float)
        outcome_misalignment[key] = {
            "episodes": len(selected),
            "queries": int(pooled.size),
            "mismatches": int((pooled < THRESHOLD).sum()),
            "pooled_rate": float((pooled < THRESHOLD).mean()) if pooled.size else None,
            "mean_episode_rate": float(np.mean(episode_rates)) if episode_rates else None,
        }
    tasks = []
    for task_id in sorted(by_task_episodes):
        task_rewards = np.asarray(by_task_queries[task_id], dtype=float)
        task_episodes = by_task_episodes[task_id]
        tasks.append(
            {
                "task_id": task_id,
                "instruction": next(e["task_instruction"] for e in episodes if int(e["task_id"]) == task_id),
                "successes": int(sum(task_episodes)),
                "episodes": len(task_episodes),
                "success_rate": float(np.mean(task_episodes)),
                "queries": int(task_rewards.size),
                "move_reward": float(task_rewards.mean()),
                "move_mismatches": int((task_rewards < THRESHOLD).sum()),
                "move_misalignment_rate": float((task_rewards < THRESHOLD).mean()),
            }
        )
    return {
        "name": name,
        "root": str(root.resolve()),
        "episodes": len(episodes),
        "successes": int(sum(bool(episode["episode_success"]) for episode in episodes)),
        "task_success_rate": float(np.mean([episode["episode_success"] for episode in episodes])),
        "queries": len(queries),
        "average_move_cosine": float(rewards.mean()),
        "move_mismatches": int((rewards < THRESHOLD).sum()),
        "move_misalignment_rate": float((rewards < THRESHOLD).mean()),
        "outcome_move_misalignment": outcome_misalignment,
        "tasks": tasks,
    }


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_html(result: Dict[str, Any]) -> str:
    models = result["models"]
    base, tuned = models["pretrained"], models["fine_tuned"]
    summary_rows = "".join(
        "<tr>"
        f"<th scope=\"row\">{html.escape(model['name'])}</th>"
        f"<td><strong>{model['successes']}/{model['episodes']}</strong><span>{_pct(model['task_success_rate'])}</span></td>"
        f"<td><strong>{model['average_move_cosine']:.3f}</strong><span>{model['queries']} policy queries</span></td>"
        f"<td><strong>{model['move_mismatches']}/{model['queries']}</strong><span>{_pct(model['move_misalignment_rate'])}</span></td>"
        f"<td><strong>{_pct(model['outcome_move_misalignment']['successful']['mean_episode_rate'])}</strong>"
        f"<span>mean across {model['outcome_move_misalignment']['successful']['episodes']} successful episodes</span>"
        f"<span>{model['outcome_move_misalignment']['successful']['mismatches']}/{model['outcome_move_misalignment']['successful']['queries']} queries "
        f"({_pct(model['outcome_move_misalignment']['successful']['pooled_rate'])} pooled)</span></td>"
        f"<td><strong>{_pct(model['outcome_move_misalignment']['failed']['mean_episode_rate'])}</strong>"
        f"<span>mean across {model['outcome_move_misalignment']['failed']['episodes']} failed episodes</span>"
        f"<span>{model['outcome_move_misalignment']['failed']['mismatches']}/{model['outcome_move_misalignment']['failed']['queries']} queries "
        f"({_pct(model['outcome_move_misalignment']['failed']['pooled_rate'])} pooled)</span></td>"
        "</tr>"
        for model in (base, tuned)
    )
    success_delta = 100 * (tuned["task_success_rate"] - base["task_success_rate"])
    cosine_delta = tuned["average_move_cosine"] - base["average_move_cosine"]
    mismatch_delta = 100 * (tuned["move_misalignment_rate"] - base["move_misalignment_rate"])
    return f"""
<section id="move-alignment-finetune-comparison">
<h2>Pretrained vs Move-alignment fine-tuned — three paired trials per task</h2>
<p class="lead">Both checkpoints were evaluated on the same nine LIBERO tasks with three new paired trials per task (base seed <code>{result['base_seed']}</code>; identical task seeds and initial-state indices). MOVE: stop uses <code>R_move = 0</code>. A query is Move-aligned when <code>R_move ≥ {THRESHOLD}</code>. Target and Subtask are excluded.</p>
<div class="table-wrap"><table class="primary"><thead><tr><th>Checkpoint</th><th>Task success</th><th>Average Move cosine</th><th>Move misalignment</th><th>Successful-episode average mismatch</th><th>Failed-episode average mismatch</th></tr></thead><tbody>{summary_rows}</tbody></table></div>
<p><strong>Fine-tuned − pretrained:</strong> task success {success_delta:+.1f} pp; average Move cosine {cosine_delta:+.3f}; Move-misalignment rate {mismatch_delta:+.1f} pp.</p>
<p><strong>Interpretation:</strong> this is a preliminary feasibility comparison (27 episodes per checkpoint), not a statistically conclusive evaluation. Query counts can differ because successful episodes terminate early.</p>
</section>""".strip()


def update_report(report: Path, comparison_html: str) -> None:
    document = report.read_text(encoding="utf-8")
    start_marker = '<section id="move-alignment-finetune-comparison">'
    if start_marker in document:
        start = document.index(start_marker)
        end = document.index("</section>", start) + len("</section>")
        document = document[:start] + comparison_html + document[end:]
    else:
        document = document.replace("</main>", comparison_html + "\n</main>", 1)
    report.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-root", type=Path, required=True)
    parser.add_argument("--fine-tuned-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--review-report", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 2,
        "base_seed": args.base_seed,
        "trials_per_task": 3,
        "move_alignment_threshold": THRESHOLD,
        "models": {
            "pretrained": summarize(args.pretrained_root, "Untouched pretrained"),
            "fine_tuned": summarize(args.fine_tuned_root, "Move-alignment fine-tuned"),
        },
    }
    comparison_html = render_html(result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_html.write_text(
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>ECoT Move-alignment comparison</title><style>body{font:16px/1.45 system-ui,sans-serif;max-width:1400px;margin:auto;padding:24px;color:#1d2430}h2{color:#17365d}.lead{font-size:18px}.table-wrap{overflow-x:auto;border:1px solid #b9c5d2;border-radius:8px}table{border-collapse:collapse;width:100%}th,td{padding:11px;border-bottom:1px solid #d7dee7;text-align:left;vertical-align:top}thead th{background:#17365d;color:white}tbody tr:nth-child(even){background:#f4f7fa}.primary td strong,.primary td span{display:block}.primary td span{color:#566474}</style></head><body>'
        + comparison_html
        + "</body></html>",
        encoding="utf-8",
    )
    update_report(args.review_report, comparison_html)
    print(args.output_html)


if __name__ == "__main__":
    main()
