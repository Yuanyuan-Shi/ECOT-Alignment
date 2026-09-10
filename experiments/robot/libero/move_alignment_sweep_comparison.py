"""Compare a paired Move-alignment checkpoint sweep and update the audit HTML."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from experiments.robot.libero.move_alignment_comparison import THRESHOLD, summarize


SECTION_ID = "move-alignment-lambda-sweep-comparison"
EPISODE_RE = re.compile(r"-task(?P<task>\d+)-episode(?P<trial>\d+)-seed(?P<seed>\d+)$")


def _parse_assignment(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {value!r}")
    name, assigned = value.split("=", 1)
    if not name or not assigned:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {value!r}")
    return name, assigned


def _schedule(root: Path) -> List[Tuple[int, int, int]]:
    episodes = json.loads((root / "episode_summary.json").read_text(encoding="utf-8"))
    schedule = []
    for episode in episodes:
        match = EPISODE_RE.search(str(episode["episode_id"]))
        if match is None:
            raise ValueError(f"cannot parse paired schedule from {episode['episode_id']!r}")
        schedule.append((int(match["task"]), int(match["trial"]), int(match["seed"])))
    return sorted(schedule)


def validate_paired(models: Mapping[str, Mapping[str, Any]], trials_per_task: int) -> None:
    reference_name = next(iter(models))
    reference_root = Path(models[reference_name]["root"])
    reference = _schedule(reference_root)
    expected = 90 * trials_per_task
    if len(reference) != expected:
        raise ValueError(f"{reference_name}: expected {expected} episodes, found {len(reference)}")
    if sorted({task for task, _, _ in reference}) != list(range(90)):
        raise ValueError(f"{reference_name}: task IDs are not exactly 0..89")
    if any(sum(task == candidate for candidate, _, _ in reference) != trials_per_task for task in range(90)):
        raise ValueError(f"{reference_name}: expected {trials_per_task} trials for every task")
    for name, model in models.items():
        actual = _schedule(Path(model["root"]))
        if actual != reference:
            raise ValueError(f"{name}: task/trial/seed schedule does not match {reference_name}")


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}%"


def _outcome_cell(model: Mapping[str, Any], outcome: str) -> str:
    stats = model["outcome_move_misalignment"][outcome]
    if not stats["episodes"]:
        return "<strong>N/A</strong><span>no episodes</span>"
    return (
        f"<strong>{_pct(stats['mean_episode_rate'])}</strong>"
        f"<span>mean across {stats['episodes']} episodes</span>"
        f"<span>{stats['mismatches']}/{stats['queries']} queries "
        f"({_pct(stats['pooled_rate'])} pooled)</span>"
    )


def render_html(result: Mapping[str, Any]) -> str:
    rows = []
    for key in result["model_order"]:
        model = result["models"][key]
        run = model.get("wandb_url")
        run_link = f'<a href="{html.escape(run)}">W&amp;B run</a>' if run else "untouched checkpoint"
        rows.append(
            "<tr>"
            f'<th scope="row">{html.escape(model["name"])}<span>{run_link}</span></th>'
            f'<td><strong>{model["successes"]}/{model["episodes"]}</strong><span>{_pct(model["task_success_rate"])}</span></td>'
            f'<td><strong>{model["average_move_cosine"]:.3f}</strong><span>{model["queries"]} policy queries</span></td>'
            f'<td><strong>{model["move_mismatches"]}/{model["queries"]}</strong><span>{_pct(model["move_misalignment_rate"])}</span></td>'
            f'<td>{_outcome_cell(model, "successful")}</td>'
            f'<td>{_outcome_cell(model, "failed")}</td>'
            "</tr>"
        )
    return f"""
<section id="{SECTION_ID}">
<h2>Move-loss weight sweep — 90 tasks, two paired trials per checkpoint</h2>
<p class="lead">The untouched pretrained checkpoint and three independently fine-tuned checkpoints were evaluated on all 90 LIBERO tasks using the same two new trials per task (base seed <code>{result['base_seed']}</code>; identical seeds and initial-state indices). A query is mismatched when <code>R_move &lt; {THRESHOLD}</code>; <code>MOVE: stop</code> has <code>R_move = 0</code>. Target and Subtask are excluded.</p>
<div class="table-wrap"><table class="primary"><thead><tr><th>Checkpoint</th><th>Task success</th><th>Average Move cosine</th><th>Overall Move mismatch</th><th>Successful-episode average mismatch</th><th>Failed-episode average mismatch</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p><strong>Outcome columns:</strong> the headline is the unweighted mean of per-episode mismatch ratios; the final line gives pooled query counts. Query totals differ because successful episodes can terminate early.</p>
<p><strong>Interpretation:</strong> this is a paired preliminary comparison, not a statistically conclusive benchmark. Each tuned run used 500 optimizer steps from the same untouched checkpoint; only <code>lambda_move</code> changed.</p>
</section>""".strip()


def update_report(report: Path, section: str) -> None:
    document = report.read_text(encoding="utf-8")
    marker = f'<section id="{SECTION_ID}">'
    if marker in document:
        start = document.index(marker)
        end = document.index("</section>", start) + len("</section>")
        document = document[:start] + section + document[end:]
    else:
        document = document.replace("</main>", section + "\n</main>", 1)
    report.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=_parse_assignment, required=True, metavar="NAME=ROOT")
    parser.add_argument("--display-name", action="append", type=_parse_assignment, default=[], metavar="NAME=TEXT")
    parser.add_argument("--wandb-run", action="append", type=_parse_assignment, default=[], metavar="NAME=URL")
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--trials-per-task", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--review-report", type=Path, required=True)
    args = parser.parse_args()

    display_names = dict(args.display_name)
    wandb_runs = dict(args.wandb_run)
    models: Dict[str, Dict[str, Any]] = {}
    order = []
    for key, root_text in args.model:
        if key in models:
            raise ValueError(f"duplicate model key: {key}")
        root = Path(root_text)
        model = summarize(root, display_names.get(key, key))
        model["wandb_url"] = wandb_runs.get(key)
        models[key] = model
        order.append(key)
    if len(models) < 2:
        raise ValueError("at least two --model arguments are required")
    validate_paired(models, args.trials_per_task)
    result = {
        "schema_version": 1,
        "base_seed": args.base_seed,
        "trials_per_task": args.trials_per_task,
        "move_alignment_threshold": THRESHOLD,
        "model_order": order,
        "models": models,
    }
    section = render_html(result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    standalone = (
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>ECoT Move-alignment sweep</title><style>body{font:16px/1.45 system-ui,sans-serif;max-width:1600px;margin:auto;padding:24px;color:#1d2430}h2{color:#17365d}.lead{font-size:18px}.table-wrap{overflow-x:auto;border:1px solid #b9c5d2;border-radius:8px}table{border-collapse:collapse;width:100%}th,td{padding:11px;border-bottom:1px solid #d7dee7;text-align:left;vertical-align:top}thead th{background:#17365d;color:white}tbody tr:nth-child(even){background:#f4f7fa}.primary td strong,.primary td span,.primary th span{display:block}.primary td span,.primary th span{color:#566474;font-weight:normal}</style></head><body>'
        + section
        + "</body></html>"
    )
    args.output_html.write_text(standalone, encoding="utf-8")
    update_report(args.review_report, section)
    print(args.output_html)


if __name__ == "__main__":
    main()
