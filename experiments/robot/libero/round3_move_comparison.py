"""Round 3 status and paired evaluation report; preserves every earlier section."""
from __future__ import annotations

import fcntl
import html
import re
import json
from pathlib import Path

from experiments.robot.libero.move_alignment_comparison import summarize, move_reward, _read_jsonl
from experiments.robot.libero.move_alignment_sweep_comparison import render_html, validate_paired, _schedule

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / 'runs/round3/status.json'
SECTION_ID = 'round3-move-alignment-comparison'
BASE = ROOT / 'experiments/robot/libero/results/move-sweep-eval-pretrained-90task-2trial-seed20261201'


def with_bins(root, name):
    model = summarize(root, name)
    values = [move_reward(row) for row in _read_jsonl(root / 'alignment_queries.jsonl')]
    counts = {'c<0': sum(c < 0 for c in values),
              '0<=c<0.5': sum(0 <= c < .5 for c in values),
              'c>=0.5': sum(c >= .5 for c in values)}
    model['cosine_bins'] = {key: {'count': count, 'percentage': 100 * count / len(values)}
                            for key, count in counts.items()}
    return model


def build_result(state):
    models = {'pretrained': with_bins(BASE, 'Untouched pretrained')}
    for key, run in state['runs'].items():
        if run['status'] != 'complete':
            continue
        models[key] = with_bins(ROOT / run['evaluation_root'], run.get("display_name", f"Round 3 lambda={run['lambda']}"))
        models[key]['wandb_url'] = run.get('wandb_url')
    validate_paired(models, 2)
    expected = [(task, trial, 20261201 + task * 2 + trial) for task in range(90) for trial in range(2)]
    for model in models.values():
        if _schedule(Path(model['root'])) != expected:
            raise ValueError('Unexpected Round 2 paired seed schedule')
    base = models['pretrained']
    for model in models.values():
        model['change_vs_pretrained'] = {
            'task_success_percentage_points': 100 * (model['task_success_rate'] - base['task_success_rate']),
            'average_move_cosine': model['average_move_cosine'] - base['average_move_cosine'],
            'move_misalignment_count': model['move_mismatches'] - base['move_mismatches'],
            'move_misalignment_percentage_points': 100 * (model['move_misalignment_rate'] - base['move_misalignment_rate']),
        }
        for outcome in ('successful', 'failed'):
            current = model['outcome_move_misalignment'][outcome]['pooled_rate']
            reference = base['outcome_move_misalignment'][outcome]['pooled_rate']
            model['change_vs_pretrained'][outcome + '_misalignment_percentage_points'] = (
                None if current is None or reference is None else 100 * (current - reference))
    return {'base_seed': 20261201, 'trials_per_task': 2, 'model_order': list(models), 'models': models,
            'status': state, 'loss': 'mean((relu(0.5-c)/1.5)^2)', 'optimizer_steps': 200,
            'warmup_steps': 20, 'dataset_queries': 1557}


def render(result):
    section = render_html(result).replace('move-alignment-lambda-sweep-comparison', SECTION_ID)
    section = section.replace('Move-loss weight sweep — 90 tasks, two paired trials per checkpoint',
                              'Round 3 — thresholded squared-hinge Move loss')
    section = section.replace('Each tuned run used 500 optimizer steps from the same untouched checkpoint; only <code>lambda_move</code> changed.',
        'Round 3 uses 200 optimizer steps, effective batch size 32 (the final batch of a full epoch has 29 queries), '
        '20 warmup steps, and <code>mean((max(0, 0.5-c)/1.5)^2)</code>. All runs start from the untouched checkpoint. '
        'The reused Round 2 dataset has 1,557 queries: 1,245 training and 312 validation.')
    section = section.replace('The untouched pretrained checkpoint and three independently fine-tuned checkpoints were evaluated',
        'Completed checkpoints shown below are evaluated')
    statuses = []
    for run in result['status']['runs'].values():
        url = run.get('wandb_url')
        link = f' — <a href="{html.escape(url)}">W&amp;B</a>' if url else ''
        label = run.get("display_name", f"lambda={run['lambda']}")
        statuses.append(f"<li>{html.escape(label)}: {html.escape(run['status'])}{link}</li>")
    pending_rows = {}
    for key, run in result["status"]["runs"].items():
        if run["status"] != "complete":
            label = html.escape(run.get("display_name", f"Round 3 lambda={run['lambda']}"))
            link = (f'<span><a href="{html.escape(run["wandb_url"])}">W&amp;B run</a></span>'
                    if run.get("wandb_url") else '')
            pending_rows[key] = (f'<tr><th scope="row">{label}{link}</th><td colspan="5">'
                                f'{html.escape(run["status"])}; rollout metrics pending</td></tr>')
    # Interleave pending and completed rows in experiment order, keeping the
    # lambda=2/5 extension ahead of every Round 4 row in all execution states.
    primary = re.search(r'<tbody>(.*?)</tbody>', section, re.S)
    completed_rows = re.findall(r'<tr>.*?</tr>', primary.group(1), re.S)
    row_by_key = dict(zip(result['model_order'], completed_rows))
    row_by_key.update(pending_rows)
    ordered = ['pretrained', *result['status']['runs']]
    body = ''.join(row_by_key[key] for key in ordered)
    section = section[:primary.start(1)] + body + section[primary.end(1):]
    rows = []
    for model in result['models'].values():
        bins = ''.join(f"<td>{b['count']} ({b['percentage']:.1f}%)</td>" for b in model['cosine_bins'].values())
        d = model['change_vs_pretrained']
        rows.append(f"<tr><th>{html.escape(model['name'])}</th>{bins}"
                    f"<td>{d['task_success_percentage_points']:+.2f} pp</td>"
                    f"<td>{d['average_move_cosine']:+.4f}</td>"
                    f"<td>{d['move_misalignment_percentage_points']:+.2f} pp ({d['move_misalignment_count']:+d} queries)</td></tr>")
    extra = '<h3>Experiment status</h3><ul>' + ''.join(statuses) + '</ul>'
    extra += ('<h3>Cosine distribution and change versus pretrained</h3><div class="table-wrap"><table>'
              '<thead><tr><th>Checkpoint</th><th>c &lt; 0</th><th>0 ≤ c &lt; 0.5</th><th>c ≥ 0.5</th>'
              '<th>Δ success</th><th>Δ cosine</th><th>Δ misalignment</th></tr></thead><tbody>'
              + ''.join(rows) + '</tbody></table></div>'
              '<p>Training cosine uses the soft-decoded predicted action chunk. Rollout cosine uses observed '
              'end-effector displacement, exactly as in Round 2. Pending evaluations have no reported result.</p>')
    if any(key.startswith("round4-") for key in result["status"]["runs"]):
        section = section.replace('Round 3 — thresholded squared-hinge Move loss',
                                  'Round 3 and failure-focused Move-alignment experiments')
        extra += ('<p><strong>Failure-focused runs:</strong> each method is tested at lambda=0.5 and lambda=1.0. '
                  'Failure-weight3 uses natural sampling and Move weights 1 for successful-episode queries, '
                  '3 for failed-episode queries, normalized by their sum. Performance loss is unweighted. '
                  'Balanced sampling uses 16 successful and 16 failed queries in every full batch of 32, '
                  'with repeated draws after an outcome pool is exhausted and unit Move weights. '
                  'The training/validation split and evaluation schedule are unchanged.</p>')
    if result['status'].get('pause'):
        resume = html.escape(result['status']['pause']['resume_at'])
        section = section.replace('</h2>', '</h2><p><strong>Experiment queues paused. '
                                  f'Automatic resume scheduled for {resume}.</strong></p>', 1)
    return section.replace('</section>', extra + '\n</section>')


def _main():
    state = json.loads(STATE.read_text())
    for extra_state in (ROOT / 'runs/round3-high-lambda/status.json', ROOT / 'runs/round4/status.json', ROOT / 'runs/round4-lambda1/status.json'):
        if extra_state.exists():
            extra = json.loads(extra_state.read_text())
            if set(state['runs']) & set(extra['runs']):
                raise ValueError('Duplicate experiment keys')
            state['runs'].update(extra['runs'])
    pause_file = ROOT / 'runs/overnight_pause.json'
    if pause_file.exists():
        pause = json.loads(pause_file.read_text())
        if not pause.get('resumed_at'):
            state['pause'] = pause
    result = build_result(state)
    section = render(result)
    out = ROOT / 'analysis_results'
    (out / 'round3_move_alignment_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'round3_move_alignment_comparison.html').write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Round 3 Move alignment</title>'
        '<style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ccc}'
        'td span,th span{display:block}.table-wrap{overflow:auto}</style></head><body>' + section + '</body></html>')
    report = out / 'index.html'
    document = report.read_text()
    # Migrate the earlier separate section into the block directly below the original table.
    document = re.sub(r'<section id="round3-policy-action-alignment">.*?</section>\n?', '', document, flags=re.S)
    marker = f'<section id="{SECTION_ID}">'
    if marker in document:
        start = document.index(marker)
        end = document.index('</section>', start) + len('</section>')
        document = document[:start] + section + document[end:]
    else:
        if '</main>' not in document:
            raise ValueError('Report has no main closing tag')
        document = document.replace('</main>', section + '\n</main>', 1)
    temporary = report.with_suffix('.html.tmp')
    temporary.write_text(document)
    temporary.replace(report)


def main():
    with (ROOT / 'analysis_results/.move_report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _main()


if __name__ == '__main__':
    main()
