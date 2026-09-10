"""Rescore saved paired rollouts using the full predicted action-chunk sum."""
from pathlib import Path
import fcntl
import hashlib
import html
import json
import numpy as np

from experiments.robot.libero.move_alignment_comparison import summarize, _read_jsonl
from experiments.robot.libero.move_alignment_sweep_comparison import validate_paired, _schedule
from experiments.robot.libero.round3_move_comparison import render, main as update_main_report

ROOT = Path(__file__).resolve().parents[3]
SECTION = 'round3-policy-action-alignment'


def policy_action_cosine(record, epsilon=1e-8):
    actions = np.asarray(record['decoded_action_chunk'], dtype=np.float64)
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite full 10x7 policy output: {record.get('episode_id')}")
    axes = record['claims']['motion_axes']
    reasoning = np.asarray([axes.get(axis,0) for axis in ('x','y','z')], dtype=float)
    delta = actions[:, :3].sum(axis=0)
    if np.linalg.norm(reasoning) == 0:
        return 0.0
    return float(np.dot(reasoning,delta) / (np.linalg.norm(reasoning)*np.linalg.norm(delta)+epsilon))


def main():
    state = json.loads((ROOT/'runs/round3/status.json').read_text())
    state['runs'].update(json.loads((ROOT/'runs/round3-high-lambda/status.json').read_text())['runs'])
    for folder in ('round4', 'round4-lambda1'):
        state['runs'].update(json.loads((ROOT/f'runs/{folder}/status.json').read_text())['runs'])
    keys = tuple(state['runs'])
    for key in keys:
        if state['runs'][key]['status'] != 'complete':
            raise RuntimeError(f'Wait for the current experiment to finish: {key}')
    if state['runs']['50']['status'] != 'complete':
        raise RuntimeError('Wait for the current lambda=5 experiment to finish')
    state['runs'] = {key:state['runs'][key] for key in keys}
    specs = {'pretrained': {'name':'Untouched pretrained',
             'root':ROOT/'experiments/robot/libero/results/move-sweep-eval-pretrained-90task-2trial-seed20261201'}}
    specs.update({key:{'name':state['runs'][key].get("display_name", f"Round 3 lambda={state['runs'][key]['lambda']}"),
                       'root':ROOT/state['runs'][key]['evaluation_root']} for key in keys})
    output = ROOT/'analysis_results'
    audit = output/'policy_action_rescoring'
    audit.mkdir(exist_ok=True)
    models = {}
    provenance = {}
    for key,spec in specs.items():
        root = spec['root']
        model = summarize(root,spec['name'],reward_fn=policy_action_cosine)
        rows = _read_jsonl(root/'alignment_queries.jsonl')
        scores = np.asarray([policy_action_cosine(row) for row in rows])
        bins = {'c<0':scores<0, '0<=c<0.5':(scores>=0)&(scores<.5), 'c>=0.5':scores>=.5}
        model['cosine_bins'] = {k:{'count':int(mask.sum()),'percentage':float(100*mask.mean())} for k,mask in bins.items()}
        model['wandb_url'] = state['runs'][key]['wandb_url'] if key!='pretrained' else None
        models[key] = model
        provenance[key] = {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                           for name in ('alignment_queries.jsonl','episode_summary.json')}
        provenance[key]['short_executed_chunks'] = sum(row['executed_chunk_length']<10 for row in rows)
        (audit/f'{key}.jsonl').write_text(''.join(json.dumps({
            'episode_id':row['episode_id'],'policy_query_index':row['policy_query_index'],
            'predicted_translation_sum':np.asarray(row['decoded_action_chunk'])[:,:3].sum(axis=0).tolist(),
            'cosine':float(c),'mismatch':bool(c<.5),'predicted_horizon':10,
            'executed_chunk_length':row['executed_chunk_length']})+'\n' for row,c in zip(rows,scores)))
    validate_paired(models,2)
    expected=[(task,trial,20261201+task*2+trial) for task in range(90) for trial in range(2)]
    for model in models.values():
        assert _schedule(Path(model['root']))==expected
        base=models['pretrained']
        model['change_vs_pretrained']={
            'task_success_percentage_points':100*(model['task_success_rate']-base['task_success_rate']),
            'average_move_cosine':model['average_move_cosine']-base['average_move_cosine'],
            'move_misalignment_percentage_points':100*(model['move_misalignment_rate']-base['move_misalignment_rate']),
            'move_misalignment_count':model['move_mismatches']-base['move_mismatches']}
    result={'models':models,'model_order':list(models),'status':state,'base_seed':20261201,
            'trials_per_task':2,'metric':'cos(reasoning_move_vector, sum(decoded_action_chunk[:, :3]))',
            'epsilon':1e-8,'threshold':.5,'predicted_horizon':10,
            'provenance':provenance,'execution':'rescored saved outputs; no training or simulation rerun'}
    section=render(result).replace('round3-move-alignment-comparison',SECTION)
    section=section.replace('Round 3 and failure-focused Move-alignment experiments','Alignment with predicted policy actions').replace('Round 3 — thresholded squared-hinge Move loss','Alignment with predicted policy actions')
    section=section.replace('Rollout cosine uses observed end-effector displacement, exactly as in Round 2.',
        'This section uses the sum of all ten decoded policy translation commands, including unexecuted trailing commands in truncated chunks.')
    section=section.replace('</h2>', '</h2><p><strong>Metric:</strong> '
        '<code>c = dot(r, sum(a_xyz[0:10])) / (norm(r) * norm(sum(a_xyz[0:10])) + 1e-8)</code>. '
        'Recomputed from the same saved episodes; task-success counts are unchanged. '
        'Training uses soft VQ decoding; these rollout outputs use decoded action tokens. '
        'The original realized-displacement results above are preserved.</p>',1)
    (output/'round3_policy_action_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'round3_policy_action_comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Policy-action alignment</title>'
        '<style>body{font:16px system-ui;margin:24px}td,th{padding:12px;border:1px solid #ddd}table{border-collapse:collapse}td span,th span{display:block}</style>'+section)
    update_main_report()
    for key,m in models.items():print(key,m['successes'],m['average_move_cosine'],m['move_mismatches'],m['queries'],m['move_misalignment_rate'])


if __name__=='__main__':
    main()
