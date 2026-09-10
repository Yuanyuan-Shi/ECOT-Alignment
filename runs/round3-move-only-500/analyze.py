"""Live and final analysis for the isolated mismatch-only ablation."""
import html
import json
from pathlib import Path
import numpy as np
from experiments.robot.libero.move_alignment_comparison import summarize, move_reward, _read_jsonl
from experiments.robot.libero.move_alignment_sweep_comparison import validate_paired, _schedule
from experiments.robot.libero.policy_action_comparison import policy_action_cosine

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / 'runs/round3-move-only-500'
RUN = ROOT / 'runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambdamoveonly500-seed20261102'
EVAL = ROOT / 'experiments/robot/libero/results/round3-move-eval-lambdamoveonly500-90task-2trial-seed20261201'
OUT = ROOT / 'analysis_results/move_only_500'

def main():
    OUT.mkdir(exist_ok=True)
    state = json.loads((QUEUE/'status.json').read_text())
    floors = json.loads((QUEUE/'dataset_diagnostics.json').read_text())
    records = _read_jsonl(RUN/'metrics.jsonl') if (RUN/'metrics.jsonl').exists() else []
    metrics = [r for r in records if 'validation/move_loss' in r or 'final_train/move_loss' in r]
    result = {'status':state, 'dataset':floors, 'metrics':metrics}
    text = ['<h1>Mismatch-only objective: 500-step Round 3 ablation</h1>',
            '<p>Status: '+html.escape(state['status'])+'</p>',
            '<p>Objective = mean((max(0, 0.5 − cosine)/1.5)²). Performance-loss weight 0; Move-loss weight 1. '
            'Fresh untouched checkpoint; Round 3 data/split/seed, batch 32, AdamW 1e-4, warmup 20, LoRA rank 16. '
            '500 updates (13th epoch partial). Performance loss is logged but excluded from backpropagation.</p>',
            '<p>Nonmonotonic rollout mismatch across lambda does not alone establish that the penalty is broken. '
            'These finite-step runs optimize a smooth loss on fixed, teacher-forced soft-decoded actions; '
            'the report thresholds realized robot displacement on checkpoint-dependent rollout states. '
            'This ablation tests the objective without performance-loss competition, but also changes the training duration from 200 to 500 steps.</p>',
            '<p>Zero-vector reasoning always has cosine 0 and loss 1/9 with no directional gradient. '
            'Thus 0% overall mismatch is impossible on any fixed dataset containing such queries under the unchanged convention. '
            'Directional-only mismatch is reported separately. A zero loss only requires cosine ≥ 0.5, not perfect cosine 1.</p>',
            '<h2>Fixed-dataset lower bounds</h2><table><tr><th>Split</th><th>Queries</th><th>Zero-vector queries</th><th>Mismatch floor</th><th>Loss floor</th></tr>']
    for key, d in floors.items():
        text.append(f"<tr><td>{key}</td><td>{d['queries']}</td><td>{d['zero_vectors']}</td><td>{d['mismatch_floor_percent']:.3f}%</td><td>{d['loss_floor']:.6f}</td></tr>")
    text.append('</table><h2>Fixed-dataset soft-decoded alignment</h2><p>Validation is measured after each epoch; final training-set evaluation uses all 1,245 queries with dropout disabled. Directional mismatch subtracts the fixed zero-vector count.</p><table><tr><th>Split</th><th>Step</th><th>Move loss</th><th>Cosine</th><th>Overall mismatch</th><th>Directional mismatch</th><th>Performance loss (diagnostic)</th></tr>')
    for row in metrics:
        prefix = 'final_train' if 'final_train/move_loss' in row else 'validation'
        floor = floors['train' if prefix == 'final_train' else 'validation']
        mismatches = row[prefix+'/move_misalignment_count']
        directional = 100*(mismatches-floor['zero_vectors'])/(floor['queries']-floor['zero_vectors'])
        text.append(f"<tr><td>{prefix}</td><td>{row['optimizer_step']}</td><td>{row[prefix+'/move_loss']:.6f}</td><td>{row[prefix+'/move_reward']:.4f}</td><td>{row[prefix+'/move_misalignment_percentage']:.3f}%</td><td>{directional:.3f}%</td><td>{row[prefix+'/performance_loss']:.4f}</td></tr>")
    text.append('</table>')
    fixed_path = OUT/'fixed_training_comparison.json'
    if fixed_path.exists():
        fixed = json.loads(fixed_path.read_text())
        comparisons = dict(fixed['models'])
        final = next((r for r in metrics if 'final_train/move_loss' in r), None)
        if final:
            comparisons['Mismatch only, 500 steps'] = {k.removeprefix('final_train/'):v for k,v in final.items() if k.startswith('final_train/')}
        text.append('<h2>Same training-set comparison</h2><p>All checkpoints evaluated on the identical 1,245 training queries with dropout disabled, teacher forcing, and soft VQ decoding. Round 3 lambda=1 uses its final 200-step adapter.</p><table><tr><th>Checkpoint</th><th>Average cosine</th><th>Overall mismatch</th><th>Directional mismatch</th><th>Move loss</th></tr>')
        for name, m in comparisons.items():
            count = m['move_misalignment_count']
            directional = count-floors['train']['zero_vectors']
            n = floors['train']['queries']-floors['train']['zero_vectors']
            text.append(f"<tr><td>{html.escape(name)}</td><td>{m['move_reward']:.4f}</td><td>{count}/1245 ({100*count/1245:.3f}%)</td><td>{directional}/{n} ({100*directional/n:.3f}%)</td><td>{m['move_loss']:.6f}</td></tr>")
        text.append('</table>')
        result['fixed_training_comparison'] = fixed
    training_outcomes = OUT/'fixed_training_outcome_comparison.json'
    test_outcomes = OUT/'test_outcome_comparison.json'
    if training_outcomes.exists() and test_outcomes.exists():
        train_split = json.loads(training_outcomes.read_text())['models']
        test_split = json.loads(test_outcomes.read_text())
        names = ['Original MiniVLA','Round 3 lambda=1','Mismatch only']
        test_names = ['pretrained','Round 3 lambda 1, 200 steps','Mismatch only, 500 steps']
        if all(name in train_split for name in names):
            result['outcome_comparison'] = {'training':train_split,'test':test_split}
            text.append('<h2>Directional policy-command mismatch by episode outcome</h2><p>Zero-vector queries excluded. Each percentage pools queries within its outcome group. Training outcomes are from the fixed source episodes; test outcomes are from each model’s own rollouts. Training uses teacher-forced soft decoding; test uses decoded rollout commands.</p><table><tr><th>Dataset / outcome</th><th>Original MiniVLA</th><th>Round 3 lambda=1</th><th>Mismatch only</th></tr>')
            for split in ('Training','Test'):
                for outcome in ('successful','failed'):
                    text.append(f'<tr><th>{split}: {outcome} episodes</th>')
                    for name,test_name in zip(names,test_names):
                        d = train_split[name]['outcomes'][outcome] if split=='Training' else test_split[test_name][outcome]
                        text.append(f"<td>{d['percent']:.2f}% ({d['mismatches']}/{d['queries']})</td>")
                    text.append('</tr>')
            text.append('</table>')
    if (OUT/'stop_execution.html').exists():
        text.append('<h2>Stop alignment and execution diagnostics</h2><p><a href="stop_execution.html">Open stop-inclusive tables and command-to-motion analysis</a></p>')
    steps = [r for r in records if 'train/move_loss' in r]
    if steps:
        result['latest_optimizer_step'] = steps[-1]['optimizer_step']
        text.append(f"<p>Latest completed optimizer step: {steps[-1]['optimizer_step']}/500.</p>")
    if state['status'] in ('analyzing', 'complete'):
        old = json.loads((ROOT/'analysis_results/round3_move_alignment_comparison.json').read_text())
        specs = [('pretrained', Path(old['models']['pretrained']['root'])),
                 ('Round 3 lambda 1, 200 steps', Path(old['models']['10']['root'])),
                 ('Mismatch only, 500 steps', EVAL)]
        result['rollout'] = {}
        for metric_name, reward_fn in [('Realized end-effector displacement', move_reward), ('Decoded policy translation commands', policy_action_cosine)]:
            models = {name:summarize(root,name,reward_fn=reward_fn) for name,root in specs}
            validate_paired(models,2)
            expected = [(task,trial,20261201+task*2+trial) for task in range(90) for trial in range(2)]
            for _,root in specs:
                if _schedule(root) != expected: raise ValueError('Unexpected evaluation schedule')
            text.append(f'<h2>{metric_name}</h2><table><tr><th>Model</th><th>Task success</th><th>Cosine</th><th>Overall mismatch</th><th>Directional mismatch</th><th>Zero-vector queries</th></tr>')
            for name,root in specs:
                m=models[name]
                rows=_read_jsonl(root/'alignment_queries.jsonl')
                directional=[r for r in rows if any(r['claims']['motion_axes'].get(a,0) for a in ('x','y','z'))]
                dm=sum(reward_fn(r)<.5 for r in directional)
                m['directional_queries']=len(directional)
                m['directional_mismatches']=dm
                m['zero_vector_queries']=len(rows)-len(directional)
                rate=f'{100*dm/len(directional):.3f}%' if directional else 'N/A'
                text.append(f"<tr><td>{name}</td><td>{m['successes']}/180</td><td>{m['average_move_cosine']:.4f}</td><td>{100*m['move_misalignment_rate']:.3f}% ({m['move_mismatches']}/{m['queries']})</td><td>{rate}</td><td>{m['zero_vector_queries']}</td></tr>")
            result['rollout'][metric_name]=models
            text.append('</table>')
        final_train=next(r for r in metrics if 'final_train/move_loss' in r)
        val=[r for r in metrics if 'validation/move_loss' in r][-1]
        reached=final_train['final_train/move_misalignment_count']==floors['train']['zero_vectors']
        text.append('<h2>Interpretation</h2><p>'+('The final model reached zero directional mismatch on the training set.' if reached else 'The final model did not reach zero directional mismatch on the training set within 500 updates.')+
                    ' Compare validation with the final training-set measurement to assess generalization. '
                    'A gap between soft-decoded fixed-data alignment and decoded rollout commands indicates a prediction/context gap; '
                    'a further gap to realized displacement reflects execution and visited-state differences. '
                    'This single-seed ablation cannot establish monotonicity or isolate training duration from removal of performance loss. '
                    'Higher task success is not required by this objective; it also does not constrain action magnitude, rotation, or gripper correctness directly.</p>')
    else:
        text.append('<p>Rollout results and final conclusions are pending. Training is followed automatically by the same 90-task, two-trial evaluation.</p>')
    if state.get('error'): text.append('<pre>'+html.escape(state['error'])+'</pre>')
    url=RUN/'wandb_run_url.txt'
    if url.exists():text.append('<p><a href="'+html.escape(url.read_text().strip())+'">W&amp;B training run</a></p>')
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    doc='<!doctype html><meta charset="utf-8"><title>Mismatch-only 500 steps</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:20px;color:#193653}table{border-collapse:collapse}th,td{padding:10px;border:1px solid #ccc}th{background:#e8eef4}</style>'+''.join(text)
    temp=OUT/'index.tmp'
    temp.write_text(doc)
    temp.replace(OUT/'index.html')

if __name__=='__main__': main()
