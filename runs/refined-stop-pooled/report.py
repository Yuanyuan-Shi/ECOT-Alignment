"""Live table for the refined-loss experiment; no incomplete rollout summaries."""
import json,html
from pathlib import Path
import numpy as np
import torch
from experiments.robot.libero.refined_move_alignment import refined_objective
from experiments.robot.libero.alignment_evaluator import parse_reasoning,_split_fields
ROOT=Path(__file__).resolve().parents[2];Q=ROOT/'runs/refined-stop-pooled';OUT=ROOT/'analysis_results/refined_stop_pooled'
NAMES={'pretrained':'Original MiniVLA','lambda1':'Refined λ=1','moveonly':'Refined mismatch only'}

def rollout(root,plan):
    episodes=json.loads((root/'episode_summary.json').read_text())
    expected={(t,plan['evaluation_base_seed']+3*t+j) for t in plan['evaluation_task_ids'] for j in range(3)}
    actual=[(e['task_id'],e['seed']) for e in episodes]
    if len(actual)!=90 or len(set(actual))!=90 or set(actual)!=expected:raise ValueError('Evaluation does not match paired 30-task/3-trial schedule')
    outcomes={e['episode_id']:bool(e['episode_success']) for e in episodes};rows=[];seen=set()
    for line in (root/'alignment_queries.jsonl').read_text().splitlines():
        r=json.loads(line);rid=(r['episode_id'],r['policy_query_index'])
        if rid in seen:raise ValueError('Duplicate query')
        seen.add(rid)
        claims=parse_reasoning(r['reasoning_raw']);v=np.array([claims.motion_axes.get(a,0) for a in ('x','y','z')],float)
        fields=_split_fields(r['reasoning_raw']).get('MOVE',[])
        stop=bool(fields) and fields[-1].rsplit('→',1)[-1].strip().lower().rstrip('.;')=='stop'
        directional=bool(np.linalg.norm(v)>0);stop=stop and not directional
        a=np.array(r['decoded_action_chunk'],float);assert a.shape==(10,7) and np.isfinite(a).all()
        _,details=refined_objective(torch.tensor(a)[None],torch.tensor(v)[None],torch.tensor([stop]))
        rows.append({'episode_success':outcomes[r['episode_id']],'eligible':directional or stop,'stop':stop,'directional':directional,
                     'mismatch':bool(details['mismatch'][0])})
    result={}
    for name,success in [('overall',None),('successful',True),('failed',False)]:
        allrows=[r for r in rows if success is None or r['episode_success']==success];g=[r for r in allrows if r['eligible']]
        n=len(g);k=sum(r['mismatch'] for r in g)
        result[name]={'queries':n,'mismatches':k,'rate':k/n if n else None,'excluded':len(allrows)-n,
                      'stop_queries':sum(r['stop'] for r in g),'stop_mismatches':sum(r['stop'] and r['mismatch'] for r in g)}
    result['successes']=sum(outcomes.values());result['episodes']=90;return result

def main():
    OUT.mkdir(exist_ok=True)
    plan=json.loads((Q/'plan.json').read_text());state=json.loads((Q/'status.json').read_text());models={}
    base=ROOT/'runs/ecot-move-alignment/refined-stop-pooled-lr5e5-lambda1-seed20261102/pretrained-training-summary.json'
    for key in NAMES:
        d={};run=state['runs'][key]
        if key=='pretrained' and base.exists():d['training']=json.loads(base.read_text())
        if key!='pretrained':
            result=ROOT/run['run_dir']/'result.json'
            if result.exists():
                saved=json.loads(result.read_text());d.update(training=saved['training'],validation=saved['validation'],selected_step=saved['selected_step'],optimizer_steps=saved['optimizer_steps'])
        if run.get('evaluation_status')=='complete':d['test']=rollout(ROOT/run['evaluation_root'],plan)
        models[key]=d
    result={'plan':plan,'state':state,'models':models}
    text=['<h1>Refined directional/stop loss: lower LR and validation early stopping</h1>',
          '<p>Fresh original MiniVLA initialization for both tuned models. λ=1 retains supervised loss; mismatch-only excludes it. Alignment loss is one pooled mean over all eligible directional and literal-stop queries. Direction: squared hinge at cosine .5; stop: squared excess net-command magnitude above .03 m. Both branches use the same 10-step command sum after clipping controller inputs and scaling by .05 m per input unit. The 3 cm stop tolerance is a chosen evaluation criterion, not a verified equivalence to the original labeling rule. Other zero-vector claims are excluded from alignment.</p>',
          '<p>Training budget 500 optimizer steps, LR 5e-5, batch32, warmup20, LoRA rank16/dropout.05. Validation every25 steps; four non-improving checks stop training. Best validation overall mismatch checkpoint is restored, including the untrained checkpoint if none improves. Validation uses separate source episodes, never rollout test scores.</p>',
          f"<p>Training {plan['train_diagnostics']['queries']} queries / {len(plan['train_episodes'])} episodes; validation {plan['validation_diagnostics']['queries']} queries / {len(plan['validation_episodes'])} episodes. Training metrics use fixed saved reasoning and soft-decoded actions with dropout disabled. Test metrics use generated reasoning and discrete policy commands. Test is 30 sampled tasks × 3 trials = 90 episodes per model, matched seeds and initial-state indices. Training outcome labels refer to fixed source episodes.</p>",
          '<table><tr><th>Policy-command mismatch</th>'+''.join('<th>'+n+'</th>' for n in NAMES.values())+'</tr>']
    for split in ['training','test']:
        for group in ['overall','successful','failed']:
            text.append('<tr><th>'+split.title()+' — '+group+'</th>')
            for key,d in models.items():
                x=d.get(split,{}).get(group)
                value=(f"{100*x['rate']:.2f}% ({x['mismatches']}/{x['queries']:,})" if x['rate'] is not None else 'N/A (no eligible queries)') if x else 'Pending'
                text.append('<td>'+value+'</td>')
            text.append('</tr>')
    text.append('<tr><th>Test task success</th>')
    for d in models.values():
        x=d.get('test');text.append('<td>'+(f"{100*x['successes']/90:.1f}% ({x['successes']}/90)" if x else 'Pending')+'</td>')
    text.append('</tr></table><h2>Progress</h2><ul>')
    for key,run in state['runs'].items():
        label=run.get('training_status','baseline')+'; evaluation: '+run.get('evaluation_status','pending')
        text.append('<li>'+NAMES[key]+': '+html.escape(label))
        if key!='pretrained':
            rd=ROOT/run['run_dir'];selection=rd/'selection.json'
            if selection.exists():
                s=json.loads(selection.read_text());text.append(f"; checked step {s['optimizer_step']}, best step {s['best_step']}, validation mismatch {100*s['best_validation_rate']:.2f}%")
            url=rd/'wandb_run_url.txt'
            if url.exists():text.append(' — <a href="'+html.escape(url.read_text().strip())+'">W&amp;B</a>')
        text.append('</li>')
    text.append('</ul><p>Sampled tasks: '+', '.join(map(str,plan['evaluation_task_ids']))+'</p>')
    if state.get('error'):text.append('<pre>'+html.escape(state['error'])+'</pre>')
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    p=OUT/'index.html';tmp=p.with_suffix('.tmp');tmp.write_text('<!doctype html><meta charset="utf-8"><title>Refined alignment experiments</title><style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:20px;color:#193653}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}th{background:#e8eef4}</style>'+''.join(text));tmp.replace(p)
if __name__=='__main__':main()
