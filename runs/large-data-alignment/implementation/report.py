"""Live report in a separate tab of the existing results HTML."""
import json,html
from pathlib import Path
from data import ROOT,Q,write
OUT=ROOT/'analysis_results/large_data_alignment'
NAMES={'pretrained':'Original MiniVLA','lambda0':'λ=0','lambda1':'λ=1','moveonly':'Mismatch only'}
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default

def main():
    OUT.mkdir(exist_ok=True);state=read(Q/'status.json',{'status':'preparing'});plan=read(Q/'plan.json',{});split=read(Q/'split.json',{})
    models={};parts=['<h1>Large-data reasoning–command alignment</h1>',f'<p>Status: <b>{html.escape(state.get("status","preparing"))}</b> · Stage: {html.escape(state.get("stage","preparation"))} · Updated: {state.get("updated_at", "pending")}</p>',
    '<p>900 episodes: 90 tasks × 10 distinct initial configurations; 720 training / 180 validation episodes. Separate test configurations: 30 matched tasks × 3 trials per model.</p>',
    '<p>Each model starts from original MiniVLA. Batch 32 (one batch per optimizer step), 2,000 steps, LR 5e-5, warmup 50; no early stopping. Checkpoints: 250, 500, 1,000, 1,500, 2,000. Validation every 250 steps. Final tests use step 2,000.</p>',
    '<p>Command displacement: sum of 10 clipped controller translations × 0.05 m. Direction aligned at cosine ≥0.5; explicit stop aligned at net magnitude ≤0.03 m. Alignment loss is one pooled eligible-query mean. Directional loss: [max(0,0.5−cosine)/1.5]². Stop loss: (net magnitude/0.03 m)², with no hinge; nonzero net movement is penalized even below the unchanged 3 cm evaluation threshold. Other zero-vector claims are excluded. Performance targets are MiniVLA-generated reasoning and actions, including failures.</p>']
    completed=state.get('collection_completed_chunks',[]);observed=len(completed)*100
    for chunk in set(range(9))-set(completed):
        ids=set();root=ROOT/f'experiments/robot/libero/results/large-data-collection-{chunk:02d}'
        for f in root.glob('rank-*/alignment_queries.jsonl'):
            for line in f.read_text().splitlines():
                try:ids.add(json.loads(line)['episode_id'])
                except json.JSONDecodeError:pass
        observed+=len(ids)
    parts.append(f'<p>Collection: {observed}/900 completed episodes ({len(completed)} fully verified batches). Training queries: {split.get("training_queries","pending")}; validation queries: {split.get("validation_queries","pending")}.</p>')
    ds=read(Q/'dataset_statistics.json')
    if ds:
        parts.append('<h2>Collected dataset: commanded Move mismatch</h2><p>Cases use eligible-query denominators. Coverage is eligible/all queries. Differences are failed minus successful; 95% intervals resample episodes within each outcome group (2,000 bootstrap samples).</p><table><tr><th>Category</th><th>Overall coverage / cases</th><th>Successful coverage / cases</th><th>Failed coverage / cases</th><th>Difference</th><th>95% CI</th></tr>')
        for label,d in ds.items():
            parts.append('<tr><th>'+label+'</th>')
            for g in ['overall','successful','failed']:
                a=d[g];rate=f'{a["rate"]*100:.2f}%' if a['rate'] is not None else 'N/A';parts.append(f'<td>{a["queries"]:,}/{a["total_queries"]:,} covered<br>{rate} ({a["mismatches"]:,}/{a["queries"]:,})</td>')
            diff=f'{d["difference"]*100:+.2f} pp' if d['difference'] is not None else 'N/A';ci=d['bootstrap_95_ci'];parts.append('<td>'+diff+'</td><td>'+(f'{ci[0]*100:+.2f} to {ci[1]*100:+.2f} pp' if ci else 'N/A')+'</td></tr>')
        parts.append('</table>')
    parts.append('<h2>Model comparison</h2><p>Training/validation: fixed source reasoning and teacher-forced soft-decoded commands. Test: generated reasoning and decoded rollout commands. Outcome breakdown uses source outcomes for training/validation and each model’s own test outcomes.</p>')
    for key in NAMES:
        d={};rd=ROOT/f'runs/ecot-move-alignment/large-data-{key}-b32-s2000'
        if key=='pretrained':
            rd=ROOT/'runs/ecot-move-alignment/large-data-lambda0-b32-s2000'
            for splitname in ['training','validation']:
                x=read(rd/f'initial-{splitname}-summary.json')
                if x:d[splitname]=x
        else:
            x=read(rd/'result.json')
            if x:d.update(training=x['training'],validation=x['validation'])
        test=read(Q/f'test-{key}.json')
        if test:d['test']=test
        models[key]=d
    parts.append('<table><tr><th>Policy-command mismatch</th>'+''.join('<th>'+n+'</th>' for n in NAMES.values())+'</tr>')
    for sp in ['training','validation','test']:
        for g in ['overall','successful','failed']:
            parts.append('<tr><th>'+sp.title()+' — '+g+'</th>')
            for d in models.values():
                x=d.get(sp,{}).get(g);value=f'{100*x["rate"]:.2f}% ({x["mismatches"]:,}/{x["queries"]:,})' if x and x['rate'] is not None else 'Pending' if not x else 'N/A';parts.append('<td>'+value+'</td>')
            parts.append('</tr>')
    training_ids=set(split.get('training_episodes',[]))
    training_episodes=[e for e in read(Q/'episodes.json',[]) if e['episode_id'] in training_ids]
    train_successes=sum(bool(e['episode_success']) for e in training_episodes)
    train_value=f'{100*train_successes/len(training_episodes):.1f}% ({train_successes}/{len(training_episodes)})' if training_episodes else 'Pending'
    parts.append('<tr><th>Train task success</th><td>'+train_value+'</td>'+''.join('<td>N/A — fixed dataset</td>' for _ in range(3))+'</tr>')
    parts.append('<tr><th>Test task success</th>')
    for d in models.values():
        t=d.get('test');parts.append('<td>'+(f'{100*t["successes"]/t["episodes"]:.1f}% ({t["successes"]}/{t["episodes"]})' if t else 'Pending')+'</td>')
    parts.append('</tr></table><p>Train task success is the original MiniVLA collection success on the 720 training episodes. Fine-tuned models are scored on these fixed queries; their task success requires fresh rollouts and is reported on the test set.</p><h2>Training progress and artifacts</h2><ul>')
    for key in ['lambda0','lambda1','moveonly']:
        rd=ROOT/f'runs/ecot-move-alignment/large-data-{key}-b32-s2000';p=read(rd/'progress.json',{});w=read(rd/'wandb.json',{});url='../../'+str(rd.relative_to(ROOT))
        parts.append(f'<li>{NAMES[key]}: step {p.get("optimizer_step",0)}/2,000 · <a href="{url}/">Logs and checkpoints</a>')
        if w:parts.append(' · W&B run '+html.escape(w['id'])+' (stored locally; sync status below)')
        u=rd/'wandb_run_url.txt'
        if u.exists():parts.append(' · <a href="'+html.escape(u.read_text().strip())+'">W&B</a>')
        parts.append('</li>')
    parts.append('</ul><p>W&B sync: '+html.escape(str(state.get('wandb_sync','pending')))+'. Offline logs remain durable during network interruptions.</p>')
    if state.get('error'):parts.append('<pre>'+html.escape(state['error'])+'</pre>')
    write(OUT/'results.json',dict(state=state,plan=plan,models=models,dataset=ds))
    text='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Large-data alignment</title><style>body{font:16px system-ui;color:#193653;margin:28px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccd5df}th{background:#eaf0f7}p{max-width:1400px}pre{white-space:pre-wrap}</style>'+''.join(parts)
    tmp=OUT/'index.tmp';tmp.write_text(text);tmp.replace(OUT/'index.html')
if __name__=='__main__':main()
