"""Validated collection assembly and command-only dataset statistics."""
import json,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from experiments.robot.libero.alignment_evaluator import parse_reasoning
from experiments.robot.libero.refined_move_alignment import literal_stop,refined_objective
ROOT=Path(__file__).resolve().parents[2];Q=ROOT/'runs/large-data-alignment'
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def score(rows):
    scored=[]
    for r in rows:
        claims=parse_reasoning(r['reasoning_raw']);v=[claims.motion_axes.get(a,0) for a in ('x','y','z')]
        a=np.asarray(r['decoded_action_chunk'],dtype=float)
        if a.shape!=(10,7) or not np.isfinite(a).all():raise ValueError('Invalid decoded chunk')
        _,d=refined_objective(torch.tensor(a)[None],torch.tensor([v],dtype=torch.float64),torch.tensor([literal_stop(r['reasoning_raw'])]))
        x={k:t[0].tolist() for k,t in d.items()};x.update(episode_id=r['episode_id'],episode_success=bool(r['episode_success']),record_id=r['episode_id']+'::'+str(r['policy_query_index']))
        scored.append(x)
    return scored
CATS={'Overall mismatch':('eligible','mismatch'),'Directional mismatch':('directional','mismatch'),'Partial directional mismatch':('directional','partial'),'Nonpositive directional mismatch':('directional','nonpositive'),'Stop mismatch':('stop','mismatch')}
def statistics(rows,bootstrap=True):
    result={}
    for r in rows:
        r['partial']=r['directional'] and 0<r['cosines']<.5;r['nonpositive']=r['directional'] and r['cosines']<=0
    for label,(coverage,case) in CATS.items():
        out={};by_outcome={}
        for name,outcome in [('overall',None),('successful',True),('failed',False)]:
            g=[r for r in rows if outcome is None or r['episode_success']==outcome];eligible=[r for r in g if r[coverage]]
            k=sum(bool(r[case]) for r in eligible);n=len(eligible)
            out[name]=dict(total_queries=len(g),episodes=len({r['episode_id'] for r in g}),queries=n,mismatches=k,rate=k/n if n else None,coverage=n/len(g) if g else None)
            if outcome is not None:
                eps={r['episode_id']:[0,0] for r in g}
                for r in eligible:eps[r['episode_id']][0]+=bool(r[case]);eps[r['episode_id']][1]+=1
                by_outcome[outcome]=np.array(list(eps.values()),dtype=float).reshape(-1,2)
        out['difference']=out['failed']['rate']-out['successful']['rate'] if all(out[k]['rate'] is not None for k in ['failed','successful']) else None
        diffs=[];rng=np.random.default_rng(20260910)
        if bootstrap and all(len(v) for v in by_outcome.values()):
            for _ in range(2000):
                vals=[]
                for o in [True,False]:
                    a=by_outcome[o];s=a[rng.integers(len(a),size=len(a))].sum(0);vals.append(s[0]/s[1] if s[1] else np.nan)
                if np.isfinite(vals).all():diffs.append(vals[1]-vals[0])
        out['bootstrap_95_ci']=np.quantile(diffs,[.025,.975]).tolist() if diffs else None;result[label]=out
    return result

def assemble():
    plan=json.loads((Q/'plan.json').read_text());records=[];episodes=[];seen=set()
    for chunk in range(9):
        root=ROOT/f'experiments/robot/libero/results/large-data-collection-{chunk:02d}'
        es=json.loads((root/'episode_summary.json').read_text());expected={(t,plan['collection_seed']+t*10+j) for t in range(chunk*10,chunk*10+10) for j in range(10)}
        assert len(es)==100 and {(e['task_id'],e['seed']) for e in es}==expected
        outcomes={e['episode_id']:bool(e['episode_success']) for e in es};episodes.extend(es)
        for line in (root/'alignment_queries.jsonl').read_text().splitlines():
            r=json.loads(line);key=(r['episode_id'],r['policy_query_index']);assert key not in seen;seen.add(key)
            r['episode_success']=outcomes[r['episode_id']];r['before_frame']=str(root/r['frames']['before'])
            with Image.open(r['before_frame']) as im:im.verify()
            assert len(r['action_tokens'])==7
            r['record_id']=r['episode_id']+'::'+str(r['policy_query_index']);r['reasoning_move_vector']=[parse_reasoning(r['reasoning_raw']).motion_axes.get(a,0) for a in ('x','y','z')]
            records.append(r)
    assert len(episodes)==900 and len({e['episode_id'] for e in episodes})==900
    trainids=set();valids=set()
    for task in range(90):
        es=sorted([e['episode_id'] for e in episodes if e['task_id']==task]);assert len(es)==10
        random.Random(plan['split_seed']+task).shuffle(es);trainids.update(es[:8]);valids.update(es[8:])
    assert not trainids&valids
    scored=score(records)
    for r in records:r['split']='training' if r['episode_id'] in trainids else 'validation'
    (Q/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    write(Q/'split.json',dict(training_episodes=sorted(trainids),validation_episodes=sorted(valids),training_queries=sum(r['split']=='training' for r in records),validation_queries=sum(r['split']=='validation' for r in records)))
    write(Q/'dataset_statistics.json',statistics(scored));write(Q/'episodes.json',episodes)
    write(Q/'dataset_complete.json',dict(episodes=900,queries=len(records),training_episodes=720,validation_episodes=180))
if __name__=='__main__':assemble()
