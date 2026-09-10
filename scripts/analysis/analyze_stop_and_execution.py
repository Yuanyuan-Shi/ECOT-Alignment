"""Read-only stop rescoring and command-to-motion diagnostics on saved rollouts."""
import json
from pathlib import Path
import numpy as np
from experiments.robot.libero.alignment_evaluator import _split_fields
from experiments.robot.libero.policy_action_comparison import policy_action_cosine
from experiments.robot.libero.move_alignment_comparison import move_reward
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis_results/move_only_500'

def move_text(r):
    values=_split_fields(r['reasoning_raw']).get('MOVE',[])
    return values[-1].rsplit('→',1)[-1].strip().lower().rstrip('.;') if values else ''
def is_stop(r):return move_text(r)=='stop'
def cos(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
def stopped(d,rule):return bool((np.linalg.norm(d) if rule=='euclidean' else np.abs(d).max())<=.03)
def main():
    prior=json.loads((OUT/'analysis.json').read_text())['rollout']['Decoded policy translation commands']
    result={'stop_threshold_m':.03,'rules':['euclidean','per_axis'],'test':{},'gap':{},'training':{}}
    for name,old in prior.items():
        rows=[json.loads(l) for l in (Path(old['root'])/'alignment_queries.jsonl').read_text().splitlines()]
        scores=[];features=[];targets=[];holdouts=[]
        for r in rows:
            a=np.array(r['decoded_action_chunk'])[:,:3];n=r['executed_chunk_length']
            cmd=.05*np.clip(a,-1,1);tr=np.array(r['executed_ee_trajectory']);delta=tr[-1]-tr[0]
            assert len(tr)==n+1
            direction=np.array([r['claims']['motion_axes'].get(k,0) for k in ('x','y','z')])
            if np.linalg.norm(direction)>0:
                directional=True
            else:directional=False
            rawcos=policy_action_cosine(r);realcos=move_reward(r)
            assert abs(cos(direction,delta)-realcos)<1e-8
            clippedcos=cos(direction,cmd[:n].sum(0))
            s={'id':r['episode_id']+'::'+str(r['policy_query_index']),'success':bool(r['episode_success']),
               'directional':directional,'stop':is_stop(r),'move':move_text(r),'raw_mismatch':rawcos<.5,
               'prefix_mismatch':cos(direction,a[:n].sum(0))<.5,'clipped_mismatch':clippedcos<.5,
               'real_mismatch':realcos<.5,'real_norm':float(np.linalg.norm(delta)),
               'command_norm':float(np.linalg.norm(cmd[:n].sum(0))),'n':n,
               'clipped':bool((np.abs(a[:n])>1).any()),
               'angle':float(np.degrees(np.arccos(np.clip(cos(cmd[:n].sum(0),delta),-1,1)))),
               'cmd_net_m':cmd.sum(0).tolist(),'real_net_m':delta.tolist()}
            scores.append(s)
            dx=np.diff(tr,axis=0)
            # Fit one scalar command coefficient and one velocity-memory coefficient,
            # holding out entire tasks (task_id divisible by 5). No causal attribution.
            for t in range(1,n):
                for axis in range(3):
                    features.append([cmd[t,axis],dx[t-1,axis]])
                    targets.append(dx[t,axis]);holdouts.append(int(r['task_id'])%5==0)
        stats={}
        for rule in result['rules']:
            stats[rule]={}
            for outcome,success in [('successful',True),('failed',False)]:
                group=[s for s in scores if s['success']==success and (s['directional'] or s['stop'])]
                stops=[s for s in group if s['stop']]
                fail=sum(s['raw_mismatch'] if s['directional'] else not stopped(s['cmd_net_m'],rule) for s in group)
                stats[rule][outcome]={'queries':len(group),'mismatches':fail,'percent':100*fail/len(group),
                                     'stop_queries':len(stops),'stop_mismatches':sum(not stopped(s['cmd_net_m'],rule) for s in stops)}
        from collections import Counter
        stats['other_zero_move_claims']=dict(Counter(s['move'] for s in scores if not s['directional'] and not s['stop']))
        result['test'][name]=stats
        d=[s for s in scores if s['directional']]
        lost=[s for s in d if not s['clipped_mismatch'] and s['real_mismatch']]
        gained=[s for s in d if s['clipped_mismatch'] and not s['real_mismatch']]
        X=np.array(features);y=np.array(targets);test=np.array(holdouts)
        ab=np.linalg.lstsq(X[~test],y[~test],rcond=None)[0]
        alpha=np.linalg.lstsq(X[~test,:1],y[~test],rcond=None)[0]
        rmse1=float(np.sqrt(np.mean((X[test,:1]@alpha-y[test])**2)))
        rmse2=float(np.sqrt(np.mean((X[test]@ab-y[test])**2)))
        gap={'queries':len(d),'raw_mismatches':sum(s['raw_mismatch'] for s in d),
             'prefix_mismatches':sum(s['prefix_mismatch'] for s in d),'clipped_prefix_mismatches':sum(s['clipped_mismatch'] for s in d),
             'real_mismatches':sum(s['real_mismatch'] for s in d),'short_chunks':sum(s['n']<10 for s in d),
             'clipped_chunks':sum(s['clipped'] for s in d),'command_aligned_real_misaligned':len(lost),'command_misaligned_real_aligned':len(gained),
             'lost_real_motion_under_3cm':sum(s['real_norm']<=.03 for s in lost),
             'lost_real_motion_under_3mm':sum(s['real_norm']<=.003 for s in lost),
             'lost_under_3mm_command_over_3cm':sum(s['real_norm']<=.003 and s['command_norm']>.03 for s in lost),
             'median_command_real_angle_deg':float(np.median([s['angle'] for s in d])),
             'lag_fit':{'command_coefficient':float(ab[0]),'previous_motion_coefficient':float(ab[1]),
                        'no_memory_rmse_m':rmse1,'with_memory_rmse_m':rmse2,'heldout_rmse_reduction_percent':100*(1-rmse2/rmse1)}}
        result['gap'][name]=gap
        (OUT/('execution_scores_'+name.replace(' ','_').replace('=','')+'.json')).write_text(json.dumps(scores,indent=2)+'\n')
    audit=[json.loads(l) for l in (ROOT/'experiments/robot/libero/results/second-round-pretrained-libero90-seed20261101/alignment_queries.jsonl').read_text().splitlines()]
    by_id={f"task{r['task_id']}-query{r['policy_query_index']}":r for r in audit}
    for name in ['Original MiniVLA','Round 3 lambda=1','Mismatch only']:
        path=OUT/('training_stop_scores_'+name.lower().replace(' ','_').replace('=','')+'.json')
        if not path.exists():continue
        rows=json.loads(path.read_text());stats={}
        for rule in result['rules']:
            stats[rule]={}
            for label,success in [('successful',True),('failed',False)]:
                group=[s for s in rows if s['episode_success']==success and (s['directional'] or is_stop(by_id[s['id']]))]
                stops=[s for s in group if not s['directional']]
                fail=sum(s['mismatch'] if s['directional'] else not stopped(s['commanded_net_m'],rule) for s in group)
                stats[rule][label]={'queries':len(group),'mismatches':fail,'percent':100*fail/len(group),
                                  'stop_queries':len(stops),'stop_mismatches':sum(not stopped(s['commanded_net_m'],rule) for s in stops)}
        result['training'][name]=stats
    (OUT/'stop_and_execution_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
