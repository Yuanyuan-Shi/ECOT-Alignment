"""Restartable collection -> dataset -> three trainings -> paired final evaluation."""
import os,json,time,subprocess,traceback,fcntl,shutil
from pathlib import Path
from data import ROOT,Q,write,score,statistics
os.chdir(ROOT);PY='/home/exx/.conda/envs/openvla/bin/python'
plan=json.loads((Q/'plan.json').read_text())
lock=(Q/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=json.loads((Q/'status.json').read_text()) if (Q/'status.json').exists() else {}
adopt_pid=state.get('active_process_pid') if state.get('status')=='running' else None
state.update(status='running',pid=os.getpid());state.pop('error',None);state.pop('process_state',None);state.pop('paused_process_ids',None)
def report():
    state['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write(Q/'status.json',state)
    subprocess.run([PY,str(Q/'report.py')],check=True)
def execute(cmd,name):
    write(Q/(name+'-command.json'),cmd)
    with (Q/(name+'.log')).open('a') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);state['active_process_pid']=p.pid;report()
        while p.poll() is None:
            try:p.wait(timeout=45)
            except subprocess.TimeoutExpired:report()
    state.pop('active_process_pid',None)
    if p.returncode:raise RuntimeError(f'{name} exited {p.returncode}; see {name}.log')
def eval_command(root,tasks,trials,offset,seed,cp,frames):
    return [PY,'-u','experiments/robot/libero/run_libero_eval.py','--model_family','prismatic','--pretrained_checkpoint',cp,'--task_suite_name','libero_90','--task_ids',','.join(map(str,tasks)),'--episodes_per_task',str(trials),'--initial_state_offset',str(offset),'--seed',str(seed),'--center_crop','False','--use_wrist_image','False','--use_cot','True','--num_open_loop_steps','10','--n_procs_per_gpu','6','--enable_alignment_evaluator','True','--alignment_cosine_similarity','True','--alignment_save_frames',str(frames),'--alignment_output_dir',str(root),'--output_dir',root.name,'--use_wandb','False','--run_id_note',root.name]
def verify(root,tasks,trials,offset,seed,frames=False):
    f=root/'episode_summary.json'
    if not f.exists():return False
    es=json.loads(f.read_text());expected={(t,seed+t*trials+j,offset+j) for t in tasks for j in range(trials)}
    actual={(e['task_id'],e['seed'],e['initial_state_index']) for e in es}
    if len(es)!=len(expected) or actual!=expected:raise ValueError('Episode schedule mismatch: '+str(root))
    seen=set();ids={e['episode_id'] for e in es}
    for line in (root/'alignment_queries.jsonl').read_text().splitlines():
        r=json.loads(line);key=(r['episode_id'],r['policy_query_index']);assert key not in seen and key[0] in ids;seen.add(key)
        if frames:assert (root/r['frames']['before']).is_file()
    assert {x[0] for x in seen}==ids
    return True
def run_eval(root,tasks,trials,offset,seed,cp,frames,name):
    if verify(root,tasks,trials,offset,seed,frames):return
    if root.exists():root.rename(root.with_name(root.name+'-interrupted-'+str(time.time_ns())))
    execute(eval_command(root,tasks,trials,offset,seed,cp,frames),name)
    assert verify(root,tasks,trials,offset,seed,frames)
def sync(key):
    rd=ROOT/f'runs/ecot-move-alignment/large-data-{key}-b32-s2000'
    # Sync all segments if a training process was resumed.
    paths=[path for path in sorted((rd/'wandb').glob('offline-run-*')) if (path/f'run-{json.loads((rd/"wandb.json").read_text())["id"]}.wandb').exists()]
    ok=True
    for path in paths:
        try:
            with (Q/'wandb-sync.log').open('a') as f:
                p=subprocess.run(['/home/exx/.conda/envs/openvla/bin/wandb','sync','--project',plan['wandb_project'],str(path)],stdout=f,stderr=subprocess.STDOUT,timeout=120)
            ok &= p.returncode==0
        except subprocess.TimeoutExpired:ok=False
    state.setdefault('wandb_sync',{})[key]='synced' if ok else 'pending retry; local logs saved'
    if ok and paths:
        # Read the actual entity from the sync output rather than guessing it.
        import re
        text=(Q/'wandb-sync.log').read_text();w=json.loads((rd/'wandb.json').read_text())
        urls=re.findall(r'https://wandb.ai/[^\s]+/runs/'+re.escape(w['id']),text)
        if urls:(rd/'wandb_run_url.txt').write_text(urls[-1]+'\n')
    report()
try:
    if adopt_pid:
        import psutil
        state['stage']='waiting for existing collection/evaluation worker';report()
        while psutil.pid_exists(adopt_pid) and psutil.Process(adopt_pid).status()!=psutil.STATUS_ZOMBIE:
            time.sleep(30);report()
        state.pop('active_process_pid',None)
    state['stage']='batch32 preflight';report()
    if not (Q/'batch32_probe.json').exists():execute([PY,'-u','vla-scripts/finetune_large_alignment.py','--probe'],'batch32-probe')
    state['collection_completed_chunks']=[]
    for chunk in range(9):
        state['stage']=f'collection batch {chunk+1}/9';report()
        root=ROOT/f'experiments/robot/libero/results/large-data-collection-{chunk:02d}'
        run_eval(root,list(range(chunk*10,chunk*10+10)),10,plan['collection_initial_state_offset'],plan['collection_seed'],plan['base_checkpoint'],True,f'collection-{chunk:02d}')
        state['collection_completed_chunks'].append(chunk);report()
    state['stage']='dataset assembly and bootstrap statistics';report()
    if not (Q/'dataset_complete.json').exists():execute([PY,str(Q/'data.py')],'assemble')
    for key in ['lambda0','lambda1','moveonly']:
        state['stage']='training '+key;report();execute([PY,'-u','vla-scripts/finetune_large_alignment.py','--mode',key],'train-'+key);sync(key)
    for key in ['pretrained','lambda0','lambda1','moveonly']:
        state['stage']='test evaluation '+key;report()
        cp=plan['base_checkpoint'] if key=='pretrained' else json.loads((ROOT/f'runs/ecot-move-alignment/large-data-{key}-b32-s2000/result.json').read_text())['checkpoint']
        root=ROOT/f'experiments/robot/libero/results/large-data-test-{key}'
        run_eval(root,plan['test_task_ids'],3,plan['test_initial_state_offset'],plan['test_seed'],cp,False,'test-'+key)
        es=json.loads((root/'episode_summary.json').read_text());outcomes={e['episode_id']:e['episode_success'] for e in es};rows=[json.loads(l) for l in (root/'alignment_queries.jsonl').read_text().splitlines()]
        for r in rows:r['episode_success']=outcomes[r['episode_id']]
        stats=statistics(score(rows));result=stats['Overall mismatch'];result.update(episodes=90,successes=sum(bool(v) for v in outcomes.values()),categories=stats);write(Q/f'test-{key}.json',result);report()
    for key in ['lambda0','lambda1','moveonly']:
        if state.get('wandb_sync',{}).get(key)!='synced':sync(key)
    execute([PY,str(Q/'log_evaluation.py')],'wandb-evaluation')
    state.update(status='complete',stage='complete',finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'));report()
except Exception:
    state.update(status='failed',error=traceback.format_exc());report();raise
