import os,json,subprocess,time
from pathlib import Path
ROOT=Path('/home/exx/Projects/ECOT-Alignment');os.chdir(ROOT)
Q=ROOT/'runs/generated-training-eval'
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
jobs=[]
for model in ('pretrained','lambda1','moveonly'):
    for rank in range(2):
        log=(Q/f'{model}-{rank}.log').open('a')
        p=subprocess.Popen(['/home/exx/.conda/envs/openvla/bin/python','-u','scripts/analysis/evaluate_generated_training.py','--model',model,'--rank',str(rank)],stdout=log,stderr=subprocess.STDOUT,env=env)
        jobs.append((model,rank,p,log))
while True:
    states=[{'model':m,'rank':r,'pid':p.pid,'exit_code':p.poll()} for m,r,p,_ in jobs]
    tmp=Q/'status.tmp';tmp.write_text(json.dumps({'status':'running' if any(x['exit_code'] is None for x in states) else 'complete' if all(x['exit_code']==0 for x in states) else 'failed','jobs':states},indent=2));tmp.replace(Q/'status.json')
    if all(x['exit_code'] is not None for x in states):break
    time.sleep(10)
