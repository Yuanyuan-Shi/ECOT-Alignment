"""Detached train/evaluate/analyze pipeline; no automatic overwrite or retry."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path('/home/exx/Projects/ECOT-Alignment')
os.chdir(ROOT)
QUEUE=ROOT/'runs/round3-move-only-500'
RUN=ROOT/'runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambdamoveonly500-seed20261102'
EVAL=ROOT/'experiments/robot/libero/results/round3-move-eval-lambdamoveonly500-90task-2trial-seed20261201'
PYTHON='/home/exx/.conda/envs/openvla/bin/python'
if RUN.exists() or EVAL.exists():raise FileExistsError('Refusing to overwrite experiment outputs')
state={'status':'queued','pid':os.getpid(),'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'run_dir':str(RUN),'evaluation_root':str(EVAL),'optimizer_steps':500}

def save():
    state['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    tmp=QUEUE/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(QUEUE/'status.json')

def report():
    save()
    subprocess.run([PYTHON,str(QUEUE/'analyze.py')],check=True)

def execute(script,log):
    with (QUEUE/log).open('x') as output:
        process=subprocess.Popen(['bash',str(QUEUE/script)],stdout=output,stderr=subprocess.STDOUT)
        state['process_pid']=process.pid
        save()
        while True:
            try:
                code=process.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                try:report()
                except Exception:traceback.print_exc()
        state.pop('process_pid',None)
        if code:raise RuntimeError(f'{script} exited {code}; inspect {log}')

try:
    state['status']='training';report()
    execute('train.sh','train.log')
    result=json.loads((RUN/'result.json').read_text())
    assert result['optimizer_steps']==500
    assert result['performance_loss_weight']==0
    assert result['move_loss_weight']==1
    state['wandb_url']=result['wandb_url']
    state['status']='evaluating';report()
    execute('evaluate.sh','eval.log')
    state['status']='analyzing';report()
    state['status']='complete';state['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');report()
except Exception:
    state['failed_phase']=state['status'];state['status']='failed';state['error']=traceback.format_exc()
    save()
    try:report()
    except Exception:traceback.print_exc()
    raise
