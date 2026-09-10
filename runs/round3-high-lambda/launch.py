"""Register both Round 4 runs, serialize GPU use, then evaluate and update the table."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/home/exx/Projects/ECOT-Alignment')
os.chdir(ROOT)
PYTHON = '/home/exx/.conda/envs/openvla/bin/python'
STATE = ROOT / 'runs/round3-high-lambda/status.json'
if STATE.exists():
    raise FileExistsError('Round 4 already registered; inspect status before recovery')
state = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'pid': os.getpid(), 'runs': {}}
processes = {}
for name, weight in [('20', 2.0), ('50', 5.0)]:
    run_dir = f'runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambda{name}-seed20261102'
    evaluation_root = f'experiments/robot/libero/results/round3-move-eval-lambda{name}-90task-2trial-seed20261201'
    gate = f'runs/round3-high-lambda/start-{name}'
    if any((ROOT / p).exists() for p in (run_dir,evaluation_root,gate)):
        raise FileExistsError(f'Refusing to overwrite {name}')
    state['runs'][name] = {'lambda': weight, 'display_name': f'Round 3 lambda={weight}',
        'status': 'waiting for GPU slot', 'name': name, 'run_dir': run_dir, 'start_gate': gate,
        'checkpoint': run_dir+'/checkpoints/step-000200-move-lora.pt', 'evaluation_root': evaluation_root}


def save():
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2)+'\n')
    tmp.replace(STATE)


def report():
    save()
    subprocess.run([PYTHON,'-m','experiments.robot.libero.round3_move_comparison'],check=True)


def refresh():
    changed = False
    for key,run in state['runs'].items():
        url = ROOT / run['run_dir'] / 'wandb_run_url.txt'
        if url.exists() and not run.get('wandb_url'):
            run['wandb_url'] = url.read_text().strip()
            changed = True
    if changed:
        report()


report()
for key,run in state['runs'].items():
    with open(f"runs/round3-high-lambda/train-{run['name']}.log",'x') as log:
        p = subprocess.Popen(['bash','runs/round3-high-lambda/train.sh',run['name'],'--start_gate',run['start_gate']],
                             stdout=log,stderr=subprocess.STDOUT)
    processes[key] = p
    run['process_pid'] = p.pid
save()
# Previous Round 3 controller alone owns the GPU until its final report finishes.
while True:
    previous = json.loads((ROOT/'runs/round4-lambda1/status.json').read_text())
    if previous.get('finished_at'):
        break
    refresh()
    time.sleep(10)

for key,run in state['runs'].items():
    p = processes[key]
    try:
        if p.poll() is not None:
            raise RuntimeError(f"Trainer exited before GPU release: {p.returncode}")
        run['status'] = 'training'
        report()
        (ROOT / run['start_gate']).touch(exist_ok=False)
        while p.poll() is None:
            refresh()
            time.sleep(10)
        if p.returncode:
            raise RuntimeError(f'Trainer exited {p.returncode}')
        result=json.loads((ROOT/run['run_dir']/'result.json').read_text())
        assert result['optimizer_steps']==200
        assert Path(result['checkpoint']).is_file()
        run['wandb_url']=result['wandb_url']
        run['status']='trained; queued for evaluation'
    except Exception:
        run['status']='training failed'
        run['error']=traceback.format_exc()
    run.pop('process_pid',None)
    report()

for key,run in state['runs'].items():
    if run['status']!='trained; queued for evaluation':
        continue
    try:
        run['status']='evaluating'
        report()
        with open(f"runs/round3-high-lambda/eval-{run['name']}.log",'x') as log:
            p=subprocess.Popen(['bash','runs/round3-high-lambda/evaluate.sh',run['name']],stdout=log,stderr=subprocess.STDOUT)
        run['process_pid']=p.pid
        save()
        if p.wait():
            raise RuntimeError(f'Evaluation exited {p.returncode}')
        run['status']='complete'
        report()  # Exact 180-episode paired schedule is checked by the report builder.
    except Exception:
        run['status']='evaluation failed'
        run['error']=traceback.format_exc()
    run.pop('process_pid',None)
    report()
state['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
report()
