"""Durable sequential train/evaluate queue for the single-GPU Round 3 sweep."""
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path('/home/exx/Projects/ECOT-Alignment')
os.chdir(ROOT)
STATE = ROOT / 'runs/round3/status.json'
PYTHON = '/home/exx/.conda/envs/openvla/bin/python'
state = json.loads(STATE.read_text())
if not all(run['status'] == 'complete' for run in state['runs'].values()):
    raise RuntimeError('Previous queue has unfinished work; refusing concurrent launch')
suffix = '10'
if suffix in state['runs']:
    raise FileExistsError('Lambda 1.0 already registered')
run_dir = f'runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambda{suffix}-seed20261102'
evaluation_root = f'experiments/robot/libero/results/round3-move-eval-lambda{suffix}-90task-2trial-seed20261201'
if (ROOT / run_dir).exists() or (ROOT / evaluation_root).exists():
    raise FileExistsError('Refusing to overwrite lambda 1.0 outputs')
with (ROOT / 'runs/round3/status_before_lambda1.json').open('x') as backup:
    json.dump(state, backup, indent=2)
state['previous_finished_at'] = state.pop('finished_at', None)
state['extension_started_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
state['pid'] = os.getpid()
state['runs'][suffix] = {'lambda': 1.0, 'status': 'queued for training', 'run_dir': run_dir,
    'checkpoint': run_dir + '/checkpoints/step-000200-move-lora.pt',
    'evaluation_root': evaluation_root}


def save():
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATE)


def report():
    save()
    subprocess.run([PYTHON, '-m', 'experiments.robot.libero.round3_move_comparison'], check=True)


def execute(command, log, run):
    with open(log, 'x') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        run['process_pid'] = process.pid
        save()
        while process.poll() is None:
            url_path = ROOT / run['run_dir'] / 'wandb_run_url.txt'
            if url_path.exists() and not run.get('wandb_url'):
                run['wandb_url'] = url_path.read_text().strip()
                report()
            time.sleep(10)
        if process.returncode:
            raise RuntimeError(f'{command} exited {process.returncode}; see {log}')
        run.pop('process_pid', None)


report()
for suffix, run in [('10', state['runs']['10'])]:
    try:
        run['status'] = 'training'
        report()
        execute(['bash', 'runs/round3/train.sh', suffix, str(run['lambda']), '32', '1'],
                f'runs/round3/train-lambda{suffix}.log', run)
        result = json.loads((ROOT / run['run_dir'] / 'result.json').read_text())
        assert result['optimizer_steps'] == 200
        run['wandb_url'] = result['wandb_url']
        run['status'] = 'trained; queued for evaluation'
    except Exception:
        run['status'] = 'training failed'
        run['error'] = traceback.format_exc()
    report()

for suffix, run in [('10', state['runs']['10'])]:
    if run['status'] != 'trained; queued for evaluation':
        continue
    try:
        run['status'] = 'evaluating'
        report()
        execute(['bash', 'runs/round3/evaluate.sh', suffix], f'runs/round3/eval-lambda{suffix}.log', run)
        run['status'] = 'complete'
        report()  # Validates all 180 paired episodes before reporting results.
    except Exception:
        run['status'] = 'evaluation failed'
        run['error'] = traceback.format_exc()
        report()
state['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
report()
