"""Publish completed paired evaluations in the current experiment project."""
import json
from pathlib import Path
import wandb

Q = Path(__file__).resolve().parent
plan = json.loads((Q / 'plan.json').read_text())
payload = {}
for key in ['pretrained', 'lambda0', 'lambda1', 'moveonly']:
    result = json.loads((Q / f'test-{key}.json').read_text())
    payload[key] = result
    payload[key]['task_success_rate'] = result['successes'] / result['episodes']
options = dict(project=plan['wandb_project'], entity=plan['wandb_entity'],
               name='paired-final-evaluation', job_type='evaluation', dir=str(Q), config=plan)
try:
    run = wandb.init(**options, mode='online', settings=wandb.Settings(init_timeout=30))
except wandb.errors.Error:
    run = wandb.init(**options, mode='offline')
run.summary.update(payload)
run.finish()
