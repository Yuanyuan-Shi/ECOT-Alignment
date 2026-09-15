#!/usr/bin/env python3
"""Train lambda 0.5/1/2, then evaluate each on the pinned 270 LIBERO cases."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

REPO = Path('/mnt1/ECOT-Alignment')
ROOT = REPO / 'pi05_policy_training'
PY = ROOT / 'data/minivla_venv/bin/python'
BASE = ROOT / 'data/minivla_reference/checkpoints/step-100000-epoch-22-loss=0.0261.pt'
RECORDS = ROOT / 'data/prior_ft_training_source/assembled/records.jsonl'
DATA_ROOT = ROOT / 'data/prior_ft_training_git/raw'
PLAN = REPO / 'runs/large-data-alignment/plan.json'
MANIFEST_PATH = ROOT / 'configs/minivla_lambda_sweep_20260914.json'
STATUS = ROOT / 'reports/minivla_lambda_sweep_status.json'
RUN_ROOT = ROOT / 'runs/minivla_lambda_sweep_20260914'

MODELS = [
    ('lambda0p5', 'Joint FT (lambda=0.5)', 0.5, 'large-data-lambda0p5-b32-s2000-20260914'),
    ('lambda1', 'Joint FT (lambda=1)', 1.0, 'large-data-lambda1-b32-s2000-20260914'),
    ('lambda2', 'Joint FT (lambda=2)', 2.0, 'large-data-lambda2-b32-s2000-20260914'),
]

ENV = os.environ.copy()
ENV.update({
    'PYTHONPATH': f"{REPO}:{ROOT/'openpi/third_party/libero'}",
    'LIBERO_CONFIG_PATH': str(ROOT/'configs/libero'),
    'PRISMATIC_DATA_ROOT': str(ROOT/'data/rlds'),
    'HF_HOME': str(ROOT/'data/minivla_reference/hf_cache'),
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
    'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1', 'TOKENIZERS_PARALLELISM': 'false',
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'WANDB_REQUIRE_ONLINE': '1',
})


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def set_status(stage: str, model: str = '', detail: str = '') -> None:
    write(STATUS, {'status': 'running', 'stage': stage, 'model': model,
                   'detail': detail, 'updated_unix': time.time()})


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8<<20), b''): digest.update(block)
    return digest.hexdigest()


def notify(title: str, detail: str) -> None:
    subprocess.run([str(ROOT/'openpi/.venv/bin/python'), str(ROOT/'scripts/notify_status.py'), title, detail], check=False)


def validate_training(key: str, weight: float, run_dir: Path) -> Path:
    result=json.loads((run_dir/'result.json').read_text())
    config=json.loads((run_dir/'training_config.json').read_text())
    wandb=json.loads((run_dir/'wandb.json').read_text())
    if wandb.get('mode') != 'online' or not wandb.get('id'):
        raise RuntimeError(f'{key}: W&B did not run online')
    if result['optimizer_steps'] != 2000 or config['alignment_loss_weight'] != weight:
        raise RuntimeError(f'{key}: incorrect steps or lambda in exported configuration')
    for split, expected in (('training',12666),('validation',3123)):
        if result[split]['overall']['queries'] != expected:
            raise RuntimeError(f'{key}: incorrect {split} eligible-query denominator')
    checkpoint=Path(result['checkpoint'])
    if not result.get('export_state_verified') or not checkpoint.is_file() or checkpoint.stat().st_size < 5_000_000_000:
        raise RuntimeError(f'{key}: final native checkpoint validation failed')
    intermediates=[p for p in run_dir.rglob('*.pt') if p != checkpoint]
    if intermediates:
        raise RuntimeError(f'{key}: unexpected retained intermediate checkpoints: {intermediates}')
    return checkpoint


def train(key: str, label: str, weight: float, directory: str) -> Path:
    run_dir=RUN_ROOT/directory
    set_status('training', key, f'lambda={weight}; progress: {run_dir}/progress.json')
    command=[str(PY), str(REPO/'vla-scripts/finetune_large_alignment.py'),
             '--mode','lambda1','--alignment-weight',str(weight),'--plan',str(PLAN),
             '--records',str(RECORDS),'--data-root',str(DATA_ROOT),
             '--base-checkpoint',str(BASE),'--run-dir',str(run_dir),
             '--final-only','--checkpoint-only-init']
    subprocess.run(command,cwd=ROOT/'data/minivla_reference',env=ENV,check=True)
    checkpoint=validate_training(key,weight,run_dir)
    set_status('training_complete',key,str(checkpoint))
    return checkpoint


def evaluate(key: str, label: str, checkpoint: Path) -> None:
    task_ids=','.join(map(str,range(90)))
    smoke=ROOT/f'reports/minivla_sweep_{key}_smoke'
    full=ROOT/f'reports/minivla_sweep_{key}_evaluation'
    summary=ROOT/f'reports/prior_ft_{key}_evaluation_summary.json'
    for path in (smoke,full):
        if path.exists(): shutil.rmtree(path)
    common=[str(PY),str(REPO/'experiments/robot/libero/run_libero_eval.py'),
            '--model_family','prismatic','--pretrained_checkpoint',str(checkpoint),
            '--task_suite_name','libero_90','--initial_state_offset','13','--seed','7',
            '--center_crop','False','--use_wrist_image','False','--use_cot','True',
            '--num_open_loop_steps','10','--enable_alignment_evaluator','True',
            '--alignment_cosine_similarity','True','--alignment_save_frames','False','--use_wandb','False']
    set_status('evaluation_smoke',key,str(checkpoint))
    subprocess.run(common+['--task_ids','0','--num_trials_per_task','1','--episodes_per_task','1',
        '--n_procs_per_gpu','1','--max_episode_steps','10','--alignment_output_dir',str(smoke),
        '--output_dir',f'minivla-sweep-{key}-smoke','--run_id_note',f'minivla-sweep-{key}-smoke'],
        cwd=ROOT/'data/minivla_reference',env=ENV,check=True)
    episodes=json.loads((smoke/'episode_summary.json').read_text())
    queries=[json.loads(x) for x in (smoke/'alignment_queries.jsonl').read_text().splitlines() if x]
    if len(episodes)!=1 or not queries or not all(len(q['decoded_action_chunk'])==10 and len(q['decoded_action_chunk'][0])==7 for q in queries):
        raise RuntimeError(f'{key}: evaluation smoke validation failed')
    set_status('evaluation_270',key,'90 tasks x 3 trials')
    subprocess.run(common+['--task_ids',task_ids,'--num_trials_per_task','3','--episodes_per_task','3',
        '--n_procs_per_gpu','6','--alignment_output_dir',str(full),
        '--output_dir',f'minivla-sweep-{key}-90task-3trial-seed7-state13',
        '--run_id_note',f'minivla-sweep-{key}-90task-3trial-seed7-state13'],
        cwd=ROOT/'data/minivla_reference',env=ENV,check=True)
    subprocess.run([str(PY),str(ROOT/'scripts/summarize_prior_ft_evaluation.py'),
        '--model-key',key,'--model-name',label,'--run-dir',str(full),
        '--checkpoint',str(checkpoint),'--output',str(summary)],cwd=REPO,env=ENV,check=True)


def main() -> None:
    manifest=json.loads(MANIFEST_PATH.read_text())
    if sha256(BASE)!=manifest['base_checkpoint_sha256'] or sha256(RECORDS)!=manifest['records_sha256']:
        raise RuntimeError('Pinned base checkpoint or training records hash changed')
    manifest['status']='approved_and_launched';manifest['launched_unix']=time.time();write(MANIFEST_PATH,manifest)
    checkpoints={}
    for key,label,weight,directory in MODELS:
        checkpoints[key]=train(key,label,weight,directory)
    set_status('all_training_complete',detail='Beginning matched evaluations')
    for key,label,_,_ in MODELS:
        evaluate(key,label,checkpoints[key])
    set_status('rendering_report',detail='Adding three rows to pi05 LIBERO-90 table')
    subprocess.run([str(PY),str(ROOT/'scripts/render_report.py')],cwd=REPO,env=ENV,check=True)
    report=(ROOT/'reports/index.html').read_text()
    for label in ('Joint FT (λ=0.5)','Joint FT (λ=1)','Joint FT (λ=2)'):
        if label not in report: raise RuntimeError(f'Report missing {label}')
    with urlopen('http://localhost:8000/index.html',timeout=10) as response:
        if response.status!=200: raise RuntimeError(f'HTML server returned {response.status}')
    write(STATUS,{'status':'complete','stage':'complete','detail':'Three trainings, 810 matched episodes, summaries, and HTML update completed','updated_unix':time.time()})
    notify('MiniVLA lambda sweep complete','lambda=0.5, 1, and 2 results are in the pi05 report tab.')


if __name__=='__main__':
    try: main()
    except Exception as exc:
        write(STATUS,{'status':'failed','stage':'failed','detail':repr(exc),'updated_unix':time.time()})
        notify('MiniVLA lambda sweep FAILED',str(exc))
        raise
