"""Detached sequential training and paired evaluation, with durable status/logs."""
import os,json,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/home/exx/Projects/ECOT-Alignment');os.chdir(ROOT)
Q=ROOT/'runs/refined-stop-generalization';PY='/home/exx/.conda/envs/openvla/bin/python'
plan=json.loads((Q/'plan.json').read_text())
state={'status':'running','pid':os.getpid(),'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'runs':{}}
for key in ['pretrained','lambda1','moveonly']:
    state['runs'][key]={'evaluation_root':f'experiments/robot/libero/results/refined-stop-fast-{key}-30task-3trial-seed20261211','evaluation_status':'pending'}
    if key!='pretrained':state['runs'][key].update(run_dir=f'runs/ecot-move-alignment/refined-stop-lr5e5-{key}-seed20261102',training_status='queued')
    for field in ['evaluation_root','run_dir']:
        if field in state['runs'][key] and (ROOT/state['runs'][key][field]).exists():raise FileExistsError('Refusing existing outputs')
def save():
    state['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');p=Q/'status.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(Q/'status.json')
def report():
    save();subprocess.run([PY,str(Q/'report.py')],check=True)
def execute(cmd,log):
    with (Q/log).open('x') as stream:
        process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
        state['active_process_pid']=process.pid;save()
        while True:
            try:code=process.wait(timeout=45);break
            except subprocess.TimeoutExpired:
                try:report()
                except Exception:traceback.print_exc()
        state.pop('active_process_pid',None)
        if code:raise RuntimeError(f'{cmd} exited {code}; see {log}')
try:
    report()
    for key in ['lambda1','moveonly']:
        state['runs'][key]['training_status']='training';report()
        execute([PY,'-u','vla-scripts/finetune_refined_alignment.py','--mode',key],f'train-{key}.log')
        result=json.loads((ROOT/state['runs'][key]['run_dir']/'result.json').read_text())
        assert result['base_checkpoint']==plan['base_checkpoint'] and result['optimizer_steps']<=500
        state['runs'][key].update(training_status='complete',checkpoint=result['checkpoint'],selected_step=result['selected_step']);report()
    for key in ['pretrained','lambda1','moveonly']:
        run=state['runs'][key];run['evaluation_status']='evaluating';report()
        cp=plan['base_checkpoint'] if key=='pretrained' else run['checkpoint']
        command=[PY,'-u','experiments/robot/libero/run_libero_eval.py','--model_family','prismatic','--pretrained_checkpoint',cp,
            '--task_suite_name','libero_90','--task_ids',','.join(map(str,plan['evaluation_task_ids'])),'--episodes_per_task','3','--seed',str(plan['evaluation_base_seed']),
            '--center_crop','False','--use_wrist_image','False','--use_cot','True','--num_open_loop_steps','10','--n_procs_per_gpu','6',
            '--enable_alignment_evaluator','True','--alignment_cosine_similarity','True','--alignment_save_frames','False',
            '--alignment_output_dir',run['evaluation_root'],'--output_dir',Path(run['evaluation_root']).name,'--use_wandb','False','--run_id_note',Path(run['evaluation_root']).name]
        (Q/f'evaluate-{key}-command.json').write_text(json.dumps(command,indent=2)+'\n')
        execute(command,f'eval-{key}.log');run['evaluation_status']='complete';report()
    state['status']='complete';state['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');report()
except Exception:
    state['status']='failed';state['error']=traceback.format_exc();save()
    try:report()
    except Exception:traceback.print_exc()
    raise
