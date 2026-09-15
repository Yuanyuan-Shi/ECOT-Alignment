"""Fixed-budget, resumable large-data LoRA experiment with offline-first W&B."""
import os,sys,json,time,random,importlib.util,shutil,argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'runs/large-data-alignment'
spec=importlib.util.spec_from_file_location('refined_trainer',ROOT/'vla-scripts/finetune_refined_alignment.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);m=r.m
class Dataset(r.Dataset):
    def __getitem__(self,i):
        x=super().__getitem__(i);x['record_id']=self.records[i].get('record_id',self.records[i]['episode_id']+'::'+str(self.records[i]['policy_query_index']));return x

def forward(vla,batch,cfg):
    from experiments.robot.libero.move_alignment_training import terminal_action_labels
    output=vla(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],pixel_values=batch['pixel_values'],labels=batch['labels'])
    tok=vla.action_tokenizer
    labels=terminal_action_labels(batch['labels'],tok.action_token_begin_idx,tok.action_token_end_idx,tok.vq_vae.vqvae_groups)
    logits=m.extract_action_code_logits(output.logits,labels,num_image_patches=vla.vision_backbone.num_patches,
        action_token_begin_idx=tok.action_token_begin_idx,action_token_end_idx=tok.action_token_end_idx,
        tokenizer_len=tok.tokenizer_len,num_codes=tok.n_bins,num_groups=tok.vq_vae.vqvae_groups)
    actions=m.differentiable_vq_decode(logits,tok.vq_vae,temperature=cfg.move_temperature)
    low,high,mask=m._action_stats(vla,output.logits.device)
    actions=m.unnormalize_actions(actions,low,high,mask)
    align,d=r.refined_objective(actions,batch['reasoning_move_vector'],batch['literal_stop'],stop_loss='net_squared')
    perf=output.loss;loss=m.combine_objective_losses(perf,align,cfg.performance_loss_weight,cfg.move_loss_weight)
    return loss,perf,align,d

def backward_effective_batch(vla,batch,cfg):
    """Exactly weight token CE and pooled eligible-query loss across microbatches."""
    size=len(batch['record_ids']);eligible=(batch['reasoning_move_vector'].norm(dim=-1)>0)|batch['literal_stop']
    token_total=int((batch['labels'][:,1:]!=-100).sum());eligible_total=int(eligible.sum())
    total_value=perf_value=align_value=0.;details=[]
    def sliced(x,start,end):
        if isinstance(x,dict):return {k:sliced(v,start,end) for k,v in x.items()}
        if isinstance(x,(torch.Tensor,list)):return x[start:end]
        return x
    for start in range(0,size,32):
        end=min(start+32,size);micro=sliced(batch,start,end)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,perf,align,d=forward(vla,micro,cfg)
            pw=int((micro['labels'][:,1:]!=-100).sum())/max(token_total,1)
            aw=int(eligible[start:end].sum())/max(eligible_total,1)
            loss=cfg.performance_loss_weight*perf*pw+cfg.move_loss_weight*align*aw
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite microbatch loss')
        loss.backward();total_value+=float(loss.detach());perf_value+=float(perf.detach())*pw;align_value+=float(align.detach())*aw
        details.append({k:v.detach() for k,v in d.items()})
        del loss,perf,align,d
    return total_value,perf_value,align_value,{k:torch.cat([x[k] for x in details]) for k in details[0]}

@torch.no_grad()
def evaluate(vla,loader,cfg,path=None):
    mode=vla.training;vla.eval();rows=[];perf_sum=0.;tokens=0
    for batch in loader:
        batch=m._to_device(batch,torch.device('cuda:0'),torch.bfloat16)
        with torch.autocast('cuda',dtype=torch.bfloat16):total,perf,align,d=forward(vla,batch,cfg)
        assert torch.isfinite(total)
        n=int((batch['labels'][:,1:]!=-100).sum());perf_sum+=float(perf)*n;tokens+=n
        values={k:v.cpu().tolist() for k,v in d.items()}
        for i,rid in enumerate(batch['record_ids']):
            x={k:v[i] for k,v in values.items()};x.update(record_id=rid,episode_success=bool(batch['episode_success'][i]));rows.append(x)
    result=r.summarize(rows);result['performance_loss']=perf_sum/tokens if tokens else 0.;result['total_loss']=cfg.performance_loss_weight*result['performance_loss']+cfg.move_loss_weight*result['alignment_loss']
    if path:Path(path).write_text(''.join(json.dumps(x)+'\n' for x in rows))
    vla.train(mode);return result

def write(path,data):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,default=str)+'\n');tmp.replace(p)
def save_torch(path,data):
    p=Path(path);tmp=p.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['lambda0','lambda1','moveonly'],default='lambda1');p.add_argument('--probe',action='store_true')
    p.add_argument('--plan',type=Path,default=Q/'plan.json');p.add_argument('--records',type=Path,default=Q/'records.jsonl')
    p.add_argument('--data-root',type=Path,help='Root containing raw/large-data-collection-* for uploaded records')
    p.add_argument('--base-checkpoint',type=Path);p.add_argument('--run-dir',type=Path)
    p.add_argument('--alignment-weight',type=float,help='Override lambda, the weight on the refined MOVE alignment loss')
    p.add_argument('--final-only',action='store_true',help='Keep only the merged final 2k checkpoint after completion')
    p.add_argument('--checkpoint-only-init',action='store_true',help='Build an empty LLM before loading the complete native checkpoint')
    args=p.parse_args()
    plan=json.loads(args.plan.read_text());
    if args.base_checkpoint:plan['base_checkpoint']=str(args.base_checkpoint.resolve())
    if args.final_only:plan['checkpoint_steps']=[]
    seed=plan['training_seed'];m.set_seed(seed)
    assert plan['batch_size']==32 and plan['gradient_accumulation_steps']==1
    move_weight=0. if args.mode=='lambda0' else 1.
    if args.alignment_weight is not None:
        if args.mode=='moveonly':raise ValueError('--alignment-weight cannot be combined with --mode moveonly')
        if args.alignment_weight<0:raise ValueError('--alignment-weight must be nonnegative')
        move_weight=args.alignment_weight
    cfg=m.MoveAlignmentFinetuneConfig(checkpoint=Path(plan['base_checkpoint']),batch_size=plan['batch_size'],gradient_accumulation_steps=1,learning_rate=plan['learning_rate'],move_loss_weight=move_weight,performance_loss_weight=0. if args.mode=='moveonly' else 1.,seed=seed)
    rd=(args.run_dir or ROOT/f'runs/ecot-move-alignment/large-data-{args.mode}-b32-s2000').resolve();rd.mkdir(parents=True,exist_ok=True)
    if (rd/'result.json').exists() and not args.probe:return
    if args.probe:
        old=json.loads((ROOT/'runs/refined-stop-pooled/plan.json').read_text());cfg.annotations=Path(old['annotations']);cfg.audit_queries=Path(old['audit_queries']);records=m.load_retained_query_records(cfg)
        train=sorted(records,key=lambda x:len(x['reasoning_raw']),reverse=True)[:128];val=train[:64]
    else:
        records=[json.loads(l) for l in args.records.read_text().splitlines()]
        if args.data_root:
            marker='/experiments/robot/libero/results/'
            for item in records:
                original=item['before_frame']
                if marker not in original:raise ValueError(f'Cannot relocate before_frame: {original}')
                item['before_frame']=str(args.data_root/(original.split(marker,1)[1]))
        train=[x for x in records if x['split']=='training'];val=[x for x in records if x['split']=='validation']
    # A native checkpoint contains the complete LLM state. Empty construction avoids
    # downloading redundant Hugging Face base weights before that state is loaded.
    vla=m.load_vla(cfg.checkpoint,load_for_training=not args.checkpoint_only_init,image_sequence_len=1).to('cuda:0');vla.requires_grad_(False);vla.vision_backbone_requires_grad=False
    for mod in (vla.action_tokenizer.vq_vae.encoder,vla.action_tokenizer.vq_vae.decoder,vla.action_tokenizer.vq_vae.vq_layer):mod.to('cuda:0').eval().requires_grad_(False)
    vla.action_tokenizer.vq_vae.device='cuda:0';vla.action_tokenizer.vq_vae.vq_layer.device=torch.device('cuda:0')
    vla.llm_backbone.llm=get_peft_model(vla.llm_backbone.llm,LoraConfig(r=16,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM',init_lora_weights='gaussian'))
    vla.llm_backbone.llm.gradient_checkpointing_enable();vla.llm_backbone.llm.enable_input_require_grads()
    collator=r.Collator(vla.llm_backbone.get_tokenizer());dataset=Dataset(train,vla)
    val_loader=DataLoader(Dataset(val,vla),batch_size=32,collate_fn=collator)
    train_loader=DataLoader(dataset,batch_size=32,collate_fn=collator)
    optimizer=torch.optim.AdamW([p for p in vla.parameters() if p.requires_grad],lr=cfg.learning_rate,weight_decay=0.)
    if args.probe:
        vla.train();torch.cuda.reset_peak_memory_stats();started=time.time()
        for start in [0,32]:
            batch=collator([dataset[i] for i in range(start,start+32)])
            optimizer.zero_grad(set_to_none=True);batch=m._to_device(batch,torch.device('cuda:0'),torch.bfloat16)
            loss,_,_,_=backward_effective_batch(vla,batch,cfg)
            assert np.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in vla.parameters() if p.grad is not None)
            optimizer.step()
        probe=dict(batch_size=32,microbatch_size=32,gradient_accumulation_steps=1,batches=2,seconds=time.time()-started,peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30,passed=True)
        write(Q/'batch32_probe.json',probe);print(probe,flush=True);return
    for name in ['config.json','dataset_statistics.json']:shutil.copy2(cfg.checkpoint.parents[1]/name,rd/name)
    write(rd/'training_config.json',{**plan,'mode':args.mode,'performance_loss_weight':cfg.performance_loss_weight,'alignment_loss_weight':cfg.move_loss_weight,
        'records':str(args.records.resolve()),'data_root':str(args.data_root.resolve()) if args.data_root else None,
        'final_only':args.final_only,'checkpoint_only_init':args.checkpoint_only_init,
        'alignment_weight_override':args.alignment_weight})
    # Keep a durable local history while publishing this round in its own project.
    project=plan['wandb_project']
    previous=json.loads((rd/'wandb.json').read_text()) if (rd/'wandb.json').exists() else {}
    same_project=previous.get('project')==project
    run_id=previous['id'] if same_project else m.wandb.util.generate_id()
    options=dict(project=project,entity=plan.get('wandb_entity'),id=run_id,name=rd.name,
                 config={**plan,'mode':args.mode},dir=str(rd),resume='allow')
    try:
        run=m.wandb.init(**options,mode=plan.get('wandb_mode','online'),settings=m.wandb.Settings(init_timeout=30))
    except m.wandb.errors.Error:
        if os.environ.get('WANDB_REQUIRE_ONLINE') == '1':
            raise
        run=m.wandb.init(**options,mode='offline')
    write(rd/'wandb.json',dict(id=run.id,directory=str(Path(run.dir).parent),project=project,mode=run.settings.mode))
    if not run.settings._offline:
        (rd/'wandb_run_url.txt').write_text(run.url+'\n')
    def flatten(value,prefix=''):
        out={}
        for key,item in value.items():
            name=prefix+key
            if isinstance(item,dict):out.update(flatten(item,name+'/'))
            elif item is not None:out[name]=item
        return out
    run.define_metric('optimizer_step')
    run.define_metric('*',step_metric='optimizer_step')
    if not same_project and (rd/'metrics.jsonl').exists():
        history={}
        for line in (rd/'metrics.jsonl').read_text().splitlines():
            payload=json.loads(line);history.setdefault(payload['optimizer_step'],{}).update(flatten(payload))
        for step_number in sorted(history):run.log(history[step_number])
    def log(payload):
        run.log(flatten(payload))
        with (rd/'metrics.jsonl').open('a') as f:f.write(json.dumps(payload)+'\n')
        write(rd/'progress.json',payload)
    step=0;epoch=0;cursor=0
    resume=rd/'resume.pt'
    if resume.exists():
        x=torch.load(resume,map_location='cpu',weights_only=False);assert x['mode']==args.mode and x['base_checkpoint']==plan['base_checkpoint'] and x.get('loss_version')==plan['loss_version'] and x.get('alignment_loss_weight',cfg.move_loss_weight)==cfg.move_loss_weight
        set_peft_model_state_dict(vla.llm_backbone.llm,x['adapter']);optimizer.load_state_dict(x['optimizer']);step=x['step'];epoch=x['epoch'];cursor=x['cursor']
        random.setstate(x['python_rng']);np.random.set_state(x['numpy_rng']);torch.set_rng_state(x['torch_rng']);torch.cuda.set_rng_state_all(x['cuda_rng'])
    else:
        with torch.random.fork_rng(devices=[0]):
            initial=evaluate(vla,val_loader,cfg);baseline=evaluate(vla,train_loader,cfg,rd/'initial-training-scores.jsonl')
        write(rd/'initial-training-summary.json',baseline);write(rd/'initial-validation-summary.json',initial);log(dict(optimizer_step=0,validation=initial,training=baseline))
    def snapshot():
        payload=dict(loss_version=plan['loss_version'],mode=args.mode,base_checkpoint=plan['base_checkpoint'],alignment_loss_weight=cfg.move_loss_weight,step=step,epoch=epoch,cursor=cursor,adapter={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(vla.llm_backbone.llm).items()},optimizer=optimizer.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        save_torch(resume,payload)
        if step in plan['checkpoint_steps']:
            save_torch(rd/f'resume-step{step:06d}.pt',payload);vla.llm_backbone.llm.save_pretrained(rd/f'adapter-step{step:06d}')
    started=time.time();vla.train();torch.cuda.reset_peak_memory_stats()
    while step<plan['max_steps']:
        order=np.random.default_rng(seed+epoch).permutation(len(train)).tolist()
        if cursor>=len(order):epoch+=1;cursor=0;continue
        indices=order[cursor:cursor+cfg.batch_size];batch=collator([dataset[i] for i in indices]);batch=m._to_device(batch,torch.device('cuda:0'),torch.bfloat16)
        optimizer.zero_grad(set_to_none=True);tick=time.time()
        loss,perf,align,d=backward_effective_batch(vla,batch,cfg)
        grad=torch.nn.utils.clip_grad_norm_([p for p in vla.parameters() if p.requires_grad],1.)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
        step+=1;lr=cfg.learning_rate*min(1.,step/plan['warmup_steps'])
        for g in optimizer.param_groups:g['lr']=lr
        optimizer.step();cursor+=len(indices)
        vals=dict(total_loss=float(loss),performance_loss=float(perf),alignment_loss=float(align),queries=len(indices),eligible_queries=int(d['eligible'].sum()),mismatches=int(d['mismatch'].sum()),directional_queries=int(d['directional'].sum()),directional_mismatches=int((d['directional']&d['mismatch']).sum()),stop_queries=int(d['stop'].sum()),stop_mismatches=int((d['stop']&d['mismatch']).sum()))
        vals.update(directional_loss=float(d['directional_penalty'][d['directional']].mean()) if d['directional'].any() else 0.,stop_loss=float(d['stop_penalty'][d['stop']].mean()) if d['stop'].any() else 0.,excluded_queries=len(indices)-int(d['eligible'].sum()))
        log(dict(optimizer_step=step,epoch=epoch+cursor/len(train),training=vals,learning_rate=lr,gradient_norm=float(grad),step_seconds=time.time()-tick,peak_gpu_allocated_gb=torch.cuda.max_memory_allocated()/2**30))
        if step%25==0:print(f'{args.mode} step={step}/2000 loss={float(loss):.6f}',flush=True)
        if step%250==0:
            if not args.final_only:
                with torch.random.fork_rng(devices=[0]):metrics=evaluate(vla,val_loader,cfg,rd/f'validation-step{step:06d}.jsonl')
                log(dict(optimizer_step=step,validation=metrics))
            snapshot()
    with torch.random.fork_rng(devices=[0]):
        final_train=evaluate(vla,train_loader,cfg,rd/'final-training-scores.jsonl');final_val=evaluate(vla,val_loader,cfg,rd/'final-validation-scores.jsonl')
    # Verify adapter merging on a fixed input before export.
    sample=m._to_device(collator([dataset[0]]),torch.device('cuda:0'),torch.float32);vla.eval()
    from torch.nn.attention import sdpa_kernel,SDPBackend
    original_dtypes=sorted({str(p.dtype) for p in vla.llm_backbone.llm.parameters()})
    vla.float()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):before=vla(input_ids=sample['input_ids'],attention_mask=sample['attention_mask'],pixel_values=sample['pixel_values']).logits.float().cpu()
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):repeat=vla(input_ids=sample['input_ids'],attention_mask=sample['attention_mask'],pixel_values=sample['pixel_values']).logits.float().cpu()
    vla.llm_backbone.llm=vla.llm_backbone.llm.merge_and_unload()
    vla.eval()
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):after=vla(input_ids=sample['input_ids'],attention_mask=sample['attention_mask'],pixel_values=sample['pixel_values']).logits.float().cpu()
    error=float((before-after).abs().max())
    write(rd/'export_diagnostics.json',dict(original_dtypes=original_dtypes,repeat_max_error=float((before-repeat).abs().max()),merge_max_error=error,tf32_disabled=True,export_dtype='float32',export_attention_backend='math'))
    assert torch.allclose(before,after,atol=.003,rtol=.003),f'Merge changed logits: {error}'
    cp=rd/'checkpoints/step-002000-move-lora.pt'
    result=dict(loss_version=plan['loss_version'],base_checkpoint=plan['base_checkpoint'],optimizer_steps=step,checkpoint=str(cp),training=final_train,validation=final_val,merge_max_logit_error=error,elapsed_seconds=time.time()-started)
    m._save_native_checkpoint(vla,cp,result)
    saved=torch.load(cp,map_location='cpu',weights_only=False)['model']
    for name in ['vision_backbone','projector','llm_backbone']:
        current=getattr(vla,name).state_dict();assert current.keys()==saved[name].keys();assert all(torch.equal(v.detach().cpu(),saved[name][k]) for k,v in current.items())
    result['export_state_verified']=True;write(rd/'result.json',result);log(dict(optimizer_step=step,final_training=final_train,final_validation=final_val));run.summary.update(result);run.finish()
    if args.final_only:
        for path in [rd/'resume.pt',*rd.glob('resume-step*.pt')]:path.unlink(missing_ok=True)
        for path in rd.glob('adapter-step*'):
            if path.is_dir():shutil.rmtree(path)
if __name__=='__main__':main()
