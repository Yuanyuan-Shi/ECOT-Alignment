"""Fresh MiniVLA LoRA with directional/stop loss and validation-selected checkpoints."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse,importlib.util,json,math,shutil,sys,time
from dataclasses import asdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from experiments.robot.libero.refined_move_alignment import refined_objective,literal_stop,episode_split,EarlyStopping
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('original_move_trainer',ROOT/'vla-scripts/finetune_move_alignment.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
Q=ROOT/'runs/refined-stop-generalization'

class Dataset(m.ReviewedQueryDataset):
    def __getitem__(self,i):
        r=super().__getitem__(i);r['literal_stop']=literal_stop(self.records[i]['reasoning_raw']);return r
class Collator(m.ReviewedQueryCollator):
    def __call__(self,items):
        r=super().__call__(items);r['literal_stop']=torch.tensor([x['literal_stop'] for x in items],dtype=torch.bool);return r

def forward(vla,batch,cfg):
    output=vla(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],pixel_values=batch['pixel_values'],labels=batch['labels'])
    tok=vla.action_tokenizer
    logits=m.extract_action_code_logits(output.logits,batch['labels'],num_image_patches=vla.vision_backbone.num_patches,
        action_token_begin_idx=tok.action_token_begin_idx,action_token_end_idx=tok.action_token_end_idx,
        tokenizer_len=tok.tokenizer_len,num_codes=tok.n_bins,num_groups=tok.vq_vae.vqvae_groups)
    actions=m.differentiable_vq_decode(logits,tok.vq_vae,temperature=cfg.move_temperature)
    low,high,mask=m._action_stats(vla,output.logits.device)
    actions=m.unnormalize_actions(actions,low,high,mask)
    alignment,d=refined_objective(actions,batch['reasoning_move_vector'],batch['literal_stop'])
    total=m.combine_objective_losses(output.loss,alignment,cfg.performance_loss_weight,1.)
    return total,output.loss,alignment,d

def summarize(rows):
    result={}
    for name,success in [('overall',None),('successful',True),('failed',False)]:
        selected=[r for r in rows if success is None or r['episode_success']==success]
        eligible=[r for r in selected if r['eligible']]
        k=sum(r['mismatch'] for r in eligible);n=len(eligible)
        result[name]={'queries':n,'mismatches':k,'rate':k/n if n else None,'excluded':len(selected)-n,
            'stop_queries':sum(r['stop'] for r in eligible),'stop_mismatches':sum(r['stop'] and r['mismatch'] for r in eligible),
            'directional_queries':sum(r['directional'] for r in eligible),'directional_mismatches':sum(r['directional'] and r['mismatch'] for r in eligible)}
    ds=[r['directional_penalty'] for r in rows if r['directional']]
    ss=[r['stop_penalty'] for r in rows if r['stop']]
    result['directional_loss']=sum(ds)/len(ds) if ds else 0
    result['stop_loss']=sum(ss)/len(ss) if ss else 0
    result['alignment_loss']=result['directional_loss']+result['stop_loss']
    return result

@torch.no_grad()
def evaluate(vla,loader,cfg,save=None):
    was_training=vla.training;vla.eval();rows=[]
    for batch in loader:
        batch=m._to_device(batch,torch.device('cuda:0'),torch.bfloat16)
        with torch.autocast('cuda',dtype=torch.bfloat16):total,perf,loss,d=forward(vla,batch,cfg)
        assert torch.isfinite(total)
        values={k:v.detach().cpu().tolist() for k,v in d.items()}
        outcomes=batch['episode_success'].cpu().tolist()
        for i,rid in enumerate(batch['record_ids']):
            row={k:v[i] for k,v in values.items()};row.update(record_id=rid,episode_success=outcomes[i]);rows.append(row)
    if save:Path(save).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    vla.train(was_training)
    return summarize(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['lambda1','moveonly'],required=True);args=p.parse_args()
    plan=json.loads((Q/'plan.json').read_text())
    cfg=m.MoveAlignmentFinetuneConfig(
        annotations=Path(plan['annotations']),audit_queries=Path(plan['audit_queries']),checkpoint=Path(plan['base_checkpoint']),
        run_name='refined-stop-lr5e5-'+args.mode+'-seed20261102',seed=20261102,
        epochs=13,batch_size=32,gradient_accumulation_steps=1,learning_rate=5e-5,warmup_steps=20,
        move_loss_type="refined_directional_stop",
        move_loss_weight=1.,performance_loss_weight=1. if args.mode=='lambda1' else 0.,
        lora_rank=16,lora_alpha=16,lora_dropout=.05,weight_decay=0,max_optimizer_steps=500,
        expected_query_count=1557,expected_task_count=90)
    assert cfg.checkpoint.resolve()==(ROOT/'artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt').resolve()
    run_dir=ROOT/cfg.run_root/cfg.run_name;run_dir.mkdir(parents=True,exist_ok=False)
    m.set_seed(cfg.seed);records=m.load_retained_query_records(cfg);train,val=episode_split(records,seed=cfg.seed)
    ids=lambda rs:[f"task{r['task_id']}-query{r['policy_query_index']}" for r in rs]
    assert ids(train)==plan['train_ids'] and ids(val)==plan['validation_ids']
    config={**asdict(cfg),'alignment_objective':'mean_directional_hinge + mean_literal_stop_net_hinge',
        'stop_tolerance_m':.03,'validation_every_steps':25,'early_stop_patience':4,'selection_metric':'validation overall eligible-query mismatch rate',
        'validation_split':'whole episodes, stratified by source outcome and presence of literal stop'}
    (run_dir/'training_config.json').write_text(json.dumps(config,indent=2,default=str)+'\n')
    (run_dir/'split_manifest.json').write_text(json.dumps({k:plan[k] for k in ['train_ids','validation_ids','train_episodes','validation_episodes']},indent=2)+'\n')
    for name in ['config.json','dataset_statistics.json']:shutil.copy2(cfg.checkpoint.parents[1]/name,run_dir/name)
    vla=m.load_vla(cfg.checkpoint,load_for_training=True,image_sequence_len=1).to('cuda:0');vla.requires_grad_(False);vla.vision_backbone_requires_grad=False
    for module in (vla.action_tokenizer.vq_vae.encoder,vla.action_tokenizer.vq_vae.decoder,vla.action_tokenizer.vq_vae.vq_layer):module.to('cuda:0').eval().requires_grad_(False)
    vla.action_tokenizer.vq_vae.device='cuda:0';vla.action_tokenizer.vq_vae.vq_layer.device=torch.device('cuda:0')
    lora=LoraConfig(r=16,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM',init_lora_weights='gaussian')
    vla.llm_backbone.llm=get_peft_model(vla.llm_backbone.llm,lora)
    vla.llm_backbone.llm.gradient_checkpointing_enable();vla.llm_backbone.llm.enable_input_require_grads()
    collator=Collator(vla.llm_backbone.get_tokenizer());gen=torch.Generator().manual_seed(cfg.seed)
    loader=DataLoader(Dataset(train,vla),batch_size=32,shuffle=True,collate_fn=collator,generator=gen)
    train_eval=DataLoader(Dataset(train,vla),batch_size=32,shuffle=False,collate_fn=collator)
    val_loader=DataLoader(Dataset(val,vla),batch_size=32,shuffle=False,collate_fn=collator)
    run=m.wandb.init(project='ecot-move-alignment',name=cfg.run_name,config=config)
    (run_dir/'wandb_run_url.txt').write_text((run.url or 'offline')+'\n')
    def log(data):
        run.log(data)
        with (run_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
    # Keep pre-training evaluation from affecting training random-number state.
    with torch.random.fork_rng(devices=[0]):
        initial=evaluate(vla,val_loader,cfg,run_dir/'validation-step000000.jsonl')
        if args.mode=='lambda1':
            baseline=evaluate(vla,train_eval,cfg,run_dir/'pretrained-training-scores.jsonl')
            (run_dir/'pretrained-training-summary.json').write_text(json.dumps(baseline,indent=2)+'\n')
    stopper=EarlyStopping();stopper.update(initial['overall']['rate'],0)
    best={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(vla.llm_backbone.llm).items()}
    vla.llm_backbone.llm.save_pretrained(run_dir/'adapter-best')
    log({'optimizer_step':0,'validation':initial,'best_step':0,'best_validation_rate':stopper.best})
    optimizer=torch.optim.AdamW([p for p in vla.parameters() if p.requires_grad],lr=cfg.learning_rate,weight_decay=0)
    step=0;stop=False;epoch=0;started=time.time();vla.train()
    while step<500 and not stop:
        epoch+=1
        for batch in loader:
            optimizer.zero_grad(set_to_none=True);batch=m._to_device(batch,torch.device('cuda:0'),torch.bfloat16)
            with torch.autocast('cuda',dtype=torch.bfloat16):total,perf,alignment,d=forward(vla,batch,cfg)
            if not torch.isfinite(total):raise RuntimeError('Nonfinite training loss')
            total.backward();grad=torch.nn.utils.clip_grad_norm_([p for p in vla.parameters() if p.requires_grad],1.)
            if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
            step+=1;lr=cfg.learning_rate*min(1.,step/20)
            for g in optimizer.param_groups:g['lr']=lr
            optimizer.step()
            log({'optimizer_step':step,'epoch':epoch,'total_loss':float(total.detach()),'performance_loss':float(perf.detach()),'alignment_loss':float(alignment.detach()),'learning_rate':lr})
            print(f'step={step}/500 total={total.item():.6f} alignment={alignment.item():.6f}',flush=True)
            if step%25==0:
                with torch.random.fork_rng(devices=[0]):metrics=evaluate(vla,val_loader,cfg,run_dir/f'validation-step{step:06d}.jsonl')
                improved,stop=stopper.update(metrics['overall']['rate'],step)
                vla.llm_backbone.llm.save_pretrained(run_dir/f'adapter-step{step:06d}')
                if improved:
                    best={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(vla.llm_backbone.llm).items()}
                    vla.llm_backbone.llm.save_pretrained(run_dir/'adapter-best')
                payload={'optimizer_step':step,'validation':metrics,'best_step':stopper.best_step,'best_validation_rate':stopper.best,'checks_without_improvement':stopper.bad_checks}
                log(payload);(run_dir/'selection.json').write_text(json.dumps(payload,indent=2)+'\n')
                print('VALIDATION',json.dumps(payload),flush=True)
            if stop or step>=500:break
    set_peft_model_state_dict(vla.llm_backbone.llm,best)
    final_train=evaluate(vla,train_eval,cfg,run_dir/'selected-training-scores.jsonl')
    final_val=evaluate(vla,val_loader,cfg,run_dir/'selected-validation-scores.jsonl')
    assert abs(final_val['overall']['rate']-stopper.best)<1e-8
    vla.llm_backbone.llm.save_pretrained(run_dir/'adapter-final')
    vla.llm_backbone.llm=vla.llm_backbone.llm.merge_and_unload()
    checkpoint=run_dir/'checkpoints'/f'step-{stopper.best_step:06d}-move-lora.pt'
    result={'base_checkpoint':str(cfg.checkpoint.resolve()),'optimizer_steps':step,'selected_step':stopper.best_step,
        'stop_reason':'validation patience exhausted' if stop else '500-step budget reached','best_validation_rate':stopper.best,
        'performance_loss_weight':cfg.performance_loss_weight,'move_loss_weight':1.,'training':final_train,'validation':final_val,
        'checkpoint':str(checkpoint),'wandb_url':run.url,'elapsed_training_seconds':time.time()-started}
    m._save_native_checkpoint(vla,checkpoint,result)
    (run_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');run.summary.update(result);run.finish()
    print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
