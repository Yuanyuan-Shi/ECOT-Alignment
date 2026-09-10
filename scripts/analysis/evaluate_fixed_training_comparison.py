"""Evaluate frozen original MiniVLA and Round 3 lambda1 on the exact training split."""
import importlib.util
import json
import sys
from pathlib import Path
import torch
from peft import PeftModel
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('move_finetune', ROOT/'vla-scripts/finetune_move_alignment.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
OUT=ROOT/'analysis_results/move_only_500/fixed_training_comparison.json'
run=ROOT/'runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambda10-seed20261102'
raw=json.loads((run/'training_config.json').read_text())
cfg=m.MoveAlignmentFinetuneConfig(**raw)
for key in ('checkpoint','annotations','audit_queries','run_root'):
    setattr(cfg,key,Path(getattr(cfg,key)))
m.set_seed(cfg.seed)
records=m.load_retained_query_records(cfg)
train,_=m.stratified_query_split(records,cfg.validation_fraction,cfg.seed)
manifest=json.loads((run/'split_manifest.json').read_text())
assert [f"task{r['task_id']}-query{r['policy_query_index']}" for r in train]==manifest['train_ids']
zeros=sum(not any(r['reasoning_move_vector']) for r in train)
result={'queries':len(train),'zero_vectors':zeros,'metric':'teacher-forced soft-decoded policy translation sum; dropout disabled','models':{}}
vla=m.load_vla(cfg.checkpoint,load_for_training=True,image_sequence_len=1).to('cuda:0')
vla.requires_grad_(False);vla.vision_backbone_requires_grad=False
for module in (vla.action_tokenizer.vq_vae.encoder,vla.action_tokenizer.vq_vae.decoder,vla.action_tokenizer.vq_vae.vq_layer):
    module.to('cuda:0').eval().requires_grad_(False)
vla.action_tokenizer.vq_vae.device='cuda:0'
vla.action_tokenizer.vq_vae.vq_layer.device=torch.device('cuda:0')
for name in ('Original MiniVLA','Round 3 lambda=1'):
    if name=='Round 3 lambda=1':
        vla.llm_backbone.llm=PeftModel.from_pretrained(vla.llm_backbone.llm,run/'adapter-final',is_trainable=False)
    loader=DataLoader(m.ReviewedQueryDataset(train,vla),batch_size=32,shuffle=False,collate_fn=m.ReviewedQueryCollator(vla.llm_backbone.get_tokenizer()),num_workers=0)
    rows=[];vla.eval()
    with torch.no_grad():
        for idx,batch in enumerate(loader):
            batch=m._to_device(batch,torch.device('cuda:0'),torch.bfloat16)
            with torch.autocast('cuda',dtype=torch.bfloat16):losses=m.compute_losses(vla,batch,cfg)
            assert all(torch.isfinite(v) for v in losses)
            rows.append((len(batch['record_ids']),[float(v.detach()) for v in losses]))
            print(f'{name}: batch {idx+1}/{len(loader)}',flush=True)
    metrics=m._average_metrics(rows)
    metrics['directional_mismatches']=metrics['move_misalignment_count']-zeros
    metrics['directional_queries']=len(train)-zeros
    metrics['directional_mismatch_percent']=100*metrics['directional_mismatches']/metrics['directional_queries']
    metrics['checkpoint']=str(cfg.checkpoint)
    if name=='Round 3 lambda=1':metrics['adapter']=str(run/'adapter-final')
    result['models'][name]=metrics
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(name,json.dumps(metrics),flush=True)
print('COMPLETE',flush=True)
