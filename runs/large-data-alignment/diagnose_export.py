from pathlib import Path
p=Path('/home/exx/Projects/ECOT-Alignment/vla-scripts/finetune_large_alignment.py')
s=p.read_text();start=s.index('    with torch.random.fork_rng(devices=[0]):\n        final_train=')
s=s[:start]+'''
    sample=m._to_device(collator([dataset[0]]),torch.device('cuda:0'),torch.float32);vla.eval();vla.float()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    print('TRAINING MODULES',[(n,type(mod).__name__) for n,mod in vla.named_modules() if mod.training],flush=True)
    captured={};outputs=[]
    def hook(name):
        def record(mod,args,out):
            if isinstance(out,tuple):out=out[0]
            if isinstance(out,torch.Tensor):captured[name]=out.detach().float().cpu()
        return record
    handles=[vla.vision_backbone.register_forward_hook(hook('vision')),vla.projector.register_forward_hook(hook('projector')),vla.llm_backbone.llm.base_model.model.model.layers[0].register_forward_hook(hook('layer0'))]
    from torch.nn.attention import sdpa_kernel,SDPBackend
    for i in range(2):
        rng=torch.cuda.get_rng_state().clone()
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):out=vla(input_ids=sample['input_ids'],attention_mask=sample['attention_mask'],pixel_values=sample['pixel_values']).logits.float().cpu()
        captured['logits']=out
        outputs.append(dict(captured));print('PASS',i,'rng_changed',not torch.equal(rng,torch.cuda.get_rng_state()),flush=True)
    for name in outputs[0]:print('REPEAT ERROR',name,float((outputs[0][name]-outputs[1][name]).abs().max()),flush=True)
    for h in handles:h.remove()
    vla.llm_backbone.llm=vla.llm_backbone.llm.merge_and_unload();vla.eval()
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):merged=vla(input_ids=sample['input_ids'],attention_mask=sample['attention_mask'],pixel_values=sample['pixel_values']).logits.float().cpu()
    error=float((merged-outputs[0]['logits']).abs().max());print('MERGE ERROR',error,flush=True)
    assert torch.allclose(merged,outputs[0]['logits'],atol=.003,rtol=.003)
    run.finish()
if __name__=='__main__':main()
'''
exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
