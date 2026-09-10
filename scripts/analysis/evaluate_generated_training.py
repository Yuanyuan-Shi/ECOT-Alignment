"""Inference-only replay of training images with the exact rollout generation path."""
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis_results/move_only_500/generated_training'
NAMES={'pretrained':'Original MiniVLA','lambda1':'Round 3 lambda=1','moveonly':'Mismatch only'}
CHECKPOINTS={
'pretrained':'artifacts/ecot-libero90/checkpoints/step-100000-epoch-22-loss=0.0261.pt',
'lambda1':'runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambda10-seed20261102/checkpoints/step-000200-move-lora.pt',
'moveonly':'runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambdamoveonly500-seed20261102/checkpoints/step-000500-move-lora.pt'}

def records():
    src=ROOT/'experiments/robot/libero/results/second-round-pretrained-libero90-seed20261101'
    rows=[json.loads(l) for l in (src/'alignment_queries.jsonl').read_text().splitlines()]
    manifest=json.loads((ROOT/'runs/ecot-move-alignment/round3-ecot-lite-libero90-hinge-lora-r16-lambda10-seed20261102/split_manifest.json').read_text())
    byid={f"task{r['task_id']}-query{r['policy_query_index']}":r for r in rows}
    selected=[]
    for rid in manifest['train_ids']:
        r=byid[rid];r['record_id']=rid;r['frame_path']=str(src/r['frames']['before']);assert Path(r['frame_path']).is_file();selected.append(r)
    assert len(selected)==1245
    return selected

def worker(model,rank,world):
    import numpy as np,torch
    from PIL import Image
    from experiments.robot.openvla_utils import get_prismatic_vla,get_prismatic_vla_action
    from experiments.robot.robot_utils import set_seed_everywhere
    from experiments.robot.libero.alignment_evaluator import parse_reasoning,_split_fields
    from experiments.robot.libero.run_libero_eval import _extract_explicit_reasoning
    torch.set_num_threads(1);set_seed_everywhere(20261102)
    OUT.mkdir(exist_ok=True,parents=True)
    path=OUT/f'{model}-rank{rank}.jsonl'
    done={r['record_id'] for r in [json.loads(l) for l in path.read_text().splitlines()]} if path.exists() else set()
    assigned=records()[rank::world]
    if all(r['record_id'] in done for r in assigned):return
    cfg=SimpleNamespace(model_family='prismatic',hf_token=ROOT/'.hf_token',pretrained_checkpoint=str(ROOT/CHECKPOINTS[model]))
    vla=get_prismatic_vla(cfg);vla.eval();vla.requires_grad_(False)
    assert vla.use_cot and not vla.use_async and vla.max_freezing_time==0
    with path.open('a',buffering=1) as stream,torch.inference_mode():
        for index,r in enumerate(assigned):
            if r['record_id'] in done:continue
            image=np.array(Image.open(r['frame_path']).convert('RGB'));assert image.shape==(224,224,3)
            info={};start=time.time()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                actions,decoded=get_prismatic_vla_action(vla,None,cfg.pretrained_checkpoint,{'full_image':image},r['task_instruction'],'libero_lm_90',center_crop=False,info_dict=info)
            a=np.asarray(actions,dtype=float).reshape(10,7);assert np.isfinite(a).all()
            # Match rollout extraction; record malformed generations explicitly rather than silently losing rows.
            try:reasoning=_extract_explicit_reasoning(decoded)
            except RuntimeError:reasoning=decoded
            claims=parse_reasoning(reasoning);vec=np.array([claims.motion_axes.get(k,0) for k in ('x','y','z')],dtype=float)
            fields=_split_fields(reasoning).get('MOVE',[])
            move=fields[-1].rsplit('→',1)[-1].strip().lower().rstrip('.;') if fields else ''
            directional=bool(np.linalg.norm(vec)>0);stop=move=='stop'
            delta=a[:,:3].sum(0);c=float(np.dot(vec,delta)/(np.linalg.norm(vec)*np.linalg.norm(delta)+1e-8))
            net=.05*np.clip(a[:,:3],-1,1).sum(0)
            mismatch=bool(c<.5) if directional else bool(np.linalg.norm(net)>.03) if stop else None
            row={'record_id':r['record_id'],'model':model,'task_id':r['task_id'],'source_episode_id':r['episode_id'],
                 'source_episode_success':bool(r['episode_success']),'frame_path':r['frame_path'],'reasoning_raw':reasoning,
                 'move':move,'reasoning_vector':vec.tolist(),'directional':directional,'stop':stop,'included':directional or stop,
                 'cosine':c,'commanded_net_m':net.tolist(),'mismatch':mismatch,'action_tokens':info['action_token_ids'],
                 'decoded_action_chunk':a.tolist(),'parse_warnings':claims.parse_warnings,'seconds':time.time()-start}
            assert len(info['action_token_ids'])==7
            stream.write(json.dumps(row)+'\n')
            if (index+1)%10==0:print(model,rank,index+1,'/',len(assigned),flush=True)
    print('COMPLETE',model,rank,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=NAMES,required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--world',type=int,default=2);a=p.parse_args();worker(a.model,a.rank,a.world)
