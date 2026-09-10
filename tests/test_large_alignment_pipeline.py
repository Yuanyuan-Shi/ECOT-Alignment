"""Regression tests for new dataset denominators and accumulated objectives."""
import ast,importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('large_data',ROOT/'runs/large-data-alignment/data.py');data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)

def test_stop_direction_partition_and_coverage():
    def row(reason,action,eid,success):
        a=torch.zeros(10,7);a[:,0]=action
        return dict(reasoning_raw='MOVE: '+reason,decoded_action_chunk=a.tolist(),episode_id=eid,episode_success=success,policy_query_index=0)
    rows=data.score([row('move forward',.1,'s1',True),row('stop',.1,'f1',False),row('close gripper',.1,'f2',False),row('stop',0.,'s2',True)])
    stats=data.statistics(rows,bootstrap=False)
    assert stats['Overall mismatch']['overall']['queries']==3
    assert stats['Stop mismatch']['overall']['queries']==2
    assert stats['Stop mismatch']['overall']['mismatches']==1
    assert stats['Directional mismatch']['overall']['queries']==1
    assert stats['Partial directional mismatch']['overall']['mismatches']+stats['Nonpositive directional mismatch']['overall']['mismatches']==stats['Directional mismatch']['overall']['mismatches']

def test_accumulated_gradient_matches_pooled_objective():
    # Load only the helper; no VLA/network/GPU setup is needed.
    tree=ast.parse((ROOT/'vla-scripts/finetune_large_alignment.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='backward_effective_batch')
    w=torch.tensor(2.,requires_grad=True)
    def forward(vla,batch,cfg):
        tokens=(batch['labels'][:,1:]!=-100).sum(1).float();perf=(w*batch['targets']*tokens).sum()/tokens.sum()
        eligible=batch['reasoning_move_vector'].norm(dim=-1)>0
        align=(w*batch['targets'][eligible]).mean() if eligible.any() else w*0
        return perf+align,perf,align,{'eligible':eligible}
    env={'torch':torch,'forward':forward};exec(compile(ast.Module(body=[fn],type_ignores=[]),'helper','exec'),env)
    batch=dict(record_ids=list(range(64)),reasoning_move_vector=torch.zeros(64,3),literal_stop=torch.zeros(64,dtype=torch.bool),labels=torch.full((64,4),-100),targets=torch.arange(1.,65.))
    batch['reasoning_move_vector'][[0,32,33,34],0]=1
    batch['labels'][:,1]=0;batch['labels'][32:,2:]=0
    cfg=SimpleNamespace(performance_loss_weight=1.,move_loss_weight=1.)
    loss,*_=env['backward_effective_batch'](None,batch,cfg)
    tokens=(batch['labels'][:,1:]!=-100).sum(1).float();expected=(batch['targets']*tokens).sum()/tokens.sum()+batch['targets'][[0,32,33,34]].mean()
    assert torch.allclose(w.grad,expected)
    assert abs(loss-float(2*expected))<1e-5


def test_alignment_ignores_action_tokens_inside_reasoning():
    from experiments.robot.libero.move_alignment_training import terminal_action_labels
    labels=torch.tensor([[-100,101,10,11,102,103,104,105,106,107,108,12]])
    original=labels.clone()
    selected=terminal_action_labels(labels,100,110,7)
    assert torch.equal(labels,original)
    assert selected[0,:4].eq(-100).all()
    assert selected[0,4:11].tolist()==[102,103,104,105,106,107,108]
