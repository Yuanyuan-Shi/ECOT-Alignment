import torch
from experiments.robot.libero.refined_move_alignment import refined_objective,episode_split,EarlyStopping,literal_stop

def test_stop_boundary_and_gradient():
    a=torch.zeros(3,10,7);a[0,:,0]=.02;a[1,:,0]=.06;a[2,:,0]=.12;a.requires_grad_()
    loss,d=refined_objective(a,torch.zeros(3,3),torch.ones(3,dtype=torch.bool))
    assert d['mismatch'].tolist()==[False,False,True]
    loss.backward();assert a.grad[2,:,:3].abs().sum()>0
    assert a.grad[:2].abs().sum()==0

def test_direction_unknown_exclusion_and_separate_means():
    a=torch.zeros(3,10,7);a[:,:,0]=.12;a.requires_grad_()
    v=torch.tensor([[1.,0,0],[-1.,0,0],[0,0,0]])
    loss,d=refined_objective(a,v,torch.zeros(3,dtype=torch.bool))
    assert d['eligible'].tolist()==[True,True,False]
    assert d['mismatch'].tolist()==[False,True,False]
    assert abs(loss.item()-.5)<1e-5
    loss.backward();assert a.grad[2].abs().sum()==0

def test_stop_and_direction_semantics():
    assert literal_stop('PLAN: act MOVE: stop GRIPPER POSITION: [1, 2]')
    assert not literal_stop('MOVE: close gripper GRIPPER POSITION: [1, 2]')
    a=torch.zeros(1,10,7);a[:,:5,0]=.2;a[:,5:,0]=-.2
    loss,d=refined_objective(a,torch.zeros(1,3),torch.ones(1,dtype=torch.bool))
    assert loss==0  # Explicit net-displacement criterion, not traveled distance.

def test_episode_split_no_leakage_and_reproducibility():
    rows=[{'episode_id':f'e{i}','episode_success':i%2==0,'reasoning_raw':'MOVE: stop' if i%3==0 else 'MOVE: up','query':j} for i in range(30) for j in range(3)]
    tr,va=episode_split(rows);tr2,va2=episode_split(rows)
    assert (tr,va)==(tr2,va2)
    assert not {r['episode_id'] for r in tr}&{r['episode_id'] for r in va}
    assert len(tr)+len(va)==len(rows)

def test_early_stop_ties_count_and_improvement_resets():
    e=EarlyStopping()
    assert e.update(.2,0)==(True,False)
    for step in [25,50,75]:assert e.update(.2,step)==(False,False)
    assert e.update(.1,100)==(True,False)
    for step in [125,150,175]:assert e.update(.11,step)==(False,False)
    assert e.update(.11,200)==(False,True)
    assert e.best_step==100
