import numpy as np
import pytest
import torch
from experiments.robot.libero.policy_action_comparison import policy_action_cosine
from experiments.robot.libero.move_alignment_comparison import move_reward
from experiments.robot.libero.move_alignment_training import move_cosines


def record():
    actions=np.zeros((10,7));actions[:,0]=-1
    return {'claims':{'motion_axes':{'x':1}},'decoded_action_chunk':actions.tolist(),
            'state_before':{'ee_position':[0,0,0]},'state_after':{'ee_position':[1,0,0]},
            'executed_chunk_length':1}


def test_policy_metric_uses_full_prediction_instead_of_realized_displacement():
    r=record()
    r['decoded_action_chunk'][0][0]=1
    assert policy_action_cosine(r)<-.99
    assert move_reward(r)>.99
    actions=torch.tensor([r['decoded_action_chunk']])
    expected=move_cosines(actions,torch.tensor([[1.,0.,0.]]))
    assert policy_action_cosine(r)==pytest.approx(expected.item(),abs=1e-6)


def test_stop_zero_translation_and_invalid_output():
    r=record();r['claims']['motion_axes']={};assert policy_action_cosine(r)==0
    r=record();r['decoded_action_chunk']=np.zeros((10,7)).tolist();assert policy_action_cosine(r)==0
    r['decoded_action_chunk']=[]
    with pytest.raises(ValueError):policy_action_cosine(r)
