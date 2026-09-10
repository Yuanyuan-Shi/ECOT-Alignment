import pytest
import torch
from experiments.robot.libero.move_alignment_training import (
    BalancedOutcomeBatchSampler, move_alignment_objective, thresholded_move_penalty, move_cosines,
)


def test_weighted_hinge_exact_normalization_and_gradient():
    actions = torch.tensor([[[1., 1., 0., 0., 0., 0., 0.]],
                            [[-1., 1., 0., 0., 0., 0., 0.]],
                            [[0., 1., 0., 0., 0., 0., 0.]]], requires_grad=True)
    reasoning = torch.tensor([[1., 0., 0.]] * 3)
    weights = torch.tensor([1., 3., 1.])
    loss, cosine, rate = move_alignment_objective(actions, reasoning,
                     loss_type='thresholded_squared_hinge', query_weights=weights)
    expected = (thresholded_move_penalty(move_cosines(actions, reasoning)) * weights).sum() / 5
    assert torch.allclose(loss, expected)
    natural, natural_cosine, natural_rate = move_alignment_objective(actions, reasoning,
                     loss_type='thresholded_squared_hinge')
    assert loss > natural
    assert cosine == natural_cosine and rate == natural_rate
    loss.backward()
    assert torch.isfinite(actions.grad).all()
    assert actions.grad[0].abs().sum() == 0
    assert actions.grad[1].abs().sum() > 0


def test_unit_weights_match_original_hinge():
    a = torch.randn(32, 10, 7)
    r = torch.randn(32, 3)
    first = move_alignment_objective(a,r,loss_type='thresholded_squared_hinge')
    second = move_alignment_objective(a,r,loss_type='thresholded_squared_hinge',query_weights=torch.ones(32))
    assert all(torch.allclose(x,y) for x,y in zip(first,second))


@pytest.mark.parametrize('size,accum', [(32,1),(8,4)])
def test_balanced_sampler_full_batches_replacement_and_seed(size, accum):
    outcomes = [True]*925 + [False]*320
    sampler = BalancedOutcomeBatchSampler(outcomes,size,accum,20261102)
    batches = list(sampler)
    assert len(batches) == 39*accum
    assert batches == list(BalancedOutcomeBatchSampler(outcomes,size,accum,20261102))
    assert batches != list(sampler)
    for batch in batches:
        assert len(batch)==size
        assert sum(outcomes[i] for i in batch)==size//2
        assert all(0<=i<len(outcomes) for i in batch)
    failed = [i for b in batches for i in b if not outcomes[i]]
    assert len(failed)==624 and len(set(failed))==320


def test_balanced_sampler_rejects_invalid_pools_and_size():
    with pytest.raises(ValueError): BalancedOutcomeBatchSampler([True]*10,32)
    with pytest.raises(ValueError): BalancedOutcomeBatchSampler([True,False],7)


def test_performance_loss_unchanged_when_only_move_is_weighted():
    # Execute the production loss function with a small stand-in VLA, avoiding
    # heavyweight model imports. This checks the actual total-loss composition.
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    source = ast.parse(Path('vla-scripts/finetune_move_alignment.py').read_text())
    fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name=='compute_losses')
    fn.returns = None
    for arg in fn.args.args: arg.annotation = None
    actions = torch.tensor([[[1.,1.,0.,0.,0.,0.,0.]], [[-1.,1.,0.,0.,0.,0.,0.]]],requires_grad=True)
    performance = torch.tensor(.4,requires_grad=True)
    ns = {'torch':torch, 'move_alignment_objective':move_alignment_objective, 'move_cosines':move_cosines,
          'extract_action_code_logits':lambda *a,**k: actions,
          'differentiable_vq_decode':lambda *a,**k: actions,
          '_action_stats':lambda *a: (None,None,None),
          'unnormalize_actions':lambda a,*args:a}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<loss>','exec'),ns)
    class VLA:
        vision_backbone=SimpleNamespace(num_patches=1)
        action_tokenizer=SimpleNamespace(action_token_begin_idx=0,action_token_end_idx=3,
            tokenizer_len=4,n_bins=2,vq_vae=SimpleNamespace(vqvae_groups=1))
        def __call__(self,**kwargs): return SimpleNamespace(logits=actions,loss=performance)
    batch={k:None for k in ('input_ids','attention_mask','pixel_values','labels')}
    batch.update(reasoning_move_vector=torch.tensor([[1.,0.,0.],[1.,0.,0.]]),episode_success=torch.tensor([True,False]))
    cfg=SimpleNamespace(move_temperature=1,move_epsilon=1e-8,move_alignment_threshold=.5,
         move_loss_type='thresholded_squared_hinge',failure_move_weight=3,move_loss_weight=.5)
    weighted=ns['compute_losses'](VLA(),batch,cfg)
    cfg.failure_move_weight=1
    natural=ns['compute_losses'](VLA(),batch,cfg)
    assert weighted[1] is performance and natural[1] is performance
    assert weighted[2] > natural[2]
    assert torch.allclose(weighted[0], performance+.5*weighted[2])
    weighted[0].backward()
    assert performance.grad == 1
