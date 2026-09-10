import torch

from experiments.robot.libero.move_alignment_training import move_alignment_objective


def test_move_only_excludes_performance_gradient_and_preserves_move_gradient():
    from experiments.robot.libero.move_alignment_training import combine_objective_losses
    performance_parameter = torch.tensor(2., requires_grad=True)
    move_parameter = torch.tensor(3., requires_grad=True)
    loss = combine_objective_losses(performance_parameter.square(), move_parameter.square(), 0., 1.)
    loss.backward()
    assert performance_parameter.grad is None
    assert move_parameter.grad == 6
    assert loss == 9


def test_default_combined_objective_is_unchanged():
    from experiments.robot.libero.move_alignment_training import combine_objective_losses
    assert torch.allclose(combine_objective_losses(torch.tensor(2.), torch.tensor(3.)), torch.tensor(2.3))


def test_move_alignment_reward_and_loss_for_aligned_motion():
    actions = torch.zeros(1, 10, 7)
    actions[:, :, 2] = 0.1
    reasoning = torch.tensor([[0.0, 0.0, 1.0]])
    loss, reward, rate = move_alignment_objective(actions, reasoning)
    assert torch.allclose(reward, torch.tensor(1.0))
    assert torch.allclose(loss, torch.tensor(0.0))
    assert torch.allclose(rate, torch.tensor(1.0))


def test_move_alignment_stop_reward_is_zero():
    actions = torch.randn(2, 10, 7)
    reasoning = torch.zeros(2, 3)
    loss, reward, rate = move_alignment_objective(actions, reasoning)
    assert torch.allclose(reward, torch.tensor(0.0))
    assert torch.allclose(loss, torch.tensor(1.0))
    assert torch.allclose(rate, torch.tensor(0.0))


def test_move_alignment_loss_backpropagates_to_predicted_actions():
    actions = torch.zeros(1, 10, 7, requires_grad=True)
    with torch.no_grad():
        actions[:, :, 0] = -0.05
        actions[:, :, 2] = 0.02
    reasoning = torch.tensor([[0.0, 0.0, 1.0]])
    loss, _, _ = move_alignment_objective(actions, reasoning)
    loss.backward()
    assert actions.grad is not None
    assert actions.grad[..., :3].abs().sum() > 0


def test_thresholded_hinge_boundary_values_and_gradients():
    from experiments.robot.libero.move_alignment_training import thresholded_move_penalty
    c = torch.tensor([-1., -0.5, 0., 0.49, 0.5, 0.75, 1.], requires_grad=True)
    penalties = thresholded_move_penalty(c)
    assert torch.all(penalties[:4] > 0)
    assert torch.equal(penalties[4:], torch.zeros(3))
    assert torch.all(penalties[:3][:-1] > penalties[:3][1:])
    assert penalties[0] == 1
    penalties.mean().backward()
    assert torch.isfinite(c.grad).all()
    assert torch.all(c.grad[:4] < 0)
    assert torch.equal(c.grad[4:], torch.zeros(3))


def test_thresholded_hinge_action_gradients_and_stop():
    actions = torch.zeros(4, 10, 7)
    actions[:, :, :3] = torch.tensor([[1., 1., 0.], [-1., 1., 0.], [0., 0., 0.], [1., 0., 0.]]).unsqueeze(1)
    actions.requires_grad_()
    reasoning = torch.tensor([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [0., 0., 0.]])
    loss, _, _ = move_alignment_objective(actions, reasoning, loss_type="thresholded_squared_hinge")
    loss.backward()
    assert torch.isfinite(actions.grad).all()
    assert actions.grad[0].abs().sum() == 0
    assert actions.grad[1].abs().sum() > 0
    assert actions.grad[3].abs().sum() == 0
    stop_loss, reward, _ = move_alignment_objective(actions[3:], reasoning[3:], loss_type="thresholded_squared_hinge")
    assert reward == 0
    assert torch.allclose(stop_loss, torch.tensor(1 / 9))
