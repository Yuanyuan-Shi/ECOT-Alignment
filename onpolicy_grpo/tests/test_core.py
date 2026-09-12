import json
from pathlib import Path

import numpy as np
import torch

from onpolicy_grpo.common import (
    alignment,
    clipped_grpo_terms,
    deterministic_resume_action,
    episode_reward,
    generated_token_mask,
    group_advantages,
    make_iteration_manifest,
    sampled_kl,
    validate_manifest,
)
from onpolicy_grpo.policy import generation_kwargs
from onpolicy_grpo.cluster_train import prepare_iteration


CFG = {
    "seed": 7,
    "task_ids": [4, 17, 23, 27, 28, 41, 65, 70, 72, 86],
    "group_size": 8,
    "training_state_min": 0,
    "training_state_max_exclusive": 30,
    "action_horizon": 10,
    "action_step_m": 0.05,
    "tau": 0.5,
    "delta_stop_m": 0.03,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
}


def test_directional_and_stop_alignment_costs():
    forward = np.zeros((10, 7)); forward[:, 0] = -0.1
    aligned = alignment("MOVE: move forward", forward, CFG)
    opposed = alignment("MOVE: move backward", forward, CFG)
    assert aligned["directional"] and np.isclose(aligned["cost"], 0)
    assert np.isclose(opposed["cost"], 1)
    stop = alignment("MOVE: stop", forward, CFG)
    rho = np.linalg.norm([0.05, 0, 0])
    assert stop["stop"] and np.isclose(stop["cost"], rho**2 / (rho**2 + 0.03**2))


def test_episode_reward_bounds_and_group_statistics():
    assert np.isclose(episode_reward(True, [0, 1], .8), .6)
    assert -.8 <= episode_reward(False, [1], .8) <= 1
    result = group_advantages([0., 1., 2., 3.], 1e-8, 1e-8)
    assert np.isclose(result["reward_mean"], 1.5)
    assert np.isclose(result["reward_std"], np.std([0., 1., 2., 3.]))
    assert abs(np.mean(result["advantages"])) < 1e-7
    assert np.isclose(np.std(result["advantages"]), 1, atol=1e-7)
    zero = group_advantages([.25] * 8, 1e-8, 1e-8)
    assert zero["zero_variance"] and zero["advantages"] == [0.] * 8


def test_ratio_clipping_sampled_kl_and_kl_sign():
    current = torch.tensor([np.log(2.), np.log(.5)], requires_grad=True)
    old = torch.zeros(2)
    reference = torch.zeros(2)
    terms = clipped_grpo_terms(current, old, reference, 1., .2, .04)
    assert torch.allclose(terms["clipped_ratio"], torch.tensor([1.2, .8], dtype=current.dtype))
    delta = torch.tensor([-1., 0., 1.])
    assert torch.allclose(sampled_kl(delta), torch.exp(delta) - delta - 1)
    no_kl = clipped_grpo_terms(current, old, current.detach(), 1., .2, 0.)["objective"]
    with_kl = clipped_grpo_terms(current, old, reference, 1., .2, .04)["objective"]
    assert torch.all(with_kl <= no_kl + 1e-8)


def test_generated_token_mask():
    ids = torch.tensor([10, 2, 11, 0, 12])
    assert generated_token_mask(ids, {0, 2}).tolist() == [True, False, True, False, True]
    assert not generated_token_mask(ids, {0, 2}, valid_action=False).any()


def test_null_top_k_explicitly_disables_hf_filtering():
    config = CFG | {"max_new_tokens": 1024}
    kwargs = generation_kwargs(config)
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == kwargs["top_p"] == 1.0
    assert kwargs["top_k"] == 0


def test_task_group_separation_and_sampling_manifest():
    manifest = make_iteration_manifest(CFG, 1)
    validate_manifest(manifest, CFG["task_ids"], 8)
    assert len(manifest["groups"]) == 10
    assert sum(len(group["episodes"]) for group in manifest["groups"]) == 80
    assert all(len({episode["task_id"] for episode in group["episodes"]}) == 1 for group in manifest["groups"])
    assert all(len({episode["initial_state"] for episode in group["episodes"]}) > 1 for group in manifest["groups"])
    assert manifest["sampling"] == {"do_sample": True, "temperature": 1., "top_p": 1., "top_k": None}


def test_frozen_roles_no_critic_and_finite_policy_gradient():
    policy = torch.nn.Linear(2, 1, bias=False)
    rollout = torch.nn.Linear(2, 1, bias=False).requires_grad_(False)
    reference = torch.nn.Linear(2, 1, bias=False).requires_grad_(False)
    old_rollout = rollout.weight.detach().clone(); old_reference = reference.weight.detach().clone()
    current = policy(torch.ones(1, 2)).flatten()
    terms = clipped_grpo_terms(current, torch.zeros_like(current), torch.zeros_like(current), 1., .2, .04)
    (-terms["objective"].mean()).backward()
    assert policy.weight.grad is not None and torch.isfinite(policy.weight.grad).all()
    assert rollout.weight.grad is None and reference.weight.grad is None
    assert torch.equal(old_rollout, rollout.weight) and torch.equal(old_reference, reference.weight)
    assert all("value" not in name.lower() and "critic" not in name.lower() for name, _ in policy.named_parameters())


def test_deterministic_iteration_boundary_resume(tmp_path: Path):
    assert deterministic_resume_action(tmp_path, 1) == "refresh"
    (tmp_path / "iteration-0001.rollout-refreshed.json").write_text("{}")
    assert deterministic_resume_action(tmp_path, 1) == "resume_after_refresh"
    (tmp_path / "iteration-0001.complete.json").write_text("{}")
    assert deterministic_resume_action(tmp_path, 1) == "complete"


def test_cluster_iteration_manifests_are_distinct_and_frozen(tmp_path: Path):
    config = CFG | {
        "iterations": 10,
        "rollout_workers": 8,
        "base_checkpoint": "/unused/original.pt",
        "run_dir": str(tmp_path),
    }
    first = prepare_iteration(tmp_path, config, 1)
    second = prepare_iteration(tmp_path, config, 2)
    manifest_1 = json.loads((first / "iteration-0001.manifest.json").read_text())
    manifest_2 = json.loads((second / "iteration-0001.manifest.json").read_text())
    assert manifest_1["iteration"] == 1 and manifest_2["iteration"] == 2
    assert manifest_1["task_order"] != manifest_2["task_order"]
    assert len(manifest_1["groups"]) == len(manifest_2["groups"]) == 10
    assert all(len(group["episodes"]) == 8 for group in manifest_2["groups"])
    assert prepare_iteration(tmp_path, config, 2) == second
