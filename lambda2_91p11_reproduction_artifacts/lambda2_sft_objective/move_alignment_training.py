"""Differentiable Move-alignment utilities for MiniVLA VQ action chunks."""

from __future__ import annotations

import math
from typing import Tuple

from torch.utils.data import Sampler

import torch
import torch.nn.functional as F


def combine_objective_losses(performance_loss, move_loss, performance_weight=1.0, move_weight=0.1):
    """Exclude performance loss entirely from autograd when its weight is zero."""
    if performance_weight < 0 or move_weight < 0 or performance_weight + move_weight == 0:
        raise ValueError("Loss weights must be nonnegative with at least one positive")
    if performance_weight == 0:
        return move_weight * move_loss
    return performance_weight * performance_loss + move_weight * move_loss


def extract_action_code_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_image_patches: int,
    action_token_begin_idx: int,
    action_token_end_idx: int,
    tokenizer_len: int,
    num_codes: int,
    num_groups: int,
) -> torch.Tensor:
    """Return `[batch, groups, codes]` logits aligned with action-token labels."""
    aligned_logits = logits[:, num_image_patches:-1]
    shifted_labels = labels[:, 1:].to(aligned_logits.device)
    action_mask = (shifted_labels > action_token_begin_idx) & (shifted_labels < action_token_end_idx)
    per_example = []
    code_token_ids = tokenizer_len - 1 - torch.arange(num_codes, device=aligned_logits.device)
    for example_logits, example_mask in zip(aligned_logits, action_mask):
        selected = example_logits[example_mask]
        if selected.shape[0] != num_groups:
            raise ValueError(
                f"Expected {num_groups} VQ action tokens per query, found {selected.shape[0]}"
            )
        per_example.append(selected.index_select(-1, code_token_ids))
    return torch.stack(per_example)


def terminal_action_labels(labels, begin_idx, end_idx, num_groups):
    """Restrict alignment extraction to the final contiguous action-token chunk.

    Generated reasoning may itself contain action-like special tokens. Preserve
    the original labels for supervised CE; use this copy only for alignment.
    """
    result=labels.clone()
    for row in result:
        positions=torch.where((row>begin_idx)&(row<end_idx))[0]
        if len(positions)<num_groups:
            raise ValueError('Missing terminal action tokens')
        tail=positions[-num_groups:]
        if not torch.equal(tail,torch.arange(tail[0],tail[0]+num_groups,device=tail.device)):
            raise ValueError('Terminal action tokens are not contiguous')
        row[:tail[0]]=-100
    return result


def differentiable_vq_decode(
    action_code_logits: torch.Tensor,
    vq_vae,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft-decode VQ code logits into a differentiable `[batch, 10, 7]` action chunk."""
    probabilities = F.softmax(action_code_logits.float() / temperature, dim=-1)
    latent = vq_vae.draw_logits_forward(probabilities)
    return vq_vae.get_action_from_latent(latent)


def unnormalize_actions(
    normalized_actions: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply MiniVLA's inference-time q01/q99 action unnormalization differentiably."""
    low = action_low.to(device=normalized_actions.device, dtype=normalized_actions.dtype)
    high = action_high.to(device=normalized_actions.device, dtype=normalized_actions.dtype)
    mask = action_mask.to(device=normalized_actions.device)
    scaled = 0.5 * (normalized_actions + 1.0) * (high - low) + low
    return torch.where(mask, scaled, normalized_actions)


def move_cosines(
    predicted_actions: torch.Tensor,
    reasoning_move_vectors: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute per-query cosine with the existing epsilon and stop convention.

    A zero reasoning vector follows the reviewed MOVE-stop convention and receives reward zero.
    """
    delta_position = predicted_actions[..., :3].sum(dim=1).float()
    reasoning = reasoning_move_vectors.to(device=delta_position.device, dtype=delta_position.dtype)
    numerator = (reasoning * delta_position).sum(dim=-1)
    reasoning_norm = torch.linalg.vector_norm(reasoning, dim=-1)
    action_norm = torch.linalg.vector_norm(delta_position, dim=-1)
    reward = numerator / (reasoning_norm * action_norm + epsilon)
    reward = torch.where(reasoning_norm > 0, reward, torch.zeros_like(reward))
    return reward


def thresholded_move_penalty(cosines: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Normalized squared hinge; retains the stop cosine of zero."""
    return (torch.relu(threshold - cosines) / (1.0 + threshold)).square()


def move_alignment_objective(
    predicted_actions: torch.Tensor,
    reasoning_move_vectors: torch.Tensor,
    *,
    epsilon: float = 1e-8,
    alignment_threshold: float = 0.5,
    loss_type: str = "cosine",
    query_weights: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reward = move_cosines(predicted_actions, reasoning_move_vectors, epsilon=epsilon)
    if loss_type == "thresholded_squared_hinge":
        penalties = thresholded_move_penalty(reward, alignment_threshold)
    elif loss_type == "cosine":
        penalties = 1.0 - reward
    else:
        raise ValueError(f"Unknown Move loss: {loss_type}")
    if query_weights is None:
        loss = penalties.mean()
    else:
        weights = query_weights.to(device=penalties.device, dtype=penalties.dtype)
        if weights.shape != penalties.shape or not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError("Query weights must be finite, positive, and match the batch")
        loss = (weights * penalties).sum() / weights.sum()
    return loss, reward.mean(), (reward >= alignment_threshold).float().mean()


class BalancedOutcomeBatchSampler(Sampler):
    """Full half-success/half-failure microbatches from training indices only.

    Each outcome pool is shuffled and cycled independently. Queries are reused
    after pool exhaustion; the smaller failure pool is consequently oversampled.
    An epoch retains the natural loader's number of effective optimizer batches.
    """
    def __init__(self, outcomes, batch_size, accumulation_steps=1, seed=0):
        if batch_size <= 0 or batch_size % 2 or accumulation_steps <= 0:
            raise ValueError("Balanced sampling requires an even positive batch size")
        self.pools = [[i for i, value in enumerate(outcomes) if bool(value) == outcome]
                      for outcome in (True, False)]
        if not all(self.pools):
            raise ValueError("Both episode outcomes must be present")
        self.batch_size = batch_size
        self.batches = math.ceil(len(outcomes) / (batch_size * accumulation_steps)) * accumulation_steps
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        remaining = [[], []]
        for _ in range(self.batches):
            batch = []
            for outcome, pool in enumerate(self.pools):
                for _ in range(self.batch_size // 2):
                    if not remaining[outcome]:
                        remaining[outcome] = [pool[i] for i in torch.randperm(len(pool), generator=generator).tolist()]
                    batch.append(remaining[outcome].pop())
            order = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[i] for i in order]
