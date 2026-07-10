"""Torch implementations of the three loss functions the loops use.

Semantics match the Tinker service contract the loops were written against:
``forward_backward`` ACCUMULATES gradients; ``optim_step`` applies + zeroes them.
Each function takes per-datum tensors (already gathered target logprobs under the
current policy) and returns a scalar batch loss.

Aggregation is token-weighted mean over the batch (loss normalized by total
weight/mask). This makes gradient magnitude invariant to batch size and sequence
length; absolute LR parity with Tinker's server-side reduction is not guaranteed
— treat LRs as backend-specific (they already are model-specific).
"""

from typing import Sequence

import torch

PPO_CLIP_EPSILON = 0.2  # standard PPO clip; Tinker's server-side epsilon is not published


def cross_entropy_loss(target_logprobs: Sequence[torch.Tensor],
                       weights: Sequence[torch.Tensor]) -> torch.Tensor:
    """Weighted NLL: -(Σ w·logp) / Σw over the batch."""
    num = sum((w * lp).sum() for lp, w in zip(target_logprobs, weights))
    den = sum(w.sum() for w in weights)
    return -num / den.clamp(min=1e-8)


def importance_sampling_loss(target_logprobs: Sequence[torch.Tensor],
                             sampled_logprobs: Sequence[torch.Tensor],
                             advantages: Sequence[torch.Tensor],
                             masks: Sequence[torch.Tensor]) -> torch.Tensor:
    """-E[ratio · A] with ratio = exp(logp_policy − logp_sampled), over masked tokens."""
    num = 0.0
    den = 0.0
    for lp, slp, adv, mask in zip(target_logprobs, sampled_logprobs, advantages, masks):
        ratio = torch.exp(lp - slp)
        num = num + (mask * ratio * adv).sum()
        den = den + mask.sum()
    return -num / torch.as_tensor(den).clamp(min=1e-8)


def ppo_loss(target_logprobs: Sequence[torch.Tensor],
             sampled_logprobs: Sequence[torch.Tensor],
             advantages: Sequence[torch.Tensor],
             masks: Sequence[torch.Tensor],
             clip_epsilon: float = PPO_CLIP_EPSILON) -> torch.Tensor:
    """Clipped PPO surrogate: -E[min(ratio·A, clip(ratio, 1±ε)·A)] over masked tokens."""
    num = 0.0
    den = 0.0
    for lp, slp, adv, mask in zip(target_logprobs, sampled_logprobs, advantages, masks):
        ratio = torch.exp(lp - slp)
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = torch.minimum(ratio * adv, clipped * adv)
        num = num + (mask * surrogate).sum()
        den = den + mask.sum()
    return -num / torch.as_tensor(den).clamp(min=1e-8)
