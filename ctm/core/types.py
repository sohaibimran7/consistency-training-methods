"""Core data types shared by the training loops and backends.

Kept dependency-light: no tinker / torch imports. ``Rollout.prompt`` is opaque
(``Any``) — for the Tinker backend it is a ``tinker.types.ModelInput``; any
backend-specific prompt container works as long as the same object round-trips
into the backend's datum construction.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel


@dataclass
class Rollout:
    """A single rollout (sampled response)."""
    tokens: list[int]
    logprobs: list[float]
    text: str
    trait_value: float
    perturbation_idx: int
    parsed_successfully: bool = True
    prompt: Optional[Any] = None


@dataclass
class RolloutResult:
    """Pre-computed result from rollout collection.
    Rates are computed internally so the full rollout data (tokens/logprobs
    for all rollouts) can be freed early. Only gradient rollouts retained."""
    train_rollouts: list[Rollout]           # Training perturbation gradient rollouts
    anchor_rollouts: list[Rollout]          # Reference perturbation gradient rollouts (for anchor)
    rates: dict[int, float | None]          # Trait rate per perturbation index (None if 0 parsed)
    rate_counts: dict[int, int]             # Number of parsed rollouts per perturbation
    n_total: int                            # Total raw rollouts (all perturbations)
    n_parsed: int                           # Parsed rollouts (all perturbations)
    resample_stats: dict = field(default_factory=dict)  # {ref,train}_{want,drawn,gave_up} (resample mode)


@dataclass
class BatchItem:
    """One datapoint's rollout results, ready for reward computation."""
    datapoint_idx: int
    datapoint: dict
    train_rollouts: list[Rollout]
    anchor_rollouts: list[Rollout]
    p_hat: dict[int, float]         # Per-perturbation trait rates (training)
    p_hat_counts: dict[int, int]    # Per-perturbation parsed rollout counts (for gap SE)
    p_ref: float                    # Reference perturbation rate
    p_ref_init: float | None        # Initial (base/anchor) reference rate
    n_total: int                    # Total raw rollouts
    n_parsed: int                   # Parsed rollouts
    n_ref_parsed: int               # Parsed ref rollouts
    n_training_parsed: int          # Parsed training rollouts
    n_ref_init_parsed: int | None = None  # Parsed rollouts behind p_ref_init (for anchor-gap SE)
    resample_stats: dict = field(default_factory=dict)  # {ref,train}_{want,drawn,gave_up} (resample mode)


class RolloutRecord(BaseModel):
    """One persisted rollout, written by ``ctm.training.rollout_log.RolloutLogger``.

    Captures not just the sampled text but the training signal it received
    (trait, reward, advantage) and the group context (p_hat / p_ref) so a rollout
    can be inspected later without replaying the run. Schema is shared with the
    reader in ``ctm.evals.analysis.rollouts`` — change it only here.
    """
    step: int                        # global training step (1-based, as logged)
    epoch: int
    datapoint_idx: int
    perturbation_idx: int
    role: Literal["train", "anchor"]  # consistency (cued) vs anchor (reference) gradient rollout
    prompt_text: str
    completion_text: str
    trait_value: float
    reward: float
    advantage: float
    p_hat: Optional[float] = None    # this rollout's perturbation rate (train role)
    p_ref: float
    p_ref_init: Optional[float] = None
