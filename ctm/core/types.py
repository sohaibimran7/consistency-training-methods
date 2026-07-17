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
    trait_value: float | None
    perturbation_idx: int
    parsed_successfully: bool = True
    answer_parsed: bool = True
    has_logprobs: bool = True
    grader_evaluated: bool = True
    grader_failed: bool = False
    prompt: Optional[Any] = None


@dataclass
class RolloutResult:
    """Pre-computed result from rollout collection."""

    train_rollouts: list[Rollout]  # Training perturbation gradient rollouts
    anchor_rollouts: list[Rollout]  # Reference perturbation gradient rollouts (for anchor)
    sampled_rollouts: list[Rollout]  # Every response sampled, including rate-only/unusable attempts
    rates: dict[int, float | None]  # Trait rate per perturbation index (None if 0 parsed)
    rate_counts: dict[int, int]  # Number of parsed rollouts per perturbation
    n_total: int  # Total raw rollouts (all perturbations)
    n_parsed: int  # Parsed rollouts (all perturbations)
    n_trait_abstained: int = 0  # Classifier failures dropped from rates/gradients
    resample_stats: dict = field(default_factory=dict)  # {ref,train}_{want,drawn,gave_up} (resample mode)


@dataclass
class BatchItem:
    """One datapoint's rollout results, ready for reward computation."""

    datapoint_idx: int
    datapoint: dict
    train_rollouts: list[Rollout]
    anchor_rollouts: list[Rollout]
    sampled_rollouts: list[Rollout]
    initial_rollouts: list[Rollout]
    p_hat: dict[int, float]  # Per-perturbation trait rates (training)
    p_hat_counts: dict[int, int]  # Per-perturbation parsed rollout counts (for gap SE)
    p_ref: float | None  # Configured aggregate of the current reference rates
    p_ref_init: float | None  # Configured aggregate of the initial reference rates (logging/fallback)
    reference_rates: dict[int, float]  # Current rate for each reference perturbation
    reference_rate_counts: dict[int, int]  # Parsed rollouts behind each current reference rate
    initial_reference_rates: dict[int, float]  # Initial/base rate for each reference perturbation
    initial_reference_rate_counts: dict[int, int]  # Parsed rollouts behind each initial reference rate
    n_total: int  # Total raw rollouts
    n_parsed: int  # Parsed rollouts
    n_ref_parsed: int  # Parsed ref rollouts
    n_training_parsed: int  # Parsed training rollouts
    n_trait_abstained: int = 0  # Classifier failures dropped from rates/gradients
    resample_stats: dict = field(default_factory=dict)  # {ref,train}_{want,drawn,gave_up} (resample mode)


class RolloutRecord(BaseModel):
    """One persisted rollout, written by ``ctm.training.rollout_log.RolloutLogger``.

    Captures sampled text and whether it reached training. Reward/advantage are
    present only for selected candidates; skipped samples carry ``skip_reason``.
    Schema is shared with ``ctm.evals.analysis.rollouts``.
    """

    step: int  # global training step (1-based, as logged)
    epoch: int
    datapoint_idx: int
    perturbation_idx: int
    role: Literal["train", "anchor", "rate", "initial_reference"]
    sample_source: Literal["policy", "anchor_model"]
    prompt_text: str
    completion_text: str
    trait_value: Optional[float]
    parsed_successfully: bool
    grader_failed: bool
    reward: Optional[float]
    advantage: Optional[float]
    skipped_from_training: bool
    skip_reason: Optional[str] = None
    p_hat: Optional[float] = None  # this rollout's perturbation rate (train role)
    p_ref: Optional[float] = None
    p_ref_init: Optional[float] = None
