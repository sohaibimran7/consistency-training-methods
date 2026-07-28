"""Backend protocols — the compute seam between the training loops and hardware.

The interface deliberately mirrors the four Tinker primitives the loops already
speak (sample / forward_backward / optim_step / save-load), because that shape is
what makes the RLCT loop portable: everything scientifically interesting happens
in pure code around these calls.

Two-phase submit/await (``submit_* → Pending*.result()``) is part of the contract:
the RL loop pipelines an optim_step submission behind an in-flight forward_backward
and overlaps rollout prefetch with training. In-process backends may resolve
eagerly and return an already-completed pending object.

Datums: both backends consume ``tinker.Datum`` as the batch container (the tinker
package builds them offline; ``tinker_cookbook``'s ``trajectory_to_data`` /
``datum_from_model_input_weights`` do the careful token shifting once, for
everyone). A local backend translates datums to tensors internally.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, Sequence, runtime_checkable

import torch

from ctm.core.config import AdamConfig, LoRAConfig


@dataclass
class SampledSequence:
    """One sampled completion: tokens plus (optionally) their sampling logprobs."""

    tokens: list[int]
    logprobs: Optional[list[float]]


@dataclass
class ForwardBackwardOutput:
    """Normalized result of a forward/backward pass.

    ``logprobs``: per-datum tensor of per-token target logprobs under the current
    policy (used for the sample-vs-train KL diagnostic and SFT NLL).
    """

    logprobs: list[torch.Tensor]
    metrics: dict[str, float]


class PendingForwardBackward(Protocol):
    async def result(self) -> ForwardBackwardOutput: ...


class PendingOptimStep(Protocol):
    async def result(self) -> None: ...


@runtime_checkable
class SamplerHandle(Protocol):
    """A handle that can sample completions for a rendered prompt.

    ``prompt`` is the renderer's prompt container (``tinker.types.ModelInput``
    for both current backends); ``stop`` is whatever the renderer's
    ``get_stop_sequences()`` returns.
    """

    async def sample(
        self,
        prompt: Any,
        *,
        max_tokens: int,
        temperature: float,
        stop: Any,
        num_samples: int,
    ) -> list[SampledSequence]: ...


@runtime_checkable
class TrainingBackend(Protocol):
    """Everything a training loop needs from the compute substrate."""

    renderer_source: Literal["tinker", "hf"]
    """Authority used to render/tokenize model chat inputs."""

    policy_samplers_are_snapshots: bool
    """Whether an existing policy sampler stays frozen after optimizer updates."""

    def setup(
        self,
        *,
        model: str,
        lora: LoRAConfig,
        resume_from: Optional[str] = None,
        resume_with_optimizer: bool = False,
    ) -> None:
        """Create/initialize the trainable model (and load a checkpoint if resuming)."""
        ...

    def policy_sampler(self, name: str) -> SamplerHandle:
        """Sampler for the CURRENT policy weights (sync; used at setup)."""
        ...

    async def refresh_policy_sampler(self, name: str) -> SamplerHandle:
        """Publish current weights and return a fresh policy sampler (in-loop refresh)."""
        ...

    def base_sampler(self) -> SamplerHandle:
        """Sampler for the frozen base model (anchor rates, distillation, KL reference)."""
        ...

    async def submit_forward_backward(self, datums: Sequence[Any], loss_fn: str) -> PendingForwardBackward:
        """Accumulate gradients for ``datums`` under ``loss_fn``
        ("cross_entropy" | "ppo" | "importance_sampling")."""
        ...

    async def submit_optim_step(self, *, learning_rate: float, adam: AdamConfig) -> PendingOptimStep:
        """Apply accumulated gradients with Adam(learning_rate, adam.*) and zero them."""
        ...

    async def incorporate_kl_penalty(
        self, datums: Sequence[Any], *, kl_coef: float, kl_discount_factor: float
    ) -> dict[str, float]:
        """Mutate ``datums``' advantages in place with a KL-to-base penalty; return metrics."""
        ...

    async def score_reference_completions(
        self,
        reference_prompts: Sequence[Any],
        completion_tokens: Sequence[Sequence[int]],
    ) -> list[list[float]]:
        """Score completions under the frozen base model on paired reference prompts.

        Each returned list contains one log-probability per completion token.  Unlike
        ``incorporate_kl_penalty``, the reference prompt may differ from the prompt
        that produced the completion; OPCT relies on this cross-prompt distinction.
        """
        ...

    async def save_checkpoint(self, *, name: str, log_dir: str | Path, loop_state: dict, kind: str) -> dict:
        """Persist weights ("sampler"), optimizer state ("state"), or "both".
        Returns a dict with "sampler_path" / "state_path" entries (backend-native URIs)."""
        ...
