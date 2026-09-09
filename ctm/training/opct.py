"""On-Policy Consistency Training (OPCT) over paired prompts.

OPCT is the online counterpart of BCT.  For a clean/reference prompt ``x`` and
a perturbed/variant prompt ``x_tilde``, the current student samples a completion
from ``x_tilde``.  A frozen run-start policy scores those exact tokens under
``x``.  The student is updated with the per-token reverse-KL estimator

    log pi_student(y_t | y_<t, x_tilde) - log pi_teacher(y_t | y_<t, x).

No trait classifier, reference-rate estimate, or GRPO reward is involved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tinker
import torch
from pydantic import BaseModel, field_validator
from tinker import types
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
from tinker_cookbook.utils.ml_log import setup_logging
from tqdm import tqdm

from ctm.backends.base import ForwardBackwardOutput, PolicyScorerHandle, SamplerHandle, TrainingBackend
from ctm.backends.renderers import get_renderer_and_tokenizer
from ctm.core.config import AdamConfig, CheckpointConfig, LoRAConfig
from ctm.core.types import RolloutRecord
from ctm.training.checkpoints import finalize_checkpoint, save_intermediate_checkpoint
from ctm.training.manifest import write_run_manifest
from ctm.training.rollout_log import RolloutLogger
from ctm.training.run_utils import build_log_dir, get_git_state, get_recommended_lr, warn_if_dirty

_log = logging.getLogger(__name__)


class OPCTGenerationConfig(BaseModel):
    """Online student-rollout configuration."""

    rollouts_per_prompt: int = 4
    # ``None`` is an explicit EOS-only policy. Backends must preserve it as
    # ``max_tokens=None`` and reject any non-EOS termination.
    max_new_tokens: int | None = 2048
    temperature: float = 0.7

    @field_validator("rollouts_per_prompt")
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("must be a positive integer")
        return value

    @field_validator("max_new_tokens")
    @classmethod
    def _positive_generation_limit_or_none(cls, value: int | None) -> int | None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError("must be a positive integer or None for EOS-only generation")
        return value

    @field_validator("temperature", mode="before")
    @classmethod
    def _non_negative_temperature(cls, value: object) -> object:
        # Preserve the established greedy compatibility mode.  The training
        # datum carries the finite behavior score reported for each emitted
        # token; a sampler that cannot provide those scores must reject it at
        # its own boundary.
        if isinstance(value, bool):
            raise ValueError("temperature must be a finite non-negative number")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            # Let Pydantic give its normal type error for non-numeric input.
            return value
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError("temperature must be a finite non-negative number")
        return value


class OPCTConfig(BaseModel):
    """Complete OPCT configuration."""

    experiment_name: str = "opct"
    run_name: str = "default"
    wandb_project: str | None = None
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora: LoRAConfig = LoRAConfig()
    optimizer: AdamConfig = AdamConfig(lr_schedule="constant")
    generation: OPCTGenerationConfig = OPCTGenerationConfig()
    n_epochs: int = 1
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    shuffle_samples: bool = True
    kl_coef: float = 1.0
    kl_discount_factor: float = 0.0
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    checkpoint: CheckpointConfig = CheckpointConfig()
    log_base_dir: str = "logs"
    rollout_log: Literal["none", "all"] = "all"
    rollout_dir: str | None = None
    reference_messages_field: str = "reference_messages"
    variant_messages_field: str = "variant_messages"
    run_metadata: dict = {}

    @field_validator("n_epochs", "batch_size", "gradient_accumulation_steps")
    @classmethod
    def _positive_loop_integer(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("must be a positive integer")
        return value

    @field_validator("kl_coef")
    @classmethod
    def _positive_kl_coef(cls, value: float) -> float:
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("kl_coef must be a finite positive number")
        return value

    @field_validator("kl_discount_factor")
    @classmethod
    def _valid_discount(cls, value: float) -> float:
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("kl_discount_factor must be in [0, 1]")
        return value

    @field_validator("reference_messages_field", "variant_messages_field")
    @classmethod
    def _non_empty_field_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("prompt field names must be non-empty strings")
        return value

    @field_validator("rollout_dir")
    @classmethod
    def _valid_rollout_dir(cls, value: str | None) -> str | None:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("rollout_dir must be a non-empty path")
        return value


def discounted_future_sum(values: torch.Tensor, discount: float) -> torch.Tensor:
    """Return ``sum_{u=t} discount**(u-t) * values[u]`` at each position."""

    if values.ndim != 1:
        raise ValueError(f"discounted_future_sum expects a 1D tensor, got shape {tuple(values.shape)}")
    if not 0 <= discount <= 1:
        raise ValueError("discount must be in [0, 1]")
    if values.numel() == 0 or discount == 0:
        return values.clone()
    if discount == 1:
        output = torch.flip(torch.cumsum(torch.flip(values, dims=(0,)), dim=0), dims=(0,))
        return output.to(dtype=values.dtype)

    # Compose adjacent discounted spans in parallel.  After each iteration,
    # output[t] contains the discounted sum over the next ``span`` values.
    # Doubling the span reduces device dispatches from one per token to O(log T)
    # while avoiding the overflow-prone division by discount**t used by the
    # usual cumsum reformulation.
    output = values.clone()
    span = 1
    span_discount = discount
    while span < values.numel():
        output = torch.cat(
            (output[:-span] + span_discount * output[span:], output[-span:]),
            dim=0,
        )
        span *= 2
        span_discount *= span_discount
    return output.to(dtype=values.dtype)


def apply_reference_reverse_kl(
    datums: Sequence[Any],
    teacher_action_logprobs: Sequence[Sequence[float]],
    *,
    student_action_logprobs: Sequence[Sequence[float]] | None = None,
    kl_coef: float,
    kl_discount_factor: float,
) -> dict[str, float]:
    """Add the OPCT reverse-KL signal to RL datum advantages in place.

    Student and teacher inputs contain raw-policy completion-token scores.  If
    ``student_action_logprobs`` is omitted, the datum's sampled logprobs are used
    for backwards compatibility; OPCT passes explicit raw scores because the
    datum must retain generation-distribution scores for importance sampling.

    The datum tensors also contain prompt positions, so the action mask is the
    authoritative alignment.  Non-action advantages are forced to zero because
    some backends remove the mask before submitting the loss.
    """

    if len(datums) != len(teacher_action_logprobs):
        raise ValueError(
            "datums and teacher_action_logprobs must have the same length, got "
            f"{len(datums)} and {len(teacher_action_logprobs)}"
        )
    if student_action_logprobs is not None and len(datums) != len(student_action_logprobs):
        raise ValueError(
            "datums and student_action_logprobs must have the same length, got "
            f"{len(datums)} and {len(student_action_logprobs)}"
        )
    if not math.isfinite(kl_coef) or kl_coef < 0:
        raise ValueError("kl_coef must be a finite non-negative number")
    if not math.isfinite(kl_discount_factor) or not 0 <= kl_discount_factor <= 1:
        raise ValueError("kl_discount_factor must be in [0, 1]")
    if not datums:
        raise ValueError("OPCT needs at least one training datum")

    reverse_kl_values: list[torch.Tensor] = []
    student_values: list[torch.Tensor] = []
    teacher_values: list[torch.Tensor] = []
    for index, (datum, raw_teacher) in enumerate(zip(datums, teacher_action_logprobs)):
        behavior = datum.loss_fn_inputs["logprobs"].to_torch().float()
        mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        existing_advantages = datum.loss_fn_inputs["advantages"].to_torch().float()
        if behavior.shape != mask.shape or behavior.shape != existing_advantages.shape:
            raise ValueError(
                f"datum {index} has inconsistent logprobs/mask/advantages shapes: "
                f"{tuple(behavior.shape)}, {tuple(mask.shape)}, {tuple(existing_advantages.shape)}"
            )
        behavior_actions = behavior[mask]
        raw_student = behavior_actions if student_action_logprobs is None else student_action_logprobs[index]
        student = torch.as_tensor(raw_student, dtype=behavior.dtype)
        teacher = torch.as_tensor(raw_teacher, dtype=behavior.dtype)
        if len(behavior_actions) != len(student):
            raise ValueError(
                f"datum {index} has {len(behavior_actions)} action tokens but {len(student)} student logprobs"
            )
        if len(student) != len(teacher):
            raise ValueError(
                f"datum {index} has {len(behavior_actions)} action tokens but {len(teacher)} teacher logprobs"
            )
        if not len(teacher):
            raise ValueError(f"datum {index} has no action tokens")
        if (
            not torch.isfinite(behavior_actions).all()
            or not torch.isfinite(student).all()
            or not torch.isfinite(teacher).all()
        ):
            raise ValueError(f"datum {index} contains non-finite behavior, student, or teacher logprobs")

        reverse_kl = student - teacher
        action_signal = -kl_coef * reverse_kl
        if kl_discount_factor > 0:
            action_signal = discounted_future_sum(action_signal, kl_discount_factor)
        updated = torch.zeros_like(existing_advantages)
        updated[mask] = existing_advantages[mask] + action_signal
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(updated)

        reverse_kl_values.append(reverse_kl)
        student_values.append(student)
        teacher_values.append(teacher)

    flat_reverse_kl = torch.cat(reverse_kl_values)
    flat_student = torch.cat(student_values)
    flat_teacher = torch.cat(teacher_values)
    return {
        "teacher_kl": float(flat_reverse_kl.mean()),
        "student_entropy": float(-flat_student.mean()),
        "teacher_cross_entropy": float(-flat_teacher.mean()),
        "teacher_scored_tokens": float(len(flat_teacher)),
    }


def _validated_messages(sample: dict, field: str, row_index: int) -> list[dict]:
    messages = sample.get(field)
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row_index} field {field!r} must be a non-empty message list")
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"row {row_index} {field}[{message_index}] must be an object")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not role.strip() or not isinstance(content, str) or not content.strip():
            raise ValueError(f"row {row_index} {field}[{message_index}] needs non-empty string role/content fields")
    return messages


def validate_opct_samples(samples: Sequence[dict], config: OPCTConfig) -> list[dict]:
    """Validate paired prompts before initializing a paid or accelerator backend."""

    if not samples:
        raise ValueError("OPCT training data is empty")
    validated = list(samples)
    for index, sample in enumerate(validated, start=1):
        if not isinstance(sample, dict):
            raise TypeError(f"row {index} must be a JSON object")
        _validated_messages(sample, config.reference_messages_field, index)
        _validated_messages(sample, config.variant_messages_field, index)
    return validated


@dataclass
class _UnscoredOPCTBatch:
    """Validated rollouts awaiting raw student/reference policy scores."""

    references: list[types.ModelInput]
    variants: list[types.ModelInput]
    completions: list[list[int]]
    sampled_logprobs: list[list[float]]
    valid_rollout_meta: list[dict | None]
    rollout_meta: list[dict]
    response_lengths: list[int]


class OPCTTrainer:
    """Backend-agnostic, fully on-policy consistency trainer."""

    def __init__(
        self,
        *,
        config: OPCTConfig,
        backend: TrainingBackend,
        resume_from: str | None = None,
        resume_with_optimizer: bool | None = None,
    ):
        self.config = config
        self.backend = backend
        self.resume_from = resume_from
        self.resume_with_optimizer = resume_with_optimizer
        self.renderer: Any = None
        self.tokenizer: Any = None
        self.sampling_client: SamplerHandle | None = None
        self.reference_policy: PolicyScorerHandle | None = None
        self._rollout_logger: RolloutLogger | None = None
        self._pending_rollout_meta: list[dict] = []
        self.setup_done = False

    def setup(self) -> None:
        if self.config.lora.seed is not None:
            random.seed(self.config.lora.seed)
        if self.resume_from and not self.backend.policy_samplers_are_snapshots:
            raise NotImplementedError(
                "OPCT resume requires an immutable run-start policy handle; "
                f"{type(self.backend).__name__} exposes live policy handles, so resuming would use the wrong teacher"
            )
        with_optimizer = False
        if self.resume_from:
            with_optimizer = (
                self.resume_with_optimizer
                if self.resume_with_optimizer is not None
                else "/weights/" in self.resume_from and "/sampler_weights/" not in self.resume_from
            )
        self.backend.setup(
            model=self.config.model,
            lora=self.config.lora,
            resume_from=self.resume_from,
            resume_with_optimizer=with_optimizer,
        )
        self.renderer, self.tokenizer = get_renderer_and_tokenizer(
            self.config.model,
            source=self.backend.renderer_source,
        )
        self.sampling_client = self.backend.policy_sampler(
            name=f"{self.config.experiment_name}_{self.config.run_name}_opct_sampler"
        )
        if self.backend.policy_samplers_are_snapshots:
            # This exact handle captures the current policy after any supported
            # checkpoint load, so it is both equal at run start and immutable.
            self.reference_policy = self.sampling_client
        else:
            # Local live handles follow optimizer updates.  Fresh local runs use
            # the immutable base copy, which equals the just-initialized policy.
            self.reference_policy = self.backend.base_sampler()
        self.setup_done = True

    def _requires_serialized_sampling_and_training(self) -> bool:
        """Whether rollout/scoring and coordinator work share GPU resources.

        Tinker and the ordinary local topology deliberately retain their
        historical overlap: OPCT can generate the next unchanged-policy group
        while the current group is in forward/backward.  A sleep/wake rollout
        backend reclaims those same GPUs for the persistent trainer replicas,
        so it advertises ``sampling_training_overlap_supported = False`` and
        requires an explicit phase transition around every group.

        The optional attribute is intentionally duck-typed.  That preserves
        the behavior of all existing backends which predate phase sharing.
        """

        supported = getattr(self.backend, "sampling_training_overlap_supported", True)
        if callable(supported):
            supported = supported()
        return not bool(supported)

    async def _enter_rollout_phase_if_needed(self) -> None:
        """Wake phase-shared rollout workers before sampling or raw scoring."""

        if not self._requires_serialized_sampling_and_training():
            return
        enter = getattr(self.backend, "enter_rollout_phase", None)
        if not callable(enter):
            raise TypeError(
                "backend disables sampling/training overlap but does not expose "
                "an async enter_rollout_phase() lifecycle method"
            )
        await enter()

    async def _enter_training_phase_if_needed(self) -> None:
        """Sleep phase-shared rollout workers before coordinator computation."""

        if not self._requires_serialized_sampling_and_training():
            return
        enter = getattr(self.backend, "enter_training_phase", None)
        if not callable(enter):
            raise TypeError(
                "backend disables sampling/training overlap but does not expose "
                "an async enter_training_phase() lifecycle method"
            )
        await enter()

    @staticmethod
    def _create_datum(prompt: types.ModelInput, tokens: list[int], logprobs: list[float]):
        transition = Transition(
            ob=prompt,
            ac=TokensWithLogprobs(tokens=tokens, maybe_logprobs=logprobs),
            reward=0.0,
            episode_done=True,
        )
        trajectory = Trajectory(
            transitions=[transition],
            final_ob=types.ModelInput.from_ints(tokens=[]),
        )
        datums = trajectory_to_data(trajectory, traj_advantage=0.0)
        if len(datums) != 1:
            raise RuntimeError(f"expected one OPCT datum per completion, got {len(datums)}")
        return datums[0]

    def _decode_tokens(self, tokens: Sequence[int]) -> str:
        """Best-effort text for provenance; decoding never blocks training."""

        try:
            decoded = self.tokenizer.decode(list(tokens))
            return decoded if isinstance(decoded, str) else ""
        except Exception:  # noqa: BLE001
            return ""

    def _log_rollouts(self, step: int, epoch: int) -> None:
        if self._rollout_logger is None or not self._pending_rollout_meta:
            self._pending_rollout_meta = []
            return
        records = [RolloutRecord(step=step, epoch=epoch, **meta) for meta in self._pending_rollout_meta]
        self._rollout_logger.log_step(records)
        self._pending_rollout_meta = []

    def _prepare_batch(
        self,
        batch: Sequence[tuple[int, dict] | dict],
    ) -> list[tuple[int, types.ModelInput, types.ModelInput]]:
        """Render one logical microbatch into reference/variant pairs."""

        pairs = []
        for fallback_idx, item in enumerate(batch):
            if isinstance(item, tuple):
                datapoint_idx, sample = item
            else:
                datapoint_idx, sample = fallback_idx, item
            reference_messages = _validated_messages(sample, self.config.reference_messages_field, datapoint_idx + 1)
            variant_messages = _validated_messages(sample, self.config.variant_messages_field, datapoint_idx + 1)
            pairs.append(
                (
                    datapoint_idx,
                    self.renderer.build_generation_prompt(reference_messages),
                    self.renderer.build_generation_prompt(variant_messages),
                )
            )
        return pairs

    async def _sample_prepared_pairs(
        self,
        pairs: Sequence[tuple[int, types.ModelInput, types.ModelInput]],
    ) -> list[list[Any]]:
        """Sample variants, using one native prompt batch when available."""

        # OPCT generates each unchanged-policy accumulation group before any
        # of its F/B work.  In a phase-shared topology this is the earliest
        # point at which vLLM may safely reclaim the rollout GPUs.
        await self._enter_rollout_phase_if_needed()
        assert self.sampling_client is not None
        prompts = [variant_prompt for _, _, variant_prompt in pairs]
        sample_kwargs = {
            "max_tokens": self.config.generation.max_new_tokens,
            "temperature": self.config.generation.temperature,
            "stop": self.renderer.get_stop_sequences(),
            "num_samples": self.config.generation.rollouts_per_prompt,
        }
        batch_sampler = getattr(self.sampling_client, "sample_batch", None)
        if callable(batch_sampler):
            sampled_groups = await batch_sampler(prompts, **sample_kwargs)
        else:
            sampled_groups = await asyncio.gather(*[self.sampling_client.sample(prompt, **sample_kwargs) for prompt in prompts])
        if len(sampled_groups) != len(pairs):
            raise RuntimeError(f"student sampler returned {len(sampled_groups)} prompt group(s); expected {len(pairs)}")
        return [list(group) for group in sampled_groups]

    async def _build_batch(
        self,
        batch: Sequence[tuple[int, dict] | dict],
        *,
        prepared_pairs: Sequence[tuple[int, types.ModelInput, types.ModelInput]] | None = None,
        sampled_groups: Sequence[Sequence[Any]] | None = None,
    ):
        assert self.sampling_client is not None
        assert self.reference_policy is not None
        self._pending_rollout_meta = []
        pairs = list(prepared_pairs) if prepared_pairs is not None else self._prepare_batch(batch)
        if len(pairs) != len(batch):
            raise ValueError(f"prepared OPCT pair count {len(pairs)} does not match microbatch size {len(batch)}")
        if sampled_groups is None:
            sampled_groups = await self._sample_prepared_pairs(pairs)
        elif len(sampled_groups) != len(pairs):
            raise ValueError(f"prefetched OPCT sample-group count {len(sampled_groups)} does not match pair count {len(pairs)}")

        result = (
            await self._build_batch_group(
                [batch],
                prepared_pair_groups=[pairs],
                sampled_groups=[sampled_groups],
            )
        )[0]
        datums, kl_metrics, response_lengths, rollout_meta = result
        self._pending_rollout_meta = rollout_meta
        return datums, kl_metrics, response_lengths

    def _collect_unscored_batch(
        self,
        pairs: Sequence[tuple[int, types.ModelInput, types.ModelInput]],
        sampled_groups: Sequence[Sequence[Any]],
    ) -> _UnscoredOPCTBatch:
        """Validate sampled sequences without changing scoring reductions."""

        references: list[types.ModelInput] = []
        variants: list[types.ModelInput] = []
        completions: list[list[int]] = []
        sampled_logprobs: list[list[float]] = []
        valid_rollout_meta: list[dict | None] = []
        rollout_meta_records: list[dict] = []
        response_lengths: list[int] = []
        for pair_index, ((datapoint_idx, reference_prompt, variant_prompt), sequences) in enumerate(zip(pairs, sampled_groups)):
            if len(sequences) != self.config.generation.rollouts_per_prompt:
                raise RuntimeError(f"student sampler returned {len(sequences)} rollouts for pair {pair_index}; expected {self.config.generation.rollouts_per_prompt}")
            for sequence in sequences:
                tokens = list(sequence.tokens)
                response_lengths.append(len(tokens))
                logprobs: list[float] = []
                skip_reason = None
                if not tokens:
                    skip_reason = "empty_completion"
                elif sequence.logprobs is None:
                    skip_reason = "missing_logprobs"
                elif len(sequence.logprobs) != len(tokens):
                    skip_reason = "misaligned_logprobs"
                else:
                    try:
                        logprobs = [float(value) for value in sequence.logprobs]
                    except (TypeError, ValueError):
                        skip_reason = "non_finite_logprobs"
                    if skip_reason is None and not all(math.isfinite(value) for value in logprobs):
                        skip_reason = "non_finite_logprobs"

                rollout_meta = None
                if self._rollout_logger is not None:
                    rollout_meta = {
                        "datapoint_idx": datapoint_idx,
                        "perturbation_idx": 0,
                        "role": "train",
                        "sample_source": "policy",
                        "prompt_text": self._decode_tokens(variant_prompt.to_ints()),
                        "prompt_context": {
                            "reference": self._decode_tokens(reference_prompt.to_ints()),
                            "variant": self._decode_tokens(variant_prompt.to_ints()),
                        },
                        "completion_text": self._decode_tokens(tokens),
                        "trait_value": None,
                        "parsed_successfully": skip_reason is None,
                        "grader_failed": False,
                        "reward": None,
                        "advantage": None,
                        "skipped_from_training": skip_reason is not None,
                        "skip_reason": skip_reason,
                        "p_hat": None,
                        "p_ref": None,
                        "p_ref_init": None,
                    }
                    rollout_meta_records.append(rollout_meta)

                if skip_reason is not None:
                    continue
                references.append(reference_prompt)
                variants.append(variant_prompt)
                completions.append(tokens)
                sampled_logprobs.append(logprobs)
                valid_rollout_meta.append(rollout_meta)

        return _UnscoredOPCTBatch(
            references=references,
            variants=variants,
            completions=completions,
            sampled_logprobs=sampled_logprobs,
            valid_rollout_meta=valid_rollout_meta,
            rollout_meta=rollout_meta_records,
            response_lengths=response_lengths,
        )

    def _finalize_scored_batch(
        self,
        prepared: _UnscoredOPCTBatch,
        student_logprobs: Sequence[Sequence[float]] | None,
        teacher_logprobs: Sequence[Sequence[float]],
        *,
        fused_student_scoring: bool = False,
    ) -> tuple[list[Any], dict[str, float], list[int], list[dict]]:
        """Apply the original per-microbatch OPCT reduction to batched scores."""

        if not prepared.completions:
            return (
                [],
                {
                    "teacher_kl": 0.0,
                    "student_entropy": 0.0,
                    "teacher_cross_entropy": 0.0,
                    "teacher_scored_tokens": 0.0,
                },
                prepared.response_lengths,
                prepared.rollout_meta,
            )
        if len(teacher_logprobs) != len(prepared.completions):
            raise RuntimeError(
                "OPCT teacher scorer result count does not match valid completions: "
                f"teacher={len(teacher_logprobs)}, completions={len(prepared.completions)}"
            )
        if student_logprobs is not None and len(student_logprobs) != len(prepared.completions):
            raise RuntimeError(
                "OPCT student scorer result count does not match valid completions: "
                f"student={len(student_logprobs)}, completions={len(prepared.completions)}"
            )

        datums = [
            self._create_datum(prompt, tokens, logprobs)
            for prompt, tokens, logprobs in zip(
                prepared.variants,
                prepared.completions,
                prepared.sampled_logprobs,
            )
        ]
        if fused_student_scoring:
            for datum, teacher in zip(datums, teacher_logprobs):
                datum.loss_fn_inputs["opct_teacher_logprobs"] = tinker.TensorData.from_torch(
                    torch.as_tensor(teacher, dtype=torch.float32)
                )
            # Raw current-policy scores, reverse-KL advantages, and these
            # aggregate placeholders are completed by the fused F/B result.
            # The generated behavior scores above remain authoritative.
            return (
                datums,
                {
                    "teacher_kl": 0.0,
                    "student_entropy": 0.0,
                    "teacher_cross_entropy": 0.0,
                    "teacher_scored_tokens": 0.0,
                },
                prepared.response_lengths,
                prepared.rollout_meta,
            )
        if student_logprobs is None:
            raise RuntimeError("non-fused OPCT batches require raw student scores")
        kl_metrics = apply_reference_reverse_kl(
            datums,
            teacher_logprobs,
            student_action_logprobs=student_logprobs,
            kl_coef=self.config.kl_coef,
            kl_discount_factor=self.config.kl_discount_factor,
        )
        for datum, raw_student, raw_teacher, rollout_meta in zip(
            datums,
            student_logprobs,
            teacher_logprobs,
            prepared.valid_rollout_meta,
        ):
            if rollout_meta is None:
                continue
            reverse_kl = torch.as_tensor(raw_student).float() - torch.as_tensor(raw_teacher).float()
            mask = datum.loss_fn_inputs["mask"].to_torch() > 0
            action_advantages = datum.loss_fn_inputs["advantages"].to_torch()[mask]
            rollout_meta["reward"] = float((-self.config.kl_coef * reverse_kl).mean())
            rollout_meta["advantage"] = float(action_advantages.mean())
        return datums, kl_metrics, prepared.response_lengths, prepared.rollout_meta

    async def _build_batch_group(
        self,
        batches: Sequence[Sequence[tuple[int, dict] | dict]],
        *,
        prepared_pair_groups: Sequence[Sequence[tuple[int, types.ModelInput, types.ModelInput]]],
        sampled_groups: Sequence[Sequence[Sequence[Any]]],
    ) -> list[tuple[list[Any], dict[str, float], list[int], list[dict]]]:
        """Score one unchanged-policy accumulation group in flat calls.

        Scores are split back at the original microbatch boundaries before KL
        metrics and advantages are reduced, so only transport batching changes.
        A fused local backend needs only the frozen-teacher call: its train-time
        differentiable forward supplies raw current-policy scores, while datums
        retain the processed behavior scores returned by generation.
        """

        assert self.sampling_client is not None
        assert self.reference_policy is not None
        if not (len(batches) == len(prepared_pair_groups) == len(sampled_groups)):
            raise ValueError("OPCT grouped batches, prepared pairs, and samples must have the same length")

        prepared_batches: list[_UnscoredOPCTBatch] = []
        for batch, pairs, samples in zip(batches, prepared_pair_groups, sampled_groups):
            if len(pairs) != len(batch):
                raise ValueError(f"prepared OPCT pair count {len(pairs)} does not match microbatch size {len(batch)}")
            if len(samples) != len(pairs):
                raise ValueError(f"prefetched OPCT sample-group count {len(samples)} does not match pair count {len(pairs)}")
            prepared_batches.append(self._collect_unscored_batch(pairs, samples))

        flat_variants = [prompt for prepared in prepared_batches for prompt in prepared.variants]
        flat_references = [prompt for prepared in prepared_batches for prompt in prepared.references]
        flat_completions = [tokens for prepared in prepared_batches for tokens in prepared.completions]
        fused_submit = getattr(self.backend, "submit_opct_forward_backward", None)
        fused_student_scoring = bool(getattr(self.backend, "supports_fused_opct_scoring", False))
        if fused_student_scoring and not callable(fused_submit):
            raise RuntimeError("backend advertises fused OPCT scoring without a submit method")

        if flat_completions:
            # Both the live student scorer (non-fused path) and the frozen
            # OPCT teacher can be rollout-worker vLLM handles.  Keep the
            # entire raw-scoring group in rollout phase; the caller switches
            # to trainer phase only after all scores have returned.
            await self._enter_rollout_phase_if_needed()
            # Generated-token scores are the processed behavior distribution.
            # Non-fused backends still need raw student scoring here. A fused
            # local backend obtains it from the differentiable training pass.
            flat_student = (
                None
                if fused_student_scoring
                else await self.sampling_client.score_completions(
                    flat_variants,
                    flat_completions,
                )
            )
            flat_teacher = await self.reference_policy.score_completions(flat_references, flat_completions)
            if len(flat_teacher) != len(flat_completions):
                raise RuntimeError(
                    "OPCT grouped teacher scorer result count does not match valid completions: "
                    f"teacher={len(flat_teacher)}, completions={len(flat_completions)}"
                )
            if flat_student is not None and len(flat_student) != len(flat_completions):
                raise RuntimeError(
                    "OPCT grouped student scorer result count does not match valid completions: "
                    f"student={len(flat_student)}, completions={len(flat_completions)}"
                )
        else:
            flat_student = None if fused_student_scoring else []
            flat_teacher = []

        output = []
        offset = 0
        for prepared in prepared_batches:
            end = offset + len(prepared.completions)
            output.append(
                self._finalize_scored_batch(
                    prepared,
                    None if flat_student is None else flat_student[offset:end],
                    flat_teacher[offset:end],
                    fused_student_scoring=fused_student_scoring,
                )
            )
            offset = end
        return output

    def _complete_fused_batch(
        self,
        datums: Sequence[Any],
        fwd_bwd: ForwardBackwardOutput,
        kl_metrics: dict[str, float],
        rollout_meta: Sequence[dict],
    ) -> None:
        """Complete fused KL reductions and provenance from pre-update scores."""

        if len(fwd_bwd.logprobs) != len(datums):
            raise RuntimeError(
                "fused OPCT forward/backward returned the wrong number of score tensors: "
                f"scores={len(fwd_bwd.logprobs)}, datums={len(datums)}"
            )

        student_action_logprobs: list[torch.Tensor] = []
        teacher_action_logprobs: list[torch.Tensor] = []
        for index, (datum, raw_values) in enumerate(zip(datums, fwd_bwd.logprobs)):
            mask = datum.loss_fn_inputs["mask"].to_torch().bool()
            raw = torch.as_tensor(raw_values).detach().float().cpu()
            if raw.shape != mask.shape:
                raise RuntimeError(
                    f"fused OPCT datum {index} raw-score/mask shape mismatch: "
                    f"raw={tuple(raw.shape)}, mask={tuple(mask.shape)}"
                )
            teacher_data = datum.loss_fn_inputs.get("opct_teacher_logprobs")
            if teacher_data is None:
                raise RuntimeError(f"fused OPCT datum {index} is missing teacher scores")
            teacher = teacher_data.to_torch().detach().float().cpu()
            if teacher.ndim != 1 or teacher.numel() != int(mask.sum()):
                raise RuntimeError(
                    f"fused OPCT datum {index} has {int(mask.sum())} action tokens but "
                    f"{teacher.numel()} teacher scores"
                )
            student_action_logprobs.append(raw[mask])
            teacher_action_logprobs.append(teacher)

        completed_metrics = apply_reference_reverse_kl(
            datums,
            teacher_action_logprobs,
            student_action_logprobs=student_action_logprobs,
            kl_coef=self.config.kl_coef,
            kl_discount_factor=self.config.kl_discount_factor,
        )
        # Replace all pre-F/B placeholders. Keeping the normalized output in
        # sync also makes the final metrics merge and progress display agree.
        kl_metrics.update(completed_metrics)
        fwd_bwd.metrics.update(completed_metrics)

        if not rollout_meta:
            return
        valid_meta = [meta for meta in rollout_meta if not meta.get("skipped_from_training", False)]
        if len(valid_meta) != len(datums):
            raise RuntimeError(
                "fused OPCT rollout provenance does not align with training datums: "
                f"valid_records={len(valid_meta)}, datums={len(datums)}"
            )
        for datum, student, teacher, meta in zip(
            datums,
            student_action_logprobs,
            teacher_action_logprobs,
            valid_meta,
        ):
            reverse_kl = student - teacher
            mask = datum.loss_fn_inputs["mask"].to_torch() > 0
            action_advantages = datum.loss_fn_inputs["advantages"].to_torch()[mask]
            meta["reward"] = float((-self.config.kl_coef * reverse_kl).mean())
            meta["advantage"] = float(action_advantages.mean())

    async def train(self, samples: Sequence[dict]) -> str:
        """Train on paired prompt rows and return the final checkpoint path."""

        samples = validate_opct_samples(samples, self.config)
        log_dir = Path(build_log_dir(self.config.log_base_dir, self.config.experiment_name, self.config.run_name))
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(
            log_dir=str(log_dir),
            wandb_project=self.config.wandb_project,
            wandb_name=self.config.run_name,
            config=self.config.model_dump(),
        )
        try:
            if self.config.rollout_log != "none":
                rollout_dir = Path(self.config.rollout_dir or str(log_dir / "rollouts"))
                rollout_index = rollout_dir / "index.json"
                existing_step_file = next(rollout_dir.glob("step_*.jsonl.zst"), None) if rollout_dir.exists() else None
                if rollout_index.exists():
                    try:
                        prior_steps = json.loads(rollout_index.read_text(encoding="utf-8")).get("steps", [])
                    except (json.JSONDecodeError, OSError, AttributeError):
                        prior_steps = ["unreadable"]
                else:
                    prior_steps = []
                if prior_steps or existing_step_file is not None:
                    raise FileExistsError(
                        f"OPCT rollout directory {rollout_dir} already contains step records. "
                        "Checkpoint loading is a warm start and does not restore the loop position; "
                        "choose a fresh run name or --rollout-dir to avoid overwriting provenance."
                    )
                self._rollout_logger = RolloutLogger(rollout_dir)
            else:
                self._rollout_logger = None
            if not self.setup_done:
                self.setup()
            git_state = get_git_state()
            warn_if_dirty(git_state)
            logger.log_hparams({"git": git_state})
            write_run_manifest(
                log_dir,
                kind="opct",
                model=self.config.model,
                backend=self.backend,
                config_dump=self.config.model_dump(),
                extra={"n_samples": len(samples)},
            )

            microbatches_per_epoch = (len(samples) + self.config.batch_size - 1) // self.config.batch_size
            optimizer_steps_per_epoch = (
                microbatches_per_epoch + self.config.gradient_accumulation_steps - 1
            ) // self.config.gradient_accumulation_steps
            total_steps = optimizer_steps_per_epoch * self.config.n_epochs
            base_lr = (
                self.config.optimizer.learning_rate
                if self.config.optimizer.learning_rate is not None
                else get_recommended_lr(self.config.model)
            )
            logger.log_hparams(
                {
                    "n_samples": len(samples),
                    "total_steps": total_steps,
                    "base_lr": base_lr,
                    "rollouts_per_prompt": self.config.generation.rollouts_per_prompt,
                }
            )
            print(
                f"OPCT Training: {len(samples)} prompt pairs, batch={self.config.batch_size}, "
                f"k={self.config.generation.rollouts_per_prompt}, {total_steps} optimizer steps, "
                f"lr={base_lr:.2e}"
            )

            def learning_rate(step: int) -> float:
                multiplier = max(
                    0.0,
                    compute_schedule_lr_multiplier(
                        lr_schedule=self.config.optimizer.lr_schedule,
                        step=step,
                        total_steps=total_steps,
                    ),
                )
                return base_lr * multiplier

            checkpoint_paths: list[str] = []
            global_step = 0
            global_microbatch = 0
            accumulated_grad_batches = 0
            indexed_samples = list(enumerate(samples))
            phase_serialized = self._requires_serialized_sampling_and_training()
            for epoch in range(self.config.n_epochs):
                epoch_samples = list(indexed_samples)
                if self.config.shuffle_samples:
                    random.shuffle(epoch_samples)
                batch_starts = list(range(0, len(epoch_samples), self.config.batch_size))
                pbar = tqdm(batch_starts, desc=f"Epoch {epoch + 1}")
                prefetched_batches: dict[
                    int,
                    tuple[list[Any], dict[str, float], list[int], list[dict]],
                ] = {}
                for microbatch_index, _ in enumerate(pbar):
                    if not prefetched_batches:
                        # Generate and raw-score the whole unchanged-policy
                        # accumulation group in native batches. Scores are split
                        # before per-microbatch KL/advantage reduction and fwd/bwd.
                        group_end = min(
                            microbatch_index + self.config.gradient_accumulation_steps,
                            len(batch_starts),
                        )
                        group_batches: list[list[tuple[int, dict]]] = []
                        group_pairs: list[list[tuple[int, types.ModelInput, types.ModelInput]]] = []
                        for grouped_index in range(microbatch_index, group_end):
                            grouped_start = batch_starts[grouped_index]
                            grouped_batch = epoch_samples[grouped_start : grouped_start + self.config.batch_size]
                            group_batches.append(grouped_batch)
                            group_pairs.append(self._prepare_batch(grouped_batch))
                        flat_pairs = [pair for prepared in group_pairs for pair in prepared]
                        flat_samples = await self._sample_prepared_pairs(flat_pairs)
                        offset = 0
                        group_samples = []
                        for prepared in group_pairs:
                            end = offset + len(prepared)
                            group_samples.append(flat_samples[offset:end])
                            offset = end
                        group_results = await self._build_batch_group(
                            group_batches,
                            prepared_pair_groups=group_pairs,
                            sampled_groups=group_samples,
                        )
                        for grouped_index, result in zip(range(microbatch_index, group_end), group_results):
                            prefetched_batches[grouped_index] = result
                    datums, kl_metrics, response_lengths, rollout_meta = prefetched_batches.pop(microbatch_index)
                    self._pending_rollout_meta = rollout_meta
                    current_lr = learning_rate(global_step)
                    should_step = (microbatch_index + 1) % self.config.gradient_accumulation_steps == 0 or microbatch_index + 1 == len(batch_starts)
                    uses_fused_scoring = bool(datums) and all(
                        "opct_teacher_logprobs" in datum.loss_fn_inputs for datum in datums
                    )

                    # Non-fused metadata is already complete. Fused metadata
                    # needs the differentiable raw scores returned below, but is
                    # still persisted before any optimizer mutation.
                    if not uses_fused_scoring:
                        self._log_rollouts(global_microbatch + 1, epoch)

                    # All generation and raw policy/reference scoring for this
                    # unchanged-policy group have completed.  In phase-shared
                    # mode the persistent HF replicas now reclaim the rollout
                    # GPUs for F/B (and, at an accumulation boundary, the
                    # optimizer).  The check also covers an empty final
                    # microbatch which still needs to flush gradients from an
                    # earlier microbatch in the group.
                    if datums or (should_step and accumulated_grad_batches > 0):
                        await self._enter_training_phase_if_needed()

                    pending_fwd_bwd = None
                    if datums:
                        fused_submit = getattr(self.backend, "submit_opct_forward_backward", None)
                        if uses_fused_scoring:
                            if not callable(fused_submit):
                                raise RuntimeError("fused OPCT datums require backend support")
                            pending_fwd_bwd = await fused_submit(
                                datums,
                                behavior_temperature=self.config.generation.temperature,
                                kl_coef=self.config.kl_coef,
                                kl_discount_factor=self.config.kl_discount_factor,
                                loss_fn=self.config.loss_fn,
                            )
                        else:
                            pending_fwd_bwd = await self.backend.submit_forward_backward(
                                datums,
                                loss_fn=self.config.loss_fn,
                            )
                        accumulated_grad_batches += 1

                    fwd_bwd = None
                    if uses_fused_scoring:
                        assert pending_fwd_bwd is not None
                        fwd_bwd = await pending_fwd_bwd.result()
                        self._complete_fused_batch(datums, fwd_bwd, kl_metrics, rollout_meta)
                        self._pending_rollout_meta = rollout_meta
                        self._log_rollouts(global_microbatch + 1, epoch)

                    pending_optim = None
                    if should_step and accumulated_grad_batches > 0:
                        pending_optim = await self.backend.submit_optim_step(
                            learning_rate=current_lr,
                            adam=self.config.optimizer,
                        )
                    if fwd_bwd is None and pending_fwd_bwd is not None:
                        fwd_bwd = await pending_fwd_bwd.result()
                    did_step = pending_optim is not None
                    if pending_optim is not None:
                        await pending_optim.result()
                        accumulated_grad_batches = 0
                        global_step += 1
                        if not phase_serialized:
                            # Preserve the historical disjoint-GPU schedule:
                            # publish immediately after the optimizer update,
                            # before logging or checkpoint materialization.
                            # OPCT's defining guarantee remains that every
                            # post-update rollout uses newly published weights.
                            self.sampling_client = await self.backend.refresh_policy_sampler(
                                name=(
                                    f"{self.config.experiment_name}_"
                                    f"{self.config.run_name}_opct_sampler_{global_step}"
                                )
                            )

                    global_microbatch += 1
                    metrics = {
                        **{f"train/{key}": value for key, value in kl_metrics.items()},
                        **({f"train/{key}": value for key, value in fwd_bwd.metrics.items()} if fwd_bwd else {}),
                        "train/lr": current_lr,
                        "train/optimizer_step": global_step,
                        "train/n_rollouts": len(datums),
                        "train/n_skipped_rollouts": len(response_lengths) - len(datums),
                        "train/avg_response_length": (
                            sum(response_lengths) / len(response_lengths) if response_lengths else 0.0
                        ),
                        "train/epoch": epoch,
                    }
                    logger.log_metrics(metrics, step=global_microbatch)
                    pbar.set_postfix(
                        {
                            "teacher_kl": f"{kl_metrics['teacher_kl']:.4f}",
                            "loss": f"{(fwd_bwd.metrics.get('loss', float('nan')) if fwd_bwd else float('nan')):.4f}",
                            "step": global_step,
                        }
                    )

                    if did_step:
                        if phase_serialized:
                            # A checkpoint hashes/serializes trainer state, so
                            # it must complete while worker vLLM engines remain
                            # asleep.  Only then may adapter publication wake
                            # them for the next unchanged-policy rollout group.
                            await self._enter_training_phase_if_needed()
                        await save_intermediate_checkpoint(
                            self.backend,
                            experiment_name=self.config.experiment_name,
                            run_name=self.config.run_name,
                            checkpoint_cfg=self.config.checkpoint,
                            global_step=global_step,
                            total_steps=total_steps,
                            epoch=epoch,
                            log_dir=log_dir,
                            checkpoint_paths=checkpoint_paths,
                            logger=logger,
                        )
                        if phase_serialized:
                            self.sampling_client = await self.backend.refresh_policy_sampler(
                                name=(
                                    f"{self.config.experiment_name}_"
                                    f"{self.config.run_name}_opct_sampler_{global_step}"
                                )
                            )

            # The final checkpoint is trainer work too.  In particular, the
            # last adapter refresh may have woken a phase-shared worker pool;
            # re-establish the barrier before it hashes or writes state.
            await self._enter_training_phase_if_needed()
            return await finalize_checkpoint(
                self.backend,
                experiment_name=self.config.experiment_name,
                run_name=self.config.run_name,
                n_epochs=self.config.n_epochs,
                save_state=self.config.checkpoint.save_state,
                global_step=global_step,
                log_dir=log_dir,
                checkpoint_paths=checkpoint_paths,
                logger=logger,
            )
        except Exception:
            _log.error("OPCT training failed:\n%s", traceback.format_exc())
            try:
                logger.close()
            except Exception:  # noqa: BLE001 -- preserve the original training failure
                _log.warning("Failed to close OPCT metric logger:\n%s", traceback.format_exc())
            raise
        finally:
            shutdown = getattr(self.backend, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:  # noqa: BLE001 -- backend shutdown is best-effort during cleanup
                    _log.warning("Backend shutdown failed:\n%s", traceback.format_exc())


__all__ = [
    "OPCTConfig",
    "OPCTGenerationConfig",
    "OPCTTrainer",
    "apply_reference_reverse_kl",
    "discounted_future_sum",
    "validate_opct_samples",
]
