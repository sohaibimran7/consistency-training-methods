"""On-Policy Consistency Training (OPCT) over paired prompts.

OPCT is the online counterpart of BCT.  For a clean/reference prompt ``x`` and
a perturbed/variant prompt ``x_tilde``, the current student samples a completion
from ``x_tilde``.  A frozen copy of the initial model scores those exact tokens
under ``x``.  The student is updated with the per-token reverse-KL estimator

    log pi_student(y_t | y_<t, x_tilde) - log pi_teacher(y_t | y_<t, x).

No trait classifier, reference-rate estimate, or GRPO reward is involved.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import traceback
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

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

from ctm.backends.base import SamplerHandle, TrainingBackend
from ctm.backends.renderers import get_renderer_and_tokenizer
from ctm.core.config import AdamConfig, CheckpointConfig, LoRAConfig
from ctm.training.checkpoints import finalize_checkpoint, save_intermediate_checkpoint
from ctm.training.manifest import write_run_manifest
from ctm.training.run_utils import build_log_dir, get_git_state, get_recommended_lr, warn_if_dirty

_log = logging.getLogger(__name__)


class OPCTGenerationConfig(BaseModel):
    """Online student-rollout configuration."""

    rollouts_per_prompt: int = 4
    max_new_tokens: int = 2048
    temperature: float = 0.7

    @field_validator("rollouts_per_prompt", "max_new_tokens")
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("must be a positive integer")
        return value

    @field_validator("temperature")
    @classmethod
    def _non_negative_temperature(cls, value: float) -> float:
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("temperature must be a finite non-negative number")
        return value


class OPCTConfig(BaseModel):
    """Complete OPCT configuration."""

    experiment_name: str = "opct"
    run_name: str = "default"
    wandb_project: Optional[str] = None
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora: LoRAConfig = LoRAConfig()
    optimizer: AdamConfig = AdamConfig(lr_schedule="constant")
    generation: OPCTGenerationConfig = OPCTGenerationConfig()
    n_epochs: int = 1
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    kl_coef: float = 1.0
    kl_discount_factor: float = 0.0
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    checkpoint: CheckpointConfig = CheckpointConfig()
    log_base_dir: str = "logs"
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


def discounted_future_sum(values: torch.Tensor, discount: float) -> torch.Tensor:
    """Return ``sum_{u=t} discount**(u-t) * values[u]`` at each position."""

    if values.ndim != 1:
        raise ValueError(f"discounted_future_sum expects a 1D tensor, got shape {tuple(values.shape)}")
    if not 0 <= discount <= 1:
        raise ValueError("discount must be in [0, 1]")
    if discount == 0:
        return values.clone()
    output = torch.empty_like(values)
    running = torch.zeros((), dtype=values.dtype, device=values.device)
    for index in range(len(values) - 1, -1, -1):
        running = values[index] + discount * running
        output[index] = running
    return output


def apply_reference_reverse_kl(
    datums: Sequence[Any],
    teacher_action_logprobs: Sequence[Sequence[float]],
    *,
    kl_coef: float,
    kl_discount_factor: float,
) -> dict[str, float]:
    """Add the OPCT reverse-KL signal to RL datum advantages in place.

    ``teacher_action_logprobs[i]`` contains only completion-token scores.  The
    datum tensors also contain prompt positions, so the action mask is the
    authoritative alignment.  Shape mismatches fail instead of silently
    training the wrong token positions.
    """

    if len(datums) != len(teacher_action_logprobs):
        raise ValueError(
            "datums and teacher_action_logprobs must have the same length, got "
            f"{len(datums)} and {len(teacher_action_logprobs)}"
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
        sampled = datum.loss_fn_inputs["logprobs"].to_torch().float()
        mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        existing_advantages = datum.loss_fn_inputs["advantages"].to_torch().float()
        if sampled.shape != mask.shape or sampled.shape != existing_advantages.shape:
            raise ValueError(
                f"datum {index} has inconsistent logprobs/mask/advantages shapes: "
                f"{tuple(sampled.shape)}, {tuple(mask.shape)}, {tuple(existing_advantages.shape)}"
            )
        sampled_actions = sampled[mask]
        teacher = torch.as_tensor(raw_teacher, dtype=sampled.dtype)
        if len(sampled_actions) != len(teacher):
            raise ValueError(
                f"datum {index} has {len(sampled_actions)} action tokens but " f"{len(teacher)} teacher logprobs"
            )
        if not len(teacher):
            raise ValueError(f"datum {index} has no action tokens")
        if not torch.isfinite(sampled_actions).all() or not torch.isfinite(teacher).all():
            raise ValueError(f"datum {index} contains non-finite student or teacher logprobs")

        reverse_kl = sampled_actions - teacher
        per_token_signal = torch.zeros_like(sampled)
        per_token_signal[mask] = -kl_coef * reverse_kl
        if kl_discount_factor > 0:
            per_token_signal = discounted_future_sum(per_token_signal, kl_discount_factor)
        # Prompt positions can receive discounted values, but the loss mask keeps
        # them out of the policy-gradient objective.
        updated = existing_advantages + per_token_signal
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(updated)

        reverse_kl_values.append(reverse_kl)
        student_values.append(sampled_actions)
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
            raise ValueError(f"row {row_index} {field}[{message_index}] must be an object")
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
            raise ValueError(f"row {index} must be a JSON object")
        _validated_messages(sample, config.reference_messages_field, index)
        _validated_messages(sample, config.variant_messages_field, index)
    return validated


class OPCTTrainer:
    """Backend-agnostic, fully on-policy consistency trainer."""

    def __init__(
        self,
        *,
        config: OPCTConfig,
        backend: TrainingBackend,
        resume_from: Optional[str] = None,
        resume_with_optimizer: Optional[bool] = None,
    ):
        self.config = config
        self.backend = backend
        self.resume_from = resume_from
        self.resume_with_optimizer = resume_with_optimizer
        self.renderer: Any = None
        self.tokenizer: Any = None
        self.sampling_client: SamplerHandle | None = None
        self.setup_done = False

    def setup(self) -> None:
        if self.config.lora.seed is not None:
            random.seed(self.config.lora.seed)
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
        self.setup_done = True

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

    async def _build_batch(self, batch: Sequence[dict]):
        assert self.sampling_client is not None
        pairs = []
        for row_index, sample in enumerate(batch, start=1):
            reference_messages = _validated_messages(sample, self.config.reference_messages_field, row_index)
            variant_messages = _validated_messages(sample, self.config.variant_messages_field, row_index)
            pairs.append(
                (
                    self.renderer.build_generation_prompt(reference_messages),
                    self.renderer.build_generation_prompt(variant_messages),
                )
            )

        sampled_groups = await asyncio.gather(
            *[
                self.sampling_client.sample(
                    variant_prompt,
                    max_tokens=self.config.generation.max_new_tokens,
                    temperature=self.config.generation.temperature,
                    stop=self.renderer.get_stop_sequences(),
                    num_samples=self.config.generation.rollouts_per_prompt,
                )
                for _, variant_prompt in pairs
            ]
        )

        datums = []
        references = []
        completions: list[list[int]] = []
        response_lengths: list[int] = []
        for pair_index, ((reference_prompt, variant_prompt), sequences) in enumerate(zip(pairs, sampled_groups)):
            if len(sequences) != self.config.generation.rollouts_per_prompt:
                raise RuntimeError(
                    f"student sampler returned {len(sequences)} rollouts for pair {pair_index}; "
                    f"expected {self.config.generation.rollouts_per_prompt}"
                )
            for rollout_index, sequence in enumerate(sequences):
                tokens = list(sequence.tokens)
                if not tokens:
                    raise RuntimeError(
                        f"student returned an empty completion for pair {pair_index}, rollout {rollout_index}"
                    )
                if sequence.logprobs is None or len(sequence.logprobs) != len(tokens):
                    raise RuntimeError(
                        f"student returned missing or misaligned logprobs for pair {pair_index}, "
                        f"rollout {rollout_index}"
                    )
                logprobs = list(sequence.logprobs)
                datums.append(self._create_datum(variant_prompt, tokens, logprobs))
                references.append(reference_prompt)
                completions.append(tokens)
                response_lengths.append(len(tokens))

        teacher_logprobs = await self.backend.score_reference_completions(references, completions)
        kl_metrics = apply_reference_reverse_kl(
            datums,
            teacher_logprobs,
            kl_coef=self.config.kl_coef,
            kl_discount_factor=self.config.kl_discount_factor,
        )
        return datums, kl_metrics, response_lengths

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
            for epoch in range(self.config.n_epochs):
                epoch_samples = list(samples)
                random.shuffle(epoch_samples)
                batch_starts = list(range(0, len(epoch_samples), self.config.batch_size))
                pbar = tqdm(batch_starts, desc=f"Epoch {epoch + 1}")
                for microbatch_index, start in enumerate(pbar):
                    batch = epoch_samples[start : start + self.config.batch_size]
                    datums, kl_metrics, response_lengths = await self._build_batch(batch)
                    current_lr = learning_rate(global_step)
                    should_step = (
                        microbatch_index + 1
                    ) % self.config.gradient_accumulation_steps == 0 or microbatch_index + 1 == len(batch_starts)

                    pending_fwd_bwd = await self.backend.submit_forward_backward(
                        datums,
                        loss_fn=self.config.loss_fn,
                    )
                    pending_optim = None
                    if should_step:
                        pending_optim = await self.backend.submit_optim_step(
                            learning_rate=current_lr,
                            adam=self.config.optimizer,
                        )
                    fwd_bwd = await pending_fwd_bwd.result()
                    if pending_optim is not None:
                        await pending_optim.result()
                        global_step += 1
                        # OPCT's defining guarantee: every post-update rollout is
                        # sampled from newly published current-policy weights.
                        self.sampling_client = await self.backend.refresh_policy_sampler(
                            name=(
                                f"{self.config.experiment_name}_{self.config.run_name}_" f"opct_sampler_{global_step}"
                            )
                        )

                    global_microbatch += 1
                    metrics = {
                        **{f"train/{key}": value for key, value in kl_metrics.items()},
                        **{f"train/{key}": value for key, value in fwd_bwd.metrics.items()},
                        "train/lr": current_lr,
                        "train/optimizer_step": global_step,
                        "train/n_rollouts": len(datums),
                        "train/avg_response_length": sum(response_lengths) / len(response_lengths),
                        "train/epoch": epoch,
                    }
                    logger.log_metrics(metrics, step=global_microbatch)
                    pbar.set_postfix(
                        {
                            "teacher_kl": f"{kl_metrics['teacher_kl']:.4f}",
                            "loss": f"{fwd_bwd.metrics.get('loss', float('nan')):.4f}",
                            "step": global_step,
                        }
                    )

                    if should_step:
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
            except Exception:
                pass
            raise
        finally:
            shutdown = getattr(self.backend, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    _log.warning("Backend shutdown failed:\n%s", traceback.format_exc())


__all__ = [
    "OPCTConfig",
    "OPCTGenerationConfig",
    "OPCTTrainer",
    "apply_reference_reverse_kl",
    "discounted_future_sum",
    "validate_opct_samples",
]
