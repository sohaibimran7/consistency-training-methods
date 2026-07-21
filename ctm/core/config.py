"""Shared training configuration classes (backend-agnostic).

Canonical home of the config models previously defined in
``cot_transparency.apis.tinker.common`` (which now re-exports from here).
"""

from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class LoRAConfig(BaseModel):
    """LoRA adapter configuration.

    The component flags are the portable Tinker interface. ``target_modules``
    is the more precise local/PEFT interface: when supplied it replaces the
    component flags and selects modules by full-name glob or terminal module
    name (for example ``["q_proj", "v_proj"]``).
    """

    rank: int = 32
    alpha: Optional[int] = None
    dropout: float = 0.0
    target_modules: Optional[list[str]] = None
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    seed: Optional[int] = None

    @field_validator("rank")
    @classmethod
    def _positive_rank(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("rank must be a positive integer")
        return value

    @field_validator("alpha")
    @classmethod
    def _positive_alpha(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and (isinstance(value, bool) or value < 1):
            raise ValueError("alpha must be a positive integer")
        return value

    @field_validator("dropout")
    @classmethod
    def _valid_dropout(cls, value: float) -> float:
        if isinstance(value, bool) or not 0.0 <= value < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return value

    @field_validator("target_modules")
    @classmethod
    def _valid_targets(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return None
        cleaned = [item.strip() for item in value]
        if not cleaned or any(not item for item in cleaned):
            raise ValueError("target_modules must contain non-empty module names")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("target_modules must not contain duplicates")
        return cleaned

    @property
    def resolved_alpha(self) -> int:
        """PEFT alpha, preserving the historical alpha/r=2 default."""

        return self.alpha if self.alpha is not None else 2 * self.rank


def resolve_lora_config(
    raw: dict,
    *,
    rank: int | None = None,
    seed: int | None = None,
    default_rank: int = 8,
) -> LoRAConfig:
    """Build a portable LoRA configuration with optional scalar overrides.

    Experiment YAML may use the complete nested configuration. The older
    scalar ``rank`` and ``seed`` CLI flags remain explicit overrides.
    """

    unknown = sorted(set(raw) - set(LoRAConfig.model_fields))
    if unknown:
        raise ValueError(f"lora_config has unknown field(s): {unknown}")
    values = {"rank": default_rank, **raw}
    if rank is not None:
        values["rank"] = rank
    if seed is not None:
        values["seed"] = seed
    config = LoRAConfig(**values)
    if config.target_modules is None and not (config.train_mlp or config.train_attn or config.train_unembed):
        raise ValueError("lora_config must enable at least one of train_mlp, train_attn, or train_unembed")
    return config


class AdamConfig(BaseModel):
    """Adam optimizer and learning rate schedule configuration."""

    learning_rate: Optional[float] = None  # None = use get_recommended_lr(model)
    lr_schedule: Literal["constant", "linear", "cosine"] = "linear"  # shared SFT+RL default; training CLIs mirror it
    beta1: float = 0.9
    beta2: float = 0.95  # cookbook default
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0


class CheckpointConfig(BaseModel):
    """Checkpointing configuration."""

    save_every_n_steps: Optional[int] = None
    save_state: bool = False  # If True, save optimizer state for resumability
    skip_near_final_steps: int = 0  # Skip intermediate checkpoints within N steps of final
