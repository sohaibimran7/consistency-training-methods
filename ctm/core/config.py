"""Shared training configuration classes (backend-agnostic).

Canonical home of the config models previously defined in
``cot_transparency.apis.tinker.common`` (which now re-exports from here).
"""

from typing import Literal, Optional

from pydantic import BaseModel


class LoRAConfig(BaseModel):
    """LoRA adapter configuration."""

    rank: int = 32
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    seed: Optional[int] = None


class AdamConfig(BaseModel):
    """Adam optimizer and learning rate schedule configuration."""

    learning_rate: Optional[float] = None  # None = use get_recommended_lr(model)
    lr_schedule: Literal["constant", "linear", "cosine"] = (
        "linear"  # shared SFT+RL default; train_sft/train_rl/train_evalaware CLIs mirror it
    )
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
