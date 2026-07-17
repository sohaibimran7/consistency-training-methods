"""Reusable refusal trait for consistency training."""

from ctm.training.refusal.judge import (
    DEFAULT_REFUSAL_GENERATION_CONFIG,
    DEFAULT_REFUSAL_MODEL,
    RefusalJudge,
    RefusalJudgeError,
    normalize_refusal_judge_options,
)

__all__ = [
    "DEFAULT_REFUSAL_GENERATION_CONFIG",
    "DEFAULT_REFUSAL_MODEL",
    "RefusalJudge",
    "RefusalJudgeError",
    "normalize_refusal_judge_options",
]
