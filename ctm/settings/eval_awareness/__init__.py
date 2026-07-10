"""Eval-awareness setting: eval-cue consistency training (see setting.py)."""

from ctm.settings.eval_awareness.cues import (
    EVAL_MARKERS,
    TRAIN_CUE,
    apply_eval_marker,
    apply_train_cue,
    natural,
)
from ctm.settings.eval_awareness.judge import misalignment_judge
from ctm.settings.eval_awareness.setting import EvalAwarenessSetting

__all__ = [
    "EVAL_MARKERS",
    "EvalAwarenessSetting",
    "TRAIN_CUE",
    "apply_eval_marker",
    "apply_train_cue",
    "misalignment_judge",
    "natural",
]
