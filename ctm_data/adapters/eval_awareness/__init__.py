"""EvalAwareBench-backed evaluation-awareness setting."""

from ctm_data.adapters.eval_awareness.data import (
    DATASET_CONFIGS,
    DATASET_ID,
    DATASET_LICENSE,
    build_prompt_families,
    materialize_eval_awareness,
)
from ctm_data.adapters.eval_awareness.setting import EvalAwarenessSetting


def create_setting(**kwargs) -> EvalAwarenessSetting:
    return EvalAwarenessSetting(**kwargs)


__all__ = [
    "DATASET_CONFIGS",
    "DATASET_ID",
    "DATASET_LICENSE",
    "EvalAwarenessSetting",
    "create_setting",
    "build_prompt_families",
    "materialize_eval_awareness",
]
