"""Trainer-facing sycophancy adapter for native mcq-bias rows."""

from ctm_data.adapters.mcq_bias.data import file_identity, load_paths, make_perturbation_fns
from ctm_data.adapters.mcq_bias.setting import (
    MCQCorrectnessPairSetting,
    SycophancySetting,
    mcq_correctness_pair_setting,
    trait_classifier,
)


def create_setting(**kwargs) -> SycophancySetting:
    return SycophancySetting(**kwargs)


__all__ = [
    "MCQCorrectnessPairSetting",
    "SycophancySetting",
    "create_setting",
    "file_identity",
    "load_paths",
    "make_perturbation_fns",
    "mcq_correctness_pair_setting",
    "trait_classifier",
]
