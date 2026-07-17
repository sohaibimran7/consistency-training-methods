"""Trainer-facing sycophancy adapter for native mcq-bias rows."""

from ctm_data.adapters.mcq_bias.data import file_identity, load_paths, make_perturbation_fns
from ctm_data.adapters.mcq_bias.setting import SycophancySetting, trait_classifier


def create_setting(**kwargs) -> SycophancySetting:
    return SycophancySetting(**kwargs)


__all__ = [
    "SycophancySetting",
    "create_setting",
    "file_identity",
    "load_paths",
    "make_perturbation_fns",
    "trait_classifier",
]
