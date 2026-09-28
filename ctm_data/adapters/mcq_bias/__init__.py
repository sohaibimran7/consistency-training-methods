"""Trainer-facing sycophancy adapter for native mcq-bias rows.

Keep the optional trainer/parser integration lazy so analysis-only modules
(notably the publication renderer) can be imported in a plotting environment
that does not install the external ``mcq_bias`` task package.
"""

from __future__ import annotations

import importlib
from typing import Any

from ctm_data.adapters.mcq_bias.data import file_identity, load_paths, make_perturbation_fns


def create_setting(**kwargs: Any) -> Any:
    from ctm_data.adapters.mcq_bias.setting import SycophancySetting

    return SycophancySetting(**kwargs)


def __getattr__(name: str) -> Any:
    if name in {"SycophancySetting", "trait_classifier", "MCQCorrectnessPairSetting", "mcq_correctness_pair_setting"}:
        setting = importlib.import_module("ctm_data.adapters.mcq_bias.setting")
        return getattr(setting, name)
    raise AttributeError(name)


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
