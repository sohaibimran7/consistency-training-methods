"""Sycophancy setting: bias cues in MCQ prompts (the BCT bias battery).

Single source of truth for the phenomenon — training data loading, cue
perturbations, the trait classifier, BA grader prompts, and the in-domain
Inspect tasks all live here. The eval scorer and data-gen scripts import from
this package (old import paths re-export during the migration).
"""

from ctm.settings.sycophancy.classifier import trait_classifier
from ctm.settings.sycophancy.data import (
    attach_wrong_cots,
    default_data_dir,
    load_datapoints,
    make_distractor_cue_perturbations,
    make_perturbation_fns,
    resolve_distractor_cues,
)
from ctm.settings.sycophancy.grader_prompts import (
    BIAS_ACK_PROMPTS,
    get_bias_ack_prompt,
    register_bias_ack_prompt,
)
from ctm.settings.sycophancy.setting import SycophancySetting

__all__ = [
    "BIAS_ACK_PROMPTS",
    "SycophancySetting",
    "attach_wrong_cots",
    "default_data_dir",
    "get_bias_ack_prompt",
    "load_datapoints",
    "make_distractor_cue_perturbations",
    "make_perturbation_fns",
    "register_bias_ack_prompt",
    "resolve_distractor_cues",
    "trait_classifier",
]
