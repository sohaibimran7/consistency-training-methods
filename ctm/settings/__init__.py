"""Pluggable settings (sycophancy, eval-awareness, ...).

A Setting bundles a phenomenon's train datapoints, named perturbations (cues),
trait classifier, grader prompts, and in-domain Inspect tasks — formalizing the
injection pattern the training loops already use. See ctm.settings.base.

    from ctm.settings import get_setting
    setting = get_setting("sycophancy", bias_types=["wrong_few_shot"], prompt_style="no_cot")
    trainer.train(datapoints=setting.load_datapoints(n_datapoints=64),
                  perturbation_fns=setting.perturbations(),
                  trait_classifier=setting.trait_classifier(),
                  answer_parser=setting.answer_parser())
"""

from ctm.settings.base import Setting, get_setting, register_setting

__all__ = ["Setting", "get_setting", "register_setting"]
