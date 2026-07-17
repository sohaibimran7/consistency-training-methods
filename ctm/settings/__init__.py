"""Pluggable consistency-training settings.

A Setting bundles only the training datapoints, prompt perturbations, trait
classifier, and answer parser. Evaluation is selected independently.

    from ctm.settings import create_setting
    setting = create_setting("my_project.adapters:create_setting", artifact_path="data/train.jsonl")
    trainer.train(datapoints=setting.load_datapoints(n_datapoints=64),
                  perturbation_fns=setting.perturbations(),
                  trait_classifier=setting.trait_classifier(),
                  answer_parser=setting.answer_parser())
"""

from ctm.settings.base import Setting, create_setting

__all__ = ["Setting", "create_setting"]
