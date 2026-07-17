"""The Setting protocol — a pluggable phenomenon to consistency-train against.

A Setting bundles everything phenomenon-specific in ONE place, formalizing the
injection pattern the training loops already use (``perturbation_fns`` +
``trait_classifier``):

- ``load_datapoints``    — training datapoints (dicts; schema is setting-internal)
- ``perturbations``      — index 0 = reference (neutral) prompt builder, 1..N = cued
- ``trait_classifier``   — (answer_text, datapoint, realized_messages) -> float in [0,1] or None to abstain
- ``answer_parser``      — optional; gates rollouts on a committed answer

Evaluation is intentionally separate: experiment configs select upstream task
factories independently of the training setting.
"""

from typing import Callable, Optional, Protocol, runtime_checkable

from ctm.importing import load_callable


@runtime_checkable
class Setting(Protocol):
    name: str

    def load_datapoints(self, **kwargs) -> list[dict]:
        """Training datapoints. Kwargs are setting-specific (paths, sizes, filters)."""
        ...

    def perturbations(self) -> list[Callable[[dict], dict]]:
        """Prompt builders: index 0 = reference/neutral, 1..N = cued variants.
        Each maps a datapoint to {"messages": [...]}."""
        ...

    def training_perturbation_indices(self) -> list[int]:
        """Which perturbation indices receive the consistency gradient (default 1..N)."""
        ...

    def trait_classifier(self) -> Callable[[str, dict, list[dict]], float | None]:
        """Classify the realized response; None excludes an unjudgeable rollout."""
        ...

    def answer_parser(self) -> Optional[Callable[[str], Optional[str]]]:
        """Optional parser gating rollouts on a committed answer (None = accept all)."""
        ...


def create_setting(factory_spec: str, **kwargs) -> Setting:
    """Construct a setting from an explicit ``module:callable`` factory."""

    setting = load_callable(factory_spec, label="setting_factory")(**kwargs)
    if not isinstance(setting, Setting):
        raise TypeError(
            f"setting factory {factory_spec!r} returned an object that does not satisfy the Setting protocol"
        )
    return setting


__all__ = ["Setting", "create_setting"]
