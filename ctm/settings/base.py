"""The Setting protocol — a pluggable phenomenon to consistency-train against.

A Setting bundles everything phenomenon-specific in ONE place, formalizing the
injection pattern the training loops already use (``perturbation_fns`` +
``trait_classifier``):

- ``load_datapoints``    — training datapoints (dicts; schema is setting-internal)
- ``perturbations``      — index 0 = reference (neutral) prompt builder, 1..N = cued
- ``trait_classifier``   — (answer_text, datapoint) -> float in [0,1]
- ``answer_parser``      — optional; gates rollouts on a committed answer
- ``tasks``              — the setting's in-domain Inspect tasks

The training loops never import a setting's internals; scripts construct a
concrete Setting (or use ``get_setting``) and hand its pieces to the trainer.
"""

from typing import Any, Callable, Optional, Protocol, runtime_checkable


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

    def trait_classifier(self) -> Callable[[str, dict], Any]:
        """(answer_text, datapoint) -> trait in [0,1]; may be sync or async."""
        ...

    def answer_parser(self) -> Optional[Callable[[str], Optional[str]]]:
        """Optional parser gating rollouts on a committed answer (None = accept all)."""
        ...

    def tasks(self, **kwargs) -> list:
        """The setting's in-domain Inspect tasks (kwargs are setting-specific)."""
        ...


_SETTING_FACTORIES: dict[str, Callable[..., Setting]] = {}


def register_setting(name: str, factory: Callable[..., Setting]) -> None:
    _SETTING_FACTORIES[name] = factory


def get_setting(name: str, **kwargs) -> Setting:
    """Construct a registered setting by name (kwargs go to its constructor)."""
    # Lazy imports keep heavy/optional deps out of `import ctm.settings`.
    if name not in _SETTING_FACTORIES:
        if name == "sycophancy":
            from ctm.settings.sycophancy.setting import SycophancySetting
            register_setting("sycophancy", SycophancySetting)
        elif name == "eval_awareness":
            from ctm.settings.eval_awareness.setting import EvalAwarenessSetting
            register_setting("eval_awareness", EvalAwarenessSetting)
        else:
            raise KeyError(f"Unknown setting: {name!r}. Known: {sorted(_SETTING_FACTORIES) + ['sycophancy', 'eval_awareness']}")
    return _SETTING_FACTORIES[name](**kwargs)
