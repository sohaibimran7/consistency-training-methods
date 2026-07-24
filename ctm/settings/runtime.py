"""Validated hand-off from a pluggable Setting to the RL trainer."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ctm.settings.base import Setting, create_setting


@dataclass(frozen=True)
class PreparedSetting:
    """All setting-owned runtime components needed by ``RLTrainer.train``."""

    setting: Setting
    datapoints: list[dict]
    perturbations: list[Callable[[dict], dict]]
    training_indices: list[int]
    trait_classifier: Callable
    answer_parser: Optional[Callable]


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


def _callable_identity(value: Callable) -> dict[str, Any]:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    identity: dict[str, Any] = {"callable": f"{module}.{qualname}"}
    provenance = getattr(value, "provenance", None)
    if callable(provenance):
        identity["provenance"] = _json_value(provenance())
    return identity


def _validate_prompt_result(value: Any, *, setting_name: str, perturbation_index: int, datapoint_index: int) -> None:
    location = f"setting {setting_name!r} perturbation {perturbation_index} " f"for datapoint {datapoint_index}"
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must return an object with messages")
    messages = value.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        raise TypeError(f"{location}.messages must be a non-empty message list")
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"{location}.messages[{message_index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip() or not isinstance(content, str) or not content.strip():
            raise TypeError(f"{location}.messages[{message_index}] must contain non-empty string role/content fields")


def setting_run_metadata(
    setting: Setting,
    *,
    setting_config: Mapping[str, Any] | None = None,
    load_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serializable setting provenance for the shared run manifest."""

    metadata: dict[str, Any] = {
        "setting": setting.name,
        "setting_config": _json_value(dict(setting_config or {})),
        "load_config": _json_value(dict(load_config or {})),
    }
    describe = getattr(setting, "run_metadata", None)
    if callable(describe):
        setting_metadata = describe()
        if not isinstance(setting_metadata, Mapping):
            raise TypeError(f"setting {setting.name!r} run_metadata() must return an object")
        metadata["setting_metadata"] = _json_value(dict(setting_metadata))
    metadata["trait_classifier_identity"] = _callable_identity(setting.trait_classifier())
    identity = getattr(setting, "training_artifact_identity", None)
    if callable(identity):
        metadata["training_artifacts"] = _json_value(identity())
    return metadata


def prepare_setting_instance(
    setting: Setting,
    *,
    load_config: Mapping[str, Any] | None = None,
) -> PreparedSetting:
    """Load and validate one setting before backend setup or paid sampling."""

    datapoints = list(setting.load_datapoints(**dict(load_config or {})))
    if not datapoints:
        raise ValueError(f"setting {setting.name!r} loaded no datapoints")
    perturbations = list(setting.perturbations())
    if len(perturbations) < 2:
        raise ValueError(f"setting {setting.name!r} needs a reference plus at least one training perturbation")
    if any(not callable(perturbation) for perturbation in perturbations):
        raise TypeError(f"setting {setting.name!r} returned a non-callable perturbation")
    # Exercise prompt construction before initializing a paid/accelerator
    # backend. Deep copies prevent a stateful custom perturbation from mutating
    # the datapoints that will later be used for training.
    for datapoint_index, datapoint in enumerate(datapoints):
        for perturbation_index, perturbation in enumerate(perturbations):
            prompt = perturbation(copy.deepcopy(datapoint))
            _validate_prompt_result(
                prompt,
                setting_name=setting.name,
                perturbation_index=perturbation_index,
                datapoint_index=datapoint_index,
            )

    training_indices = list(setting.training_perturbation_indices())
    if not training_indices:
        raise ValueError(f"setting {setting.name!r} returned no training perturbation indices")
    if len(training_indices) != len(set(training_indices)):
        raise ValueError(f"setting {setting.name!r} returned duplicate training perturbation indices")
    invalid = [index for index in training_indices if index < 1 or index >= len(perturbations)]
    if invalid:
        raise ValueError(
            f"setting {setting.name!r} returned invalid training indices {invalid}; "
            f"valid variant indices are 1..{len(perturbations) - 1}"
        )

    classifier = setting.trait_classifier()
    if not callable(classifier):
        raise TypeError(f"setting {setting.name!r} returned a non-callable trait classifier")
    parser = setting.answer_parser()
    if parser is not None and not callable(parser):
        raise TypeError(f"setting {setting.name!r} returned a non-callable answer parser")
    return PreparedSetting(
        setting=setting,
        datapoints=datapoints,
        perturbations=perturbations,
        training_indices=training_indices,
        trait_classifier=classifier,
        answer_parser=parser,
    )


def prepare_setting(
    factory_spec: str,
    *,
    setting_config: Mapping[str, Any] | None = None,
    load_config: Mapping[str, Any] | None = None,
) -> PreparedSetting:
    """Construct an explicitly selected setting, then prepare its components."""

    return prepare_setting_instance(
        create_setting(factory_spec, **dict(setting_config or {})),
        load_config=load_config,
    )
