"""Presentation metadata for chart-ready mcq-bias results.

Scientific identity remains in the experiment configuration and result rows.
This registry only supplies reusable labels, ordering, and visual styles.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PresentationRegistry:
    models: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    biases: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    training_types: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    methods: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    ordering: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


DEFAULT_REGISTRY_PATH = Path(__file__).with_name("plot_registry.toml")


def load_presentation_registry(path: str | Path | None = None) -> PresentationRegistry:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not registry_path.is_file():
        if path is None:
            return PresentationRegistry()
        raise ValueError(f"presentation registry does not exist: {registry_path}")
    try:
        with registry_path.open("rb") as handle:
            document = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid presentation registry {registry_path}: {exc}") from exc
    allowed = {"models", "biases", "training_types", "methods", "ordering"}
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(f"presentation registry has unknown sections: {unknown}")
    sections: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    for section in ("models", "biases", "training_types", "methods"):
        value = document.get(section, {})
        if not isinstance(value, Mapping) or any(not isinstance(item, Mapping) for item in value.values()):
            raise ValueError(f"presentation registry section {section!r} must contain tables")
        sections[section] = value
    ordering = document.get("ordering", {})
    if not isinstance(ordering, Mapping):
        raise TypeError("presentation registry ordering must be a table")
    normalized_ordering = {}
    for key, value in ordering.items():
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"presentation registry ordering.{key} must be an array of strings")
        normalized_ordering[str(key)] = tuple(value)
    return PresentationRegistry(**sections, ordering=normalized_ordering)


def registry_labels(entries: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(key): str(value["display_name"])
        for key, value in entries.items()
        if isinstance(value.get("display_name"), str)
    }
