"""MCQ-bias presentation-registry compatibility wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ctm_data.adapters._plot_registry import PresentationRegistry as _PresentationRegistry
from ctm_data.adapters._plot_registry import load_presentation_registry as _load
from ctm_data.adapters._plot_registry import registry_labels

DEFAULT_REGISTRY_PATH = Path(__file__).with_name("plot_registry.toml")


@dataclass(frozen=True)
class PresentationRegistry(_PresentationRegistry):
    """Backwards-compatible MCQ registry with a named ``biases`` view."""

    @property
    def biases(self) -> Any:
        return self.categories


def load_presentation_registry(path: str | Path | None = None) -> PresentationRegistry:
    registry = _load(path, default_path=DEFAULT_REGISTRY_PATH, category_section="biases")
    return PresentationRegistry(
        models=registry.models,
        categories=registry.categories,
        training_types=registry.training_types,
        methods=registry.methods,
        ordering=registry.ordering,
    )


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "PresentationRegistry",
    "load_presentation_registry",
    "registry_labels",
]
