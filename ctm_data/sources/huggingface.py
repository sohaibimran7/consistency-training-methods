"""Explicit, lazily imported Hugging Face dataset loading."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from ctm_data.sources.base import LoadedRows, SourceIdentity, materialize_mapping_rows


@dataclass(frozen=True, slots=True)
class HuggingFaceSource:
    """A fully specified Hugging Face dataset/config/split/revision selection."""

    dataset: str
    config: str
    split: str
    revision: str

    def __post_init__(self) -> None:
        for field_name in ("dataset", "config", "split", "revision"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Hugging Face {field_name} must be a non-empty string")

    @property
    def identity(self) -> SourceIdentity:
        return SourceIdentity(
            loader="huggingface",
            location=self.dataset,
            config=self.config,
            split=self.split,
            revision=self.revision,
        )

    def load(self) -> LoadedRows:
        """Materialize every selected row as a plain dictionary."""

        try:
            datasets = import_module("datasets")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "HuggingFaceSource requires the optional 'datasets' package; install project requirements"
            ) from exc
        load_dataset = getattr(datasets, "load_dataset", None)
        if not callable(load_dataset):
            raise TypeError("the imported 'datasets' package does not provide callable load_dataset")

        selected = load_dataset(
            self.dataset,
            self.config,
            split=self.split,
            revision=self.revision,
        )
        identity = self.identity
        rows = materialize_mapping_rows(selected, source=identity)
        return LoadedRows(rows=rows, source=identity)


def load_huggingface_rows(
    dataset: str,
    *,
    config: str,
    split: str,
    revision: str,
) -> LoadedRows:
    """Load one fully specified Hugging Face selection without filtering it."""

    return HuggingFaceSource(dataset=dataset, config=config, split=split, revision=revision).load()


__all__ = ["HuggingFaceSource", "load_huggingface_rows"]
