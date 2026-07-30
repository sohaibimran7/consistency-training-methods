"""Policy-neutral interfaces shared by concrete row sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class SourceRowError(ValueError):
    """Raised when a source does not yield mapping-shaped rows."""


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Serializable identity for one explicit source selection.

    ``location`` is either a local path or a remote dataset identifier. The
    remaining fields record selection arguments without interpreting them.
    """

    loader: str
    location: str
    format: str | None = None
    config: str | None = None
    split: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("loader", "location"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"source {field_name} must be a non-empty string")
        for field_name in ("format", "config", "split", "revision"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"source {field_name} must be None or a non-empty string")

    def as_dict(self) -> dict[str, str]:
        """Return the identity as a plain mapping, omitting unused fields."""

        values = {
            "loader": self.loader,
            "location": self.location,
            "format": self.format,
            "config": self.config,
            "split": self.split,
            "revision": self.revision,
        }
        return {key: value for key, value in values.items() if value is not None}

    @property
    def label(self) -> str:
        """Short label used in deterministic validation errors."""

        return f"{self.loader} source {self.location!r}"


@dataclass(frozen=True, slots=True)
class LoadedRows:
    """Materialized plain rows together with the source that produced them."""

    rows: list[dict[str, Any]]
    source: SourceIdentity


@runtime_checkable
class RowSource(Protocol):
    """Structural interface implemented by local and remote row sources."""

    @property
    def identity(self) -> SourceIdentity:
        """Return the complete source selection identity."""

        ...

    def load(self) -> LoadedRows:
        """Materialize this source as plain mapping rows."""

        ...


def materialize_mapping_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: SourceIdentity,
) -> list[dict[str, Any]]:
    """Copy an iterable of mappings into plain dictionaries without filtering."""

    if isinstance(rows, (str, bytes, bytearray, Mapping)):
        raise SourceRowError(f"{source.label} must yield an iterable of row mappings")
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise SourceRowError(f"{source.label} must yield an iterable of row mappings") from exc

    materialized: list[dict[str, Any]] = []
    for index, row in enumerate(iterator, start=1):
        if not isinstance(row, Mapping):
            raise SourceRowError(f"{source.label} row {index} must be a mapping, got {type(row).__name__}")
        copied = dict(row)
        for key in copied:
            if not isinstance(key, str):
                raise SourceRowError(f"{source.label} row {index} has a non-string column name: {key!r}")
        materialized.append(copied)
    return materialized


__all__ = [
    "LoadedRows",
    "RowSource",
    "SourceIdentity",
    "SourceRowError",
    "materialize_mapping_rows",
]
