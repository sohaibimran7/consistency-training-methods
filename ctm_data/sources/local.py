"""Explicit local JSON, JSONL, CSV, and TSV row loading."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ctm_data.sources.base import LoadedRows, SourceIdentity, SourceRowError, materialize_mapping_rows

LocalFormat = Literal["json", "jsonl", "csv", "tsv"]
SUPPORTED_LOCAL_FORMATS: frozenset[str] = frozenset({"json", "jsonl", "csv", "tsv"})


@dataclass(frozen=True, slots=True)
class LocalSource:
    """One local file plus its explicitly selected serialization format."""

    path: str | Path
    format: LocalFormat
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.format not in SUPPORTED_LOCAL_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_LOCAL_FORMATS))
            raise ValueError(f"unsupported local format {self.format!r}; expected one of: {supported}")
        if not isinstance(self.encoding, str) or not self.encoding.strip():
            raise ValueError("encoding must be a non-empty string")

    @property
    def identity(self) -> SourceIdentity:
        return SourceIdentity(
            loader="local",
            location=str(cast(Path, self.path).resolve()),
            format=self.format,
        )

    def load(self) -> LoadedRows:
        """Decode the complete file without schema conversion or filtering."""

        path = cast(Path, self.path)
        if not path.is_file():
            raise FileNotFoundError(f"local source is not a file: {path}")

        if self.format == "json":
            raw_rows = _read_json(path, encoding=self.encoding)
        elif self.format == "jsonl":
            raw_rows = _read_jsonl(path, encoding=self.encoding)
        else:
            delimiter = "," if self.format == "csv" else "\t"
            raw_rows = _read_delimited(path, encoding=self.encoding, delimiter=delimiter, format_name=self.format)

        identity = self.identity
        rows = materialize_mapping_rows(raw_rows, source=identity)
        return LoadedRows(rows=rows, source=identity)


def load_local_rows(
    path: str | Path,
    *,
    format: LocalFormat,
    encoding: str = "utf-8",
) -> LoadedRows:
    """Load an explicitly formatted local file and return rows plus identity."""

    return LocalSource(path=path, format=format, encoding=encoding).load()


def _read_json(path: Path, *, encoding: str) -> list[Any]:
    try:
        with path.open("r", encoding=encoding) as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SourceRowError(
            f"local JSON source {str(path)!r} is invalid at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, list):
        raise SourceRowError(f"local JSON source {str(path)!r} must contain a top-level array of rows")
    return value


def _read_jsonl(path: Path, *, encoding: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding=encoding, newline=None) as handle:
        for line_number, physical_line in enumerate(handle, start=1):
            # TextIO recognizes only physical newline conventions here. Avoid
            # str.splitlines(), which also splits JSON strings at U+2028/U+2029.
            candidate = physical_line.rstrip("\r\n")
            if not candidate.strip(" \t"):
                continue
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise SourceRowError(
                    f"local JSONL source {str(path)!r} line {line_number} is invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, Mapping):
                raise SourceRowError(f"local JSONL source {str(path)!r} line {line_number} must decode to an object")
            rows.append(value)
    return rows


def _read_delimited(
    path: Path,
    *,
    encoding: str,
    delimiter: str,
    format_name: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SourceRowError(f"local {format_name.upper()} source {str(path)!r} has no header row") from exc
        except csv.Error as exc:
            raise SourceRowError(
                f"local {format_name.upper()} source {str(path)!r} has an invalid header: {exc}"
            ) from exc

        if not header or any(not column.strip() for column in header):
            raise SourceRowError(f"local {format_name.upper()} source {str(path)!r} header names must be non-empty")
        duplicates = sorted({column for column in header if header.count(column) > 1})
        if duplicates:
            raise SourceRowError(
                f"local {format_name.upper()} source {str(path)!r} has duplicate header names: {duplicates}"
            )

        try:
            for row_number, values in enumerate(reader, start=2):
                if not values:
                    continue
                if len(values) != len(header):
                    raise SourceRowError(
                        f"local {format_name.upper()} source {str(path)!r} row {row_number} has "
                        f"{len(values)} values for {len(header)} columns"
                    )
                rows.append(dict(zip(header, values, strict=True)))
        except csv.Error as exc:
            raise SourceRowError(
                f"local {format_name.upper()} source {str(path)!r} is invalid near line {reader.line_num}: {exc}"
            ) from exc
    return rows


__all__ = ["SUPPORTED_LOCAL_FORMATS", "LocalFormat", "LocalSource", "load_local_rows"]
