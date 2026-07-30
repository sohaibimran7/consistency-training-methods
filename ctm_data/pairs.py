"""Canonical reference/variant prompt-pair row validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

REFERENCE_MESSAGES_FIELD = "reference_messages"
VARIANT_MESSAGES_FIELD = "variant_messages"
PAIR_MESSAGE_FIELDS = (REFERENCE_MESSAGES_FIELD, VARIANT_MESSAGES_FIELD)


class PairRowError(ValueError):
    """Raised when a canonical paired-prompt row is malformed."""


def canonical_pair_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy one canonical pair while retaining every extra field."""

    return _canonical_pair_row(row, location="pair row")


def canonical_pair_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and copy an iterable of canonical paired-prompt rows."""

    if isinstance(rows, (str, bytes, bytearray, Mapping)):
        raise PairRowError("pair rows must be an iterable of mappings")
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise PairRowError("pair rows must be an iterable of mappings") from exc
    return [_canonical_pair_row(row, location=f"pair row {index}") for index, row in enumerate(iterator, start=1)]


def make_pair_row(
    *,
    reference_messages: Sequence[Mapping[str, Any]],
    variant_messages: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a canonical pair with optional flattened caller metadata."""

    if metadata is None:
        row: dict[str, Any] = {}
    elif not isinstance(metadata, Mapping):
        raise PairRowError("pair metadata must be a mapping")
    else:
        row = dict(metadata)
    conflicts = sorted(set(PAIR_MESSAGE_FIELDS).intersection(row))
    if conflicts:
        raise PairRowError(f"pair metadata cannot override canonical message fields: {conflicts}")
    row[REFERENCE_MESSAGES_FIELD] = reference_messages
    row[VARIANT_MESSAGES_FIELD] = variant_messages
    return canonical_pair_row(row)


def _canonical_pair_row(row: Mapping[str, Any], *, location: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise PairRowError(f"{location} must be a mapping")
    normalized = dict(row)
    for field in PAIR_MESSAGE_FIELDS:
        if field not in row:
            raise PairRowError(f"{location} is missing {field!r}")
        messages = row[field]
        if isinstance(messages, (str, bytes, bytearray, Mapping)) or not isinstance(messages, Sequence):
            raise PairRowError(f"{location} {field!r} must be a non-empty sequence of message mappings")
        if not messages:
            raise PairRowError(f"{location} {field!r} must not be empty")

        copied_messages: list[dict[str, Any]] = []
        for message_index, message in enumerate(messages, start=1):
            if not isinstance(message, Mapping):
                raise PairRowError(f"{location} {field!r} message {message_index} must be a mapping")
            copied_messages.append(dict(message))
        normalized[field] = copied_messages
    return normalized


__all__ = [
    "PAIR_MESSAGE_FIELDS",
    "REFERENCE_MESSAGES_FIELD",
    "VARIANT_MESSAGES_FIELD",
    "PairRowError",
    "canonical_pair_row",
    "canonical_pair_rows",
    "make_pair_row",
]
