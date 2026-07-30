"""Versioned row schema and stable identities for the paper reproduction artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

PAPER_ID = "irpan_2510_27062"
ARTIFACT_SCHEMA = f"ctm_data.{PAPER_ID}"
SCHEMA_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class RecordSchemaError(ValueError):
    """A reproduction row is malformed or its recorded digest is stale."""


def canonical_json(value: Any) -> str:
    """Return the one canonical JSON encoding used for all identities."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def normalize_text(value: str) -> str:
    """Normalize text for stable IDs without changing internal whitespace."""

    if not isinstance(value, str):
        raise RecordSchemaError("text values must be strings")
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_json(value: Any) -> Any:
    """Recursively normalize strings and mapping order for identity material."""

    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, Mapping):
        return {str(key): normalize_json(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [normalize_json(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise RecordSchemaError(f"unsupported JSON value for stable identity: {type(value).__name__}")


def require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RecordSchemaError(f"{field} must be a lowercase SHA-256 digest")
    return value


def stable_example_id(source: str, source_key: str, payload: Mapping[str, Any]) -> str:
    """Build a stable, content-bound ID for one upstream evaluation point."""

    source_name = _require_name(source, field="source")
    normalized_key = normalize_text(source_key)
    if not normalized_key:
        raise RecordSchemaError("source_key must be non-empty")
    digest = sha256_json({"source": source_name, "source_key": normalized_key, "payload": normalize_json(payload)})
    return f"{PAPER_ID}:{source_name}:{digest[:24]}"


def make_source_record(
    *,
    record_type: str,
    source: str,
    source_key: str,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a normalized source row whose ID survives file relocation."""

    normalized_payload = _require_mapping(payload, field="payload")
    normalized_metadata = _require_mapping(metadata or {}, field="metadata")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": _require_name(record_type, field="record_type"),
        "example_id": stable_example_id(source, source_key, normalized_payload),
        "source": _require_name(source, field="source"),
        "source_key": normalize_text(source_key),
        "parent_hashes": [],
        "payload": normalized_payload,
        "metadata": normalized_metadata,
    }
    record["content_sha256"] = _record_digest(record)
    return record


def make_derived_record(
    *,
    record_type: str,
    example_id: str,
    source: str,
    source_key: str,
    payload: Mapping[str, Any],
    parent_hashes: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a derived row with explicit row-level lineage."""

    if not isinstance(example_id, str) or not example_id.startswith(f"{PAPER_ID}:"):
        raise RecordSchemaError(f"example_id must begin with {PAPER_ID!r}")
    parents = list(parent_hashes)
    if not parents:
        raise RecordSchemaError("derived records require at least one parent_hash")
    for index, digest in enumerate(parents):
        require_sha256(digest, field=f"parent_hashes[{index}]")
    if len(parents) != len(set(parents)):
        raise RecordSchemaError("parent_hashes must not contain duplicates")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": _require_name(record_type, field="record_type"),
        "example_id": example_id,
        "source": _require_name(source, field="source"),
        "source_key": normalize_text(source_key),
        "parent_hashes": parents,
        "payload": _require_mapping(payload, field="payload"),
        "metadata": _require_mapping(metadata or {}, field="metadata"),
    }
    record["content_sha256"] = _record_digest(record)
    return record


def validate_record(record: Mapping[str, Any], *, expected_type: str | None = None) -> dict[str, Any]:
    """Validate and return a plain copy of one immutable row."""

    required = {
        "schema_version",
        "record_type",
        "example_id",
        "source",
        "source_key",
        "parent_hashes",
        "payload",
        "metadata",
        "content_sha256",
    }
    missing = sorted(required - set(record))
    extra = sorted(set(record) - required)
    if missing or extra:
        raise RecordSchemaError(f"record keys mismatch: missing={missing}, extra={extra}")
    plain = dict(record)
    if plain["schema_version"] != SCHEMA_VERSION:
        raise RecordSchemaError(f"schema_version must be {SCHEMA_VERSION}")
    record_type = _require_name(plain["record_type"], field="record_type")
    if expected_type is not None and record_type != expected_type:
        raise RecordSchemaError(f"record_type is {record_type!r}, expected {expected_type!r}")
    if not isinstance(plain["example_id"], str) or not plain["example_id"].startswith(f"{PAPER_ID}:"):
        raise RecordSchemaError("example_id is not in this paper namespace")
    _require_name(plain["source"], field="source")
    if not isinstance(plain["source_key"], str) or not plain["source_key"]:
        raise RecordSchemaError("source_key must be a non-empty string")
    if not isinstance(plain["parent_hashes"], list):
        raise RecordSchemaError("parent_hashes must be a list")
    for index, digest in enumerate(plain["parent_hashes"]):
        require_sha256(digest, field=f"parent_hashes[{index}]")
    if len(plain["parent_hashes"]) != len(set(plain["parent_hashes"])):
        raise RecordSchemaError("parent_hashes must not contain duplicates")
    _require_mapping(plain["payload"], field="payload")
    _require_mapping(plain["metadata"], field="metadata")
    recorded_digest = require_sha256(plain["content_sha256"], field="content_sha256")
    actual_digest = _record_digest(plain)
    if recorded_digest != actual_digest:
        raise RecordSchemaError(f"record digest mismatch: recorded {recorded_digest}, computed {actual_digest}")
    return plain


def _require_name(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _NAME_RE.fullmatch(value) is None:
        raise RecordSchemaError(f"{field} must match {_NAME_RE.pattern!r}")
    return value


def _require_mapping(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordSchemaError(f"{field} must be a mapping")
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise RecordSchemaError(f"{field} must normalize to an object")
    return normalized


def _record_digest(record: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in record.items() if key != "content_sha256"})


__all__ = [
    "ARTIFACT_SCHEMA",
    "PAPER_ID",
    "SCHEMA_VERSION",
    "RecordSchemaError",
    "canonical_json",
    "make_derived_record",
    "make_source_record",
    "normalize_json",
    "normalize_text",
    "require_sha256",
    "sha256_bytes",
    "sha256_json",
    "sha256_text",
    "stable_example_id",
    "validate_record",
]
