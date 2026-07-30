"""Canonical JSON, text normalization, and content identities shared by CTM.

These primitives intentionally contain no paper, artifact, or provider policy.
Callers own the material they hash; this module only makes the byte contract
consistent across data adapters, manifests, and generation provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class IdentityValueError(ValueError):
    """A value cannot participate in CTM's canonical identity contract."""


def canonical_json(value: Any) -> str:
    """Return CTM's stable JSON encoding without accepting NaN/Infinity."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError("sha256_bytes payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("sha256_text value must be a string")
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def normalize_text(value: str) -> str:
    """Normalize identity text while preserving internal whitespace."""

    if not isinstance(value, str):
        raise IdentityValueError("text values must be strings")
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_json(value: Any) -> Any:
    """Recursively normalize JSON-compatible identity material."""

    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, Mapping):
        return {str(key): normalize_json(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [normalize_json(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise IdentityValueError(f"unsupported JSON value for stable identity: {type(value).__name__}")


def require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise IdentityValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "IdentityValueError",
    "canonical_json",
    "normalize_json",
    "normalize_text",
    "require_sha256",
    "sha256_bytes",
    "sha256_json",
    "sha256_text",
]
