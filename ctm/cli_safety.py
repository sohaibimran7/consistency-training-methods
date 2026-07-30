"""Shared safeguards for CLI configuration that is echoed or persisted.

The training and evaluation CLIs intentionally print exact commands and
resolved configuration before an expensive run.  Secrets therefore belong in
provider environment variables, never in one of the JSON configuration flags.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence, Set
from pathlib import Path

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "cookie",
    "cookies",
    "credential",
    "credentials",
    "password",
    "proxy_authorization",
    "secret",
    "set_cookie",
    "token",
    "access_token",
    "auth_token",
    "client_secret",
    "x_api_key",
}
_SECRET_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_credential",
    "_credentials",
    "_password",
    "_secret",
    "_token",
)
_HEADER_CONTAINERS = {
    "default_headers",
    "extra_headers",
    "headers",
    "http_headers",
    "request_headers",
}


def normalized_config_key(key: object) -> str:
    """Normalize common header/config spelling variants for comparisons."""

    return str(key).strip().lower().replace("-", "_")


def is_secret_key(key: object) -> bool:
    """Whether a configuration key conventionally contains a credential."""

    normalized = normalized_config_key(key)
    return normalized in _SECRET_KEYS or normalized.endswith(_SECRET_SUFFIXES)


def is_header_container(key: object) -> bool:
    """Whether a mapping may contain arbitrary sensitive HTTP header values."""

    return normalized_config_key(key) in _HEADER_CONTAINERS


def reject_inline_secrets(value: object, *, path: str) -> None:
    """Reject secrets before a CLI echoes its argv or resolved configuration."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if is_secret_key(key) or is_header_container(key):
                raise ValueError(
                    f"{item_path} may contain credentials; use the provider's environment variable instead"
                )
            reject_inline_secrets(item, path=item_path)
    elif isinstance(value, (Sequence, Set)) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            reject_inline_secrets(item, path=f"{path}[{index}]")


def redact_secrets(value: object) -> object:
    """Return a recursively redacted, JSON-friendly copy of configuration data."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if is_secret_key(key) or is_header_container(key):
                result[str(key)] = "<redacted>"
            else:
                result[str(key)] = redact_secrets(item)
        return result
    if isinstance(value, (Sequence, Set)) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_secrets(item) for item in value]
    return value


def parse_json_object(value: str | None, *, label: str) -> dict:
    """Parse an inline JSON object or a path to a JSON file."""

    if not value:
        return {}
    stripped = value.lstrip()
    if stripped.startswith(("{", "[")):
        text = value
    else:
        candidate = Path(value).expanduser()
        try:
            text = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
        except OSError:
            text = value
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be a JSON object or path to a JSON file: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to an object")
    return parsed


__all__ = [
    "is_header_container",
    "is_secret_key",
    "parse_json_object",
    "redact_secrets",
    "reject_inline_secrets",
]
