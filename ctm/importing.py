"""Small helpers for explicit ``module:callable`` composition."""

from __future__ import annotations

import importlib
from collections.abc import Callable


def load_callable(spec: str, *, label: str) -> Callable:
    """Load a callable without maintaining a package-specific registry."""

    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"{label} must use the form 'module:callable'")
    module_name, attribute = spec.rsplit(":", 1)
    if not module_name or not attribute:
        raise ValueError(f"{label} must use the form 'module:callable'")
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot load {label} {spec!r}: {exc}") from exc
    if not callable(value):
        raise TypeError(f"{label} {spec!r} is not callable")
    return value


__all__ = ["load_callable"]
