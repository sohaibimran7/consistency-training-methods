"""Local, reproducible ELEPHANT AITA-NTA-FLIP benchmark support.

The package intentionally contains only the judge-free binary moral
sycophancy benchmark.  It does not download data, call a model provider, or
ship the underlying Reddit-derived text in the repository.  ``prepare``
stages a separately obtained official archive into a local immutable
artifact; ``tasks`` exposes it through the repository's generic Inspect
runner; and ``preflight`` scores completed EvalLogs locally.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BENCHMARK",
    "CONCURRENCY_CONFIG",
    "EXPECTED_PAIRS",
    "GENERATION_CONFIG",
    "MAX_CONNECTIONS",
    "MANIFEST_SCHEMA",
    "NUM_SHARDS",
    "PROMPT_SUFFIX",
    "RUNTIME_GENERATION_CONFIG",
]


def __getattr__(name: str) -> Any:
    """Lazily expose constants without pre-importing deployment CLIs."""

    if name not in __all__:
        raise AttributeError(name)
    value = getattr(importlib.import_module(f"{__name__}.prepare"), name)
    globals()[name] = value
    return value
