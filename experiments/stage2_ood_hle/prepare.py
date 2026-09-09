"""Frozen bias-order contract for the Stage 2 IID/HLE OOD diagnostic.

The evaluation-data materializer owns any future full manifest.  Analysis
imports this one ordering source instead of maintaining its own independently
sorted list, and can additionally bind itself to a copied frozen manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "stage2-ood-hle-manifest-v1"
TRAINING_BIAS = "wrong_argument"
# Preserve the frozen evaluation order; do not alphabetise at analysis time.
HELDOUT_BIASES = (
    "suggested_answer",
    "distractor_fact",
    "post_hoc",
    "spurious_few_shot_squares",
    "wrong_few_shot",
)


def _validate_biases(value: Any, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{label} must be a non-empty array of unique bias names")
    return tuple(value)


def load_bias_contract(path: str | Path) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    """Load the exact ordered bias contract from a frozen Stage 2 manifest."""

    manifest = Path(path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Stage 2 OOD manifest does not exist: {manifest}")
    payload = manifest.read_bytes()
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Stage 2 OOD manifest is invalid JSON: {manifest}") from exc
    if not isinstance(document, Mapping) or document.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Stage 2 OOD manifest must use schema {MANIFEST_SCHEMA!r}")
    training_bias = document.get("training_bias")
    if not isinstance(training_bias, str) or not training_bias:
        raise ValueError("Stage 2 OOD manifest training_bias must be non-empty")
    heldout = _validate_biases(document.get("held_out_biases"), label="Stage 2 OOD manifest held_out_biases")
    if training_bias in heldout:
        raise ValueError("Stage 2 OOD manifest held_out_biases must exclude training_bias")
    return training_bias, heldout, {
        "path": str(manifest),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "schema": MANIFEST_SCHEMA,
    }


__all__ = ["HELDOUT_BIASES", "MANIFEST_SCHEMA", "TRAINING_BIAS", "load_bias_contract"]
