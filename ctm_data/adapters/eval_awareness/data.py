"""Adapt explicitly selected EvalAwareBench rows into training families."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm.artifacts import read_verified_artifact_manifest
from ctm.settings.families import (
    FAMILY_SCHEMA_VERSION,
    FamilyValidationError,
    select_fixed_variants,
    stable_digest,
    write_frozen_artifact,
)

DATASET_ID = "aisa-group/EvalAwareBench"
DATASET_CONFIGS = ("prompts", "prompts_safety", "prompts_capability")
DATASET_LICENSE = "CC-BY-NC-4.0"
VALID_VALENCES = ("safety", "capability")
_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def _normalize_row(row: Mapping[str, Any], *, row_number: int) -> dict[str, Any]:
    required = {
        "task_id",
        "task_name",
        "valence",
        "description",
        "factors_varied",
        "num_factors_varied",
        "config",
        "prompt",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"row {row_number} is missing required fields: {missing}")
    if row["valence"] not in VALID_VALENCES:
        raise ValueError(f"row {row_number}.valence must be one of {list(VALID_VALENCES)}")
    for field in ("task_id", "task_name", "description", "prompt"):
        if not isinstance(row[field], str) or (field != "description" and not row[field]):
            raise ValueError(f"row {row_number}.{field} must be a string")
    if not isinstance(row["config"], dict):
        raise ValueError(f"row {row_number}.config must be an object")
    if not isinstance(row["num_factors_varied"], int) or isinstance(row["num_factors_varied"], bool):
        raise ValueError(f"row {row_number}.num_factors_varied must be an integer")
    factors = row["factors_varied"]
    if row["num_factors_varied"] == 0 and factors == ["none (baseline)"]:
        factors = []
    if not isinstance(factors, list) or any(not isinstance(factor, str) for factor in factors):
        raise ValueError(f"row {row_number}.factors_varied must be a string list")
    factors = sorted(set(factors))
    if len(factors) != row["num_factors_varied"]:
        raise ValueError(f"row {row_number}.num_factors_varied does not match factors_varied")
    return {**dict(row), "factors_varied": factors}


def normalize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("EvalAwareBench source rows must not be empty")
    return [_normalize_row(row, row_number=index) for index, row in enumerate(rows, start=1)]


def factor_side_name(factors: Sequence[str]) -> str:
    """Stable, human-readable name for one non-baseline condition."""

    normalized = sorted(set(factors))
    return "+".join(normalized) if normalized else "baseline"


def build_prompt_families(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_variants: int,
    seed: str = "42",
    factors: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build direction-neutral matched families from EvalAwareBench rows.

    ``factors`` selects rows whose ``factors_varied`` exactly match that set.
    Artifacts always use the natural baseline as their canonical storage side;
    the setting reorients the pair from explicit experiment configuration.
    """

    if n_variants < 1:
        raise ValueError("n_variants must be >= 1")
    if factors is not None and (not factors or any(not isinstance(factor, str) or not factor for factor in factors)):
        raise ValueError("factors must contain one or more non-empty factor names")
    factor_filter = tuple(sorted(set(factors))) if factors is not None else None
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalize_rows(rows):
        grouped[(row["task_id"], row["valence"])].append(row)

    families = []
    for (task_id, valence), group in sorted(grouped.items()):
        source_id = f"{task_id}:{valence}"
        baselines = [row for row in group if row["num_factors_varied"] == 0]
        if len(baselines) != 1:
            raise ValueError(f"task {task_id!r}/{valence} has {len(baselines)} baseline rows; expected 1")
        baseline = baselines[0]
        varied_rows = [
            row
            for row in group
            if row is not baseline and (factor_filter is None or tuple(sorted(row["factors_varied"])) == factor_filter)
        ]
        candidates = []
        for row in varied_rows:
            identity = json.dumps(
                {"config": row["config"], "factors": row["factors_varied"], "prompt": row["prompt"]},
                sort_keys=True,
                separators=(",", ":"),
            )
            candidates.append(
                {
                    "variant_id": f"{source_id}:{stable_digest(identity, seed='evalaware-variant')[:20]}",
                    "side": factor_side_name(row["factors_varied"]),
                    "messages": [{"role": "user", "content": row["prompt"]}],
                    "axes": {
                        "factors_varied": row["factors_varied"],
                        "num_factors_varied": row["num_factors_varied"],
                        "config": row["config"],
                    },
                }
            )
        try:
            selected = select_fixed_variants(candidates, source_id=source_id, n_variants=n_variants, seed=seed)
        except FamilyValidationError as exc:
            filter_description = f" for exact factors {list(factor_filter)}" if factor_filter is not None else ""
            raise ValueError(f"{exc}{filter_description}") from exc

        families.append(
            {
                "schema_version": FAMILY_SCHEMA_VERSION,
                "source_id": source_id,
                "source": DATASET_ID,
                "reference_messages": [{"role": "user", "content": baseline["prompt"]}],
                "variants": selected,
                "metadata": {
                    "task_name": baseline["task_name"],
                    "valence": valence,
                    "description": baseline["description"],
                    "factor_filter": list(factor_filter) if factor_filter is not None else None,
                    "available_sides": ["baseline"] + sorted({variant["side"] for variant in selected}),
                },
            }
        )
    families.sort(key=lambda family: stable_digest(family["source_id"], seed=seed))
    return families


def materialize_eval_awareness(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    n_variants: int,
    source_revision: str,
    source_config: str = "prompts",
    source_license: str = DATASET_LICENSE,
    seed: str = "42",
    factors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Freeze training families from exactly the supplied source rows."""

    if source_config not in DATASET_CONFIGS:
        raise ValueError(f"source_config must be one of {list(DATASET_CONFIGS)}")
    if _IMMUTABLE_REVISION_RE.fullmatch(source_revision) is None:
        raise ValueError("source_revision must be a full 40-hex dataset commit")
    if not isinstance(source_license, str) or not source_license.strip():
        raise ValueError("source_license must be a non-empty string")
    normalized = normalize_rows(rows)
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sorted(normalized, key=lambda row: json.dumps(row, sort_keys=True))
    ).encode()
    families = build_prompt_families(
        normalized,
        n_variants=n_variants,
        seed=seed,
        factors=factors,
    )
    return write_frozen_artifact(
        output_path,
        families,
        provenance={
            "dataset_id": DATASET_ID,
            "dataset_config": source_config,
            "source_revision": source_revision.lower(),
            "source_license": source_license,
            "source_row_count": len(normalized),
            "source_rows_sha256": hashlib.sha256(payload).hexdigest(),
            "n_variants": n_variants,
            "seed": seed,
            "factor_filter": sorted(set(factors)) if factors is not None else None,
        },
    )


def validate_artifact_manifest(path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Require EvalAwareBench provenance on an already verified manifest."""

    if manifest.get("provenance", {}).get("dataset_id") != DATASET_ID:
        raise ValueError(f"{path} is not an EvalAwareBench training artifact")
    return dict(manifest)


def read_artifact_manifest(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    manifest = read_verified_artifact_manifest(path, expected_schema=expected_schema)
    return validate_artifact_manifest(path, manifest)


def read_jsonl(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path_like in paths:
        path = Path(path_like)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
                rows.append(row)
    if not rows:
        raise ValueError("source JSONL files contained no rows")
    return rows
