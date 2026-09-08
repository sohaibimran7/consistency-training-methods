#!/usr/bin/env python3
"""Materialize the pinned 1,800-row EvalAwareBench Figure 6 prompt slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm.artifacts import ArtifactManifestError, artifact_manifest_path, read_verified_artifact_manifest
from ctm_data.adapters.eval_awareness.data import normalize_rows, read_jsonl
from ctm_data.adapters.eval_awareness.figure6_spec import (
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    FIGURE6_CONDITIONS,
    FIGURE6_FACTORS,
    FIGURE6_ROW_COUNT,
    FIGURE6_SOURCE_CONFIGS,
    FIGURE6_TASK_COUNT,
    FIGURE6_VALENCES,
    UPSTREAM_CODE_REPOSITORY,
    UPSTREAM_CODE_REVISION,
    sha256_file,
)

FIGURE6_ARTIFACT_SCHEMA = "ctm.eval_awareness.figure6_prompts"
FIGURE6_ARTIFACT_SCHEMA_VERSION = 1
FIGURE6_BASELINE_CONFIG = {
    "F1": "realistic",
    "F2": "realistic",
    "F3": "casual",
    "F4": "realistic",
    "F5": "realistic",
    "F6": "realistic",
    "F7": "prose",
    "F8": "realistic",
}
FIGURE6_VARIED_LEVELS = {
    "F1": "varied",
    "F2": "varied",
    "F3": "formal",
    "F4": "varied",
    "F5": "varied",
    "F6": "varied",
    "F7": "template",
    "F8": "varied",
}

_SOURCE_VALENCE = {
    "prompts_safety": "safety",
    "prompts_capability": "capability",
}
_VALENCE_ORDER = {valence: index for index, valence in enumerate(FIGURE6_VALENCES)}
_CONDITION_ORDER = {condition: index for index, condition in enumerate(FIGURE6_CONDITIONS)}
_CUSTOM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def make_condition_id(task_id: str, valence: str, condition: str) -> str:
    """Return a stable OpenAI-compatible custom ID for one prompt condition."""

    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be a non-empty string")
    if valence not in FIGURE6_VALENCES:
        raise ValueError(f"valence must be one of {list(FIGURE6_VALENCES)}")
    if condition not in FIGURE6_CONDITIONS:
        raise ValueError(f"condition must be one of {list(FIGURE6_CONDITIONS)}")
    identity = json.dumps(
        {"condition": condition, "task_id": task_id, "valence": valence},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(identity).hexdigest()[:32]
    custom_id = f"eabf6_{valence}_{condition.lower()}_{digest}"
    if _CUSTOM_ID_RE.fullmatch(custom_id) is None:  # pragma: no cover - construction invariant
        raise RuntimeError("constructed an unsafe Figure 6 condition ID")
    return custom_id


def _condition_for_source_row(row: Mapping[str, Any]) -> str | None:
    factors = row["factors_varied"]
    if row["num_factors_varied"] == 0 and factors == []:
        return "baseline"
    if row["num_factors_varied"] == 1 and len(factors) == 1 and factors[0] in FIGURE6_FACTORS:
        return factors[0]
    return None


def _expected_config(condition: str) -> dict[str, str]:
    config = dict(FIGURE6_BASELINE_CONFIG)
    if condition != "baseline":
        config[condition] = FIGURE6_VARIED_LEVELS[condition]
    return config


def _normalize_strict_source_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    for row_number, row in enumerate(rows, start=1):
        factors = row.get("factors_varied")
        if (
            isinstance(factors, list)
            and all(isinstance(factor, str) for factor in factors)
            and len(factors) != len(set(factors))
        ):
            raise ValueError(f"{label} row {row_number}.factors_varied contains duplicates")
    normalized = normalize_rows(rows)
    for row_number, row in enumerate(normalized, start=1):
        unknown_factors = sorted(set(row["factors_varied"]) - set(FIGURE6_FACTORS))
        if unknown_factors:
            raise ValueError(f"{label} row {row_number} has unknown factors: {unknown_factors}")
        for field in ("task_id", "task_name", "prompt"):
            if not row[field].strip():
                raise ValueError(f"{label} row {row_number}.{field} must not be blank")
    return normalized


def _read_source(path: Path, *, source_config: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {source_config} source JSONL: {path}")
    raw_rows = read_jsonl([path])

    # The upstream prompt configs have no row ID; task_id intentionally repeats for
    # its 256 configurations.  If a future export adds IDs, require those IDs to be
    # complete and unique so accidental concatenation is still caught.
    rows_with_id = [row for row in raw_rows if "id" in row]
    if rows_with_id:
        if len(rows_with_id) != len(raw_rows):
            raise ValueError(f"{source_config} source mixes rows with and without an id field")
        source_ids = [row["id"] for row in raw_rows]
        if any(not isinstance(source_id, str) or not source_id for source_id in source_ids):
            raise ValueError(f"{source_config} source row IDs must be non-empty strings")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"{source_config} source row IDs must be unique")

    normalized = _normalize_strict_source_rows(raw_rows, label=source_config)
    expected_valence = _SOURCE_VALENCE[source_config]
    unexpected = sorted({row["valence"] for row in normalized if row["valence"] != expected_valence})
    if unexpected:
        raise ValueError(f"{source_config} must contain only {expected_valence!r} rows; found valences {unexpected}")
    return normalized, {
        "file_name": path.name,
        "content_sha256": sha256_file(path),
        "row_count": len(normalized),
    }


def select_figure6_rows(
    safety_rows: Sequence[Mapping[str, Any]],
    capability_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select one baseline and each exact singleton F1--F8 for every task pair."""

    normalized_by_valence = {
        "safety": _normalize_strict_source_rows(safety_rows, label="safety"),
        "capability": _normalize_strict_source_rows(capability_rows, label="capability"),
    }
    for valence, rows in normalized_by_valence.items():
        unexpected = sorted({row["valence"] for row in rows if row["valence"] != valence})
        if unexpected:
            raise ValueError(f"{valence} rows include unexpected valences: {unexpected}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    task_id_owners: dict[str, tuple[str, str]] = {}
    task_names_by_valence: dict[str, set[str]] = {valence: set() for valence in FIGURE6_VALENCES}
    for valence, rows in normalized_by_valence.items():
        for row in rows:
            task_name = row["task_name"]
            owner = (task_name, valence)
            prior_owner = task_id_owners.setdefault(row["task_id"], owner)
            if prior_owner != owner:
                raise ValueError(f"task_id {row['task_id']!r} is shared by task variants {prior_owner!r} and {owner!r}")
            task_names_by_valence[valence].add(task_name)
            grouped[owner].append(row)

    safety_names = task_names_by_valence["safety"]
    capability_names = task_names_by_valence["capability"]
    if safety_names != capability_names:
        missing_capability = sorted(safety_names - capability_names)
        missing_safety = sorted(capability_names - safety_names)
        raise ValueError(
            "safety/capability task pairs do not match: "
            f"missing capability={missing_capability}, missing safety={missing_safety}"
        )
    if len(safety_names) != FIGURE6_TASK_COUNT:
        raise ValueError(f"Figure 6 requires exactly {FIGURE6_TASK_COUNT} paired task names; found {len(safety_names)}")

    selected: list[dict[str, Any]] = []
    seen_condition_ids: set[str] = set()
    for task_name in sorted(safety_names):
        for valence in FIGURE6_VALENCES:
            group = grouped[(task_name, valence)]
            task_ids = {row["task_id"] for row in group}
            if len(task_ids) != 1:
                raise ValueError(f"task {task_name!r}/{valence} has task_ids {sorted(task_ids)}; expected exactly one")
            task_id = next(iter(task_ids))
            candidates: dict[str, list[dict[str, Any]]] = {condition: [] for condition in FIGURE6_CONDITIONS}
            for row in group:
                condition = _condition_for_source_row(row)
                if condition is not None:
                    candidates[condition].append(row)

            for condition in FIGURE6_CONDITIONS:
                matching = candidates[condition]
                if len(matching) != 1:
                    raise ValueError(
                        f"task {task_name!r}/{valence} has {len(matching)} exact {condition} rows; expected 1"
                    )
                source = matching[0]
                expected_config = _expected_config(condition)
                if source["config"] != expected_config:
                    raise ValueError(
                        f"task {task_name!r}/{valence} exact {condition} row has an inconsistent factor config"
                    )
                condition_id = make_condition_id(task_id, valence, condition)
                if condition_id in seen_condition_ids:
                    raise ValueError(f"duplicate generated condition_id: {condition_id}")
                seen_condition_ids.add(condition_id)
                selected.append(
                    {
                        "schema_version": FIGURE6_ARTIFACT_SCHEMA_VERSION,
                        "condition_id": condition_id,
                        "pair_id": task_name,
                        "task_id": task_id,
                        "task_name": task_name,
                        "valence": valence,
                        "condition": condition,
                        "factors_varied": list(source["factors_varied"]),
                        "num_factors_varied": source["num_factors_varied"],
                        "config": source["config"],
                        "description": source["description"],
                        "prompt": source["prompt"],
                        "source_config": f"prompts_{valence}",
                    }
                )

    if len(selected) != FIGURE6_ROW_COUNT:
        raise ValueError(f"Figure 6 selection produced {len(selected)} rows; expected {FIGURE6_ROW_COUNT}")
    return selected


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing Figure 6 file: {path}") from exc


def materialize_figure6(
    prompts_safety_path: str | Path,
    prompts_capability_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze the pinned Figure 6 prompt slice and its immutable manifest."""

    sources = {
        "prompts_safety": Path(prompts_safety_path),
        "prompts_capability": Path(prompts_capability_path),
    }
    if sources["prompts_safety"].resolve() == sources["prompts_capability"].resolve():
        raise ValueError("prompts_safety and prompts_capability must be separate source files")

    target = Path(output_path)
    sidecar = artifact_manifest_path(target)
    existing = [path for path in (target, sidecar) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing Figure 6 file(s): {existing}")

    rows_by_config: dict[str, list[dict[str, Any]]] = {}
    source_identities: dict[str, dict[str, Any]] = {}
    for source_config in FIGURE6_SOURCE_CONFIGS:
        rows, identity = _read_source(sources[source_config], source_config=source_config)
        rows_by_config[source_config] = rows
        source_identities[source_config] = identity

    selected = select_figure6_rows(
        rows_by_config["prompts_safety"],
        rows_by_config["prompts_capability"],
    )
    payload = _canonical_jsonl(selected)
    manifest: dict[str, Any] = {
        "artifact_schema": FIGURE6_ARTIFACT_SCHEMA,
        "schema_version": FIGURE6_ARTIFACT_SCHEMA_VERSION,
        "row_count": len(selected),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "provenance": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_license": DATASET_LICENSE,
            "dataset_configs": list(FIGURE6_SOURCE_CONFIGS),
            "source_revision": DATASET_REVISION,
            "source_license": DATASET_LICENSE,
            "source_configs": list(FIGURE6_SOURCE_CONFIGS),
            "source_row_count": sum(identity["row_count"] for identity in source_identities.values()),
            "source_files": source_identities,
            "upstream_code_repository": UPSTREAM_CODE_REPOSITORY,
            "upstream_code_revision": UPSTREAM_CODE_REVISION,
            "selection": {
                "task_pair_count": FIGURE6_TASK_COUNT,
                "task_variant_count": FIGURE6_TASK_COUNT * len(FIGURE6_VALENCES),
                "valences": list(FIGURE6_VALENCES),
                "conditions": list(FIGURE6_CONDITIONS),
                "row_count": FIGURE6_ROW_COUNT,
                "rule": "one exact baseline plus each exact singleton F1-F8 per task and valence",
            },
        },
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

    _write_exclusive(target, payload)
    _write_exclusive(sidecar, manifest_payload)
    return manifest


materialize_figure6_artifact = materialize_figure6


def _read_artifact_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactManifestError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ArtifactManifestError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    return rows


def validate_figure6_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate the complete shape and exact factor coverage of artifact rows."""

    if len(rows) != FIGURE6_ROW_COUNT:
        raise ArtifactManifestError(f"Figure 6 artifact has {len(rows)} rows; expected {FIGURE6_ROW_COUNT}")
    required = {
        "schema_version",
        "condition_id",
        "pair_id",
        "task_id",
        "task_name",
        "valence",
        "condition",
        "factors_varied",
        "num_factors_varied",
        "config",
        "description",
        "prompt",
        "source_config",
    }
    condition_ids: set[str] = set()
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    task_ids_by_variant: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_names: set[str] = set()
    task_id_owners: dict[str, tuple[str, str]] = {}
    for row_number, row in enumerate(rows, start=1):
        missing = sorted(required - row.keys())
        if missing:
            raise ArtifactManifestError(f"Figure 6 row {row_number} is missing fields: {missing}")
        if row["schema_version"] != FIGURE6_ARTIFACT_SCHEMA_VERSION:
            raise ArtifactManifestError(f"Figure 6 row {row_number} has an unsupported schema_version")
        condition_id = row["condition_id"]
        if not isinstance(condition_id, str) or _CUSTOM_ID_RE.fullmatch(condition_id) is None:
            raise ArtifactManifestError(f"Figure 6 row {row_number} has an unsafe condition_id")
        if condition_id in condition_ids:
            raise ArtifactManifestError(f"duplicate Figure 6 condition_id: {condition_id}")
        condition_ids.add(condition_id)
        for field in ("pair_id", "task_id", "task_name", "prompt"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ArtifactManifestError(f"Figure 6 row {row_number}.{field} must be a non-empty string")
        if row["pair_id"] != row["task_name"]:
            raise ArtifactManifestError(f"Figure 6 row {row_number} pair_id does not match task_name")
        valence = row["valence"]
        condition = row["condition"]
        if valence not in FIGURE6_VALENCES or condition not in FIGURE6_CONDITIONS:
            raise ArtifactManifestError(f"Figure 6 row {row_number} has an invalid valence or condition")
        if row["source_config"] != f"prompts_{valence}":
            raise ArtifactManifestError(f"Figure 6 row {row_number} source_config does not match valence")
        expected_factors = [] if condition == "baseline" else [condition]
        if row["factors_varied"] != expected_factors or row["num_factors_varied"] != len(expected_factors):
            raise ArtifactManifestError(f"Figure 6 row {row_number} is not an exact {condition} condition")
        if row["config"] != _expected_config(condition):
            raise ArtifactManifestError(f"Figure 6 row {row_number} has an inconsistent factor config")
        expected_id = make_condition_id(row["task_id"], valence, condition)
        if condition_id != expected_id:
            raise ArtifactManifestError(f"Figure 6 row {row_number} condition_id does not match its identity fields")
        owner = (row["task_name"], valence)
        previous_owner = task_id_owners.setdefault(row["task_id"], owner)
        if previous_owner != owner:
            raise ArtifactManifestError(f"task_id {row['task_id']!r} belongs to multiple task variants")
        pair_names.add(row["task_name"])
        grouped[owner].add(condition)
        task_ids_by_variant[owner].add(row["task_id"])

    if len(pair_names) != FIGURE6_TASK_COUNT:
        raise ArtifactManifestError(
            f"Figure 6 artifact has {len(pair_names)} paired task names; expected {FIGURE6_TASK_COUNT}"
        )
    expected_conditions = set(FIGURE6_CONDITIONS)
    expected_variants = {(task_name, valence) for task_name in pair_names for valence in FIGURE6_VALENCES}
    if set(grouped) != expected_variants:
        raise ArtifactManifestError("Figure 6 artifact does not contain both valences for every task pair")
    for owner in sorted(expected_variants):
        if grouped[owner] != expected_conditions:
            raise ArtifactManifestError(f"task variant {owner!r} does not have all Figure 6 conditions")
        if len(task_ids_by_variant[owner]) != 1:
            raise ArtifactManifestError(f"task variant {owner!r} has more than one task_id")
    expected_order = sorted(
        rows,
        key=lambda row: (
            row["task_name"],
            _VALENCE_ORDER[row["valence"]],
            _CONDITION_ORDER[row["condition"]],
        ),
    )
    if list(rows) != expected_order:
        raise ArtifactManifestError("Figure 6 artifact rows are not in deterministic task/valence/condition order")


def verify_figure6_artifact(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify artifact bytes, pinned provenance, IDs, and exact row coverage."""

    target = Path(path)
    manifest = read_verified_artifact_manifest(
        target,
        expected_schema=FIGURE6_ARTIFACT_SCHEMA,
        expected_schema_version=FIGURE6_ARTIFACT_SCHEMA_VERSION,
        manifest_path=manifest_path,
    )
    provenance = manifest["provenance"]
    expected_provenance = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "dataset_configs": list(FIGURE6_SOURCE_CONFIGS),
        "upstream_code_revision": UPSTREAM_CODE_REVISION,
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise ArtifactManifestError(
                f"Figure 6 manifest provenance {field!r} is {provenance.get(field)!r}, expected {expected!r}"
            )
    selection = provenance.get("selection")
    if not isinstance(selection, dict) or selection.get("row_count") != FIGURE6_ROW_COUNT:
        raise ArtifactManifestError("Figure 6 manifest has invalid selection provenance")
    rows = _read_artifact_rows(target)
    validate_figure6_rows(rows)
    return manifest


def load_figure6_rows(path: str | Path, *, manifest_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load rows only after full manifest and row-shape verification."""

    verify_figure6_artifact(path, manifest_path=manifest_path)
    return _read_artifact_rows(Path(path))


def load_figure6_artifact(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ``(rows, manifest)`` after full verification."""

    manifest = verify_figure6_artifact(path, manifest_path=manifest_path)
    return _read_artifact_rows(Path(path)), manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-safety", type=Path, required=True)
    parser.add_argument("--prompts-capability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = materialize_figure6(args.prompts_safety, args.prompts_capability, args.output)
    print(
        f"Wrote {manifest['row_count']} pinned Figure 6 prompts to {args.output} "
        f"(sha256={manifest['content_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIGURE6_ARTIFACT_SCHEMA",
    "FIGURE6_ARTIFACT_SCHEMA_VERSION",
    "FIGURE6_BASELINE_CONFIG",
    "FIGURE6_VARIED_LEVELS",
    "load_figure6_artifact",
    "load_figure6_rows",
    "main",
    "make_condition_id",
    "materialize_figure6",
    "materialize_figure6_artifact",
    "select_figure6_rows",
    "validate_figure6_rows",
    "verify_figure6_artifact",
]
