"""Offline normalization of an explicitly acquired HarmBench export.

This module deliberately has no dataset-client dependency.  Callers acquire a
HarmBench export themselves, pin its upstream revision when one is known, and
pass the local file here.  The local file digest is always recorded as an
additional immutable source identity.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from scripts.irpan_2510_27062.artifacts import (
    producer_identity,
    require_local_artifact,
    write_artifact,
)
from scripts.irpan_2510_27062.partitions import (
    HARM_BENCH_IDENTITY_FIELD,
    SOURCE_IDENTITY_FIELDS,
    TRAINING,
    PartitionError,
    extract_source_identity,
    harmbench_partition_provenance,
    partition_harmbench_ids,
    require_partition,
)
from scripts.irpan_2510_27062.schema import (
    RecordSchemaError,
    make_source_record,
    normalize_text,
    require_sha256,
    sha256_bytes,
)
from scripts.irpan_2510_27062.source_registry import require_source

HARM_BENCH_SOURCE = "centerforaisafety/HarmBench"
DEFAULT_ID_FIELD = HARM_BENCH_IDENTITY_FIELD
DEFAULT_PROMPT_FIELD = "Behavior"
NORMALIZATION_VERSION = "reconstruction_v2"


class HarmBenchSourceError(ValueError):
    """A local HarmBench export cannot be normalized without ambiguity."""


def read_local_harmbench_export(path: str | Path) -> list[dict[str, Any]]:
    """Read a local JSON, JSONL, or CSV export without performing acquisition."""

    source = require_source("harmbench")
    target = require_local_artifact(path, source_key=source.key, acquisition_url=source.official_url)
    suffix = target.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows = _read_jsonl(target)
    elif suffix == ".json":
        try:
            decoded = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HarmBenchSourceError(f"invalid JSON in {target}: {exc}") from exc
        if not isinstance(decoded, list) or not all(isinstance(row, dict) for row in decoded):
            raise HarmBenchSourceError(f"{target} must contain a JSON array of objects")
        rows = [dict(row) for row in decoded]
    elif suffix == ".csv":
        with target.open(encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        raise HarmBenchSourceError(f"unsupported HarmBench export format {suffix!r}; use .jsonl, .json, or .csv")
    if not rows:
        raise HarmBenchSourceError(f"HarmBench export contains no rows: {target}")
    return rows


def normalize_harmbench_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    subset: str,
    split: str,
    source_revision: str | None = None,
    source_file_sha256: str | None = None,
    id_field: str = DEFAULT_ID_FIELD,
    prompt_field: str = DEFAULT_PROMPT_FIELD,
) -> list[dict[str, Any]]:
    """Normalize explicit HarmBench rows into stable, content-bound records.

    ``subset`` and ``split`` are required reconstruction choices because the
    paper does not report them.  At least one immutable upstream identity must
    be supplied: an upstream revision or the exact local-file SHA-256.
    Source IDs and prompts are preserved after only the repository-wide text
    normalization (NFC, newline normalization, and outer whitespace trimming).
    """

    subset_value = _nonempty(subset, field="subset")
    split_value = _nonempty(split, field="split")
    id_field_value = _nonempty(id_field, field="id_field")
    if id_field_value not in SOURCE_IDENTITY_FIELDS["harmbench"]:
        raise HarmBenchSourceError(
            "id_field must identify HarmBench BehaviorID; generic export IDs cannot be partition keys"
        )
    prompt_field_value = _nonempty(prompt_field, field="prompt_field")
    revision = _optional_nonempty(source_revision, field="source_revision")
    file_digest = None
    if source_file_sha256 is not None:
        try:
            file_digest = require_sha256(source_file_sha256, field="source_file_sha256")
        except RecordSchemaError as exc:
            raise HarmBenchSourceError(str(exc)) from exc
    if revision is None and file_digest is None:
        raise HarmBenchSourceError("HarmBench normalization requires source_revision or source_file_sha256")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    prompt_owners: dict[str, str] = {}
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise HarmBenchSourceError(f"HarmBench row {index} must be an object")
        prompt = raw.get(prompt_field_value)
        if not isinstance(prompt, str) or not normalize_text(prompt):
            raise HarmBenchSourceError(f"HarmBench row {index} has missing or non-string {prompt_field_value!r}")
        prompt = normalize_text(prompt)
        try:
            source_id, identity_field = extract_source_identity(
                "harmbench",
                raw,
                prompt=prompt,
                allow_prompt_fallback=False,
            )
        except PartitionError as exc:
            raise HarmBenchSourceError(f"HarmBench row {index}: {exc}") from exc
        if source_id in seen_ids:
            raise HarmBenchSourceError(f"duplicate HarmBench source ID {source_id!r}")
        if prompt in prompt_owners:
            raise HarmBenchSourceError(
                f"duplicate HarmBench prompt for source IDs {prompt_owners[prompt]!r} and {source_id!r}"
            )
        seen_ids.add(source_id)
        prompt_owners[prompt] = source_id
        normalized.append(
            make_source_record(
                record_type="harmbench_prompt",
                source="harmbench",
                source_key=source_id,
                payload={"prompt": prompt, "source_id": source_id},
                metadata={
                    "upstream_source": HARM_BENCH_SOURCE,
                    "subset": subset_value,
                    "split": split_value,
                    "source_revision": revision,
                    "source_file_sha256": file_digest,
                    "id_field": identity_field,
                    "configured_id_field": id_field_value,
                    "prompt_field": prompt_field_value,
                    "normalization_version": NORMALIZATION_VERSION,
                    "paper_status": "split/subset/revision paper-unspecified",
                },
            )
        )
    if not normalized:
        raise HarmBenchSourceError("HarmBench rows contain no prompts")
    return sorted(normalized, key=lambda row: (row["source_key"], row["content_sha256"]))


def materialize_harmbench_source(
    input_path: str | Path,
    output_path: str | Path,
    *,
    subset: str,
    split: str,
    source_revision: str | None = None,
    expected_file_sha256: str | None = None,
    id_field: str = DEFAULT_ID_FIELD,
    prompt_field: str = DEFAULT_PROMPT_FIELD,
    partition: str = TRAINING,
) -> dict[str, Any]:
    """Publish an already partitioned HarmBench training source artifact.

    Membership is checked against the fixed reconstruction.  A configured
    partition that conflicts with any stable example ID fails rather than
    silently moving or dropping rows.
    """

    partition_spec = require_partition("harmbench", partition, role=TRAINING)
    source_path = Path(input_path)
    rows = read_local_harmbench_export(source_path)
    actual_file_sha256 = sha256_bytes(source_path.read_bytes())
    if expected_file_sha256 is not None:
        try:
            expected = require_sha256(expected_file_sha256, field="expected_file_sha256")
        except RecordSchemaError as exc:
            raise HarmBenchSourceError(str(exc)) from exc
        if expected != actual_file_sha256:
            raise HarmBenchSourceError(
                f"HarmBench file digest mismatch: expected {expected}, computed {actual_file_sha256}"
            )
    normalized = normalize_harmbench_rows(
        rows,
        subset=subset,
        split=split,
        source_revision=source_revision,
        source_file_sha256=actual_file_sha256,
        id_field=id_field,
        prompt_field=prompt_field,
    )
    partition_harmbench_ids(
        (row["source_key"] for row in normalized),
        configured_partition=partition_spec.partition,
        configured_role=partition_spec.role,
    )
    partition_provenance = harmbench_partition_provenance(partition_spec.partition)
    config = {
        "normalization_version": NORMALIZATION_VERSION,
        "upstream_source": HARM_BENCH_SOURCE,
        "subset": normalize_text(subset),
        "split": normalize_text(split),
        "source_revision": _optional_nonempty(source_revision, field="source_revision"),
        "source_file_sha256": actual_file_sha256,
        "id_field": id_field,
        "prompt_field": prompt_field,
        "partition": partition_spec.partition,
        "partition_reconstruction": partition_provenance,
    }
    return write_artifact(
        output_path,
        normalized,
        artifact_kind="normalized_harmbench",
        role=TRAINING,
        producer=producer_identity("normalize_harmbench", __file__),
        config=config,
        provenance={
            "upstream_license": require_source("harmbench").license,
            "redistribution": require_source("harmbench").redistribution,
            "local_input_path": str(source_path),
            "partition_reconstruction": partition_provenance,
        },
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HarmBenchSourceError(f"invalid JSON in {path} line {line_number}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise HarmBenchSourceError(f"{path} line {line_number} must be a JSON object")
        rows.append(decoded)
    return rows


def _nonempty(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not normalize_text(value):
        raise HarmBenchSourceError(f"{field} must be a non-empty string")
    return normalize_text(value)


def _optional_nonempty(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, field=field)


__all__ = [
    "DEFAULT_ID_FIELD",
    "DEFAULT_PROMPT_FIELD",
    "HARM_BENCH_SOURCE",
    "NORMALIZATION_VERSION",
    "HarmBenchSourceError",
    "materialize_harmbench_source",
    "normalize_harmbench_rows",
    "read_local_harmbench_export",
]
