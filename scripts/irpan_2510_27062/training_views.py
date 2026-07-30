"""Canonical, immutable paired-prompt training views for Irpan et al.

The paper adapter owns a single plain-JSONL boundary shared by ACT, AttCT,
MLPCT, OPCT, and the BCT target pipeline.  Adapter records are deliberately
flattened at this boundary, but their identity and lineage remain in every row
and in a verified sidecar manifest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ctm.artifacts import (
    ArtifactManifestError,
    artifact_manifest_path,
    read_verified_artifact_manifest,
    write_atomic_bytes,
)
from ctm_data.pairs import PairRowError, canonical_pair_rows
from scripts.irpan_2510_27062.artifacts import (
    MANIFEST_SHA256_FIELD,
    producer_identity,
    read_artifact,
)
from scripts.irpan_2510_27062.partitions import PartitionError, require_artifact_role
from scripts.irpan_2510_27062.schema import (
    PAPER_ID,
    canonical_json,
    normalize_json,
    normalize_text,
    require_sha256,
    sha256_bytes,
    sha256_json,
    sha256_text,
)

TRAINING_VIEW_SCHEMA = f"ctm_data.{PAPER_ID}.training_view"
TRAINING_VIEW_SCHEMA_VERSION = 1
TRAINING_VIEW_ARTIFACT_KIND = "canonical_training_view"
TRAINING_ROLE = "training"

SYCOPHANCY_DOMAIN = "sycophancy"
JAILBREAK_DOMAIN = "jailbreak"

_DOMAIN_INPUTS = {
    SYCOPHANCY_DOMAIN: ("sycophancy_prompt_pairs", "sycophancy_prompt_pair"),
    JAILBREAK_DOMAIN: ("act_training_exports", "act_training_export"),
}
_PAIR_REQUIRED_KEYS = {
    "pair_id",
    "source_id",
    "example_id",
    "domain",
    "source",
    "variant_id",
    "metadata",
    "reference_messages",
    "variant_messages",
}
_PAIR_OPTIONAL_KEYS = {
    "alignment_text",
    "correct_label",
    "suggested_wrong_label",
    "choices",
    "choice_labels",
}


class TrainingViewError(ValueError):
    """A source artifact cannot form the canonical paired training view."""


def materialize_training_view(
    source_path: str | Path,
    output_path: str | Path,
    *,
    domain: str,
) -> dict[str, Any]:
    """Export one verified ``role=training`` adapter artifact as plain JSONL.

    The accepted kind and record type are fixed per domain.  In particular,
    evaluation artifacts and structurally similar rows cannot cross this
    boundary merely because they happen to contain prompt-looking fields.
    """

    expected_kind, expected_type = _domain_spec(domain)
    try:
        source_rows, source_manifest = read_artifact(
            source_path,
            expected_kind=expected_kind,
            expected_role=TRAINING_ROLE,
        )
    except (ArtifactManifestError, TypeError, ValueError) as exc:
        raise TrainingViewError(str(exc)) from exc

    try:
        rows = canonical_pair_rows(_build_rows(source_rows, domain=domain, expected_type=expected_type))
    except PairRowError as exc:  # pragma: no cover - paper validation is intentionally stricter
        raise TrainingViewError(str(exc)) from exc
    source_identity = _source_identity(source_path, source_manifest)
    config = {
        "domain": domain,
        "input_artifact_kind": expected_kind,
        "input_record_type": expected_type,
        "output_schema": TRAINING_VIEW_SCHEMA,
        "output_schema_version": TRAINING_VIEW_SCHEMA_VERSION,
    }
    return _write_plain_artifact(
        output_path,
        rows,
        artifact_schema=TRAINING_VIEW_SCHEMA,
        schema_version=TRAINING_VIEW_SCHEMA_VERSION,
        artifact_kind=TRAINING_VIEW_ARTIFACT_KIND,
        role=TRAINING_ROLE,
        producer=producer_identity("irpan-canonical-training-view", __file__),
        config=config,
        parent_artifacts=[source_identity],
        provenance={
            "source_parent_identity": source_identity,
            "source_manifest_sha256": source_identity[MANIFEST_SHA256_FIELD],
            "stable_sort": ["domain", "source", "example_id", "variant_id", "pair_id"],
        },
    )


def write_training_view(
    source_path: str | Path,
    output_path: str | Path,
    *,
    domain: str,
) -> dict[str, Any]:
    """Alias for :func:`materialize_training_view`."""

    return materialize_training_view(source_path, output_path, domain=domain)


def read_training_view(
    path: str | Path,
    *,
    expected_domain: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify a canonical pair file, its sidecar, role, order, and every row."""

    rows, manifest = _read_plain_artifact(
        path,
        expected_schema=TRAINING_VIEW_SCHEMA,
        expected_schema_version=TRAINING_VIEW_SCHEMA_VERSION,
        expected_kind=TRAINING_VIEW_ARTIFACT_KIND,
        expected_role=TRAINING_ROLE,
    )
    try:
        validated = _validate_training_view_rows(canonical_pair_rows(rows))
    except PairRowError as exc:
        raise TrainingViewError(str(exc)) from exc
    provenance = manifest["provenance"]
    config = provenance["config"]
    configured_domain = config.get("domain")
    _domain_spec(configured_domain)
    actual_domains = {row["domain"] for row in validated}
    if actual_domains != {configured_domain}:
        raise TrainingViewError(
            f"training view rows have domains {sorted(actual_domains)!r}, but manifest config names "
            f"{configured_domain!r}"
        )
    expected_kind, expected_type = _domain_spec(configured_domain)
    if config.get("input_artifact_kind") != expected_kind or config.get("input_record_type") != expected_type:
        raise TrainingViewError("training view manifest input kind/type do not match its configured domain")
    if len(provenance["parent_artifacts"]) != 1:
        raise TrainingViewError("training view must have exactly one source parent artifact")
    parent = provenance["parent_artifacts"][0]
    source_parent = provenance.get("source_parent_identity")
    if source_parent != parent:
        raise TrainingViewError("training view source_parent_identity differs from parent_artifacts[0]")
    if parent.get("role") != TRAINING_ROLE:
        raise TrainingViewError("training view parent artifact is not role=training")
    if provenance.get("source_manifest_sha256") != parent.get(MANIFEST_SHA256_FIELD):
        raise TrainingViewError("training view source manifest digest differs from its parent identity")
    if {row["metadata"]["source_record_type"] for row in validated} != {expected_type}:
        raise TrainingViewError("training view row metadata record types differ from manifest provenance")
    if expected_domain is not None:
        _domain_spec(expected_domain)
        if actual_domains != {expected_domain}:
            raise TrainingViewError(
                f"training view domains are {sorted(actual_domains)!r}, expected only {expected_domain!r}"
            )
    return validated, manifest


def _build_rows(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    expected_type: str,
) -> list[dict[str, Any]]:
    if not source_rows:
        raise TrainingViewError("canonical training view requires at least one source row")
    output: list[dict[str, Any]] = []
    seen_source_records: set[str] = set()
    for index, source_row in enumerate(source_rows, start=1):
        record_type = source_row.get("record_type")
        if record_type != expected_type:
            raise TrainingViewError(
                f"{domain} source row {index} has record_type {record_type!r}, expected {expected_type!r}"
            )
        record_sha256 = _sha256(source_row.get("content_sha256"), field=f"source row {index}.content_sha256")
        if record_sha256 in seen_source_records:
            raise TrainingViewError(f"duplicate source record digest {record_sha256!r}")
        seen_source_records.add(record_sha256)
        if domain == SYCOPHANCY_DOMAIN:
            output.append(_sycophancy_pair(source_row))
        else:
            output.append(_jailbreak_pair(source_row))
    return _validate_training_view_rows(sorted(output, key=_sort_key))


def _sycophancy_pair(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(record.get("payload"), field="sycophancy payload")
    example_id = _text(record.get("example_id"), field="sycophancy example_id")
    clean_prompt = _text(payload.get("clean_prompt"), field="clean_prompt")
    wrapped_prompt = _text(payload.get("wrapped_prompt"), field="wrapped_prompt")
    if clean_prompt not in wrapped_prompt:
        raise TrainingViewError(f"sycophancy clean prompt is not contained in wrapped prompt for {example_id!r}")
    correct_label = _text(payload.get("correct_label"), field="correct_label")
    wrong_label = _text(payload.get("suggested_wrong_label"), field="suggested_wrong_label")
    if correct_label == wrong_label:
        raise TrainingViewError(f"sycophancy suggested label equals the gold label for {example_id!r}")
    choices = _choices(payload.get("choices"), field="choices")
    choice_labels = [choice["label"] for choice in choices]
    if correct_label not in choice_labels or wrong_label not in choice_labels:
        raise TrainingViewError(f"sycophancy labels are not both present in choices for {example_id!r}")
    base = {
        "example_id": example_id,
        "domain": SYCOPHANCY_DOMAIN,
        "source": _text(record.get("source"), field="source"),
        "variant_id": f"wrong_suggestion:{wrong_label}",
        "metadata": _row_metadata(record, extra={"variant_kind": "wrong_suggestion"}),
        "reference_messages": [{"role": "user", "content": clean_prompt}],
        "variant_messages": [{"role": "user", "content": wrapped_prompt}],
        "alignment_text": clean_prompt,
        "correct_label": correct_label,
        "suggested_wrong_label": wrong_label,
        "choices": choices,
        "choice_labels": choice_labels,
    }
    pair_id = _make_pair_id(base)
    return {"pair_id": pair_id, "source_id": pair_id, **base}


def _jailbreak_pair(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(record.get("payload"), field="jailbreak ACT payload")
    candidate_id = _text(payload.get("candidate_id"), field="candidate_id")
    source_id = _text(payload.get("source_id"), field="source_id")
    if candidate_id != source_id:
        raise TrainingViewError(f"ACT candidate_id/source_id mismatch for {candidate_id!r}")
    marker = ":wrapper:"
    if marker not in candidate_id:
        raise TrainingViewError(f"jailbreak candidate_id has no {marker!r} base-cluster boundary: {candidate_id!r}")
    example_id, wrapper_id = candidate_id.rsplit(marker, 1)
    if not example_id or not wrapper_id:
        raise TrainingViewError(f"malformed jailbreak candidate_id {candidate_id!r}")
    reference_messages = _messages(payload.get("reference_messages"), field="reference_messages")
    variant_messages = _messages(payload.get("variant_messages"), field="variant_messages")
    alignment_text = _text(payload.get("alignment_text"), field="alignment_text")
    if not _messages_contain(reference_messages, alignment_text) or not _messages_contain(
        variant_messages, alignment_text
    ):
        raise TrainingViewError(f"alignment_text is not present on both sides for {candidate_id!r}")
    _verify_optional_hash(payload, "alignment_text", alignment_text)
    _verify_optional_hash(payload, "clean_prompt", reference_messages[-1]["content"])
    _verify_optional_hash(payload, "wrapped_prompt", variant_messages[-1]["content"])
    base = {
        "example_id": example_id,
        "domain": JAILBREAK_DOMAIN,
        "source": _text(record.get("source"), field="source"),
        "variant_id": candidate_id,
        "metadata": _row_metadata(
            record,
            extra={
                "candidate_id": candidate_id,
                "wrapper_id": wrapper_id,
                "training_export_version": payload.get("training_export_version"),
            },
        ),
        "reference_messages": reference_messages,
        "variant_messages": variant_messages,
        "alignment_text": alignment_text,
    }
    pair_id = _make_pair_id(base)
    return {"pair_id": pair_id, "source_id": pair_id, **base}


def _validate_training_view_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise TrainingViewError("canonical training view is empty")
    validated: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise TrainingViewError(f"training view row {index} must be an object")
        missing = sorted(_PAIR_REQUIRED_KEYS - set(raw))
        extra = sorted(set(raw) - _PAIR_REQUIRED_KEYS - _PAIR_OPTIONAL_KEYS)
        if missing or extra:
            raise TrainingViewError(f"training view row {index} keys mismatch: missing={missing}, extra={extra}")
        row = dict(raw)
        pair_id = _text(row["pair_id"], field=f"row {index}.pair_id")
        if pair_id in pair_ids:
            raise TrainingViewError(f"duplicate pair_id {pair_id!r}")
        pair_ids.add(pair_id)
        domain = _text(row["domain"], field=f"row {index}.domain")
        _domain_spec(domain)
        normalized: dict[str, Any] = {
            "pair_id": pair_id,
            "source_id": _text(row["source_id"], field=f"row {index}.source_id"),
            "example_id": _text(row["example_id"], field=f"row {index}.example_id"),
            "domain": domain,
            "source": _text(row["source"], field=f"row {index}.source"),
            "variant_id": _text(row["variant_id"], field=f"row {index}.variant_id"),
            "metadata": _mapping(row["metadata"], field=f"row {index}.metadata"),
            "reference_messages": _messages(row["reference_messages"], field=f"row {index}.reference_messages"),
            "variant_messages": _messages(row["variant_messages"], field=f"row {index}.variant_messages"),
        }
        if normalized["source_id"] != pair_id:
            raise TrainingViewError(f"row {index} source_id must equal pair_id")
        alignment_text = row.get("alignment_text")
        if alignment_text is not None:
            alignment = _text(alignment_text, field=f"row {index}.alignment_text")
            if not _messages_contain(normalized["reference_messages"], alignment) or not _messages_contain(
                normalized["variant_messages"], alignment
            ):
                raise TrainingViewError(f"row {index} alignment_text is not present on both prompt sides")
            normalized["alignment_text"] = alignment
        if domain == SYCOPHANCY_DOMAIN:
            correct = _text(row.get("correct_label"), field=f"row {index}.correct_label")
            wrong = _text(row.get("suggested_wrong_label"), field=f"row {index}.suggested_wrong_label")
            if correct == wrong:
                raise TrainingViewError(f"row {index} sycophancy labels must differ")
            normalized["correct_label"] = correct
            normalized["suggested_wrong_label"] = wrong
            choices = _choices(row.get("choices"), field=f"row {index}.choices")
            labels = [_text(value, field=f"row {index}.choice_labels") for value in row.get("choice_labels", [])]
            expected_labels = [choice["label"] for choice in choices]
            if labels != expected_labels:
                raise TrainingViewError(f"row {index} choice_labels do not match ordered choices")
            if correct not in labels or wrong not in labels:
                raise TrainingViewError(f"row {index} sycophancy labels are not present in choice_labels")
            normalized["choices"] = choices
            normalized["choice_labels"] = labels
        elif {"correct_label", "suggested_wrong_label", "choices", "choice_labels"} & set(row):
            raise TrainingViewError(f"row {index} jailbreak pair cannot carry sycophancy gold labels")
        identity_material = {key: value for key, value in normalized.items() if key not in {"pair_id", "source_id"}}
        expected_pair_id = _make_pair_id(identity_material)
        if pair_id != expected_pair_id:
            raise TrainingViewError(
                f"row {index} pair_id mismatch: recorded {pair_id!r}, computed {expected_pair_id!r}"
            )
        validated.append(normalized)
    ordered = sorted(validated, key=_sort_key)
    if list(validated) != ordered:
        raise TrainingViewError("training view rows are not in canonical deterministic order")
    return validated


def _make_pair_id(material: Mapping[str, Any]) -> str:
    return f"{PAPER_ID}:training_pair:{sha256_json(material)}"


def _sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["domain"]),
        str(row["source"]),
        str(row["example_id"]),
        str(row["variant_id"]),
        str(row["pair_id"]),
    )


def _row_metadata(record: Mapping[str, Any], *, extra: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "source_record_type": record.get("record_type"),
        "source_record_content_sha256": _sha256(record.get("content_sha256"), field="source record digest"),
        "source_record_parent_hashes": [
            _sha256(value, field="source parent hash") for value in record.get("parent_hashes", [])
        ],
        "source_key": _text(record.get("source_key"), field="source_key"),
        "source_record_metadata": _mapping(record.get("metadata"), field="source record metadata"),
        **dict(extra),
    }
    return _mapping(metadata, field="pair metadata")


def _verify_optional_hash(payload: Mapping[str, Any], stem: str, value: str) -> None:
    field = f"{stem}_sha256"
    if field in payload and _sha256(payload[field], field=field) != sha256_text(value):
        raise TrainingViewError(f"{field} does not match {stem}")


def _domain_spec(domain: str) -> tuple[str, str]:
    if domain not in _DOMAIN_INPUTS:
        raise TrainingViewError(f"domain must be one of {sorted(_DOMAIN_INPUTS)}, got {domain!r}")
    return _DOMAIN_INPUTS[domain]


def _messages(value: Any, *, field: str) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise TrainingViewError(f"{field} must be a non-empty message list")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise TrainingViewError(f"{field}[{index}] must be an object")
        if set(message) != {"role", "content"}:
            raise TrainingViewError(f"{field}[{index}] must have exactly role/content fields")
        messages.append(
            {
                "role": _text(message.get("role"), field=f"{field}[{index}].role"),
                "content": _text(message.get("content"), field=f"{field}[{index}].content"),
            }
        )
    if messages[-1]["role"] != "user":
        raise TrainingViewError(f"{field} must end with a user message")
    return messages


def _messages_contain(messages: Sequence[Mapping[str, str]], text: str) -> bool:
    return any(text in message["content"] for message in messages)


def _choices(value: Any, *, field: str) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) < 2:
        raise TrainingViewError(f"{field} must contain at least two choices")
    choices: list[dict[str, str]] = []
    labels: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {"label", "text"}:
            raise TrainingViewError(f"{field}[{index}] must have exactly label/text fields")
        label = _text(raw.get("label"), field=f"{field}[{index}].label")
        if label in labels:
            raise TrainingViewError(f"{field} contains duplicate label {label!r}")
        labels.add(label)
        choices.append({"label": label, "text": _text(raw.get("text"), field=f"{field}[{index}].text")})
    return choices


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingViewError(f"{field} must be an object")
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise TrainingViewError(f"{field} must normalize to an object")
    return normalized


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not normalize_text(value):
        raise TrainingViewError(f"{field} must be a non-empty string")
    return normalize_text(value)


def _sha256(value: Any, *, field: str) -> str:
    try:
        return require_sha256(value, field=field)
    except ValueError as exc:
        raise TrainingViewError(str(exc)) from exc


def _source_identity(path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TrainingViewError("source artifact manifest has no provenance object")
    role = provenance.get("role")
    if role != TRAINING_ROLE:
        raise TrainingViewError(f"source artifact role is {role!r}, expected {TRAINING_ROLE!r}")
    return {
        "path": str(Path(path)),
        "artifact_schema": manifest.get("artifact_schema"),
        "schema_version": manifest.get("schema_version"),
        "artifact_kind": provenance.get("artifact_kind"),
        "role": role,
        "row_count": manifest.get("row_count"),
        "content_sha256": _sha256(manifest.get("content_sha256"), field="source manifest content_sha256"),
        MANIFEST_SHA256_FIELD: _sha256(
            manifest.get(MANIFEST_SHA256_FIELD), field=f"source manifest {MANIFEST_SHA256_FIELD}"
        ),
    }


def _write_plain_artifact(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact_schema: str,
    schema_version: int,
    artifact_kind: str,
    role: str,
    producer: Mapping[str, Any],
    config: Mapping[str, Any],
    parent_artifacts: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(path)
    sidecar = artifact_manifest_path(target)
    if target.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact pair: {target} / {sidecar}")
    if not rows:
        raise ArtifactManifestError("immutable training artifact must contain at least one row")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
        raise ArtifactManifestError("schema_version must be a positive integer")
    if not all(isinstance(value, str) and value for value in (artifact_schema, artifact_kind)):
        raise ArtifactManifestError("artifact schema, kind, and role must be non-empty strings")
    try:
        validated_role = require_artifact_role(role)
    except PartitionError as exc:
        raise ArtifactManifestError(str(exc)) from exc
    producer_copy = _mapping(producer, field="producer")
    if not isinstance(producer_copy.get("name"), str) or not producer_copy["name"]:
        raise ArtifactManifestError("producer has no name")
    _sha256(producer_copy.get("code_sha256"), field="producer.code_sha256")
    config_copy = _mapping(config, field="config")
    parents = [_mapping(parent, field="parent artifact identity") for parent in parent_artifacts]
    if not parents:
        raise ArtifactManifestError("derived artifact requires at least one parent artifact identity")
    extra = _mapping(provenance or {}, field="provenance")
    reserved = {
        "paper_id",
        "artifact_kind",
        "role",
        "created_at_utc",
        "producer",
        "config",
        "config_sha256",
        "parent_artifacts",
    }
    overlap = sorted(reserved & set(extra))
    if overlap:
        raise ArtifactManifestError(f"custom provenance cannot replace reserved fields: {overlap}")
    payload = b"".join((canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows)
    manifest_without_digest = {
        "artifact_schema": artifact_schema,
        "schema_version": schema_version,
        "row_count": len(rows),
        "content_sha256": sha256_bytes(payload),
        "provenance": {
            "paper_id": PAPER_ID,
            "artifact_kind": artifact_kind,
            "role": validated_role,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "producer": producer_copy,
            "config": config_copy,
            "config_sha256": sha256_json(config_copy),
            "parent_artifacts": parents,
            **extra,
        },
    }
    manifest = {
        **manifest_without_digest,
        MANIFEST_SHA256_FIELD: sha256_json(manifest_without_digest),
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_atomic_bytes(target, payload)
    write_atomic_bytes(sidecar, manifest_payload)
    return _verified_plain_manifest(
        target,
        expected_schema=artifact_schema,
        expected_schema_version=schema_version,
        expected_kind=artifact_kind,
        expected_role=role,
    )


def _read_plain_artifact(
    path: str | Path,
    *,
    expected_schema: str,
    expected_schema_version: int,
    expected_kind: str,
    expected_role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = Path(path)
    manifest = _verified_plain_manifest(
        target,
        expected_schema=expected_schema,
        expected_schema_version=expected_schema_version,
        expected_kind=expected_kind,
        expected_role=expected_role,
    )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactManifestError(f"invalid JSON in {target} line {line_number}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ArtifactManifestError(f"{target} line {line_number} must be a JSON object")
        rows.append(decoded)
    return rows, manifest


def _verified_plain_manifest(
    path: str | Path,
    *,
    expected_schema: str,
    expected_schema_version: int,
    expected_kind: str,
    expected_role: str,
) -> dict[str, Any]:
    manifest = read_verified_artifact_manifest(
        path,
        expected_schema=expected_schema,
        expected_schema_version=expected_schema_version,
    )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ArtifactManifestError("artifact manifest has no provenance object")
    if provenance.get("role") != expected_role:
        raise ArtifactManifestError(f"artifact role is {provenance.get('role')!r}, expected {expected_role!r}")
    recorded_manifest_hash = _sha256(
        manifest.get(MANIFEST_SHA256_FIELD), field=f"artifact manifest {MANIFEST_SHA256_FIELD}"
    )
    unsigned_manifest = {key: value for key, value in manifest.items() if key != MANIFEST_SHA256_FIELD}
    if recorded_manifest_hash != sha256_json(unsigned_manifest):
        raise ArtifactManifestError("artifact manifest integrity digest mismatch")
    if provenance.get("paper_id") != PAPER_ID:
        raise ArtifactManifestError(f"artifact paper_id is {provenance.get('paper_id')!r}, expected {PAPER_ID!r}")
    if provenance.get("artifact_kind") != expected_kind:
        raise ArtifactManifestError(f"artifact kind is {provenance.get('artifact_kind')!r}, expected {expected_kind!r}")
    producer = provenance.get("producer")
    if not isinstance(producer, Mapping) or not isinstance(producer.get("name"), str):
        raise ArtifactManifestError("artifact manifest has no producer identity")
    _sha256(producer.get("code_sha256"), field="producer.code_sha256")
    config = provenance.get("config")
    if not isinstance(config, Mapping):
        raise ArtifactManifestError("artifact manifest has no config object")
    recorded_config_hash = _sha256(provenance.get("config_sha256"), field="config_sha256")
    if recorded_config_hash != sha256_json(config):
        raise ArtifactManifestError("artifact config_sha256 does not match its config object")
    parents = provenance.get("parent_artifacts")
    if not isinstance(parents, list) or not parents:
        raise ArtifactManifestError("artifact manifest has no parent artifact identities")
    for index, parent in enumerate(parents):
        if not isinstance(parent, Mapping):
            raise ArtifactManifestError(f"parent_artifacts[{index}] must be an object")
        _sha256(parent.get("content_sha256"), field=f"parent_artifacts[{index}].content_sha256")
        if MANIFEST_SHA256_FIELD in parent:
            _sha256(parent.get(MANIFEST_SHA256_FIELD), field=f"parent_artifacts[{index}].{MANIFEST_SHA256_FIELD}")
    return manifest


__all__ = [
    "JAILBREAK_DOMAIN",
    "SYCOPHANCY_DOMAIN",
    "TRAINING_ROLE",
    "TRAINING_VIEW_ARTIFACT_KIND",
    "TRAINING_VIEW_SCHEMA",
    "TRAINING_VIEW_SCHEMA_VERSION",
    "TrainingViewError",
    "materialize_training_view",
    "read_training_view",
    "write_training_view",
]
