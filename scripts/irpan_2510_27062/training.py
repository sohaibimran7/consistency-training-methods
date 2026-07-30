"""Lineage-preserving BCT/ACT exports for retained jailbreak candidates.

Fresh BCT targets come only from the clean member of the paired completion
artifact used by this DAG.  Separately imported stale targets have their own
record type, require a provider/model/revision/date identity, and can only be
exported through the explicitly stale API.  They are never silently promoted
to fresh targets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.irpan_2510_27062.artifacts import producer_identity, read_artifact, write_artifact
from scripts.irpan_2510_27062.partitions import TRAINING
from scripts.irpan_2510_27062.schema import (
    RecordSchemaError,
    make_derived_record,
    normalize_json,
    normalize_text,
    require_sha256,
    sha256_bytes,
    sha256_json,
    sha256_text,
    validate_record,
)
from scripts.irpan_2510_27062.wrappers import (
    read_external_result_export,
    validate_external_identity,
)

TRAINING_EXPORT_VERSION = "reconstruction_v1"
STALE_TARGET_IMPORT_VERSION = "reconstruction_v1"
FRESH_TARGET = "fresh_clean_completion"
STALE_TARGET = "stale_external_completion"


class TrainingExportError(ValueError):
    """Retained candidates and targets cannot form an auditable export."""


def build_bct_training_rows(
    retained_rows: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map each wrapped prompt to its paired, freshly generated clean target."""

    retained = _validated_retained(retained_rows)
    completions = _unique_records(completion_rows, expected_type="external_completion")
    completion_pairs = _completion_pairs(completions)
    rows: list[dict[str, Any]] = []
    for candidate in retained:
        payload = candidate["payload"]
        candidate_id = payload["candidate_id"]
        if candidate_id not in completion_pairs:
            raise TrainingExportError(f"fresh completions missing retained candidate {candidate_id!r}")
        clean_completion = completion_pairs[candidate_id]["clean"]
        wrapped_completion = completion_pairs[candidate_id]["wrapped"]
        retained_completion_hashes = _hash_mapping(
            payload.get("completion_content_sha256"),
            field="completion_content_sha256",
            keys={"clean", "wrapped"},
        )
        if retained_completion_hashes != {
            "clean": clean_completion["content_sha256"],
            "wrapped": wrapped_completion["content_sha256"],
        }:
            raise TrainingExportError(f"fresh completion artifact differs from judged lineage for {candidate_id!r}")
        target_payload = clean_completion["payload"]
        if target_payload["candidate_content_sha256"] != payload["candidate_content_sha256"]:
            raise TrainingExportError(f"fresh target has stale candidate lineage for {candidate_id!r}")
        target = _text(target_payload, "response", context=clean_completion["example_id"])
        target_sha256 = _sha256(target_payload.get("response_sha256"), field="response_sha256")
        if target_sha256 != sha256_text(target):
            raise TrainingExportError(f"fresh target response digest mismatch for {candidate_id!r}")
        generator = _mapping(target_payload.get("generator"), field="generator")
        generator_sha256 = _sha256(target_payload.get("generator_identity_sha256"), field="generator_identity_sha256")
        if generator_sha256 != sha256_json(generator):
            raise TrainingExportError(f"fresh target generator identity mismatch for {candidate_id!r}")
        export_id = f"{candidate_id}:training:bct:fresh:{target_sha256[:16]}"
        rows.append(
            make_derived_record(
                record_type="bct_training_export",
                example_id=export_id,
                source="harmbench",
                source_key=f"{candidate['source_key']}::bct::fresh",
                payload={
                    "source_id": candidate_id,
                    "candidate_id": candidate_id,
                    "messages": [
                        {"role": "user", "content": payload["wrapped_prompt"]},
                        {"role": "assistant", "content": target},
                    ],
                    "wrapped_prompt_sha256": payload["wrapped_prompt_sha256"],
                    "target_completion": target,
                    "target_completion_sha256": target_sha256,
                    "target_freshness": FRESH_TARGET,
                    "target_origin_condition": "clean",
                    "target_completion_request_id": target_payload["request_id"],
                    "target_completion_content_sha256": clean_completion["content_sha256"],
                    "target_generator": generator,
                    "target_generator_identity_sha256": generator_sha256,
                    "target_prompt_template_version": target_payload["prompt_template_version"],
                    "target_prompt_template_sha256": target_payload["prompt_template_sha256"],
                    "candidate_content_sha256": payload["candidate_content_sha256"],
                    "retained_content_sha256": candidate["content_sha256"],
                    "training_export_version": TRAINING_EXPORT_VERSION,
                },
                parent_hashes=[candidate["content_sha256"], clean_completion["content_sha256"]],
                metadata={
                    "training_method": "bct",
                    "target_is_fresh": True,
                    "compatible_messages_field": "messages",
                    "model_calls_performed": 0,
                },
            )
        )
    return rows


def build_act_training_rows(retained_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Export prompt pairs using the repository's ACT field conventions."""

    retained = _validated_retained(retained_rows)
    rows: list[dict[str, Any]] = []
    for candidate in retained:
        payload = candidate["payload"]
        candidate_id = payload["candidate_id"]
        alignment = _mapping(payload.get("alignment"), field="alignment")
        alignment_strategy = _text(alignment, "strategy", context=candidate_id)
        alignment_text = _text(alignment, "alignment_text", context=candidate_id)
        shared_suffix = _text(payload, "shared_suffix", context=candidate_id)
        alignment_suffix = _text(alignment, "shared_suffix", context=candidate_id)
        clean_prompt = payload["clean_prompt"]
        wrapped_prompt = payload["wrapped_prompt"]
        if alignment_strategy != "core_request_as_shared_terminal_span":
            raise TrainingExportError(f"unsupported ACT alignment strategy for {candidate_id!r}")
        if alignment_suffix != shared_suffix or alignment_text != shared_suffix:
            raise TrainingExportError(f"ACT alignment metadata is inconsistent for {candidate_id!r}")
        if alignment_text not in clean_prompt or alignment_text not in wrapped_prompt:
            raise TrainingExportError(f"ACT alignment text is not present on both sides for {candidate_id!r}")
        if not clean_prompt.endswith(shared_suffix) or not wrapped_prompt.endswith(shared_suffix):
            raise TrainingExportError(f"ACT shared suffix is not preserved on both sides for {candidate_id!r}")
        export_id = f"{candidate_id}:training:act:{TRAINING_EXPORT_VERSION}"
        rows.append(
            make_derived_record(
                record_type="act_training_export",
                example_id=export_id,
                source="harmbench",
                source_key=f"{candidate['source_key']}::act",
                payload={
                    "source_id": candidate_id,
                    "candidate_id": candidate_id,
                    "variant_messages": [{"role": "user", "content": wrapped_prompt}],
                    "reference_messages": [{"role": "user", "content": clean_prompt}],
                    "alignment_text": alignment_text,
                    "alignment_text_sha256": sha256_text(alignment_text),
                    "alignment_text_field": "alignment_text",
                    "alignment_strategy": alignment_strategy,
                    "shared_suffix": shared_suffix,
                    "shared_suffix_sha256": payload["shared_suffix_sha256"],
                    "clean_prompt_sha256": payload["clean_prompt_sha256"],
                    "wrapped_prompt_sha256": payload["wrapped_prompt_sha256"],
                    "candidate_content_sha256": payload["candidate_content_sha256"],
                    "retained_content_sha256": candidate["content_sha256"],
                    "training_export_version": TRAINING_EXPORT_VERSION,
                },
                parent_hashes=[candidate["content_sha256"]],
                metadata={
                    "training_method": "act",
                    "compatible_reference_messages_field": "reference_messages",
                    "compatible_variant_messages_field": "variant_messages",
                    "compatible_alignment_text_field": "alignment_text",
                    "model_calls_performed": 0,
                },
            )
        )
    return rows


def build_training_exports(
    retained_rows: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the primary fresh-target BCT export and the ACT pair export."""

    return build_bct_training_rows(retained_rows, completion_rows), build_act_training_rows(retained_rows)


def extract_training_payloads(
    export_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
) -> list[dict[str, Any]]:
    """Expose convention-compatible payloads while retaining artifact lineage IDs.

    The immutable artifact remains the source of truth.  This helper produces
    the objects consumed by ``ctm.training.sft`` and adds the canonical record
    digest so derived training files can cite their origin.
    """

    expected = {
        "bct": {"bct_training_export", "bct_stale_training_export"},
        "act": {"act_training_export"},
    }
    if method not in expected:
        raise TrainingExportError("method must be 'bct' or 'act'")
    payloads: list[dict[str, Any]] = []
    for row in export_rows:
        try:
            plain = validate_record(row)
        except RecordSchemaError as exc:
            raise TrainingExportError(str(exc)) from exc
        if plain["record_type"] not in expected[method]:
            raise TrainingExportError(
                f"{method} payload extraction does not accept record_type {plain['record_type']!r}"
            )
        payload = dict(plain["payload"])
        payload["artifact_record_content_sha256"] = plain["content_sha256"]
        payload["artifact_parent_hashes"] = list(plain["parent_hashes"])
        payloads.append(payload)
    return payloads


def import_stale_targets(
    retained_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    *,
    target_model: Mapping[str, Any],
    input_manifest_sha256: str,
) -> list[dict[str, Any]]:
    """Import explicitly stale targets with provider/model/revision/date identity."""

    retained = _validated_retained(retained_rows)
    retained_by_id = {row["payload"]["candidate_id"]: row for row in retained}
    try:
        model_identity = validate_external_identity(
            target_model,
            role="stale target model",
            require_revision_and_date=True,
        )
    except ValueError as exc:
        raise TrainingExportError(str(exc)) from exc
    input_digest = _sha256(input_manifest_sha256, field="input_manifest_sha256")
    imported: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(result_rows, start=1):
        if not isinstance(raw, Mapping):
            raise TrainingExportError(f"stale target result {index} must be an object")
        candidate_id = raw.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise TrainingExportError(f"stale target result {index} has no candidate_id")
        if candidate_id in imported:
            raise TrainingExportError(f"duplicate stale target candidate ID {candidate_id!r}")
        target = raw.get("target")
        if not isinstance(target, str) or not normalize_text(target):
            raise TrainingExportError(f"stale target for {candidate_id!r} has no non-empty target")
        target = normalize_text(target)
        target_sha256 = sha256_text(target)
        supplied_digest = raw.get("target_sha256")
        if supplied_digest is not None and _sha256(supplied_digest, field="target_sha256") != target_sha256:
            raise TrainingExportError(f"stale target digest mismatch for {candidate_id!r}")
        imported[candidate_id] = {
            "target": target,
            "target_sha256": target_sha256,
            "metadata": _mapping(raw.get("metadata", {}), field=f"stale target {candidate_id!r} metadata"),
        }
    _exact_ids(retained_by_id, imported, label="stale targets")

    model_sha256 = sha256_json(model_identity)
    stale_rows: list[dict[str, Any]] = []
    for candidate in retained:
        candidate_id = candidate["payload"]["candidate_id"]
        result = imported[candidate_id]
        stale_id = f"{candidate_id}:stale_target:{model_sha256[:12]}:{result['target_sha256'][:12]}"
        stale_rows.append(
            make_derived_record(
                record_type="stale_target",
                example_id=stale_id,
                source="harmbench",
                source_key=f"{candidate['source_key']}::stale_target",
                payload={
                    "stale_target_id": stale_id,
                    "candidate_id": candidate_id,
                    "candidate_content_sha256": candidate["payload"]["candidate_content_sha256"],
                    "retained_content_sha256": candidate["content_sha256"],
                    "target": result["target"],
                    "target_sha256": result["target_sha256"],
                    "target_freshness": STALE_TARGET,
                    "target_model": model_identity,
                    "target_model_identity_sha256": model_sha256,
                    "target_model_revision": model_identity["revision"],
                    "target_generation_date": model_identity["date"],
                    "input_manifest_sha256": input_digest,
                },
                parent_hashes=[candidate["content_sha256"]],
                metadata={
                    "import_version": STALE_TARGET_IMPORT_VERSION,
                    "external_result_metadata": result["metadata"],
                    "target_is_fresh": False,
                    "never_promote_to_fresh": True,
                    "model_called_by_adapter": False,
                },
            )
        )
    return stale_rows


def build_stale_bct_training_rows(
    retained_rows: Sequence[Mapping[str, Any]],
    stale_target_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a separately labelled BCT export from explicitly stale targets."""

    retained = _validated_retained(retained_rows)
    stale = _unique_records(stale_target_rows, expected_type="stale_target")
    stale_by_candidate: dict[str, dict[str, Any]] = {}
    for row in stale:
        payload = row["payload"]
        candidate_id = _text(payload, "candidate_id", context=row["example_id"])
        if candidate_id in stale_by_candidate:
            raise TrainingExportError(f"duplicate stale target for candidate {candidate_id!r}")
        if payload.get("target_freshness") != STALE_TARGET:
            raise TrainingExportError(f"stale target row is not labelled stale for {candidate_id!r}")
        model = _mapping(payload.get("target_model"), field="target_model")
        try:
            validate_external_identity(
                model,
                role="stale target model",
                require_revision_and_date=True,
            )
        except ValueError as exc:
            raise TrainingExportError(str(exc)) from exc
        stale_by_candidate[candidate_id] = row
    retained_by_id = {row["payload"]["candidate_id"]: row for row in retained}
    _exact_ids(retained_by_id, stale_by_candidate, label="stale target records")

    exports: list[dict[str, Any]] = []
    for candidate in retained:
        payload = candidate["payload"]
        candidate_id = payload["candidate_id"]
        stale_row = stale_by_candidate[candidate_id]
        target_payload = stale_row["payload"]
        if target_payload["retained_content_sha256"] != candidate["content_sha256"]:
            raise TrainingExportError(f"stale target has stale retained lineage for {candidate_id!r}")
        target = _text(target_payload, "target", context=stale_row["example_id"])
        target_sha256 = _sha256(target_payload.get("target_sha256"), field="target_sha256")
        if target_sha256 != sha256_text(target):
            raise TrainingExportError(f"stale target response digest mismatch for {candidate_id!r}")
        export_id = f"{candidate_id}:training:bct:stale:{target_sha256[:16]}"
        exports.append(
            make_derived_record(
                record_type="bct_stale_training_export",
                example_id=export_id,
                source="harmbench",
                source_key=f"{candidate['source_key']}::bct::stale",
                payload={
                    "source_id": candidate_id,
                    "candidate_id": candidate_id,
                    "messages": [
                        {"role": "user", "content": payload["wrapped_prompt"]},
                        {"role": "assistant", "content": target},
                    ],
                    "wrapped_prompt_sha256": payload["wrapped_prompt_sha256"],
                    "target_completion": target,
                    "target_completion_sha256": target_sha256,
                    "target_freshness": STALE_TARGET,
                    "stale_target_content_sha256": stale_row["content_sha256"],
                    "target_model": target_payload["target_model"],
                    "target_model_identity_sha256": target_payload["target_model_identity_sha256"],
                    "candidate_content_sha256": payload["candidate_content_sha256"],
                    "retained_content_sha256": candidate["content_sha256"],
                    "training_export_version": TRAINING_EXPORT_VERSION,
                },
                parent_hashes=[candidate["content_sha256"], stale_row["content_sha256"]],
                metadata={
                    "training_method": "bct",
                    "target_is_fresh": False,
                    "never_promote_to_fresh": True,
                    "compatible_messages_field": "messages",
                    "model_calls_performed": 0,
                },
            )
        )
    return exports


def materialize_training_exports(
    retained_path: str | Path,
    completion_path: str | Path,
    bct_output_path: str | Path,
    act_output_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish the primary fresh BCT and ACT artifacts."""

    if Path(bct_output_path).resolve() == Path(act_output_path).resolve():
        raise TrainingExportError("BCT and ACT outputs must be different paths")
    retained, retained_manifest = read_artifact(
        retained_path,
        expected_kind="retained_vulnerabilities",
        expected_role=TRAINING,
    )
    completions, completion_manifest = read_artifact(
        completion_path,
        expected_kind="external_completions",
        expected_role=TRAINING,
    )
    bct_rows, act_rows = build_training_exports(retained, completions)
    producer = producer_identity("export_jailbreak_training_data", __file__)
    bct_target_hashes = {row["payload"]["candidate_id"]: row["payload"]["target_completion_sha256"] for row in bct_rows}
    bct_manifest = write_artifact(
        bct_output_path,
        bct_rows,
        artifact_kind="bct_training_exports",
        role=TRAINING,
        producer=producer,
        config={
            "training_export_version": TRAINING_EXPORT_VERSION,
            "method": "bct",
            "target_freshness": FRESH_TARGET,
            "retained_manifest_sha256": sha256_json(retained_manifest),
            "completion_manifest_sha256": sha256_json(completion_manifest),
        },
        parent_artifacts=[retained_path, completion_path],
        provenance={
            "one_target_per_retained_candidate": True,
            "target_hashes": bct_target_hashes,
            "target_hashes_sha256": sha256_json(bct_target_hashes),
            "model_calls_performed": 0,
        },
    )
    act_manifest = write_artifact(
        act_output_path,
        act_rows,
        artifact_kind="act_training_exports",
        role=TRAINING,
        producer=producer,
        config={
            "training_export_version": TRAINING_EXPORT_VERSION,
            "method": "act",
            "reference_messages_field": "reference_messages",
            "variant_messages_field": "variant_messages",
            "alignment_text_field": "alignment_text",
            "retained_manifest_sha256": sha256_json(retained_manifest),
        },
        parent_artifacts=[retained_path],
        provenance={"one_pair_per_retained_candidate": True, "model_calls_performed": 0},
    )
    return bct_manifest, act_manifest


def materialize_stale_targets(
    retained_path: str | Path,
    external_results_path: str | Path,
    output_path: str | Path,
    *,
    target_model: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a separately typed import of stale clean-prompt targets."""

    retained, retained_manifest = read_artifact(
        retained_path,
        expected_kind="retained_vulnerabilities",
        expected_role=TRAINING,
    )
    result_path = Path(external_results_path)
    results = read_external_result_export(result_path)
    try:
        model_identity = validate_external_identity(
            target_model,
            role="stale target model",
            require_revision_and_date=True,
        )
    except ValueError as exc:
        raise TrainingExportError(str(exc)) from exc
    stale_rows = import_stale_targets(
        retained,
        results,
        target_model=model_identity,
        input_manifest_sha256=sha256_json(retained_manifest),
    )
    target_hashes = {row["payload"]["candidate_id"]: row["payload"]["target_sha256"] for row in stale_rows}
    return write_artifact(
        output_path,
        stale_rows,
        artifact_kind="stale_targets",
        role=TRAINING,
        producer=producer_identity("import_stale_jailbreak_targets", __file__),
        config={
            "import_version": STALE_TARGET_IMPORT_VERSION,
            "target_freshness": STALE_TARGET,
            "target_model": model_identity,
            "target_model_identity_sha256": sha256_json(model_identity),
            "retained_manifest_sha256": sha256_json(retained_manifest),
            "external_results_sha256": sha256_bytes(result_path.read_bytes()),
        },
        parent_artifacts=[retained_path],
        provenance={
            "external_results_path": str(result_path),
            "target_hashes": target_hashes,
            "target_hashes_sha256": sha256_json(target_hashes),
            "target_is_fresh": False,
            "model_calls_performed": 0,
        },
    )


def materialize_stale_bct_export(
    retained_path: str | Path,
    stale_target_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Publish a BCT artifact that remains unambiguously labelled stale."""

    retained, retained_manifest = read_artifact(
        retained_path,
        expected_kind="retained_vulnerabilities",
        expected_role=TRAINING,
    )
    stale_targets, stale_manifest = read_artifact(
        stale_target_path,
        expected_kind="stale_targets",
        expected_role=TRAINING,
    )
    rows = build_stale_bct_training_rows(retained, stale_targets)
    return write_artifact(
        output_path,
        rows,
        artifact_kind="bct_stale_training_exports",
        role=TRAINING,
        producer=producer_identity("export_stale_jailbreak_bct_data", __file__),
        config={
            "training_export_version": TRAINING_EXPORT_VERSION,
            "method": "bct",
            "target_freshness": STALE_TARGET,
            "retained_manifest_sha256": sha256_json(retained_manifest),
            "stale_target_manifest_sha256": sha256_json(stale_manifest),
        },
        parent_artifacts=[retained_path, stale_target_path],
        provenance={"target_is_fresh": False, "model_calls_performed": 0},
    )


def _validated_retained(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    retained = _unique_records(rows, expected_type="vulnerable_candidate")
    if not retained:
        raise TrainingExportError("training export requires at least one retained candidate")
    for row in retained:
        payload = row["payload"]
        candidate_id = _text(payload, "candidate_id", context=row["example_id"])
        if candidate_id != row["example_id"]:
            raise TrainingExportError(f"retained candidate ID mismatch in {row['example_id']!r}")
        for prompt_field, digest_field in (
            ("clean_prompt", "clean_prompt_sha256"),
            ("wrapped_prompt", "wrapped_prompt_sha256"),
            ("shared_suffix", "shared_suffix_sha256"),
        ):
            value = _text(payload, prompt_field, context=candidate_id)
            if _sha256(payload.get(digest_field), field=digest_field) != sha256_text(value):
                raise TrainingExportError(f"{digest_field} mismatch for {candidate_id!r}")
        _sha256(payload.get("candidate_content_sha256"), field="candidate_content_sha256")
        _hash_mapping(
            payload.get("completion_content_sha256"),
            field="completion_content_sha256",
            keys={"clean", "wrapped"},
        )
    return sorted(retained, key=lambda row: row["example_id"])


def _completion_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        payload = row["payload"]
        candidate_id = _text(payload, "candidate_id", context=row["example_id"])
        condition = _text(payload, "condition", context=row["example_id"])
        if condition not in {"clean", "wrapped"}:
            raise TrainingExportError(f"invalid completion condition {condition!r}")
        group = grouped.setdefault(candidate_id, {})
        if condition in group:
            raise TrainingExportError(f"duplicate {condition} completion for {candidate_id!r}")
        group[condition] = row
    for candidate_id, conditions in grouped.items():
        if set(conditions) != {"clean", "wrapped"}:
            raise TrainingExportError(
                f"candidate {candidate_id!r} requires exactly clean/wrapped completions; got {sorted(conditions)}"
            )
    return grouped


def _unique_records(rows: Sequence[Mapping[str, Any]], *, expected_type: str) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        try:
            plain = validate_record(row, expected_type=expected_type)
        except RecordSchemaError as exc:
            raise TrainingExportError(str(exc)) from exc
        identity = plain["example_id"]
        if identity in seen:
            raise TrainingExportError(f"duplicate {expected_type} example_id {identity!r}")
        seen.add(identity)
        validated.append(plain)
    return validated


def _exact_ids(expected: Mapping[str, Any], actual: Mapping[str, Any], *, label: str) -> None:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise TrainingExportError(f"{label} ID mismatch: missing={missing}, extra={extra}")


def _text(payload: Mapping[str, Any], field: str, *, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not normalize_text(value):
        raise TrainingExportError(f"{context} needs non-empty string {field!r}")
    return normalize_text(value)


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingExportError(f"{field} must be an object")
    normalized = normalize_json(value)
    if not isinstance(normalized, dict):
        raise TrainingExportError(f"{field} must normalize to an object")
    return normalized


def _hash_mapping(value: Any, *, field: str, keys: set[str]) -> dict[str, str]:
    mapping = _mapping(value, field=field)
    if set(mapping) != keys:
        raise TrainingExportError(f"{field} must have exactly keys {sorted(keys)}")
    return {key: _sha256(mapping[key], field=f"{field}.{key}") for key in sorted(keys)}


def _sha256(value: Any, *, field: str) -> str:
    try:
        return require_sha256(value, field=field)
    except RecordSchemaError as exc:
        raise TrainingExportError(str(exc)) from exc


__all__ = [
    "FRESH_TARGET",
    "STALE_TARGET",
    "STALE_TARGET_IMPORT_VERSION",
    "TRAINING_EXPORT_VERSION",
    "TrainingExportError",
    "build_act_training_rows",
    "build_bct_training_rows",
    "build_stale_bct_training_rows",
    "build_training_exports",
    "extract_training_payloads",
    "import_stale_targets",
    "materialize_stale_bct_export",
    "materialize_stale_targets",
    "materialize_training_exports",
]
