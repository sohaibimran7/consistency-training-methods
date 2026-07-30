"""Strict, offline BCT target artifacts derived from canonical pair views.

This module performs no model or network calls.  It materializes clean-prompt
requests, strictly imports separately produced responses, and joins the
verified targets back onto the variant prompts as trainer-native JSONL.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm.artifacts import ArtifactManifestError
from ctm_data.adapters.irpan_2510_27062.artifacts import MANIFEST_SHA256_FIELD, producer_identity
from ctm_data.adapters.irpan_2510_27062.provenance import (
    ProvenanceError,
    make_generator_identity,
    validate_generator_identity,
)
from ctm_data.adapters.irpan_2510_27062.schema import (
    PAPER_ID,
    normalize_json,
    normalize_text,
    require_sha256,
    sha256_json,
    sha256_text,
)
from ctm_data.adapters.irpan_2510_27062.training_views import (
    TRAINING_ROLE,
    TRAINING_VIEW_SCHEMA,
    TRAINING_VIEW_SCHEMA_VERSION,
    TrainingViewError,
    _read_plain_artifact,
    _write_plain_artifact,
    read_training_view,
)

BCT_TARGET_REQUEST_SCHEMA = f"ctm_data.{PAPER_ID}.bct_target_request"
BCT_TARGET_REQUEST_SCHEMA_VERSION = 1
BCT_TARGET_REQUEST_ARTIFACT_KIND = "bct_target_requests"
BCT_TARGET_REQUEST_ROLE = TRAINING_ROLE

BCT_TARGET_SCHEMA = f"ctm_data.{PAPER_ID}.bct_target"
BCT_TARGET_SCHEMA_VERSION = 1
BCT_TARGET_ARTIFACT_KIND = "bct_targets"
BCT_TARGET_ROLE = TRAINING_ROLE

BCT_TRAINING_SCHEMA = f"ctm_data.{PAPER_ID}.bct_training"
BCT_TRAINING_SCHEMA_VERSION = 1
BCT_TRAINING_ARTIFACT_KIND = "bct_training_jsonl"

_REQUEST_KEYS = {
    "pair_id",
    "source_id",
    "example_id",
    "domain",
    "source",
    "clean_prompt",
    "clean_prompt_sha256",
    "reference_messages",
    "reference_messages_sha256",
    "pair_record_sha256",
    "training_view_content_sha256",
    "training_view_manifest_sha256",
    "request_record_sha256",
}
_TARGET_KEYS = {
    "pair_id",
    "source_id",
    "example_id",
    "domain",
    "source",
    "clean_prompt",
    "clean_prompt_sha256",
    "reference_messages",
    "reference_messages_sha256",
    "response",
    "response_sha256",
    "generator_identity",
    "generator_identity_sha256",
    "decoding_parameters",
    "decoding_parameters_sha256",
    "pair_record_sha256",
    "request_record_sha256",
    "request_artifact_content_sha256",
    "training_view_content_sha256",
    "metadata",
    "target_record_sha256",
}
_TRAINING_KEYS = {
    "pair_id",
    "source_id",
    "example_id",
    "domain",
    "source",
    "messages",
    "clean_prompt_sha256",
    "reference_messages_sha256",
    "variant_messages_sha256",
    "response_sha256",
    "generator_identity_sha256",
    "decoding_parameters_sha256",
    "metadata",
    "training_record_sha256",
}


class BCTTargetError(ValueError):
    """BCT request, target, or training lineage is incomplete or inconsistent."""


def make_external_generator_identity(
    *,
    generator_id: str,
    provider: str,
    model: str,
    revision: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    """Explicitly map external ``revision/date`` vocabulary to canonical keys."""

    try:
        return make_generator_identity(
            generator_id=generator_id,
            provider=provider,
            model=model,
            model_revision=revision,
            model_immutable_date=date,
        )
    except ProvenanceError as exc:
        raise BCTTargetError(str(exc)) from exc


def make_fixture_generator_identity(fixture_id: str) -> dict[str, Any]:
    """Build a canonical, immutable identity for deterministic offline fixtures."""

    stable_id = _text(fixture_id, field="fixture_id")
    try:
        return make_generator_identity(
            generator_id=f"fixture:{stable_id}",
            provider="offline_fixture",
            model="bct_target_fixture",
            model_revision=f"fixture:{stable_id}",
        )
    except ProvenanceError as exc:  # pragma: no cover - fields above are fixed and valid
        raise BCTTargetError(str(exc)) from exc


def materialize_bct_target_requests(
    training_view_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Publish clean/reference-only requests from a verified training view."""

    try:
        pairs, view_manifest = read_training_view(training_view_path)
    except (ArtifactManifestError, TrainingViewError) as exc:
        raise BCTTargetError(str(exc)) from exc
    view_content_sha256 = _sha256(view_manifest.get("content_sha256"), field="training view content_sha256")
    view_manifest_sha256 = _sha256(
        view_manifest.get(MANIFEST_SHA256_FIELD), field=f"training view {MANIFEST_SHA256_FIELD}"
    )
    requests: list[dict[str, Any]] = []
    for pair in pairs:
        reference_messages = _prompt_messages(pair["reference_messages"], field=f"{pair['pair_id']}.reference_messages")
        clean_prompt = _clean_prompt(reference_messages)
        base = {
            "pair_id": pair["pair_id"],
            "source_id": pair["pair_id"],
            "example_id": pair["example_id"],
            "domain": pair["domain"],
            "source": pair["source"],
            "clean_prompt": clean_prompt,
            "clean_prompt_sha256": sha256_text(clean_prompt),
            "reference_messages": reference_messages,
            "reference_messages_sha256": sha256_json(reference_messages),
            "pair_record_sha256": sha256_json(pair),
            "training_view_content_sha256": view_content_sha256,
            "training_view_manifest_sha256": view_manifest_sha256,
        }
        requests.append({**base, "request_record_sha256": sha256_json(base)})
    requests = _validate_request_rows(sorted(requests, key=lambda row: row["pair_id"]))
    view_identity = _artifact_identity(training_view_path, view_manifest)
    return _write_plain_artifact(
        output_path,
        requests,
        artifact_schema=BCT_TARGET_REQUEST_SCHEMA,
        schema_version=BCT_TARGET_REQUEST_SCHEMA_VERSION,
        artifact_kind=BCT_TARGET_REQUEST_ARTIFACT_KIND,
        role=BCT_TARGET_REQUEST_ROLE,
        producer=producer_identity(
            "irpan-bct-target-request-builder",
            __file__,
            Path(__file__).with_name("training_views.py"),
        ),
        config={
            "input_schema": TRAINING_VIEW_SCHEMA,
            "input_schema_version": TRAINING_VIEW_SCHEMA_VERSION,
            "clean_prompt_source": "reference_messages",
            "model_calls_performed": 0,
        },
        parent_artifacts=[view_identity],
        provenance={
            "training_view_identity": view_identity,
            "training_view_manifest_sha256": view_manifest_sha256,
            "external_execution_required": True,
            "model_calls_performed": 0,
        },
    )


def read_bct_target_requests(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify a target-request artifact and every clean-prompt binding."""

    rows, manifest = _read_plain_artifact(
        path,
        expected_schema=BCT_TARGET_REQUEST_SCHEMA,
        expected_schema_version=BCT_TARGET_REQUEST_SCHEMA_VERSION,
        expected_kind=BCT_TARGET_REQUEST_ARTIFACT_KIND,
        expected_role=BCT_TARGET_REQUEST_ROLE,
    )
    validated = _validate_request_rows(rows)
    provenance = manifest["provenance"]
    parents = provenance["parent_artifacts"]
    if len(parents) != 1 or provenance.get("training_view_identity") != parents[0]:
        raise BCTTargetError("target-request training-view identity differs from its sole parent artifact")
    parent = parents[0]
    if parent.get("role") != TRAINING_ROLE or parent.get("artifact_schema") != TRAINING_VIEW_SCHEMA:
        raise BCTTargetError("target-request parent is not a canonical role=training view")
    view_content_hashes = {row["training_view_content_sha256"] for row in validated}
    view_manifest_hashes = {row["training_view_manifest_sha256"] for row in validated}
    if view_content_hashes != {parent.get("content_sha256")}:
        raise BCTTargetError("target-request rows disagree with parent training-view content identity")
    if view_manifest_hashes != {parent.get(MANIFEST_SHA256_FIELD)}:
        raise BCTTargetError("target-request rows disagree with parent training-view manifest identity")
    if provenance.get("training_view_manifest_sha256") != parent.get(MANIFEST_SHA256_FIELD):
        raise BCTTargetError("target-request provenance has a stale training-view manifest digest")
    return validated, manifest


def import_bct_target_results(
    request_path: str | Path,
    result_rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    generator_identity: Mapping[str, Any],
    decoding_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly import offline responses and publish targets keyed by ``pair_id``.

    ``generator_identity`` must already use the canonical provenance vocabulary
    (``generator_id``, ``provider``, ``model``, ``model_revision``, and
    ``model_immutable_date``).  Use :func:`make_external_generator_identity`
    for an explicit legacy ``revision/date`` mapping.
    """

    try:
        requests, request_manifest = read_bct_target_requests(request_path)
    except (ArtifactManifestError, BCTTargetError) as exc:
        raise BCTTargetError(str(exc)) from exc
    generator = _generator_identity(generator_identity)
    generator_sha256 = generator["identity_sha256"]
    decoding = _mapping(decoding_parameters, field="decoding_parameters")
    decoding_sha256 = sha256_json(decoding)
    results = _validate_offline_results(result_rows)
    request_by_id = {row["pair_id"]: row for row in requests}
    result_by_id = {row["pair_id"]: row for row in results}
    _exact_ids(request_by_id, result_by_id, label="offline BCT target results")

    request_content_sha256 = _sha256(
        request_manifest.get("content_sha256"), field="target request artifact content_sha256"
    )
    targets: list[dict[str, Any]] = []
    for request in requests:
        pair_id = request["pair_id"]
        result = result_by_id[pair_id]
        for supplied_field, request_field in (
            ("clean_prompt_sha256", "clean_prompt_sha256"),
            ("reference_messages_sha256", "reference_messages_sha256"),
            ("request_record_sha256", "request_record_sha256"),
        ):
            supplied = result.get(supplied_field)
            if supplied is not None and supplied != request[request_field]:
                raise BCTTargetError(f"{supplied_field} mismatch for pair_id {pair_id!r}")
        response = result["response"]
        response_sha256 = sha256_text(response)
        base = {
            "pair_id": pair_id,
            "source_id": pair_id,
            "example_id": request["example_id"],
            "domain": request["domain"],
            "source": request["source"],
            "clean_prompt": request["clean_prompt"],
            "clean_prompt_sha256": request["clean_prompt_sha256"],
            "reference_messages": request["reference_messages"],
            "reference_messages_sha256": request["reference_messages_sha256"],
            "response": response,
            "response_sha256": response_sha256,
            "generator_identity": generator,
            "generator_identity_sha256": generator_sha256,
            "decoding_parameters": decoding,
            "decoding_parameters_sha256": decoding_sha256,
            "pair_record_sha256": request["pair_record_sha256"],
            "request_record_sha256": request["request_record_sha256"],
            "request_artifact_content_sha256": request_content_sha256,
            "training_view_content_sha256": request["training_view_content_sha256"],
            "metadata": result["metadata"],
        }
        targets.append({**base, "target_record_sha256": sha256_json(base)})
    targets = _validate_target_rows(targets, generator=generator, decoding=decoding)
    request_identity = _artifact_identity(request_path, request_manifest)
    request_provenance = request_manifest["provenance"]
    return _write_plain_artifact(
        output_path,
        targets,
        artifact_schema=BCT_TARGET_SCHEMA,
        schema_version=BCT_TARGET_SCHEMA_VERSION,
        artifact_kind=BCT_TARGET_ARTIFACT_KIND,
        role=BCT_TARGET_ROLE,
        producer=producer_identity(
            "irpan-offline-bct-target-importer",
            __file__,
            Path(__file__).with_name("training_views.py"),
            Path(__file__).with_name("provenance.py"),
        ),
        config={
            "generator_identity": generator,
            "generator_identity_sha256": generator_sha256,
            "decoding_parameters": decoding,
            "decoding_parameters_sha256": decoding_sha256,
            "model_calls_performed": 0,
        },
        parent_artifacts=[request_identity],
        provenance={
            "target_request_identity": request_identity,
            "target_request_manifest_sha256": _sha256(
                request_manifest.get(MANIFEST_SHA256_FIELD), field=f"target request {MANIFEST_SHA256_FIELD}"
            ),
            "training_view_identity": request_provenance.get("training_view_identity"),
            "training_view_content_sha256": requests[0]["training_view_content_sha256"],
            "generator_identity": generator,
            "generator_identity_sha256": generator_sha256,
            "decoding_parameters": decoding,
            "decoding_parameters_sha256": decoding_sha256,
            "offline_results_sha256": sha256_json(results),
            "external_execution_required": False,
            "model_calls_performed": 0,
        },
    )


def materialize_bct_targets(
    request_path: str | Path,
    result_rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    generator_identity: Mapping[str, Any],
    decoding_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Alias for :func:`import_bct_target_results`."""

    return import_bct_target_results(
        request_path,
        result_rows,
        output_path,
        generator_identity=generator_identity,
        decoding_parameters=decoding_parameters,
    )


def read_bct_targets(
    path: str | Path,
    *,
    expected_generator_identity: Mapping[str, Any] | None = None,
    expected_decoding_parameters: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify target rows, provenance hashes, and optional pinned identities."""

    rows, manifest = _read_plain_artifact(
        path,
        expected_schema=BCT_TARGET_SCHEMA,
        expected_schema_version=BCT_TARGET_SCHEMA_VERSION,
        expected_kind=BCT_TARGET_ARTIFACT_KIND,
        expected_role=BCT_TARGET_ROLE,
    )
    provenance = manifest["provenance"]
    generator = _generator_identity(provenance.get("generator_identity"))
    if provenance.get("generator_identity_sha256") != generator["identity_sha256"]:
        raise BCTTargetError("target manifest generator_identity_sha256 mismatch")
    decoding = _mapping(provenance.get("decoding_parameters"), field="manifest decoding_parameters")
    if provenance.get("decoding_parameters_sha256") != sha256_json(decoding):
        raise BCTTargetError("target manifest decoding_parameters_sha256 mismatch")
    if expected_generator_identity is not None and generator != _generator_identity(expected_generator_identity):
        raise BCTTargetError("target generator identity differs from the expected identity")
    if expected_decoding_parameters is not None and decoding != _mapping(
        expected_decoding_parameters, field="expected_decoding_parameters"
    ):
        raise BCTTargetError("target decoding parameters differ from the expected parameters")
    validated = _validate_target_rows(rows, generator=generator, decoding=decoding)
    config = provenance["config"]
    if config.get("generator_identity") != generator or config.get("generator_identity_sha256") != generator[
        "identity_sha256"
    ]:
        raise BCTTargetError("target manifest config and provenance generator identities differ")
    if config.get("decoding_parameters") != decoding or config.get("decoding_parameters_sha256") != sha256_json(
        decoding
    ):
        raise BCTTargetError("target manifest config and provenance decoding parameters differ")
    parents = provenance["parent_artifacts"]
    if len(parents) != 1 or provenance.get("target_request_identity") != parents[0]:
        raise BCTTargetError("target request identity differs from the target artifact's sole parent")
    request_parent = parents[0]
    if (
        request_parent.get("role") != TRAINING_ROLE
        or request_parent.get("artifact_schema") != BCT_TARGET_REQUEST_SCHEMA
    ):
        raise BCTTargetError("target artifact parent is not a canonical training target-request artifact")
    if {row["request_artifact_content_sha256"] for row in validated} != {
        request_parent.get("content_sha256")
    }:
        raise BCTTargetError("target rows disagree with target-request artifact lineage")
    if provenance.get("target_request_manifest_sha256") != request_parent.get(MANIFEST_SHA256_FIELD):
        raise BCTTargetError("target provenance has a stale target-request manifest digest")
    view_hash = _sha256(
        provenance.get("training_view_content_sha256"), field="manifest training_view_content_sha256"
    )
    if {row["training_view_content_sha256"] for row in validated} != {view_hash}:
        raise BCTTargetError("target rows disagree with manifest training-view lineage")
    view_identity = provenance.get("training_view_identity")
    if not isinstance(view_identity, Mapping) or view_identity.get("content_sha256") != view_hash:
        raise BCTTargetError("target manifest training-view identity does not match its lineage hash")
    return validated, manifest


def materialize_bct_training_data(
    training_view_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Join verified targets onto variant prompts and publish trainer JSONL."""

    try:
        pairs, view_manifest = read_training_view(training_view_path)
        targets, target_manifest = read_bct_targets(target_path)
    except (ArtifactManifestError, TrainingViewError, BCTTargetError) as exc:
        raise BCTTargetError(str(exc)) from exc
    view_content_sha256 = _sha256(view_manifest.get("content_sha256"), field="training view content_sha256")
    target_provenance = target_manifest["provenance"]
    if target_provenance.get("training_view_content_sha256") != view_content_sha256:
        raise BCTTargetError("BCT target artifact was generated from a different training view")
    pair_by_id = {row["pair_id"]: row for row in pairs}
    target_by_id = {row["pair_id"]: row for row in targets}
    _exact_ids(pair_by_id, target_by_id, label="BCT target artifact")
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        target = target_by_id[pair_id]
        _validate_pair_target_join(pair, target, view_content_sha256=view_content_sha256)
        assistant = {"role": "assistant", "content": target["response"]}
        messages = [*pair["variant_messages"], assistant]
        metadata = {
            "training_view_content_sha256": view_content_sha256,
            "target_artifact_content_sha256": _sha256(
                target_manifest.get("content_sha256"), field="target artifact content_sha256"
            ),
            "target_request_artifact_content_sha256": target["request_artifact_content_sha256"],
            "pair_record_sha256": target["pair_record_sha256"],
            "request_record_sha256": target["request_record_sha256"],
            "target_record_sha256": target["target_record_sha256"],
            "generator_identity": target["generator_identity"],
            "decoding_parameters": target["decoding_parameters"],
            "pair_metadata": pair["metadata"],
        }
        base = {
            "pair_id": pair_id,
            "source_id": pair_id,
            "example_id": pair["example_id"],
            "domain": pair["domain"],
            "source": pair["source"],
            "messages": messages,
            "clean_prompt_sha256": target["clean_prompt_sha256"],
            "reference_messages_sha256": target["reference_messages_sha256"],
            "variant_messages_sha256": sha256_json(pair["variant_messages"]),
            "response_sha256": target["response_sha256"],
            "generator_identity_sha256": target["generator_identity_sha256"],
            "decoding_parameters_sha256": target["decoding_parameters_sha256"],
            "metadata": _mapping(metadata, field="BCT training metadata"),
        }
        rows.append({**base, "training_record_sha256": sha256_json(base)})
    rows = _validate_bct_training_rows(sorted(rows, key=lambda row: row["pair_id"]))
    view_identity = _artifact_identity(training_view_path, view_manifest)
    target_identity = _artifact_identity(target_path, target_manifest)
    return _write_plain_artifact(
        output_path,
        rows,
        artifact_schema=BCT_TRAINING_SCHEMA,
        schema_version=BCT_TRAINING_SCHEMA_VERSION,
        artifact_kind=BCT_TRAINING_ARTIFACT_KIND,
        role=TRAINING_ROLE,
        producer=producer_identity(
            "irpan-bct-training-exporter",
            __file__,
            Path(__file__).with_name("training_views.py"),
        ),
        config={
            "message_source": "variant_messages_plus_verified_target",
            "generator_identity_sha256": target_provenance["generator_identity_sha256"],
            "decoding_parameters_sha256": target_provenance["decoding_parameters_sha256"],
            "model_calls_performed": 0,
        },
        parent_artifacts=[view_identity, target_identity],
        provenance={
            "training_view_identity": view_identity,
            "target_artifact_identity": target_identity,
            "target_request_identity": target_provenance.get("target_request_identity"),
            "generator_identity": target_provenance["generator_identity"],
            "generator_identity_sha256": target_provenance["generator_identity_sha256"],
            "decoding_parameters": target_provenance["decoding_parameters"],
            "decoding_parameters_sha256": target_provenance["decoding_parameters_sha256"],
            "model_calls_performed": 0,
        },
    )


def export_bct_training_jsonl(
    training_view_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Alias for :func:`materialize_bct_training_data`."""

    return materialize_bct_training_data(training_view_path, target_path, output_path)


def read_bct_training_data(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify exported BCT JSONL and its ``role=training`` sidecar."""

    rows, manifest = _read_plain_artifact(
        path,
        expected_schema=BCT_TRAINING_SCHEMA,
        expected_schema_version=BCT_TRAINING_SCHEMA_VERSION,
        expected_kind=BCT_TRAINING_ARTIFACT_KIND,
        expected_role=TRAINING_ROLE,
    )
    validated = _validate_bct_training_rows(rows)
    provenance = manifest["provenance"]
    parents = provenance["parent_artifacts"]
    if len(parents) != 2:
        raise BCTTargetError("BCT training artifact requires exactly the training-view and target parents")
    parent_by_schema = {parent.get("artifact_schema"): parent for parent in parents}
    if set(parent_by_schema) != {TRAINING_VIEW_SCHEMA, BCT_TARGET_SCHEMA}:
        raise BCTTargetError("BCT training artifact parents have unexpected schemas")
    view_parent = parent_by_schema[TRAINING_VIEW_SCHEMA]
    target_parent = parent_by_schema[BCT_TARGET_SCHEMA]
    if provenance.get("training_view_identity") != view_parent or provenance.get("target_artifact_identity") != target_parent:
        raise BCTTargetError("BCT training named parent identities differ from parent_artifacts")
    if view_parent.get("role") != TRAINING_ROLE or target_parent.get("role") != TRAINING_ROLE:
        raise BCTTargetError("BCT training parents must both retain role=training")
    if {row["metadata"]["training_view_content_sha256"] for row in validated} != {
        view_parent.get("content_sha256")
    }:
        raise BCTTargetError("BCT training rows disagree with training-view parent identity")
    if {row["metadata"]["target_artifact_content_sha256"] for row in validated} != {
        target_parent.get("content_sha256")
    }:
        raise BCTTargetError("BCT training rows disagree with target parent identity")
    generator = _generator_identity(provenance.get("generator_identity"))
    decoding = _mapping(provenance.get("decoding_parameters"), field="manifest decoding_parameters")
    if provenance.get("generator_identity_sha256") != generator["identity_sha256"]:
        raise BCTTargetError("BCT training manifest generator identity hash mismatch")
    if provenance.get("decoding_parameters_sha256") != sha256_json(decoding):
        raise BCTTargetError("BCT training manifest decoding parameters hash mismatch")
    if {row["generator_identity_sha256"] for row in validated} != {generator["identity_sha256"]}:
        raise BCTTargetError("BCT training rows disagree with manifest generator identity")
    if {row["decoding_parameters_sha256"] for row in validated} != {sha256_json(decoding)}:
        raise BCTTargetError("BCT training rows disagree with manifest decoding parameters")
    return validated, manifest


def _validate_request_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise BCTTargetError("BCT target request artifact is empty")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        row = _exact_mapping(raw, keys=_REQUEST_KEYS, field=f"target request row {index}")
        pair_id = _text(row["pair_id"], field=f"request row {index}.pair_id")
        if pair_id in seen:
            raise BCTTargetError(f"duplicate BCT target request pair_id {pair_id!r}")
        seen.add(pair_id)
        if row["source_id"] != pair_id:
            raise BCTTargetError(f"request row {index} source_id must equal pair_id")
        reference = _prompt_messages(row["reference_messages"], field=f"request row {index}.reference_messages")
        clean_prompt = _text(row["clean_prompt"], field=f"request row {index}.clean_prompt")
        if clean_prompt != _clean_prompt(reference):
            raise BCTTargetError(f"request row {index} clean_prompt differs from the final reference user prompt")
        normalized = {
            "pair_id": pair_id,
            "source_id": pair_id,
            "example_id": _text(row["example_id"], field=f"request row {index}.example_id"),
            "domain": _text(row["domain"], field=f"request row {index}.domain"),
            "source": _text(row["source"], field=f"request row {index}.source"),
            "clean_prompt": clean_prompt,
            "clean_prompt_sha256": _sha256(row["clean_prompt_sha256"], field="clean_prompt_sha256"),
            "reference_messages": reference,
            "reference_messages_sha256": _sha256(
                row["reference_messages_sha256"], field="reference_messages_sha256"
            ),
            "pair_record_sha256": _sha256(row["pair_record_sha256"], field="pair_record_sha256"),
            "training_view_content_sha256": _sha256(
                row["training_view_content_sha256"], field="training_view_content_sha256"
            ),
            "training_view_manifest_sha256": _sha256(
                row["training_view_manifest_sha256"], field="training_view_manifest_sha256"
            ),
        }
        if normalized["clean_prompt_sha256"] != sha256_text(clean_prompt):
            raise BCTTargetError(f"request row {index} clean_prompt_sha256 mismatch")
        if normalized["reference_messages_sha256"] != sha256_json(reference):
            raise BCTTargetError(f"request row {index} reference_messages_sha256 mismatch")
        request_hash = _sha256(row["request_record_sha256"], field="request_record_sha256")
        if request_hash != sha256_json(normalized):
            raise BCTTargetError(f"request row {index} request_record_sha256 mismatch")
        validated.append({**normalized, "request_record_sha256": request_hash})
    ordered = sorted(validated, key=lambda row: row["pair_id"])
    if validated != ordered:
        raise BCTTargetError("BCT target request rows are not in canonical pair_id order")
    return validated


def _validate_offline_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise BCTTargetError("offline BCT target results must be a non-empty sequence")
    required = {"pair_id", "response"}
    optional = {
        "response_sha256",
        "clean_prompt_sha256",
        "reference_messages_sha256",
        "request_record_sha256",
        "metadata",
    }
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise BCTTargetError(f"offline target result {index} must be an object")
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required - optional)
        if missing or extra:
            raise BCTTargetError(f"offline target result {index} keys mismatch: missing={missing}, extra={extra}")
        pair_id = _text(raw["pair_id"], field=f"offline target result {index}.pair_id")
        if pair_id in seen:
            raise BCTTargetError(f"duplicate offline BCT target result pair_id {pair_id!r}")
        seen.add(pair_id)
        response = _text(raw["response"], field=f"offline target result {pair_id!r}.response")
        response_hash = sha256_text(response)
        if "response_sha256" in raw and _sha256(raw["response_sha256"], field="response_sha256") != response_hash:
            raise BCTTargetError(f"response_sha256 mismatch for pair_id {pair_id!r}")
        row: dict[str, Any] = {
            "pair_id": pair_id,
            "response": response,
            "response_sha256": response_hash,
            "metadata": _mapping(raw.get("metadata", {}), field=f"result {pair_id!r} metadata"),
        }
        for field in ("clean_prompt_sha256", "reference_messages_sha256", "request_record_sha256"):
            if field in raw:
                row[field] = _sha256(raw[field], field=field)
        validated.append(row)
    return sorted(validated, key=lambda row: row["pair_id"])


def _validate_target_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    generator: Mapping[str, Any],
    decoding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not rows:
        raise BCTTargetError("BCT target artifact is empty")
    expected_generator = _generator_identity(generator)
    expected_decoding = _mapping(decoding, field="decoding_parameters")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        row = _exact_mapping(raw, keys=_TARGET_KEYS, field=f"target row {index}")
        pair_id = _text(row["pair_id"], field=f"target row {index}.pair_id")
        if pair_id in seen:
            raise BCTTargetError(f"duplicate BCT target pair_id {pair_id!r}")
        seen.add(pair_id)
        if row["source_id"] != pair_id:
            raise BCTTargetError(f"target row {index} source_id must equal pair_id")
        reference = _prompt_messages(row["reference_messages"], field=f"target row {index}.reference_messages")
        clean_prompt = _text(row["clean_prompt"], field=f"target row {index}.clean_prompt")
        response = _text(row["response"], field=f"target row {index}.response")
        actual_generator = _generator_identity(row["generator_identity"])
        actual_decoding = _mapping(row["decoding_parameters"], field=f"target row {index}.decoding_parameters")
        if actual_generator != expected_generator:
            raise BCTTargetError(f"target row {index} generator identity differs from manifest provenance")
        if actual_decoding != expected_decoding:
            raise BCTTargetError(f"target row {index} decoding parameters differ from manifest provenance")
        normalized = {
            "pair_id": pair_id,
            "source_id": pair_id,
            "example_id": _text(row["example_id"], field=f"target row {index}.example_id"),
            "domain": _text(row["domain"], field=f"target row {index}.domain"),
            "source": _text(row["source"], field=f"target row {index}.source"),
            "clean_prompt": clean_prompt,
            "clean_prompt_sha256": _sha256(row["clean_prompt_sha256"], field="clean_prompt_sha256"),
            "reference_messages": reference,
            "reference_messages_sha256": _sha256(
                row["reference_messages_sha256"], field="reference_messages_sha256"
            ),
            "response": response,
            "response_sha256": _sha256(row["response_sha256"], field="response_sha256"),
            "generator_identity": actual_generator,
            "generator_identity_sha256": _sha256(
                row["generator_identity_sha256"], field="generator_identity_sha256"
            ),
            "decoding_parameters": actual_decoding,
            "decoding_parameters_sha256": _sha256(
                row["decoding_parameters_sha256"], field="decoding_parameters_sha256"
            ),
            "pair_record_sha256": _sha256(row["pair_record_sha256"], field="pair_record_sha256"),
            "request_record_sha256": _sha256(row["request_record_sha256"], field="request_record_sha256"),
            "request_artifact_content_sha256": _sha256(
                row["request_artifact_content_sha256"], field="request_artifact_content_sha256"
            ),
            "training_view_content_sha256": _sha256(
                row["training_view_content_sha256"], field="training_view_content_sha256"
            ),
            "metadata": _mapping(row["metadata"], field=f"target row {index}.metadata"),
        }
        if clean_prompt != _clean_prompt(reference):
            raise BCTTargetError(f"target row {index} clean_prompt differs from reference messages")
        expected_hashes = {
            "clean_prompt_sha256": sha256_text(clean_prompt),
            "reference_messages_sha256": sha256_json(reference),
            "response_sha256": sha256_text(response),
            "generator_identity_sha256": actual_generator["identity_sha256"],
            "decoding_parameters_sha256": sha256_json(actual_decoding),
        }
        for field, expected in expected_hashes.items():
            if normalized[field] != expected:
                raise BCTTargetError(f"target row {index} {field} mismatch")
        target_hash = _sha256(row["target_record_sha256"], field="target_record_sha256")
        if target_hash != sha256_json(normalized):
            raise BCTTargetError(f"target row {index} target_record_sha256 mismatch")
        validated.append({**normalized, "target_record_sha256": target_hash})
    ordered = sorted(validated, key=lambda row: row["pair_id"])
    if validated != ordered:
        raise BCTTargetError("BCT target rows are not in canonical pair_id order")
    return validated


def _validate_bct_training_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise BCTTargetError("BCT training artifact is empty")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        row = _exact_mapping(raw, keys=_TRAINING_KEYS, field=f"BCT training row {index}")
        pair_id = _text(row["pair_id"], field=f"BCT training row {index}.pair_id")
        if pair_id in seen:
            raise BCTTargetError(f"duplicate BCT training pair_id {pair_id!r}")
        seen.add(pair_id)
        if row["source_id"] != pair_id:
            raise BCTTargetError(f"BCT training row {index} source_id must equal pair_id")
        messages = _training_messages(row["messages"], field=f"BCT training row {index}.messages")
        metadata = _mapping(row["metadata"], field=f"BCT training row {index}.metadata")
        normalized = {
            "pair_id": pair_id,
            "source_id": pair_id,
            "example_id": _text(row["example_id"], field=f"BCT training row {index}.example_id"),
            "domain": _text(row["domain"], field=f"BCT training row {index}.domain"),
            "source": _text(row["source"], field=f"BCT training row {index}.source"),
            "messages": messages,
            "clean_prompt_sha256": _sha256(row["clean_prompt_sha256"], field="clean_prompt_sha256"),
            "reference_messages_sha256": _sha256(
                row["reference_messages_sha256"], field="reference_messages_sha256"
            ),
            "variant_messages_sha256": _sha256(
                row["variant_messages_sha256"], field="variant_messages_sha256"
            ),
            "response_sha256": _sha256(row["response_sha256"], field="response_sha256"),
            "generator_identity_sha256": _sha256(
                row["generator_identity_sha256"], field="generator_identity_sha256"
            ),
            "decoding_parameters_sha256": _sha256(
                row["decoding_parameters_sha256"], field="decoding_parameters_sha256"
            ),
            "metadata": metadata,
        }
        if normalized["variant_messages_sha256"] != sha256_json(messages[:-1]):
            raise BCTTargetError(f"BCT training row {index} variant_messages_sha256 mismatch")
        if normalized["response_sha256"] != sha256_text(messages[-1]["content"]):
            raise BCTTargetError(f"BCT training row {index} response_sha256 mismatch")
        generator = _generator_identity(metadata.get("generator_identity"))
        decoding = _mapping(metadata.get("decoding_parameters"), field="metadata.decoding_parameters")
        if normalized["generator_identity_sha256"] != generator["identity_sha256"]:
            raise BCTTargetError(f"BCT training row {index} generator provenance mismatch")
        if normalized["decoding_parameters_sha256"] != sha256_json(decoding):
            raise BCTTargetError(f"BCT training row {index} decoding provenance mismatch")
        record_hash = _sha256(row["training_record_sha256"], field="training_record_sha256")
        if record_hash != sha256_json(normalized):
            raise BCTTargetError(f"BCT training row {index} training_record_sha256 mismatch")
        validated.append({**normalized, "training_record_sha256": record_hash})
    ordered = sorted(validated, key=lambda row: row["pair_id"])
    if validated != ordered:
        raise BCTTargetError("BCT training rows are not in canonical pair_id order")
    return validated


def _validate_pair_target_join(
    pair: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    view_content_sha256: str,
) -> None:
    pair_id = pair["pair_id"]
    for field in ("pair_id", "example_id", "domain", "source"):
        if pair[field] != target[field]:
            raise BCTTargetError(f"BCT target {field} mismatch for pair_id {pair_id!r}")
    reference = pair["reference_messages"]
    clean_prompt = _clean_prompt(reference)
    expected = {
        "clean_prompt": clean_prompt,
        "clean_prompt_sha256": sha256_text(clean_prompt),
        "reference_messages": reference,
        "reference_messages_sha256": sha256_json(reference),
        "pair_record_sha256": sha256_json(pair),
        "training_view_content_sha256": view_content_sha256,
    }
    for field, value in expected.items():
        if target[field] != value:
            raise BCTTargetError(f"BCT target {field} mismatch for pair_id {pair_id!r}")


def _generator_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BCTTargetError("generator_identity must be an object using canonical provenance keys")
    try:
        return validate_generator_identity(value)
    except ProvenanceError as exc:
        raise BCTTargetError(str(exc)) from exc


def _prompt_messages(value: Any, *, field: str) -> list[dict[str, str]]:
    messages = _base_messages(value, field=field)
    if messages[-1]["role"] != "user":
        raise BCTTargetError(f"{field} must end with a user message")
    return messages


def _training_messages(value: Any, *, field: str) -> list[dict[str, str]]:
    messages = _base_messages(value, field=field)
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise BCTTargetError(f"{field} must contain a prompt followed by an assistant response")
    if messages[-2]["role"] != "user":
        raise BCTTargetError(f"{field} assistant response must directly follow a user prompt")
    return messages


def _base_messages(value: Any, *, field: str) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise BCTTargetError(f"{field} must be a non-empty message list")
    messages: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {"role", "content"}:
            raise BCTTargetError(f"{field}[{index}] must have exactly role/content fields")
        messages.append(
            {
                "role": _text(raw.get("role"), field=f"{field}[{index}].role"),
                "content": _text(raw.get("content"), field=f"{field}[{index}].content"),
            }
        )
    return messages


def _clean_prompt(reference_messages: Sequence[Mapping[str, str]]) -> str:
    return reference_messages[-1]["content"]


def _exact_mapping(value: Any, *, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BCTTargetError(f"{field} must be an object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise BCTTargetError(f"{field} keys mismatch: missing={missing}, extra={extra}")
    return dict(value)


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BCTTargetError(f"{field} must be an object")
    try:
        normalized = normalize_json(value)
        sha256_json(normalized)
    except (TypeError, ValueError) as exc:
        raise BCTTargetError(f"{field} must contain finite canonical JSON values: {exc}") from exc
    if not isinstance(normalized, dict):
        raise BCTTargetError(f"{field} must normalize to an object")
    return normalized


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not normalize_text(value):
        raise BCTTargetError(f"{field} must be a non-empty string")
    return normalize_text(value)


def _sha256(value: Any, *, field: str) -> str:
    try:
        return require_sha256(value, field=field)
    except ValueError as exc:
        raise BCTTargetError(str(exc)) from exc


def _exact_ids(expected: Mapping[str, Any], actual: Mapping[str, Any], *, label: str) -> None:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise BCTTargetError(f"{label} ID mismatch: missing={missing}, extra={extra}")


def _artifact_identity(path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise BCTTargetError("artifact manifest has no provenance object")
    return {
        "path": str(Path(path)),
        "artifact_schema": manifest.get("artifact_schema"),
        "schema_version": manifest.get("schema_version"),
        "artifact_kind": provenance.get("artifact_kind"),
        "role": provenance.get("role"),
        "row_count": manifest.get("row_count"),
        "content_sha256": _sha256(manifest.get("content_sha256"), field="artifact content_sha256"),
        MANIFEST_SHA256_FIELD: _sha256(
            manifest.get(MANIFEST_SHA256_FIELD), field=f"artifact {MANIFEST_SHA256_FIELD}"
        ),
    }


__all__ = [
    "BCT_TARGET_ARTIFACT_KIND",
    "BCT_TARGET_REQUEST_ARTIFACT_KIND",
    "BCT_TARGET_REQUEST_ROLE",
    "BCT_TARGET_REQUEST_SCHEMA",
    "BCT_TARGET_ROLE",
    "BCT_TARGET_SCHEMA",
    "BCT_TRAINING_ARTIFACT_KIND",
    "BCT_TRAINING_SCHEMA",
    "BCTTargetError",
    "export_bct_training_jsonl",
    "import_bct_target_results",
    "make_external_generator_identity",
    "make_fixture_generator_identity",
    "materialize_bct_target_requests",
    "materialize_bct_targets",
    "materialize_bct_training_data",
    "read_bct_target_requests",
    "read_bct_targets",
    "read_bct_training_data",
]
