from __future__ import annotations

import copy

import pytest

from ctm_data.adapters.irpan_2510_27062.provenance import (
    EXTERNAL_MODEL,
    FRESH_SELF_GENERATED,
    OLDER_MODEL_REVISION,
    PROVENANCE_SCHEMA_VERSION,
    STALE_EXTERNAL,
    ProvenanceError,
    build_fresh_target_provenance,
    build_generation_provenance,
    build_stale_target_provenance,
    make_generator_identity,
    make_response_manifest_entry,
    ordered_response_manifest_sha256,
    require_fresh_self_generated,
    require_stale_external,
    stable_example_manifest_sha256,
    validate_generation_provenance,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _current_generator() -> dict:
    return make_generator_identity(
        generator_id="self-generation-runner-v1",
        provider="fixture-provider",
        model="fixture/model",
        model_revision="revision-current-002",
        model_immutable_date="2025-06-01",
    )


def _responses() -> list[dict[str, str]]:
    return [
        make_response_manifest_entry(example_id="example-b", response="response B"),
        make_response_manifest_entry(example_id="example-a", response="response A"),
    ]


def _fresh_provenance() -> dict:
    return build_fresh_target_provenance(
        input_artifact_sha256=SHA_A,
        input_manifest_sha256=SHA_B,
        example_ids=["example-b", "example-a"],
        generator_identity=_current_generator(),
        prompt_template_sha256=SHA_C,
        decoding_parameters={"temperature": 0.0, "max_tokens": 256, "stop": []},
        ordered_response_manifest=_responses(),
        parent_artifact_sha256=SHA_D,
        generated_at_utc="2025-07-01T14:30:00+03:00",
        metadata={"fixture": True},
    )


def test_fresh_provenance_records_every_required_identity_and_recomputable_hash() -> None:
    provenance = _fresh_provenance()
    assert provenance["provenance_schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert provenance["input_artifact_sha256"] == SHA_A
    assert provenance["input_manifest_sha256"] == SHA_B
    assert provenance["parent_artifact_sha256"] == SHA_D
    assert provenance["example_manifest"] == ["example-a", "example-b"]
    assert provenance["example_manifest_sha256"] == stable_example_manifest_sha256(
        ["example-b", "example-a"]
    )
    assert [entry["example_id"] for entry in provenance["ordered_response_manifest"]] == [
        "example-b",
        "example-a",
    ]
    assert provenance["ordered_response_manifest_sha256"] == ordered_response_manifest_sha256(_responses())
    assert len(provenance["decoding_parameters_sha256"]) == 64
    assert provenance["generated_at_utc"] == "2025-07-01T11:30:00Z"
    assert provenance["target_freshness"] == FRESH_SELF_GENERATED
    assert provenance["stale_reason"] is None
    assert provenance["generator_identity"] == provenance["reference_generator_identity"]
    assert require_fresh_self_generated(provenance) == provenance


def test_missing_provenance_fields_fail_closed() -> None:
    provenance = _fresh_provenance()
    for field in (
        "input_artifact_sha256",
        "input_manifest_sha256",
        "example_manifest_sha256",
        "generator_identity",
        "prompt_template_sha256",
        "decoding_parameters_sha256",
        "ordered_response_manifest_sha256",
        "parent_artifact_sha256",
        "generated_at_utc",
        "target_freshness",
    ):
        incomplete = copy.deepcopy(provenance)
        incomplete.pop(field)
        with pytest.raises(ProvenanceError, match="missing"):
            validate_generation_provenance(incomplete)


def test_internal_and_expected_hash_mismatches_are_detected() -> None:
    provenance = _fresh_provenance()
    tampered_decoding = copy.deepcopy(provenance)
    tampered_decoding["decoding_parameters"]["temperature"] = 0.7
    with pytest.raises(ProvenanceError, match="decoding parameters hash mismatch"):
        validate_generation_provenance(tampered_decoding)

    reordered = copy.deepcopy(provenance)
    reordered["ordered_response_manifest"].reverse()
    with pytest.raises(ProvenanceError, match="ordered response manifest hash mismatch"):
        validate_generation_provenance(reordered)

    with pytest.raises(ProvenanceError, match="expected identity mismatch.*input_artifact_sha256"):
        validate_generation_provenance(provenance, expected={"input_artifact_sha256": SHA_B})
    with pytest.raises(ProvenanceError, match="expected identity mismatch.*model"):
        validate_generation_provenance(
            provenance,
            expected={"generator_identity": {"model": "other/model"}},
        )


def test_generator_requires_immutable_revision_or_date_and_timezone() -> None:
    with pytest.raises(ProvenanceError, match="revision and/or"):
        make_generator_identity(generator_id="gen", provider="provider", model="model")
    with pytest.raises(ProvenanceError, match="mutable"):
        make_generator_identity(
            generator_id="gen",
            provider="provider",
            model="model",
            model_revision="latest",
        )
    date_only = make_generator_identity(
        generator_id="gen",
        provider="provider",
        model="model",
        model_immutable_date="2025-01-01",
    )
    assert date_only["model_revision"] is None

    kwargs = {
        "input_artifact_sha256": SHA_A,
        "input_manifest_sha256": SHA_B,
        "example_ids": ["example-a", "example-b"],
        "generator_identity": _current_generator(),
        "prompt_template_sha256": SHA_C,
        "decoding_parameters": {},
        "ordered_response_manifest": _responses(),
        "parent_artifact_sha256": SHA_D,
    }
    with pytest.raises(ProvenanceError, match="include a timezone"):
        build_fresh_target_provenance(**kwargs, generated_at_utc="2025-07-01T12:00:00")
    future_identity = make_generator_identity(
        generator_id="gen",
        provider="provider",
        model="model",
        model_immutable_date="2026-01-01",
    )
    with pytest.raises(ProvenanceError, match="cannot be later"):
        build_fresh_target_provenance(
            **{**kwargs, "generator_identity": future_identity},
            generated_at_utc="2025-07-01T12:00:00Z",
        )


def test_stale_external_requires_distinct_explicit_external_identity() -> None:
    stale_generator = make_generator_identity(
        generator_id="external-target-export-v1",
        provider="other-provider",
        model="other/model",
        model_revision="immutable-external-001",
        model_immutable_date="2025-01-01",
    )
    stale = build_stale_target_provenance(
        input_artifact_sha256=SHA_A,
        input_manifest_sha256=SHA_B,
        example_ids=["example-a", "example-b"],
        stale_generator_identity=stale_generator,
        current_generator_identity=_current_generator(),
        stale_reason=EXTERNAL_MODEL,
        prompt_template_sha256=SHA_C,
        decoding_parameters={"temperature": 0.2},
        ordered_response_manifest=_responses(),
        parent_artifact_sha256=SHA_D,
        generated_at_utc="2025-07-01T12:00:00Z",
    )
    assert stale["target_freshness"] == STALE_EXTERNAL
    assert stale["stale_reason"] == EXTERNAL_MODEL
    assert require_stale_external(stale) == stale
    with pytest.raises(ProvenanceError, match="expected identity mismatch.*target_freshness"):
        require_fresh_self_generated(stale)

    with pytest.raises(ProvenanceError, match="distinct old/external"):
        build_stale_target_provenance(
            input_artifact_sha256=SHA_A,
            input_manifest_sha256=SHA_B,
            example_ids=["example-a", "example-b"],
            stale_generator_identity=_current_generator(),
            current_generator_identity=_current_generator(),
            stale_reason=EXTERNAL_MODEL,
            prompt_template_sha256=SHA_C,
            decoding_parameters={},
            ordered_response_manifest=_responses(),
            parent_artifact_sha256=SHA_D,
            generated_at_utc="2025-07-01T12:00:00Z",
        )


def test_old_revision_stays_stale_and_cannot_be_silently_marked_fresh() -> None:
    old_generator = make_generator_identity(
        generator_id="self-generation-runner-v1",
        provider="fixture-provider",
        model="fixture/model",
        model_revision="revision-old-001",
        model_immutable_date="2025-01-01",
    )
    common = {
        "input_artifact_sha256": SHA_A,
        "input_manifest_sha256": SHA_B,
        "example_ids": ["example-a", "example-b"],
        "prompt_template_sha256": SHA_C,
        "decoding_parameters": {"temperature": 0.0},
        "ordered_response_manifest": _responses(),
        "parent_artifact_sha256": SHA_D,
        "generated_at_utc": "2025-07-01T12:00:00Z",
    }
    stale = build_stale_target_provenance(
        **common,
        stale_generator_identity=old_generator,
        current_generator_identity=_current_generator(),
        stale_reason=OLDER_MODEL_REVISION,
    )
    assert stale["target_freshness"] == STALE_EXTERNAL
    assert stale["stale_reason"] == OLDER_MODEL_REVISION

    with pytest.raises(ProvenanceError, match="fresh_self_generated requires"):
        build_generation_provenance(
            **common,
            generator_identity=old_generator,
            reference_generator_identity=_current_generator(),
            target_freshness=FRESH_SELF_GENERATED,
        )
    with pytest.raises(ProvenanceError, match="use older_model_revision"):
        build_stale_target_provenance(
            **common,
            stale_generator_identity=old_generator,
            current_generator_identity=_current_generator(),
            stale_reason=EXTERNAL_MODEL,
        )


def test_response_and_example_manifests_detect_identity_conflicts() -> None:
    with pytest.raises(ProvenanceError, match="duplicate example_id"):
        stable_example_manifest_sha256(["example-a", "example-a"])
    bad_responses = [
        make_response_manifest_entry(example_id="example-a", response="first"),
        make_response_manifest_entry(example_id="example-c", response="extra"),
    ]
    with pytest.raises(ProvenanceError, match="identity mismatch"):
        build_fresh_target_provenance(
            input_artifact_sha256=SHA_A,
            input_manifest_sha256=SHA_B,
            example_ids=["example-a", "example-b"],
            generator_identity=_current_generator(),
            prompt_template_sha256=SHA_C,
            decoding_parameters={},
            ordered_response_manifest=bad_responses,
            parent_artifact_sha256=SHA_D,
            generated_at_utc="2025-07-01T12:00:00Z",
        )
