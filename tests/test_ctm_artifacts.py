from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ctm.artifacts import (
    MANIFEST_SHA256_FIELD,
    ArtifactManifestError,
    artifact_manifest_path,
    parent_artifact_identity,
    producer_identity,
    read_verified_jsonl_artifact,
    verify_data_manifest_binding,
    verify_data_manifest_bindings,
    write_verified_jsonl_artifact,
)
from ctm.identity import canonical_json, sha256_json

ARTIFACT_SCHEMA = "ctm.test.rows"
SCHEMA_VERSION = 1


def _write(path: Path, rows: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    return write_verified_jsonl_artifact(
        path,
        rows,
        artifact_schema=ARTIFACT_SCHEMA,
        schema_version=SCHEMA_VERSION,
        provenance={
            "producer": producer_identity("test-ctm-artifacts", __file__),
            "role": "training",
            "config": {"seed": 42, "nested": {"mode": "fixture", "unused": True}},
        },
        **kwargs,
    )


def test_verified_jsonl_round_trip_and_identities(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        {"z": 1, "text": "café"},
        {"nested": {"value": 2}, "id": "second"},
    ]

    manifest = _write(path, rows, nonempty=True)

    expected_payload = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    assert path.read_bytes() == expected_payload
    unsigned_manifest = {key: value for key, value in manifest.items() if key != MANIFEST_SHA256_FIELD}
    assert manifest[MANIFEST_SHA256_FIELD] == sha256_json(unsigned_manifest)

    loaded, verified = read_verified_jsonl_artifact(
        path,
        expected_schema=ARTIFACT_SCHEMA,
        expected_schema_version=SCHEMA_VERSION,
        expected_provenance={"config": {"nested": {"mode": "fixture"}}},
    )
    assert loaded == rows
    assert verified == manifest
    assert parent_artifact_identity(path) == {
        "path": str(path),
        "artifact_schema": ARTIFACT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "row_count": 2,
        "content_sha256": manifest["content_sha256"],
        MANIFEST_SHA256_FIELD: manifest[MANIFEST_SHA256_FIELD],
    }

    producer = manifest["provenance"]["producer"]
    assert producer["name"] == "test-ctm-artifacts"
    assert producer["files"][0]["name"] == Path(__file__).name
    assert producer["code_sha256"] == sha256_json(producer["files"])


def test_payload_tampering_fails_verification(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _write(path, [{"id": 1}])
    path.write_text('{"id":2}\n', encoding="utf-8")

    with pytest.raises(ArtifactManifestError, match="artifact digest mismatch"):
        read_verified_jsonl_artifact(path)


def test_explicit_data_manifest_binding_verifies_self_hashed_artifact(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    manifest = _write(path, [{"id": 1}])

    assert verify_data_manifest_binding(path, artifact_manifest_path(path)) == manifest
    assert verify_data_manifest_bindings([path], [artifact_manifest_path(path)]) == [manifest]

    with pytest.raises(ArtifactManifestError, match="same count"):
        verify_data_manifest_bindings([path], [])


def test_bct_target_manifest_binding_checks_path_digest_and_row_count(tmp_path: Path) -> None:
    path = tmp_path / "bct.jsonl"
    path.write_text('{"id":1}\n', encoding="utf-8")
    sidecar = tmp_path / "bct.manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": "ctm_bct_targets",
        "row_count": 1,
        "outputs": {
            "main": {
                "path": str(path.resolve()),
                "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        },
    }
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_data_manifest_binding(path, sidecar) == manifest
    path.write_text('{"id":2}\n', encoding="utf-8")
    with pytest.raises(ArtifactManifestError, match="digest mismatch"):
        verify_data_manifest_binding(path, sidecar)


def test_manifest_self_hash_detects_field_tampering(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _write(path, [{"id": 1}])
    sidecar = artifact_manifest_path(path)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["provenance"]["config"]["seed"] = 7
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactManifestError, match="manifest integrity digest mismatch"):
        read_verified_jsonl_artifact(path)


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"config": {"seed": 7}}, "provenance.config.seed"),
        ({"config": {"missing": True}}, "provenance.config.missing"),
    ],
)
def test_expected_provenance_must_be_recursive_subset(
    tmp_path: Path,
    expected: dict[str, Any],
    message: str,
) -> None:
    path = tmp_path / "rows.jsonl"
    _write(path, [{"id": 1}])

    with pytest.raises(ArtifactManifestError, match=message):
        read_verified_jsonl_artifact(path, expected_provenance=expected)


def test_writer_refuses_to_overwrite_either_artifact_path(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _write(path, [{"id": 1}])

    with pytest.raises(FileExistsError, match="refusing to overwrite immutable artifact pair"):
        _write(path, [{"id": 2}])

    sidecar_only_target = tmp_path / "sidecar-only.jsonl"
    artifact_manifest_path(sidecar_only_target).write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite immutable artifact pair"):
        _write(sidecar_only_target, [{"id": 1}])


def test_row_validator_controls_written_and_loaded_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"

    def validate(row: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(row.get("id"), int):
            raise TypeError("id must be an integer")
        return {**row, "validated": True}

    _write(path, [{"id": 1}], row_validator=validate, nonempty=True)
    rows, _manifest = read_verified_jsonl_artifact(path, row_validator=validate)
    assert rows == [{"id": 1, "validated": True}]

    invalid = tmp_path / "invalid.jsonl"
    with pytest.raises(ArtifactManifestError, match="id must be an integer"):
        _write(invalid, [{"id": "one"}], row_validator=validate)
    assert not invalid.exists()
    assert not artifact_manifest_path(invalid).exists()


def test_nonempty_artifacts_reject_empty_input_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"

    with pytest.raises(ArtifactManifestError, match="at least one row"):
        _write(path, [], nonempty=True)
    assert not path.exists()
    assert not artifact_manifest_path(path).exists()
