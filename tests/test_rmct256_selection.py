"""Focused offline tests for the immutable RMCT-256 training selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.rmct_256 import selection


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def _recovered_row(index: int) -> dict:
    dataset = "logiqa" if index % 2 == 0 else "hellaswag"
    question = f"Question {index}?\n\nAnswer choices:\n(A) correct\n(B) wrong"
    argument = f"Stored wrong argument {index}."
    return {
        "question": question,
        "question_id": f"q-{index:04d}",
        "source_dataset": dataset,
        "prompt_style": "none",
        "unbiased_messages": [{"role": "user", "content": question + selection.NONE_ANSWER_FORMAT_TERMINAL}],
        "biased_messages": [
            {
                "role": "user",
                "content": selection.LEGACY_G4_WRONG_ARGUMENT_TEMPLATE.format(argument=argument, question=question)
                + selection.NONE_ANSWER_FORMAT_TERMINAL,
            }
        ],
        "bias_type": "wrong_argument",
        "ground_truth": "A",
        "biased_option": "B",
        "biasing_text": argument,
    }


def _write_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, stage2_overlap: bool = False) -> dict[str, Path]:
    """Make a small but complete immutable-source chain with an n=256 prefix."""

    rows = [_recovered_row(index) for index in range(selection.SELECTION_ROWS)]
    recovered_payload = _jsonl(rows)
    recovered_source = tmp_path / "recovered-none.jsonl"
    recovered_source.write_bytes(recovered_payload)
    recovered_hash = _sha256(recovered_payload)

    recovery_manifest = {
        "schema_version": selection.RECOVERY_SCHEMA_VERSION,
        "kind": selection.RECOVERED_MANIFEST_KIND,
        "transform": {
            "version": selection.RECOVERY_TRANSFORM_VERSION,
            "source_prompt_style": "encourage_cot",
            "target_prompt_style": "none",
            "bias_type": "wrong_argument",
        },
        "source": {"content_sha256": "a" * 64, "row_count": len(rows)},
        "output": {"content_sha256": recovered_hash, "row_count": len(rows)},
        "selection": {"row_count": len(rows), "source_dataset_counts": dict(selection.SELECTION_COUNTS)},
    }
    recovered_manifest = tmp_path / "recovered-none.manifest.json"
    recovered_manifest.write_bytes(_json(recovery_manifest))

    canonical_rows = [
        selection._canonicalize_wrong_argument_row(row, path=recovered_source, line_number=index)
        for index, row in enumerate(rows, start=1)
    ]
    canonical_payload = _jsonl(canonical_rows)
    canonical_source = tmp_path / "canonical-n256.jsonl"
    canonical_source.write_bytes(canonical_payload)
    canonical_hash = _sha256(canonical_payload)
    canonical_manifest = {
        "artifact_schema": selection.CANONICAL_PAIR_SCHEMA,
        "schema_version": 1,
        "row_count": len(canonical_rows),
        "content_sha256": canonical_hash,
        "provenance": {
            "source": {"content_sha256": recovered_hash, "row_count": len(rows)},
            "selection": {"limit": len(canonical_rows)},
            "transform": {
                "name": selection.CANONICAL_PAIR_TRANSFORM,
                "argument_field": "biasing_text",
                "reference_field": "unbiased_messages",
                "variant_field": "biased_messages",
                "invariant": "variant last user content ends with exact reference last user content",
            },
        },
    }
    canonical_manifest_path = tmp_path / "canonical-n256.manifest.json"
    canonical_manifest_path.write_bytes(_json(canonical_manifest))

    original64_ids = [row["question_id"] for row in rows[: selection.ORIGINAL64_ROWS]]
    original64_manifest = {
        "schema_version": 1,
        "kind": selection.ORIGINAL64_REFERENCE_KIND,
        "source": {"content_sha256": recovered_hash},
        "rmct_first64": {
            "row_count": selection.ORIGINAL64_ROWS,
            "counts_by_dataset": dict(selection.ORIGINAL64_COUNTS),
            "question_ids": original64_ids,
            "question_ids_sha256": selection._ids_sha256(original64_ids),
            "source_rows_1_based_inclusive": [1, selection.ORIGINAL64_ROWS],
        },
    }
    original64_path = tmp_path / "original64.manifest.json"
    original64_path.write_bytes(_json(original64_manifest))

    stage2_ids = ["stage2-0", "stage2-1"]
    if stage2_overlap:
        stage2_ids[0] = rows[0]["question_id"]
    stage2_counts = {"logiqa": 1, "hellaswag": 1}
    artifact = {
        "row_count": 2,
        "counts_by_dataset": stage2_counts,
        "content_sha256": _sha256(b"fixture-stage2-unbiased"),
        "question_ids": stage2_ids,
        "question_ids_sha256": selection._ids_sha256(stage2_ids),
    }
    stage2_manifest = {
        "schema": selection.STAGE2_MANIFEST_SCHEMA,
        "schema_version": 1,
        "kind": selection.STAGE2_MANIFEST_KIND,
        "populations": {
            "in_domain": {
                "row_count": 2,
                "counts_by_dataset": stage2_counts,
                "artifacts": {name: dict(artifact) for name in selection.STAGE2_IN_DOMAIN_ARTIFACTS},
            }
        },
    }
    stage2_path = tmp_path / "stage2.manifest.json"
    stage2_path.write_bytes(_json(stage2_manifest))

    monkeypatch.setattr(selection, "CANONICAL_SOURCE_ROWS", len(canonical_rows))
    monkeypatch.setattr(selection, "CANONICAL_SOURCE_COUNTS", dict(selection.SELECTION_COUNTS))
    monkeypatch.setattr(selection, "CANONICAL_SOURCE_SHA256", canonical_hash)
    monkeypatch.setattr(selection, "CANONICAL_SOURCE_MANIFEST_SHA256", _sha256(canonical_manifest_path.read_bytes()))
    monkeypatch.setattr(selection, "RECOVERED_NONE_ROWS", len(rows))
    monkeypatch.setattr(selection, "RECOVERED_NONE_COUNTS", dict(selection.SELECTION_COUNTS))
    monkeypatch.setattr(selection, "RECOVERED_NONE_SOURCE_SHA256", recovered_hash)
    monkeypatch.setattr(selection, "RECOVERED_NONE_MANIFEST_SHA256", _sha256(recovered_manifest.read_bytes()))
    monkeypatch.setattr(selection, "LEGACY_COT_SOURCE_SHA256", "a" * 64)
    monkeypatch.setattr(
        selection, "ORIGINAL64_REFERENCE_MANIFEST_SHA256", _sha256(original64_path.read_bytes())
    )
    monkeypatch.setattr(selection, "ORIGINAL64_REFERENCE_IDS_SHA256", selection._ids_sha256(original64_ids))
    monkeypatch.setattr(selection, "STAGE2_MANIFEST_SHA256", _sha256(stage2_path.read_bytes()))
    monkeypatch.setattr(selection, "STAGE2_IN_DOMAIN_ROWS", 2)
    monkeypatch.setattr(selection, "STAGE2_IN_DOMAIN_COUNTS", stage2_counts)
    monkeypatch.setattr(selection, "STAGE2_IN_DOMAIN_CONTENT_SHA256", artifact["content_sha256"])
    monkeypatch.setattr(selection, "STAGE2_IN_DOMAIN_IDS_SHA256", selection._ids_sha256(stage2_ids))

    return {
        "canonical_source": canonical_source,
        "canonical_manifest": canonical_manifest_path,
        "recovered_source": recovered_source,
        "recovered_manifest": recovered_manifest,
        "original64_manifest": original64_path,
        "stage2_manifest": stage2_path,
    }


def _materialize(inputs: dict[str, Path], output: Path) -> selection.MaterializedSelection:
    return selection.materialize_rmct256_selection(
        canonical_source=inputs["canonical_source"],
        canonical_source_manifest=inputs["canonical_manifest"],
        recovered_none_source=inputs["recovered_source"],
        recovered_none_manifest=inputs["recovered_manifest"],
        original64_reference_manifest=inputs["original64_manifest"],
        stage2_manifest=inputs["stage2_manifest"],
        output_dir=output,
    )


def _verify(result: selection.MaterializedSelection, inputs: dict[str, Path]) -> dict:
    return selection.verify_selection_manifest(
        result.manifest_path,
        selection_path=result.data_path,
        canonical_source=inputs["canonical_source"],
        canonical_source_manifest=inputs["canonical_manifest"],
        recovered_none_source=inputs["recovered_source"],
        recovered_none_manifest=inputs["recovered_manifest"],
        original64_reference_manifest=inputs["original64_manifest"],
        stage2_manifest=inputs["stage2_manifest"],
    )


def test_materializes_content_addressed_exact_prefix_and_full_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _write_fixture(tmp_path, monkeypatch)
    inputs_before = {name: path.read_bytes() for name, path in inputs.items()}
    source_before = inputs_before["canonical_source"]
    result = _materialize(inputs, tmp_path / "selection")

    assert result.data_status == result.manifest_status == "written"
    assert result.data_path.name == f"rmct-256-training-{result.data_sha256}.jsonl"
    assert result.manifest_path.name == f"rmct-256-training-manifest-{result.manifest_sha256}.json"
    assert result.data_path.read_bytes() == source_before
    assert {name: path.read_bytes() for name, path in inputs.items()} == inputs_before

    document = _verify(result, inputs)
    assert document["selection"]["row_count"] == 256
    assert document["selection"]["counts_by_dataset"] == {"logiqa": 128, "hellaswag": 128}
    assert document["original64"]["question_ids"] == document["selection"]["question_ids"][:64]
    assert document["stage2_in_domain_exclusion"]["overlap_count"] == 0
    assert document["assertions"]["original64_is_strict_question_id_subset_of_rmct256"] is True

    replay = _materialize(inputs, tmp_path / "selection")
    assert replay.data_path == result.data_path
    assert replay.manifest_path == result.manifest_path
    assert replay.data_status == replay.manifest_status == "resumed"


def test_refuses_any_overlap_with_frozen_stage2_in_domain_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _write_fixture(tmp_path, monkeypatch, stage2_overlap=True)

    with pytest.raises(ValueError, match="overlaps frozen Stage-2 in-domain IDs"):
        _materialize(inputs, tmp_path / "selection")


def test_verifier_rejects_changed_content_even_when_manifest_is_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_fixture(tmp_path, monkeypatch)
    result = _materialize(inputs, tmp_path / "selection")
    payload = result.data_path.read_bytes()
    result.data_path.write_bytes(payload.replace(b"Question 0?", b"Question X?", 1))

    with pytest.raises(ValueError):
        _verify(result, inputs)


def test_full_verifier_requires_every_source_proof_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _write_fixture(tmp_path, monkeypatch)
    result = _materialize(inputs, tmp_path / "selection")

    with pytest.raises(ValueError, match="full source proof requires explicit input path"):
        selection.verify_selection_manifest(result.manifest_path, selection_path=result.data_path)


def test_rejects_canonical_source_that_does_not_reconstruct_from_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_fixture(tmp_path, monkeypatch)
    source_rows = [json.loads(line) for line in inputs["canonical_source"].read_bytes().splitlines()]
    source_rows[17]["biasing_text"] = "tampered but self-consistent hash"
    # Keep the raw canonical pair deliberately stale: source and its manifest
    # hashes are updated, so only the content-level reconstruction can catch it.
    tampered_payload = _jsonl(source_rows)
    inputs["canonical_source"].write_bytes(tampered_payload)
    canonical_manifest = json.loads(inputs["canonical_manifest"].read_text())
    canonical_manifest["content_sha256"] = _sha256(tampered_payload)
    inputs["canonical_manifest"].write_bytes(_json(canonical_manifest))
    monkeypatch.setattr(selection, "CANONICAL_SOURCE_SHA256", _sha256(tampered_payload))
    monkeypatch.setattr(selection, "CANONICAL_SOURCE_MANIFEST_SHA256", _sha256(inputs["canonical_manifest"].read_bytes()))

    with pytest.raises(ValueError, match="does not byte-match the approved recovered-no-CoT transform"):
        _materialize(inputs, tmp_path / "selection")
