"""Offline contracts for the RMCT-256 convergence-segment manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.rmct_256 import selection as parent
from experiments.rmct_256_convergence import selection


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)


def _row(index: int) -> dict:
    dataset = "logiqa" if index % 2 == 0 else "hellaswag"
    clean = f"Clean prompt {index}"
    return {
        "question": f"Question {index}?",
        "question_id": f"q-{index:03d}",
        "source_dataset": dataset,
        "prompt_style": "none",
        "unbiased_messages": [{"role": "user", "content": clean}],
        "biased_messages": [{"role": "user", "content": f"Wrong argument {index}\n\n{clean}"}],
        "bias_type": "wrong_argument",
        "ground_truth": "A",
        "biased_option": "B",
        "biasing_text": f"Wrong argument {index}",
        "consistency_pair_transform": parent.CANONICAL_PAIR_TRANSFORM,
    }


def _parent_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict]:
    rows = [_row(index) for index in range(selection.PARENT_ROWS)]
    payload = _jsonl(rows)
    data_path = tmp_path / f"rmct-256-training-{_sha256(payload)}.jsonl"
    data_path.write_bytes(payload)
    ids = [row["question_id"] for row in rows]
    document = {
        "selection": {
            "filename": data_path.name,
            "content_sha256": _sha256(payload),
            "byte_count": len(payload),
            "row_count": selection.PARENT_ROWS,
            "counts_by_dataset": dict(parent.SELECTION_COUNTS),
            "question_ids": ids,
            "question_ids_sha256": parent._ids_sha256(ids),
            "source_rows_1_based_inclusive": [1, selection.PARENT_ROWS],
            "selection_method": "exact_ordered_source_prefix_without_shuffle_or_reserialization",
        }
    }
    manifest_payload = _json(document)
    manifest_path = tmp_path / f"rmct-256-training-manifest-{_sha256(manifest_payload)}.json"
    manifest_path.write_bytes(manifest_payload)

    def fake_parent_verifier(path, *, selection_path, verify_sources, **_):
        assert Path(path) == manifest_path
        assert Path(selection_path) == data_path
        assert isinstance(verify_sources, bool)
        return document

    monkeypatch.setattr(parent, "verify_selection_manifest", fake_parent_verifier)
    return data_path, manifest_path, document


def _source_kwargs() -> dict[str, str]:
    # The parent verifier is mocked in these unit tests. Production
    # materialization separately requires every real immutable source input.
    return {
        "canonical_source": "canonical.jsonl",
        "canonical_source_manifest": "canonical.manifest.json",
        "recovered_none_source": "recovered.jsonl",
        "recovered_none_manifest": "recovered.manifest.json",
        "original64_reference_manifest": "original64.manifest.json",
        "stage2_manifest": "stage2.manifest.json",
    }


def test_materializes_four_byte_exact_balanced_disjoint_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_path, parent_manifest, parent_document = _parent_fixture(tmp_path, monkeypatch)
    payload_before = parent_path.read_bytes()

    result = selection.materialize_rmct256_convergence_segments(
        parent_selection=parent_path,
        parent_selection_manifest=parent_manifest,
        output_dir=tmp_path / "segments",
        **_source_kwargs(),
    )

    assert result.manifest_status == "written"
    assert result.manifest_path.name == f"{selection.MANIFEST_FILENAME_PREFIX}{result.manifest_sha256}.json"
    document = json.loads(result.manifest_path.read_bytes())
    segments = document["segments"]
    assert [(segment["index"], segment["row_offset"], segment["row_count"]) for segment in segments] == [
        (0, 0, 64),
        (1, 64, 64),
        (2, 128, 64),
        (3, 192, 64),
    ]
    assert all(segment["counts_by_dataset"] == {"logiqa": 32, "hellaswag": 32} for segment in segments)
    assert all(
        segment["question_ids"] == parent_document["selection"]["question_ids"][
            segment["row_offset"] : segment["row_offset"] + segment["row_count"]
        ]
        for segment in segments
    )
    segment_ids = [set(segment["question_ids"]) for segment in segments]
    assert all(first.isdisjoint(second) for index, first in enumerate(segment_ids) for second in segment_ids[index + 1 :])
    assert [question_id for segment in segments for question_id in segment["question_ids"]] == parent_document["selection"][
        "question_ids"
    ]
    assert parent_path.read_bytes() == payload_before

    verified = selection.verify_rmct256_convergence_segments_manifest(
        result.manifest_path,
        parent_selection=parent_path,
        parent_selection_manifest=parent_manifest,
        verify_parent_sources=False,
        expected_manifest_sha256=result.manifest_sha256,
    )
    assert verified == document

    replay = selection.materialize_rmct256_convergence_segments(
        parent_selection=parent_path,
        parent_selection_manifest=parent_manifest,
        output_dir=tmp_path / "segments",
        **_source_kwargs(),
    )
    assert replay == result.__class__(
        manifest_path=result.manifest_path,
        manifest_sha256=result.manifest_sha256,
        manifest_status="resumed",
    )


def test_transport_verifier_rejects_parent_drift_or_wrong_expected_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_path, parent_manifest, _ = _parent_fixture(tmp_path, monkeypatch)
    result = selection.materialize_rmct256_convergence_segments(
        parent_selection=parent_path,
        parent_selection_manifest=parent_manifest,
        output_dir=tmp_path / "segments",
        **_source_kwargs(),
    )

    with pytest.raises(ValueError, match="expected launch contract"):
        selection.verify_rmct256_convergence_segments_manifest(
            result.manifest_path,
            parent_selection=parent_path,
            parent_selection_manifest=parent_manifest,
            verify_parent_sources=False,
            expected_manifest_sha256="0" * 64,
        )

    parent_path.write_bytes(parent_path.read_bytes().replace(b"Clean prompt 0", b"Changed prompt 0"))
    with pytest.raises(ValueError, match="selection record differs"):
        selection.verify_rmct256_convergence_segments_manifest(
            result.manifest_path,
            parent_selection=parent_path,
            parent_selection_manifest=parent_manifest,
            verify_parent_sources=False,
            expected_manifest_sha256=result.manifest_sha256,
        )


def test_full_verification_requires_every_parent_source_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_path, parent_manifest, _ = _parent_fixture(tmp_path, monkeypatch)
    result = selection.materialize_rmct256_convergence_segments(
        parent_selection=parent_path,
        parent_selection_manifest=parent_manifest,
        output_dir=tmp_path / "segments",
        **_source_kwargs(),
    )

    with pytest.raises(ValueError, match="full convergence-segment proof requires explicit input path"):
        selection.verify_rmct256_convergence_segments_manifest(
            result.manifest_path,
            parent_selection=parent_path,
            parent_selection_manifest=parent_manifest,
        )
