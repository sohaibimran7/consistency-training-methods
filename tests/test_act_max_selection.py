"""Offline contracts for the immutable ACT-Max data selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.act_max import selection


def _row(index: int) -> dict[str, object]:
    dataset = "logiqa" if index % 2 else "hellaswag"
    question = f"Question {index}?"
    return {
        "bias_type": "wrong_argument",
        "biased_messages": [{"role": "user", "content": f"legacy biased {index}"}],
        "biased_option": "B",
        "biasing_text": f"stored wrong argument {index}",
        "ground_truth": "A",
        "prompt_style": "none",
        "question": question,
        "question_id": f"q-{index}",
        "source_dataset": dataset,
        "unbiased_messages": [{"role": "user", "content": question}],
    }


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def small_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Replace expensive frozen-source revalidation with a tiny equivalent fixture."""

    rows = [_row(index) for index in range(1, 11)]
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    canonical_prefix = selection._canonical_jsonl(
        [
            selection._qwen35_source_contract._canonicalize_wrong_argument_row(
                row, path=source, line_number=index
            )
            for index, row in enumerate(rows[:4], start=1)
        ]
    )

    def fake_recovered_and_canonical(**_kwargs: object):
        return source, rows, {"fixture": True}, canonical_prefix

    def fake_iid_and_stage2(**_kwargs: object):
        return ["q-5", "q-6"], {"fixture_iid": True}, {"fixture_stage2": True}

    monkeypatch.setattr(selection, "_verify_recovered_and_canonical", fake_recovered_and_canonical)
    monkeypatch.setattr(selection, "_verify_iid_and_stage2", fake_iid_and_stage2)
    monkeypatch.setattr(selection, "RECOVERED_ROWS", 10)
    monkeypatch.setattr(selection, "IID_ROWS", 2)
    monkeypatch.setattr(selection, "ACT_MAX_ROWS", 8)
    monkeypatch.setattr(selection, "ACT_MAX_COUNTS", {"logiqa": 4, "hellaswag": 4})
    monkeypatch.setattr(selection, "IID_SOURCE_ROWS", (5, 6))
    monkeypatch.setattr(selection, "SELECTED_SOURCE_SPANS", ((1, 4), (7, 10)))

    stage2 = tmp_path / "stage2.json"
    _json(stage2, {"populations": {"hle": {"artifacts": {"unbiased": {"question_ids": ["hle-q"]}}}}})
    placeholders = {
        "recovered_none_source": source,
        "recovered_none_manifest": tmp_path / "recovered.manifest.json",
        "canonical_prefix": tmp_path / "canonical.jsonl",
        "canonical_prefix_manifest": tmp_path / "canonical.manifest.json",
        "iid_reference_manifest": tmp_path / "iid.manifest.json",
        "iid_heldout": tmp_path / "iid.jsonl",
        "stage2_manifest": stage2,
    }
    for key, path in placeholders.items():
        if path != source and path != stage2:
            path.write_text("{}\n", encoding="utf-8")
    return placeholders


def test_materializes_all_and_only_the_non_iid_rows(small_contract: dict[str, Path], tmp_path: Path) -> None:
    result = selection.materialize_act_max_selection(output_dir=tmp_path / "out", **small_contract)

    assert result.data_status == result.manifest_status == "written"
    assert result.data_path.name == f"act-max-training-{result.data_sha256}.jsonl"
    assert result.manifest_path.name == f"act-max-training-manifest-{result.manifest_sha256}.json"
    rows = [json.loads(line) for line in result.data_path.read_text(encoding="utf-8").splitlines()]
    assert [row["question_id"] for row in rows] == ["q-1", "q-2", "q-3", "q-4", "q-7", "q-8", "q-9", "q-10"]
    document = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert document["selection"]["counts_by_dataset"] == {"hellaswag": 4, "logiqa": 4}
    assert document["iid_headline_reservation"]["overlap_count"] == 0
    assert document["stage2"]["source_hle_overlap_count"] == 0
    assert "path" not in json.dumps(document)  # portable across staging roots

    verified = selection.verify_act_max_selection(
        selection=result.data_path,
        selection_manifest=result.manifest_path,
        **small_contract,
    )
    assert verified["assertions"]["selected_rows_are_all_and_only_non_iid_source_rows"] is True

    resumed = selection.materialize_act_max_selection(output_dir=tmp_path / "out", **small_contract)
    assert resumed.data_status == resumed.manifest_status == "resumed"


def test_verifier_rejects_tampered_selection(small_contract: dict[str, Path], tmp_path: Path) -> None:
    result = selection.materialize_act_max_selection(output_dir=tmp_path / "out", **small_contract)
    payload = result.data_path.read_bytes()
    result.data_path.write_bytes(payload.replace(b"Question 1?", b"Question X?", 1))

    with pytest.raises(ValueError, match="selection bytes"):
        selection.verify_act_max_selection(
            selection=result.data_path,
            selection_manifest=result.manifest_path,
            **small_contract,
        )


def test_rejects_a_holdout_that_does_not_match_the_fixed_source_gap(
    small_contract: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def wrong_iid(**_kwargs: object):
        return ["q-1", "q-6"], {"fixture_iid": True}, {"fixture_stage2": True}

    monkeypatch.setattr(selection, "_verify_iid_and_stage2", wrong_iid)
    with pytest.raises(ValueError, match="selected source positions"):
        selection.materialize_act_max_selection(output_dir=tmp_path / "out", **small_contract)
