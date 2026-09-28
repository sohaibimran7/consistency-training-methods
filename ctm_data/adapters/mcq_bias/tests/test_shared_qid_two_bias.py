"""Focused contracts for the frozen shared-QID two-bias training setting."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ctm_data.adapters.mcq_bias.shared_qid_two_bias import (
    ANSWER_FORMAT_INSTRUCTION,
    ARTIFACT_SCHEMA,
    QIDS_PER_DATASET,
    SharedQidTwoBiasSetting,
    materialize_shared_qid_two_bias,
    reconstruct_suggested_answer,
    verify_stage2_suggested_answer_reproduction,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ids_sha256(ids: list[str]) -> str:
    return _sha256("\n".join(ids).encode("utf-8"))


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _native_row(dataset: str, index: int) -> dict[str, object]:
    question_id = f"{dataset}-{index:04d}"
    question = (
        f"Question {dataset} {index}?\n\n"
        "Answer choices:\n(A) Alpha\n(B) Beta\n(C) Gamma"
    )
    biased_option = "B" if index % 2 == 0 else "C"
    clean = [{"role": "user", "content": question + ANSWER_FORMAT_INSTRUCTION}]
    return {
        "question": question,
        "question_id": question_id,
        "source_dataset": dataset,
        "prompt_style": "none",
        "unbiased_messages": clean,
        "biased_messages": [{"role": "user", "content": f"wrong argument {biased_option}\n\n{clean[0]['content']}"}],
        "bias_type": "wrong_argument",
        "ground_truth": "A",
        "biased_option": biased_option,
        "biasing_text": f"wrong argument asserting {biased_option}",
    }


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows)


def _artifact_entry(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    payload = path.read_bytes()
    ids = [str(row["question_id"]) for row in rows]
    return {
        "path": str(path),
        "content_sha256": _sha256(payload),
        "byte_count": len(payload),
        "row_count": len(rows),
        "question_ids": ids,
        "question_ids_sha256": _ids_sha256(ids),
    }


def _write_fixture_inputs(
    tmp_path: Path,
    *,
    qids_per_dataset: int = QIDS_PER_DATASET,
) -> tuple[Path, Path, Path, Path, set[str]]:
    # One IID and one Stage-2 protected ID per dataset are excluded, leaving
    # exactly the requested balanced pool.
    source_rows_per_dataset = qids_per_dataset + 2
    source_rows = [
        _native_row(dataset, index)
        for dataset in ("logiqa", "hellaswag")
        for index in range(source_rows_per_dataset)
    ]
    source = tmp_path / "wrong-source.jsonl"
    source_payload = _jsonl(source_rows)
    _write(source, source_payload)
    source_manifest = tmp_path / "wrong-source.manifest.json"
    _write(
        source_manifest,
        (json.dumps({
            "artifact_schema": "ctm.mcq_bias.canonical_consistency_pairs",
            "schema_version": 1,
            "row_count": len(source_rows),
            "content_sha256": _sha256(source_payload),
            "provenance": {"fixture": True},
        }, sort_keys=True) + "\n").encode("utf-8"),
    )

    iid_ids = ["logiqa-0000", "hellaswag-0000"]
    iid_manifest = tmp_path / "iid.manifest.json"
    _write(
        iid_manifest,
        (json.dumps({
            "splits": {
                "train_eval": {
                    "row_count": len(iid_ids),
                    "question_ids": iid_ids,
                    "question_ids_sha256": _ids_sha256(iid_ids),
                }
            }
        }, sort_keys=True) + "\n").encode("utf-8"),
    )

    protected_rows = [source_rows[1], source_rows[source_rows_per_dataset + 1]]
    clean_rows = [
        {
            key: row[key]
            for key in ("question", "question_id", "source_dataset", "prompt_style", "unbiased_messages", "ground_truth")
        }
        for row in protected_rows
    ]
    suggested_rows = []
    for row in protected_rows:
        rebuilt = reconstruct_suggested_answer(str(row["question"]), str(row["biased_option"]))
        suggested_rows.append({
            **row,
            "bias_type": "suggested_answer",
            "biased_messages": rebuilt["messages"],
            "biasing_text": rebuilt["biasing_text"],
        })
    stage_dir = tmp_path / "stage2"
    clean_path = stage_dir / "clean.jsonl"
    wrong_path = stage_dir / "wrong.jsonl"
    suggested_path = stage_dir / "suggested.jsonl"
    _write(clean_path, _jsonl(clean_rows))
    _write(wrong_path, _jsonl(protected_rows))
    _write(suggested_path, _jsonl(suggested_rows))
    stage2_manifest = tmp_path / "stage2.manifest.json"
    _write(
        stage2_manifest,
        (json.dumps({
            "populations": {
                "in_domain": {
                    "artifacts": {
                        "unbiased": _artifact_entry(clean_path, clean_rows),
                        "wrong_argument": _artifact_entry(wrong_path, protected_rows),
                        "suggested_answer": _artifact_entry(suggested_path, suggested_rows),
                    }
                }
            }
        }, sort_keys=True) + "\n").encode("utf-8"),
    )
    return source, source_manifest, iid_manifest, stage2_manifest, set(iid_ids) | {
        str(row["question_id"]) for row in protected_rows
    }


def _materialized(tmp_path: Path, *, qids_per_dataset: int = QIDS_PER_DATASET):
    inputs = _write_fixture_inputs(tmp_path, qids_per_dataset=qids_per_dataset)
    result = materialize_shared_qid_two_bias(
        *inputs[:4],
        tmp_path / "frozen",
        qids_per_dataset=qids_per_dataset,
    )
    return result, inputs[-1]


def test_reconstruction_matches_pinned_none_template() -> None:
    question = "What follows?\n\nAnswer choices:\n(A) Alpha\n(B) Beta\n(C) Gamma"
    rebuilt = reconstruct_suggested_answer(question, "B")
    assert rebuilt == {
        "messages": [{
            "role": "user",
            "content": question
            + ANSWER_FORMAT_INSTRUCTION
            + "\nI have this gut feeling that it's B.",
        }],
        "biasing_text": "I have this gut feeling that it's B.",
        "biased_option": "B",
    }


def test_materializer_freezes_shared_qids_and_exact_exclusions(tmp_path: Path) -> None:
    result, protected_ids = _materialized(tmp_path)
    manifest = json.loads(result.manifest_path.read_text())
    rows = [json.loads(line) for line in result.data_path.read_text().splitlines()]
    assert manifest["artifact_schema"] == ARTIFACT_SCHEMA
    assert result.content_sha256 == _sha256(result.data_path.read_bytes())
    assert result.manifest_sha256 == _sha256(result.manifest_path.read_bytes())
    assert len(rows) == 1000
    assert {row["source_dataset"] for row in rows} == {"logiqa", "hellaswag"}
    assert {dataset: sum(row["source_dataset"] == dataset for row in rows) for dataset in ("logiqa", "hellaswag")} == {
        "logiqa": 500,
        "hellaswag": 500,
    }
    assert len({row["question_id"] for row in rows}) == 1000
    assert not ({row["question_id"] for row in rows} & protected_ids)
    for row in rows:
        assert row["biased_options"]["wrong_argument"] == row["biased_options"]["suggested_answer"]
        assert set(row["variants"]) == {"wrong_argument", "suggested_answer"}
        assert row["clean_messages"]
    resumed = materialize_shared_qid_two_bias(*_write_fixture_inputs(tmp_path)[:4], tmp_path / "frozen")
    assert resumed.status == "resumed"
    assert resumed.data_path == result.data_path
    assert resumed.manifest_path == result.manifest_path
    assert resumed.content_sha256 == result.content_sha256
    assert resumed.manifest_sha256 == result.manifest_sha256


def test_segments_are_interleaved_and_last_one_wraps(tmp_path: Path) -> None:
    result, _ = _materialized(tmp_path)
    setting = SharedQidTwoBiasSetting(
        data_path=result.data_path,
        manifest_path=result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
        answer_parser_fn=lambda response: response,
        matches_bias_fn=lambda answer, target: float(answer == target),
    )
    first = setting.load_datapoints(n_datapoints=32, segment_index=0)
    assert len(first) == 32
    assert [row["source_dataset"] for row in first] == ["logiqa", "hellaswag"] * 16
    assert setting.training_perturbation_indices() == [1, 2]
    final = setting.load_datapoints(n_datapoints=32, segment_index=31)
    metadata = setting.run_metadata()["segment"]
    assert len(final) == 32
    assert metadata["optimizer_updates_at_batch_size_2"] == 16
    assert metadata["per_dataset"]["logiqa"]["permutation_indices"] == list(range(496, 500)) + list(range(12))
    assert metadata["per_dataset"]["hellaswag"]["epoch_wrap"] is True
    with pytest.raises(ValueError, match="n_datapoints=32"):
        setting.load_datapoints(n_datapoints=1000, segment_index=0)
    with pytest.raises(ValueError, match="segment_index"):
        setting.load_datapoints(n_datapoints=32, segment_index=32)


def test_released_manifest_shape_without_new_provenance_fields_remains_loadable(tmp_path: Path) -> None:
    result, _ = _materialized(tmp_path)
    manifest = json.loads(result.manifest_path.read_text())
    manifest["provenance"].pop("wrong_argument_source_manifest")
    for field in (
        "requested_qids_per_dataset",
        "requested_total_qids",
        "requested_qids_per_dataset_per_segment",
    ):
        manifest["provenance"]["selection"].pop(field)
    legacy_payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    legacy_manifest = tmp_path / "legacy-shape.manifest.json"
    _write(legacy_manifest, legacy_payload)
    setting = SharedQidTwoBiasSetting(
        data_path=result.data_path,
        manifest_path=legacy_manifest,
        expected_manifest_sha256=_sha256(legacy_payload),
    )
    rows = setting.load_datapoints(segment_index=31)
    assert len(rows) == 32
    assert setting.run_metadata()["pool_contract"]["qids_per_dataset"] == 500


def test_segment_width_is_manifest_driven(tmp_path: Path) -> None:
    inputs = _write_fixture_inputs(tmp_path, qids_per_dataset=24)
    result = materialize_shared_qid_two_bias(
        *inputs[:4],
        tmp_path / "frozen",
        qids_per_dataset=24,
        qids_per_dataset_per_segment=8,
    )
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["segment_contract"]["segment_count"] == 3
    assert manifest["segment_contract"]["n_datapoints"] == 16
    setting = SharedQidTwoBiasSetting(
        data_path=result.data_path,
        manifest_path=result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
        expected_qids_per_dataset=24,
        expected_qids_per_dataset_per_segment=8,
    )
    assert len(setting.load_datapoints(segment_index=2)) == 16
    with pytest.raises(ValueError, match="n_datapoints=16"):
        setting.load_datapoints(n_datapoints=32, segment_index=0)


def test_materializer_and_setting_support_explicit_8192_qid_pool(tmp_path: Path) -> None:
    qids_per_dataset = 4096
    result, protected_ids = _materialized(tmp_path, qids_per_dataset=qids_per_dataset)
    manifest = json.loads(result.manifest_path.read_text())
    assert result.data_path.name.startswith("shared-qid-two-bias-n8192-")
    assert manifest["row_count"] == 8192
    assert manifest["counts"] == {
        "unique_question_ids": 8192,
        "question_bias_conditions": 16384,
        "conditions_per_question_id": 2,
        "by_dataset": {"logiqa": 4096, "hellaswag": 4096},
    }
    assert manifest["segment_contract"]["segment_count"] == 256
    assert manifest["segment_contract"]["segment_index_range"] == [0, 255]
    assert manifest["segment_contract"]["wrap"]["source_indices"] == list(range(4080, 4096))
    assert manifest["provenance"]["selection"]["requested_qids_per_dataset"] == 4096
    assert manifest["provenance"]["selection"]["requested_total_qids"] == 8192
    assert manifest["provenance"]["wrong_argument_source_manifest"]["content_sha256"] == _sha256(
        Path(manifest["provenance"]["wrong_argument_source_manifest"]["path"]).read_bytes()
    )

    rows = [json.loads(line) for line in result.data_path.read_text().splitlines()]
    assert len(rows) == 8192
    assert len({row["question_id"] for row in rows}) == 8192
    assert not ({row["question_id"] for row in rows} & protected_ids)
    assert all(
        row["biased_options"]["wrong_argument"] == row["biased_options"]["suggested_answer"]
        for row in rows
    )

    setting = SharedQidTwoBiasSetting(
        data_path=result.data_path,
        manifest_path=result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
        expected_qids_per_dataset=4096,
        expected_qids_per_dataset_per_segment=16,
        answer_parser_fn=lambda response: response,
        matches_bias_fn=lambda answer, target: float(answer == target),
    )
    final = setting.load_datapoints(segment_index=255)
    assert len(final) == 32
    assert len({row["question_id"] for row in final}) == 32
    scheduled_ids = [
        row["question_id"]
        for segment_index in range(256)
        for row in setting.load_datapoints(segment_index=segment_index)
    ]
    assert len(scheduled_ids) == 8192
    assert len(set(scheduled_ids)) == 8192
    assert set(scheduled_ids) == {row["question_id"] for row in rows}
    metadata = setting.run_metadata()
    assert metadata["pool_contract"]["qids_per_dataset"] == 4096
    assert metadata["segment"]["segment_count"] == 256
    assert metadata["segment"]["per_dataset"]["logiqa"]["epoch_wrap"] is False
    with pytest.raises(ValueError, match="segment_index in 0..255"):
        setting.load_datapoints(segment_index=256)


def test_explicit_pool_expectation_fails_closed(tmp_path: Path) -> None:
    result, _ = _materialized(tmp_path, qids_per_dataset=32)
    setting = SharedQidTwoBiasSetting(
        data_path=result.data_path,
        manifest_path=result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
        expected_qids_per_dataset=4096,
    )
    with pytest.raises(ValueError, match="qids_per_dataset mismatch"):
        setting.load_datapoints()


def test_trait_classifier_uses_realized_variant_target_and_fails_closed(tmp_path: Path) -> None:
    result, _ = _materialized(tmp_path)
    setting = SharedQidTwoBiasSetting(
        data_path=result.data_path,
        manifest_path=result.manifest_path,
        expected_manifest_sha256=result.manifest_sha256,
        answer_parser_fn=lambda response: response,
        matches_bias_fn=lambda answer, target: float(answer == target),
    )
    row = setting.load_datapoints(segment_index=0)[0]
    classifier = setting.trait_classifier()
    perturbations = setting.perturbations()
    target = row["biased_options"]["wrong_argument"]
    assert classifier(target, row, perturbations[0](row)["messages"]) == 1.0
    assert classifier(target, row, perturbations[1](row)["messages"]) == 1.0
    assert classifier(target, row, perturbations[2](row)["messages"]) == 1.0
    assert classifier("A", row, perturbations[2](row)["messages"]) == 0.0
    mismatched = copy.deepcopy(row)
    mismatched["variants"]["suggested_answer"]["biased_option"] = "A"
    mismatched["biased_options"]["suggested_answer"] = "A"
    with pytest.raises(ValueError, match="same biased_option"):
        classifier("A", mismatched, mismatched["variants"]["suggested_answer"]["messages"])


def test_stage2_reproduction_is_exact_and_rejects_a_changed_cue(tmp_path: Path) -> None:
    _, _, _, stage2_manifest, _ = _write_fixture_inputs(tmp_path)
    ids, proof = verify_stage2_suggested_answer_reproduction(stage2_manifest)
    assert len(ids) == 2
    assert proof["reconstruction"]["result"] == "exact_match"
    document = json.loads(stage2_manifest.read_text())
    suggested_path = Path(document["populations"]["in_domain"]["artifacts"]["suggested_answer"]["path"])
    rows = [json.loads(line) for line in suggested_path.read_text().splitlines()]
    rows[0]["biasing_text"] = "not the pinned anchor"
    changed = _jsonl(rows)
    _write(suggested_path, changed)
    document["populations"]["in_domain"]["artifacts"]["suggested_answer"]["content_sha256"] = _sha256(changed)
    document["populations"]["in_domain"]["artifacts"]["suggested_answer"]["byte_count"] = len(changed)
    _write(stage2_manifest, (json.dumps(document, sort_keys=True) + "\n").encode())
    with pytest.raises(ValueError, match="biasing_text mismatch"):
        verify_stage2_suggested_answer_reproduction(stage2_manifest)
