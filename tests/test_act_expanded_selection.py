"""Contracts for the offline expanded-ACT question freezer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.act_expanded import selection


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_bytes(selection._canonical_jsonl(rows))
    return path


def _options(identity: str) -> list[str]:
    return [f"correct {identity}", f"wrong {identity}"]


def _raw(question: str, *, identity: str, answer: int = 0) -> dict[str, object]:
    return {"question": question, "options": _options(identity), "ground_truth_idx": answer}


def _legacy_row(dataset: str, question: str, question_id: str) -> dict[str, object]:
    """Include old target fields to prove that the freezer never republishes them."""

    return {
        "question": question,
        "question_id": question_id,
        "source_dataset": dataset,
        "ground_truth": "A",
        "biasing_text": f"old target that must never be reused for {question_id}",
        "biased_messages": [{"role": "user", "content": "old biased prompt"}],
        "unbiased_messages": [{"role": "user", "content": question}],
    }


@pytest.fixture
def small_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path | dict[str, Path]]:
    """A small cross-format fixture with a duplicate source row and old target text."""

    monkeypatch.setattr(
        selection,
        "SOURCE_SPECS",
        {
            "logiqa": {
                "repository": "fixture/logiqa",
                "revision": "logiqa-revision",
                "split": "train",
                "raw_row_count": 6,
                "unique_canonical_count": 5,
            },
            "hellaswag": {
                "repository": "fixture/hellaswag",
                "revision": "hellaswag-revision",
                "split": "train",
                "raw_row_count": 6,
                "unique_canonical_count": 6,
            },
        },
    )
    monkeypatch.setattr(
        selection,
        "CORE_CROSS_FORMAT_EXPECTATIONS",
        {
            "logiqa": {
                "source_physical_rows": 6,
                "source_unique_rows": 5,
                "legacy_collision_physical_rows": 3,
                "legacy_collision_unique_rows": 3,
                "fresh_safe_physical_rows": 3,
                "fresh_safe_unique_rows": 2,
            },
            "hellaswag": {
                "source_physical_rows": 6,
                "source_unique_rows": 6,
                "legacy_collision_physical_rows": 1,
                "legacy_collision_unique_rows": 1,
                "fresh_safe_physical_rows": 5,
                "fresh_safe_unique_rows": 5,
            },
        },
    )
    monkeypatch.setattr(selection, "LEGACY_POPULATION_COUNTS", {"logiqa": 3, "hellaswag": 3})
    monkeypatch.setattr(selection, "LEGACY_TRAINING_COUNTS", {"logiqa": 2, "hellaswag": 2})
    monkeypatch.setattr(selection, "EXISTING_FROZEN_IID_COUNTS", {"logiqa": 1, "hellaswag": 1})
    monkeypatch.setattr(selection, "FRESH_IID_RESERVE_PER_DATASET", 1)

    # The LogiQA source and legacy row differ in spacing/case/quote glyphs;
    # they must still collide after repository-rendered MCQ normalisation.
    legacy_l_iid = selection._render_canonical_input("Legacy iid", _options("legacy-l-iid"))
    legacy_l_one = selection._render_canonical_input("Legacy one ‘quoted’", _options("legacy-l-one"))
    legacy_l_two = selection._render_canonical_input("Legacy two", _options("legacy-l-two"))
    legacy_h_iid = selection._render_canonical_input("Hella iid", _options("legacy-h-iid"))
    legacy_h_one = selection._render_canonical_input("Hella one", _options("legacy-h-one"))
    legacy_h_two = selection._render_canonical_input("Hella two", _options("legacy-h-two"))

    logiqa_raw = _write_jsonl(
        tmp_path / "logiqa.raw.jsonl",
        [
            _raw("legacy iid", identity="legacy-l-iid"),
            _raw("Legacy   one 'quoted'", identity="legacy-l-one"),
            _raw("Legacy two", identity="legacy-l-two"),
            _raw("fresh logiqa three", identity="fresh-l-three"),
            _raw("fresh logiqa four", identity="fresh-l-four"),
            _raw("fresh logiqa three", identity="fresh-l-three"),  # physical duplicate, one canonical fresh question
        ],
    )
    hellaswag_raw = _write_jsonl(
        tmp_path / "hellaswag.raw.jsonl",
        [
            _raw("Hella iid", identity="legacy-h-iid"),
            _raw("fresh hella one", identity="fresh-h-one"),
            _raw("fresh hella two", identity="fresh-h-two"),
            _raw("fresh hella three", identity="fresh-h-three"),
            _raw("fresh hella four", identity="fresh-h-four"),
            _raw("fresh hella five", identity="fresh-h-five"),
        ],
    )
    snapshots = tmp_path / "snapshots"
    logiqa_snapshot = selection.materialize_source_snapshot(dataset="logiqa", raw_source=logiqa_raw, output_dir=snapshots)
    hellaswag_snapshot = selection.materialize_source_snapshot(dataset="hellaswag", raw_source=hellaswag_raw, output_dir=snapshots)

    all_rows = [
        _legacy_row("logiqa", legacy_l_iid, "legacy-l-iid"),
        _legacy_row("logiqa", legacy_l_one, "legacy-l-one"),
        _legacy_row("logiqa", legacy_l_two, "legacy-l-two"),
        _legacy_row("hellaswag", legacy_h_iid, "legacy-h-iid"),
        _legacy_row("hellaswag", legacy_h_one, "legacy-h-one"),
        _legacy_row("hellaswag", legacy_h_two, "legacy-h-two"),
    ]
    legacy_population = _write_jsonl(tmp_path / "legacy-population.jsonl", all_rows)
    legacy_training = _write_jsonl(
        tmp_path / "legacy-training.jsonl",
        [all_rows[1], all_rows[2], all_rows[4], all_rows[5]],
    )
    existing_iid = _write_jsonl(tmp_path / "existing-iid.jsonl", [all_rows[0], all_rows[3]])
    prohibited = _write_jsonl(
        tmp_path / "prohibited-hle.jsonl",
        [{"question": selection._render_canonical_input("unrelated hle", ["a", "b"]), "question_id": "hle-1"}],
    )
    return {
        "logiqa_source": logiqa_snapshot.data_path,
        "logiqa_manifest": logiqa_snapshot.manifest_path,
        "hellaswag_source": hellaswag_snapshot.data_path,
        "hellaswag_manifest": hellaswag_snapshot.manifest_path,
        "legacy_population": legacy_population,
        "legacy_training": legacy_training,
        "existing_iid": existing_iid,
        "prohibited_populations": {"hle": prohibited},
    }


def test_audit_reports_both_requested_capacity_conditions(small_contract: dict[str, Path | dict[str, Path]], tmp_path: Path) -> None:
    audit = selection.materialize_cross_format_audit(output_dir=tmp_path / "audit", **small_contract)
    assert audit.status == "written"
    document = json.loads(audit.path.read_text(encoding="utf-8"))

    assert document["cross_format_mapping"]["logiqa"]["legacy_collision_physical_rows"] == 3
    assert document["cross_format_mapping"]["logiqa"]["legacy_collision_unique_rows"] == 3
    assert document["cross_format_mapping"]["logiqa"]["fresh_safe_unique_rows"] == 2
    # 2 legacy ACT-Max + 2 fresh LogiQA / 2 legacy + 5 fresh HellaSwag.
    assert document["capacity_report"]["existing_frozen_iid_only"]["question_candidate_counts_by_dataset"] == {
        "logiqa": 4,
        "hellaswag": 7,
    }
    # A distinct new IID reserve is taken only from the fresh safe pool.
    assert document["capacity_report"]["with_additional_fresh_iid_100_per_dataset"]["question_candidate_counts_by_dataset"] == {"logiqa": 3, "hellaswag": 6}
    assert document["generator_policy"]["legacy_biasing_text_must_not_be_reused"] is True

    assert selection.verify_cross_format_audit(cross_format_audit=audit.path, **small_contract) == document
    resumed = selection.materialize_cross_format_audit(output_dir=tmp_path / "audit", **small_contract)
    assert resumed.status == "resumed"


def test_selection_is_question_only_and_replays_its_audit(small_contract: dict[str, Path | dict[str, Path]], tmp_path: Path) -> None:
    audit = selection.materialize_cross_format_audit(output_dir=tmp_path / "audit", **small_contract)
    result = selection.materialize_expanded_act_selection(cross_format_audit=audit.path, output_dir=tmp_path / "selection", **small_contract)
    assert result.fresh_iid_status == result.candidate_status == result.manifest_status == "written"
    fresh_iid = [json.loads(line) for line in result.fresh_iid_path.read_text(encoding="utf-8").splitlines()]
    candidates = [json.loads(line) for line in result.candidate_path.read_text(encoding="utf-8").splitlines()]
    assert selection._row_counts(fresh_iid) == {"logiqa": 1, "hellaswag": 1}
    assert selection._row_counts(candidates) == {"logiqa": 3, "hellaswag": 6}
    assert {row["candidate_id"] for row in fresh_iid}.isdisjoint(row["candidate_id"] for row in candidates)
    assert {row["source_origin"] for row in candidates} == {
        "legacy_act_max_question",
        "pinned_hf_train_question",
    }
    for row in [*fresh_iid, *candidates]:
        assert not {"biasing_text", "biased_messages", "unbiased_messages"} & set(row)
        assert row["selection_role"] in {"fresh_iid_reserve", "homogeneous_gemma_generation_candidate"}

    manifest = selection.verify_expanded_act_selection(
        cross_format_audit=audit.path,
        fresh_iid_selection=result.fresh_iid_path,
        candidate_selection=result.candidate_path,
        selection_manifest=result.manifest_path,
        **small_contract,
    )
    contract = manifest["homogeneous_argument_generation_contract"]
    assert contract["model"] == "google/gemma-4-31b-it"
    assert contract["legacy_biasing_text_is_not_a_valid_input_or_output"] is True

    resumed = selection.materialize_expanded_act_selection(cross_format_audit=audit.path, output_dir=tmp_path / "selection", **small_contract)
    assert resumed.fresh_iid_status == resumed.candidate_status == resumed.manifest_status == "resumed"


def test_prohibited_population_removes_a_fresh_content_collision(small_contract: dict[str, Path | dict[str, Path]], tmp_path: Path) -> None:
    prohibited = small_contract["prohibited_populations"]
    assert isinstance(prohibited, dict)
    _write_jsonl(
        prohibited["hle"],
        [
            {
                # The foreign ID cannot be relied on; normalized rendered MCQ
                # content must exclude this fresh LogiQA question.
                "question": selection._render_canonical_input("fresh logiqa four", _options("fresh-l-four")),
                "question_id": "foreign-hle-id",
            }
        ],
    )
    audit = selection.materialize_cross_format_audit(output_dir=tmp_path / "audit", **small_contract)
    document = json.loads(audit.path.read_text(encoding="utf-8"))
    metrics = document["post_prohibited_filter"]["logiqa"]
    assert metrics["fresh_safe_unique_rows_before_prohibited"] == 2
    assert metrics["excluded_by_prohibited_population_unique_rows"] == 1
    assert metrics["fresh_eligible_unique_rows_after_prohibited"] == 1
    assert len(metrics["excluded_source_question_ids_sha256"]) == 64
    assert document["capacity_report"]["with_additional_fresh_iid_100_per_dataset"]["question_candidate_counts_by_dataset"]["logiqa"] == 2


def test_verifier_rejects_substituted_candidate_bytes(small_contract: dict[str, Path | dict[str, Path]], tmp_path: Path) -> None:
    audit = selection.materialize_cross_format_audit(output_dir=tmp_path / "audit", **small_contract)
    result = selection.materialize_expanded_act_selection(cross_format_audit=audit.path, output_dir=tmp_path / "selection", **small_contract)
    result.candidate_path.write_bytes(result.candidate_path.read_bytes().replace(b"fresh", b"altered", 1))

    with pytest.raises(ValueError, match="candidate selection bytes"):
        selection.verify_expanded_act_selection(
            cross_format_audit=audit.path,
            fresh_iid_selection=result.fresh_iid_path,
            candidate_selection=result.candidate_path,
            selection_manifest=result.manifest_path,
            **small_contract,
        )


def test_balanced_target_helpers_are_strict() -> None:
    assert selection.final_training_count(19, optimizer_granularity=8) == 16
    selection.require_hellaswag_capacity(16, target_count=16)
    with pytest.raises(ValueError, match="need at least 17"):
        selection.require_hellaswag_capacity(16, target_count=17)


def test_fresh_only_selection_is_balanced_and_excludes_all_legacy_questions(
    small_contract: dict[str, Path | dict[str, Path]], tmp_path: Path
) -> None:
    audit = selection.materialize_cross_format_audit(output_dir=tmp_path / "audit", **small_contract)
    result = selection.materialize_expanded_act_selection(
        cross_format_audit=audit.path,
        output_dir=tmp_path / "selection",
        candidate_source_mode="fresh_only",
        n_total=2,
        **small_contract,
    )
    rows = [json.loads(line) for line in result.candidate_path.read_text(encoding="utf-8").splitlines()]
    assert [row["source_dataset"] for row in rows] == ["logiqa", "hellaswag"]
    assert [row["selection_rank"] for row in rows] == [0, 0]
    assert {row["source_origin"] for row in rows} == {"pinned_hf_train_question"}
    manifest = selection.verify_expanded_act_selection(
        cross_format_audit=audit.path,
        fresh_iid_selection=result.fresh_iid_path,
        candidate_selection=result.candidate_path,
        selection_manifest=result.manifest_path,
        candidate_source_mode="fresh_only",
        n_total=2,
        **small_contract,
    )
    assert manifest["candidate_source_mode"] == "fresh_only"
    assert manifest["requested_balanced_n_total"] == 2
    assert manifest["question_candidates"]["counts_by_dataset"] == {"logiqa": 1, "hellaswag": 1}


def test_fresh_only_allows_prohibited_population_to_overlap_unused_legacy_training(
    small_contract: dict[str, Path | dict[str, Path]], tmp_path: Path
) -> None:
    prohibited = small_contract["prohibited_populations"]
    assert isinstance(prohibited, dict)
    legacy_rows = [
        json.loads(line)
        for line in Path(small_contract["legacy_training"]).read_text(encoding="utf-8").splitlines()
    ]
    _write_jsonl(prohibited["hle"], [legacy_rows[0]])
    audit = selection.materialize_cross_format_audit(output_dir=tmp_path / "audit", **small_contract)
    audit_document = json.loads(audit.path.read_text(encoding="utf-8"))
    assert audit_document["legacy_training_prohibited_overlap"]["logiqa"]["row_count"] == 1

    safe_mixed = selection.materialize_expanded_act_selection(
        cross_format_audit=audit.path,
        output_dir=tmp_path / "legacy-plus-fresh",
        **small_contract,
    )
    mixed_rows = [
        json.loads(line)
        for line in safe_mixed.candidate_path.read_text(encoding="utf-8").splitlines()
    ]
    assert legacy_rows[0]["question_id"] not in {row["source_question_id"] for row in mixed_rows}
    result = selection.materialize_expanded_act_selection(
        cross_format_audit=audit.path,
        output_dir=tmp_path / "fresh-only",
        candidate_source_mode="fresh_only",
        n_total=2,
        **small_contract,
    )
    rows = [json.loads(line) for line in result.candidate_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["source_origin"] == "pinned_hf_train_question" for row in rows)
