"""Offline tests for the thin EvalAwareBench training adapter."""

from __future__ import annotations

import asyncio
import json

import pytest

from ctm_data.adapters.eval_awareness.data import (
    DATASET_ID,
    DATASET_LICENSE,
    build_prompt_families,
    materialize_eval_awareness,
)

TEST_REVISION = "a" * 40


def _row(task: str, valence: str, factors: tuple[str, ...]) -> dict:
    label = "+".join(factors) if factors else "baseline"
    return {
        "task_id": task,
        "task_name": task,
        "valence": valence,
        "description": f"description-{task}-{valence}",
        "factors_varied": list(factors) if factors else ["none (baseline)"],
        "num_factors_varied": len(factors),
        "config": {factor: True for factor in factors},
        "prompt": f"prompt-{task}-{valence}-{label}",
    }


def _rows(tasks=("task-a", "task-b")) -> list[dict]:
    rows = []
    for task in tasks:
        for valence in ("safety", "capability"):
            rows.extend(
                [
                    _row(task, valence, ()),
                    _row(task, valence, ("formal_tone",)),
                    _row(task, valence, ("benchmark_framing",)),
                ]
            )
    return rows


def test_family_builder_uses_exactly_the_supplied_rows():
    selected = _rows(("task-a",))

    families = build_prompt_families(selected, n_variants=2, seed="unit")

    assert {family["source_id"] for family in families} == {"task-a:safety", "task-a:capability"}
    assert all(len(family["variants"]) == 2 for family in families)
    assert all("task-b" not in family["reference_messages"][0]["content"] for family in families)


def test_fixed_k_is_deterministic_but_not_a_train_eval_split():
    rows = _rows(("task-a",))
    assert build_prompt_families(rows, n_variants=1, seed="unit") == build_prompt_families(
        list(reversed(rows)), n_variants=1, seed="unit"
    )


def test_official_f6_artifact_is_direction_neutral():
    rows = [
        _row("task-a", "safety", ()),
        _row("task-a", "safety", ("F6",)),
        _row("task-a", "safety", ("F8",)),
    ]

    family = build_prompt_families(rows, n_variants=1, factors=["F6"])[0]

    assert family["reference_messages"][0]["content"].endswith("baseline")
    assert family["variants"][0]["messages"][0]["content"].endswith("F6")
    assert family["variants"][0]["side"] == "F6"
    assert family["metadata"]["factor_filter"] == ["F6"]
    assert family["metadata"]["available_sides"] == ["baseline", "F6"]


def test_exact_factor_selection_is_strict():
    with pytest.raises(ValueError, match=r"exact factors \['F6'\]"):
        build_prompt_families(_rows(("task-a",)), n_variants=1, factors=["F6"])


def test_family_builder_requires_one_baseline_and_enough_variants():
    without_baseline = [row for row in _rows(("task-a",)) if row["num_factors_varied"]]
    with pytest.raises(ValueError, match="baseline rows"):
        build_prompt_families(without_baseline, n_variants=1)
    with pytest.raises(ValueError, match="needs 3"):
        build_prompt_families(_rows(("task-a",)), n_variants=3)


def test_materialization_records_source_provenance(tmp_path):
    path = tmp_path / "train.jsonl"
    manifest = materialize_eval_awareness(
        _rows(),
        path,
        n_variants=1,
        source_revision=TEST_REVISION,
    )
    assert manifest["artifact_schema"] == "ctm.prompt_families"
    assert manifest["provenance"] == {
        "dataset_id": DATASET_ID,
        "dataset_config": "prompts",
        "source_revision": TEST_REVISION,
        "source_license": DATASET_LICENSE,
        "source_row_count": len(_rows()),
        "source_rows_sha256": manifest["provenance"]["source_rows_sha256"],
        "n_variants": 1,
        "seed": "42",
        "factor_filter": None,
    }
    assert len(manifest["provenance"]["source_rows_sha256"]) == 64


def test_materializer_rejects_mutable_revision_and_overwrite(tmp_path):
    with pytest.raises(ValueError, match="full 40-hex"):
        materialize_eval_awareness(
            _rows(),
            tmp_path / "mutable.jsonl",
            n_variants=1,
            source_revision="main",
        )
    path = tmp_path / "frozen.jsonl"
    materialize_eval_awareness(
        _rows(),
        path,
        n_variants=1,
        source_revision=TEST_REVISION,
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_eval_awareness(
            _rows(),
            path,
            n_variants=1,
            source_revision=TEST_REVISION,
        )


def test_build_cli_only_builds_training_families(tmp_path):
    from ctm_data.adapters.eval_awareness.builder import main

    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in _rows()))
    output = tmp_path / "train.jsonl"
    assert (
        main(
            [
                "--input-jsonl",
                str(source),
                "--output",
                str(output),
                "--n-variants",
                "1",
                "--source-revision",
                TEST_REVISION,
                "--factors",
                "benchmark_framing",
            ]
        )
        == 0
    )
    assert output.exists()


def test_setting_uses_explicit_sides_from_training_artifact(tmp_path, monkeypatch):
    from ctm_data.adapters.eval_awareness import setting as setting_module

    class FakeJudge:
        def __init__(self, model=None, **_):
            self.model = model

        async def __call__(self, response, datapoint, realized_messages):
            return float("refuse" in response)

    monkeypatch.setattr(setting_module, "RefusalJudge", FakeJudge)
    path = tmp_path / "train.jsonl"
    materialize_eval_awareness(
        _rows(),
        path,
        n_variants=1,
        factors=["benchmark_framing"],
        source_revision=TEST_REVISION,
    )
    setting = setting_module.EvalAwarenessSetting(
        data_path=path,
        n_variants=1,
        reference_side="baseline",
        train_side="benchmark_framing",
        grader_model="grader",
    )
    datapoints = setting.load_datapoints(n_datapoints=1)
    assert len(datapoints) == 1
    assert len(setting.perturbations()) == 2
    assert setting.training_perturbation_indices() == [1]
    assert setting.answer_parser() is None
    realized = setting.perturbations()[1](datapoints[0])["messages"]
    assert asyncio.run(setting.trait_classifier()("I refuse", datapoints[0], realized)) == 1.0


def test_setting_can_reverse_a_single_neutral_pair(tmp_path, monkeypatch):
    from ctm_data.adapters.eval_awareness import setting as setting_module

    monkeypatch.setattr(setting_module, "RefusalJudge", lambda model=None, **_: lambda *_: 0.0)
    path = tmp_path / "train.jsonl"
    materialize_eval_awareness(
        _rows(),
        path,
        n_variants=1,
        factors=["benchmark_framing"],
        source_revision=TEST_REVISION,
    )
    setting = setting_module.EvalAwarenessSetting(
        data_path=path,
        n_variants=1,
        reference_side="benchmark_framing",
        train_side="baseline",
    )

    datapoint = setting.load_datapoints(n_datapoints=1)[0]
    perturbations = setting.perturbations()

    assert perturbations[0](datapoint)["messages"][0]["content"].endswith("benchmark_framing")
    assert perturbations[1](datapoint)["messages"][0]["content"].endswith("baseline")
