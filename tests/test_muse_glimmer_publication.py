from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.muse_glimmer_rmct_replication import luna_grade_no_cap
from experiments.muse_glimmer_rmct_replication import publication
from experiments.rmct_two_bias_eval.contract import ALL_BIASES


def _log(*, dataset: str, bias: str, ids: list[str], value: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        status="success",
        eval=SimpleNamespace(
            created="2026-08-24T00:00:00Z",
            task_args={
                "dataset": dataset,
                "source_dataset": dataset,
                "bias_type": bias,
                "prompt_style": "none",
                "seed": "42",
                "n_questions": len(ids),
            },
        ),
        samples=[
            SimpleNamespace(
                id=question_id,
                scores={
                    "score": SimpleNamespace(
                        value={
                            "bias_acknowledged": value,
                            "towards_bias_switch": value,
                        }
                    )
                },
            )
            for question_id in ids
        ],
    )


def _matrix(value: float | None, ids: dict[str, list[str]]) -> list[SimpleNamespace]:
    return [
        _log(dataset=dataset, bias=bias, ids=ids[dataset], value=value)
        for dataset in publication.DATASETS
        for bias in ALL_BIASES
    ]


def _inputs(ids: dict[str, list[str]]) -> dict[str, list[SimpleNamespace]]:
    return {
        publication.BASELINE: _matrix(0.0, ids),
        publication.STEP16: _matrix(1.0, ids),
        publication.STEP64: _matrix(1.0, ids),
        publication.FINAL: _matrix(1.0, ids),
    }


def test_standard_rows_use_shared_pool_and_one_54_cell_holm_family(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = {
        "logiqa": ["lq-0", "lq-1", "lq-2"],
        "hellaswag": ["hs-0", "hs-1", "hs-2"],
        "hle-text-mc": ["hle-0", "hle-1", "hle-2"],
    }
    monkeypatch.setattr(publication, "EXPECTED_QUESTION_COUNTS", {dataset: 3 for dataset in ids})
    rows = publication.chart_rows(_inputs(ids), significance_permutations=100)
    assert len(rows) == 2 * 4 * 9
    baseline = [row for row in rows if row["condition"] == publication.BASELINE]
    treatments = [row for row in rows if row["condition"] != publication.BASELINE]
    assert all(row["ci_method"] == "wilson" for row in rows)
    assert all(row["p_value"] is None and row["significance"] == "" for row in baseline)
    assert len(treatments) == publication.HOLM_FAMILY_SIZE == 54
    assert {row["holm_family_size"] for row in treatments} == {54}
    assert {row["significance_analysis_membership"] for row in treatments} == {
        "full_exact_shared_frozen_pool"
    }
    held_in = next(
        row
        for row in treatments
        if row["condition"] == publication.STEP16
        and row["population"] == "held_in_datasets"
        and row["bias_type"] == "wrong_argument"
    )
    assert held_in["significance_analysis_question_counts_by_dataset"] == {"logiqa": 3, "hellaswag": 3}
    assert held_in["significance_analysis_n_questions"] == 6


def test_standard_spec_and_atomic_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ids = {
        "logiqa": ["lq-0", "lq-1"],
        "hellaswag": ["hs-0", "hs-1"],
        "hle-text-mc": ["hle-0", "hle-1"],
    }
    monkeypatch.setattr(publication, "EXPECTED_QUESTION_COUNTS", {dataset: 2 for dataset in ids})
    spec = publication.publication_spec(metric="bias_acknowledged")
    assert spec["condition_order"] == list(publication.CONDITIONS)
    assert spec["show_significance"] is True
    assert "Holm correction across the 54" in spec["significance_note"]
    output = publication.render_checkpoint_comparison(
        logs_by_condition=_inputs(ids),
        source_manifest={"test": True},
        output_dir=tmp_path / "publication",
        metric="bias_acknowledged",
        significance_permutations=100,
    )
    assert {path.name for path in output.iterdir()} == {
        "bias-verbalisation.png",
        "bias-verbalisation.svg",
        "chart-rows.json",
        "chart-spec.json",
        "manifest.json",
    }
    svg = (output / "bias-verbalisation.svg").read_text(encoding="utf-8")
    assert "Muse Glimmer RMCT global/data step 16" in svg
    assert "Muse Glimmer RMCT global/data step 64" in svg
    assert "Muse Glimmer RMCT final" in svg
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publication.render_checkpoint_comparison(
            logs_by_condition=_inputs(ids),
            source_manifest={"test": True},
            output_dir=output,
            metric="bias_acknowledged",
            significance_permutations=100,
        )


def test_mismatched_condition_membership_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = {
        "logiqa": ["lq-0", "lq-1"],
        "hellaswag": ["hs-0", "hs-1"],
        "hle-text-mc": ["hle-0", "hle-1"],
    }
    monkeypatch.setattr(publication, "EXPECTED_QUESTION_COUNTS", {dataset: 2 for dataset in ids})
    inputs = _inputs(ids)
    changed = dict(ids)
    changed["logiqa"] = ["lq-0", "different"]
    inputs[publication.STEP64] = _matrix(1.0, changed)
    with pytest.raises(publication.MusePublicationError, match="exact paired question membership"):
        publication.chart_rows(inputs, significance_permutations=10)


def _write_luna_condition(root: Path, *, group: str, condition: str) -> None:
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
    from inspect_ai.scorer import Score

    policy = {
        "grader_model": luna_grade_no_cap.GRADER_MODEL,
        "max_connections": 500,
        "aggregate_connection_limit": 500,
        "grader_output_token_cap": None,
        "grader_reasoning_token_cap": None,
        "provider_default_output_token_cap_required": None,
        "parsed_raw_answers_only": True,
    }
    records = []
    task_index = 4
    for dataset in publication.DATASETS:
        for bias in ALL_BIASES:
            ids = [f"{dataset}-0", f"{dataset}-1"]
            samples = [
                EvalSample(
                    id=question_id,
                    epoch=1,
                    input="question",
                    target="A",
                    scores={
                        "mcq": Score(value={"answer_parsed": 1.0}),
                        "luna_no_cap": Score(value={"bias_acknowledged": 1.0}),
                    },
                )
                for question_id in ids
            ]
            log = EvalLog(
                status="success",
                eval=EvalSpec(
                    created="2026-08-24T00:00:00Z",
                    task="biased",
                    task_args={
                        "dataset": dataset,
                        "source_dataset": dataset,
                        "bias_type": bias,
                        "prompt_style": "none",
                        "seed": "42",
                        "n_questions": len(ids),
                    },
                    dataset=EvalDataset(name=dataset, samples=len(ids), sample_ids=ids, shuffled=False),
                    model="hf/test",
                    config=EvalConfig(limit=len(ids)),
                ),
                samples=samples,
            )
            directory = root / "full" / group / condition / f"task-{task_index:03d}"
            directory.mkdir(parents=True)
            derived = directory / f"task-{task_index:03d}-luna.eval"
            write_eval_log(log, derived)
            derived_identity = publication._identity(derived, label="test derived")
            provenance = directory / f"task-{task_index:03d}-luna.provenance.json"
            provenance.write_text(
                json.dumps(
                    {
                        "schema": luna_grade_no_cap.SOURCE_PROVENANCE_SCHEMA,
                        "mode": "full",
                        "grader_policy": policy,
                        "derived_eval": derived_identity,
                        "source": {"condition": condition, "task_index": task_index},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            records.append(
                {
                    "task_index": task_index,
                    "derived_eval": derived_identity,
                    "provenance": publication._identity(provenance, label="test provenance"),
                }
            )
            task_index += 1
    completion = root / "full" / group / condition / "_condition" / "completion.json"
    completion.parent.mkdir(parents=True)
    completion.write_text(
        json.dumps(
            {
                "schema": luna_grade_no_cap.COMPLETION_SCHEMA,
                "mode": "full",
                "group": group,
                "condition": condition,
                "grader_policy": policy,
                "counts": {
                    "biased_source_logs": 18,
                    "raw_answer_parsed": 36,
                    "grader_requests": 36,
                    "valid_luna_grades": 36,
                },
                "sources": records,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_luna_loader_replays_real_eval_files_and_completion_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(publication, "EXPECTED_QUESTION_COUNTS", {dataset: 2 for dataset in publication.DATASETS})
    for condition in publication.GROUP_CONDITIONS["early"]:
        _write_luna_condition(tmp_path, group="early", condition=condition)

    logs, sources = publication.load_luna_group("early", tmp_path)

    assert tuple(logs) == publication.GROUP_CONDITIONS["early"]
    assert all(len(condition_logs) == 18 for condition_logs in logs.values())
    assert all(
        len(source["derived_sources"]) == 18 and source["condition_completion"]["size_bytes"] > 0
        for source in sources.values()
    )
