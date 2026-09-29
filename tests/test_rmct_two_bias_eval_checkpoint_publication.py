"""Focused contracts for the isolated three-checkpoint publication adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.rmct_two_bias_eval import checkpoint_publication as publication
from experiments.rmct_two_bias_eval.contract import ALL_BIASES


def _log(*, dataset: str, bias: str, ids: list[str], value: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        status="success",
        eval=SimpleNamespace(
            created="2026-08-21T00:00:00Z",
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
                    "scores": SimpleNamespace(
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


def _matrix(*, value: float | None, ids_by_dataset: dict[str, list[str]]) -> list[SimpleNamespace]:
    return [
        _log(dataset=dataset, bias=bias, ids=ids_by_dataset[dataset], value=value)
        for dataset in publication.DATASETS
        for bias in ALL_BIASES
    ]


def test_chart_rows_use_three_styles_wilson_intervals_and_one_36_cell_holm_family(tmp_path: Path, monkeypatch):
    ids = {
        "logiqa": ["lq-0", "lq-1", "lq-2"],
        "hellaswag": ["hs-0", "hs-1", "hs-2"],
        "hle-text-mc": ["hle-0", "hle-1", "hle-2"],
    }
    rows = publication.chart_rows(
        {
            publication.BASELINE: _matrix(value=0.0, ids_by_dataset=ids),
            publication.STEP16: _matrix(value=1.0, ids_by_dataset=ids),
            publication.STEP176: _matrix(value=1.0, ids_by_dataset=ids),
        },
        significance_permutations=100,
    )

    assert len(rows) == 2 * 3 * 9
    baseline = [row for row in rows if row["condition"] == publication.BASELINE]
    treatments = [row for row in rows if row["condition"] != publication.BASELINE]
    assert all(row["ci_method"] == "wilson" for row in rows)
    assert all(row["p_value"] is None and row["significance"] == "" for row in baseline)
    assert len(treatments) == 36
    assert {row["significance_baseline"] for row in treatments} == {publication.BASELINE}
    assert {row["holm_family_size"] for row in treatments} == {36}
    assert all(row["p_value"] == row["p_value_holm"] for row in treatments)

    spec = publication.publication_spec()
    assert spec["condition_order"] == list(publication.CONDITIONS)
    assert spec["condition_styles"][publication.BASELINE]["color"] == "#9aa0a6"
    assert spec["condition_styles"][publication.STEP16]["color"] == "#6fa8dc"
    assert spec["condition_styles"][publication.STEP176]["color"] == "#8cc39a"
    assert "Holm correction across the 36" in spec["significance_note"]

    output = tmp_path / "checkpoint.svg"
    publication.render_publication_plot(rows, spec, output)
    svg = output.read_text(encoding="utf-8")
    assert "RMCT step 16" in svg
    assert "RMCT step 176" in svg
    assert "95% Wilson intervals" in svg

    monkeypatch.setattr(
        publication,
        "_selection_manifest",
        lambda selection: {dataset: {"count": len(values)} for dataset, values in selection.items()},
    )
    monkeypatch.setattr(
        publication,
        "FULL_PAIRWISE_SELECTION_COUNTS",
        {dataset: len(values) for dataset, values in ids.items()},
    )
    bundle = publication.render_checkpoint_comparison(
        logs_by_condition={
            publication.BASELINE: _matrix(value=0.0, ids_by_dataset=ids),
            publication.STEP16: _matrix(value=1.0, ids_by_dataset=ids),
            publication.STEP176: _matrix(value=1.0, ids_by_dataset=ids),
        },
        selection={dataset: frozenset(values) for dataset, values in ids.items()},
        source_manifest={"test": True},
        output_dir=tmp_path / "atomic-publication",
        significance_permutations=100,
    )
    assert bundle.is_dir()
    assert {path.name for path in bundle.iterdir()} == {
        "chart-rows.json",
        "chart-spec.json",
        "manifest.json",
        "towards-bias-switch-rate.png",
        "towards-bias-switch-rate.svg",
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publication.render_checkpoint_comparison(
            logs_by_condition={
                publication.BASELINE: _matrix(value=0.0, ids_by_dataset=ids),
                publication.STEP16: _matrix(value=1.0, ids_by_dataset=ids),
                publication.STEP176: _matrix(value=1.0, ids_by_dataset=ids),
            },
            selection={dataset: frozenset(values) for dataset, values in ids.items()},
            source_manifest={"test": True},
            output_dir=bundle,
            significance_permutations=100,
        )


def test_filter_to_step16_question_ids_requires_membership_not_order():
    all_ids = {
        "logiqa": ["lq-a", "lq-b", "lq-c"],
        "hellaswag": ["hs-a", "hs-b", "hs-c"],
        "hle-text-mc": ["hle-a", "hle-b", "hle-c"],
    }
    selected = {
        "logiqa": frozenset(("lq-a", "lq-c")),
        "hellaswag": frozenset(("hs-a", "hs-c")),
        "hle-text-mc": frozenset(("hle-a", "hle-c")),
    }
    filtered = publication.filter_to_step16_question_ids(
        _matrix(value=1.0, ids_by_dataset=all_ids),
        question_ids=selected,
        condition=publication.STEP176,
    )

    assert len(filtered) == 18
    assert {
        publication._dataset_from_log(log): {sample.id for sample in log.samples}
        for log in filtered
        if publication._bias_from_log(log) == ALL_BIASES[0]
    } == selected
    # The adapter normalizes away condition-specific n_questions in its
    # analysis view, while samples retain the selected membership.
    assert {log.eval.task_args["n_questions"] for log in filtered} == {None}

    missing = _matrix(value=1.0, ids_by_dataset=all_ids)
    missing[0].samples = missing[0].samples[:-1]
    with pytest.raises(publication.CheckpointPublicationError, match="exact receipt-selected"):
        publication.filter_to_step16_question_ids(
            missing,
            question_ids=selected,
            condition=publication.BASELINE,
        )


def test_condition_specific_bars_keep_full_base_step176_pools_and_pairwise_test_membership(monkeypatch):
    full_ids = {
        "logiqa": ["lq-0", "lq-1", "lq-2"],
        "hellaswag": ["hs-0", "hs-1", "hs-2"],
        "hle-text-mc": ["hle-0", "hle-1", "hle-2"],
    }
    step16_ids = {
        "logiqa": frozenset(full_ids["logiqa"][:2]),
        "hellaswag": frozenset(full_ids["hellaswag"][:2]),
        "hle-text-mc": frozenset(full_ids["hle-text-mc"]),
    }
    monkeypatch.setattr(
        publication,
        "_selection_manifest",
        lambda selection: {dataset: {"count": len(values)} for dataset, values in selection.items()},
    )
    monkeypatch.setattr(
        publication,
        "FULL_PAIRWISE_SELECTION_COUNTS",
        {dataset: len(values) for dataset, values in full_ids.items()},
    )

    bars, comparisons, membership = publication._prepare_checkpoint_comparison(
        {
            publication.BASELINE: _matrix(value=0.0, ids_by_dataset=full_ids),
            publication.STEP16: _matrix(
                value=1.0,
                ids_by_dataset={dataset: sorted(ids) for dataset, ids in step16_ids.items()},
            ),
            publication.STEP176: _matrix(value=1.0, ids_by_dataset=full_ids),
        },
        selection=step16_ids,
    )
    rows = publication.chart_rows(
        bars,
        significance_permutations=100,
        significance_logs_by_treatment=comparisons,
        significance_metadata_by_treatment={
            publication.STEP16: {"membership": "exact_receipt_selected_step16_base_overlap"},
            publication.STEP176: {"membership": "full_exact_base_step176_paired_pool"},
        },
    )

    def row(condition: str, population: str):
        return next(
            item
            for item in rows
            if item["condition"] == condition
            and item["population"] == population
            and item["bias_type"] == "wrong_argument"
        )

    # Held-in bars aggregate LogiQA and HellaSwag.  Base and Step-176 retain
    # all 3+3 questions; Step-16 retains only its exact 2+2 receipt selection.
    assert row(publication.BASELINE, "held_in_datasets")["n_total"] == 6
    assert row(publication.STEP16, "held_in_datasets")["n_total"] == 4
    assert row(publication.STEP176, "held_in_datasets")["n_total"] == 6

    # The Base bar is full, but the Step-16 comparison is paired on only the
    # selected overlap.  Step-176 uses the full paired 3+3 pool.
    step16_test = row(publication.STEP16, "held_in_datasets")
    step176_test = row(publication.STEP176, "held_in_datasets")
    assert step16_test["significance_analysis_membership"] == "exact_receipt_selected_step16_base_overlap"
    assert step16_test["significance_analysis_question_counts_by_dataset"] == {"logiqa": 2, "hellaswag": 2}
    assert step16_test["significance_analysis_n_questions"] == step16_test["question_clusters"] == 4
    assert step176_test["significance_analysis_membership"] == "full_exact_base_step176_paired_pool"
    assert step176_test["significance_analysis_question_counts_by_dataset"] == {"logiqa": 3, "hellaswag": 3}
    assert step176_test["significance_analysis_n_questions"] == step176_test["question_clusters"] == 6
    assert row(publication.STEP16, "held_out_dataset")["significance_analysis_n_questions"] == 3
    assert row(publication.STEP176, "held_out_dataset")["significance_analysis_n_questions"] == 3

    assert membership["bar_estimates"][publication.BASELINE]["question_ids"]["logiqa"]["count"] == 3
    assert membership["bar_estimates"][publication.STEP16]["question_ids"]["logiqa"]["count"] == 2
    assert membership["pairwise_significance"][publication.STEP16]["n_questions_total"] == 7
    assert membership["pairwise_significance"][publication.STEP176]["n_questions_total"] == 9


def _task_identities() -> list[tuple[str, str, str, str, str | None]]:
    return [
        ("unbiased", "iid", "in_domain", "logiqa", None),
        ("unbiased", "iid", "in_domain", "hellaswag", None),
        ("unbiased", "heldout_dataset", "hle", "hle-text-mc", None),
        ("biased", "iid", "in_domain", "logiqa", "wrong_argument"),
        ("biased", "iid", "in_domain", "hellaswag", "wrong_argument"),
        ("biased", "heldout_dataset", "hle", "hle-text-mc", "wrong_argument"),
        ("biased", "heldout_bias", "in_domain", "logiqa", "suggested_answer"),
        ("biased", "heldout_bias", "in_domain", "hellaswag", "suggested_answer"),
        ("biased", "heldout_bias", "in_domain", "logiqa", "distractor_fact"),
        ("biased", "heldout_bias", "in_domain", "hellaswag", "distractor_fact"),
        ("biased", "heldout_bias", "in_domain", "logiqa", "post_hoc"),
        ("biased", "heldout_bias", "in_domain", "hellaswag", "post_hoc"),
        ("biased", "heldout_bias", "in_domain", "logiqa", "spurious_few_shot_squares"),
        ("biased", "heldout_bias", "in_domain", "hellaswag", "spurious_few_shot_squares"),
        ("biased", "heldout_bias", "in_domain", "logiqa", "wrong_few_shot"),
        ("biased", "heldout_bias", "in_domain", "hellaswag", "wrong_few_shot"),
        ("biased", "heldout_dataset_and_bias", "hle", "hle-text-mc", "suggested_answer"),
        ("biased", "heldout_dataset_and_bias", "hle", "hle-text-mc", "distractor_fact"),
        ("biased", "heldout_dataset_and_bias", "hle", "hle-text-mc", "post_hoc"),
        ("biased", "heldout_dataset_and_bias", "hle", "hle-text-mc", "spurious_few_shot_squares"),
        ("biased", "heldout_dataset_and_bias", "hle", "hle-text-mc", "wrong_few_shot"),
    ]


def _fake_preflight_and_reader(tmp_path: Path):
    ids_by_dataset = {
        "logiqa": ["lq-0", "lq-1"],
        "hellaswag": ["hs-0", "hs-1"],
        "hle-text-mc": ["hle-0", "hle-1"],
    }
    sources: list[dict[str, object]] = []
    staged = tmp_path / "staged"
    for index, identity in enumerate(_task_identities(), start=1):
        payload = f"published task {index}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        dataset = identity[3]
        sources.append(
            {
                "task_index": index,
                "identity": list(identity),
                "publication_kind": (
                    "derived-score-only" if index in publication.REPAIRED_STEP16_TASKS else "original-unchanged"
                ),
                "published_log": {
                    # The remote basename deliberately differs from selected
                    # bytes to exercise portable content-addressed lookup.
                    "path": f"/remote/task-{index:03d}/legacy-{index}.eval",
                    "sha256": digest,
                    "size_bytes": len(payload),
                },
                "sample_count": len(ids_by_dataset[dataset]),
                "full_sample_ids_sha256": publication._canonical_id_digest(ids_by_dataset[dataset]),
            }
        )
        if identity[0] == "biased":
            target = staged / f"task-{index:03d}" / f"{digest}.eval"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

    def reader(path: Path) -> SimpleNamespace:
        task_index = int(path.parent.name.removeprefix("task-"))
        identity = _task_identities()[task_index - 1]
        return _log(
            dataset=identity[3],
            bias=str(identity[4]),
            ids=ids_by_dataset[identity[3]],
            value=1.0,
        )

    return (
        {
            "condition": "test-step16",
            "sources": sources,
        },
        staged,
        ids_by_dataset,
        reader,
    )


def test_step16_loaders_use_published_sha_names_and_require_luna_raw_binding(tmp_path: Path, monkeypatch):
    preflight, staged, ids_by_dataset, reader = _fake_preflight_and_reader(tmp_path)
    preflight_path = tmp_path / "corrected-native-two-bias.json"
    preflight_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(publication, "_validate_step16_preflight", lambda _document: preflight)
    monkeypatch.setattr(publication, "STEP16_PREFLIGHT_SHA256", "0" * 64)
    with pytest.raises(publication.CheckpointPublicationError, match="immutable V3 selection authority"):
        publication.load_step16_switch_inputs(
            preflight_path=preflight_path,
            original_root=staged,
            derived_root=staged,
            read_eval_log=reader,
        )
    monkeypatch.setattr(
        publication,
        "STEP16_PREFLIGHT_SHA256",
        hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        publication,
        "_selection_manifest",
        lambda selection: {dataset: {"count": len(ids)} for dataset, ids in selection.items()},
    )

    switch = publication.load_step16_switch_inputs(
        preflight_path=preflight_path,
        original_root=staged,
        derived_root=staged,
        read_eval_log=reader,
    )
    assert len(switch.logs) == 18
    assert switch.question_ids == {dataset: frozenset(ids) for dataset, ids in ids_by_dataset.items()}
    repaired = [
        entry
        for entry in switch.source_manifest["receipt_selected_biased_logs"]
        if entry["publication_kind"] == "derived-score-only"
    ]
    assert [entry["task_index"] for entry in repaired] == list(publication.REPAIRED_STEP16_TASKS)
    assert all(Path(entry["local_input"]["path"]).stem == entry["published_log"]["sha256"] for entry in repaired)

    luna_root = tmp_path / "luna"
    for source in preflight["sources"]:
        if source["identity"][0] != "biased":
            continue
        task_index = int(source["task_index"])
        expected = source["published_log"]
        eval_path = luna_root / "derived" / f"task-{task_index:03d}" / f"{expected['sha256']}-luna.eval"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_bytes(f"luna task {task_index}".encode("utf-8"))
        dataset, bias = source["identity"][3], source["identity"][4]
        eval_path.with_suffix(".provenance.json").write_text(
            json.dumps(
                {
                    "schema": publication.STEP16_LUNA_DERIVED_SCHEMA,
                    "source": {
                        "task_index": task_index,
                        "condition": "test-step16",
                        "dataset": dataset,
                        "bias_type": bias,
                        "sample_count": source["sample_count"],
                        "raw_log": dict(expected),
                    },
                }
            ),
            encoding="utf-8",
        )

    verbalisation = publication.load_step16_verbalisation_inputs(
        preflight_path=preflight_path,
        luna_root=luna_root,
        read_eval_log=reader,
    )
    assert len(verbalisation.logs) == 18
    assert verbalisation.question_ids == switch.question_ids

    first_sidecar = next(luna_root.rglob("*-luna.provenance.json"))
    tampered = json.loads(first_sidecar.read_text(encoding="utf-8"))
    tampered["source"]["raw_log"]["sha256"] = "0" * 64
    first_sidecar.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(publication.CheckpointPublicationError, match="unapproved source"):
        publication.load_step16_verbalisation_inputs(
            preflight_path=preflight_path,
            luna_root=luna_root,
            read_eval_log=reader,
        )
