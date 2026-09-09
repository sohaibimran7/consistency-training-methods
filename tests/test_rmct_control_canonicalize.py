"""Focused offline tests for accelerated RMCT-control canonicalization."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from experiments.stage1_iid_diagnostic import analyze
from experiments.stage1_iid_diagnostic_none import canonicalize_rmct_control as canonicalize


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_frozen_splits(tmp_path: Path) -> tuple[Path, dict, dict[tuple[str, str], tuple[str, ...]]]:
    """Make a small manifest-shaped frozen suite with real 100+100 cells."""

    expected: dict[tuple[str, str], tuple[str, ...]] = {}
    split_entries: dict[str, dict] = {}
    first64: list[str] = []
    for split in analyze.SPLITS:
        rows: list[dict] = []
        for index in range(100):
            for dataset in analyze.DATASETS:
                question_id = f"{split}-{dataset}-{index:03d}"
                rows.append({"question_id": question_id, "source_dataset": dataset})
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        ids = [row["question_id"] for row in rows]
        split_entries[split] = {"path": str(path), "question_ids": ids}
        for dataset in analyze.DATASETS:
            expected[(split, dataset)] = tuple(
                row["question_id"] for row in rows if row["source_dataset"] == dataset
            )
        if split == "train_eval":
            first64 = ids[:64]
    manifest = {
        "splits": split_entries,
        "rmct_first64": {
            "question_ids": first64,
            "question_ids_sha256": hashlib.sha256(
                "".join(f"{question_id}\n" for question_id in first64).encode()
            ).hexdigest(),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path, manifest, expected


def _observations(expected: dict[tuple[str, str], tuple[str, ...]]) -> list[analyze.Observation]:
    rows: list[analyze.Observation] = []
    for split, dataset in sorted(expected):
        for index, question_id in enumerate(expected[(split, dataset)]):
            clean_matches_bias = 0 if index % 3 else 1
            toward = int(clean_matches_bias == 0 and index % 5 == 0)
            away = int(clean_matches_bias == 1 and index % 7 == 0)
            rows.append(
                analyze.Observation(
                    condition=canonicalize.ALIAS_CONDITION,
                    split=split,
                    dataset=dataset,
                    question_id=question_id,
                    joint_parse=True,
                    clean_matches_bias=clean_matches_bias,
                    toward=toward,
                    away=away,
                    total_switch=toward + away,
                    luna=index % 2,
                    generation_cap_hit=False,
                    grader_cap_hit=False,
                )
            )
    return rows


def _sources(tmp_path: Path) -> list[dict]:
    result: list[dict] = []
    for split, dataset in sorted(canonicalize._EXPECTED_CELLS):
        eval_path = tmp_path / "graded" / canonicalize.ALIAS_CONDITION / split / f"{dataset}.eval"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_bytes(f"immutable {split} {dataset}".encode())
        result.append(
            {
                "condition": canonicalize.ALIAS_CONDITION,
                "split": split,
                "dataset": dataset,
                "graded_log": str(eval_path),
                "graded_log_sha256": _sha256(eval_path),
                "samples": 100,
                "source_log": f"/raw/{split}/{dataset}.eval",
                "source_sha256": "a" * 64,
            }
        )
    return result


def _alias_analysis(manifest_path: Path, manifest: dict, rows: list[analyze.Observation], sources: list[dict]) -> dict:
    cells = analyze.grouped_report(rows, set(manifest["rmct_first64"]["question_ids"]))
    return {
        "schema": analyze.ANALYSIS_SCHEMA,
        "grader_model": analyze.DEFAULT_LUNA_GRADER_MODEL,
        "grader_max_tokens": 4096,
        "inspect_rescore_model": analyze.INSPECT_RESCORE_MODEL,
        "diagnostic_manifest": str(manifest_path),
        "diagnostic_manifest_sha256": _sha256(manifest_path),
        "metric_definitions": {
            "tbsr": "P(biased answer = bias answer | clean answer != bias answer, jointly parsed)",
            "away_from_bias": "P(biased answer != bias answer | clean answer = bias answer, jointly parsed)",
            "total_switch": "P(biased answer differs from clean answer | jointly parsed)",
            "luna_yes": "P(Luna YES | Luna verdict parsed)",
        },
        "sources": sources,
        "cells": cells,
    }


@pytest.fixture
def canonical_inputs(tmp_path: Path, monkeypatch):
    manifest_path, manifest, expected = _write_frozen_splits(tmp_path)
    rows = _observations(expected)
    sources = _sources(tmp_path)
    alias = _alias_analysis(manifest_path, manifest, rows, sources)
    alias_path = tmp_path / "alias-analysis.json"
    alias_path.write_text(json.dumps(alias, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # The production validator is covered in the no-CoT preparation tests.
    # Here it returns the same frozen object so this test focuses on the alias
    # report/EvalLog derivation boundary, including real split-file ID reads.
    monkeypatch.setattr(canonicalize, "_validate_frozen_manifest", lambda path: (manifest, 64))
    calls: list[tuple[Path, int]] = []

    def fake_load(graded_root, *, grader_max_tokens):
        calls.append((Path(graded_root), grader_max_tokens))
        return rows, sources

    monkeypatch.setattr(canonicalize.analyze, "load_graded_logs", fake_load)
    return {
        "alias": alias,
        "alias_path": alias_path,
        "graded_root": tmp_path / "graded",
        "manifest_path": manifest_path,
        "manifest": manifest,
        "rows": rows,
        "sources": sources,
        "calls": calls,
    }


def test_canonicalization_recomputes_cells_first64_and_preserves_immutable_inputs(canonical_inputs, tmp_path):
    inputs = canonical_inputs
    alias_before = inputs["alias_path"].read_bytes()
    eval_before = {source["graded_log"]: Path(source["graded_log"]).read_bytes() for source in inputs["sources"]}
    output = tmp_path / "analysis" / "rmct-control-final.json"
    provenance = tmp_path / "analysis" / "rmct-control-canonicalization.json"

    assert canonicalize.canonicalize(
        inputs["alias_path"],
        inputs["graded_root"],
        inputs["manifest_path"],
        output=output,
        provenance_output=provenance,
    ) == ("written", "written")
    assert canonicalize.canonicalize(
        inputs["alias_path"],
        inputs["graded_root"],
        inputs["manifest_path"],
        output=output,
        provenance_output=provenance,
    ) == ("resumed", "resumed")

    report = json.loads(output.read_text())
    proof = json.loads(provenance.read_text())
    assert set(report["cells"]) == {
        "rmct-control/train_eval",
        "rmct-control/heldout_in_domain",
    }
    assert report["cells"]["rmct-control/train_eval"]["rmct_first64"]["pooled"]["counts"]["samples"] == 64
    assert all(source["condition"] == canonicalize.CANONICAL_CONDITION for source in report["sources"])
    assert all(source["source_condition"] == canonicalize.ALIAS_CONDITION for source in report["sources"])
    assert proof["checks"] == {
        "frozen_manifest_validated": True,
        "graded_eval_log_ids_exactly_match_manifest": True,
        "alias_analysis_sources_exactly_match_graded_eval_logs": True,
        "pooled_and_per_dataset_cells_equivalent_except_condition_label": True,
        "raw_or_graded_logs_modified": False,
    }
    assert inputs["calls"] == [(inputs["graded_root"].resolve(), 4096), (inputs["graded_root"].resolve(), 4096)]
    assert inputs["alias_path"].read_bytes() == alias_before
    assert {path: Path(path).read_bytes() for path in eval_before} == eval_before


def test_canonicalization_accepts_manifest_equivalent_eval_log_order(canonical_inputs, tmp_path, monkeypatch):
    """Inspect may persist samples in a different order from frozen JSONL."""

    inputs = canonical_inputs
    reordered_rows = list(reversed(inputs["rows"]))
    monkeypatch.setattr(
        canonicalize.analyze,
        "load_graded_logs",
        lambda *_args, **_kwargs: (reordered_rows, inputs["sources"]),
    )

    report, _ = canonicalize.build_canonical_report(
        inputs["alias_path"], inputs["graded_root"], inputs["manifest_path"]
    )

    assert report["cells"]["rmct-control/train_eval"]["rmct_first64"]["pooled"]["counts"]["samples"] == 64


def test_canonicalization_rejects_alias_cell_that_cannot_be_reproduced(canonical_inputs, tmp_path):
    inputs = canonical_inputs
    altered = json.loads(inputs["alias_path"].read_text())
    altered["cells"][f"{canonicalize.ALIAS_CONDITION}/train_eval"]["pooled"]["counts"]["samples"] += 1
    inputs["alias_path"].write_text(json.dumps(altered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "analysis" / "rmct-control-final.json"
    provenance = tmp_path / "analysis" / "rmct-control-canonicalization.json"

    with pytest.raises(ValueError, match="not exactly reproducible"):
        canonicalize.canonicalize(
            inputs["alias_path"],
            inputs["graded_root"],
            inputs["manifest_path"],
            output=output,
            provenance_output=provenance,
        )

    assert not output.exists()
    assert not provenance.exists()


def test_canonicalization_rejects_log_ids_outside_frozen_cell(canonical_inputs, tmp_path, monkeypatch):
    inputs = canonical_inputs
    changed_rows = list(inputs["rows"])
    changed_rows[0] = dataclasses.replace(changed_rows[0], question_id="not-in-frozen-manifest")
    monkeypatch.setattr(
        canonicalize.analyze,
        "load_graded_logs",
        lambda *_args, **_kwargs: (changed_rows, inputs["sources"]),
    )

    with pytest.raises(ValueError, match="IDs do not exactly match the frozen manifest"):
        canonicalize.build_canonical_report(
            inputs["alias_path"], inputs["graded_root"], inputs["manifest_path"]
        )


def test_canonicalization_refuses_to_publish_inside_graded_log_root(canonical_inputs, tmp_path):
    inputs = canonical_inputs
    output = inputs["graded_root"] / "rmct-control-final.json"

    with pytest.raises(ValueError, match="outside the immutable graded-log root"):
        canonicalize.canonicalize(
            inputs["alias_path"],
            inputs["graded_root"],
            inputs["manifest_path"],
            output=output,
            provenance_output=tmp_path / "analysis" / "rmct-control-canonicalization.json",
        )

    assert not output.exists()
