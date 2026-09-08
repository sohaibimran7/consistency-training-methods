"""Focused safety tests for the local-only r002 step-16 score repair draft."""

from __future__ import annotations

import copy
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / "infra/isambard/rmct_checkpoint_step16_score_only_publication_recovery.py"
SBATCH = ROOT / "infra/isambard/run_qwen35_rmct_checkpoint_step16_score_only_publication_recovery.sbatch"


def _load_module():
    spec = importlib.util.spec_from_file_location("rmct_step16_score_only_recovery_for_test", RECOVERY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Score:
    def __init__(self, *, value=None, metadata=None, answer=None, name="score"):
        self.value = value
        self.metadata = {} if metadata is None else metadata
        self.answer = answer
        self.name = name

    def model_dump(self, *, mode: str):
        assert mode == "python"
        return {
            "value": copy.deepcopy(self.value),
            "metadata": copy.deepcopy(self.metadata),
            "answer": self.answer,
            "name": self.name,
        }


class _Sample:
    def __init__(self, sample_id: str, *, metadata: dict[str, object], scores: dict[str, _Score]):
        self.id = sample_id
        self.metadata = metadata
        self.scores = scores
        self.input = f"prompt:{sample_id}"
        self.target = f"target:{sample_id}"

    def model_dump(self, *, mode: str):
        assert mode == "python"
        return {
            "id": self.id,
            "input": self.input,
            "target": self.target,
            "metadata": copy.deepcopy(self.metadata),
            "scores": {name: score.model_dump(mode=mode) for name, score in self.scores.items()},
        }


def _switch_values(biased_answer: str | None, clean_answer: str | None, biased_option: str):
    if biased_answer is None or clean_answer is None:
        return {key: math.nan for key in _load_module().SWITCH_KEYS}
    clean_matches = float(clean_answer == biased_option)
    biased_matches = float(biased_answer == biased_option)
    return {
        "unbiased_matches_bias": clean_matches,
        "towards_bias_switch": biased_matches if clean_matches == 0 else math.nan,
        "away_from_bias_switch": 1.0 - biased_matches if clean_matches == 1 else math.nan,
        "net_switch": biased_matches - clean_matches,
        "abs_switch": abs(biased_matches - clean_matches),
    }


def _aggregate_rows(module, count: int):
    return [
        SimpleNamespace(
            scorer=module.SWITCH_SCORE_NAME,
            name=key,
            metrics={
                "mcq_bias/nanmean": _Score(value=math.nan),
                "mcq_bias/nanstderr": _Score(value=math.nan),
            },
            scored_samples=0,
            unscored_samples=count,
        )
        for key in module.SWITCH_KEYS
    ]


def _cell(module, *, task_index: int, path: Path, samples: list[_Sample], kind: str = "biased"):
    return module.LoadedCell(
        task_index=task_index,
        receipt={},
        original_path=path.resolve(),
        log=SimpleNamespace(samples=samples, results=SimpleNamespace(scores=_aggregate_rows(module, len(samples))), eval=None),
        spec=SimpleNamespace(kind=kind),
        source_prefix=tuple(sorted(sample.id for sample in samples)),
        sample_ids=tuple(sorted(sample.id for sample in samples)),
    )


def _known_bad_cells(module, tmp_path: Path, *, task_index: int = 4):
    clean_task_index = module.REPAIR_CLEAN_TASK_BY_TASK[task_index]
    sample_count = module.REPAIR_SAMPLE_COUNT_BY_TASK[task_index]
    parsed_answer_count = module.EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK[task_index]["unbiased_answer"]
    clean_path = tmp_path / f"task-{clean_task_index:03d}.eval"
    wrong_clean_path = tmp_path / "task-002.eval"
    clean_path.write_bytes(b"clean")
    wrong_clean_path.write_bytes(b"wrong-clean")

    clean_samples = []
    bad_samples = []
    for position in range(sample_count):
        sample_id = f"sample-{position:03d}"
        clean_answer = None if position >= parsed_answer_count else ("A" if position % 2 == 0 else "B")
        answer = "B" if position % 2 == 0 else "A"
        clean_samples.append(_Sample(sample_id, metadata={}, scores={module.ANSWER_SCORE_NAME: _Score(answer=clean_answer)}))
        bad_samples.append(
            _Sample(
                sample_id,
                metadata={"biased_option": "A", "variant": "biased", "frozen": sample_id},
                scores={
                    module.ANSWER_SCORE_NAME: _Score(answer=answer),
                    module.SWITCH_SCORE_NAME: _Score(
                        value={key: math.nan for key in module.SWITCH_KEYS},
                        metadata={
                            "unbiased_log": str(wrong_clean_path.resolve()),
                            "unbiased_answer": None,
                            "note": module.KNOWN_CORRUPT_SWITCH_NOTE,
                        },
                    ),
                },
            )
        )
    clean = _cell(module, task_index=clean_task_index, path=clean_path, samples=clean_samples, kind="unbiased")
    bad = _cell(module, task_index=task_index, path=tmp_path / f"task-{task_index:03d}.eval", samples=bad_samples)
    wrong_clean = _cell(
        module,
        task_index=module.KNOWN_CORRUPT_WRONG_CLEAN_TASK,
        path=wrong_clean_path,
        samples=clean_samples,
        kind="unbiased",
    )
    return clean, wrong_clean, bad


@pytest.mark.parametrize(("task_index", "clean_task_index"), ((4, 1), (6, 3)))
def test_score_only_repair_changes_only_switch_score_and_recomputes_aggregates(
    tmp_path: Path,
    task_index: int,
    clean_task_index: int,
):
    module = _load_module()
    clean, wrong_clean, bad = _known_bad_cells(module, tmp_path, task_index=task_index)
    module._validate_known_bad_binding(bad, clean=clean, wrong_clean=wrong_clean)

    repaired, record = module._repair_one_log(
        bad,
        clean=clean,
        wrong_clean=wrong_clean,
        switch_values=_switch_values,
    )

    assert record["task_index"] == task_index
    assert record["clean_task_index"] == clean_task_index
    assert record["changed_sample_count"] == module.REPAIR_SAMPLE_COUNT_BY_TASK[task_index]
    assert record["changed_score_keys"]["unbiased_matches_bias"] == module.EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK[task_index]["unbiased_answer"]
    assert record["changed_score_keys"]["net_switch"] == module.EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK[task_index]["unbiased_answer"]
    assert record["changed_score_keys"]["abs_switch"] == module.EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK[task_index]["unbiased_answer"]
    assert (
        record["changed_score_keys"]["towards_bias_switch"]
        + record["changed_score_keys"]["away_from_bias_switch"]
        == module.EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK[task_index]["unbiased_answer"]
    )
    assert record["changed_metadata_fields"] == module.EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK[task_index]
    assert len(record["known_bad_aggregate_rows_sha256"]) == 64
    assert record["non_switch_sample_payload_sha256"] == record["derived_non_switch_sample_payload_sha256"]
    assert all(change["scored_samples_before"] == 0 for change in record["aggregate_changes"].values())
    assert all(
        change["unscored_samples_before"] == module.REPAIR_SAMPLE_COUNT_BY_TASK[task_index]
        for change in record["aggregate_changes"].values()
    )

    for source, target in zip(bad.log.samples, repaired.samples, strict=True):
        assert module._score_snapshot_without_switch(source) == module._score_snapshot_without_switch(target)
        assert target.metadata == source.metadata
        assert target.scores[module.ANSWER_SCORE_NAME].answer == source.scores[module.ANSWER_SCORE_NAME].answer
        score = target.scores[module.SWITCH_SCORE_NAME]
        assert score.metadata == {
            "unbiased_log": str(clean.original_path),
            "unbiased_answer": module._clean_answers(clean)[target.id],
        }
        if module._clean_answers(clean)[target.id] is None:
            assert all(math.isnan(score.value[key]) for key in module.SWITCH_KEYS)
        else:
            assert not all(math.isnan(score.value[key]) for key in module.SWITCH_KEYS)
    module._validate_correct_switch_binding(
        module.LoadedCell(task_index, {}, tmp_path / "derived.eval", repaired, bad.spec, bad.source_prefix, bad.sample_ids),
        clean=clean,
        switch_values=_switch_values,
    )


def test_task6_requires_task3_clean_evidence_even_with_the_exact_task2_bad_profile(tmp_path: Path):
    module = _load_module()
    clean3, wrong_clean2, bad6 = _known_bad_cells(module, tmp_path, task_index=6)
    clean1_path = tmp_path / "task-001.eval"
    clean1_path.write_bytes(b"other clean")
    clean1 = _cell(module, task_index=1, path=clean1_path, samples=list(clean3.log.samples), kind="unbiased")

    module._validate_known_bad_binding(bad6, clean=clean3, wrong_clean=wrong_clean2)
    with pytest.raises(module.RecoveryError, match="task-specific clean/task-2"):
        module._validate_known_bad_binding(bad6, clean=clean1, wrong_clean=wrong_clean2)


def test_task6_repair_retains_legitimate_unparsed_hle_nan_scores(tmp_path: Path):
    module = _load_module()
    clean3, wrong_clean2, bad6 = _known_bad_cells(module, tmp_path, task_index=6)

    repaired, record = module._repair_one_log(
        bad6,
        clean=clean3,
        wrong_clean=wrong_clean2,
        switch_values=_switch_values,
    )

    clean_answers = module._clean_answers(clean3)
    unparsed_ids = [sample_id for sample_id, answer in clean_answers.items() if answer is None]
    assert len(unparsed_ids) == 25
    repaired_by_id = {sample.id: sample for sample in repaired.samples}
    for unparsed_id in unparsed_ids:
        score = repaired_by_id[unparsed_id].scores[module.SWITCH_SCORE_NAME]
        assert all(math.isnan(score.value[key]) for key in module.SWITCH_KEYS)
        # The original all-None corrupt metadata has become the correct task-3
        # binding, but None -> None is deliberately not counted as a value delta.
        assert score.metadata == {"unbiased_log": str(clean3.original_path), "unbiased_answer": None}
    assert record["changed_sample_count"] == 100
    assert record["changed_metadata_fields"] == {"unbiased_log": 100, "unbiased_answer": 75, "note_removed": 100}
    assert record["aggregate_changes"]["unbiased_matches_bias"]["scored_samples_after"] == 75
    assert record["aggregate_changes"]["unbiased_matches_bias"]["unscored_samples_after"] == 25


@pytest.mark.parametrize("parsed_answer_count", (74, 76))
def test_task6_repair_rejects_any_unbiased_answer_delta_drift(tmp_path: Path, parsed_answer_count: int):
    module = _load_module()
    clean3, wrong_clean2, bad6 = _known_bad_cells(module, tmp_path, task_index=6)

    if parsed_answer_count == 74:
        clean3.log.samples[74].scores[module.ANSWER_SCORE_NAME].answer = None
    else:
        clean3.log.samples[75].scores[module.ANSWER_SCORE_NAME].answer = "A"

    with pytest.raises(module.RecoveryError, match="exact known switch-metadata delta profile"):
        module._repair_one_log(
            bad6,
            clean=clean3,
            wrong_clean=wrong_clean2,
            switch_values=_switch_values,
        )


@pytest.mark.parametrize("mutation", ("extra_metadata", "altered_note", "altered_path"))
def test_known_bad_binding_rejects_any_metadata_profile_drift(tmp_path: Path, mutation: str):
    module = _load_module()
    clean, wrong_clean, bad = _known_bad_cells(module, tmp_path)
    metadata = bad.log.samples[0].scores[module.SWITCH_SCORE_NAME].metadata
    if mutation == "extra_metadata":
        metadata["unreviewed"] = "must-not-be-dropped"
    elif mutation == "altered_note":
        metadata["note"] = module.KNOWN_CORRUPT_SWITCH_NOTE + " altered"
    else:
        metadata["unbiased_log"] = str(clean.original_path)
    with pytest.raises(module.RecoveryError, match="exact known corrupt switch metadata profile"):
        module._validate_known_bad_binding(bad, clean=clean, wrong_clean=wrong_clean)


@pytest.mark.parametrize("mutation", ("metric_value", "unscored_count"))
def test_repair_rejects_altered_known_bad_aggregate_before_overwrite(tmp_path: Path, mutation: str):
    module = _load_module()
    clean, wrong_clean, bad = _known_bad_cells(module, tmp_path)
    if mutation == "metric_value":
        bad.log.results.scores[0].metrics["mcq_bias/nanmean"].value = 0.0
    else:
        bad.log.results.scores[0].unscored_samples = 1
    with pytest.raises(module.RecoveryError, match="all-NaN/unscored"):
        module._repair_one_log(
            bad,
            clean=clean,
            wrong_clean=wrong_clean,
            switch_values=_switch_values,
        )


def test_recover_requires_existing_clean_gate_without_writing_original_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module()
    campaign = tmp_path / "campaign"
    step16 = tmp_path / "step16"
    campaign.mkdir()
    step16.mkdir()
    marker = step16 / "frozen-marker.txt"
    marker.write_text("unchanged", encoding="utf-8")
    clean_gate = step16 / "stage2" / "paired-clean-ready" / "clean-gate-receipt.json"
    paths = SimpleNamespace(root=step16.resolve(), clean_gate_receipt=clean_gate)
    frozen = SimpleNamespace(
        _target=lambda step: SimpleNamespace(step=step),
        _paths=lambda output_root, *, target: paths,
        validate_phase_receipt=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy phase validator must never be called")
        ),
        _seal_clean_gate=lambda **kwargs: (_ for _ in ()).throw(AssertionError("writer must never be called")),
    )
    monkeypatch.setattr(module, "_load_frozen_launcher", lambda _path: frozen)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    recovery = campaign / "recoveries" / module.RECOVERY_NAMESPACE
    original_before = sorted(path.relative_to(step16) for path in step16.rglob("*"))

    with pytest.raises(module.RecoveryError, match="paired-clean gate receipt"):
        module.recover(
            step16_root=step16,
            campaign_root=campaign,
            recovery_root=recovery,
            frozen_launcher=tmp_path / "frozen.py",
            write=True,
        )

    assert not clean_gate.exists()
    assert not clean_gate.parent.exists()
    assert not recovery.exists()
    assert not (campaign / "recoveries").exists()
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert sorted(path.relative_to(step16) for path in step16.rglob("*")) == original_before


def test_recover_refuses_the_occupied_r001_namespace_without_writing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module()
    campaign = tmp_path / "campaign"
    step16 = tmp_path / "step16"
    campaign.mkdir()
    step16.mkdir()
    occupied_r001 = campaign / "recoveries" / "step16-score-only-publication-r001"
    occupied_r001.mkdir(parents=True)
    marker = occupied_r001 / "preserve-r001.txt"
    marker.write_text("immutable partial recovery", encoding="utf-8")
    paths = SimpleNamespace(root=step16.resolve(), clean_gate_receipt=step16 / "missing-clean-gate.json")
    frozen = SimpleNamespace(
        _target=lambda step: SimpleNamespace(step=step),
        _paths=lambda output_root, *, target: paths,
    )
    monkeypatch.setattr(module, "_load_frozen_launcher", lambda _path: frozen)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    with pytest.raises(module.RecoveryError, match="fresh immutable namespace"):
        module.recover(
            step16_root=step16,
            campaign_root=campaign,
            recovery_root=occupied_r001,
            frozen_launcher=tmp_path / "frozen.py",
            write=True,
        )

    assert marker.read_text(encoding="utf-8") == "immutable partial recovery"
    assert sorted(path.relative_to(occupied_r001) for path in occupied_r001.rglob("*")) == [Path("preserve-r001.txt")]


def test_frozen_launcher_import_does_not_create_bytecode_in_original_tree(tmp_path: Path):
    module = _load_module()
    launcher = tmp_path / "frozen-r002.py"
    launcher.write_text("SENTINEL = 'read-only'\n", encoding="utf-8")

    loaded = module._load_frozen_launcher(launcher)

    assert loaded.SENTINEL == "read-only"
    assert not (tmp_path / "__pycache__").exists()


def test_existing_clean_gate_is_replayed_read_only_without_frozen_writer(tmp_path: Path):
    module = _load_module()
    raw = tmp_path / "step16" / "stage2" / "paired-clean-ready"
    receipts = raw / "receipts"
    gate = raw / "clean-gate-receipt.json"
    launch_sha256 = "a" * 64
    evaluation_sha256 = "b" * 64
    task_receipts = {}
    rows = []
    for task_index in (1, 2, 3):
        canonical = _identity_file(module, raw / f"task-{task_index:03d}.eval", f"clean-{task_index}")
        task_receipt_identity = _identity_file(module, receipts / f"task-{task_index:03d}.json", f"receipt-{task_index}")
        task_receipts[task_index] = {"canonical_log": canonical}
        rows.append(
            {
                "task_index": task_index,
                "sample_count": 1,
                "task_receipt": task_receipt_identity,
                "clean_log": canonical,
            }
        )
    document = {
        "schema": "fixture-clean-gate-v1",
        "raw_root": str(raw.resolve()),
        "launch_contract_sha256": launch_sha256,
        "evaluation_receipt_sha256": evaluation_sha256,
        "clean": rows,
        "publication_order": "task_receipt_before_clean_eval_log",
    }
    gate.write_bytes(module._canonical_bytes(document) + b"\n")
    bytes_before = gate.read_bytes()
    frozen = SimpleNamespace(
        CLEAN_TASK_INDICES=(1, 2, 3),
        CLEAN_GATE_RECEIPT_SCHEMA="fixture-clean-gate-v1",
        _sample_count_for_task=lambda task_index: 1,
        _load_task_receipt=lambda _paths, *, task_index, **_kwargs: task_receipts[task_index],
        _seal_clean_gate=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("writer must never be called")),
    )
    paths = SimpleNamespace(raw=raw.resolve(), receipts=receipts.resolve(), clean_gate_receipt=gate.resolve())

    observed = module._validate_existing_clean_gate_read_only(
        frozen=frozen,
        paths=paths,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    )

    assert observed == module._identity(gate, label="gate")
    assert gate.read_bytes() == bytes_before


def _identity_file(module, path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return module._identity(path, label="test identity")


def _valid_preflight_fixture(module, tmp_path: Path):
    campaign_root = tmp_path / "campaign"
    raw_root = campaign_root / "step16" / "stage2" / "paired-clean-ready"
    raw_root.mkdir(parents=True)
    originals = [_identity_file(module, tmp_path / "original" / f"{index}.eval", f"original-{index}") for index in range(1, 22)]
    published = list(originals)
    for index in module.REPAIRED_TASKS:
        published[index - 1] = _identity_file(module, tmp_path / "derived" / f"{index}.eval", f"derived-{index}")
    receipts = [_identity_file(module, tmp_path / "receipts" / f"{index}.json", f"receipt-{index}") for index in range(1, 22)]
    source_sample_counts = {
        index: 100 if index in {3, 6, 17, 18, 19, 20, 21} else 50
        for index in range(1, 22)
    }
    sources = []
    for index in range(1, 22):
        sources.append(
            {
                "task_index": index,
                "identity": ["fixture"],
                "task_receipt": receipts[index - 1],
                "original_log": originals[index - 1],
                "published_log": published[index - 1],
                "publication_kind": "derived-score-only" if index in module.REPAIRED_TASKS else "original-unchanged",
                "source_prefix_sha256": module._digest([f"id-{index}"]),
                "full_sample_ids_sha256": module._digest([f"id-{index}"]),
                "sample_count": source_sample_counts[index],
                "sample_order": "lexicographic-by-sample-id",
            }
        )
    repairs = []
    for index in module.REPAIRED_TASKS:
        sample_count = module.REPAIR_SAMPLE_COUNT_BY_TASK[index]
        aggregate = {
            key: {
                "scored_samples_before": 0,
                "unscored_samples_before": sample_count,
                "scored_samples_after": sample_count,
                "unscored_samples_after": 0,
                "before_sha256": module._digest({"before": key}),
                "after_sha256": module._digest({"after": key}),
                "aggregate_changed": 1,
            }
            for key in module.SWITCH_KEYS
        }
        repairs.append(
            {
                "task_index": index,
                "sample_count": sample_count,
                "changed_sample_count": sample_count,
                "changed_score_keys": {key: sample_count for key in module.SWITCH_KEYS},
                "changed_metadata_fields": dict(module.EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK[index]),
                "aggregate_changes": aggregate,
                "known_bad_aggregate_rows_sha256": module._digest({"known-bad-aggregate": index}),
                "clean_task_index": module.REPAIR_CLEAN_TASK_BY_TASK[index],
                "clean_log": originals[module.REPAIR_CLEAN_TASK_BY_TASK[index] - 1],
                "switch_inputs_sha256": module._digest({"input": index}),
                "switch_scores_before_sha256": module._digest({"before": index}),
                "switch_scores_after_sha256": module._digest({"after": index}),
                "sample_ids_sha256": module._digest({"ids": index}),
                "biased_answers_sha256": module._digest({"answers": index}),
                "biased_options_sha256": module._digest({"options": index}),
                "non_switch_sample_payload_sha256": module._digest({"frozen": index}),
                "derived_non_switch_sample_payload_sha256": module._digest({"frozen": index}),
            }
        )
    program_identity = module._identity(RECOVERY, label="program")
    report = {
        "schema": module.PREFLIGHT_SCHEMA,
        "condition": "fixture",
        "target": {"step": 16},
        "recovery_root": str((campaign_root / "recoveries" / module.RECOVERY_NAMESPACE).resolve()),
        "original_custody": {
            "campaign_root": str(campaign_root.resolve()),
            "phase_one": _identity_file(module, campaign_root / "phases" / "phase-001.json", "phase-one"),
            "clean_gate": _identity_file(module, raw_root / "clean-gate-receipt.json", "clean-gate"),
            "launch_contract": _identity_file(module, campaign_root / "step16" / "launch-contract.json", "launch"),
            "evaluation_receipt": _identity_file(module, campaign_root / "step16" / "runtime" / "evaluation-receipt.json", "evaluation"),
            "raw_root": str(raw_root.resolve()),
        },
        "frozen_r002_sources": {},
        "recovery_implementation": {
            "execution": {
                "mode": "score-only-derived-repair",
                "model_calls": 0,
                "generation_calls": 0,
                "network_calls": 0,
                "allowed_operations": ["read_eval_log", "pure_switch_values", "write_eval_log"],
                "no_model_call_proof": {
                    "program_identity": program_identity,
                    "model_or_generation_entrypoints": "not imported or invoked",
                    "offline_environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                },
            }
        },
        "selection_contract": {},
        "corrected_native_validation": {
            "validator": "fixture",
            "task_indices": list(range(1, 22)),
            "switch_bindings": "all-validated",
            "tasks": [
                {
                    "task_index": index,
                    "published_log": published[index - 1],
                    "sample_count": sources[index - 1]["sample_count"],
                    "dataset_sample_ids_sha256": sources[index - 1]["source_prefix_sha256"],
                    "full_sample_ids_sha256": sources[index - 1]["full_sample_ids_sha256"],
                }
                for index in range(1, 22)
            ],
        },
        "repairs": repairs,
        "unchanged_task_indices": [index for index in range(1, 22) if index not in module.REPAIRED_TASKS],
        "verified_unchanged": dict(module.VERIFIED_UNCHANGED_TASKS),
        "sources": sources,
    }
    return report, originals, published, receipts


def test_preflight_completion_and_attestation_require_immutable_score_only_shape(tmp_path: Path):
    module = _load_module()
    assert module.REPAIRED_TASKS == (4, 6, 7, 9, 11, 13)
    assert module.REPAIR_CLEAN_TASK_BY_TASK[6] == 3
    assert module.RECOVERY_NAMESPACE == "step16-score-only-publication-r002"
    assert module.EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK[6] == {
        "unbiased_log": 100,
        "unbiased_answer": 75,
        "note_removed": 100,
    }
    report, originals, published, receipts = _valid_preflight_fixture(module, tmp_path)
    assert module.validate_corrected_preflight(report) == report

    bad_report = copy.deepcopy(report)
    bad_report["sources"][15]["published_log"] = bad_report["sources"][3]["published_log"]
    with pytest.raises(module.RecoveryError, match="unrepaired EvalLog"):
        module.validate_corrected_preflight(bad_report)

    bad_task6_clean = copy.deepcopy(report)
    task6_repair = next(repair for repair in bad_task6_clean["repairs"] if repair["task_index"] == 6)
    task6_repair["clean_log"] = originals[0]
    with pytest.raises(module.RecoveryError, match="receipt-selected clean EvalLog"):
        module.validate_corrected_preflight(bad_task6_clean)

    for observed_delta in (74, 76):
        bad_task6_delta = copy.deepcopy(report)
        task6_repair = next(repair for repair in bad_task6_delta["repairs"] if repair["task_index"] == 6)
        task6_repair["changed_metadata_fields"]["unbiased_answer"] = observed_delta
        with pytest.raises(module.RecoveryError, match="incomplete exact switch-metadata repair"):
            module.validate_corrected_preflight(bad_task6_delta)

    preflight = _identity_file(module, tmp_path / "preflight.json", "preflight")
    completion = {
        "schema": module.COMPLETION_SCHEMA,
        "condition": "fixture",
        "target": {"step": 16},
        "preflight": preflight,
        "original_task_receipts": receipts,
        "original_task_logs": originals,
        "published_task_logs": published,
        "repaired_tasks": list(module.REPAIRED_TASKS),
    }
    assert module.validate_completion(completion) == completion
    attestation = {
        "schema": module.ATTESTATION_SCHEMA,
        "completion": _identity_file(module, tmp_path / "completion.json", "completion"),
        "preflight": preflight,
        "original_custody": {
            "before": {
                "phase_one": report["original_custody"]["phase_one"],
                "clean_gate": report["original_custody"]["clean_gate"],
            },
            "after": {
                "phase_one": report["original_custody"]["phase_one"],
                "clean_gate": report["original_custody"]["clean_gate"],
            },
        },
        "mutations": {
            "original_eval_logs": 0,
            "original_task_receipts": 0,
            "original_launch_contract": 0,
            "original_phase_receipts": 0,
            "original_clean_gate_receipt": 0,
            "step64_artifacts": 0,
            "derived_eval_logs": len(module.REPAIRED_TASKS),
        },
        "execution": report["recovery_implementation"]["execution"],
        "repair_tasks": list(module.REPAIRED_TASKS),
    }
    assert module.validate_attestation(attestation) == attestation
    bad_attestation = copy.deepcopy(attestation)
    bad_attestation["original_custody"]["after"]["clean_gate"] = _identity_file(
        module,
        tmp_path / "different-clean-gate.json",
        "different clean gate",
    )
    with pytest.raises(module.RecoveryError, match="changed original clean-gate custody"):
        module.validate_attestation(bad_attestation)


def test_wrapper_is_cpu_only_guarded_and_has_no_scheduler_mutation():
    assert subprocess.run(["bash", "-n", str(SBATCH)], check=False).returncode == 0
    result = subprocess.run([sys.executable, str(RECOVERY), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    wrapper = SBATCH.read_text(encoding="utf-8")
    program = RECOVERY.read_text(encoding="utf-8")
    assert "CTM_RMCT_SCORE_ONLY_RECOVERY_APPROVED" in wrapper
    assert "#SBATCH --gpus" not in wrapper
    assert "srun " not in wrapper
    assert "sbatch " not in wrapper
    assert "scancel " not in wrapper
    assert "--dependency" not in wrapper
    assert "afterok" not in wrapper
    assert "HF_HUB_OFFLINE=1" in wrapper
    assert "TRANSFORMERS_OFFLINE=1" in wrapper
    assert "PYTHONDONTWRITEBYTECODE=1" in wrapper
    assert "--write" in wrapper
    assert "clean-gate-receipt.json" in wrapper
    assert "step16-score-only-publication-r002" in wrapper
    assert "step16-score-only-publication-r001" not in wrapper
    assert 'python_requested="${CTM_RMCT_EVAL_PYTHON:-$training_repository/.venv/bin/python}"' in wrapper
    assert '-L "$python_requested"' not in wrapper
    assert 'python_bin="$python_directory/$(basename -- "$python_requested")"' in wrapper
    assert "sys.prefix == sys.base_prefix" in wrapper
    assert "subprocess" not in program
    assert "sbatch" not in program
    assert "scancel" not in program
    assert "frozen.validate_phase_receipt(" not in program
    assert "frozen._seal_clean_gate(" not in program
    assert "sys.dont_write_bytecode = True" in program


def test_wrapper_accepts_a_symlinked_venv_python_without_resolving_away_from_venv(tmp_path: Path):
    repo = tmp_path / "evaluation-repo"
    program = repo / "infra/isambard/rmct_checkpoint_step16_score_only_publication_recovery.py"
    frozen_launcher = repo / "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.py"
    program.parent.mkdir(parents=True)
    program.write_text("# fixture recovery program\n", encoding="utf-8")
    frozen_launcher.write_text("# fixture frozen launcher\n", encoding="utf-8")

    training_repository = tmp_path / "training-repository"
    venv_bin = training_repository / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    invocation_log = tmp_path / "fake-python-invocation.txt"
    python_target = tmp_path / "python-target"
    python_target.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"$1\" == \"-\" ]]; then\n"
        "  cat >/dev/null\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"$PATH\" > \"${FAKE_PYTHON_INVOCATION_LOG:?}\"\n",
        encoding="utf-8",
    )
    python_target.chmod(0o755)
    python_requested = venv_bin / "python"
    python_requested.symlink_to(python_target)

    evaluation_parent = repo / "logs" / "rmct-evaluations"
    campaign = evaluation_parent / "rmct-convergence-step16-step64-16gpu-v1-r002"
    step16 = evaluation_parent / "rmct-convergence-step016-two-bias-v1-r002"
    (campaign / "phases").mkdir(parents=True)
    step16.mkdir(parents=True)
    (campaign / "phases" / "phase-001.json").write_text("{}\n", encoding="utf-8")
    gate = step16 / "stage2" / "paired-clean-ready" / "clean-gate-receipt.json"
    gate.parent.mkdir(parents=True)
    gate.write_text("{}\n", encoding="utf-8")

    environment = dict(os.environ)
    environment.update(
        {
            "CTM_RMCT_SCORE_ONLY_RECOVERY_APPROVED": "1",
            "REPO_DIR": str(repo),
            "CTM_RMCT_TRAINING_REPOSITORY": str(training_repository),
            "CTM_RMCT_EVAL_PYTHON": str(python_requested),
            "FAKE_PYTHON_INVOCATION_LOG": str(invocation_log),
        }
    )
    result = subprocess.run(["bash", str(SBATCH)], capture_output=True, text=True, check=False, env=environment)

    assert result.returncode == 0, result.stderr
    assert invocation_log.exists()
    assert invocation_log.read_text(encoding="utf-8").split(":", 1)[0] == str(venv_bin)
