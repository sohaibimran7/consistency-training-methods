from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.stage1_iid_diagnostic import gate_analysis


def _score(value):
    return SimpleNamespace(value=value)


def _sample(question_id: str, *, dataset: str, index: int, toward_count: int):
    if index == 0:
        # Both legacy None and Inspect's current NaN null sentinel must mean a
        # genuinely unscored pair, rather than a zero-valued non-switch.
        null = float("nan")
        values = {
            "unbiased_matches_bias": null,
            "towards_bias_switch": null,
            "away_from_bias_switch": null,
            "net_switch": null,
            "abs_switch": null,
        }
    elif index <= 59:
        toward = int(index <= toward_count)
        values = {
            "unbiased_matches_bias": 0,
            "towards_bias_switch": toward,
            "away_from_bias_switch": None,
            "net_switch": toward,
            "abs_switch": toward,
        }
    else:
        values = {
            "unbiased_matches_bias": 1,
            "towards_bias_switch": None,
            "away_from_bias_switch": 0,
            "net_switch": 0,
            "abs_switch": 0,
        }
    return SimpleNamespace(
        id=question_id,
        metadata={
            "variant": "biased",
            "source_dataset": dataset,
            "bias_type": "wrong_argument",
            "prompt_style": "encourage_cot",
        },
        scores={"paired-switch": _score(values)},
    )


def _log(*, variant: str, split: str, dataset: str, toward_count: int, ids: list[str] | None = None):
    question_ids = ids or [f"{dataset}-{split}-{index:03d}" for index in range(100)]
    variant_file = None if variant == "native" else f"/frozen/{split}-canonical-pairs.jsonl"
    task_args = {
        "dataset": dataset,
        "split": split,
        "bias_type": "wrong_argument",
        "question_ids_from": question_ids,
        "variant_file": variant_file,
        "unbiased_log": f"/logs/act/{split}/clean.eval",
        "prompt_style": "encourage_cot",
        "include_bias_acknowledged": False,
        "grader_model": None,
        "source_sha256": "frozen-stage1-source",
    }
    metadata = {
        "source_dataset": dataset,
        "split": split,
        "bias_type": "wrong_argument",
        "question_ids_from": question_ids,
        "variant_file": variant_file,
        "unbiased_log": f"/logs/act/{split}/clean.eval",
        "prompt_style": "encourage_cot",
        "include_bias_acknowledged": False,
        "grader_model": None,
        "source_identity_digest": "frozen-stage1-source",
    }
    return SimpleNamespace(
        status="success",
        eval=SimpleNamespace(
            task="stage1_iid_biased",
            task_args=task_args,
            metadata=metadata,
            created="2026-07-31T00:00:00Z",
        ),
        samples=[
            _sample(question_id, dataset=dataset, index=index, toward_count=toward_count)
            for index, question_id in enumerate(question_ids)
        ],
    )


def _install_fake_inspect(monkeypatch, logs):
    inspect_module = types.ModuleType("inspect_ai")
    inspect_log = types.ModuleType("inspect_ai.log")

    def list_eval_logs(root, **_kwargs):
        return sorted(str(path) for path in Path(root).rglob("*.eval"))

    def read_eval_log(path, **_kwargs):
        return logs[Path(path).resolve()]

    inspect_log.list_eval_logs = list_eval_logs
    inspect_log.read_eval_log = read_eval_log
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_module)
    monkeypatch.setitem(sys.modules, "inspect_ai.log", inspect_log)


def _populate_variant(tmp_path: Path, *, variant: str, toward_count: int, logs: dict[Path, object]) -> Path:
    root = tmp_path / variant
    for split in gate_analysis.SPLITS:
        for dataset in gate_analysis.DATASETS:
            path = root / split / f"{dataset}.eval"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{variant}/{split}/{dataset}".encode("utf-8"))
            logs[path.resolve()] = _log(
                variant=variant,
                split=split,
                dataset=dataset,
                toward_count=toward_count,
            )
    return root


def _fixture_roots(tmp_path, monkeypatch):
    logs: dict[Path, object] = {}
    native = _populate_variant(tmp_path, variant="native", toward_count=20, logs=logs)
    canonical = _populate_variant(tmp_path, variant="canonical", toward_count=5, logs=logs)
    _install_fake_inspect(monkeypatch, logs)
    return native, canonical, logs


def test_raw_gate_report_uses_local_parser_eligibility_and_conditional_tbsr(tmp_path, monkeypatch):
    native, canonical, _ = _fixture_roots(tmp_path, monkeypatch)

    report = gate_analysis.build_report(native, canonical)

    assert report["schema"] == gate_analysis.ANALYSIS_SCHEMA
    assert report["analysis_mode"] == "raw_local_paired_switch_scores_only"
    assert set(report["cells"]) == {
        "native/train_eval",
        "native/heldout_in_domain",
        "canonical/train_eval",
        "canonical/heldout_in_domain",
    }
    native_train = report["cells"]["native/train_eval"]["pooled"]
    canonical_heldout = report["cells"]["canonical/heldout_in_domain"]["pooled"]
    assert native_train["counts"] == {
        "samples": 200,
        "joint_parsed": 198,
        "joint_parse_failures": 2,
        "eligible_clean_answer_not_bias_answer": 118,
        "ineligible_clean_answer_equals_bias_answer": 80,
        "toward_bias_switches": 40,
    }
    assert native_train["rates"] == {
        "parser_coverage": {"numerator": 198, "denominator": 200, "rate": 0.99},
        "eligible_coverage": {"numerator": 118, "denominator": 198, "rate": 118 / 198},
        "tbsr": {"numerator": 40, "denominator": 118, "rate": 40 / 118},
    }
    assert canonical_heldout["rates"]["tbsr"] == {
        "numerator": 10,
        "denominator": 118,
        "rate": 10 / 118,
    }
    assert all(source["variant_file"] is None for source in report["sources"] if source["prompt_variant"] == "native")
    assert all(source["variant_file"] for source in report["sources"] if source["prompt_variant"] == "canonical")
    assert "luna" not in json.dumps(report, sort_keys=True).lower()

    output = tmp_path / "report.json"
    assert gate_analysis.write_report(output, report) == "written"
    assert gate_analysis.write_report(output, report) == "resumed"
    changed = json.loads(json.dumps(report))
    changed["condition"] = "different-act"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gate_analysis.write_report(output, changed)


def test_raw_gate_rejects_duplicate_successful_cell(tmp_path, monkeypatch):
    native, canonical, logs = _fixture_roots(tmp_path, monkeypatch)
    duplicate = native / "train_eval" / "logiqa-retry.eval"
    duplicate.write_bytes(b"duplicate")
    logs[duplicate.resolve()] = _log(
        variant="native",
        split="train_eval",
        dataset="logiqa",
        toward_count=20,
    )

    with pytest.raises(ValueError, match="duplicate successful ACT gate log"):
        gate_analysis.build_report(native, canonical)


def test_raw_gate_accepts_inspect_omitting_native_default_variant_file(tmp_path, monkeypatch):
    native, canonical, logs = _fixture_roots(tmp_path, monkeypatch)
    path = native / "train_eval" / "logiqa.eval"
    log = logs[path.resolve()]
    log.eval.task_args.pop("variant_file")
    log.eval.metadata.pop("variant_file")

    report = gate_analysis.build_report(native, canonical)

    source = next(
        source
        for source in report["sources"]
        if source["prompt_variant"] == "native" and source["split"] == "train_eval" and source["dataset"] == "logiqa"
    )
    assert source["variant_file"] is None


def test_raw_gate_rejects_missing_required_cell(tmp_path, monkeypatch):
    native, canonical, logs = _fixture_roots(tmp_path, monkeypatch)
    missing = canonical / "heldout_in_domain" / "hellaswag.eval"
    logs.pop(missing.resolve())
    missing.unlink()

    with pytest.raises(ValueError, match="canonical ACT gate logs are incomplete"):
        gate_analysis.build_report(native, canonical)


def test_raw_gate_rejects_native_canonical_population_mismatch(tmp_path, monkeypatch):
    native, canonical, logs = _fixture_roots(tmp_path, monkeypatch)
    path = canonical / "train_eval" / "logiqa.eval"
    mismatched_ids = [f"logiqa-train_eval-different-{index:03d}" for index in range(100)]
    logs[path.resolve()] = _log(
        variant="canonical",
        split="train_eval",
        dataset="logiqa",
        toward_count=5,
        ids=mismatched_ids,
    )

    with pytest.raises(ValueError, match="native/canonical question_ids_from differ"):
        gate_analysis.build_report(native, canonical)
