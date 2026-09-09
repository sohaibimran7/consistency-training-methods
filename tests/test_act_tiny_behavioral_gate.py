"""Focused CPU-only contracts for the tiny native-HF ACT behavioral gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.act_repair_gate.behavioral_gate import SCHEMA, build_attestation, write_attestation
from experiments.act_repair_gate.direct_answer import _select_balanced_rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(*, switches: int, eligible: int) -> dict:
    return {
        "pooled": {
            "counts": {"toward_bias_switches": switches, "eligible": eligible},
            "metrics": {
                "direct_tbsr": {
                    "numerator": switches,
                    "denominator": eligible,
                    "rate": switches / eligible,
                }
            },
        }
    }


def _report(*, adapter: Path, train: Path, heldout: Path, trained_switches: int = 0) -> dict:
    sources = {}
    for split, path in (("train_eval", train), ("heldout_in_domain", heldout)):
        sources[split] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "source_samples": 200,
            "samples": 8,
            "selected_counts_by_dataset": {"hellaswag": 4, "logiqa": 4},
            "question_ids_sha256": "a" * 64,
        }
    return {
        "schema": "act-repair-direct-answer-v1",
        "protocol": {
            "backend": "transformers_peft_hf_only",
            "selection": {"kind": "balanced_dataset_prefix", "limit_per_dataset": 4},
        },
        "model": "Qwen/Qwen3.5-9B",
        "adapter": {
            "path": str(adapter.resolve()),
            "adapter_model_sha256": _sha256(adapter / "adapter_model.safetensors"),
            "condition_name": "act",
        },
        "sources": sources,
        "conditions": {
            "untrained": {
                "cells": {
                    "train_eval": _cell(switches=2, eligible=8),
                    "heldout_in_domain": _cell(switches=1, eligible=8),
                }
            },
            "act": {
                "cells": {
                    "train_eval": _cell(switches=trained_switches, eligible=8),
                    "heldout_in_domain": _cell(switches=0, eligible=8),
                }
            },
        },
    }


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    train.write_text('{"frozen":"train"}\n', encoding="utf-8")
    heldout.write_text('{"frozen":"heldout"}\n', encoding="utf-8")
    report = tmp_path / "report.json"
    return adapter, train, heldout, report


def test_tiny_behavioral_gate_hash_binds_a_passing_native_hf_report(tmp_path):
    adapter, train, heldout, report_path = _fixture_paths(tmp_path)
    report_path.write_text(json.dumps(_report(adapter=adapter, train=train, heldout=heldout)), encoding="utf-8")

    attestation = build_attestation(
        report_path=report_path,
        adapter=adapter,
        train_data=train,
        heldout_data=heldout,
        expected_limit_per_dataset=4,
    )

    assert attestation["schema"] == SCHEMA
    assert attestation["gate"]["passed"] is True
    assert attestation["gate"]["untrained"] == {
        "toward_bias_switches": 2,
        "eligible": 8,
        "direct_tbsr": 0.25,
    }
    assert attestation["sources"]["train_eval"]["selected_counts_by_dataset"] == {
        "hellaswag": 4,
        "logiqa": 4,
    }
    output = write_attestation(tmp_path / "attestation.json", attestation)
    assert json.loads(output.read_text(encoding="utf-8")) == attestation
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_attestation(output, attestation)


def test_tiny_behavioral_gate_records_a_failed_directional_result_without_rewriting_evidence(tmp_path):
    adapter, train, heldout, report_path = _fixture_paths(tmp_path)
    report_path.write_text(
        json.dumps(_report(adapter=adapter, train=train, heldout=heldout, trained_switches=2)), encoding="utf-8"
    )

    attestation = build_attestation(
        report_path=report_path,
        adapter=adapter,
        train_data=train,
        heldout_data=heldout,
        expected_limit_per_dataset=4,
    )

    assert attestation["gate"]["baseline_has_signal"] is True
    assert attestation["gate"]["strict_direct_tbsr_reduction"] is False
    assert attestation["gate"]["passed"] is False


def test_balanced_direct_answer_selection_is_deterministic_and_rejects_incomplete_groups():
    rows = [
        {"question_id": "l1", "source_dataset": "logiqa"},
        {"question_id": "h1", "source_dataset": "hellaswag"},
        {"question_id": "l2", "source_dataset": "logiqa"},
        {"question_id": "h2", "source_dataset": "hellaswag"},
        {"question_id": "l3", "source_dataset": "logiqa"},
    ]
    selected = _select_balanced_rows(rows, limit_per_dataset=2)

    assert [row["question_id"] for row in selected] == ["l1", "h1", "l2", "h2"]
    with pytest.raises(ValueError, match="cannot build balanced"):
        _select_balanced_rows(rows, limit_per_dataset=3)
