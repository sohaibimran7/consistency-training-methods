"""Focused tests for the offline final Stage 1 IID matrix merger."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.stage1_iid_diagnostic import analyze, plot
from experiments.stage1_iid_diagnostic_none import merge_final_matrix as merge


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _summary(samples: int, *, clean_not_bias: int, clean_equals_bias: int, luna_yes: int) -> dict:
    towards = clean_not_bias // 2
    away = clean_equals_bias // 4
    total = towards + away
    luna_given_towards = min(towards, luna_yes)
    return {
        "counts": {
            "samples": samples,
            "joint_parsed": samples,
            "joint_parse_failures": 0,
            "clean_answer_not_bias_answer": clean_not_bias,
            "clean_answer_equals_bias_answer": clean_equals_bias,
            "luna_parsed": samples,
            "luna_parse_failures": 0,
            "generation_max_token_cap_hits": 0,
            "grader_max_token_cap_hits": 0,
        },
        "rates": {
            "tbsr": {"numerator": towards, "denominator": clean_not_bias, "rate": towards / clean_not_bias},
            "away_from_bias": {
                "numerator": away,
                "denominator": clean_equals_bias,
                "rate": away / clean_equals_bias,
            },
            "total_switch": {"numerator": total, "denominator": samples, "rate": total / samples},
            "luna_yes": {"numerator": luna_yes, "denominator": samples, "rate": luna_yes / samples},
            "luna_yes_given_towards_bias_switch": {
                "numerator": luna_given_towards,
                "denominator": towards,
                "rate": luna_given_towards / towards,
            },
        },
    }


def _pooled(parts: dict[str, dict]) -> dict:
    counts = {key: sum(parts[dataset]["counts"][key] for dataset in merge.DATASETS) for key in merge.COUNT_KEYS}
    rates = {}
    for metric in merge.RATE_KEYS:
        numerator = sum(parts[dataset]["rates"][metric]["numerator"] for dataset in merge.DATASETS)
        denominator = sum(parts[dataset]["rates"][metric]["denominator"] for dataset in merge.DATASETS)
        rates[metric] = {
            "numerator": numerator,
            "denominator": denominator,
            "rate": numerator / denominator if denominator else None,
        }
    return {"counts": counts, "rates": rates}


def _cell(raw_condition: str, split: str, *, include_first64: bool) -> dict:
    per_dataset = {dataset: _summary(100, clean_not_bias=90, clean_equals_bias=10, luna_yes=60) for dataset in merge.DATASETS}
    result = {
        "condition": raw_condition,
        "split": split,
        "pooled": _pooled(per_dataset),
        "per_dataset": per_dataset,
    }
    if include_first64:
        subset = {dataset: _summary(32, clean_not_bias=30, clean_equals_bias=2, luna_yes=20) for dataset in merge.DATASETS}
        result["rmct_first64"] = {"pooled": _pooled(subset), "per_dataset": subset}
    return result


def _population() -> dict:
    return {
        "source_sha256": merge.FROZEN_SOURCE_SHA256,
        "source_rows": 3000,
        "source_counts_by_dataset": {"hellaswag": 1500, "logiqa": 1500},
        "bias_type": "wrong_argument",
        "prompt_style": "none",
        "splits": {
            split: {
                "content_sha256": merge.FROZEN_SPLITS[split]["content_sha256"],
                "question_ids_sha256": merge.FROZEN_SPLITS[split]["question_ids_sha256"],
                "row_count": 200,
                "counts_by_dataset": {"hellaswag": 100, "logiqa": 100},
            }
            for split in merge.SPLITS
        },
        "cell_question_ids_sha256": {f"{split}/{dataset}": merge.FROZEN_CELL_QUESTION_IDS_SHA256[(split, dataset)] for split in merge.SPLITS for dataset in merge.DATASETS},
        "rmct_first64": {
            "question_ids_sha256": merge.FROZEN_RMCT_FIRST64_IDS_SHA256,
            "row_count": 64,
            "counts_by_dataset": {"hellaswag": 32, "logiqa": 32},
        },
    }


def _write(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _report_document(raw_conditions: list[str], *, source_conditions: dict[str, str] | None = None) -> tuple[dict, dict[str, str]]:
    report_manifest_sha = _sha("manifest/" + ",".join(raw_conditions))
    sources = []
    raw_hashes: dict[str, str] = {}
    for raw_condition in raw_conditions:
        for split in merge.SPLITS:
            for dataset in merge.DATASETS:
                key = f"{raw_condition}/{split}/{dataset}"
                raw_hash = _sha("raw/" + key)
                raw_hashes[key] = raw_hash
                source = {
                    "condition": raw_condition,
                    "split": split,
                    "dataset": dataset,
                    "graded_log": f"/graded/{key}.eval",
                    "graded_log_sha256": _sha("graded/" + key),
                    "samples": 100,
                    "source_log": f"/raw/{key}.eval",
                    "source_sha256": raw_hash,
                }
                if source_conditions and raw_condition in source_conditions:
                    source["source_condition"] = source_conditions[raw_condition]
                sources.append(source)
    cells = {}
    for raw_condition in raw_conditions:
        canonical = merge.CONDITION_ALIASES[raw_condition]
        for split in merge.SPLITS:
            cells[f"{raw_condition}/{split}"] = _cell(
                raw_condition,
                split,
                include_first64=canonical in {"rmct", "rmct-control"} and split == "train_eval",
            )
    return (
        {
            "schema": analyze.ANALYSIS_SCHEMA,
            "grader_model": analyze.DEFAULT_LUNA_GRADER_MODEL,
            "grader_max_tokens": 1024,
            "inspect_rescore_model": analyze.INSPECT_RESCORE_MODEL,
            "diagnostic_manifest": "/host/manifest.json",
            "diagnostic_manifest_sha256": report_manifest_sha,
            "metric_definitions": copy.deepcopy(merge.EXPECTED_METRIC_DEFINITIONS),
            "sources": sources,
            "cells": cells,
        },
        raw_hashes,
    )


def _preflight_document(
    raw_condition: str,
    raw_hashes: dict[str, str],
    *,
    report_manifest_sha: str,
    runtime_profile: str = "vllm",
) -> dict:
    sources = []
    for split in merge.SPLITS:
        for dataset in merge.DATASETS:
            key = f"{raw_condition}/{split}/{dataset}"
            sources.append(
                {
                    "split": split,
                    "dataset": dataset,
                    "sample_count": 100,
                    "source_identity_digest": merge.FROZEN_SOURCE_SHA256,
                    "prompt_style": "none",
                    "question_ids_sha256": merge.FROZEN_CELL_QUESTION_IDS_SHA256[(split, dataset)],
                    "raw_log": f"/raw/{key}.eval",
                    "raw_log_sha256": raw_hashes[key],
                }
            )
    contract = {
        "source_sha256": merge.FROZEN_SOURCE_SHA256,
        "bias_type": "wrong_argument",
        "prompt_style": "none",
        "expected_base_model": "Qwen/Qwen3.5-9B",
        "include_bias_acknowledged": False,
        "grader_model": None,
        "runtime_profile": runtime_profile,
    }
    if runtime_profile == "hf-peft":
        contract.update(
            {
                "expected_checkpoint": f"/checkpoints/{raw_condition}",
                "expected_max_connections": 8,
                "checkpoint_artifact_identity": {
                    "adapter_model_sha256": _sha(f"adapter-model/{raw_condition}"),
                    "adapter_config_sha256": _sha(f"adapter-config/{raw_condition}"),
                    "checkpoint_manifest_sha256": _sha(f"checkpoint-manifest/{raw_condition}"),
                },
            }
        )
    elif raw_condition != "base-vllm":
        contract.update(
            {
                "expected_checkpoint": f"/checkpoints/{raw_condition}-vllm-compat",
                "vllm_compatibility_adapter_identity": {
                    "adapter_model_sha256": _sha(f"vllm-adapter-model/{raw_condition}"),
                    "adapter_config_sha256": _sha(f"vllm-adapter-config/{raw_condition}"),
                    "compatibility_manifest_sha256": _sha(f"vllm-compatibility-manifest/{raw_condition}"),
                    "parity_attestation_sha256": _sha(f"vllm-parity-attestation/{raw_condition}"),
                    "source_adapter_model_sha256": _sha(f"vllm-source-adapter-model/{raw_condition}"),
                },
            }
        )
    return {
        "schema": merge.PREFLIGHT_SCHEMA,
        "condition": raw_condition,
        "contract": contract,
        "frozen_splits": {
            split: {
                "path": f"/host/{split}.jsonl",
                "sha256": merge.FROZEN_SPLITS[split]["content_sha256"],
            }
            for split in merge.SPLITS
        },
        "manifest": "/host/manifest.json",
        "manifest_sha256": report_manifest_sha,
        "sources": sources,
    }


def _inputs(tmp_path: Path, monkeypatch, *, accelerated_rmct_control: bool = False) -> dict:
    population_path = tmp_path / "population-manifest.json"
    population_path.write_text("{}\n", encoding="utf-8")
    population = _population()
    monkeypatch.setattr(
        merge,
        "load_population_fingerprint",
        lambda _path: (population_path.resolve(), population_path.read_bytes(), population),
    )
    groups = [
        ["base-vllm", "repaired-act-vllm-compat", "attct-vllm-compat", "mlpct-vllm-compat"],
        ["bct-main-hf-peft", "bct-control-hf-peft"],
        ["opct-vllm"],
        ["rmct"],
        ["rmct-control"],
    ]
    reports: list[Path] = []
    preflights: list[Path] = []
    report_documents: dict[str, dict] = {}
    for number, raw_conditions in enumerate(groups):
        source_conditions = None
        if accelerated_rmct_control and raw_conditions == ["rmct-control"]:
            source_conditions = {"rmct-control": "rmct-control-b8-accelerated"}
        report, raw_hashes = _report_document(raw_conditions, source_conditions=source_conditions)
        report_path = _write(tmp_path / f"report-{number}.json", report)
        reports.append(report_path)
        for raw_condition in raw_conditions:
            preflight_condition = "rmct-control-b8-accelerated" if accelerated_rmct_control and raw_condition == "rmct-control" else raw_condition
            # The RMCT canonical report source hashes are necessarily sourced
            # from the operational alias's raw logs.
            adjusted_hashes = {key.replace(raw_condition, preflight_condition, 1): value for key, value in raw_hashes.items() if key.startswith(raw_condition + "/")}
            preflight = _preflight_document(
                preflight_condition,
                adjusted_hashes,
                report_manifest_sha=report["diagnostic_manifest_sha256"],
                runtime_profile=(
                    "hf-peft"
                    if raw_condition in {"bct-main-hf-peft", "bct-control-hf-peft"}
                    else "vllm"
                ),
            )
            preflights.append(_write(tmp_path / f"preflight-{preflight_condition}.json", preflight))
        report_documents[raw_conditions[0]] = report
    return {
        "population": population_path,
        "reports": reports,
        "preflights": preflights,
        "documents": report_documents,
    }


def test_final_matrix_merger_builds_exact_plot_ready_eighteen_cells_and_preserves_inputs(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in [*inputs["reports"], *inputs["preflights"], inputs["population"]]}

    adapter, provenance = merge.build_final_matrix(inputs["reports"], inputs["preflights"], inputs["population"])

    plot.validate_report(adapter)
    assert set(adapter["cells"]) == {f"{condition}/{split}" for condition in merge.CANONICAL_CONDITIONS for split in merge.SPLITS}
    assert adapter["cells"]["untrained/train_eval"]["condition"] == "untrained"
    assert adapter["cells"]["act/heldout_in_domain"]["condition"] == "act"
    assert len(adapter["sources"]) == 36
    assert provenance["checks"]["analysis_raw_hashes_match_preflight_raw_hashes"] is True
    assert provenance["checks"]["bct_main_and_control_use_backend_matched_native_hf_peft"] is True
    assert provenance["checks"]["bct_main_and_control_bind_raw_adapter_config_and_manifest_hashes"] is True
    assert provenance["checks"]["vllm_lora_conditions_bind_compatibility_adapter_and_parity_attestation"] is True
    act_preflight = next(
        record
        for record in provenance["raw_preflight_reports"]
        if record["canonical_condition"] == "act"
    )
    assert act_preflight["vllm_compatibility_adapter_identity"] == {
        "adapter_model_sha256": _sha("vllm-adapter-model/repaired-act-vllm-compat"),
        "adapter_config_sha256": _sha("vllm-adapter-config/repaired-act-vllm-compat"),
        "compatibility_manifest_sha256": _sha("vllm-compatibility-manifest/repaired-act-vllm-compat"),
        "parity_attestation_sha256": _sha("vllm-parity-attestation/repaired-act-vllm-compat"),
        "source_adapter_model_sha256": _sha("vllm-source-adapter-model/repaired-act-vllm-compat"),
    }
    assert {path: path.read_bytes() for path in before} == before

    output = tmp_path / "analysis" / "final-matrix.json"
    proof = tmp_path / "analysis" / "final-matrix-provenance.json"
    assert merge.write_final_matrix(adapter, provenance, output=output, provenance_output=proof, inputs=list(before)) == (
        "written",
        "written",
    )
    assert merge.write_final_matrix(adapter, provenance, output=output, provenance_output=proof, inputs=list(before)) == (
        "resumed",
        "resumed",
    )


def test_final_matrix_merger_accepts_backend_matched_bct_main_hf_alias():
    assert merge._canonical_condition("bct-main-hf-peft", origin="native-HF BCT main") == "bct"


def test_final_matrix_merger_rejects_backend_mismatched_bct_pair(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path, monkeypatch)
    main_preflight = next(
        path
        for path in inputs["preflights"]
        if json.loads(path.read_text())["condition"] == "bct-main-hf-peft"
    )
    document = json.loads(main_preflight.read_text())
    document["contract"].pop("expected_checkpoint")
    document["contract"].pop("expected_max_connections")
    document["contract"].pop("checkpoint_artifact_identity")
    document["contract"]["runtime_profile"] = "vllm"
    document["contract"]["expected_checkpoint"] = "/checkpoints/bct-main-vllm-compat"
    document["contract"]["vllm_compatibility_adapter_identity"] = {
        "adapter_model_sha256": _sha("vllm-adapter-model/bct-main"),
        "adapter_config_sha256": _sha("vllm-adapter-config/bct-main"),
        "compatibility_manifest_sha256": _sha("vllm-compatibility-manifest/bct-main"),
        "parity_attestation_sha256": _sha("vllm-parity-attestation/bct-main"),
        "source_adapter_model_sha256": _sha("vllm-source-adapter-model/bct-main"),
    }
    _write(main_preflight, document)

    with pytest.raises(ValueError, match="BCT main/control evidence must both use the backend-matched native HF/PEFT runtime"):
        merge.build_final_matrix(inputs["reports"], inputs["preflights"], inputs["population"])


def test_final_matrix_merger_requires_raw_bct_checkpoint_artifact_identity(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path, monkeypatch)
    main_preflight = next(
        path
        for path in inputs["preflights"]
        if json.loads(path.read_text())["condition"] == "bct-main-hf-peft"
    )
    document = json.loads(main_preflight.read_text())
    document["contract"].pop("checkpoint_artifact_identity")
    _write(main_preflight, document)

    with pytest.raises(ValueError, match="BCT main/control evidence must bind the raw adapter"):
        merge.build_final_matrix(inputs["reports"], inputs["preflights"], inputs["population"])


def test_final_matrix_merger_requires_vllm_compatibility_adapter_identity(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path, monkeypatch)
    attct_preflight = next(
        path
        for path in inputs["preflights"]
        if json.loads(path.read_text())["condition"] == "attct-vllm-compat"
    )
    document = json.loads(attct_preflight.read_text())
    document["contract"].pop("vllm_compatibility_adapter_identity")
    _write(attct_preflight, document)

    with pytest.raises(ValueError, match="adapted vLLM contract must bind compatibility-adapter"):
        merge.build_final_matrix(inputs["reports"], inputs["preflights"], inputs["population"])


def test_final_matrix_merger_rejects_raw_hash_not_bound_to_preflight(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path, monkeypatch)
    preflight = json.loads(inputs["preflights"][0].read_text())
    preflight["sources"][0]["raw_log_sha256"] = _sha("wrong raw log")
    _write(inputs["preflights"][0], preflight)

    with pytest.raises(ValueError, match="does not bind to its preflight raw-log SHA-256"):
        merge.build_final_matrix(inputs["reports"], inputs["preflights"], inputs["population"])


def test_final_matrix_merger_rejects_historical_cot_preflight(tmp_path, monkeypatch):
    """Legacy ``encourage_cot`` diagnostics cannot enter the final no-CoT matrix."""

    inputs = _inputs(tmp_path, monkeypatch)
    preflight_path = next(
        path
        for path in inputs["preflights"]
        if json.loads(path.read_text())["condition"] == "attct-vllm-compat"
    )
    preflight = json.loads(preflight_path.read_text())
    preflight["contract"]["prompt_style"] = "encourage_cot"
    for source in preflight["sources"]:
        source["prompt_style"] = "encourage_cot"
    _write(preflight_path, preflight)

    with pytest.raises(ValueError, match="contract 'prompt_style' is not pinned to the final no-CoT diagnostic"):
        merge.build_final_matrix(inputs["reports"], inputs["preflights"], inputs["population"])


def test_final_matrix_merger_requires_and_checks_accelerated_rmct_control_canonicalization(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path, monkeypatch, accelerated_rmct_control=True)

    with pytest.raises(ValueError, match="requires its canonicalization provenance"):
        merge.build_final_matrix(inputs["reports"], inputs["preflights"], inputs["population"])

    canonical_report = next(path for path in inputs["reports"] if "rmct-control" in path.read_text())
    report = json.loads(canonical_report.read_text())
    canonicalization = {
        "schema": "stage1-iid-rmct-control-canonicalization-v1",
        "alias_condition": "rmct-control-b8-accelerated",
        "canonical_condition": "rmct-control",
        "canonical_analysis_sha256": _sha256_file(canonical_report),
        "diagnostic_manifest_sha256": report["diagnostic_manifest_sha256"],
        "checks": {
            "frozen_manifest_validated": True,
            "graded_eval_log_ids_exactly_match_manifest": True,
            "alias_analysis_sources_exactly_match_graded_eval_logs": True,
            "pooled_and_per_dataset_cells_equivalent_except_condition_label": True,
            "raw_or_graded_logs_modified": False,
        },
    }
    canonicalization_path = _write(tmp_path / "rmct-control-canonicalization.json", canonicalization)
    adapter, provenance = merge.build_final_matrix(
        inputs["reports"],
        inputs["preflights"],
        inputs["population"],
        rmct_control_canonicalization=canonicalization_path,
    )
    plot.validate_report(adapter)
    assert provenance["checks"]["accelerated_rmct_control_canonicalization_attested"] is True


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
