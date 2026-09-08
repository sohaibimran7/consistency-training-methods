"""CPU-only contracts for the fresh repaired-ACT pre-launch evidence chain."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.act_repair_gate.behavioral_gate import build_attestation, write_attestation
from experiments.rmct_paper_vast_dense_models.stage1 import repaired_act_evaluation_guard as guard
from experiments.stage1_iid_diagnostic import gate_analysis
from experiments.stage1_iid_diagnostic_none import raw_preflight


ROOT = Path(__file__).parent.parent
WRAPPER = ROOT / "scripts" / "ctm_repaired_act_long_eval_guard_20260803.sh"


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


def _direct_answer_report(*, adapter: Path, train: Path, heldout: Path) -> dict:
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
        "model": guard.MODEL,
        "adapter": {
            "path": str(adapter.resolve()),
            "adapter_model_sha256": _sha256(adapter / "adapter_model.safetensors"),
            "condition_name": "repaired-act",
        },
        "sources": sources,
        "conditions": {
            "untrained": {
                "cells": {
                    "train_eval": _cell(switches=2, eligible=8),
                    "heldout_in_domain": _cell(switches=1, eligible=8),
                }
            },
            "repaired-act": {
                "cells": {
                    "train_eval": _cell(switches=0, eligible=8),
                    "heldout_in_domain": _cell(switches=0, eligible=8),
                }
            },
        },
    }


def _write_vllm_compat(raw: Path, destination: Path) -> None:
    destination.mkdir()
    weights = destination / "adapter_model.safetensors"
    weights.write_bytes(b"translated-adapter-weights")
    (destination / "adapter_config.json").write_text('{"base_model_name_or_path":"Qwen/Qwen3.5-9B"}\n')
    manifest = destination / "compatibility-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "qwen35-vllm-compat-adapter-v1",
                "source": {
                    "path": str(raw.resolve()),
                    "adapter_model_sha256": _sha256(raw / "adapter_model.safetensors"),
                },
                "destination": {"path": str(destination.resolve()), "adapter_model_sha256": _sha256(weights)},
                "translation": {
                    "source_prefix": "base_model.model.model.layers.",
                    "destination_prefix": "base_model.model.model.language_model.layers.",
                    "tensor_count": 2,
                    "translated_tensor_count": 2,
                    "translated_tensor_names_sha256": "b" * 64,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = destination.parent / "runtime-parity.json"
    variants = ["full", "linear_only", "self_attn_only", "evaluator_path"]
    report.write_text(
        json.dumps(
            {
                "adapter": {"path": str(destination.resolve()), "adapter_model_sha256": _sha256(weights)},
                "results": {variant: {"verdict": "hf_vllm_effects_agree"} for variant in variants},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (destination / "vllm-parity-attestation.json").write_text(
        json.dumps(
            {
                "schema": "qwen35-vllm-parity-attestation-v1",
                "adapter_path": str(destination.resolve()),
                "adapter_model_sha256": _sha256(weights),
                "model": guard.MODEL,
                "report_path": str(report.resolve()),
                "report_sha256": _sha256(report),
                "required_variants": variants,
                "verdicts": {variant: "hf_vllm_effects_agree" for variant in variants},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    raw = tmp_path / "fresh-raw-adapter"
    raw.mkdir()
    (raw / "adapter_model.safetensors").write_bytes(b"fresh-raw-adapter-weights")
    (raw / "adapter_config.json").write_text('{"base_model_name_or_path":"Qwen/Qwen3.5-9B"}\n')
    (raw / "manifest.json").write_text(
        json.dumps({"backend": "local", "model": guard.MODEL, "lora": True}), encoding="utf-8"
    )
    state = tmp_path / "outputs.json"
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": guard.FRESH_EXPERIMENT,
                "execution_target": guard.FRESH_TARGET,
                "training_checkpoints": {guard.FRESH_TRAINING_COMMAND: f"file://{raw.resolve()}"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    train = tmp_path / "canonical-repaired-act-train-n200.jsonl"
    heldout = tmp_path / "canonical-repaired-act-heldout-n200.jsonl"
    train.write_text('{"frozen":"train"}\n', encoding="utf-8")
    heldout.write_text('{"frozen":"heldout"}\n', encoding="utf-8")
    report = tmp_path / "direct-answer-report.json"
    report.write_text(json.dumps(_direct_answer_report(adapter=raw, train=train, heldout=heldout)), encoding="utf-8")
    tiny = tmp_path / "tiny-gate-attestation.json"
    tiny_attestation = build_attestation(
        report_path=report,
        adapter=raw,
        train_data=train,
        heldout_data=heldout,
        expected_limit_per_dataset=4,
    )
    write_attestation(tiny, tiny_attestation)
    compat = tmp_path / "vllm-compat-adapter"
    _write_vllm_compat(raw, compat)
    return {"raw": raw, "state": state, "tiny": tiny, "compat": compat, "chain": tmp_path / "chain.json"}


def _build_vllm_chain(paths: dict[str, Path]) -> dict:
    return guard.build_repaired_act_evaluation_chain_attestation(
        training_output_state=paths["state"],
        checkpoint=f"file://{paths['raw'].resolve()}",
        tiny_gate_attestation=paths["tiny"],
        runtime_profile="vllm",
        vllm_compat_adapter=paths["compat"],
    )


def test_chain_binds_fresh_target_state_passed_hf_gate_and_vllm_parity(tmp_path):
    paths = _fixture(tmp_path)
    chain = _build_vllm_chain(paths)

    assert chain["fresh_training"]["training_command"] == guard.FRESH_TRAINING_COMMAND
    assert chain["raw_checkpoint"]["path"] == str(paths["raw"].resolve())
    assert chain["tiny_native_hf_gate"]["adapter_model_sha256"] == _sha256(paths["raw"] / "adapter_model.safetensors")
    compat = chain["runtime"]["vllm_compatibility_adapter"]
    assert compat["source_adapter_model_sha256"] == _sha256(paths["raw"] / "adapter_model.safetensors")
    assert compat["parity_attestation_sha256"] == _sha256(paths["compat"] / "vllm-parity-attestation.json")

    output, status = guard.write_repaired_act_evaluation_chain_attestation(paths["chain"], chain)
    assert status == "written"
    assert output == paths["chain"].resolve()
    assert guard.validate_repaired_act_evaluation_chain_attestation(
        output,
        expected_runtime_profile="vllm",
        expected_vllm_compat_adapter=paths["compat"],
    )["attestation"]["sha256"] == _sha256(output)
    _, resumed = guard.write_repaired_act_evaluation_chain_attestation(output, chain)
    assert resumed == "resumed"


def test_chain_rejects_mismatched_training_state_tiny_gate_and_compat_source(tmp_path):
    paths = _fixture(tmp_path)
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    state["training_checkpoints"][guard.FRESH_TRAINING_COMMAND] = "file:///different/raw-adapter"
    paths["state"].write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _build_vllm_chain(paths)

    state["training_checkpoints"][guard.FRESH_TRAINING_COMMAND] = f"file://{paths['raw'].resolve()}"
    paths["state"].write_text(json.dumps(state), encoding="utf-8")
    tiny = json.loads(paths["tiny"].read_text(encoding="utf-8"))
    tiny["gate"]["passed"] = False
    paths["tiny"].write_text(json.dumps(tiny), encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly revalidate|did not pass"):
        _build_vllm_chain(paths)

    # Restore an exact, passing tiny attestation and corrupt the compatibility
    # source hash. The parity document remains valid for the served copy, but
    # the copy is no longer attributable to this fresh raw checkpoint.
    report = paths["tiny"].parent / "direct-answer-report.json"
    train = paths["tiny"].parent / "canonical-repaired-act-train-n200.jsonl"
    heldout = paths["tiny"].parent / "canonical-repaired-act-heldout-n200.jsonl"
    restored = build_attestation(
        report_path=report,
        adapter=paths["raw"],
        train_data=train,
        heldout_data=heldout,
        expected_limit_per_dataset=4,
    )
    paths["tiny"].write_text(json.dumps(restored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = paths["compat"] / "compatibility-manifest.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["source"]["adapter_model_sha256"] = "0" * 64
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="source adapter hash"):
        _build_vllm_chain(paths)


def test_prelaunch_wrapper_creates_chain_before_executing_long_evaluator(tmp_path):
    paths = _fixture(tmp_path)
    marker = tmp_path / "evaluator-started"
    environment = {
        **os.environ,
        "CTM_REPAIRED_ACT_REPO": str(ROOT),
        "CTM_REPAIRED_ACT_PY": sys.executable,
        "CTM_REPAIRED_ACT_TRAINING_OUTPUT_STATE": str(paths["state"]),
        "CTM_REPAIRED_ACT_RAW_CHECKPOINT": f"file://{paths['raw'].resolve()}",
        "CTM_REPAIRED_ACT_TINY_GATE_ATTESTATION": str(paths["tiny"]),
        "CTM_REPAIRED_ACT_VLLM_COMPAT_ADAPTER": str(paths["compat"]),
        "CTM_REPAIRED_ACT_CHAIN_ATTESTATION": str(paths["chain"]),
    }
    command = [
        "bash",
        str(WRAPPER),
        "--",
        "/bin/sh",
        "-c",
        'test -f "$CTM_REPAIRED_ACT_CHAIN_ATTESTATION" && printf invoked > "$1"',
        "sh",
        str(marker),
    ]
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8") == "invoked"
    assert paths["chain"].is_file()


def test_prelaunch_wrapper_does_not_exec_evaluator_when_chain_is_invalid(tmp_path):
    paths = _fixture(tmp_path)
    marker = tmp_path / "must-not-run"
    tiny = json.loads(paths["tiny"].read_text(encoding="utf-8"))
    tiny["gate"]["passed"] = False
    paths["tiny"].write_text(json.dumps(tiny), encoding="utf-8")
    environment = {
        **os.environ,
        "CTM_REPAIRED_ACT_REPO": str(ROOT),
        "CTM_REPAIRED_ACT_PY": sys.executable,
        "CTM_REPAIRED_ACT_TRAINING_OUTPUT_STATE": str(paths["state"]),
        "CTM_REPAIRED_ACT_RAW_CHECKPOINT": str(paths["raw"]),
        "CTM_REPAIRED_ACT_TINY_GATE_ATTESTATION": str(paths["tiny"]),
        "CTM_REPAIRED_ACT_VLLM_COMPAT_ADAPTER": str(paths["compat"]),
        "CTM_REPAIRED_ACT_CHAIN_ATTESTATION": str(paths["chain"]),
    }
    result = subprocess.run(
        ["bash", str(WRAPPER), "--", "/bin/sh", "-c", 'printf bad > "$1"', "sh", str(marker)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert not marker.exists()


@pytest.mark.parametrize("condition", ("repaired-act-vllm-compat", "repaired-act-hf-peft"))
def test_fresh_repaired_act_raw_preflight_cannot_silently_omit_the_chain(condition):
    with pytest.raises(ValueError, match="requires --repaired-act-chain-attestation"):
        raw_preflight.preflight_raw_logs(
            "unused-raw-root",
            "unused-manifest.json",
            split_files={},
            condition=condition,
        )


def test_raw_preflight_revalidates_and_records_the_fresh_act_chain(tmp_path, monkeypatch):
    paths = _fixture(tmp_path)
    chain = _build_vllm_chain(paths)
    guard.write_repaired_act_evaluation_chain_attestation(paths["chain"], chain)
    source_digest = "c" * 64
    split_files = {}
    expected_ids = {}
    loaded = {}
    raw_root = tmp_path / "raw"
    for split in ("train_eval", "heldout_in_domain"):
        split_file = tmp_path / f"{split}.jsonl"
        split_file.write_text("{}\n", encoding="utf-8")
        split_files[split] = split_file
        for dataset in ("hellaswag", "logiqa"):
            question_id = f"{split}-{dataset}"
            expected_ids[(split, dataset)] = (question_id,)
            log = raw_root / split / f"{dataset}.eval"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("raw\n", encoding="utf-8")
            loaded[(split, dataset)] = gate_analysis.LoadedLog(
                header=gate_analysis.LogHeader(
                    prompt_variant="native",
                    split=split,
                    dataset=dataset,
                    question_ids=(question_id,),
                    variant_file=None,
                    unbiased_log=f"/clean/{split}/{dataset}.eval",
                    prompt_style="none",
                    source_identity_digest=source_digest,
                    created="2026-08-03T00:00:00Z",
                ),
                path=log,
                sha256=_sha256(log),
                rows=(object(),),
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    compat_identity = chain["runtime"]["vllm_compatibility_adapter"]
    monkeypatch.setattr(raw_preflight, "SOURCE_SHA256", source_digest)
    monkeypatch.setattr(raw_preflight, "_validate_runtime_profile", lambda **_kwargs: None)
    monkeypatch.setattr(raw_preflight, "_validate_checkpoint_artifact_identity", lambda **_kwargs: None)
    monkeypatch.setattr(
        raw_preflight,
        "_validate_vllm_compatibility_adapter_identity",
        lambda **_kwargs: {
            name: compat_identity[name]
            for name in (
                "adapter_model_sha256",
                "adapter_config_sha256",
                "compatibility_manifest_sha256",
                "parity_attestation_sha256",
                "source_adapter_model_sha256",
            )
        },
    )
    monkeypatch.setattr(
        raw_preflight,
        "_manifest_and_expected_ids",
        lambda *_args, **_kwargs: ({"source": {"content_sha256": source_digest}}, expected_ids),
    )
    monkeypatch.setattr(gate_analysis, "scan_variant_logs", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(raw_preflight, "_assert_expected_model", lambda *_args, **_kwargs: "vllm/fresh-act")

    report = raw_preflight.preflight_raw_logs(
        raw_root,
        manifest,
        split_files=split_files,
        condition="repaired-act-vllm-compat",
        expected_base_model=guard.MODEL,
        expected_checkpoint=str(paths["compat"]),
        runtime_profile="vllm",
        repaired_act_chain_attestation=paths["chain"],
    )

    identity = report["contract"]["repaired_act_evaluation_chain_identity"]
    assert identity["chain_attestation_path"] == str(paths["chain"].resolve())
    assert identity["chain_attestation_sha256"] == _sha256(paths["chain"])
    assert identity["raw_adapter_model_sha256"] == _sha256(paths["raw"] / "adapter_model.safetensors")
    assert identity["vllm_compatibility_adapter_sha256"] == _sha256(paths["compat"] / "adapter_model.safetensors")
