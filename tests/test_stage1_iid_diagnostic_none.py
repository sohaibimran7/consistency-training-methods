"""Contracts for the isolated no-CoT 100+100 Stage 1 diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ctm_data.adapters.mcq_bias.recover_legacy_cot_to_none import (
    LEGACY_G4_WRONG_ARGUMENT_TEMPLATE,
    NONE_ANSWER_FORMAT_TERMINAL,
)
from experiments.rmct_paper_vast_dense_models.stage1 import supervised_recovery_none_prepare as recovery
from experiments.stage1_iid_diagnostic import gate_analysis
from experiments.stage1_iid_diagnostic_none import config
from experiments.stage1_iid_diagnostic_none import grade_luna as none_grade_luna
from experiments.stage1_iid_diagnostic_none import prepare
from experiments.stage1_iid_diagnostic_none import raw_preflight


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)


def _none_row(index: int) -> dict:
    dataset = "logiqa" if index % 2 == 0 else "hellaswag"
    question = f"Question {index}?\n\nAnswer choices:\n(A) correct\n(B) wrong"
    argument = f"Stored wrong argument {index}."
    return {
        "question": question,
        "question_id": f"q-{index:04d}",
        "source_dataset": dataset,
        "prompt_style": "none",
        "unbiased_messages": [{"role": "user", "content": question + NONE_ANSWER_FORMAT_TERMINAL}],
        "biased_messages": [
            {
                "role": "user",
                "content": LEGACY_G4_WRONG_ARGUMENT_TEMPLATE.format(argument=argument, question=question)
                + NONE_ANSWER_FORMAT_TERMINAL,
            }
        ],
        "bias_type": "wrong_argument",
        "ground_truth": "A",
        "biased_option": "B",
        "biasing_text": argument,
    }


def _write_attested_none_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, list[dict]]:
    rows = [_none_row(index) for index in range(prepare.SOURCE_ROWS)]
    source = tmp_path / "recovered-none.jsonl"
    payload = _jsonl(rows)
    source.write_bytes(payload)
    source_hash = _sha256(payload)
    legacy_hash = "a" * 64
    monkeypatch.setattr(prepare, "SOURCE_SHA256", source_hash)
    monkeypatch.setattr(recovery, "RECOVERED_NONE_SOURCE_SHA256", source_hash)
    monkeypatch.setattr(recovery, "LEGACY_COT_SOURCE_SHA256", legacy_hash)
    manifest = {
        "schema_version": recovery.RECOVERY_SCHEMA_VERSION,
        "kind": recovery.RECOVERED_MANIFEST_KIND,
        "transform": {
            "version": recovery.RECOVERY_TRANSFORM_VERSION,
            "source_prompt_style": "encourage_cot",
            "target_prompt_style": "none",
            "bias_type": "wrong_argument",
        },
        "source": {"content_sha256": legacy_hash, "row_count": prepare.SOURCE_ROWS},
        "output": {"content_sha256": source_hash, "row_count": prepare.SOURCE_ROWS},
        "selection": {
            "row_count": prepare.SOURCE_ROWS,
            "source_dataset_counts": prepare.SOURCE_COUNTS,
        },
    }
    source_manifest = tmp_path / "recovered-none.manifest.json"
    source_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return source, source_manifest, rows


def test_prepare_freezes_attested_none_offsets_and_source_bytes(tmp_path, monkeypatch):
    source, source_manifest, rows = _write_attested_none_source(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic-none"

    manifest = prepare.prepare_iid_diagnostic_none(source, source_manifest, output)

    train = manifest["splits"]["train_eval"]
    heldout = manifest["splits"]["heldout_in_domain"]
    assert train["question_ids"] == [row["question_id"] for row in rows[:200]]
    assert heldout["question_ids"] == [row["question_id"] for row in rows[2048:2248]]
    assert train["counts_by_dataset"] == {"logiqa": 100, "hellaswag": 100}
    assert heldout["counts_by_dataset"] == {"logiqa": 100, "hellaswag": 100}
    assert set(train["question_ids"]).isdisjoint(heldout["question_ids"])
    assert manifest["source"]["prompt_style"] == "none"
    assert manifest["source"]["content_sha256"] == _sha256(source.read_bytes())
    assert manifest["rmct_first64"]["question_ids"] == [row["question_id"] for row in rows[:64]]
    assert Path(train["path"]).read_bytes().splitlines()[0] == source.read_bytes().splitlines()[0]

    checked = prepare.validate_manifest(output / prepare.DEFAULT_MANIFEST_FILENAME, verify_source=True)
    assert checked == manifest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare.prepare_iid_diagnostic_none(source, source_manifest, output)


def test_prepare_rejects_source_without_the_attested_none_prompt_body(tmp_path, monkeypatch):
    source, source_manifest, rows = _write_attested_none_source(tmp_path, monkeypatch)
    rows[0]["prompt_style"] = "encourage_cot"
    altered_payload = _jsonl(rows)
    source.write_bytes(altered_payload)
    altered_hash = _sha256(altered_payload)
    monkeypatch.setattr(prepare, "SOURCE_SHA256", altered_hash)
    monkeypatch.setattr(recovery, "RECOVERED_NONE_SOURCE_SHA256", altered_hash)
    document = json.loads(source_manifest.read_text())
    document["output"]["content_sha256"] = altered_hash
    source_manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="none-style wrong_argument row"):
        prepare.prepare_iid_diagnostic_none(source, source_manifest, tmp_path / "out")


def test_validate_manifest_rejects_rehashed_split_that_no_longer_matches_source(tmp_path, monkeypatch):
    source, source_manifest, _ = _write_attested_none_source(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic-none"
    manifest = prepare.prepare_iid_diagnostic_none(source, source_manifest, output)
    train_path = Path(manifest["splits"]["train_eval"]["path"])
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["question"] = "substituted question with the same ID"
    train_payload = _jsonl(rows)
    train_path.write_bytes(train_payload)
    manifest_path = output / prepare.DEFAULT_MANIFEST_FILENAME
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["splits"]["train_eval"]["content_sha256"] = _sha256(train_payload)
    document["splits"]["train_eval"]["byte_count"] = len(train_payload)
    manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Hashes/IDs alone are insufficient; the final task path requires this
    # stronger source-byte verification.
    prepare.validate_manifest(manifest_path, verify_source=False)
    with pytest.raises(ValueError, match="train_eval content does not byte-match source rows 1-200"):
        prepare.validate_manifest(manifest_path, verify_source=True)


def test_task_factory_uses_only_none_contract_and_exact_selected_ids(tmp_path, monkeypatch):
    source, source_manifest, _ = _write_attested_none_source(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic-none"
    manifest = prepare.prepare_iid_diagnostic_none(source, source_manifest, output)
    from experiments.stage1_iid_diagnostic_none import tasks

    calls: list[tuple[str, dict]] = []

    def fake_unbiased(**kwargs):
        calls.append(("clean", kwargs))
        return SimpleNamespace(kind="clean", kwargs=kwargs)

    def fake_biased(**kwargs):
        calls.append(("biased", kwargs))
        return SimpleNamespace(kind="biased", kwargs=kwargs)

    monkeypatch.setattr(tasks, "stage1_iid_unbiased", fake_unbiased)
    monkeypatch.setattr(tasks, "stage1_iid_biased", fake_biased)
    result = tasks.diagnostic_tasks(
        manifest=output / prepare.DEFAULT_MANIFEST_FILENAME,
        split="heldout_in_domain",
        unbiased_log="logs/final-none/heldout",
    )

    assert [item.kind for item in result] == ["clean", "clean", "biased", "biased"]
    assert [kwargs["dataset"] for _, kwargs in calls] == ["logiqa", "hellaswag", "logiqa", "hellaswag"]
    assert all(kwargs["prompt_style"] == "none" for _, kwargs in calls)
    assert all(kwargs["source_sha256"] == manifest["source"]["content_sha256"] for _, kwargs in calls)
    assert all(len(kwargs["question_ids_from"]) == 100 for _, kwargs in calls)
    assert set(calls[0][1]["question_ids_from"]).isdisjoint(calls[1][1]["question_ids_from"])
    assert all(kwargs.get("variant_file") is None for _, kwargs in calls)
    with pytest.raises(ValueError, match="prompt_style"):
        tasks.diagnostic_tasks("unused", "train_eval", "logs", prompt_style="encourage_cot")


def test_pinned_config_revalidates_manifest_and_preserves_native_reasoning_mode(tmp_path, monkeypatch):
    source, source_manifest, _ = _write_attested_none_source(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic-none"
    prepare.prepare_iid_diagnostic_none(source, source_manifest, output)

    args = config.task_args(
        manifest=output / prepare.DEFAULT_MANIFEST_FILENAME,
        split="train_eval",
        unbiased_log="logs/final-none/train_eval",
    )

    assert args["prompt_style"] == "none"
    assert config.TASK_FACTORY == "experiments.stage1_iid_diagnostic_none.tasks:diagnostic_tasks"
    assert config.GENERATION_CONFIG["max_tokens"] == 20480
    assert "enable_thinking" not in config.GENERATION_CONFIG
    assert config.VLLM_MODEL_ARGS["provider"] == "vllm"


def _loaded_none_log(
    tmp_path: Path,
    *,
    split: str,
    dataset: str,
    ids: tuple[str, ...],
    source_sha256: str,
    prompt_style: str = "none",
) -> gate_analysis.LoadedLog:
    path = tmp_path / "raw" / split / f"{dataset}.eval"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{split}/{dataset}".encode())
    header = gate_analysis.LogHeader(
        prompt_variant="native",
        split=split,
        dataset=dataset,
        question_ids=ids,
        variant_file=None,
        unbiased_log=f"/clean/{split}/{dataset}.eval",
        prompt_style=prompt_style,
        source_identity_digest=source_sha256,
        created="2026-08-01T00:00:00Z",
    )
    return gate_analysis.LoadedLog(
        header=header,
        path=path,
        sha256=_sha256(path.read_bytes()),
        rows=tuple(object() for _ in ids),
    )


def test_raw_preflight_binds_later_grading_only_to_none_logs(tmp_path, monkeypatch):
    source, source_manifest, _ = _write_attested_none_source(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic-none"
    manifest = prepare.prepare_iid_diagnostic_none(source, source_manifest, output)
    source_hash = manifest["source"]["content_sha256"]
    monkeypatch.setattr(raw_preflight, "SOURCE_SHA256", source_hash)

    loaded: dict[tuple[str, str], gate_analysis.LoadedLog] = {}
    for split in gate_analysis.SPLITS:
        rows = [
            json.loads(line)
            for line in Path(manifest["splits"][split]["path"]).read_text(encoding="utf-8").splitlines()
        ]
        for dataset in prepare.DATASETS:
            ids = tuple(row["question_id"] for row in rows if row["source_dataset"] == dataset)
            loaded[(split, dataset)] = _loaded_none_log(
                tmp_path,
                split=split,
                dataset=dataset,
                ids=ids,
                source_sha256=source_hash,
            )
    monkeypatch.setattr(gate_analysis, "scan_variant_logs", lambda *_args, **_kwargs: loaded)

    report = raw_preflight.preflight_raw_logs(
        tmp_path / "raw",
        output / prepare.DEFAULT_MANIFEST_FILENAME,
        split_files={
            split: manifest["splits"][split]["path"]
            for split in ("train_eval", "heldout_in_domain")
        },
        condition="attct",
    )

    assert report["schema"] == raw_preflight.PREFLIGHT_SCHEMA
    assert report["contract"]["prompt_style"] == "none"
    assert report["contract"]["source_sha256"] == source_hash
    assert {source["sample_count"] for source in report["sources"]} == {100}

    first_key = ("train_eval", "logiqa")
    original = loaded[first_key]
    loaded[first_key] = _loaded_none_log(
        tmp_path,
        split=first_key[0],
        dataset=first_key[1],
        ids=original.header.question_ids,
        source_sha256=source_hash,
        prompt_style="encourage_cot",
    )
    with pytest.raises(ValueError, match="unexpected prompt style"):
        raw_preflight.preflight_raw_logs(
            tmp_path / "raw",
            output / prepare.DEFAULT_MANIFEST_FILENAME,
            split_files={
                split: manifest["splits"][split]["path"]
                for split in ("train_eval", "heldout_in_domain")
            },
            condition="attct",
        )


def test_hf_peft_preflight_binds_the_exact_raw_adapter_artifacts(tmp_path, monkeypatch):
    source, source_manifest, _ = _write_attested_none_source(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic-none"
    manifest = prepare.prepare_iid_diagnostic_none(source, source_manifest, output)
    source_hash = manifest["source"]["content_sha256"]
    monkeypatch.setattr(raw_preflight, "SOURCE_SHA256", source_hash)

    loaded: dict[tuple[str, str], gate_analysis.LoadedLog] = {}
    for split in gate_analysis.SPLITS:
        rows = [
            json.loads(line)
            for line in Path(manifest["splits"][split]["path"]).read_text(encoding="utf-8").splitlines()
        ]
        for dataset in prepare.DATASETS:
            loaded[(split, dataset)] = _loaded_none_log(
                tmp_path,
                split=split,
                dataset=dataset,
                ids=tuple(row["question_id"] for row in rows if row["source_dataset"] == dataset),
                source_sha256=source_hash,
            )
    monkeypatch.setattr(gate_analysis, "scan_variant_logs", lambda *_args, **_kwargs: loaded)

    checkpoint = tmp_path / "raw-adapter"
    checkpoint.mkdir()
    adapter_model = checkpoint / "adapter_model.safetensors"
    adapter_config = checkpoint / "adapter_config.json"
    checkpoint_manifest = checkpoint / "manifest.json"
    adapter_model.write_bytes(b"adapter-tensors")
    adapter_config.write_bytes(b'{"base_model_name_or_path":"Qwen/Qwen3.5-9B"}\n')
    checkpoint_manifest.write_bytes(b'{"model":"Qwen/Qwen3.5-9B","lora":true}\n')
    expected = {
        "adapter_model_sha256": _sha256(adapter_model.read_bytes()),
        "adapter_config_sha256": _sha256(adapter_config.read_bytes()),
        "checkpoint_manifest_sha256": _sha256(checkpoint_manifest.read_bytes()),
    }
    monkeypatch.setattr(
        raw_preflight,
        "_assert_expected_hf_peft_model",
        lambda *_args, **_kwargs: (
            "hf/Qwen/Qwen3.5-9B",
            {
                "profile": "hf-peft",
                "provider": "hf",
                "device": "cuda:0",
                "dtype": "bfloat16",
                "max_connections": 8,
                "checkpoint": str(checkpoint),
                "checkpoint_backend": "local",
                "base_model": "Qwen/Qwen3.5-9B",
            },
        ),
    )

    kwargs = {
        "split_files": {split: manifest["splits"][split]["path"] for split in ("train_eval", "heldout_in_domain")},
        "condition": "bct-main-hf-peft",
        "expected_base_model": "Qwen/Qwen3.5-9B",
        "expected_checkpoint": str(checkpoint),
        "runtime_profile": "hf-peft",
        "expected_max_connections": 8,
        **{f"expected_{key}": value for key, value in expected.items()},
    }
    report = raw_preflight.preflight_raw_logs(tmp_path / "raw", output / prepare.DEFAULT_MANIFEST_FILENAME, **kwargs)
    assert report["contract"]["checkpoint_artifact_identity"] == expected

    adapter_model.write_bytes(b"different-adapter-tensors")
    with pytest.raises(ValueError, match="adapter_model.safetensors SHA-256"):
        raw_preflight.preflight_raw_logs(tmp_path / "raw", output / prepare.DEFAULT_MANIFEST_FILENAME, **kwargs)

    with pytest.raises(ValueError, match="supply all of"):
        raw_preflight.preflight_raw_logs(
            tmp_path / "raw",
            output / prepare.DEFAULT_MANIFEST_FILENAME,
            **{key: value for key, value in kwargs.items() if key != "expected_adapter_config_sha256"},
        )


def test_vllm_preflight_binds_verified_compatibility_adapter_and_parity_artifacts(tmp_path, monkeypatch):
    source, source_manifest, _ = _write_attested_none_source(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic-none"
    manifest = prepare.prepare_iid_diagnostic_none(source, source_manifest, output)
    source_hash = manifest["source"]["content_sha256"]
    monkeypatch.setattr(raw_preflight, "SOURCE_SHA256", source_hash)

    loaded: dict[tuple[str, str], gate_analysis.LoadedLog] = {}
    for split in gate_analysis.SPLITS:
        rows = [
            json.loads(line)
            for line in Path(manifest["splits"][split]["path"]).read_text(encoding="utf-8").splitlines()
        ]
        for dataset in prepare.DATASETS:
            loaded[(split, dataset)] = _loaded_none_log(
                tmp_path,
                split=split,
                dataset=dataset,
                ids=tuple(row["question_id"] for row in rows if row["source_dataset"] == dataset),
                source_sha256=source_hash,
            )
    monkeypatch.setattr(gate_analysis, "scan_variant_logs", lambda *_args, **_kwargs: loaded)

    checkpoint = tmp_path / "vllm-compat-adapter"
    checkpoint.mkdir()
    adapter_model = checkpoint / "adapter_model.safetensors"
    adapter_config = checkpoint / "adapter_config.json"
    compatibility_manifest = checkpoint / "compatibility-manifest.json"
    parity_attestation = checkpoint / "vllm-parity-attestation.json"
    parity_report = tmp_path / "runtime-parity.json"
    adapter_model.write_bytes(b"translated-adapter-tensors")
    adapter_config.write_bytes(b'{"base_model_name_or_path":"Qwen/Qwen3.5-9B"}\n')
    raw_adapter_digest = _sha256(b"raw-adapter-tensors")
    compatibility_manifest.write_text(
        json.dumps(
            {
                "schema": "qwen35-vllm-compat-adapter-v1",
                "source": {
                    "path": str((tmp_path / "raw-adapter").resolve()),
                    "adapter_model_sha256": raw_adapter_digest,
                },
                "destination": {
                    "path": str(checkpoint.resolve()),
                    "adapter_model_sha256": _sha256(adapter_model.read_bytes()),
                },
                "translation": {
                    "source_prefix": "base_model.model.model.layers.",
                    "destination_prefix": "base_model.model.model.language_model.layers.",
                    "tensor_count": 4,
                    "translated_tensor_count": 4,
                    "translated_tensor_names_sha256": _sha256(b"translated-names"),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    variants = ["full", "linear_only", "self_attn_only", "evaluator_path"]
    parity_report.write_text(
        json.dumps(
            {
                "adapter": {
                    "path": str(checkpoint.resolve()),
                    "adapter_model_sha256": _sha256(adapter_model.read_bytes()),
                },
                "results": {variant: {"verdict": "hf_vllm_effects_agree"} for variant in variants},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    parity_report_payload = parity_report.read_bytes()
    parity_attestation.write_text(
        json.dumps(
            {
                "schema": "qwen35-vllm-parity-attestation-v1",
                "adapter_path": str(checkpoint.resolve()),
                "adapter_model_sha256": _sha256(adapter_model.read_bytes()),
                "model": "Qwen/Qwen3.5-9B",
                "report_path": str(parity_report.resolve()),
                "report_sha256": _sha256(parity_report_payload),
                "required_variants": variants,
                "verdicts": {variant: "hf_vllm_effects_agree" for variant in variants},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        raw_preflight,
        "_assert_expected_model",
        lambda *_args, **_kwargs: f"vllm/Qwen/Qwen3.5-9B:{checkpoint}",
    )

    kwargs = {
        "split_files": {split: manifest["splits"][split]["path"] for split in ("train_eval", "heldout_in_domain")},
        "condition": "attct-vllm-compat",
        "expected_base_model": "Qwen/Qwen3.5-9B",
        "expected_checkpoint": str(checkpoint),
        "runtime_profile": "vllm",
    }
    report = raw_preflight.preflight_raw_logs(tmp_path / "raw", output / prepare.DEFAULT_MANIFEST_FILENAME, **kwargs)
    expected = {
        "adapter_model_sha256": _sha256(adapter_model.read_bytes()),
        "adapter_config_sha256": _sha256(adapter_config.read_bytes()),
        "compatibility_manifest_sha256": _sha256(compatibility_manifest.read_bytes()),
        "parity_attestation_sha256": _sha256(parity_attestation.read_bytes()),
        "source_adapter_model_sha256": raw_adapter_digest,
    }
    assert report["contract"]["vllm_compatibility_adapter_identity"] == expected

    parity_report.write_text('{"tampered":"parity-report"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="valid immutable Qwen3.5 HF/vLLM parity attestation"):
        raw_preflight.preflight_raw_logs(tmp_path / "raw", output / prepare.DEFAULT_MANIFEST_FILENAME, **kwargs)

    parity_report.write_bytes(parity_report_payload)
    manifest_document = json.loads(compatibility_manifest.read_text())
    manifest_document["destination"]["adapter_model_sha256"] = _sha256(b"wrong-adapter")
    compatibility_manifest.write_text(json.dumps(manifest_document), encoding="utf-8")
    with pytest.raises(ValueError, match="destination adapter SHA-256"):
        raw_preflight.preflight_raw_logs(tmp_path / "raw", output / prepare.DEFAULT_MANIFEST_FILENAME, **kwargs)


def test_bct_hf_recovery_wrappers_pin_main_and_control_raw_checkpoint_bytes():
    root = Path(__file__).parent.parent
    expected = {
        "ctm_bct_main_hf_recovery_20260802.sh": {
            "adapter": "b7b6d2545797894e9f976f497ee6e5c8b09c26a3b2185aa36be82e4c308285ac",
            "config": "b29d45fb65363175b99a9230d4bf33adefddf26e155e7b13d33d02c96e6319a9",
        },
        "ctm_bct_control_hf_recovery_20260802.sh": {
            "adapter": "84efdc5b38f488ad49ef3659518408ec4f93cf624bf6f30b535cbcf897ad19f5",
            "config": "a3814737831811f7b46f7df475cb220bec20c80f803f1af6c0ce9cd7d98f88ac",
        },
    }
    shared_manifest = "8612a1651547ebc94348036d93bfdc71dc07f7e91ba9194c67fdc173182a5b2d"
    for name, identities in expected.items():
        text = (root / "scripts" / name).read_text(encoding="utf-8")
        assert identities["adapter"] in text
        assert identities["config"] in text
        assert shared_manifest in text
        assert "--expected-adapter-model-sha256 \"$ADAPTER_MODEL_SHA256\"" in text
        assert "--expected-adapter-config-sha256 \"$ADAPTER_CONFIG_SHA256\"" in text
        assert "--expected-checkpoint-manifest-sha256 \"$CHECKPOINT_MANIFEST_SHA256\"" in text

    watcher = (root / "scripts" / "ctm_bct_hf_offhost_watcher_20260802.sh").read_text(encoding="utf-8")
    for identities in expected.values():
        assert identities["adapter"] in watcher
        assert identities["config"] in watcher
    assert shared_manifest in watcher
    assert "--expected-adapter-model-sha256 \"$EXPECTED_ADAPTER_MODEL_SHA256\"" in watcher
    assert "--expected-adapter-config-sha256 \"$EXPECTED_ADAPTER_CONFIG_SHA256\"" in watcher
    assert "--expected-checkpoint-manifest-sha256 \"$EXPECTED_CHECKPOINT_MANIFEST_SHA256\"" in watcher
    raw_preflight_path = "experiments/stage1_iid_diagnostic_none/raw_preflight.py"
    raw_preflight_hash = _sha256((root / raw_preflight_path).read_bytes())
    assert f'"{raw_preflight_path}": "{raw_preflight_hash}"' in watcher


def test_bct_hf_recovery_wrappers_are_portable_and_backend_matched():
    root = Path(__file__).parent.parent
    wrappers = {
        "ctm_bct_main_hf_recovery_20260802.sh": "BCT_MAIN",
        "ctm_bct_control_hf_recovery_20260802.sh": "BCT_CONTROL",
    }
    shared_model_args = '''--model-args '{"provider":"hf","device":"cuda:0","dtype":"bfloat16"}' '''
    shared_generation = '''--generation-config '{"max_tokens":20480,"temperature":1.0,"top_p":0.95,"top_k":20,"max_connections":8}' '''
    for name, prefix in wrappers.items():
        text = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "HOST_ROOT=${CTM_BCT_HOST_ROOT:-/workspace}" in text
        assert f'EVAL_ROOT=${{{prefix}_EVAL_ROOT:-"$REPO/logs/evals/' in text
        assert f'STAGE=${{{prefix}_STAGE:-"$REPO/artifacts/' in text
        assert f'RUNNER_ROOT=${{{prefix}_RUNNER_ROOT:-"$RUN/runners/' in text
        assert '"$EVAL_ROOT" "$pair"' in text
        assert '"$REPO/$EVAL_ROOT"' not in text
        assert "--isolate-tasks" in text
        assert shared_model_args in text
        assert shared_generation in text
        assert "--task-index 1 --task-index 3" in text
        assert "--task-index 2 --task-index 4" in text


def test_luna_handoff_requires_the_none_preflight_contract(tmp_path, monkeypatch):
    source, source_manifest, _ = _write_attested_none_source(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic-none"
    manifest = prepare.prepare_iid_diagnostic_none(source, source_manifest, output)
    source_hash = manifest["source"]["content_sha256"]
    monkeypatch.setattr(raw_preflight, "SOURCE_SHA256", source_hash)
    monkeypatch.setattr(none_grade_luna, "SOURCE_SHA256", source_hash)
    loaded: dict[tuple[str, str], gate_analysis.LoadedLog] = {}
    for split in gate_analysis.SPLITS:
        rows = [
            json.loads(line)
            for line in Path(manifest["splits"][split]["path"]).read_text(encoding="utf-8").splitlines()
        ]
        for dataset in prepare.DATASETS:
            loaded[(split, dataset)] = _loaded_none_log(
                tmp_path,
                split=split,
                dataset=dataset,
                ids=tuple(row["question_id"] for row in rows if row["source_dataset"] == dataset),
                source_sha256=source_hash,
            )
    monkeypatch.setattr(gate_analysis, "scan_variant_logs", lambda *_args, **_kwargs: loaded)
    report = raw_preflight.preflight_raw_logs(
        tmp_path / "raw",
        output / prepare.DEFAULT_MANIFEST_FILENAME,
        split_files={
            split: manifest["splits"][split]["path"]
            for split in ("train_eval", "heldout_in_domain")
        },
        condition="mlpct",
    )
    report_path = tmp_path / "none-preflight.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    observed: dict[str, object] = {}

    def fake_grade_all(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return []

    monkeypatch.setattr(none_grade_luna.shared_grade_luna, "grade_all", fake_grade_all)
    assert none_grade_luna.grade_all(
        tmp_path / "staged-none-logs",
        tmp_path / "derived-none-luna",
        preflight_report=report_path,
    ) == []
    assert observed["kwargs"]["preflight_report"] == report_path

    report["contract"]["prompt_style"] = "encourage_cot"
    with pytest.raises(ValueError, match="final no-CoT contract"):
        none_grade_luna.validate_none_preflight_report(report)
