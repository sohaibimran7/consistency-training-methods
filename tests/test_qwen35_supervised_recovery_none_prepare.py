"""Offline provenance tests for the Qwen3.5 no-CoT SFT recovery inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ctm.training.bct_targets import BCT_TARGET_SCHEMA_VERSION
from ctm_data.adapters.mcq_bias.recover_legacy_cot_to_none import (
    LEGACY_G4_WRONG_ARGUMENT_TEMPLATE,
    NONE_ANSWER_FORMAT_TERMINAL,
)
from experiments.rmct_paper_vast_dense_models.stage1 import supervised_recovery_none_prepare as prep


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


def _write_recovered_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, list[dict]]:
    rows = [_none_row(index) for index in range(prep.RECOVERED_NONE_ROWS)]
    source = tmp_path / "recovered-none.jsonl"
    payload = _jsonl(rows)
    source.write_bytes(payload)
    source_hash = _sha256(payload)
    legacy_hash = "a" * 64
    monkeypatch.setattr(prep, "RECOVERED_NONE_SOURCE_SHA256", source_hash)
    monkeypatch.setattr(prep, "LEGACY_COT_SOURCE_SHA256", legacy_hash)
    manifest = {
        "schema_version": prep.RECOVERY_SCHEMA_VERSION,
        "kind": prep.RECOVERED_MANIFEST_KIND,
        "transform": {
            "version": prep.RECOVERY_TRANSFORM_VERSION,
            "source_prompt_style": "encourage_cot",
            "target_prompt_style": "none",
            "bias_type": "wrong_argument",
        },
        "source": {"content_sha256": legacy_hash, "row_count": prep.RECOVERED_NONE_ROWS},
        "output": {"content_sha256": source_hash, "row_count": prep.RECOVERED_NONE_ROWS},
        "selection": {
            "row_count": prep.RECOVERED_NONE_ROWS,
            "source_dataset_counts": prep.RECOVERED_COUNTS,
        },
    }
    manifest_path = tmp_path / "recovered-none.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, manifest_path, rows


def _write_frozen_instruction_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    rows = [
        {
            "source_id": f"instruction-{index:04d}",
            "messages": [
                {"role": "user", "content": f"Instruction {index}"},
                {"role": "assistant", "content": f"Base completion {index}"},
            ],
        }
        for index in range(prep.FROZEN_INSTRUCTION_ROWS)
    ]
    payload = _jsonl(rows)
    target_hash = _sha256(payload)
    source_hash = "b" * 64
    monkeypatch.setattr(prep, "FROZEN_INSTRUCTION_TARGET_SHA256", target_hash)
    monkeypatch.setattr(prep, "FROZEN_INSTRUCTION_SOURCE_SHA256", source_hash)

    main = tmp_path / "instruction-targets.jsonl"
    control = tmp_path / "instruction-targets-control.jsonl"
    main.write_bytes(payload)
    control.write_bytes(payload)
    manifest = {
        "schema_version": BCT_TARGET_SCHEMA_VERSION,
        "kind": "ctm_bct_targets",
        "model": prep.MODEL,
        "backend": "FrozenBaseVLLMBackend",
        "row_count": prep.FROZEN_INSTRUCTION_ROWS,
        "generation": prep.TARGET_GENERATION,
        "fields": {
            "source_messages": "reference_messages",
            "main_messages": "variant_messages",
            "control_messages": "reference_messages",
        },
        "source_files": [{"content_sha256": source_hash, "row_count": prep.FROZEN_INSTRUCTION_ROWS}],
        "outputs": {
            "main": {"content_sha256": target_hash},
            "control": {"content_sha256": target_hash},
        },
    }
    manifest_path = tmp_path / "instruction-targets.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return main, control, manifest_path


def _write_fresh_bct_targets(tmp_path: Path, source: Path, rows: list[dict]) -> tuple[Path, Path, Path]:
    main_rows: list[dict] = []
    control_rows: list[dict] = []
    for index, row in enumerate(rows[: prep.STANDARD_TRAINING_ROWS]):
        assistant = {"role": "assistant", "content": f"Frozen base completion {index}"}
        main_rows.append({"source_id": row["question_id"], "messages": [*row["biased_messages"], assistant]})
        control_rows.append({"source_id": row["question_id"], "messages": [*row["unbiased_messages"], assistant]})
    main_payload, control_payload = _jsonl(main_rows), _jsonl(control_rows)
    main = tmp_path / "fresh-bct-main.jsonl"
    control = tmp_path / "fresh-bct-control.jsonl"
    main.write_bytes(main_payload)
    control.write_bytes(control_payload)
    manifest = {
        "schema_version": BCT_TARGET_SCHEMA_VERSION,
        "kind": "ctm_bct_targets",
        "model": prep.MODEL,
        "backend": "FrozenBaseVLLMBackend",
        "row_count": prep.STANDARD_TRAINING_ROWS,
        "generation": prep.FRESH_BCT_TARGET_GENERATION,
        "fields": {
            "source_messages": "unbiased_messages",
            "main_messages": "biased_messages",
            "control_messages": "unbiased_messages",
        },
        "source_files": [
            {
                "content_sha256": _sha256(source.read_bytes()),
                "row_count": prep.RECOVERED_NONE_ROWS,
            }
        ],
        "outputs": {
            "main": {"content_sha256": _sha256(main_payload)},
            "control": {"content_sha256": _sha256(control_payload)},
        },
    }
    manifest_path = tmp_path / "fresh-bct.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return main, control, manifest_path


def test_prepare_rebuilds_canonical_none_artifacts_and_reuses_only_verified_instruction_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, source_manifest, rows = _write_recovered_source(tmp_path, monkeypatch)
    instruction_main, instruction_control, instruction_manifest = _write_frozen_instruction_targets(
        tmp_path, monkeypatch
    )
    output = tmp_path / "prepared"

    first = prep.prepare_supervised_recovery_inputs(
        source=source,
        source_manifest=source_manifest,
        instruction_main=instruction_main,
        instruction_control=instruction_control,
        instruction_manifest=instruction_manifest,
        output_dir=output,
    )
    second = prep.prepare_supervised_recovery_inputs(
        source=source,
        source_manifest=source_manifest,
        instruction_main=instruction_main,
        instruction_control=instruction_control,
        instruction_manifest=instruction_manifest,
        output_dir=output,
    )

    assert set(first["statuses"]) == {
        "repaired_act_train",
        "repaired_act_heldout",
        "repaired_act_splits_manifest",
        "standard_canonical",
        "repaired_act_train_canonical",
        "repaired_act_heldout_canonical",
        "prepared_inputs_manifest",
    }
    assert all(status == "written" for status in first["statuses"].values())
    assert all(status == "resumed" for status in second["statuses"].values())
    assert (output / "canonical-consistency-pairs-n2048.jsonl").is_file()
    assert (output / "canonical-repaired-act-train-n200.jsonl").is_file()
    assert (output / "canonical-repaired-act-heldout-n200.jsonl").is_file()
    prepared = json.loads((output / "supervised-recovery-inputs.manifest.json").read_text())
    assert prepared["kind"] == prep.PREPARED_INPUTS_KIND
    assert prepared["standard_training"]["counts_by_dataset"] == {"logiqa": 1024, "hellaswag": 1024}
    assert prepared["repaired_act"]["counts_by_dataset"] == {"logiqa": 100, "hellaswag": 100}
    assert prepared["assertions"]["prompt_dependent_artifacts_derived_from_recovered_none_source"] is True

    canonical_rows = [
        json.loads(line) for line in (output / "canonical-consistency-pairs-n2048.jsonl").read_text().splitlines()
    ]
    assert len(canonical_rows) == prep.STANDARD_TRAINING_ROWS
    assert canonical_rows[0]["biased_messages"][0]["content"].endswith(
        canonical_rows[0]["unbiased_messages"][0]["content"]
    )
    assert canonical_rows[0]["question_id"] == rows[0]["question_id"]


def test_fresh_bct_target_verifier_rejects_a_prompt_not_bound_to_the_none_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, source_manifest, rows = _write_recovered_source(tmp_path, monkeypatch)
    main, control, target_manifest = _write_fresh_bct_targets(tmp_path, source, rows)

    identity = prep.verify_fresh_bct_targets(
        source=source,
        source_manifest=source_manifest,
        main=main,
        control=control,
        manifest_path=target_manifest,
    )
    assert identity["main"]["row_count"] == prep.STANDARD_TRAINING_ROWS

    corrupted = [json.loads(line) for line in control.read_text().splitlines()]
    corrupted[0]["messages"][0]["content"] = "old CoT prompt"
    control.write_bytes(_jsonl(corrupted))
    manifest = json.loads(target_manifest.read_text())
    manifest["outputs"]["control"]["content_sha256"] = _sha256(control.read_bytes())
    target_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="recovered none clean prompt"):
        prep.verify_fresh_bct_targets(
            source=source,
            source_manifest=source_manifest,
            main=main,
            control=control,
            manifest_path=target_manifest,
        )
