"""Focused CPU-only contracts for the fresh Aug-03 on-policy eval handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.stage1_iid_diagnostic_none import stage_luna
from scripts import launch_fresh_onpolicy_evals as launcher


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(tmp_path: Path, name: str = "checkpoint") -> Path:
    checkpoint = tmp_path / name
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"fresh-adapter")
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": launcher.MODEL}), encoding="utf-8"
    )
    (checkpoint / "manifest.json").write_text(
        json.dumps({"backend": "local", "model": launcher.MODEL, "lora": True}), encoding="utf-8"
    )
    return checkpoint


def _training_identity(tmp_path: Path, target: str) -> dict:
    definition = launcher.target_spec(target)
    checkpoint = _checkpoint(tmp_path, target)
    return {
        "target": target,
        "experiment": definition.experiment,
        "training_command": definition.training_command,
        "training_plan": {"path": str(tmp_path / "plan.yaml"), "sha256": "a" * 64, "resolved_target_plan": {"path": str(tmp_path / "resolved-plan.yaml"), "sha256": "b" * 64}},
        "target_output_state": {"path": str(tmp_path / "outputs.json"), "sha256": "c" * 64, "published_checkpoint": f"file://{checkpoint}"},
        "checkpoint": launcher._local_lora_checkpoint_identity(checkpoint),
        "authored_plan_name": definition.experiment,
    }


def _stage1_identity(tmp_path: Path) -> dict:
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    train.write_text('{"split":"train"}\n', encoding="utf-8")
    heldout.write_text('{"split":"heldout"}\n', encoding="utf-8")
    return {
        "path": str(tmp_path / "stage1-manifest.json"),
        "sha256": "1" * 64,
        "prompt_style": "none",
        "splits": {
            "train_eval": {"path": str(train), "sha256": _sha256(train), "row_count": 200, "question_ids_sha256": "2" * 64},
            "heldout_in_domain": {"path": str(heldout), "sha256": _sha256(heldout), "row_count": 200, "question_ids_sha256": "3" * 64},
        },
        "raw_task_count": 8,
        "metrics": ["conditional_tbsr", "luna_bias_acknowledgement"],
    }


def _stage2_identity(tmp_path: Path) -> dict:
    return {
        "path": str(tmp_path / "stage2-manifest.json"),
        "sha256": "4" * 64,
        "schema": "stage2-ood-hle-manifest-v1",
        "prompt_style": "none",
        "raw_task_count": 21,
        "clean_task_count": 3,
        "biased_task_count": 18,
        "populations": ["in_domain", "hle"],
        "metrics": ["conditional_tbsr", "luna_bias_acknowledgement"],
    }


def _patch_local_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str) -> dict:
    training = _training_identity(tmp_path, target)
    stage1 = _stage1_identity(tmp_path)
    stage2 = _stage2_identity(tmp_path)
    monkeypatch.setattr(launcher, "validate_target_training_output", lambda **_kwargs: training)
    monkeypatch.setattr(launcher, "validate_stage1_manifest", lambda _manifest: stage1)
    monkeypatch.setattr(launcher, "validate_stage2_manifest", lambda _manifest: stage2)
    return training


def _by_name(contract: dict) -> dict[str, dict]:
    return {command["name"]: command for command in contract["commands"]}


def test_rmct_contract_uses_fresh_native_hf_peft_and_explicit_stage_token_caps(tmp_path, monkeypatch):
    training = _patch_local_inputs(monkeypatch, tmp_path, "rmct-main")

    contract = launcher.build_launch_contract(
        target="rmct-main",
        checkpoint=training["checkpoint"]["path"],
        target_output_state=training["target_output_state"]["path"],
        stage1_manifest=tmp_path / "stage1-manifest.json",
        stage2_manifest=tmp_path / "stage2-manifest.json",
        output_root=tmp_path / "evidence",
        python="/opt/fresh/python",
    )

    assert contract["schema"] == launcher.LAUNCH_SCHEMA
    assert contract["runtime"]["profile"] == "hf-peft"
    assert contract["evaluation"]["stage1"]["required_splits"] == ["train_eval", "heldout_in_domain"]
    assert contract["evaluation"]["stage2"]["required_populations"] == ["in_domain", "hle"]
    commands = _by_name(contract)
    stage1_raw = commands["stage1_raw_generation"]["argv"]
    assert "--persistent-vllm-server" not in stage1_raw
    assert json.loads(stage1_raw[stage1_raw.index("--model-args") + 1])["provider"] == "hf"
    assert "--expected-adapter-model-sha256" in commands["stage1_raw_preflight"]["argv"]
    assert commands["stage1_luna_verbalisation"]["argv"][-1] == "1024"
    assert "--grader-max-tokens" in commands["stage1_tbsr_and_luna_analysis"]["argv"]
    analysis_tokens = commands["stage1_tbsr_and_luna_analysis"]["argv"]
    assert analysis_tokens[analysis_tokens.index("--grader-max-tokens") + 1] == "1024"
    assert analysis_tokens[-2:] == ["--output", str(tmp_path / "evidence" / "stage1" / "analysis" / "rmct-main.json")]
    assert commands["stage2_luna_verbalisation"]["argv"][-1] == "256"
    assert contract["evaluation"]["stage1"]["raw_generation_has_luna"] is False
    assert contract["evaluation"]["stage2"]["raw_generation_has_luna"] is False


def test_opct_contract_requires_translated_attested_adapter_before_emitting_vllm(tmp_path, monkeypatch):
    training = _patch_local_inputs(monkeypatch, tmp_path, "opct")
    compatibility = {
        "profile": "vllm",
        "served_compatibility_adapter": {"path": str(tmp_path / "compat"), "adapter_model_sha256": "5" * 64},
        "translation_manifest": {"path": str(tmp_path / "compatibility-manifest.json"), "sha256": "6" * 64},
        "runtime_parity": {"path": str(tmp_path / "vllm-parity-attestation.json"), "sha256": "7" * 64, "schema": "qwen35-vllm-parity-attestation-v1", "runtime_reports": []},
    }
    monkeypatch.setattr(launcher, "validate_opct_vllm_compatibility", lambda **_kwargs: compatibility)

    with pytest.raises(ValueError, match="requires --vllm-compat-adapter"):
        launcher.build_launch_contract(
            target="opct",
            checkpoint=training["checkpoint"]["path"],
            target_output_state=training["target_output_state"]["path"],
            stage1_manifest=tmp_path / "stage1-manifest.json",
            stage2_manifest=tmp_path / "stage2-manifest.json",
            output_root=tmp_path / "evidence-missing",
        )

    contract = launcher.build_launch_contract(
        target="opct",
        checkpoint=training["checkpoint"]["path"],
        target_output_state=training["target_output_state"]["path"],
        stage1_manifest=tmp_path / "stage1-manifest.json",
        stage2_manifest=tmp_path / "stage2-manifest.json",
        output_root=tmp_path / "evidence",
        vllm_compat_adapter=tmp_path / "compat",
    )
    commands = _by_name(contract)
    raw = commands["stage1_raw_generation"]["argv"]
    assert contract["runtime"]["profile"] == "vllm"
    assert contract["runtime"]["checkpoint"] == str(tmp_path / "compat")
    assert "--persistent-vllm-server" in raw
    assert json.loads(raw[raw.index("--model-args") + 1])["provider"] == "vllm"
    assert contract["runtime"]["vllm_compatibility"] == compatibility


def test_target_scoped_output_must_name_the_explicit_checkpoint_and_matching_resolved_plan(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)
    state = tmp_path / "outputs.json"
    resolved = tmp_path / "resolved-plan.yaml"
    source = tmp_path / "source.yaml"
    target = launcher.target_spec("rmct-main")
    expected_plan = "name: fresh-target\n"
    source.write_text("name: fresh-source\n", encoding="utf-8")
    monkeypatch.setattr(
        launcher,
        "_current_plan",
        lambda _target, _plan: (source, {"name": target.experiment}, {}, expected_plan),
    )
    resolved.write_text(expected_plan, encoding="utf-8")
    state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": target.experiment,
                "execution_target": target.target,
                "training_checkpoints": {target.training_command: f"file://{checkpoint}"},
            }
        ),
        encoding="utf-8",
    )

    identity = launcher.validate_target_training_output(
        target=target.target,
        checkpoint=checkpoint,
        target_output_state=state,
        target_resolved_plan=resolved,
    )
    assert identity["checkpoint"]["path"] == str(checkpoint)
    assert identity["target_output_state"]["sha256"] == _sha256(state)

    state_document = json.loads(state.read_text(encoding="utf-8"))
    state_document["training_checkpoints"][target.training_command] = "file:///somewhere-else"
    state.write_text(json.dumps(state_document), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        launcher.validate_target_training_output(
            target=target.target,
            checkpoint=checkpoint,
            target_output_state=state,
            target_resolved_plan=resolved,
        )


def test_contract_writer_is_write_once_and_refuses_a_used_new_output_root(tmp_path):
    root = tmp_path / "evidence"
    output = root / "launch.json"
    contract = {"outputs": {"root": str(root)}, "value": 1}
    launcher.validate_new_output_layout(contract, output)
    assert launcher.write_launch_contract(output, contract) == "written"
    assert launcher.write_launch_contract(output, contract) == "resumed"
    with pytest.raises(FileExistsError, match="differing"):
        launcher.write_launch_contract(output, {"outputs": {"root": str(root)}, "value": 2})

    used = tmp_path / "used"
    used.mkdir()
    (used / "old.log").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        launcher.validate_new_output_layout({"outputs": {"root": str(used)}}, used / "launch.json")
    with pytest.raises(ValueError, match="must not be nested"):
        launcher.validate_new_output_layout(
            {"outputs": {"root": str(tmp_path / "new"), "stage1_raw": str(tmp_path / "new" / "stage1" / "raw")}},
            tmp_path / "new" / "stage1" / "raw" / "launch.json",
        )


def test_written_contract_revalidates_before_generation_and_detects_tampering(tmp_path, monkeypatch):
    training = _patch_local_inputs(monkeypatch, tmp_path, "rmct-main")
    output_root = tmp_path / "evidence"
    output = output_root / "launch.json"
    contract = launcher.build_launch_contract(
        target="rmct-main",
        checkpoint=training["checkpoint"]["path"],
        target_output_state=training["target_output_state"]["path"],
        stage1_manifest=tmp_path / "stage1-manifest.json",
        stage2_manifest=tmp_path / "stage2-manifest.json",
        output_root=output_root,
        contract_output=output,
    )
    launcher.validate_new_output_layout(contract, output)
    assert launcher.write_launch_contract(output, contract) == "written"
    assert launcher.validate_launch_contract(output)["contract"] == contract
    assert contract["commands"][0]["name"] == "revalidate_fresh_contract"

    altered = json.loads(output.read_text(encoding="utf-8"))
    altered["condition"] = "different-condition"
    output.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly revalidate"):
        launcher.validate_launch_contract(output)


def test_opct_runtime_parity_report_must_name_both_the_fresh_hf_and_served_vllm_adapters(tmp_path):
    raw = launcher._local_lora_checkpoint_identity(_checkpoint(tmp_path, "raw"))
    compat = tmp_path / "compat"
    compat.mkdir()
    served_hash = "8" * 64
    report = tmp_path / "runtime-parity.json"
    report.write_text(
        json.dumps(
            {
                "schema": "qwen35-lora-runtime-parity-v1",
                "model": launcher.MODEL,
                "adapter": {
                    "path": str(compat),
                    "adapter_model_sha256": served_hash,
                    "hf_path": raw["path"],
                    "hf_adapter_model_sha256": raw["adapter_model_sha256"],
                },
            }
        ),
        encoding="utf-8",
    )
    identity = launcher._runtime_parity_report_identity(
        report,
        raw_checkpoint=raw,
        compatibility={"path": str(compat), "adapter_model_sha256": served_hash},
    )
    assert identity == {"path": str(report), "sha256": _sha256(report)}

    document = json.loads(report.read_text(encoding="utf-8"))
    document["adapter"]["hf_adapter_model_sha256"] = "0" * 64
    report.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="fresh adapter pair"):
        launcher._runtime_parity_report_identity(
            report,
            raw_checkpoint=raw,
            compatibility={"path": str(compat), "adapter_model_sha256": served_hash},
        )


def test_stage1_stager_copies_only_hash_bound_logs_and_resumes_exact_bytes(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    sources = []
    for split in ("train_eval", "heldout_in_domain"):
        for dataset in ("logiqa", "hellaswag"):
            path = raw_root / f"{split}-{dataset}.eval"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{split}-{dataset}".encode("utf-8"))
            sources.append({"split": split, "dataset": dataset, "raw_log": str(path), "raw_log_sha256": _sha256(path)})
    report = {"condition": "rmct-main", "raw_root": str(raw_root), "sources": sources}
    monkeypatch.setattr(stage_luna, "validate_none_preflight_report", lambda _path: report)

    output = tmp_path / "staged"
    first = stage_luna.stage_from_preflight(tmp_path / "preflight.json", output)
    assert {status for _, status in first} == {"staged"}
    assert {path.relative_to(output).parts[1] for path, _ in first} == {"train_eval", "heldout_in_domain"}
    assert {status for _, status in stage_luna.stage_from_preflight(tmp_path / "preflight.json", output)} == {"resumed"}

    corrupted = first[0][0]
    corrupted.write_bytes(b"wrong")
    with pytest.raises(FileExistsError, match="differing"):
        stage_luna.stage_from_preflight(tmp_path / "preflight.json", output)
