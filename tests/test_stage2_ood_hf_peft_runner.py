from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.stage2_ood_hle import hf_peft_runner as runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(tmp_path: Path, monkeypatch, *, condition: str = "bct-hf-peft") -> Path:
    original = runner.CONDITIONS[condition]
    checkpoint = tmp_path / original.checkpoint_name
    checkpoint.mkdir(parents=True)
    adapter = checkpoint / "adapter_model.safetensors"
    config = checkpoint / "adapter_config.json"
    manifest = checkpoint / "manifest.json"
    adapter.write_bytes(b"raw-peft-adapter")
    config.write_text(json.dumps({"base_model_name_or_path": runner.BASE_MODEL}, sort_keys=True))
    manifest.write_bytes(b"raw-localbackend-manifest")
    monkeypatch.setitem(
        runner.CONDITIONS,
        condition,
        runner.HFPEFTCondition(
            checkpoint_name=checkpoint.name,
            adapter_model_sha256=_sha256(adapter),
            adapter_config_sha256=_sha256(config),
            manifest_sha256=_sha256(manifest),
        ),
    )
    monkeypatch.setattr(
        runner,
        "read_local_checkpoint",
        lambda value: (Path(value).resolve(), {"model": runner.BASE_MODEL, "lora": True}),
    )
    return checkpoint


def _patch_stage2_matrix(tmp_path: Path, monkeypatch, *, task_count: int = 21):
    manifest = tmp_path / "stage2-manifest.json"
    manifest.write_bytes(b"frozen-stage2")
    specs = [SimpleNamespace(kind="unbiased") for _ in range(3)] + [SimpleNamespace(kind="biased") for _ in range(task_count - 3)]
    monkeypatch.setattr(runner, "validate_manifest", lambda _path: {"validated": True})
    monkeypatch.setattr(runner, "ood_task_specs", lambda _path: specs)
    return manifest


def test_native_hf_peft_contract_binds_raw_adapter_and_full_stage2_matrix(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path, monkeypatch)
    manifest = _patch_stage2_matrix(tmp_path, monkeypatch)
    raw_log_dir = tmp_path / "raw-no-luna" / "bct-hf-peft"

    contract = runner.build_launch_contract(
        condition="bct-hf-peft",
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_log_dir,
        max_connections=1,
    )

    assert contract["schema"] == runner.LAUNCH_SCHEMA
    assert contract["mode"] == "raw-no-luna"
    assert contract["checkpoint"]["path"] == str(checkpoint.resolve())
    assert contract["stage2"]["task_count"] == 21
    assert contract["stage2"]["unbiased_task_count"] == 3
    assert contract["stage2"]["biased_task_count"] == 18
    assert contract["stage2"]["prompt_style"] == "none"
    assert contract["stage2"]["include_bias_acknowledged"] is False
    assert contract["model_args"] == {"provider": "hf", "device": "cuda:0", "dtype": "bfloat16"}
    assert contract["generation_config"] == {
        "max_tokens": 20480,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_connections": 1,
    }
    assert contract["execution"] == {
        "max_tasks": 1,
        "isolate_tasks": True,
        "persistent_vllm_server": False,
    }


def test_native_hf_peft_contract_rejects_wrong_raw_adapter_or_task_matrix(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path, monkeypatch)
    manifest = _patch_stage2_matrix(tmp_path, monkeypatch)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"different-adapter")

    with pytest.raises(ValueError, match="adapter_model.safetensors SHA-256"):
        runner.build_launch_contract(
            condition="bct-hf-peft",
            checkpoint=checkpoint,
            manifest=manifest,
            raw_log_dir=tmp_path / "raw",
            max_connections=1,
        )

    checkpoint = _checkpoint(tmp_path / "second", monkeypatch)
    manifest = _patch_stage2_matrix(tmp_path / "second", monkeypatch, task_count=20)
    with pytest.raises(ValueError, match=r"3 clean \+ 18 biased"):
        runner.build_launch_contract(
            condition="bct-hf-peft",
            checkpoint=checkpoint,
            manifest=manifest,
            raw_log_dir=tmp_path / "raw-second",
            max_connections=1,
        )


def test_rmct_native_hf_contract_is_a_distinct_raw_adapter_identity(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path, monkeypatch, condition="rmct-hf-peft")
    manifest = _patch_stage2_matrix(tmp_path, monkeypatch)

    contract = runner.build_launch_contract(
        condition="rmct-hf-peft",
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=tmp_path / "raw-no-luna" / "rmct-hf-peft",
        max_connections=8,
    )

    assert contract["checkpoint"]["checkpoint_name"] == checkpoint.name
    assert contract["checkpoint"]["adapter_model_sha256"] != runner.CONDITIONS["rmct-control-hf-peft"].adapter_model_sha256


def test_fresh_rmct_native_hf_conditions_are_pinned_to_the_current_phase2_adapters():
    assert runner.condition_spec("rmct-hf-peft") == runner.HFPEFTCondition(
        checkpoint_name="rmct_paper_isambard_phase2_qwen3_5_9b_rng_repair_4gpu_20260803_rate-matching-lr-1e-4",
        adapter_model_sha256="0ea90421fa2f81fee390288aca21b57d942945ebea10afc28b2a5267d244d707",
        adapter_config_sha256="0ddae2df7fd16cc50e03a7bdaa4118175c805e08c1d1a714d03107c8f943c603",
        manifest_sha256="d846cd872e392eafd21be095e4fcab1694a5b8cf9b798f4c7a7e4ed118ddadb5",
    )
    assert runner.condition_spec("rmct-control-hf-peft") == runner.HFPEFTCondition(
        checkpoint_name="rmct_paper_isambard_phase2_qwen3_5_9b_rng_repair_4gpu_20260803_rate-matching-control-lr-1e-4",
        adapter_model_sha256="bded56fd53c606d8b589667d2afcda8213d6c9ef692d2101ed42598f53663f14",
        adapter_config_sha256="384cbfb571704f03e308a224d4659e22ab68f392a7d66c3cd9c67d547a3cc4b2",
        manifest_sha256="d846cd872e392eafd21be095e4fcab1694a5b8cf9b798f4c7a7e4ed118ddadb5",
    )
    assert runner.condition_spec("rmct") is runner.condition_spec("rmct-hf-peft")
    assert runner.condition_spec("rmct-control") is runner.condition_spec("rmct-control-hf-peft")


def test_opct_phase2_native_hf_condition_is_pinned_to_verified_custody_identity():
    assert runner.condition_spec("opct-phase2-hf-peft") == runner.HFPEFTCondition(
        checkpoint_name="rmct_paper_isambard_phase2_qwen3_5_9b_opct_rng_repair_4gpu_20260803_opct-lr-1e-4",
        adapter_model_sha256="d893a6c202e5c9b0a1d00358b1f656d21ad99168613758046e64749845d8a7dd",
        adapter_config_sha256="c68aca369a9b5c31449b54c56f9d6c0df113f19cbf2b95fd8d6b57eef7c7b5e6",
        manifest_sha256="cef600b8c8bfe5e3d874b717e3e9ee993f0d114b525cd89de0f5aafa4b6e0d0e",
    )


def test_launch_contract_is_immutable_and_positive_batch_is_required(tmp_path):
    output = tmp_path / "launch-contract.json"
    assert runner.write_launch_contract(output, {"value": 1}) == "written"
    assert runner.write_launch_contract(output, {"value": 1}) == "resumed"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.write_launch_contract(output, {"value": 2})
    with pytest.raises(ValueError, match="positive integer"):
        runner._strict_positive_integer(0, label="max_connections")


def test_hf_peft_shell_launcher_is_syntactically_valid_and_raw_only():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "ctm_stage2_ood_hf_peft_raw_condition_20260802.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")

    assert "raw-no-luna" in text
    assert "CTM_OOD_ARCHIVE_EXISTING" in text
    assert "--local-checkpoint" in text
    assert '"provider":"hf"' in text
    assert '"prompt_style": "none"' in text
    assert '"max_tokens": 20480' in text
    assert "--persistent-vllm-server" not in text
    # The native-HF launcher uses a one-task Inspect child budget, so grouped
    # task indices must be explicitly isolated rather than silently truncated.
    assert "--isolate-tasks" in text
