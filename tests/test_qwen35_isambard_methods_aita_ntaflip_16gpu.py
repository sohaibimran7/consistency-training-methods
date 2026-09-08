from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "infra/isambard/run_qwen35_methods_aita_ntaflip_16gpu.py"
SBATCH = ROOT / "infra/isambard/run_qwen35_methods_aita_ntaflip_16gpu.sbatch"
WORKER = ROOT / "infra/isambard/run_qwen35_methods_aita_ntaflip_16gpu_worker.sh"


def _module():
    spec = importlib.util.spec_from_file_location("methods_aita_test_module", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_five_methods_use_a_16_then_4_pair_preserving_plan() -> None:
    module = _module()
    phases = module._phase_plan()
    assert [len(phase["cells"]) for phase in phases] == [16, 4]
    assert phases[0]["allocated_gpus"] == 16
    assert phases[1]["allocated_gpus"] == 4
    assert len({(cell["condition"], cell["shard_index"]) for phase in phases for cell in phase["cells"]}) == 20
    assert {condition.optimizer_step for condition in module.CONDITIONS} == {32, 128, 256, 4000}
    assert all(condition.source_kind == "recovered-final-native-hf" for condition in module.CONDITIONS)


def test_methods_launch_surface_has_no_output_token_cap() -> None:
    module = _module()
    module.assert_no_token_cap_mapping(module.RUNTIME_GENERATION_CONFIG, label="test runtime")
    text = "\n".join(path.read_text(encoding="utf-8") for path in (LAUNCHER, SBATCH, WORKER))
    for field in (
        "max_new_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "completion_max_tokens",
    ):
        assert field not in text
    assert 'if (( $# != 7 )); then' in WORKER.read_text(encoding="utf-8")
    assert 'run_phase phase-1 4 16 16' in SBATCH.read_text(encoding="utf-8")
    assert 'run_phase phase-2 1 4 4' in SBATCH.read_text(encoding="utf-8")


def test_recovered_adapter_requires_final_manifest_and_stage2_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    adapter = tmp_path / "demo"
    provenance_root = tmp_path / "provenance"
    adapter.mkdir()
    provenance_root.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B"}), encoding="utf-8"
    )
    (adapter / "manifest.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3.5-9B",
                "lora": True,
                "loop_state": {"step": 7, "final": True},
            }
        ),
        encoding="utf-8",
    )
    (adapter / "README.md").write_text("checkpoint", encoding="utf-8")
    (provenance_root / "demo.json").write_text(
        json.dumps({"source": {"adapter_model_sha256": _sha256(adapter / "adapter_model.safetensors")}}),
        encoding="utf-8",
    )
    spec = module.RawAdapterSpec(
        directory_name="demo",
        canonical_source="demo-final",
        final_step=7,
        adapter_model_sha256=_sha256(adapter / "adapter_model.safetensors"),
        adapter_config_sha256=_sha256(adapter / "adapter_config.json"),
        manifest_sha256=_sha256(adapter / "manifest.json"),
        readme_sha256=_sha256(adapter / "README.md"),
        provenance_filename="demo.json",
        provenance_sha256=_sha256(provenance_root / "demo.json"),
    )
    monkeypatch.setitem(module.RAW_ADAPTER_REGISTRY, "demo", spec)

    record = module._checked_raw_adapter(checkpoint_directory=tmp_path, condition_name="demo")
    assert record["source"] == "recovered-final-native-hf"
    assert record["checkpoint"]["final_step"] == 7
    assert record["checkpoint"]["stage2_provenance"]["sha256"] == spec.provenance_sha256

    manifest = json.loads((adapter / "manifest.json").read_text(encoding="utf-8"))
    manifest["loop_state"]["final"] = False
    (adapter / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(module.EvaluationError):
        module._checked_raw_adapter(checkpoint_directory=tmp_path, condition_name="demo")
