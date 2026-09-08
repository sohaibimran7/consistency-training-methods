"""Static safety contracts for the r4 Isambard recovery integration."""

from __future__ import annotations

import importlib.util
from hashlib import sha256
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.rmct_convergence_r4_recovery import plan as r4
from scripts.run_experiment import _argument_tokens


ROOT = Path(__file__).parent.parent
LAUNCHER = ROOT / "infra/isambard/run_qwen35_rmct_convergence_r4_recovery_deadline.sh"
SEGMENT_SBATCH = ROOT / "infra/isambard/run_qwen35_rmct_convergence_r4_recovery_segment.sbatch"
VERIFIER = ROOT / "infra/isambard/verify_rmct_convergence_r4_recovery_production_ready.py"


def _load_verifier_module():
    spec = importlib.util.spec_from_file_location("rmct_r4_recovery_readiness_verifier_for_test", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_r4_entrypoints_are_syntax_valid_explicitly_executable_and_never_submit():
    for script in (LAUNCHER, SEGMENT_SBATCH):
        assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0
        assert script.stat().st_mode & 0o111
    result = subprocess.run([sys.executable, str(VERIFIER), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "sbatch " not in LAUNCHER.read_text(encoding="utf-8")


def test_r4_launcher_and_sbatch_are_one_window_fail_closed_recovery_entrypoints():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    sbatch = SEGMENT_SBATCH.read_text(encoding="utf-8")
    assert "--segment-index 4..31" in launcher
    assert "start_segment_index=4" in launcher
    assert 'parent_run_prefix="rmct-convergence-gcall-r2-mb49152-r3"' in launcher
    assert 'parent_run_name="${parent_run_prefix}-s004"' in launcher
    assert 'run_prefix="rmct-convergence-gcall-r2-mb40960-r4"' in launcher
    assert "6031786 is never a" in launcher
    assert "qwen3_5_9b_rmct_convergence_gcall_r2_mb40960_r4_isambard_20260818.yaml" in launcher
    assert "experiments.rmct_convergence_r4_recovery.plan render" in launcher
    assert "experiments.rmct_convergence_r4_recovery.plan execute" in launcher
    assert '"$boundary" decision --directory "$parent_decisions" --segment-index "$((start_segment_index - 1))"' in launcher
    assert "local_forward_microbatch_max_tokens=40960" in launcher
    assert "verify-sealed-segment-custody" in launcher
    assert launcher.count("verify-sealed-segment-custody") >= 2
    assert launcher.index('"$python_bin" -m experiments.rmct_convergence_r4_recovery.plan execute') < launcher.rindex("verify-sealed-segment-custody") < launcher.rindex('"$python_bin" "$boundary" seal')
    assert "checkpoint path is linked; refuse recovery" in launcher
    assert "refuse ambiguous replay" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "--dependency=afterok:<sealed-predecessor-job>" in sbatch
    assert "CTM_RMCT_SEGMENT_INDEX" in sbatch and "[4, 31]" in sbatch
    assert ".rmct-convergence-gcall-r2-mb40960-r4-production-source-ready" in sbatch
    assert "rmct-convergence-gcall-r2-mb40960-r4-production-ready.json" in sbatch
    assert "verify-source-ready" in sbatch
    assert sbatch.index("verify-source-ready") < sbatch.index("exec srun ")
    assert "--max-segments" not in sbatch


def test_r4_verifier_binds_failure_parent_artifact_exact_40960_preflight_and_compiled_delta():
    verifier = _load_verifier_module()
    assert verifier.RUN_PREFIX == r4.RUN_PREFIX
    assert verifier.PARENT_RUN_PREFIX == "rmct-convergence-gcall-r2-mb49152-r3"
    assert verifier.PARENT_RUN_NAME == "rmct-convergence-gcall-r2-mb49152-r3-s004"
    assert verifier.START_SEGMENT_INDEX == 4
    assert verifier.RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS == 40960
    assert verifier.FAILURE_PARENT_ARTIFACT_SHA256 == r4.FAILURE_PARENT_ARTIFACT_SHA256
    assert "load_strict_local_rl_resume_state" in VERIFIER.read_text(encoding="utf-8")
    assert "r3_ready.verify_sealed_segment_custody" in VERIFIER.read_text(encoding="utf-8")
    assert "controller.require_continue" in VERIFIER.read_text(encoding="utf-8")
    assert "6031786" in VERIFIER.read_text(encoding="utf-8")
    assert "failed_attempt_evidence" in VERIFIER.read_text(encoding="utf-8")
    assert "40,960-token" in VERIFIER.read_text(encoding="utf-8")
    compiled = verifier._compiled_contract(ROOT)
    assert compiled["compiler_module"] == "experiments.rmct_convergence_r4_recovery.plan"
    assert compiled["frozen_hyperparameters"]["continuation"]["logical_start_segment_index"] == 4
    assert compiled["frozen_hyperparameters"]["continuation"]["execution_delta"] == {
        "local_forward_microbatch_max_tokens": {"from": 49152, "to": 40960}
    }


def test_r4_failed_attempt_evidence_requires_every_exact_remote_file_and_hash(monkeypatch, tmp_path):
    verifier = _load_verifier_module()
    payloads = {
        "slurm_output": b"slurm failure output\n",
        "training_command": b"command\n",
        "training_started": b"marker\n",
        "metrics": b"metrics\n",
        "logs": b"logs\n",
        "config": b"config\n",
        "manifest": b"manifest\n",
    }
    evidence = {
        name: {"path": f"remote/{name}", "sha256": sha256(payload).hexdigest()}
        for name, payload in payloads.items()
    }
    for name, payload in payloads.items():
        path = tmp_path / evidence[name]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    monkeypatch.setattr(verifier, "FAILED_ATTEMPT_EVIDENCE", evidence)
    identities = verifier._failed_attempt_evidence_identity(tmp_path)
    assert set(identities) == set(evidence)
    assert identities["slurm_output"]["sha256"] == evidence["slurm_output"]["sha256"]

    (tmp_path / evidence["metrics"]["path"]).write_bytes(b"different")
    with pytest.raises(verifier.ReadyError, match="metrics evidence differs"):
        verifier._failed_attempt_evidence_identity(tmp_path)


def _write_complete_checkpoint(path: Path, *, step: int = 80, manifest_kind: str = "both", omit: str | None = None) -> None:
    path.mkdir(parents=True)
    contents: dict[str, bytes] = {
        "adapter_config.json": b"{}\n",
        "adapter_model.safetensors": b"adapter",
        "optimizer.pt": b"optimizer",
        "replicated_training_rng.pt": b"rng",
    }
    manifest = {
        "backend": "local",
        "kind": manifest_kind,
        "loop_state": {
            "global_step": step,
            "optimizer_step": step,
            "step": step,
            "completed_epochs": 1,
            "accumulated_grads": 0,
            "final": True,
        },
    }
    replicated = {
        "schema": 1,
        "checkpoint_kind": "both",
        "world_size": 4,
        "train_logical_indices": [0, 1, 2, 3],
        "process_group_backend": "nccl",
        "optimizer_timeout_seconds": 120.0,
        "device_type": "cuda",
        "state_hash": "a" * 64,
        "rng_state_file": "replicated_training_rng.pt",
    }
    contents["manifest.json"] = json.dumps(manifest).encode("utf-8")
    contents["replicated_training_manifest.json"] = json.dumps(replicated).encode("utf-8")
    for filename, payload in contents.items():
        if filename != omit:
            (path / filename).write_bytes(payload)


def test_r4_strict_checkpoint_identity_rejects_missing_bundle_files_and_manifest_drift(monkeypatch, tmp_path):
    verifier = _load_verifier_module()
    import ctm.training.resume_state as resume_state

    monkeypatch.setattr(
        resume_state,
        "load_strict_local_rl_resume_state",
        lambda _checkpoint: SimpleNamespace(global_step=80, optimizer_step=80, completed_epochs=1),
    )
    missing_adapter = tmp_path / "missing-adapter"
    _write_complete_checkpoint(missing_adapter, omit="adapter_model.safetensors")
    with pytest.raises(verifier.ReadyError, match="adapter_model"):
        verifier._strict_checkpoint_identity(tmp_path, missing_adapter, expected_segment_index=4, label="test")

    missing_replicated = tmp_path / "missing-replicated"
    _write_complete_checkpoint(missing_replicated, omit="replicated_training_rng.pt")
    with pytest.raises(verifier.ReadyError, match="replicated_training_rng"):
        verifier._strict_checkpoint_identity(tmp_path, missing_replicated, expected_segment_index=4, label="test")

    drifted_manifest = tmp_path / "drifted-manifest"
    _write_complete_checkpoint(drifted_manifest, manifest_kind="sampler")
    with pytest.raises(verifier.ReadyError, match="final local kind='both'"):
        verifier._strict_checkpoint_identity(tmp_path, drifted_manifest, expected_segment_index=4, label="test")


def test_r4_preflight_requires_the_staged_passed_40960_probe(monkeypatch, tmp_path):
    verifier = _load_verifier_module()
    result_relative = "evidence/result.json"
    contract_relative = "evidence/contract.json"
    result_path = tmp_path / result_relative
    contract_path = tmp_path / contract_relative
    result_path.parent.mkdir(parents=True)
    contract = {
        "schema": "qwen35-phase-shared-preflight-v1",
        "kind": "non_production_phase_shared_preflight_contract",
        "non_production": True,
        "production_output_touched": False,
        "config": {
            "model": verifier.MODEL,
            "forward_microbatch_max_datums": 8,
            "target_logprob_chunk_size": 2048,
            "packing_budgets": [20480, 40960, 49152],
        },
        "resolved": {
            "gradient_checkpoint_layers": "all",
            "world_size": 4,
            "visible_devices": ["0", "1", "2", "3"],
            "training_gpus": [{"logical_index": index} for index in range(4)],
            "rollout_gpus": [{"logical_index": index} for index in range(4)],
        },
    }
    result = {
        "schema": "qwen35-phase-shared-preflight-v1",
        "passed": True,
        "phase_shared": {
            "rank_zero_packing_sweep": [{
                "packing_budget": 40960,
                "padded_token_slots": 40960,
                "passed": True,
                "scope": "rank_zero_capacity_only_with_all_vllm_workers_asleep",
            }],
            "packing_workers_asleep_memory": {
                "nvidia_smi_rows": [f"{index}, GPU-{index}, NVIDIA GH200 120GB" for index in range(4)]
            },
            "fixed_update_loss_parity": {"passed": True},
            "fixed_update_parameter_parity": {"passed": True},
            "fixed_update_pre_optimizer_gradient_parity": {"gate": {"passed": True}},
        },
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(verifier, "PREFLIGHT_RESULT_RELATIVE", result_relative)
    monkeypatch.setattr(verifier, "PREFLIGHT_CONTRACT_RELATIVE", contract_relative)
    monkeypatch.setattr(verifier, "PREFLIGHT_RESULT_SHA256", verifier._sha256(result_path))
    monkeypatch.setattr(verifier, "PREFLIGHT_CONTRACT_SHA256", verifier._sha256(contract_path))
    assert verifier._preflight_evidence_identity(tmp_path)["semantics"]["local_forward_microbatch_max_tokens"] == 40960

    result["phase_shared"]["rank_zero_packing_sweep"][0]["passed"] = False
    result_path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(verifier, "PREFLIGHT_RESULT_SHA256", verifier._sha256(result_path))
    with pytest.raises(verifier.ReadyError, match="40,960-token"):
        verifier._preflight_evidence_identity(tmp_path)


def test_r4_checkpoint_recovery_requires_its_exact_command_marker_and_ready_receipt(monkeypatch, tmp_path):
    verifier = _load_verifier_module()
    index = 4
    plan_path = tmp_path / verifier.PLAN_RELATIVE
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("frozen-r4-plan\n", encoding="utf-8")
    snapshot = tmp_path / "hf" / "snapshots" / verifier.BASE_SNAPSHOT
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    ready_path = tmp_path / "artifacts" / "r4-ready.json"
    ready_path.parent.mkdir(parents=True)
    ready_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        verifier,
        "verify_receipt",
        lambda root, receipt: {"base_snapshot": {"repo_id": verifier.MODEL, "revision": verifier.BASE_SNAPSHOT, "snapshot_path": str(snapshot.resolve())}},
    )
    run = r4.run_name(index)
    segment_dir = tmp_path / "logs" / verifier.CONDITION / run / "segment"
    segment_dir.mkdir(parents=True)
    checkpoint = r4.final_checkpoint_path(tmp_path, index)
    _write_complete_checkpoint(checkpoint)
    import ctm.training.resume_state as resume_state
    monkeypatch.setattr(
        resume_state,
        "load_strict_local_rl_resume_state",
        lambda _checkpoint: SimpleNamespace(global_step=80, optimizer_step=80, completed_epochs=1),
    )
    command_path = segment_dir / "training-command.json"
    expected_args = r4.segment_args(tmp_path, index, model_path=snapshot)
    command = {
        "schema": r4.COMMAND_SCHEMA,
        "condition": verifier.CONDITION,
        "run_prefix": verifier.RUN_PREFIX,
        "logical_segment_index": index,
        "plan": {"path": str(plan_path.resolve()), "sha256": verifier._sha256(plan_path)},
        "model": {"repo_id": verifier.MODEL, "revision": verifier.BASE_SNAPSHOT, "snapshot_path": str(snapshot.resolve())},
        "segment": r4.segment_record(tmp_path, index),
        "argv": ["/usr/bin/python3", str((tmp_path / "scripts" / "train_rlct.py").resolve()), *_argument_tokens(expected_args)],
        "environment_contract": {
            "cuda_visible_devices_preserved": True,
            "topology_profile": verifier.TOPOLOGY_PROFILE,
            "phase_shared": True,
            "gradient_checkpointing_layers": "all",
            "local_forward_microbatch_max_datums": 8,
            "local_forward_microbatch_max_tokens": 40960,
            "local_target_logprob_chunk_size": 2048,
        },
    }
    command_path.write_text(json.dumps(command), encoding="utf-8")
    marker_path = segment_dir / "training-started.json"
    marker = {
        "schema": "rmct-convergence-training-started-v1",
        "segment_index": index,
        "command_attestation": {"path": str(command_path.resolve()), "sha256": verifier._sha256(command_path)},
        "ready_receipt": {"path": str(ready_path.resolve()), "sha256": verifier._sha256(ready_path)},
    }
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    identity = verifier.verify_sealed_segment_custody(tmp_path, segment_index=index, ready_receipt=ready_path)
    assert identity["run_name"] == run
    marker["ready_receipt"]["sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(verifier.ReadyError, match="marker readiness"):
        verifier.verify_sealed_segment_custody(tmp_path, segment_index=index, ready_receipt=ready_path)
