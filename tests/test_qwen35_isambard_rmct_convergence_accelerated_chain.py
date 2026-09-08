"""Static safety contracts for the r3 accelerated Isambard continuation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parent.parent
LAUNCHER = ROOT / "infra/isambard/run_qwen35_rmct_convergence_accelerated_deadline.sh"
SEGMENT_SBATCH = ROOT / "infra/isambard/run_qwen35_rmct_convergence_accelerated_segment.sbatch"
VERIFIER = ROOT / "infra/isambard/verify_rmct_convergence_accelerated_production_ready.py"


def _load_verifier_module():
    spec = importlib.util.spec_from_file_location("rmct_accelerated_readiness_verifier_for_test", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_accelerated_r3_entrypoints_are_syntax_valid_and_explicitly_executable():
    for script in (LAUNCHER, SEGMENT_SBATCH):
        assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0
        assert script.stat().st_mode & 0o111
    result = subprocess.run([sys.executable, str(VERIFIER), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_r3_launcher_cannot_replay_segment_zero_and_uses_the_external_r2_parent_once():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "--segment-index 1..31" in text
    assert 'start_segment_index=1' in text
    assert 'last_segment_index=31' in text
    assert 'parent_run_prefix="rmct-convergence-gcall-r2"' in text
    assert 'parent_run_name="${parent_run_prefix}-s001"' in text
    assert 'run_prefix="rmct-convergence-gcall-r2-mb49152-r3"' in text
    assert "rmct-convergence-gcall-r2-mb49152-r3-production-ready.json" in text
    assert "qwen3_5_9b_rmct_convergence_gcall_r2_mb49152_r3_isambard_20260814.yaml" in text
    assert "experiments.rmct_convergence_accelerated.plan render" in text
    assert "experiments.rmct_convergence_accelerated.plan execute" in text
    assert "for (( index=start_segment_index; index<=last_segment_index; index++ ))" in text
    assert "parent_predecessor" in text
    assert '"$boundary" decision --directory "$parent_decisions" --segment-index 0' in text
    assert "--predecessor-receipt \"$predecessor\" --max-optimizer-steps 512" in text
    assert "local_forward_microbatch_max_tokens=49152" in text
    assert "RMCT_CONVERGENCE_ACCELERATED_PARENT_NOOP=1" in text
    assert "refuse ambiguous replay" in text
    assert "verify-sealed-segment-custody" in text
    assert "checkpoint path is linked; refuse recovery" in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "--no-shuffle-datapoints" not in text  # fixed solely by the frozen compiler
    # The inherited rollout parity sidecar must be resumed, never created by r3.
    assert "inherited immutable rollout worker-parity sidecar is absent" in text
    assert '--output-dir "$worker_parity_dir" --model-snapshot "$model_snapshot" --resume' in text
    assert '"$python_bin" "$verify_ready" --repository "$repo_root" --receipt "$ready_receipt"' in text
    assert text.index("ensure_worker_parity") < text.rindex('"$python_bin" "$verify_ready"')


def test_r3_successor_sbatch_is_one_explicit_dependency_safe_window_and_captures_its_own_receipt():
    text = SEGMENT_SBATCH.read_text(encoding="utf-8")
    assert "--dependency=afterok:<sealed-predecessor-job>" in text
    assert "#SBATCH --partition=workq" in text
    assert "#SBATCH --time=12:00:00" in text
    assert "CTM_RMCT_SEGMENT_INDEX" in text
    assert "[1, 31]" in text
    assert ".rmct-convergence-gcall-r2-mb49152-r3-production-source-ready" in text
    assert "rmct-convergence-gcall-r2-mb49152-r3-production-ready.json" in text
    assert "verify-source-ready" in text
    assert text.index("verify-source-ready") < text.index("exec srun ")
    assert "--resume" in text
    capture = '"$python_bin" "$verify_script" capture --repository "$repo_root" --output "$ready_receipt"'
    verify = '"$python_bin" "$verify_script" verify --repository "$repo_root" --receipt "$ready_receipt"'
    launch = 'exec "$launcher" --ready-receipt "$ready_receipt" --segment-index "$segment_index" --yes'
    assert capture in text and verify in text and launch in text
    assert text.index(capture) < text.index(verify) < text.index(launch)
    assert "--max-segments" not in text
    assert "--segment-index 0" not in text


def test_r3_verifier_binds_fixed_parent_artifact_preflight_and_compiled_physical_delta():
    verifier = _load_verifier_module()
    text = VERIFIER.read_text(encoding="utf-8")
    assert verifier.RUN_PREFIX == "rmct-convergence-gcall-r2-mb49152-r3"
    assert verifier.PARENT_RUN_PREFIX == "rmct-convergence-gcall-r2"
    assert verifier.PARENT_RUN_NAME == "rmct-convergence-gcall-r2-s001"
    assert verifier.CONTINUATION_PARENT_ARTIFACT_SHA256 == "b4ad494f7b20fc20af4610489a2ccff315d651a0e2636957a353a89065a1f906"
    assert "da5783f1aca63cd7f7df500cd85e3460f204ddf2398417c8f44a5c9fadcdd0e9" in text
    assert "a71c0d74cb65b3b3aba56eb61f03e5ab25fe5cf28e909448b60fe62e051c6394" in text
    assert "load_strict_local_rl_resume_state" in text
    assert "controller.require_continue" in text
    assert "local_forward_microbatch_max_tokens" in text and "49152" in text
    assert "logical segments 1..31" in text
    assert "local_gradient_checkpointing_layers" in text
    assert "verify-sealed-segment-custody" in text
    for critical in (
        "ctm/artifacts.py",
        "ctm/backends/run_metadata.py",
        "ctm/identity.py",
        "ctm/provenance.py",
        "ctm/experiments/records.py",
        "ctm/cli_safety.py",
        "ctm/__init__.py",
        "ctm/backends/__init__.py",
        "ctm/backends/base.py",
        "ctm/backends/local/__init__.py",
        "ctm/backends/renderers.py",
        "experiments/rmct_convergence_accelerated/plan.py",
        "ctm/backends/local/losses.py",
        "ctm/backends/local/mlp_hooks.py",
        "ctm/backends/local/vllm_sampler.py",
        "ctm/core/advantages.py",
        "ctm/core/__init__.py",
        "ctm/core/config.py",
        "ctm/core/rewards.py",
        "ctm/core/types.py",
        "ctm/settings/runtime.py",
        "ctm/settings/base.py",
        "ctm/settings/__init__.py",
        "ctm/importing.py",
        "ctm/training/checkpoints.py",
        "ctm/training/__init__.py",
        "ctm/training/consistency_losses.py",
        "ctm/training/manifest.py",
        "ctm/training/rollout_log.py",
        "ctm/training/run_utils.py",
        "ctm_data/adapters/mcq_bias/data.py",
        "ctm_data/__init__.py",
        "ctm_data/adapters/__init__.py",
        "ctm_data/adapters/mcq_bias/__init__.py",
        "infra/vastai/preflight_qwen35_phase_shared.py",
        "run_qwen35_rmct_convergence_accelerated_deadline.sh",
        "run_qwen35_rmct_convergence_accelerated_segment.sbatch",
        "rmct_convergence_segment_boundary.py",
        "preflight_qwen35_rmct_convergence_worker_parity.py",
    ):
        assert any(path.endswith(critical) for path in verifier.CRITICAL_SOURCES)
    assert verifier.PREFLIGHT_RESULT_RELATIVE.endswith("same-gh200-all-gc-preflight-result.json")
    assert verifier.PREFLIGHT_RESULT_SHA256 == "da5783f1aca63cd7f7df500cd85e3460f204ddf2398417c8f44a5c9fadcdd0e9"
    assert verifier.PREFLIGHT_CONTRACT_RELATIVE.endswith("same-gh200-all-gc-preflight-contract.json")
    assert verifier.PREFLIGHT_CONTRACT_SHA256 == "a71c0d74cb65b3b3aba56eb61f03e5ab25fe5cf28e909448b60fe62e051c6394"

    compiled = verifier._compiled_contract(ROOT)
    assert compiled["compiler_module"] == "experiments.rmct_convergence_accelerated.plan"
    assert compiled["frozen_hyperparameters"]["continuation"]["logical_start_segment_index"] == 1
    assert compiled["frozen_hyperparameters"]["topology"]["gradient_checkpointing_layers"] == "all"


def test_preflight_evidence_requires_the_exact_staged_passed_all_gc_four_gh200_contract(monkeypatch, tmp_path):
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
            "rank_zero_packing_sweep": [
                {
                    "packing_budget": 49152,
                    "padded_token_slots": 49152,
                    "passed": True,
                    "scope": "rank_zero_capacity_only_with_all_vllm_workers_asleep",
                }
            ],
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

    evidence = verifier._preflight_evidence_identity(tmp_path)
    assert evidence["semantics"] == {
        "hardware_scope": "same-GH200",
        "gpu_count": 4,
        "gradient_checkpointing_layers": "all",
        "local_forward_microbatch_max_datums": 8,
        "local_forward_microbatch_max_tokens": 49152,
        "local_target_logprob_chunk_size": 2048,
        "passed": True,
    }

    result["phase_shared"]["rank_zero_packing_sweep"][0]["passed"] = False
    result_path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(verifier, "PREFLIGHT_RESULT_SHA256", verifier._sha256(result_path))
    with pytest.raises(verifier.ReadyError, match="49,152-token"):
        verifier._preflight_evidence_identity(tmp_path)


def test_verifier_reports_readiness_failures_as_argparse_errors_not_name_errors(tmp_path):
    verifier = _load_verifier_module()
    with pytest.raises(SystemExit) as exit_status:
        verifier.main(
            [
                "verify-source-ready",
                "--repository",
                str(tmp_path),
                "--receipt",
                str(tmp_path / "absent-source-ready.json"),
            ]
        )
    assert exit_status.value.code == 2
    with pytest.raises(SystemExit) as no_command:
        verifier.main([])
    assert no_command.value.code == 2


def test_parent_training_command_and_marker_bind_the_full_non_resume_gcall_r2_s001_argv(tmp_path):
    verifier = _load_verifier_module()
    from experiments.rmct_convergence import plan as gcall
    from scripts.run_experiment import _argument_tokens

    plan_path = tmp_path / verifier.PARENT_PLAN_RELATIVE
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("frozen-gcall-r2-plan\n", encoding="utf-8")
    snapshot = tmp_path / "hf" / "snapshots" / verifier.BASE_SNAPSHOT
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    command_path = tmp_path / verifier.PARENT_TRAINING_COMMAND_RELATIVE
    command_path.parent.mkdir(parents=True)
    expected_args = gcall.segment_args(tmp_path, 0, model_path=snapshot, run_prefix=verifier.PARENT_RUN_PREFIX)
    command = {
        "schema": gcall.COMMAND_SCHEMA,
        "condition": verifier.CONDITION,
        "run_prefix": verifier.PARENT_RUN_PREFIX,
        "plan": {"path": str(plan_path.resolve()), "sha256": verifier._sha256(plan_path)},
        "model": {"repo_id": verifier.MODEL, "revision": verifier.BASE_SNAPSHOT, "snapshot_path": str(snapshot.resolve())},
        "segment": gcall.segment_record(tmp_path, 0, run_prefix=verifier.PARENT_RUN_PREFIX),
        "argv": ["/usr/bin/python3", str((tmp_path / "scripts/train_rlct.py").resolve()), *_argument_tokens(expected_args)],
        "environment_contract": {
            "cuda_visible_devices_preserved": True,
            "topology_profile": verifier.TOPOLOGY_PROFILE,
            "phase_shared": True,
            "gradient_checkpointing_layers": "all",
        },
    }
    command_path.write_text(json.dumps(command), encoding="utf-8")
    ready_path = tmp_path / verifier.PARENT_READY_RECEIPT_RELATIVE
    ready_path.parent.mkdir(parents=True)
    ready_path.write_text(
        json.dumps(
            {
                "schema": verifier.PARENT_READY_SCHEMA,
                "condition": verifier.CONDITION,
                "ready": True,
                "compiled_contract": {
                    "compiler_module": verifier.PARENT_COMPILER_MODULE,
                    "model_constants": {"model": verifier.MODEL, "base_snapshot": verifier.BASE_SNAPSHOT},
                    "frozen_hyperparameters": {"run_prefix": verifier.PARENT_RUN_PREFIX},
                },
                "source_ready": {
                    "schema": verifier.PARENT_SOURCE_READY_SCHEMA,
                    "condition": verifier.CONDITION,
                    "source_ready": True,
                },
            }
        ),
        encoding="utf-8",
    )
    marker_path = tmp_path / verifier.PARENT_TRAINING_STARTED_RELATIVE
    marker = {
        "schema": "rmct-convergence-training-started-v1",
        "segment_index": 0,
        "command_attestation": {"path": str(command_path.resolve()), "sha256": verifier._sha256(command_path)},
        "ready_receipt": {"path": str(ready_path.resolve()), "sha256": verifier._sha256(ready_path)},
    }
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    provenance = verifier._parent_training_provenance_identity(tmp_path)
    assert provenance["semantics"]["resume"] is False
    assert provenance["semantics"]["run_name"] == verifier.PARENT_RUN_NAME
    assert provenance["command"]["sha256"] == verifier._sha256(command_path)
    assert provenance["training_started"]["sha256"] == verifier._sha256(marker_path)

    original_ready = ready_path.read_bytes()
    malformed_ready = json.loads(original_ready)
    malformed_ready["run_prefix"] = verifier.PARENT_RUN_PREFIX
    ready_path.write_text(json.dumps(malformed_ready), encoding="utf-8")
    with pytest.raises(verifier.ReadyError, match="unexpected gcall-r2 readiness schema"):
        verifier._parent_training_provenance_identity(tmp_path)
    ready_path.write_bytes(original_ready)

    command["argv"].extend(["--resume-from", "file:///forbidden"])
    command_path.write_text(json.dumps(command), encoding="utf-8")
    with pytest.raises(verifier.ReadyError, match="non-resume gcall-r2 s001 argv"):
        verifier._parent_training_provenance_identity(tmp_path)


def test_parent_checkpoint_requires_and_hashes_every_strict_four_rank_sidecar(tmp_path):
    verifier = _load_verifier_module()
    checkpoint = tmp_path / "logs" / "parent-checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "manifest.json").write_text(json.dumps({"backend": "local", "kind": "both"}), encoding="utf-8")
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
    (checkpoint / "replicated_training_manifest.json").write_text(json.dumps(replicated), encoding="utf-8")
    (checkpoint / "replicated_training_rng.pt").write_bytes(b"strict-rng")
    state = SimpleNamespace(global_step=16, optimizer_step=16, completed_epochs=1)

    identity = verifier._parent_checkpoint_identity(tmp_path, checkpoint, state=state)
    assert set(identity["files"]) == {
        "adapter_config",
        "adapter_model",
        "optimizer",
        "manifest",
        "replicated_training_manifest",
        "replicated_training_rng",
    }
    assert identity["replicated_training"]["world_size"] == 4
    assert identity["replicated_training"]["train_logical_indices"] == [0, 1, 2, 3]
    assert identity["replicated_training"]["state_hash"] == "a" * 64

    replicated["world_size"] = 3
    (checkpoint / "replicated_training_manifest.json").write_text(json.dumps(replicated), encoding="utf-8")
    with pytest.raises(verifier.ReadyError, match="strict four-GPU"):
        verifier._parent_checkpoint_identity(tmp_path, checkpoint, state=state)


def test_r3_checkpoint_recovery_requires_its_exact_command_marker_and_ready_receipt(monkeypatch, tmp_path):
    verifier = _load_verifier_module()
    from experiments.rmct_convergence_accelerated import plan as accelerated
    from scripts.run_experiment import _argument_tokens

    index = 1
    plan_path = tmp_path / verifier.PLAN_RELATIVE
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("frozen-r3-plan\n", encoding="utf-8")
    snapshot = tmp_path / "hf" / "snapshots" / verifier.BASE_SNAPSHOT
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    ready_path = tmp_path / "artifacts" / "r3-ready.json"
    ready_path.parent.mkdir(parents=True)
    ready_path.write_text("{}", encoding="utf-8")
    ready = {
        "base_snapshot": {
            "repo_id": verifier.MODEL,
            "revision": verifier.BASE_SNAPSHOT,
            "snapshot_path": str(snapshot.resolve()),
        }
    }
    monkeypatch.setattr(verifier, "verify_receipt", lambda root, receipt: ready)

    run = accelerated.run_name(index)
    segment_dir = tmp_path / "logs" / verifier.CONDITION / run / "segment"
    segment_dir.mkdir(parents=True)
    accelerated.final_checkpoint_path(tmp_path, index).mkdir(parents=True)
    command_path = segment_dir / "training-command.json"
    expected_args = accelerated.segment_args(tmp_path, index, model_path=snapshot)
    command = {
        "schema": accelerated.COMMAND_SCHEMA,
        "condition": verifier.CONDITION,
        "run_prefix": verifier.RUN_PREFIX,
        "logical_segment_index": index,
        "plan": {"path": str(plan_path.resolve()), "sha256": verifier._sha256(plan_path)},
        "model": {"repo_id": verifier.MODEL, "revision": verifier.BASE_SNAPSHOT, "snapshot_path": str(snapshot.resolve())},
        "segment": accelerated.segment_record(tmp_path, index),
        "argv": [
            "/usr/bin/python3",
            str((tmp_path / "scripts" / "train_rlct.py").resolve()),
            *_argument_tokens(expected_args),
        ],
        "environment_contract": {
            "cuda_visible_devices_preserved": True,
            "topology_profile": verifier.TOPOLOGY_PROFILE,
            "phase_shared": True,
            "gradient_checkpointing_layers": "all",
            "local_forward_microbatch_max_datums": 8,
            "local_forward_microbatch_max_tokens": 49152,
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

    identity = verifier.verify_sealed_segment_custody(
        tmp_path, segment_index=index, ready_receipt=ready_path
    )
    assert identity["run_name"] == run
    assert identity["command"]["sha256"] == verifier._sha256(command_path)
    assert identity["training_started"]["sha256"] == verifier._sha256(marker_path)

    marker["ready_receipt"] = {"path": str(ready_path.resolve()), "sha256": "0" * 64}
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(verifier.ReadyError, match="marker readiness"):
        verifier.verify_sealed_segment_custody(tmp_path, segment_index=index, ready_receipt=ready_path)
