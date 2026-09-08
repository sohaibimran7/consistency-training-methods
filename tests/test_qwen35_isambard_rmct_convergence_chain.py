"""Static safety contracts for the new RMCT-convergence Isambard path."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
LAUNCHER = ROOT / "infra/isambard/run_qwen35_rmct_convergence_deadline.sh"
SEGMENT_SBATCH = ROOT / "infra/isambard/run_qwen35_rmct_convergence_segment.sbatch"
SUBMITTER = ROOT / "infra/isambard/submit_qwen35_rmct_convergence_chain.sh"
VERIFIER = ROOT / "infra/isambard/verify_rmct_convergence_production_ready.py"
BOUNDARY = ROOT / "infra/isambard/rmct_convergence_segment_boundary.py"
BROKER = ROOT / "infra/isambard/run_rmct_convergence_deadline_broker.sbatch"
SEGMENT0_SBATCH = ROOT / "infra/isambard/run_qwen35_rmct_convergence_segment0_interactive.sbatch"
SEGMENT0_SUBMITTER = ROOT / "infra/isambard/submit_qwen35_rmct_convergence_segment0_when_ready.sh"
GCALL_R2_BOOTSTRAP = ROOT / "infra/isambard/run_qwen35_rmct_convergence_gcall_r2_workq_bootstrap.sbatch"
GCALL_R2_SEGMENT = ROOT / "infra/isambard/run_qwen35_rmct_convergence_gcall_r2_segment.sbatch"


def test_production_entrypoints_are_syntax_valid_and_broker_wires_the_fixed_launcher():
    for script in (LAUNCHER, SEGMENT_SBATCH, SUBMITTER, BROKER, GCALL_R2_BOOTSTRAP, GCALL_R2_SEGMENT):
        assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0
    assert LAUNCHER.stat().st_mode & 0o111
    assert "run_qwen35_rmct_convergence_deadline.sh" in BROKER.read_text(encoding="utf-8")
    assert "verify_rmct_convergence_production_ready.py" in BROKER.read_text(encoding="utf-8")


def test_gcall_r2_workq_bootstrap_binds_the_recovery_namespace_and_only_seals_segment_zero():
    text = GCALL_R2_BOOTSTRAP.read_text(encoding="utf-8")
    assert "#SBATCH --partition=workq" in text
    assert "#SBATCH --time=12:00:00" in text
    assert ".rmct-convergence-gcall-r2-production-source-ready" in text
    assert "rmct-convergence-gcall-r2-production-ready.json" in text
    assert "verify-source-ready" in text
    assert "--resume" in text
    assert text.index("verify-source-ready") < text.index("exec srun ")
    assert text.index('"$python_bin" "$parity_helper" --output-dir "$parity_dir" --model-snapshot "$model_snapshot" --resume') < text.index(
        '"$python_bin" "$verify_script" capture'
    )
    assert text.index('"$python_bin" "$verify_script" capture') < text.rindex(
        '"$python_bin" "$verify_script" verify'
    )
    assert "--segment-index 0 --yes" in text
    assert "--max-segments 2" not in text
    assert "bootstrap will not create it" in text


def test_gcall_r2_successor_requires_an_explicit_nonzero_dependency_safe_segment_index():
    text = GCALL_R2_SEGMENT.read_text(encoding="utf-8")
    assert "#SBATCH --partition=workq" in text
    assert "#SBATCH --time=12:00:00" in text
    assert "CTM_RMCT_SEGMENT_INDEX" in text
    assert "[1, 31]" in text
    assert ".rmct-convergence-gcall-r2-production-source-ready" in text
    assert "rmct-convergence-gcall-r2-production-ready.json" in text
    assert "verify-source-ready" in text
    assert "--resume" in text
    assert '"$python_bin" "$verify_script" verify --repository "$repo_root" --receipt "$ready_receipt"' in text
    assert 'exec "$launcher" --ready-receipt "$ready_receipt" --segment-index "$segment_index" --yes' in text
    assert "capture --repository" not in text
    assert "--max-segments" not in text


def test_launcher_preserves_sealed_segment_semantics_and_noops_after_terminal_controller_receipts():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "--segment-index 0..31" in text
    assert "--max-segments" in text
    assert "RMCT_CONVERGENCE_SEGMENT_REUSED=1" in text
    assert "RMCT_CONVERGENCE_SUCCESSOR_NOOP=1" in text
    assert "experiments.rmct_convergence.controller extract-source" in text
    assert "experiments.rmct_convergence.controller guard" in text
    assert "--max-optimizer-steps 512" in text
    assert "refuse ambiguous replay" in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "benchmark_gate_required=false" in text
    assert "rmct-convergence-gcall-r2" in text
    assert "qwen3_5_9b_rmct_convergence_gcall_r2_isambard_20260814.yaml" in text
    assert "rmct-convergence-gcall-r2-production-ready.json" in text
    assert "ensure_worker_parity" in text
    assert text.rindex("ensure_worker_parity") < text.rindex('"$python_bin" "$verify_ready"')
    assert "--no-shuffle-datapoints" not in text  # frozen in the compiler, never shell-overridable


def test_readiness_verifier_and_boundary_helper_are_importable_and_have_fixed_interfaces():
    for path in (VERIFIER, BOUNDARY):
        result = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
    verifier = VERIFIER.read_text(encoding="utf-8")
    assert "rmct-convergence-gcall-r2-production-ready-v1" in verifier
    assert "rmct-convergence-gcall-r2-production-source-ready-v1" in verifier
    assert "capture-source-ready" in verifier
    assert "verify-source-ready" in verifier
    assert "comparative_optimality_validated" in verifier
    assert "critical_sources" in verifier
    assert "worker_parity" in verifier
    assert "WORKER_PARITY_SUCCESS" in verifier
    assert "recovery-parent.json" in verifier
    assert "b2a5c3d63323f43563a0281a4efb402a7778cc1e9a8d4a363ada90a9eee89e15" in verifier
    assert "local_gradient_checkpointing_layers" in verifier
    assert "--resume" in verifier
    for critical in (
        "experiments/rmct_convergence/__init__.py",
        "ctm/backends/cli.py",
        "ctm/training/rl.py",
        "scripts/run_experiment.py",
        "rmct_convergence_segment_boundary.py",
        "requirements.txt",
        "preflight_qwen35_rmct_convergence_worker_parity.py",
        "infra/vastai/preflight_qwen35_phase_shared.py",
        "run_qwen35_rmct_convergence_gcall_r2_workq_bootstrap.sbatch",
        "run_qwen35_rmct_convergence_gcall_r2_segment.sbatch",
    ):
        assert critical in verifier
    boundary = BOUNDARY.read_text(encoding="utf-8")
    assert "rmct-convergence-checkpoint-receipt-v1" not in boundary  # imported literal controller schemas
    assert "load_strict_local_rl_resume_state" in boundary


def test_queue_submitter_is_explicit_and_does_not_call_sbatch_without_yes():
    text = SUBMITTER.read_text(encoding="utf-8")
    assert "--yes" in text
    assert "--dependency=\"afterok:$previous_job\"" in text
    assert "verify_rmct_convergence_production_ready.py" in text
    assert "run_qwen35_rmct_convergence_segment.sbatch" in text


def test_broker_forces_offline_resolution_before_readiness_verification():
    text = BROKER.read_text(encoding="utf-8")
    assert "export HF_HUB_OFFLINE=1" in text
    assert "export TRANSFORMERS_OFFLINE=1" in text
    assert text.index("export HF_HUB_OFFLINE=1") < text.index('"$repo_root/.venv/bin/python" "$verify_script"')


def test_segment_zero_submission_waits_before_allocation_then_gates_parity_readiness_and_training():
    for script in (SEGMENT0_SBATCH, SEGMENT0_SUBMITTER):
        assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0
        assert script.stat().st_mode & 0o111

    submitter = SEGMENT0_SUBMITTER.read_text(encoding="utf-8")
    assert "CTM_RMCT_SOURCE_READY_WAIT_SECONDS:-1200" in submitter
    assert ".rmct-convergence-production-ready" in submitter
    assert submitter.index("while true") < submitter.index("job=\"$(sbatch")
    assert "CTM_RMCT_SOURCE_READY_SHA256" in submitter
    assert "verify-source-ready" in submitter

    allocated = SEGMENT0_SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --time=08:00:00" in allocated
    assert "HF_HOME=\"$scratch_root/ctm/huggingface\"" in allocated
    assert "HF_HUB_OFFLINE=1" in allocated
    assert "TRANSFORMERS_OFFLINE=1" in allocated
    assert "ctm_job_tmp" in allocated
    assert allocated.count("exec srun ") == 1
    assert "--resume" in allocated
    assert "verify-source-ready" in allocated
    assert allocated.index("$python_bin \"$parity_helper\"") < allocated.index(
        "$python_bin \"$verify_script\" capture"
    )
    assert allocated.index("$python_bin \"$verify_script\" capture") < allocated.index(
        "$python_bin \"$verify_script\" verify"
    )
    assert allocated.index("$python_bin \"$verify_script\" verify") < allocated.index(
        "--segment-index 0 --yes"
    )


def _load_verifier_module():
    spec = importlib.util.spec_from_file_location("rmct_readiness_verifier_for_test", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_worker_parity_identity_is_nonempty_and_binds_all_completed_sidecar_files(monkeypatch, tmp_path):
    verifier = _load_verifier_module()
    parity_dir = tmp_path / verifier.WORKER_PARITY_DIRECTORY_RELATIVE
    parity_dir.mkdir(parents=True)
    snapshot = tmp_path / "offline-snapshot"
    snapshot.mkdir()
    attestation = parity_dir / verifier.WORKER_PARITY_ATTESTATION
    attestation.write_text(
        json.dumps({"schema": "qwen35-rollout-worker-parity-attestation-v1"}), encoding="utf-8"
    )
    attestation_sha = verifier._sha256(attestation)
    (parity_dir / verifier.WORKER_PARITY_RESULT).write_text(
        json.dumps(
            {
                "schema": "rmct-convergence-worker-parity-fastpath-v1",
                "status": "passed",
                "passed": True,
                "model_snapshot": str(snapshot),
                "attestation_sha256": attestation_sha,
            }
        ),
        encoding="utf-8",
    )
    (parity_dir / verifier.WORKER_PARITY_SUCCESS).write_text("passed\n", encoding="utf-8")
    helper = tmp_path / "infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# helper stub\n", encoding="utf-8")

    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(_args[0], returncode=0, stdout="ok", stderr=""),
    )
    identity = verifier._worker_parity_identity(tmp_path, snapshot={"snapshot_path": str(snapshot)})
    assert identity == {
        "directory": verifier.WORKER_PARITY_DIRECTORY_RELATIVE,
        "helper_resume_validated": True,
        "files": {
            "attestation": verifier._identity(
                tmp_path,
                f"{verifier.WORKER_PARITY_DIRECTORY_RELATIVE}/{verifier.WORKER_PARITY_ATTESTATION}",
                label="expected",
            ),
            "result": verifier._identity(
                tmp_path,
                f"{verifier.WORKER_PARITY_DIRECTORY_RELATIVE}/{verifier.WORKER_PARITY_RESULT}",
                label="expected",
            ),
            "success": verifier._identity(
                tmp_path,
                f"{verifier.WORKER_PARITY_DIRECTORY_RELATIVE}/{verifier.WORKER_PARITY_SUCCESS}",
                label="expected",
            ),
        },
        "result_schema": "rmct-convergence-worker-parity-fastpath-v1",
        "attestation_schema": "qwen35-rollout-worker-parity-attestation-v1",
    }


def test_source_ready_manifest_capture_and_verify_are_recomputable(monkeypatch, tmp_path):
    verifier = _load_verifier_module()
    expected = {
        "schema": verifier.SOURCE_READY_SCHEMA,
        "condition": verifier.CONDITION,
        "source_ready": True,
        "plan": {"path": "plan", "sha256": "a" * 64, "size_bytes": 1},
        "data": {"data": {"sha256": "b" * 64}, "manifest": {"sha256": "c" * 64}},
        "recovery_parent": {"path": "recovery-parent.json", "sha256": "d" * 64, "size_bytes": 1},
        "critical_sources": {"one.py": {"sha256": "d" * 64}},
        "execution_disclosure": {
            "topology_profile": verifier.TOPOLOGY_PROFILE,
            "run_prefix": verifier.RUN_PREFIX,
            "activation_checkpointing_layers": "all",
            "deadline_execution_choice": True,
            "comparative_optimality_validated": False,
            "benchmark_gate_required": False,
        },
    }
    monkeypatch.setattr(verifier, "build_source_ready_receipt", lambda _repository: expected)
    receipt = tmp_path / verifier.SOURCE_READY_FILENAME
    assert verifier._write_immutable(receipt, expected) == "written"
    assert verifier.verify_source_ready_receipt(tmp_path, receipt) == expected


def test_final_readiness_build_binds_mocked_snapshot_and_parity_identity(monkeypatch, tmp_path):
    verifier = _load_verifier_module()
    source_ready = {
        "schema": verifier.SOURCE_READY_SCHEMA,
        "condition": verifier.CONDITION,
        "source_ready": True,
        "plan": {"path": "plan", "sha256": "a" * 64, "size_bytes": 1},
        "data": {"data": {"sha256": "b" * 64}, "manifest": {"sha256": "c" * 64}},
        "recovery_parent": {"path": "recovery-parent.json", "sha256": "d" * 64, "size_bytes": 1},
        "critical_sources": {"one.py": {"sha256": "d" * 64}},
    }
    snapshot = {
        "repo_id": verifier.MODEL,
        "revision": verifier.BASE_SNAPSHOT,
        "snapshot_path": "/offline/snapshot",
        "config_sha256": "e" * 64,
        "offline_only": True,
    }
    parity = {"directory": verifier.WORKER_PARITY_DIRECTORY_RELATIVE, "helper_resume_validated": True}
    compiled = {"compiled_plan_sha256": "f" * 64}
    runtime = {"environment": {"HF_HUB_OFFLINE": "1"}}
    monkeypatch.setattr(verifier, "build_source_ready_receipt", lambda _repository: source_ready)
    monkeypatch.setattr(verifier, "_snapshot_identity", lambda: snapshot)
    monkeypatch.setattr(verifier, "_compiled_contract", lambda _root: compiled)
    monkeypatch.setattr(verifier, "_runtime", lambda: runtime)

    seen: list[tuple[Path, dict[str, str]]] = []

    def parity_identity(root, *, snapshot):
        seen.append((root, snapshot))
        return parity

    monkeypatch.setattr(verifier, "_worker_parity_identity", parity_identity)
    receipt = verifier.build_receipt(tmp_path)
    assert receipt["base_snapshot"] == snapshot
    assert receipt["worker_parity"] == parity
    assert receipt["source_ready"] == source_ready
    assert receipt["recovery_parent"] == source_ready["recovery_parent"]
    assert receipt["compiled_contract"] == compiled
    assert receipt["runtime"] == runtime
    assert seen == [(tmp_path.resolve(), snapshot)]
