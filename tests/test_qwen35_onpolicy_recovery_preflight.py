"""No-GPU contracts for the Qwen3.5 on-policy launch gates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.rmct_paper_vast_dense_models.stage1 import onpolicy_recovery_preflight as gate


ROOT = Path(__file__).parent.parent
GENERIC_WORKER_SCRIPT = ROOT / "infra" / "vastai" / "preflight_qwen35_rollout_workers.sh"
LAUNCHER = ROOT / "infra" / "vastai" / "run_qwen35_onpolicy_recovery.sh"


def _generic_preflight_command(
    *,
    experiment: str,
    run_name: str,
    resume: bool = False,
    script: Path = GENERIC_WORKER_SCRIPT,
) -> list[str]:
    command = [
        "bash",
        str(script),
        "--label",
        "unit",
        "--experiment",
        experiment,
        "--run",
        run_name,
        "--worker-gpus",
        "1,2,3,4,5,6,7",
        "--worker-gpu-mem-util",
        "0.75",
        "--worker-max-model-len",
        "32768",
        "--worker-max-num-seqs",
        "256",
        "--worker-max-num-batched-tokens",
        "8192",
        "--worker-seed-base",
        "42",
        "--target-logprob-chunk-size",
        "2048",
    ]
    if resume:
        command.append("--resume-attestation")
    return command


def _identity(payload: bytes, path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": sum(1 for line in payload.splitlines() if line.strip()),
    }


def _write_minimal_attested_opct_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, dict[str, object]]:
    """Build a no-GPU target sidecar with the real plan compiler in the loop."""

    root = tmp_path / "repo-root"
    source = tmp_path / "pairs.jsonl"
    source.write_text('{"unbiased_messages": [{"role": "user", "content": "q"}], "biased_messages": [{"role": "user", "content": "q"}]}\n', encoding="utf-8")
    manifest = tmp_path / "pairs.manifest.json"
    manifest.write_text('{"recovery": true}\n', encoding="utf-8")
    source_payload = source.read_bytes()

    def exact_verifier(observed_source: str | Path, observed_manifest: str | Path) -> list[dict[str, object]]:
        assert Path(observed_source).resolve() == source.resolve()
        assert Path(observed_manifest).resolve() == manifest.resolve()
        return [{}]

    monkeypatch.setattr(gate, "PROJECT_ROOT", root)
    monkeypatch.setattr(gate, "verify_recovered_none_pairs", exact_verifier)
    monkeypatch.setattr(gate, "RECOVERED_NONE_ROWS", 1)
    monkeypatch.setattr(gate, "RECOVERED_NONE_SOURCE_SHA256", hashlib.sha256(source_payload).hexdigest())
    # Worker-side rebuilding is separately covered by qwen35 compatibility
    # tests. This focused contract test needs only prove it is invoked before
    # the target can be accepted.
    monkeypatch.setattr(gate, "_validate_worker_parity_attestation", lambda **_kwargs: {"validated": True})

    # The target sidecar seals the training-script bytes. The plan compiler
    # remains the real one, while this disposable root supplies the script
    # identity that a real launcher/child would share.
    training_script = root / "scripts" / "train_opct.py"
    training_script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "train_opct.py", training_script)

    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "\n".join(
            [
                "name: unit_onpolicy",
                "training:",
                "  - name: opct_unit",
                "    target: opct",
                '    command: ["${python}", "scripts/train_opct.py"]',
                "    args:",
                "      backend: local",
                "      local_sampler: vllm",
                "      local_device: cuda:0",
                "      local_rollout_gpus: 1,2,3,4,5,6,7",
                "      local_rollout_gpu_mem_util: 0.75",
                "      local_rollout_seed_base: 42",
                "      local_vllm_max_model_len: 32768",
                "      local_vllm_max_num_seqs: 256",
                "      local_vllm_max_num_batched_tokens: 8192",
                "      local_target_logprob_chunk_size: 2048",
                "      model: Qwen/Qwen3.5-9B",
                f"      data: [{json.dumps(str(source) + ':1')}]",
                f"      data_manifest: [{json.dumps(str(manifest))}]",
                "      experiment_name: ${experiment}",
                "      run_name: unit-run",
                "      require_onpolicy_target_attestation: true",
                "      lr: 0.0001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_attestation = root / "logs" / "unit_onpolicy" / "unit-run" / "preflight" / "source.json"
    gate.attest_recovered_none_source(
        source=source,
        source_manifest=manifest,
        output=source_attestation,
    )
    worker_attestation = (
        root
        / "logs"
        / "unit_onpolicy"
        / "unit-run"
        / "rollout_workers"
        / "qwen35-rollout-worker-parity-attestation.json"
    )
    worker_attestation.parent.mkdir(parents=True)
    worker_attestation.write_text('{"worker": "stub"}\n', encoding="utf-8")
    target_attestation = root / "logs" / "unit_onpolicy" / "unit-run" / "preflight" / "target.json"
    result = gate.attest_onpolicy_target(
        plan=plan,
        target="opct",
        source=source,
        source_manifest=manifest,
        source_attestation=source_attestation,
        worker_parity_attestation=worker_attestation,
        output=target_attestation,
        experiment_name="unit_onpolicy",
        run_name="unit-run",
        worker_gpus="1,2,3,4,5,6,7",
        worker_gpu_mem_util=0.75,
        worker_max_model_len=32768,
        worker_max_num_seqs=256,
        worker_max_num_batched_tokens=8192,
        target_logprob_chunk_size=2048,
        cuda_visible_devices="0,1,2,3,4,5,6,7",
    )
    assert result["status"] == "written"
    return plan, source, target_attestation, result["attestation"]


def test_source_attestation_calls_exact_recovered_source_verifier_and_binds_both_input_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "recovered-none.jsonl"
    source_payload = b'{"example": 1}\n'
    source.write_bytes(source_payload)
    manifest = tmp_path / "recovery.manifest.json"
    manifest_payload = b'{"manifest": true}\n'
    manifest.write_bytes(manifest_payload)
    output = tmp_path / "source-attestation.json"
    calls: list[tuple[Path, Path]] = []

    def exact_verifier(observed_source: str | Path, observed_manifest: str | Path) -> list[dict[str, object]]:
        calls.append((Path(observed_source), Path(observed_manifest)))
        return [{}]

    monkeypatch.setattr(gate, "verify_recovered_none_pairs", exact_verifier)
    monkeypatch.setattr(gate, "RECOVERED_NONE_ROWS", 1)
    monkeypatch.setattr(gate, "RECOVERED_NONE_SOURCE_SHA256", hashlib.sha256(source_payload).hexdigest())

    first = gate.attest_recovered_none_source(source=source, source_manifest=manifest, output=output)
    second = gate.attest_recovered_none_source(source=source, source_manifest=manifest, output=output)

    assert first["status"] == "written"
    assert second["status"] == "resumed"
    assert calls == [(source.resolve(), manifest.resolve()), (source.resolve(), manifest.resolve())]
    recorded = json.loads(output.read_text(encoding="utf-8"))
    assert recorded["schema"] == gate.ATTESTATION_SCHEMA
    assert recorded["source"] == _identity(source_payload, source)
    assert recorded["source_manifest"] == _identity(manifest_payload, manifest)
    assert recorded["verification"] == {
        "kind": "exact_legacy_g4_cot_to_none_content_proof",
        "verifier": "supervised_recovery_none_prepare.verify_recovered_none_pairs",
        "expected_recovered_source_sha256": hashlib.sha256(source_payload).hexdigest(),
        "expected_row_count": 1,
    }

    source.write_bytes(b'{"example": 2}\n')
    with pytest.raises(FileExistsError, match="refusing to overwrite different recovered-source attestation"):
        gate.attest_recovered_none_source(source=source, source_manifest=manifest, output=output)


def test_source_preflight_dry_run_invokes_the_exact_verifier_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    source = tmp_path / "source.jsonl"
    manifest = tmp_path / "manifest.json"
    source.write_text('{"source": true}\n', encoding="utf-8")
    manifest.write_text('{"manifest": true}\n', encoding="utf-8")
    calls: list[tuple[Path, Path]] = []

    def exact_verifier(observed_source: str | Path, observed_manifest: str | Path) -> list[dict[str, object]]:
        calls.append((Path(observed_source), Path(observed_manifest)))
        return [{}]

    monkeypatch.setattr(gate, "verify_recovered_none_pairs", exact_verifier)
    monkeypatch.setattr(gate, "RECOVERED_NONE_ROWS", 1)
    gate.main(["--source", str(source), "--source-manifest", str(manifest), "--dry-run"])

    assert calls == [(source.resolve(), manifest.resolve())]
    assert "QWEN35_RECOVERED_NONE_SOURCE_PREFLIGHT_DRY_RUN=" in capsys.readouterr().out
    assert not list(tmp_path.glob("*attestation*"))


def test_generic_worker_preflight_dry_run_derives_immutable_default_status_path_without_gpu_or_write():
    result = subprocess.run(
        [
            "bash",
            str(GENERIC_WORKER_SCRIPT),
            "--label",
            "unit",
            "--experiment",
            "qwen35-gate-unit",
            "--run",
            "opct-lr-1e-4",
            "--worker-gpus",
            "1,2,3,4,5,6,7",
            "--worker-gpu-mem-util",
            "0.75",
            "--worker-max-model-len",
            "32768",
            "--worker-max-num-seqs",
            "256",
            "--worker-max-num-batched-tokens",
            "8192",
            "--worker-seed-base",
            "42",
            "--target-logprob-chunk-size",
            "2048",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    status = ROOT / "logs" / "qwen35-gate-unit" / "opct-lr-1e-4" / "rollout_workers"
    assert result.returncode == 0, result.stderr
    assert "QWEN35_ROLLOUT_WORKER_PREFLIGHT_DRY_RUN=1" in result.stdout
    assert f"status_dir={status}" in result.stdout
    assert f"attestation={status}/qwen35-rollout-worker-parity-attestation.json" in result.stdout
    assert "rollout_worker_logical_gpus=1,2,3,4,5,6,7" in result.stdout
    assert "worker_max_model_len=32768" in result.stdout
    assert "worker_max_num_seqs=256" in result.stdout
    assert not status.exists()


def test_generic_worker_preflight_requires_real_existing_evidence_for_resume_mode():
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    result = subprocess.run(
        [
            "bash",
            str(GENERIC_WORKER_SCRIPT),
            "--label",
            "unit",
            "--experiment",
            "qwen35-gate-missing-resume",
            "--run",
            "opct-lr-1e-4",
            "--worker-gpus",
            "1,2,3,4,5,6,7",
            "--worker-gpu-mem-util",
            "0.75",
            "--worker-max-model-len",
            "32768",
            "--worker-max-num-seqs",
            "256",
            "--worker-max-num-batched-tokens",
            "8192",
            "--worker-seed-base",
            "42",
            "--target-logprob-chunk-size",
            "2048",
            "--resume-attestation",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--resume-attestation requires existing immutable evidence" in result.stderr


def test_generic_worker_preflight_resume_rejects_a_direct_production_session_not_bootstrap_workers(
    tmp_path: Path,
):
    """A direct RolloutWorkerPool session blocks resume even before adapters exist.

    The copied script makes its derived repository root a disposable temporary
    tree.  The nested ``workers/session-*`` fixture mirrors bootstrap evidence;
    only the direct ``rollout_workers/session-*`` child is the production
    marker, so this also guards against accidentally scanning recursively.
    """

    root = tmp_path / "repo"
    copied = root / "infra" / "vastai" / GENERIC_WORKER_SCRIPT.name
    copied.parent.mkdir(parents=True)
    shutil.copy2(GENERIC_WORKER_SCRIPT, copied)
    experiment, run_name = "unit", "resume"
    status = root / "logs" / experiment / run_name / "rollout_workers"
    attestation = status / "qwen35-rollout-worker-parity-attestation.json"
    attestation.parent.mkdir(parents=True)
    attestation.write_text("{}\n", encoding="utf-8")
    bootstrap_session = status / "workers" / "session-bootstrap" / "ipc"
    bootstrap_session.mkdir(parents=True)
    production_session = status / "session-production-before-ipc"
    production_session.mkdir()

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    result = subprocess.run(
        _generic_preflight_command(
            experiment=experiment,
            run_name=run_name,
            resume=True,
            script=copied,
        ),
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert f"detected production rollout session: {production_session}" in result.stderr
    assert "workers/session-bootstrap" not in result.stderr


def test_generic_worker_preflight_resume_rejects_the_immutable_training_started_marker(
    tmp_path: Path,
):
    """A crash after launch hand-off cannot reuse the on-policy namespace."""

    root = tmp_path / "repo"
    copied = root / "infra" / "vastai" / GENERIC_WORKER_SCRIPT.name
    copied.parent.mkdir(parents=True)
    shutil.copy2(GENERIC_WORKER_SCRIPT, copied)
    experiment, run_name = "unit", "started"
    status = root / "logs" / experiment / run_name / "rollout_workers"
    attestation = status / "qwen35-rollout-worker-parity-attestation.json"
    marker = status / "qwen35-onpolicy-training-started.json"
    attestation.parent.mkdir(parents=True)
    attestation.write_text("{}\n", encoding="utf-8")
    marker.write_text('{"schema": "qwen35-onpolicy-training-started-v1"}\n', encoding="utf-8")

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    result = subprocess.run(
        _generic_preflight_command(
            experiment=experiment,
            run_name=run_name,
            resume=True,
            script=copied,
        ),
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert f"detected immutable training-started marker: {marker}" in result.stderr


def test_target_sidecar_rejects_a_mutated_final_child_argument_and_source_before_backend_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan, source, target_attestation, document = _write_minimal_attested_opct_target(tmp_path, monkeypatch)
    child_argv = list(document["child_argv"])
    interpreter = str(document["interpreter_argv0"])
    training_script = tmp_path / "repo-root" / "scripts" / "train_opct.py"

    provenance = gate.validate_onpolicy_target_attestation_for_child(
        attestation=target_attestation,
        expected_training_script="scripts/train_opct.py",
        child_argv=child_argv,
        interpreter=interpreter,
        training_script_path=training_script,
        cuda_visible_devices="0,1,2,3,4,5,6,7",
    )
    assert provenance["target"] == "opct"
    assert provenance["path"] == str(target_attestation.resolve())
    assert document["interpreter_argv0"] == sys.executable
    assert document["interpreter"]["path"] == str(Path(sys.executable).resolve())
    assert len(document["interpreter"]["content_sha256"]) == 64
    assert document["training_script_identity"]["path"] == str(training_script.resolve())
    assert document["cuda_visible_devices"] == ["0", "1", "2", "3", "4", "5", "6", "7"]

    mutated_argv = list(child_argv)
    lr_index = mutated_argv.index("--lr") + 1
    mutated_argv[lr_index] = "0.0002"
    with pytest.raises(ValueError, match="actual on-policy child argv"):
        gate.validate_onpolicy_target_attestation_for_child(
            attestation=target_attestation,
            expected_training_script="scripts/train_opct.py",
            child_argv=mutated_argv,
            interpreter=interpreter,
            training_script_path=training_script,
            cuda_visible_devices="0,1,2,3,4,5,6,7",
        )

    with pytest.raises(ValueError, match="actual on-policy CUDA_VISIBLE_DEVICES"):
        gate.validate_onpolicy_target_attestation_for_child(
            attestation=target_attestation,
            expected_training_script="scripts/train_opct.py",
            child_argv=child_argv,
            interpreter=interpreter,
            training_script_path=training_script,
            cuda_visible_devices="0,1,2,3,4,5,6,8",
        )

    source.write_text('{"mutated": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="attested recovered source changed after attestation"):
        gate.validate_onpolicy_target_attestation_for_child(
            attestation=target_attestation,
            expected_training_script="scripts/train_opct.py",
            child_argv=child_argv,
            interpreter=interpreter,
            training_script_path=training_script,
            cuda_visible_devices="0,1,2,3,4,5,6,7",
        )


def test_target_sidecar_rejects_a_changed_target_argument_when_runner_rereads_the_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    plan, _source, target_attestation, document = _write_minimal_attested_opct_target(tmp_path, monkeypatch)
    gate.validate_onpolicy_target_attestation_for_runner(
        attestation=target_attestation,
        plan=plan,
        target="opct",
        child_argv=document["child_argv"],
        interpreter=document["interpreter_argv0"],
        cuda_visible_devices="0,1,2,3,4,5,6,7",
    )

    with pytest.raises(ValueError, match=r"runner on-policy interpreter argv\[0\]"):
        gate.validate_onpolicy_target_attestation_for_runner(
            attestation=target_attestation,
            plan=plan,
            target="opct",
            child_argv=document["child_argv"],
            interpreter="/different/python",
            cuda_visible_devices="0,1,2,3,4,5,6,7",
        )

    with pytest.raises(ValueError, match="runner on-policy CUDA_VISIBLE_DEVICES"):
        gate.validate_onpolicy_target_attestation_for_runner(
            attestation=target_attestation,
            plan=plan,
            target="opct",
            child_argv=document["child_argv"],
            interpreter=document["interpreter_argv0"],
            cuda_visible_devices="0,1,2,3,4,5,6,8",
        )

    plan.write_text(plan.read_text(encoding="utf-8").replace("lr: 0.0001", "lr: 0.0002"), encoding="utf-8")
    with pytest.raises(ValueError, match="authored on-policy YAML changed after attestation"):
        gate.validate_onpolicy_target_attestation_for_runner(
            attestation=target_attestation,
            plan=plan,
            target="opct",
            child_argv=document["child_argv"],
            interpreter=document["interpreter_argv0"],
            cuda_visible_devices="0,1,2,3,4,5,6,7",
        )


def test_target_contract_rejects_a_plan_source_different_from_the_source_attested_by_the_launcher():
    plan = ROOT / "experiments" / "rmct_paper_vast_dense_models" / "stage1" / "qwen3_5_9b_rmct_paper_fidelity_20260803.yaml"
    source = ROOT / "artifacts" / "rmct-hle-dense-models-shared-qwen3.5-none-20260801" / "data" / "distractor-argument-pairs.jsonl"
    manifest = source.with_suffix(".manifest.json")

    report = gate.verify_onpolicy_target_contract(
        plan=plan,
        target="rmct-main",
        source=source,
        source_manifest=manifest,
        experiment_name="rmct_paper_vast_dense_qwen3_5_9b_batching_repair_20260803",
        run_name="rate-matching-lr-1e-4",
        worker_gpus="1,2,3,4,5,6,7",
        worker_gpu_mem_util=0.75,
        worker_max_model_len=32768,
        worker_max_num_seqs=256,
        worker_max_num_batched_tokens=8192,
        target_logprob_chunk_size=2048,
    )

    assert report["schema"] == gate.TARGET_CONTRACT_SCHEMA
    assert report["source_path"] == str(source.resolve())
    with pytest.raises(ValueError, match="rmct-main frozen pair source"):
        gate.verify_onpolicy_target_contract(
            plan=plan,
            target="rmct-main",
            source=ROOT / "artifacts" / "not-the-frozen-source.jsonl",
            source_manifest=manifest,
            experiment_name="rmct_paper_vast_dense_qwen3_5_9b_batching_repair_20260803",
            run_name="rate-matching-lr-1e-4",
            worker_gpus="1,2,3,4,5,6,7",
            worker_gpu_mem_util=0.75,
            worker_max_model_len=32768,
            worker_max_num_seqs=256,
            worker_max_num_batched_tokens=8192,
            target_logprob_chunk_size=2048,
        )


@pytest.mark.parametrize(
    ("target", "experiment", "run_name", "command_name"),
    [
        (
            "rmct-main",
            "rmct_paper_vast_dense_qwen3_5_9b_batching_repair_20260803",
            "rate-matching-lr-1e-4",
            "rate_matching_lr1",
        ),
        (
            "opct",
            "rmct_paper_vast_dense_qwen3_5_9b_opct_recovery_20260803",
            "opct-lr-1e-4",
            "opct_lr1",
        ),
    ],
)
def test_combined_launcher_dry_run_wires_exact_source_and_worker_gates_before_target_plan(
    target: str,
    experiment: str,
    run_name: str,
    command_name: str,
):
    environment = dict(os.environ)
    environment["CTM_PYTHON"] = sys.executable
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run(
        ["bash", str(LAUNCHER), target, "--dry-run"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    source = ROOT / "artifacts" / "rmct-hle-dense-models-shared-qwen3.5-none-20260801" / "data" / "distractor-argument-pairs.jsonl"
    manifest = source.with_suffix(".manifest.json")
    identity = ROOT / "logs" / experiment / run_name / "preflight" / "qwen35-recovered-none-source-attestation.json"
    assert result.returncode == 0, result.stderr
    assert "QWEN35_ONPOLICY_RECOVERY_LAUNCH_DRY_RUN=1" in result.stdout
    assert f"source={source}" in result.stdout
    assert f"source_manifest={manifest}" in result.stdout
    assert f"source_attestation={identity}" in result.stdout
    assert "source_gate=python -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_recovery_preflight" in result.stdout
    assert "QWEN35_ONPOLICY_TARGET_CONTRACT=" in result.stdout
    assert "QWEN35_ROLLOUT_WORKER_PREFLIGHT_DRY_RUN=1" in result.stdout
    assert f"[training:{command_name}]" in result.stdout
    assert not identity.exists()
