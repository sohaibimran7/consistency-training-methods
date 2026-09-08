"""Focused safety tests for the isolated step-16/64 16-GPU evaluator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.py"
SBATCH = ROOT / "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.sbatch"
WORKER = ROOT / "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu_worker.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("rmct_checkpoint_two_bias_16gpu_for_test", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _strict_replay_fixture(module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Construct only the local receipt files needed to exercise replay logic."""

    target = module.APPROVED_TARGETS[16]
    paths = module._paths(tmp_path / target.condition, target=target)
    paths.root.mkdir()
    _write_json(paths.contract, {"fixture": True})
    adapter = paths.runtime / "compatibility" / "adapter-0001"
    adapter.mkdir(parents=True)
    attestation = adapter / "vllm-parity-attestation.json"
    attestation.write_text('{"fixture":true}\n', encoding="utf-8")
    raw = {
        "path": str(tmp_path / "sealed-raw-checkpoint"),
        "adapter_model_sha256": "a" * 64,
        "adapter_config_sha256": "b" * 64,
    }
    sampler = {"fixture": "frozen-sampler"}
    compatibility = {
        "path": str(adapter),
        "adapter_model_sha256": "c" * 64,
        "adapter_config_sha256": "d" * 64,
        "compatibility_manifest": {"path": str(adapter / "compatibility-manifest.json"), "sha256": "e" * 64},
        "parity_attestation": {"path": str(attestation), "sha256": module._sha256_file(attestation), "schema": "fixture"},
        "parity_reports": [{"path": str(adapter / "report.json"), "sha256": "f" * 64}],
        "source_raw_checkpoint": raw["path"],
        "source_raw_adapter_model_sha256": raw["adapter_model_sha256"],
        "base_model": str(module.MODEL_SNAPSHOT),
    }
    calls: list[Path] = []

    def fake_compatibility_identity(candidate: Path, *, raw: object, sampler: object):
        calls.append(candidate)
        return compatibility

    monkeypatch.setattr(module, "_validated_launch_runtime_policy", lambda launch: (raw, sampler))
    monkeypatch.setattr(module, "_validate_translation_only", lambda raw, adapter: None)
    monkeypatch.setattr(module, "_compatibility_identity", fake_compatibility_identity)
    runtime = module._runtime_receipt_document(
        raw=raw,
        sampler=sampler,
        compatibility=compatibility,
        parity_attestation=module._identity(attestation, label="fixture attestation"),
    )
    _write_json(paths.runtime_receipt, runtime)
    launch = {
        "condition": target.condition,
        "target": {
            "step": target.step,
            "condition": target.condition,
            "run_prefix": target.run_prefix,
            "run_name": target.run_name,
            "segment_index": target.segment_index,
        },
        "checkpoint_custody": {"fixture": "sealed-custody"},
        "deployment_manifest": {"path": str(paths.deployment_manifest), "fixture": "exact-substrate"},
    }
    evaluation = module._build_evaluation_receipt(launch=launch, runtime=runtime, paths=paths)
    _write_json(paths.evaluation_receipt, evaluation)
    return target, paths, launch, runtime, evaluation, adapter, calls


def _launch_replay_fixture(module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a fully deterministic read-only launch replay fixture."""

    target = module.APPROVED_TARGETS[16]
    paths = module._paths(tmp_path / target.condition, target=target)
    paths.root.mkdir()
    repository = tmp_path / "sealed-training-repository"
    repository.mkdir()
    source = tmp_path / "source-stage2-manifest.json"
    source.write_text('{"source":true}\n', encoding="utf-8")
    artifacts = tmp_path / "stage2-artifacts"
    artifacts.mkdir()
    parity_data = tmp_path / "train-eval-n200.jsonl"
    parity_data.write_text("fixture\n", encoding="utf-8")
    parity_manifest = tmp_path / "parity-manifest.json"
    parity_manifest.write_text('{"fixture":true}\n', encoding="utf-8")
    target_record = {
        "step": target.step,
        "condition": target.condition,
        "run_prefix": target.run_prefix,
        "run_name": target.run_name,
        "segment_index": target.segment_index,
    }
    raw = {
        "path": str(repository / "checkpoint"),
        "adapter_model_sha256": "a" * 64,
        "adapter_config_sha256": "b" * 64,
    }
    custody = {"target": target_record, "checkpoint": raw, "fixture": "sealed"}
    snapshot = {"path": str(tmp_path / "snapshot"), "config": {"sha256": "c" * 64}}
    deployment = {
        "schema": "fixture-deployment",
        "manifest_path": str(paths.deployment_manifest),
        "manifest_sha256": "d" * 64,
        "source_manifest_sha256": module._sha256_file(source),
        "artifact_root": str(artifacts),
        "artifacts": {"in_domain/clean": {"sha256": "e" * 64}},
    }
    substrate = {"fixture": "exact-substrate"}
    parity = {"path": str(parity_data), "canonical_manifest": {"path": str(parity_manifest)}}
    critical_state = {"fixture.py": {"path": "/fixture.py", "sha256": "f" * 64, "size_bytes": 1}}
    sampler = module._sampler_runtime(("0", "1", "2", "3"))
    monkeypatch.setattr(module, "validate_approved_target", lambda repository, step: dict(custody))
    monkeypatch.setattr(module, "_validate_model_snapshot", lambda: dict(snapshot))
    monkeypatch.setattr(module, "_replay_deployment_manifest", lambda **kwargs: dict(deployment))
    monkeypatch.setattr(module, "_validate_two_bias_substrate", lambda manifest: dict(substrate))
    monkeypatch.setattr(module, "_parity_data", lambda data, manifest: dict(parity))
    monkeypatch.setattr(module, "_critical_source_identities", lambda: dict(critical_state))
    launch = {
        "schema": module.LAUNCH_SCHEMA,
        "condition": target.condition,
        "target": target_record,
        "repository": str(repository),
        "checkpoint_custody": custody,
        "model_snapshot": snapshot,
        "source_stage2_manifest": module._identity(source, label="fixture source"),
        "stage2_artifact_root": str(artifacts),
        "deployment_manifest": {
            "path": str(paths.deployment_manifest),
            "provenance": deployment,
            "substrate": substrate,
        },
        "parity_data": parity,
        "critical_sources": dict(critical_state),
        "runtime": module._launch_runtime_contract(sampler),
        "matrix": module._launch_matrix_contract(),
        "outputs": module._launch_outputs_contract(paths),
        "policy": module._launch_policy_contract(),
    }
    _write_json(paths.contract, launch)
    return target, paths, launch, critical_state


def _identity(path: Path, token: int) -> dict[str, object]:
    return {"path": str(path), "sha256": f"{token:064x}", "size_bytes": 1}


def _native_report_fixture(module, tmp_path: Path) -> dict[str, object]:
    """Build a shape-valid report so negative tests exercise r002 validation."""

    target = module.APPROVED_TARGETS[16]
    raw_root = tmp_path / "paired-clean-ready"
    identities = module._expected_r002_task_identities()
    clean_by_population_dataset: dict[tuple[str, str], dict[str, object]] = {}
    sources: list[dict[str, object]] = []
    for task_index, (kind, regime, population, dataset, bias_type) in enumerate(identities, start=1):
        qid_sha = f"{(100 + (hash((population, dataset)) % 1000)):064x}"
        source: dict[str, object] = {
            "task_index": task_index,
            "kind": kind,
            "regime": regime,
            "population": population,
            "dataset": dataset,
            "bias_type": bias_type,
            "raw_log": str(raw_root / f"task-{task_index:03d}" / "log.eval"),
            "raw_log_sha256": f"{task_index:064x}",
            "created": "2026-08-21T00:00:00Z",
            "sample_count": module._sample_count_for_task(task_index),
            "expected_sample_count": module._sample_count_for_task(task_index),
            "question_ids_sha256": qid_sha,
            "frozen_file": str(tmp_path / f"frozen-{task_index}.jsonl"),
            "frozen_file_sha256": f"{(200 + task_index):064x}",
            "source_identity_digest": "stage2-ood-hle-2x2:" + "a" * 64,
            "prompt_style": "none",
            "model": "vllm/Qwen/Qwen3.5-9B:/fixture/adapter",
            "runtime": {"profile": "vllm"},
            "task_receipt": _identity(tmp_path / f"task-{task_index:03d}.json", 300 + task_index),
            "evaluation_bias_status": None if kind == "unbiased" else ("seen" if bias_type in module.SEEN_BIASES else "held_out"),
        }
        if kind == "unbiased":
            clean_by_population_dataset[(population, dataset)] = source
        else:
            clean = clean_by_population_dataset[(population, dataset)]
            source["unbiased_log"] = str(raw_root)
            source["paired_clean"] = {
                "raw_log": clean["raw_log"],
                "raw_log_sha256": clean["raw_log_sha256"],
                "question_ids_sha256": clean["question_ids_sha256"],
                "source_identity_digest": clean["source_identity_digest"],
            }
            # The native validator requires the biased subset to equal its
            # matching clean subset, not merely have the same count.
            source["question_ids_sha256"] = clean["question_ids_sha256"]
        sources.append(source)
    return {
        "schema": module.NATIVE_PREFLIGHT_SCHEMA,
        "condition": target.condition,
        "target": {"step": target.step, "run_name": target.run_name, "segment_index": target.segment_index},
        "evaluation_receipt": _identity(tmp_path / "evaluation-receipt.json", 401),
        "raw_root": str(raw_root),
        "clean_gate": _identity(tmp_path / "clean-gate-receipt.json", 402),
        "sampling": module._sampling_contract(),
        "science": {
            "all_biases": list(module.ALL_BIASES),
            "seen_biases": list(module.SEEN_BIASES),
            "held_out_biases": list(module.HELD_OUT_BIASES),
            "include_bias_acknowledged": False,
            "grader_model": None,
        },
        "sources": sources,
    }


def test_entrypoints_are_executable_syntax_valid_and_do_not_operate_the_scheduler():
    assert LAUNCHER.stat().st_mode & 0o111
    assert SBATCH.stat().st_mode & 0o111
    assert WORKER.stat().st_mode & 0o111
    assert subprocess.run(["bash", "-n", str(SBATCH)], check=False).returncode == 0
    assert subprocess.run(["bash", "-n", str(WORKER)], check=False).returncode == 0
    result = subprocess.run([sys.executable, str(LAUNCHER), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    launcher = LAUNCHER.read_text(encoding="utf-8")
    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "subprocess.run([\"sbatch\"" not in launcher
    assert "sbatch " not in sbatch
    assert "scancel " not in sbatch
    assert "--dependency" not in sbatch
    assert "afterok" not in sbatch
    assert "#SBATCH --nodes=4" in sbatch
    assert "#SBATCH --gpus-per-node=4" in sbatch
    assert "#SBATCH --cpus-per-gpu=16" in sbatch
    assert "#SBATCH --ntasks=16" not in sbatch
    assert "#SBATCH --ntasks-per-node=4" not in sbatch
    assert "#SBATCH --cpus-per-task=16" not in sbatch
    assert "--nodes=1 --ntasks=1 --gpus-per-task=4 --cpus-per-task=64" in sbatch
    assert "--nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=16" in sbatch
    assert "--kill-on-bad-exit=0" in sbatch
    assert "seal_phase \"$phase\" || true" in sbatch
    assert "stage2-ood-hle-2x2-20260802-r1-source/manifest.json" in sbatch
    assert "rmct-convergence-step16-step64-16gpu-v1-r002" in sbatch
    assert "rmct-convergence-step016-two-bias-v1-r002" in sbatch
    assert "rmct-convergence-step064-two-bias-v1-r002" in sbatch
    # The evaluation venv's Python is normally a symlink; it must be accepted
    # while the more meaningful sys.prefix guard remains in force.
    assert '-L "$python_requested"' not in sbatch
    assert "sys.prefix == sys.base_prefix" in sbatch
    worker = WORKER.read_text(encoding="utf-8")
    assert '"$python_bin" || -L "$python_bin"' not in worker
    assert '[[ ! -e "$python_bin" || ! -x "$python_bin" ]]' in worker


def test_only_reviewed_step16_and_step64_targets_are_accepted_with_exact_identities():
    module = _load_module()
    assert set(module.APPROVED_TARGETS) == {16, 64}
    first = module.APPROVED_TARGETS[16]
    second = module.APPROVED_TARGETS[64]
    assert (first.run_name, first.segment_index, first.condition) == (
        "rmct-convergence-gcall-r2-s001",
        0,
        "rmct-convergence-step016-two-bias-v1-r002",
    )
    assert (second.run_name, second.segment_index, second.condition) == (
        "rmct-convergence-gcall-r2-mb49152-r3-s004",
        3,
        "rmct-convergence-step064-two-bias-v1-r002",
    )
    assert first.checkpoint_relative.endswith("rmct-convergence_rmct-convergence-gcall-r2-s001")
    assert second.checkpoint_relative.endswith("rmct-convergence_rmct-convergence-gcall-r2-mb49152-r3-s004")
    assert first.checkpoint_receipt_relative.endswith("segment/checkpoint-receipt.json")
    assert second.completion_receipt_relative.endswith("segment/completion-receipt.json")
    with pytest.raises(module.EvaluationError, match="only the reviewed"):
        module._target(176)


def test_exact_three_phase_plan_covers_each_target_cell_once_with_two_idle_slots_per_phase():
    module = _load_module()
    assert module.PHASES == {
        1: {16: tuple(range(1, 15))},
        2: {16: tuple(range(15, 22)), 64: tuple(range(1, 8))},
        3: {64: tuple(range(8, 22))},
    }
    cells = [(step, index) for phase in module.PHASES.values() for step, indices in phase.items() for index in indices]
    assert len(cells) == 42
    assert len(set(cells)) == 42
    assert set(cells) == {(step, index) for step in (16, 64) for index in range(1, 22)}
    assert [sum(len(indices) for indices in phase.values()) for phase in module.PHASES.values()] == [14, 14, 14]


def test_r002_mixed_sample_plan_is_exact_per_task_and_total_and_reaches_task_commands(tmp_path: Path):
    module = _load_module()
    assert module.IID_CLEAN_TASK_INDICES == (1, 2)
    assert module.HLE_CLEAN_TASK_INDICES == (3,)
    assert module.IID_BIASED_TASK_INDICES == (4, 5, *range(7, 17))
    assert module.HLE_BIASED_TASK_INDICES == (6, *range(17, 22))
    assert {index for index, count in module.TASK_SAMPLE_COUNTS.items() if count == 50} == set(module.IID_TASK_INDICES)
    assert {index for index, count in module.TASK_SAMPLE_COUNTS.items() if count == 100} == set(module.HLE_TASK_INDICES)
    assert module.TOTAL_SAMPLES_PER_CHECKPOINT == 1400
    assert sum(row["sample_count"] for row in module._task_sample_count_records()) == 1400
    assert module._sampling_contract()["total_samples_per_checkpoint"] == 1400
    assert [row["task_index"] for row in module._sampling_contract()["task_identities"]] == list(range(1, 22))

    paths = module._paths(tmp_path / module.APPROVED_TARGETS[16].condition, target=module.APPROVED_TARGETS[16])
    for task_index, expected_limit in ((1, 50), (2, 50), (3, 100), (4, 50), (6, 100), (16, 50), (17, 100), (21, 100)):
        command = module._task_command(
            launch={"deployment_manifest": {"path": str(tmp_path / "manifest.json")}},
            receipt={"runtime": {"checkpoint": str(tmp_path / "adapter")}},
            paths=paths,
            attempt=tmp_path / f"attempt-{task_index}",
            task_index=task_index,
            python="/unit/python",
        )
        assert _option_value(command, "--limit") == str(expected_limit)


def test_r002_static_task_identity_and_full_question_pools_reject_reordering_or_drift():
    module = _load_module()
    pools: dict[tuple[str, str], tuple[str, ...]] = {}
    specs: list[SimpleNamespace] = []
    for task_index, identity in enumerate(module._expected_r002_task_identities(), start=1):
        kind, regime, population, dataset, bias_type = identity
        key = (population, dataset)
        pools.setdefault(key, tuple(f"{population}-{dataset}-{item:03d}" for item in range(100)))
        specs.append(
            SimpleNamespace(
                kind=kind,
                regime=regime,
                population=population,
                dataset=dataset,
                bias_type=bias_type,
                question_ids=pools[key],
            )
        )
    module._validate_r002_sampling_matrix(specs)

    reordered = list(specs)
    reordered[3] = SimpleNamespace(**{**vars(reordered[3]), "dataset": "hellaswag"})
    with pytest.raises(module.EvaluationError, match="task-4"):
        module._validate_r002_sampling_matrix(reordered)

    drifted = list(specs)
    drifted[6] = SimpleNamespace(**{**vars(drifted[6]), "question_ids": tuple(reversed(drifted[6].question_ids))})
    with pytest.raises(module.EvaluationError, match="shares one ordered"):
        module._validate_r002_sampling_matrix(drifted)


def test_native_r002_preflight_rejects_count_pair_and_index_identity_tampering(tmp_path: Path):
    module = _load_module()
    report = _native_report_fixture(module, tmp_path)
    assert module.validate_native_preflight(report) == report

    count_tampered = deepcopy(report)
    count_tampered["sources"][5]["sample_count"] = 50  # task 6 is HLE and must retain 100.
    with pytest.raises(module.EvaluationError, match="sample-count"):
        module.validate_native_preflight(count_tampered)

    pair_tampered = deepcopy(report)
    pair_tampered["sources"][3]["question_ids_sha256"] = "f" * 64
    with pytest.raises(module.EvaluationError, match="matching clean question-id subset"):
        module.validate_native_preflight(pair_tampered)

    identity_tampered = deepcopy(report)
    identity_tampered["sources"][3]["dataset"] = "hellaswag"
    with pytest.raises(module.EvaluationError, match="identity differs"):
        module.validate_native_preflight(identity_tampered)


def test_converter_and_parity_reservations_keep_the_destination_absent_and_preserve_markers(tmp_path: Path):
    module = _load_module()
    root = tmp_path / "runtime"
    adapter = module._reserve_uncreated_output_destination(root / "compatibility", label="adapter")
    adapter_marker = adapter.parent / ".adapter-0001.reservation.json"
    assert not adapter.exists()
    assert json.loads(adapter_marker.read_text(encoding="utf-8"))["destination"] == str(adapter)
    # This models the converter's documented destination contract: its output
    # leaf must not exist until the converter itself creates it.
    assert not adapter.exists()
    adapter.mkdir()
    parity = module._reserve_uncreated_output_destination(root / "parity-attempts", label="parity")
    parity_marker = parity.parent / ".parity-0001.reservation.json"
    assert not parity.exists()
    assert json.loads(parity_marker.read_text(encoding="utf-8"))["destination"] == str(parity)
    parity.mkdir()
    next_adapter = module._reserve_uncreated_output_destination(root / "compatibility", label="adapter")
    assert next_adapter.name == "adapter-0002"
    assert adapter_marker.exists()  # failed/interrupted reservation evidence is never deleted

    text = LAUNCHER.read_text(encoding="utf-8")
    assert '_reserve_uncreated_output_destination(compatibility_root, label="adapter")' in text
    assert '_reserve_uncreated_output_destination(parity_root, label="parity")' in text
    assert "_next_attempt(paths.attempts" in text  # only ordinary task log dirs are precreated


def test_frozen_r005_generation_controls_and_switch_wait_contract_are_retained():
    module = _load_module()
    assert module.GENERATION_CONFIG == {
        "max_tokens": 20480,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "extra_body": {"top_k": 20},
    }
    assert module.VLLM_MODEL_ARGS == {
        "provider": "vllm",
        "gpu_memory_utilization": 0.9,
        "max_model_len": 32768,
        "language_model_only": True,
        "max_num_seqs": 256,
        "gdn_prefill_backend": "triton",
    }
    assert module.VLLM_VERSION == "0.21.0"
    assert module.PARITY_VARIANTS == ("full", "linear_only", "self_attn_only", "evaluator_path")
    assert module.PARITY_MAX_LOGPROBS == 29
    assert module.PARITY_LOGPROBS_MODE == "processed_logprobs"
    assert module.PARITY_SCORE_TRANSPORT == "allowed_token_ids_restricted_softmax"

    text = LAUNCHER.read_text(encoding="utf-8")
    assert '"total_samples_per_checkpoint": TOTAL_SAMPLES_PER_CHECKPOINT' in text
    assert module.TOTAL_SAMPLES_PER_CHECKPOINT == 1400
    assert '"switch_scorer_wait_timeout_seconds": 3600' in text
    assert "clean_before_biased" not in text
    assert "_require_clean_barrier" not in text


def test_critical_sources_bind_runner_and_stage2_materialization_dependencies():
    module = _load_module()
    assert {
        "ctm/evals/runner.py",
        "experiments/stage2_ood_hle/materialize.py",
        "experiments/stage2_ood_hle/prepare.py",
        "experiments/rmct_two_bias_eval/raw_preflight.py",
    }.issubset(module.CRITICAL_SOURCES)


def test_snapshot_aware_native_runtime_accepts_the_pinned_snapshot_and_rejects_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Exercise r005's real header checker without importing heavy Stage-2 deps."""

    module = _load_module()
    source = LAUNCHER.read_text(encoding="utf-8")
    native_preflight = source[source.index("def _native_two_bias_preflight") : source.index("def _is_sha256")]
    assert "r005_raw_preflight._assert_snapshot_vllm_runtime" in source
    assert "mechanical._validate_runtime_contract" not in native_preflight
    assert "mechanical._assert_runtime" not in native_preflight
    fake_legacy = SimpleNamespace(
        _attribute=lambda value, name, default=None: value.get(name, default)
        if isinstance(value, dict)
        else getattr(value, name, default)
    )
    fake_stage2 = SimpleNamespace(raw_preflight=fake_legacy)
    monkeypatch.setitem(sys.modules, "experiments.stage2_ood_hle", fake_stage2)
    monkeypatch.setitem(sys.modules, "experiments.stage2_ood_hle.raw_preflight", fake_legacy)

    checkpoint = str(tmp_path / "runtime" / "compatibility" / "adapter-0001")
    runtime = {
        "profile": "vllm",
        "base_model": str(module.MODEL_SNAPSHOT),
        "model_args": dict(module.VLLM_MODEL_ARGS),
        "generation_config": dict(module.GENERATION_CONFIG),
        "checkpoint": checkpoint,
    }
    header = SimpleNamespace(
        eval=SimpleNamespace(
            model=f"vllm/{module.MODEL_SNAPSHOT}:{checkpoint}",
            metadata={
                "checkpoint": checkpoint,
                "checkpoint_backend": "local",
                "model_args": dict(module.VLLM_MODEL_ARGS),
                "generation_config": dict(module.GENERATION_CONFIG),
            },
            model_args={key: value for key, value in module.VLLM_MODEL_ARGS.items() if key != "provider"},
            model_generate_config=dict(module.GENERATION_CONFIG),
        )
    )
    path = tmp_path / "raw" / "task-001.eval"

    model, observed = module._assert_early_snapshot_vllm_runtime(header, path=path, runtime=runtime)
    assert model == f"vllm/{module.MODEL_SNAPSHOT}:{checkpoint}"
    assert observed == runtime

    alias_header = deepcopy(header)
    alias_header.eval.model = f"vllm/{module.MODEL_ALIAS}:{checkpoint}"
    with pytest.raises(module.EvaluationError, match="wrong parity-attested vLLM model"):
        module._assert_early_snapshot_vllm_runtime(alias_header, path=path, runtime=runtime)

    changed_model_args = deepcopy(header)
    changed_model_args.eval.metadata["model_args"]["gdn_prefill_backend"] = "flashinfer"
    with pytest.raises(module.EvaluationError, match="metadata.model_args"):
        module._assert_early_snapshot_vllm_runtime(changed_model_args, path=path, runtime=runtime)

    changed_generation = deepcopy(header)
    changed_generation.eval.model_generate_config["top_k"] = 99
    with pytest.raises(module.EvaluationError, match="native-vLLM generation top_k"):
        module._assert_early_snapshot_vllm_runtime(changed_generation, path=path, runtime=runtime)


def test_clean_gate_is_the_shared_raw_root_and_clean_receipt_precedes_publication(tmp_path: Path):
    module = _load_module()
    output = tmp_path / module.APPROVED_TARGETS[16].condition
    paths = module._paths(output, target=module.APPROVED_TARGETS[16])
    assert paths.raw == output / "stage2" / "paired-clean-ready"
    assert paths.clean_gate_receipt.parent == paths.raw

    command = module._task_command(
        launch={"deployment_manifest": {"path": str(tmp_path / "manifest.json")}},
        receipt={"runtime": {"checkpoint": str(tmp_path / "adapter")}},
        paths=paths,
        attempt=tmp_path / "attempt",
        task_index=4,
        python="/unit/python",
    )
    assert command.count("--task-index") == 1
    assert _option_value(command, "--task-index") == "4"
    task_args = json.loads(_option_value(command, "--task-args"))
    assert task_args["unbiased_log"] == str(paths.raw)
    assert task_args["include_bias_acknowledged"] is False
    assert _option_value(command, "--limit") == "50"
    assert "--persistent-vllm-server" in command
    assert "--isolate-tasks" in command

    text = LAUNCHER.read_text(encoding="utf-8")
    clean_branch = text.index("if task_index in CLEAN_TASK_INDICES:\n        _write_immutable_json", text.index("def _promote_attempt"))
    gate_copy = text.index("_copy_immutable_file(selected, canonical, label=f\"task-{task_index} clean gate EvalLog\")", clean_branch)
    assert clean_branch < gate_copy


def test_operator_facing_roots_reject_direct_symlink_spellings(tmp_path: Path):
    module = _load_module()
    target_root = tmp_path / module.APPROVED_TARGETS[16].condition
    target_root.mkdir()
    target_link = tmp_path / "target-link"
    target_link.symlink_to(target_root, target_is_directory=True)
    with pytest.raises(module.EvaluationError, match="must not be a symlink|regular directory"):
        module._paths(target_link, target=module.APPROVED_TARGETS[16])

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    campaign_link = tmp_path / "campaign-link"
    campaign_link.symlink_to(campaign, target_is_directory=True)
    with pytest.raises(module.EvaluationError, match="regular directory"):
        module._campaign_path(campaign_link)


def test_local_worker_mapping_uses_logical_rank_not_global_cuda_token_identity():
    worker = WORKER.read_text(encoding="utf-8")
    assert "rank=${SLURM_PROCID:-}" in worker
    assert "1:13) step=16; task_index=14" in worker
    assert "2:6) step=16; task_index=21" in worker
    assert "2:7) step=64; task_index=1" in worker
    assert "3:13) step=64; task_index=21" in worker
    assert "*) exit 0 ;;" in worker
    assert "local GPU token \"0\" is valid" in worker
    assert "CUDA_VISIBLE_DEVICES" in worker
    assert "--phase \"$phase\"" in worker


def test_parity_gpu_tokens_are_local_four_token_identity_and_reject_duplicates():
    module = _load_module()
    assert module._parse_four_gpu_tokens("GPU-a,GPU-b,GPU-c,GPU-d") == ("GPU-a", "GPU-b", "GPU-c", "GPU-d")
    with pytest.raises(module.EvaluationError, match="four distinct"):
        module._parse_four_gpu_tokens("GPU-a,GPU-a,GPU-c,GPU-d")
    sampler = module._sampler_runtime(("0", "1", "2", "3"))
    assert sampler["vllm_device_tokens"] == ["0", "1", "2", "3"]
    assert sampler["parallel_isolated_vllm_variants"] is True
    assert sampler["one_adapter_per_server"] is True


def test_later_phase_admission_is_receipt_gated_and_step64_prepare_requires_phase_one():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "validate_worker_phase_admission" in text
    assert "step-64 prepare requires phase-one campaign/root/receipt evidence" in text
    assert "validate_phase_receipt(\n                    args.step16_phase_receipt" in text
    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "seal_phase 1" in sbatch
    assert "prepare_step 64 \"$step64_root\" \"$phase_one_receipt\"" in sbatch
    assert "seal_phase 2" in sbatch
    assert "run_phase 3" in sbatch


def test_strict_evaluation_replay_recomputes_attested_runtime_and_rejects_all_evaluation_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_module()
    target, paths, launch, runtime, evaluation, adapter, calls = _strict_replay_fixture(module, tmp_path, monkeypatch)
    launch_sha256 = module._sha256_file(paths.contract)

    loaded, loaded_sha256 = module._load_evaluation_receipt(
        paths,
        target=target,
        launch=launch,
        launch_sha256=launch_sha256,
    )
    assert loaded == evaluation
    assert loaded_sha256 == module._sha256_file(paths.evaluation_receipt)
    assert calls == [adapter]

    mutations = {
        "checkpoint": lambda record: record["runtime"].__setitem__("checkpoint", "/tmp/not-attested"),
        "runtime": lambda record: record["runtime"].__setitem__("model_args", {"provider": "hf"}),
        "substrate": lambda record: record.__setitem__("stage2_substrate", {"forged": True}),
        "custody": lambda record: record.__setitem__("checkpoint_custody", {"forged": True}),
    }
    for label, mutate in mutations.items():
        altered = json.loads(json.dumps(evaluation))
        mutate(altered)
        _write_json(paths.evaluation_receipt, altered)
        with pytest.raises(module.EvaluationError, match="exact replayed"):
            module._load_evaluation_receipt(
                paths,
                target=target,
                launch=launch,
                launch_sha256=launch_sha256,
            )
        _write_json(paths.evaluation_receipt, evaluation)


def test_runtime_replay_rejects_external_adapter_path_before_any_evaluation_log_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_module()
    _target, paths, launch, runtime, _evaluation, _adapter, _calls = _strict_replay_fixture(module, tmp_path, monkeypatch)
    altered = json.loads(json.dumps(runtime))
    altered["compatibility_adapter"]["path"] = "/tmp/not-attested"
    _write_json(paths.runtime_receipt, altered)
    with pytest.raises(module.EvaluationError, match="compatibility adapter"):
        module._load_runtime_receipt(paths=paths, launch=launch)


def test_runtime_replay_rejects_a_changed_launch_serving_policy():
    module = _load_module()
    sampler = module._sampler_runtime(("0", "1", "2", "3"))
    raw = {
        "path": "/sealed/checkpoint",
        "adapter_model_sha256": "a" * 64,
        "adapter_config_sha256": "b" * 64,
    }
    launch = {
        "checkpoint_custody": {"checkpoint": raw},
        "runtime": {
            "profile": "vllm",
            "sampler": sampler,
            "model_args": dict(module.VLLM_MODEL_ARGS),
            "generation_config": dict(module.GENERATION_CONFIG),
            "hf_fallback_permitted": False,
            "requires_peft_translation_and_strict_parity": True,
            "persistent_vllm_server": True,
        },
    }
    assert module._validated_launch_runtime_policy(launch) == (raw, sampler)
    launch["runtime"]["model_args"] = {"provider": "hf"}
    with pytest.raises(module.EvaluationError, match="serving/parity policy"):
        module._validated_launch_runtime_policy(launch)


def test_launch_replay_rejects_coordinated_custody_deployment_and_runtime_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_module()
    target, paths, launch, _critical_state = _launch_replay_fixture(module, tmp_path, monkeypatch)
    loaded, digest = module._load_launch(paths, target=target)
    assert loaded == launch
    assert digest == module._sha256_file(paths.contract)

    coordinated = json.loads(json.dumps(launch))
    # Treat all three receipts as coordinated attacker inputs: launch replay
    # must reject the forged custody before runtime/evaluation can be trusted.
    coordinated["checkpoint_custody"]["checkpoint"]["path"] = "/tmp/not-sealed-checkpoint"
    coordinated["runtime"]["sampler"]["vllm_device_tokens"] = ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]
    _write_json(paths.contract, coordinated)
    _write_json(paths.runtime_receipt, {"coordinated": "forged-runtime"})
    _write_json(paths.evaluation_receipt, {"coordinated": "forged-evaluation"})
    with pytest.raises(module.EvaluationError, match="raw-checkpoint identity|exact replayed"):
        module._load_launch(paths, target=target)

    _write_json(paths.contract, launch)
    for _field, mutate in (
        ("source", lambda value: value["source_stage2_manifest"].__setitem__("sha256", "0" * 64)),
        ("parity", lambda value: value["parity_data"].__setitem__("path", "/tmp/forged-parity.jsonl")),
        ("deployment", lambda value: value["deployment_manifest"].__setitem__("substrate", {"forged": True})),
        ("deployment-provenance", lambda value: value["deployment_manifest"].__setitem__("provenance", {"forged": True})),
        ("custody", lambda value: value.__setitem__("checkpoint_custody", {"forged": True})),
        (
            "matrix",
            lambda value: value["matrix"]["sampling"]["task_sample_counts"][0].__setitem__("sample_count", 99),
        ),
        ("output", lambda value: value["outputs"].__setitem__("raw", "/tmp/forged-raw")),
        ("policy", lambda value: value["policy"].__setitem__("self_submits", True)),
    ):
        altered = json.loads(json.dumps(launch))
        mutate(altered)
        _write_json(paths.contract, altered)
        with pytest.raises(module.EvaluationError, match="exact replayed|strict checkpoint custody"):
            module._load_launch(paths, target=target)
    _write_json(paths.contract, launch)


def test_launch_replay_rejects_critical_source_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    target, paths, _launch, critical_state = _launch_replay_fixture(module, tmp_path, monkeypatch)
    module._load_launch(paths, target=target)
    critical_state["fixture.py"] = {"path": "/fixture.py", "sha256": "0" * 64, "size_bytes": 1}
    with pytest.raises(module.EvaluationError, match="exact replayed"):
        module._load_launch(paths, target=target)
