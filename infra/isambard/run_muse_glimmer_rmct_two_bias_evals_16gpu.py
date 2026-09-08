#!/usr/bin/env python3
"""Run one three-phase, two-condition Muse Glimmer RMCT evaluation campaign.

The ``early`` campaign evaluates the pinned base and global/data-step 16.  The
``late`` campaign evaluates global/data-step 64 and the trajectory's authored
final checkpoint (the first passing convergence window, or the hard-cap
checkpoint).  Each campaign contains 42 frozen Stage-2 cells and executes in
three sequential 14-worker phases on one 16-GPU allocation.

This module deliberately reuses the audited native-HF mixed-count custody
implementation.  Before delegating any operation, it replaces every
model/runtime/training-specific boundary with the frozen Muse contract.  A
generation command is admissible only through the generic native-HF EOS-only
sampler, with no output-token cap in its command, task, or effective config.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.muse_glimmer_rmct_replication import plan  # noqa: E402
from infra.isambard import muse_glimmer_rmct_segment_amendment as training_amendment  # noqa: E402
from infra.isambard import muse_glimmer_rmct_segment_contract as training_contract  # noqa: E402
from infra.isambard import run_qwen35_rmct_all_hf_peft_two_bias_16gpu as audited  # noqa: E402


GROUPS = ("early", "late")
GENERATION_CONFIG = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "max_connections": 8,
}
HF_MODEL_ARGS = {
    "device": "cuda:0",
    "dtype": "bfloat16",
    "do_sample": True,
    "hf_language_model_only": True,
}
HF_LOCAL_MODEL_ARGS = {"provider": "hf", **HF_MODEL_ARGS}
EVALUATOR_PACKAGE_VERSIONS = {
    "inspect-ai": "0.3.260",
    "torch": "2.13.0+cu129",
    "transformers": "5.15.1",
    "peft": "0.19.1",
    "safetensors": "0.8.0",
}
TOKEN_CAP_NAMES = frozenset(
    {
        "max_tokens",
        "max_new_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "generation_token_cap",
        "output_token_cap",
        "completion_token_cap",
    }
)
PHASE_ASSIGNMENTS = {
    1: tuple((0, task_index) for task_index in range(1, 15)),
    2: tuple((0, task_index) for task_index in range(15, 22))
    + tuple((1, task_index) for task_index in range(1, 8)),
    3: tuple((1, task_index) for task_index in range(8, 22)),
}


def _assert_no_token_cap(value: Any, *, label: str) -> None:
    """Reject every concrete token-cap spelling while permitting explicit null."""

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            spelling = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(raw_key))
            normalized = re.sub(r"[^a-zA-Z0-9]+", "_", spelling).strip("_").lower()
            if normalized in TOKEN_CAP_NAMES and nested is not None:
                raise audited.EvaluationError(f"{label} contains forbidden output-token cap {raw_key}={nested!r}")
            _assert_no_token_cap(nested, label=f"{label}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_no_token_cap(nested, label=f"{label}[{index}]")


def _snapshot_from_receipt() -> Path:
    receipt = PROJECT_ROOT / plan.RUNTIME_RECEIPT_DIR / "model-snapshot.json"
    document = audited._read_json(receipt, label="Muse model snapshot receipt")
    snapshot = Path(str(document.get("snapshot_path", ""))).resolve()
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.name != plan.MODEL_REVISION:
        raise audited.EvaluationError("Muse model snapshot receipt does not name the pinned revision directory")
    if not (snapshot / "config.json").is_file():
        raise audited.EvaluationError("pinned Muse model snapshot has no config.json")
    return snapshot


def _checkpoint_source(repository: Path, segment_index: int, *, label: str) -> dict[str, Any]:
    training_amendment.validate_attestation(repository)
    training_amendment.install()
    receipt = training_contract.validate_receipt(repository, segment_index)
    convergence = training_contract.validate_convergence(repository, segment_index)
    checkpoint = training_contract.checkpoint_path(repository, segment_index).resolve()
    expected_step = (segment_index + 1) * plan.UPDATES_PER_SEGMENT
    if receipt.get("checkpoint") != str(checkpoint):
        raise audited.EvaluationError(f"{label} segment receipt names a different checkpoint")
    artifacts = receipt.get("checkpoint_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise audited.EvaluationError(f"{label} checkpoint receipt has no artifact identities")
    files: dict[str, dict[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise audited.EvaluationError(f"{label} checkpoint receipt has an invalid artifact identity")
        path = Path(record["path"])
        identity = audited._identity(path, label=f"{label} checkpoint {path.name}")
        if identity != dict(record):
            raise audited.EvaluationError(f"{label} checkpoint artifact changed after training receipt")
        files[path.name] = identity
    required = {"adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "manifest.json"}
    if not required <= set(files):
        raise audited.EvaluationError(f"{label} checkpoint lacks the complete resumable file set")
    loop = receipt.get("loop_state")
    if (
        not isinstance(loop, Mapping)
        or loop.get("global_step") != expected_step
        or isinstance(loop.get("optimizer_step"), bool)
        or not isinstance(loop.get("optimizer_step"), int)
        or not 0 <= loop["optimizer_step"] <= expected_step
    ):
        raise audited.EvaluationError(f"{label} checkpoint has invalid global/optimizer step custody")
    return {
        "source": "raw-training-checkpoint",
        "step": expected_step,
        "checkpoint_axis": "global_training_batch_and_frozen_data_window",
        "global_step": expected_step,
        "optimizer_step": loop["optimizer_step"],
        "condition": plan.CONDITION_NAME,
        "segment_index": segment_index,
        "checkpoint": {
            "path": str(checkpoint),
            "global_step": expected_step,
            "optimizer_step": loop["optimizer_step"],
            "full_resumability_required": True,
            "files": files,
        },
        "boundary_amendment": audited._identity(
            training_amendment.amendment_path(repository),
            label="Muse boundary amendment attestation",
        ),
        "segment_receipt": audited._identity(
            training_contract.receipt_path(repository, segment_index),
            label=f"{label} segment receipt",
        ),
        "convergence_receipt": audited._identity(
            training_contract.convergence_path(repository, segment_index),
            label=f"{label} convergence receipt",
        ),
        "convergence": dict(convergence),
    }


def _final_segment(repository: Path) -> int:
    training_amendment.validate_attestation(repository)
    training_amendment.install()
    last_sealed = -1
    first_passing: int | None = None
    for segment_index in range(plan.TOTAL_SEGMENTS):
        receipt = training_contract.receipt_path(repository, segment_index)
        if not receipt.exists():
            break
        training_contract.validate_receipt(repository, segment_index)
        convergence = training_contract.validate_convergence(repository, segment_index)
        last_sealed = segment_index
        if first_passing is None and convergence.get("passed") is True:
            first_passing = segment_index
    completed_global_step = (last_sealed + 1) * plan.UPDATES_PER_SEGMENT
    if last_sealed < 3:
        raise audited.EvaluationError("Muse evaluation requires the sealed global-step 64 comparison checkpoint")
    if first_passing is not None and completed_global_step >= plan.MINIMUM_COMPARISON_OPTIMIZER_STEP:
        return first_passing
    if last_sealed == plan.TOTAL_SEGMENTS - 1:
        return last_sealed
    raise audited.EvaluationError("Muse trajectory is not yet converged and has not reached its hard cap")


def _condition_runtime_records(
    *,
    training_repository: str | Path,
    snapshot: Mapping[str, Any],
    evaluator: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    repository = Path(training_repository).resolve()
    records: dict[str, dict[str, Any]] = {}
    names = {condition.name for condition in audited.CONDITIONS}
    if "base" in names:
        records["base"] = audited._base_runtime(snapshot=snapshot, evaluator=evaluator)
    if "step016" in names:
        source = _checkpoint_source(repository, 0, label="step-16")
        records["step016"] = audited._trained_runtime(source, snapshot=snapshot, evaluator=evaluator)
    if "step064" in names:
        source = _checkpoint_source(repository, 3, label="step-64")
        records["step064"] = audited._trained_runtime(source, snapshot=snapshot, evaluator=evaluator)
    if "final" in names:
        segment_index = _final_segment(repository)
        source = _checkpoint_source(repository, segment_index, label="final")
        records["final"] = audited._trained_runtime(source, snapshot=snapshot, evaluator=evaluator)
    if set(records) != names:
        raise audited.EvaluationError("Muse runtime records differ from the selected two-condition campaign")
    return records


def _runtime_policy() -> dict[str, Any]:
    from ctm.evals.hf_eos_only import runtime_policy

    return runtime_policy()


def _validate_evaluator_environment(*, require_one_gpu: bool) -> dict[str, Any]:
    result = _ORIGINAL_VALIDATE_ENVIRONMENT(require_one_gpu=require_one_gpu)
    expected = {
        "CTM_HF_EOS_ONLY_NO_TOKEN_CAP": "1",
        "CTM_HF_EOS_ONLY_EXPECTED_INSPECT": EVALUATOR_PACKAGE_VERSIONS["inspect-ai"],
        "CTM_HF_EOS_ONLY_EXPECTED_TRANSFORMERS": EVALUATOR_PACKAGE_VERSIONS["transformers"],
    }
    if any(os.environ.get(name) != value for name, value in expected.items()):
        raise audited.EvaluationError("Muse evaluator lacks the exact generic EOS-only runtime environment")
    policy = _runtime_policy()
    if policy.get("output_token_cap") is not None or policy.get("termination") != "model_eos_only":
        raise audited.EvaluationError("Muse generic HF runtime does not attest uncapped EOS-only generation")
    result = dict(result)
    result["eos_only_no_token_cap"] = policy
    return result


def _configuration_mapping(value: Any) -> dict[str, Any]:
    mapped = audited._configuration_mapping(value)
    return dict(mapped) if isinstance(mapped, Mapping) else {}


def _assert_hf_runtime(header_log: Any, *, path: Path, runtime: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    from experiments.stage2_ood_hle import raw_preflight as mechanical

    evaluation = mechanical._attribute(header_log, "eval")
    metadata = _configuration_mapping(mechanical._attribute(evaluation, "metadata", {}))
    model = str(mechanical._attribute(evaluation, "model", "") or "")
    expected_model = f"hf/{audited.MODEL_SNAPSHOT}"
    if model != expected_model or runtime.get("model") != expected_model:
        raise audited.EvaluationError(f"Muse EvalLog has wrong pinned model identity: {path}")
    expected_args = HF_MODEL_ARGS if runtime.get("profile") == "hf-base" else HF_LOCAL_MODEL_ARGS
    if _configuration_mapping(metadata.get("model_args")) != expected_args:
        raise audited.EvaluationError(f"Muse EvalLog has wrong requested HF model arguments: {path}")
    requested_generation = _configuration_mapping(metadata.get("generation_config"))
    if requested_generation != GENERATION_CONFIG:
        raise audited.EvaluationError(f"Muse EvalLog has wrong requested generation configuration: {path}")
    effective_generation = _configuration_mapping(mechanical._attribute(evaluation, "model_generate_config", {}))
    _assert_no_token_cap(effective_generation, label=f"Muse effective generation config ({path})")
    if any(effective_generation.get(name) != value for name, value in GENERATION_CONFIG.items()):
        raise audited.EvaluationError(f"Muse EvalLog has wrong effective generation setting: {path}")
    if metadata.get("no_token_cap_runtime_policy") != _runtime_policy():
        raise audited.EvaluationError(f"Muse EvalLog lacks the generic EOS-only runtime attestation: {path}")
    native_args = _configuration_mapping(mechanical._attribute(evaluation, "model_args", {}))
    for name, value in {"device": "cuda:0", "dtype": "bfloat16", "do_sample": True}.items():
        if native_args.get(name) != value:
            raise audited.EvaluationError(f"Muse EvalLog has wrong native-HF {name}: {path}")
    observed: dict[str, Any] = {
        "profile": str(runtime.get("profile")),
        "provider": "hf",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "do_sample": True,
        "hf_language_model_only": True,
        "max_connections": GENERATION_CONFIG["max_connections"],
        "base_model": str(audited.MODEL_SNAPSHOT),
        "eos_only_no_token_cap": _runtime_policy(),
    }
    if runtime.get("profile") == "hf-peft":
        checkpoint = runtime.get("checkpoint")
        if (
            not isinstance(checkpoint, str)
            or metadata.get("checkpoint") != checkpoint
            or metadata.get("checkpoint_backend") != "local"
            or metadata.get("base_model") != str(audited.MODEL_SNAPSHOT)
        ):
            raise audited.EvaluationError(f"Muse EvalLog does not bind its raw PEFT checkpoint: {path}")
        observed["checkpoint"] = checkpoint
        observed["checkpoint_backend"] = "local"
    elif runtime.get("profile") == "hf-base":
        if metadata.get("model") != expected_model:
            raise audited.EvaluationError(f"Muse base EvalLog lacks its direct model identity: {path}")
    else:
        raise audited.EvaluationError("Muse EvalLog runtime is neither native HF base nor PEFT")
    return model, observed


def _assert_base_hf_runtime(header_log: Any, *, path: Path) -> tuple[str, dict[str, Any]]:
    runtime = {
        "profile": "hf-base",
        "model": f"hf/{audited.MODEL_SNAPSHOT}",
    }
    return _assert_hf_runtime(header_log, path=path, runtime=runtime)


def _task_command(
    *,
    python: str,
    contract: Mapping[str, Any],
    condition: Any,
    condition_paths: Any,
    task_index: int,
    attempt: Path,
) -> list[str]:
    row = audited._condition_row(contract, condition)
    runtime = row.get("runtime")
    deployment = contract.get("deployment_manifest")
    if not isinstance(runtime, Mapping) or not isinstance(deployment, Mapping):
        raise audited.EvaluationError("Muse campaign lacks runtime/deployment custody")
    task_args = {
        "manifest": str(deployment["path"]),
        "unbiased_log": str(condition_paths.raw),
        "prompt_style": "none",
        "include_bias_acknowledged": False,
        "hf_eos_only_no_token_cap": True,
    }
    command = [python, str(PROJECT_ROOT / "scripts" / "run_evals.py"), "--task-factory", audited.TASK_FACTORY]
    if runtime.get("profile") == "hf-base":
        command.extend(["--model", f"hf/{audited.MODEL_SNAPSHOT}"])
        model_args = HF_MODEL_ARGS
    elif runtime.get("profile") == "hf-peft":
        checkpoint = runtime.get("checkpoint")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
            raise audited.EvaluationError("Muse trained condition has no absolute PEFT checkpoint")
        command.extend(["--local-checkpoint", checkpoint, "--base-model", str(audited.MODEL_SNAPSHOT)])
        model_args = HF_LOCAL_MODEL_ARGS
    else:
        raise audited.EvaluationError("Muse condition has an unsupported HF runtime profile")
    if dict(runtime.get("model_args", {})) != model_args or dict(runtime.get("generation_config", {})) != GENERATION_CONFIG:
        raise audited.EvaluationError("Muse runtime changed its frozen HF/decode controls")
    command.extend(
        [
            "--task-args",
            json.dumps(task_args, sort_keys=True, separators=(",", ":")),
            "--model-args",
            json.dumps(model_args, sort_keys=True, separators=(",", ":")),
            "--generation-config",
            json.dumps(GENERATION_CONFIG, sort_keys=True, separators=(",", ":")),
            "--log-dir",
            str(attempt),
            "--limit",
            str(audited._sample_count_for_task(task_index)),
            "--max-tasks",
            "1",
            "--isolate-tasks",
            "--task-index",
            str(task_index),
            "--yes",
        ]
    )
    _assert_no_token_cap(task_args, label="Muse task args")
    _assert_no_token_cap(model_args, label="Muse model args")
    _assert_no_token_cap(GENERATION_CONFIG, label="Muse generation config")
    _assert_no_token_cap(command, label="Muse command")
    return command


def _topology_contract() -> dict[str, Any]:
    conditions = list(audited.CONDITIONS)
    if len(conditions) != 2:
        raise audited.EvaluationError("Muse three-phase campaign requires exactly two conditions")
    phases: list[dict[str, Any]] = []
    seen: list[tuple[str, int]] = []
    for phase, assignments in PHASE_ASSIGNMENTS.items():
        rows = []
        for rank, (condition_offset, task_index) in enumerate(assignments):
            condition = conditions[condition_offset]
            rows.append(
                {
                    "rank": rank,
                    "condition": condition.name,
                    "task_index": task_index,
                    "sample_count": audited._sample_count_for_task(task_index),
                }
            )
            seen.append((condition.name, task_index))
        phases.append({"phase": phase, "assignments": rows, "idle_ranks": [14, 15]})
    expected = {(condition.name, task_index) for condition in conditions for task_index in range(1, 22)}
    if set(seen) != expected or len(seen) != len(expected):
        raise audited.EvaluationError("Muse three-phase topology does not cover exactly 42 cells")
    return {
        "nodes": 4,
        "gpus_per_node": 4,
        "workers": 16,
        "one_gpu_per_worker": True,
        "phases": phases,
        "phase_count": 3,
        "workers_per_phase": 14,
        "clean_and_biased_launch_together": True,
        "switch_scorer_clean_wait": True,
    }


def _policy_contract() -> dict[str, bool]:
    return {
        "self_submits": False,
        "scheduler_clients": False,
        "chains_successors": False,
        "cancels_jobs": False,
        "automatic_retries": False,
        "incomplete_attempts_require_operator": True,
        "overwrite_differing_logs": False,
        "partial_attempts_preserved": True,
        "clean_receipt_gate_required": False,
        "switch_scorer_waits_for_clean": True,
        "strict_checkpoint_and_base_custody": True,
        "output_token_cap_forbidden": True,
        "eos_only_termination_required": True,
    }


def _validate_all_sample_termination(*, condition_paths: Any) -> None:
    from inspect_ai.log import read_eval_log

    for path in sorted(condition_paths.raw.rglob("*.eval")):
        log = read_eval_log(str(path), header_only=False)
        samples = getattr(log, "samples", None)
        if not isinstance(samples, list) or not samples:
            raise audited.EvaluationError(f"Muse canonical EvalLog has no samples: {path}")
        for sample in samples:
            output = getattr(sample, "output", None)
            metadata = getattr(output, "metadata", {})
            if not isinstance(metadata, Mapping) or (
                metadata.get("ctm_no_output_token_cap") is not True
                or metadata.get("ctm_termination") != "model_eos_only"
            ):
                raise audited.EvaluationError(f"Muse sample lacks EOS-only/no-cap output attestation: {path}")


def _mixed_preflight(**kwargs: Any) -> dict[str, Any]:
    _validate_all_sample_termination(condition_paths=kwargs["condition_paths"])
    return _ORIGINAL_MIXED_PREFLIGHT(**kwargs)


def _worker_without_clean_barrier(**kwargs: Any) -> dict[str, Any]:
    biased = audited.BIASED_TASK_INDICES
    try:
        # The installed switch scorer owns the clean-log wait.  Temporarily
        # remove only the audited launcher's earlier gate admission check;
        # task identity, switch pairing, receipts, and final clean gate remain.
        audited.BIASED_TASK_INDICES = ()
        return audited.worker(**kwargs)
    finally:
        audited.BIASED_TASK_INDICES = biased


def _phase_rows(phase: int) -> list[dict[str, Any]]:
    topology = _topology_contract()
    rows = topology["phases"][phase - 1]["assignments"]
    return [dict(row) for row in rows]


def _seal_phase(args: argparse.Namespace) -> dict[str, Any]:
    contract, paths = audited._load_campaign(
        campaign_root=args.campaign_root,
        training_repository=args.training_repository,
        source_stage2_manifest=args.source_stage2_manifest,
        stage2_artifact_root=args.stage2_artifact_root,
    )
    launch_sha = audited._sha256_file(paths.contract)
    selected: list[dict[str, Any]] = []
    for row in _phase_rows(args.phase):
        condition = audited._condition(row["condition"])
        condition_paths = audited._condition_paths(paths, condition)
        _evaluation, evaluation_sha = audited._load_evaluation_receipt(contract, paths, condition)
        _runtime, runtime_sha = audited._load_runtime_receipt(contract, paths, condition)
        receipt = audited._load_task_receipt(
            contract=contract,
            paths=paths,
            condition=condition,
            condition_paths=condition_paths,
            task_index=row["task_index"],
            launch_sha256=launch_sha,
            runtime_sha256=runtime_sha,
            evaluation_sha256=evaluation_sha,
        )
        if receipt is None:
            raise audited.EvaluationError(
                f"phase {args.phase} is incomplete at {condition.name}/task-{row['task_index']}"
            )
        selected.append(
            {
                **row,
                "task_receipt": audited._identity(
                    audited._task_receipt_path(condition_paths, row["task_index"]),
                    label=f"phase {args.phase} task receipt",
                ),
            }
        )
    document = {
        "schema": "muse-glimmer-rmct-two-bias-phase-v1",
        "campaign": audited.CAMPAIGN_NAME,
        "group": args.group,
        "phase": args.phase,
        "launch_contract_sha256": launch_sha,
        "assignments": selected,
    }
    destination = paths.root / "phases" / f"phase-{args.phase:03d}.json"
    status = audited._write_immutable_json(destination, document, label=f"Muse phase {args.phase} receipt")
    return {"status": status, "phase_receipt": audited._identity(destination, label="Muse phase receipt")}


def _configure(group: str) -> None:
    if group not in GROUPS:
        raise audited.EvaluationError(f"unknown Muse evaluation group {group!r}")
    snapshot = _snapshot_from_receipt()
    audited.MODEL_SNAPSHOT = snapshot
    audited.INSPECT_VERSION = EVALUATOR_PACKAGE_VERSIONS["inspect-ai"]
    audited.EVALUATOR_PACKAGE_VERSIONS = dict(EVALUATOR_PACKAGE_VERSIONS)
    audited.GENERATION_CONFIG = dict(GENERATION_CONFIG)
    audited.HF_MODEL_ARGS = dict(HF_MODEL_ARGS)
    audited.HF_LOCAL_MODEL_ARGS = dict(HF_LOCAL_MODEL_ARGS)
    audited.LAUNCH_SCHEMA = "muse-glimmer-rmct-two-bias-16gpu-launch-v1"
    audited.RUNTIME_RECEIPT_SCHEMA = "muse-glimmer-rmct-two-bias-16gpu-runtime-v1"
    audited.EVALUATION_RECEIPT_SCHEMA = "muse-glimmer-rmct-two-bias-16gpu-evaluation-v1"
    audited.TASK_RECEIPT_SCHEMA = "muse-glimmer-rmct-two-bias-16gpu-task-receipt-v1"
    audited.CLEAN_GATE_RECEIPT_SCHEMA = "muse-glimmer-rmct-two-bias-16gpu-clean-gate-v1"
    audited.PREFLIGHT_SCHEMA = "muse-glimmer-rmct-two-bias-16gpu-preflight-v1"
    audited.COMPLETION_SCHEMA = "muse-glimmer-rmct-two-bias-16gpu-completion-v1"
    audited.CAMPAIGN_NAME = f"muse-glimmer-rmct-two-bias-{group}-16gpu-v1"
    if group == "early":
        conditions = (
            audited.Condition("base", "base", None, "pinned-base-snapshot", "muse-glimmer-base-two-bias-v1"),
            # The inherited Condition field is named ``optimizer_step`` for
            # the Qwen campaign.  Under the attested Muse boundary amendment,
            # 16 identifies a global/data-batch boundary while the realized
            # optimizer count is read from raw checkpoint custody (13 at this
            # boundary).  Keep the inherited field null rather than publishing
            # a false optimizer-step claim.
            audited.Condition(
                "step016",
                "global/data-step-016",
                None,
                "raw-training-checkpoint",
                "muse-glimmer-step016-two-bias-v1",
            ),
        )
    else:
        conditions = (
            audited.Condition(
                "step064",
                "global/data-step-064",
                None,
                "raw-training-checkpoint",
                "muse-glimmer-step064-two-bias-v1",
            ),
            audited.Condition("final", "final", None, "raw-training-checkpoint", "muse-glimmer-final-two-bias-v1"),
        )
    audited.CONDITIONS = conditions
    audited._CONDITION_BY_NAME = {condition.name: condition for condition in conditions}
    audited.CRITICAL_SOURCES = (
        "infra/isambard/run_muse_glimmer_rmct_two_bias_evals_16gpu.py",
        "infra/isambard/run_muse_glimmer_rmct_two_bias_evals_16gpu.sbatch",
        "infra/isambard/run_muse_glimmer_rmct_two_bias_evals_16gpu_worker.sh",
        "infra/isambard/run_qwen35_rmct_all_hf_peft_two_bias_16gpu.py",
        "scripts/run_evals.py",
        "ctm/evals/runner.py",
        "ctm/evals/local_model.py",
        "ctm/evals/hf_eos_only.py",
        "experiments/elephant_aita_ntaflip/no_cap_hf.py",
        "experiments/elephant_aita_ntaflip/prepare.py",
        "experiments/muse_glimmer_rmct_replication/plan.py",
        "infra/isambard/muse_glimmer_rmct_segment_contract.py",
        "infra/isambard/muse_glimmer_rmct_segment_amendment.py",
        "experiments/stage2_ood_hle/tasks.py",
        "experiments/stage2_ood_hle/materialize.py",
        "experiments/stage2_ood_hle/prepare.py",
        "experiments/stage2_ood_hle/raw_preflight.py",
        "experiments/stage1_iid_diagnostic/raw_preflight.py",
        "experiments/stage1_iid_diagnostic_none/prepare.py",
        "experiments/rmct_paper_vast_dense_models/stage1/supervised_recovery_none_prepare.py",
        "experiments/rmct_tbsr/constants.py",
        "experiments/rmct_two_bias_eval/contract.py",
        "experiments/rmct_two_bias_eval/deployment.py",
    )
    audited._condition_runtime_records = _condition_runtime_records
    audited._validate_evaluator_environment = _validate_evaluator_environment
    audited._assert_base_hf_runtime = _assert_base_hf_runtime
    audited._assert_hf_runtime = _assert_hf_runtime
    audited._task_command = _task_command
    audited._topology_contract = _topology_contract
    audited._policy_contract = _policy_contract
    audited._mixed_preflight = _mixed_preflight
    _assert_no_token_cap(GENERATION_CONFIG, label="Muse frozen generation config")
    plan._validate_spec(plan.FROZEN_SPEC)


_ORIGINAL_VALIDATE_ENVIRONMENT = audited._validate_evaluator_environment
_ORIGINAL_MIXED_PREFLIGHT = audited._mixed_preflight


def _extract_group(argv: Sequence[str]) -> tuple[str, list[str]]:
    values = list(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--group", choices=GROUPS, required=True)
    namespace, remaining = parser.parse_known_args(values)
    return namespace.group, remaining


def _phase_parser(group: str, argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal one completed Muse evaluation phase")
    parser.add_argument("command", choices=["seal-phase"])
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--training-repository", required=True, type=Path)
    parser.add_argument("--source-stage2-manifest", required=True, type=Path)
    parser.add_argument("--stage2-artifact-root", required=True, type=Path)
    parser.add_argument("--phase", required=True, type=int, choices=(1, 2, 3))
    result = parser.parse_args(list(argv))
    result.group = group
    return result


def main(argv: Sequence[str] | None = None) -> int:
    group, remaining = _extract_group(list(argv) if argv is not None else sys.argv[1:])
    try:
        _configure(group)
        if remaining and remaining[0] == "seal-phase":
            result = _seal_phase(_phase_parser(group, remaining))
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if remaining and remaining[0] == "worker":
            original = audited.worker
            audited.worker = _worker_without_clean_barrier
            try:
                return audited.main(remaining)
            finally:
                audited.worker = original
        return audited.main(remaining)
    except (audited.EvaluationError, FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVALUATOR_PACKAGE_VERSIONS",
    "GENERATION_CONFIG",
    "GROUPS",
    "HF_LOCAL_MODEL_ARGS",
    "HF_MODEL_ARGS",
    "PHASE_ASSIGNMENTS",
    "main",
]
