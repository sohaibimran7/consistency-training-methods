"""Frozen compiler for the deadline-driven RMCT convergence trajectory.

The experiment deliberately has one scientific configuration and one execution
topology.  It is *not* a topology comparison: the four-GH200, fully
phase-shared placement was selected to meet the deadline and has not been
comparatively optimized.  Each emitted target is one sealed 16-optimizer-step
window over a deterministic, shared-QID 32-question slice.

The small command-line interface is consumed only by the protected Isambard
launcher.  It resolves the exact local Hugging Face snapshot into the otherwise
portable compiled command, writes an immutable command attestation, and then
can execute that command without changing CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MODEL = "Qwen/Qwen3.5-9B"
BASE_SNAPSHOT = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CONDITION_NAME = "rmct-convergence"
# The original 16-layer activation-checkpointing attempt is retained as an
# immutable, separately authored plan.  This recovery is deliberately in a
# new run namespace because job 6011179 reached no optimizer step and wrote no
# checkpoint; it must restart from the pinned base rather than look resumable.
RUN_PREFIX = "rmct-convergence-gcall-r2"
ORIGINAL_RUN_PREFIX = CONDITION_NAME
TOPOLOGY_PROFILE = "phase-shared-four-lane"
GPU_COUNT = 4
UPDATES_PER_SEGMENT = 16
TOTAL_SEGMENTS = 32
HARD_CAP_OPTIMIZER_STEPS = TOTAL_SEGMENTS * UPDATES_PER_SEGMENT
DATAPOINTS_PER_SEGMENT = 32
BATCH_SIZE = 2
BASE_ROLLOUT_SEED = 42

DATA_PATH = (
    "artifacts/rmct-shared-qid-two-bias-20260813/"
    "shared-qid-two-bias-n1000-cc7842566093e10d3867514e40b55d62ba56521fee24e2ba3695036295199075.jsonl"
)
DATA_SHA256 = "cc7842566093e10d3867514e40b55d62ba56521fee24e2ba3695036295199075"
MANIFEST_PATH = (
    "artifacts/rmct-shared-qid-two-bias-20260813/"
    "shared-qid-two-bias-n1000-cc7842566093e10d3867514e40b55d62ba56521fee24e2ba3695036295199075"
    ".manifest-eac0682fe0286126cc5928253e2e2ef4968eee50775a65feb21d061eec6853cc.json"
)
MANIFEST_SHA256 = "eac0682fe0286126cc5928253e2e2ef4968eee50775a65feb21d061eec6853cc"
SETTING_FACTORY = "ctm_data.adapters.mcq_bias.shared_qid_two_bias:create_shared_qid_two_bias_setting"
WORKER_PARITY_ATTESTATION = (
    "artifacts/rmct-convergence-worker-parity-20260813/"
    "qwen35-rollout-worker-parity-attestation.json"
)
CONVERGENCE_SCHEMA = "rmct_convergence_plan_v1"
SEGMENT_SCHEMA = "rmct_convergence_segment_v1"
COMMAND_SCHEMA = "rmct-convergence-segment-command-v1"

RECOVERY_SCHEMA = "rmct-convergence-clean-recovery-v1"
RECOVERY_PARENT_ARTIFACT = "artifacts/rmct-convergence-gcall-r2-20260814/recovery-parent.json"
RECOVERY_PARENT_ARTIFACT_SHA256 = "b2a5c3d63323f43563a0281a4efb402a7778cc1e9a8d4a363ada90a9eee89e15"
RECOVERY_METADATA: dict[str, Any] = {
    "schema": RECOVERY_SCHEMA,
    "recovery_of": {
        "prior_job_id": "6011179",
        "prior_attempt_ended": "pre_optimizer",
        "successful_optimizer_steps": 0,
        "checkpoint_written": False,
        "restart_parent": "pinned_base_snapshot",
        "artifacts_sha256": {
            "training_started_marker": "33a3d176cc3df0a25e9e6629dad2e96a65e62e58fb19256eca7933f94b9d6a6c",
            "training_command": "affed38d5ccd1645d3af926ac528ee9869e5547e3fae197293e5fd40dc4191fc",
            "metrics": "63bad54abaccfdcbad5b60d7c0ab2a1401554babd57c59b70840a3f499e06e92",
            "slurm_log": "a485cdadf67cef9cc8b96454f1507e09472d1ffc82f7e0a55293b2a060324df2",
            "production_ready": "5741a59dcaeb9a2136bd00b8569d01f88ea6751d70b084ba98fc28eadfa96509",
            "source_ready": "67fc52b0ed64937f352b0fa64607a0aa84bfdfacab37e6c32f8242f873c09e7d",
            "rollout_worker_parity_attestation": "38765b2f61925880ef6cbc50af7fb2868725337b3ad26c9ac7243c26304b6f37",
        },
    },
    "execution_delta": {
        "activation_checkpointing_layers": {
            "from": "first16",
            "to": "all32",
            "local_gradient_checkpointing_layers_cli": "omitted",
        },
        "reason": "30,720-token singleton proof established that the physical caps cannot address the pre-optimizer failure",
    },
    "scientific_contract_unchanged": True,
    "recovery_parent_artifact": {
        "path": RECOVERY_PARENT_ARTIFACT,
        "sha256": RECOVERY_PARENT_ARTIFACT_SHA256,
    },
    "same_hardware_all_gradient_checkpointing_evidence": {
        "result_sha256": "da5783f1aca63cd7f7df500cd85e3460f204ddf2398417c8f44a5c9fadcdd0e9",
        "contract_sha256": "a71c0d74cb65b3b3aba56eb61f03e5ab25fe5cf28e909448b60fe62e051c6394",
    },
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PlanError(ValueError):
    """The authored plan or requested segment cannot be executed safely."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < TOTAL_SEGMENTS:
        raise PlanError(f"segment_index must be an integer in [0, {TOTAL_SEGMENTS - 1}]")
    return value


def _safe_run_prefix(value: str) -> str:
    if value not in {ORIGINAL_RUN_PREFIX, RUN_PREFIX}:
        raise PlanError(f"unsupported RMCT convergence run prefix: {value!r}")
    return value


def run_name(segment_index: int, *, run_prefix: str = ORIGINAL_RUN_PREFIX) -> str:
    return f"{_safe_run_prefix(run_prefix)}-s{_safe_index(segment_index) + 1:03d}"


def target_name(segment_index: int, *, run_prefix: str = ORIGINAL_RUN_PREFIX) -> str:
    return f"{_safe_run_prefix(run_prefix)}-s{_safe_index(segment_index) + 1:03d}"


def final_checkpoint_path(
    repository: str | Path, segment_index: int, *, run_prefix: str = ORIGINAL_RUN_PREFIX
) -> Path:
    root = Path(repository).resolve()
    name = run_name(segment_index, run_prefix=run_prefix)
    return root / "logs" / CONDITION_NAME / name / "checkpoints" / f"{CONDITION_NAME}_{name}"


def _parent(repository: str | Path, segment_index: int, *, run_prefix: str = ORIGINAL_RUN_PREFIX) -> dict[str, Any]:
    index = _safe_index(segment_index)
    if index == 0:
        return {
            "kind": "pinned_base_snapshot",
            "resume": False,
            "base_snapshot": {"repo_id": MODEL, "revision": BASE_SNAPSHOT, "offline_only": True},
        }
    previous = index - 1
    checkpoint = final_checkpoint_path(repository, previous, run_prefix=run_prefix)
    return {
        "kind": "sealed_strict_final_checkpoint",
        "resume": True,
        "segment_index": previous,
        "target": target_name(previous, run_prefix=run_prefix),
        "run_name": run_name(previous, run_prefix=run_prefix),
        "uri": f"file://{checkpoint}",
        "optimizer_step": (previous + 1) * UPDATES_PER_SEGMENT,
        "expected_kind": "both",
        "expected_final": True,
        "resume_with_optimizer": True,
        "resume_state_required": True,
    }


def _frozen_spec(*, recovery: bool) -> dict[str, Any]:
    """Return one exact authored contract, never a mutable profile merge."""

    gradient_checkpointing_layers: int | str = "all" if recovery else 16
    spec: dict[str, Any] = {
        "topology_profiles": {
            TOPOLOGY_PROFILE: {
                "gpu_count": GPU_COUNT,
                "training_gpus": "all",
                "rollout_gpus": "all",
                "phase_shared": True,
                "deadline_execution_choice": True,
                "comparative_optimality_validated": False,
                # This is execution metadata.  In the recovery profile,
                # ``all`` means all 32 Qwen backbone layers and the compiler
                # must omit a limiting CLI flag.
                "gradient_checkpointing_layers": gradient_checkpointing_layers,
            }
        },
        "model": MODEL,
        "base_snapshot": {"repo_id": MODEL, "revision": BASE_SNAPSHOT, "offline_only": True},
        "setting": {
            "factory": SETTING_FACTORY,
            "setting_config": {
                "data_path": DATA_PATH,
                "manifest_path": MANIFEST_PATH,
                "expected_manifest_sha256": MANIFEST_SHA256,
            },
            "data_content_sha256": DATA_SHA256,
            "prompt_style": "none",
            "perturbations": ["clean", "wrong_argument", "suggested_answer"],
            "training_indices": [1, 2],
            "segment_datapoints": DATAPOINTS_PER_SEGMENT,
            "segment_count": TOTAL_SEGMENTS,
            "batch_composition": "one_logiqa_plus_one_hellaswag_per_B2_optimizer_update",
            "ordering": "setting_interleaved_logiqa_hellaswag_and_trainer_no_shuffle",
        },
        "local": {
            "backend": "local",
            "dtype": "bfloat16",
            "sampler": "vllm",
            "gpu_memory_utilization": 0.34,
            "rollout_gpu_memory_utilization": 0.34,
            "vllm_language_model_only": True,
            "vllm_max_num_seqs": 256,
            "vllm_max_num_batched_tokens": 8192,
            "vllm_max_model_len": 32768,
            "vllm_gdn_prefill_backend": "triton",
            "forward_microbatch_max_datums": 8,
            "forward_microbatch_max_tokens": 20480,
            "target_logprob_chunk_size": 2048,
            "gradient_checkpointing": True,
            "gradient_checkpointing_layers": gradient_checkpointing_layers,
            "phase_shared": True,
            "worker_parity_attestation": WORKER_PARITY_ATTESTATION,
        },
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "train_mlp": True,
            "train_attn": True,
            "train_unembed": False,
        },
        "optimizer": {
            "learning_rate": 0.0001,
            "learning_rate_schedule": "constant",
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
        },
        "method": {
            "loss": "ppo",
            "advantage_estimator": "grpo_normalized",
            "normalization": "pooled",
            "kl_coefficient": 0.05,
            "kl_discount_factor": 0.0,
            "ppo_clip_epsilon": 0.2,
            "anchor_weight": 0.0,
            "anchor_model": "base",
        },
        "sampling": {
            "reference_rollouts": 96,
            "training_rollouts": 96,
            "consistency_rollouts": 96,
            "anchor_rollouts": 96,
            "temperature": 1.0,
            "max_new_tokens": 20480,
            "unparsed_handling": "discard",
            "max_resample_attempts": 4,
            "snr_mode": "soft",
            "snr_z": 2.0,
            "snr_normalizer": "trait_std",
        },
        "loop": {
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": 1,
            "epochs": 1,
            "refresh_every": 1,
            "shuffle_datapoints": False,
            "checkpoint_every_optimizer_steps": UPDATES_PER_SEGMENT,
            "save_state": True,
        },
        "convergence": {
            "controller_schema": "rmct-convergence-checkpoint-window-decision-v1",
            "source_metrics_schema": "rmct-convergence-source-metrics-v1",
            "updates_per_segment": UPDATES_PER_SEGMENT,
            "hard_cap_segments": TOTAL_SEGMENTS,
            "hard_cap_optimizer_steps": HARD_CAP_OPTIMIZER_STEPS,
            "cap_label": "capped_not_converged",
            "minimum_coverage": 0.85,
            "maximum_weighted_abs_gap": 0.10,
            "maximum_abs_change": 0.01,
            "first_eligible_segment_index": 1,
            "first_eligible_optimizer_step": 32,
        },
    }
    if recovery:
        spec["recovery"] = RECOVERY_METADATA
    return spec


def _validate_spec(spec: Mapping[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    if not isinstance(spec, Mapping):
        raise PlanError("RMCT convergence spec must be an object")
    actual = dict(spec)
    original = _frozen_spec(recovery=False)
    recovery = _frozen_spec(recovery=True)
    if actual == original:
        return original, ORIGINAL_RUN_PREFIX, None
    if actual == recovery:
        return recovery, RUN_PREFIX, dict(RECOVERY_METADATA)

    # Give a useful structural error before the final immutable-contract
    # rejection, while keeping the two authored profiles strictly separate.
    expected = recovery if "recovery" in actual else original
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unknown = sorted(set(actual) - set(expected))
        if missing or unknown:
            raise PlanError(f"RMCT convergence spec keys differ: missing={missing}, unknown={unknown}")
        raise PlanError("RMCT convergence spec differs from the frozen production contract")
    raise AssertionError("unreachable RMCT convergence contract selection")


def _validate_profile(spec: Mapping[str, Any], requested: str | None) -> dict[str, Any]:
    if requested != TOPOLOGY_PROFILE:
        raise PlanError(
            f"RMCT convergence requires explicit topology profile {TOPOLOGY_PROFILE!r}; "
            "it is a deadline-driven execution choice, not a benchmark optimum"
        )
    profile = spec["topology_profiles"][TOPOLOGY_PROFILE]
    if not isinstance(profile, Mapping):  # guarded by exact spec, kept for type checkers
        raise PlanError("phase-shared topology profile must be an object")
    return dict(profile)


def segment_record(
    repository: str | Path, segment_index: int, *, run_prefix: str = ORIGINAL_RUN_PREFIX
) -> dict[str, Any]:
    """Return the fully attested immutable identity for one segment."""

    index = _safe_index(segment_index)
    prefix = _safe_run_prefix(run_prefix)
    return {
        "schema": SEGMENT_SCHEMA,
        "condition": CONDITION_NAME,
        "run_prefix": prefix,
        "segment_index": index,
        "target": target_name(index, run_prefix=prefix),
        "run_name": run_name(index, run_prefix=prefix),
        "optimizer_step_start": index * UPDATES_PER_SEGMENT + 1,
        "optimizer_step_end": (index + 1) * UPDATES_PER_SEGMENT,
        "optimizer_steps": UPDATES_PER_SEGMENT,
        "base_qids": DATAPOINTS_PER_SEGMENT,
        "batch_size": BATCH_SIZE,
        "expected_batches": UPDATES_PER_SEGMENT,
        "dataset_balance": "one_logiqa_plus_one_hellaswag_per_optimizer_update",
        "parent": _parent(repository, index, run_prefix=prefix),
        # One immutable four-lane parity receipt binds the vLLM worker seed
        # contract.  Keep it constant across sealed data windows so every
        # segment validates against exactly the same preflight sidecar.
        "rollout_seed_base": BASE_ROLLOUT_SEED,
    }


def segment_args(
    repository: str | Path,
    segment_index: int,
    *,
    model_path: str | Path | None = None,
    run_prefix: str = ORIGINAL_RUN_PREFIX,
) -> dict[str, Any]:
    """Emit the exact ``train_rlct.py`` argument map for one sealed window."""

    index = _safe_index(segment_index)
    root = Path(repository).resolve()
    prefix = _safe_run_prefix(run_prefix)
    record = segment_record(root, index, run_prefix=prefix)
    parent = record["parent"]
    model = str(Path(model_path).resolve()) if model_path is not None else MODEL
    if model_path is not None:
        snapshot = Path(model)
        if snapshot.name != BASE_SNAPSHOT or snapshot.is_symlink() or not (snapshot / "config.json").is_file():
            raise PlanError("model_path must be the regular pinned Qwen snapshot directory containing config.json")
    setting_config = {
        "data_path": str(root / DATA_PATH),
        "manifest_path": str(root / MANIFEST_PATH),
        "expected_manifest_sha256": MANIFEST_SHA256,
    }
    args: dict[str, Any] = {
        "backend": "local",
        "local_dtype": "bfloat16",
        "local_sampler": "vllm",
        "local_gpu_mem_util": 0.34,
        "local_rollout_gpu_mem_util": 0.34,
        "local_vllm_language_model_only": True,
        "local_vllm_max_num_seqs": 256,
        "local_vllm_max_num_batched_tokens": 8192,
        "local_vllm_max_model_len": 32768,
        "local_vllm_gdn_prefill_backend": "triton",
        "local_forward_microbatch_max_datums": 8,
        "local_forward_microbatch_max_tokens": 20480,
        "local_target_logprob_chunk_size": 2048,
        "local_gradient_checkpointing": True,
        "local_phase_shared": True,
        "local_training_gpus": "all",
        "local_rollout_gpus": "all",
        "local_rollout_seed_base": record["rollout_seed_base"],
        "local_qwen35_rollout_parity_attestation": str(root / WORKER_PARITY_ATTESTATION),
        "model": model,
        "setting_factory": SETTING_FACTORY,
        "setting_config": setting_config,
        "load_config": {"n_datapoints": DATAPOINTS_PER_SEGMENT, "segment_index": index},
        "n_datapoints": DATAPOINTS_PER_SEGMENT,
        "experiment_name": CONDITION_NAME,
        "run_name": record["run_name"],
        "seed": BASE_ROLLOUT_SEED,
        "lora_config": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "train_mlp": True,
            "train_attn": True,
            "train_unembed": False,
            "seed": BASE_ROLLOUT_SEED,
        },
        "lr": 0.0001,
        "lr_schedule": "constant",
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "kl_coef": 0.05,
        "kl_discount_factor": 0.0,
        "local_ppo_clip_epsilon": 0.2,
        "anchor_weight": 0.0,
        "anchor_model": "base",
        "loss_fn": "ppo",
        "advantage_estimator": "grpo_normalized",
        "normalization": "pooled",
        "n_ref_rollouts": 96,
        "n_train_rollouts": 96,
        "n_consistency_rollouts": 96,
        "n_anchor_rollouts": 96,
        "temperature": 1.0,
        "max_new_tokens": 20480,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "n_epochs": 1,
        "refresh_every": 1,
        "no_shuffle_datapoints": True,
        "checkpoint_every": UPDATES_PER_SEGMENT,
        "save_state": True,
        "unparsed_handling": "discard",
        "max_resample_attempts": 4,
        "snr_mode": "soft",
        "snr_z": 2.0,
        "snr_normalizer": "trait_std",
        "yes": True,
    }
    # The recovery is memory-only: no value is passed because the local CLI's
    # unset checkpoint-layer limit means checkpoint every backbone layer.
    if prefix == ORIGINAL_RUN_PREFIX:
        args["local_gradient_checkpointing_layers"] = 16
    if parent["resume"]:
        args.update(
            {
                "resume_from": parent["uri"],
                "resume_with_optimizer": True,
                "resume_state_required": True,
            }
        )
    return args


def _training_entry(
    repository: str | Path, segment_index: int, *, run_prefix: str = ORIGINAL_RUN_PREFIX
) -> dict[str, Any]:
    record = segment_record(repository, segment_index, run_prefix=run_prefix)
    return {
        "name": f"{run_prefix.replace('-', '_')}_s{segment_index + 1:03d}",
        "target": record["target"],
        "gpu_count": GPU_COUNT,
        "command": ["${python}", "scripts/train_rlct.py"],
        "args": segment_args(repository, segment_index, run_prefix=run_prefix),
    }


def compile_experiment(*, name: str, spec: Mapping[str, Any], topology_profile: str | None = None) -> dict[str, Any]:
    """Expand the canonical YAML into its 32 explicit sealed boundaries."""

    if name != CONDITION_NAME:
        raise PlanError(f"condition name must be {CONDITION_NAME!r}")
    frozen, run_prefix, recovery = _validate_spec(spec)
    profile = _validate_profile(frozen, topology_profile)
    segments = [segment_record(_PROJECT_ROOT, index, run_prefix=run_prefix) for index in range(TOTAL_SEGMENTS)]
    compiled: dict[str, Any] = {
        "name": name,
        "training_only": True,
        "onpolicy_topology_profile": TOPOLOGY_PROFILE,
        "onpolicy_topology": {
            "gpu_count": GPU_COUNT,
            "training_gpus": "all",
            "rollout_gpus": "all",
            "phase_shared": True,
            "execution_choice": "deadline_driven_not_comparative_optimality_validated",
        },
        "rmct_convergence": {
            "schema": CONVERGENCE_SCHEMA,
            "condition": CONDITION_NAME,
            "run_prefix": run_prefix,
            "base_snapshot": frozen["base_snapshot"],
            "data": {
                "path": DATA_PATH,
                "content_sha256": DATA_SHA256,
                "manifest_path": MANIFEST_PATH,
                "manifest_sha256": MANIFEST_SHA256,
            },
            "topology": profile,
            "optimizer": frozen["optimizer"],
            "method": frozen["method"],
            "sampling": frozen["sampling"],
            "loop": frozen["loop"],
            "convergence": frozen["convergence"],
            "segments": segments,
        },
        "training": [_training_entry(_PROJECT_ROOT, index, run_prefix=run_prefix) for index in range(TOTAL_SEGMENTS)],
    }
    if recovery is not None:
        compiled["rmct_convergence"]["recovery"] = recovery
    return compiled


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise PlanError(f"{label} must be a regular file: {path}")


def _load_compiled_plan(plan: Path) -> dict[str, Any]:
    from scripts.run_experiment import load_experiment

    _regular_file(plan, label="authored RMCT convergence YAML")
    return load_experiment(plan, topology_profile=TOPOLOGY_PROFILE)


def _immutable_write(path: Path, document: Mapping[str, Any]) -> str:
    payload = _canonical_json(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PlanError(f"command-attestation parent must be a regular directory: {path.parent}")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        _regular_file(path, label="command attestation")
        if path.read_bytes() != payload:
            raise PlanError(f"refusing to overwrite different command attestation: {path}")
        return "resumed"
    return "written"


def command_attestation(
    *, repository: str | Path, plan: str | Path, segment_index: int, model_snapshot: str | Path
) -> dict[str, Any]:
    root = Path(repository).resolve()
    plan_path = Path(plan).resolve()
    compiled = _load_compiled_plan(plan_path)
    index = _safe_index(segment_index)
    # Compiling validates the exact canonical YAML.  Rebuild the selected
    # entry for the durable repository rather than comparing paths expanded in
    # this developer checkout with the allocation's repository checkout.
    compiled_entry = compiled["training"][index]
    convergence = compiled.get("rmct_convergence")
    if not isinstance(convergence, Mapping):
        raise PlanError("compiled RMCT convergence plan lacks its custody block")
    run_prefix = convergence.get("run_prefix")
    if not isinstance(run_prefix, str):
        raise PlanError("compiled RMCT convergence plan lacks a run prefix")
    entry = _training_entry(root, index, run_prefix=run_prefix)
    if compiled_entry["target"] != entry["target"] or compiled_entry["gpu_count"] != entry["gpu_count"]:
        raise PlanError("compiled target identity diverges from the frozen segment contract")
    args = segment_args(root, index, model_path=model_snapshot, run_prefix=run_prefix)
    # The plan's static entry is intentionally model-ID portable.  All other
    # args must agree, and launch resolves the model to the verified snapshot.
    expected = dict(entry["args"])
    expected["model"] = args["model"]
    if expected != args:
        raise PlanError("dynamic segment command diverges from the compiled frozen plan")
    from scripts.run_experiment import _argument_tokens

    argv = [sys.executable, str(root / "scripts" / "train_rlct.py"), *_argument_tokens(args)]
    return {
        "schema": COMMAND_SCHEMA,
        "condition": CONDITION_NAME,
        "run_prefix": run_prefix,
        "plan": {"path": str(plan_path), "sha256": _sha256_bytes(plan_path.read_bytes())},
        "model": {"repo_id": MODEL, "revision": BASE_SNAPSHOT, "snapshot_path": args["model"]},
        "segment": segment_record(root, index, run_prefix=run_prefix),
        "argv": argv,
        "environment_contract": {
            "cuda_visible_devices_preserved": True,
            "topology_profile": TOPOLOGY_PROFILE,
            "phase_shared": True,
            "gradient_checkpointing_layers": "all" if run_prefix == RUN_PREFIX else 16,
        },
    }


def _render_or_execute(args: argparse.Namespace, *, execute: bool) -> int:
    document = command_attestation(
        repository=args.repository,
        plan=args.plan,
        segment_index=args.segment_index,
        model_snapshot=args.model_snapshot,
    )
    status = _immutable_write(args.output, document)
    if not execute:
        print(_canonical_json({"path": str(Path(args.output).resolve()), "status": status}).decode(), end="")
        return 0
    if not args.yes:
        raise PlanError("execute requires --yes")
    process = subprocess.run(document["argv"], cwd=Path(args.repository).resolve(), env=dict(os.environ), check=False)
    return int(process.returncode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("render", "write one immutable exact training command"), ("execute", "write and run one command")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--repository", required=True, type=Path)
        command.add_argument("--plan", required=True, type=Path)
        command.add_argument("--segment-index", required=True, type=int)
        command.add_argument("--model-snapshot", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "execute":
            command.add_argument("--yes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "render":
            return _render_or_execute(args, execute=False)
        if args.command == "execute":
            return _render_or_execute(args, execute=True)
    except (OSError, PlanError, ValueError) as exc:
        _parser().error(str(exc))
    raise AssertionError(f"unexpected command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())


__all__ = [
    "BASE_SNAPSHOT",
    "BATCH_SIZE",
    "COMMAND_SCHEMA",
    "CONDITION_NAME",
    "CONVERGENCE_SCHEMA",
    "DATA_PATH",
    "DATA_SHA256",
    "DATAPOINTS_PER_SEGMENT",
    "GPU_COUNT",
    "HARD_CAP_OPTIMIZER_STEPS",
    "MANIFEST_PATH",
    "MANIFEST_SHA256",
    "MODEL",
    "ORIGINAL_RUN_PREFIX",
    "RECOVERY_METADATA",
    "RECOVERY_PARENT_ARTIFACT",
    "RECOVERY_PARENT_ARTIFACT_SHA256",
    "RECOVERY_SCHEMA",
    "RUN_PREFIX",
    "SEGMENT_SCHEMA",
    "SETTING_FACTORY",
    "TOPOLOGY_PROFILE",
    "TOTAL_SEGMENTS",
    "UPDATES_PER_SEGMENT",
    "WORKER_PARITY_ATTESTATION",
    "command_attestation",
    "compile_experiment",
    "final_checkpoint_path",
    "run_name",
    "segment_args",
    "segment_record",
    "target_name",
]
