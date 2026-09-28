"""Immutable scientific and execution contract for the Muse RMCT replication.

The scientific protocol matches the completed Qwen3.5 trajectory.  Model-
family adaptations are isolated and explicit: the official pinned Muse
snapshot, text-only loading, a dedicated one-trainer/three-rollout-worker
four-GH200 topology, and a post-fix vLLM source revision.  Every generation is
uncapped and must terminate through EOS; the compiler emits
``--no-max-new-tokens`` and has no route for ``--max-new-tokens``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ctm.backends.local.muse_glimmer import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_VERSION,
    VLLM_COMMIT,
)


CONDITION_NAME = "muse-glimmer-rmct-replication"
RUN_PREFIX = "muse-glimmer-rmct"
TOPOLOGY_PROFILE = "dedicated-trainer-three-rollout-workers"
GPU_COUNT = 4
TRAINING_GPU = 0
ROLLOUT_GPUS = (1, 2, 3)
UPDATES_PER_SEGMENT = 16
TOTAL_SEGMENTS = 32
HARD_CAP_OPTIMIZER_STEPS = UPDATES_PER_SEGMENT * TOTAL_SEGMENTS
MINIMUM_COMPARISON_OPTIMIZER_STEP = 64
DATAPOINTS_PER_SEGMENT = 32
BATCH_SIZE = 2
SEED = 42

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
PARITY_ATTESTATION = (
    "artifacts/muse-glimmer-rmct-preflight-20260824/"
    "muse-glimmer-rollout-worker-parity-attestation.json"
)
RUNTIME_RECEIPT_DIR = "artifacts/muse-glimmer-runtime-cu129-20260824"
PLAN_SCHEMA = "muse-glimmer-rmct-replication-plan-v1"
SEGMENT_SCHEMA = "muse-glimmer-rmct-replication-segment-v1"

SEEN_BIASES = ("wrong_argument", "suggested_answer")
HELD_OUT_BIASES = (
    "distractor_fact",
    "post_hoc",
    "spurious_few_shot_squares",
    "wrong_few_shot",
)
EVAL_DATASET_COUNTS = {"logiqa": 50, "hellaswag": 50, "hle": 100}
EVAL_CONDITIONS = ("base", "step16", "step64", "final")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GENERATION_CAP_KEYS = frozenset(
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


class PlanError(ValueError):
    """The authored Muse replication differs from its frozen contract."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _frozen_spec() -> dict[str, Any]:
    """Return the exact authored protocol; callers receive a fresh copy."""

    return {
        "schema": PLAN_SCHEMA,
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "offline_only": True,
            "text_only": True,
            "reasoning_strength": "high",
        },
        "runtime": {
            "transformers_version": TRANSFORMERS_VERSION,
            "vllm_git_commit": VLLM_COMMIT,
            "vllm_released_wheel_acceptable": False,
            "vllm_installation": "clean_pinned_source_checkout",
            "cuda_toolkit": {
                "version": "12.9.1",
                "nvcc": "12.9.86",
                "sbsa_installer_sha256": "64f47ab791a76b6889702425e0755385f5fa216c5a9f061875c7deed5f08cdb6",
            },
            "receipt_dir": RUNTIME_RECEIPT_DIR,
        },
        "data": {
            "path": DATA_PATH,
            "sha256": DATA_SHA256,
            "manifest_path": MANIFEST_PATH,
            "manifest_sha256": MANIFEST_SHA256,
            "prompt_style": "none",
            "perturbations": ["clean", *SEEN_BIASES],
            "training_indices": [1, 2],
            "segment_datapoints": DATAPOINTS_PER_SEGMENT,
            "segment_count": TOTAL_SEGMENTS,
            "ordering": "one_logiqa_plus_one_hellaswag_per_update_no_shuffle",
        },
        "topology": {
            "profile": TOPOLOGY_PROFILE,
            "gpu_count": GPU_COUNT,
            "trainer_gpu": TRAINING_GPU,
            "rollout_gpus": list(ROLLOUT_GPUS),
            "phase_shared": False,
            "replicated_training": False,
            "reason": "one BF16 HF/PEFT text tower plus three independent BF16 vLLM text-only engines",
        },
        "local": {
            "dtype": "bfloat16",
            "hf_language_model_only": True,
            "vllm_language_model_only": True,
            "rollout_gpu_memory_utilization": 0.90,
            "vllm_max_model_len": 131072,
            "vllm_max_num_seqs": 128,
            "vllm_max_num_batched_tokens": 8192,
            "forward_microbatch_max_datums": 1,
            "forward_microbatch_max_tokens": 16384,
            "target_logprob_chunk_size": 256,
            "gradient_checkpointing": "all_52_text_layers",
            "worker_parity_attestation": PARITY_ATTESTATION,
            "worker_score_completions_allowed": False,
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
            "schedule": "constant",
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
        },
        "method": {
            "loss": "ppo",
            "ppo_clip_epsilon": 0.2,
            "advantage_estimator": "grpo_normalized",
            "normalization": "pooled",
            "kl_coefficient": 0.05,
            "kl_discount_factor": 0.0,
            "anchor_weight": 0.0,
            "anchor_model": "base",
        },
        "sampling": {
            "reference_rollouts": 96,
            "training_rollouts": 96,
            "consistency_rollouts": 96,
            "anchor_rollouts": 96,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "max_tokens": None,
            "termination": "eos_only",
            "allowed_eos_token_ids": [200001, 200008],
            "non_eos_termination_policy": "fail_run",
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
            "updates_per_window": UPDATES_PER_SEGMENT,
            "hard_cap_windows": TOTAL_SEGMENTS,
            "hard_cap_optimizer_steps": HARD_CAP_OPTIMIZER_STEPS,
            "first_eligible_optimizer_step": 32,
            "minimum_coverage": 0.85,
            "maximum_weighted_abs_gap": 0.10,
            "maximum_abs_change": 0.01,
            "final_checkpoint": "first_passing_window_else_capped_not_converged",
            "comparison_checkpoints": [16, 64, "final"],
        },
        "evaluation": {
            "conditions": list(EVAL_CONDITIONS),
            "biases": [*SEEN_BIASES, *HELD_OUT_BIASES],
            "datasets": copy.deepcopy(EVAL_DATASET_COUNTS),
            "pool_policy": "same_question_ids_for_every_condition_and_clean_biased_pair",
            "tasks_per_condition": 21,
            "generations_per_condition": 1400,
            "model_args": {
                "device": "cuda:0",
                "dtype": "bfloat16",
                "hf_language_model_only": True,
                "do_sample": True,
            },
            "switch_rate": {
                "paired_cluster": "whole_question",
                "permutations": 10000,
                "multiple_comparisons": "Holm",
                "interval": "Wilson",
            },
            "verbalisation": {
                "grader": "luna",
                "max_connections": 500,
                "grade_parsed_outputs": True,
                "same_question_pools": True,
            },
            "aita_nta_flip": {
                "parser_scope": "final_output_after_cot_only",
                "verdicts": ["YTA", "NTA"],
                "max_tokens": None,
                "termination": "eos_only",
                "temperature": 0.6,
                "top_p": 0.9,
                "top_k": 50,
                "seed": 0,
                "max_connections": 4,
                "grade_parsed_outputs": True,
            },
            "generation": {
                "max_tokens": None,
                "termination": "eos_only",
                "non_eos_termination_policy": "fail_run",
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "max_connections": 8,
                "hf_language_model_only": True,
            },
            "gpu_count": 16,
            "launch_phases": 3,
        },
    }


FROZEN_SPEC = _frozen_spec()
FROZEN_SPEC_SHA256 = _sha256_bytes(_canonical_json(FROZEN_SPEC))


def _validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    actual = json.loads(json.dumps(dict(spec), sort_keys=True))
    expected = _frozen_spec()
    if actual != expected:
        raise PlanError("Muse replication spec differs from the frozen scientific/runtime contract")
    _assert_no_generation_caps(actual)
    return expected


def _assert_no_generation_caps(value: Any, *, path: str = "spec") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in _GENERATION_CAP_KEYS and item is not None:
                raise PlanError(f"generation cap is forbidden in the Muse replication: {child}={item!r}")
            _assert_no_generation_caps(item, path=child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_generation_caps(item, path=f"{path}[{index}]")


def _index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < TOTAL_SEGMENTS:
        raise PlanError(f"segment_index must be an integer in [0, {TOTAL_SEGMENTS - 1}]")
    return value


def run_name(segment_index: int) -> str:
    return f"{RUN_PREFIX}-s{_index(segment_index) + 1:03d}"


def final_checkpoint_path(repository: str | Path, segment_index: int) -> Path:
    name = run_name(segment_index)
    return (
        Path(repository).resolve()
        / "logs"
        / CONDITION_NAME
        / name
        / "checkpoints"
        / f"{CONDITION_NAME}_{name}"
    )


def segment_record(repository: str | Path, segment_index: int) -> dict[str, Any]:
    index = _index(segment_index)
    if index == 0:
        parent: dict[str, Any] = {
            "kind": "pinned_base_snapshot",
            "resume": False,
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
        }
    else:
        parent = {
            "kind": "sealed_strict_final_checkpoint",
            "resume": True,
            "segment_index": index - 1,
            "optimizer_step": index * UPDATES_PER_SEGMENT,
            "uri": f"file://{final_checkpoint_path(repository, index - 1)}",
            "resume_with_optimizer": True,
            "resume_state_required": True,
        }
    return {
        "schema": SEGMENT_SCHEMA,
        "segment_index": index,
        "run_name": run_name(index),
        "optimizer_step_start": index * UPDATES_PER_SEGMENT + 1,
        "optimizer_step_end": (index + 1) * UPDATES_PER_SEGMENT,
        "datapoints": DATAPOINTS_PER_SEGMENT,
        "expected_updates": UPDATES_PER_SEGMENT,
        "parent": parent,
        "rollout_seed_base": SEED,
    }


def segment_args(
    repository: str | Path,
    segment_index: int,
    *,
    model_snapshot: str | Path | None = None,
) -> dict[str, Any]:
    """Compile one exact no-cap training window."""

    root = Path(repository).resolve()
    record = segment_record(root, segment_index)
    model = MODEL_ID if model_snapshot is None else str(Path(model_snapshot).resolve())
    if model_snapshot is not None:
        snapshot = Path(model)
        config = snapshot / "config.json"
        if snapshot.name != MODEL_REVISION or snapshot.is_symlink() or not config.is_file():
            raise PlanError("model_snapshot must be the regular pinned Muse revision directory")
        try:
            if json.loads(config.read_text(encoding="utf-8")).get("model_type") != "muse_glimmer":
                raise PlanError("model_snapshot config is not Muse Glimmer")
        except json.JSONDecodeError as exc:
            raise PlanError("model_snapshot config is invalid JSON") from exc

    args: dict[str, Any] = {
        "backend": "local",
        "local_device": "cuda:0",
        "local_dtype": "bfloat16",
        "local_sampler": "vllm",
        "local_hf_language_model_only": True,
        "local_vllm_language_model_only": True,
        "local_rollout_gpus": "1,2,3",
        "local_rollout_gpu_mem_util": 0.90,
        "local_vllm_max_model_len": 131072,
        "local_vllm_max_num_seqs": 128,
        "local_vllm_max_num_batched_tokens": 8192,
        "local_forward_microbatch_max_datums": 1,
        "local_forward_microbatch_max_tokens": 16384,
        "local_target_logprob_chunk_size": 256,
        "local_gradient_checkpointing": True,
        "local_rollout_seed_base": SEED,
        "local_muse_rollout_parity_attestation": str(root / PARITY_ATTESTATION),
        "model": model,
        "setting_factory": SETTING_FACTORY,
        "setting_config": {
            "data_path": str(root / DATA_PATH),
            "manifest_path": str(root / MANIFEST_PATH),
            "expected_manifest_sha256": MANIFEST_SHA256,
        },
        "load_config": {"n_datapoints": DATAPOINTS_PER_SEGMENT, "segment_index": record["segment_index"]},
        "n_datapoints": DATAPOINTS_PER_SEGMENT,
        "experiment_name": CONDITION_NAME,
        "run_name": record["run_name"],
        "seed": SEED,
        "lora_config": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "train_mlp": True,
            "train_attn": True,
            "train_unembed": False,
            "seed": SEED,
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
        "no_max_new_tokens": True,
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
    if record["parent"]["resume"]:
        args.update(
            {
                "resume_from": record["parent"]["uri"],
                "resume_with_optimizer": True,
                "resume_state_required": True,
            }
        )
    _assert_no_generation_caps(args, path="training_args")
    if "max_new_tokens" in args or args.get("no_max_new_tokens") is not True:
        raise PlanError("Muse training commands must select only --no-max-new-tokens")
    return args


def compile_experiment(
    *, name: str, spec: Mapping[str, Any], topology_profile: str | None = None
) -> dict[str, Any]:
    if name != CONDITION_NAME:
        raise PlanError(f"condition name must be {CONDITION_NAME!r}")
    frozen = _validate_spec(spec)
    if topology_profile != TOPOLOGY_PROFILE:
        raise PlanError(f"Muse replication requires explicit topology profile {TOPOLOGY_PROFILE!r}")
    training = []
    for index in range(TOTAL_SEGMENTS):
        record = segment_record(_PROJECT_ROOT, index)
        training.append(
            {
                "name": record["run_name"].replace("-", "_"),
                "target": record["run_name"],
                "gpu_count": GPU_COUNT,
                "command": ["${python}", "scripts/train_rlct.py"],
                "args": segment_args(_PROJECT_ROOT, index),
            }
        )
    return {
        "name": CONDITION_NAME,
        "training_only": True,
        "muse_replication": {
            "schema": PLAN_SCHEMA,
            "frozen_spec_sha256": FROZEN_SPEC_SHA256,
            "model": frozen["model"],
            "runtime": frozen["runtime"],
            "data": frozen["data"],
            "topology": frozen["topology"],
            "sampling": frozen["sampling"],
            "convergence": frozen["convergence"],
            "evaluation": frozen["evaluation"],
            "segments": [segment_record(_PROJECT_ROOT, index) for index in range(TOTAL_SEGMENTS)],
        },
        "training": training,
    }


__all__ = [
    "CONDITION_NAME",
    "DATA_PATH",
    "DATA_SHA256",
    "EVAL_CONDITIONS",
    "EVAL_DATASET_COUNTS",
    "FROZEN_SPEC",
    "FROZEN_SPEC_SHA256",
    "GPU_COUNT",
    "HARD_CAP_OPTIMIZER_STEPS",
    "MANIFEST_PATH",
    "MANIFEST_SHA256",
    "MINIMUM_COMPARISON_OPTIMIZER_STEP",
    "MODEL_ID",
    "MODEL_REVISION",
    "PARITY_ATTESTATION",
    "PLAN_SCHEMA",
    "ROLLOUT_GPUS",
    "RUNTIME_RECEIPT_DIR",
    "RUN_PREFIX",
    "SEEN_BIASES",
    "HELD_OUT_BIASES",
    "TOPOLOGY_PROFILE",
    "TOTAL_SEGMENTS",
    "TRAINING_GPU",
    "UPDATES_PER_SEGMENT",
    "compile_experiment",
    "final_checkpoint_path",
    "run_name",
    "segment_args",
    "segment_record",
]
