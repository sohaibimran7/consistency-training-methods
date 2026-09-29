"""Compile the fixed Isambard RMCT-256 convergence trajectory.

The condition is deliberately a sequence of independently attested local-RL
segments, rather than one 256-row invocation with an informal stopping rule.
Each segment sees one immutable 64-row slice, writes a completed local
``kind='both'`` checkpoint after its 16th update, and the next segment names
that final checkpoint's deterministic path explicitly.  This gives the
protected launcher a static parent interface to attest before model startup.

There are exactly four passes through the four frozen blocks.  Extending the
trajectory is a new experiment-plan decision: this factory has no pass-count
argument and rejects any altered hard cap.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any


MODEL = "Qwen/Qwen3.5-9B"
BASE_SNAPSHOT = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CONDITION_NAME = "rmct256-convergence-isambard-4x64-20260811"
ARTIFACT_ROOT = "artifacts/rmct-hle-qwen3.5-9b-dense-rmct256-convergence-isambard-4x64-20260811"
FIGURE_ROOT = "figures/rmct-hle-qwen3.5-9b-dense-rmct256-convergence-isambard-4x64-20260811"
SELECTION_PATH = (
    "artifacts/rmct-256-training-20260804/"
    "rmct-256-training-7602aca7f92312e24b884a8dd4f290a5c2374350e51cd3b6bf3fffbcf216a55a.jsonl"
)
SELECTION_MANIFEST_PATH = (
    "artifacts/rmct-256-training-20260804/"
    "rmct-256-training-manifest-5c18fa31ddaaca76256bef15cc54ddfa0f103287b0f5c940fd0a87cc8e2e179e.json"
)
SEGMENT_MANIFEST_PATH = (
    "artifacts/rmct-256-convergence-segments-20260811/"
    "rmct-256-convergence-segments-manifest-2f81e40297c594f1a05413fded238ac956a345f4fa6d3a636288aa4a9edccdc7.json"
)
SELECTION_CONTENT_SHA256 = "7602aca7f92312e24b884a8dd4f290a5c2374350e51cd3b6bf3fffbcf216a55a"
SELECTION_MANIFEST_SHA256 = "5c18fa31ddaaca76256bef15cc54ddfa0f103287b0f5c940fd0a87cc8e2e179e"
SEGMENT_MANIFEST_SHA256 = "2f81e40297c594f1a05413fded238ac956a345f4fa6d3a636288aa4a9edccdc7"

TOPOLOGY_PROFILE = "four-gpu"
GPU_COUNT = 4
COORDINATOR_DEVICE = "cuda:0"
ROLLOUT_GPUS = (1, 2, 3)

HARD_CAP_PASSES = 4
SEGMENTS_PER_PASS = 4
ROWS_PER_SEGMENT = 64
UPDATES_PER_SEGMENT = 16
TOTAL_SEGMENTS = HARD_CAP_PASSES * SEGMENTS_PER_PASS
TOTAL_UPDATES = TOTAL_SEGMENTS * UPDATES_PER_SEGMENT
ROW_OFFSETS = (0, 64, 128, 192)
BASE_ROLLOUT_SEED = 42

SEGMENT_METADATA_SCHEMA = "rmct256_convergence_segment_v1"
CONVERGENCE_SCHEMA = "rmct256_convergence_plan_v1"
_SAFE_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_SPEC_KEYS = frozenset(
    {
        "topology_profiles",
        "model",
        "base_snapshot",
        "isambard_runtime",
        "artifact_root",
        "figure_root",
        "training_only",
        "onpolicy_target_attestation",
        "tracking",
        "selection",
        "setting_config",
        "lora",
        "local",
        "sampling_defaults",
        "optimizer",
        "rate_matching",
        "convergence",
    }
)


def _object(value: Any, *, label: str, keys: set[str] | frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    result = dict(value)
    unknown = sorted(set(result) - set(keys))
    missing = sorted(set(keys) - set(result))
    if missing or unknown:
        details = [
            *(f"missing {missing}" for _ in [None] if missing),
            *(f"unknown {unknown}" for _ in [None] if unknown),
        ]
        raise ValueError(f"{label}: {', '.join(details)}")
    return result


def _safe_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or not _SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"{label} must start with a letter or digit and contain only letters, digits, dots, underscores, and hyphens"
        )
    return value


def _repository_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    if value.startswith("/") or value.startswith("file:") or "${" in value:
        raise ValueError(f"{label} must be a literal repository-relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} must be a normalized repository-relative path")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lower-case SHA-256 hex digest")
    return value


def _exact(value: Any, expected: Any, *, label: str) -> Any:
    if value != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {value!r}")
    return value


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _profile(spec: Mapping[str, Any], requested: str | None) -> tuple[str, dict[str, Any]]:
    raw_profiles = spec["topology_profiles"]
    if not isinstance(raw_profiles, Mapping) or set(raw_profiles) != {TOPOLOGY_PROFILE}:
        raise ValueError(
            f"RMCT-256 convergence topology_profiles must contain only {TOPOLOGY_PROFILE!r}"
        )
    if requested is None:
        raise ValueError(
            f"RMCT-256 convergence requires an explicit topology profile; choose {TOPOLOGY_PROFILE!r}"
        )
    if requested != TOPOLOGY_PROFILE:
        raise ValueError(
            f"unknown RMCT-256 convergence topology profile {requested!r}; choose {TOPOLOGY_PROFILE!r}"
        )
    profile = _object(
        raw_profiles[TOPOLOGY_PROFILE],
        label=f"topology_profiles.{TOPOLOGY_PROFILE}",
        keys={
            "target_prefix",
            "run_name_prefix",
            "gpu_count",
            "coordinator_device",
            "rollout_gpus",
            "gradient_checkpointing_layers",
        },
    )
    profile["target_prefix"] = _safe_name(
        profile["target_prefix"], label=f"topology_profiles.{TOPOLOGY_PROFILE}.target_prefix"
    )
    profile["run_name_prefix"] = _safe_name(
        profile["run_name_prefix"], label=f"topology_profiles.{TOPOLOGY_PROFILE}.run_name_prefix"
    )
    _exact(profile["target_prefix"], "rmct256-convergence", label="RMCT-256 convergence target prefix")
    _exact(
        profile["run_name_prefix"],
        "rmct256-convergence-isambard-4x64-20260811",
        label="RMCT-256 convergence run-name prefix",
    )
    _exact(profile["gpu_count"], GPU_COUNT, label="RMCT-256 convergence gpu_count")
    _exact(
        profile["coordinator_device"], COORDINATOR_DEVICE, label="RMCT-256 convergence coordinator device"
    )
    _exact(profile["rollout_gpus"], list(ROLLOUT_GPUS), label="RMCT-256 convergence rollout GPUs")
    _exact(
        profile["gradient_checkpointing_layers"],
        "all",
        label="RMCT-256 convergence gradient checkpointing scope",
    )
    return requested, profile


def _validate_base_snapshot(value: Any) -> dict[str, Any]:
    snapshot = _object(
        value,
        label="base_snapshot",
        keys={"repo_id", "revision", "offline_only"},
    )
    _exact(snapshot["repo_id"], MODEL, label="base_snapshot.repo_id")
    _exact(snapshot["revision"], BASE_SNAPSHOT, label="base_snapshot.revision")
    _exact(snapshot["offline_only"], True, label="base_snapshot.offline_only")
    return snapshot


def _validate_isambard_runtime(value: Any) -> dict[str, Any]:
    runtime = _object(
        value,
        label="isambard_runtime",
        keys={
            "platform",
            "architecture",
            "vllm_version",
            "vllm_cuda",
            "transformers_version",
            "gdn_prefill_backend",
        },
    )
    expected = {
        "platform": "isambard-ai-gh200",
        "architecture": "aarch64",
        "vllm_version": "0.21.0",
        "vllm_cuda": "12.9",
        "transformers_version": "5.5.4",
        "gdn_prefill_backend": "triton",
    }
    for key, expected_value in expected.items():
        _exact(runtime[key], expected_value, label=f"isambard_runtime.{key}")
    return runtime


def _validate_selection(value: Any) -> dict[str, Any]:
    selection = _object(
        value,
        label="selection",
        keys={
            "pairs_path",
            "selection_manifest",
            "selection_content_sha256",
            "selection_manifest_sha256",
            "segment_manifest",
            "segment_manifest_sha256",
        },
    )
    for key in ("pairs_path", "selection_manifest", "segment_manifest"):
        selection[key] = _repository_path(selection[key], label=f"selection.{key}")
    for key in (
        "selection_content_sha256",
        "selection_manifest_sha256",
        "segment_manifest_sha256",
    ):
        selection[key] = _sha256(selection[key], label=f"selection.{key}")
    _exact(selection["pairs_path"], SELECTION_PATH, label="selection.pairs_path")
    _exact(selection["selection_manifest"], SELECTION_MANIFEST_PATH, label="selection.selection_manifest")
    _exact(selection["segment_manifest"], SEGMENT_MANIFEST_PATH, label="selection.segment_manifest")
    _exact(selection["selection_content_sha256"], SELECTION_CONTENT_SHA256, label="selection.selection_content_sha256")
    _exact(selection["selection_manifest_sha256"], SELECTION_MANIFEST_SHA256, label="selection.selection_manifest_sha256")
    _exact(selection["segment_manifest_sha256"], SEGMENT_MANIFEST_SHA256, label="selection.segment_manifest_sha256")
    return selection


def _validate_setting_config(value: Any) -> dict[str, Any]:
    """Freeze the biased-prompt member of each RMCT training pair.

    ``SycophancySetting`` treats a true control flag as an unbiased-vs-
    unbiased pair.  This condition instead requires the frozen biased prompt,
    so the false value is an explicit scientific contract rather than a
    reliance on the setting factory's default.
    """

    setting_config = _object(value, label="setting_config", keys={"control"})
    _exact(setting_config["control"], False, label="setting_config.control")
    return setting_config


def _validate_lora(value: Any) -> dict[str, Any]:
    lora = _object(
        value,
        label="lora",
        keys={"rank", "alpha", "dropout", "train_mlp", "train_attn", "train_unembed"},
    )
    expected = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.0,
        "train_mlp": True,
        "train_attn": True,
        "train_unembed": False,
    }
    for key, expected_value in expected.items():
        _exact(lora[key], expected_value, label=f"lora.{key}")
    return lora


def _validate_local(value: Any) -> dict[str, Any]:
    local = _object(
        value,
        label="local",
        keys={
            "dtype",
            "sampler",
            "gpu_memory_utilization",
            "rollout_gpu_memory_utilization",
            "rollout_seed_base",
            "vllm_language_model_only",
            "vllm_max_num_seqs",
            "vllm_max_num_batched_tokens",
            "vllm_max_model_len",
            "vllm_gdn_prefill_backend",
            "gradient_checkpointing",
            "gradient_checkpointing_layers",
            "forward_microbatch_max_datums",
            "forward_microbatch_max_tokens",
            "target_logprob_chunk_size",
        },
    )
    expected = {
        "dtype": "bfloat16",
        "sampler": "vllm",
        "gpu_memory_utilization": 0.34,
        "rollout_gpu_memory_utilization": 0.75,
        "rollout_seed_base": BASE_ROLLOUT_SEED,
        "vllm_language_model_only": True,
        "vllm_max_num_seqs": 256,
        "vllm_max_num_batched_tokens": 8192,
        "vllm_max_model_len": 32768,
        "vllm_gdn_prefill_backend": "triton",
        "gradient_checkpointing": True,
        "gradient_checkpointing_layers": "all",
        "forward_microbatch_max_datums": 8,
        "forward_microbatch_max_tokens": 20480,
        "target_logprob_chunk_size": 2048,
    }
    for key, expected_value in expected.items():
        _exact(local[key], expected_value, label=f"local.{key}")
    return local


def _validate_sampling_defaults(value: Any) -> dict[str, Any]:
    defaults = _object(
        value,
        label="sampling_defaults",
        keys={"unparsed_handling", "max_resample_attempts", "snr_mode", "snr_z", "snr_normalizer"},
    )
    expected = {
        "unparsed_handling": "discard",
        "max_resample_attempts": 4,
        "snr_mode": "soft",
        "snr_z": 2.0,
        "snr_normalizer": "trait_std",
    }
    for key, expected_value in expected.items():
        _exact(defaults[key], expected_value, label=f"sampling_defaults.{key}")
    return defaults


def _validate_optimizer(value: Any) -> dict[str, Any]:
    optimizer = _object(
        value,
        label="optimizer",
        keys={"learning_rate", "learning_rate_schedule", "beta1", "beta2", "eps", "weight_decay", "grad_clip_norm"},
    )
    expected = {
        "learning_rate": 0.0001,
        "learning_rate_schedule": "constant",
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
    }
    for key, expected_value in expected.items():
        _exact(optimizer[key], expected_value, label=f"optimizer.{key}")
    return optimizer


def _validate_rate_matching(value: Any) -> dict[str, Any]:
    rate_matching = _object(
        value,
        label="rate_matching",
        keys={
            "datapoints",
            "rollouts",
            "batch_size",
            "epochs",
            "temperature",
            "max_new_tokens",
            "learning_rate_schedule",
            "kl_coefficient",
            "kl_discount_factor",
            "ppo_clip_epsilon",
            "anchor_weight",
            "anchor_model",
            "loss",
            "advantage_estimator",
            "normalization",
            "gradient_accumulation_steps",
            "refresh_every",
            "checkpoint_every",
            "save_state",
        },
    )
    rollouts = _object(
        rate_matching["rollouts"],
        label="rate_matching.rollouts",
        keys={"reference", "training", "consistency", "anchor"},
    )
    expected = {
        "datapoints": ROWS_PER_SEGMENT,
        "rollouts": {"reference": 96, "training": 96, "consistency": 96, "anchor": 96},
        "batch_size": 4,
        "epochs": 1,
        "temperature": 1.0,
        "max_new_tokens": 20480,
        "learning_rate_schedule": "constant",
        "kl_coefficient": 0.05,
        "kl_discount_factor": 0.0,
        "ppo_clip_epsilon": 0.2,
        "anchor_weight": 0.0,
        "anchor_model": "base",
        "loss": "ppo",
        "advantage_estimator": "grpo_normalized",
        "normalization": "pooled",
        "gradient_accumulation_steps": 1,
        "refresh_every": 1,
        "checkpoint_every": UPDATES_PER_SEGMENT,
        "save_state": True,
    }
    _exact(rollouts, expected["rollouts"], label="rate_matching.rollouts")
    for key, expected_value in expected.items():
        if key != "rollouts":
            value = rate_matching[key]
            if key in {"kl_coefficient", "kl_discount_factor", "ppo_clip_epsilon"}:
                value = _finite_number(value, label=f"rate_matching.{key}")
            _exact(value, expected_value, label=f"rate_matching.{key}")
    return rate_matching


def _validate_convergence(value: Any) -> dict[str, Any]:
    convergence = _object(
        value,
        label="convergence",
        keys={
            "mode",
            "hard_cap_passes",
            "segments_per_pass",
            "rows_per_segment",
            "updates_per_segment",
            "row_offsets",
            "minimum_completed_passes",
            "full_pass_min_delta",
            "non_improving_pass_patience",
            "retain",
            "cap_outcome",
        },
    )
    expected = {
        "mode": "optimizer_data_segment",
        "hard_cap_passes": HARD_CAP_PASSES,
        "segments_per_pass": SEGMENTS_PER_PASS,
        "rows_per_segment": ROWS_PER_SEGMENT,
        "updates_per_segment": UPDATES_PER_SEGMENT,
        "row_offsets": list(ROW_OFFSETS),
        "minimum_completed_passes": 1,
        "full_pass_min_delta": 0.01,
        "non_improving_pass_patience": 2,
        "retain": "retain_all_16_step_checkpoints",
        "cap_outcome": "capped_while_improving",
    }
    for key, expected_value in expected.items():
        _exact(convergence[key], expected_value, label=f"convergence.{key}")
    return convergence


def _run_name(prefix: str, *, pass_index: int, segment_index: int) -> str:
    return f"{prefix}-p{pass_index:02d}-s{segment_index + 1:02d}"


def _target_name(prefix: str, *, pass_index: int, segment_index: int) -> str:
    return f"{prefix}-p{pass_index:02d}-s{segment_index + 1:02d}"


def _final_checkpoint_uri(*, experiment_name: str, run_name: str) -> str:
    """Return the only checkpoint URI eligible to parent the next segment.

    A final local-RL checkpoint has no step suffix.  ``${project_root}`` is
    resolved by the normal runner before target attestation, making this URI
    static and sealable while remaining checkout-location independent.
    """

    return (
        f"file://${{project_root}}/logs/{experiment_name}/{run_name}/checkpoints/"
        f"{experiment_name}_{run_name}"
    )


def compile_experiment(
    *,
    name: str,
    spec: Mapping[str, Any],
    topology_profile: str | None = None,
) -> dict[str, Any]:
    """Compile all 16 statically linked convergence segments.

    The compiler is intentionally not parameterized by a pass count, initial
    checkpoint, or rollout seed.  The only selectable property is the one
    authored four-GPU Isambard profile, and every later segment resumes from
    the exact final directory expected from its immediate predecessor.
    """

    if not isinstance(spec, Mapping):
        raise ValueError("RMCT-256 convergence spec must be an object")
    raw = dict(spec)
    unknown = sorted(set(raw) - _SPEC_KEYS)
    missing = sorted(_SPEC_KEYS - set(raw))
    if missing or unknown:
        details = [
            *(f"missing {missing}" for _ in [None] if missing),
            *(f"unknown {unknown}" for _ in [None] if unknown),
        ]
        raise ValueError(f"RMCT-256 convergence spec: {', '.join(details)}")

    profile_name, profile = _profile(raw, topology_profile)
    _exact(name, CONDITION_NAME, label="RMCT-256 convergence condition name")
    _exact(raw["model"], MODEL, label="model")
    base_snapshot = _validate_base_snapshot(raw["base_snapshot"])
    isambard_runtime = _validate_isambard_runtime(raw["isambard_runtime"])
    artifact_root = _repository_path(raw["artifact_root"], label="artifact_root")
    figure_root = _repository_path(raw["figure_root"], label="figure_root")
    _exact(artifact_root, ARTIFACT_ROOT, label="artifact_root")
    _exact(figure_root, FIGURE_ROOT, label="figure_root")
    _exact(raw["training_only"], True, label="training_only")
    _exact(raw["onpolicy_target_attestation"], True, label="onpolicy_target_attestation")
    tracking = _object(raw["tracking"], label="tracking", keys={"wandb_project"})
    _exact(
        tracking["wandb_project"],
        "rmct256_convergence_isambard_4x64_20260811",
        label="tracking.wandb_project",
    )
    selection = _validate_selection(raw["selection"])
    setting_config = _validate_setting_config(raw["setting_config"])
    lora = _validate_lora(raw["lora"])
    local = _validate_local(raw["local"])
    sampling_defaults = _validate_sampling_defaults(raw["sampling_defaults"])
    optimizer = _validate_optimizer(raw["optimizer"])
    rate_matching = _validate_rate_matching(raw["rate_matching"])
    _exact(
        rate_matching["learning_rate_schedule"],
        optimizer["learning_rate_schedule"],
        label="rate_matching.learning_rate_schedule",
    )
    convergence = _validate_convergence(raw["convergence"])
    method = {
        "loss_fn": rate_matching["loss"],
        "kl_coefficient": rate_matching["kl_coefficient"],
        "kl_discount_factor": rate_matching["kl_discount_factor"],
        "ppo_clip_epsilon": rate_matching["ppo_clip_epsilon"],
    }

    training: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for global_segment_index in range(TOTAL_SEGMENTS):
        pass_index = global_segment_index // SEGMENTS_PER_PASS + 1
        segment_index = global_segment_index % SEGMENTS_PER_PASS
        row_offset = ROW_OFFSETS[segment_index]
        checkpoint_step = (global_segment_index + 1) * UPDATES_PER_SEGMENT
        target = _target_name(
            profile["target_prefix"], pass_index=pass_index, segment_index=segment_index
        )
        run_name = _run_name(
            profile["run_name_prefix"], pass_index=pass_index, segment_index=segment_index
        )

        if global_segment_index == 0:
            parent: dict[str, Any] = {
                "kind": "pinned_base_snapshot",
                "resume": False,
                "base_snapshot": dict(base_snapshot),
            }
            resume_args: dict[str, Any] = {}
        else:
            parent_global_index = global_segment_index - 1
            parent_pass_index = parent_global_index // SEGMENTS_PER_PASS + 1
            parent_segment_index = parent_global_index % SEGMENTS_PER_PASS
            parent_target = _target_name(
                profile["target_prefix"],
                pass_index=parent_pass_index,
                segment_index=parent_segment_index,
            )
            parent_run_name = _run_name(
                profile["run_name_prefix"],
                pass_index=parent_pass_index,
                segment_index=parent_segment_index,
            )
            parent_uri = _final_checkpoint_uri(experiment_name=name, run_name=parent_run_name)
            parent = {
                "kind": "strict_final_checkpoint",
                "resume": True,
                "global_segment_index": parent_global_index,
                "target": parent_target,
                "run_name": parent_run_name,
                "uri": parent_uri,
                "checkpoint_step": (parent_global_index + 1) * UPDATES_PER_SEGMENT,
                "expected_kind": "both",
                "expected_final": True,
                "resume_with_optimizer": True,
                "resume_state_required": True,
            }
            resume_args = {
                "resume_from": parent_uri,
                "resume_with_optimizer": True,
                "resume_state_required": True,
            }

        metadata = {
            "schema": SEGMENT_METADATA_SCHEMA,
            "condition": name,
            "base_snapshot_commit": BASE_SNAPSHOT,
            "global_segment_index": global_segment_index,
            "pass_index": pass_index,
            "segment_index": segment_index,
            "checkpoint_step": checkpoint_step,
            "updates": UPDATES_PER_SEGMENT,
            "questions": ROWS_PER_SEGMENT,
            "row_offset": row_offset,
            "segment_namespace": f"{name}/p{pass_index:02d}/s{segment_index + 1:02d}",
            "selection_content_sha256": selection["selection_content_sha256"],
            "selection_manifest_sha256": selection["selection_manifest_sha256"],
            "segment_manifest": selection["segment_manifest"],
            "segment_manifest_sha256": selection["segment_manifest_sha256"],
            "setting_config": dict(setting_config),
            "parent": parent,
            "base_snapshot": dict(base_snapshot),
            "isambard_runtime": dict(isambard_runtime),
            # Record the exact AdamW configuration alongside the segment
            # identity.  The same values are also emitted as explicit CLI
            # arguments below, so neither target attestation nor execution
            # can silently inherit mutable trainer defaults.
            "optimizer": dict(optimizer),
            # These are PPO/RMCT method parameters, not AdamW settings.  They
            # are separately sealed because both formerly had runtime defaults.
            "method": dict(method),
        }
        load_config = {
            "n_datapoints": ROWS_PER_SEGMENT,
            "row_offset": row_offset,
            "rmct256_segment_index": segment_index,
            "selection_manifest": selection["selection_manifest"],
            "rmct256_convergence_manifest": selection["segment_manifest"],
            "rmct256_convergence_manifest_sha256": selection["segment_manifest_sha256"],
            # This is metadata-only.  The setting independently validates the
            # concrete fields above so this object cannot alter the slice.
            "rmct256_convergence_metadata": metadata,
        }
        args = {
            "backend": "local",
            "local_dtype": local["dtype"],
            "local_sampler": local["sampler"],
            "local_gpu_mem_util": local["gpu_memory_utilization"],
            "local_rollout_gpu_mem_util": local["rollout_gpu_memory_utilization"],
            "local_rollout_seed_base": BASE_ROLLOUT_SEED + len(ROLLOUT_GPUS) * global_segment_index,
            "local_vllm_language_model_only": local["vllm_language_model_only"],
            "local_vllm_max_num_seqs": local["vllm_max_num_seqs"],
            "local_vllm_max_num_batched_tokens": local["vllm_max_num_batched_tokens"],
            "local_vllm_max_model_len": local["vllm_max_model_len"],
            "local_vllm_gdn_prefill_backend": local["vllm_gdn_prefill_backend"],
            # Omitting local_gradient_checkpointing_layers means all layers;
            # the authored profile and metadata explicitly pin that choice.
            "local_gradient_checkpointing": local["gradient_checkpointing"],
            "local_forward_microbatch_max_datums": local["forward_microbatch_max_datums"],
            "local_forward_microbatch_max_tokens": local["forward_microbatch_max_tokens"],
            "local_target_logprob_chunk_size": local["target_logprob_chunk_size"],
            "local_device": profile["coordinator_device"],
            "local_rollout_gpus": ",".join(str(gpu) for gpu in profile["rollout_gpus"]),
            "model": MODEL,
            "setting_factory": "ctm_data.adapters.mcq_bias:create_setting",
            "setting_config": {"data_paths": [selection["pairs_path"]], **setting_config},
            "load_config": load_config,
            "experiment_name": "${experiment}",
            "run_name": run_name,
            "seed": BASE_ROLLOUT_SEED,
            "lora_config": {**lora, "seed": BASE_ROLLOUT_SEED},
            "lr": optimizer["learning_rate"],
            "lr_schedule": optimizer["learning_rate_schedule"],
            "beta1": optimizer["beta1"],
            "beta2": optimizer["beta2"],
            "eps": optimizer["eps"],
            "weight_decay": optimizer["weight_decay"],
            "grad_clip_norm": optimizer["grad_clip_norm"],
            "kl_coef": rate_matching["kl_coefficient"],
            "kl_discount_factor": rate_matching["kl_discount_factor"],
            "local_ppo_clip_epsilon": rate_matching["ppo_clip_epsilon"],
            "anchor_weight": rate_matching["anchor_weight"],
            "anchor_model": rate_matching["anchor_model"],
            "loss_fn": rate_matching["loss"],
            "advantage_estimator": rate_matching["advantage_estimator"],
            "normalization": rate_matching["normalization"],
            "n_ref_rollouts": rate_matching["rollouts"]["reference"],
            "n_train_rollouts": rate_matching["rollouts"]["training"],
            "n_consistency_rollouts": rate_matching["rollouts"]["consistency"],
            "n_anchor_rollouts": rate_matching["rollouts"]["anchor"],
            "temperature": rate_matching["temperature"],
            "max_new_tokens": rate_matching["max_new_tokens"],
            "batch_size": rate_matching["batch_size"],
            "gradient_accumulation_steps": rate_matching["gradient_accumulation_steps"],
            "refresh_every": rate_matching["refresh_every"],
            "n_epochs": rate_matching["epochs"],
            "checkpoint_every": rate_matching["checkpoint_every"],
            "save_state": rate_matching["save_state"],
            "unparsed_handling": sampling_defaults["unparsed_handling"],
            "max_resample_attempts": sampling_defaults["max_resample_attempts"],
            "snr_mode": sampling_defaults["snr_mode"],
            "snr_z": sampling_defaults["snr_z"],
            "snr_normalizer": sampling_defaults["snr_normalizer"],
            "wandb_project": tracking["wandb_project"],
            "require_onpolicy_target_attestation": True,
            **resume_args,
            "yes": True,
        }
        command_name = f"rmct256_convergence_p{pass_index:02d}_s{segment_index + 1:02d}"
        training.append(
            {
                "name": command_name,
                "target": target,
                "gpu_count": profile["gpu_count"],
                "command": ["${python}", "scripts/train_rlct.py"],
                "args": args,
            }
        )
        segments.append(
            {
                "name": command_name,
                "target": target,
                "run_name": run_name,
                "metadata": metadata,
                "resume_from": args.get("resume_from"),
            }
        )

    return {
        "name": name,
        "training_only": True,
        "onpolicy_target_attestation": True,
        "onpolicy_topology_profile": profile_name,
        "onpolicy_topology": {
            "gpu_count": GPU_COUNT,
            "coordinator_device": COORDINATOR_DEVICE,
            "rollout_gpus": list(ROLLOUT_GPUS),
        },
        "rmct256_convergence": {
            "schema": CONVERGENCE_SCHEMA,
            "mode": convergence["mode"],
            "condition": name,
            "artifact_root": artifact_root,
            "figure_root": figure_root,
            "base_snapshot": dict(base_snapshot),
            "base_snapshot_commit": BASE_SNAPSHOT,
            "isambard_runtime": dict(isambard_runtime),
            "optimizer": dict(optimizer),
            "method": dict(method),
            "selection": dict(selection),
            "setting_config": dict(setting_config),
            "plateau": {
                "minimum_completed_passes": convergence["minimum_completed_passes"],
                "full_pass_min_delta": convergence["full_pass_min_delta"],
                "non_improving_pass_patience": convergence["non_improving_pass_patience"],
                "decision_boundary": "completed_full_pass_only",
                "retain": convergence["retain"],
                "cap_outcome": convergence["cap_outcome"],
            },
            "hard_cap": {
                "passes": convergence["hard_cap_passes"],
                "segments": TOTAL_SEGMENTS,
                "updates": TOTAL_UPDATES,
                "extension_policy": "new_explicit_plan_required",
            },
            "segments": segments,
        },
        "training": training,
    }


__all__ = [
    "BASE_ROLLOUT_SEED",
    "BASE_SNAPSHOT",
    "CONDITION_NAME",
    "CONVERGENCE_SCHEMA",
    "HARD_CAP_PASSES",
    "ROWS_PER_SEGMENT",
    "SEGMENTS_PER_PASS",
    "TOTAL_SEGMENTS",
    "TOTAL_UPDATES",
    "UPDATES_PER_SEGMENT",
    "compile_experiment",
]
