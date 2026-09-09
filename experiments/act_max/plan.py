"""Compile the profile-bound, native-prompt ACT-Max training plan.

The sole profile is intentionally explicit even though ACT itself is a
single-GPU paired-forward method.  This prevents a launcher from silently
changing the logical resource shape or run namespace while leaving the
scientific condition's name unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]*")
_REQUIRED_SPEC_KEYS = {
    "topology_profiles",
    "model",
    "artifact_root",
    "recovered_none_source",
    "recovered_none_manifest",
    "canonical_prefix",
    "canonical_prefix_manifest",
    "iid_reference_manifest",
    "iid_heldout",
    "stage2_manifest",
    "selection",
    "selection_manifest",
}


def _require_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or not _NAME.fullmatch(value):
        raise ValueError(f"{label} must be a non-empty safe execution name")
    return value


def _require_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.startswith("/"):
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    return value


def _profile(spec: Mapping[str, Any], requested: str | None) -> tuple[str, dict[str, Any]]:
    profiles = spec.get("topology_profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("ACT-Max topology_profiles must be a non-empty object")
    names = sorted(profiles)
    if requested is None:
        raise ValueError(f"ACT-Max requires an explicit topology profile; choose one of {names}")
    raw = profiles.get(requested)
    if not isinstance(raw, Mapping):
        raise ValueError(f"unknown ACT-Max topology profile {requested!r}; choose one of {names}")
    value = dict(raw)
    expected = {"target", "run_name", "gpu_count"}
    if set(value) != expected:
        raise ValueError(f"ACT-Max topology_profiles.{requested} must contain exactly {sorted(expected)}")
    value["target"] = _require_name(value["target"], label=f"ACT-Max profile {requested} target")
    value["run_name"] = _require_name(value["run_name"], label=f"ACT-Max profile {requested} run_name")
    if value["gpu_count"] != 1:
        raise ValueError(f"ACT-Max profile {requested} must allocate exactly one GPU")
    return requested, value


def compile_experiment(
    *,
    name: str,
    spec: Mapping[str, Any],
    topology_profile: str | None = None,
) -> dict[str, Any]:
    """Expand the one selected ACT-Max resource profile into normal stages."""

    unknown = sorted(set(spec) - _REQUIRED_SPEC_KEYS)
    missing = sorted(_REQUIRED_SPEC_KEYS - set(spec))
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ValueError(f"ACT-Max spec has {', '.join(details)}")
    requested, profile = _profile(spec, topology_profile)
    model = spec["model"]
    if model != "Qwen/Qwen3.5-9B":
        raise ValueError("ACT-Max is pinned to Qwen/Qwen3.5-9B")

    paths = {key: _require_path(spec[key], label=f"ACT-Max spec.{key}") for key in _REQUIRED_SPEC_KEYS - {"topology_profiles", "model", "artifact_root"}}
    artifact_root = _require_path(spec["artifact_root"], label="ACT-Max spec.artifact_root")
    variables = {"model": model, "artifact_root": artifact_root, **paths}
    target = profile["target"]

    return {
        "name": name,
        "variables": variables,
        "supervised_topology_profile": requested,
        "supervised_topology": {"gpu_count": 1, "device": "cuda:0"},
        # The verifier is first in the selected target's stage sequence. It
        # fails before model initialisation if an operator stages a stale data
        # selection, an altered IID reservation, or a different Stage-2 suite.
        "data_preparation": [
            {
                "name": "verify_act_max_selection",
                "target": target,
                "resource": "cpu",
                "command": ["${python}", "-m", "experiments.act_max.selection", "verify"],
                "args": {
                    "recovered_none_source": "${recovered_none_source}",
                    "recovered_none_manifest": "${recovered_none_manifest}",
                    "canonical_prefix": "${canonical_prefix}",
                    "canonical_prefix_manifest": "${canonical_prefix_manifest}",
                    "iid_reference_manifest": "${iid_reference_manifest}",
                    "iid_heldout": "${iid_heldout}",
                    "stage2_manifest": "${stage2_manifest}",
                    "selection": "${selection}",
                    "selection_manifest": "${selection_manifest}",
                },
            }
        ],
        "training": [
            {
                "name": "act_max_none",
                "target": target,
                "resource": "gpu",
                "gpu_count": 1,
                "command": ["${python}", "scripts/train_bct.py"],
                "args": {
                    "backend": "local",
                    "local_dtype": "bfloat16",
                    "local_sampler": "hf",
                    "local_gradient_checkpointing": True,
                    "local_forward_microbatch_max_datums": 1,
                    "local_forward_microbatch_max_tokens": 20480,
                    "model": "${model}",
                    "method": "act",
                    "data": ["${selection}:2800"],
                    "data_manifest": ["${selection_manifest}"],
                    "reference_messages_field": "unbiased_messages",
                    "variant_messages_field": "biased_messages",
                    "require_full_reference_suffix_alignment": True,
                    "qwen35_consistency_preflight": True,
                    "method_config": {"weight": 0.00005, "layer_selection": "all", "normalize": False},
                    "lora_config": {
                        "rank": 8,
                        "alpha": 16,
                        "dropout": 0.05,
                        "target_modules": [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "in_proj_qkv",
                            "in_proj_z",
                            "in_proj_b",
                            "in_proj_a",
                            "out_proj",
                        ],
                        "train_mlp": False,
                        "train_attn": False,
                        "train_unembed": False,
                        "seed": 42,
                    },
                    "optimizer_config": {
                        "learning_rate": 0.0001,
                        "lr_schedule": "constant",
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "eps": 0.00000001,
                        "weight_decay": 0.01,
                        "grad_clip_norm": 1.0,
                    },
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    # A twenty-pass replay of the old n=200 repair would make
                    # the data scaling result a 14x optimisation-budget change.
                    # Two full passes retain the high-data supervised-method
                    # convention and expose every retained row twice.
                    "epochs": 2,
                    "minimum_optimizer_steps": 5600,
                    "save_every": 700,
                    "save_state": True,
                    "experiment_name": "${experiment}",
                    "run_name": profile["run_name"],
                    "wandb_project": "rmct_paper_vast_dense_models_act_max_20260804",
                    "yes": True,
                },
            }
        ],
    }


__all__ = ["compile_experiment"]
