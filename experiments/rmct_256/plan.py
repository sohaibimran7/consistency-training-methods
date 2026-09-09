"""Compile the RMCT-256 plan with an explicitly selected worker topology.

The scientific condition is authored once, but the coordinator/worker layout
is not an ambient launcher choice.  A caller must choose one of the authored
profiles; the selected layout is then part of the compiled command, target,
run namespace, and on-policy target attestation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from scripts.rmct_paper_vast_more_methods.experiment_factory import compile_experiment as compile_mcq_bias_experiment


_PROFILE_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]*")


def _require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or not _PROFILE_NAME.fullmatch(value):
        raise ValueError(
            f"{label} must start with a letter or digit and contain only letters, digits, dots, underscores, and hyphens"
        )
    return value


def _select_profile(value: Any, *, requested: str | None) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("topology_profiles must be a non-empty object")
    profiles = dict(value)
    names = sorted(profiles)
    if requested is None:
        raise ValueError(f"RMCT-256 requires an explicit topology profile; choose one of {names}")
    if requested not in profiles:
        raise ValueError(f"unknown RMCT-256 topology profile {requested!r}; choose one of {names}")
    raw = profiles[requested]
    if not isinstance(raw, Mapping):
        raise ValueError(f"topology_profiles.{requested} must be an object")
    profile = dict(raw)
    expected = {"target", "run_name", "gpu_count", "rollout_gpus"}
    unknown = sorted(set(profile) - expected)
    missing = sorted(expected - set(profile))
    if missing or unknown:
        details = [
            *(f"missing {missing}" for _ in [None] if missing),
            *(f"unknown {unknown}" for _ in [None] if unknown),
        ]
        raise ValueError(f"topology_profiles.{requested}: {', '.join(details)}")

    profile["target"] = _require_string(profile["target"], label=f"topology_profiles.{requested}.target")
    profile["run_name"] = _require_string(profile["run_name"], label=f"topology_profiles.{requested}.run_name")
    gpu_count = profile["gpu_count"]
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 2:
        raise ValueError(f"topology_profiles.{requested}.gpu_count must be an integer of at least 2")
    rollout_gpus = profile["rollout_gpus"]
    if (
        not isinstance(rollout_gpus, list)
        or any(isinstance(gpu, bool) or not isinstance(gpu, int) for gpu in rollout_gpus)
        or rollout_gpus != list(range(1, gpu_count))
    ):
        raise ValueError(
            f"topology_profiles.{requested}.rollout_gpus must be the complete logical worker range "
            f"[1, ..., {gpu_count - 1}]"
        )
    return requested, profile


def compile_experiment(
    *,
    name: str,
    spec: Mapping[str, Any],
    topology_profile: str | None = None,
) -> dict[str, Any]:
    """Expand exactly one RMCT-256 topology profile into a normal command plan."""

    mutable_spec = dict(spec)
    if "execution" in mutable_spec:
        raise ValueError("RMCT-256 execution is selected only through topology_profiles")
    requested, profile = _select_profile(
        mutable_spec.pop("topology_profiles", None),
        requested=topology_profile,
    )
    mutable_spec["execution"] = {
        "publication_owner": "rmct256-from-base-coordinator",
        "allocations": [
            {
                "target": profile["target"],
                "stage": "training",
                "commands": ["rate_matching_lr1"],
                "gpu_count": profile["gpu_count"],
                "local_device": "cuda:0",
                "rollout_gpus": profile["rollout_gpus"],
                "gradient_checkpointing_layers": "all",
            }
        ],
    }
    compiled = compile_mcq_bias_experiment(name=name, spec=mutable_spec)
    training = compiled.get("training")
    if not isinstance(training, list) or len(training) != 1:
        raise AssertionError("RMCT-256 must compile exactly one training command")
    entry = training[0]
    if entry.get("name") != "rate_matching_lr1":
        raise AssertionError("RMCT-256 must compile the rate_matching_lr1 command")
    args = entry.get("args")
    if not isinstance(args, dict):
        raise AssertionError("RMCT-256 training command has no args object")
    args["run_name"] = profile["run_name"]
    compiled["onpolicy_topology_profile"] = requested
    compiled["onpolicy_topology"] = {
        "gpu_count": profile["gpu_count"],
        "coordinator_device": "cuda:0",
        "rollout_gpus": list(profile["rollout_gpus"]),
    }
    return compiled


__all__ = ["compile_experiment"]
