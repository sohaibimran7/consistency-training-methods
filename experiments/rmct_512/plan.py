"""Compile the protected RMCT-512 continuation plan.

The plan deliberately contains an unresolved ``${checkpoint}`` placeholder.
Only the continuation launcher supplies it after sealing an immutable parent
checkpoint contract; a bare generic runner invocation therefore fails before
any model initialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from experiments.rmct_256.plan import _select_profile
from scripts.rmct_paper_vast_more_methods.experiment_factory import compile_experiment as compile_mcq_bias_experiment


def compile_experiment(
    *,
    name: str,
    spec: Mapping[str, Any],
    topology_profile: str | None = None,
) -> dict[str, Any]:
    """Expand one explicit coordinator/worker profile for RMCT-512."""

    mutable_spec = dict(spec)
    if "execution" in mutable_spec:
        raise ValueError("RMCT-512 execution is selected only through topology_profiles")
    requested, profile = _select_profile(
        mutable_spec.pop("topology_profiles", None),
        requested=topology_profile,
    )
    mutable_spec["execution"] = {
        "publication_owner": "rmct512-continuation-coordinator",
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
        raise AssertionError("RMCT-512 must compile exactly one continuation command")
    entry = training[0]
    if entry.get("name") != "rate_matching_lr1":
        raise AssertionError("RMCT-512 must compile the rate_matching_lr1 command")
    args = entry.get("args")
    if not isinstance(args, dict):
        raise AssertionError("RMCT-512 continuation command has no args object")
    args["run_name"] = profile["run_name"]
    compiled["onpolicy_topology_profile"] = requested
    compiled["onpolicy_topology"] = {
        "gpu_count": profile["gpu_count"],
        "coordinator_device": "cuda:0",
        "rollout_gpus": list(profile["rollout_gpus"]),
    }
    return compiled


__all__ = ["compile_experiment"]

