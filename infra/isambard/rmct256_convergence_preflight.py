"""Compile the isolated four-GH200 preflight form of RMCT256 segment zero.

The production factory deliberately permits only its one scientific condition
and one set of run names.  A real worker transport probe therefore cannot use
that plan directly without leaving immutable preflight evidence inside the
production segment-0 namespace.  This small factory derives a separate,
attestable *preflight* plan from the production plan.  It changes only the
experiment/run namespace; the selected target and every training argument
remain the compiled segment-0 values.

It is invoked only through the immutable YAML minted by
``rmct256_convergence_segment_contract.py preflight-plan``.  The factory
re-hashes and recompiles the production YAML, so a stale or hand-edited
preflight YAML fails before any worker model starts.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infra.isambard import rmct256_convergence_segment_contract as contract


PREFLIGHT_SCHEMA = contract.PREFLIGHT_SCHEMA
PREFLIGHT_EXPERIMENT = contract.PREFLIGHT_EXPERIMENT
PREFLIGHT_RUN_NAME = contract.PREFLIGHT_RUN_NAME


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _spec_value(spec: Mapping[str, Any], key: str) -> Any:
    if key not in spec:
        raise ValueError(f"RMCT256 preflight spec is missing {key}")
    return spec[key]


def compile_preflight_experiment(
    *,
    name: str,
    spec: Mapping[str, Any],
    topology_profile: str | None = None,
) -> dict[str, Any]:
    """Return the segment-0 command under its non-production namespace."""

    if not isinstance(spec, Mapping):
        raise ValueError("RMCT256 preflight spec must be an object")
    expected_keys = {
        "schema",
        "production_plan",
        "production_plan_identity",
        "target",
        "preflight_run_name",
        "production_entry_sha256",
    }
    if set(spec) != expected_keys:
        raise ValueError(f"RMCT256 preflight spec keys must be exactly {sorted(expected_keys)}")
    if name != PREFLIGHT_EXPERIMENT:
        raise ValueError(f"RMCT256 preflight experiment name must be {PREFLIGHT_EXPERIMENT!r}")
    if topology_profile != contract.TOPOLOGY_PROFILE:
        raise ValueError(f"RMCT256 preflight requires topology profile {contract.TOPOLOGY_PROFILE!r}")
    if _spec_value(spec, "schema") != PREFLIGHT_SCHEMA:
        raise ValueError("RMCT256 preflight spec has an unexpected schema")
    if _spec_value(spec, "target") != contract.segment_for_index(0).target:
        raise ValueError("RMCT256 preflight may attest only the static segment-0 target")
    if _spec_value(spec, "preflight_run_name") != PREFLIGHT_RUN_NAME:
        raise ValueError("RMCT256 preflight has an unexpected isolated run namespace")

    production_plan_raw = _spec_value(spec, "production_plan")
    if not isinstance(production_plan_raw, str) or not production_plan_raw:
        raise ValueError("RMCT256 preflight production_plan must be a non-empty path")
    production_plan = Path(production_plan_raw).resolve()
    recorded_identity = _spec_value(spec, "production_plan_identity")
    if not isinstance(recorded_identity, Mapping):
        raise ValueError("RMCT256 preflight has no production plan identity")
    current_identity = contract.file_identity(production_plan, label="RMCT256 production plan for preflight")
    if current_identity != dict(recorded_identity):
        raise ValueError("RMCT256 production plan changed after the preflight plan was minted")

    # Import lazily so this factory remains importable by isolated tests before
    # a caller has established the repository execution context.
    from scripts import run_experiment

    compiled = run_experiment.load_experiment(production_plan, topology_profile=topology_profile)
    if compiled.get("name") != contract.CONDITION:
        raise ValueError("RMCT256 preflight source plan is not the protected production condition")
    expected_topology = {
        "gpu_count": 4,
        "coordinator_device": "cuda:0",
        "rollout_gpus": [1, 2, 3],
    }
    if compiled.get("onpolicy_topology") != expected_topology:
        raise ValueError("RMCT256 preflight source plan does not retain the four-GH200 topology")
    training = compiled.get("training")
    if not isinstance(training, list):
        raise ValueError("RMCT256 preflight source plan has no training list")
    selected = [entry for entry in training if isinstance(entry, Mapping) and entry.get("target") == spec["target"]]
    if len(selected) != 1:
        raise ValueError("RMCT256 preflight source plan must contain exactly one segment-0 target")
    entry = copy.deepcopy(dict(selected[0]))
    if _spec_value(spec, "production_entry_sha256") != _canonical_sha256(selected[0]):
        raise ValueError("RMCT256 preflight production target changed after the preflight plan was minted")
    args = entry.get("args")
    if not isinstance(args, dict):
        raise ValueError("RMCT256 preflight segment-0 target has no argument object")
    expected_run = contract.segment_for_index(0).run_name
    if args.get("run_name") != expected_run:
        raise ValueError("RMCT256 preflight source target has an unexpected production run name")
    # This is the sole semantic difference from the production command.  The
    # compiled child still uses the same target, selected data, optimizer,
    # model, worker options, and target-attestation requirement.
    args["run_name"] = PREFLIGHT_RUN_NAME
    entry["args"] = args

    return {
        "name": name,
        "training_only": True,
        "onpolicy_target_attestation": True,
        "onpolicy_topology_profile": contract.TOPOLOGY_PROFILE,
        "onpolicy_topology": expected_topology,
        "rmct256_preflight": {
            "schema": PREFLIGHT_SCHEMA,
            "production_plan": current_identity,
            "production_target": contract.segment_for_index(0).target,
            "production_run_name": expected_run,
            "preflight_run_name": PREFLIGHT_RUN_NAME,
            "production_entry_sha256": _canonical_sha256(selected[0]),
        },
        "training": [entry],
    }


__all__ = ["PREFLIGHT_EXPERIMENT", "PREFLIGHT_RUN_NAME", "PREFLIGHT_SCHEMA", "compile_preflight_experiment"]
