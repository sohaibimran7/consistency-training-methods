"""Shared execution-topology metadata for local training entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ctm.backends.cli import PhaseSharedCLIConfig


def phase_shared_run_metadata(phase_shared: PhaseSharedCLIConfig | None) -> dict[str, object]:
    """Return reproducible runtime provenance for an opt-in shared-GPU run.

    The placement is deliberately recorded separately from scientific settings:
    phase sharing changes how the configured update is executed, rather than
    which data, objective, rollout budgets, or optimizer hyperparameters the
    experiment uses.
    """

    if phase_shared is None:
        return {}
    topology = phase_shared.topology
    return {
        "phase_shared": {
            "schema_version": "local_phase_shared_v1",
            "execution_only": True,
            "execution_semantics": (
                "runtime topology only; configured objective, data selection, rollout budgets, "
                "and optimizer hyperparameters are unchanged"
            ),
            "visible_devices": list(topology.visible_devices),
            "training_world_size": topology.world_size,
            "training_ranks": [
                {
                    "rank": rank.rank,
                    "logical_index": rank.gpu.logical_index,
                    "device_token": rank.gpu.device_token,
                    "publisher": rank.is_publisher,
                }
                for rank in topology.training_ranks
            ],
            "rollout_workers": [
                {
                    "worker_id": worker_id,
                    "logical_index": gpu.logical_index,
                    "device_token": gpu.device_token,
                }
                for worker_id, gpu in enumerate(topology.rollout_gpus)
            ],
            "overlap": [
                {"logical_index": gpu.logical_index, "device_token": gpu.device_token} for gpu in topology.overlap
            ],
            "coordinator": {
                "rank": 0,
                "logical_index": topology.coordinator.logical_index,
                "device_token": topology.coordinator.device_token,
                "device": phase_shared.coordinator_device,
                "canonical_adapter_publisher": True,
            },
            "vllm_sleep_lifecycle": {
                "enabled": True,
                "sleep_level": 1,
                "rollout_phase": "workers awake; sampling and scoring permitted",
                "training_phase": "workers sleep before replicated trainer work; sampling and scoring prohibited",
                "publication": "rank 0 verifies replica state, publishes the adapter, then all workers acknowledge before rollout resumes",
                "transition_failure_policy": "fail_closed",
            },
            "rollout_worker_timeouts_seconds": {
                "startup": phase_shared.rollout.start_timeout_seconds,
                "request": phase_shared.rollout.request_timeout_seconds,
            },
            "replica_timeouts_seconds": {
                "startup": phase_shared.replica_start_timeout_seconds,
                "command": phase_shared.replica_command_timeout_seconds,
                "shutdown": phase_shared.replica_shutdown_timeout_seconds,
            },
        }
    }
