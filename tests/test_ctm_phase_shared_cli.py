"""CPU-only contracts for opt-in phase-shared CLI composition.

These tests mock constructors deliberately: building a topology must not probe
CUDA, load a model, start vLLM, or start child processes.  They cover small and
large allocations and a nonzero/non-contiguous rank-zero placement so the CLI
cannot accidentally grow a GPU-0 or four-GPU assumption.
"""

from __future__ import annotations

import argparse
import pickle
from typing import Any

import pytest

from ctm.backends.cli import (
    add_backend_args,
    build_backend,
    build_base_generation_backend,
    describe_backend,
    resolve_phase_shared_args,
    resolve_rollout_parallel_args,
)
from ctm.backends.local.replicated import LocalBackendConstructorSpec


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    return parser.parse_args(argv)


class _FakeLocalBackend:
    """Pickleable LocalBackend stand-in used to inspect the constructor recipe."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.device = kwargs["device"]


class _FakeReplicatedTrainingBackend:
    def __init__(self, training_backend: Any, **kwargs: Any) -> None:
        self.training_backend = training_backend
        self.kwargs = kwargs


class _FakeRolloutParallelBackend:
    def __init__(self, training_backend: Any, **kwargs: Any) -> None:
        self.training_backend = training_backend
        self.kwargs = kwargs


def _patch_constructors(monkeypatch: pytest.MonkeyPatch) -> None:
    from ctm.backends.local import engine, replicated, rollout_workers

    monkeypatch.setattr(engine, "LocalBackend", _FakeLocalBackend)
    monkeypatch.setattr(replicated, "ReplicatedTrainingBackend", _FakeReplicatedTrainingBackend)
    monkeypatch.setattr(rollout_workers, "RolloutParallelBackend", _FakeRolloutParallelBackend)


@pytest.mark.parametrize("gpu_count", [2, 4, 8])
def test_phase_shared_defaults_to_every_visible_gpu_and_composes_all_three_layers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    gpu_count: int,
):
    _patch_constructors(monkeypatch)
    visible = ",".join(f"GPU-{index}" for index in range(gpu_count))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    args = _parse(
        [
            "--backend",
            "local",
            "--local-phase-shared",
            "--local-rollout-status-dir",
            str(tmp_path / f"status-{gpu_count}"),
        ]
    )

    backend = build_backend(args)

    assert isinstance(backend, _FakeRolloutParallelBackend)
    replicated = backend.training_backend
    assert isinstance(replicated, _FakeReplicatedTrainingBackend)
    rank_zero = replicated.training_backend
    assert isinstance(rank_zero, _FakeLocalBackend)
    assert rank_zero.device == "cuda:0"
    assert tuple(gpu.logical_index for gpu in replicated.kwargs["topology"].train_gpus) == tuple(range(gpu_count))
    assert [(gpu.logical_index, gpu.device_token) for gpu in backend.kwargs["gpus"]] == [
        (index, f"GPU-{index}") for index in range(gpu_count)
    ]
    assert rank_zero.kwargs["vllm_options"]["enable_sleep_mode"] is True
    assert backend.kwargs["worker_vllm_options"]["enable_sleep_mode"] is True

    child_spec = replicated.kwargs["child_backend_spec"]
    assert isinstance(child_spec, LocalBackendConstructorSpec)
    assert child_spec.constructor is _FakeLocalBackend
    child_spec.validate()
    assert dict(child_spec.kwargs) == {key: value for key, value in rank_zero.kwargs.items() if key != "device"}
    assert "gradient_reducer" not in child_spec.kwargs
    # Prove this is a portable child recipe, rather than a closure or a live
    # CUDA/model object held by rank zero.
    pickle.dumps(child_spec)


def test_phase_shared_uses_first_training_gpu_as_rank_zero_and_allows_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    _patch_constructors(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    args = _parse(
        [
            "--backend",
            "local",
            "--local-phase-shared",
            "--local-training-gpus",
            "3,1",
            "--local-rollout-gpus",
            "1,3",
            "--local-rollout-status-dir",
            str(tmp_path / "status"),
            "--local-rollout-start-timeout-seconds",
            "11",
            "--local-rollout-request-timeout-seconds",
            "22",
            "--local-replica-start-timeout-seconds",
            "33",
            "--local-replica-command-timeout-seconds",
            "44",
            "--local-replica-shutdown-timeout-seconds",
            "55",
        ]
    )

    resolved = resolve_phase_shared_args(args)
    assert resolved is not None
    assert resolved.coordinator_device == "cuda:3"
    assert args.local_device == "cuda:3"
    assert [gpu.logical_index for gpu in resolved.topology.train_gpus] == [3, 1]
    assert [gpu.logical_index for gpu in resolved.topology.rollout_gpus] == [1, 3]
    assert [gpu.logical_index for gpu in resolved.topology.overlap] == [3, 1]
    assert resolve_rollout_parallel_args(args) == resolved.rollout

    backend = build_backend(args)
    replicated = backend.training_backend
    assert replicated.training_backend.device == "cuda:3"
    assert replicated.kwargs["start_timeout_seconds"] == 33.0
    assert replicated.kwargs["command_timeout_seconds"] == 44.0
    assert replicated.kwargs["shutdown_timeout_seconds"] == 55.0
    assert [(gpu.logical_index, gpu.device_token) for gpu in backend.kwargs["gpus"]] == [
        (1, "GPU-b"),
        (3, "GPU-d"),
    ]
    rendered = describe_backend(args)
    assert "rank0=cuda:3" in rendered
    assert "train=[3,1]" in rendered
    assert "rollout=[1,3]" in rendered
    assert "overlap=[3,1]" in rendered
    assert "rollout_timeouts=11/22s" in rendered
    assert "replica_timeouts=33/44/55s" in rendered
    assert "vllm_sleep=training_phase" in rendered


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--local-phase-shared"], "requires --backend local"),
        (["--backend", "local", "--local-phase-shared", "--local-sampler", "hf"], "requires --local-sampler vllm"),
        (["--backend", "local", "--local-phase-shared", "--local-full-finetune"], "requires LoRA"),
        (["--backend", "local", "--local-phase-shared", "--local-device-map", "auto"], "incompatible with --local-device-map"),
        (["--backend", "local", "--local-training-gpus", "0,1"], "requires --local-phase-shared"),
    ],
)
def test_phase_shared_rejects_incompatible_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    argv: list[str],
    message: str,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    args = _parse([*argv, "--local-rollout-status-dir", str(tmp_path / "status")])
    with pytest.raises(ValueError, match=message):
        build_backend(args)


def test_phase_shared_requires_explicit_visible_allocation(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = _parse(
        [
            "--backend",
            "local",
            "--local-phase-shared",
            "--local-rollout-status-dir",
            str(tmp_path / "status"),
        ]
    )
    with pytest.raises(ValueError, match="explicit non-empty CUDA_VISIBLE_DEVICES"):
        build_backend(args)


def test_phase_shared_rejects_a_legacy_coordinator_device_that_does_not_match_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")
    args = _parse(
        [
            "--backend",
            "local",
            "--local-phase-shared",
            "--local-training-gpus",
            "2,0",
            "--local-device",
            "cuda:0",
            "--local-rollout-status-dir",
            str(tmp_path / "status"),
        ]
    )
    with pytest.raises(ValueError, match="first --local-training-gpus entry"):
        build_backend(args)


def test_phase_shared_cannot_be_accidentally_used_for_frozen_base_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    args = _parse(
        [
            "--backend",
            "local",
            "--local-phase-shared",
            "--local-rollout-status-dir",
            str(tmp_path / "status"),
        ]
    )
    with pytest.raises(ValueError, match="training topology"):
        build_base_generation_backend(args, model="unit/model", default_status_dir=tmp_path / "base")
