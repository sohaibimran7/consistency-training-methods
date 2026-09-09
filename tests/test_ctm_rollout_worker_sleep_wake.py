"""CPU-only lifecycle contracts for phase-shared rollout workers.

These tests deliberately exercise the pool protocol through fake endpoints,
rather than requiring CUDA or vLLM.  A phase-shared deployment may use two,
four, or eight allocated GPUs; the worker count is therefore parametrized
instead of encoding a particular host layout.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ctm.backends.base import SampledSequence
from ctm.backends.local.rollout_workers import (
    RolloutGPU,
    RolloutParallelBackend,
    RolloutWorkerError,
    RolloutWorkerPool,
    _write_payload,
)


class _LifecycleEndpoint:
    """In-memory endpoint implementing the worker lifecycle protocol."""

    def __init__(
        self,
        worker_id: int,
        gpu: RolloutGPU,
        *,
        sleep_enabled: bool,
        fail_sleep: bool = False,
    ):
        self.worker_id = worker_id
        self.gpu = gpu
        self.sleep_enabled = sleep_enabled
        self.fail_sleep = fail_sleep
        self.adapter_version = 0
        self.sleeping = False
        self.launched = False
        self.closed = False
        self.commands: list[dict] = []
        self.ipc_dir: Path | None = None

    def launch_sync(self) -> None:
        self.launched = True

    def wait_ready_sync(self) -> dict:
        return {
            "ok": True,
            "op": "ready",
            "worker_id": self.worker_id,
            "adapter_version": self.adapter_version,
            "sleep_enabled": self.sleep_enabled,
            "sleeping": self.sleeping,
        }

    def call_sync(self, command: dict, **_kwargs) -> dict:
        self.commands.append(command)
        op = command["op"]
        response = {
            "ok": True,
            "op": op,
            "request_id": command["request_id"],
            "worker_id": self.worker_id,
            "adapter_version": self.adapter_version,
            "sleep_enabled": self.sleep_enabled,
            "sleeping": self.sleeping,
        }
        if op == "health":
            return response
        if op == "sleep":
            if self.fail_sleep:
                raise RolloutWorkerError(f"fake worker {self.worker_id} refuses sleep")
            self.sleeping = True
            response["sleeping"] = True
            return response
        if op == "wake":
            self.sleeping = False
            response["sleeping"] = False
            return response
        if op == "load_adapter":
            self.adapter_version = int(command["adapter_version"])
            response["adapter_version"] = self.adapter_version
            return response
        if op == "sample_batch":
            assert not self.sleeping
            assert self.ipc_dir is not None
            records = [
                (
                    assignment["prompt_index"],
                    sample_index,
                    SampledSequence(tokens=[assignment["prompt_index"], sample_index], logprobs=[-0.5, -0.25]),
                )
                for assignment in command["assignments"]
                for sample_index in assignment["sample_indices"]
            ]
            response["result"] = _write_payload(
                records,
                ipc_dir=self.ipc_dir,
                worker_id=self.worker_id,
                request_id=command["request_id"],
            )
            response["adapter_version"] = self.adapter_version
            return response
        if op == "score_batch":
            assert not self.sleeping
            assert self.ipc_dir is not None
            records = [
                (
                    assignment["score_index"],
                    0,
                    SampledSequence(
                        tokens=list(assignment["completion_tokens"]),
                        logprobs=[-0.5] * len(assignment["completion_tokens"]),
                    ),
                )
                for assignment in command["assignments"]
            ]
            response["result"] = _write_payload(
                records,
                ipc_dir=self.ipc_dir,
                worker_id=self.worker_id,
                request_id=command["request_id"],
            )
            response["adapter_version"] = self.adapter_version
            return response
        raise AssertionError(op)

    async def call(self, command: dict) -> dict:
        return self.call_sync(command)

    def close_sync(self) -> None:
        self.closed = True


def _pool(
    tmp_path: Path,
    *,
    worker_count: int,
    sleep_enabled: bool,
    fail_sleep_worker: int | None = None,
) -> tuple[RolloutWorkerPool, list[_LifecycleEndpoint]]:
    gpus = tuple(RolloutGPU(index + 1, f"GPU-{index + 1}") for index in range(worker_count))
    endpoints = [
        _LifecycleEndpoint(
            index,
            gpu,
            sleep_enabled=sleep_enabled,
            fail_sleep=index == fail_sleep_worker,
        )
        for index, gpu in enumerate(gpus)
    ]
    engine_kwargs = {"seed": 11}
    if sleep_enabled:
        engine_kwargs["enable_sleep_mode"] = True
    pool = RolloutWorkerPool(
        model="fake/model",
        gpus=gpus,
        engine_kwargs=engine_kwargs,
        status_dir=tmp_path,
        endpoints=endpoints,
    )
    for endpoint in endpoints:
        endpoint.ipc_dir = pool.ipc_dir
    return pool, endpoints


@pytest.mark.parametrize("worker_count", [1, 3, 7], ids=["two-gpu", "four-gpu", "eight-gpu"])
def test_sleep_wake_barriers_are_idempotent_for_arbitrary_worker_counts(tmp_path, worker_count):
    """One coordinator plus N workers covers 2-, 4-, and 8-GPU allocations."""

    pool, endpoints = _pool(tmp_path, worker_count=worker_count, sleep_enabled=True)
    pool.start()

    asyncio.run(pool.sleep())
    asyncio.run(pool.sleep())
    assert pool.sleeping is True
    assert pool.phase == "training"
    assert pool.sampling_training_overlap_supported is False
    assert all(endpoint.sleeping for endpoint in endpoints)
    assert all([command["op"] for command in endpoint.commands].count("sleep") == 1 for endpoint in endpoints)

    asyncio.run(pool.wake())
    asyncio.run(pool.wake())
    assert pool.sleeping is False
    assert pool.phase == "rollout"
    assert not any(endpoint.sleeping for endpoint in endpoints)
    assert all([command["op"] for command in endpoint.commands].count("wake") == 1 for endpoint in endpoints)
    pool.shutdown()


def test_sleep_disabled_is_a_true_noop(tmp_path):
    pool, endpoints = _pool(tmp_path, worker_count=3, sleep_enabled=False)
    pool.start()

    asyncio.run(pool.sleep())
    asyncio.run(pool.wake())

    assert pool.sleeping is False
    assert pool.sampling_training_overlap_supported is True
    assert all(command["op"] not in {"sleep", "wake"} for endpoint in endpoints for command in endpoint.commands)
    pool.shutdown()


def test_partial_sleep_failure_is_fatal_before_training_can_start(tmp_path):
    pool, endpoints = _pool(tmp_path, worker_count=3, sleep_enabled=True, fail_sleep_worker=1)
    pool.start()

    with pytest.raises(RolloutWorkerError, match="refuses sleep"):
        asyncio.run(pool.sleep())

    assert pool.sleeping is False
    # Once a lifecycle barrier is ambiguous, the pool must not be reused for
    # either rollout or adapter publication.
    with pytest.raises(RolloutWorkerError, match="unhealthy"):
        asyncio.run(pool.wake())
    assert any(command["op"] == "sleep" for command in endpoints[0].commands)
    assert any(command["op"] == "sleep" for command in endpoints[1].commands)
    pool.shutdown()


def test_policy_publication_wakes_every_worker_before_the_version_barrier(tmp_path):
    pool, endpoints = _pool(tmp_path, worker_count=3, sleep_enabled=True)
    pool.start()
    asyncio.run(pool.sleep())

    asyncio.run(pool.publish_adapter(tmp_path, version=1))

    assert pool.sleeping is False
    for endpoint in endpoints:
        operations = [command["op"] for command in endpoint.commands]
        assert operations[-3:] == ["sleep", "wake", "load_adapter"]
        assert endpoint.adapter_version == 1
    pool.shutdown()


@pytest.mark.parametrize("operation", ["sample", "score"])
def test_rollout_operations_fail_closed_until_the_explicit_rollout_phase_barrier(tmp_path, operation):
    pool, endpoints = _pool(tmp_path, worker_count=3, sleep_enabled=True)
    pool.start()
    pool.publish_adapter_sync(tmp_path, version=1)
    asyncio.run(pool.sleep())

    if operation == "sample":
        with pytest.raises(RolloutWorkerError, match="training phase"):
            asyncio.run(
                pool.sample_batch(
                    [[1]],
                    max_tokens=8,
                    temperature=0.7,
                    stop=[],
                    num_samples=1,
                    use_base=False,
                )
            )
        asyncio.run(pool.wake())
        batches = asyncio.run(
            pool.sample_batch(
                [[1], [2]],
                max_tokens=8,
                temperature=0.7,
                stop=[],
                num_samples=2,
                use_base=False,
            )
        )
        assert [[sequence.tokens for sequence in group] for group in batches] == [
            [[0, 0], [0, 1]],
            [[1, 0], [1, 1]],
        ]
        request_op = "sample_batch"
    else:
        with pytest.raises(RolloutWorkerError, match="training phase"):
            asyncio.run(pool.score_completions([[1]], [[4]], use_base=False))
        asyncio.run(pool.wake())
        scores = asyncio.run(pool.score_completions([[1], [2], [3]], [[4], [5], [6]], use_base=False))
        assert scores == [[-0.5], [-0.5], [-0.5]]
        request_op = "score_batch"

    assert pool.sleeping is False
    for endpoint in endpoints:
        operations = [command["op"] for command in endpoint.commands]
        assert operations.index("sleep") < operations.index("wake") < operations.index(request_op)
    pool.shutdown()


@pytest.mark.parametrize("operation", ["sample", "score"])
def test_training_phase_barrier_waits_for_an_inflight_rollout_operation(tmp_path, operation):
    """Sleeping must never overtake a pipe request that still owns worker VRAM."""

    pool, endpoints = _pool(tmp_path, worker_count=1, sleep_enabled=True)
    endpoint = endpoints[0]
    pool.start()
    pool.publish_adapter_sync(tmp_path, version=1)

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        original_call = endpoint.call

        async def blocking_call(command):
            if command["op"] == ("sample_batch" if operation == "sample" else "score_batch"):
                started.set()
                await release.wait()
            return await original_call(command)

        endpoint.call = blocking_call
        if operation == "sample":
            request = asyncio.create_task(
                pool.sample_batch(
                    [[1]],
                    max_tokens=8,
                    temperature=0.7,
                    stop=[],
                    num_samples=1,
                    use_base=False,
                )
            )
        else:
            request = asyncio.create_task(pool.score_completions([[1]], [[4]], use_base=False))
        await started.wait()
        transition = asyncio.create_task(pool.enter_training_phase())
        await asyncio.sleep(0)
        assert not transition.done()
        assert not any(command["op"] == "sleep" for command in endpoint.commands)
        release.set()
        await request
        await transition

    asyncio.run(scenario())
    operations = [command["op"] for command in endpoint.commands]
    request_op = "sample_batch" if operation == "sample" else "score_batch"
    assert operations.index(request_op) < operations.index("sleep")
    assert pool.phase == "training"
    assert pool.sleeping is True
    pool.shutdown()


class _PhaseTransitionPool:
    """Minimal phase-aware worker pool for backend ordering contracts."""

    def __init__(self, events: list[str], *, phase: str = "training"):
        self.events = events
        self.phase = phase
        self.sleep_enabled = True
        self._training_phase_active = phase == "training"
        self.sleeping = self._training_phase_active
        self.published: list[tuple[Path, int]] = []

    async def enter_training_phase(self) -> None:
        self.events.append("pool.enter_training_phase")
        self.phase = "training"
        self._training_phase_active = True
        self.sleeping = True

    async def enter_rollout_phase(self) -> None:
        self.events.append("pool.enter_rollout_phase")
        self.phase = "rollout"
        self._training_phase_active = False
        self.sleeping = False

    async def publish_adapter(self, adapter_path: Path, *, version: int) -> None:
        self.events.append("pool.publish_adapter")
        self.published.append((Path(adapter_path), version))


class _SnapshotModel:
    def __init__(self, events: list[str]):
        self.events = events

    def save_pretrained(self, path: str) -> None:
        self.events.append("trainer.snapshot_adapter")
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "adapter_model.safetensors").write_bytes(b"adapter")


class _PhaseTransitionTrainer:
    def __init__(self, events: list[str]):
        self.events = events
        self.model = _SnapshotModel(events)

    async def enter_rollout_phase(self) -> None:
        self.events.append("trainer.enter_rollout_phase")


def _phase_transition_backend(tmp_path: Path, events: list[str], *, phase: str = "training"):
    pool = _PhaseTransitionPool(events, phase=phase)
    backend = RolloutParallelBackend(
        _PhaseTransitionTrainer(events),
        gpus=(),
        status_dir=tmp_path / "status",
        pool=pool,
    )
    backend._model_name = "fake/model"
    backend._adapter_root = tmp_path / "adapters"
    backend._adapter_root.mkdir(parents=True)
    return backend, pool


def test_backend_releases_trainer_memory_before_worker_wake_only_on_training_transition(tmp_path):
    events: list[str] = []
    backend, _pool = _phase_transition_backend(tmp_path, events)

    asyncio.run(backend.enter_rollout_phase())

    assert events == ["trainer.enter_rollout_phase", "pool.enter_rollout_phase"]

    # The pool wake is itself idempotent, but repeating the request while
    # workers are already in rollout must not redundantly disrupt the trainer.
    asyncio.run(backend.enter_rollout_phase())
    assert events == [
        "trainer.enter_rollout_phase",
        "pool.enter_rollout_phase",
        "pool.enter_rollout_phase",
    ]


def test_refresh_releases_trainer_memory_before_snapshot_and_worker_publication(tmp_path):
    events: list[str] = []
    backend, pool = _phase_transition_backend(tmp_path, events)

    asyncio.run(backend.refresh_policy_sampler("after-update"))

    assert events.index("trainer.enter_rollout_phase") < events.index("trainer.snapshot_adapter")
    assert events.index("trainer.snapshot_adapter") < events.index("pool.publish_adapter")
    assert pool.published[0][1] == 1
