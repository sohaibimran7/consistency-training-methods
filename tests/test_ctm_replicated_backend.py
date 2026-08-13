"""CPU contracts for the exact replicated LocalBackend coordinator.

The process test intentionally uses Gloo and a tiny pickleable fake backend.
It exercises the same persistent rank-0/peer command protocol used by NCCL,
without needing a GPU or a model download in unit tests.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

import ctm.backends.local.replicated as replicated_module
from ctm.backends.base import ForwardBackwardOutput
from ctm.backends.local.phase_shared import resolve_phase_shared_topology
from ctm.backends.local.replicated import (
    LocalBackendConstructorSpec,
    ReplicatedBackendError,
    ReplicatedBackendPoisonedError,
    ReplicatedTrainingBackend,
    ReplicatedWorkerResult,
    aggregate_sharded_metrics,
    datum_token_cost,
    trainable_state_hash,
)
from ctm.backends.local.rollout_workers import RolloutParallelBackend
from ctm.core.config import AdamConfig, LoRAConfig


class _Pending:
    def __init__(self, value: Any) -> None:
        self.value = value

    async def result(self) -> Any:
        return self.value


@dataclass(frozen=True)
class _ModelInput:
    tokens: tuple[int, ...]

    def to_ints(self) -> list[int]:
        return list(self.tokens)


@dataclass(frozen=True)
class _Datum:
    value: float
    token_count: int

    @property
    def model_input(self) -> _ModelInput:
        return _ModelInput(tuple(range(self.token_count)))


class _TinyReplicaBackend:
    """Small module-level fake that is safe to construct in spawned processes."""

    renderer_source = "hf"
    policy_samplers_are_snapshots = False
    sampler = "hf"
    use_lora = True

    def __init__(self, *, device: str = "cpu", gradient_reducer=None) -> None:
        self.device = device
        self.model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1.0)
        self.gradient_reducer = gradient_reducer
        self._accumulations = 0
        self.save_calls = 0

    def set_gradient_reducer(self, reducer) -> None:
        self.gradient_reducer = reducer

    def setup(self, *, resume_from=None, **_kwargs) -> None:
        if resume_from:
            raw = str(resume_from)
            directory = Path(raw.removeprefix("file://"))
            self.model.load_state_dict(torch.load(directory / "fake_weight.pt", map_location="cpu", weights_only=True))

    def logical_loss_denominator(self, datums, _loss_fn):
        return torch.tensor(sum(datum.value for datum in datums), dtype=torch.float32)

    async def submit_forward_backward(self, datums, loss_fn, *, global_loss_denominator=None):
        assert loss_fn in {"ppo", "importance_sampling", "cross_entropy"}
        denominator = max(float(global_loss_denominator), 1e-8)
        inputs = [float(datum.value) for datum in datums]
        numerator = sum(inputs)
        loss = -(self.model.weight.reshape(()) * numerator / denominator)
        loss.backward()
        self._accumulations += 1
        return _Pending(
            ForwardBackwardOutput(
                logprobs=[torch.tensor([value]) for value in inputs],
                metrics={"loss": float(loss.detach())},
            )
        )

    async def submit_opct_forward_backward(self, datums, **kwargs):
        return await self.submit_forward_backward(
            datums,
            kwargs["loss_fn"],
            global_loss_denominator=kwargs["global_loss_denominator"],
        )

    async def submit_optim_step(self, *, learning_rate, adam):
        del adam
        if self.gradient_reducer == "torch.distributed":
            from ctm.backends.local.engine import sum_trainable_gradients_torch_distributed

            sum_trainable_gradients_torch_distributed(list(self.model.parameters()))
        elif callable(self.gradient_reducer):
            self.gradient_reducer(list(self.model.parameters()))
        self.model.weight.data.add_(self.model.weight.grad, alpha=-learning_rate)
        self.model.zero_grad(set_to_none=True)
        self._accumulations = 0
        return _Pending(None)

    async def incorporate_kl_penalty(self, datums, *, kl_coef, kl_discount_factor):
        del datums, kl_coef, kl_discount_factor
        return {"kl_policy_base": 0.0}

    def policy_sampler(self, name):
        return name

    async def refresh_policy_sampler(self, name):
        return name

    def base_sampler(self):
        return "base"

    async def save_checkpoint(self, *, name, log_dir, loop_state, kind):
        del loop_state
        self.save_calls += 1
        directory = Path(log_dir) / "checkpoints" / name
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "fake_weight.pt")
        uri = f"file://{directory.resolve()}"
        return {"sampler_path": uri, "state_path": uri if kind in {"state", "both"} else None}

    def shutdown(self) -> None:
        return None


class _PeerFailsBeforeOptimizerCollectiveBackend(_TinyReplicaBackend):
    """Child-only backend that exits before joining the gradient SUM."""

    async def submit_optim_step(self, *, learning_rate, adam):
        del learning_rate, adam
        raise RuntimeError("injected peer failure before optimizer collective")


def _topology(count: int):
    return resolve_phase_shared_topology(
        train_gpus_spec="all",
        rollout_gpus_spec="all",
        cuda_visible_devices=",".join(f"fake-{index}" for index in range(count)),
    )


class _ExactChildProcess:
    """No-op process double that records whether force teardown was exact."""

    def __init__(self, *, hung: bool = False) -> None:
        self._alive = True
        self.hung = hung
        self.terminated = False
        self.join_states: list[tuple[bool, float | None]] = []

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        self.join_states.append((self.terminated, timeout))


class _RecordingQueue:
    def __init__(self) -> None:
        self.puts: list[object] = []
        self.closed = False

    def put(self, value, timeout=None) -> None:
        del timeout
        self.puts.append(value)

    def close(self) -> None:
        self.closed = True


class _PublicationPool:
    """Minimal sleeping worker-pool double for outer publication ordering."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.sleep_enabled = True
        self.phase = "training"
        self.published: list[tuple[Path, int]] = []

    async def publish_adapter(self, adapter_path: Path, *, version: int) -> None:
        self.events.append("pool.publish_adapter")
        self.published.append((Path(adapter_path), version))


class _PublicationModel:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def save_pretrained(self, path: str) -> None:
        self.events.append("trainer.snapshot_adapter")
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "adapter_model.safetensors").write_bytes(b"adapter")


class _PublicationTrainer:
    """A narrow trainer double exposing the two outer lifecycle hooks."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.model = _PublicationModel(events)

    async def verify_policy_publication(self) -> None:
        self.events.append("trainer.verify_policy_publication")

    async def enter_rollout_phase(self) -> None:
        self.events.append("trainer.enter_rollout_phase")


def _manually_ready_backend(*, world_size: int = 2, shutdown_timeout_seconds: float = 0.01):
    """Create a ready wrapper with exact process doubles, never spawning children."""

    rank_zero = _TinyReplicaBackend(device="cpu")
    backend = ReplicatedTrainingBackend(
        rank_zero,
        topology=_topology(world_size),
        child_backend_spec=LocalBackendConstructorSpec(_TinyReplicaBackend),
        process_group_backend="gloo",
        device_type="cpu",
        start_timeout_seconds=1,
        command_timeout_seconds=1,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
    )
    processes = {rank: _ExactChildProcess() for rank in range(1, world_size)}
    queues = {rank: _RecordingQueue() for rank in range(1, world_size)}
    backend._initialized = True
    backend._processes = processes
    backend._command_queues = queues
    backend._result_queue = _RecordingQueue()
    return backend, rank_zero, processes, queues


def _outer_publication_backend(tmp_path: Path, trainer, pool: _PublicationPool) -> RolloutParallelBackend:
    backend = RolloutParallelBackend(
        trainer,
        gpus=(),
        status_dir=tmp_path / "status",
        pool=pool,
    )
    backend._model_name = "fake/model"
    backend._adapter_root = tmp_path / "adapters"
    backend._adapter_root.mkdir(parents=True)
    return backend


def _assert_force_torn_down(backend, processes, queues) -> None:
    assert backend.poisoned
    assert all(process.terminated for process in processes.values())
    assert backend._processes == {}
    assert backend._command_queues == {}
    assert all(queue.puts == [] for queue in queues.values())


def _assert_no_unsafe_retry(backend) -> None:
    with pytest.raises(ReplicatedBackendPoisonedError):
        asyncio_run(backend.submit_optim_step(learning_rate=0.1, adam=AdamConfig()))


def test_replicated_wrapper_cpu_gloo_restores_order_and_exact_gradient():
    """The persistent rank-zero/peer path is testable without CUDA."""

    rank_zero = _TinyReplicaBackend(device="cpu")
    backend = ReplicatedTrainingBackend(
        rank_zero,
        topology=_topology(2),
        child_backend_spec=LocalBackendConstructorSpec(_TinyReplicaBackend),
        process_group_backend="gloo",
        device_type="cpu",
        start_timeout_seconds=30,
        command_timeout_seconds=30,
        shutdown_timeout_seconds=5,
    )
    try:
        backend.setup(model="fake", lora=LoRAConfig(rank=1, seed=42))
        datums = [_Datum(1.0, 9), _Datum(2.0, 2), _Datum(3.0, 13), _Datum(4.0, 4), _Datum(5.0, 7)]
        pending = asyncio_run(backend.submit_forward_backward(datums, "ppo"))
        result = asyncio_run(pending.result())
        assert [float(value.item()) for value in result.logprobs] == [1, 2, 3, 4, 5]
        assert result.metrics["loss"] == pytest.approx(-1.0)
        optim_pending = asyncio_run(backend.submit_optim_step(learning_rate=0.25, adam=AdamConfig()))
        asyncio_run(optim_pending.result())
        # Global objective is -w * sum(values)/sum(values) = -w, hence the
        # exact one-rank gradient is -1 and one SGD-style step makes 1.25.
        assert float(rank_zero.model.weight.detach().reshape(())) == pytest.approx(1.25)
        assert not backend.poisoned
    finally:
        backend.shutdown()


def test_peer_precollective_optimizer_failure_uses_bounded_operation_timeout_not_startup_timeout():
    """A peer error before SUM cannot retain rank zero for the startup timeout."""

    rank_zero = _TinyReplicaBackend(device="cpu")
    backend = ReplicatedTrainingBackend(
        rank_zero,
        topology=_topology(2),
        child_backend_spec=LocalBackendConstructorSpec(_PeerFailsBeforeOptimizerCollectiveBackend),
        process_group_backend="gloo",
        device_type="cpu",
        # A deliberately much longer startup limit makes this distinguish the
        # optimizer-only group from the default rendezvous group.
        start_timeout_seconds=30,
        optimizer_timeout_seconds=1,
        command_timeout_seconds=10,
        shutdown_timeout_seconds=2,
    )
    try:
        backend.setup(model="fake", lora=LoRAConfig(rank=1, seed=42))
        assert callable(rank_zero.gradient_reducer)
        pending = asyncio_run(backend.submit_forward_backward([_Datum(2.0, 4)], "ppo"))
        asyncio_run(pending.result())

        started = time.monotonic()
        with pytest.raises((RuntimeError, ReplicatedBackendError)):
            asyncio_run(backend.submit_optim_step(learning_rate=0.25, adam=AdamConfig()))
        elapsed = time.monotonic() - started

        assert backend.poisoned
        # Include modest process-spawn/teardown slack, but this must remain
        # decisively below the 30 second default-process-group startup limit.
        assert elapsed < 6
        assert elapsed < backend.start_timeout_seconds / 2
        assert backend._processes == {}
        assert backend._optimizer_process_group is None
    finally:
        backend.shutdown()


@pytest.mark.parametrize("gpu_count", [2, 4, 8])
def test_replicated_wrapper_exposes_arbitrary_topology_world_sizes(gpu_count):
    """Larger GPU counts use the same topology contract before launch."""

    backend = ReplicatedTrainingBackend(
        _TinyReplicaBackend(device="cpu"),
        topology=_topology(gpu_count),
        child_backend_spec=LocalBackendConstructorSpec(_TinyReplicaBackend),
        process_group_backend="gloo",
        device_type="cpu",
    )
    assert backend.world_size == gpu_count
    assert backend.topology.training_ranks[-1].rank == gpu_count - 1


def test_replicated_setup_fails_closed_for_lora_dropout_before_launch():
    backend = ReplicatedTrainingBackend(
        _TinyReplicaBackend(device="cpu"),
        topology=_topology(2),
        child_backend_spec=LocalBackendConstructorSpec(_TinyReplicaBackend),
        process_group_backend="gloo",
        device_type="cpu",
    )
    with pytest.raises(ReplicatedBackendError, match="dropout=0"):
        backend.setup(model="fake", lora=LoRAConfig(rank=1, dropout=0.1, seed=42))


def test_replicated_setup_fails_closed_for_lora_rank_above_vllm_limit_before_launch(monkeypatch):
    rank_zero = _TinyReplicaBackend(device="cpu")
    backend = ReplicatedTrainingBackend(
        rank_zero,
        topology=_topology(2),
        child_backend_spec=LocalBackendConstructorSpec(_TinyReplicaBackend),
        process_group_backend="gloo",
        device_type="cpu",
    )

    def unexpected_worker_launch(*_args, **_kwargs):
        pytest.fail("LoRA-rank validation must run before worker launch")

    def unexpected_rank_zero_setup(**_kwargs):
        pytest.fail("LoRA-rank validation must run before rank-zero model setup")

    monkeypatch.setattr(backend, "_start_process_group_and_workers", unexpected_worker_launch)
    monkeypatch.setattr(rank_zero, "setup", unexpected_rank_zero_setup)

    with pytest.raises(ReplicatedBackendError, match=r"LoRA rank <= 64.*got rank=65"):
        backend.setup(model="fake", lora=LoRAConfig(rank=65, seed=42))

    assert not backend._initialized
    assert not backend.poisoned
    assert backend._processes == {}


def test_replicated_checkpoint_sidecar_allows_same_topology_strict_resume(tmp_path):
    first_rank_zero = _TinyReplicaBackend(device="cpu")
    first = ReplicatedTrainingBackend(
        first_rank_zero,
        topology=_topology(2),
        child_backend_spec=LocalBackendConstructorSpec(_TinyReplicaBackend),
        process_group_backend="gloo",
        device_type="cpu",
        start_timeout_seconds=30,
        command_timeout_seconds=30,
        shutdown_timeout_seconds=5,
    )
    try:
        first.setup(model="fake", lora=LoRAConfig(rank=1, seed=42))
        pending = asyncio_run(first.submit_forward_backward([_Datum(2.0, 4)], "ppo"))
        asyncio_run(pending.result())
        optim = asyncio_run(first.submit_optim_step(learning_rate=0.25, adam=AdamConfig()))
        asyncio_run(optim.result())
        saved = asyncio_run(first.save_checkpoint(name="strict", log_dir=tmp_path, loop_state={}, kind="both"))
        directory = Path(saved["sampler_path"][len("file://") :])
        assert (directory / "replicated_training_manifest.json").is_file()
        assert (directory / "replicated_training_rng.pt").is_file()
        manifest = json.loads((directory / "replicated_training_manifest.json").read_text(encoding="utf-8"))
        assert manifest["optimizer_timeout_seconds"] == 120.0
    finally:
        first.shutdown()

    resumed_rank_zero = _TinyReplicaBackend(device="cpu")
    resumed = ReplicatedTrainingBackend(
        resumed_rank_zero,
        topology=_topology(2),
        child_backend_spec=LocalBackendConstructorSpec(_TinyReplicaBackend),
        process_group_backend="gloo",
        device_type="cpu",
        start_timeout_seconds=30,
        command_timeout_seconds=30,
        shutdown_timeout_seconds=5,
    )
    try:
        resumed.setup(
            model="fake",
            lora=LoRAConfig(rank=1, seed=42),
            resume_from=saved["sampler_path"],
            resume_with_optimizer=True,
        )
        assert trainable_state_hash(resumed_rank_zero) == trainable_state_hash(first_rank_zero)
    finally:
        resumed.shutdown()


def asyncio_run(awaitable):
    """Keep test helpers picklable by avoiding an async pytest plugin."""

    import asyncio

    return asyncio.run(awaitable)


def test_metric_aggregation_rejects_mismatched_global_bookkeeping():
    outputs = {
        0: ForwardBackwardOutput(logprobs=[], metrics={"loss": -0.2, "teacher_scored_tokens": 10.0}),
        1: ForwardBackwardOutput(logprobs=[], metrics={"loss": -0.3, "teacher_scored_tokens": 11.0}),
    }
    with pytest.raises(ReplicatedBackendError, match="global metric"):
        aggregate_sharded_metrics(outputs)


def test_token_cost_and_hash_are_explicit_and_stable():
    datum = _Datum(1.0, 7)
    assert datum_token_cost(datum) == 7
    backend = _TinyReplicaBackend()
    first = trainable_state_hash(backend)
    second = trainable_state_hash(backend)
    assert first == second
    with torch.no_grad():
        backend.model.weight.add_(1)
    assert trainable_state_hash(backend) != first


def test_partial_optimizer_dispatch_poison_force_tears_down_exact_children(monkeypatch):
    """A failure after rank 1 receives an optimizer command is terminal.

    It is unsafe to leave rank 1 holding an accumulated gradient and then let
    the coordinator retry against a new set of peers.  The wrapper must poison
    itself and terminate the exact child PIDs immediately.
    """

    backend, _rank_zero, processes, queues = _manually_ready_backend(world_size=3)
    dispatched: list[int] = []

    def partial_put(rank, command):
        dispatched.append(rank)
        if rank == 2:
            raise ReplicatedBackendError("injected second-peer dispatch failure")

    monkeypatch.setattr(backend, "_put_command", partial_put)
    try:
        with pytest.raises(ReplicatedBackendError, match="dispatch failure"):
            asyncio_run(backend.submit_optim_step(learning_rate=0.1, adam=AdamConfig()))
        assert dispatched == [1, 2]
        _assert_force_torn_down(backend, processes, queues)
        _assert_no_unsafe_retry(backend)
    finally:
        backend.shutdown()


def test_partial_forward_backward_dispatch_poison_force_tears_down_exact_children(monkeypatch):
    """A peer that accepted F/B cannot be left with unreconciled gradients."""

    backend, _rank_zero, processes, queues = _manually_ready_backend(world_size=3)
    dispatched: list[int] = []

    def partial_put(rank, command):
        dispatched.append(rank)
        if rank == 2:
            raise ReplicatedBackendError("injected second-peer F/B dispatch failure")

    monkeypatch.setattr(backend, "_put_command", partial_put)
    try:
        with pytest.raises(ReplicatedBackendError, match="F/B dispatch failure"):
            asyncio_run(
                backend.submit_forward_backward(
                    [_Datum(1.0, 8), _Datum(2.0, 4), _Datum(3.0, 6)],
                    "ppo",
                )
            )
        assert dispatched == [1, 2]
        _assert_force_torn_down(backend, processes, queues)
        _assert_no_unsafe_retry(backend)
    finally:
        backend.shutdown()


def test_checkpoint_hash_mismatch_poison_force_tears_down_and_does_not_save(tmp_path, monkeypatch):
    """A pre-checkpoint replica hash mismatch cannot publish rank-zero state."""

    backend, rank_zero, processes, queues = _manually_ready_backend()
    rank_zero_hash = "0" * 64
    peer_hash = "f" * 64

    def fake_dispatch(operation, payload):
        assert operation == "state_hash"
        assert payload == {}
        return 71

    def fake_collect(command_id, expected_ranks, timeout_seconds=None):
        assert command_id == 71
        assert expected_ranks == {1}
        assert timeout_seconds is None
        return {
            1: ReplicatedWorkerResult(
                rank=1,
                command_id=71,
                ok=True,
                payload={"state_hash": peer_hash},
            )
        }

    monkeypatch.setattr(backend, "_dispatch", fake_dispatch)
    monkeypatch.setattr(backend, "_collect_results", fake_collect)
    monkeypatch.setattr(replicated_module, "trainable_state_hash", lambda _backend: rank_zero_hash)
    try:
        with pytest.raises(ReplicatedBackendError, match="diverged"):
            asyncio_run(backend.save_checkpoint(name="mismatch", log_dir=tmp_path, loop_state={}, kind="both"))
        assert rank_zero.save_calls == 0
        _assert_force_torn_down(backend, processes, queues)
        with pytest.raises(ReplicatedBackendPoisonedError):
            asyncio_run(backend.save_checkpoint(name="retry", log_dir=tmp_path, loop_state={}, kind="both"))
    finally:
        backend.shutdown()


def test_checkpoint_rng_verifier_failure_poison_force_tears_down_after_unpublished_sidecar(tmp_path, monkeypatch):
    """A malformed peer RNG acknowledgement cannot leave a resumable checkpoint."""

    backend, rank_zero, processes, queues = _manually_ready_backend()

    async def hash_check_ok(*, phase):
        assert phase == "before checkpoint"

    def fake_dispatch(operation, payload):
        assert operation == "rng_state"
        assert payload == {}
        return 72

    def fake_collect(command_id, expected_ranks, timeout_seconds=None):
        assert command_id == 72
        assert expected_ranks == {1}
        assert timeout_seconds is None
        return {
            1: ReplicatedWorkerResult(
                rank=1,
                command_id=72,
                ok=True,
                payload={"rng_state": {"not": "a complete valid state"}},
            )
        }

    monkeypatch.setattr(backend, "_verify_replica_hashes", hash_check_ok)
    monkeypatch.setattr(backend, "_dispatch", fake_dispatch)
    monkeypatch.setattr(backend, "_collect_results", fake_collect)
    try:
        with pytest.raises(ReplicatedBackendError, match="RNG"):
            asyncio_run(backend.save_checkpoint(name="bad-rng", log_dir=tmp_path, loop_state={}, kind="both"))
        assert rank_zero.save_calls == 1
        checkpoint_dir = tmp_path / "checkpoints" / "bad-rng"
        assert (checkpoint_dir / "replicated_training_manifest.json").exists() is False
        _assert_force_torn_down(backend, processes, queues)
        with pytest.raises(ReplicatedBackendPoisonedError):
            asyncio_run(backend.refresh_policy_sampler("unsafe-retry"))
    finally:
        backend.shutdown()


def test_force_teardown_of_hung_peers_skips_graceful_wait_and_terminates_first():
    """Force cleanup must not wait for a queue response from a hung peer."""

    backend, _rank_zero, processes, queues = _manually_ready_backend(shutdown_timeout_seconds=60)
    for process in processes.values():
        process.hung = True
    try:
        backend._teardown_processes(force=True)
        assert all(process.terminated for process in processes.values())
        # Immediate force teardown must terminate before its first join. This
        # distinguishes it from a graceful wait that could retain paid GPUs for
        # the full shutdown timeout after an unsafe failure.
        assert all(process.join_states and process.join_states[0][0] is True for process in processes.values())
        assert all(queue.puts == [] for queue in queues.values())
    finally:
        backend.shutdown()


def test_outer_refresh_verifies_replicas_before_release_snapshot_and_worker_publish(tmp_path):
    """Publication verification is the first irreversible outer refresh step."""

    events: list[str] = []
    pool = _PublicationPool(events)
    outer = _outer_publication_backend(tmp_path, _PublicationTrainer(events), pool)

    asyncio_run(outer.refresh_policy_sampler("updated-policy"))

    assert events == [
        "trainer.verify_policy_publication",
        "trainer.enter_rollout_phase",
        "trainer.snapshot_adapter",
        "pool.publish_adapter",
    ]
    assert pool.published[0][1] == 1


def test_outer_refresh_replica_verification_failure_poison_tears_down_before_publish(tmp_path, monkeypatch):
    """The outer adapter path cannot bypass a failed replicated hash barrier."""

    replicated, _rank_zero, processes, queues = _manually_ready_backend()
    events: list[str] = []
    pool = _PublicationPool(events)
    outer = _outer_publication_backend(tmp_path, replicated, pool)

    def fake_dispatch(operation, payload):
        assert operation == "state_hash"
        assert payload == {}
        return 81

    def fake_collect(command_id, expected_ranks, timeout_seconds=None):
        assert command_id == 81
        assert expected_ranks == {1}
        assert timeout_seconds is None
        return {
            1: ReplicatedWorkerResult(
                rank=1,
                command_id=81,
                ok=True,
                payload={"state_hash": "f" * 64},
            )
        }

    monkeypatch.setattr(replicated, "_dispatch", fake_dispatch)
    monkeypatch.setattr(replicated, "_collect_results", fake_collect)
    monkeypatch.setattr(replicated_module, "trainable_state_hash", lambda _backend: "0" * 64)
    try:
        with pytest.raises(ReplicatedBackendError, match="diverged"):
            asyncio_run(outer.refresh_policy_sampler("unsafe-policy"))
        _assert_force_torn_down(replicated, processes, queues)
        assert events == []
        assert pool.published == []
        with pytest.raises(ReplicatedBackendPoisonedError):
            asyncio_run(outer.refresh_policy_sampler("retry"))
    finally:
        replicated.shutdown()
