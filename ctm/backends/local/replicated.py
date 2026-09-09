"""Exact, phase-shared replicated training for :class:`LocalBackend`.

This module is deliberately an orchestration layer rather than another loss
implementation.  It lets one rank-zero local trainer remain the public
``TrainingBackend`` while persistent peer processes hold identical replicas of
the trainable model.  A caller can therefore compose it as::

    LocalBackend -> ReplicatedTrainingBackend -> RolloutParallelBackend

The outer rollout backend continues to make rank zero the only adapter
publisher.  During a training phase, vLLM workers can sleep and every GPU in a
``PhaseSharedTopology.train_gpus`` set can contribute to the HF/PEFT update.

Correctness rules
-----------------

* rank zero receives the complete post-KL logical batch and computes its one
  denominator before sharding;
* datums are assigned by deterministic token-cost LPT sharding, never by a
  hard-coded device number or contiguous question range;
* every rank backpropagates its local numerator divided by that same global
  denominator; LocalBackend's optimizer-boundary reducer then SUMs LoRA
  gradients exactly once;
* rank zero restores logprob outputs to the original datum order and is the
  only checkpoint/adapter publisher; and
* a startup or post-optimizer replica-hash mismatch poisons the whole backend.

The transport is purposely small and explicit.  It uses a one-node
``torch.distributed`` process group for gradient collectives and ordinary
``multiprocessing`` queues for rank-zero commands/results.  This keeps model
and optimizer ownership inside ``LocalBackend`` rather than trying to wrap its
selected-token fast path in naïve DDP (which would silently bypass the wrapper
in current LocalBackend implementations).

Only globally-normalized token losses (cross entropy, PPO, importance
sampling, and the OPCT variant) are admitted.  ACT/AttCT/MLPCT use a different
paired-objective normalisation in LocalBackend; accepting them here without a
separate exact sharded formulation would be scientifically wrong, so they fail
closed.

The initialization group deliberately keeps its long startup timeout.  Gradient
SUMs instead use a second, identically-ranked process group with a short,
operation-specific timeout.  A peer which fails before joining an optimizer
collective therefore cannot hold rank zero in the startup rendezvous timeout.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import queue
import random
import socket
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import torch

from ctm.backends.base import ForwardBackwardOutput
from ctm.backends.local.phase_shared import (
    IndexedResult,
    PhaseSharedTopology,
    TokenCostShard,
    plan_token_cost_balanced_shards,
    restore_sharded_results,
)
from ctm.core.config import AdamConfig, LoRAConfig

_GLOBAL_NORMALIZED_LOSS_FNS = frozenset({"cross_entropy", "ppo", "importance_sampling"})
_SPECIAL_GLOBAL_METRICS = frozenset({"teacher_scored_tokens", "scored_tokens", "global_denominator"})
# Keep this synchronized with VLLMSampler's default ``max_lora_rank``.  A
# phase-shared rollout worker can otherwise accept startup only to fail when it
# loads the freshly trained adapter after the first update.
_VLLM_MAX_LORA_RANK = 64
REPLICATED_PROTOCOL_VERSION = 1
_RESUME_MANIFEST_NAME = "replicated_training_manifest.json"
_RESUME_RNG_NAME = "replicated_training_rng.pt"


class ReplicatedBackendError(RuntimeError):
    """A replicated local-training operation could not complete safely."""


class ReplicatedBackendPoisonedError(ReplicatedBackendError):
    """A previous replica/process failure made further use unsafe."""


@dataclass(frozen=True)
class LocalBackendConstructorSpec:
    """A picklable constructor contract for nonzero LocalBackend replicas.

    ``constructor`` must be a module-level callable (normally
    ``ctm.backends.local.engine.LocalBackend``), and ``kwargs`` must describe a
    fresh replica without an already-live model/vLLM process.  The coordinator
    receives its concrete ``training_backend`` separately, because it is the
    sole public sampler/checkpoint publisher.

    The worker device and the optimizer-boundary gradient reducer are injected
    by :class:`ReplicatedTrainingBackend`; callers must not bake either into
    ``kwargs``.  Failing early rather than letting multiprocessing discover an
    unpicklable closure hours into a run is intentional.
    """

    constructor: Callable[..., Any]
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not callable(self.constructor):
            raise TypeError("LocalBackendConstructorSpec.constructor must be callable")
        prohibited = {"device", "gradient_reducer"} & set(self.kwargs)
        if prohibited:
            joined = ", ".join(sorted(prohibited))
            raise ValueError(
                "LocalBackendConstructorSpec.kwargs must not specify "
                f"{joined}; replicated orchestration injects it per rank"
            )
        try:
            pickle.dumps(self)
        except BaseException as exc:
            raise TypeError(
                "LocalBackendConstructorSpec must be pickleable for persistent "
                "spawned worker processes; use a module-level constructor and "
                "serializable kwargs"
            ) from exc

    def build(self, *, device: str, gradient_reducer: Any | None) -> Any:
        self.validate()
        kwargs = copy.deepcopy(dict(self.kwargs))
        kwargs["device"] = device
        if gradient_reducer is not None:
            kwargs["gradient_reducer"] = gradient_reducer
        return self.constructor(**kwargs)


@dataclass(frozen=True)
class ReplicatedWorkerResult:
    """A serializable peer response labelled by logical training rank."""

    rank: int
    command_id: int
    ok: bool
    payload: Any = None
    error: str | None = None
    traceback_text: str | None = None


@dataclass(frozen=True)
class _WorkerCommand:
    command_id: int
    operation: str
    payload: Mapping[str, Any] = field(default_factory=dict)


class _ResolvedPending:
    """A local eager pending result preserving the public backend protocol."""

    def __init__(self, value: Any) -> None:
        self._value = value

    async def result(self) -> Any:
        return self._value


def _free_loopback_port() -> int:
    """Reserve a short-lived local rendezvous port for a one-node process group."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rank_device(topology: PhaseSharedTopology, rank: int, *, device_type: str) -> str:
    if device_type == "cpu":
        return "cpu"
    if device_type != "cuda":
        raise ValueError("device_type must be 'cuda' or 'cpu'")
    return f"cuda:{topology.training_ranks[rank].gpu.logical_index}"


def _checkpoint_directory(path: str | Path) -> Path:
    raw = str(path)
    raw = raw.removeprefix("file://")
    directory = Path(raw)
    if directory.is_symlink() or not directory.is_dir():
        raise ReplicatedBackendError(f"replicated checkpoint must be a regular directory: {directory}")
    return directory


def _capture_rng_state(*, device_type: str, device: str) -> dict[str, Any]:
    """Capture the stochastic state actually owned by one trainer process."""

    payload: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    try:
        import numpy as np

        payload["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover - NumPy is normally present, but is optional here
        payload["numpy"] = None
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise ReplicatedBackendError("cannot capture CUDA RNG state because torch.cuda is unavailable")
        payload["torch_cuda"] = torch.cuda.get_rng_state(torch.device(device)).cpu()
        payload["device"] = device
    else:
        payload["torch_cuda"] = None
        payload["device"] = "cpu"
    return payload


def _validate_rng_state(payload: Mapping[str, Any], *, device_type: str, device: str) -> None:
    """Validate one rank's checkpoint RNG payload without mutating live RNGs."""

    if not isinstance(payload, Mapping):
        raise ReplicatedBackendError("replicated RNG state must be a mapping")
    required = {"python", "torch_cpu", "torch_cuda", "device"}
    missing = sorted(required - set(payload))
    if missing:
        raise ReplicatedBackendError(f"replicated RNG state is missing fields: {missing}")
    try:
        probe = random.Random()
        probe.setstate(payload["python"])
    except (TypeError, ValueError) as exc:
        raise ReplicatedBackendError("replicated Python RNG state is invalid") from exc
    cpu_state = payload["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8 or cpu_state.ndim != 1:
        raise ReplicatedBackendError("replicated RNG torch_cpu field must be a one-dimensional uint8 tensor")
    recorded_device = payload["device"]
    expected_device = device if device_type == "cuda" else "cpu"
    if recorded_device != expected_device:
        raise ReplicatedBackendError(
            f"replicated RNG state belongs to device {recorded_device!r}, expected {expected_device!r}"
        )
    cuda_state = payload["torch_cuda"]
    if device_type == "cuda":
        if not isinstance(cuda_state, torch.Tensor) or cuda_state.dtype != torch.uint8 or cuda_state.ndim != 1:
            raise ReplicatedBackendError("replicated CUDA RNG state must be a one-dimensional uint8 tensor")
    elif cuda_state is not None:
        raise ReplicatedBackendError("cannot use a CUDA RNG state in a CPU replicated checkpoint")
    numpy_state = payload.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np

            probe_numpy = np.random.RandomState()
            probe_numpy.set_state(numpy_state)
        except ImportError as exc:  # pragma: no cover - NumPy is normally present
            raise ReplicatedBackendError("checkpoint contains NumPy RNG state but NumPy is unavailable") from exc
        except (TypeError, ValueError) as exc:
            raise ReplicatedBackendError("replicated NumPy RNG state is invalid") from exc


def _restore_rng_state(payload: Mapping[str, Any], *, device_type: str, device: str) -> None:
    """Restore an exact per-rank RNG state after model/optimizer construction."""

    _validate_rng_state(payload, device_type=device_type, device=device)
    random.setstate(payload["python"])
    cpu_state = payload["torch_cpu"]
    assert isinstance(cpu_state, torch.Tensor)
    torch.set_rng_state(cpu_state.cpu())
    numpy_state = payload.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np

            np.random.set_state(numpy_state)
        except ImportError as exc:  # pragma: no cover - protects strict state restore on minimal installs
            raise ReplicatedBackendError("checkpoint contains NumPy RNG state but NumPy is unavailable") from exc
    if device_type == "cuda":
        cuda_state = payload["torch_cuda"]
        assert isinstance(cuda_state, torch.Tensor)
        torch.cuda.set_rng_state(cuda_state.cpu(), device=torch.device(device))


def trainable_state_hash(backend: Any) -> str:
    """Hash trainable model state deterministically without trusting rank order.

    This intentionally hashes names, metadata, and raw bytes.  It is heavier
    than an all-reduced scalar checksum, but it is a fail-closed publication
    guard: equal checksums cannot hide a different adapter tensor layout.
    """

    model = getattr(backend, "model", None)
    if model is None or not callable(getattr(model, "named_parameters", None)):
        raise ReplicatedBackendError("replica has no initialized model for a trainable-state hash")
    digest = hashlib.sha256()
    found = False
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            found = True
            tensor = parameter.detach().contiguous()
            # ``bfloat16.numpy()`` is unsupported in some torch/NumPy pairs;
            # byte-viewing first is portable and represents all dtypes exactly.
            raw = tensor.view(torch.uint8).cpu().numpy().tobytes()
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(raw)
    if not found:
        raise ReplicatedBackendError("replica exposes no trainable parameters to hash")
    return digest.hexdigest()


def datum_token_cost(datum: Any) -> int:
    """Return one deterministic conservative scheduling cost for a datum.

    LocalBackend's bounded forwards pay predominantly for sequence length;
    the input token count is the only universally available, tokenizer-native
    estimate at this layer.  It is deliberately separate so a future packing
    scheduler can supply a more detailed cost model without changing the
    replicated correctness contract.
    """

    model_input = getattr(datum, "model_input", None)
    to_ints = getattr(model_input, "to_ints", None)
    if not callable(to_ints):
        raise TypeError("replicated training datum must expose model_input.to_ints()")
    tokens = to_ints()
    cost = len(tokens)
    if cost < 0:  # defensive for unusual sequence implementations
        raise ValueError("datum token cost must be non-negative")
    return max(1, int(cost))


def _coerce_metric(value: Any, *, key: str, rank: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplicatedBackendError(f"rank {rank} returned non-numeric forward/backward metric {key!r}: {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ReplicatedBackendError(f"rank {rank} returned non-finite metric {key!r}: {numeric!r}")
    return numeric


def aggregate_sharded_metrics(outputs_by_rank: Mapping[int, ForwardBackwardOutput]) -> dict[str, float]:
    """Combine local normalized outputs into one logical-batch metric dict.

    LocalBackend returns the local numerator divided by the shared global
    denominator.  Those values therefore sum exactly across ranks.  A small
    number of bookkeeping fields already contain the global denominator itself
    and are instead required to agree exactly (within floating point noise).
    """

    all_keys = set().union(*(output.metrics for output in outputs_by_rank.values()))
    metrics: dict[str, float] = {}
    for key in sorted(all_keys):
        values: list[tuple[int, float]] = []
        for rank, output in sorted(outputs_by_rank.items()):
            if key not in output.metrics:
                raise ReplicatedBackendError(f"rank {rank} omitted metric {key!r}; per-rank metric schemas must match")
            values.append((rank, _coerce_metric(output.metrics[key], key=key, rank=rank)))
        if key in _SPECIAL_GLOBAL_METRICS:
            reference = values[0][1]
            if any(not math.isclose(value, reference, rel_tol=1e-6, abs_tol=1e-6) for _, value in values[1:]):
                raise ReplicatedBackendError(
                    f"replicas disagreed about global metric {key!r}: "
                    + ", ".join(f"rank {rank}={value}" for rank, value in values)
                )
            metrics[key] = reference
        else:
            metrics[key] = sum(value for _, value in values)
    return metrics


async def _await_backend_operation(backend: Any, operation: str, payload: Mapping[str, Any]) -> Any:
    """Invoke one LocalBackend-shaped async operation and resolve its pending value."""

    method = getattr(backend, operation, None)
    if not callable(method):
        raise ReplicatedBackendError(f"replica backend has no callable {operation}()")
    result = await method(**dict(payload))
    pending_result = getattr(result, "result", None)
    if callable(pending_result):
        return await pending_result()
    return result


def _configure_optimizer_gradient_reducer(backend: Any, *, process_group: Any) -> None:
    """Bind LocalBackend's one-SUM reducer to the short-lived optimizer group.

    The default group retains the generous model-load/rendezvous timeout.  A
    separate group lets a failed peer abort an optimizer collective on the
    prompt operation timeout without changing startup behaviour or the exact
    SUM (rather than mean) reduction semantics.
    """

    setter = getattr(backend, "set_gradient_reducer", None)
    if not callable(setter):
        raise ReplicatedBackendError(
            "replicated local backend lacks set_gradient_reducer(); exact optimizer synchronization is unavailable"
        )
    from ctm.backends.local.engine import sum_trainable_gradients_torch_distributed

    def reduce_trainable_gradients(parameters: Sequence[torch.nn.Parameter]) -> None:
        sum_trainable_gradients_torch_distributed(parameters, process_group=process_group)

    setter(reduce_trainable_gradients)


def _new_optimizer_process_group(
    *,
    world_size: int,
    backend: str,
    timeout_seconds: float,
) -> Any:
    """Create the all-rank collective group used only at optimizer boundaries."""

    import torch.distributed as dist

    return dist.new_group(
        ranks=list(range(world_size)),
        backend=backend,
        timeout=timedelta(seconds=timeout_seconds),
    )


def _worker_main(
    *,
    rank: int,
    topology: PhaseSharedTopology,
    constructor_spec: LocalBackendConstructorSpec,
    init_method: str,
    process_group_backend: str,
    process_group_timeout_seconds: float,
    optimizer_timeout_seconds: float,
    device_type: str,
    command_queue: Any,
    result_queue: Any,
    setup_payload: Mapping[str, Any],
    resume_rng_state: Mapping[str, Any] | None,
) -> None:
    """Persistent nonzero-rank worker entry point (must remain module-level)."""

    backend: Any | None = None
    dist_initialized = False
    optimizer_process_group: Any | None = None
    try:
        import torch.distributed as dist

        device = _rank_device(topology, rank, device_type=device_type)
        if device_type == "cuda":
            if not torch.cuda.is_available():
                raise ReplicatedBackendError("CUDA replicated worker requested but torch.cuda is unavailable")
            torch.cuda.set_device(torch.device(device))
        dist.init_process_group(
            backend=process_group_backend,
            init_method=init_method,
            rank=rank,
            world_size=topology.world_size,
            timeout=timedelta(seconds=process_group_timeout_seconds),
        )
        dist_initialized = True
        # Every rank creates this subgroup in the same deterministic point in
        # startup, before loading model state.  It is deliberately separate
        # from the long-lived default group so an optimizer-only collective
        # can fail promptly when a peer exits before entering it.
        optimizer_process_group = _new_optimizer_process_group(
            world_size=topology.world_size,
            backend=process_group_backend,
            timeout_seconds=optimizer_timeout_seconds,
        )
        backend = constructor_spec.build(
            device=device,
            gradient_reducer=None,
        )
        backend.setup(**dict(setup_payload))
        _configure_optimizer_gradient_reducer(backend, process_group=optimizer_process_group)
        if resume_rng_state is not None:
            _restore_rng_state(resume_rng_state, device_type=device_type, device=device)
        result_queue.put(
            ReplicatedWorkerResult(
                rank=rank,
                command_id=0,
                ok=True,
                payload={"state_hash": trainable_state_hash(backend)},
            )
        )

        while True:
            command = command_queue.get()
            if not isinstance(command, _WorkerCommand):
                raise ReplicatedBackendError(f"worker received unsupported command {command!r}")
            if command.operation == "shutdown":
                result_queue.put(ReplicatedWorkerResult(rank=rank, command_id=command.command_id, ok=True))
                break
            try:
                if command.operation == "state_hash":
                    payload = {"state_hash": trainable_state_hash(backend)}
                elif command.operation == "rng_state":
                    payload = {
                        "rng_state": _capture_rng_state(
                            device_type=device_type,
                            device=device,
                        )
                    }
                elif command.operation == "enter_rollout_phase":
                    asyncio.run(_await_backend_operation(backend, "enter_rollout_phase", {}))
                    payload = {"training_memory_released": True}
                elif command.operation == "forward_backward":
                    payload = awaitable_result = asyncio.run(
                        _await_backend_operation(backend, "submit_forward_backward", command.payload)
                    )
                    if not isinstance(awaitable_result, ForwardBackwardOutput):
                        raise ReplicatedBackendError(
                            "replica submit_forward_backward() did not resolve to ForwardBackwardOutput"
                        )
                elif command.operation == "opct_forward_backward":
                    payload = awaitable_result = asyncio.run(
                        _await_backend_operation(backend, "submit_opct_forward_backward", command.payload)
                    )
                    if not isinstance(awaitable_result, ForwardBackwardOutput):
                        raise ReplicatedBackendError(
                            "replica submit_opct_forward_backward() did not resolve to ForwardBackwardOutput"
                        )
                elif command.operation == "optim_step":
                    asyncio.run(_await_backend_operation(backend, "submit_optim_step", command.payload))
                    payload = {"state_hash": trainable_state_hash(backend)}
                else:
                    raise ReplicatedBackendError(f"unknown replicated worker operation {command.operation!r}")
            except BaseException as exc:  # noqa: BLE001 - relay exact remote diagnostic
                result_queue.put(
                    ReplicatedWorkerResult(
                        rank=rank,
                        command_id=command.command_id,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                        traceback_text=traceback.format_exc(),
                    )
                )
                # A failed command may leave ranks at different optimizer/
                # collective boundaries. Continuing would risk a later NCCL
                # deadlock or mixed adapter state, so this worker is terminal.
                break
            else:
                result_queue.put(
                    ReplicatedWorkerResult(rank=rank, command_id=command.command_id, ok=True, payload=payload)
                )
    except BaseException as exc:  # noqa: BLE001 - parent must receive startup failures too
        try:
            result_queue.put(
                ReplicatedWorkerResult(
                    rank=rank,
                    command_id=0,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback_text=traceback.format_exc(),
                )
            )
        except BaseException:  # noqa: BLE001, S110 - parent receives the startup error; cleanup cannot replace it
            pass
    finally:
        if backend is not None:
            shutdown = getattr(backend, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except BaseException:  # noqa: BLE001, S110 - preserve the worker's original training failure
                    pass
        if dist_initialized:
            try:
                import torch.distributed as dist

                if optimizer_process_group is not None:
                    dist.destroy_process_group(optimizer_process_group)
            except BaseException:  # noqa: BLE001, S110 - preserve the worker's original training failure
                pass
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group()
            except BaseException:  # noqa: BLE001, S110 - avoid masking a startup/worker failure during teardown
                pass


class ReplicatedTrainingBackend:
    """Rank-zero wrapper for exact, phase-shared replicated LocalBackend updates.

    The caller supplies an already-created rank-zero ``training_backend`` and a
    serializable recipe for independently creating the other replicas.  The
    topology's first explicit training GPU is rank zero; the constructor rejects
    ambiguous ``cuda`` placement when that GPU is not logical device zero.

    ``process_group_backend='nccl'`` is the production default.  ``gloo`` plus
    ``device_type='cpu'`` exists solely for isolated protocol tests and small
    CPU fake backends; it does not make long-model training practical on CPU.
    """

    renderer_source = "hf"
    policy_samplers_are_snapshots = False

    def __init__(
        self,
        training_backend: Any,
        *,
        topology: PhaseSharedTopology,
        child_backend_spec: LocalBackendConstructorSpec | None = None,
        process_group_backend: Literal["nccl", "gloo"] = "nccl",
        device_type: Literal["cuda", "cpu"] = "cuda",
        start_timeout_seconds: float = 1800.0,
        optimizer_timeout_seconds: float = 120.0,
        command_timeout_seconds: float = 7200.0,
        shutdown_timeout_seconds: float = 30.0,
        master_addr: str = "127.0.0.1",
        master_port: int | None = None,
        process_start_method: str = "spawn",
        verify_state_after_optim_step: bool = True,
    ) -> None:
        if topology.world_size < 1:
            raise ValueError("replicated training topology must contain at least one training GPU")
        if process_group_backend not in {"nccl", "gloo"}:
            raise ValueError("process_group_backend must be 'nccl' or 'gloo'")
        if device_type not in {"cuda", "cpu"}:
            raise ValueError("device_type must be 'cuda' or 'cpu'")
        if process_group_backend == "nccl" and device_type != "cuda":
            raise ValueError("NCCL replicated training requires device_type='cuda'")
        if not isinstance(verify_state_after_optim_step, bool):
            raise TypeError("verify_state_after_optim_step must be a boolean")
        for label, value in (
            ("start_timeout_seconds", start_timeout_seconds),
            ("optimizer_timeout_seconds", optimizer_timeout_seconds),
            ("command_timeout_seconds", command_timeout_seconds),
            ("shutdown_timeout_seconds", shutdown_timeout_seconds),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{label} must be a finite positive number")
        if not isinstance(master_addr, str) or not master_addr.strip():
            raise ValueError("master_addr must be a non-empty hostname/address")
        if master_port is not None and (
            isinstance(master_port, bool) or not isinstance(master_port, int) or not 1 <= master_port <= 65535
        ):
            raise ValueError("master_port must be None or an integer in [1, 65535]")
        try:
            self._mp = mp.get_context(process_start_method)
        except ValueError as exc:
            raise ValueError(f"unknown multiprocessing start method {process_start_method!r}") from exc

        self.training_backend = training_backend
        self.topology = topology
        self.child_backend_spec = child_backend_spec
        self.process_group_backend = process_group_backend
        self.device_type = device_type
        self.start_timeout_seconds = float(start_timeout_seconds)
        self.optimizer_timeout_seconds = float(optimizer_timeout_seconds)
        self.command_timeout_seconds = float(command_timeout_seconds)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self.master_addr = master_addr.strip()
        self.master_port = master_port
        self.verify_state_after_optim_step = verify_state_after_optim_step

        self._command_queues: dict[int, Any] = {}
        self._result_queue: Any | None = None
        self._processes: dict[int, mp.Process] = {}
        self._next_command_id = 1
        self._initialized = False
        self._poison_reason: str | None = None
        self._shutdown = False
        self._operation_lock = asyncio.Lock()
        self._rank_zero_state_hash: str | None = None
        self._init_method: str | None = None
        self._optimizer_process_group: Any | None = None
        self._operation_history: list[dict[str, Any]] = []

        if topology.world_size > 1:
            if child_backend_spec is None:
                raise ValueError(
                    "child_backend_spec is required when replicated topology world_size is greater than one"
                )
            child_backend_spec.validate()
        elif child_backend_spec is not None:
            child_backend_spec.validate()

    def __getattr__(self, name: str) -> Any:
        """Expose rank-zero LocalBackend attributes to outer rollout wrappers."""

        return getattr(self.training_backend, name)

    @property
    def sampling_training_overlap_supported(self) -> bool:
        # Phase sharing intentionally serializes sampling and training.  The
        # RL scheduler must not prefetch into the period in which vLLM sleeps.
        return False

    @property
    def poisoned(self) -> bool:
        return self._poison_reason is not None

    @property
    def poison_reason(self) -> str | None:
        return self._poison_reason

    @property
    def world_size(self) -> int:
        return self.topology.world_size

    @property
    def operation_history(self) -> tuple[Mapping[str, Any], ...]:
        """Small in-memory audit trail for runner logs and failure reports."""

        return tuple(dict(event) for event in self._operation_history)

    def setup(
        self,
        *,
        model: str,
        lora: LoRAConfig,
        resume_from: str | None = None,
        resume_with_optimizer: bool = False,
    ) -> None:
        """Start peer replicas and load one identical rank-zero/peer state."""

        self._ensure_usable()
        if self._initialized:
            raise ReplicatedBackendError("ReplicatedTrainingBackend.setup() may be called only once")
        if lora.rank > _VLLM_MAX_LORA_RANK:
            raise ReplicatedBackendError(
                "exact phase-shared replicated training requires LoRA rank "
                f"<= {_VLLM_MAX_LORA_RANK}, because its colocated vLLM rollout engines are configured "
                f"with max_lora_rank={_VLLM_MAX_LORA_RANK}; got rank={lora.rank}. "
                "Lower lora.rank or update the vLLM max_lora_rank implementation and this guard together."
            )
        if float(lora.dropout) != 0.0:
            raise ReplicatedBackendError(
                "exact replicated LocalBackend training currently requires LoRA dropout=0. "
                "Independent rank RNG streams would otherwise change the logical update; "
                "implement a rank-consistent stochastic-equivalence design before enabling it."
            )
        if lora.seed is None:
            raise ReplicatedBackendError(
                "exact replicated LocalBackend training requires an explicit LoRA seed so independently "
                "constructed replicas have identical adapter initialization."
            )
        self._validate_rank_zero_device()
        resume_contract = self._load_resume_contract(
            resume_from=resume_from,
            resume_with_optimizer=resume_with_optimizer,
        )
        setup_payload = {
            "model": model,
            "lora": lora,
            "resume_from": resume_from,
            "resume_with_optimizer": resume_with_optimizer,
        }

        try:
            if self.world_size == 1:
                self.training_backend.setup(**setup_payload)
                if resume_contract is not None:
                    _restore_rng_state(
                        resume_contract["rng_by_rank"][0],
                        device_type=self.device_type,
                        device=_rank_device(self.topology, 0, device_type=self.device_type),
                    )
                self._rank_zero_state_hash = trainable_state_hash(self.training_backend)
                self._initialized = True
                self._record_operation(
                    "setup",
                    model=model,
                    resumed=bool(resume_contract),
                    optimizer_timeout_seconds=self.optimizer_timeout_seconds,
                )
                return

            self._start_process_group_and_workers(
                setup_payload,
                resume_rng_by_rank=None if resume_contract is None else resume_contract["rng_by_rank"],
            )
            self.training_backend.setup(**setup_payload)
            if resume_contract is not None:
                _restore_rng_state(
                    resume_contract["rng_by_rank"][0],
                    device_type=self.device_type,
                    device=_rank_device(self.topology, 0, device_type=self.device_type),
                )
            rank_zero_hash = trainable_state_hash(self.training_backend)
            peer_results = self._collect_results(command_id=0, expected_ranks=set(range(1, self.world_size)))
            peer_hashes = {
                rank: self._extract_state_hash(result.payload, rank=rank) for rank, result in peer_results.items()
            }
            self._assert_replica_hashes(rank_zero_hash, peer_hashes, phase="initial setup")
            if resume_contract is not None and rank_zero_hash != resume_contract["state_hash"]:
                raise ReplicatedBackendError(
                    "strict replicated resume loaded a different trainable state than the checkpoint sidecar: "
                    f"expected {resume_contract['state_hash']}, got {rank_zero_hash}"
                )
            self._rank_zero_state_hash = rank_zero_hash
            self._initialized = True
            self._record_operation(
                "setup",
                model=model,
                resumed=bool(resume_contract),
                optimizer_timeout_seconds=self.optimizer_timeout_seconds,
            )
        except BaseException as exc:
            self._poison(f"setup failed: {type(exc).__name__}: {exc}")
            self._teardown_processes(force=True)
            raise

    def policy_sampler(self, name: str) -> Any:
        self._ensure_ready()
        return self.training_backend.policy_sampler(name)

    async def refresh_policy_sampler(self, name: str) -> Any:
        """Publish only the rank-zero adapter after a verified synchronized step."""

        self._ensure_ready()
        async with self._operation_lock:
            self._ensure_ready()
            try:
                if self.world_size > 1:
                    await self._verify_replica_hashes(phase="before adapter publication")
                return await self.training_backend.refresh_policy_sampler(name)
            except BaseException as exc:
                self._poison(f"adapter publication failed: {type(exc).__name__}: {exc}")
                self._teardown_processes(force=True)
                raise

    async def verify_policy_publication(self) -> None:
        """Unconditionally verify every replica before rank-zero publication.

        The outer rollout backend owns the actual adapter snapshot and vLLM
        wake, so calling ``refresh_policy_sampler`` here would incorrectly boot
        rank zero's unused in-process sampler.  This narrow hook supplies the
        missing hash barrier without changing resource ownership.
        """

        self._ensure_ready()
        async with self._operation_lock:
            self._ensure_ready()
            try:
                await self._verify_replica_hashes(phase="before adapter publication")
                self._record_operation("policy_publication_verified")
            except BaseException as exc:
                self._poison(f"policy publication verification failed: {type(exc).__name__}: {exc}")
                self._teardown_processes(force=True)
                raise

    def base_sampler(self) -> Any:
        self._ensure_ready()
        return self.training_backend.base_sampler()

    async def enter_rollout_phase(self) -> None:
        """Synchronize and release every trainer cache before vLLM wakes.

        The outer :class:`RolloutParallelBackend` owns the actual worker wake.
        This inner barrier only proves that all rank-local CUDA work is complete
        and that unused caching-allocator blocks have been returned first.
        """

        self._ensure_ready()
        async with self._operation_lock:
            self._ensure_ready()
            try:
                command_id: int | None = None
                if self.world_size > 1:
                    command_id = self._dispatch("enter_rollout_phase", {})
                await _await_backend_operation(self.training_backend, "enter_rollout_phase", {})
                if command_id is not None:
                    peer_results = await asyncio.to_thread(
                        self._collect_results,
                        command_id,
                        set(range(1, self.world_size)),
                    )
                    for rank, response in peer_results.items():
                        payload = response.payload
                        if not isinstance(payload, Mapping) or payload.get("training_memory_released") is not True:
                            raise ReplicatedBackendError(
                                f"rank {rank} did not acknowledge trainer-memory release before rollout"
                            )
                self._record_operation("training_memory_released")
            except BaseException as exc:
                self._poison(f"rollout memory-release barrier failed: {type(exc).__name__}: {exc}")
                self._teardown_processes(force=True)
                raise

    async def incorporate_kl_penalty(
        self,
        datums: Sequence[Any],
        *,
        kl_coef: float,
        kl_discount_factor: float,
    ) -> dict[str, float]:
        """Keep post-rollout KL mutation authoritative on rank zero.

        The full logical batch is scored/mutated before deterministic sharding,
        so GRPO/PPO advantages cannot depend on which replica later owns a
        datum.  This is deliberate rather than a missed parallelization.
        """

        self._ensure_ready()
        async with self._operation_lock:
            self._ensure_ready()
            return await self.training_backend.incorporate_kl_penalty(
                datums,
                kl_coef=kl_coef,
                kl_discount_factor=kl_discount_factor,
            )

    async def submit_forward_backward(
        self,
        datums: Sequence[Any],
        loss_fn: str,
        *,
        global_loss_denominator: torch.Tensor | float | None = None,
    ) -> _ResolvedPending:
        """Shard one exact globally-normalized LocalBackend objective."""

        if loss_fn not in _GLOBAL_NORMALIZED_LOSS_FNS:
            raise ValueError(
                "ReplicatedTrainingBackend supports only globally-normalized "
                "cross_entropy, ppo, and importance_sampling objectives; "
                f"got {loss_fn!r}"
            )
        output = await self._submit_sharded_forward_backward(
            datums,
            operation="forward_backward",
            loss_fn=loss_fn,
            extra_payload={},
            global_loss_denominator=global_loss_denominator,
        )
        return _ResolvedPending(output)

    async def submit_opct_forward_backward(
        self,
        datums: Sequence[Any],
        *,
        behavior_temperature: float,
        kl_coef: float,
        kl_discount_factor: float,
        loss_fn: str,
        global_loss_denominator: torch.Tensor | float | None = None,
    ) -> _ResolvedPending:
        """Replicate the OPCT raw-score update using the same exact contract."""

        if loss_fn not in {"importance_sampling", "ppo"}:
            raise ValueError(f"OPCT replicated training supports importance_sampling or ppo, got {loss_fn!r}")
        output = await self._submit_sharded_forward_backward(
            datums,
            operation="opct_forward_backward",
            loss_fn=loss_fn,
            extra_payload={
                "behavior_temperature": behavior_temperature,
                "kl_coef": kl_coef,
                "kl_discount_factor": kl_discount_factor,
            },
            global_loss_denominator=global_loss_denominator,
        )
        return _ResolvedPending(output)

    async def submit_optim_step(self, *, learning_rate: float, adam: AdamConfig) -> _ResolvedPending:
        """Synchronously reduce gradients and step every replica once."""

        self._ensure_ready()
        async with self._operation_lock:
            self._ensure_ready()
            if self.world_size == 1:
                result = await _await_backend_operation(
                    self.training_backend,
                    "submit_optim_step",
                    {"learning_rate": learning_rate, "adam": adam},
                )
                self._rank_zero_state_hash = trainable_state_hash(self.training_backend)
                return _ResolvedPending(result)

            try:
                # Dispatch belongs inside the fail-closed region.  A queue
                # failure after an earlier peer accepted the command can leave
                # that peer blocked in the NCCL reducer; the exception path
                # below must therefore poison and terminate the exact child
                # processes rather than allow a retry.
                command_id = self._dispatch(
                    "optim_step",
                    {"learning_rate": learning_rate, "adam": adam},
                )
                # Both parent and workers enter LocalBackend's reducer once;
                # its NCCL SUM is the barrier that makes optimizer state move
                # together.  Do not await peer Queue results before rank zero
                # enters, otherwise NCCL would deadlock.
                result = await _await_backend_operation(
                    self.training_backend,
                    "submit_optim_step",
                    {"learning_rate": learning_rate, "adam": adam},
                )
                peer_results = await asyncio.to_thread(
                    self._collect_results,
                    command_id,
                    set(range(1, self.world_size)),
                    timeout_seconds=self.optimizer_timeout_seconds,
                )
                rank_zero_hash = trainable_state_hash(self.training_backend)
                peer_hashes = {
                    rank: self._extract_state_hash(response.payload, rank=rank)
                    for rank, response in peer_results.items()
                }
                if self.verify_state_after_optim_step:
                    self._assert_replica_hashes(rank_zero_hash, peer_hashes, phase="optimizer step")
                self._rank_zero_state_hash = rank_zero_hash
                self._record_operation(
                    "optimizer_step",
                    command_id=command_id,
                    optimizer_timeout_seconds=self.optimizer_timeout_seconds,
                )
                return _ResolvedPending(result)
            except BaseException as exc:
                self._poison(f"optimizer step failed: {type(exc).__name__}: {exc}")
                self._teardown_processes(force=True)
                raise

    async def save_checkpoint(self, *, name: str, log_dir: str | Path, loop_state: dict, kind: str) -> dict:
        """Save only rank-zero state after proving replicas still agree."""

        self._ensure_ready()
        async with self._operation_lock:
            self._ensure_ready()
            try:
                if self.world_size > 1:
                    await self._verify_replica_hashes(phase="before checkpoint")
                result = await self.training_backend.save_checkpoint(
                    name=name,
                    log_dir=log_dir,
                    loop_state=loop_state,
                    kind=kind,
                )
                if not isinstance(result, dict):
                    raise ReplicatedBackendError("rank-zero save_checkpoint() did not return a dict")
                await self._write_resume_contract(result, kind=kind)
                self._record_operation("checkpoint", name=name, kind=kind)
                return result
            except BaseException as exc:
                self._poison(f"checkpoint failed: {type(exc).__name__}: {exc}")
                self._teardown_processes(force=True)
                raise

    def shutdown(self) -> None:
        """Ask peers to exit, preserve diagnostics on failure, and release rank zero."""

        if self._shutdown:
            return
        self._shutdown = True
        try:
            self._teardown_processes(force=self.poisoned)
        finally:
            shutdown = getattr(self.training_backend, "shutdown", None)
            if callable(shutdown):
                shutdown()

    def _validate_rank_zero_device(self) -> None:
        if self.device_type == "cpu":
            return
        expected = _rank_device(self.topology, 0, device_type="cuda")
        actual_raw = str(getattr(self.training_backend, "device", ""))
        try:
            actual = torch.device(actual_raw)
        except (RuntimeError, TypeError) as exc:
            raise ReplicatedBackendError(
                f"rank-zero LocalBackend has invalid device {actual_raw!r}; expected {expected!r}"
            ) from exc
        expected_device = torch.device(expected)
        # ``cuda`` is only unambiguous when the topology coordinator is the
        # first inherited visible GPU.  Reject it on e.g. train_gpus='3,1'.
        if (
            actual.type != "cuda"
            or (actual.index is not None and actual.index != expected_device.index)
            or (actual.index is None and expected_device.index != 0)
        ):
            raise ReplicatedBackendError(
                "rank-zero LocalBackend placement does not match topology coordinator: "
                f"backend device={actual_raw!r}, expected {expected!r}. "
                "Construct LocalBackend with the explicit topology-selected device."
            )

    def _start_process_group_and_workers(
        self,
        setup_payload: Mapping[str, Any],
        *,
        resume_rng_by_rank: Mapping[int, Mapping[str, Any]] | None,
    ) -> None:
        if self.child_backend_spec is None:
            raise AssertionError("validated in constructor")
        try:
            import torch.distributed as dist
        except ImportError as exc:  # pragma: no cover - standard torch builds include it
            raise ReplicatedBackendError("torch.distributed is required for replicated local training") from exc
        if dist.is_initialized():
            raise ReplicatedBackendError(
                "cannot create ReplicatedTrainingBackend inside an already initialized "
                "torch.distributed process group; use one rank-zero wrapper per local job"
            )
        if self.device_type == "cuda":
            if not torch.cuda.is_available():
                raise ReplicatedBackendError("CUDA replicated training requested but torch.cuda is unavailable")
            torch.cuda.set_device(torch.device(_rank_device(self.topology, 0, device_type="cuda")))

        port = self.master_port if self.master_port is not None else _free_loopback_port()
        self._init_method = f"tcp://{self.master_addr}:{port}"
        self._result_queue = self._mp.Queue()
        self._command_queues = {rank: self._mp.Queue() for rank in range(1, self.world_size)}
        for rank in range(1, self.world_size):
            process = self._mp.Process(
                target=_worker_main,
                kwargs={
                    "rank": rank,
                    "topology": self.topology,
                    "constructor_spec": self.child_backend_spec,
                    "init_method": self._init_method,
                    "process_group_backend": self.process_group_backend,
                    "process_group_timeout_seconds": self.start_timeout_seconds,
                    "optimizer_timeout_seconds": self.optimizer_timeout_seconds,
                    "device_type": self.device_type,
                    "command_queue": self._command_queues[rank],
                    "result_queue": self._result_queue,
                    "setup_payload": dict(setup_payload),
                    "resume_rng_state": None if resume_rng_by_rank is None else resume_rng_by_rank[rank],
                },
                name=f"ctm-replicated-trainer-rank-{rank}",
                daemon=False,
            )
            process.start()
            self._processes[rank] = process

        # Child ranks must be running before rank zero initializes; all ranks
        # then rendezvous before any replica loads model state.
        dist.init_process_group(
            backend=self.process_group_backend,
            init_method=self._init_method,
            rank=0,
            world_size=self.world_size,
            timeout=timedelta(seconds=self.start_timeout_seconds),
        )
        self._optimizer_process_group = _new_optimizer_process_group(
            world_size=self.world_size,
            backend=self.process_group_backend,
            timeout_seconds=self.optimizer_timeout_seconds,
        )
        _configure_optimizer_gradient_reducer(
            self.training_backend,
            process_group=self._optimizer_process_group,
        )

    def _planned_shards(self, datums: Sequence[Any]) -> tuple[TokenCostShard[Any], ...]:
        values = tuple(datums)
        costs = tuple(datum_token_cost(datum) for datum in values)
        return plan_token_cost_balanced_shards(values, costs, ranks=self.topology.training_ranks)

    def _global_denominator(
        self,
        datums: Sequence[Any],
        loss_fn: str,
        supplied: torch.Tensor | float | None,
    ) -> float:
        if supplied is not None:
            if isinstance(supplied, torch.Tensor):
                if supplied.numel() != 1:
                    raise ValueError("global_loss_denominator must be scalar")
                value = float(supplied.detach().cpu())
            else:
                value = float(supplied)
        else:
            method = getattr(self.training_backend, "logical_loss_denominator", None)
            if not callable(method):
                raise ReplicatedBackendError(
                    "rank-zero LocalBackend lacks logical_loss_denominator(), required for exact replicated updates"
                )
            local = method(datums, loss_fn)
            if isinstance(local, torch.Tensor):
                if local.numel() != 1:
                    raise ReplicatedBackendError("rank-zero logical_loss_denominator() returned a non-scalar")
                value = float(local.detach().cpu())
            else:
                value = float(local)
        if not math.isfinite(value) or value < 0:
            raise ReplicatedBackendError(f"global loss denominator must be finite and non-negative, got {value!r}")
        return value

    async def _submit_sharded_forward_backward(
        self,
        datums: Sequence[Any],
        *,
        operation: Literal["forward_backward", "opct_forward_backward"],
        loss_fn: str,
        extra_payload: Mapping[str, Any],
        global_loss_denominator: torch.Tensor | float | None,
    ) -> ForwardBackwardOutput:
        self._ensure_ready()
        values = tuple(datums)
        async with self._operation_lock:
            self._ensure_ready()
            shards = self._planned_shards(values)
            denominator = self._global_denominator(values, loss_fn, global_loss_denominator)
            rank_zero_shard = shards[0]
            base_payload = dict(extra_payload)
            base_payload.update({"loss_fn": loss_fn, "global_loss_denominator": denominator})

            try:
                command_id: int | None = None
                if self.world_size > 1:
                    command_id = self._dispatch_shards(operation, shards[1:], base_payload)
                rank_zero_payload = dict(base_payload)
                rank_zero_payload["datums"] = tuple(item.value for item in rank_zero_shard.items)
                method_name = (
                    "submit_forward_backward" if operation == "forward_backward" else "submit_opct_forward_backward"
                )
                rank_zero_output = await _await_backend_operation(
                    self.training_backend,
                    method_name,
                    rank_zero_payload,
                )
                if not isinstance(rank_zero_output, ForwardBackwardOutput):
                    raise ReplicatedBackendError(f"rank zero {method_name}() did not resolve to ForwardBackwardOutput")

                outputs_by_rank: dict[int, ForwardBackwardOutput] = {0: rank_zero_output}
                if command_id is not None:
                    peer_results = await asyncio.to_thread(
                        self._collect_results,
                        command_id,
                        set(range(1, self.world_size)),
                    )
                    for rank, result in peer_results.items():
                        if not isinstance(result.payload, ForwardBackwardOutput):
                            raise ReplicatedBackendError(
                                f"rank {rank} {method_name}() did not return ForwardBackwardOutput"
                            )
                        outputs_by_rank[rank] = result.payload

                indexed_by_rank = {
                    rank: tuple(
                        IndexedResult(item.original_index, logprob)
                        for item, logprob in zip(shard.items, outputs_by_rank[rank].logprobs)
                    )
                    for rank, shard in enumerate(shards)
                }
                # A missing logprob for a nonempty batch is not recoverable;
                # never let zip silently turn it into a shorter public result.
                for rank, shard in enumerate(shards):
                    if len(outputs_by_rank[rank].logprobs) != len(shard.items):
                        raise ReplicatedBackendError(
                            f"rank {rank} returned {len(outputs_by_rank[rank].logprobs)} logprob vectors for "
                            f"{len(shard.items)} assigned datums"
                        )
                logprobs = restore_sharded_results(shards, indexed_by_rank)
                return ForwardBackwardOutput(logprobs=logprobs, metrics=aggregate_sharded_metrics(outputs_by_rank))
            except BaseException as exc:
                self._poison(f"{operation} failed: {type(exc).__name__}: {exc}")
                # A partially dispatched F/B command can leave an earlier
                # peer with accumulated gradients while another rank never
                # received the same logical batch. That state is not safely
                # retryable: terminate these exact replicas before returning
                # the error, just as the optimizer path does for a partial
                # collective entry.
                self._teardown_processes(force=True)
                raise

    def _dispatch_shards(
        self,
        operation: str,
        shards: Sequence[TokenCostShard[Any]],
        base_payload: Mapping[str, Any],
    ) -> int:
        if {shard.rank for shard in shards} != set(range(1, self.world_size)):
            raise ReplicatedBackendError("peer shard layout does not match replicated world size")
        command_id = self._next_command_id
        self._next_command_id += 1
        for shard in shards:
            payload = dict(base_payload)
            payload["datums"] = tuple(item.value for item in shard.items)
            self._put_command(shard.rank, _WorkerCommand(command_id, operation, payload))
        return command_id

    def _dispatch(self, operation: str, payload: Mapping[str, Any]) -> int:
        command_id = self._next_command_id
        self._next_command_id += 1
        for rank in range(1, self.world_size):
            self._put_command(rank, _WorkerCommand(command_id, operation, dict(payload)))
        return command_id

    def _put_command(self, rank: int, command: _WorkerCommand) -> None:
        process = self._processes.get(rank)
        if process is None or not process.is_alive():
            raise ReplicatedBackendError(f"replica rank {rank} is not alive before command {command.operation!r}")
        try:
            self._command_queues[rank].put(command, timeout=self.command_timeout_seconds)
        except BaseException as exc:
            raise ReplicatedBackendError(
                f"could not dispatch {command.operation!r} to replica rank {rank} within timeout"
            ) from exc

    def _collect_results(
        self,
        command_id: int,
        expected_ranks: set[int],
        timeout_seconds: float | None = None,
    ) -> dict[int, ReplicatedWorkerResult]:
        if not expected_ranks:
            return {}
        if self._result_queue is None:
            raise ReplicatedBackendError("replicated worker result queue is not initialized")
        if timeout_seconds is None:
            timeout_seconds = self.start_timeout_seconds if command_id == 0 else self.command_timeout_seconds
        deadline = time.monotonic() + timeout_seconds
        results: dict[int, ReplicatedWorkerResult] = {}
        while set(results) != expected_ranks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                alive = {rank: process.is_alive() for rank, process in self._processes.items()}
                missing = sorted(expected_ranks - set(results))
                raise ReplicatedBackendError(
                    f"timed out waiting for replicated worker command {command_id}; missing ranks={missing}, alive={alive}"
                )
            try:
                message = self._result_queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                dead = [
                    rank
                    for rank in expected_ranks
                    if not self._processes.get(rank, None) or not self._processes[rank].is_alive()
                ]
                if dead:
                    raise ReplicatedBackendError(
                        f"replicated worker process(es) died waiting for command {command_id}: ranks={dead}"
                    )
                continue
            if not isinstance(message, ReplicatedWorkerResult):
                raise ReplicatedBackendError(f"received malformed replicated worker response {message!r}")
            if message.command_id != command_id:
                raise ReplicatedBackendError(
                    f"received stale/out-of-order worker response command_id={message.command_id}, expected={command_id}"
                )
            if message.rank not in expected_ranks:
                raise ReplicatedBackendError(
                    f"received worker response from unexpected rank {message.rank}; expected {sorted(expected_ranks)}"
                )
            if message.rank in results:
                raise ReplicatedBackendError(f"received duplicate worker response from rank {message.rank}")
            if not message.ok:
                diagnostic = message.error or "unknown remote error"
                if message.traceback_text:
                    diagnostic += f"\nremote traceback:\n{message.traceback_text}"
                raise ReplicatedBackendError(f"replica rank {message.rank} failed command {command_id}: {diagnostic}")
            results[message.rank] = message
        return results

    async def _verify_replica_hashes(self, *, phase: str) -> None:
        if self.world_size <= 1:
            self._rank_zero_state_hash = trainable_state_hash(self.training_backend)
            return
        command_id = self._dispatch("state_hash", {})
        peer_results = await asyncio.to_thread(
            self._collect_results,
            command_id,
            set(range(1, self.world_size)),
        )
        rank_zero_hash = trainable_state_hash(self.training_backend)
        peer_hashes = {
            rank: self._extract_state_hash(response.payload, rank=rank) for rank, response in peer_results.items()
        }
        self._assert_replica_hashes(rank_zero_hash, peer_hashes, phase=phase)
        self._rank_zero_state_hash = rank_zero_hash

    async def _write_resume_contract(self, result: Mapping[str, Any], *, kind: str) -> None:
        """Persist rank-local RNG plus topology provenance beside rank-zero state.

        A LocalBackend optimizer checkpoint is enough only if every replica has
        the same parameter/optimizer tensors *and* each rank's stochastic
        stream resumes at the saved point.  This sidecar makes that extra
        contract explicit.  It is intentionally written for sampler-only
        checkpoints too, but only ``state``/``both`` manifests may be used for
        strict optimizer-state resume.
        """

        sampler_path = result.get("sampler_path") or result.get("state_path")
        if not isinstance(sampler_path, (str, Path)):
            raise ReplicatedBackendError("rank-zero checkpoint result has no sampler_path/state_path for provenance")
        directory = _checkpoint_directory(sampler_path)
        rng_by_rank: dict[int, Mapping[str, Any]] = {
            0: _capture_rng_state(
                device_type=self.device_type,
                device=_rank_device(self.topology, 0, device_type=self.device_type),
            )
        }
        if self.world_size > 1:
            command_id = self._dispatch("rng_state", {})
            peer_results = await asyncio.to_thread(
                self._collect_results,
                command_id,
                set(range(1, self.world_size)),
            )
            for rank, response in peer_results.items():
                payload = response.payload
                if not isinstance(payload, Mapping) or not isinstance(payload.get("rng_state"), Mapping):
                    raise ReplicatedBackendError(f"rank {rank} did not return a valid RNG state")
                rng_state = payload["rng_state"]
                _validate_rng_state(
                    rng_state,
                    device_type=self.device_type,
                    device=_rank_device(self.topology, rank, device_type=self.device_type),
                )
                rng_by_rank[rank] = rng_state
        if set(rng_by_rank) != set(range(self.world_size)):
            raise ReplicatedBackendError("could not capture RNG state for every replicated training rank")

        state_hash = trainable_state_hash(self.training_backend)
        if self._rank_zero_state_hash is not None and state_hash != self._rank_zero_state_hash:
            raise ReplicatedBackendError(
                "rank-zero trainable state changed between replica verification and checkpoint provenance capture"
            )
        manifest = {
            "schema": REPLICATED_PROTOCOL_VERSION,
            "checkpoint_kind": kind,
            "world_size": self.world_size,
            "train_logical_indices": [rank.gpu.logical_index for rank in self.topology.training_ranks],
            "process_group_backend": self.process_group_backend,
            "optimizer_timeout_seconds": self.optimizer_timeout_seconds,
            "device_type": self.device_type,
            "state_hash": state_hash,
            "rng_state_file": _RESUME_RNG_NAME,
        }
        rng_path = directory / _RESUME_RNG_NAME
        manifest_path = directory / _RESUME_MANIFEST_NAME
        rng_staging = directory / f".{_RESUME_RNG_NAME}.tmp-{os.getpid()}-{time.time_ns()}"
        manifest_staging = directory / f".{_RESUME_MANIFEST_NAME}.tmp-{os.getpid()}-{time.time_ns()}"
        # Do not erase failed staging evidence. A future strict resume will
        # refuse incomplete sidecars rather than guessing which file won.
        torch.save({"schema": REPLICATED_PROTOCOL_VERSION, "rng_by_rank": rng_by_rank}, rng_staging)
        os.replace(rng_staging, rng_path)
        manifest_staging.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(manifest_staging, manifest_path)

    def _load_resume_contract(
        self,
        *,
        resume_from: str | None,
        resume_with_optimizer: bool,
    ) -> dict[str, Any] | None:
        """Validate an immutable sidecar before a strict replicated resume.

        The historical single-rank checkpoint format lacks per-rank RNG and
        world-size provenance.  It remains usable for sampler-only warm starts,
        but strict optimizer-state resume intentionally fails closed unless the
        checkpoint was written by this wrapper.
        """

        if not resume_with_optimizer:
            return None
        if not resume_from:
            raise ReplicatedBackendError("resume_with_optimizer=True requires an explicit replicated checkpoint")
        directory = _checkpoint_directory(resume_from)
        manifest_path = directory / _RESUME_MANIFEST_NAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ReplicatedBackendError(
                "strict replicated optimizer resume requires "
                f"{_RESUME_MANIFEST_NAME}; checkpoint was not saved by ReplicatedTrainingBackend"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplicatedBackendError(f"cannot read replicated resume manifest {manifest_path}") from exc
        if not isinstance(manifest, Mapping):
            raise ReplicatedBackendError("replicated resume manifest must be a JSON object")
        if manifest.get("schema") != REPLICATED_PROTOCOL_VERSION:
            raise ReplicatedBackendError(
                "unsupported replicated checkpoint protocol schema "
                f"{manifest.get('schema')!r}; expected {REPLICATED_PROTOCOL_VERSION}"
            )
        if manifest.get("checkpoint_kind") not in {"state", "both"}:
            raise ReplicatedBackendError(
                "strict replicated optimizer resume requires a checkpoint saved with kind='state' or kind='both'"
            )
        if manifest.get("world_size") != self.world_size:
            raise ReplicatedBackendError(
                "strict replicated resume requires the same training world size: "
                f"checkpoint={manifest.get('world_size')!r}, current={self.world_size}"
            )
        saved_indices = manifest.get("train_logical_indices")
        current_indices = [rank.gpu.logical_index for rank in self.topology.training_ranks]
        if saved_indices != current_indices:
            raise ReplicatedBackendError(
                "strict replicated resume requires the same ordered logical training-GPU layout: "
                f"checkpoint={saved_indices!r}, current={current_indices!r}"
            )
        if manifest.get("process_group_backend") != self.process_group_backend:
            raise ReplicatedBackendError(
                "strict replicated resume requires the same collective backend: "
                f"checkpoint={manifest.get('process_group_backend')!r}, current={self.process_group_backend!r}"
            )
        if manifest.get("device_type") != self.device_type:
            raise ReplicatedBackendError(
                "strict replicated resume requires the same device type: "
                f"checkpoint={manifest.get('device_type')!r}, current={self.device_type!r}"
            )
        state_hash = manifest.get("state_hash")
        self._extract_state_hash({"state_hash": state_hash}, rank=0)
        rng_filename = manifest.get("rng_state_file")
        if rng_filename != _RESUME_RNG_NAME:
            raise ReplicatedBackendError("replicated resume manifest references an unsupported RNG-state filename")
        rng_path = directory / rng_filename
        if rng_path.is_symlink() or not rng_path.is_file():
            raise ReplicatedBackendError(f"replicated checkpoint RNG state is missing: {rng_path}")
        try:
            rng_document = torch.load(rng_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, pickle.UnpicklingError) as exc:
            raise ReplicatedBackendError(f"cannot load replicated RNG state {rng_path}") from exc
        if not isinstance(rng_document, Mapping) or rng_document.get("schema") != REPLICATED_PROTOCOL_VERSION:
            raise ReplicatedBackendError("replicated RNG sidecar has an unsupported schema")
        raw_rng_by_rank = rng_document.get("rng_by_rank")
        if not isinstance(raw_rng_by_rank, Mapping):
            raise ReplicatedBackendError("replicated RNG sidecar has no rank-indexed RNG states")
        rng_by_rank: dict[int, Mapping[str, Any]] = {}
        for rank in range(self.world_size):
            state = raw_rng_by_rank.get(rank)
            if not isinstance(state, Mapping):
                raise ReplicatedBackendError(f"replicated RNG sidecar is missing a valid state for rank {rank}")
            rng_by_rank[rank] = state
        return {"state_hash": state_hash, "rng_by_rank": rng_by_rank}

    def _record_operation(self, operation: str, **details: Any) -> None:
        self._operation_history.append(
            {
                "protocol_version": REPLICATED_PROTOCOL_VERSION,
                "operation": operation,
                "timestamp_unix": time.time(),
                "world_size": self.world_size,
                **details,
            }
        )

    @staticmethod
    def _extract_state_hash(payload: Any, *, rank: int) -> str:
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state_hash"), str):
            raise ReplicatedBackendError(f"rank {rank} did not return a valid trainable-state hash")
        state_hash = payload["state_hash"]
        if len(state_hash) != 64 or any(character not in "0123456789abcdef" for character in state_hash):
            raise ReplicatedBackendError(f"rank {rank} returned malformed SHA-256 state hash {state_hash!r}")
        return state_hash

    @staticmethod
    def _assert_replica_hashes(rank_zero_hash: str, peer_hashes: Mapping[int, str], *, phase: str) -> None:
        mismatched = {rank: value for rank, value in peer_hashes.items() if value != rank_zero_hash}
        if mismatched:
            details = f"rank0={rank_zero_hash}; " + ", ".join(
                f"rank{rank}={value}" for rank, value in sorted(mismatched.items())
            )
            raise ReplicatedBackendError(f"trainable replica state diverged during {phase}: {details}")

    def _ensure_usable(self) -> None:
        if self._shutdown:
            raise ReplicatedBackendError("ReplicatedTrainingBackend has been shut down")
        if self._poison_reason is not None:
            raise ReplicatedBackendPoisonedError(
                f"ReplicatedTrainingBackend is poisoned and cannot safely continue: {self._poison_reason}"
            )

    def _ensure_ready(self) -> None:
        self._ensure_usable()
        if not self._initialized:
            raise ReplicatedBackendError("ReplicatedTrainingBackend.setup() must complete before use")

    def _poison(self, reason: str) -> None:
        if self._poison_reason is None:
            self._poison_reason = reason
            self._record_operation("poisoned", reason=reason)

    def _teardown_processes(self, *, force: bool) -> None:
        # Try a normal queue-level shutdown only while the protocol is still
        # healthy.  After a failed collective it can block forever, in which
        # case process termination is safer than keeping paid GPUs alive.
        if self._processes and not force:
            command_id = self._next_command_id
            self._next_command_id += 1
            for rank, process in self._processes.items():
                if process.is_alive():
                    try:
                        self._command_queues[rank].put(_WorkerCommand(command_id, "shutdown"), timeout=1.0)
                    except BaseException:  # noqa: BLE001 - force shutdown is the only safe recovery path
                        force = True
                        break
            if not force:
                try:
                    self._collect_results(
                        command_id,
                        set(self._processes),
                        timeout_seconds=self.shutdown_timeout_seconds,
                    )
                except BaseException:  # noqa: BLE001 - failed graceful shutdown must not block exact child cleanup
                    force = True

        if force:
            # A peer may be blocked inside a collective that can never finish
            # after partial command dispatch.  Terminate only these precisely
            # tracked child PIDs immediately instead of paying one full
            # graceful timeout per rank before taking the same action.
            for process in self._processes.values():
                if process.is_alive():
                    process.terminate()
        # The bounded optimizer subgroup is never reused after shutdown.  In
        # a failure path its communicator may be in an aborted state, so drop
        # the rank-zero handle before waiting on children.  This is separate
        # from the default startup group, whose longer timeout must not govern
        # an already-failed optimizer boundary.
        self._destroy_optimizer_process_group()
        for process in self._processes.values():
            process.join(timeout=self.shutdown_timeout_seconds)
        for process in self._processes.values():
            if process.is_alive():
                # A deadlocked peer cannot safely own an adapter copy or keep
                # a GPU allocation after the public job exits. ``terminate``
                # acts only on these exact child PIDs; no filesystem cleanup is
                # involved and diagnostics remain in their existing log dirs.
                process.terminate()
                process.join(timeout=self.shutdown_timeout_seconds)
        self._processes.clear()
        self._command_queues.clear()
        if self._result_queue is not None:
            try:
                self._result_queue.close()
            except BaseException:  # noqa: BLE001, S110 - queue close must not prevent process-group teardown
                pass
            self._result_queue = None
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except BaseException:  # noqa: BLE001, S110 - preserve original diagnostics while releasing the group
            pass

    def _destroy_optimizer_process_group(self) -> None:
        """Best-effort release of rank zero's bounded optimizer subgroup."""

        process_group = self._optimizer_process_group
        self._optimizer_process_group = None
        if process_group is None:
            return
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group(process_group)
        except BaseException:  # noqa: BLE001, S110 - original replica failure remains authoritative
            pass


__all__ = [
    "LocalBackendConstructorSpec",
    "ReplicatedBackendError",
    "ReplicatedBackendPoisonedError",
    "ReplicatedTrainingBackend",
    "ReplicatedWorkerResult",
    "aggregate_sharded_metrics",
    "datum_token_cost",
    "trainable_state_hash",
]
