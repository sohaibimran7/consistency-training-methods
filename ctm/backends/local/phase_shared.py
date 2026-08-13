"""GPU-count-independent contracts for phase-shared local training.

The rollout and optimization phases are intentionally described by two
independent logical-GPU sets.  They may be disjoint (the historical layout),
partially overlapping, or identical (all GPUs serve both phases).  Keeping the
resolver pure and CUDA-discovery-free makes the same contract usable for two,
four, eight, or any other number of GPUs inside an explicit allocation.

The module deliberately separates *planning* from *runtime orchestration*:

* topology and token-cost sharding are deterministic, CPU-only pure helpers;
* :class:`PhaseSharedController` records the only valid lifecycle transitions
  and the canonical-publisher contract; and
* ``PhaseSharedCollectives`` is a small adapter protocol for a future
  ``torch.distributed`` integration.

It does **not** launch rank processes, sleep vLLM engines, construct DDP/FSDP,
or move model weights.  Those actions must be performed and acknowledged by
the backend layer before it confirms the corresponding controller transition.
This fail-closed boundary is intentional: a local state-machine transition is
not evidence that a remote worker actually released GPU memory.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@dataclass(frozen=True)
class PhaseGPU:
    """One logical GPU within the inherited ``CUDA_VISIBLE_DEVICES`` list."""

    logical_index: int
    device_token: str


@dataclass(frozen=True)
class PhaseSharedTopology:
    """Resolved trainer and rollout placements for a phase-shared run."""

    visible_devices: tuple[str, ...]
    train_gpus: tuple[PhaseGPU, ...]
    rollout_gpus: tuple[PhaseGPU, ...]

    @property
    def world_size(self) -> int:
        """Number of replicated training ranks."""

        return len(self.train_gpus)

    @property
    def overlap(self) -> tuple[PhaseGPU, ...]:
        """Training GPUs that are also used for rollout generation."""

        rollout_indices = {gpu.logical_index for gpu in self.rollout_gpus}
        return tuple(gpu for gpu in self.train_gpus if gpu.logical_index in rollout_indices)

    @property
    def coordinator(self) -> PhaseGPU:
        """Canonical policy publisher (the first explicitly ordered train GPU)."""

        return self.train_gpus[0]

    @property
    def training_ranks(self) -> tuple[TrainingRank, ...]:
        """Logical replicated-training ranks in the explicit training order.

        Rank ``0`` means *the first training GPU in this topology*, not CUDA
        device ``0``.  For example, a ``train_gpus_spec`` of ``"3,1"`` makes
        logical GPU 3 the rank-0 publisher and logical GPU 1 rank 1.
        """

        return resolve_training_ranks(self)


def _visible_devices(raw: str | None) -> tuple[str, ...]:
    value = (raw or "").strip()
    if not value or value in {"-1", "NoDevFiles"}:
        raise ValueError("phase-shared execution requires an explicit non-empty CUDA_VISIBLE_DEVICES allocation")
    devices = tuple(token.strip() for token in value.split(","))
    if any(not token for token in devices):
        raise ValueError(f"invalid CUDA_VISIBLE_DEVICES={value!r}")
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate device tokens")
    return devices


def _resolve_gpu_spec(
    spec: str | None,
    *,
    label: str,
    visible_devices: tuple[str, ...],
) -> tuple[PhaseGPU, ...]:
    raw = "all" if spec is None else spec.strip()
    if not raw:
        raise ValueError(f"{label} GPU list must not be empty")
    if raw.lower() == "all":
        indices = list(range(len(visible_devices)))
    else:
        pieces = [piece.strip() for piece in raw.split(",")]
        if any(not piece for piece in pieces):
            raise ValueError(f"invalid {label} GPU list: {spec!r}")
        try:
            indices = [int(piece) for piece in pieces]
        except ValueError as exc:
            raise ValueError(f"{label} GPUs must be 'all' or comma-separated logical integer indices") from exc
    if not indices:
        raise ValueError(f"{label} GPU list must not be empty")
    if any(index < 0 for index in indices):
        raise ValueError(f"{label} GPU logical indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label} GPU logical indices must be unique")
    outside = [index for index in indices if index >= len(visible_devices)]
    if outside:
        raise ValueError(
            f"{label} logical GPU index/indices {outside} are outside CUDA_VISIBLE_DEVICES "
            f"({len(visible_devices)} device(s))"
        )
    return tuple(PhaseGPU(index, visible_devices[index]) for index in indices)


def resolve_phase_shared_topology(
    *,
    train_gpus_spec: str | None,
    rollout_gpus_spec: str | None,
    cuda_visible_devices: str | None,
    allow_overlap: bool = True,
) -> PhaseSharedTopology:
    """Resolve an arbitrary-size trainer/rollout topology without probing CUDA.

    ``None`` and ``"all"`` both select every inherited visible device.  Specs
    are logical indices relative to ``CUDA_VISIBLE_DEVICES`` and retain their
    explicit order; the first training entry is therefore the canonical policy
    publisher.  Overlap is expected for phase sharing but can be prohibited for
    layouts whose two phases must remain simultaneously resident.
    """

    visible = _visible_devices(cuda_visible_devices)
    train = _resolve_gpu_spec(train_gpus_spec, label="training", visible_devices=visible)
    rollout = _resolve_gpu_spec(rollout_gpus_spec, label="rollout", visible_devices=visible)
    overlap = {gpu.logical_index for gpu in train} & {gpu.logical_index for gpu in rollout}
    if overlap and not allow_overlap:
        raise ValueError(
            "training and rollout GPU sets overlap while overlap is disabled: "
            + ",".join(str(index) for index in sorted(overlap))
        )
    return PhaseSharedTopology(
        visible_devices=visible,
        train_gpus=train,
        rollout_gpus=rollout,
    )


@dataclass(frozen=True)
class TrainingRank:
    """A replicated-training rank and its topology-selected GPU.

    The rank number is local to the training process group.  It never implies
    a physical device ordinal, so the same plan works with arbitrary
    ``CUDA_VISIBLE_DEVICES`` orderings and world sizes.
    """

    rank: int
    gpu: PhaseGPU

    @property
    def is_publisher(self) -> bool:
        """Whether this rank owns canonical adapter publication."""

        return self.rank == 0


def resolve_training_ranks(topology: PhaseSharedTopology) -> tuple[TrainingRank, ...]:
    """Return the ordered rank-to-GPU mapping for ``topology``.

    This is intentionally a pure conversion rather than a CUDA or distributed
    discovery operation.  A launcher must still create exactly this many
    processes and make their distributed rank agree with the returned mapping.
    """

    return tuple(TrainingRank(rank=index, gpu=gpu) for index, gpu in enumerate(topology.train_gpus))


def _validate_rank_layout(ranks: Sequence[TrainingRank]) -> tuple[TrainingRank, ...]:
    resolved = tuple(ranks)
    if not resolved:
        raise ValueError("token-cost sharding requires at least one training rank")
    expected = tuple(range(len(resolved)))
    actual = tuple(rank.rank for rank in resolved)
    if actual != expected:
        raise ValueError(
            "training ranks must be ordered contiguously from 0; " f"got {actual!r}, expected {expected!r}"
        )
    gpu_indices = tuple(rank.gpu.logical_index for rank in resolved)
    if len(set(gpu_indices)) != len(gpu_indices):
        raise ValueError("training ranks must map to distinct logical GPUs")
    return resolved


def _validate_token_cost(cost: Any, *, index: int) -> int:
    if isinstance(cost, bool) or not isinstance(cost, Integral):
        raise TypeError(f"token cost at original index {index} must be an integer, got {type(cost).__name__}")
    value = int(cost)
    if value < 0:
        raise ValueError(f"token cost at original index {index} must be non-negative, got {value}")
    return value


@dataclass(frozen=True)
class IndexedWorkItem(Generic[T]):
    """One logical work item with stable input position and estimated cost."""

    original_index: int
    token_cost: int
    value: T

    def __post_init__(self) -> None:
        if isinstance(self.original_index, bool) or not isinstance(self.original_index, Integral):
            raise TypeError("original_index must be an integer")
        if int(self.original_index) < 0:
            raise ValueError("original_index must be non-negative")
        _validate_token_cost(self.token_cost, index=int(self.original_index))


@dataclass(frozen=True)
class TokenCostShard(Generic[T]):
    """One rank's deterministic token-cost-balanced work assignment."""

    training_rank: TrainingRank
    items: tuple[IndexedWorkItem[T], ...]
    token_cost: int

    def __post_init__(self) -> None:
        if self.token_cost < 0:
            raise ValueError("shard token_cost must be non-negative")
        actual = sum(item.token_cost for item in self.items)
        if self.token_cost != actual:
            raise ValueError(f"shard token_cost={self.token_cost} does not match contained cost={actual}")
        indices = self.original_indices
        if len(set(indices)) != len(indices):
            raise ValueError("a shard cannot contain an original index more than once")
        if indices != tuple(sorted(indices)):
            raise ValueError("shard items must be restored to ascending original-index order")

    @property
    def rank(self) -> int:
        return self.training_rank.rank

    @property
    def original_indices(self) -> tuple[int, ...]:
        """Input positions owned by this rank, in canonical restore order."""

        return tuple(item.original_index for item in self.items)


def plan_token_cost_balanced_shards(
    values: Sequence[T],
    token_costs: Sequence[int],
    *,
    ranks: Sequence[TrainingRank],
) -> tuple[TokenCostShard[T], ...]:
    """Deterministically assign logical work to arbitrary replicated ranks.

    The assignment uses stable longest-processing-time first (LPT): items are
    considered by descending token cost with original index as the tie-breaker;
    each is assigned to the least-loaded rank, with rank as the final
    tie-breaker.  This makes plans reproducible and balances highly variable
    completion lengths better than contiguous question sharding.

    Items within each returned shard are put back in original-index order.  The
    latter matters because callers can safely reconstruct public API outputs
    even when rank execution completes in a different order.
    """

    resolved_ranks = _validate_rank_layout(ranks)
    if len(values) != len(token_costs):
        raise ValueError(
            "values and token_costs must have the same length, " f"got {len(values)} and {len(token_costs)}"
        )

    items = tuple(
        IndexedWorkItem(original_index=index, token_cost=_validate_token_cost(cost, index=index), value=value)
        for index, (value, cost) in enumerate(zip(values, token_costs))
    )
    assignments: list[list[IndexedWorkItem[T]]] = [[] for _ in resolved_ranks]
    loads = [0 for _ in resolved_ranks]

    # Stable LPT.  ``min`` visits ranks in order, so the explicit rank in the
    # key gives a deterministic tie-break even if the implementation changes.
    for item in sorted(items, key=lambda candidate: (-candidate.token_cost, candidate.original_index)):
        target = min(range(len(resolved_ranks)), key=lambda rank: (loads[rank], resolved_ranks[rank].rank))
        assignments[target].append(item)
        loads[target] += item.token_cost

    return tuple(
        TokenCostShard(
            training_rank=rank,
            items=tuple(sorted(assignment, key=lambda item: item.original_index)),
            token_cost=loads[position],
        )
        for position, (rank, assignment) in enumerate(zip(resolved_ranks, assignments))
    )


def plan_token_cost_balanced_indices(
    token_costs: Sequence[int],
    *,
    world_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Return just the deterministic index assignment for a logical world size.

    This convenience form is useful in CPU-only tests and deliberately does
    not invent CUDA identities.  Production callers should use
    :func:`plan_token_cost_balanced_shards` with topology-derived ranks.
    """

    if isinstance(world_size, bool) or not isinstance(world_size, Integral) or int(world_size) <= 0:
        raise ValueError("world_size must be a positive integer")
    size = int(world_size)
    synthetic_ranks = tuple(
        TrainingRank(rank=index, gpu=PhaseGPU(logical_index=index, device_token=f"rank-{index}"))
        for index in range(size)
    )
    shards = plan_token_cost_balanced_shards(tuple(range(len(token_costs))), token_costs, ranks=synthetic_ranks)
    return tuple(shard.original_indices for shard in shards)


@dataclass(frozen=True)
class IndexedResult(Generic[T]):
    """A rank-produced value labelled with the original logical input index."""

    original_index: int
    value: T

    def __post_init__(self) -> None:
        if isinstance(self.original_index, bool) or not isinstance(self.original_index, Integral):
            raise TypeError("original_index must be an integer")
        if int(self.original_index) < 0:
            raise ValueError("original_index must be non-negative")


def index_shard_results(shard: TokenCostShard[Any], values: Sequence[T]) -> tuple[IndexedResult[T], ...]:
    """Attach the shard's canonical indices to in-shard-order results.

    This helper is deliberately strict.  It is safe only when the rank itself
    preserved the returned shard order; otherwise the rank should emit
    :class:`IndexedResult` directly and let :func:`restore_original_order`
    validate it.
    """

    if len(values) != len(shard.items):
        raise ValueError(
            f"rank {shard.rank} returned {len(values)} result(s) for a shard containing {len(shard.items)} item(s)"
        )
    return tuple(IndexedResult(item.original_index, value) for item, value in zip(shard.items, values))


def restore_original_order(
    indexed_results: Iterable[IndexedResult[T]],
    *,
    expected_count: int | None = None,
) -> list[T]:
    """Validate and restore results to their original input order.

    Missing, duplicate, negative, or non-contiguous indices are errors rather
    than silently producing a reordered or partially filled result.  This is
    especially important after an asynchronous multi-rank execution.
    """

    values_by_index: dict[int, T] = {}
    for result in indexed_results:
        if not isinstance(result, IndexedResult):
            raise TypeError("restore_original_order requires IndexedResult values")
        index = int(result.original_index)
        if index in values_by_index:
            raise ValueError(f"duplicate result for original index {index}")
        values_by_index[index] = result.value

    if expected_count is None:
        count = 0 if not values_by_index else max(values_by_index) + 1
    else:
        if isinstance(expected_count, bool) or not isinstance(expected_count, Integral) or int(expected_count) < 0:
            raise ValueError("expected_count must be a non-negative integer")
        count = int(expected_count)
    expected = set(range(count))
    actual = set(values_by_index)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing original indices {missing}")
        if unexpected:
            details.append(f"unexpected original indices {unexpected}")
        raise ValueError("cannot restore original order: " + "; ".join(details))
    return [values_by_index[index] for index in range(count)]


def restore_sharded_results(
    shards: Sequence[TokenCostShard[Any]],
    results_by_rank: Mapping[int, Iterable[IndexedResult[T]]],
) -> list[T]:
    """Restore rank results while checking each rank returned only its work.

    The function accepts rank-local completion order, but makes accidental
    cross-rank result mixing, omissions, and duplicates visible immediately.
    """

    expected_ranks = {shard.rank for shard in shards}
    actual_ranks = set(results_by_rank)
    if actual_ranks != expected_ranks:
        raise ValueError(
            "results_by_rank must contain exactly the planned ranks; "
            f"missing={sorted(expected_ranks - actual_ranks)}, extra={sorted(actual_ranks - expected_ranks)}"
        )
    expected_by_rank = {shard.rank: set(shard.original_indices) for shard in shards}
    combined: list[IndexedResult[T]] = []
    for rank in sorted(expected_ranks):
        rank_results = tuple(results_by_rank[rank])
        actual_indices: set[int] = set()
        for result in rank_results:
            if not isinstance(result, IndexedResult):
                raise TypeError("results_by_rank values must contain IndexedResult values")
            if result.original_index in actual_indices:
                raise ValueError(f"rank {rank} returned duplicate original index {result.original_index}")
            actual_indices.add(result.original_index)
        if actual_indices != expected_by_rank[rank]:
            raise ValueError(
                f"rank {rank} returned indices {sorted(actual_indices)}, " f"expected {sorted(expected_by_rank[rank])}"
            )
        combined.extend(rank_results)
    return restore_original_order(combined, expected_count=sum(len(shard.items) for shard in shards))


class PhaseSharedLifecycle(str, Enum):
    """Explicit controller lifecycle states.

    The transitional states intentionally make the external sleep/wake and
    adapter-acknowledgement barriers visible.  A controller can enter training
    only after the backend confirms the rollout pool is quiesced; it returns to
    rollout only after every worker has acknowledged the published policy.
    """

    NEW = "new"
    ROLLOUT = "rollout"
    QUIESCING_ROLLOUT = "quiescing_rollout"
    TRAINING = "training"
    PUBLISHING = "publishing"
    WAKING_ROLLOUT = "waking_rollout"
    FAILED = "failed"
    CLOSED = "closed"


class PhaseSharedRuntimeError(RuntimeError):
    """An unsafe lifecycle or collective operation was attempted."""


@dataclass(frozen=True)
class PhaseTransition:
    """A pure controller transition and the external action it requires."""

    previous: PhaseSharedLifecycle
    current: PhaseSharedLifecycle
    adapter_version: int
    required_runtime_action: str


@dataclass(frozen=True)
class PolicyPublication:
    """Rank-0's immutable declaration of the adapter made canonical.

    ``artifact_reference`` is deliberately opaque: it can be a filesystem path,
    content hash, object-store URI, or adapter manifest identifier.  The
    backend must validate and load the referenced artifact before acknowledging
    the wake barrier.
    """

    adapter_version: int
    publisher_rank: int
    artifact_reference: str

    def __post_init__(self) -> None:
        if isinstance(self.adapter_version, bool) or not isinstance(self.adapter_version, Integral):
            raise TypeError("adapter_version must be an integer")
        if int(self.adapter_version) < 0:
            raise ValueError("adapter_version must be non-negative")
        if self.publisher_rank != 0:
            raise ValueError("only logical training rank 0 may publish a canonical adapter")
        if not isinstance(self.artifact_reference, str) or not self.artifact_reference.strip():
            raise ValueError("artifact_reference must be a non-empty string")


class GradientReduction(str, Enum):
    """How replicated parameter gradients are combined after local backward.

    ``SUM`` is the explicit LocalBackend contract: every rank uses the shared
    global denominator, then its gradient reducer SUMs LoRA gradients once at
    the optimizer boundary.  ``MEAN`` is for a conventional DDP wrapper which
    averages gradients automatically.  The difference is a factor of world
    size, so it is always explicit rather than inferred from a process group.
    """

    SUM = "sum"
    MEAN = "mean"


@runtime_checkable
class PhaseSharedCollectives(Protocol):
    """Minimal rank transport expected by :class:`PhaseSharedController`.

    A thin ``torch.distributed`` adapter can implement this protocol with
    ``dist.all_reduce(..., ReduceOp.SUM)``, ``dist.barrier()``, and
    ``dist.broadcast_object_list()``.  Tags are observability/call-order names;
    an adapter may record them even if its underlying collective API has no tag
    argument.

    The controller intentionally does not import torch.  That keeps topology
    planning and CPU-only tests independent of CUDA and makes launch ownership
    explicit in the backend.
    """

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def all_reduce_sum(self, value: float, *, tag: str) -> float: ...

    def barrier(self, *, tag: str) -> None: ...

    def broadcast_object(self, value: Any | None, *, src_rank: int, tag: str) -> Any: ...


@dataclass(frozen=True)
class GlobalDenominator:
    """The exact global normalizer for a manually SUM-reduced loss.

    With the LocalBackend's explicit optimizer-boundary ``SUM`` reducer, each
    rank backpropagates its local *sum* numerator divided by this one global
    denominator.  Summing gradients then exactly yields the logical global
    ``sum / denominator`` objective.  A conventional DDP wrapper averages
    gradients instead, so it needs an explicit world-size multiplier.  Internal
    forward microbatches must reuse this same object rather than normalizing
    independently.
    """

    local_denominator: float
    global_denominator: float
    normalizer: float
    world_size: int
    tag: str
    gradient_reduction: GradientReduction = GradientReduction.SUM

    @property
    def local_sum_scale(self) -> float:
        """Scale each local numerator before backward under the configured reducer."""

        if self.gradient_reduction is GradientReduction.SUM:
            return 1.0 / self.normalizer
        return self.world_size / self.normalizer

    @property
    def manual_sum_scale(self) -> float:
        """Scale required by LocalBackend's explicit optimizer-boundary SUM reducer."""

        return 1.0 / self.normalizer

    @property
    def ddp_mean_scale(self) -> float:
        """Scale required when a DDP wrapper averages gradients across ranks."""

        return self.world_size / self.normalizer

    @property
    def ddp_sum_scale(self) -> float:
        """Backward-compatible alias for :attr:`ddp_mean_scale`.

        The name predates the explicit reduction-mode distinction.  New code
        should use ``manual_sum_scale`` for LocalBackend's manual SUM reducer
        or ``ddp_mean_scale`` for conventional averaging DDP.
        """

        return self.ddp_mean_scale

    def scale_local_sum(self, local_sum: T, *, negate: bool = False) -> T:
        """Return the tensor/scalar loss contribution for one local sum.

        ``local_sum`` can be a torch tensor without this module importing
        torch.  Set ``negate=True`` for objectives represented as a reward or
        PPO surrogate that the optimizer should minimize as ``-sum / denom``.
        """

        sign = -1.0 if negate else 1.0
        return local_sum * (sign * self.local_sum_scale)  # type: ignore[operator]


@dataclass(frozen=True)
class ReducedSums:
    """Named local and global SUM reductions for logging and exact metrics."""

    local: Mapping[str, float]
    global_: Mapping[str, float]
    tag: str


def _as_finite_float(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite scalar number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if nonnegative and result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _collective_rank(collectives: PhaseSharedCollectives) -> int:
    rank = getattr(collectives, "rank", None)
    if isinstance(rank, bool) or not isinstance(rank, Integral) or int(rank) < 0:
        raise PhaseSharedRuntimeError("collective adapter must expose a non-negative integer rank")
    return int(rank)


def _collective_world_size(collectives: PhaseSharedCollectives) -> int:
    world_size = getattr(collectives, "world_size", None)
    if isinstance(world_size, bool) or not isinstance(world_size, Integral) or int(world_size) <= 0:
        raise PhaseSharedRuntimeError("collective adapter must expose a positive integer world_size")
    return int(world_size)


def reduce_named_sums(
    collectives: PhaseSharedCollectives,
    local_sums: Mapping[str, float],
    *,
    tag: str,
) -> ReducedSums:
    """All-reduce named scalar sums in deterministic key order.

    Every rank must call this with the same keys and tag in the same phase.  A
    backend should derive names statically from the loss configuration; silently
    allowing rank-specific metric keys could deadlock a real collective group.
    """

    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("collective tag must be a non-empty string")
    _collective_rank(collectives)
    _collective_world_size(collectives)
    ordered = tuple(sorted(local_sums))
    if len(set(ordered)) != len(ordered):  # defensive for unusual Mapping implementations
        raise ValueError("local_sums keys must be unique")
    local = {name: _as_finite_float(local_sums[name], label=f"local_sums[{name!r}]") for name in ordered}
    global_sums: dict[str, float] = {}
    for name in ordered:
        reduced = collectives.all_reduce_sum(local[name], tag=f"{tag}/sum/{name}")
        global_sums[name] = _as_finite_float(reduced, label=f"global sum for {name!r}")
    return ReducedSums(local=local, global_=global_sums, tag=tag)


def prepare_global_denominator(
    collectives: PhaseSharedCollectives,
    local_denominator: float,
    *,
    tag: str,
    minimum: float = 1e-8,
    gradient_reduction: GradientReduction | str = GradientReduction.SUM,
) -> GlobalDenominator:
    """SUM-reduce a token denominator before local forward/backward work.

    ``minimum`` mirrors the existing local backend's empty-batch guard.  The
    un-clamped global denominator is retained for metrics, while ``normalizer``
    is what loss scaling uses.  ``gradient_reduction`` defaults to ``"sum"``
    because LocalBackend's forthcoming reducer SUMs gradients at the optimizer
    boundary; choose ``"mean"`` only when a DDP wrapper averages them.  This
    operation is intentionally separate from reduction of the post-backward
    numerator: the denominator is needed before each rank constructs its local
    loss graph.
    """

    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("collective tag must be a non-empty string")
    local = _as_finite_float(local_denominator, label="local_denominator", nonnegative=True)
    lower_bound = _as_finite_float(minimum, label="minimum", nonnegative=True)
    if lower_bound <= 0:
        raise ValueError("minimum must be greater than zero")
    try:
        reduction = GradientReduction(gradient_reduction)
    except (TypeError, ValueError) as exc:
        raise ValueError("gradient_reduction must be 'sum' or 'mean'") from exc
    world_size = _collective_world_size(collectives)
    _collective_rank(collectives)
    global_value = _as_finite_float(
        collectives.all_reduce_sum(local, tag=f"{tag}/sum/denominator"),
        label="global_denominator",
        nonnegative=True,
    )
    return GlobalDenominator(
        local_denominator=local,
        global_denominator=global_value,
        normalizer=max(global_value, lower_bound),
        world_size=world_size,
        tag=tag,
        gradient_reduction=reduction,
    )


class PhaseSharedController:
    """Fail-closed local controller for one replicated-training rank.

    The class is deliberately safe to instantiate once per distributed rank.
    Rank 0 is the canonical *training rank* publisher, derived from the
    topology's explicit order; it is never assumed to be physical GPU 0.  The
    outer backend remains responsible for process launch, actual vLLM
    sleep/wake calls, DDP/FSDP construction, and error propagation.

    A typical runtime sequence is::

        controller.start_rollout()
        controller.request_training()       # backend sleeps/acks all workers
        controller.confirm_rollouts_quiesced()
        denominator = controller.prepare_global_denominator(...)
        ... local SUM losses scaled with denominator.local_sum_scale ...
        controller.complete_training()
        publication = controller.make_canonical_publication(... on rank 0 ...)
        controller.broadcast_and_accept_publication(publication)
        ... backend wakes/loads/acks all workers ...
        controller.confirm_rollouts_ready()

    Any failed external barrier should call :meth:`fail`; the controller never
    guesses that a partially completed transition is safe to resume.
    """

    def __init__(
        self,
        topology: PhaseSharedTopology,
        *,
        rank: int = 0,
        collectives: PhaseSharedCollectives | None = None,
        initial_adapter_version: int = 0,
    ) -> None:
        self.topology = topology
        self._ranks = _validate_rank_layout(topology.training_ranks)
        if isinstance(rank, bool) or not isinstance(rank, Integral) or not 0 <= int(rank) < len(self._ranks):
            raise ValueError(f"rank must be in [0, {len(self._ranks)}), got {rank!r}")
        if isinstance(initial_adapter_version, bool) or not isinstance(initial_adapter_version, Integral):
            raise TypeError("initial_adapter_version must be an integer")
        if int(initial_adapter_version) < 0:
            raise ValueError("initial_adapter_version must be non-negative")
        self._rank = int(rank)
        self._collectives = collectives
        self._adapter_version = int(initial_adapter_version)
        self._state = PhaseSharedLifecycle.NEW
        self._failure_reason: str | None = None
        if collectives is not None:
            self._validate_collectives(collectives)

    @property
    def rank(self) -> TrainingRank:
        """This process's logical training rank and selected GPU."""

        return self._ranks[self._rank]

    @property
    def rank_index(self) -> int:
        """This process's integer training rank."""

        return self._rank

    @property
    def world_size(self) -> int:
        return len(self._ranks)

    @property
    def publisher(self) -> TrainingRank:
        """The one rank allowed to declare a canonical adapter publication."""

        return self._ranks[0]

    @property
    def is_publisher(self) -> bool:
        return self.rank.is_publisher

    @property
    def state(self) -> PhaseSharedLifecycle:
        return self._state

    @property
    def adapter_version(self) -> int:
        return self._adapter_version

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def collectives(self) -> PhaseSharedCollectives | None:
        """The injected adapter, if this controller participates in a group."""

        return self._collectives

    def plan_token_cost_shards(
        self,
        values: Sequence[T],
        token_costs: Sequence[int],
    ) -> tuple[TokenCostShard[T], ...]:
        """Plan CPU-only work assignments for this topology's training group."""

        return plan_token_cost_balanced_shards(values, token_costs, ranks=self._ranks)

    def start_rollout(self) -> PhaseTransition:
        """Enter the initial ready-to-rollout state after an external load ack."""

        return self._transition(
            allowed={PhaseSharedLifecycle.NEW},
            target=PhaseSharedLifecycle.ROLLOUT,
            action="serve_rollouts",
        )

    def request_training(self) -> PhaseTransition:
        """Request the external vLLM sleep/quiesce barrier."""

        return self._transition(
            allowed={PhaseSharedLifecycle.ROLLOUT},
            target=PhaseSharedLifecycle.QUIESCING_ROLLOUT,
            action="sleep_and_quiesce_rollout_workers",
        )

    def confirm_rollouts_quiesced(self) -> PhaseTransition:
        """Enter training only after every relevant rollout worker acknowledged sleep."""

        return self._transition(
            allowed={PhaseSharedLifecycle.QUIESCING_ROLLOUT},
            target=PhaseSharedLifecycle.TRAINING,
            action="run_replicated_training",
        )

    def complete_training(self) -> PhaseTransition:
        """Move from training to the rank-0 canonical-publication barrier."""

        return self._transition(
            allowed={PhaseSharedLifecycle.TRAINING},
            target=PhaseSharedLifecycle.PUBLISHING,
            action="publish_rank_zero_adapter",
        )

    def make_canonical_publication(self, artifact_reference: str) -> PolicyPublication:
        """Let only rank 0 nominate the next adapter version for publication."""

        self._require_state({PhaseSharedLifecycle.PUBLISHING})
        if not self.is_publisher:
            raise PhaseSharedRuntimeError(
                f"rank {self._rank} cannot publish; canonical publisher is rank {self.publisher.rank} "
                f"on logical GPU {self.publisher.gpu.logical_index}"
            )
        return PolicyPublication(
            adapter_version=self._adapter_version + 1,
            publisher_rank=self.publisher.rank,
            artifact_reference=artifact_reference,
        )

    def accept_publication(self, publication: PolicyPublication) -> PhaseTransition:
        """Validate one rank-0 publication before asking workers to wake/load it."""

        self._require_state({PhaseSharedLifecycle.PUBLISHING})
        if not isinstance(publication, PolicyPublication):
            raise TypeError("publication must be a PolicyPublication")
        if publication.publisher_rank != self.publisher.rank:
            raise PhaseSharedRuntimeError("publication did not originate from the canonical rank")
        expected_version = self._adapter_version + 1
        if publication.adapter_version != expected_version:
            raise PhaseSharedRuntimeError(
                f"publication version {publication.adapter_version} is invalid; expected {expected_version}"
            )
        self._adapter_version = publication.adapter_version
        return self._transition(
            allowed={PhaseSharedLifecycle.PUBLISHING},
            target=PhaseSharedLifecycle.WAKING_ROLLOUT,
            action="wake_rollout_workers_and_ack_adapter",
        )

    def broadcast_and_accept_publication(self, publication: PolicyPublication | None = None) -> PolicyPublication:
        """Broadcast rank-0's publication through the injected rank transport.

        This is a coordination hook, not a distributed launcher.  All ranks
        must call it in the same lifecycle state.  A missing or malformed
        broadcast fails closed rather than falling back to a local publication.
        """

        self._require_state({PhaseSharedLifecycle.PUBLISHING})
        collectives = self._require_collectives()
        if self.is_publisher:
            if publication is None:
                raise PhaseSharedRuntimeError("publisher rank must provide a PolicyPublication to broadcast")
            if not isinstance(publication, PolicyPublication):
                raise TypeError("publisher must broadcast a PolicyPublication")
        elif publication is not None:
            raise PhaseSharedRuntimeError("non-publisher ranks must not supply a local publication")
        received = collectives.broadcast_object(
            publication if self.is_publisher else None,
            src_rank=self.publisher.rank,
            tag=self._collective_tag("publication"),
        )
        if not isinstance(received, PolicyPublication):
            raise PhaseSharedRuntimeError("publication broadcast returned no valid PolicyPublication")
        self.accept_publication(received)
        return received

    def confirm_rollouts_ready(self) -> PhaseTransition:
        """Return to rollout only after all workers loaded the exact publication."""

        return self._transition(
            allowed={PhaseSharedLifecycle.WAKING_ROLLOUT},
            target=PhaseSharedLifecycle.ROLLOUT,
            action="serve_rollouts",
        )

    def prepare_global_denominator(
        self,
        local_denominator: float,
        *,
        label: str = "loss",
        minimum: float = 1e-8,
        gradient_reduction: GradientReduction | str = GradientReduction.SUM,
    ) -> GlobalDenominator:
        """Create the exact replicated-training normalization context."""

        self._require_state({PhaseSharedLifecycle.TRAINING})
        return prepare_global_denominator(
            self._require_collectives(),
            local_denominator,
            tag=self._collective_tag(f"denominator/{label}"),
            minimum=minimum,
            gradient_reduction=gradient_reduction,
        )

    def reduce_named_sums(self, local_sums: Mapping[str, float], *, label: str = "metrics") -> ReducedSums:
        """Reduce exact scalar SUMs during training in a stable call order."""

        self._require_state({PhaseSharedLifecycle.TRAINING})
        return reduce_named_sums(
            self._require_collectives(),
            local_sums,
            tag=self._collective_tag(f"metrics/{label}"),
        )

    def barrier(self, *, label: str) -> None:
        """Call an explicit backend-owned collective barrier with a stable tag."""

        if not isinstance(label, str) or not label.strip():
            raise ValueError("barrier label must be a non-empty string")
        self._require_state(
            {
                PhaseSharedLifecycle.QUIESCING_ROLLOUT,
                PhaseSharedLifecycle.TRAINING,
                PhaseSharedLifecycle.PUBLISHING,
                PhaseSharedLifecycle.WAKING_ROLLOUT,
            }
        )
        self._require_collectives().barrier(tag=self._collective_tag(f"barrier/{label}"))

    def fail(self, reason: str) -> PhaseTransition:
        """Make a failed external barrier terminal for this controller instance."""

        if self._state is PhaseSharedLifecycle.CLOSED:
            raise PhaseSharedRuntimeError("cannot fail a closed phase-shared controller")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failure reason must be a non-empty string")
        previous = self._state
        self._state = PhaseSharedLifecycle.FAILED
        self._failure_reason = reason.strip()
        return PhaseTransition(
            previous=previous,
            current=self._state,
            adapter_version=self._adapter_version,
            required_runtime_action="stop_and_preserve_diagnostics",
        )

    def close(self) -> PhaseTransition:
        """Close the local protocol object after the outer backend tears down safely."""

        if self._state is PhaseSharedLifecycle.CLOSED:
            return PhaseTransition(
                previous=self._state,
                current=self._state,
                adapter_version=self._adapter_version,
                required_runtime_action="none",
            )
        previous = self._state
        self._state = PhaseSharedLifecycle.CLOSED
        return PhaseTransition(
            previous=previous,
            current=self._state,
            adapter_version=self._adapter_version,
            required_runtime_action="release_backend_resources",
        )

    def _validate_collectives(self, collectives: PhaseSharedCollectives) -> None:
        rank = _collective_rank(collectives)
        world_size = _collective_world_size(collectives)
        if rank != self._rank:
            raise ValueError(f"collective adapter rank {rank} does not match controller rank {self._rank}")
        if world_size != self.world_size:
            raise ValueError(
                f"collective adapter world_size {world_size} does not match topology world_size {self.world_size}"
            )

    def _require_collectives(self) -> PhaseSharedCollectives:
        if self._collectives is None:
            raise PhaseSharedRuntimeError(
                "this operation needs a PhaseSharedCollectives adapter; "
                "topology planning alone cannot synchronize distributed ranks"
            )
        self._validate_collectives(self._collectives)
        return self._collectives

    def _require_state(self, allowed: set[PhaseSharedLifecycle]) -> None:
        if self._state not in allowed:
            allowed_text = ", ".join(state.value for state in sorted(allowed, key=lambda state: state.value))
            raise PhaseSharedRuntimeError(
                f"operation is unsafe in lifecycle state {self._state.value!r}; expected one of {allowed_text}"
            )

    def _transition(
        self,
        *,
        allowed: set[PhaseSharedLifecycle],
        target: PhaseSharedLifecycle,
        action: str,
    ) -> PhaseTransition:
        self._require_state(allowed)
        previous = self._state
        self._state = target
        return PhaseTransition(
            previous=previous,
            current=target,
            adapter_version=self._adapter_version,
            required_runtime_action=action,
        )

    def _collective_tag(self, suffix: str) -> str:
        return f"phase-shared/v{self._adapter_version}/{self._state.value}/{suffix}"


__all__ = [
    "GlobalDenominator",
    "GradientReduction",
    "IndexedResult",
    "IndexedWorkItem",
    "PhaseGPU",
    "PhaseSharedCollectives",
    "PhaseSharedController",
    "PhaseSharedLifecycle",
    "PhaseSharedRuntimeError",
    "PhaseSharedTopology",
    "PhaseTransition",
    "PolicyPublication",
    "ReducedSums",
    "TokenCostShard",
    "TrainingRank",
    "index_shard_results",
    "plan_token_cost_balanced_indices",
    "plan_token_cost_balanced_shards",
    "prepare_global_denominator",
    "reduce_named_sums",
    "resolve_phase_shared_topology",
    "resolve_training_ranks",
    "restore_original_order",
    "restore_sharded_results",
]
