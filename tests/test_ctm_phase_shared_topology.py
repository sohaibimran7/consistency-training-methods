"""Contracts for GPU-count-independent phase-shared topology resolution."""

from __future__ import annotations

import pytest

from ctm.backends.local.phase_shared import (
    GradientReduction,
    IndexedResult,
    PhaseGPU,
    PhaseSharedController,
    PhaseSharedLifecycle,
    PhaseSharedRuntimeError,
    PhaseSharedTopology,
    PolicyPublication,
    TrainingRank,
    index_shard_results,
    plan_token_cost_balanced_indices,
    plan_token_cost_balanced_shards,
    prepare_global_denominator,
    reduce_named_sums,
    resolve_phase_shared_topology,
    restore_original_order,
    restore_sharded_results,
)

_UNSET = object()


class _FakeCollectives:
    """CPU-only collective recorder for the phase-shared contracts."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        reduced_value=_UNSET,
        broadcast_value=_UNSET,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self._reduced_value = reduced_value
        self._broadcast_value = broadcast_value
        self.sum_calls: list[tuple[float, str]] = []
        self.barrier_calls: list[str] = []
        self.broadcast_calls: list[tuple[object | None, int, str]] = []

    def all_reduce_sum(self, value: float, *, tag: str) -> float:
        self.sum_calls.append((value, tag))
        if self._reduced_value is _UNSET:
            return value
        if callable(self._reduced_value):
            return self._reduced_value(value, tag)
        return self._reduced_value

    def barrier(self, *, tag: str) -> None:
        self.barrier_calls.append(tag)

    def broadcast_object(self, value, *, src_rank: int, tag: str):
        self.broadcast_calls.append((value, src_rank, tag))
        if self._broadcast_value is _UNSET:
            return value
        return self._broadcast_value


def _visible(count: int) -> str:
    return ",".join(f"GPU-{index}" for index in range(count))


def _pairs(gpus):
    return [(gpu.logical_index, gpu.device_token) for gpu in gpus]


@pytest.mark.parametrize("gpu_count", [2, 4, 8])
def test_default_phase_shared_topology_uses_every_visible_gpu(gpu_count):
    """Two-, four-, and eight-GPU hosts differ only in allocation length."""

    topology = resolve_phase_shared_topology(
        train_gpus_spec=None,
        rollout_gpus_spec="all",
        cuda_visible_devices=_visible(gpu_count),
    )

    assert isinstance(topology, PhaseSharedTopology)
    expected = [(index, f"GPU-{index}") for index in range(gpu_count)]
    assert _pairs(topology.train_gpus) == expected
    assert _pairs(topology.rollout_gpus) == expected
    assert topology.visible_devices == tuple(f"GPU-{index}" for index in range(gpu_count))
    assert topology.world_size == gpu_count
    assert _pairs(topology.overlap) == expected
    assert (topology.coordinator.logical_index, topology.coordinator.device_token) == expected[0]


@pytest.mark.parametrize(
    ("visible", "train", "rollout", "expected_train", "expected_rollout"),
    [
        (
            "GPU-a,MIG-b",
            "1,0",
            "0,1",
            [(1, "MIG-b"), (0, "GPU-a")],
            [(0, "GPU-a"), (1, "MIG-b")],
        ),
        (
            "GPU-a,7,MIG-c,GPU-z",
            "3,1,0,2",
            "2,3,1,0",
            [(3, "GPU-z"), (1, "7"), (0, "GPU-a"), (2, "MIG-c")],
            [(2, "MIG-c"), (3, "GPU-z"), (1, "7"), (0, "GPU-a")],
        ),
        (
            "GPU-a,1,MIG-c,GPU-d,4,GPU-f,MIG-g,7",
            "6,2,0,1,4,7,5,3",
            "7,3,6,0",
            [
                (6, "MIG-g"),
                (2, "MIG-c"),
                (0, "GPU-a"),
                (1, "1"),
                (4, "4"),
                (7, "7"),
                (5, "GPU-f"),
                (3, "GPU-d"),
            ],
            [(7, "7"), (3, "GPU-d"), (6, "MIG-g"), (0, "GPU-a")],
        ),
    ],
    ids=["two-gpu", "four-gpu", "eight-gpu"],
)
def test_explicit_topology_preserves_logical_order_and_does_not_assume_gpu_zero(
    visible,
    train,
    rollout,
    expected_train,
    expected_rollout,
):
    topology = resolve_phase_shared_topology(
        train_gpus_spec=train,
        rollout_gpus_spec=rollout,
        cuda_visible_devices=visible,
    )

    assert _pairs(topology.train_gpus) == expected_train
    assert _pairs(topology.rollout_gpus) == expected_rollout
    assert topology.world_size == len(expected_train)
    assert (topology.coordinator.logical_index, topology.coordinator.device_token) == expected_train[0]
    assert _pairs(topology.overlap) == [
        (logical_index, device_token)
        for logical_index, device_token in expected_train
        if logical_index in {index for index, _ in expected_rollout}
    ]


def test_disjoint_train_and_rollout_sets_are_allowed_only_when_overlap_is_disabled():
    topology = resolve_phase_shared_topology(
        train_gpus_spec="2,0",
        rollout_gpus_spec="3,1",
        cuda_visible_devices="GPU-a,GPU-b,GPU-c,GPU-d",
        allow_overlap=False,
    )

    assert _pairs(topology.train_gpus) == [(2, "GPU-c"), (0, "GPU-a")]
    assert _pairs(topology.rollout_gpus) == [(3, "GPU-d"), (1, "GPU-b")]
    assert topology.overlap == ()
    assert topology.coordinator.logical_index == 2


def test_overlap_can_be_forbidden_explicitly():
    with pytest.raises(ValueError, match="overlap"):
        resolve_phase_shared_topology(
            train_gpus_spec="0,2",
            rollout_gpus_spec="2,3",
            cuda_visible_devices="GPU-a,GPU-b,GPU-c,GPU-d",
            allow_overlap=False,
        )


@pytest.mark.parametrize(
    ("train", "rollout", "visible", "message"),
    [
        (None, None, None, "CUDA_VISIBLE_DEVICES"),
        (None, None, "", "CUDA_VISIBLE_DEVICES"),
        (None, None, "GPU-a,,GPU-b", "invalid CUDA_VISIBLE_DEVICES"),
        (None, None, "GPU-a,GPU-a", "duplicate"),
        ("", "all", "GPU-a,GPU-b", "must not be empty"),
        ("all", "", "GPU-a,GPU-b", "must not be empty"),
        ("0,0", "all", "GPU-a,GPU-b", "must be unique"),
        ("all", "1,1", "GPU-a,GPU-b", "must be unique"),
        ("2", "all", "GPU-a,GPU-b", "outside CUDA_VISIBLE_DEVICES"),
        ("all", "-1", "GPU-a,GPU-b", "non-negative"),
        ("x", "all", "GPU-a,GPU-b", "logical integer"),
    ],
)
def test_phase_shared_topology_fails_closed_on_ambiguous_or_unsafe_specs(train, rollout, visible, message):
    with pytest.raises(ValueError, match=message):
        resolve_phase_shared_topology(
            train_gpus_spec=train,
            rollout_gpus_spec=rollout,
            cuda_visible_devices=visible,
        )


def test_lpt_shards_preserve_arbitrary_training_rank_order_and_stable_ties():
    """LPT must balance by token cost, not by physical-GPU or input order."""

    topology = resolve_phase_shared_topology(
        train_gpus_spec="3,1,0",
        rollout_gpus_spec="3,1,0",
        cuda_visible_devices=_visible(4),
    )
    shards = plan_token_cost_balanced_shards(
        values=("a", "b", "c", "d", "e"),
        token_costs=(9, 8, 7, 6, 5),
        ranks=topology.training_ranks,
    )

    # The first declared training GPU is rank 0 even though it is logical GPU
    # 3.  Stable LPT assigns ties to the lower training rank, not GPU ordinal.
    assert [
        (shard.rank, shard.training_rank.gpu.logical_index, shard.original_indices, shard.token_cost)
        for shard in shards
    ] == [
        (0, 3, (0,), 9),
        (1, 1, (1, 4), 13),
        (2, 0, (2, 3), 13),
    ]
    assert plan_token_cost_balanced_indices((4, 4, 4, 4, 4), world_size=2) == ((0, 2, 4), (1, 3))


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_lpt_index_plans_are_deterministic_and_cover_every_item_for_any_gpu_count(world_size):
    token_costs = (21, 1, 20, 2, 19, 3, 18, 4, 17, 5, 16, 6, 15)

    first = plan_token_cost_balanced_indices(token_costs, world_size=world_size)
    second = plan_token_cost_balanced_indices(token_costs, world_size=world_size)

    assert first == second
    assert len(first) == world_size
    assert sorted(index for shard in first for index in shard) == list(range(len(token_costs)))


def test_lpt_supports_empty_shards_and_empty_input_without_dropping_ranks():
    topology = resolve_phase_shared_topology(
        train_gpus_spec="2,0,3,1",
        rollout_gpus_spec="all",
        cuda_visible_devices=_visible(4),
    )

    one_item_shards = plan_token_cost_balanced_shards(
        values=("only",),
        token_costs=(17,),
        ranks=topology.training_ranks,
    )
    assert [(shard.rank, shard.original_indices, shard.token_cost) for shard in one_item_shards] == [
        (0, (0,), 17),
        (1, (), 0),
        (2, (), 0),
        (3, (), 0),
    ]
    assert restore_sharded_results(
        one_item_shards,
        {
            0: (IndexedResult(0, "finished"),),
            1: (),
            2: (),
            3: (),
        },
    ) == ["finished"]

    empty_shards = plan_token_cost_balanced_shards(values=(), token_costs=(), ranks=topology.training_ranks)
    assert [(shard.rank, shard.original_indices, shard.token_cost) for shard in empty_shards] == [
        (0, (), 0),
        (1, (), 0),
        (2, (), 0),
        (3, (), 0),
    ]
    assert restore_sharded_results(empty_shards, {rank: () for rank in range(4)}) == []


def test_sharded_results_restore_original_order_despite_rank_and_completion_order():
    topology = resolve_phase_shared_topology(
        train_gpus_spec="2,0,1",
        rollout_gpus_spec="all",
        cuda_visible_devices=_visible(3),
    )
    values = ("zero", "one", "two", "three", "four", "five")
    shards = plan_token_cost_balanced_shards(
        values=values,
        token_costs=(10, 1, 9, 2, 8, 3),
        ranks=topology.training_ranks,
    )

    # Mimic arbitrary rank completion order and arbitrary in-rank return order.
    results_by_rank = {}
    for shard in reversed(shards):
        indexed = index_shard_results(shard, [f"done:{item.value}" for item in shard.items])
        results_by_rank[shard.rank] = tuple(reversed(indexed))

    assert restore_sharded_results(shards, results_by_rank) == [f"done:{value}" for value in values]


@pytest.mark.parametrize(
    ("results", "expected_count", "message"),
    [
        ((IndexedResult(0, "a"), IndexedResult(0, "again")), 1, "duplicate"),
        ((IndexedResult(0, "a"), IndexedResult(2, "c")), 3, "missing"),
        ((IndexedResult(0, "a"), IndexedResult(2, "c")), 2, "unexpected"),
    ],
)
def test_restore_original_order_rejects_duplicate_missing_and_unexpected_indices(results, expected_count, message):
    with pytest.raises(ValueError, match=message):
        restore_original_order(results, expected_count=expected_count)


def test_restore_sharded_results_rejects_missing_ranks_and_cross_rank_items():
    ranks = (
        TrainingRank(rank=0, gpu=PhaseGPU(3, "GPU-3")),
        TrainingRank(rank=1, gpu=PhaseGPU(1, "GPU-1")),
    )
    shards = plan_token_cost_balanced_shards(
        values=("a", "b"),
        token_costs=(8, 7),
        ranks=ranks,
    )

    with pytest.raises(ValueError, match="exactly the planned ranks"):
        restore_sharded_results(shards, {0: (IndexedResult(0, "a"),)})
    with pytest.raises(ValueError, match="expected"):
        restore_sharded_results(
            shards,
            {
                0: (IndexedResult(1, "wrong-rank"),),
                1: (IndexedResult(0, "wrong-rank"),),
            },
        )


@pytest.mark.parametrize(
    ("ranks", "message"),
    [
        (
            (
                TrainingRank(rank=1, gpu=PhaseGPU(0, "GPU-0")),
                TrainingRank(rank=0, gpu=PhaseGPU(1, "GPU-1")),
            ),
            "ordered contiguously",
        ),
        (
            (
                TrainingRank(rank=0, gpu=PhaseGPU(0, "GPU-0")),
                TrainingRank(rank=1, gpu=PhaseGPU(0, "GPU-0-again")),
            ),
            "distinct logical GPUs",
        ),
    ],
)
def test_lpt_fails_closed_on_invalid_training_rank_layout(ranks, message):
    with pytest.raises(ValueError, match=message):
        plan_token_cost_balanced_shards(values=("item",), token_costs=(1,), ranks=ranks)


def _two_rank_topology() -> PhaseSharedTopology:
    return resolve_phase_shared_topology(
        train_gpus_spec="3,1",
        rollout_gpus_spec="3,1",
        cuda_visible_devices=_visible(4),
    )


def _advance_to_publishing(controller: PhaseSharedController) -> None:
    controller.start_rollout()
    controller.request_training()
    controller.confirm_rollouts_quiesced()
    controller.complete_training()


def test_controller_requires_external_phase_barriers_and_uses_first_training_rank_as_publisher():
    controller = PhaseSharedController(_two_rank_topology())

    assert controller.rank_index == 0
    assert controller.rank.gpu.logical_index == 3
    assert controller.publisher.gpu.logical_index == 3
    assert controller.is_publisher
    with pytest.raises(PhaseSharedRuntimeError, match="unsafe"):
        controller.request_training()

    assert controller.start_rollout().current is PhaseSharedLifecycle.ROLLOUT
    transition = controller.request_training()
    assert transition.current is PhaseSharedLifecycle.QUIESCING_ROLLOUT
    assert transition.required_runtime_action == "sleep_and_quiesce_rollout_workers"
    with pytest.raises(PhaseSharedRuntimeError, match="unsafe"):
        controller.complete_training()

    assert controller.confirm_rollouts_quiesced().current is PhaseSharedLifecycle.TRAINING
    assert controller.complete_training().current is PhaseSharedLifecycle.PUBLISHING
    with pytest.raises(PhaseSharedRuntimeError, match="unsafe"):
        controller.confirm_rollouts_ready()

    publication = controller.make_canonical_publication("adapter-sha-1")
    assert publication == PolicyPublication(1, 0, "adapter-sha-1")
    assert controller.accept_publication(publication).current is PhaseSharedLifecycle.WAKING_ROLLOUT
    assert controller.confirm_rollouts_ready().current is PhaseSharedLifecycle.ROLLOUT
    assert controller.adapter_version == 1


def test_controller_fails_closed_after_invalid_publication_or_external_failure():
    controller = PhaseSharedController(_two_rank_topology(), rank=1)
    _advance_to_publishing(controller)

    with pytest.raises(PhaseSharedRuntimeError, match="cannot publish"):
        controller.make_canonical_publication("forbidden")
    with pytest.raises(PhaseSharedRuntimeError, match="invalid; expected 1"):
        controller.accept_publication(PolicyPublication(2, 0, "wrong-version"))
    assert controller.state is PhaseSharedLifecycle.PUBLISHING
    assert controller.adapter_version == 0

    failed = controller.fail("rollout worker rejected adapter publication")
    assert failed.previous is PhaseSharedLifecycle.PUBLISHING
    assert failed.current is PhaseSharedLifecycle.FAILED
    assert failed.required_runtime_action == "stop_and_preserve_diagnostics"
    assert controller.failure_reason == "rollout worker rejected adapter publication"
    with pytest.raises(PhaseSharedRuntimeError, match="unsafe"):
        controller.accept_publication(PolicyPublication(1, 0, "would-be-valid"))


def test_publication_broadcast_requires_rank_zero_and_a_valid_shared_artifact():
    topology = _two_rank_topology()
    publication = PolicyPublication(1, 0, "adapter-sha-1")
    publisher_collectives = _FakeCollectives(rank=0, world_size=2, broadcast_value=publication)
    follower_collectives = _FakeCollectives(rank=1, world_size=2, broadcast_value=publication)
    publisher = PhaseSharedController(topology, rank=0, collectives=publisher_collectives)
    follower = PhaseSharedController(topology, rank=1, collectives=follower_collectives)
    _advance_to_publishing(publisher)
    _advance_to_publishing(follower)

    assert publisher.broadcast_and_accept_publication(publication) == publication
    assert follower.broadcast_and_accept_publication() == publication
    assert publisher.state is PhaseSharedLifecycle.WAKING_ROLLOUT
    assert follower.state is PhaseSharedLifecycle.WAKING_ROLLOUT
    assert publisher_collectives.broadcast_calls == [
        (publication, 0, "phase-shared/v0/publishing/publication")
    ]
    assert follower_collectives.broadcast_calls == [
        (None, 0, "phase-shared/v0/publishing/publication")
    ]

    malformed_collectives = _FakeCollectives(rank=0, world_size=2, broadcast_value=None)
    malformed = PhaseSharedController(topology, collectives=malformed_collectives)
    _advance_to_publishing(malformed)
    with pytest.raises(PhaseSharedRuntimeError, match="no valid PolicyPublication"):
        malformed.broadcast_and_accept_publication(publication)
    assert malformed.state is PhaseSharedLifecycle.PUBLISHING
    with pytest.raises(PhaseSharedRuntimeError, match="unsafe"):
        malformed.confirm_rollouts_ready()


def test_global_denominator_is_exact_for_manual_sum_and_ddp_mean_reducers():
    # The fake reports a global denominator of 12 to every rank.  Under a
    # manual SUM reducer each local sum scales by 1/12; under a conventional
    # averaging DDP reducer it must instead scale by world_size/12.
    collectives = _FakeCollectives(rank=0, world_size=4, reduced_value=12.0)
    manual_sum = prepare_global_denominator(
        collectives,
        3,
        tag="objective/manual",
        gradient_reduction=GradientReduction.SUM,
    )
    ddp_mean = prepare_global_denominator(
        collectives,
        3,
        tag="objective/ddp",
        gradient_reduction="mean",
    )

    assert manual_sum.global_denominator == 12.0
    assert manual_sum.normalizer == 12.0
    assert manual_sum.local_sum_scale == pytest.approx(1 / 12)
    assert manual_sum.manual_sum_scale == pytest.approx(1 / 12)
    assert ddp_mean.local_sum_scale == pytest.approx(4 / 12)
    assert ddp_mean.ddp_mean_scale == pytest.approx(4 / 12)

    local_numerators = (2.0, 3.0, 5.0, 7.0)
    expected_global_objective = sum(local_numerators) / 12
    assert sum(manual_sum.scale_local_sum(value) for value in local_numerators) == pytest.approx(
        expected_global_objective
    )
    assert sum(ddp_mean.scale_local_sum(value) for value in local_numerators) / 4 == pytest.approx(
        expected_global_objective
    )
    assert manual_sum.scale_local_sum(2.0, negate=True) == pytest.approx(-2 / 12)
    assert [tag for _, tag in collectives.sum_calls] == [
        "objective/manual/sum/denominator",
        "objective/ddp/sum/denominator",
    ]


def test_global_denominator_clamps_empty_global_batch_and_named_sums_are_stably_tagged():
    collectives = _FakeCollectives(
        rank=0,
        world_size=2,
        reduced_value=lambda value, _tag: value * 2,
    )
    denominator = prepare_global_denominator(
        collectives,
        0,
        tag="empty",
        minimum=0.25,
    )
    assert denominator.global_denominator == 0
    assert denominator.normalizer == 0.25
    assert denominator.local_sum_scale == pytest.approx(4.0)

    reduced = reduce_named_sums(collectives, {"z_loss": 1.5, "a_reward": 2.0}, tag="training")
    assert list(reduced.local) == ["a_reward", "z_loss"]
    assert reduced.global_ == {"a_reward": 4.0, "z_loss": 3.0}
    assert [tag for _, tag in collectives.sum_calls] == [
        "empty/sum/denominator",
        "training/sum/a_reward",
        "training/sum/z_loss",
    ]
