"""CPU-only contract tests for rollout data parallelism.

No worker here imports vLLM or touches CUDA.  Fake endpoints exercise the same
partition, adapter barrier, IPC descriptor, merge, failure, and shutdown path as
real process endpoints.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
from tinker import types

from ctm.backends.base import SampledSequence
from ctm.backends.local.rollout_workers import (
    FrozenBaseVLLMBackend,
    RolloutGPU,
    RolloutParallelBackend,
    RolloutWorkerError,
    RolloutWorkerPool,
    _WorkerProcessConfig,
    _read_payload,
    _remove_payload,
    _worker_main,
    _write_payload,
    resolve_base_only_gpus,
    resolve_rollout_gpus,
)


def _fake_spawn_worker(connection, config):
    """Picklable process target exercising the real pipe/process endpoint."""

    version = 0
    connection.send(
        {
            "ok": True,
            "op": "ready",
            "worker_id": config.worker_id,
            "pid": os.getpid(),
            "adapter_version": version,
        }
    )
    while True:
        command = connection.recv()
        op = command["op"]
        response = {
            "ok": True,
            "op": op,
            "request_id": command["request_id"],
            "worker_id": config.worker_id,
        }
        if op == "health":
            response.update(pid=os.getpid(), adapter_version=version)
            connection.send(response)
        elif op == "load_adapter":
            version = command["adapter_version"]
            response["adapter_version"] = version
            connection.send(response)
        elif op == "sample_batch":
            records = [
                (
                    assignment["prompt_index"],
                    sample_index,
                    SampledSequence(
                        tokens=[config.worker_id, sample_index],
                        logprobs=[-0.25, -0.5],
                    ),
                )
                for assignment in command["assignments"]
                for sample_index in assignment["sample_indices"]
            ]
            response.update(
                adapter_version=version,
                result=_write_payload(
                    records,
                    ipc_dir=Path(config.ipc_dir),
                    worker_id=config.worker_id,
                    request_id=command["request_id"],
                ),
            )
            connection.send(response)
        elif op in {"score_batch", "score_batch_uncapped_eos_tail"}:
            records = [
                (
                    assignment["score_index"],
                    0,
                    SampledSequence(
                        tokens=list(assignment["completion_tokens"]),
                        logprobs=[-(config.worker_id + token / 100.0) for token in assignment["completion_tokens"]],
                    ),
                )
                for assignment in command["assignments"]
            ]
            response.update(
                adapter_version=version,
                result=_write_payload(
                    records,
                    ipc_dir=Path(config.ipc_dir),
                    worker_id=config.worker_id,
                    request_id=command["request_id"],
                ),
            )
            connection.send(response)
        elif op == "shutdown":
            response["adapter_version"] = version
            connection.send(response)
            break
        else:
            raise AssertionError(op)
    connection.close()


class FakeEndpoint:
    def __init__(
        self,
        worker_id: int,
        gpu: RolloutGPU,
        *,
        fail_sample: bool = False,
        fail_score: bool = False,
        corrupt_score: str | None = None,
    ):
        self.worker_id = worker_id
        self.gpu = gpu
        self.fail_sample = fail_sample
        self.fail_score = fail_score
        self.corrupt_score = corrupt_score
        self.ipc_dir: Path | None = None
        self.adapter_version = 0
        self.commands = []
        self.launched = False
        self.closed = False

    def launch_sync(self):
        self.launched = True

    def wait_ready_sync(self):
        return {
            "ok": True,
            "op": "ready",
            "worker_id": self.worker_id,
            "adapter_version": self.adapter_version,
        }

    def call_sync(self, command, **_kwargs):
        self.commands.append(command)
        op = command["op"]
        if op == "health":
            return {
                "ok": True,
                "op": op,
                "request_id": command["request_id"],
                "worker_id": self.worker_id,
                "adapter_version": self.adapter_version,
            }
        if op == "load_adapter":
            self.adapter_version = command["adapter_version"]
            return {
                "ok": True,
                "op": op,
                "request_id": command["request_id"],
                "worker_id": self.worker_id,
                "adapter_version": self.adapter_version,
            }
        if op in {"score_batch", "score_batch_uncapped_eos_tail"}:
            if self.fail_score:
                raise RolloutWorkerError(f"fake worker {self.worker_id} score failed")
            records = []
            for assignment in reversed(command["assignments"]):
                score_index = assignment["score_index"]
                completion = list(assignment["completion_tokens"])
                records.append(
                    (
                        score_index,
                        0,
                        SampledSequence(
                            tokens=completion,
                            logprobs=[-(score_index + token / 1000.0 + self.worker_id / 100.0) for token in completion],
                        ),
                    )
                )
            if records and self.corrupt_score == "tokens":
                records[0][2].tokens[-1] += 1
            elif records and self.corrupt_score == "missing_logprobs":
                records[0][2].logprobs = None
            elif records and self.corrupt_score == "nonfinite":
                records[0][2].logprobs[-1] = float("nan")
            elif records and self.corrupt_score == "duplicate":
                records.append(records[0])
            elif records and self.corrupt_score == "omit":
                records.pop()
            assert self.ipc_dir is not None
            descriptor = _write_payload(
                records,
                ipc_dir=self.ipc_dir,
                worker_id=self.worker_id,
                request_id=command["request_id"],
            )
            return {
                "ok": True,
                "op": op,
                "request_id": command["request_id"],
                "worker_id": self.worker_id,
                "adapter_version": self.adapter_version,
                "result": descriptor,
            }
        if op != "sample_batch":
            raise AssertionError(op)
        if self.fail_sample:
            raise RolloutWorkerError(f"fake worker {self.worker_id} failed")
        if not command["use_base"]:
            assert command["adapter_version"] == self.adapter_version
        records = []
        # Deliberately reverse both assignment and slot order. The coordinator
        # must restore original prompt/sample order rather than response order.
        for assignment in reversed(command["assignments"]):
            prompt_index = assignment["prompt_index"]
            for sample_index in reversed(assignment["sample_indices"]):
                records.append(
                    (
                        prompt_index,
                        sample_index,
                        SampledSequence(
                            tokens=[prompt_index, sample_index, self.worker_id],
                            logprobs=[-(sample_index + 0.125)] * 3,
                        ),
                    )
                )
        assert self.ipc_dir is not None
        descriptor = _write_payload(
            records,
            ipc_dir=self.ipc_dir,
            worker_id=self.worker_id,
            request_id=command["request_id"],
        )
        return {
            "ok": True,
            "op": op,
            "request_id": command["request_id"],
            "worker_id": self.worker_id,
            "adapter_version": self.adapter_version,
            "result": descriptor,
        }

    async def call(self, command):
        return self.call_sync(command)

    def close_sync(self):
        self.closed = True


def make_pool(
    tmp_path,
    worker_count: int,
    *,
    failing_worker: int | None = None,
    enable_lora: bool = True,
    first_logical_gpu: int = 1,
    failing_score_worker: int | None = None,
    corrupt_score_worker: int | None = None,
    corrupt_score: str | None = None,
):
    gpus = tuple(
        RolloutGPU(index + first_logical_gpu, f"GPU-{index + first_logical_gpu}") for index in range(worker_count)
    )
    endpoints = [
        FakeEndpoint(
            index,
            gpu,
            fail_sample=index == failing_worker,
            fail_score=index == failing_score_worker,
            corrupt_score=corrupt_score if index == corrupt_score_worker else None,
        )
        for index, gpu in enumerate(gpus)
    ]
    pool = RolloutWorkerPool(
        model="fake/model",
        gpus=gpus,
        engine_kwargs={"gpu_memory_utilization": 0.9},
        status_dir=tmp_path,
        enable_lora=enable_lora,
        endpoints=endpoints,
    )
    for endpoint in endpoints:
        endpoint.ipc_dir = pool.ipc_dir
    return pool, endpoints


def test_pool_pins_processed_logprobs_mode_for_every_worker(tmp_path):
    pool, _endpoints = make_pool(tmp_path / "processed", 2)
    assert pool.engine_kwargs["logprobs_mode"] == "processed_logprobs"

    gpus = [RolloutGPU(1, "GPU-one")]
    with pytest.raises(ValueError, match="require logprobs_mode='processed_logprobs'"):
        RolloutWorkerPool(
            model="fake/model",
            gpus=gpus,
            engine_kwargs={"logprobs_mode": "raw_logprobs"},
            status_dir=tmp_path / "incompatible",
            endpoints=[FakeEndpoint(0, gpus[0])],
        )


def test_pool_derives_distinct_engine_seed_for_every_worker(tmp_path):
    gpus = tuple(RolloutGPU(index + 1, f"GPU-{index + 1}") for index in range(3))
    pool = RolloutWorkerPool(
        model="fake/model",
        gpus=gpus,
        engine_kwargs={"gpu_memory_utilization": 0.9, "seed": 42},
        status_dir=tmp_path,
    )

    assert pool.engine_kwargs["seed"] == 42
    assert pool.engine_seed_base == 42
    assert pool.engine_seed_policy == "base_plus_worker_id_v1"
    assert pool.worker_engine_seeds == (42, 43, 44)
    assert [endpoint._config.engine_kwargs["seed"] for endpoint in pool._endpoints] == [42, 43, 44]
    assert all(endpoint._config.engine_kwargs["gpu_memory_utilization"] == 0.9 for endpoint in pool._endpoints)

    configured = json.loads(pool.log_path.read_text().splitlines()[0])
    assert configured["engine_seed_base"] == 42
    assert configured["engine_seed_policy"] == "base_plus_worker_id_v1"
    assert configured["worker_engine_seeds"] == [42, 43, 44]
    assert [worker["engine_seed"] for worker in configured["workers"]] == [42, 43, 44]
    pool.shutdown()


def test_pool_draws_one_entropy_seed_then_derives_worker_streams(monkeypatch, tmp_path):
    monkeypatch.setattr("ctm.backends.local.rollout_workers.secrets.randbelow", lambda upper: 700)
    gpus = (RolloutGPU(0, "GPU-a"), RolloutGPU(1, "GPU-b"))
    pool = RolloutWorkerPool(
        model="fake/model",
        gpus=gpus,
        engine_kwargs={},
        status_dir=tmp_path,
    )

    assert pool.engine_kwargs["seed"] == 700
    assert pool.worker_engine_seeds == (700, 701)
    assert [endpoint._config.engine_kwargs["seed"] for endpoint in pool._endpoints] == [700, 701]
    pool.shutdown()


@pytest.mark.parametrize("seed", [-1, 2**31 - 1, True, 1.5])
def test_pool_rejects_seed_base_that_cannot_cover_every_worker(tmp_path, seed):
    gpus = (RolloutGPU(0, "GPU-a"), RolloutGPU(1, "GPU-b"))
    with pytest.raises(ValueError, match="rollout worker vLLM seed"):
        RolloutWorkerPool(
            model="fake/model",
            gpus=gpus,
            engine_kwargs={"seed": seed},
            status_dir=tmp_path / str(seed),
        )


@pytest.mark.parametrize("worker_count", [1, 2, 3])
def test_partition_and_merge_preserve_every_prompt_sample_slot(tmp_path, worker_count):
    pool, endpoints = make_pool(tmp_path, worker_count)
    pool.start()
    pool.publish_adapter_sync(tmp_path, version=1)

    batches = asyncio.run(
        pool.sample_batch(
            [[10], [20]],
            max_tokens=32,
            temperature=1.0,
            stop=[2],
            num_samples=7,
            use_base=False,
        )
    )

    assert len(batches) == 2
    for prompt_index, sequences in enumerate(batches):
        assert [sequence.tokens[0:2] for sequence in sequences] == [
            [prompt_index, sample_index] for sample_index in range(7)
        ]
        assert [sequence.tokens[2] for sequence in sequences] == [
            (prompt_index * 7 + sample_index) % worker_count for sample_index in range(7)
        ]
        assert [sequence.logprobs for sequence in sequences] == [
            [-(sample_index + 0.125)] * 3 for sample_index in range(7)
        ]

    sampled_slots = []
    per_worker_counts = []
    for endpoint in endpoints:
        command = next(command for command in endpoint.commands if command["op"] == "sample_batch")
        worker_slots = [
            (assignment["prompt_index"], sample_index)
            for assignment in command["assignments"]
            for sample_index in assignment["sample_indices"]
        ]
        per_worker_counts.append(len(worker_slots))
        sampled_slots.extend(worker_slots)
    assert sorted(sampled_slots) == [(prompt, sample) for prompt in range(2) for sample in range(7)]
    assert len(sampled_slots) == len(set(sampled_slots))
    assert max(per_worker_counts) - min(per_worker_counts) <= 1
    assert list(pool.ipc_dir.iterdir()) == []


def test_uneven_three_worker_partition_is_stable(tmp_path):
    pool, endpoints = make_pool(tmp_path, 3)
    pool.start()
    pool.publish_adapter_sync(tmp_path, version=1)

    result = asyncio.run(
        pool.sample_batch([[1]], max_tokens=4, temperature=0.7, stop=[], num_samples=5, use_base=False)
    )[0]

    assert [sequence.tokens[1:] for sequence in result] == [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 0],
        [4, 1],
    ]
    counts = [
        sum(
            len(assignment["sample_indices"])
            for command in endpoint.commands
            if command["op"] == "sample_batch"
            for assignment in command["assignments"]
        )
        for endpoint in endpoints
    ]
    assert counts == [2, 2, 1]


def test_score_partition_and_merge_are_stable_exact_and_policy_versioned(tmp_path):
    pool, endpoints = make_pool(tmp_path, 3)
    pool.start()
    pool.publish_adapter_sync(tmp_path, version=4)
    prompts = [[100 + index] for index in range(8)]
    completions = [[index + 1, index + 11] for index in range(8)]

    scores = asyncio.run(pool.score_completions(prompts, completions, use_base=False))

    expected_scores = [
        [-(score_index + token / 1000.0 + (score_index % 3) / 100.0) for token in completion]
        for score_index, completion in enumerate(completions)
    ]
    assert all(actual == pytest.approx(expected) for actual, expected in zip(scores, expected_scores))
    commands = [
        next(command for command in endpoint.commands if command["op"] == "score_batch") for endpoint in endpoints
    ]
    assert [[assignment["score_index"] for assignment in command["assignments"]] for command in commands] == [
        [0, 3, 6],
        [1, 4, 7],
        [2, 5],
    ]
    assert all(command["adapter_version"] == 4 and command["use_base"] is False for command in commands)
    assert [assignment["completion_tokens"] for command in commands for assignment in command["assignments"]] == [
        completions[index] for index in [0, 3, 6, 1, 4, 7, 2, 5]
    ]
    assert list(pool.ipc_dir.iterdir()) == []


def test_uncapped_eos_tail_score_uses_distinct_fail_closed_worker_operation(tmp_path):
    pool, endpoints = make_pool(tmp_path, 3)
    pool.start()

    scores = asyncio.run(
        pool.score_completions_uncapped_eos_tail(
            [[10], [20], [30]],
            [[1], [2], [3]],
            use_base=True,
        )
    )

    assert len(scores) == 3
    commands = [
        command
        for endpoint in endpoints
        for command in endpoint.commands
        if command["op"] == "score_batch_uncapped_eos_tail"
    ]
    assert len(commands) == 3
    assert all(command["use_base"] is True for command in commands)


def test_base_scoring_needs_no_adapter_but_policy_scoring_does(tmp_path):
    pool, endpoints = make_pool(tmp_path, 2)
    pool.start()

    base = asyncio.run(pool.score_completions([[1], [2]], [[3], [4, 5]], use_base=True))
    assert base[0] == pytest.approx([-0.003])
    assert base[1] == pytest.approx([-1.014, -1.015])
    with pytest.raises(RolloutWorkerError, match="acknowledged adapter"):
        asyncio.run(pool.score_completions([[1]], [[2]], use_base=False))
    assert all(
        command["use_base"] is True
        for endpoint in endpoints
        for command in endpoint.commands
        if command["op"] == "score_batch"
    )


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("tokens", "misaligned completion tokens"),
        ("missing_logprobs", "returned None logprob"),
        ("nonfinite", "non-finite logprob"),
        ("duplicate", "duplicate scoring slot"),
        ("omit", "omitted 1 scoring slot"),
    ],
)
def test_malformed_worker_score_payload_fails_closed_and_marks_pool_unhealthy(tmp_path, corruption, message):
    pool, _endpoints = make_pool(
        tmp_path,
        1,
        corrupt_score_worker=0,
        corrupt_score=corruption,
    )
    pool.start()

    with pytest.raises(RolloutWorkerError, match=message):
        asyncio.run(pool.score_completions([[1]], [[2, 3]], use_base=True))
    with pytest.raises(RolloutWorkerError, match="unhealthy"):
        asyncio.run(pool.score_completions([[1]], [[2]], use_base=True))
    assert list(pool.ipc_dir.iterdir()) == []


def test_score_worker_failure_is_fatal_and_cleans_other_payloads(tmp_path):
    pool, _endpoints = make_pool(tmp_path, 2, failing_score_worker=1)
    pool.start()

    with pytest.raises(RolloutWorkerError, match="fake worker 1 score failed"):
        asyncio.run(pool.score_completions([[1], [2]], [[3], [4]], use_base=True))
    assert list(pool.ipc_dir.iterdir()) == []


def test_score_inputs_are_validated_before_worker_calls(tmp_path):
    pool, endpoints = make_pool(tmp_path, 2)
    pool.start()

    with pytest.raises(ValueError, match="same length"):
        asyncio.run(pool.score_completions([[1]], [], use_base=True))
    with pytest.raises(ValueError, match="prompt 0 is empty"):
        asyncio.run(pool.score_completions([[]], [[1]], use_base=True))
    with pytest.raises(ValueError, match="completion 0 is empty"):
        asyncio.run(pool.score_completions([[1]], [[]], use_base=True))
    assert asyncio.run(pool.score_completions([], [], use_base=True)) == []
    assert all(not any(command["op"] == "score_batch" for command in endpoint.commands) for endpoint in endpoints)


def test_ignore_eos_is_forwarded_to_every_worker(tmp_path):
    pool, endpoints = make_pool(tmp_path, 2)
    pool.start()
    pool.publish_adapter_sync(tmp_path, version=1)

    asyncio.run(
        pool.sample_batch(
            [[1]],
            max_tokens=20_480,
            temperature=0.7,
            stop=[],
            num_samples=3,
            use_base=False,
            ignore_eos=True,
        )
    )

    commands = [command for endpoint in endpoints for command in endpoint.commands if command["op"] == "sample_batch"]
    assert len(commands) == 2
    assert all(command["ignore_eos"] is True for command in commands)


def test_pool_preserves_uncapped_max_tokens_and_rejects_nonpositive_caps(tmp_path):
    pool, endpoints = make_pool(tmp_path, 2)
    pool.start()

    sampled = asyncio.run(
        pool.sample_batch(
            [[1]],
            max_tokens=None,
            temperature=0.7,
            stop=[2],
            num_samples=1,
            use_base=True,
        )
    )

    assert len(sampled) == 1
    commands = [command for endpoint in endpoints for command in endpoint.commands if command["op"] == "sample_batch"]
    assert len(commands) == 2
    assert all(command["max_tokens"] is None for command in commands)
    for invalid_max_tokens in (0, -1):
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            asyncio.run(
                pool.sample_batch(
                    [[1]],
                    max_tokens=invalid_max_tokens,
                    temperature=0.7,
                    stop=[2],
                    num_samples=1,
                    use_base=True,
                )
            )


def test_base_sampling_and_versioned_policy_refresh(tmp_path):
    pool, endpoints = make_pool(tmp_path, 2)
    pool.start()

    # Frozen-base sampling needs no adapter. Policy sampling does.
    base = asyncio.run(pool.sample_batch([[1]], max_tokens=4, temperature=1.0, stop=[], num_samples=3, use_base=True))
    assert len(base[0]) == 3
    with pytest.raises(RolloutWorkerError, match="acknowledged adapter"):
        asyncio.run(pool.sample_batch([[1]], max_tokens=4, temperature=1.0, stop=[], num_samples=1, use_base=False))

    pool.publish_adapter_sync(tmp_path, version=4)
    asyncio.run(pool.sample_batch([[1]], max_tokens=4, temperature=1.0, stop=[], num_samples=3, use_base=False))
    asyncio.run(pool.publish_adapter(tmp_path, version=5))
    asyncio.run(pool.sample_batch([[1]], max_tokens=4, temperature=1.0, stop=[], num_samples=3, use_base=False))

    for endpoint in endpoints:
        policy_commands = [
            command for command in endpoint.commands if command["op"] == "sample_batch" and not command["use_base"]
        ]
        assert [command["adapter_version"] for command in policy_commands] == [4, 5]
        assert endpoint.adapter_version == 5
    assert pool.adapter_version == 5


def test_frozen_base_backend_preserves_uncapped_max_tokens_and_restores_prompt_order(tmp_path):
    pool, endpoints = make_pool(
        tmp_path / "pool",
        3,
        enable_lora=False,
        first_logical_gpu=0,
    )
    backend = FrozenBaseVLLMBackend(
        model="fake/model",
        engine_kwargs={"gpu_memory_utilization": 0.9},
        pool=pool,
    )

    backend.setup(model="fake/model")
    sampled = asyncio.run(
        backend.base_sampler().sample_batch(
            [
                types.ModelInput.from_ints(tokens=[10]),
                types.ModelInput.from_ints(tokens=[20]),
                types.ModelInput.from_ints(tokens=[30]),
                types.ModelInput.from_ints(tokens=[40]),
            ],
            max_tokens=None,
            temperature=0.7,
            stop=[2],
            num_samples=1,
        )
    )

    assert not hasattr(backend, "model")
    assert pool.adapter_version == 0
    assert [[sequence.tokens[:2] for sequence in group] for group in sampled] == [
        [[0, 0]],
        [[1, 0]],
        [[2, 0]],
        [[3, 0]],
    ]
    assert all(not any(command["op"] == "load_adapter" for command in endpoint.commands) for endpoint in endpoints)
    assert all(
        command["max_tokens"] is None
        for endpoint in endpoints
        for command in endpoint.commands
        if command["op"] == "sample_batch"
    )
    sampled_workers = [
        endpoint.worker_id
        for endpoint in endpoints
        if any(
            assignment["sample_indices"]
            for command in endpoint.commands
            if command["op"] == "sample_batch"
            for assignment in command["assignments"]
        )
    ]
    assert sampled_workers == [0, 1, 2]

    with pytest.raises(RolloutWorkerError, match="does not accept adapter"):
        pool.publish_adapter_sync(tmp_path, version=1)
    backend.shutdown()


def test_frozen_base_backend_forwards_uncapped_max_tokens_to_in_process_sampler():
    sample_calls = []

    class FakeSampler:
        def __init__(self, **_kwargs):
            pass

        def sample_batch(self, prompts, **kwargs):
            sample_calls.append((prompts, kwargs))
            return [[SampledSequence(tokens=[42], logprobs=[-0.5])] for _ in prompts]

        def shutdown(self):
            return None

    backend = FrozenBaseVLLMBackend(model="fake/model", sampler_factory=FakeSampler)
    backend.setup(model="fake/model")

    sampled = asyncio.run(
        backend.sample_base_batch(
            [[10], [20]],
            max_tokens=None,
            temperature=0.7,
            stop=[2],
            num_samples=1,
        )
    )

    assert [[sequence.tokens for sequence in group] for group in sampled] == [[[42]], [[42]]]
    assert sample_calls == [
        (
            [[10], [20]],
            {
                "max_tokens": None,
                "temperature": 0.7,
                "stop": [2],
                "num_samples": 1,
                "use_base": True,
            },
        )
    ]
    backend.shutdown()


def test_frozen_base_backend_worker_failure_is_fatal_and_leaves_no_partial_payload(tmp_path):
    pool, _endpoints = make_pool(
        tmp_path / "pool",
        2,
        failing_worker=1,
        enable_lora=False,
        first_logical_gpu=0,
    )
    backend = FrozenBaseVLLMBackend(model="fake/model", pool=pool)
    backend.setup(model="fake/model")

    with pytest.raises(RolloutWorkerError, match="fake worker 1 failed"):
        asyncio.run(
            backend.base_sampler().sample_batch(
                [types.ModelInput.from_ints(tokens=[10]), types.ModelInput.from_ints(tokens=[20])],
                max_tokens=8,
                temperature=0.7,
                stop=[],
                num_samples=1,
            )
        )
    with pytest.raises(RolloutWorkerError, match="unhealthy"):
        asyncio.run(
            backend.base_sampler().sample_batch(
                [types.ModelInput.from_ints(tokens=[30])],
                max_tokens=8,
                temperature=0.7,
                stop=[],
                num_samples=1,
            )
        )
    assert list(pool.ipc_dir.iterdir()) == []
    backend.shutdown()


def test_worker_failure_propagates_and_marks_pool_unhealthy(tmp_path):
    pool, _endpoints = make_pool(tmp_path, 2, failing_worker=1)
    pool.start()
    pool.publish_adapter_sync(tmp_path, version=1)

    with pytest.raises(RolloutWorkerError, match="fake worker 1 failed"):
        asyncio.run(pool.sample_batch([[1]], max_tokens=4, temperature=1.0, stop=[], num_samples=4, use_base=False))
    with pytest.raises(RolloutWorkerError, match="unhealthy"):
        asyncio.run(pool.sample_batch([[1]], max_tokens=4, temperature=1.0, stop=[], num_samples=1, use_base=False))
    assert list(pool.ipc_dir.iterdir()) == []


def test_cancelled_old_policy_request_drains_before_refresh(tmp_path):
    gpu = RolloutGPU(1, "GPU-one")
    endpoint = FakeEndpoint(0, gpu)
    pool = RolloutWorkerPool(
        model="fake/model",
        gpus=[gpu],
        engine_kwargs={},
        status_dir=tmp_path,
        endpoints=[endpoint],
    )
    endpoint.ipc_dir = pool.ipc_dir
    pool.start()
    pool.publish_adapter_sync(tmp_path, version=1)

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        original_call = endpoint.call

        async def blocking_call(command):
            if command["op"] == "sample_batch":
                started.set()
                await release.wait()
            return await original_call(command)

        endpoint.call = blocking_call
        sample = asyncio.create_task(
            pool.sample_batch([[1]], max_tokens=4, temperature=1.0, stop=[], num_samples=2, use_base=False)
        )
        await started.wait()
        sample.cancel()
        refresh = asyncio.create_task(pool.publish_adapter(tmp_path, version=2))
        await asyncio.sleep(0)
        assert not refresh.done()  # the adapter barrier cannot overtake v1 generation
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await sample
        await refresh

    asyncio.run(scenario())
    operations = [command["op"] for command in endpoint.commands]
    assert operations[-2:] == ["sample_batch", "load_adapter"]
    assert endpoint.adapter_version == pool.adapter_version == 2
    assert list(pool.ipc_dir.iterdir()) == []


def test_cancelled_old_policy_score_drains_before_refresh(tmp_path):
    gpu = RolloutGPU(1, "GPU-one")
    endpoint = FakeEndpoint(0, gpu)
    pool = RolloutWorkerPool(
        model="fake/model",
        gpus=[gpu],
        engine_kwargs={},
        status_dir=tmp_path,
        endpoints=[endpoint],
    )
    endpoint.ipc_dir = pool.ipc_dir
    pool.start()
    pool.publish_adapter_sync(tmp_path, version=1)

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        original_call = endpoint.call

        async def blocking_call(command):
            if command["op"] == "score_batch":
                started.set()
                await release.wait()
            return await original_call(command)

        endpoint.call = blocking_call
        score = asyncio.create_task(pool.score_completions([[1]], [[2, 3]], use_base=False))
        await started.wait()
        score.cancel()
        refresh = asyncio.create_task(pool.publish_adapter(tmp_path, version=2))
        await asyncio.sleep(0)
        assert not refresh.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await score
        await refresh

    asyncio.run(scenario())
    operations = [command["op"] for command in endpoint.commands]
    assert operations[-2:] == ["score_batch", "load_adapter"]
    assert endpoint.adapter_version == pool.adapter_version == 2
    assert list(pool.ipc_dir.iterdir()) == []


def test_shutdown_closes_every_worker_and_is_idempotent(tmp_path):
    pool, endpoints = make_pool(tmp_path, 3)
    pool.start()
    pool.shutdown()
    pool.shutdown()
    assert all(endpoint.closed for endpoint in endpoints)


def test_spawn_endpoint_uses_one_real_process_per_gpu(tmp_path):
    gpus = (RolloutGPU(1, "GPU-one"), RolloutGPU(2, "GPU-two"))
    pool = RolloutWorkerPool(
        model="fake/model",
        gpus=gpus,
        engine_kwargs={},
        status_dir=tmp_path,
        start_timeout_seconds=10,
        request_timeout_seconds=10,
        shutdown_timeout_seconds=5,
        worker_entrypoint_for_tests=_fake_spawn_worker,
    )
    pool.start()
    child_processes = [endpoint._process for endpoint in pool._endpoints]
    assert all(process.pid != os.getpid() and process.is_alive() for process in child_processes)
    assert len({process.pid for process in child_processes}) == 2

    pool.publish_adapter_sync(tmp_path, version=1)
    result = asyncio.run(
        pool.sample_batch([[9]], max_tokens=4, temperature=1.0, stop=[], num_samples=5, use_base=False)
    )[0]
    assert [sequence.tokens for sequence in result] == [[0, 0], [1, 1], [0, 2], [1, 3], [0, 4]]
    scores = asyncio.run(
        pool.score_completions(
            [[10], [20], [30]],
            [[4, 5], [6], [7, 8]],
            use_base=False,
        )
    )
    assert scores == [[-0.04, -0.05], [-1.06], [-0.07, -0.08]]

    pool.shutdown()
    assert all(not process.is_alive() for process in child_processes)


def test_worker_sample_log_records_elapsed_tokens_and_throughput(tmp_path, monkeypatch):
    from ctm.backends.local import vllm_sampler

    sampler_inits = []
    sample_calls = []
    score_calls = []

    class FakeSampler:
        def __init__(self, **kwargs):
            sampler_inits.append(kwargs)

        def sample_batch(self, prompts, *, num_samples, **kwargs):
            sample_calls.append((prompts, num_samples, kwargs))
            return [
                [SampledSequence(tokens=[tokens[0], 99], logprobs=[-0.1, -0.2]) for _ in range(num_samples)]
                for tokens in prompts
            ]

        def score_completions(self, prompts, completions, *, use_base):
            score_calls.append((prompts, completions, use_base))
            return [[-(token / 10.0) for token in tokens] for tokens in completions]

        def shutdown(self):
            return None

    class FakeConnection:
        def __init__(self, commands):
            self.commands = list(commands)
            self.responses = []

        def recv(self):
            return self.commands.pop(0)

        def send(self, response):
            self.responses.append(response)

        def close(self):
            return None

    monkeypatch.setattr(vllm_sampler, "VLLMSampler", FakeSampler)
    ipc_dir = tmp_path / "ipc"
    log_path = tmp_path / "worker.jsonl"
    connection = FakeConnection(
        [
            {
                "op": "sample_batch",
                "request_id": "sample-1",
                "assignments": [
                    {"prompt_index": 0, "prompt_tokens": [10], "sample_indices": [0]},
                    {"prompt_index": 1, "prompt_tokens": [20], "sample_indices": [0]},
                ],
                "max_tokens": None,
                "temperature": 0.7,
                "stop": [2],
                "use_base": True,
                "adapter_version": 0,
            },
            {
                "op": "score_batch",
                "request_id": "score-1",
                "assignments": [
                    {
                        "score_index": 3,
                        "prompt_tokens": [30, 31],
                        "completion_tokens": [40, 41],
                    }
                ],
                "use_base": True,
                "adapter_version": 0,
            },
            {"op": "shutdown", "request_id": "shutdown-1"},
        ]
    )
    config = _WorkerProcessConfig(
        worker_id=0,
        gpu=RolloutGPU(0, "GPU-a"),
        model="fake/model",
        enable_lora=False,
        engine_kwargs={"gpu_memory_utilization": 0.9, "seed": 42},
        ipc_dir=str(ipc_dir),
        log_path=str(log_path),
    )

    _worker_main(connection, config)

    assert sampler_inits == [
        {
            "model": "fake/model",
            "enable_lora": False,
            "gpu_memory_utilization": 0.9,
            "seed": 42,
        }
    ]
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    completed = next(event for event in events if event["event"] == "sample_completed")
    boot = next(event for event in events if event["event"] == "boot")
    ready = next(event for event in events if event["event"] == "ready")
    assert boot["engine_seed"] == ready["engine_seed"] == 42
    assert completed["record_count"] == 2
    assert completed["completion_tokens"] == 4
    assert completed["elapsed_seconds"] >= 0
    assert completed["output_tokens_per_second"] >= 0
    assert sample_calls == [
        (
            [[10], [20]],
            1,
            {
                "max_tokens": None,
                "temperature": 0.7,
                "stop": [2],
                "use_base": True,
                "ignore_eos": False,
            },
        )
    ]
    assert score_calls == [([[30, 31]], [[40, 41]], True)]
    score_completed = next(event for event in events if event["event"] == "score_completed")
    assert score_completed["record_count"] == 1
    assert score_completed["completion_tokens"] == 2
    score_response = next(response for response in connection.responses if response.get("op") == "score_batch")
    decoded = _read_payload(score_response["result"], ipc_dir=ipc_dir)
    _remove_payload(score_response["result"], ipc_dir)
    assert decoded == [(3, 0, SampledSequence(tokens=[40, 41], logprobs=[-4.0, -4.1]))]


def test_binary_ipc_round_trips_exact_tokens_logprobs_and_none(tmp_path):
    records = [
        (0, 2, SampledSequence(tokens=[0, 2**40, -3], logprobs=[-0.1, float("-inf"), 1.25])),
        (1, 0, SampledSequence(tokens=[], logprobs=None)),
    ]
    descriptor = _write_payload(records, ipc_dir=tmp_path, worker_id=0, request_id="exact")
    decoded = _read_payload(descriptor, ipc_dir=tmp_path)
    _remove_payload(descriptor, tmp_path)

    assert [(p, s, sequence.tokens) for p, s, sequence in decoded] == [
        (0, 2, [0, 2**40, -3]),
        (1, 0, []),
    ]
    assert decoded[0][2].logprobs == [-0.1, float("-inf"), 1.25]
    assert decoded[1][2].logprobs is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("spec", "visible", "coordinator", "expected"),
    [
        ("1", "4,7", "cuda:0", [(1, "7")]),
        ("1,2", "GPU-a,GPU-b,MIG-c", "cuda:0", [(1, "GPU-b"), (2, "MIG-c")]),
    ],
)
def test_gpu_resolution_is_explicit_logical_to_supplied_token(spec, visible, coordinator, expected):
    resolved = resolve_rollout_gpus(
        spec,
        cuda_visible_devices=visible,
        coordinator_device=coordinator,
    )
    assert [(gpu.logical_index, gpu.device_token) for gpu in resolved] == expected


@pytest.mark.parametrize(
    ("spec", "visible", "coordinator", "message"),
    [
        ("1", None, "cuda:0", "CUDA_VISIBLE_DEVICES"),
        ("1", "0,1", "cuda", "explicit --local-device"),
        ("0", "0,1", "cuda:0", "cannot also host"),
        ("2", "0,1", "cuda:0", "outside CUDA_VISIBLE_DEVICES"),
        ("1,1", "0,1", "cuda:0", "must be unique"),
    ],
)
def test_gpu_resolution_rejects_ambiguous_or_unsafe_assignments(spec, visible, coordinator, message):
    with pytest.raises(ValueError, match=message):
        resolve_rollout_gpus(
            spec,
            cuda_visible_devices=visible,
            coordinator_device=coordinator,
        )


def test_base_only_gpu_resolution_uses_every_explicit_logical_gpu_including_zero():
    resolved = resolve_base_only_gpus(
        "0,1,2",
        cuda_visible_devices="GPU-a,7,MIG-c",
    )

    assert [(gpu.logical_index, gpu.device_token) for gpu in resolved] == [
        (0, "GPU-a"),
        (1, "7"),
        (2, "MIG-c"),
    ]


def test_base_only_gpu_resolution_rejects_outside_allocation():
    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        resolve_base_only_gpus("0,2", cuda_visible_devices="GPU-a,GPU-b")


class FakePolicyModel:
    def __init__(self):
        self.snapshots = []

    def save_pretrained(self, path):
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "adapter_model.safetensors").write_bytes(b"adapter")
        self.snapshots.append(destination)


class FakeTrainingBackend:
    renderer_source = "hf"
    sampler = "vllm"
    use_lora = True
    vllm_options = {"gpu_memory_utilization": 0.8}

    def __init__(self):
        self.model = FakePolicyModel()
        self.setup_calls = []
        self.shutdown_calls = 0
        self.local_sampler_calls = 0
        self.score_calls = []

    def setup(self, **kwargs):
        self.setup_calls.append(kwargs)

    def policy_sampler(self, _name):
        self.local_sampler_calls += 1
        raise AssertionError("coordinator vLLM sampler must remain cold")

    def base_sampler(self):
        self.local_sampler_calls += 1
        raise AssertionError("coordinator vLLM sampler must remain cold")

    def _score_completions(self, prompts, completions, *, use_base):
        self.score_calls.append((prompts, completions, use_base))
        return [[-0.5] * len(tokens) for tokens in completions]

    def shutdown(self):
        self.shutdown_calls += 1


def test_backend_proxy_snapshots_refreshes_and_scores_on_rollout_workers(tmp_path):
    pool, endpoints = make_pool(tmp_path / "pool", 2)
    training_backend = FakeTrainingBackend()
    backend = RolloutParallelBackend(
        training_backend,
        gpus=pool.gpus,
        status_dir=tmp_path / "unused",
        pool=pool,
    )
    backend.setup(model="fake/model", lora=object())

    assert training_backend.local_sampler_calls == 0
    assert pool.adapter_version == 1
    assert all(endpoint.adapter_version == 1 for endpoint in endpoints)
    assert (pool.status_dir / "adapters" / "v00000001" / "ctm_rollout_adapter.json").is_file()

    asyncio.run(backend.refresh_policy_sampler("step-1"))
    assert pool.adapter_version == 2
    assert all(endpoint.adapter_version == 2 for endpoint in endpoints)

    handle = backend.policy_sampler("policy")
    sampled = asyncio.run(
        handle.sample_batch(
            [
                types.ModelInput.from_ints(tokens=[10]),
                types.ModelInput.from_ints(tokens=[20]),
                types.ModelInput.from_ints(tokens=[30]),
            ],
            max_tokens=None,
            temperature=0.7,
            stop=[],
            num_samples=2,
        )
    )
    assert [[sequence.tokens[:2] for sequence in group] for group in sampled] == [
        [[0, 0], [0, 1]],
        [[1, 0], [1, 1]],
        [[2, 0], [2, 1]],
    ]
    assert all(
        len([command for command in endpoint.commands if command["op"] == "sample_batch"]) == 1
        for endpoint in endpoints
    )

    async def sample_concurrently():
        return await asyncio.gather(
            *[
                handle.sample(
                    types.ModelInput.from_ints(tokens=[token]),
                    max_tokens=None,
                    temperature=0.7,
                    stop=[],
                    num_samples=2,
                )
                for token in (40, 50, 60)
            ]
        )

    coalesced = asyncio.run(sample_concurrently())
    assert len(coalesced) == 3
    assert all(
        len([command for command in endpoint.commands if command["op"] == "sample_batch"]) == 2
        for endpoint in endpoints
    )
    assert all(
        command["max_tokens"] is None
        for endpoint in endpoints
        for command in endpoint.commands
        if command["op"] == "sample_batch"
    )
    scores = asyncio.run(
        handle.score_completions(
            [types.ModelInput.from_ints(tokens=[1])],
            [[2, 3]],
        )
    )
    assert scores == [[-0.002, -0.003]]
    assert not hasattr(handle, "generation_logprobs_are_raw_policy")
    score_commands = [
        command for endpoint in endpoints for command in endpoint.commands if command["op"] == "score_batch"
    ]
    assert len(score_commands) == 2
    assert all(command["use_base"] is False and command["adapter_version"] == 2 for command in score_commands)
    assert training_backend.score_calls == []
    assert training_backend.local_sampler_calls == 0

    backend.shutdown()
    backend.shutdown()
    assert training_backend.shutdown_calls == 1
