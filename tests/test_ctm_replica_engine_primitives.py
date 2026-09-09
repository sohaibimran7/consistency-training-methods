"""CPU contracts for exact replicated LocalBackend primitives.

These tests deliberately stop at the engine boundary.  They compare manually
summed shard gradients against the existing one-process objective, leaving
process launch and collectives to the replicated-backend tests.
"""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest
import torch
from tinker import types
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.supervised.common import datum_from_model_input_weights

from ctm.backends.local.engine import LocalBackend, LocalSamplerHandle
from ctm.core.config import AdamConfig, LoRAConfig


class _TinyCausalLM(torch.nn.Module):
    """Small deterministic CausalLM-shaped module with no dropout."""

    def __init__(self, *, vocab_size: int = 31, hidden_size: int = 11) -> None:
        super().__init__()
        torch.manual_seed(41)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, vocab_size)
        self.forward_calls = 0

    def forward(self, *, input_ids, attention_mask=None, use_cache=None):
        assert use_cache is False
        self.forward_calls += 1
        hidden = self.embedding(input_ids)
        positions = torch.arange(1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.cumsum(dim=1) / positions[None, :, None]
        return SimpleNamespace(logits=self.projection(hidden))


def _backend(*, gradient_reducer=None) -> LocalBackend:
    backend = LocalBackend(
        device="cpu",
        use_lora=False,
        model_instance=_TinyCausalLM(),
        forward_microbatch_max_datums=None,
        forward_microbatch_max_tokens=None,
        gradient_reducer=gradient_reducer,
    )
    backend.setup(model="replica-engine-tiny", lora=LoRAConfig(rank=2))
    return backend


def _identical_backends(count: int) -> list[LocalBackend]:
    template = _TinyCausalLM()
    backends = []
    for _ in range(count):
        backend = LocalBackend(
            device="cpu",
            use_lora=False,
            model_instance=copy.deepcopy(template),
            forward_microbatch_max_datums=None,
            forward_microbatch_max_tokens=None,
        )
        backend.setup(model="replica-engine-tiny", lora=LoRAConfig(rank=2))
        backends.append(backend)
    return backends


def _sft_datum(tokens: tuple[int, ...], *, prompt_tokens: int):
    weights = torch.tensor([0.0] * prompt_tokens + [1.0] * (len(tokens) - prompt_tokens))
    return datum_from_model_input_weights(types.ModelInput.from_ints(tokens=list(tokens)), weights)


def _rl_datum(
    backend: LocalBackend,
    *,
    prompt: tuple[int, ...],
    action: tuple[int, ...],
    advantage: float,
):
    tokens = prompt + action
    probe = datum_from_model_input_weights(
        types.ModelInput.from_ints(tokens=list(tokens)),
        torch.ones(len(tokens)),
    )
    with torch.no_grad():
        target_logprobs = backend._target_logprobs([probe])[0]
    transition = Transition(
        ob=types.ModelInput.from_ints(tokens=list(prompt)),
        ac=TokensWithLogprobs(
            tokens=list(action),
            maybe_logprobs=target_logprobs[len(prompt) - 1 :].tolist(),
        ),
        reward=0.0,
        episode_done=True,
    )
    trajectory = Trajectory(transitions=[transition], final_ob=types.ModelInput.from_ints(tokens=[]))
    return trajectory_to_data(trajectory, traj_advantage=advantage)[0]


def _datums(backend: LocalBackend, loss_fn: str):
    if loss_fn == "cross_entropy":
        return [
            _sft_datum((2, 3, 4, 5, 6, 7), prompt_tokens=2),
            _sft_datum((8, 7, 6, 5, 4), prompt_tokens=1),
            _sft_datum((3, 5, 7, 9, 11, 13, 15), prompt_tokens=3),
            _sft_datum((14, 12, 10, 8), prompt_tokens=1),
        ]

    datums = [
        _rl_datum(backend, prompt=(2, 3), action=(4, 5, 6), advantage=1.25),
        _rl_datum(backend, prompt=(7,), action=(8, 9), advantage=-0.75),
        _rl_datum(backend, prompt=(10, 11), action=(12, 13, 14), advantage=0.5),
        _rl_datum(backend, prompt=(15, 16, 17), action=(18, 19), advantage=2.0),
    ]
    # Put PPO on both clipping branches and avoid a vacuous IS ratio of one.
    for datum, offset in zip(datums, (-0.45, 0.30, -0.25, 0.15)):
        sampled = datum.loss_fn_inputs["logprobs"].to_torch()
        datum.loss_fn_inputs["logprobs"] = types.TensorData.from_torch(sampled + offset)
    return datums


async def _forward_backward(
    backend: LocalBackend,
    datums,
    loss_fn: str,
    *,
    global_loss_denominator: torch.Tensor | None = None,
):
    pending = await backend.submit_forward_backward(
        datums,
        loss_fn,
        global_loss_denominator=global_loss_denominator,
    )
    return await pending.result()


async def _optim_step(backend: LocalBackend, *, adam: AdamConfig) -> None:
    pending = await backend.submit_optim_step(learning_rate=1e-3, adam=adam)
    await pending.result()


def _assert_summed_shard_gradients(reference: LocalBackend, *shards: LocalBackend) -> None:
    for reference_parameter, *shard_parameters in zip(
        reference.model.parameters(),
        *(shard.model.parameters() for shard in shards),
    ):
        assert reference_parameter.grad is not None
        total = torch.zeros_like(reference_parameter.grad)
        for parameter in shard_parameters:
            if parameter.grad is not None:
                total.add_(parameter.grad)
        torch.testing.assert_close(total, reference_parameter.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("loss_fn", ["cross_entropy", "importance_sampling", "ppo"])
def test_global_denominator_shards_match_one_logical_batch(loss_fn):
    whole, first_shard, second_shard = _identical_backends(3)
    datums = _datums(whole, loss_fn)
    left, right = datums[:2], datums[2:]
    global_denominator = whole.logical_loss_denominator(datums, loss_fn)

    whole_output = asyncio.run(_forward_backward(whole, datums, loss_fn))
    left_output = asyncio.run(
        _forward_backward(
            first_shard,
            left,
            loss_fn,
            global_loss_denominator=global_denominator,
        )
    )
    right_output = asyncio.run(
        _forward_backward(
            second_shard,
            right,
            loss_fn,
            global_loss_denominator=global_denominator,
        )
    )

    shard_denominator = first_shard.logical_loss_denominator(left, loss_fn)
    shard_denominator = shard_denominator + second_shard.logical_loss_denominator(right, loss_fn)
    torch.testing.assert_close(shard_denominator, global_denominator)
    assert left_output.metrics["loss"] + right_output.metrics["loss"] == pytest.approx(
        whole_output.metrics["loss"],
        abs=1e-6,
    )
    _assert_summed_shard_gradients(whole, first_shard, second_shard)


@pytest.mark.parametrize("loss_fn", ["cross_entropy", "importance_sampling", "ppo"])
def test_global_denominator_empty_shard_is_zero_contribution(loss_fn):
    whole, populated, empty = _identical_backends(3)
    datums = _datums(whole, loss_fn)
    global_denominator = whole.logical_loss_denominator(datums, loss_fn)

    whole_output = asyncio.run(_forward_backward(whole, datums, loss_fn))
    populated_output = asyncio.run(
        _forward_backward(
            populated,
            datums,
            loss_fn,
            global_loss_denominator=global_denominator,
        )
    )
    empty_output = asyncio.run(
        _forward_backward(
            empty,
            [],
            loss_fn,
            global_loss_denominator=global_denominator,
        )
    )

    assert populated_output.metrics["loss"] == pytest.approx(whole_output.metrics["loss"], abs=1e-6)
    assert empty_output.logprobs == []
    assert empty_output.metrics["loss"] == pytest.approx(0.0)
    assert empty.model.forward_calls == 0
    assert empty._gradient_accumulations == 1
    assert all(parameter.grad is None for parameter in empty.model.parameters())
    _assert_summed_shard_gradients(whole, populated, empty)


@pytest.mark.parametrize("loss_fn", ["cross_entropy", "importance_sampling", "ppo"])
def test_global_denominator_preserves_multi_submission_accumulation(loss_fn):
    whole, first_shard, second_shard = _identical_backends(3)
    datums = _datums(whole, loss_fn)
    batches = (datums[:2], datums[2:])

    for batch in batches:
        global_denominator = whole.logical_loss_denominator(batch, loss_fn)
        asyncio.run(_forward_backward(whole, batch, loss_fn))
        asyncio.run(
            _forward_backward(
                first_shard,
                batch[:1],
                loss_fn,
                global_loss_denominator=global_denominator,
            )
        )
        asyncio.run(
            _forward_backward(
                second_shard,
                batch[1:],
                loss_fn,
                global_loss_denominator=global_denominator,
            )
        )

    assert whole._gradient_accumulations == first_shard._gradient_accumulations == 2
    assert second_shard._gradient_accumulations == 2
    _assert_summed_shard_gradients(whole, first_shard, second_shard)

    # Simulate the exact SUM reducer before each rank's optimizer step.  The
    # engine still performs its historical average over the two public
    # submissions, so the post-step parameters must match the unsharded run.
    for first_parameter, second_parameter in zip(first_shard.model.parameters(), second_shard.model.parameters()):
        if first_parameter.grad is None:
            assert second_parameter.grad is None
            continue
        if second_parameter.grad is not None:
            first_parameter.grad.add_(second_parameter.grad)
    adam = AdamConfig(weight_decay=0.0, grad_clip_norm=0.0)
    asyncio.run(_optim_step(whole, adam=adam))
    asyncio.run(_optim_step(first_shard, adam=adam))
    for whole_parameter, shard_parameter in zip(whole.model.parameters(), first_shard.model.parameters()):
        torch.testing.assert_close(shard_parameter, whole_parameter, atol=1e-6, rtol=1e-5)


def test_gradient_reducer_runs_once_after_accumulation_before_clipping(monkeypatch):
    events: list[str] = []

    def reducer(parameters):
        events.append("reduce")
        assert parameters
        assert all(parameter.requires_grad for parameter in parameters)
        assert all(parameter.grad is not None for parameter in parameters)

    backend = _backend(gradient_reducer=reducer)
    backend.model.projection.bias.requires_grad_(False)
    datums = _datums(backend, "cross_entropy")
    asyncio.run(_forward_backward(backend, datums[:2], "cross_entropy"))
    asyncio.run(_forward_backward(backend, datums[2:], "cross_entropy"))

    original_clip = torch.nn.utils.clip_grad_norm_

    def record_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        events.append("clip")
        assert events == ["reduce", "clip"]
        return original_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    asyncio.run(_optim_step(backend, adam=AdamConfig(weight_decay=0.0, grad_clip_norm=0.5)))

    assert events == ["reduce", "clip"]


def test_default_gradient_reducer_is_a_noop(monkeypatch):
    backend = _backend()
    assert backend._gradient_reducer is None

    import ctm.backends.local.engine as engine_module

    def should_not_reduce(_parameters):
        raise AssertionError("default LocalBackend must not invoke a gradient reducer")

    monkeypatch.setattr(engine_module, "sum_trainable_gradients_torch_distributed", should_not_reduce)
    datums = _datums(backend, "cross_entropy")
    asyncio.run(_forward_backward(backend, datums, "cross_entropy"))
    asyncio.run(_optim_step(backend, adam=AdamConfig(weight_decay=0.0, grad_clip_norm=0.0)))


def test_replicated_opct_empty_shard_reaches_optimizer_collective_without_forward():
    backend = _backend()

    async def run_empty_shard():
        pending = await backend.submit_opct_forward_backward(
            [],
            behavior_temperature=0.7,
            kl_coef=1.0,
            kl_discount_factor=0.0,
            loss_fn="importance_sampling",
            global_loss_denominator=17.0,
        )
        return await pending.result()

    output = asyncio.run(run_empty_shard())

    assert output.logprobs == []
    assert output.metrics == {
        "loss": 0.0,
        "teacher_kl": 0.0,
        "student_entropy": 0.0,
        "teacher_cross_entropy": 0.0,
        "teacher_scored_tokens": 17.0,
    }
    assert backend.model.forward_calls == 0
    assert backend._gradient_accumulations == 1
    assert all(parameter.grad is None for parameter in backend.model.parameters())


class _SleepSampler:
    def __init__(self) -> None:
        self.sleeping = False
        self.sleep_transitions = 0
        self.wake_transitions = 0

    def sleep(self) -> bool:
        if self.sleeping:
            return True
        self.sleeping = True
        self.sleep_transitions += 1
        return True

    def wake_up(self) -> bool:
        if not self.sleeping:
            return True
        self.sleeping = False
        self.wake_transitions += 1
        return True


def test_inprocess_sleep_lifecycle_is_idempotent_and_keeps_lazy_boot(monkeypatch):
    backend = _backend()
    backend.sampler = "vllm"
    backend.vllm_options = {"enable_sleep_mode": True}
    assert backend.sampling_training_overlap_supported is False
    assert backend._vllm is None

    async def before_boot():
        await backend.enter_training_phase()
        await backend.enter_rollout_phase()

    asyncio.run(before_boot())
    assert backend._vllm is None

    sampler = _SleepSampler()
    boot_calls: list[bool] = []

    def boot_sampler() -> None:
        boot_calls.append(True)
        backend._vllm = sampler

    monkeypatch.setattr(backend, "_ensure_vllm", boot_sampler)
    handle = backend.policy_sampler("policy")
    assert isinstance(handle, LocalSamplerHandle)
    assert boot_calls == [True]

    async def exercise_lifecycle():
        await backend.enter_training_phase()
        await backend.enter_training_phase()
        await backend.enter_rollout_phase()
        await backend.enter_rollout_phase()

    asyncio.run(exercise_lifecycle())
    assert sampler.sleep_transitions == 1
    assert sampler.wake_transitions == 1
    assert sampler.sleeping is False

    plain = _backend()
    assert plain.sampling_training_overlap_supported is True
