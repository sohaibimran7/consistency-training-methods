"""RL scheduler contract for phase-shared rollout/training GPUs."""

from __future__ import annotations

import asyncio
import random
import re
from unittest.mock import MagicMock, patch

import torch
from tinker import types

from ctm.backends.base import ForwardBackwardOutput, SampledSequence
from ctm.core.config import AdamConfig, CheckpointConfig, LoRAConfig
from ctm.core.rewards import ConsistencyReward
from ctm.training.rl import GenerationConfig, RateEstimationConfig, RLConfig, RLTrainer, TrainingLoopConfig, TrainingSamplingConfig

_A, _B = 65, 66


class _Tokenizer:
    def decode(self, tokens):
        return "The answer is (A)" if tokens and tokens[0] == _A else "The answer is (B)"


class _Renderer:
    def build_generation_prompt(self, messages):
        content = " ".join(str(message["content"]) for message in messages)
        question_id = int(re.search(r"Q(\d+)", content).group(1))
        cued = int("bias" in content)
        return types.ModelInput.from_ints(tokens=[question_id * 2 + cued])

    def parse_response(self, tokens):
        raise RuntimeError("exercise decode fallback")

    def get_stop_sequences(self):
        return []


class _Sampler:
    def __init__(self, events):
        self.events = events

    async def sample(self, prompt, *, max_tokens, temperature, stop, num_samples):
        encoded = prompt.to_ints()[0]
        question_id, cued = divmod(encoded, 2)
        self.events.append(("sample", question_id, cued))
        return [
            SampledSequence(
                tokens=[_A if ((index % 4 != 0) if cued else (index % 4 == 0)) else _B],
                logprobs=[-0.1],
                finish_reason="stop",
            )
            for index in range(num_samples)
        ]


class _PhaseSharedBackend:
    """Small backend whose pending F/B exposes a scheduling yield point."""

    renderer_source = "tinker"
    sampling_training_overlap_supported = False

    def __init__(self, events):
        self.events = events
        self.shutdown_calls = 0

    def setup(self, **_kwargs):
        return None

    def policy_sampler(self, _name):
        return _Sampler(self.events)

    def base_sampler(self):
        return _Sampler(self.events)

    async def refresh_policy_sampler(self, name):
        del name
        self.events.append(("refresh_policy_sampler",))
        return _Sampler(self.events)

    async def enter_rollout_phase(self):
        self.events.append(("rollout_phase",))

    async def enter_training_phase(self):
        self.events.append(("training_phase",))

    async def submit_forward_backward(self, datums, loss_fn):
        self.events.append(("fwd_submitted", loss_fn))
        lengths = [datum.loss_fn_inputs["target_tokens"].to_torch().shape[0] for datum in datums]
        events = self.events

        class _Pending:
            async def result(self):
                events.append(("fwd_result_started",))
                # If the RL scheduler had refilled its prefetch queue, the
                # pending rollout task would become runnable at this yield.
                await asyncio.sleep(0)
                events.append(("fwd_result_completed",))
                return ForwardBackwardOutput(
                    logprobs=[-0.1 * torch.ones(length) for length in lengths],
                    metrics={"loss": 0.5},
                )

        return _Pending()

    async def submit_optim_step(self, *, learning_rate, adam):
        self.events.append(("optim_submitted", learning_rate))

        class _Pending:
            async def result(self):
                return None

        return _Pending()

    async def incorporate_kl_penalty(self, datums, *, kl_coef, kl_discount_factor):
        raise AssertionError("KL is disabled for this scheduling-only test")

    async def save_checkpoint(self, *, name, log_dir, loop_state, kind):
        self.events.append(("checkpoint_started", name))
        # Make a scheduler yield point explicit: a rollout wake scheduled too
        # early would become observable between these two events.
        await asyncio.sleep(0)
        self.events.append(("checkpoint_completed", name))
        return {"sampler_path": f"fake://{name}", "state_path": None}

    def shutdown(self):
        self.shutdown_calls += 1


def _perturbations():
    return [
        lambda datapoint: {"messages": [{"role": "user", "content": datapoint["question"]}]},
        lambda datapoint: {"messages": [{"role": "user", "content": "bias " + datapoint["question"]}]},
    ]


def _trait(answer_text, _datapoint, _messages):
    return float("(A)" in answer_text)


def test_phase_shared_rl_scheduler_never_prefetches_while_forward_backward_is_pending(tmp_path):
    events = []
    backend = _PhaseSharedBackend(events)
    config = RLConfig(
        experiment_name="phase-shared",
        run_name="prefetch-barrier",
        lora=LoRAConfig(rank=4, seed=0),
        optimizer=AdamConfig(learning_rate=1e-4),
        reference_rate=RateEstimationConfig(perturbation_indices=[0], n_rollouts=4),
        training=TrainingSamplingConfig(
            perturbation_indices=[1],
            n_rollouts_for_rate=4,
            n_rollouts_for_consistency=None,
        ),
        # Two batches in one accumulation window are the case that normally
        # prefetches batch two while batch one trains.
        loop=TrainingLoopConfig(batch_size=1, gradient_accumulation_steps=2, refresh_policy_every_n_steps=0, n_epochs=1),
        generation=GenerationConfig(max_new_tokens=8, temperature=0.7),
        checkpoint=CheckpointConfig(),
        kl_coef=0.0,
        anchor_weight=0.0,
        log_base_dir=str(tmp_path / "logs"),
        rollout_dir=str(tmp_path / "rollouts"),
    )
    trainer = RLTrainer(config=config, backend=backend, reward_function=ConsistencyReward())
    trainer.setup_done = True
    trainer.renderer = _Renderer()
    trainer.tokenizer = _Tokenizer()
    trainer.sampling_client = _Sampler(events)
    trainer.base_sampling_client = _Sampler(events)
    trainer.anchor_sampling_client = trainer.base_sampling_client

    random.seed(0)
    with patch("ctm.training.rl.setup_logging", return_value=MagicMock()):
        asyncio.run(
            trainer.train(
                datapoints=[{"question": "Q0"}, {"question": "Q1"}],
                perturbation_fns=_perturbations(),
                trait_classifier=_trait,
            )
        )

    first_training = events.index(("training_phase",))
    first_submit = events.index(("fwd_submitted", "ppo"))
    first_started = events.index(("fwd_result_started",))
    first_completed = events.index(("fwd_result_completed",))
    assert first_training < first_submit < first_started < first_completed
    assert not any(event[0] == "sample" for event in events[first_started + 1 : first_completed])

    # The next rollout is queued only after the first F/B result is complete
    # and the explicit rollout barrier has restored worker resources.
    next_sample = next(index for index, event in enumerate(events) if index > first_completed and event[0] == "sample")
    assert any(event == ("rollout_phase",) for event in events[first_completed + 1 : next_sample])
    assert backend.shutdown_calls == 1


def test_phase_shared_rl_serializes_checkpoint_before_refresh_and_next_rollout_wake(tmp_path):
    """A checkpoint is trainer work, not a gap in the phase barrier.

    The first of two optimizer steps produces a scheduled intermediate
    checkpoint.  Its async completion must precede both policy publication and
    the rollout wake that permits the second batch to sample.
    """

    events = []
    backend = _PhaseSharedBackend(events)
    config = RLConfig(
        experiment_name="phase-shared",
        run_name="checkpoint-barrier",
        lora=LoRAConfig(rank=4, seed=0),
        optimizer=AdamConfig(learning_rate=1e-4),
        reference_rate=RateEstimationConfig(perturbation_indices=[0], n_rollouts=4),
        training=TrainingSamplingConfig(
            perturbation_indices=[1],
            n_rollouts_for_rate=4,
            n_rollouts_for_consistency=None,
        ),
        loop=TrainingLoopConfig(batch_size=1, gradient_accumulation_steps=1, refresh_policy_every_n_steps=1, n_epochs=1),
        generation=GenerationConfig(max_new_tokens=8, temperature=0.7),
        checkpoint=CheckpointConfig(save_every_n_steps=1, skip_near_final_steps=0),
        kl_coef=0.0,
        anchor_weight=0.0,
        log_base_dir=str(tmp_path / "logs"),
        rollout_dir=str(tmp_path / "rollouts"),
    )
    trainer = RLTrainer(config=config, backend=backend, reward_function=ConsistencyReward())
    trainer.setup_done = True
    trainer.renderer = _Renderer()
    trainer.tokenizer = _Tokenizer()
    trainer.sampling_client = _Sampler(events)
    trainer.base_sampling_client = _Sampler(events)
    trainer.anchor_sampling_client = trainer.base_sampling_client

    random.seed(0)
    with patch("ctm.training.rl.setup_logging", return_value=MagicMock()):
        asyncio.run(
            trainer.train(
                datapoints=[{"question": "Q0"}, {"question": "Q1"}],
                perturbation_fns=_perturbations(),
                trait_classifier=_trait,
            )
        )

    checkpoint_started = next(index for index, event in enumerate(events) if event[0] == "checkpoint_started")
    checkpoint_completed = next(index for index, event in enumerate(events) if event[0] == "checkpoint_completed")
    refresh = next(index for index, event in enumerate(events) if event == ("refresh_policy_sampler",))
    next_rollout_phase = next(
        index for index, event in enumerate(events) if index > refresh and event == ("rollout_phase",)
    )
    next_sample = next(index for index, event in enumerate(events) if index > next_rollout_phase and event[0] == "sample")

    assert checkpoint_started < checkpoint_completed < refresh < next_rollout_phase < next_sample
