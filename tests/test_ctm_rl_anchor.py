import asyncio

import pytest
from tinker import types

from ctm.backends.base import SampledSequence
from ctm.core.advantages import normalize_grouped
from ctm.core.rewards import ConsistencyReward
from ctm.core.types import Rollout
from ctm.training.rl import RateEstimationConfig, RLConfig, RLTrainer, TrainingLoopConfig, TrainingSamplingConfig


class _UnusedBackend:
    """Backend placeholder for unit tests that exercise rollout accounting only."""


def _rollout(reference_index: int, trait: float) -> Rollout:
    return Rollout(
        tokens=[65],
        logprobs=[-0.1],
        text="answer",
        trait_value=trait,
        perturbation_idx=reference_index,
    )


def test_per_item_normalization_is_the_default_and_pooled_remains_available():
    assert TrainingLoopConfig().normalize == "per_item"
    assert TrainingLoopConfig(normalize="pooled").normalize == "pooled"

    rewards = [0.0, 2.0, 0.0, 10.0]
    slices = [(0, 2), (2, 4)]
    assert normalize_grouped(rewards, slices) == pytest.approx([-1.0, 1.0, -1.0, 1.0])
    assert normalize_grouped(rewards, slices, mode="pooled") != pytest.approx([-1.0, 1.0, -1.0, 1.0])


def test_anchor_reward_uses_each_references_own_current_and_initial_rate():
    rollouts = [
        _rollout(0, 1.0),
        _rollout(0, 0.0),
        _rollout(1, 1.0),
        _rollout(1, 0.0),
    ]
    rewards = ConsistencyReward().compute_anchor_rewards(
        rollouts,
        reference_rates={0: 0.75, 1: 0.25},
        initial_reference_rates={0: 0.25, 1: 0.75},
    )
    assert rewards == pytest.approx([-0.125, 0.375, 0.375, -0.125])


class _IndexRenderer:
    def build_generation_prompt(self, messages):
        return types.ModelInput.from_ints(tokens=[int(messages[0]["content"])])

    def parse_response(self, tokens):
        raise RuntimeError("exercise decode fallback")

    def get_stop_sequences(self):
        return []


class _AnswerTokenizer:
    def decode(self, tokens):
        return "(A)" if tokens and tokens[0] == 65 else "(B)"


class _CountingSampler:
    def __init__(self):
        self.calls = []

    async def sample(self, prompt, *, max_tokens, temperature, stop, num_samples):
        self.calls.append((prompt.to_ints()[0], num_samples))
        return [SampledSequence(tokens=[65 if i % 2 == 0 else 66], logprobs=[-0.1]) for i in range(num_samples)]


class _RetrySampler:
    def __init__(self):
        self.calls = 0

    async def sample(self, prompt, *, max_tokens, temperature, stop, num_samples):
        self.calls += 1
        logprobs = None if self.calls == 1 else [-0.1]
        return [SampledSequence(tokens=[65], logprobs=logprobs) for _ in range(num_samples)]


def test_rate_only_collection_samples_only_requested_reference_indices():
    config = RLConfig(
        reference_rate=RateEstimationConfig(perturbation_indices=[0, 1], n_rollouts=4),
        training=TrainingSamplingConfig(perturbation_indices=[2], n_rollouts_for_rate=9),
        anchor_weight=0.0,
    )
    trainer = RLTrainer(config=config, backend=_UnusedBackend())
    trainer.renderer = _IndexRenderer()
    trainer.tokenizer = _AnswerTokenizer()
    sampler = _CountingSampler()
    trainer.sampling_client = sampler

    perturbations = [lambda _dp, idx=idx: {"messages": [{"role": "user", "content": str(idx)}]} for idx in range(3)]
    result = asyncio.run(
        trainer._collect_rollouts(
            {},
            perturbations,
            lambda answer, _dp, _messages: float("(A)" in answer),
            answer_parser=lambda answer: answer,
            rates_only=True,
            requested_indices=[0, 1],
        )
    )

    assert sampler.calls == [(0, 4), (1, 4)]
    assert set(result.rates) == {0, 1}
    assert result.train_rollouts == []
    assert result.anchor_rollouts == []


def test_trait_abstentions_are_excluded_from_rates_and_counted():
    config = RLConfig(
        reference_rate=RateEstimationConfig(perturbation_indices=[0], n_rollouts=4),
        training=TrainingSamplingConfig(perturbation_indices=[1], n_rollouts_for_rate=4),
        anchor_weight=0.0,
    )
    trainer = RLTrainer(config=config, backend=_UnusedBackend())
    trainer.renderer = _IndexRenderer()
    trainer.tokenizer = _AnswerTokenizer()
    trainer.sampling_client = _CountingSampler()
    perturbations = [lambda _dp, idx=idx: {"messages": [{"role": "user", "content": str(idx)}]} for idx in range(2)]

    result = asyncio.run(
        trainer._collect_rollouts(
            {},
            perturbations,
            lambda _answer, _dp, _messages: None,
            answer_parser=lambda answer: answer,
            rates_only=True,
            requested_indices=[0],
        )
    )

    assert result.rates == {0: None}
    assert result.rate_counts == {0: 0}
    assert result.n_parsed == 0
    assert result.n_trait_abstained == 4


def test_resampling_retains_failed_attempts_for_logging():
    config = RLConfig(
        reference_rate=RateEstimationConfig(perturbation_indices=[0], n_rollouts=2),
        training=TrainingSamplingConfig(perturbation_indices=[1], n_rollouts_for_rate=2),
        anchor_weight=0.0,
        unparsed_handling="resample",
        max_resample_attempts=2,
    )
    trainer = RLTrainer(config=config, backend=_UnusedBackend())
    trainer.renderer = _IndexRenderer()
    trainer.tokenizer = _AnswerTokenizer()
    trainer.sampling_client = _RetrySampler()
    perturbations = [lambda _dp, idx=idx: {"messages": [{"role": "user", "content": str(idx)}]} for idx in range(2)]

    result = asyncio.run(
        trainer._collect_rollouts(
            {},
            perturbations,
            lambda answer, _dp, _messages: float("(A)" in answer),
            answer_parser=lambda answer: answer,
            rates_only=True,
            requested_indices=[0],
        )
    )

    assert len(result.sampled_rollouts) == 4
    assert result.rate_counts == {0: 2}
    rejected = result.sampled_rollouts[:2]
    assert all(not rollout.has_logprobs for rollout in rejected)
    assert all(not rollout.grader_evaluated for rollout in rejected)
