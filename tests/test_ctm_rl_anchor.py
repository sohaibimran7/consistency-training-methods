import asyncio
import random

import pytest
from tinker import types

from ctm.backends.base import SampledSequence
from ctm.core.advantages import normalize_grouped
from ctm.core.rewards import ConsistencyReward
from ctm.core.types import BatchItem, Rollout
from ctm.training.rl import (
    RateEstimationConfig,
    RLConfig,
    RLTrainer,
    TrainingLoopConfig,
    TrainingSamplingConfig,
    _epoch_datapoint_batches,
)


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


def test_datapoint_shuffle_defaults_to_the_historical_epoch_shuffle():
    assert TrainingLoopConfig().shuffle_datapoints is True

    random.seed(17)
    actual = _epoch_datapoint_batches(6, 2, shuffle_datapoints=True)
    random.seed(17)
    expected_order = list(range(6))
    random.shuffle(expected_order)

    assert actual == [expected_order[0:2], expected_order[2:4], expected_order[4:6]]


def test_no_datapoint_shuffle_preserves_each_supplied_batch_order():
    assert TrainingLoopConfig(shuffle_datapoints=False).shuffle_datapoints is False

    assert _epoch_datapoint_batches(6, 2, shuffle_datapoints=False) == [[0, 1], [2, 3], [4, 5]]
    # A partial final batch remains in source order too.
    assert _epoch_datapoint_batches(5, 2, shuffle_datapoints=False) == [[0, 1], [2, 3], [4]]


def test_four_item_pooled_advantages_match_official_global_standardization():
    """Paper-style b=4 must concatenate rewards before standardizing them once.

    The upstream RMCT trainer's ``_build_training_batch`` concatenates the
    consistency rewards for all four items, then calls its global
    ``_normalize_advantages`` once.  This fixture gives each item a different
    gap magnitude, so per-item normalization must produce a different result.
    """

    def make_item(idx: int, traits: list[float], p_hat: float, p_ref: float) -> BatchItem:
        rollouts = [
            Rollout(
                tokens=[65],
                logprobs=[-0.1],
                text="answer",
                trait_value=trait,
                perturbation_idx=1,
                prompt=types.ModelInput.from_ints(tokens=[idx + 1]),
            )
            for trait in traits
        ]
        return BatchItem(
            datapoint_idx=idx,
            datapoint={"question": f"Q{idx}"},
            train_rollouts=rollouts,
            anchor_rollouts=[],
            sampled_rollouts=rollouts,
            initial_rollouts=[],
            p_hat={1: p_hat},
            p_hat_counts={1: len(rollouts)},
            p_ref=p_ref,
            p_ref_init=None,
            reference_rates={0: p_ref},
            reference_rate_counts={0: len(rollouts)},
            initial_reference_rates={},
            initial_reference_rate_counts={},
            n_total=len(rollouts),
            n_parsed=len(rollouts),
            n_ref_parsed=len(rollouts),
            n_training_parsed=len(rollouts),
        )

    # The listed p_hats are the empirical rates of their trait vectors.  Gaps
    # are deliberately unequal (.25, .25, .50, .50), which is what a pooled
    # b=4 baseline preserves and per-item normalization removes.
    items = [
        make_item(0, [1.0, 0.0, 0.0, 0.0], p_hat=0.25, p_ref=0.0),
        make_item(1, [1.0, 1.0, 0.0, 0.0], p_hat=0.50, p_ref=0.25),
        make_item(2, [1.0, 1.0, 1.0, 0.0], p_hat=0.75, p_ref=0.25),
        make_item(3, [1.0, 1.0, 0.0, 0.0], p_hat=0.50, p_ref=0.0),
    ]

    def standardize(values: list[float]) -> list[float]:
        mean = sum(values) / len(values)
        std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
        return [(value - mean) / std for value in values]

    expected_rewards = [
        -(item.p_hat[1] - item.p_ref) * (float(rollout.trait_value) - item.p_hat[1])
        for item in items
        for rollout in item.train_rollouts
    ]
    official_global_advantages = standardize(expected_rewards)

    def build_with(normalize: str) -> tuple[list, list[float], list[float], list[float], list[tuple]]:
        trainer = RLTrainer(
            config=RLConfig(
                anchor_weight=0.0,
                training=TrainingSamplingConfig(perturbation_indices=[1]),
                loop=TrainingLoopConfig(batch_size=4, gradient_accumulation_steps=1, normalize=normalize),
            ),
            backend=_UnusedBackend(),
        )
        return trainer._build_training_batch(items)

    pooled_datums, pooled_rewards, pooled_anchor_rewards, pooled_advantages, _ = build_with("pooled")
    _, _, _, per_item_advantages, _ = build_with("per_item")

    assert len(pooled_datums) == 16
    assert pooled_rewards == pytest.approx(expected_rewards)
    assert pooled_anchor_rewards == []
    assert pooled_advantages == pytest.approx(official_global_advantages)

    expected_per_item_advantages = [
        advantage
        for start in range(0, len(expected_rewards), 4)
        for advantage in standardize(expected_rewards[start : start + 4])
    ]
    assert per_item_advantages == pytest.approx(expected_per_item_advantages)
    assert max(abs(pooled - per_item) for pooled, per_item in zip(pooled_advantages, per_item_advantages)) > 0.1


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
        return [SampledSequence(tokens=[65 if i % 2 == 0 else 66], logprobs=[-0.1], finish_reason="stop") for i in range(num_samples)]


class _RetrySampler:
    def __init__(self):
        self.calls = 0

    async def sample(self, prompt, *, max_tokens, temperature, stop, num_samples):
        self.calls += 1
        logprobs = None if self.calls == 1 else [-0.1]
        return [SampledSequence(tokens=[65], logprobs=logprobs, finish_reason="stop") for _ in range(num_samples)]


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
