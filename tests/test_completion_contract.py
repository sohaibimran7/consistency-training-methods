"""Incomplete responses must never become negative traits or gradients."""
import asyncio
from types import SimpleNamespace

import pytest

from ctm.backends.base import SampledSequence
from ctm.backends.local.vllm_sampler import VLLMSampler
from ctm.training.rl import RLConfig, RLTrainer, RateEstimationConfig, TrainingSamplingConfig
from tests.test_ctm_rl_anchor import _UnusedBackend, _IndexRenderer, _AnswerTokenizer


@pytest.mark.parametrize("reason", ["length", "unknown", "error"])
def test_incomplete_parseable_answer_excluded_before_grader(reason):
    config = RLConfig(reference_rate=RateEstimationConfig(perturbation_indices=[0], n_rollouts=1),
                      training=TrainingSamplingConfig(perturbation_indices=[1], n_rollouts_for_rate=1))
    trainer = RLTrainer(config, backend=_UnusedBackend())
    trainer.renderer = _IndexRenderer()
    trainer.tokenizer = _AnswerTokenizer()
    class Sampler:
        async def sample(self, *args, **kwargs):
            return [SampledSequence([65], [-.1], finish_reason=reason)]
    trainer.sampling_client = Sampler()
    calls = []
    def grade(*args):
        calls.append(args)
        return 0.0
    result = asyncio.run(trainer._collect_rollouts({}, [
        lambda _: {"messages": [{"role": "user", "content": "0"}]},
        lambda _: {"messages": [{"role": "user", "content": "1"}]},
    ], grade, answer_parser=lambda _: "A"))
    assert not calls
    assert result.rate_counts == {0: 0, 1: 0}
    assert result.train_rollouts == []
    assert all(r.finish_reason == reason and r.raw_logprobs == [-.1] for r in result.sampled_rollouts)


@pytest.mark.parametrize("values", [[-.1], [float("nan"), -.2], [float("-inf"), -.2]])
def test_malformed_logprobs_fail(values):
    with pytest.raises(ValueError):
        SampledSequence([1, 2], values, finish_reason="stop").validate()


def test_vllm_preserves_length_status():
    c = SimpleNamespace(token_ids=[1], logprobs=[{1: SimpleNamespace(logprob=-.1)}], finish_reason="length")
    sequence = VLLMSampler._extract_sequences(SimpleNamespace(outputs=[c]))[0]
    assert sequence.finish_reason == "length"
