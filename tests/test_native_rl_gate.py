from types import SimpleNamespace
import pytest
from experiments.rmct_restart_20260928.native_rl_gate import validate_rollout_records


def record(**changes):
    return SimpleNamespace(**(dict(skipped_from_training=False, completion_tokens=[1],
        sampled_logprobs=[-.1], finish_reason='stop', parsed_successfully=True,
        grader_failed=False, reward=1., advantage=.5) | changes))


def test_valid_evidence():
    validate_rollout_records([record()])


@pytest.mark.parametrize('changes', [
    dict(completion_tokens=[], sampled_logprobs=[]), dict(reward=None),
    dict(advantage=float('nan')), dict(grader_failed=True),
    dict(finish_reason='length'), dict(parsed_successfully=False),
    dict(sampled_logprobs=[]), dict(sampled_logprobs=[float('-inf')]),
])
def test_bad_evidence_rejected(changes):
    with pytest.raises(RuntimeError):
        validate_rollout_records([record(**changes)])


def test_excluded_length_may_lack_reward():
    validate_rollout_records([record(), record(skipped_from_training=True,
        finish_reason='length', parsed_successfully=False, reward=None, advantage=None)])
