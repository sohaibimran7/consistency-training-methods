from types import SimpleNamespace
import pytest
from experiments.rmct_restart_20260928.native_rl_gate import validate_rollout_records


def record(**changes):
    return SimpleNamespace(**(dict(skipped_from_training=False, completion_tokens=[1],
        sampled_logprobs=[-.1], finish_reason='stop', parsed_successfully=True,
        grader_failed=False, reward=1., advantage=.5) | changes))


def test_valid_evidence():
    validate_rollout_records([record()])


def test_real_adapter_update_required():
    import torch
    from experiments.rmct_restart_20260928.native_rl_gate import validate_updated_adapter
    a = torch.ones(2, 2)
    for tensors in ({}, {'lora_A.weight': a},
                    {'lora_A.weight': a, 'lora_B.weight': torch.zeros(2, 2)},
                    {'lora_A.weight': a, 'lora_B.weight': torch.full((2, 2), float('nan'))}):
        with pytest.raises(RuntimeError):
            validate_updated_adapter(tensors)
    assert validate_updated_adapter({'lora_A.weight': a, 'lora_B.weight': a * .001}) > 0


def test_real_frozen_setting_validated_before_gate_subset(tmp_path):
    from ctm_data.adapters.mcq_bias.tests.test_shared_qid_two_bias import _write_fixture_inputs
    from ctm_data.adapters.mcq_bias.shared_qid_two_bias import materialize_shared_qid_two_bias, SharedQidTwoBiasSetting
    from ctm.settings.runtime import prepare_setting_instance
    from experiments.rmct_restart_20260928.native_rl_worker import select_first_batch
    inputs = _write_fixture_inputs(tmp_path, qids_per_dataset=32)
    frozen = materialize_shared_qid_two_bias(*inputs[:4], tmp_path/'frozen', qids_per_dataset=32)
    setting = SharedQidTwoBiasSetting(data_path=frozen.data_path, manifest_path=frozen.manifest_path,
        expected_manifest_sha256=frozen.manifest_sha256, expected_qids_per_dataset=32)
    with pytest.raises(ValueError, match='n_datapoints=32'):
        prepare_setting_instance(setting, load_config={'n_datapoints': 2, 'segment_index': 0})
    prepared = prepare_setting_instance(setting, load_config={'n_datapoints': 32, 'segment_index': 0})
    subset = select_first_batch(prepared)
    assert len(prepared.datapoints) == 32
    assert subset.datapoints == prepared.datapoints[:2]
    assert subset.answer_parser is prepared.answer_parser


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


@pytest.mark.parametrize('family', ['shared', 'gemma'])
def test_one_bias_gate_trains_first_four_qids_for_both_families(family):
    from ctm_data.adapters.mcq_bias.shared_qid_one_bias import SharedQidOneBiasSetting
    from experiments.rmct_restart_20260928.gemma_production_setting import OneBiasGemmaSetting
    from experiments.rmct_restart_20260928.native_rl_worker import select_first_batch
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Prepared:
        setting: object
        datapoints: list

    cls = SharedQidOneBiasSetting if family == 'shared' else OneBiasGemmaSetting
    assert select_first_batch(Prepared(cls.__new__(cls), list(range(16)))).datapoints == [0, 1, 2, 3]
    with pytest.raises(ValueError, match='16-QID'):
        select_first_batch(Prepared(cls.__new__(cls), list(range(32))))
