import pytest

from ctm_data.adapters.mcq_bias import shared_qid_one_bias as one
from experiments.rmct_restart_20260928.gemma_production_setting import OneBiasGemmaSetting
from tests.test_gemma_methods_plan import _manifest


def setting(tmp_path):
    manifest = _manifest()
    path = one.freeze_manifest(manifest, tmp_path)
    s = OneBiasGemmaSetting.__new__(OneBiasGemmaSetting)
    s.one_bias_manifest_path = path
    s.one_bias_manifest = one.load_manifest(path, expected_sha256=one.manifest_identity(manifest))
    s.one_bias_manifest_sha256 = one.manifest_identity(manifest)
    s._loaded_segment = None
    from tests.test_gemma_methods_plan import plan as reference_plan
    from ctm_data.adapters.mcq_bias.shared_qid_two_bias import DATUM_SCHEMA, SCHEMA_VERSION
    def msg(text):
        return [{'role': 'user', 'content': text}]
    rows = {f'q{i}': {'datum_schema': DATUM_SCHEMA, 'schema_version': SCHEMA_VERSION, 'question_id': f'q{i}',
            'source_dataset': ('logiqa', 'hellaswag')[i % 2], 'question': 'q', 'ground_truth': 'A',
            'prompt_style': 'none', 'clean_messages': msg('clean'),
            'biased_options': {b: 'B' for b in reference_plan.BIASES},
            'variants': {b: {'messages': msg(b), 'biased_option': 'B', 'biasing_text': b} for b in reference_plan.BIASES},
            'provenance': {'wrong_argument_source_line_number': i + 1}} for i in range(8)}
    s._load_verified = lambda: ({}, rows)
    s._matches_bias_fn = lambda answer, target: float(answer == target)
    return s, manifest


def test_one_arm_slices_follow_shared_manifest_without_cycling(tmp_path):
    s, manifest = setting(tmp_path)
    first = s.load_datapoints(4, attempt_offset=0)
    second = s.load_datapoints(4, attempt_offset=1)
    got = [(d['question_id'], d['bias']) for d in first + second]
    assert got == [(a['question_id'], a['bias']) for a in manifest['assignments']]
    assert all('variants' not in d for d in first)
    assert len(s.perturbations()) == 2 and s.training_perturbation_indices() == [1]
    with pytest.raises(ValueError, match='exhausted'):
        s.load_datapoints(4, attempt_offset=2)
    with pytest.raises(ValueError, match='four-QID'):
        s.load_datapoints(6, attempt_offset=0)


def test_trait_uses_assigned_target_for_clean_and_cue(tmp_path):
    s, _ = setting(tmp_path)
    s.answer_parser = lambda: (lambda response: response)
    d = s.load_datapoints(4, attempt_offset=0)[0]
    classify = s.trait_classifier()
    assert classify('B', d, d['clean_messages']) == 1.0
    assert classify('A', d, d['variant']['messages']) == 0.0
    with pytest.raises(ValueError, match='neither'):
        classify('B', d, [{'role': 'user', 'content': 'other'}])
