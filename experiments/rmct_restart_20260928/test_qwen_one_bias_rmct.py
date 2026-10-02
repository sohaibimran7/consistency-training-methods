"""One-bias Qwen RMCT recipe, slicing and progress; CPU only, no training."""
import json
from pathlib import Path

import pytest

from experiments.rmct_restart_20260928 import prepare, qwen_one_bias_rmct as rm
from experiments.rmct_restart_20260928.qwen_progress import next_encounter_slice, progress

HERE = Path(__file__).resolve().parent
KW = dict(repo='/canonical/repo', python='/canonical/venv/bin/python', python_prefix='/canonical/venv',
          commit='a' * 40, run_name='qwen-one-bias', data='/data/pool7680', manifest='/data/manifest7680',
          attestation='/new/preflight.json')


def plan():
    baseline = json.loads((HERE / 'fixtures/original_recipe.json').read_text())
    return prepare.build_one_bias(baseline, one_bias_manifest='/data/one-bias.json',
                                  one_bias_manifest_sha256='f' * 64, **KW)


def value(argv, flag):
    return argv[argv.index(flag) + 1]


def test_recipe_changes_only_data_exposure():
    two = prepare.build(json.loads((HERE / 'fixtures/original_recipe.json').read_text()), **KW)['argv']
    one = plan()['argv']
    changed = {two[i] for i in range(len(two)) if two[i] != one[i]}
    assert value(one, '--batch-size') == '4' and value(one, '--n-datapoints') == '16'
    assert json.loads(value(one, '--load-config')) == {'n_datapoints': 16, 'attempt_offset': 0}
    assert value(one, '--setting-factory') == prepare.ONE_BIAS_FACTORY
    config = json.loads(value(one, '--setting-config'))
    assert config['expected_qids_per_dataset'] == 3840 and config['one_bias_manifest_sha256'] == 'f' * 64
    for flag in ('--lr', '--kl-coef', '--n-train-rollouts', '--max-new-tokens', '--temperature', '--lora-config'):
        assert value(one, flag) == value(two, flag)
    assert len(one) == len(two) and changed  # same flags, only values differ


def test_slices_never_cross_validation_boundary_or_pool_end():
    state = progress(0, 0)
    assert next_encounter_slice(state, 64)['batch_count'] == 16
    state = progress(60, 50, no_progress_batches=3)
    assert next_encounter_slice(state, 64) == dict(segment_index=3, batch_offset=12, batch_count=4)
    assert next_encounter_slice(progress(1910, 1500), 1920)['batch_count'] == 10
    with pytest.raises(ValueError, match='boundary'):
        next_encounter_slice(progress(64, 60), 100)


def test_command_uses_absolute_encounter_cursor_and_strict_resume():
    p = plan()
    before = progress(80, 70)
    parent = dict(progress=before, checkpoint='/ckpt/parent')
    argv = rm.command(p, before, dict(segment_index=5, batch_offset=0, batch_count=16), parent)
    assert json.loads(value(argv, '--load-config')) == {'n_datapoints': 64, 'attempt_offset': 80}
    assert value(argv, '--run-name') == 'qwen-one-bias-b000080'
    assert value(argv, '--resume-from') == 'file:///ckpt/parent' and '--resume-state-required' in argv
    with pytest.raises(ValueError, match='replays or skips'):
        rm.command(p, before, dict(segment_index=5, batch_offset=1, batch_count=4), parent)
    with pytest.raises(ValueError, match='Missing parent'):
        rm.command(p, before, dict(segment_index=5, batch_offset=0, batch_count=4), None)


def test_normalized_progress_counts_no_update_batches_as_encounters():
    seal = dict(progress=progress(82, 64), checkpoint='/ckpt', files={})
    record = rm.normalized_progress(seal, model='m', source_commit='s', campaign_id='c')
    assert record['encounter_attempt'] == 82 and record['encountered_qid_bias_examples'] == 328
    assert record['actual_optimizer_step'] == 64 and record['trailing_skip_files'] == []


def test_target_cannot_cross_unvalidated_boundary(tmp_path):
    with pytest.raises(ValueError, match='unvalidated'):
        rm.train_to_encounters(plan(), {'source_root': str(tmp_path)}, tmp_path, 65, {}, None, {})
