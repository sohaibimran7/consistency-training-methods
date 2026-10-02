"""One-bias Qwen validation boundaries and progress normalization; no GPU work."""
import json

import pytest

from experiments.rmct_restart_20260928 import qwen_one_bias_validation as ob
from experiments.rmct_restart_20260928 import qwen_validation as v
from experiments.rmct_restart_20260928.qwen_checkpoint import identity


def sealed(tmp_path, step=64):
    run_dir = tmp_path / 'opct'
    checkpoint = run_dir / 'checkpoints' / f'step-{step:06d}'
    checkpoint.mkdir(parents=True)
    exposure = {'one_bias_manifest_sha256': 'm', 'encounter_attempt': step, 'actual_optimizer_step': step,
                'encountered_qid_bias_examples': 4 * step, 'encountered_prefix_sha256': 'p',
                'encountered_bias_counts': {}, 'trailing_skip_files': [], 'validation_due': True}
    convergence = {'step': step, 'attempts': step, 'decision': 'continue', 'pending': []}
    for name in ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt'):
        (checkpoint / name).write_bytes(name.encode())
    (checkpoint / 'manifest.json').write_text(json.dumps({'loop_state': {'convergence': convergence, 'exposure': exposure}}))
    receipt = {'schema': 'ctm-grouped-qid-resume-v1', 'method': 'opct', 'convergence': convergence,
               'checkpoint': str(checkpoint.relative_to(run_dir)), 'exposure': exposure,
               'checkpoint_files': {n: identity(checkpoint / n)['sha256'] for n in ob.FIVE_METHOD_FILES}}
    path = run_dir / 'receipts' / f'step-{step:06d}.json'
    path.parent.mkdir()
    path.write_text(json.dumps(receipt))
    return run_dir, path, checkpoint


def test_progress_carries_encounters_and_exact_file_identities(tmp_path):
    run_dir, receipt, checkpoint = sealed(tmp_path)
    progress = ob.five_method_progress(run_dir, receipt, campaign_id='c', source_commit='s', model='m')
    assert progress['encountered_qid_bias_examples'] == 256 and progress['encounter_attempt'] == 64
    assert progress['actual_optimizer_step'] == progress['next_attempt_index'] == 64
    assert progress['checkpoint_files']['optimizer.pt'] == identity(checkpoint / 'optimizer.pt')
    assert ob.boundary(progress) == 256


def test_progress_rejects_changed_checkpoint_bytes(tmp_path):
    run_dir, receipt, checkpoint = sealed(tmp_path)
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='differ from receipt'):
        ob.five_method_progress(run_dir, receipt, campaign_id='c', source_commit='s', model='m')


@pytest.mark.parametrize('encounters,ok', [(256, True), (512, True), (7680, True), (128, False), (0, False), (300, False)])
def test_boundary_is_256_encounters_or_exhaustion(encounters, ok):
    progress = {'encountered_qid_bias_examples': encounters}
    if ok:
        assert ob.boundary(progress) == encounters
    else:
        with pytest.raises(ValueError, match='256-encounter'):
            ob.boundary(progress)


def test_executor_accepts_one_bias_schema_only_on_grid(monkeypatch, tmp_path):
    from experiments.rmct_restart_20260928 import qwen_validation_executor as ex
    base = {'schema': ob.SCHEMA, 'settings': v.SETTINGS, 'campaign_id': 'c', 'step': 31,
            'encountered_qid_bias_examples': 124, 'validation_sha256': v.PROMPT_SHA}
    with pytest.raises(ValueError, match='encounter boundary'):
        ex.check_contract(base, tmp_path)
    base['encountered_qid_bias_examples'] = 256
    # Past the grid check the contract continues to the deployed-source gate.
    monkeypatch.setattr(ex.subprocess, 'check_output', lambda *a, **k: 'other')
    with pytest.raises(ValueError, match='deployed source'):
        ex.check_contract(dict(base, source_commit='s'), tmp_path)


def test_first_window_is_fresh_and_bounded_without_validation(tmp_path):
    assert ob.gated_budget(run_dir=tmp_path, step=0, requested=128, contract_path=None, folder=None,
                           manifest=None, model_path=None) == 64


def test_selection_contract_is_the_shared_v2_encounter_contract(tmp_path):
    from experiments.rmct_restart_20260928 import validation_selection as s
    population = tmp_path / 'validation.json'
    population.write_text('{}')
    contract = ob.selection_contract(campaign_id='c', method='bct', model='m', source_commit='x', population=population)
    assert s.encounter_mode(contract) and contract['policy'] == s.ENCOUNTER_POLICY
    s.check_contract(contract)
