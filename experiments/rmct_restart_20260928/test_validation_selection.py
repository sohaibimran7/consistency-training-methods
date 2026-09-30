"""Selection, replay and stopping tests using immutable saved artifacts."""
import json
import pytest

from experiments.rmct_restart_20260928 import validation_selection as s


@pytest.fixture
def campaign(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return s.identity(path)
    contract = dict(schema='ctm-tbsr-selection-contract-v1', policy=s.POLICY,
                    campaign_id='fresh', method='bct', model='gemma', source_commit='source',
                    response_count=600, pair_count=400, settings=dict(enable_thinking=True),
                    population=save('population.json', dict(rows=list(range(600)))))
    contract_record = save('contract.json', contract)
    def checkpoint(step, attempts=None):
        attempts = step + 6 if attempts is None else attempts
        files = {'manifest.json': save(f'checkpoint-{step}/manifest.json',
                                      dict(optimizer_step=step, next_attempt_index=attempts))}
        progress = dict(schema='ctm-training-progress-v1', campaign_id='fresh', method='bct',
                        model='gemma', source_commit='source', actual_optimizer_step=step,
                        next_attempt_index=attempts, sampled_batches=None,
                        checkpoint=str(tmp_path / f'checkpoint-{step}'), checkpoint_files=files)
        return save(f'progress-{step}.json', progress), progress
    def validation(step, n, d=100, attempts=None):
        p, progress = checkpoint(step, attempts)
        metrics = dict(step=step, campaign_id='fresh', response_count=600,
                       checkpoint_files=progress['checkpoint_files'], towards_switches=n,
                       eligible_pairs=d, tbsr=n/d)
        return p, save(f'validation-{step}.json', metrics), progress
    def verify_checkpoint(progress, contract):
        manifest = s.read_verified(progress['checkpoint_files']['manifest.json'])
        if (manifest['optimizer_step'] != progress['actual_optimizer_step']
                or manifest['next_attempt_index'] != progress['next_attempt_index']):
            raise ValueError('Saved optimizer/cursor mismatch')
        return progress
    def verify_validation(record, progress, contract):
        return s.read_verified(record)
    return dict(root=tmp_path/'selection', save=save, contract=contract_record,
                checkpoint=checkpoint, validation=validation,
                verify_checkpoint=verify_checkpoint, verify_validation=verify_validation)


def accept(c, step, n, d=100, attempts=None):
    p, v, progress = c['validation'](step, n, d, attempts)
    result = s.accept_validation(c['root'], c['contract'], p, v,
                                verify_checkpoint=c['verify_checkpoint'],
                                verify_validation=c['verify_validation'])
    return result, p, v, progress


def test_ties_select_earliest_and_two_nonimprovements_stop(campaign):
    c = campaign
    state, _, _, _ = accept(c, 64, 10)
    assert not state['stopped']
    state, _, _, _ = accept(c, 128, 20, 200)
    assert state['history'][-1]['best_step'] == 64
    state, _, _, _ = accept(c, 192, 12)
    assert state['stopped'] and state['stop_reason'] == 'patience'
    selected = s.selected_evaluation_manifest(state)
    assert selected['actual_optimizer_step'] == 64
    assert state['latest']['metrics']['step'] == 192
    assert selected['selection_status'] == 'terminal'
    with pytest.raises(ValueError, match='stopping event'):
        accept(c, 256, 1)


def test_improvement_resets_patience_with_exact_rational_comparison(campaign):
    c = campaign
    accept(c, 64, 1, 3)
    accept(c, 128, 2, 6)
    state, _, _, _ = accept(c, 192, 1, 4)
    assert state['history'][-1] == dict(step=192, best_step=192, nonimproving_checks=0)
    assert not state['stopped']


def test_identical_ingestion_is_idempotent_and_conflict_rejected(campaign):
    c = campaign
    state, p, v, progress = accept(c, 64, 10)
    repeated = s.accept_validation(c['root'], c['contract'], p, v,
                                  verify_checkpoint=c['verify_checkpoint'],
                                  verify_validation=c['verify_validation'])
    assert repeated == state and len(repeated['history']) == 1
    other = c['save']('other-validation.json', {**s.read_verified(v), 'towards_switches': 11, 'tbsr': .11})
    with pytest.raises(ValueError, match='Conflicting'):
        s.accept_validation(c['root'], c['contract'], p, other,
                            verify_checkpoint=c['verify_checkpoint'], verify_validation=c['verify_validation'])


@pytest.mark.parametrize('step', [12, 16, 65, 128])
def test_missing_or_false_boundary_does_not_create_receipt(campaign, step):
    c = campaign
    with pytest.raises(ValueError):
        accept(c, step, 10)
    assert not list(c['root'].glob('validation/*.json'))


def test_missing_validation_blocks_next_window_and_skips_do_not_consume_update_budget(campaign):
    c = campaign
    empty = s.replay([], c['contract'], verify_checkpoint=c['verify_checkpoint'],
                     verify_validation=c['verify_validation'])
    _, p = c['checkpoint'](12, attempts=16)
    assert s.continuation_budget(p, empty, requested_updates=64) == 52
    _, boundary = c['checkpoint'](64, attempts=70)
    with pytest.raises(ValueError, match='requires validation'):
        s.continuation_budget(boundary, empty, requested_updates=64)
    state, _, _, progress = accept(c, 64, 10, attempts=70)
    assert s.continuation_budget(progress, state, requested_updates=100) == 64


def test_saved_evidence_is_rechecked_on_every_replay(campaign):
    c = campaign
    state, p, v, progress = accept(c, 64, 10)
    entry = state['latest']
    from pathlib import Path
    Path(v['path']).write_text('{}')
    with pytest.raises(ValueError, match='identity changed'):
        s.replay([entry], c['contract'], verify_checkpoint=c['verify_checkpoint'],
                 verify_validation=c['verify_validation'])


def test_native_validator_failure_cannot_be_replaced_with_success_flag(campaign):
    c = campaign
    p, v, _ = c['validation'](64, 10)
    def fail(*args): raise ValueError('Wrong native template')
    with pytest.raises(ValueError, match='Wrong native template'):
        s.accept_validation(c['root'], c['contract'], p, v,
                            verify_checkpoint=c['verify_checkpoint'], verify_validation=fail)
    assert not list(c['root'].glob('validation/*.json'))


def test_backward_consumed_cursor_and_wrong_checkpoint_validation_rejected(campaign):
    c = campaign
    accept(c, 64, 10, attempts=200)
    with pytest.raises(ValueError, match='cursor'):
        accept(c, 128, 9, attempts=150)
    c = campaign
    p, v, progress = c['validation'](128, 9, attempts=201)
    metric = s.read_verified(v)
    metric['checkpoint_files'] = {}
    wrong = c['save']('wrong-checkpoint.json', metric)
    with pytest.raises(ValueError, match='identity differs'):
        s.accept_validation(c['root'], c['contract'], p, wrong,
                            verify_checkpoint=c['verify_checkpoint'], verify_validation=c['verify_validation'])
