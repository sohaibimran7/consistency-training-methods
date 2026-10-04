"""Encounter-cadence (256 QIDs incl. no-update batches) selection tests."""
import json

import pytest

from experiments.rmct_restart_20260928 import validation_selection as s


@pytest.fixture
def c(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return s.identity(path)
    contract = dict(schema='ctm-tbsr-selection-contract-v2-encounters', policy=s.ENCOUNTER_POLICY,
                    campaign_id='fresh', method='bct', model='gemma', source_commit='source',
                    response_count=600, pair_count=400, settings=dict(enable_thinking=True),
                    population=save('population.json', dict(rows=list(range(600)))))
    record = save('contract.json', contract)

    def progress(step, attempts, cursor):
        files = {'manifest.json': save(f'ck-{step}-{cursor}/manifest.json', dict(step=step))}
        skips = [save(f'skips/attempt-{a:07d}.json', dict(attempt=a)) for a in range(attempts, cursor)]
        value = dict(schema='ctm-training-progress-v1', campaign_id='fresh', method='bct', model='gemma',
                     source_commit='source', actual_optimizer_step=step, next_attempt_index=attempts,
                     encounter_attempt=cursor, encountered_qid_bias_examples=4 * cursor,
                     trailing_skip_files=skips, sampled_batches=None,
                     checkpoint=str(tmp_path / f'ck-{step}-{cursor}'), checkpoint_files=files)
        return save(f'progress-{step}-{cursor}.json', value), value

    def accept(step, attempts, cursor, n, d=100):
        p, value = progress(step, attempts, cursor)
        v = save(f'validation-{cursor}.json', dict(step=step, campaign_id='fresh', response_count=600,
                 checkpoint_files=value['checkpoint_files'], towards_switches=n, eligible_pairs=d, tbsr=n / d))
        return s.accept_validation(tmp_path / 'sel', record, p, v, verify_checkpoint=lambda pr, ct: pr,
                                   verify_validation=lambda r, pr, ct: s.read_verified(r))
    return dict(record=record, progress=progress, accept=accept, root=tmp_path / 'sel')


def test_boundaries_count_no_update_batches_and_budget_in_attempts(c):
    # 64 attempts = 256 encounters; last 4 attempts were skipped (no update).
    state = c['accept'](55, 60, 64, 10)
    assert state['latest'] and not state['stopped']
    assert (c['root'] / 'validation' / 'encounter-000256.json').is_file()
    # Best accepted at 64 -> patience horizon 64 + 2*64 = 192 (train ahead of
    # the unvalidated 128 boundary, never past what patience could require).
    assert s.encounter_horizon(state) == 192
    _, progress = c['progress'](70, 100, 100)
    assert s.continuation_budget(progress, state, requested_updates=999) == 92
    _, at_boundary = c['progress'](80, 128, 128)
    assert s.continuation_budget(at_boundary, state, requested_updates=10) == 10
    _, at_horizon = c['progress'](150, 192, 192)
    assert s.continuation_budget(at_horizon, state, requested_updates=10) == 0
    _, beyond = c['progress'](160, 200, 200)
    with pytest.raises(ValueError, match='outside'):
        s.continuation_budget(beyond, state, requested_updates=10)


def test_horizon_tracks_best_not_latest(c):
    c['accept'](60, 64, 64, 10)
    state = c['accept'](120, 128, 128, 30)  # worse: best stays at 64
    assert s.encounter_horizon(state) == 192
    state = c['accept'](180, 192, 192, 5)   # better: horizon moves to 192 + 128
    assert s.encounter_horizon(state) == 320


def test_non_boundary_and_missing_skip_evidence_rejected(c):
    with pytest.raises(ValueError, match='256-encounter'):
        c['accept'](40, 50, 50, 10)
    p, value = c['progress'](55, 60, 64)
    value['trailing_skip_files'] = value['trailing_skip_files'][:-1]
    with pytest.raises(ValueError, match='skip evidence'):
        s.check_progress(value, s.read_verified(c['record']))


def test_patience_and_pool_exhaustion_stop_reasons(c):
    c['accept'](60, 64, 64, 10)
    c['accept'](120, 128, 128, 20)
    state = c['accept'](180, 192, 192, 30)
    assert state['stopped'] and state['stop_reason'] == 'patience'
    assert state['selected']['metrics']['step'] == 60


def test_v1_policy_unchanged():
    assert s.POLICY == dict(metric='tbsr', interval=64, patience=2, min_delta=0, tie='earliest',
                            max_optimizer_updates=4096)
    assert s.interval_attempts() == 64 and s.max_attempts() == 1920


def test_data_matched_budget_extends_only_to_fixed_target(c, monkeypatch):
    c['accept'](60, 64, 64, 10)
    c['accept'](120, 128, 128, 20)
    state = c['accept'](180, 192, 192, 30)        # patience stop at 192 (best 64)
    assert state['stopped']
    _, at_stop = c['progress'](180, 192, 192)
    assert s.continuation_budget(at_stop, state, requested_updates=16) == 0
    assert s.data_matched_budget(at_stop, state, requested_updates=16, target_attempts=256) == 16
    _, near = c['progress'](230, 250, 250)
    assert s.data_matched_budget(near, state, requested_updates=16, target_attempts=256) == 6
    _, done = c['progress'](236, 256, 256)
    assert s.data_matched_budget(done, state, requested_updates=16, target_attempts=256) == 0
    monkeypatch.setenv(s.DATA_MATCHED_ENV, '256')
    assert s.data_matched_target() == 256
    monkeypatch.setenv(s.DATA_MATCHED_ENV, '100')
    with pytest.raises(ValueError, match='256-encounter'):
        s.data_matched_target()


def test_data_matched_budget_keeps_patience_budget_when_larger(c):
    state = c['accept'](60, 64, 64, 10)            # horizon 192, not stopped
    _, progress = c['progress'](70, 100, 100)
    assert s.data_matched_budget(progress, state, requested_updates=999, target_attempts=128) == 92
