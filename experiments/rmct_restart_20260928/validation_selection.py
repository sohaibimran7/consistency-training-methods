"""Model-independent TBSR selection over externally verified artifacts.

Model-specific validators own native templates, completion parsing, scheduler
checks and optimizer/RNG restoration. This module requires their verifier on
every ingestion/replay; stored success flags do not grant continuation.
"""
import json
import os
from pathlib import Path

from experiments.rmct_restart_20260928.qwen_checkpoint import identity

POLICY = dict(metric='tbsr', interval=64, patience=2, min_delta=0,
              tie='earliest', max_optimizer_updates=4096)
# Encounter cadence (user-approved 2026-10-02 for the Gemma one-bias campaign):
# validate every 256 encountered QID/bias examples, counting sampled batches
# that produced no optimizer update. Batches are 4 QIDs, so 64 attempts.
ENCOUNTER_POLICY = dict(metric='tbsr', interval_encounters=256, qids_per_attempt=4, patience=2,
                        min_delta=0, tie='earliest', max_encounters=7680)
SCHEMAS = {'ctm-tbsr-selection-contract-v1': POLICY,
           'ctm-tbsr-selection-contract-v2-encounters': ENCOUNTER_POLICY}


def encounter_mode(contract):
    return contract.get('schema') == 'ctm-tbsr-selection-contract-v2-encounters'


def interval_attempts():
    return ENCOUNTER_POLICY['interval_encounters'] // ENCOUNTER_POLICY['qids_per_attempt']


def max_attempts():
    return ENCOUNTER_POLICY['max_encounters'] // ENCOUNTER_POLICY['qids_per_attempt']


def position(progress, contract):
    """Selection coordinate: optimizer step (v1) or consumed attempts incl. trailing skips (v2)."""
    return progress['encounter_attempt'] if encounter_mode(contract) else progress['actual_optimizer_step']


def boundary_name(progress, contract):
    if encounter_mode(contract):
        return f"encounter-{ENCOUNTER_POLICY['qids_per_attempt'] * progress['encounter_attempt']:06d}.json"
    return f"step-{progress['actual_optimizer_step']:06d}.json"


def read_verified(record):
    if identity(record['path']) != record:
        raise ValueError('Artifact identity changed')
    return json.loads(Path(record['path']).read_text())


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def check_contract(contract):
    if contract.get('schema') not in SCHEMAS or contract.get('policy') != SCHEMAS[contract['schema']]:
        raise ValueError('Unknown selection policy')
    for field in ('campaign_id', 'method', 'model', 'source_commit'):
        if not isinstance(contract.get(field), str) or not contract[field]:
            raise ValueError(f'Missing {field}')
    if contract['method'] not in ('bct', 'opct', 'rmct', 'act', 'attct', 'mlpct'):
        raise ValueError('Unknown method')
    if type(contract.get('response_count')) is not int or contract['response_count'] != 600:
        raise ValueError('Expected 600 validation prompts')
    if type(contract.get('pair_count')) is not int or contract['pair_count'] != 400:
        raise ValueError('Expected 400 possible biased pairs')
    if contract.get('settings', {}).get('enable_thinking') is not True:
        raise ValueError('Thinking-enabled validation required')
    read_verified(contract['population'])


def check_progress(progress, contract):
    if progress.get('schema') != 'ctm-training-progress-v1':
        raise ValueError('Verified training progress required')
    for key in ('campaign_id', 'method', 'model', 'source_commit'):
        if progress.get(key) != contract[key]:
            raise ValueError('Training/selection identity differs')
    step, attempts = progress['actual_optimizer_step'], progress['next_attempt_index']
    if type(step) is not int or type(attempts) is not int or not 0 <= step <= attempts:
        raise ValueError('Invalid separate progress counters')
    if step > POLICY['max_optimizer_updates']:
        raise ValueError('Optimizer maximum exceeded')
    if encounter_mode(contract):
        # Weights are those of the last update; every later attempt up to the
        # encounter cursor must be an evidenced skipped/no-signal batch.
        cursor = progress.get('encounter_attempt')
        if type(cursor) is not int or not attempts <= cursor <= max_attempts():
            raise ValueError('Invalid encounter cursor')
        if progress.get('encountered_qid_bias_examples') != ENCOUNTER_POLICY['qids_per_attempt'] * cursor:
            raise ValueError('Encounter count disagrees with cursor')
        if len(progress.get('trailing_skip_files', [])) != cursor - attempts:
            raise ValueError('Trailing consumed attempts lack skip evidence')
        for record in progress['trailing_skip_files']:
            if identity(record['path']) != record:
                raise ValueError('Skip evidence changed')
    sampled = progress.get('sampled_batches')
    if sampled is not None and (type(sampled) is not int or not 0 <= sampled <= attempts):
        raise ValueError('Invalid sampled batch count')
    files = progress['checkpoint_files']
    if not files or not isinstance(files, dict):
        raise ValueError('Missing checkpoint file identities')
    for record in files.values():
        if identity(record['path']) != record:
            raise ValueError('Checkpoint bytes changed')
        if not Path(record['path']).resolve().is_relative_to(Path(progress['checkpoint']).resolve()):
            raise ValueError('Checkpoint file outside checkpoint directory')
    return progress


def _verified_entry(entry, contract_record, verify_checkpoint, verify_validation):
    """Callbacks must verify full model-specific lineage and saved outputs."""
    contract = read_verified(contract_record)
    check_contract(contract)
    if entry.get('schema') != 'ctm-tbsr-selection-entry-v1' or entry['contract'] != contract_record:
        raise ValueError('Selection entry belongs to another contract')
    progress = check_progress(read_verified(entry['progress']), contract)
    if verify_checkpoint(progress, contract) != progress:
        raise ValueError('Checkpoint verifier must return the verified progress')
    # The validator reproduces scores from saved requests/responses, verifies
    # actual model/template/settings and scheduler completion, and binds the
    # exact checkpoint file map. None of these are inferred from metric flags.
    read_verified(entry['validation'])
    metrics = verify_validation(entry['validation'], progress, contract)
    step = progress['actual_optimizer_step']
    if encounter_mode(contract):
        cursor = progress['encounter_attempt']
        if cursor <= 0 or (cursor % interval_attempts() and cursor != max_attempts()):
            raise ValueError('Validation requires a 256-encounter boundary')
    elif type(step) is not int or step < 64 or step % 64:
        raise ValueError('Validation requires actual 64-update boundary')
    for key, expected in [('step', step), ('campaign_id', contract['campaign_id']),
                          ('response_count', 600), ('checkpoint_files', progress['checkpoint_files'])]:
        if metrics.get(key) != expected:
            raise ValueError('Validation checkpoint/coverage identity differs')
    n, d = metrics['towards_switches'], metrics['eligible_pairs']
    if type(n) is not int or type(d) is not int or not 0 <= n <= d <= 400 or d == 0:
        raise ValueError('Invalid TBSR counts')
    if metrics.get('tbsr') != n / d:
        raise ValueError('TBSR/count disagreement')
    if metrics != entry['metrics']:
        raise ValueError('Stored validation metrics changed')
    return progress, metrics


def replay(entries, contract_record, *, verify_checkpoint, verify_validation):
    contract = read_verified(contract_record)
    check_contract(contract)
    best = latest = None
    stale = 0
    history = []
    previous_attempts = -1
    for index, entry in enumerate(entries):
        if stale >= POLICY['patience']:
            raise ValueError('Validation after the first stopping event')
        progress, metrics = _verified_entry(entry, contract_record, verify_checkpoint, verify_validation)
        step = progress['actual_optimizer_step']
        if encounter_mode(contract):
            cursor = progress['encounter_attempt']
            expected = min((index + 1) * interval_attempts(), max_attempts())
            if cursor != expected or cursor <= previous_attempts:
                raise ValueError('Nonchronological validation or consumed-data cursor')
            previous_attempts = cursor
        else:
            if step != (index + 1) * 64 or progress['next_attempt_index'] <= previous_attempts:
                raise ValueError('Nonchronological validation or consumed-data cursor')
            previous_attempts = progress['next_attempt_index']
        latest = entry
        if (best is None or metrics['towards_switches'] * best['metrics']['eligible_pairs']
                < best['metrics']['towards_switches'] * metrics['eligible_pairs']):
            best, stale = entry, 0
        else:
            stale += 1
        history.append(dict(step=step, best_step=best['metrics']['step'], nonimproving_checks=stale,
                            **({'encounter_attempt': progress['encounter_attempt']} if encounter_mode(contract) else {})))
    if encounter_mode(contract):
        reached_maximum = bool(latest and read_verified(latest['progress'])['encounter_attempt'] == max_attempts())
    else:
        reached_maximum = bool(latest and latest['metrics']['step'] == POLICY['max_optimizer_updates'])
    return dict(schema='ctm-tbsr-selection-state-v1', contract=contract_record,
                latest=latest, selected=best, history=history,
                stopped=stale >= POLICY['patience'] or reached_maximum,
                stop_reason='patience' if stale >= POLICY['patience'] else
                ('pool_exhausted' if encounter_mode(contract) else 'maximum') if reached_maximum else None)


def entries_from_folder(folder, contract=None):
    encounters = contract is not None and encounter_mode(contract)
    paths = sorted((Path(folder) / 'validation').glob('encounter-*.json' if encounters else 'step-*.json'))
    entries = []
    for index, path in enumerate(paths):
        expected = (f"encounter-{min(256 * (index + 1), ENCOUNTER_POLICY['max_encounters']):06d}.json"
                    if encounters else f'step-{64 * (index + 1):06d}.json')
        if path.name != expected:
            raise ValueError('Missing or unexpected selection receipt')
        entries.append(read_verified(identity(path)))
    return entries


def accept_validation(folder, contract_record, progress_record, validation_record, *,
                      verify_checkpoint, verify_validation):
    contract = read_verified(contract_record)
    check_contract(contract)
    progress = check_progress(read_verified(progress_record), contract)
    if verify_checkpoint(progress, contract) != progress:
        raise ValueError('Checkpoint verifier must return verified progress')
    read_verified(validation_record)
    metrics = verify_validation(validation_record, progress, contract)
    entry = dict(schema='ctm-tbsr-selection-entry-v1', contract=contract_record,
                 progress=progress_record, validation=validation_record, metrics=metrics)
    entries = entries_from_folder(folder, contract)
    path = Path(folder) / 'validation' / boundary_name(progress, contract)
    if path.exists():
        if json.loads(path.read_text()) != entry:
            raise ValueError('Conflicting receipt for an already accepted checkpoint')
        return replay(entries, contract_record, verify_checkpoint=verify_checkpoint, verify_validation=verify_validation)
    result = replay([*entries, entry], contract_record,
                    verify_checkpoint=verify_checkpoint, verify_validation=verify_validation)
    write_exclusive(path, entry)
    return result


def continuation_budget(progress, state, *, requested_updates):
    """Pure budget; call only with freshly verified progress/replayed state."""
    if type(requested_updates) is not int or requested_updates <= 0:
        raise ValueError('Positive actual-update budget required')
    contract=read_verified(state['contract'])
    check_contract(contract)
    check_progress(progress, contract)
    if encounter_mode(contract):
        # Budget unit is sampled-batch attempts (4 encounters each), not updates.
        cursor = progress['encounter_attempt']
        if state['stopped'] or cursor >= max_attempts():
            return 0
        accepted = read_verified(state['latest']['progress'])['encounter_attempt'] if state['latest'] else 0
        if cursor < accepted or cursor > accepted + interval_attempts():
            raise ValueError('Progress outside the currently authorized window')
        if cursor == accepted + interval_attempts():
            raise ValueError('Current boundary still requires validation')
        return min(requested_updates, accepted + interval_attempts() - cursor, max_attempts() - cursor)
    step = progress['actual_optimizer_step']
    if type(step) is not int or not 0 <= step <= POLICY['max_optimizer_updates']:
        raise ValueError('Invalid actual optimizer progress')
    if state['stopped'] or step == POLICY['max_optimizer_updates']:
        return 0
    accepted = state['latest']['metrics']['step'] if state['latest'] else 0
    if step < accepted or step > accepted + 64:
        raise ValueError('Progress outside the currently authorized window')
    if step == accepted + 64:
        raise ValueError('Current boundary still requires validation')
    if accepted and progress['next_attempt_index'] < read_verified(state['latest']['progress'])['next_attempt_index']:
        raise ValueError('Consumed-data cursor moved backwards')
    return min(requested_updates, accepted + 64 - step, POLICY['max_optimizer_updates'] - step)


def bootstrap_budget(contract_record, start_record, *, requested_updates, verify_start):
    """Budget the fresh first window before a training checkpoint exists."""
    contract = read_verified(contract_record)
    check_contract(contract)
    start = read_verified(start_record)
    if start.get('schema') != 'ctm-training-start-v1' or start.get('contract') != contract_record:
        raise ValueError('Fresh start contract mismatch')
    for key in ('campaign_id','method','model','source_commit'):
        if start.get(key) != contract[key]:
            raise ValueError('Fresh start identity mismatch')
    for key in ('actual_optimizer_step','next_attempt_index'):
        if type(start.get(key)) is not int or start[key] != 0:
            raise ValueError('Bootstrap requires fresh zero progress')
    if start.get('resume_from') is not None or start.get('optimizer') != 'fresh':
        raise ValueError('Bootstrap requires base weights and fresh optimizer')
    # Adapter verifies actual source, original base/data/native gate evidence
    # and exclusive start claims before returning this same immutable record.
    if verify_start(start,contract) != start:
        raise ValueError('Fresh start evidence not verified')
    if type(requested_updates) is not int or requested_updates <= 0:
        raise ValueError('Positive actual-update budget required')
    return min(requested_updates, interval_attempts() if encounter_mode(contract) else 64)


def selected_evaluation_manifest(state):
    """Materialize the selected identity from a freshly replayed state."""
    selected = state['selected']
    if selected is None:
        raise ValueError('No verified selected checkpoint')
    contract=read_verified(state['contract'])
    check_contract(contract)
    progress = check_progress(read_verified(selected['progress']),contract)
    return dict(schema='ctm-selected-checkpoint-evaluation-v1', contract=state['contract'],
                progress=selected['progress'], validation=selected['validation'],
                actual_optimizer_step=progress['actual_optimizer_step'],
                checkpoint=progress['checkpoint'], checkpoint_files=progress['checkpoint_files'],
                selection_status='terminal' if state['stopped'] else 'provisional')
