"""Gemma fresh-campaign exposure protocol: one QID once, one assigned bias.

User-approved (2026-10-02): all six methods share the frozen 7,680-QID pool in
its frozen interleaved order; each update consumes four distinct QIDs, each with
exactly one hash-assigned cue (four biased examples). No QID repeats; skipped or
no-signal batches still consume their four encounters. Validation cadence is
deliberately left unset until the user decides it.
"""
from __future__ import annotations

from pathlib import Path

from ctm_data.adapters.mcq_bias import shared_qid_one_bias as protocol

ASSIGNMENT_SEED = 42
POOL_SCOPE = 'shared_frozen_7680_all_six_methods'
QIDS_PER_UPDATE = 4
BIASES_PER_QID = 1
VALIDATION_CADENCE = None  # pending explicit user decision; drivers refuse to cross a boundary without it


def manifest_for(pool, *, pool_sha256, manifest_sha256, order_sha256):
    """Deterministic manifest over the frozen ordered pool (identical for every method)."""
    if len(pool) != 7680:
        raise ValueError('Gemma one-bias campaign uses exactly the frozen 7,680-QID pool')
    return protocol.build_manifest(pool, seed=ASSIGNMENT_SEED, source={
        'pool_sha256': pool_sha256, 'pool_manifest_sha256': manifest_sha256, 'order_sha256': order_sha256})


def freeze(root, pool, **identity):
    manifest = manifest_for(pool, **identity)
    return protocol.freeze_manifest(manifest, Path(root) / 'one-bias'), manifest


def update_rows(manifest, by_id, attempt):
    """The four (datum, bias) inputs for a zero-based sampled-batch attempt."""
    assignments = protocol.encounter_slice(manifest, attempt, QIDS_PER_UPDATE)
    rows = [by_id[a['question_id']] for a in assignments]
    for row, assignment in zip(rows, assignments):
        if protocol.digest(protocol._validate_shared_datum(row, location=row['question_id'])) != assignment['datum_sha256']:
            raise ValueError('Pool datum differs from frozen one-bias assignment')
    return rows, [a['bias'] for a in assignments]


def max_attempts(manifest):
    return manifest['max_encounters'] // QIDS_PER_UPDATE


def contract_block(manifest):
    return {'protocol': protocol.PROTOCOL, 'protocol_version': protocol.PROTOCOL_VERSION,
            'assignment_seed': ASSIGNMENT_SEED, 'pool_scope': POOL_SCOPE,
            'manifest_sha256': protocol.manifest_identity(manifest),
            'assignment_sha256': manifest['assignment_sha256'], 'bias_counts': manifest['bias_counts'],
            'qids_per_update': QIDS_PER_UPDATE, 'biases_per_qid': BIASES_PER_QID,
            'paired_rows_per_update': QIDS_PER_UPDATE * BIASES_PER_QID,
            'max_encounters': manifest['max_encounters'], 'max_attempts': max_attempts(manifest),
            'primary_budget': protocol.PRIMARY_BUDGET, 'cycling_allowed': False,
            'skipped_batches_consume_encounters': True, 'validation_cadence': VALIDATION_CADENCE,
            'user_approval': '2026-10-02: shared 7,680 pool; 4 distinct QIDs x 1 sampled bias per update'}
