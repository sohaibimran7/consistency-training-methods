"""Separate trainer progress from selection; no loss-based convergence."""
import hashlib
import json
import math

from experiments.gemma4_methods.one_bias import QIDS_PER_UPDATE


def bounded_end(step, budget):
    if type(step) is not int or step < 0 or type(budget) is not int or budget < 1:
        raise ValueError('Valid actual-update step and positive budget required')
    if step and step % 64 == 0:
        raise RuntimeError('Validation boundary requires shared-controller acceptance; adapter not integrated')
    return min(step + budget, (step // 64 + 1) * 64)


def record_update(state, *, attempt, loss, question_ids):
    if not math.isfinite(loss) or attempt < state['step'] or len(set(question_ids)) != QIDS_PER_UPDATE or len(question_ids) != QIDS_PER_UPDATE:
        raise ValueError('Invalid actual optimizer update')
    step = state['step'] + 1
    pending = [*state.get('pending', []), loss]
    return {**state, 'step': step, 'attempts': attempt+1, 'decision': 'continue',
            'pending': [] if step % 16 == 0 else pending,
            'last_update_question_ids': list(question_ids),
            'last_update_question_ids_sha256': hashlib.sha256(json.dumps(question_ids).encode()).hexdigest()}
