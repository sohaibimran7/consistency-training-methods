"""Separate consumed batches from real optimizer updates (batch size two).

No training or migration authority is granted by these arithmetic helpers.
"""
SEGMENT_BATCHES = 16
POOL_BATCHES = 500


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def progress(sampled_batches, optimizer_updates, *, no_progress_batches=0):
    integer(sampled_batches, 'sampled_batches')
    integer(optimizer_updates, 'optimizer_updates')
    integer(no_progress_batches, 'no_progress_batches')
    if optimizer_updates > sampled_batches or no_progress_batches > sampled_batches:
        raise ValueError('Impossible progress counters')
    return dict(sampled_batches=sampled_batches, optimizer_updates=optimizer_updates,
                no_progress_batches=no_progress_batches,
                segment_index=sampled_batches // SEGMENT_BATCHES,
                batch_offset=sampled_batches % SEGMENT_BATCHES)


def next_slice(state, target):
    current = progress(state['sampled_batches'], state['optimizer_updates'],
                       no_progress_batches=state['no_progress_batches'])
    if current != state:
        raise ValueError('Cursor inconsistent with consumed batches')
    integer(target, 'target')
    if target < 64 or target % 64 or state['optimizer_updates'] > target:
        raise ValueError('Invalid optimizer validation boundary')
    if state['optimizer_updates'] == target:
        return None
    if state['no_progress_batches'] >= POOL_BATCHES:
        raise ValueError('No optimizer progress across one full pool; not convergence')
    count = min(SEGMENT_BATCHES - state['batch_offset'],
                target - state['optimizer_updates'],
                POOL_BATCHES - state['no_progress_batches'])
    return dict(segment_index=state['segment_index'], batch_offset=state['batch_offset'], batch_count=count)


def advance(state, selection, optimizer_updates):
    integer(optimizer_updates, 'optimizer_updates')
    if (selection['segment_index'], selection['batch_offset']) != (state['segment_index'], state['batch_offset']):
        raise ValueError('Selection does not start at saved cursor')
    count = selection['batch_count']
    if type(count) is not int or not 1 <= count <= SEGMENT_BATCHES - state['batch_offset']:
        raise ValueError('Invalid slice size')
    delta = optimizer_updates - state['optimizer_updates']
    if not 0 <= delta <= count:
        raise ValueError('Optimizer update delta cannot exceed consumed batches')
    # Conservative child-level no-progress guard. Any verified optimizer
    # progress resets it; all-zero children consume its finite budget.
    stale = 0 if delta else state['no_progress_batches'] + count
    return progress(state['sampled_batches'] + count, optimizer_updates, no_progress_batches=stale)


def validate_loop(loop, state):
    if any(loop.get(k) != state['sampled_batches'] for k in ('step', 'global_step')):
        raise ValueError('Consumed batch counter mismatch')
    if loop.get('optimizer_step') != state['optimizer_updates']:
        raise ValueError('Optimizer counter mismatch')
    if loop.get('final') is not True or loop.get('accumulated_grads') != 0:
        raise ValueError('Not a completed child at optimizer boundary')


ONE_BIAS_MAX_BATCHES = 1920   # 7,680 shared QIDs / 4 per sampled batch
ONE_BIAS_VALIDATION_BATCHES = 64  # 256 encountered QID/bias examples


def next_encounter_slice(state, boundary, max_batches=ONE_BIAS_MAX_BATCHES):
    """One-bias child slice in sampled batches; never crosses a 64-batch boundary or the pool end.

    No-update batches consume encounters, so boundaries are sampled-batch counts,
    not optimizer updates. There is no no-progress cap: the finite pool is the bound.
    """
    if state != progress(state['sampled_batches'], state['optimizer_updates'],
                         no_progress_batches=state['no_progress_batches']):
        raise ValueError('Cursor inconsistent with consumed batches')
    if (type(boundary) is not int or boundary <= state['sampled_batches']
            or (boundary % ONE_BIAS_VALIDATION_BATCHES and boundary != max_batches)):
        raise ValueError('Invalid encounter validation boundary')
    count = min(SEGMENT_BATCHES - state['batch_offset'], boundary - state['sampled_batches'],
                max_batches - state['sampled_batches'])
    if count <= 0:
        raise ValueError('Finite one-bias pool exhausted; never cycle QIDs')
    return dict(segment_index=state['segment_index'], batch_offset=state['batch_offset'], batch_count=count)
