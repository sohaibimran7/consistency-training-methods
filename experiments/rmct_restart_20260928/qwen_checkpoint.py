"""Strict clean-lineage checkpoint seals; no legacy campaign dependencies."""
import hashlib
import json
from pathlib import Path


def identity(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not path.stat().st_size:
        raise ValueError(f'Missing/empty/nonregular checkpoint file: {path}')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return dict(path=str(path),sha256=digest.hexdigest(),bytes=path.stat().st_size)


def validate_metadata(manifest,replicated,step):
    loop=manifest.get('loop_state',{})
    if manifest.get('backend')!='local' or manifest.get('kind')!='both':raise ValueError('Wrong checkpoint kind')
    if any(loop.get(k)!=step for k in ('global_step','optimizer_step','step')):
        raise ValueError('Checkpoint step mismatch')
    if loop.get('final') is not True or loop.get('accumulated_grads')!=0:
        raise ValueError('Not final optimizer boundary')
    validate_replicated(replicated)


def validate_replicated(replicated):
    if (replicated.get('checkpoint_kind')!='both' or replicated.get('world_size')!=4
        or replicated.get('train_logical_indices')!=[0,1,2,3]
        or replicated.get('process_group_backend')!='nccl' or replicated.get('device_type')!='cuda'
        or replicated.get('rng_state_file')!='replicated_training_rng.pt'):
        raise ValueError('Missing four-rank optimizer/RNG contract')


def seal(checkpoint, *, campaign_root, campaign_id, step, command, parent=None):
    from ctm.training.resume_state import load_strict_local_rl_resume_state
    checkpoint=Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():raise ValueError('Regular checkpoint directory required')
    checkpoint=checkpoint.resolve();root=Path(campaign_root).resolve()
    if not checkpoint.is_relative_to(root):raise ValueError('Checkpoint outside fresh campaign')
    if not campaign_id or type(step) is not int or step<16 or step%16:raise ValueError('Invalid step')
    if step==16:
        if parent is not None or any(a.startswith('--resume') for a in command):raise ValueError('Initial segment must be fresh')
    else:
        if not parent or parent['campaign_id']!=campaign_id or parent['step']!=step-16:
            raise ValueError('Wrong clean parent')
        for file in parent['files'].values():
            if identity(file['path'])!=file:raise ValueError('Parent bytes changed')
        required=['--resume-from','--resume-with-optimizer','--resume-state-required']
        if any(flag not in command for flag in required):raise ValueError('Strict resume flags required')
        if command[command.index('--resume-from')+1]!='file://'+parent['checkpoint']:
            raise ValueError('Command resumes different parent')
    files={name:identity(checkpoint/name) for name in ('adapter_config.json','adapter_model.safetensors',
           'optimizer.pt','manifest.json','replicated_training_manifest.json','replicated_training_rng.pt')}
    manifest=json.loads((checkpoint/'manifest.json').read_text())
    replicated=json.loads((checkpoint/'replicated_training_manifest.json').read_text())
    validate_metadata(manifest,replicated,step)
    state=load_strict_local_rl_resume_state(checkpoint)
    if state.global_step!=step or state.optimizer_step!=step or state.completed_epochs!=manifest['loop_state']['completed_epochs']:
        raise ValueError('Strict resume state mismatch')
    return dict(schema='rmct-clean-checkpoint-v1',campaign_id=campaign_id,step=step,
                checkpoint=str(checkpoint),files=files,parent=parent,command=command)


FILES = ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
         'manifest.json', 'replicated_training_manifest.json', 'replicated_training_rng.pt')


def checkpoint_state(checkpoint, before, selection):
    """Read real counters without changing any saved checkpoint bytes."""
    from ctm.training.resume_state import load_strict_local_rl_resume_state
    from experiments.rmct_restart_20260928.qwen_progress import advance, validate_loop
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError('Regular checkpoint directory required')
    files = {name: identity(checkpoint / name) for name in FILES}
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    if manifest.get('backend') != 'local' or manifest.get('kind') != 'both':
        raise ValueError('Wrong checkpoint kind')
    loop = manifest['loop_state']
    after = advance(before, selection, loop['optimizer_step'])
    validate_loop(loop, after)
    if (loop.get('segment_start_global_step') != before['sampled_batches']
            or loop.get('segment_step') != selection['batch_count']):
        raise ValueError('Checkpoint did not consume exactly the requested slice')
    validate_replicated(json.loads((checkpoint / 'replicated_training_manifest.json').read_text()))
    state = load_strict_local_rl_resume_state(checkpoint)
    if (state.global_step != after['sampled_batches'] or state.optimizer_step != after['optimizer_updates']
            or state.completed_epochs != loop['completed_epochs']):
        raise ValueError('Strict resume state mismatch')
    return after, files


def seal_v2(checkpoint, *, campaign_id, before, selection, command, binding, parent=None):
    """A completed child: batch cursor and optimizer count are distinct."""
    from experiments.rmct_restart_20260928.qwen_progress import progress
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink():
        raise ValueError('Symlink checkpoint')
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to(Path(binding['source_root']) / 'logs'):
        raise ValueError('Checkpoint outside bound source logs')
    if parent is None:
        if before != progress(0, 0) or any(a.startswith('--resume') for a in command):
            raise ValueError('Initial child must be fresh')
    else:
        if parent['campaign_id'] != campaign_id or parent['progress'] != before:
            raise ValueError('Wrong parent counters or campaign')
        if parent['binding'] != binding:
            raise ValueError('Parent must be explicitly migrated to the current binding')
        for record in parent['files'].values():
            if identity(record['path']) != record:
                raise ValueError('Parent bytes changed')
        for flag in ('--resume-from', '--resume-with-optimizer', '--resume-state-required'):
            if command.count(flag) != 1:
                raise ValueError('Strict resume flags required exactly once')
        if command[command.index('--resume-from') + 1] != 'file://' + parent['checkpoint']:
            raise ValueError('Wrong resumed checkpoint')
    after, files = checkpoint_state(checkpoint, before, selection)
    return dict(schema='rmct-clean-checkpoint-v2', campaign_id=campaign_id,
                step=after['optimizer_updates'], progress=after, before=before, selection=selection,
                continuation_mode='optimizer_data_segment', bitwise_rollout_continuation=False,
                checkpoint=str(checkpoint), files=files, command=command, binding=binding, parent=parent)
