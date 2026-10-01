"""Fresh Gemma coordinator RNG checkpoints; no legacy checkpoint mutation.

This does NOT capture private vLLM worker RNG streams. Native continuation
verification must account for that separately; trainer state is not proof of
bitwise-identical resumed rollout sampling.
"""
import json
import hashlib
import os
import uuid
from pathlib import Path

from ctm.training.resume_state import capture_runtime_rng_state, restore_runtime_rng_state


def require_coordinator_rng(state):
    from ctm.training.resume_state import RUNTIME_RNG_SCHEMA
    if (state.get('schema') != RUNTIME_RNG_SCHEMA
            or not isinstance(state.get('python_random_state'),list)
            or not state.get('torch_cpu_rng_state_base64')
            or not state.get('torch_cuda_rng_state_base64')
            or type(state.get('torch_cuda_coordinator_device')) is not int):
        raise ValueError('Complete CUDA trainer RNG state required')
    return state


class RNGCheckpointBackend:
    def __init__(self, backend, publication=None):
        self.backend = backend
        self.publication=publication

    async def save_checkpoint(self, **kwargs):
        rng = require_coordinator_rng(capture_runtime_rng_state())
        kwargs['loop_state'] = {**kwargs['loop_state'], 'runtime_rng':rng,
                               'rollout_worker_rng_serialized':False}
        result=await self.backend.save_checkpoint(**kwargs)
        if self.publication is not None:
            from experiments.gemma4_methods.reference.plan import immutable_json
            directory=Path(kwargs['log_dir'])/'checkpoints'/kwargs['name']
            immutable_json(directory/'publication.json',self.publication)
        return result


async def seal_checkpoint(backend, **kwargs):
    from experiments.gemma4_methods.reference.train import seal_checkpoint as original_seal
    publication={k:kwargs[k] for k in ('method','state','plan_hash','window_metrics')}
    return await original_seal(RNGCheckpointBackend(backend,publication),**kwargs)


def make_progress(receipt):
    from experiments.gemma4_methods.train import ORDER_SHA
    state=receipt['convergence']
    return {'schema':'gemma-trainer-progress-draft-v1','plan_sha256':receipt['plan_sha256'],
        'actual_optimizer_step':state['step'],'next_attempt_index':state['attempts'],
        'attempted_batches':state['attempts'],'sampled_batches':None,
        'sampled_batches_note':'derive from generation events; attempts can include cached skip replay',
        'ordered_pool_sha256':ORDER_SHA,'consumed_qid_position':2*state['attempts'],
        'last_update_question_ids':state['last_update_question_ids'],
        'latest_checkpoint':receipt['checkpoint'],'checkpoint_files':receipt['checkpoint_files'],
        'selected_checkpoint':None,'validation_required':state['step']%64==0,
        'rng_metadata':'manifest coordinator RNG saved/read-back restored; private vLLM worker RNG not serialized; native resume gate required',
        'controller_adapter':'pending; not authorization to continue past boundary'}


def recover_publication(run_dir,plan_hash,method):
    """Rebuild a complete fresh lineage ONLY from immutable checkpoints.

    Preserve original metrics by digest, reconstructing authoritative rows from
    checkpoint publication sidecars rather than keeping duplicate retry rows.
    Incomplete staging directories are never counted as saved updates.
    """
    from experiments.gemma4_methods.reference import train as helpers,plan
    rows=[]
    latest=None
    for index,directory in enumerate(sorted((run_dir/'checkpoints').glob('step-*')),1):
        if directory.is_symlink() or directory.name!=f'step-{index:06d}':
            raise ValueError('Incomplete or foreign checkpoint sequence')
        files=helpers.checkpoint_identity(directory)
        manifest=json.loads((directory/'manifest.json').read_text())
        publication=json.loads((directory/'publication.json').read_text())
        loop=manifest['loop_state']
        state=publication['state']
        if (publication['method'],publication['plan_hash'],state['step'],manifest['kind']) != (method,plan_hash,index,'both'):
            raise ValueError('Immutable checkpoint publication lineage differs')
        if loop['convergence']!=state or loop['method']!=method or loop['plan_sha256']!=plan_hash:
            raise ValueError('Native checkpoint metadata differs from publication')
        require_coordinator_rng(loop['runtime_rng'])
        metrics=publication['window_metrics']
        if json.loads((directory/'window-metrics.json').read_text())!=metrics:
            raise ValueError('Immutable checkpoint metrics differ')
        if not metrics or metrics[-1]['step']!=index or metrics[-1]['attempt']+1!=state['attempts']:
            raise ValueError('Saved update/cursor metric differs')
        if metrics[-1]['question_ids']!=state['last_update_question_ids']:
            raise ValueError('Saved update QIDs differ')
        rows.append(metrics[-1])
        latest={'schema':'ctm-grouped-qid-resume-v1','method':method,'plan_sha256':plan_hash,
            'convergence':state,'checkpoint':str(directory.relative_to(run_dir)),'checkpoint_files':files}
        plan.immutable_json(run_dir/'receipts'/f'step-{index:06d}.json',latest)
        plan.immutable_json(run_dir/'progress'/f'step-{index:06d}.json',make_progress(latest))
    log=run_dir/'metrics.jsonl'
    payload=b''.join((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode() for row in rows)
    original=log.read_bytes() if log.exists() else b''
    if original!=payload:
        archive=run_dir/'recovery'/f'metrics-{hashlib.sha256(original).hexdigest()}.jsonl'
        archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists():
            if archive.read_bytes()!=original:
                raise ValueError('Recovery archive conflict')
        else:
            with archive.open('xb') as stream:
                stream.write(original);stream.flush();os.fsync(stream.fileno())
        temporary=run_dir/f'.recovered-metrics-{uuid.uuid4().hex}.tmp'
        with temporary.open('xb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,log)
    if latest:
        helpers.atomic_json(run_dir/'state.json',latest)
    elif (run_dir/'state.json').exists():
        raise ValueError('State pointer exists without a durable checkpoint')
    return latest


def restore_coordinator_rng(checkpoint):
    """After model setup, restore and read back actual trainer RNG state."""
    path = str(checkpoint)
    if path.startswith('file://'):
        path = path[7:]
    manifest = json.loads((Path(path)/'manifest.json').read_text())
    rng = require_coordinator_rng(manifest['loop_state']['runtime_rng'])
    restore_runtime_rng_state(rng,require_torch=True)
    if capture_runtime_rng_state() != rng:
        raise ValueError('Restored trainer RNG differs from saved state')
    return rng
