"""Fresh Gemma coordinator RNG checkpoints; no legacy checkpoint mutation.

This does NOT capture private vLLM worker RNG streams. Native continuation
verification must account for that separately; trainer state is not proof of
bitwise-identical resumed rollout sampling.
"""
import json
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
    def __init__(self, backend):
        self.backend = backend

    async def save_checkpoint(self, **kwargs):
        rng = require_coordinator_rng(capture_runtime_rng_state())
        kwargs['loop_state'] = {**kwargs['loop_state'], 'runtime_rng':rng,
                               'rollout_worker_rng_serialized':False}
        return await self.backend.save_checkpoint(**kwargs)


async def seal_checkpoint(backend, **kwargs):
    from experiments.gemma4_methods.reference.train import seal_checkpoint as original_seal
    return await original_seal(RNGCheckpointBackend(backend),**kwargs)


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
