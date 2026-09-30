"""Scheduled, load-only diagnostic. Never generate, backward, step or save.

Uses the production replicated setup and additionally materializes the lazy
AdamW state on every GPU. Outputs are diagnostic receipts, never checkpoints.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from ctm.backends.local.engine import LocalBackend
from experiments.rmct_restart_20260928.qwen_validation_executor import read, write


def state_digest(value):
    """Device-independent digest of exact tensor/array state and containers."""
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous()
            h.update(repr(('tensor',str(t.dtype),tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            h.update(repr(('dict',len(x))).encode())
            for k in sorted(x,key=repr):visit(k);visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(repr((type(x).__name__,len(x))).encode())
            for item in x:visit(item)
        elif hasattr(x,'dtype') and hasattr(x,'tobytes'):
            h.update(repr(('array',str(x.dtype),tuple(x.shape))).encode());h.update(x.tobytes())
        elif x is None or type(x) in (str,int,float,bool):
            raw=repr((type(x).__name__,x)).encode();h.update(str(len(raw)).encode()+b':'+raw)
        else:raise TypeError(f'Unsupported state value {type(x)}')
    visit(value)
    return h.hexdigest()


def materialize_optimizer(backend, *, learning_rate, adam):
    """Same AdamW construction/load as production, deliberately without step."""
    pending=backend._pending_optimizer_state
    if not isinstance(pending,dict) or not pending.get('state'):
        raise ValueError('Missing nonempty staged optimizer state')
    expected=state_digest(pending)
    optimizer=torch.optim.AdamW([p for p in backend.model.parameters() if p.requires_grad],
        lr=learning_rate,betas=(adam['beta1'],adam['beta2']),eps=adam['eps'],weight_decay=adam['weight_decay'])
    optimizer.load_state_dict(pending)
    actual=state_digest(optimizer.state_dict())
    if actual!=expected:raise ValueError('Restored optimizer state differs from saved state')
    if any(p.grad is not None for group in optimizer.param_groups for p in group['params']):
        raise ValueError('Unexpected gradients in load-only gate')
    backend._optimizer=optimizer;backend._pending_optimizer_state=None
    return actual


class LoadOnlyBackend(LocalBackend):
    def __init__(self, *, gate_directory, gate_optimizer, **kwargs):
        self.gate_directory=Path(gate_directory);self.gate_optimizer=gate_optimizer
        super().__init__(**kwargs)

    def setup(self, **kwargs):
        if not kwargs.get('resume_from') or kwargs.get('resume_with_optimizer') is not True:
            raise ValueError('Saved optimizer checkpoint required')
        super().setup(**kwargs)
        digest=materialize_optimizer(self,**self.gate_optimizer)
        device=str(self.device)
        if not device.startswith('cuda:'):raise ValueError('GPU load required')
        write(self.gate_directory/f'optimizer-{device.replace(":","-")}.json',
              dict(device=device,optimizer_state_sha256=digest,resume_from=kwargs['resume_from'],
                   optimizer_step_calls=0,generation_calls=0))

    async def submit_optim_step(self, **kwargs):raise RuntimeError('Optimizer step forbidden')
    async def submit_forward_backward(self, *args, **kwargs):raise RuntimeError('Backward forbidden')
    async def save_checkpoint(self, *args, **kwargs):raise RuntimeError('Checkpoint write forbidden')
    def policy_sampler(self, *args, **kwargs):raise RuntimeError('Sampling forbidden')
    def base_sampler(self, *args, **kwargs):raise RuntimeError('Sampling forbidden')


def load_and_observe(trainer, output):
    from ctm.backends.local.replicated import (ReplicatedTrainingBackend,LocalBackendConstructorSpec,
        _capture_rng_state,_rank_device)
    from ctm.backends.local.rollout_workers import RolloutParallelBackend
    from ctm.training.resume_state import restore_runtime_rng_state,capture_runtime_rng_state
    if not isinstance(trainer.backend,RolloutParallelBackend):raise ValueError('Wrong production backend')
    replicated=trainer.backend.training_backend
    if not isinstance(replicated,ReplicatedTrainingBackend) or replicated.world_size!=4:
        raise ValueError('Four-rank replicated backend required')
    if trainer.resume_state is None or (trainer.resume_state.global_step,trainer.resume_state.optimizer_step)!=(16,12):
        raise ValueError('Wrong saved counters')
    spec=replicated.child_backend_spec
    if spec.constructor is not LocalBackend:raise ValueError('Unexpected replica constructor')
    kwargs=dict(spec.kwargs,gate_directory=str(output),gate_optimizer=dict(
        learning_rate=trainer.config.optimizer.learning_rate,adam=trainer.config.optimizer.model_dump()))
    replicated.child_backend_spec=LocalBackendConstructorSpec(LoadOnlyBackend,kwargs=kwargs)
    replicated.training_backend=LoadOnlyBackend(device=_rank_device(replicated.topology,0,device_type='cuda'),**kwargs)
    try:
        replicated.setup(model=trainer.config.model,lora=trainer.config.lora,
                         resume_from=trainer.resume_from,resume_with_optimizer=True)
        expected=replicated._load_resume_contract(resume_from=trainer.resume_from,resume_with_optimizer=True)
        command_id=replicated._dispatch('rng_state',{})
        peers=replicated._collect_results(command_id,set(range(1,4)))
        actual={0:_capture_rng_state(device_type='cuda',device=_rank_device(replicated.topology,0,device_type='cuda'))}
        actual.update({rank:response.payload['rng_state'] for rank,response in peers.items()})
        if set(actual)!=set(range(4)):raise ValueError('Missing rank RNG response')
        rng_hashes={}
        for rank in range(4):
            rng_hashes[str(rank)]=state_digest(actual[rank])
            if rng_hashes[str(rank)]!=state_digest(expected['rng_by_rank'][rank]):
                raise ValueError(f'Rank {rank} RNG not restored exactly')
        restore_runtime_rng_state(trainer.resume_state.runtime_rng,require_torch=True)
        coordinator_rng_sha256=state_digest(capture_runtime_rng_state())
        if coordinator_rng_sha256!=state_digest(trainer.resume_state.runtime_rng):
            raise ValueError('Coordinator RNG not restored exactly')
        records=[read(output/f'optimizer-cuda-{rank}.json') for rank in range(4)]
        if len({r['optimizer_state_sha256'] for r in records})!=1:
            raise ValueError('Optimizer states differ across ranks')
        if any(r['resume_from']!=trainer.resume_from for r in records):raise ValueError('Wrong rank parent')
        return dict(sampled_batches=16,optimizer_updates=12,next_batch=16,
            optimizer_rank_records=records,rng_sha256_by_rank=rng_hashes,
            coordinator_rng_restored=True,coordinator_rng_sha256=coordinator_rng_sha256,
            adapter_state_hash=replicated._rank_zero_state_hash,
            operations=list(replicated.operation_history),production_resume_forbidden=True,
            generation_calls=0,backward_calls=0,optimizer_step_calls=0,
            limitation='No vLLM private RNG restoration; no training or rollout throughput measurement')
    finally:replicated.shutdown()


def main():
    from scripts import train_rlct
    from experiments.rmct_restart_20260928.qwen_train_window import (
        plan_binding,recover_v2,verify_seal_v2,command_v2,gates)
    from experiments.rmct_restart_20260928.qwen_progress import next_slice
    from experiments.rmct_restart_20260928.qwen_launch import source_check
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','output','preflight','rl-gate','regression'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Scheduled allocation required')
    plan=read(a.plan);binding=plan_binding(plan,a.plan);root=Path(binding['source_root'])
    if Path(__file__).resolve()!=root/'experiments/rmct_restart_20260928/qwen_resume_load_gate.py':
        raise ValueError('Noncanonical gate source')
    if os.path.abspath(sys.executable)!=os.path.abspath(plan['argv'][0]) or sys.prefix!=plan['python_prefix']:
        raise ValueError('Wrong interpreter')
    checked=gates(plan,a.plan,a.preflight,a.rl_gate,a.regression,root)
    recovered=recover_v2(plan,binding);verify_seal_v2(recovered,plan,binding)
    out=a.output.resolve()
    if out.is_relative_to(root) or out.is_relative_to(Path(recovered['checkpoint'])):
        raise ValueError('Gate output must be separate')
    out.mkdir(parents=True,exist_ok=False)
    argv=command_v2(plan,recovered['progress'],next_slice(recovered['progress'],64),recovered)
    write(out/'started.json',dict(binding=binding,gates=checked,checkpoint=recovered,
          job_id=os.environ['SLURM_JOB_ID'],mode='load_only',argv=argv))
    original=train_rlct.RLTrainer;observed=[]
    class Probe(original):
        def setup(self):observed.append(load_and_observe(self,out));raise Finished()
        async def train(self,**kwargs):raise RuntimeError('Training forbidden')
    class Finished(Exception):pass
    train_rlct.RLTrainer=Probe
    try:
        try:train_rlct.main(argv[2:])
        except Finished:pass
    finally:train_rlct.RLTrainer=original
    if len(observed)!=1:raise ValueError('Missing load evidence')
    verify_seal_v2(recovered,plan,binding);source_check(root,binding['source_commit'])
    write(out/'receipt.json',dict(schema='rmct-saved-resume-load-v1',status='passed',binding=binding,
          gates=checked,checkpoint=recovered,job_id=os.environ['SLURM_JOB_ID'],**observed[0]))


if __name__=='__main__':main()
