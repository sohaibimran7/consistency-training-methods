"""CPU tests of load-only diagnostics, not a GPU resume receipt."""
import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from experiments.rmct_restart_20260928.qwen_resume_load_gate import state_digest,materialize_optimizer,LoadOnlyBackend
from experiments.rmct_restart_20260928.qwen_train_window import require_resume_load_gate
from experiments.rmct_restart_20260928.qwen_validation_executor import write


def fixture():
    model=torch.nn.Linear(2,1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001)
    model(torch.ones(1,2)).sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    return SimpleNamespace(model=model,_pending_optimizer_state=copy.deepcopy(optimizer.state_dict()))


def test_materializes_without_step_or_backward():
    backend=fixture();before=state_digest(backend.model.state_dict());pending=state_digest(backend._pending_optimizer_state)
    with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('step forbidden')):
        digest=materialize_optimizer(backend,learning_rate=0.0001,
            adam=dict(beta1=.9,beta2=.999,eps=1e-8,weight_decay=.01))
    assert digest==pending
    assert backend._pending_optimizer_state is None
    assert state_digest(backend.model.state_dict())==before


def test_missing_state_fails():
    backend=fixture();backend._pending_optimizer_state=None
    with pytest.raises(ValueError):materialize_optimizer(backend,learning_rate=.0001,adam={})


def test_changed_loaded_state_fails():
    backend=fixture()
    with patch.object(torch.optim.AdamW,'load_state_dict',return_value=None):
        with pytest.raises(ValueError,match='differs'):
            materialize_optimizer(backend,learning_rate=.0001,adam=dict(beta1=.9,beta2=.999,eps=1e-8,weight_decay=.01))


def test_digest_exactness():
    x={'step':torch.tensor(12.),'weights':torch.tensor([1.,2.]),'rng':(1,[2,3])}
    assert state_digest(x)==state_digest(copy.deepcopy(x))
    changed=copy.deepcopy(x);changed['weights'][0]=3
    assert state_digest(x)!=state_digest(changed)
    # A container boundary must not collide with a different nesting shape.
    assert state_digest([[1],2]) != state_digest([[1,2]])


@pytest.mark.parametrize('method',['submit_optim_step','submit_forward_backward','save_checkpoint'])
def test_training_operations_forbidden(method):
    with pytest.raises(RuntimeError):asyncio.run(getattr(LoadOnlyBackend,method)(None))


@pytest.mark.parametrize('method',['policy_sampler','base_sampler'])
def test_sampling_forbidden(method):
    with pytest.raises(RuntimeError):getattr(LoadOnlyBackend,method)(None)


@pytest.fixture
def receipt(tmp_path):
    parent={'checkpoint':'/saved/checkpoint','progress':{'sampled_batches':16,'optimizer_updates':12}}
    binding={'source_commit':'same'};gates={'native':'same'}
    rows=[dict(device=f'cuda:{r}',optimizer_state_sha256='a'*64,
               resume_from='file:///saved/checkpoint',generation_calls=0,optimizer_step_calls=0) for r in range(4)]
    value=dict(schema='rmct-saved-resume-load-v1',status='passed',binding=binding,gates=gates,
        checkpoint=parent,production_resume_forbidden=True,sampled_batches=16,optimizer_updates=12,next_batch=16,
        coordinator_rng_restored=True,coordinator_rng_sha256='b'*64,operations=[{'operation':'setup'}],
        generation_calls=0,backward_calls=0,optimizer_step_calls=0,optimizer_rank_records=rows,
        rng_sha256_by_rank={str(r):'c'*64 for r in range(4)},adapter_state_hash='d'*64,job_id='123')
    return tmp_path/'load.json',value,parent,binding,gates


def test_load_receipt_requires_completed_same_parent_job(receipt):
    path,value,parent,binding,gates=receipt;write(path,value)
    with patch('experiments.rmct_restart_20260928.qwen_train_window.subprocess.check_output',return_value='COMPLETED|0:0\n'):
        assert require_resume_load_gate(path,parent,binding,gates)['path']==str(path)
    with patch('experiments.rmct_restart_20260928.qwen_train_window.subprocess.check_output',return_value='RUNNING|0:0\n'):
        with pytest.raises(ValueError,match='scheduler completion'):
            require_resume_load_gate(path,parent,binding,gates)


@pytest.mark.parametrize('field,value',[
    ('optimizer_updates',16),('next_batch',12),('generation_calls',1),('backward_calls',1),
    ('optimizer_step_calls',1),('optimizer_step_calls',False),('coordinator_rng_restored',False),
    ('coordinator_rng_sha256','invalid'),('operations',[{'operation':'checkpoint'}]),
    ('rng_sha256_by_rank',{}),('adapter_state_hash','invalid'),('job_id','untrusted'),
    ('binding',{'source_commit':'other'}),('gates',{'native':'other'})])
def test_changed_or_incomplete_load_receipt_rejected_before_scheduler(receipt,field,value):
    path,data,parent,binding,gates=receipt;data[field]=value;write(path,data)
    with patch('experiments.rmct_restart_20260928.qwen_train_window.subprocess.check_output') as scheduler:
        with pytest.raises(ValueError):require_resume_load_gate(path,parent,binding,gates)
        scheduler.assert_not_called()


def test_missing_or_duplicated_rank_cannot_pass(receipt):
    path,data,parent,binding,gates=receipt
    data['optimizer_rank_records'][3]=data['optimizer_rank_records'][2]
    write(path,data)
    with pytest.raises(ValueError,match='duplicated GPU rank'):
        require_resume_load_gate(path,parent,binding,gates)


def test_recovery_requires_load_gate_path():
    with pytest.raises(ValueError,match='completed optimizer/RNG'):
        require_resume_load_gate(None,{}, {}, {})
