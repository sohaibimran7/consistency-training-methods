"""Validation boundary checks; no GPU or remote work."""
import sys
import pytest
from experiments.rmct_restart_20260928 import qwen_validation_prepare as v


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    binding = dict(plan={'path': 'plan.json'}, source_commit='fixed', source_root=str(tmp_path))
    plan = dict(argv=[sys.executable, 'train.py', '--model', 'model'], python_prefix=sys.prefix)
    receipt = dict(binding=binding, step=64, progress=dict(optimizer_updates=64))
    monkeypatch.setattr(v, 'read', lambda path: plan)
    monkeypatch.setattr(v, 'plan_binding', lambda p, path: binding)
    calls = []
    def verify(r, p, b):
        calls.append(r)
        return r
    monkeypatch.setattr(v, 'verify_seal_v2', verify)
    return receipt, plan, calls, tmp_path


def test_calls_full_verifier(fixture):
    receipt, plan, calls, root = fixture
    assert v.verified_validation_checkpoint(receipt, root, 'fixed', 'model') == receipt
    assert calls == [receipt]


@pytest.mark.parametrize('step,updates', [(12,12), (16,12), (0,0), (True,True), (64,12), (65,65), (64.0,64)])
def test_rejects_non_optimizer_boundary(fixture, step, updates):
    receipt, _, _, root = fixture
    receipt.update(step=step, progress=dict(optimizer_updates=updates))
    with pytest.raises(ValueError, match='actual 64-update'):
        v.verified_validation_checkpoint(receipt, root, 'fixed', 'model')


@pytest.mark.parametrize('field', ['source', 'root', 'model', 'python', 'prefix'])
def test_rejects_runtime_mismatch(fixture, field):
    receipt, plan, calls, root = fixture
    commit, model = 'fixed', 'model'
    if field == 'source': commit = 'other'
    if field == 'root': root = root / 'other'
    if field == 'model': model = 'other'
    if field == 'python': plan['argv'][0] = '/other/python'
    if field == 'prefix': plan['python_prefix'] = '/other'
    with pytest.raises(ValueError):
        v.verified_validation_checkpoint(receipt, root, commit, model)
    assert not calls


def test_verifier_failure_propagates(fixture, monkeypatch):
    receipt, _, _, root = fixture
    def fail(*args): raise ValueError('tampered lineage')
    monkeypatch.setattr(v, 'verify_seal_v2', fail)
    with pytest.raises(ValueError, match='tampered lineage'):
        v.verified_validation_checkpoint(receipt, root, 'fixed', 'model')
