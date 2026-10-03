import json
import subprocess

import pytest

from experiments.gemma4_methods import launch_guard


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.email', 't@t'); git(tmp_path, 'config', 'user.name', 't')
    for name in ('train.py', 'validation.py'):
        (tmp_path / name).write_text('v1\n')
    git(tmp_path, 'add', '.'); git(tmp_path, 'commit', '-qm', 'parent')
    parent = git(tmp_path, 'rev-parse', 'HEAD')
    monkeypatch.setattr(launch_guard, 'INCORPORATION_BASE', parent)
    amendments = tmp_path / 'amend.json'
    amendments.write_text(json.dumps({'amendments': [
        {'parent': parent, 'execution_only_paths': ['validation.py', 'amend.json']}]}))
    monkeypatch.setattr(launch_guard, 'AMENDMENTS', amendments)
    return tmp_path, parent


def test_exact_commit_still_accepted(repo):
    root, parent = repo
    git(root, 'add', '.'); git(root, 'commit', '-qm', 'record')
    assert launch_guard.check_source(root, git(root, 'rev-parse', 'HEAD')) == root.resolve()


def test_execution_only_successor_continues_parent_run(repo):
    root, parent = repo
    (root / 'validation.py').write_text('sharded\n')
    git(root, 'add', '.'); git(root, 'commit', '-qm', 'amend')
    assert launch_guard.check_source(root, parent) == root.resolve()


def test_successor_touching_training_code_is_rejected(repo):
    root, parent = repo
    (root / 'train.py').write_text('changed\n')
    git(root, 'add', '.'); git(root, 'commit', '-qm', 'science change')
    with pytest.raises(ValueError, match='non-execution files: train.py'):
        launch_guard.check_source(root, parent)


def test_unlisted_parent_and_dirty_tree_rejected(repo):
    root, parent = repo
    git(root, 'add', '.'); git(root, 'commit', '-qm', 'x')
    with pytest.raises(ValueError, match='Exact clean'):
        launch_guard.check_source(root, 'f' * 40)
    (root / 'validation.py').write_text('dirty\n')
    with pytest.raises(ValueError, match='Exact clean'):
        launch_guard.check_source(root, parent)


def test_recovery_ignores_inference_view_directories(tmp_path):
    import re
    names = ['step-000001', 'step-000002', 'step-000002.gemma4-vllm-v1']
    assert [n for n in names if re.fullmatch(r'step-\d{6}', n)] == names[:2]


def test_shard_count_requires_one_gpu_per_shard(monkeypatch):
    from experiments.gemma4_methods import validation
    monkeypatch.setenv('GEMMA_VALIDATION_SHARDS', '4')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    with pytest.raises(ValueError, match='own visible GPU'):
        validation._shard_count()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    assert validation._shard_count() == (4, ['0', '1', '2', '3'])
    rows = list(range(600))
    assert sorted(x for i in range(4) for x in rows[i::4]) == rows
