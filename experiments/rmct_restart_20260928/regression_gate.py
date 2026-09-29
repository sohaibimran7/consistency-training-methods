"""Run restart-critical regressions in the deployed interpreter; immutable receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.rmct_restart_20260928.qwen_launch import source_check, sha, write

TESTS = [
    'tests/test_completion_contract.py', 'tests/test_native_rl_gate.py',
    'tests/test_restart_preflight.py', 'tests/test_ctm_backend_integration.py',
    'tests/test_ctm_rl_anchor.py', 'tests/test_ctm_rl_phase_sharing.py',
    'tests/test_ctm_rollout_log.py', 'tests/test_ctm_rollout_workers.py',
    'tests/test_mcq_bias_parser_compat.py', 'tests/test_parser_terminal_pilot.py',
    'tests/test_mcq_bias_answer_safety.py', 'tests/test_gemma_thinking_attestation.py',
    'tests/test_terminal_context_safety.py',
    'experiments/rmct_restart_20260928/test_prepare.py',
    'experiments/rmct_restart_20260928/test_qwen_validation.py',
    'experiments/rmct_restart_20260928/test_qwen_runtime_boundaries.py',
    'experiments/rmct_restart_20260928/test_qwen_controller.py',
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.source_root.resolve()
    source_check(root, a.source_commit)
    if Path(__file__).resolve() != root/'experiments/rmct_restart_20260928/regression_gate.py':
        raise RuntimeError('Noncanonical regression gate')
    out = a.output.resolve()
    if out == root or root in out.parents:
        raise RuntimeError('Evidence must be outside source')
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONHOME', None)
    with (out/'pytest.txt').open('x') as log:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *TESTS],
                       cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    source_check(root, a.source_commit)
    write(out/'receipt.json', {'schema': 'rmct-integrated-regression-v1', 'status': 'passed',
        'source_commit': a.source_commit, 'source_root': str(root),
        'python': sys.executable, 'sys_prefix': sys.prefix,
        'tests': {name: sha(root/name) for name in TESTS}, 'pytest_log_sha256': sha(out/'pytest.txt'),
        'optimizer_work_authorized': False})


if __name__ == '__main__':
    main()
