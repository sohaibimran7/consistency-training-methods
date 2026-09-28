"""Disposable one-update native RMCT check. Never resume its output in production.

Uses the approved production token/rollout/optimizer settings on the first two
training QIDs. Requires a passed same-commit native worker preflight. A successful
exit is not sufficient: saved completion evidence and a real update are checked.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from experiments.rmct_restart_20260928.qwen_launch import sha, source_check, write


def validate_rollout_records(records):
    """Validate saved training evidence, not merely process exit status."""
    if not records or not any(not r.skipped_from_training for r in records):
        raise RuntimeError('No usable training rollouts')
    for record in records:
        if len(record.completion_tokens) != len(record.sampled_logprobs):
            raise RuntimeError('Saved token/logprob evidence mismatch')
        if any(not math.isfinite(v) for v in record.sampled_logprobs):
            raise RuntimeError('Nonfinite sampling evidence')
        if record.finish_reason != 'stop' and (record.parsed_successfully or not record.skipped_from_training):
            raise RuntimeError('Incomplete completion entered training')
        if not record.skipped_from_training:
            if not record.completion_tokens or not record.parsed_successfully or record.grader_failed:
                raise RuntimeError('Included rollout lacks valid evidence')
            if any(v is None or not math.isfinite(v) for v in (record.reward, record.advantage)):
                raise RuntimeError('Included rollout lacks finite reward/advantage')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Scheduled allocation required')
    plan = json.loads(a.plan.read_text())
    argv = list(plan['argv'])
    root = Path(argv[1]).resolve().parents[1]
    commit = plan['incorporated_commit']
    source_check(root, commit)
    if Path(__file__).resolve() != root/'experiments/rmct_restart_20260928/native_rl_gate.py':
        raise RuntimeError('Noncanonical gate import')
    if os.path.abspath(sys.executable) != os.path.abspath(argv[0]) or sys.prefix != plan['python_prefix']:
        raise RuntimeError('Interpreter identity mismatch')
    receipt = json.loads(a.preflight.read_text())
    if receipt['source_commit'] != commit or receipt['optimizer_work_authorized'] is not False:
        raise RuntimeError('Wrong preflight receipt')
    evidence = a.preflight.parent
    for name, key in [('cpu.json', 'cpu_receipt_sha256'), ('native/result.json', 'native_result_sha256'),
                      ('native/qwen35-rollout-worker-parity-attestation.json', 'attestation_sha256')]:
        if sha(evidence/name) != receipt[key]:
            raise RuntimeError('Preflight evidence changed')
    if receipt['plan_sha256'] != sha(a.plan):
        raise RuntimeError('Plan changed after preflight')
    if any(arg.startswith('--resume') for arg in argv):
        raise RuntimeError('Fresh base required')
    def replace(flag, value):
        if argv.count(flag) != 1:
            raise RuntimeError('Missing/duplicate flag: ' + flag)
        argv[argv.index(flag)+1] = value
    out = a.output.resolve()
    if out == root or root in out.parents:
        raise RuntimeError('Disposable work must be outside deployment')
    out.mkdir(parents=True, exist_ok=False)
    # Preserve the frozen 32-QID segment contract. The disposable wrapper
    # validates the entire segment before selecting one batch; production's
    # setting and CLI are never weakened to accept an invalid two-QID segment.
    argv[1] = str(root/'experiments/rmct_restart_20260928/native_rl_worker.py')
    replace('--checkpoint-every', '1')
    replace('--experiment-name', 'disposable-native-rl-gate')
    replace('--run-name', 'one-update')
    write(out/'started.json', {'argv': argv, 'source_commit': commit,
        'job_id': os.environ['SLURM_JOB_ID'], 'production_resume_forbidden': True})
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONHOME', None)
    subprocess.run(argv, cwd=out, env=env, check=True)
    from ctm.evals.analysis.rollouts import iter_rollouts
    from ctm.training.resume_state import load_strict_local_rl_resume_state
    indices = list(out.glob('logs/**/rollouts/index.json'))
    if len(indices) != 1:
        raise RuntimeError('Unique rollout evidence required')
    records = list(iter_rollouts(indices[0].parent))
    validate_rollout_records(records)
    states = []
    for path in out.glob('logs/**/checkpoints/**/manifest.json'):
        manifest = json.loads(path.read_text())
        if manifest.get('loop_state', {}).get('final'):
            states.append(load_strict_local_rl_resume_state(path.parent))
    if len(states) != 1 or states[0].optimizer_step != 1 or states[0].global_step != 1:
        raise RuntimeError('Exactly one completed real optimizer update required')
    from safetensors import safe_open
    import torch
    adapter = states[0].checkpoint_dir/'adapter_model.safetensors'
    with safe_open(adapter, framework='pt', device='cpu') as handle:
        keys = list(handle.keys())
        if not keys or any(not torch.isfinite(handle.get_tensor(key)).all() for key in keys):
            raise RuntimeError('Empty/nonfinite updated adapter')
    metrics_path = indices[0].parent.parent/'metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    updates = [m for m in metrics if m.get('train/optimizer_step') == 1]
    if len(updates) != 1 or updates[0].get('train/skipped_empty_batch') != 0:
        raise RuntimeError('No unique non-skipped update metrics')
    if any(isinstance(v, float) and not math.isfinite(v) for m in metrics for v in m.values()):
        raise RuntimeError('Nonfinite training metrics')
    source_check(root, commit)
    write(out/'receipt.json', {'schema': 'rmct-native-rl-gate-v1', 'status': 'passed',
        'source_commit': commit, 'plan_sha256': sha(a.plan), 'preflight_sha256': sha(a.preflight),
        'optimizer_steps': 1, 'production_resume_forbidden': True,
        'rollout_count': len(records), 'rollout_index_sha256': sha(indices[0]),
        'metrics_sha256': sha(metrics_path), 'adapter_sha256': sha(adapter),
        'subset_sha256': sha(out/'disposable-subset.json'),
        'checkpoint_manifest_sha256': sha(states[0].checkpoint_dir/'manifest.json'),
        'limitations': ['one batch is not a convergence or full-run capacity guarantee']})


if __name__ == '__main__':
    main()
