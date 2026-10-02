"""Scheduled first-segment launcher; never submits jobs or resumes old weights.

Preflight-only until all integration and native RL regression gates exist.
No optimizer launch or gate retries are permitted by this draft.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())


def source_check(repo, commit):
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if actual!=commit: raise RuntimeError('Deployed HEAD differs from incorporated commit')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip():
        raise RuntimeError('Tracked deployment differs from incorporated commit')


def cpu_gate_check(receipt, repo, commit, python, prefix):
    # Explicit integration contract, not an inferred status from file existence.
    if receipt.get('schema')!='rmct-restart-cpu-v1' or receipt.get('status')!='cpu_checks_passed':
        raise RuntimeError('CPU gate receipt not passed')
    if receipt.get('optimizer_work_authorized') is not False:
        raise RuntimeError('CPU receipt must not authorize optimizer work')
    if (receipt.get('source_commit')!=commit or receipt.get('source_root')!=str(repo)
        or receipt.get('cwd')!=str(repo) or receipt.get('python')!=python
        or receipt.get('sys_prefix')!=prefix):
        raise RuntimeError('CPU gate source identity mismatch')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation-manifest',type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Run inside scheduled allocation')
    plan=json.loads(a.plan.read_text()); argv=plan['argv']
    repo=Path(argv[1]).resolve().parents[1]; commit=plan['incorporated_commit']
    if os.path.abspath(sys.executable)!=os.path.abspath(argv[0]):
        raise RuntimeError('Launcher Python is not planned Python')
    if os.path.abspath(sys.prefix)!=os.path.abspath(plan['python_prefix']):
        raise RuntimeError('Launcher environment prefix differs from plan')
    own=Path(__file__).resolve()
    if own!=repo/'experiments/rmct_restart_20260928/qwen_launch.py':
        raise RuntimeError('Launcher is not from the canonical deployment')
    subprocess.run(['git','ls-files','--error-unmatch',str(own.relative_to(repo))],cwd=repo,check=True,capture_output=True)
    if any(x.startswith('--resume') for x in argv): raise RuntimeError('Fresh-start only')
    load=json.loads(argv[argv.index('--load-config')+1])
    # One-bias plans start at absolute sampled batch 0; two-bias plans at segment 0.
    if (load.get('attempt_offset') if 'attempt_offset' in load else load.get('segment_index'))!=0:
        raise RuntimeError('Fresh segment0 required')
    if argv[argv.index('--max-new-tokens')+1]!='20480': raise RuntimeError('Cap mismatch')
    source_check(repo,commit)
    out=a.output.resolve()
    if out.is_relative_to(repo) or a.plan.resolve().is_relative_to(repo):
        raise RuntimeError('Plans and outputs must be outside clean deployment')
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(repo),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONHOME',None)
    model=argv[argv.index('--model')+1]
    native=out/'native'
    expected=native/'qwen35-rollout-worker-parity-attestation.json'
    if Path(argv[argv.index('--local-qwen35-rollout-parity-attestation')+1]).resolve()!=expected:
        raise RuntimeError('Plan must bind THIS fresh native attestation')
    write(out/'started.json',dict(job_id=os.environ['SLURM_JOB_ID'],source_commit=commit,
          source_root=str(repo),python=sys.executable,sys_prefix=sys.prefix,plan_sha256=sha(a.plan),
          validation_manifest_sha256=sha(a.validation_manifest),training_requested=False))
    cpu=out/'cpu.json'
    subprocess.run([sys.executable,'-m','experiments.rmct_restart_20260928.restart_preflight','--source-root',str(repo),
        '--source-commit',commit,'--model',model,'--family','qwen',
        '--validation-manifest',str(a.validation_manifest.resolve()),'--output',str(cpu)],
        cwd=repo,env=env,check=True)
    cpu_gate_check(json.loads(cpu.read_text()),repo,commit,sys.executable,sys.prefix)
    subprocess.run([sys.executable,str(repo/'infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py'),
        '--model-snapshot',model,'--output-dir',str(native)],cwd=repo,env=env,check=True)
    result=json.loads((native/'result.json').read_text())
    if not (native/'SUCCESS').is_file() or result.get('passed') is not True or result.get('status')!='passed':
        raise RuntimeError('Fresh native preflight failed')
    if result.get('model_snapshot')!=str(Path(model).resolve()) or result.get('attestation_sha256')!=sha(expected):
        raise RuntimeError('Native model/attestation mismatch')
    source_check(repo,commit)
    write(out/'preflight-results.json',dict(source_commit=commit,source_root=str(repo),
          optimizer_work_authorized=False,
          cpu_receipt_sha256=sha(cpu),native_result_sha256=sha(native/'result.json'),
          attestation_sha256=sha(expected),plan_sha256=sha(a.plan),
          limitations=['native synthetic CE probe is not full PPO/GRPO training regression',
                      'matching memory settings does not establish full-length workload capacity']))
    # Native worker parity is not scientific RL completion-gating clearance.
    # Do not train until the additional integrated gates have an agreed API.


if __name__=='__main__': main()
