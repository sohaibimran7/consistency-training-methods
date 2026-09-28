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


def cpu_gate_check(receipt, repo, commit):
    # Explicit integration contract, not an inferred status from file existence.
    if receipt.get('schema')!='rmct-restart-cpu-v1' or receipt.get('status')!='cpu_checks_passed':
        raise RuntimeError('CPU gate receipt not passed')
    if receipt.get('optimizer_work_authorized') is not False:
        raise RuntimeError('CPU receipt must not authorize optimizer work')
    if receipt.get('source_commit')!=commit:
        raise RuntimeError('CPU gate source identity mismatch')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation-manifest',type=Path,required=True)
    p.add_argument('--cpu-module',required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Run inside scheduled allocation')
    plan=json.loads(a.plan.read_text()); argv=plan['argv']
    repo=Path(argv[1]).resolve().parents[1]; commit=plan['incorporated_commit']
    if Path(sys.executable).resolve()!=Path(argv[0]).resolve():
        raise RuntimeError('Launcher Python is not planned Python')
    if any(x.startswith('--resume') for x in argv): raise RuntimeError('Fresh-start only')
    if json.loads(argv[argv.index('--load-config')+1])['segment_index']!=0:
        raise RuntimeError('Fresh segment0 required')
    if argv[argv.index('--max-new-tokens')+1]!='20480': raise RuntimeError('Cap mismatch')
    source_check(repo,commit)
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(repo),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONHOME',None)
    model=argv[argv.index('--model')+1]
    native=out/'native'
    expected=native/'qwen35-rollout-worker-parity-attestation.json'
    if Path(argv[argv.index('--local-qwen35-rollout-parity-attestation')+1]).resolve()!=expected:
        raise RuntimeError('Plan must bind THIS fresh native attestation')
    write(out/'started.json',dict(job_id=os.environ['SLURM_JOB_ID'],source_commit=commit,
          source_root=str(repo),python=sys.executable,plan_sha256=sha(a.plan),
          validation_manifest_sha256=sha(a.validation_manifest),training_requested=False))
    cpu=out/'cpu.json'
    subprocess.run([sys.executable,'-m',a.cpu_module,'--source-root',str(repo),
        '--source-commit',commit,'--model',model,'--family','qwen',
        '--validation-manifest',str(a.validation_manifest.resolve()),'--output',str(cpu)],
        cwd=repo,env=env,check=True)
    cpu_gate_check(json.loads(cpu.read_text()),repo,commit)
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
