"""Train at most one 64-update clean window after same-commit gates.

No submission, automatic retries, historical resumes or automatic chaining.
The disposable native-RL checkpoint is never used as a production parent.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from experiments.rmct_restart_20260928.qwen_launch import source_check,sha
from experiments.rmct_restart_20260928.qwen_checkpoint import seal
from experiments.rmct_restart_20260928.qwen_validation import replay,verify_saved
from experiments.rmct_restart_20260928.qwen_validation_executor import read,write


def gates(plan,plan_path,preflight,rl_gate,regression,repo):
    commit=plan['incorporated_commit'];source_check(repo,commit)
    # This is an actual repository ancestry check, not a manually authored
    # incorporation boolean. Deployment must fetch origin/main explicitly.
    subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/main'],cwd=repo,check=True)
    main=subprocess.check_output(['git','rev-parse','origin/main'],cwd=repo,text=True).strip()
    p=read(preflight);r=read(rl_gate)
    if p['source_commit']!=commit or p['plan_sha256']!=sha(plan_path):raise ValueError('Preflight binding mismatch')
    if (r.get('schema')!='rmct-native-rl-gate-v1' or r.get('status')!='passed'
        or r.get('source_commit')!=commit or r.get('plan_sha256')!=sha(plan_path)
        or r.get('preflight_sha256')!=sha(preflight) or r.get('optimizer_steps')!=1
        or r.get('production_resume_forbidden') is not True):
        raise ValueError('Native RL gate incomplete or differently bound')
    from experiments.rmct_restart_20260928.regression_gate import TESTS
    g=read(regression)
    if (g.get('schema')!='rmct-integrated-regression-v1' or g.get('status')!='passed'
        or g.get('source_commit')!=commit or g.get('source_root')!=str(repo)
        or g.get('python')!=sys.executable or g.get('sys_prefix')!=sys.prefix
        or g.get('optimizer_work_authorized') is not False):raise ValueError('Wrong integrated regression receipt')
    if g.get('tests')!={name:sha(repo/name) for name in TESTS}:raise ValueError('Regression source changed')
    if g.get('pytest_log_sha256')!=sha(Path(regression).parent/'pytest.txt'):raise ValueError('Regression log changed')
    for name,key in [('cpu.json','cpu_receipt_sha256'),('native/result.json','native_result_sha256'),
                     ('native/qwen35-rollout-worker-parity-attestation.json','attestation_sha256')]:
        if sha(Path(preflight).parent/name)!=p[key]:raise ValueError('Native preflight evidence changed')
    return dict(source_commit=commit,origin_main=main,preflight_sha256=sha(preflight),
                native_rl_sha256=sha(rl_gate),regression_sha256=sha(regression))


def command(plan,index,parent):
    argv=list(plan['argv'])
    if any(x.startswith('--resume') for x in argv):raise ValueError('Initial plan must be fresh')
    if index==0 and parent is not None:raise ValueError('Fresh run cannot have a parent')
    if index>0 and (not parent or parent['step']!=index*16):raise ValueError('Missing previous clean segment')
    base=argv[argv.index('--run-name')+1]
    argv[argv.index('--run-name')+1]=f'{base}-s{index+1:03d}'
    load=json.loads(argv[argv.index('--load-config')+1]);load['segment_index']=index
    argv[argv.index('--load-config')+1]=json.dumps(load,sort_keys=True)
    if parent:
        argv+=['--resume-from','file://'+parent['checkpoint'],'--resume-with-optimizer','--resume-state-required']
    return argv


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Scheduled allocation required')
    plan=read(args.plan);repo=Path(plan['argv'][1]).resolve().parents[1]
    if Path(__file__).resolve()!=repo/'experiments/rmct_restart_20260928/qwen_train_window.py':raise ValueError('Noncanonical controller')
    if os.path.abspath(sys.executable)!=os.path.abspath(plan['argv'][0]) or sys.prefix!=plan['python_prefix']:
        raise ValueError('Wrong Python environment')
    if args.target<64 or args.target%64:raise ValueError('64-update boundaries required')
    root=args.campaign.resolve()
    if root.is_relative_to(repo):raise ValueError('Evidence must live outside source')
    campaign_id=plan['argv'][plan['argv'].index('--run-name')+1]
    checked=gates(plan,args.plan,args.preflight,args.rl_gate,args.regression,repo)
    history=[]
    if args.target>64:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(plan['argv'][plan['argv'].index('--model')+1],local_files_only=True)
        for step in range(64,args.target,64):
            folder=root/'validation'/str(step)
            verify_saved(folder,args.validation_manifest,tokenizer)
            verified=read(folder/'verified-complete.json')
            if verified['score_sha256']!=sha(folder/'score.json') or verified['decision_sha256']!=sha(folder/'decision.json'):
                raise ValueError('Validation final seal changed')
            row=read(folder/'score.json')
            job=read(folder/'completed-job.json')['job_id']
            state=subprocess.check_output(['sacct','-X','-j',str(job),'--noheader','--format=State,ExitCode','-P'],text=True).strip()
            if state!='COMPLETED|0:0':raise ValueError('Validation scheduler completion unverified')
            history.append(row)
        if replay(history,campaign_id=campaign_id)['stopped']:raise ValueError('Patience reached')
    window=root/f'window-{args.target}'
    write(window/'attempt.json',dict(job_id=os.environ['SLURM_JOB_ID'],plan_sha256=sha(args.plan),gates=checked))
    env=dict(os.environ,PYTHONPATH=str(repo),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONHOME',None)
    parent=None
    first=(args.target-64)//16
    if first:parent=read(root/'segments'/str(first-1)/'complete.json')
    for index in range(first,args.target//16):
        argv=command(plan,index,parent)
        run_name=argv[argv.index('--run-name')+1];experiment=argv[argv.index('--experiment-name')+1]
        run_dir=repo/'logs'/experiment/run_name
        if run_dir.exists():raise ValueError('Uncertain/existing production run; no replay')
        folder=root/'segments'/str(index)
        write(folder/'started.json',dict(argv=argv,plan_sha256=sha(args.plan),gates=checked))
        subprocess.run(argv,cwd=repo,env=env,check=True)
        checkpoint=run_dir/'checkpoints'/f'{experiment}_{run_name}'
        parent=seal(checkpoint,campaign_root=repo/'logs'/experiment,campaign_id=campaign_id,
                    step=(index+1)*16,command=argv,parent=parent)
        write(folder/'complete.json',parent)
    write(window/'training-complete.json',dict(target=args.target,checkpoint=parent,
          validation_required_before_advancement=True,gates=checked))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','campaign','preflight','rl-gate','regression','validation-manifest'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--target',type=int,required=True)
    run(p.parse_args())
