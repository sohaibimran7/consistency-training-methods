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
from experiments.rmct_restart_20260928.qwen_checkpoint import seal, seal_v2, identity, checkpoint_state
from experiments.rmct_restart_20260928.qwen_progress import progress, next_slice
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


def validate_parent(plan,index,parent,repo):
    """Revalidate the exact clean parent BEFORE a child can use its weights.

    Resealing checks required file paths/hashes, final four-rank metadata and
    strict optimizer/RNG resume state. The expected path comes from this plan,
    not from the untrusted saved parent receipt.
    """
    if index==0:
        if parent is not None:raise ValueError('Fresh segment cannot have a parent')
        return
    argv=plan['argv']
    campaign_id=argv[argv.index('--run-name')+1]
    experiment=argv[argv.index('--experiment-name')+1]
    if (not parent or parent.get('schema')!='rmct-clean-checkpoint-v1'
        or parent.get('campaign_id')!=campaign_id or parent.get('step')!=index*16):
        raise ValueError('Wrong clean parent identity before training')
    run_name=f'{campaign_id}-s{index:03d}'
    expected=Path(repo).resolve()/'logs'/experiment/run_name/'checkpoints'/f'{experiment}_{run_name}'
    if parent.get('checkpoint')!=str(expected):raise ValueError('Wrong parent path before training')
    grandparent=parent.get('parent')
    expected_command=command(plan,index-1,grandparent)
    if parent.get('command')!=expected_command:raise ValueError('Parent recipe differs from current clean plan')
    current=seal(expected,campaign_root=Path(repo).resolve()/'logs'/experiment,
                 campaign_id=campaign_id,step=index*16,command=expected_command,parent=grandparent)
    if current!=parent:raise ValueError('Parent seal or bytes changed before training')


def execute_segment(plan,index,parent,repo,env):
    validate_parent(plan,index,parent,repo)
    subprocess.run(command(plan,index,parent),cwd=repo,env=env,check=True)


def plan_binding(plan, plan_path):
    if read(plan_path) != plan:
        raise ValueError('Plan changed in memory or on disk')
    return dict(plan=identity(plan_path), source_commit=plan['incorporated_commit'],
                source_root=str(Path(plan['argv'][1]).resolve().parents[1]))


def command_v2(plan, before, selection, parent):
    argv = list(plan['argv'])
    if any(x.startswith('--resume') for x in argv):
        raise ValueError('Base recipe must not contain resume arguments')
    for flag, value in (('--batch-size', '2'), ('--n-epochs', '1'), ('--gradient-accumulation-steps', '1')):
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != value:
            raise ValueError('Counter slicing requires the unchanged two-QID, one-epoch recipe')
    if '--no-shuffle-datapoints' not in argv:
        raise ValueError('Frozen order required')
    if before != progress(before['sampled_batches'], before['optimizer_updates'],
                          no_progress_batches=before['no_progress_batches']):
        raise ValueError('Inconsistent cursor')
    if (selection['segment_index'], selection['batch_offset']) != (before['segment_index'], before['batch_offset']):
        raise ValueError('Slice replays or skips consumed data')
    count = selection['batch_count']
    if type(count) is not int or not 1 <= count <= 16 - before['batch_offset']:
        raise ValueError('Invalid batch slice')
    if parent is None and before != progress(0, 0):
        raise ValueError('Missing parent')
    if parent is not None and parent['progress'] != before:
        raise ValueError('Parent cursor differs')
    base = argv[argv.index('--run-name') + 1]
    argv[argv.index('--run-name') + 1] = f"{base}-b{before['sampled_batches']:06d}"
    load = json.loads(argv[argv.index('--load-config') + 1])
    if load.get('n_datapoints') != 32 or load.get('cycle_segments') is not True:
        raise ValueError('Frozen full-segment contract required')
    load.update(selection)
    argv[argv.index('--load-config') + 1] = json.dumps(load, sort_keys=True)
    if parent is not None:
        argv += ['--resume-from', 'file://' + parent['checkpoint'], '--resume-with-optimizer', '--resume-state-required']
    return argv


def checkpoint_path(argv, repo):
    name = argv[argv.index('--run-name') + 1]
    experiment = argv[argv.index('--experiment-name') + 1]
    return Path(repo) / 'logs' / experiment / name / 'checkpoints' / f'{experiment}_{name}'


def checked_record(record):
    if identity(record['path']) != record:
        raise ValueError('Evidence identity changed')
    return read(record['path'])


def verify_data_slice(argv, selection):
    from ctm_data.adapters.mcq_bias.shared_qid_two_bias import SharedQidTwoBiasSetting
    config=json.loads(argv[argv.index('--setting-config')+1])
    load=json.loads(argv[argv.index('--load-config')+1])
    setting=SharedQidTwoBiasSetting(**config)
    selected=setting.load_datapoints(**load)
    full_load=dict(load);full_load.pop('batch_offset',None);full_load.pop('batch_count',None)
    full=setting.load_datapoints(**full_load)
    start=2*selection['batch_offset'];end=start+2*selection['batch_count']
    if len(full)!=32 or selected!=full[start:end]:
        raise ValueError('Runtime loader does not implement the exact frozen batch slice')
    return [row['question_id'] for row in selected]


def recover_v2(plan, binding):
    """Explicit, hash-pinned migration of the failed first child only.

    This reads old bytes; it does not modify the failed source/checkpoint,
    manufacture a successful old seal, or start a process.
    """
    contract_record = plan['recovery_contract']
    contract = checked_record(contract_record)
    if (contract.get('schema') != 'rmct-first-child-recovery-contract-v2'
            or contract.get('destination_source_commit') != binding['source_commit']):
        raise ValueError('Wrong recovery contract/source')
    original = checked_record(contract['origin_plan'])
    started = checked_record(contract['origin_started'])
    audit = checked_record(contract['audit'])
    old_repo = Path(original['argv'][1]).resolve().parents[1]
    source_check(old_repo, original['incorporated_commit'])
    # Source and new preflight paths may change. Scientific recipe and runtime
    # executable/prefix must not; no broader argv normalization is allowed.
    def recipe(p):
        a = list(p['argv']); a[1] = '<source>/scripts/train_rlct.py'
        flag = '--local-qwen35-rollout-parity-attestation'
        if a.count(flag) != 1:
            raise ValueError('Missing exact runtime attestation flag')
        a[a.index(flag) + 1] = '<fresh-attestation>'
        return a
    if recipe(original) != recipe(plan) or original['python_prefix'] != plan['python_prefix']:
        raise ValueError('Recovery changes the training recipe or Python runtime')
    expected_command = command(original, 0, None)
    if (started.get('argv') != expected_command or started.get('plan_sha256') != contract['origin_plan']['sha256']
            or started.get('gates', {}).get('source_commit') != original['incorporated_commit']):
        raise ValueError('Original start receipt is not bound to the original plan/source')
    checkpoint = checkpoint_path(expected_command, old_repo)
    if str(checkpoint) != contract['checkpoint'] or audit.get('checkpoint') != str(checkpoint):
        raise ValueError('Wrong recovered checkpoint path')
    before = progress(0, 0)
    selection = dict(segment_index=0, batch_offset=0, batch_count=16)
    after, files = checkpoint_state(checkpoint, before, selection)
    if after != progress(16, 12):
        raise ValueError('This recovery only covers the audited 16-batch/12-update child')
    for name, record in files.items():
        if audit['files'].get(name) != {k: record[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Checkpoint does not match saved failed-job evidence')
    manifest = read(checkpoint / 'manifest.json')
    loop = dict(manifest['loop_state']); rng = loop.pop('runtime_rng')
    if loop != audit['loop']:
        raise ValueError('Audit counters differ from checkpoint')
    import hashlib
    # Audit hashes the sorted JSON representation, not a restored vLLM RNG.
    if sorted(rng) != sorted(audit['runtime_rng_keys']):
        raise ValueError('Audited RNG fields differ')
    if hashlib.sha256(json.dumps(rng, sort_keys=True).encode()).hexdigest() != audit['runtime_rng_sha256']:
        raise ValueError('Audited RNG bytes differ')
    metrics = [r for r in audit['metrics'] if 'train/optimizer_step' in r]
    if len(metrics) != 16 or [r['step'] for r in metrics] != list(range(1, 17)):
        raise ValueError('Incomplete first-child batch history')
    previous = 0
    for row in metrics:
        delta = row['train/optimizer_step'] - previous
        if delta not in (0, 1) or bool(row['train/skipped_empty_batch']) != (delta == 0):
            raise ValueError('Inconsistent optimizer/skip history')
        previous = row['train/optimizer_step']
    if previous != 12 or audit.get('scheduler_state') != 'FAILED':
        raise ValueError('Not the audited failed child')
    # Verify the entire artifact and derive consumed IDs from the manifest's
    # permutations, never the raw JSONL first 32 rows.
    from ctm_data.adapters.mcq_bias.shared_qid_two_bias import SharedQidTwoBiasSetting
    setting = SharedQidTwoBiasSetting(**json.loads(original['argv'][original['argv'].index('--setting-config') + 1]))
    if str(setting.data_path) != audit['data_path'] or sha(setting.data_path) != audit['data_sha256']:
        raise ValueError('Original training data changed')
    rows = setting.load_datapoints(**json.loads(original['argv'][original['argv'].index('--load-config') + 1]))
    ids = [r['question_id'] for r in rows]
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError('Wrong original consumed population')
    campaign = plan['argv'][plan['argv'].index('--run-name') + 1]
    if contract.get('campaign_id') != campaign:
        raise ValueError('Wrong recovery campaign')
    return dict(schema='rmct-clean-recovery-v2', campaign_id=campaign, step=12, progress=after,
                continuation_mode='optimizer_data_segment', bitwise_rollout_continuation=False,
                checkpoint=str(checkpoint), files=files, binding=binding, parent=None,
                recovery_contract=contract_record, command=expected_command, consumed_question_ids=ids,
                original_source_commit=original['incorporated_commit'])


def verify_seal_v2(receipt, plan, binding):
    """Reproduce a full v2 lineage against pinned plan and checkpoint bytes.

    Returns the verified receipt, or raises before any child can be launched.
    The recovery root is accepted only through plan.recovery_contract.
    """
    if plan_binding(plan, binding['plan']['path']) != binding:
        raise ValueError('Plan/source binding mismatch')
    chain = []; node = receipt
    while node is not None:
        if node.get('binding') != binding:
            raise ValueError('Unmigrated source or plan boundary')
        chain.append(node); node = node.get('parent')
    parent = None
    for saved in reversed(chain):
        if saved.get('schema') == 'rmct-clean-recovery-v2':
            if parent is not None:
                raise ValueError('Recovery must be the lineage root')
            current = recover_v2(plan, binding)
        elif saved.get('schema') == 'rmct-clean-checkpoint-v2':
            before = parent['progress'] if parent else progress(0, 0)
            argv = command_v2(plan, before, saved['selection'], parent)
            current = seal_v2(checkpoint_path(argv, binding['source_root']), campaign_id=saved['campaign_id'],
                before=before, selection=saved['selection'], command=argv, binding=binding, parent=parent)
            if saved['campaign_id'] != plan['argv'][plan['argv'].index('--run-name') + 1]:
                raise ValueError('Wrong campaign')
        else:
            raise ValueError('Unknown checkpoint schema; no implicit legacy migration')
        if current != saved:
            raise ValueError('Checkpoint seal, recipe or lineage changed')
        parent = current
    if parent is None:
        raise ValueError('Empty checkpoint lineage')
    return parent


def train_to_boundary(plan, binding, root, target, checked, parent, env):
    """Each child consumes new data and cannot overshoot real update target."""
    repo = Path(binding['source_root'])
    campaign_id = plan['argv'][plan['argv'].index('--run-name') + 1]
    while True:
        if parent is not None:
            verify_seal_v2(parent, plan, binding)
        before = parent['progress'] if parent else progress(0, 0)
        selection = next_slice(before, target)
        if selection is None:
            return parent
        argv = command_v2(plan, before, selection, parent)
        selected_ids=verify_data_slice(argv,selection)
        checkpoint = checkpoint_path(argv, repo)
        if checkpoint.parent.parent.exists():
            raise ValueError('Uncertain/existing production run; no replay')
        folder = root / 'batches' / str(before['sampled_batches'])
        # Exclusive start receipt makes any uncertain child non-replayable.
        write(folder / 'started.json', dict(argv=argv, binding=binding, gates=checked,
              before=before, selection=selection, target=target, consumed_question_ids=selected_ids))
        subprocess.run(argv, cwd=repo, env=env, check=True)
        parent = seal_v2(checkpoint, campaign_id=campaign_id, before=before, selection=selection,
                         command=argv, binding=binding, parent=parent)
        write(folder / 'complete.json', parent)


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
    binding=plan_binding(plan,args.plan)
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
    parent=None
    if args.target>64:
        previous=read(root/f'window-{args.target-64}'/'training-complete.json')
        if previous['target']!=args.target-64 or previous.get('binding')!=binding:
            raise ValueError('Previous window binding mismatch')
        parent=verify_seal_v2(previous['checkpoint'],plan,binding)
        if parent['step']!=args.target-64:
            raise ValueError('Previous window not at exact optimizer boundary')
    elif 'recovery_contract' in plan:
        parent=recover_v2(plan,binding)
        # The old job must be terminal; an audit alone is not a live lock.
        contract=checked_record(plan['recovery_contract'])
        audit=checked_record(contract['audit'])
        state=subprocess.check_output(['sacct','-X','-j',str(audit['job']),'--noheader','--format=State,ExitCode','-P'],text=True).strip()
        if state!='FAILED|1:0':raise ValueError('Original failed job is not confirmed terminal')
    window=root/f'window-{args.target}'
    write(window/'attempt.json',dict(job_id=os.environ['SLURM_JOB_ID'],plan_sha256=sha(args.plan),gates=checked))
    env=dict(os.environ,PYTHONPATH=str(repo),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONHOME',None)
    if parent is not None:write(window/'parent.json',parent)
    parent=train_to_boundary(plan,binding,root,args.target,checked,parent,env)
    write(window/'training-complete.json',dict(target=args.target,checkpoint=parent,
          validation_required_before_advancement=True,gates=checked,binding=binding))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','campaign','preflight','rl-gate','regression','validation-manifest'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--target',type=int,required=True)
    run(p.parse_args())
