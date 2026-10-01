"""Fresh native Gemma RMCT controller; no Qwen checkpoint-schema substitution.

Preserves two-QID groups,96rollouts/stage and cyclic1000QID order. Durable
sampled-batch and optimizer counters are separate. Validation owns stopping.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.gemma4_methods.selection_adapter import GemmaVerifiers,file_identity,read_verified
from experiments.gemma4_methods.native_hooks import runtime_restore,scheduler_complete
from experiments.gemma4_methods.launch_guard import check_source
from experiments.gemma4_methods.reference.plan import immutable_json
from experiments.rmct_restart_20260928.qwen_progress import progress,next_slice,advance,validate_loop


def command(plan,before,selection,parent):
    argv=list(plan['argv'])
    if any(x.startswith('--resume') for x in argv):raise ValueError('Fresh recipe required')
    if before!=progress(before['sampled_batches'],before['optimizer_updates'],
            no_progress_batches=before['no_progress_batches']):raise ValueError('Invalid counters')
    if (selection['segment_index'],selection['batch_offset'])!=(before['segment_index'],before['batch_offset']):
        raise ValueError('Slice cursor mismatch')
    count=selection['batch_count']
    if type(count) is not int or not 1<=count<=16-before['batch_offset']:
        raise ValueError('Whole two-QID groups required')
    for flag,value in (('--batch-size','2'),('--n-epochs','1'),('--gradient-accumulation-steps','1'),
                       ('--max-new-tokens','20480')):
        if argv.count(flag)!=1 or argv[argv.index(flag)+1]!=value:
            raise ValueError('Scientific recipe changed:'+flag)
    if '--no-shuffle-datapoints' not in argv:raise ValueError('Frozen order required')
    base=argv[argv.index('--run-name')+1]
    argv[argv.index('--run-name')+1]=f"{base}-b{before['sampled_batches']:06d}"
    # Gemma's ContinuingSharedQidSetting expects an ABSOLUTE attempted cursor,
    # unlike the Qwen segment-relative slice loader. Never interchange them.
    load={'n_datapoints':2*count,'segment_index':before['segment_index'],
          'batch_offset':before['sampled_batches']}
    argv[argv.index('--load-config')+1]=json.dumps(load,sort_keys=True)
    argv[argv.index('--n-datapoints')+1]=str(2*count)
    if parent is None:
        if before!=progress(0,0):raise ValueError('Missing native parent')
    else:
        if parent['progress']!=before:raise ValueError('Parent counters differ')
        for record in parent['files'].values():read_verified_bytes(record)
        argv+=['--resume-from','file://'+parent['checkpoint'],'--resume-with-optimizer','--resume-state-required']
    return argv


def read_verified_bytes(record):
    if file_identity(record['path'])!=record:raise ValueError('Checkpoint bytes changed')


def seal(checkpoint,plan_record,before,selection,argv,parent):
    from ctm.training.resume_state import load_strict_local_rl_resume_state
    checkpoint=Path(checkpoint).resolve()
    files={str(p.relative_to(checkpoint)):file_identity(p) for p in checkpoint.rglob('*') if p.is_file()}
    if not {'adapter_config.json','adapter_model.safetensors','optimizer.pt','manifest.json'}<=set(files):
        raise ValueError('Incomplete native Gemma checkpoint')
    manifest=read_verified(files['manifest.json'])
    if manifest['backend']!='local' or manifest['kind']!='both':raise ValueError('Native both checkpoint required')
    loop=manifest['loop_state']
    after=advance(before,selection,loop['optimizer_step'])
    validate_loop(loop,after)
    if loop['segment_start_global_step']!=before['sampled_batches'] or loop['segment_step']!=selection['batch_count']:
        raise ValueError('Native checkpoint consumed wrong batch slice')
    state=load_strict_local_rl_resume_state(checkpoint)
    if (state.global_step,state.optimizer_step)!=(after['sampled_batches'],after['optimizer_updates']):
        raise ValueError('Strict native resume counters disagree')
    plan=read_verified(plan_record)
    if manifest['model']!=plan['argv'][plan['argv'].index('--model')+1]:raise ValueError('Native model differs')
    if argv!=command(plan,before,selection,parent):raise ValueError('Native command differs')
    return {'schema':'gemma-rmct-native-checkpoint-v1','plan':plan_record,'progress':after,
        'before':before,'selection':selection,'command':argv,'checkpoint':str(checkpoint),'files':files,
        'parent':parent,'continuation_mode':'optimizer_data_segment','bitwise_rollout_continuation':False}


def verify_seal(receipt):
    if receipt['schema']!='gemma-rmct-native-checkpoint-v1':raise ValueError('Native Gemma seal required')
    for item in receipt['files'].values():read_verified_bytes(item)
    parent=receipt['parent']
    if parent is not None:verify_seal(parent)
    if seal(receipt['checkpoint'],receipt['plan'],receipt['before'],receipt['selection'],receipt['command'],parent)!=receipt:
        raise ValueError('Native seal differs on replay')
    return receipt


def normalized(receipt,contract):
    verify_seal(receipt)
    state=receipt['progress']
    return {'schema':'ctm-training-progress-v1',
        **{k:contract[k] for k in ('campaign_id','method','model','source_commit')},
        'actual_optimizer_step':state['optimizer_updates'],'next_attempt_index':state['sampled_batches'],
        'sampled_batches':state['sampled_batches'],'checkpoint':receipt['checkpoint'],
        'checkpoint_files':receipt['files'],'rmct_native_seal':receipt}


class RMCTVerifiers(GemmaVerifiers):
    def __init__(self,*,processor):
        super().__init__(processor=processor,verify_runtime_checkpoint=runtime_restore,
                         verify_scheduler=scheduler_complete)

    def verify_checkpoint(self,value,contract):
        from experiments.rmct_restart_20260928.validation_selection import check_progress
        check_progress(value,contract)
        if contract['method']!='rmct':raise ValueError('RMCT native controller only')
        receipt=verify_seal(value['rmct_native_seal'])
        plan=read_verified(receipt['plan'])
        if plan['incorporated_commit']!=contract['source_commit']:
            raise ValueError('Native checkpoint/source selection differs')
        if normalized(receipt,contract)!=value:raise ValueError('Normalized native counters differ')
        return self.runtime(value,contract)


def verify_start(start,contract):
    from experiments.gemma4_methods.native_method_probe import verify_context
    import types
    repo=Path(__file__).resolve().parents[2]
    check_source(repo,contract['source_commit'])
    if contract['method']!='rmct' or not contract.get('approval_reference'):
        raise ValueError('Scoped fresh RMCT approval required')
    plan=read_verified(start['native_plan'])
    if plan['incorporated_commit']!=contract['source_commit'] or plan['initial_optimizer_step']!=0:
        raise ValueError('Fresh same-source plan required')
    verify_context(types.SimpleNamespace(repository=repo,commit=contract['source_commit'],
        model=contract['model'],cpu_receipt=start['cpu_receipt']['path']))
    gate=read_verified(start['native_rl_gate'])
    if (gate['schema']!='gemma-native-rmct-gate-v1' or gate['plan']!=start['native_plan']
        or gate['cpu_receipt']!=start['cpu_receipt'] or gate['optimizer_updates']!=1
        or gate['production_resume_forbidden'] is not True
        or gate['nonzero_adapter_b_tensors']<=0 or gate['nonzero_optimizer_averages']<=0):
        raise ValueError('Native RL gate incomplete')
    scheduler_complete({'job_id':gate['job_id']},contract)
    for item in gate['checkpoint_files'].values():read_verified_bytes(item)
    parity=read_verified(start['native_parity'])
    if (parity.get('status')!='passed'
        or parity['provenance']['source_commit']!=contract['source_commit']
        or parity['provenance']['cpu_receipt_sha256']!=start['cpu_receipt']['sha256']):
        raise ValueError('Fresh HF-vLLM parity missing')
    scheduler_complete({'job_id':start['parity_job']},contract)
    return start


def train_window(args):
    from transformers import AutoProcessor
    from ctm.evals.local_model import Gemma4UnifiedTextProcessor
    from experiments.rmct_restart_20260928.validation_selection import bootstrap_budget,continuation_budget
    repo=Path(__file__).resolve().parents[2]
    contract_record=file_identity(args.contract)
    contract=read_verified(contract_record)
    check_source(repo,contract['source_commit'])
    plan_record=file_identity(args.plan)
    plan=read_verified(plan_record)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        processor=Gemma4UnifiedTextProcessor(AutoProcessor.from_pretrained(contract['model'],local_files_only=True))
        adapter=RMCTVerifiers(processor=processor)
        pointer=output/'latest.json'
        parent=read_verified(file_identity(pointer)) if pointer.exists() else None
        if parent is None:
            budget=bootstrap_budget(contract_record,file_identity(args.start),requested_updates=16,verify_start=verify_start)
            before=progress(0,0)
        else:
            value=normalized(parent,contract)
            adapter.verify_checkpoint(value,contract)
            budget=continuation_budget(value,adapter.replay(args.folder,contract_record),requested_updates=16)
            before=parent['progress']
        if not budget:return
        target=before['optimizer_updates']+budget
        while before['optimizer_updates']<target:
            # next_slice bounds skips to one full pool and never crosses64.
            boundary=((before['optimizer_updates']//64)+1)*64
            selection=next_slice(before,boundary)
            selection['batch_count']=min(selection['batch_count'],target-before['optimizer_updates'])
            argv=command(plan,before,selection,parent)
            config=json.loads(argv[argv.index('--setting-config')+1])
            from experiments.rmct_restart_20260928.gemma_production_setting import create_setting
            datapoints=create_setting(**config).load_datapoints(**json.loads(argv[argv.index('--load-config')+1]))
            if len(datapoints)!=selection['batch_count']*2:raise ValueError('Native slice size differs')
            name=argv[argv.index('--run-name')+1]
            claim=output/'claims'/f'{name}.json'
            immutable_json(claim,{'job_id':os.environ['SLURM_JOB_ID'],'argv':argv,
                'question_ids':[r['question_id'] for r in datapoints]})
            experiment=argv[argv.index('--experiment-name')+1]
            checkpoint=repo/'logs'/experiment/name/'checkpoints'/f'{experiment}_{name}'
            if checkpoint.exists():raise ValueError('Existing unsealed child; explicit recovery required')
            subprocess.run(argv,cwd=repo,check=True)
            parent=seal(checkpoint,plan_record,before,selection,argv,parent)
            immutable_json(output/'receipts'/f'{name}.json',parent)
            from experiments.gemma4_methods.reference.train import atomic_json
            atomic_json(pointer,parent)
            value=normalized(parent,contract)
            immutable_json(output/'progress'/f"step-{value['actual_optimizer_step']:06d}-attempt-{value['next_attempt_index']:06d}.json",value)
            before=parent['progress']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('contract','plan','start','folder','output'):p.add_argument('--'+name,type=Path,required=True)
    train_window(p.parse_args())
