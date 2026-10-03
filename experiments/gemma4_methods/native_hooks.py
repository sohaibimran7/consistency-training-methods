"""Concrete five-method hooks: native receipts, live scheduler and restore.

No stub success flags and no RMCT progress-schema substitution.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

from experiments.gemma4_methods.selection_adapter import GemmaVerifiers,file_identity,read_verified
from experiments.gemma4_methods.launch_guard import check_source
from experiments.gemma4_methods import train


def scheduler_complete(receipt,contract):
    if not isinstance(receipt,dict) or set(receipt)!= {'job_id'}:
        raise ValueError('Explicit native scheduler job identity required')
    job=str(receipt['job_id'])
    if not job.replace('_','').isdigit():
        raise ValueError('Invalid scheduler job identity')
    result=subprocess.check_output(['sacct','-j',job,'-X','-n','-P','-o','JobIDRaw,State,ExitCode'],text=True)
    rows=[line.split('|') for line in result.splitlines() if line.strip()]
    matched=[row for row in rows if row[0]==job]
    if len(matched)!=1 or matched[0][1:3]!=['COMPLETED','0:0']:
        raise ValueError('Native scheduler completion not verified')
    return True


def runtime_restore(progress,contract):
    root=Path(__file__).resolve().parents[2]
    check_source(root,contract['source_commit'])
    with tempfile.TemporaryDirectory(prefix='gemma-native-restore-') as folder:
        output=Path(folder)/'restored.json'
        subprocess.run([sys.executable,'-B','-m','experiments.gemma4_methods.native_checkpoint_restore',
            '--repository',str(root),'--commit',contract['source_commit'],
            '--checkpoint',progress['checkpoint'],'--model',contract['model'],
            '--method',contract['method'],'--output',str(output)],check=True)
        restored=json.loads(output.read_text())
        if (restored['checkpoint_files']!=progress['checkpoint_files']
                or restored['model']!=contract['model'] or restored['method']!=contract['method']
                or restored['source_commit']!=contract['source_commit'] or restored['optimizer_updates']!=0):
            raise ValueError('Actual native restore evidence differs')
    return progress


def normalize(run_dir,contract,at_attempt=None):
    """Normalized progress at the latest checkpoint, or at encounter boundary ``at_attempt``.

    With ``at_attempt`` (training may have run ahead to the patience horizon),
    use the last update before that boundary; the attempts after it up to the
    boundary must all be immutable skip records.
    """
    from experiments.gemma4_methods.reference import train as helpers
    pointer=json.loads((run_dir/'state.json').read_text())
    if at_attempt is not None:
        logged=[json.loads(line) for line in (run_dir/'metrics.jsonl').read_text().splitlines()]
        before=[r for r in logged if r['attempt']<at_attempt]
        if not before or [r['step'] for r in before]!=list(range(1,len(before)+1)):
            raise ValueError('No contiguous saved update before the requested boundary')
        pointer=json.loads((run_dir/'receipts'/f'step-{len(before):06d}.json').read_text())
        if pointer['convergence']['attempts']!=before[-1]['attempt']+1:
            raise ValueError('Boundary receipt cursor differs from its update')
    state=pointer['convergence']
    step,attempt=state['step'],state['attempts']
    checkpoint=(run_dir/pointer['checkpoint']).resolve()
    if helpers.checkpoint_identity(checkpoint)!=pointer['checkpoint_files']:
        raise ValueError('Durable checkpoint pointer bytes differ')
    original=run_dir/'progress'/f'step-{step:06d}.json'
    if not original.is_file():
        raise ValueError('Original actual-update progress missing')
    rows=[json.loads(line) for line in (run_dir/'metrics.jsonl').read_text().splitlines()]
    # Immutable prefix snapshots exclude only a provably unsaved later tail.
    prefix=[r for r in rows if r['step']<=step]
    if [r['step'] for r in prefix]!=list(range(1,step+1)):
        raise ValueError('Metrics prefix does not match durable optimizer history')
    # Consumed batches after the last update (no weight change) still advance
    # the encounter cursor; each must have its immutable skip record.
    trailing=[]
    while ((at_attempt is None or attempt+len(trailing)<at_attempt)
           and (run_dir/'skips'/f'attempt-{attempt+len(trailing):07d}.json').is_file()):
        path=run_dir/'skips'/f'attempt-{attempt+len(trailing):07d}.json'
        row=json.loads(path.read_text())
        if row['attempt']!=attempt+len(trailing) or row['step']!=step or row.get('optimizer_update') is not False:
            raise ValueError('Trailing skip record disagrees with durable progress')
        trailing.append(file_identity(path))
    if at_attempt is not None and attempt+len(trailing)!=at_attempt:
        raise ValueError('Training has not yet consumed the requested boundary')
    evidence_dir=run_dir/'normalized'/f'step-{step:06d}-attempt-{attempt+len(trailing):07d}'
    evidence_dir.mkdir(parents=True,exist_ok=True)
    metrics=evidence_dir/'metrics.jsonl'
    payload=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in prefix)
    if metrics.exists():
        if metrics.read_text()!=payload:
            raise ValueError('Previously sealed metrics prefix differs')
    else:
        with metrics.open('x') as stream:
            stream.write(payload)
    skips=[]
    for path in sorted((run_dir/'skips').glob('attempt-*.json')):
        row=json.loads(path.read_text())
        if row['attempt']<attempt:
            skips.append(file_identity(path))
    from experiments.gemma4_methods.one_bias import QIDS_PER_UPDATE
    exposure={'metric_files':[file_identity(metrics)],'skip_files':skips}
    exposure_path=evidence_dir/'exposure.json'
    from experiments.gemma4_methods.reference.plan import immutable_json
    immutable_json(exposure_path,exposure)
    progress={'schema':'ctm-training-progress-v1',
        **{k:contract[k] for k in ('campaign_id','method','model','source_commit')},
        'actual_optimizer_step':step,'next_attempt_index':attempt,'sampled_batches':None,
        'encounter_attempt':attempt+len(trailing),'encountered_qid_bias_examples':QIDS_PER_UPDATE*(attempt+len(trailing)),
        'trailing_skip_files':trailing,
        'checkpoint':str(checkpoint),'checkpoint_files':{k:file_identity(checkpoint/k) for k in pointer['checkpoint_files']},
        'gemma_original_progress':file_identity(original),'gemma_exposure':file_identity(exposure_path)}
    immutable_json(evidence_dir/'progress.json',progress)
    return progress


def verify_start(start,contract,args):
    root=Path(__file__).resolve().parents[2]
    check_source(root,contract['source_commit'])
    if contract['method'] not in train.reference.METHODS:
        raise ValueError('Five-method native hook cannot substitute RMCT progress')
    if not contract.get('approval_reference'):
        raise ValueError('Scoped user approval reference required')
    run_dir=args.root.resolve()/'runs'/contract['method']
    if (run_dir/'state.json').exists() or any((run_dir/'checkpoints').glob('step-*')):
        raise ValueError('Fresh initialization cannot reuse a production checkpoint')
    frozen=json.loads((args.root/'contract.json').read_text())
    if start['training_contract']!=file_identity(args.root/'contract.json'):
        raise ValueError('Fresh training contract differs')
    train.verify(args.root.resolve())
    if frozen['contract']['prompt_mode'].get('enable_thinking') is not True:
        raise ValueError('Thinking-enabled training contract required')
    cpu=read_verified(start['cpu_receipt'])
    from experiments.gemma4_methods.native_method_probe import verify_context
    current=verify_context(types.SimpleNamespace(repository=str(root),commit=contract['source_commit'],
        model=contract['model'],cpu_receipt=start['cpu_receipt']['path']))
    if current!=cpu:
        raise ValueError('CPU evidence changed during bootstrap consumption')
    update=read_verified(start['native_update'])
    restored=read_verified(start['native_restore'])
    if (cpu['schema'],cpu['status'],cpu['source_commit']) != ('rmct-restart-cpu-v1','cpu_checks_passed',contract['source_commit']):
        raise ValueError('Current CPU/native source differs')
    if (update['schema'],restored['schema']) != ('gemma-native-method-update-v1','gemma-native-method-restore-v1'):
        raise ValueError('Concrete native method update/restore required')
    for receipt in (update,restored):
        if any(receipt[k]!=contract[k] for k in ('method','model','source_commit')):
            raise ValueError('Native method evidence lineage differs')
        scheduler_complete({'job_id':receipt['slurm_job_id']},contract)
    if restored['update']!=start['native_update'] or update['cpu_receipt']!=start['cpu_receipt']:
        raise ValueError('Native proof artifact binding differs')
    if update['production_optimizer_updates']!=0 or update['not_a_scientific_parent'] is not True:
        raise ValueError('Native proof must not reuse scientific training')
    if (len(update['losses'])!=4 or not all(math.isfinite(x) for x in update['losses'])
            or update['gradients']['positive_gradient_tensors']<=0
            or restored['optimizer_parameter_states']<=0 or restored['actual_adapter_tensors']<=0):
        raise ValueError('Concrete finite nonzero method/optimizer evidence missing')
    if update['implementation']!=file_identity(root/'experiments/gemma4_methods/native_method_probe.py'):
        raise ValueError('Native method probe implementation changed')
    if cpu['python']!=sys.executable or Path(cpu['source_root']).resolve()!=root:
        raise ValueError('Native CPU runtime/source differs')
    for item in cpu['sources']:
        actual=file_identity(item['path'])
        if any(actual[k]!=item[k] for k in ('sha256','bytes')):
            raise ValueError('Native CPU source bytes changed')
    from experiments.gemma4_methods.one_bias import QIDS_PER_UPDATE
    # First one-bias batch: one BCT target per QID; four OPCT rollouts per pair.
    expected_samples={'bct':QIDS_PER_UPDATE,'opct':4*QIDS_PER_UPDATE}.get(contract['method'],0)
    if len(update['samples'])!=expected_samples:
        raise ValueError('Native rollout sample coverage changed')
    if expected_samples and update['generation_cap_including_reasoning']!=20480:
        raise ValueError('Native rollout cap differs from approved training cap')
    for file in update['checkpoint_files'].values():
        read_bytes=file_identity(file['path'])
        if read_bytes!=file:
            raise ValueError('Native checkpoint bytes changed')
    data_order=read_verified(contract['data_order'])
    pool=train.ordered_pool(args.root.resolve())
    if [r['question_id'] for r in pool]!=data_order['ordered_qids']:
        raise ValueError('Fresh exact consumed-QID pool changed')
    return start


def create(args):
    from transformers import AutoProcessor
    from ctm.evals.local_model import Gemma4UnifiedTextProcessor
    if not args.selection_contract:
        raise ValueError('Explicit scientific contract required')
    contract=read_verified(file_identity(args.selection_contract))
    start_record=file_identity(os.environ['GEMMA_NATIVE_START_RECORD'])
    processor=Gemma4UnifiedTextProcessor(AutoProcessor.from_pretrained(contract['model'],local_files_only=True))
    adapter=GemmaVerifiers(processor=processor,verify_runtime_checkpoint=runtime_restore,
                          verify_scheduler=scheduler_complete)
    return types.SimpleNamespace(adapter=adapter,start_record=start_record,
        normalized_progress=normalize,verify_start=lambda start,c:verify_start(start,c,args))
