"""Actual fresh Gemma96-rollout multiworker RL update and separate restore."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.gemma4_methods.selection_adapter import file_identity,read_verified
from experiments.gemma4_methods.reference.plan import immutable_json
from experiments.gemma4_methods.native_method_probe import verify_context
from experiments.gemma4_methods.native_hooks import runtime_restore


def main(args):
    import types
    plan_record=file_identity(args.plan)
    plan=read_verified(plan_record)
    repo=Path(__file__).resolve().parents[2]
    model=plan['argv'][plan['argv'].index('--model')+1]
    cpu=verify_context(types.SimpleNamespace(repository=repo,commit=plan['incorporated_commit'],
        model=model,cpu_receipt=args.cpu_receipt))
    if os.path.abspath(plan['argv'][0])!=os.path.abspath(sys.executable):raise ValueError('Native interpreter mismatch')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    argv=list(plan['argv'])
    argv[1]=str(repo/'experiments/rmct_restart_20260928/native_rl_worker.py')
    argv[argv.index('--experiment-name')+1]='disposable-native-rl-gate'
    argv[argv.index('--run-name')+1]='one-update'
    immutable_json(output/'claim.json',{'plan':plan_record,'cpu_receipt':file_identity(args.cpu_receipt),
        'argv':argv,'job_id':os.environ['SLURM_JOB_ID'],'production_resume_forbidden':True})
    # The disposable worker uses its scheduled output cwd for subset evidence;
    # the canonical source is imported only through the reviewed PYTHONPATH.
    subprocess.run(argv,cwd=output,check=True)
    # train_rlct derives logs from cwd, not from the source checkout.
    checkpoint=output/'logs/disposable-native-rl-gate/one-update/checkpoints/disposable-native-rl-gate_one-update'
    if not checkpoint.exists():
        checkpoint=repo/'logs/disposable-native-rl-gate/one-update/checkpoints/disposable-native-rl-gate_one-update'
    files={str(p.relative_to(checkpoint)):file_identity(p) for p in checkpoint.rglob('*') if p.is_file()}
    manifest=read_verified(files['manifest.json'])
    loop=manifest['loop_state']
    if (manifest['kind'],loop['global_step'],loop['optimizer_step'],loop['final'],loop['accumulated_grads'])!=('both',1,1,True,0):
        raise ValueError('Actual one-update native RL boundary missing')
    import torch
    from safetensors.torch import load_file
    tensors=load_file(str(checkpoint/'adapter_model.safetensors'))
    b=[value for name,value in tensors.items() if 'lora_B' in name]
    optimizer=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
    averages=[value['exp_avg'] for value in optimizer['state'].values() if 'exp_avg' in value]
    if (not b or not averages or not all(torch.isfinite(x).all() for x in [*b,*averages])
        or not any(torch.count_nonzero(x)>0 for x in b)
        or not any(torch.count_nonzero(x)>0 for x in averages)):
        raise ValueError('Native RL update has no finite nonzero adapter/optimizer effect')
    progress={'checkpoint':str(checkpoint),'checkpoint_files':files}
    runtime_restore(progress,{'source_commit':plan['incorporated_commit'],'model':model,'method':'rmct'})
    immutable_json(output/'receipt.json',{'schema':'gemma-native-rmct-gate-v1','plan':plan_record,
        'cpu_receipt':file_identity(args.cpu_receipt),'job_id':os.environ['SLURM_JOB_ID'],
        'optimizer_updates':1,'production_resume_forbidden':True,'checkpoint_files':files,
        'nonzero_adapter_b_tensors':sum(int(torch.count_nonzero(x)>0) for x in b),
        'nonzero_optimizer_averages':sum(int(torch.count_nonzero(x)>0) for x in averages)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','cpu-receipt','output'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
