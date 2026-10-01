"""Disposable native method update and separate-process optimizer restore gate.

Uses two original training QIDs. Diagnostic adapter/optimizer files are NEVER
scientific parents. No change to production rollout counts or generation caps.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

from experiments.gemma4_methods import train
from experiments.gemma4_methods.reference import plan,train as helpers
from experiments.gemma4_methods.selection_adapter import file_identity,native_prompt
from experiments.gemma4_methods.launch_guard import check_source


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def verify_context(args):
    root=check_source(args.repository,args.commit)
    cpu=json.loads(Path(args.cpu_receipt).read_text())
    if (cpu.get('schema'),cpu.get('status'),cpu.get('source_commit')) != (
            'rmct-restart-cpu-v1','cpu_checks_passed',args.commit):
        raise ValueError('Current shared CPU gate required')
    if Path(cpu['source_root']).resolve()!=root or Path(cpu['python']).resolve()!=Path(sys.executable).resolve():
        raise ValueError('CPU source/interpreter changed')
    for item in cpu['sources']+cpu['model_files']:
        actual=file_identity(item['path'])
        if any(actual[k]!=item[k] for k in ('sha256','bytes')):
            raise ValueError('CPU-attested source/model bytes changed')
    for name,version in cpu['dependencies'].items():
        if importlib.metadata.version(name)!=version:
            raise ValueError('CPU-attested dependency changed')
    if Path(args.model).name!=train.REVISION:
        raise ValueError('Pinned original Gemma required')
    weights=[x for x in cpu['model_files'] if Path(x['path']).name=='model.safetensors']
    if len(weights)!=1 or Path(weights[0]['path']).resolve()!=(Path(args.model)/'model.safetensors').resolve():
        raise ValueError('Requested model differs from CPU evidence')
    for module in (sys.modules[__name__],train,plan,helpers):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError('Method probe imported a foreign implementation')
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Native GPU gate must run in a scheduled allocation')
    return cpu


def same_state(left,right):
    """Exact actual tensor/state equality, not a saved success flag."""
    import torch
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(same_state(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):
        return type(left)==type(right) and len(left)==len(right) and all(same_state(a,b) for a,b in zip(left,right))
    return type(left)==type(right) and left==right


async def run(args):
    cpu=verify_context(args)
    import torch
    from transformers import AutoModelForImageTextToText,AutoProcessor
    from ctm.backends.local.engine import LocalBackend
    from ctm.backends.local.rollout_workers import RolloutParallelBackend,resolve_rollout_gpus
    from ctm.backends.renderers import HuggingFaceChatTemplateRenderer
    from ctm.core.config import LoRAConfig,AdamConfig
    from ctm.training.consistency_data import build_consistency_datums_with_audit,require_full_reference_suffix_alignment
    from ctm.training.sft import METHOD_LOSS_FNS
    from experiments.gemma4_methods.checkpoint import RNGCheckpointBackend,restore_coordinator_rng
    if args.stage=='update':
        args.output.mkdir(parents=True,exist_ok=False)
    else:
        previous=json.loads((args.output/'update.json').read_text())
        for item in previous['checkpoint_files'].values():
            if file_identity(item['path'])!=item:
                raise ValueError('Disposable checkpoint bytes changed')
        if previous['method']!=args.method or previous['source_commit']!=args.commit:
            raise ValueError('Restore gate lineage mismatch')
    online=args.method in ('bct','opct')
    torch.manual_seed(42)
    model=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.bfloat16,
        local_files_only=True,**({'attn_implementation':'eager'} if not online else {}))
    targets=[name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear)
        and 'language_model' in name.split('.') and
        (('self_attn' in name.split('.') or 'mlp' in name.split('.')) if online else name.rsplit('.',1)[-1] in ('q_proj','v_proj'))]
    if not targets:
        raise ValueError('No native text LoRA targets')
    lora=LoRAConfig(rank=8,alpha=16,dropout=0,train_mlp=online,train_attn=online,
                   train_unembed=False,seed=42,target_modules=targets)
    local=LocalBackend(device='cuda:0',dtype=torch.bfloat16,model_instance=model,
        sampler='vllm' if online and args.stage=='update' else 'hf',gradient_checkpointing=True,
        consistency_loss_options=plan.LOSS_OPTIONS[args.method],
        forward_microbatch_max_datums=1,forward_microbatch_max_tokens=40960,target_logprob_chunk_size=2048)
    backend=local
    if online and args.stage=='update':
        gpus=resolve_rollout_gpus('1,2,3',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],coordinator_device='cuda:0')
        backend=RolloutParallelBackend(local,gpus=gpus,status_dir=args.output/'workers',
            worker_vllm_options={'dtype':'bfloat16','gpu_memory_utilization':.85,'max_num_seqs':32,
                'max_num_batched_tokens':8192,'enforce_eager':False,'generation_config':'vllm','seed':42})
    checkpoint=args.output/'checkpoints'/'diagnostic-only'
    backend.setup(model=args.model,lora=lora,resume_from='file://'+str(checkpoint.resolve()) if args.stage=='restore' else None,
                  resume_with_optimizer=args.stage=='restore')
    try:
        if args.stage=='restore':
            expected=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
            if not same_state(expected,local._pending_optimizer_state):
                raise ValueError('Loaded pending optimizer state differs')
            parameters=[p for p in local.model.parameters() if p.requires_grad]
            adam=AdamConfig(**plan.contract()['optimizer'])
            local._optimizer=torch.optim.AdamW(parameters,lr=1e-4,betas=(adam.beta1,adam.beta2),
                                               eps=adam.eps,weight_decay=adam.weight_decay)
            local._optimizer.load_state_dict(local._pending_optimizer_state)
            local._pending_optimizer_state=None
            if not same_state(expected,local._optimizer.state_dict()):
                raise ValueError('Materialized optimizer state differs')
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            actual=get_peft_model_state_dict(local.model)
            if not same_state(load_file(str(checkpoint/'adapter_model.safetensors')),actual):
                raise ValueError('Restored actual adapter tensors differ')
            rng=restore_coordinator_rng(checkpoint)
            save(args.output/'restore.json',{'schema':'gemma-native-method-restore-v1',
                'source_commit':args.commit,'method':args.method,'model':args.model,
                'update':file_identity(args.output/'update.json'),'slurm_job_id':os.environ['SLURM_JOB_ID'],
                'optimizer_parameter_states':len(local._optimizer.state_dict()['state']),
                'actual_adapter_tensors':len(actual),'coordinator_rng':rng,
                'production_optimizer_updates':0,'rollout_worker_rng_serialized':False})
            return
        processor=train.alignment_processor(AutoProcessor.from_pretrained(args.model,local_files_only=True))
        renderer=HuggingFaceChatTemplateRenderer(processor,chat_template_kwargs={'enable_thinking':True})
        qids=train.ordered_pool(args.data_root)[:2]
        pairs=plan.paired_rows(qids,method=args.method)
        prompts=[{'messages':pair[side],'tokens':native_prompt(processor,pair[side])}
                 for pair in pairs for side in ('reference_messages','variant_messages')]
        plan.OUTPUT_TOKEN_CAP=train.CAP
        helpers.validate_completion=train.validate_completion
        generated=[]
        def record(tokens,assignment):
            raw=processor.decode(tokens,skip_special_tokens=False)
            if not raw.startswith('<|channel>thought\n') or '<channel|>' not in raw:
                raise ValueError('Actual sampled response lacks native thinking channel')
            generated.append({'assignment':assignment,'tokens':list(tokens),'raw_text':raw})
        if args.method=='bct':
            for row in qids:
                tokens=await helpers.bct_target(row,backend=backend,renderer=renderer,
                    cache_dir=args.output/'disposable-targets',plan_hash=hashlib.sha256(Path(args.cpu_receipt).read_bytes()).hexdigest())
                from ctm.backends.base import SampledSequence
                if train.validate_completion(SampledSequence(tokens=tokens,logprobs=[]),backend,
                    renderer=renderer,max_tokens=train.CAP)[1]!='model_eos':
                    raise ValueError('Truncated BCT gate target')
                record(tokens,{'question_id':row['question_id']})
            losses,detail=await helpers.supervised_update(args.method,qids,pairs,backend=backend,renderer=renderer,
                tokenizer=processor,cache_dir=args.output/'disposable-targets',
                plan_hash=hashlib.sha256(Path(args.cpu_receipt).read_bytes()).hexdigest(),preflight_path=None)
        elif args.method=='opct':
            from ctm.training.opct import OPCTTrainer,OPCTConfig,OPCTGenerationConfig
            trainer=OPCTTrainer(config=OPCTConfig(model=args.model,lora=lora,optimizer=AdamConfig(**plan.contract()['optimizer']),
                generation=OPCTGenerationConfig(rollouts_per_prompt=4,max_new_tokens=train.CAP,temperature=.7),
                batch_size=1,gradient_accumulation_steps=4,shuffle_samples=False,kl_coef=2.,kl_discount_factor=.9,
                loss_fn='importance_sampling'),backend=backend)
            trainer.renderer,trainer.tokenizer=renderer,processor
            trainer.sampling_client=backend.policy_sampler(name='diagnostic-only')
            trainer.reference_policy=backend.base_sampler()
            trainer.setup_done=True
            original=trainer._sample_prepared_pairs
            async def checked(prepared):
                groups=await original(prepared)
                for i,group in enumerate(groups):
                    for j,sample in enumerate(group):
                        tokens,finish=train.validate_completion(sample,backend,renderer=renderer,max_tokens=train.CAP)
                        if finish!='model_eos' or sample.logprobs is None:
                            raise ValueError('Truncated/unusable OPCT gate sample')
                        record(tokens,{'pair':i,'sample':j})
                return groups
            trainer._sample_prepared_pairs=checked
            losses,detail=await helpers.opct_update(trainer,pairs,backend=backend)
        else:
            datums,audit=build_consistency_datums_with_audit(processor,pairs)
            require_full_reference_suffix_alignment(audit)
            if len(datums)!=4:
                raise ValueError('Native paired-row coverage changed')
            losses=[]
            for datum in datums:
                pending=await backend.submit_forward_backward([datum],loss_fn=METHOD_LOSS_FNS[args.method])
                losses.append(float((await pending.result()).metrics['loss']))
        if len(losses)!=4 or not all(math.isfinite(x) for x in losses):
            raise ValueError('Invalid native method losses')
        gradients=helpers.assert_gradients(backend)
        await (await backend.submit_optim_step(learning_rate=1e-4,adam=AdamConfig(**plan.contract()['optimizer']))).result()
        await RNGCheckpointBackend(backend).save_checkpoint(name='diagnostic-only',log_dir=args.output,kind='both',
            loop_state={'step':1,'method':args.method,'diagnostic_only':True})
        save(args.output/'update.json',{'schema':'gemma-native-method-update-v1','source_commit':args.commit,
            'method':args.method,'model':args.model,'cpu_receipt':file_identity(args.cpu_receipt),
            'implementation':file_identity(__file__),'slurm_job_id':os.environ['SLURM_JOB_ID'],
            'question_ids':[row['question_id'] for row in qids],'native_prompts':prompts,
            'losses':losses,'gradients':gradients,'samples':generated,
            'checkpoint_files':{str(p.relative_to(checkpoint)):file_identity(p) for p in checkpoint.rglob('*') if p.is_file()},
            'diagnostic_optimizer_updates':1,'production_optimizer_updates':0,
            'generation_cap_including_reasoning':train.CAP if online else None,
            'not_a_scientific_parent':True})
    finally:
        backend.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['update','restore'])
    p.add_argument('--method',choices=plan.METHODS,required=True)
    for name in ('repository','commit','model','cpu-receipt'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    asyncio.run(run(p.parse_args()))
