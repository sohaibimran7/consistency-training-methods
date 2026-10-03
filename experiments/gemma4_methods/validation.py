"""Native token-preserving 600-response validation and shared-controller ingestion.

Generation and ingestion are separate scheduled jobs: ingestion requires the
actual generation job to have completed. No uncertain request is retried.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.gemma4_methods.selection_adapter import file_identity, read_verified, native_prompt
from experiments.gemma4_methods.reference.plan import immutable_json
from experiments.gemma4_methods.launch_guard import check_source


def check_boundary(progress, contract):
    from experiments.rmct_restart_20260928.validation_selection import check_progress, check_contract
    check_contract(contract)
    check_progress(progress, contract)
    from experiments.rmct_restart_20260928.validation_selection import (
        encounter_mode, interval_attempts, max_attempts)
    step=progress['actual_optimizer_step']
    if encounter_mode(contract):
        cursor=progress['encounter_attempt']
        if step<1 or cursor<=0 or (cursor%interval_attempts() and cursor!=max_attempts()):
            raise ValueError('Validation requires a 256-encounter boundary with a saved update')
    elif step<64 or step%64:
        raise ValueError('Validation requires an actual64-update boundary')
    return step


def saved_response(request, sequence, processor, stops, cap):
    tokens=list(sequence.token_ids)
    reason=sequence.finish_reason
    if not tokens or len(tokens)>cap or reason not in ('stop','length'):
        raise ValueError('Unknown/invalid native generation termination')
    if any(type(t) is not int or t<0 for t in tokens):
        raise ValueError('Invalid native response tokens')
    if reason=='stop' and tokens[-1] not in stops:
        raise ValueError('Native EOS not retained in sampled tokens')
    return {'sample_id':request['sample_id'],'generated_token_ids':tokens,
            'raw_text':processor.decode(tokens,skip_special_tokens=False),'finish_reason':reason}


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _load(args):
    contract=read_verified(file_identity(args.contract))
    check_source(Path(__file__).resolve().parents[2],contract['source_commit'])
    progress=read_verified(file_identity(args.progress))
    step=check_boundary(progress,contract)
    rows=read_verified(contract['population'])['rows']
    if len(rows)!=600 or len({r['sample_id'] for r in rows})!=600:
        raise ValueError('Exact600response population required')
    settings=contract['settings']
    required={'enable_thinking':True,'max_tokens':20480,'temperature':1.0,'top_p':.95,'top_k':20}
    if any(settings.get(k)!=v for k,v in required.items()):
        raise ValueError('Approved validation settings changed')
    return contract,progress,step,rows


def _generate_rows(contract,progress,rows,output):
    """Generate and immutably save one response per row (fixed per-sample seed)."""
    from transformers import AutoProcessor
    from ctm.evals.local_model import Gemma4UnifiedTextProcessor
    from ctm.backends.local.vllm_sampler import VLLMSampler
    from vllm import SamplingParams
    processor=Gemma4UnifiedTextProcessor(AutoProcessor.from_pretrained(contract['model'],local_files_only=True))
    settings=contract['settings']
    sampler=VLLMSampler(contract['model'],enable_lora=True,dtype='bfloat16',
        gpu_memory_utilization=.85,max_num_seqs=32,max_num_batched_tokens=8192,
        max_lora_rank=8,enforce_eager=False,generation_config='vllm')
    try:
        sampler.advance_policy(progress['checkpoint'])
        config=json.loads((Path(contract['model'])/'generation_config.json').read_text())
        bad_words=[processor.decode([t],skip_special_tokens=False) for t in config.get('suppress_tokens',[])]
        for offset in range(0,len(rows),32):
            group=rows[offset:offset+32]
            requests=[]
            parameters=[]
            for row in group:
                seed=int(_sha(row['sample_id'])[:8],16)%2147483647
                request={'sample_id':row['sample_id'],'prompt_token_ids':native_prompt(processor,row['messages']),
                    'model':contract['model'],'settings':settings,'checkpoint_files':progress['checkpoint_files'],
                    'seed':seed,'bad_words':bad_words}
                path=output/'requests'/(_sha(row['sample_id'])+'.json')
                immutable_json(path,request)
                requests.append((request,file_identity(path)))
                parameters.append(SamplingParams(n=1,max_tokens=20480,temperature=1.,top_p=.95,top_k=20,
                    stop_token_ids=settings['stop_token_ids'],seed=seed,bad_words=bad_words))
            outputs=sampler.engine.generate([sampler._api.TokensPrompt(prompt_token_ids=r['prompt_token_ids'])
                for r,ref in requests],parameters,lora_request=sampler._policy_lora_request(),use_tqdm=False)
            if len(outputs)!=len(requests):raise ValueError('Incomplete validation batch')
            for (request,ref),result in zip(requests,outputs):
                if list(result.prompt_token_ids)!=request['prompt_token_ids'] or len(result.outputs)!=1:
                    raise ValueError('Native returned prompt/sample assignment differs')
                response=saved_response(request,result.outputs[0],processor,settings['stop_token_ids'],20480)
                response['request']=ref
                immutable_json(output/'responses'/(_sha(request['sample_id'])+'.json'),response)
    finally:
        sampler.shutdown()


def _shard_count():
    shards=int(os.environ.get('GEMMA_VALIDATION_SHARDS','1'))
    visible=[d for d in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if d]
    if shards<1 or (shards>1 and len(visible)<shards):
        raise ValueError('Each validation shard needs its own visible GPU')
    return shards,visible


def generate(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Scheduled validation required')
    contract,progress,step,rows=_load(args)
    from experiments.gemma4_methods.native_hooks import runtime_restore
    if contract['method']=='rmct':
        from experiments.rmct_restart_20260928.gemma_controller import runtime_restore
    runtime_restore(progress,contract)
    shards,visible=_shard_count()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    immutable_json(output/'claim.json',{'job_id':os.environ['SLURM_JOB_ID'],
        'contract':file_identity(args.contract),'progress':file_identity(args.progress),'step':step})
    if shards==1:
        _generate_rows(contract,progress,rows,output)
    else:
        # Same prompts, settings and fixed per-sample seeds; only the GPU that
        # serves each prompt differs. Strided shards, one vLLM process per GPU.
        import subprocess,sys
        children=[]
        for index in range(shards):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':visible[index]}
            children.append(subprocess.Popen([sys.executable,'-B','-m','experiments.gemma4_methods.validation',
                'generate-shard','--shard',str(index),'--shards',str(shards),'--contract',str(args.contract),
                '--progress',str(args.progress),'--output',str(output),'--folder',str(args.folder)],env=env))
        if any(child.wait()!=0 for child in children):
            raise RuntimeError('A validation shard failed; no evidence published')
    samples={}
    for row in rows:
        request=output/'requests'/(_sha(row['sample_id'])+'.json')
        response=output/'responses'/(_sha(row['sample_id'])+'.json')
        if not request.is_file() or not response.is_file():
            raise ValueError('Missing validation request/response for '+row['sample_id'])
        samples[row['sample_id']]={'request':file_identity(request),'response':file_identity(response)}
    import subprocess
    executing=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip()
    evidence={'schema':'gemma-native-validation-evidence-v1',
        **{k:contract[k] for k in ('campaign_id','method','model','source_commit')},
        'checkpoint_files':progress['checkpoint_files'],'scheduler':{'job_id':os.environ['SLURM_JOB_ID']},
        'execution':{'commit':executing,'shards':shards},'samples':samples}
    immutable_json(output/'evidence.json',evidence)


def generate_shard(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Scheduled validation required')
    contract,progress,step,rows=_load(args)
    output=args.output.resolve()
    claim=json.loads((output/'claim.json').read_text())
    if claim['job_id']!=os.environ['SLURM_JOB_ID'] or not 0<=args.shard<args.shards:
        raise ValueError('Shard does not belong to this validation job')
    _generate_rows(contract,progress,rows[args.shard::args.shards],output)


def accept(args):
    from transformers import AutoProcessor
    from ctm.evals.local_model import Gemma4UnifiedTextProcessor
    from experiments.gemma4_methods.selection_adapter import GemmaVerifiers
    from experiments.gemma4_methods.native_hooks import runtime_restore,scheduler_complete
    from experiments.rmct_restart_20260928.validation_selection import accept_validation
    contract_record=file_identity(args.contract)
    contract=read_verified(contract_record)
    check_source(Path(__file__).resolve().parents[2],contract['source_commit'])
    processor=Gemma4UnifiedTextProcessor(AutoProcessor.from_pretrained(contract['model'],local_files_only=True))
    if contract['method']=='rmct':
        from experiments.rmct_restart_20260928.gemma_controller import RMCTVerifiers
        adapter=RMCTVerifiers(processor=processor)
    else:
        adapter=GemmaVerifiers(processor=processor,verify_runtime_checkpoint=runtime_restore,verify_scheduler=scheduler_complete)
    return accept_validation(args.folder,contract_record,file_identity(args.progress),
        file_identity(args.output/'evidence.json'),verify_checkpoint=adapter.verify_checkpoint,
        verify_validation=adapter.verify_validation)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['generate','generate-shard','accept'])
    for name in ('contract','progress','output','folder'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    a=p.parse_args()
    actions={'generate':generate,'generate-shard':generate_shard,'accept':accept}
    print(json.dumps(actions[a.action](a),default=str))
