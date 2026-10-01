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
    step=progress['actual_optimizer_step']
    if step<64 or step%64:
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


def generate(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Scheduled validation required')
    contract=read_verified(file_identity(args.contract))
    check_source(Path(__file__).resolve().parents[2],contract['source_commit'])
    progress=read_verified(file_identity(args.progress))
    step=check_boundary(progress,contract)
    from transformers import AutoProcessor
    from ctm.evals.local_model import Gemma4UnifiedTextProcessor
    from ctm.backends.local.vllm_sampler import VLLMSampler
    from vllm import SamplingParams
    processor=Gemma4UnifiedTextProcessor(AutoProcessor.from_pretrained(contract['model'],local_files_only=True))
    from experiments.gemma4_methods.native_hooks import runtime_restore
    if contract['method']=='rmct':
        from experiments.rmct_restart_20260928.gemma_controller import runtime_restore
    runtime_restore(progress,contract)
    rows=read_verified(contract['population'])['rows']
    if len(rows)!=600 or len({r['sample_id'] for r in rows})!=600:
        raise ValueError('Exact600response population required')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    immutable_json(output/'claim.json',{'job_id':os.environ['SLURM_JOB_ID'],
        'contract':file_identity(args.contract),'progress':file_identity(args.progress),'step':step})
    sampler=VLLMSampler(contract['model'],enable_lora=True,dtype='bfloat16',
        gpu_memory_utilization=.85,max_num_seqs=32,max_num_batched_tokens=8192,
        max_lora_rank=8,enforce_eager=False,generation_config='vllm')
    try:
        sampler.advance_policy(progress['checkpoint'])
        settings=contract['settings']
        required={'enable_thinking':True,'max_tokens':20480,'temperature':1.0,'top_p':.95,'top_k':20}
        if any(settings.get(k)!=v for k,v in required.items()):
            raise ValueError('Approved validation settings changed')
        config=json.loads((Path(contract['model'])/'generation_config.json').read_text())
        bad_words=[processor.decode([t],skip_special_tokens=False) for t in config.get('suppress_tokens',[])]
        samples={}
        for offset in range(0,600,32):
            group=rows[offset:offset+32]
            requests=[]
            parameters=[]
            for row in group:
                seed=int(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8],16)%2147483647
                request={'sample_id':row['sample_id'],'prompt_token_ids':native_prompt(processor,row['messages']),
                    'model':contract['model'],'settings':settings,'checkpoint_files':progress['checkpoint_files'],
                    'seed':seed,'bad_words':bad_words}
                path=output/'requests'/(hashlib.sha256(row['sample_id'].encode()).hexdigest()+'.json')
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
                path=output/'responses'/(hashlib.sha256(request['sample_id'].encode()).hexdigest()+'.json')
                immutable_json(path,response)
                samples[request['sample_id']]={'request':ref,'response':file_identity(path)}
        evidence={'schema':'gemma-native-validation-evidence-v1',
            **{k:contract[k] for k in ('campaign_id','method','model','source_commit')},
            'checkpoint_files':progress['checkpoint_files'],'scheduler':{'job_id':os.environ['SLURM_JOB_ID']},
            'samples':samples}
        immutable_json(output/'evidence.json',evidence)
    finally:
        sampler.shutdown()


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
    p.add_argument('action',choices=['generate','accept'])
    for name in ('contract','progress','output','folder'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(generate(a) if a.action=='generate' else accept(a),default=str))
