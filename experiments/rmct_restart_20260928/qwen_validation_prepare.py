"""Translate one clean checkpoint and measure its evaluator-path adapter effect.

Runs only inside an allocated GPU step. Uses explicitly pinned historical probe
inputs/token protocol, never historical adapters, runtime paths or scores.
"""
import argparse
import os
from pathlib import Path
import sys
from experiments.rmct_restart_20260928 import qwen_validation as v
from experiments.rmct_restart_20260928.qwen_checkpoint import identity
from experiments.rmct_restart_20260928.qwen_launch import source_check
from experiments.rmct_restart_20260928.qwen_validation_executor import read,write,check_contract
from experiments.rmct_restart_20260928.qwen_validation_producer import scheduler_complete
from experiments.rmct_restart_20260928.qwen_validation_server import check_executable


def prepare(a):
    import shutil
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Scheduled GPU allocation required')
    root=Path(__file__).resolve().parents[2];source_check(root,a.source_commit)
    scheduler_complete(a.training_job)
    v.population(a.manifest)
    sealed=read(a.checkpoint_seal)
    if sealed['schema']!='rmct-clean-checkpoint-v1' or sealed['step']%64:raise ValueError('Expected clean64-update boundary')
    for f in sealed['files'].values():
        if identity(f['path'])!=f:raise ValueError('Sealed checkpoint changed')
    if v.sha(a.reference_report)!=a.reference_sha:raise ValueError('Probe reference changed')
    reference=read(a.reference_report);data=reference['data']
    if v.sha(data['path'])!=data['sha256']:raise ValueError('Probe population changed')
    out=a.folder.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'preparation-claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],
          checkpoint_seal_sha256=v.sha(a.checkpoint_seal),reference_sha256=a.reference_sha,source_commit=a.source_commit))
    from experiments.act_repair_gate.vllm_compat_adapter import make_compat_adapter
    from experiments.act_repair_gate import runtime_parity as backend
    from transformers import AutoTokenizer
    check_executable(shutil.which('vllm'),sys.prefix)
    adapter=out/'adapter';make_compat_adapter(sealed['checkpoint'],adapter)
    translation=read(adapter/'compatibility-manifest.json')
    raw=sealed['files']['adapter_model.safetensors']
    translated=v.sha(adapter/'adapter_model.safetensors')
    if translation['source']['adapter_model_sha256']!=raw['sha256'] or translation['destination']['adapter_model_sha256']!=translated:
        raise ValueError('Adapter translation identity mismatch')
    if v.sha(adapter/'adapter_config.json')!=sealed['files']['adapter_config.json']['sha256']:
        raise ValueError('Adapter configuration/scope changed')
    prompts=backend.load_prompts(data['path'],limit=data['samples'])
    if [p.question_id for p in prompts]!=data['question_ids']:raise ValueError('Probe QID mismatch')
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    ids=[backend._chat_prompt_ids(tokenizer,p.messages) for p in prompts]
    if [len(p) for p in ids]!=data['chat_prompt_token_lengths']:raise ValueError('Probe rendering changed')
    protocol=reference['token_protocol'];wanted=protocol['requested_token_ids'];refs=protocol['reference_token_ids']
    device=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')[0]
    if not device:raise ValueError('Missing allocated GPU')
    caches={key:out/'probe-cache'/leaf for key,leaf in backend.PARALLEL_VLLM_CACHE_ENVIRONMENT}
    for p in caches.values():p.mkdir(parents=True,exist_ok=False)
    log=out/'probe-server.log';url=f'http://127.0.0.1:{a.port}/v1'
    process=backend._start_vllm(model=a.model,port=a.port,max_model_len=98304,
        gpu_memory_utilization=.90,max_loras=1,log_path=log,enforce_eager=True,
        gdn_prefill_backend='triton',device_token=device,cache_directories=caches)
    try:
        if backend._await_server(process,url,log)!=a.model:raise ValueError('Wrong served model')
        backend._load_vllm_adapters(url,{str(adapter):str(adapter)})
        exposed=backend._vllm_model_ids(url)
        base=backend._vllm_scores(base_url=url,model_name=a.model,prompt_token_ids=ids,token_ids=wanted)
        adapted=backend._vllm_scores(base_url=url,model_name=str(adapter),prompt_token_ids=ids,token_ids=wanted)
        effect=[]
        for i,tokens in enumerate(wanted):effect.extend(backend._effect_vector(base[i],adapted[i],reference_token_id=refs[i],token_ids=tokens))
        report=dict(model=a.model,numerical_parity_claimed=False,
            adapter=dict(path=str(adapter),hf_adapter_model_sha256=raw['sha256'],adapter_model_sha256=translated),
            backends={'vllm':{'runtime_adapter_names':{'evaluator_path':str(adapter)},'model_ids_after_load':{'evaluator_path':exposed}}},
            results={'evaluator_path':{'vllm_effect_vector':effect}},data=data,token_protocol=protocol,
            probe_prompt_token_ids=ids,base_scores=base,adapter_scores=adapted)
        write(out/'activity-report.json',report)
    finally:backend._stop_process(process)
    contract=dict(schema='rmct-clean-validation-v1',campaign_id=sealed['campaign_id'],step=sealed['step'],
        source_commit=a.source_commit,model=a.model,adapter=str(adapter),settings=v.SETTINGS,
        validation_sha256=v.PROMPT_SHA,raw_adapter=raw,
        translated_adapter=identity(adapter/'adapter_model.safetensors'),
        activity_report=identity(out/'activity-report.json'),
        checkpoint_seal=identity(a.checkpoint_seal),translation=identity(adapter/'compatibility-manifest.json'),
        raw_config=sealed['files']['adapter_config.json'],translated_config=identity(adapter/'adapter_config.json'))
    check_contract(contract,root)
    write(out/'contract.json',contract)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint-seal','reference-report','folder','manifest'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('reference-sha','model','source-commit','training-job'):p.add_argument('--'+name,required=True)
    p.add_argument('--port',type=int,default=19789)
    prepare(p.parse_args())
