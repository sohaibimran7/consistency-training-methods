"""Bounded new-campaign validation worker against a prepared local vLLM server.

Does not prepare adapters, start servers, submit jobs or advance training.
Every checkpoint needs fresh activity evidence before this worker is invoked.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
from urllib.parse import urlparse
from experiments.rmct_restart_20260928 import qwen_validation as v


def read(p): return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def check_contract(contract,root):
    if contract['schema'] not in ('rmct-clean-validation-v1','qwen-one-bias-validation-v1') or contract['settings']!=v.SETTINGS:
        raise ValueError('Not new validation contract')
    if contract['schema']=='qwen-one-bias-validation-v1':
        # Fresh one-bias campaigns validate on the 256-encounter grid instead.
        encounters=contract['encountered_qid_bias_examples']
        if (not contract['campaign_id'] or type(encounters) is not int or encounters<=0
                or (encounters%256 and encounters!=7680)):
            raise ValueError('Invalid campaign/encounter boundary')
    elif not contract['campaign_id'] or contract['step']<64 or contract['step']%64:
        raise ValueError('Invalid campaign/checkpoint')
    if contract['validation_sha256']!=v.PROMPT_SHA: raise ValueError('Population changed')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if commit!=contract['source_commit']: raise ValueError('Wrong deployed source')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip():
        raise ValueError('Dirty deployed source')
    for key in ('raw_adapter','translated_adapter','activity_report','checkpoint_seal','translation','raw_config','translated_config'):
        evidence=contract[key]
        if v.sha(evidence['path'])!=evidence['sha256']: raise ValueError(f'Changed {key}')
    report=read(contract['activity_report']['path'])
    adapter=report['adapter'];backend=report['backends']['vllm']
    if adapter['hf_adapter_model_sha256']!=contract['raw_adapter']['sha256'] or adapter['adapter_model_sha256']!=contract['translated_adapter']['sha256']:
        raise ValueError('Activity/checkpoint mismatch')
    if adapter['path']!=contract['adapter'] or backend['runtime_adapter_names']['evaluator_path']!=contract['adapter']:
        raise ValueError('Wrong evaluator adapter')
    if report['model']!=contract['model']: raise ValueError('Wrong base model')
    values=report['results']['evaluator_path']['vllm_effect_vector']
    if not values or any(type(x) not in (int,float) or not math.isfinite(x) for x in values) or max(map(abs,values))<=1e-5:
        raise ValueError('No finite nonzero adapter effect')
    if Path(contract['translated_adapter']['path']).parent!=Path(contract['adapter']):
        raise ValueError('Requested adapter does not contain pinned weights')


def worker(folder,manifest,rank,endpoint):
    folder=Path(folder).resolve();root=Path(__file__).resolve().parents[2]
    if not os.environ.get('SLURM_JOB_ID'): raise ValueError('Scheduled allocation required')
    if rank not in range(4): raise ValueError('Four ranks required')
    url=urlparse(endpoint)
    if url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost') or url.path!='/v1':
        raise ValueError('Prepared local /v1 endpoint required')
    contract=read(folder/'contract.json');check_contract(contract,root)
    rows=v.population(manifest)
    assigned=rows[rank::4]
    # Never blindly retry an uncertain request, even if no output was saved.
    write(folder/f'worker-{rank}-claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],
          contract_sha256=v.sha(folder/'contract.json'),rank=rank))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(contract['model'],local_files_only=True)
    with urllib.request.urlopen(endpoint+'/models',timeout=30) as response:
        exposed={r['id'] for r in json.load(response)['data']}
    if not {contract['model'],contract['adapter']}<=exposed: raise ValueError('Exact base/adapter not exposed')
    def generate(row):
        sid=row['sample_id'];reqpath=folder/'requests'/f'{sid}.json'
        request=dict(model=contract['adapter'],prompt=v.render_tokens(tokenizer,row['messages']),
            temperature=1.,top_p=.95,top_k=20,max_tokens=20480,
            seed=int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)%2147483647)
        write(reqpath,dict(sample_id=sid,request=request,contract_sha256=v.sha(folder/'contract.json')))
        raw=urllib.request.Request(endpoint+'/completions',data=json.dumps(request).encode(),
                                   headers={'Content-Type':'application/json'},method='POST')
        with urllib.request.urlopen(raw,timeout=21600) as response: result=json.load(response)
        if len(result.get('choices',[]))!=1: raise ValueError('Expected one completion')
        choice=result['choices'][0]
        write(folder/'responses'/f'{sid}.json',dict(sample_id=sid,text=choice['text'],
              finish_reason=choice['finish_reason'],response=result,usage=result.get('usage'),
              request_sha256=v.sha(reqpath),contract_sha256=v.sha(folder/'contract.json')))
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(generate,assigned))
    write(folder/f'worker-{rank}-complete.json',dict(count=len(assigned),contract_sha256=v.sha(folder/'contract.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--endpoint',required=True)
    a=p.parse_args();worker(a.folder,a.manifest,a.rank,a.endpoint)
