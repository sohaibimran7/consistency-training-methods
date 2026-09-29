"""Prepare/validate a distinct 65536-output campaign, without inference or network.

Historical panels are immutable inputs. Validation requires externally collected,
hash-bound serving/tokenization evidence. Passing this offline contract is not
itself permission to execute or proof that a long-context server has been tested.
"""
import argparse
import hashlib
import json
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(panel_path,output,campaign_id,context_limit):
    if not campaign_id or context_limit<=65536:
        raise ValueError('Distinct campaign ID and context limit greater than65536 required')
    old=json.loads(panel_path.read_text())
    if old.get('status')!='frozen_reviewed_for_execution':raise ValueError('Reviewed source panel required')
    config=dict(old['target_config'],max_tokens=65536)
    rows=[]
    for row in old['rows']:
        if digest(row['messages'])!=row['messages_sha256']:raise ValueError('Source prompt hash mismatch')
        rows.append(dict(source_id=row['source_id'],condition=row['condition'],messages_sha256=row['messages_sha256']))
    result=dict(schema_version=1,campaign_id=campaign_id,status='prepared_not_runtime_validated',
        source_panel=dict(path=str(panel_path),sha256=sha(panel_path)),models=old['models'],rows=rows,
        target_config=config,seed=20260910,epochs=1,requested_context_limit=context_limit,
        generation_policy=dict(default_max_tokens=65536,exceptions={'logiqa':20480,'hellaswag':20480},
            exceptions_used=[],scope='All rows in this eval-gaming campaign use65536; exceptions are unrelated datasets, never factor/scenario aliases.'),
        no_fallback=True,reuse_historical_8192=False,
        requirements=['Fresh output namespace and serving allocation attestations',
            'Configured and effective server context must equal requested_context_limit',
            'Pinned model revisions and complete long-context/rope configuration evidence',
            'Exact server-rendered token counts including chat template, generation prefix and any tools for every role/prompt',
            'prompt_tokens +65536 <= effective context; never shrink output ceiling or truncate inputs',
            'Live runner must enforce this guard before every request; no live execution implemented in this preparer'],
        incompatibilities=['Historical runner and attestation contract fix context32768 and output8192; incompatible with this configuration',
            'Requested context length is a proposal, NOT evidence of native model support or validated rope extension'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2)
    return result

def validate(config,evidence):
    if config['target_config']['max_tokens']!=65536 or config.get('no_fallback') is not True:
        raise ValueError('Output policy mismatch or silent fallback enabled')
    if config.get('reuse_historical_8192') is not False:raise ValueError('Historical short-ceiling reuse forbidden')
    limit=config['requested_context_limit']
    if type(limit) is not int or limit<=65536:raise ValueError('Context cannot accommodate prompt plus output')
    if evidence.get('campaign_sha256')!=digest(config):raise ValueError('Evidence belongs to different campaign')
    if set(evidence.get('roles',{}))!=set(config['models']):raise ValueError('Missing/extra serving roles')
    expected={ (role,r['source_id'],r['condition']):r for role in config['models'] for r in config['rows'] }
    if len(expected)!=len(config['models'])*len(config['rows']):raise ValueError('Duplicate prompt identities')
    for role,pin in config['models'].items():
        server=evidence['roles'][role]
        if [server.get('model'),server.get('revision')]!=pin:raise ValueError('Model revision mismatch')
        for key in ('configured_context_limit','effective_context_limit'):
            if type(server.get(key)) is not int or server[key]!=limit:raise ValueError('Serving context mismatch')
        if server.get('long_context_configuration_verified') is not True:
            raise ValueError('Native/extended context configuration is unverified')
        for field in ('attestation_sha256','tokenizer_configuration_sha256','model_configuration_sha256'):
            value=server.get(field)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Missing hashed serving/tokenizer/configuration provenance')
    seen=set();maximum=0
    for receipt in evidence.get('token_counts',[]):
        identity=(receipt['role'],receipt['source_id'],receipt['condition'])
        if identity not in expected or identity in seen:raise ValueError('Unexpected/duplicate token count')
        seen.add(identity);row=expected[identity];server=evidence['roles'][receipt['role']]
        if receipt.get('messages_sha256')!=row['messages_sha256']:raise ValueError('Token count prompt mismatch')
        if receipt.get('tokenizer_configuration_sha256')!=server['tokenizer_configuration_sha256']:
            raise ValueError('Token count tokenizer mismatch')
        if receipt.get('render_contract')!={'enable_thinking':True,'add_generation_prompt':True,'tools':None}:
            raise ValueError('Must count exact rendered text-only request with thinking/generation prefix')
        n=receipt.get('prompt_tokens')
        if type(n) is not int or n<=0:raise ValueError('Invalid prompt token count')
        if n+65536>limit:raise ValueError('Context overflow: do not truncate prompt or lower max_tokens')
        maximum=max(maximum,n)
    if seen!=set(expected):raise ValueError('Incomplete role-by-prompt token coverage')
    return dict(status='offline_context_contract_validated',requests=len(seen),max_prompt_tokens=maximum,
        output_ceiling=65536,context_limit=limit,minimum_headroom=limit-maximum-65536,
        note='Saved evidence validated offline; no live endpoint, authorization, or generation claim.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--panel',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--campaign-id',required=True);a.add_argument('--context-limit',type=int,required=True)
    b=sub.add_parser('validate');b.add_argument('--config',type=Path,required=True);b.add_argument('--evidence',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':
        result=prepare(args.panel,args.output,args.campaign_id,args.context_limit)
        print(json.dumps(dict(status=result['status'],output=str(args.output),configuration_digest=digest(result))))
    else:print(json.dumps(validate(json.loads(args.config.read_text()),json.loads(args.evidence.read_text())),indent=2))
