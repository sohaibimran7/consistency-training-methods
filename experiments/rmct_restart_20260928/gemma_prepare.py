"""Prepare a fresh Gemma RMCT plan and native-proof-bound start, never submit."""
import argparse
import json
from pathlib import Path

from experiments.gemma4_methods.selection_adapter import file_identity,read_verified
from experiments.gemma4_methods.reference.plan import immutable_json
from experiments.gemma4_methods.launch_guard import check_source
from experiments.rmct_restart_20260928.gemma_production_plan import build
from experiments.rmct_restart_20260928.qwen_validation import population,PROMPT_SHA
from experiments.rmct_restart_20260928.validation_selection import ENCOUNTER_POLICY,check_contract
from experiments.gemma4_methods.reference.plan import POOL_SHA as DATA_SHA256,MANIFEST_SHA as MANIFEST_SHA256


def prepare_plan(args):
    check_source(args.repository,args.commit)
    population(args.population)
    if file_identity(args.data)['sha256']!=DATA_SHA256 or file_identity(args.manifest)['sha256']!=MANIFEST_SHA256:
        raise ValueError('Shared 7,680-QID pool identity differs')
    # The same deterministic one-bias manifest as the five file-driven methods.
    from experiments.gemma4_methods import train as methods
    root=args.data.resolve().parents[len(Path(methods.reference.POOL).parts)-1]
    if (root/methods.reference.POOL).resolve()!=args.data.resolve() or (root/methods.reference.MANIFEST).resolve()!=args.manifest.resolve():
        raise ValueError('Pool paths must be the shared frozen data-root layout')
    from experiments.gemma4_methods import one_bias
    path,manifest=one_bias.freeze(args.output,methods.ordered_pool(root),pool_sha256=DATA_SHA256,
        manifest_sha256=MANIFEST_SHA256,order_sha256=methods.ORDER_SHA)
    provenance=json.loads(args.parity_provenance.read_text())
    if provenance['source_commit']!=args.commit or provenance['cpu_receipt_sha256']!=file_identity(args.cpu_receipt)['sha256']:
        raise ValueError('Actual same-source text LoRA target provenance required')
    value=build(repo=str(args.repository.resolve()),python=args.python,model=str(args.model),
        targets=provenance['targets'],data=str(args.data),manifest=str(args.manifest),
        one_bias_manifest=str(path.resolve()),one_bias_manifest_sha256=one_bias.protocol.manifest_identity(manifest),
        commit=args.commit,
        run_name=args.campaign,approval_reference=args.approval_reference,validation_sha256=PROMPT_SHA)
    immutable_json(args.output/'plan.json',value)
    config=json.loads((args.model/'generation_config.json').read_text())
    stops=config['eos_token_id'];stops=[stops] if type(stops) is int else stops
    contract={'schema':'ctm-tbsr-selection-contract-v2-encounters','policy':ENCOUNTER_POLICY,
        'campaign_id':args.campaign,'method':'rmct','model':str(args.model),'source_commit':args.commit,
        'population':file_identity(args.population),'response_count':600,'pair_count':400,
        'approval_reference':args.approval_reference,
        'settings':{'enable_thinking':True,'max_tokens':20480,'temperature':1.,'top_p':.95,'top_k':20,'stop_token_ids':stops}}
    check_contract(contract)
    immutable_json(args.output/'selection/contract.json',contract)
    return value


def prepare_start(args):
    contract=read_verified(file_identity(args.output/'selection/contract.json'))
    check_source(args.repository,contract['source_commit'])
    plan_record=file_identity(args.output/'plan.json')
    gate=read_verified(file_identity(args.rl_gate))
    if gate['plan']!=plan_record or gate['optimizer_updates']!=1 or gate['production_resume_forbidden'] is not True:
        raise ValueError('Actual disposable native RL gate required')
    parity=read_verified(file_identity(args.parity))
    if parity['status']!='passed':raise ValueError('Actual native parity required')
    from experiments.gemma4_methods.native_hooks import scheduler_complete
    scheduler_complete({'job_id':gate['job_id']},contract)
    scheduler_complete({'job_id':args.parity_job},contract)
    immutable_json(args.output/'selection/start.json',{'schema':'ctm-training-start-v1',
        **{k:contract[k] for k in ('campaign_id','method','model','source_commit')},
        'contract':file_identity(args.output/'selection/contract.json'),
        'actual_optimizer_step':0,'next_attempt_index':0,'encounter_attempt':0,'resume_from':None,'optimizer':'fresh',
        'native_plan':plan_record,'cpu_receipt':file_identity(args.cpu_receipt),
        'native_rl_gate':file_identity(args.rl_gate),'native_parity':file_identity(args.parity),
        'parity_job':args.parity_job})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','start'])
    for name in ('repository','output','cpu-receipt'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('model','population','data','manifest','parity-provenance','rl-gate','parity'):
        p.add_argument('--'+name,type=Path)
    for name in ('commit','python','campaign','approval-reference','parity-job'):p.add_argument('--'+name)
    a=p.parse_args()
    required=('model','population','data','manifest','parity_provenance','commit','python','campaign','approval_reference') if a.action=='plan' else ('rl_gate','parity','parity_job')
    if not all(getattr(a,key) for key in required):p.error('Missing action-specific inputs')
    prepare_plan(a) if a.action=='plan' else prepare_start(a)
