"""Freeze approved fresh-run records after actual scheduled native gates."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.gemma4_methods import train
from experiments.gemma4_methods.reference import plan
from experiments.gemma4_methods.selection_adapter import file_identity,read_verified
from experiments.gemma4_methods.launch_guard import check_source
from experiments.gemma4_methods.native_hooks import scheduler_complete
from experiments.rmct_restart_20260928.validation_selection import ENCOUNTER_POLICY,check_contract
from experiments.rmct_restart_20260928.qwen_validation import population


def prepare(args):
    repo=check_source(args.repository,args.commit)
    if not args.approval_reference.strip():raise ValueError('Explicit scoped human approval required')
    population(args.population)
    cpu=read_verified(file_identity(args.cpu_receipt))
    if cpu['source_commit']!=args.commit or cpu['model_files'][0]['sha256']!='5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d':
        raise ValueError('Current native CPU/publisher receipt required')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    for relative,digest in ((plan.POOL,plan.POOL_SHA),(plan.MANIFEST,plan.MANIFEST_SHA)):
        source=args.data_root/relative
        if plan.sha256(source)!=digest:raise ValueError('Frozen7680QID training data changed')
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    plan.immutable_json(root/'thinking-run-approval.json',{'run_root':str(root),'scope':'training',
        'enable_thinking':True,'generated_token_cap_including_reasoning':20480,
        'user_approval_reference':args.approval_reference})
    train.freeze(root,repo)
    ids=[r['question_id'] for r in train.ordered_pool(root)]
    plan.immutable_json(root/'data-order.json',{'ordered_qids':ids,'ordered_qids_sha256':train.ORDER_SHA})
    config=json.loads((args.model/'generation_config.json').read_text())
    stops=config['eos_token_id'];stops=[stops] if type(stops) is int else stops
    if not stops or any(type(t) is not int or t<0 for t in stops):raise ValueError('Native stop ids required')
    for method in plan.METHODS:
        matched=[p for p in args.native.glob('*/update.json') if json.loads(p.read_text())['method']==method]
        if len(matched)!=1:raise ValueError('Unique actual method gate required:'+method)
        update=matched[0];restore=update.with_name('restore.json')
        u=json.loads(update.read_text());r=json.loads(restore.read_text())
        if any(x['source_commit']!=args.commit or x['model']!=str(args.model) for x in (u,r)):
            raise ValueError('Gate source/model differs')
        scheduler_complete({'job_id':u['slurm_job_id']},{})
        scheduler_complete({'job_id':r['slurm_job_id']},{})
        if r['optimizer_parameter_states']<=0 or r['actual_adapter_tensors']<=0:
            raise ValueError('Actual native restore evidence missing')
        folder=root/'selection'/method;folder.mkdir(parents=True)
        contract={'schema':'ctm-tbsr-selection-contract-v2-encounters','policy':ENCOUNTER_POLICY,
            'campaign_id':args.campaign+'-'+method,'method':method,'model':str(args.model),
            'source_commit':args.commit,'population':file_identity(args.population),'response_count':600,'pair_count':400,
            'data_order':file_identity(root/'data-order.json'),'approval_reference':args.approval_reference,
            'settings':{'enable_thinking':True,'max_tokens':20480,'temperature':1.,'top_p':.95,'top_k':20,'stop_token_ids':stops}}
        check_contract(contract)
        plan.immutable_json(folder/'contract.json',contract)
        plan.immutable_json(folder/'start.json',{'schema':'ctm-training-start-v1',
            **{k:contract[k] for k in ('campaign_id','method','model','source_commit')},
            'contract':file_identity(folder/'contract.json'),'actual_optimizer_step':0,'next_attempt_index':0,
            'encounter_attempt':0,'one_bias_manifest':json.loads((root/'contract.json').read_text())['one_bias_manifest'],
            'resume_from':None,'optimizer':'fresh','training_contract':file_identity(root/'contract.json'),
            'cpu_receipt':file_identity(args.cpu_receipt),'native_update':file_identity(update),'native_restore':file_identity(restore)})
    return {'root':str(root),'source_commit':args.commit,'methods':list(plan.METHODS)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repository','model','output','data-root','cpu-receipt','population','native'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('commit','campaign','approval-reference'):p.add_argument('--'+name,required=True)
    print(json.dumps(prepare(p.parse_args())))
