import json,hashlib
from pathlib import Path
from common import artifact_directory
R=artifact_directory("compute-matched-selection-20260926")
D=json.loads((R/'inventory.json').read_text())
S=json.loads((R.parent/'compute-audit-20260917/summary.json').read_text())
out={}
for m,d in D.items():
    if m=='rmct':continue
    attempts={a['job']:a['rows'] for a in d['attempts']}
    cps={p['step']:p for p in d['checkpoints'] if p['adapter_exists']}
    ledger=[]
    for step,cp in sorted(cps.items()):
        # First attempt to reach a sealed checkpoint. Recovery import duplicates
        # have identical step labels; discard tails do not create sealed weights.
        elapsed=0;lo=hi=est=None
        for job in S[m]['jobs']:
            rows=attempts[job['job']]
            if any(r['step']==step for r in rows):
                total=sum(r['seconds'] for r in rows)
                used=sum(r['seconds'] for r in rows if r['step']<=step)
                lo=elapsed+used/3600*job['gpus']
                hi=elapsed+job['gpu_hours']
                est=elapsed+job['gpu_hours']*used/total
                break
            elapsed+=job['gpu_hours']
        assert est is not None
        rows=[r for r in d['rows'] if r['step']<=step]
        ids=[q for r in rows for q in r['question_ids']]
        ledger.append({'step':step,'path':cp['path'],'loop_gpu_hours':sum(r['seconds'] for r in rows)/3600,
                       'allocation_estimate':est,'allocation_bound_low':lo,'allocation_bound_high':hi,
                       'qid_exposures':len(ids),'unique_qids':len(set(ids)),'paired_bias_rows':len(ids)*2})
    out[m]=ledger
ledger=[];cost=0
for j in S['rmct']['jobs']:
    cost+=j['gpu_hours'];step=(len(ledger)+1)*16
    cp=next(p for p in D['rmct']['checkpoints'] if p['step']==step)
    ledger.append({'step':step,'path':cp['path'],'allocation_estimate':cost,'allocation_bound_low':cost,'allocation_bound_high':cost,
                   'qid_exposures_configured':2*step,'paired_bias_rows_configured':4*step})
out['rmct']=ledger
budget=ledger[0]['allocation_estimate']
ceiling={m:max((r for r in rows if r['allocation_bound_high']<=budget+1e-8),key=lambda r:r['step'],default=None) for m,rows in out.items()}
for r in ceiling.values():
    if r:r['shortfall_fraction']=1-r['allocation_estimate']/budget
internal_budget=out['mlpct'][-1]['loop_gpu_hours']
internal={m:max((r for r in out[m] if r['loop_gpu_hours']<=internal_budget+1e-10),key=lambda r:r['step']) for m in ['act','attct','mlpct']}
result={'accounting':'GPU allocation/loop resources, NOT measured model FLOPs','inventory_sha256':hashlib.sha256((R/'inventory.json').read_bytes()).hexdigest(),'ledger':out,'resource_ceiling':budget,'resource_ceiling_selections_not_compute_matched':ceiling,'internal_loop_budget':internal_budget,'internal_only_proxy_selections':internal}
(R/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
print('resource ceiling',budget)
for m,r in ceiling.items():print(m,{k:v for k,v in r.items() if k!='path'})
print('internal',[(m,r['step'],r['loop_gpu_hours'],r['unique_qids']) for m,r in internal.items()])
