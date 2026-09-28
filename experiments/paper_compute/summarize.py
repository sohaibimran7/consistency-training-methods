import json,re,hashlib
from pathlib import Path

from common import artifact_directory
ROOT=artifact_directory("compute-audit-20260917")
raw=json.loads((ROOT/'remote.json').read_text())
def accounting(text):
    rows=[]
    for line in text.splitlines():
        if not line.strip():continue
        job,name,state,elapsed,tres,start,end=line.split('|')
        gpu=int(re.search(r'(?:^|,)gres/gpu=(\d+)',tres)[1])
        rows.append(dict(job=job,state=state,seconds=int(elapsed),gpus=gpu,gpu_hours=int(elapsed)*gpu/3600))
    assert len({r['job'] for r in rows})==len(rows)
    return rows
summary={}
for m,d in raw.items():
    jobs=accounting(d['accounting'])
    assert len(jobs)==len({a['job'] for a in d['attempts']})
    expected={'act':992,'attct':880,'mlpct':736,'bct':288,'opct':384}[m]
    assert d['sealed_steps']==list(range(1,expected+1))
    summary[m]={'terminal_step':expected,'gpu_hours_audited_lineage':sum(j['gpu_hours'] for j in jobs),
                'committed_update_loop_hours':d['sealed_seconds']/3600,'jobs':jobs,
                'scope':'Jobs with recorded metrics in prefix-v2 (internal methods), or cap-v4 plus recovery-v5 (BCT/OPCT). Excludes evaluation, shared preparation and earlier failed namespaces.'}
jobs=accounting((ROOT/'rmct-continuation-sacct.txt').read_text())
assert len(jobs)==11
trace=json.loads((ROOT/'initial-trace.json').read_text())
receipts=json.loads((ROOT/'rmct-initial.json').read_text())['segments']
initial_accounting={r['job']:r for r in accounting(trace['accounting'])}
initial_jobs=[]
for name,receipt in receipts.items():
    matches=[r for r in trace['logs'] if r['runs']==[name]]
    assert len(matches)==1,(name,matches)
    job=initial_accounting[matches[0]['job']]
    assert job['state']=='COMPLETED'
    assert receipt['completion-receipt.json']['sealed']
    initial_jobs.append(dict(job,run=name,receipt=receipt['completion-receipt.json']))
assert len(initial_jobs)==11
assert sorted(r['receipt']['optimizer_step_end'] for r in initial_jobs)==list(range(16,177,16))
initial_hours=sum(j['gpu_hours'] for j in initial_jobs)
continuation_hours=sum(j['gpu_hours'] for j in jobs)
summary['rmct']={'terminal_step':352,'start_step':1,'gpu_hours_audited_lineage':initial_hours+continuation_hours,
                 'initial_176_gpu_hours':initial_hours,'continuation_gpu_hours':continuation_hours,
                 'jobs':initial_jobs+jobs,'scope':'All 22 successful production segments, steps 1-352. Failed/debug attempts and evaluation excluded.'}
base=summary['mlpct']['gpu_hours_audited_lineage']
for m,d in summary.items():
    d['relative_to_mlpct']=d.get('gpu_hours_audited_lineage',d.get('gpu_hours_lower_bound'))/base
summary['provenance']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'remote.json',ROOT/'rmct-continuation-sacct.txt',ROOT/'rmct-initial.json',ROOT/'initial-trace.json']}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({m:{k:v for k,v in d.items() if k not in ['jobs','scope']} for m,d in summary.items() if m!='provenance'},indent=2))
