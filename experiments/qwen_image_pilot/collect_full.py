"""Read completed image-suite logs without changing original evidence."""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
from inspect_ai.log import read_eval_log
from full_suite import write
p=argparse.ArgumentParser()
p.add_argument('--run-root',type=Path,action='append',required=True,help='Repeat for each original/recovery run root')
p.add_argument('--output',type=Path,required=True,help='New collected output directory')
p.add_argument('--expected-count',type=int,default=10800)
a=p.parse_args();roots=a.run_root
rows=[];logs=[]
for root in roots:
    for path in sorted((root/'runs').glob('*/rank-*/logs/*.eval')):
        log=read_eval_log(path)
        key=path.parents[2].name
        logs.append({'path':str(path),'model':key,'status':log.status,'samples':len(log.samples or [])})
        for s in log.samples or []:
            score=next(iter((s.scores or {}).values()),None)
            choices=s.output.choices if s.output else []
            rows.append({'model':key,'id':s.id,'qid':s.metadata['question_id'],'dataset':s.metadata['dataset'],'condition':s.metadata['condition'],'ground_truth':s.target,'biased_option':s.metadata['biased_option'],'answer':score.answer if score else None,'scores':score.value if score else {},'error':s.error.model_dump(mode='json') if s.error else None,'stop_reason':choices[0].stop_reason if choices else None,'usage':s.output.usage.model_dump(mode='json') if s.output and s.output.usage else None,'log':str(path)})
        print(key,len(log.samples or []),flush=True)
        del log
assert len(rows)==a.expected_count
assert len({(r['model'],r['id']) for r in rows})==a.expected_count
summary=[]
for model in sorted({r['model'] for r in rows}):
    rr=[r for r in rows if r['model']==model]
    summary.append({'model':model,'samples':len(rr),'errors':sum(bool(r['error']) for r in rr),'parsed':sum(r['scores'].get('parsed',0) for r in rr),'correct':sum(r['scores'].get('accuracy',0) for r in rr),'stop_reasons':dict(Counter(r['stop_reason'] for r in rr))})
out=a.output;out.mkdir(parents=True,exist_ok=False)
write(out/'rows.json',rows);write(out/'summary.json',summary);write(out/'logs.json',logs);write(out/'errors.json',[r for r in rows if r['error']])
print(json.dumps(summary,indent=2))
