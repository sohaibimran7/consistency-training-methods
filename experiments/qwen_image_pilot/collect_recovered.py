"""Export successful timeout recoveries; never overwrite original logs."""
from pathlib import Path
import argparse
from recovery import overlay
from full_suite import read,write
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True,help='Original suite artifact root')
source=p.add_mutually_exclusive_group(required=True)
source.add_argument('--retry-root',type=Path)
source.add_argument('--retry-rows',type=Path,help='Exported rows for offline collection')
p.add_argument('--output',type=Path,required=True,help='New directory; never overwrite')
a=p.parse_args();original=a.root;root=a.retry_root
original_rows=read(original/'collected/rows.json')
rows=[]
if a.retry_rows:rows=read(a.retry_rows)
else:
    from inspect_ai.log import read_eval_log
for path in sorted((root/'runs').glob('*/rank-*/logs/*.eval')) if root else []:
    log=read_eval_log(path)
    if log.status not in ('success','error'):raise ValueError(f'Unfinished log: {path}')
    for s in log.samples or []:
        key=path.parents[2].name
        if s.error:
            rows.append(dict(model=key,id=s.id,qid=s.metadata['question_id'],dataset=s.metadata['dataset'],condition=s.metadata['condition'],ground_truth=s.target,biased_option=s.metadata['biased_option'],error=s.error.model_dump(mode='json'),log=str(path)))
            continue
        score=next(iter(s.scores.values()))
        rows.append({'model':key,'id':s.id,'qid':s.metadata['question_id'],'dataset':s.metadata['dataset'],'condition':s.metadata['condition'],'ground_truth':s.target,'biased_option':s.metadata['biased_option'],'answer':score.answer,'scores':score.value,'error':None,'stop_reason':s.output.choices[0].stop_reason,'usage':s.output.usage.model_dump(mode='json') if s.output.usage else None,'log':str(path)})
merged,recovered,unresolved=overlay(original_rows,rows)
a.output.mkdir(parents=True,exist_ok=False)
write(a.output/'recovered-rows.json',recovered)
write(a.output/'unresolved-manifest.json',unresolved)
write(a.output/'merged-rows.json',merged)
print(f'{len(recovered)} recovered; {len(unresolved)} unresolved')
