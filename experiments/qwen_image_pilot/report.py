"""Validate embedded images and summarize completed pilot logs without inference."""
import argparse
import base64
import collections
import hashlib
import json
import importlib.metadata
from pathlib import Path
from inspect_ai.log import read_eval_log

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root
manifest=json.loads((root/'manifest.json').read_text())
expected={c['id']:c for c in manifest['cases']}
rows=[];image_checks=0;probs={};statuses=[];api_events=0;request_configs=[]
for path in sorted((root/'logs').glob('*.eval')):
    log=read_eval_log(path,resolve_attachments=True)
    statuses.append({'log':str(path),'status':log.status,'samples':len(log.samples or [])})
    for s in log.samples or []:
        case=expected[s.id]
        # Check the actual logged input bytes, not only the metadata claim.
        blobs=[]
        for msg in s.input:
            if not isinstance(msg.content,list): continue
            for content in msg.content:
                if content.type=='image':
                    assert content.image.startswith('data:image/png;base64,')
                    blobs.append(hashlib.sha256(base64.b64decode(content.image.split(',',1)[1])).hexdigest())
        assert blobs==[manifest['images'][x]['sha256'] for x in case['images']],s.id
        image_checks+=len(blobs)
        api_events+=sum(e.event=='model' and getattr(e,'call',None) is not None for e in s.events)
        for e in s.events:
            if e.event=='model' and getattr(e,'call',None) is not None:
                req=e.call.request
                request_configs.append({k:req[k] for k in ('model','temperature','top_p','seed','max_tokens','max_completion_tokens','extra_body','logprobs','top_logprobs') if k in req})
        choice=s.output.choices[0] if s.output and s.output.choices else None
        if choice and choice.logprobs and choice.logprobs.content:
            first=choice.logprobs.content[0]
            probs[log.eval.model.rsplit('/',1)[-1],s.id]={t.token:t.logprob for t in first.top_logprobs or []}
        score=next(iter((s.scores or {}).values()),None)
        rows.append({'model':log.eval.model,'id':s.id,'condition':case['condition'],'qid':case['question_id'],
            'answer':score.answer if score else None,'scores':score.value if score else {},
            'stop_reason':choice.stop_reason if choice else None,'error':str(s.error) if s.error else None,
            'input_tokens':s.output.usage.input_tokens if s.output and s.output.usage else None,
            'output_tokens':s.output.usage.output_tokens if s.output and s.output.usage else None})

effects=[]
for case in manifest['cases']:
    b=probs.get(('base',case['id']),{});r=probs.get(('rmct',case['id']),{})
    shared=b.keys()&r.keys()
    if shared:
        effects.append({'id':case['id'],'max_shared_first_token_logprob_delta':max(abs(b[k]-r[k]) for k in shared)})
summary={'logs':statuses,'sample_count':len(rows),'expected_samples':2*len(expected),'embedded_image_hash_checks':image_checks,
    'logged_model_api_events':api_events,'adapter_output_distribution_checks':effects,
    'adapter_effect_observed':any(x['max_shared_first_token_logprob_delta']>1e-3 for x in effects),
    'errors':sum(bool(x['error']) for x in rows),'truncations':sum(x['stop_reason'] in ('max_tokens','model_length') for x in rows)}
summary['runtime_versions']={name:importlib.metadata.version(name) for name in ('inspect-ai','transformers','torch','vllm','pillow','safetensors')}
summary['unique_logged_request_configs']=list({json.dumps(v,sort_keys=True):v for v in request_configs}.values())
summary['pilot_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}
summary['upstream_source_sha256']={str(p.relative_to(root/'vendor')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'vendor').rglob('*.py')}
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'log-validation.json').write_text(json.dumps(summary,indent=2)+'\n')
(a.output/'scored-rows.json').write_text(json.dumps(rows,indent=2)+'\n')
groups=collections.defaultdict(list)
for row in rows:groups[row['model'],row['condition']].append(row)
lines=['# Qwen image pilot','',f"{len(rows)}/{2*len(expected)} samples; {image_checks} embedded-image hashes verified.",'',
    'Three HLE questions selected by length; these are feasibility observations, not population estimates.','',
    '| Model | Configuration | n | Correct | Parsed | Bias answer | Truncated |','|---|---|---:|---:|---:|---:|---:|']
for (model,condition),group in sorted(groups.items()):
    counts=[sum(x['scores'].get(k,0) for x in group) for k in ('accuracy','parsed','bias_answer')]
    if condition=='are_you_sure':counts[2]='n/a'
    trunc=sum(x['stop_reason'] in ('max_tokens','model_length') for x in group)
    lines.append(f'| {model} | {condition} | {len(group)} | {counts[0]} | {counts[1]} | {counts[2]} | {trunc} |')
lines.extend(['','## Viewable images',''])
for q in manifest['selected_qids']:
    for suffix in ('clean','spurious__composite','wrong__composite'):
        name=q+'__'+suffix+'.png'
        lines.append(f'- [{q[:8]} {suffix}: rendered](images/{name}) / [processed](processed/{name})')
lines.extend(['','## Inspect logs',''])
for item in statuses:lines.append(f"- [{Path(item['log']).name}](logs/{Path(item['log']).name}) ({item['status']})")
(a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='adapter_output_distribution_checks'},indent=2))
