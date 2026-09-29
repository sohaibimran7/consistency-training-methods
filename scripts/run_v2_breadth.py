"""Generate an explicitly reviewed V2-breadth panel; no authoring or grading.

Offline audit is the default. One native Inspect epoch per distinct source/arm.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

import inspect_ai
from inspect_ai import Task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.solver import generate
from inspect_ai.log import read_eval_log
from scripts.lasr_transfer_tasks import MODELS, TARGET_CONFIG
from scripts.run_lasr_transfer import validate_attestation
from scripts.prepare_v2_breadth import digest
from scripts.lasr_transfer_tasks import canonical_messages

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def panel_tasks(panel,role):
    if panel.get('status')!='frozen_reviewed_for_execution':raise ValueError('panel is not reviewed and frozen')
    if panel['models']!={k:list(v) for k,v in MODELS.items()}:raise ValueError('model pins differ')
    if panel['target_config']!=TARGET_CONFIG:raise ValueError('sampler differs')
    if panel['epochs']!=1:raise ValueError('breadth campaign must use one epoch')
    seen=set();samples=[]
    for row in panel['rows']:
        if digest(row['messages'])!=row['messages_sha256'] or digest(row['source_messages'])!=row['source_sha256']:
            raise ValueError('message hash binding changed')
        identity=(row['source_id'],row['condition'])
        if identity in seen:raise ValueError('duplicate source-arm')
        seen.add(identity)
        if row['review']['decision'] not in ('accepted','accepted_diagnostic'):raise ValueError('unaccepted cell')
        if not row['review']['reason'] or not row['review']['reviewer']:raise ValueError('missing substantive review')
        messages=[]
        for m in row['messages']:
            if m['role']=='system':messages.append(ChatMessageSystem(content=m['content']))
            elif m['role']=='user':messages.append(ChatMessageUser(content=m['content']))
            else:raise ValueError('unexpected message role for text-only panel')
        metadata=dict(protocol='v2_breadth_singletons_v1',source_id=row['source_id'],family=row['family'],
            condition=row['condition'],source_messages=row['source_messages'],source_metadata=row['source_metadata'],
            review=row['review'],model_role=role,seed=20260910,source_sha256=row['source_sha256'])
        samples.append(Sample(id=row['source_id']+'--'+row['condition'],input=messages,target=row.get('target',''),metadata=metadata))
    if not samples:raise ValueError('empty panel')
    for source_id,_ in seen:
        if (source_id,'B') not in seen:raise ValueError('factor lacks matched baseline')
    # Native AM uses generate(tool_calls="none"); this text-only panel has no
    # tools, no sandbox, and no extra system-message solver or injected instructions.
    return [Task(name='v2_breadth_'+role,dataset=MemoryDataset(samples),solver=generate(tool_calls='none'),
        scorer=None,metrics=[],epochs=1,message_limit=3,config=GenerateConfig(**TARGET_CONFIG,seed=20260910),
        metadata={'protocol':'v2_breadth_singletons_v1','source_count':len({x[0] for x in seen})})]

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('endpoint redirects forbidden')

def main(args):
    path_map=json.loads(args.path_map.read_text()) if args.path_map else {}
    if not isinstance(path_map,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in path_map.items()):
        raise ValueError('path map must map exact archived path strings to local paths')
    def local(path):return Path(path_map.get(str(path),str(path)))
    panel=json.loads(args.panel.read_text());tasks=panel_tasks(panel,args.role)
    for path,expected in panel.get('bindings',{}).items():
        if sha(local(path))!=expected:raise ValueError('frozen source/review binding changed')
    summary=dict(stage='offline_audit',panel_sha256=sha(args.panel),role=args.role,
        samples=sum(len(t.dataset) for t in tasks),epochs=1,scoring=False)
    if args.reuse:
        reuse=json.loads(args.reuse.read_text())
        if reuse['panel_sha256']!=sha(args.panel):raise ValueError('reuse panel mismatch')
        skip=set()
        for r in reuse['rows']:
            if r['role']!=args.role:continue
            if sha(local(r['source_log']))!=r['source_log_sha256'] or sha(local(r['manifest_path']))!=r['manifest_sha256']:raise ValueError('reuse evidence changed')
            if (r['model'],r['revision'])!=MODELS[args.role]:raise ValueError('reuse checkpoint differs')
            log=read_eval_log(local(r['source_log']),resolve_attachments=True)
            sample=next(s for s in log.samples if (s.id,s.epoch)==(r['sample_id'],r['epoch']))
            events=[e for e in sample.events if e.event=='model']
            if len(events)!=1 or events[0].config.seed!=20260910:raise ValueError('reuse event contract differs')
            event=events[0]
            if any(getattr(event.config,k)!=v for k,v in TARGET_CONFIG.items()):raise ValueError('reuse decoding differs')
            sid=r['source_id']+'--'+r['condition']
            expected=next(s for s in tasks[0].dataset if s.id==sid)
            if canonical_messages(event.input)!=canonical_messages(expected.input):raise ValueError('reuse prompt differs')
            if sid in skip:raise ValueError('duplicate reuse')
            skip.add(sid)
        tasks[0].dataset=MemoryDataset([s for s in tasks[0].dataset if s.id not in skip])
        summary.update(reused=len(skip),new_samples=len(tasks[0].dataset),reuse_sha256=sha(args.reuse))
    if not args.execute:print(json.dumps(summary));return
    if not args.runtime_dir or not args.job_id or not args.output:raise ValueError('live execution requires runtime, exact job, fresh output')
    if args.output.exists():raise ValueError('output already exists; never automatically rerun')
    allocation=json.loads((args.runtime_dir/'allocation.json').read_text())
    att=json.loads((args.runtime_dir/args.role/'attestation.json').read_text())
    validate_attestation(att,args.role)
    if allocation['job_id']!=args.job_id or att['job_id']!=args.job_id:raise ValueError('wrong serving job')
    if att['node'].split('.')[0]!=allocation['nodes'][args.role]:raise ValueError('wrong serving node')
    key_path=args.runtime_dir/'api-key'
    if key_path.stat().st_mode & 0o077:raise ValueError('credential file permissions too broad')
    key=key_path.read_text().strip()
    endpoint=f"http://127.0.0.1:{dict(base=18100,mo_mid=18101,mo_post=18102)[args.role]}/v1"
    opener=build_opener(ProxyHandler({}),NoRedirect())
    with opener.open(Request(endpoint+'/models',headers={'Authorization':'Bearer '+key}),timeout=20) as f:data=json.load(f)
    if len(data['data'])!=1 or data['data'][0]['id']!=MODELS[args.role][0] or data['data'][0].get('max_model_len')!=32768:raise ValueError('live endpoint mismatch')
    args.output.mkdir(parents=True)
    summary.update(stage='generation_dispatched',job_id=args.job_id,attestation=att,implementation_sha256=sha(__file__))
    (args.output/'manifest.json').write_text(json.dumps(summary,indent=2))
    model=get_model('vllm/'+MODELS[args.role][0],base_url=endpoint,api_key=key,config=GenerateConfig(**TARGET_CONFIG,seed=20260910))
    logs=inspect_ai.eval(tasks,model=model,epochs=1,score=False,log_dir=str(args.output/'logs'),display='plain')
    result=dict(statuses=[l.status for l in logs],paths=[l.location for l in logs],scored=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--role',choices=MODELS,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--runtime-dir',type=Path);p.add_argument('--job-id');p.add_argument('--output',type=Path);p.add_argument('--reuse',type=Path)
    p.add_argument('--path-map',type=Path,help='Exact archived-to-local path JSON; original hashes remain mandatory')
    main(p.parse_args())
