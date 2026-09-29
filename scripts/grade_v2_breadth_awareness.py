"""One-pass saved pre-final reasoning grading with the existing belief rubric."""
import asyncio,json,argparse
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from inspect_ai.model import get_model,GenerateConfig,ChatMessageSystem,ChatMessageUser
from scripts.peer_action_belief import RUBRIC,response_schema,split_passages,validate_timeline
from scripts.grade_rogueqwen_flattery import sha,write
from scripts.prepare_v2_breadth_grading import BASE,ROOT

async def main(execute,source=None,out=None):
    source=source or BASE/'grading-v1/inventory.json';data=json.loads(source.read_text())
    out=out or BASE/'grading-v1/awareness';out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for r in data['rows']:
        if r['status']!='valid' or not r['reasoning'].strip():continue
        refs={}
        for n,s in enumerate(split_passages(r['reasoning']),1):
            key=f'T1.R{n:03}';refs[key]=dict(s,id=key,ordinal=n,assistant_turn=1,field='reasoning')
        jobs.append(dict(key=r['key'],identity=r['identity'],source=r['source'],refs=refs))
    plan=dict(source_sha256=sha(source),rubric=RUBRIC,schema=response_schema(),jobs=jobs,
        model='openrouter/openai/gpt-5.6-luna',reasoning_effort='medium',max_tokens=8192,retries=0,
        scope='Pre-final reasoning only; model/factor/outcome blinded. Latest expressed stance, not hidden awareness.',
        authorization='User approved V2 breadth transcripts and rubrics for paid Luna grading.')
    if not (out/'plan.json').exists():write(out/'plan.json',plan)
    assert json.loads((out/'plan.json').read_text())==plan
    print('planned',len(jobs),flush=True)
    if not execute:return
    for path,h in {j['source']['path']:j['source']['sha256'] for j in jobs}.items():assert sha(path)==h
    load_dotenv(ROOT/'.env',override=False)
    model=get_model(plan['model'],config=GenerateConfig(reasoning_effort='medium',max_tokens=8192,max_retries=0,max_connections=16,cache=False,
        extra_body={'provider':{'only':['openai'],'allow_fallbacks':False,'require_parameters':True}},
        response_schema={'name':'v2_breadth_belief','strict':True,'json_schema':response_schema()}),max_retries=0)
    sem=asyncio.Semaphore(16)
    async def one(j):
        async with sem:
            claim=out/'claims'/f"{j['key']}.json";dest=out/'results'/f"{j['key']}.json"
            if claim.exists():return
            write(claim,dict(key=j['key'],attempts=1,implementation_sha256=sha(__file__)))
            rec=dict(key=j['key'],identity=j['identity'])
            try:
                payload={'transcript':[{'role':'assistant','turn':1,'passages':list(j['refs'].values()),'tool_calls':[]}]}
                result=await model.generate([ChatMessageSystem(content=RUBRIC),ChatMessageUser(content=json.dumps(payload))],cache=False)
                rec['raw_output']=result.model_dump(mode='json')
                assert result.model in (plan['model'],'openai/gpt-5.6-luna')
                assert len(result.choices)==1 and result.choices[0].stop_reason=='stop'
                value=validate_timeline(result.completion,j['refs'])
                rec.update(status='validated',judgment=value,label=value['updates'][-1]['label'] if value['updates'] else 'NO_EVIDENCE')
            except Exception as e:rec.update(status='judge_invalid',error=type(e).__name__+': '+str(e))
            write(dest,rec)
    pending=[j for j in jobs if not (out/'claims'/f"{j['key']}.json").exists()]
    await asyncio.gather(*(one(j) for j in pending[:16]))
    first=[json.loads((out/'results'/f"{j['key']}.json").read_text()) for j in pending[:16]]
    if first and not any(r['status']=='validated' for r in first):print('STOP: first wave invalid');return
    await asyncio.gather(*(one(j) for j in pending[16:]))
    counts=Counter(json.loads(p.read_text())['status'] for p in (out/'results').glob('*.json'))
    if not (out/'completed.json').exists():write(out/'completed.json',dict(counts))
    print(dict(counts),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--inventory',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();asyncio.run(main(args.execute,args.inventory,args.output))
