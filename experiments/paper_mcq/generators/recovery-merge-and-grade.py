"""Merge immutable successful generations; grade new outputs; standard plots."""
import asyncio,copy,hashlib,importlib.util,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO))
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from mcq_bias.scorers import mcq_bias_scorer
from ctm_data.adapters.mcq_bias.luna_scorer_no_cap import luna_bias_acknowledged_no_cap_scorer
from dotenv import load_dotenv

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

async def merge():
    load_dotenv('/Users/work/consistency-training-methods/.env')
    data=read(ROOT.parent/'paper-behavioural-plots-20260917/samples.json')
    maps={m:{(r['dataset'],r['bias'],r['qid']):r for r in rows} for m,rows in data.items()}
    originals=copy.deepcopy(maps)
    updates={};sources={}
    for p in sorted((ROOT/'inputs').glob('*/raw/shard-*/*.eval')):
        log=read_eval_log(str(p));sources[str(p)]=sha(p)
        for s in log.samples or []:
            if s.error or not s.output or not s.output.choices or s.output.error:continue
            md=s.metadata
            if 'rmct_topup' in md:
                info=md['rmct_topup'];method='rmct352';kind='topup'
            else:
                info=md['cap64k_source'];method=info['method'];kind='recovery'
            key=(method,info['dataset'],info['bias'],info['qid'])
            assert key not in updates,('duplicate',key)
            updates[key]=(s,log,dict(path=str(p),sha256=sources[str(p)]),kind)
    assert sum(v[3]=='topup' for v in updates.values())==700
    assert sum(v[3]=='recovery' for v in updates.values())==552
    print('Loaded 1252 valid new outputs; parsing and grading',flush=True)
    parse=mcq_bias_scorer();grader=luna_bias_acknowledged_no_cap_scorer(max_connections=500)
    clean={};audit=[];semaphore=asyncio.Semaphore(500)
    async def one(key,record):
        method,ds,bias,qid=key;s,log,source,kind=record
        state=TaskState(model=log.eval.model,sample_id=s.id,epoch=s.epoch,input=s.input,
            messages=s.messages,output=s.output,metadata=s.metadata)
        score=await parse(state,Target(s.target))
        answer=score.answer if score.value['answer_parsed']==1 else None
        ack=None
        if bias is not None and answer is not None:
            identity=hashlib.sha256(json.dumps(dict(key=key,output=s.output.model_dump(mode='json')),sort_keys=True).encode()).hexdigest()
            receipt=ROOT/'grades'/f'{identity}.json'
            if receipt.exists():grade=read(receipt)
            else:
                state.scores={'mcq_bias_scorer':score}
                async with semaphore:result=await grader(state,Target(s.target))
                value=result.value['bias_acknowledged']
                if value not in (0,1):raise ValueError(('invalid grade',key))
                grade=dict(key=key,source=source,score=result.model_dump(mode='json'))
                write(receipt,grade)
                print('Graded',method,ds,bias,qid,flush=True)
            ack=grade['score']['value']['bias_acknowledged']
        audit.append(dict(method=method,dataset=ds,bias=bias,qid=qid,kind=kind,source=source,
                          parsed=answer is not None,graded=ack is not None,
                          tokens=s.output.usage.output_tokens,stop=s.output.choices[0].stop_reason))
        if bias is None:clean[method,ds,qid]=answer
        else:
            k=(ds,bias,qid)
            if kind=='topup':
                assert k not in maps[method]
                r=copy.deepcopy(maps['base'][k]);r.update(b=answer,u=None,ack=ack)
            else:
                r=copy.deepcopy(maps[method][k]);r.update(b=answer,ack=ack)
            maps[method][k]=r
    results=await asyncio.gather(*(one(k,v) for k,v in updates.items()),return_exceptions=True)
    errors=[r for r in results if isinstance(r,BaseException)]
    if errors:
        write(ROOT/'errors.json',[str(e) for e in errors]);raise RuntimeError(f'{len(errors)} failed records: {errors[0]}')
    for method,rows in maps.items():
        for (ds,bias,qid),r in rows.items():
            if (method,ds,qid) in clean:r['u']=clean[method,ds,qid]
            if method=='rmct352' and (ds,bias,qid) not in originals[method]:
                assert (method,ds,qid) in clean
    missing=[r for r in read(ROOT.parent/'unparsed-diagnosis-20260917/diagnosis.json')
             if r['reason']=='token_limit' and (r['method'],r['dataset'],r['bias'],r['qid']) not in updates]
    assert len(missing)==12 and all(r['method']=='bct' for r in missing)
    write(ROOT/'missing-bct.json',missing)
    write(ROOT/'samples.json',{m:list(rows.values()) for m,rows in maps.items()})
    write(ROOT/'merge-audit.json',audit);write(ROOT/'source-hashes.json',sources)
    print('MERGED',len(updates),'new generations; missing BCT=12',flush=True)

def plot():
    spec=importlib.util.spec_from_file_location('prior',ROOT.parent/'paper-behavioural-plots-20260917/build.py')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    b.ROOT=ROOT;b.PERM=108000
    data=read(ROOT/'samples.json')
    from ctm_data.adapters.mcq_bias.plot import render_publication_plot
    template=read(ROOT.parent/'methods-verbalisation-all-seven-20260916/bias_acknowledged-vs-base/chart-spec.json')
    def value(r,metric):
        if metric=='bias_acknowledged':return float(r['ack']) if r['b'] is not None and r['ack'] in (0,1) else None
        if r['b'] is None or r['u'] is None or r['u']==r['option']:return None
        return float(r['b']==r['option'])
    b.value=value
    for metric,name,ylabel in [('towards_bias_switch','towards-bias-switch-rate','Towards-bias switch rate'),
                               ('bias_acknowledged','bias-verbalisation','Bias verbalised (Luna YES | valid grade)')]:
        rows=[]
        for pop in ['held_in_datasets','held_out_dataset']:
            for bias in template['bias_order']:
                def select(m):
                    return [r for r in data[m] if (r['dataset']=='hle-text-mc')==(pop=='held_out_dataset')
                        and (r['bias']==bias or bias=='overall_mean' or bias=='seen_mean' and r['bias'] in b.SEEN
                             or bias=='held_out_mean' and r['bias'] not in b.SEEN)]
                for m in b.METHODS:
                    rr=select(m);tag=f'{metric}/{pop}/{bias}/{m}'
                    r=dict(population=pop,bias_type=bias,condition=m,**b.estimate(rr,metric,tag))
                    if m!='base':r.update(b.contrast(rr,select('base'),metric,tag))
                    rows.append(r)
        b.holm(rows)
        for r in rows:
            r.update(model='qwen3.5-9b',metric=metric,mean=r['estimate'],stderr=0.,ci_lower=r['low'],ci_upper=r['high'],
                n_scored=r['n_pairs'],n_total=r['n_pairs'],significance=r.get('marker','') if r.get('marker')!='ns' else '',
                bias_status='seen' if r['bias_type'] in b.SEEN else 'aggregate' if r['bias_type'].endswith('_mean') else 'held_out')
            if r['bias_type'].endswith('_mean'):r['bias_group']=r['bias_type'].removesuffix('_mean')
        chart=copy.deepcopy(template);chart.update(metric=metric,ylabel=ylabel)
        chart['ylim']=[0,1.15]
        chart['condition_labels']['bct']='BCT (12 reruns missing)'
        chart['bias_labels'].update(seen_mean='Seen pooled',held_out_mean='Held-out pooled',overall_mean='Overall pooled')
        chart['significance_note']='PROVISIONAL selective 64k recovery + RMCT IID top-up. All methods: 100 QIDs per dataset. Original 20k outputs retained unless a completed targeted 64k rerun exists.\n12 missing BCT reruns retain original unparsed outputs; excluded from parsed-response metrics, never imputed. Other non-cap parse failures unchanged. New top-up responses use 20k.\n95% dataset-stratified QID-cluster bootstrap (10,000). Stars vs Base: 108,000 whole-QID swaps, Holm-108 per figure; * p<.05, ** p<.01, *** p<.001.\nTBSR: jointly parsed, clean answer not the biased option. Verbalisation: parsed biased responses with valid grades. Old grades retained; new responses graded with uncapped Luna.'
        write(ROOT/f'{name}-rows.json',rows);write(ROOT/f'{name}-spec.json',chart)
        for ext in ['png','pdf','svg']:render_publication_plot(rows,chart,ROOT/f'{name}.{ext}')
        print('PLOTTED',name,flush=True)

if __name__=='__main__':
    if sys.argv[1]=='merge':asyncio.run(merge())
    elif sys.argv[1]=='plot':plot()
