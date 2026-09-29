
from matplotlib.figure import Figure as _Figure
if not getattr(_Figure, '_ctm_parser_warning', False):
 _old_save = _Figure.savefig
 def _fixed_save(self, *args, **kwargs):
  if not getattr(self, '_ctm_warned', False):
   self.text(.5, .998, 'Corrected MCQ answers + selective 64k reruns + RMCT IID top-up (100 QIDs/dataset). 12 BCT reruns missing. Base clean-dependent values PROVISIONAL.', ha='center', va='top', fontsize=7, color='#9b2226', bbox=dict(facecolor='white',alpha=.95,edgecolor='none'))
   self._ctm_warned=True
  return _old_save(self,*args,**kwargs)
 _Figure.savefig=_fixed_save
 _Figure._ctm_parser_warning=True
"""CPU-only paper-style analyses of immutable saved evaluation outputs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctm-paper-behavioural-mpl-20260917')
import csv, hashlib, json, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/work/.codex/worktrees/d6d6/consistency-training-methods')
sys.path.insert(0,str(REPO))
from inspect_ai.log import read_eval_log
from experiments.rmct_two_bias_eval import checkpoint_publication as standard
from ctm_data.adapters.mcq_bias.method_presentation import method_color

METHODS=['base','act','attct','mlpct','bct','opct','rmct352']
LABELS=['Base','ACT','AttCT','MLPCT','BCT','OPCT','RMCT']
DATASETS=['logiqa','hellaswag','hle-text-mc']
TITLES=['LogiQA','HellaSwag','HLE (held out)']
SEEN=set(standard.SEEN_BIASES)
BOOT=10000
PERM=1000000
sources={}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def verify(r):
    p=r['path']; h=sha(p); assert h==r['sha256'],p; sources[p]=h; return p
def seed(s):return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)
def extract():
    manifest=read(REPO/'artifacts/methods-complete-with-opct-20260916/towards_bias_switch-vs-base/manifest.json')
    verbal=read(REPO/'artifacts/methods-verbalisation-all-seven-20260916/complete.json')
    result={};coverage=[]
    for method in METHODS:
        clean={}
        for entry in manifest['audit']['pairing'].get(method,[]):
            ds=entry['dataset']
            if ds in clean:continue
            log=read_eval_log(verify(entry['clean_log']))
            clean[ds]={str(s.id):(s.scores['mcq_bias_scorer'].answer,s.target) for s in log.samples}
        for r in manifest['source_logs'][method]:verify(r)
        rows=[]; base_clean={}
        for r in verbal['sources'][method]:
            log=read_eval_log(verify(r));ds=standard._dataset_from_log(log);bias=standard._bias_from_log(log)
            for s in log.samples:
                q=str(s.id);score=s.scores['mcq_bias_scorer'];b=score.answer
                if score.value['answer_parsed']!=1:b=None
                gold=score.metadata['ground_truth'];option=s.metadata['biased_option']
                assert gold==s.target
                if method=='base':
                    u=s.scores['switch_scorer'].metadata.get('unbiased_answer')
                    if (ds,q) in base_clean:assert base_clean[ds,q]==u
                    base_clean[ds,q]=u
                else:
                    u,target=clean[ds][q];assert target==gold
                for answer in [b,u]:assert answer is None or (isinstance(answer,str) and len(answer)==1 and answer.isalpha()),answer
                ack=standard._sample_metric(s,'bias_acknowledged') if b is not None else None
                if b is not None:assert ack in (0,1),(method,ds,q,ack)
                prompt=s.model_dump(mode='json',include={'input'})['input']
                if isinstance(prompt,list):prompt=[{k:v for k,v in x.items() if k!='id'} for x in prompt]
                rows.append(dict(dataset=ds,bias=bias,qid=q,b=b,u=u,gold=gold,option=option,ack=ack,
                    prompt_hash=hashlib.sha256(json.dumps(prompt,sort_keys=True).encode()).hexdigest()))
        assert len({(r['dataset'],r['bias'],r['qid']) for r in rows})==len(rows)
        result[method]=rows
        for ds in DATASETS:
            rr=[r for r in rows if r['dataset']==ds];qs={r['qid'] for r in rr}
            coverage.append(dict(method=method,dataset=ds,questions=len(qs),generated_pairs=len(rr),
                biased_parsed=sum(r['b'] is not None for r in rr),jointly_parsed=sum(r['b'] is not None and r['u'] is not None for r in rr),
                clean_parsed_questions=len({r['qid'] for r in rr if r['u'] is not None}),promoted_gold=sum(r['option']==r['gold'] for r in rr)))
        print('loaded',method,len(rows),flush=True)
    base={(r['dataset'],r['bias'],r['qid']):r for r in result['base']}
    for method,rows in result.items():
        for r in rows:
            ref=base[r['dataset'],r['bias'],r['qid']]
            assert all(r[k]==ref[k] for k in ['gold','option','prompt_hash']),(method,r)
    write('samples.json',result);write('coverage.json',coverage);write('sources.json',sources)
    return result

def value(r,metric):
    b,u,g,s=r['b'],r['u'],r['gold'],r['option']
    if metric=='accuracy_all_generated':return float(b==g)
    if b is None or u is None:return None
    c=b==g;uc=u==g;sw=b==s
    vals={'clean_accuracy':float(uc),'biased_accuracy':float(c),'accuracy_gap':float(uc)-float(c),
          'brr':float(sw)-float(u==s),'bias_rate':float(sw),'invariance':float(b==u),'avoidance':float(not sw),
          'correct_stable':float(uc and c),'correct_to_wrong':float(uc and not c),
          'wrong_to_correct':float(not uc and c),'wrong_stable':float(not uc and not c and b==u),
          'wrong_to_other_wrong':float(not uc and not c and b!=u)}
    if metric in vals:return vals[metric]
    if u==s or r['ack'] is None:return None
    a=r['ack']==1
    return {'ack':float(a),'towards':float(sw),'ack_resist':float(a and not sw),'ack_switch':float(a and sw),
            'silent_resist':float(not a and not sw),'silent_switch':float(not a and sw)}[metric]

def select(rows,pop):
    if pop in DATASETS:return [r for r in rows if r['dataset']==pop]
    if pop=='iid_seen':return [r for r in rows if r['dataset']!='hle-text-mc' and r['bias'] in SEEN]
    if pop=='iid_held':return [r for r in rows if r['dataset']!='hle-text-mc' and r['bias'] not in SEEN]
    raise ValueError(pop)

def estimate(rows,metric,tag):
    clusters=defaultdict(lambda:[0,0])
    for r in rows:
        v=value(r,metric)
        if v is not None:
            a=clusters[r['dataset'],r['qid']];a[0]+=v;a[1]+=1
    assert clusters,(metric,tag)
    byds=defaultdict(list)
    for (ds,q),v in sorted(clusters.items()):byds[ds].append(v)
    rng=np.random.default_rng(seed(tag));num=np.zeros(BOOT);den=np.zeros(BOOT)
    for ds,values in byds.items():
        a=np.array(values)
        for lo in range(0,BOOT,500):
            ix=rng.integers(len(a),size=(min(500,BOOT-lo),len(a)));sample=a[ix].sum(axis=1)
            num[lo:lo+len(ix)]+=sample[:,0];den[lo:lo+len(ix)]+=sample[:,1]
    total=np.array(list(clusters.values())).sum(axis=0);lo,hi=np.quantile(num/den,[.025,.975])
    return dict(estimate=float(total[0]/total[1]),low=float(lo),high=float(hi),n_pairs=int(total[1]),n_qids=len(clusters))

def contrast(rows,base,metric,tag):
    def keyed(rr):return {(r['dataset'],r['bias'],r['qid']):r for r in rr if value(r,metric) is not None}
    a,b=keyed(rows),keyed(base);keys=sorted(a.keys()&b.keys());assert keys
    deltas=defaultdict(float)
    for k in keys:deltas[k[0],k[2]]+=value(a[k],metric)-value(b[k],metric)
    x=np.array(list(deltas.values())); observed=x.sum()
    # On jointly eligible records, denominators agree. Swapping method labels for
    # whole QIDs is exactly a sign flip of each cluster's integer-valued sum.
    # Group equal absolute sums: independent Binomial(count, .5) gives the exact
    # same randomization distribution, avoiding a million-by-QID matrix.
    weights,counts=np.unique(np.abs(x[x!=0]),return_counts=True)
    rng=np.random.default_rng(seed(tag));extreme=0
    for start in range(0,PERM,10000):
        n=min(10000,PERM-start);draw=np.zeros(n)
        for w,c in zip(weights,counts):draw+=w*(2*rng.binomial(int(c),.5,size=n)-c)
        extreme+=int(np.count_nonzero(abs(draw)>=abs(observed)-1e-12))
    return dict(p_raw=(extreme+1)/(PERM+1),paired_difference=float(observed/len(keys)),
                matched_treatment_estimate=sum(value(a[k],metric) for k in keys)/len(keys),
                matched_base_estimate=sum(value(b[k],metric) for k in keys)/len(keys),
                paired_n=len(keys),paired_qids=len(deltas),seed=seed(tag),permutations=PERM)

def holm(rows):
    ordered=sorted([r for r in rows if 'p_raw' in r],key=lambda r:r['p_raw']); running=0
    for i,r in enumerate(ordered):
        running=max(running,min(1,r['p_raw']*(len(ordered)-i)));r['p_holm']=running
        r['marker']='***' if running<.001 else '**' if running<.01 else '*' if running<.05 else 'ns'
        r['holm_family_size']=len(ordered)

def analyse(data):
    families={
      'accuracy':(['clean_accuracy','biased_accuracy','accuracy_gap'],DATASETS),
      'paper_metrics':(['brr','bias_rate','invariance'],DATASETS),
      'tradeoff':(['clean_accuracy','avoidance'],DATASETS),
      'generalisation':(['brr'],['iid_seen','iid_held','hle-text-mc']),
      'joint':(['ack','towards','ack_resist','ack_switch','silent_resist','silent_switch'],DATASETS),
      'transitions':(['correct_stable','correct_to_wrong','wrong_to_correct','wrong_stable','wrong_to_other_wrong'],DATASETS),
      'sensitivity':(['accuracy_all_generated'],DATASETS)}
    rows=[]
    for family,(metrics,pops) in families.items():
        part=[]
        for pop in pops:
            for metric in metrics:
                for method in METHODS:
                    tag=f'{family}/{pop}/{metric}/{method}';rr=select(data[method],pop)
                    row=dict(family=family,population=pop,metric=metric,method=method,**estimate(rr,metric,tag))
                    if method!='base':row.update(contrast(rr,select(data['base'],pop),metric,tag))
                    part.append(row)
        holm(part);rows.extend(part);write('statistics.json',rows)
        print('analysed',family,len(part),flush=True)
    write('statistics.json',rows)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (ROOT/'statistics.csv').open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    return rows

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
COLORS=[method_color(m) for m in METHODS]
def lookup(rows,f,p,k,m):return next(r for r in rows if (r['family'],r['population'],r['metric'],r['method'])==(f,p,k,m))
def save(fig,name,footer,top=.94):
    footer += '\nPROVISIONAL: Base clean responses unavailable; Base clean-dependent values and comparisons retain legacy clean labels.'
    fig.text(.5,.015,footer,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,top))
    for ext in ['png','svg','pdf']:fig.savefig(ROOT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
def ticks(ax):
    ax.set_xticks(range(7),LABELS,rotation=35,ha='right');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
def bars(ax,rows,f,p,k):
    for i,m in enumerate(METHODS):
        r=lookup(rows,f,p,k,m);y=r['estimate']*100
        ax.bar(i,y,color=COLORS[i],width=.72)
        ax.errorbar(i,y,yerr=[[max(0,y-r['low']*100)],[max(0,r['high']*100-y)]],color='#333',capsize=2,fmt='none',lw=.8)
        if m!='base':ax.annotate(r['marker'],(i,r['high']*100),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    ticks(ax);ax.margins(y=.22)
def render(rows):
    note='95% QID-cluster bootstrap intervals; stars vs Base: 1M paired cluster swaps, Holm within figure. Full pools shown; tests use shared eligible pairs.'
    fig,axs=plt.subplots(2,3,figsize=(16,8));fig.suptitle('Clean vs biased accuracy and paired accuracy penalty',fontsize=17)
    for j,(ds,title) in enumerate(zip(DATASETS,TITLES)):
        ax=axs[0,j]
        for i,m in enumerate(METHODS):
            for metric,offset,hatch in [('clean_accuracy',-.19,'///'),('biased_accuracy',.19,None)]:
                r=lookup(rows,'accuracy',ds,metric,m);y=r['estimate']*100
                ax.bar(i+offset,y,width=.36,color=COLORS[i],hatch=hatch,alpha=.55 if hatch else 1)
                ax.errorbar(i+offset,y,yerr=[[max(0,y-r['low']*100)],[max(0,r['high']*100-y)]],fmt='none',color='#333',lw=.7,capsize=1)
        ax.set_title(title);ticks(ax);ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,105)
        bars(axs[1,j],rows,'accuracy',ds,'accuracy_gap');axs[1,j].set_ylabel('Clean − biased accuracy (pp)');axs[1,j].axhline(0,color='#555',lw=.7)
    axs[0,0].legend(handles=[plt.Rectangle((0,0),1,1,color='#aaa',hatch='///',alpha=.55,label='Clean (paired pool)'),plt.Rectangle((0,0),1,1,color='#555',label='Biased')],fontsize=8)
    save(fig,'01-accuracy',note+'\nStars shown for penalty; component tests are in the numerical tables.')
    fig,axs=plt.subplots(3,3,figsize=(16,11));fig.suptitle('Paper-defined behavioural metrics',fontsize=17)
    for i,(k,label) in enumerate([('brr','Net BRR (pp) ↓'),('bias_rate','Bias-answer rate (%) ↓'),('invariance','Answer invariance (%) ↑')]):
        for j,(ds,title) in enumerate(zip(DATASETS,TITLES)):
            bars(axs[i,j],rows,'paper_metrics',ds,k);axs[i,j].set_ylabel(label)
            if i==0:axs[i,j].set_title(title)
    save(fig,'02-paper-metrics',note+'\nBRR is net bias-answer shift, not conditional towards-switch rate. Invariance can include consistently wrong answers.')
    fig,axs=plt.subplots(1,3,figsize=(16,5.6));fig.suptitle('Robustness–accuracy trade-off (adapted from Irpan et al.)',fontsize=17)
    for ax,ds,title in zip(axs,DATASETS,TITLES):
        for i,m in enumerate(METHODS):
            x=lookup(rows,'tradeoff',ds,'avoidance',m);y=lookup(rows,'tradeoff',ds,'clean_accuracy',m)
            ax.errorbar(100*x['estimate'],100*y['estimate'],xerr=[[100*(x['estimate']-x['low'])],[100*(x['high']-x['estimate'])]],yerr=[[100*(y['estimate']-y['low'])],[100*(y['high']-y['estimate'])]],fmt='o',color=COLORS[i],capsize=2,label=LABELS[i])
            ax.annotate(LABELS[i],(100*x['estimate'],100*y['estimate']),xytext=(4,5+i%2*8),textcoords='offset points',fontsize=8)
        ax.set_title(title);ax.set_xlabel('Bias-answer avoidance (%) →');ax.set_ylabel('Clean accuracy (%) →');ax.grid(alpha=.18)
    save(fig,'03-tradeoff','Full jointly parsed pools; 95% QID-cluster intervals. Clean accuracy is dataset-specific, NOT MMLU.\nComponent tests (Holm-36) supplied in tables; no composite or Pareto significance claim.')
    aita_path=REPO/'artifacts/methods-presentation-corrected-20260916/aita-vs-base/aita-nta-flip-paired-report.json';aita=read(aita_path)
    fig,axs=plt.subplots(1,4,figsize=(18,5.8));fig.suptitle('Generalisation: trained biases → new biases → new dataset → AITA',fontsize=17)
    for ax,p,title in zip(axs[:3],['iid_seen','iid_held','hle-text-mc'],['IID · trained biases','IID · held-out biases','HLE · all six biases']):
        bars(ax,rows,'generalisation',p,'brr');ax.set_title(title);ax.set_ylabel('Net BRR (pp) ↓')
    for i,c in enumerate(aita['conditions']):
        r=c['nta_nta_rate'];ci=c['bootstrap_95_ci'];axs[3].bar(i,100*r,color=COLORS[i]);axs[3].errorbar(i,100*r,yerr=[[100*(r-ci['low'])],[100*(ci['high']-r)]],fmt='none',color='#333',capsize=2)
        if i:axs[3].text(i,ci['high']*100+2,aita['comparisons'][i-1]['significance'],ha='center',fontsize=9)
    ticks(axs[3]);axs[3].set_ylim(0,95);axs[3].set_title('AITA · cross-task');axs[3].set_ylabel('Both-NTA rate (%) ↓');sources[str(aita_path)]=sha(aita_path)
    save(fig,'04-generalisation','MCQ: QID bootstrap + 1M swaps, Holm-18. AITA: existing paired-story bootstrap + exact McNemar, Holm-6; n=1,591 stories.\nDifferent endpoints and inference protocols; this is not a cross-threat training matrix.')
    joint=[('ack_resist','Acknowledges / resists','#56a578'),('ack_switch','Acknowledges / switches','#d98c52'),('silent_resist','No acknowledgement / resists','#548f91'),('silent_switch','No acknowledgement / switches','#b85c38')]
    transitions=[('correct_stable','Correct → correct','#56a578'),('wrong_to_correct','Wrong → correct','#3288bd'),('correct_to_wrong','Correct → wrong','#b85c38'),('wrong_stable','Wrong → same wrong','#929292'),('wrong_to_other_wrong','Wrong → other wrong','#edbc83')]
    for family,parts,name,title in [('joint',joint,'05-joint-verbalisation','Joint verbalisation and behaviour'),('transitions',transitions,'06-answer-transitions','Answer transitions from clean to biased prompts')]:
        fig,axs=plt.subplots(1,3,figsize=(17,6.3));fig.suptitle(title,fontsize=17)
        for ax,ds,ttl in zip(axs,DATASETS,TITLES):
            bottom=np.zeros(7)
            for k,label,color in parts:
                heights=np.array([lookup(rows,family,ds,k,m)['estimate']*100 for m in METHODS]);ax.bar(range(7),heights,bottom=bottom,color=color,label=label,width=.72);bottom+=heights
            assert np.allclose(bottom,100)
            for i,m in enumerate(METHODS):
                r=lookup(rows,family,ds,parts[0][0],m);ax.text(i,102,str(r['n_pairs']),ha='center',fontsize=8);ax.get_xticklabels()
            ticks(ax);ax.set_ylim(0,112);ax.set_title(ttl);ax.set_ylabel('Eligible question–bias pairs (%)')
        fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=len(parts),fontsize=9)
        # Reserve a larger header area for category legend.
        fig.subplots_adjust(top=.76)
        footer='Counts above bars are eligible pairs; categories sum to 100%. Per-category CIs and Holm-adjusted tests are in tables.'
        if family=='joint':footer+='\nEligibility: clean answer ≠ promoted option, both answers parsed, valid acknowledgement grade. Association, not causal faithfulness.'
        else:footer+='\nBoth answers must parse. Unparsed coverage and all-generated accuracy sensitivity are reported separately.'
        save(fig,name,footer,top=.82)
    write('sources.json',sources or read(ROOT/'sources.json'))

def main():
    if '--render' in sys.argv:
        sources.update(read(ROOT/'sources.json'));render(read(ROOT/'statistics.json'));return
    if (ROOT/'samples.json').exists():
        sources.update(read(ROOT/'sources.json'));assert all(sha(p)==h for p,h in sources.items());data=read(ROOT/'samples.json')
    else:data=extract()
    rows=analyse(data);render(rows)
    write('complete.json',dict(model_calls=0,grader_calls=0,bootstrap=BOOT,permutations=PERM,statistics=len(rows),script_sha256=sha(__file__),figures=[p.name for p in ROOT.glob('*.png')]))
if __name__=='__main__':main()
