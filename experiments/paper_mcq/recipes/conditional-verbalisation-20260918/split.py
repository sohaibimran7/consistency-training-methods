
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
"""Conditional acknowledgement: dataset and bias generalisation, 50k swaps."""
import importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'paper-behavioural-plots-20260917'
spec=importlib.util.spec_from_file_location('prior',PRIOR/'build.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.ROOT=ROOT
DATA=b.read(PRIOR/'samples.json');PERM=50000;BOOT=10000
def select(m,d,g):
    return [r for r in DATA[m] if ((r['dataset']!='hle-text-mc')==(d=='seen')) and ((r['bias'] in b.SEEN)==(g=='seen'))]
def counts(r):
    switch=r['b'] is not None and r['u'] is not None and r['u']!=r['option'] and r['b']==r['option'] and r['ack'] is not None
    return np.array([int(r['ack']) if switch else 0,int(switch)])
def estimate(rr,tag):
    clusters=defaultdict(lambda:np.zeros(2))
    for r in rr:clusters[r['dataset'],r['qid']]+=counts(r)
    total=np.sum(list(clusters.values()),axis=0)
    if total[1]==0:return dict(estimate=None,low=None,high=None,acknowledged=0,n_switches=0,n_qids=len(clusters),bootstrap_valid=0)
    rng=np.random.default_rng(b.seed(tag));num=np.zeros(BOOT);den=np.zeros(BOOT)
    for ds in b.DATASETS:
        a=np.array([v for (d,q),v in clusters.items() if d==ds])
        if not len(a):continue
        sample=a[rng.integers(len(a),size=(BOOT,len(a)))].sum(axis=1);num+=sample[:,0];den+=sample[:,1]
    valid=den>0;lo,hi=np.quantile(num[valid]/den[valid],[.025,.975])
    return dict(estimate=float(total[0]/total[1]),low=float(lo),high=float(hi),acknowledged=int(total[0]),n_switches=int(total[1]),n_qids=len(clusters),bootstrap_valid=int(valid.sum()))
def contrast(rr,base,tag):
    def keyed(rows):return {(r['dataset'],r['bias'],r['qid']):r for r in rows}
    a,c=keyed(rr),keyed(base);keys=sorted(a.keys()&c.keys())
    clusters=defaultdict(lambda:np.zeros(4))
    for k in keys:clusters[k[0],k[2]]+=np.r_[counts(a[k]),counts(c[k])]
    v=np.array(list(clusters.values()));av,cv=v[:,:2],v[:,2:];at,ct=av.sum(0),cv.sum(0)
    if at[1]==0 or ct[1]==0:return dict(test_unavailable='No switched responses in one matched condition',matched_method_switches=int(at[1]),matched_base_switches=int(ct[1]))
    observed=at[0]/at[1]-ct[0]/ct[1];rng=np.random.default_rng(b.seed(tag));extreme=0;valid=0;invalid=0
    delta=cv-av
    while valid<PERM:
        size=min(2000,PERM-valid);swaps=rng.integers(0,2,size=(size,len(v)))
        change=swaps@delta;aa=at+change;cc=ct-change
        ok=(aa[:,1]>0)&(cc[:,1]>0);invalid+=int((~ok).sum())
        diff=aa[ok,0]/aa[ok,1]-cc[ok,0]/cc[ok,1]
        extreme+=int((abs(diff)>=abs(observed)-1e-12).sum());valid+=len(diff)
    return dict(p_raw=(extreme+1)/(PERM+1),paired_difference=float(observed),matched_method_estimate=float(at[0]/at[1]),matched_base_estimate=float(ct[0]/ct[1]),matched_method_switches=int(at[1]),matched_base_switches=int(ct[1]),matched_qids=len(v),matched_records=len(keys),permutations=valid,undefined_permutations_redrawn=invalid,permutation_seed=b.seed(tag))
rows=[]
for d in ['seen','held']:
    for g in ['seen','held']:
        for m in b.METHODS:
            tag=f'conditional-split/{d}/{g}/{m}';rr=select(m,d,g)
            r=dict(dataset_group=d,bias_group=g,method=m,**estimate(rr,tag+'/boot'))
            if m!='base':r.update(contrast(rr,select('base',d,g),tag+'/perm'))
            rows.append(r)
        print('completed',d,g,flush=True)
b.holm(rows)
for r in rows:
    if 'p_raw' in r:
        r['resolution_floor']=r['p_raw']==1/(PERM+1)
        if r['resolution_floor'] and r['p_holm']>=.05:r['marker']='unresolved'
b.write('split-statistics-50k.json',rows)
fig,axs=b.plt.subplots(2,2,figsize=(14,10),sharey=True)
fig.suptitle('Acknowledgement conditional on a towards-bias switch',fontsize=17)
for i,g in enumerate(['seen','held']):
    for j,d in enumerate(['seen','held']):
        ax=axs[i,j]
        for x,m in enumerate(b.METHODS):
            r=next(r for r in rows if (r['dataset_group'],r['bias_group'],r['method'])==(d,g,m));y=100*r['estimate']
            ax.bar(x,y,color=b.COLORS[x]);ax.errorbar(x,y,yerr=[[max(0,y-100*r['low'])],[max(0,100*r['high']-y)]],fmt='none',color='#333',capsize=3)
            ax.text(x,106,f"{r['acknowledged']}/{r['n_switches']}",ha='center',fontsize=9)
            ax.text(x,100*r['high']+1.5,r.get('marker',''),ha='center',fontsize=9)
        ax.set_title(('Seen biases' if g=='seen' else 'Held-out biases')+' · '+('Seen datasets: LogiQA + HellaSwag' if d=='seen' else 'Held-out dataset: HLE'))
        b.ticks(ax);ax.set_ylim(0,115);ax.set_yticks(range(0,101,20));ax.set_ylabel('Acknowledgement among towards-bias switches (%)')
b.save(fig,'acknowledgement-given-switch-dataset-bias-split-50k','Labels = acknowledged / switched. Both answers parsed, clean ≠ promoted and biased = promoted; valid grade.\nBars: full pools, each model’s own switch subset. 95% dataset-stratified QID bootstrap (10k); all-success intervals degenerate.\nStars vs Base: 50,000 whole-QID label swaps on shared records; Holm-24. * <.05, ** <.01, *** <.001; ns otherwise.\nTests use method-specific switch denominators, NOT only cases where both models switch. Selection differences are not causal effects.\nRMCT step 352; known parser limitations unchanged. No model calls.',top=.94)
b.write('split-provenance-50k.json',dict(samples_sha256=b.sha(PRIOR/'samples.json'),script_sha256=b.sha(__file__),bootstrap=BOOT,permutations=PERM,holm_family=24,model_calls=0,grader_calls=0))
