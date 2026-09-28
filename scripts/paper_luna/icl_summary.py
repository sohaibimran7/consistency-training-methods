"""Recompute historical ICL aggregate curves, AUROC and paired contrasts offline."""
import json, os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctm-monitor-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import cli, load_icl
args=cli('icl');OUT=args.output
state=load_icl(args)
maps=state['maps'];common=state['common'];clusters=state['clusters'];ix=state['ix'];weights=state['weights']
methods=['base','bct','rmct352'];names=['Base','BCT','RMCT']
conds=['zero_shot','distribution_only','examples_only','both'];cn=['Zero-shot','Distribution only','Examples only','Both'];colors=['#777777','#408cbe','#df9c35','#8a63b8']
def auc(y,g,w):
 p=np.bincount(g,weights=y*w);n=np.bincount(g,weights=(1-y)*w);den=p.sum()*n.sum()
 return float(np.sum(p*(np.cumsum(n)-n/2))/den) if den else np.nan
stats={};boot={};curves={}
fig,axes=plt.subplots(2,3,figsize=(16,9),sharey=True)
for col,(m,name) in enumerate(zip(methods,names)):
 for c,label,color in zip(conds,cn,colors):
  rr=[maps[m,c][k] for k in common];y=np.array([r['target'] for r in rr]);s=np.array([r['score'] for r in rr]);_,g=np.unique(s,return_inverse=True)
  a=auc(y,g,np.ones(len(y)));b=np.array([auc(y,g,w[ix]) for w in weights]);boot[m,c]=b
  stats[m,c]={'auroc':a,'ci95':np.nanpercentile(b,[2.5,97.5]).tolist(),'positive':int(y.sum()),'n':len(y)}
  ts=np.r_[np.inf,np.unique(s)[::-1]];fp=np.array([np.mean(s[y==0]>=t) for t in ts]);fn=np.array([np.mean(s[y==1]<t) for t in ts])
  assert abs(a-np.sum(np.diff(fp)*(2-fn[:-1]-fn[1:])/2))<1e-10
  curves[m,c]=[{'threshold':None if not np.isfinite(t) else float(t),'fpr':float(x),'fnr':float(z)} for t,x,z in zip(ts,fp,fn)]
  for ax in axes[:,col]:ax.plot(fp,fn,lw=2,color=color,label=f'{label}: {a:.3f}')
 for row in [0,1]:
  ax=axes[row,col];ax.plot([0,1],[1,0],':',color='#cccccc');ax.set(xlim=(0,1 if row==0 else .2),ylim=(0,1),xlabel='False-positive rate',title=name+(' · low-FPR zoom' if row else ''));ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
 axes[0,col].legend(fontsize=8)
axes[0,0].set_ylabel('False-negative rate');axes[1,0].set_ylabel('False-negative rate')
fig.suptitle(f'Luna · CoT-only · xhigh · {len(common)} matched cases\nFour model-specific examples, one per quadrant · lower-left is better')
fig.text(.5,.015,'Same evaluation cases in every panel; demonstration QIDs excluded. Descriptive threshold curves; no confidence bands.\nLabels are observed towards-bias switches, not direct evidence of causal influence.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.065,1,.93]);fig.savefig(OUT/'fnr-fpr.png',dpi=180);fig.savefig(OUT/'fnr-fpr.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,6));x=np.arange(3);width=.19
for j,(c,label,color) in enumerate(zip(conds,cn,colors)):
 vals=np.array([stats[m,c]['auroc'] for m in methods]);lo=np.array([stats[m,c]['ci95'][0] for m in methods]);hi=np.array([stats[m,c]['ci95'][1] for m in methods]);xx=x+(j-1.5)*width
 ax.bar(xx,vals,width,color=color,label=label);ax.errorbar(xx,vals,yerr=[vals-lo,hi-vals],fmt='none',color='black',capsize=3)
 for xpos,v,h in zip(xx,vals,hi):ax.text(xpos,h+.012,f'{v:.3f}',ha='center',fontsize=9)
ax.set(xticks=x,xticklabels=names,ylim=(0,1),ylabel='AUROC (higher is better)');ax.axhline(.5,color='#aaaaaa',ls=':');ax.spines[['top','right']].set_visible(False);ax.legend(ncol=2,loc='lower left')
ax.set_title(f'Luna in-context monitoring · xhigh\n{len(common)} matched cases / {len(clusters)} question clusters')
fig.text(.5,.015,f'95% paired question-cluster bootstrap intervals, stratified by dataset; {args.bootstrap:,} resamples.\nSingle fixed example bank per model; intervals do not capture variation from example selection.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(OUT/'auroc.png',dpi=180);fig.savefig(OUT/'auroc.pdf');plt.close(fig)
contrasts={}
for m in methods:
 for c in conds[1:]:contrasts[m+':'+c+' minus zero_shot']={'difference':stats[m,c]['auroc']-stats[m,'zero_shot']['auroc'],'ci95':np.nanpercentile(boot[m,c]-boot[m,'zero_shot'],[2.5,97.5]).tolist()}
report={'n':len(common),'question_clusters':len(clusters),'metrics':{m+':'+c:v for (m,c),v in stats.items()},'paired_differences':contrasts,'intervals':f'{args.bootstrap} paired question cluster bootstrap, dataset stratified; unadjusted 95%; no significance stars','caveats':'One example bank; varying prompt lengths. Same original system and target JSON, added calibration context only. Observed TBS labels versus causal-influence prompt.'}
(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'curve-points.json').write_text(json.dumps({m+':'+c:v for (m,c),v in curves.items()})+'\n')
print(json.dumps(report,indent=2))
