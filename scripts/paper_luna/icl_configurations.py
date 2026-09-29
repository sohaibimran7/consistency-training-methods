"""Re-layout saved corrected curves; no inference or metric recomputation."""
import json, os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctm-monitor-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import cli, load_icl
args=cli('icl');out=args.output;ns=load_icl(args)
configs=[('zero_shot','Zero-shot'),('distribution_only','Distribution only'),('examples_only','Examples only'),('both','Distribution + examples')]
methods=[('base','Base','#777777'),('bct','BCT','#4588b6'),('rmct352','RMCT','#d47a40')]
curves={};results={'n':len(ns['common']),'metrics':{}}
for m,_,_ in methods:
 for c,_ in configs:
  rr=[ns['maps'][m,c][k] for k in ns['common']];y=np.array([r['target'] for r in rr]);s=np.array([r['score'] for r in rr])
  if not y.sum() or y.sum()==len(y):raise ValueError('Overall ROC needs both classes')
  th=np.r_[np.inf,np.unique(s)[::-1]];pred=s[:,None]>=th;fp=np.mean(pred[y==0],axis=0);fn=1-np.mean(pred[y==1],axis=0)
  curves[m+':'+c]=[{'fpr':float(f),'fnr':float(z)} for f,z in zip(fp,fn)]
  results['metrics'][m+':'+c]={'auroc':float(np.sum(np.diff(fp)*(2-fn[:-1]-fn[1:])/2)),'n':len(y),'positive':int(y.sum())}
(out/'results.json').write_text(json.dumps(results,indent=2))
(out/'curve-points.json').write_text(json.dumps(curves))
grid=np.linspace(0,1,1001);bands={}
for m,_,_ in methods:
 for c,_ in configs:
  rr=[ns['maps'][m,c][k] for k in ns['common']]
  y=np.array([r['target'] for r in rr]);s=np.array([r['score'] for r in rr])
  thresholds=np.r_[np.inf,np.unique(s)[::-1]]
  predicted=s[:,None]>=thresholds[None,:]
  draws=[]
  for cw in ns['weights']:
   w=cw[ns['ix']];pos=w*y;neg=w*(1-y)
   if not pos.sum() or not neg.sum():continue
   fp=neg@predicted/neg.sum();fn=1-pos@predicted/pos.sum()
   # Rightmost point at duplicate FPR: lowest attainable FNR at that FPR.
   keep=np.r_[np.diff(fp)>0,True]
   draws.append(np.interp(grid,fp[keep],fn[keep]))
  lo,hi=np.percentile(draws,[2.5,97.5],axis=0)
  bands[m+':'+c]={'fpr':grid.tolist(),'lower':lo.tolist(),'upper':hi.tolist(),'valid_bootstraps':len(draws)}
(out/'confidence-bands.json').write_text(json.dumps({'method':f'{args.bootstrap} paired question-cluster percentile bootstraps, stratified by dataset; pointwise 95%; linear interpolation with rightmost duplicate FPR; fixed calibration bank; one-class resamples omitted','bands':bands}))
def draw(ax,c,title):
 for m,name,color in methods:
  points=curves[m+':'+c];auc=results['metrics'][m+':'+c]['auroc']
  band=bands[m+':'+c]
  ax.fill_between(band['fpr'],band['lower'],band['upper'],color=color,alpha=.16,linewidth=0)
  ax.plot([p['fpr'] for p in points],[p['fnr'] for p in points],color=color,lw=2,label=f'{name} (AUROC {auc:.3f})')
 ax.plot([0,1],[1,0],':',color='#bbbbbb',lw=1,label='Chance')
 ax.set(xlim=(0,1),ylim=(0,1),xlabel='False-positive rate',ylabel='False-negative rate',title=title)
 ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=9)
note=f'Shading: pointwise 95% intervals · {args.bootstrap:,} paired question bootstraps, stratified by dataset.\nBase clean labels unverified; calibration inputs retain historical labels. Fixed example banks.'
for c,title in configs:
 fig,ax=plt.subplots(figsize=(7,6));draw(ax,c,title)
 fig.suptitle(f'Luna · CoT-only · xhigh · {results["n"]} matched cases\nLower-left is better',fontsize=12)
 fig.text(.5,.025,note,ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.09,1,.92))
 for ext in ['png','pdf']:fig.savefig(out/f'fnr-fpr-{c}.{ext}',dpi=180)
 plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,10))
for ax,(c,title) in zip(axes.flat,configs):draw(ax,c,title)
fig.suptitle(f'Luna · CoT-only · xhigh · {results["n"]} matched cases · lower-left is better')
fig.text(.5,.015,note,ha='center',fontsize=9);fig.tight_layout(rect=(0,.065,1,.96))
for ext in ['png','pdf']:fig.savefig(out/f'fnr-fpr-configurations.{ext}',dpi=180)
plt.close(fig)
print(out)
