import json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctm-monitor-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import cli, validate_rows
args=cli('rare');O=args.output
rows=json.loads(args.data.read_text());counts=json.loads(args.counts.read_text());validate_rows(rows)
configs=[('zero_shot','Zero-shot'),('distribution_only','Distribution only'),('examples_only','Examples only'),('both','Distribution + examples')]
methods=[('base','Base','#777777'),('bct','BCT','#4588b6'),('rmct352','RMCT','#d47a40')]
clusters=sorted({(r['dataset'],r['qid']) for r in rows});ci={k:i for i,k in enumerate(clusters)}
rng=np.random.default_rng(args.seed);B=args.bootstrap;cw=np.zeros((B,len(clusters)))
for ds in sorted({k[0] for k in clusters}):
 ids=[i for i,k in enumerate(clusters) if k[0]==ds];cw[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=B)
grid=np.linspace(0,1,501);report={};roc={};ecdfs={};comparison={};bands={}
def calc(s,y,w):
 order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order];ww=w[order]
 ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
 pos=ww*yy;neg=ww*(1-yy)
 if not pos.sum() or not neg.sum():return None
 fp=np.r_[0,np.cumsum(neg)[ends]/neg.sum()];fn=np.r_[1,1-np.cumsum(pos)[ends]/pos.sum()]
 auc=float(np.sum(np.diff(fp)*(2-fn[:-1]-fn[1:])/2));return fp,fn,auc
def summary_curve(s,y,w,weights):
 point=calc(s,y,w)
 if point is None:return None,{'n':len(y),'positive':int(y.sum()),'auroc':None}
 draws=[];aucs=np.full(B,np.nan)
 for i,ww in enumerate(weights):
  result=calc(s,y,ww)
  if result is None:continue
  fp,fn,a=result;keep=np.r_[np.diff(fp)>0,True];draws.append(np.interp(grid,fp[keep],fn[keep]));aucs[i]=a
 lo,hi=np.percentile(draws,[2.5,97.5],axis=0)
 return (*point,lo,hi,aucs),{'n':len(y),'positive':int(y.sum()),'auroc':point[2],'ci95':np.nanpercentile(aucs,[2.5,97.5]).tolist(),'valid_bootstraps':len(draws)}
for m,_,_ in methods:
 rr=[r for r in rows if r['method']==m];a=np.array([r['ack'] for r in rr]);y=np.array([r['target'] for r in rr]);base=np.array([r['baseline'] for r in rr]);wboot=cw[:,[ci[r['dataset'],r['qid']] for r in rr]]
 w=np.ones(len(rr));wb=wboot.copy()
 for aa in [0,1]:
  for yy in [0,1]:
   mask=(a==aa)&(y==yy);N=counts[m]['quadrants'][f'{aa},{yy}']['population'];w[mask]=N/mask.sum()
   den=wboot[:,mask].sum(axis=1);wb[:,mask]=np.divide(wboot[:,mask]*N,den[:,None],out=np.zeros_like(wboot[:,mask]),where=den[:,None]>0)
 # Omit draws missing a population stratum rather than silently changing its mass.
 valid=np.ones(B,dtype=bool)
 for aa in [0,1]:
  for yy in [0,1]:valid &= wboot[:,(a==aa)&(y==yy)].sum(axis=1)>0
 wb[~valid]=0
 for c,_ in configs:
  s=np.array([r['scores'][c] for r in rr]);report[m+':'+c]={}
  for aa in [0,1]:
   mask=a==aa;result,stats=summary_curve(s[mask],y[mask],np.ones(mask.sum()),wboot[:,mask]);roc[m,c,aa]=result;report[m+':'+c][f'ack={aa}']=stats
   for yy in [0,1]:
    mask=(a==aa)&(y==yy);ww=wboot[:,mask];den=ww.sum(axis=1);keep=den>0
    curves=ww[keep]@(s[mask,None]<=grid)/den[keep,None];lo,hi=np.percentile(curves,[2.5,97.5],axis=0)
    ecdfs[m,c,aa,yy]=(int(mask.sum()),np.mean(s[mask,None]<=grid,axis=0),lo,hi)
  for mode,mask,weights,boot in [('historical',base,np.ones(base.sum()),wboot[:,base]),('weighted',np.ones(len(rr),dtype=bool),w,wb)]:
   result,stats=summary_curve(s[mask],y[mask],weights,boot);comparison[m,c,mode]=result;report[m+':'+c][mode]=stats
  delta=comparison[m,c,'weighted'][-1]-comparison[m,c,'historical'][-1]
  report[m+':'+c]['weighted_minus_historical']={'difference':comparison[m,c,'weighted'][2]-comparison[m,c,'historical'][2],'ci95':np.nanpercentile(delta,[2.5,97.5]).tolist(),'valid_paired_draws':int(np.isfinite(delta).sum())}
  pairs=[]
  for ap in [0,1]:
   for an in [0,1]:
    p=s[(a==ap)&(y==1)];n=s[(a==an)&(y==0)];pairauc=float(np.mean((p[:,None]>n)+.5*(p[:,None]==n)))
    q=counts[m]['quadrants'];mass=q[f'{ap},1']['population']/sum(q[f'{k},1']['population'] for k in [0,1])*q[f'{an},0']['population']/sum(q[f'{k},0']['population'] for k in [0,1])
    pairs.append({'positive_ack':ap,'negative_ack':an,'pair_auroc':pairauc,'population_pair_mass':mass,'contribution':pairauc*mass})
  assert abs(sum(p['contribution'] for p in pairs)-comparison[m,c,'weighted'][2])<1e-10
  report[m+':'+c]['pair_contributions']=pairs
def style(ax):
 ax.set(xlim=(0,1),ylim=(0,1));ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
def line(ax,result,color,label,ls='-'):
 if result is None:ax.plot([],[],color=color,label=label+' undefined');return
 fp,fn,auc,lo,hi,_=result;ax.fill_between(grid,lo,hi,color=color,alpha=.14,lw=0);ax.plot(fp,fn,color=color,lw=2,ls=ls,label=label+f' AUC {auc:.3f}')
note=f'Pointwise 95% question-cluster bootstrap bands ({args.bootstrap:,} draws); model-specific populations, no global intersection.\nBase clean labels unverified; historical calibration labels retained. Rare cells have limited support.'
for c,title in configs:
 fig,axes=plt.subplots(2,2,figsize=(12,9))
 for i,aa in enumerate([1,0]):
  for j,yy in enumerate([0,1]):
   ax=axes[i,j]
   for m,name,color in methods:
    n,v,lo,hi=ecdfs[m,c,aa,yy];ax.fill_between(grid,lo,hi,color=color,alpha=.14,lw=0);ax.plot(grid,v,color=color,lw=2,label=f'{name} n={n}')
   style(ax);ax.set(title=('Acknowledged' if aa else 'Silent')+' / '+('switch' if yy else 'no switch'),xlabel='Luna influence score',ylabel='Fraction scoring ≤ x');ax.legend(fontsize=9)
 fig.suptitle(title+' · cached common cases + all available rare cases\nNo switch: lower scores desirable; switch: higher scores desirable');fig.text(.5,.01,note,ha='center',fontsize=8);fig.tight_layout(rect=(0,.06,1,.94))
 for ext in ['png','pdf']:fig.savefig(O/f'quadrants-{c}.{ext}',dpi=160)
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,6))
 for ax,aa in zip(axes,[1,0]):
  for m,name,color in methods:
   st=report[m+':'+c][f'ack={aa}'];line(ax,roc[m,c,aa],color,f'{name} ({st["positive"]}+/{st["n"]-st["positive"]}−)')
  style(ax);ax.plot([0,1],[1,0],':',color='#bbb');ax.set(title='Acknowledged' if aa else 'Silent',xlabel='False-positive rate',ylabel='False-negative rate');ax.legend(fontsize=8)
 fig.suptitle(title+' · diagnostic FNR–FPR by acknowledgement');fig.text(.5,.01,note+'\nUnweighted diagnostic strata; not population estimates. One-class bootstrap draws omitted.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.1,1,.94))
 for ext in ['png','pdf']:fig.savefig(O/f'fnr-fpr-{c}.{ext}',dpi=160)
 plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(16,6))
 for ax,(m,name,color) in zip(axes,methods):
  line(ax,comparison[m,c,'historical'],'#777777','Existing comparison','--');line(ax,comparison[m,c,'weighted'],color,'Population reweighted')
  style(ax);ax.set(title=name,xlabel='False-positive rate',ylabel='False-negative rate');ax.legend(fontsize=8)
 fig.suptitle(title+' · sampling sensitivity (not a randomized comparison)');fig.text(.5,.01,'Weighted curves calibrate all four quadrants to eligible-population counts and include cross-acknowledgement pairs.\nHistorical inclusion probabilities unknown: weighting does not establish population unbiasedness. No new random sample.\n'+note,ha='center',fontsize=8);fig.tight_layout(rect=(0,.15,1,.94))
 for ext in ['png','pdf']:fig.savefig(O/f'weighted-vs-existing-{c}.{ext}',dpi=160)
 plt.close(fig)
for key,result in list(roc.items())+list(comparison.items()):
 if result is not None:bands[str(key)]={'fpr':result[0].tolist(),'fnr':result[1].tolist(),'band_grid':grid.tolist(),'lower':result[3].tolist(),'upper':result[4].tolist()}
(O/'results.json').write_text(json.dumps(report,indent=2));(O/'curves.json').write_text(json.dumps(bands));print('Complete',O)
