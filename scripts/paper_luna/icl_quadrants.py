"""Saved-score acknowledgement strata, with paired question bootstraps."""
import json,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/ctm-monitor-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import cli, load_icl
args=cli('icl');out=args.output;ns=load_icl(args)
configs=[('zero_shot','Zero-shot'),('distribution_only','Distribution only'),('examples_only','Examples only'),('both','Distribution + examples')]
methods=[('base','Base','#777777'),('bct','BCT','#4588b6'),('rmct352','RMCT','#d47a40')]
grid=np.linspace(0,1,501);report={};roc={};dist={}
for m,_,_ in methods:
 for c,_ in configs:
  rr=[ns['maps'][m,c][k] for k in ns['common']]
  y=np.array([r['target'] for r in rr]);s=np.array([r['score'] for r in rr]);a=np.array([r['ack'] if r['ack'] in [0,1] else -1 for r in rr])
  report[m+':'+c]={'total':len(rr),'missing_ack':int((a==-1).sum()),'strata':{}}
  for ack in [1,0]:
   mask=a==ack;ix=ns['ix'][mask];yy=y[mask];ss=s[mask];n1=int(yy.sum());n0=int(len(yy)-n1)
   st={'n':len(yy),'switches':n1,'non_switches':n0};report[m+':'+c]['strata'][str(ack)]=st
   for target in [0,1]:
    sub=mask&(y==target);scores=s[sub];w=ns['weights'][:,ns['ix'][sub]].astype(float);den=w.sum(axis=1);valid=den>0
    if len(scores):
     values=(w[valid]@(scores[:,None]<=grid[None,:]))/den[valid,None]
     lo,hi=np.percentile(values,[2.5,97.5],axis=0)
     dist[m,c,ack,target]=(len(scores),np.mean(scores[:,None]<=grid[None,:],axis=0),lo,hi)
    else:dist[m,c,ack,target]=(0,None,None,None)
   if not n1 or not n0:roc[m,c,ack]=None;continue
   th=np.r_[np.inf,np.unique(ss)[::-1]];pred=ss[:,None]>=th
   fp=np.mean(pred[yy==0],axis=0);fn=1-np.mean(pred[yy==1],axis=0)
   st['auroc']=float(np.sum(np.diff(fp)*(2-fn[:-1]-fn[1:])/2))
   draws=[]
   for cw in ns['weights']:
    w=cw[ix];pos=w*yy;neg=w*(1-yy)
    if not pos.sum() or not neg.sum():continue
    f=neg@pred/neg.sum();z=1-pos@pred/pos.sum();keep=np.r_[np.diff(f)>0,True]
    draws.append(np.interp(grid,f[keep],z[keep]))
   lo,hi=np.percentile(draws,[2.5,97.5],axis=0);st['valid_bootstraps']=len(draws)
   roc[m,c,ack]=(fp,fn,lo,hi)
def style(ax):
 ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.set(xlim=(0,1),ylim=(0,1))
note=f'Pointwise 95% bands: {args.bootstrap:,} paired question bootstraps, stratified by dataset; fixed calibration banks.\nStratum membership differs by target model. Base clean labels unverified; historical calibration labels retained.'
def drawroc(ax,c,ack):
 for m,name,color in methods:
  st=report[m+':'+c]['strata'][str(ack)];r=roc[m,c,ack]
  if r is None:
   ax.plot([],[],color=color,label=f'{name}: undefined ({st["switches"]}+ / {st["non_switches"]}−)');continue
  fp,fn,lo,hi=r;ax.fill_between(grid,lo,hi,color=color,alpha=.15,lw=0)
  ax.plot(fp,fn,color=color,lw=2,label=f'{name}: AUC {st["auroc"]:.3f} ({st["switches"]}+ / {st["non_switches"]}−)')
 ax.plot([0,1],[1,0],':',color='#bbb');style(ax);ax.set(xlabel='False-positive rate',ylabel='False-negative rate',title='Acknowledged' if ack else 'Not acknowledged');ax.legend(fontsize=8)
for c,title in configs:
 fig,axes=plt.subplots(2,2,figsize=(12,9))
 for row,ack in enumerate([1,0]):
  for col,t in enumerate([0,1]):
   ax=axes[row,col]
   for m,name,color in methods:
    n,ecdf,lo,hi=dist[m,c,ack,t]
    if n:ax.fill_between(grid,lo,hi,color=color,alpha=.15,lw=0);ax.plot(grid,ecdf,color=color,lw=2,label=f'{name} (n={n})')
    else:ax.plot([],[],color=color,label=f'{name} (n=0)')
   style(ax);ax.set(xlabel='Luna influence score (0–1)',ylabel='Fraction of responses scoring ≤ x',title=('Acknowledged' if ack else 'Not acknowledged')+' / '+('switch' if t else 'no switch'));ax.legend(fontsize=9)
 fig.suptitle(f'{title} · Luna xhigh · score distributions by acknowledgement × TBS\nNo switch: lower scores desirable; switch: higher scores desirable')
 fig.text(.5,.015,note,ha='center',fontsize=8);fig.tight_layout(rect=(0,.065,1,.93))
 for ext in ['png','pdf']:fig.savefig(out/f'score-quadrants-{c}.{ext}',dpi=170)
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,5.8))
 for ax,ack in zip(axes,[1,0]):drawroc(ax,c,ack)
 fig.suptitle(f'{title} · Luna xhigh · FNR–FPR by acknowledgement · lower-left is better')
 fig.text(.5,.015,note+'\nLegend counts: + switches / − non-switches. One-class resamples omitted.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.10,1,.94))
 for ext in ['png','pdf']:fig.savefig(out/f'fnr-fpr-acknowledgement-{c}.{ext}',dpi=170)
 plt.close(fig)
(out/'results.json').write_text(json.dumps({'matched_population':len(ns['common']),'question_clusters':len(ns['clusters']),'metrics':report,'notes':note},indent=2))
print(out)
