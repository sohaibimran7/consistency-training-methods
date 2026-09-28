"""Descriptive condition-mean OLS with source-block bootstrap confidence bands."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]/'experiments/eval_awareness/v2-breadth-20260924/analysis-v1'
ROLES=['base','mo_mid','mo_post'];NAMES=['Base Qwen','MO mid','MO post']
FAMILIES=['am-blackmail','am-leaking','am-murder','instrumentaleval']
TITLES=['Blackmail','Information leaking','Rescue obstruction','InstrumentalEval']
CONDS=['B']+[f'F{i}' for i in range(1,9)]

def fit(x,y):
    if len(x)<3 or np.ptp(x)<1e-9:return None
    slope=np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)
    return float(slope),float(y.mean()-slope*x.mean())

def bootstrap(rows,conditions,grid,seed,nboot=5000):
    sources=sorted({r['source_id'] for r in rows})
    a=np.zeros((len(sources),len(conditions)));p=a.copy();valid=a.copy()
    for r in rows:
        if r['condition'] not in conditions or r['combined'] is None or r['propensity'] is None:continue
        i=sources.index(r['source_id']);j=conditions.index(r['condition'])
        a[i,j]=r['combined'];p[i,j]=r['propensity'];valid[i,j]=1
    rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(sources),np.full(len(sources),1/len(sources)),size=nboot)
    count=weights@valid
    xs=np.divide(weights@a,count,out=np.full(count.shape,np.nan),where=count>0)*100
    ys=np.divide(weights@p,count,out=np.full(count.shape,np.nan),where=count>0)*100
    # Keep the set of factor conditions fixed; missing conditions or constant X
    # make a draw non-estimable, rather than replacing missing values by zero.
    coefs=[fit(x,y) for x,y in zip(xs,ys) if np.isfinite(x).all()]
    coefs=np.array([c for c in coefs if c is not None])
    if len(coefs)<100:return dict(draws=nboot,estimable=len(coefs),sources=len(sources))
    predicted=coefs[:,0,None]*grid+coefs[:,1,None]
    lo,hi=np.percentile(predicted,[2.5,97.5],axis=0)
    return dict(draws=nboot,estimable=len(coefs),sources=len(sources),lower=lo.tolist(),upper=hi.tolist(),slope_ci=np.percentile(coefs[:,0],[2.5,97.5]).tolist())

def main(report=None,out=None):
    report=report or BASE/'report.json';out=out or BASE
    out.mkdir(parents=True,exist_ok=True)
    data=json.loads(report.read_text());stats=[]
    fig,axes=plt.subplots(4,3,figsize=(14,14),sharex=True,sharey=True)
    colors=['#467ba8','#b26729','#773e91']
    for fi,family in enumerate(FAMILIES):
      for ri,role in enumerate(ROLES):
        ax=axes[fi,ri];color=colors[ri]
        cells=[c for c in data['cells'] if (c['family'],c['role'])==(family,role) and c['valid']]
        x=np.array([100*c['awareness'] for c in cells]);y=np.array([100*c['propensity'] for c in cells])
        coef=fit(x,y);entry=dict(family=family,role=role,conditions=len(cells),method='unweighted OLS over condition means')
        if coef is not None:
            grid=np.linspace(x.min(),x.max(),150);slope,intercept=coef
            group=[r for r in data['rows'] if (r['family'],r['role'])==(family,role)]
            boot=bootstrap(group,[c['condition'] for c in cells],grid,seed=240924+fi*3+ri)
            entry.update(slope=slope,intercept=intercept,x=grid.tolist(),bootstrap=boot)
            if 'lower' in boot:ax.fill_between(grid,boot['lower'],boot['upper'],color=color,alpha=.18,lw=0)
            ax.plot(grid,slope*grid+intercept,color=color,lw=1.9,zorder=2)
            note=f'slope {slope:+.2f}'
            if 'slope_ci' in boot:note+='  ['+', '.join(f'{v:+.2f}' for v in boot['slope_ci'])+']'
            note+=f"\nBootstrap estimable: {boot['estimable']}/{boot['draws']}"
            if np.ptp(y)<1e-9:note+='\nNo observed outcome variation'
        else:
            entry['fit_status']='not_identifiable_constant_x_or_too_few_conditions'
            note='No fitted slope: no X variation' if np.ptp(x)<1e-9 else 'Too few conditions to fit'
        ax.text(.03,.97,note,transform=ax.transAxes,va='top',fontsize=8,color='#333',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=2))
        labels={}
        for c,xx,yy in zip(cells,x,y):
            ax.scatter(xx,yy,s=25+10*c['valid'],marker='D' if c['condition']=='B' else 'o',color=color,alpha=.8,zorder=3)
            labels.setdefault((xx,yy),[]).append(c['condition'])
        for idx,(xy,ls) in enumerate(labels.items()):
            ax.annotate(','.join(ls),xy,xytext=(-5 if xy[0]>80 else 5,7 if idx%2==0 else -12),ha='right' if xy[0]>80 else 'left',textcoords='offset points',fontsize=7)
        ax.set_title(TITLES[fi]+' · '+NAMES[ri],fontsize=11)
        ax.set_xlim(-5,115);ax.set_ylim(-8,125);ax.set_yticks([0,25,50,75,100]);ax.grid(alpha=.15)
        if fi==3:ax.set_xlabel('VAE OR flattery (%)')
        if ri==0:ax.set_ylabel('Misalignment propensity (%)')
        stats.append(entry)
    fig.suptitle('V2 breadth · best-fit lines with 95% bootstrap bands',fontsize=18,y=.987)
    fig.text(.5,.96,'Unweighted condition-mean OLS; shaded pointwise confidence bands, not prediction intervals. Provisional.',ha='center',fontsize=10)
    fig.text(.5,.047,'5,000 source-block resamples retain factor variants together. Non-estimable draws are excluded; counts shown in each panel.',ha='center',fontsize=9)
    fig.text(.5,.029,'AM resamples goal configurations within ONE narrative per panel; intervals do not establish generalization to new narratives.',ha='center',fontsize=9)
    fig.text(.5,.011,'Conditional on observed valid judgments: bands omit judge error, missingness bias, and generation randomness. Flat bands do not imply certainty.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95))
    for ext in ('png','svg'):fig.savefig(out/f'awareness-propensity-scatter-fits.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
    (out/'fit-statistics.json').write_text(json.dumps(dict(panels=stats,seed=240924,bootstrap_draws=5000),indent=2)+'\n')
    print(json.dumps([{k:v for k,v in s.items() if k not in ('x','bootstrap')}|({'estimable':s['bootstrap']['estimable']} if 'bootstrap' in s else {}) for s in stats],indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=BASE/'report.json')
    parser.add_argument('--output',type=Path,default=BASE)
    args=parser.parse_args();main(args.report,args.output)
