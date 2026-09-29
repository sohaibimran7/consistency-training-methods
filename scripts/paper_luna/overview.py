"""Historical Luna overview: offline aggregation without injected external recipes."""
import argparse
import hashlib
import json

def correct_labels(labels, samples):
    lookup={(m,r['dataset'],r['bias'],r['qid']):r for m,rows in samples.items() for r in rows}
    out=[];changed=excluded=unverified=missing=0
    for label in labels:
        r=lookup.get((label['method'],label['dataset'],label['bias'],label['qid']))
        if r is None:
            missing+=1
            continue
        eligible=r['b'] is not None and r['u'] is not None and r['u']!=r['option']
        target=int(r['b']==r['option']) if eligible else None
        changed+=eligible and target!=label['target']
        excluded+=not eligible
        unverified+=not r['clean_verified']
        if eligible:
            out.append(dict(label,target=target,corrected_eligible=eligible,clean_verified=r['clean_verified']))
    return out,dict(original_label_rows=len(labels),excluded=excluded,changed_eligible_targets=changed,unverified_base_clean_rows=unverified,missing_source=missing)

def main():
    """Offline analysis only. Common-case comparisons and paired QID bootstrap."""
    import json, os
    from pathlib import Path
    from collections import Counter
    import numpy as np
    os.environ.setdefault('MPLCONFIGDIR','/tmp/ctm-monitor-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--planned-requests',type=int,default=12092)
    p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20260920)
    args=p.parse_args()
    if args.bootstrap<2:p.error('At least two bootstrap draws required')
    OUT=args.output
    OUT.mkdir(parents=True,exist_ok=True)
    METHODS=['bct','rmct352']; EFFORTS=['minimal','xhigh']; VIEWS=['cot_only','cot_sequential','prompt_cot','prompt_only']
    NAMES={'bct':'BCT','rmct352':'RMCT','cot_only':'CoT only','cot_sequential':'Sequential CoT','prompt_cot':'Prompt + CoT','prompt_only':'Prompt only'}
    results=[json.loads(l) for l in args.ledger.read_text().splitlines()]
    good={r['request_id']:r for r in results if r['ok']}
    labels,audit=correct_labels(json.loads(args.labels.read_text()),json.loads(args.samples.read_text()))
    rows=[dict(r,score=good[r['request_id']]['score']/100) for r in labels if r['request_id'] in good]
    maps={(m,e,v):{r['case_id']:r for r in rows if (r['method'],r['effort'],r['view'])==(m,e,v)} for m in METHODS for e in EFFORTS for v in VIEWS}
    common=sorted(set.intersection(*(set(v) for v in maps.values())))
    if not common:raise ValueError('No common eligible cases across the 16 conditions')
    reference=[maps['bct','minimal','cot_only'][k] for k in common]
    clusters=sorted({(r['dataset'],r['qid']) for r in reference}); ci={k:i for i,k in enumerate(clusters)}
    indices=np.array([ci[r['dataset'],r['qid']] for r in reference])
    rng=np.random.default_rng(args.seed); B=args.bootstrap
    weights=np.zeros((B,len(clusters)),dtype=np.int16)
    for ds in sorted({k[0] for k in clusters}):
     ids=[i for i,k in enumerate(clusters) if k[0]==ds]
     weights[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=B)
    
    def auc(y,s,w=None):
     y=np.asarray(y);s=np.asarray(s);w=np.ones(len(y)) if w is None else w
     _,groups=np.unique(s,return_inverse=True)
     pos=np.bincount(groups,weights=w*y);neg=np.bincount(groups,weights=w*(1-y))
     den=pos.sum()*neg.sum()
     return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/den) if den else float('nan')
    
    def bounds(a):
     a=np.asarray(a);valid=a[np.isfinite(a)]
     return [float(x) for x in np.percentile(valid,[2.5,97.5])] if len(valid) else [None,None]
    
    def basic(rr):
     y=np.array([r['target'] for r in rr]);s=np.array([r['score'] for r in rr])
     return {'n':len(rr),'positive':int(y.sum()),'questions':len({(r['dataset'],r['qid']) for r in rr}),
             'auroc':auc(y,s),'brier':float(np.mean((s-y)**2)) if len(rr) else None}
    
    stats={};boot={}; curves={}
    for key,lookup in maps.items():
     rr=[lookup[k] for k in common];y=np.array([r['target'] for r in rr]);s=np.array([r['score'] for r in rr])
     _,g=np.unique(s,return_inverse=True);ng=g.max()+1
     bs=np.empty(B)
     for b,w in enumerate(weights):
      ww=w[indices];pos=np.bincount(g,weights=ww*y,minlength=ng);neg=np.bincount(g,weights=ww*(1-y),minlength=ng)
      den=pos.sum()*neg.sum();bs[b]=np.sum(pos*(np.cumsum(neg)-.5*neg))/den if den else np.nan
     boot[key]=bs; stats['/'.join(key)]={**basic(rr),'ci95':bounds(bs),'bootstrap_valid':int(np.isfinite(bs).sum())}
     thresholds=np.r_[np.inf,np.unique(s)[::-1],-np.inf]
     curves[key]=([float(np.mean(s[y==0]>=t)) for t in thresholds],[float(np.mean(s[y==1]<t)) for t in thresholds])
    
    contrasts=[]
    for e in EFFORTS:
     for v in VIEWS:
      k1=('rmct352',e,v);k0=('bct',e,v)
      contrasts.append({'contrast':'RMCT minus BCT','effort':e,'view':v,'difference':stats['/'.join(k1)]['auroc']-stats['/'.join(k0)]['auroc'],'ci95':bounds(boot[k1]-boot[k0])})
    for m in METHODS:
     for e in EFFORTS:
      k1=(m,e,'prompt_cot');k0=(m,e,'prompt_only')
      contrasts.append({'contrast':'CoT uplift','method':m,'effort':e,'difference':stats['/'.join(k1)]['auroc']-stats['/'.join(k0)]['auroc'],'ci95':bounds(boot[k1]-boot[k0])})
     for v in VIEWS:
      k1=(m,'xhigh',v);k0=(m,'minimal',v)
      contrasts.append({'contrast':'xhigh minus minimal','method':m,'view':v,'difference':stats['/'.join(k1)]['auroc']-stats['/'.join(k0)]['auroc'],'ci95':bounds(boot[k1]-boot[k0])})
    
    available=[];breakdowns=[]
    for key,lookup in maps.items():
     available.append(dict(zip(['method','effort','view'],key),**basic(list(lookup.values()))))
     for ds in ['logiqa','hellaswag','hle-text-mc']:
      rr=[lookup[k] for k in common if lookup[k]['dataset']==ds]
      breakdowns.append(dict(zip(['method','effort','view'],key),dataset=ds,**basic(rr)))
     for bias in sorted({r['bias'] for r in reference}):
      rr=[lookup[k] for k in common if lookup[k]['bias']==bias]
      breakdowns.append(dict(zip(['method','effort','view'],key),bias=bias,**basic(rr)))
    
    usage={}
    for e in EFFORTS:
     rr=[r for r in results if r.get('effort')==e and r.get('response',{}).get('usage')]
     usage[e]={'reported_cost_usd':sum(r['response']['usage'].get('cost',0) or 0 for r in rr),'responses_with_usage':len(rr),
               'mean_reasoning_tokens':float(np.mean([r['response']['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) or 0 for r in rr]))}
    report={'successful_requests':len(good),'planned_requests':args.planned_requests,'failed_requests':args.planned_requests-len(good),'common_cases':len(common),'common_questions':len(clusters),
     'common_dataset_counts':dict(Counter(r['dataset'] for r in reference)),'population':'Exact question-bias intersection with all four views and both reasoning levels successful for both methods; all-available secondary results separate.',
     'bootstrap':f'{B} paired question-cluster bootstrap resamples stratified by dataset; intervals unadjusted descriptive 95%; not multiplicity-adjusted significance tests.',
     'metrics':stats,'contrasts':contrasts,'all_available':available,'breakdowns':breakdowns,'usage':usage}
    (OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    colors={'bct':'#4588b6','rmct352':'#d47a40'}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    fig,axs=plt.subplots(1,3,figsize=(16,5),gridspec_kw={'width_ratios':[1.25,1.25,.85]})
    for ax,e in zip(axs[:2],EFFORTS):
     for j,m in enumerate(METHODS):
      vals=[stats[f'{m}/{e}/{v}'] for v in VIEWS];x=np.arange(4)+(j-.5)*.34;y=np.array([v['auroc'] for v in vals]);lo=np.array([v['ci95'][0] for v in vals]);hi=np.array([v['ci95'][1] for v in vals])
      ax.bar(x,y,width=.32,color=colors[m],label=NAMES[m]);ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='none',color='#333333',capsize=3)
      for xx,yy,top in zip(x,y,hi):ax.text(xx,top+.018,f'{yy:.3f}',ha='center',fontsize=9)
     ax.axhline(.5,color='gray',ls='--',lw=1);ax.set_ylim(0,1.10);ax.set_yticks(np.linspace(0,1,6));ax.set_xticks(range(4),[NAMES[v] for v in VIEWS],rotation=20,ha='right');ax.set_title(e+' reasoning');ax.set_ylabel('AUROC (higher is better)')
    axs[0].legend(loc='lower left')
    for j,m in enumerate(METHODS):
     cc=[next(c for c in contrasts if c['contrast']=='CoT uplift' and c['method']==m and c['effort']==e) for e in EFFORTS]
     y=np.array([c['difference'] for c in cc]);lo=np.array([c['ci95'][0] for c in cc]);hi=np.array([c['ci95'][1] for c in cc]);x=np.arange(2)+(j-.5)*.3
     axs[2].errorbar(x,y,yerr=[y-lo,hi-y],fmt='o',color=colors[m],capsize=4,label=NAMES[m])
    axs[2].axhline(0,color='gray',ls='--');axs[2].set_xticks(range(2),EFFORTS);axs[2].set_title('CoT uplift');axs[2].set_ylabel('AUROC: prompt + CoT minus prompt only')
    fig.suptitle(f'Luna monitorability · {len(common)} matched question–bias cases · {len(clusters)} question clusters')
    fig.text(.5,.01,f'Same cases across methods, views and efforts. 95% paired question-cluster bootstrap intervals; {B:,} resamples. Observed switches, not causal labels.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.91]);fig.text(.5,.925,'Historical RMCT; parser-corrected evaluation labels do not repair training rewards.',ha='center',va='top',fontsize=7,color='#9b2226');fig.savefig(OUT/'monitorability-overview.png');fig.savefig(OUT/'monitorability-overview.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(15,8),sharex=True,sharey=True)
    for i,e in enumerate(EFFORTS):
     for j,v in enumerate(VIEWS):
      ax=axs[i,j]
      for m in METHODS:
       x,y=curves[m,e,v];ax.plot(x,y,color=colors[m],label=NAMES[m])
      ax.plot([0,1],[1,0],':',color='gray');ax.set_title(f'{NAMES[v]} · {e}');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('False-positive rate');ax.set_ylabel('False-negative rate')
    axs[0,0].legend();fig.suptitle('Monitoring error trade-offs · lower-left is better');fig.tight_layout(rect=[0,0,1,.92]);fig.text(.5,.94,'Parser-corrected outcomes; saved scores reused. Historical RMCT training caveat applies.',ha='center',va='top',fontsize=7,color='#9b2226');fig.savefig(OUT/'fnr-fpr-curves.png');fig.savefig(OUT/'fnr-fpr-curves.pdf');plt.close(fig)
    print(json.dumps({k:report[k] for k in ['successful_requests','common_cases','common_questions','common_dataset_counts','metrics','contrasts','usage']},indent=2))
    
    (OUT/'label-audit.json').write_text(json.dumps(audit,indent=2))
    manifest={'inputs':[{'path':str(x.resolve()),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in (args.samples,args.ledger,args.labels)],'bootstrap':B,'seed':args.seed,'planned_requests':args.planned_requests,'warnings':['Historical RMCT training is not repaired by evaluation relabelling.','Observed switches are not causal labels. Missing and invalid cases excluded, not negative.','Full 16-condition intersection; available-score secondary cohorts differ.']}
    (OUT/'input-provenance.json').write_text(json.dumps(manifest,indent=2))
    

if __name__ == '__main__':
    main()
