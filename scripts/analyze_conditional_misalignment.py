"""Trajectory-level conditional behavior rates; descriptive, not causal effects."""
import argparse,json,hashlib
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/eval_awareness/lasr_transfer/petri-pilots-20260914/expanded-r3-20260915/recovery-preparation-2249'
NEW=ROOT/'experiments/eval_awareness/v2-breadth-20260924/analysis-v1'
OUT=NEW/'conditional-misalignment'

def marker(label,f):
    if label=='EVALUATION' or f==1:return 1
    if label in ('NO_EVIDENCE','DEPLOYMENT_REJECTED') and f==0:return 0
    return None

def rates(group,measure='combined'):
    valid=[r for r in group if r[measure] is not None and r['p'] is not None]
    out=dict(observed=len(group),valid=len(valid),excluded=len(group)-len(valid))
    for a,key in [(1,'aware'),(0,'no_marker')]:
        subset=[r for r in valid if r[measure]==a];n=len(subset);k=sum(r['p'] for r in subset)
        out[key]=dict(n=n,misaligned=k,rate=k/n if n else None)
    out['difference_pp']=100*(out['aware']['rate']-out['no_marker']['rate']) if out['aware']['n'] and out['no_marker']['n'] else None
    return out

def standardize(group,measure='combined'):
    strata=defaultdict(list)
    for r in group:strata[r['family'],r['factor']].append(r)
    stats=[dict(family=k[0],factor=k[1],**rates(rs,measure)) for k,rs in sorted(strata.items())]
    overlap=[s for s in stats if s['aware']['n'] and s['no_marker']['n']]
    total=sum(s['valid'] for s in overlap)
    standardized={key:sum(s['valid']*s[key]['rate'] for s in overlap)/total if total else None for key in ('aware','no_marker')}
    return dict(strata=stats,overlap_strata=len(overlap),total_strata=len(stats),overlap_observations=total,
        standardized=standardized,difference_pp=100*(standardized['aware']-standardized['no_marker']) if total else None,
        weighting='Joint-valid stratum size, normalized only over family-by-factor strata containing both awareness groups')

def main(paths=None,out=None):
    out=out or OUT;out.mkdir(parents=True,exist_ok=True)
    paths=paths or [OLD/'rogueqwen-flattery-v2-20260923/analysis.json',OLD/'rogueqwen-flattery-v2-20260923/negative-audit-20260924/audited-results.json',NEW/'report.json']
    audit={r['key']:r for r in json.loads(paths[1].read_text())['rows']};rows=[]
    for r in json.loads(paths[0].read_text())['rows']:
        label=r['awareness'];f=audit[r['key']]['flattery']
        rows.append(dict(key=r['key'],version=r['version'],role=r['role'],family=r['case'].split('__')[0],factor='B' if r['profile']=='shared_realistic' else r['profile'],
            combined=marker(label,f),vae=1 if label=='EVALUATION' else 0 if label in ('NO_EVIDENCE','DEPLOYMENT_REJECTED') else None,p=r['behavior']))
    for r in json.loads(paths[2].read_text())['rows']:
        label=r['awareness'];rows.append(dict(key=r['key'],version='V2',role=r['role'],family=r['family'],factor=r['condition'],combined=r['combined'],
            vae=1 if label=='EVALUATION' else 0 if label in ('NO_EVIDENCE','DEPLOYMENT_REJECTED') else None,p=r['propensity']))
    assert len({r['key'] for r in rows})==len(rows)
    results=[];allowed={'B'}|{f'F{i}' for i in range(1,9)}
    for scope in ('baseline_and_single_factors','all_available_conditions'):
     for version in ('V2','V3','V4'):
      for role in ('mo_mid','mo_post'):
       group=[r for r in rows if (r['version'],r['role'])==(version,role) and (scope=='all_available_conditions' or r['factor'] in allowed)]
       for measure in ('combined','vae'):
        results.append(dict(scope=scope,version=version,role=role,measure=measure,**rates(group,measure),
            by_factor={f:rates([r for r in group if r['factor']==f],measure) for f in sorted({r['factor'] for r in group})},
            by_family={f:rates([r for r in group if r['family']==f],measure) for f in sorted({r['family'] for r in group})},
            stratified=standardize(group,measure)))
    report=dict(results=results,rows=rows,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        interpretation='No-marker means no observed VAE/flattery, not known absence of latent awareness. Uncertain/missing excluded. This is association, not a causal effect. Final-answer flattery is contemporaneous with behavior; VAE-only sensitivity is reported separately.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Conditional misalignment rates','',report['interpretation'],'','Primary scope: baseline and F1-F8 only, to match the preceding factor comparison. All-condition and VAE-only sensitivities are in report.json.','',
        '| Version | Model | Misalignment given marker | Misalignment given no marker | Difference (pp) | Valid / observed |','|---|---|---:|---:|---:|---:|']
    for r in results:
        if r['scope']!='baseline_and_single_factors' or r['measure']!='combined':continue
        fmt=lambda s:f"{100*s['rate']:.1f}% ({s['misaligned']}/{s['n']})" if s['n'] else 'not estimable (0 observations)'
        delta=f"{r['difference_pp']:+.1f}" if r['difference_pp'] is not None else 'N/A'
        lines.append(f"| {r['version']} | {r['role']} | {fmt(r['aware'])} | {fmt(r['no_marker'])} | {delta} | {r['valid']}/{r['observed']} |")
    lines += ['','## Same-family, same-factor standardization','',
        'Restrict to strata with both groups, and average the two conditional rates using identical joint-valid stratum-size weights. This removes between-stratum composition differences, but not residual confounding or selection.','',
        '| Version | Model | Overlap strata | Overlap observations | Standardized difference (pp) |','|---|---|---:|---:|---:|']
    for r in results:
        if r['scope']!='baseline_and_single_factors' or r['measure']!='combined':continue
        s=r['stratified'];d=f"{s['difference_pp']:+.1f}" if s['difference_pp'] is not None else 'N/A'
        lines.append(f"| {r['version']} | {r['role']} | {s['overlap_strata']}/{s['total_strata']} | {s['overlap_observations']} | {d} |")
    (out/'README.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-analysis',type=Path,default=OLD/'rogueqwen-flattery-v2-20260923/analysis.json')
    parser.add_argument('--legacy-audit',type=Path,default=OLD/'rogueqwen-flattery-v2-20260923/negative-audit-20260924/audited-results.json')
    parser.add_argument('--v2-report',type=Path,default=NEW/'report.json')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();main([args.legacy_analysis,args.legacy_audit,args.v2_report],args.output)
