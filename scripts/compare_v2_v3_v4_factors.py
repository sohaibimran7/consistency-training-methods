"""Offline matched-baseline contrasts; no generation, grading, or imputation."""
import argparse,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/eval_awareness/lasr_transfer/petri-pilots-20260914/expanded-r3-20260915/recovery-preparation-2249'
NEW=ROOT/'experiments/eval_awareness/v2-breadth-20260924/analysis-v1'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--legacy-analysis',type=Path,default=OLD/'rogueqwen-flattery-v2-20260923/analysis.json')
parser.add_argument('--legacy-audit',type=Path,default=OLD/'rogueqwen-flattery-v2-20260923/negative-audit-20260924/audited-results.json')
parser.add_argument('--v2-report',type=Path,default=NEW/'report.json')
parser.add_argument('--output',type=Path,default=NEW/'cross-version-factor-analysis')
args=parser.parse_args()
OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
paths=[args.legacy_analysis,args.legacy_audit,args.v2_report]
raw=json.loads(paths[0].read_text())['rows'];audit={r['key']:r for r in json.loads(paths[1].read_text())['rows']}
rows=[]
for r in raw:
    f=audit[r['key']]['flattery'];a=r['awareness']
    c=1 if a=='EVALUATION' or f else 0 if a in ('NO_EVIDENCE','DEPLOYMENT_REJECTED') else None
    rows.append(dict(key=r['key'],version=r['version'],role=r['role'],family=r['case'].split('__')[0],source=r['case'].split('__')[0]+'::'+str(r['epoch']),factor='B' if r['profile']=='shared_realistic' else r['profile'],a=c,p=r['behavior']))
for r in json.loads(paths[2].read_text())['rows']:
    rows.append(dict(key=r['key'],version='V2',role=r['role'],family=r['family'],source=r['source_id'],factor=r['condition'],a=r['combined'],p=r['propensity']))
lookup={(r['version'],r['role'],r['source'],r['factor']):r for r in rows}
assert len(lookup)==len(rows)
contrasts=[]
for v in ('V2','V3','V4'):
 for role in ('mo_mid','mo_post'):
  for f in [f'F{i}' for i in range(1,9)]:
   group=[r for r in rows if (r['version'],r['role'],r['factor'])==(v,role,f)]
   for family in [None]+sorted({r['family'] for r in group}):
    selected=[r for r in group if family is None or r['family']==family];pairs=[]
    for r in selected:
        b=lookup.get((v,role,r['source'],'B'))
        if b and all(t[k] is not None for t in (r,b) for k in ('a','p')):pairs.append((r,b))
    n=len(pairs)
    contrasts.append(dict(version=v,role=role,factor=f,family=family or 'pooled',observed=len(selected),matched=n,
        delta_awareness_pp=100*sum(r['a']-b['a'] for r,b in pairs)/n if n else None,
        delta_propensity_pp=100*sum(r['p']-b['p'] for r,b in pairs)/n if n else None,
        factor_awareness=100*sum(r['a'] for r,b in pairs)/n if n else None,
        baseline_awareness=100*sum(b['a'] for r,b in pairs)/n if n else None,
        factor_propensity=100*sum(r['p'] for r,b in pairs)/n if n else None,
        baseline_propensity=100*sum(b['p'] for r,b in pairs)/n if n else None,
        sources=[r['source'] for r,b in pairs],keys=[(r['key'],b['key']) for r,b in pairs]))
result=dict(contrasts=contrasts,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    rule='Within checkpoint, match factor to B on source (V2) or scenario/epoch (V3/V4); require both outcomes valid in both arms. Positive marker=VAE EVALUATION or flattery; uncertainty is missing, not zero.',
    limitations=['V2 is new breadth adaptation, not old two-repeat rescue-only pilot','F8 differs: V2 output contract versus V3/V4 explicit disclosure','V3/V4 audited flattery versus V2 single-pass flattery','V2 family mix differs by factor support; no pooled causal estimate','V3/V4 repeated seeds are not independent narratives','Epoch matching is design alignment, not proof of common random numbers','This stricter cohort differs from older marginal-difference figures'])
(OUT/'contrasts.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Cross-version single-factor effects','',result['rule'],'','Each entry is delta awareness / delta propensity in percentage points, followed by matched n. V2 pooled values are descriptive across the supported family mix, not a common population effect.','']
for role in ('mo_mid','mo_post'):
    lines += ['## '+role,'','| Factor | V2 breadth | V3 | V4 |','|---|---:|---:|---:|']
    for f in [f'F{i}' for i in range(1,9)]:
        vals=[]
        for v in ('V2','V3','V4'):
            c=next(c for c in contrasts if (c['version'],c['role'],c['factor'],c['family'])==(v,role,f,'pooled'))
            vals.append(f"{c['delta_awareness_pp']:+.1f} / {c['delta_propensity_pp']:+.1f} (n={c['matched']})" if c['matched'] else 'unavailable')
        lines.append('| '+f+' | '+' | '.join(vals)+' |')
    lines.append('')
lines += ['## Limits','']+['- '+x for x in result['limitations']]
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
