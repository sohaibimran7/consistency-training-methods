"""Compare explicit-path offline reproductions to immutable historical results.

Run the commands in the handoff runbook first. No model, network, or writes.
"""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
V2=ROOT/'experiments/eval_awareness/v2-breadth-20260924'
OLD=ROOT/'experiments/eval_awareness/lasr_transfer/petri-pilots-20260914/expanded-r3-20260915/recovery-preparation-2249'

def read(path):return json.loads(path.read_text())

def main(output):
    checks=[]
    def compare(old,new,fields,label):
        a,b=read(old),read(new)
        for field in fields:assert a[field]==b[field],(label,field)
        checks.append(label)
    compare(V2/'analysis-v1/report.json',output/'v2/report.json',['rows','cells','coverage','limitations'],'V2 rows, denominators and figures data')
    compare(V2/'analysis-v1/cross-version-factor-analysis/contrasts.json',output/'contrasts/contrasts.json',['contrasts','rule','limitations'],'matched factor contrasts')
    compare(V2/'analysis-v1/conditional-misalignment/report.json',output/'conditional/report.json',['rows','results','interpretation'],'conditional rates and standardization')
    compare(V2/'analysis-v1/fit-statistics.json',output/'fits/fit-statistics.json',['panels','seed','bootstrap_draws'],'OLS source-bootstrap statistics')
    compare(OLD/'analysis-vae-or-flattery-20260924/figure-data.json',output/'legacy/figure-data.json',['cells','correlations','combined_positive','unresolved','rule','sensitivity'],'legacy overlays and associations')
    a=read(V2/'grading-v1/awareness-basis-repair.json');b=read(output/'awareness-basis-repair.json')
    strip=lambda rows:[{k:v for k,v in r.items() if k!='source_path'} for r in rows]
    assert strip(a['rows'])==strip(b['rows'])
    assert a['aliases']==b['aliases'] and a['source_plan_sha256']==b['source_plan_sha256']
    checks.append('offline alias repair identical judgments and source hashes')
    a=read(V2/'grading-v1/native/plan.json');b=read(output/'native-plan/plan.json')
    assert a['jobs']==b['jobs'] and a['ie_prompt_sha256']==b['ie_prompt_sha256']
    checks.append('native grading inputs all656 jobs identical')
    print(json.dumps(dict(passed=checks,ignored='Relocated provenance path strings and new implementation hash only; no outcome differences allowed.'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
