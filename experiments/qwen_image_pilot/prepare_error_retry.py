"""Build an additive recovery plan for recorded request errors only."""
import argparse
from pathlib import Path
from full_suite import read,write,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--original',type=Path,required=True);a=p.parse_args()
errors=read(a.original/'collected/errors.json');assert len(errors)==2
selection={r['model']:[r['id']] for r in errors};assert set(selection)=={'qwen-bct','gemma-rmct'}
write(a.root/'retry-selection.json',selection)
plan=read(a.original/'launch-plan.json')
for i in [9,15]:plan['workers'][i]['rank']=0;plan['workers'][i]['shards']=1
plan['code_sha256']={n:sha(a.root/n) for n in ['full_suite.py','full_worker.py','full.sbatch']}
plan['retry_selection_sha256']=sha(a.root/'retry-selection.json')
plan['retry_only_errors']=errors
write(a.root/'launch-plan.json',plan)
