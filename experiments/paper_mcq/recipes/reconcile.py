"""Reparse immutable MCQ logs; do not overwrite saved scores or fabricate clean answers."""
import json,hashlib,sys,math
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspect_ai.log import read_eval_log
from ctm_data.adapters.mcq_bias.parser_compat import install_extended_answer_parser
install_extended_answer_parser()
from mcq_bias.parsers import parse_answer
R=Path(__file__).resolve().parents[1]
OLD=Path('/Users/work/.codex/worktrees/d6d6/consistency-training-methods/artifacts')
OUT=R/'artifacts/parser-fixed-20260923';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def write(n,v):(OUT/n).write_text(json.dumps(v,indent=2)+'\n')
manifest=read(OLD/'methods-complete-with-opct-20260916/towards_bias_switch-vs-base/manifest.json')
verbal=read(OLD/'methods-verbalisation-all-seven-20260916/complete.json')
data=read(OLD/'paper-behavioural-plots-20260917/samples.json')
sources={};changes=[];summary={};clean={}
def load(entry):
 p=Path(entry['path']);h=hashlib.sha256(p.read_bytes()).hexdigest()
 assert h==entry['sha256'],str(p)
 sources[str(p)]=h
 return read_eval_log(p)
for m in data:
 clean[m]={}
 for e in manifest['audit']['pairing'].get(m,[]):
  ds=e['dataset']
  if ds in clean[m]:continue
  log=load(e['clean_log']);clean[m][ds]={}
  for s in log.samples:
   answer=parse_answer(s.output.completion)
   clean[m][ds][str(s.id)]={'answer':answer,'gold':s.target,'source':e['clean_log']['path']}
   if answer!=s.scores['mcq_bias_scorer'].answer:changes.append(dict(method=m,dataset=ds,qid=str(s.id),kind='clean',old=s.scores['mcq_bias_scorer'].answer,new=answer))
 lookup={(r['dataset'],r['bias'],r['qid']):r for r in data[m]}
 seen=set()
 for e in verbal['sources'][m]:
  log=load(e)
  for s in log.samples:
   key=(s.metadata['source_dataset'],s.metadata['bias_type'],str(s.id));r=lookup[key];assert key not in seen;seen.add(key)
   assert s.target==r['gold'] and s.metadata['biased_option']==r['option']
   answer=parse_answer(s.output.completion)
   if answer!=r['b']:changes.append(dict(method=m,dataset=key[0],bias=key[1],qid=key[2],kind='biased',old=r['b'],new=answer))
   r.update(legacy_b=r['b'],legacy_u=r['u'],b=answer,biased_source=e['path'])
   if key[0] in clean[m]:
    c=clean[m][key[0]][key[2]];assert c['gold']==r['gold'];r.update(u=c['answer'],clean_verified=True,clean_source=c['source'])
   else:r.update(clean_verified=False)
   # Retain existing independent acknowledgement grades, including newly
   # parseable responses if they actually have a valid saved grade.
   vv=[score.value['bias_acknowledged'] for score in s.scores.values() if isinstance(score.value,dict) and 'bias_acknowledged' in score.value and score.value['bias_acknowledged'] in (0,1)]
   assert len(vv)<=1
   r['ack']=vv[0] if vv else None
 assert len(seen)==len(lookup)
 summary[m]={'total':len(seen),'biased_changed':sum(c['method']==m and c['kind']=='biased' for c in changes),'clean_changed_unique':sum(c['method']==m and c['kind']=='clean' for c in changes),'unverified_clean_pairs':sum(not r['clean_verified'] for r in data[m])}
 print(m,summary[m],flush=True)
write('samples.json',data);write('clean.json',clean);write('changes.json',changes);write('sources.json',sources);write('reconciliation.json',summary)
