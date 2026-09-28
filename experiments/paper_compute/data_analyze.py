"""Reproduce count-matched candidates without claiming identical training data."""
import json,hashlib
from pathlib import Path
from collections import Counter
from common import artifact_directory, unique_match
ROOT=artifact_directory("data-matched-checkpoints-20260926")
ARTIFACT_ROOT=ROOT.parent
METHODS=['bct','act','attct','mlpct','opct','rmct']
def read(p):return json.loads(p.read_text())
def digest(x):return hashlib.sha256(json.dumps(x,separators=(',',':'),sort_keys=True).encode()).hexdigest()
def order(p):
 d=read(p)['datasets'];return [q for pair in zip(d['logiqa']['permutation'],d['hellaswag']['permutation']) for q in pair]
SMALL=unique_match(ARTIFACT_ROOT/'rmct-shared-qid-two-bias-20260813', '*n1000-cc784*.manifest-eac*.json')
LARGE=unique_match(ARTIFACT_ROOT/'act-expanded-shared-8192-20260910/shared-two-bias', '*.manifest-*.json')
orders={'rmct':order(SMALL),'others':order(LARGE)}
raw=read(ROOT/'remote-inventory.json');g=read(ROOT/'gemma-rmct.json');extra=read(ROOT/'exposure-receipts.json')
inventory={fam:{m:{} for m in METHODS} for fam in ['qwen','gemma']}
metrics={fam:{m:{} for m in METHODS} for fam in ['qwen','gemma']}
for run in raw['runs']:
 fam='gemma' if 'gemma' in run['root'] else 'qwen';m=run['method']
 for cp in run['checkpoints']:
  step=int(Path(cp['path']).name.split('-')[-1]);c=cp['manifest.json']['loop_state']['convergence']
  assert c['step']==step
  st=cp.get('safetensors_structure',{})
  assert st.get('max_offset')==st.get('payload_bytes') and st.get('tensor_count',0)>0
  inventory[fam][m][step]={'path':cp['path'],'optimizer_updates':step,'attempted_groups':c.get('attempts',step),'files':cp['files'],'adapter_structure_valid':True,'best_own_loss_step':c.get('best_step')}
 for r in run['sealed_metrics']:
  if r['step'] in metrics[fam][m]:assert metrics[fam][m][r['step']]['question_ids']==r['question_ids']
  metrics[fam][m][r['step']]=r
for run in raw['rmct']:
 rec=run.get('segment/checkpoint-receipt.json',{});step=rec.get('optimizer_step')
 if step is None:continue # copied parent or failed attempt, not a new committed segment
 cps=[cp for cp in run['checkpoints'] if cp['manifest.json']['loop_state'].get('final')]
 assert len(cps)==1
 cp=cps[0];loop=cp['manifest.json']['loop_state'];assert loop['optimizer_step']==step
 inventory['qwen']['rmct'][step]={'path':cp['path'],'optimizer_updates':step,'attempted_groups':loop['global_step'],'files':cp['files'],'adapter_structure_valid':cp['safetensors_structure']['max_offset']==cp['safetensors_structure']['payload_bytes']}
for cp in g['checkpoints']:
 l=cp['manifest']['loop_state'];a=l['global_step'];s=l['optimizer_step']
 inventory['gemma']['rmct'][a]={'path':cp['path'],'optimizer_updates':s,'attempted_groups':a,'files':cp['files'],'adapter_structure_valid':None}
for fam in ['qwen','gemma']:
 for m in METHODS[:-1]:
  rs=metrics[fam][m];assert sorted(rs)==list(range(1,max(rs)+1))
  for s,r in rs.items():
   a=r.get('attempt',s-1)
   assert r['question_ids']==orders['others'][2*a:2*a+2]
for f,rs in extra['qwen_rmct_metrics'].items():
 for r in rs:
  if 'train/optimizer_step' in r:
   assert r['step']==r['train/optimizer_step'] and not r['train/skipped_empty_batch']
   metrics['qwen']['rmct'][r['step']]=r
assert sorted(metrics['qwen']['rmct'])==list(range(1,353))
for source in [g['optimizer_metrics'],extra['gemma_early_optimizer_metrics']]:
 for rs in source.values():
  for r in rs:
   s=r['train/optimizer_step'];assert s==r['step'] and not r['train/skipped_empty_batch']
   metrics['gemma']['rmct'][s]=r
assert sorted(metrics['gemma']['rmct'])==list(range(1,193))
def summary(fam,m,cp):
 a=cp['attempted_groups'];s=cp['optimizer_updates'];ids=orders['rmct' if m=='rmct' else 'others'][:2*a]
 assert len(ids)==2*a and len(set(ids))==len(ids)
 if m=='rmct':
  globals_=[metrics[fam][m][i].get('gemma/source_global_step',i) for i in range(1,s+1)]
  selected=[q for t in globals_ for q in orders['rmct'][2*(t-1):2*t]]
 else:selected=[q for i in range(1,s+1) for q in metrics[fam][m][i]['question_ids']]
 return {**cp,'unique_questions_encountered':len(set(ids)),'qid_bias_opportunities_encountered':2*len(ids),'questions_in_updating_groups':len(selected),'qid_bias_opportunities_in_updating_groups':2*len(selected),'skipped_groups':a-s,'question_order_sha256':digest(ids),'optimized_group_qid_order_sha256':digest(selected),'repeated_qid_exposures':len(ids)-len(set(ids)),'question_pool_size':len(orders['rmct' if m=='rmct' else 'others'])}
latest={fam:{m:summary(fam,m,max(cps.values(),key=lambda x:(x['optimizer_updates'],x['attempted_groups']))) for m,cps in ms.items()} for fam,ms in inventory.items()}
def candidates(fam,budget):
 result={}
 for m,cps in inventory[fam].items():
  xs=[c for c in cps.values() if c['attempted_groups']*2==budget]
  if not xs:return None
  result[m]=summary(fam,m,max(xs,key=lambda x:x['optimizer_updates']))
 return result
def budgets(fam,methods=METHODS):
 return set.intersection(*[{c['attempted_groups']*2 for c in inventory[fam][m].values()} for m in methods])
maximum={fam:max(budgets(fam)) for fam in inventory}
common=max(budgets('qwen')&budgets('gemma'))
exact_matches={}
for fam in inventory:
 exact=[]
 for n in sorted(budgets(fam)):
  cs=candidates(fam,n)
  assert set(orders['rmct'][:n]) != set(orders['others'][:n]), 'Unexpected exact question-set match requires review'
  if len({c['question_order_sha256'] for c in cs.values()})==1 and len({c['optimized_group_qid_order_sha256'] for c in cs.values()})==1:exact.append(n)
 exact_matches[fam]=exact
assert exact_matches=={'qwen':[],'gemma':[]}
subsets={}
for fam in inventory:
 matches=[]
 for n in budgets(fam,METHODS[:-1]):
  cs={m:summary(fam,m,next(c for c in inventory[fam][m].values() if c['attempted_groups']*2==n)) for m in METHODS[:-1]}
  if len({c['optimized_group_qid_order_sha256'] for c in cs.values()})==1:matches.append(n)
 n=max(matches);subsets[fam]={'unique_qids':n,'optimizer_updates':n//2}
comparisons={}
for n in sorted(set(maximum.values())|{common}):
 small=set(orders['rmct'][:n]);big=set(orders['others'][:n])
 comparisons[str(n)]={'rmct_vs_other_intersection':len(small&big),'each_set_size':n,'jaccard':len(small&big)/len(small|big),'identical_order':orders['rmct'][:n]==orders['others'][:n]}
out={'schema':'ctm-data-matched-checkpoint-audit-v1','observed_utc':raw['observed_utc'],'method_order':METHODS,'latest':latest,'max_question_count_budgets':maximum,'max_cross_family_question_count_budget':common,'within_family_question_count_candidates':{fam:candidates(fam,n) for fam,n in maximum.items()},'cross_family_question_count_candidates':{fam:candidates(fam,common) for fam in inventory},'exact_all_method_data_matches':exact_matches,'exact_non_rmct_qid_sequence_subset':subsets,'pool_intersection':len(set(orders['rmct'])&set(orders['others'])),'pool_order_digests':{k:digest(v) for k,v in orders.items()},'candidate_qid_overlap':comparisons,'retained_steps':{fam:{m:sorted(cps) for m,cps in ms.items()} for fam,ms in inventory.items()},'retained_step_note':'Gemma RMCT indexed by global attempted group; all others by optimizer update.','loadability_scope':'Remote file existence/config/manifest and safetensors structural checks only; no fresh model load or GPU evaluation.','limits':['Matching number of encountered questions is not matching exact QIDs or gradient contributions.','Updating-group question counts are opportunities, not proof every question or sampled response had nonzero gradient.','Historical Gemma thinking-off; no corrected thinking-on checkpoint lineup exists.','Failed/replayed attempts outside retained lineage excluded; sampled token/repeated rollout exposure is not matched.']}
(ROOT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'maxima':maximum,'cross_family':common,'exact_matches':exact_matches,'non_rmct':subsets,'overlaps':comparisons,'latest':{fam:{m:{k:c[k] for k in ['optimizer_updates','attempted_groups','unique_questions_encountered','questions_in_updating_groups']} for m,c in ms.items()} for fam,ms in latest.items()}},indent=2))
