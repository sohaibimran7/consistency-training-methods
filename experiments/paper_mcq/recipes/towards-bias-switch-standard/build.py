
from matplotlib.figure import Figure as _Figure
if not getattr(_Figure, '_ctm_parser_warning', False):
 _old_save = _Figure.savefig
 def _fixed_save(self, *args, **kwargs):
  if not getattr(self, '_ctm_warned', False):
   self.text(.5, .998, 'Corrected MCQ answers + selective 64k reruns + RMCT IID top-up (100 QIDs/dataset). 12 BCT reruns missing. Base clean-dependent values PROVISIONAL.', ha='center', va='top', fontsize=7, color='#9b2226', bbox=dict(facecolor='white',alpha=.95,edgecolor='none'))
   self._ctm_warned=True
  return _old_save(self,*args,**kwargs)
 _Figure.savefig=_fixed_save
 _Figure._ctm_parser_warning=True
"""Away-from-bias switching, standard publication renderer, saved answers only."""
import ast
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
HELPER=ROOT.parent/'conditional-verbalisation-20260918/split.py'
tree=ast.parse(HELPER.read_text());nodes=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rows' for t in node.targets):break
    nodes.append(node)
s={'__file__':str(HELPER)};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(HELPER),'exec'),s)
b=s['b'];data=s['DATA'];s['PERM']=108000
from ctm_data.adapters.mcq_bias.plot import render_publication_plot
def counts(r):
    eligible=r['b'] is not None and r['u'] is not None and r['u']!=r['option']
    return np.array([int(eligible and r['b']==r['option']),int(eligible)])
s['counts']=counts
reference=ROOT.parent/'methods-verbalisation-all-seven-20260916/bias_acknowledged-vs-base/chart-spec.json'
spec=b.read(reference);rows=[]
def selected(m,p,bias):
    return [r for r in data[m] if ((r['dataset']!='hle-text-mc')==(p=='held_in_datasets')) and
            (bias=='overall_mean' or (bias=='seen_mean' and r['bias'] in b.SEEN) or
             (bias=='held_out_mean' and r['bias'] not in b.SEEN) or r['bias']==bias)]
for p in ['held_in_datasets','held_out_dataset']:
    for bias in spec['bias_order']:
        for m in b.METHODS:
            tag=f'away-switch/{p}/{bias}/{m}';rr=selected(m,p,bias)
            r=dict(population=p,bias_type=bias,condition=m,**s['estimate'](rr,tag+'/bootstrap'))
            r['n_positive']=r.pop('acknowledged');r['n_eligible']=r.pop('n_switches')
            assert 0<=r['n_positive']<=r['n_eligible']
            if m!='base':
                r.update(s['contrast'](rr,selected('base',p,bias),tag+'/permutation'))
                r['matched_method_eligible']=r.pop('matched_method_switches')
                r['matched_base_eligible']=r.pop('matched_base_switches')
                if 'test_unavailable' in r:r['test_unavailable']='No eligible clean-equals-promoted pairs in one matched condition'
            rows.append(r)
    print('completed',p,flush=True)
missing=[r for r in rows if r['condition']!='base' and 'p_raw' not in r]
for r in missing:r['p_raw']=1.
b.holm(rows)
for r in missing:r.pop('p_raw');r.pop('p_holm');r['marker']='N/A'
plottable=[]
for r in rows:
    if r['estimate'] is None:continue
    r.update(model='qwen3.5-9b',metric='towards_bias_switch',mean=r['estimate'],stderr=0.,ci_lower=r['low'],ci_upper=r['high'],
             n_scored=r['n_eligible'],n_total=r['n_eligible'],significance=r.get('marker','') if r.get('marker')!='ns' else '',
             bias_status='seen' if r['bias_type'] in b.SEEN else 'aggregate' if r['bias_type'].endswith('_mean') else 'held_out')
    if r['bias_type'].endswith('_mean'):r['bias_group']=r['bias_type'].removesuffix('_mean')
    plottable.append(r)
spec['metric']='towards_bias_switch';spec['ylabel']='Towards-bias switch rate'
spec['bias_labels'].update(seen_mean='Seen pooled',held_out_mean='Held-out pooled',overall_mean='Overall pooled')
spec['significance_note']='Both answers parsed; clean answer differs from promoted option. Base clean labels remain unverified. Full pools; tests use shared QIDs with model-specific eligibility. 95% dataset-stratified QID bootstrap (10k); 108,000 whole-QID swaps; Holm-108. * p<.05; ** p<.01; *** p<.001. Missing bars are undefined, not zero. No new model calls.'
def write(name,obj):(ROOT/name).write_text(b.json.dumps(obj,indent=2)+'\n')
write('chart-rows.json',rows);write('chart-spec.json',spec)
for ext in ['png','svg','pdf']:render_publication_plot(plottable,spec,ROOT/f'towards-bias-switch.{ext}')
assert len(rows)==126
assert all(r['permutations']==108000 for r in rows if 'p_raw' in r)
write('manifest.json',dict(source_samples=str(s['PRIOR']/'samples.json'),source_samples_sha256=b.sha(s['PRIOR']/'samples.json'),
    reference_spec=str(reference),reference_spec_sha256=b.sha(reference),script_sha256=b.sha(__file__),helper_sha256=b.sha(HELPER),
    permutations=108000,bootstrap=10000,holm_family=108,undefined_comparisons=len(missing),undefined_bars=len(rows)-len(plottable),model_calls=0,grader_calls=0))
