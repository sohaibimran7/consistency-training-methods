
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
"""Use the established main-paper renderer for the conditional endpoint."""
import ast,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
# Load the analysis definitions without executing split.py's plotting driver:
# a new publication must not rewrite the earlier split figures.
helper=ROOT/'split.py'
tree=ast.parse(helper.read_text(),filename=str(helper));definitions=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rows' for t in node.targets):break
    definitions.append(node)
s={'__file__':str(helper)}
exec(compile(ast.Module(body=definitions,type_ignores=[]),str(helper),'exec'),s)
PERM=int(os.environ.get('CTM_CONDITIONAL_PERMUTATIONS','50000'))
s['PERM']=PERM;b=s['b'];data=s['DATA']
from ctm_data.adapters.mcq_bias.plot import render_publication_plot
REFERENCE=ROOT.parent/'methods-verbalisation-all-seven-20260916/bias_acknowledged-vs-base'
spec=b.read(REFERENCE/'chart-spec.json');rows=[]
def selected(m,p,bias):
    return [r for r in data[m] if ((r['dataset']!='hle-text-mc')==(p=='held_in_datasets')) and
            (bias=='overall_mean' or (bias=='seen_mean' and r['bias'] in b.SEEN) or
             (bias=='held_out_mean' and r['bias'] not in b.SEEN) or r['bias']==bias)]
for p in ['held_in_datasets','held_out_dataset']:
    for bias in spec['bias_order']:
        for m in b.METHODS:
            tag=f'conditional-paper/{p}/{bias}/{m}';rr=selected(m,p,bias)
            r=dict(population=p,bias_type=bias,condition=m,**s['estimate'](rr,tag+'/boot'))
            if m!='base':r.update(s['contrast'](rr,selected('base',p,bias),tag+'/perm'))
            rows.append(r)
    print('paper completed',p,flush=True)
missing=[r for r in rows if r['condition']!='base' and 'p_raw' not in r]
for r in missing:r['p_raw']=1.
b.holm(rows)
for r in missing:
    r.pop('p_raw');r.pop('p_holm');r['marker']='N/A'
for r in rows:
    assert r['estimate'] is not None,'Standard renderer must not display undefined values as zero'
    r.update(model='qwen3.5-9b',metric='acknowledgement_given_towards_switch',mean=r['estimate'],stderr=0.,
             ci_lower=r['low'],ci_upper=r['high'],n_scored=r['n_switches'],n_total=r['n_switches'],
             significance=r.get('marker','') if r.get('marker')!='ns' else '',
             bias_status='seen' if r['bias_type'] in b.SEEN else 'aggregate' if r['bias_type'].endswith('_mean') else 'held_out')
    if r['bias_type'].endswith('_mean'):r['bias_group']=r['bias_type'].removesuffix('_mean')
spec['metric']='acknowledgement_given_towards_switch'
spec['ylabel']='Bias verbalised | towards-bias switch'
spec['bias_labels'].update(seen_mean='Seen pooled',held_out_mean='Held-out pooled',overall_mean='Overall pooled')
spec['significance_note']='Acknowledgement conditional on a towards-bias switch: both answers parsed, clean ≠ promoted option, biased = promoted option, valid grade. n = switched responses.\nFull pools shown; each model has its own switch subset. Stars vs Base: 50,000 whole-QID swaps on shared records, method-specific switch denominators; Holm across 108 contrasts.\n95% QID-cluster bootstrap intervals (10k; stratified by dataset); 0%/100% groups have degenerate intervals. Pooled summaries are ratios of summed counts, not equal-bias averages.\nSelection differences are not causal effects. Answer parser corrected; original responses unchanged. * p<.05; ** p<.01; *** p<.001. No new model/grader calls.'
spec['significance_note']=spec['significance_note'].replace('50,000',f'{PERM:,}')
out=ROOT/('main-paper-style' if PERM==50000 else f'main-paper-style-{PERM}');out.mkdir(exist_ok=True)
(out/'chart-rows.json').write_text(b.json.dumps(rows,indent=2)+'\n')
(out/'chart-spec.json').write_text(b.json.dumps(spec,indent=2)+'\n')
for ext in ['png','svg','pdf']:render_publication_plot(rows,spec,out/f'conditional-bias-verbalisation.{ext}')
(out/'manifest.json').write_text(b.json.dumps(dict(reference_spec=str(REFERENCE/'chart-spec.json'),reference_spec_sha256=b.sha(REFERENCE/'chart-spec.json'),samples_sha256=b.sha(s['PRIOR']/'samples.json'),script_sha256=b.sha(__file__),permutations=PERM,holm_family=108,minimum_raw_p=1/(PERM+1),worst_case_holm_floor=108/(PERM+1),model_calls=0,grader_calls=0),indent=2)+'\n')
