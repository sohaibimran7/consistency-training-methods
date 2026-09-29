
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
"""All-generated biased accuracy using saved answers and standard figures."""
import importlib.util,copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'paper-behavioural-plots-20260917'
loader=importlib.util.spec_from_file_location('prior',PRIOR/'build.py');b=importlib.util.module_from_spec(loader);loader.loader.exec_module(b)
b.ROOT=ROOT;b.PERM=108000
from ctm_data.adapters.mcq_bias.plot import render_publication_plot
data=b.read(PRIOR/'samples.json');metric='accuracy_all_generated'
reference=ROOT.parent/'methods-verbalisation-all-seven-20260916/bias_acknowledged-vs-base/chart-spec.json'
template=b.read(reference)
def select(m,p,bias):
    datasets=b.DATASETS[:2] if p=='held_in_datasets' else ['hle-text-mc'] if p=='held_out_dataset' else [p]
    return [r for r in data[m] if r['dataset'] in datasets and (bias=='overall_mean' or bias=='seen_mean' and r['bias'] in b.SEEN or bias=='held_out_mean' and r['bias'] not in b.SEEN or r['bias']==bias)]
def analyse(pops,biases,family):
    rows=[]
    for p in pops:
        for bias in biases:
            for m in b.METHODS:
                rr=select(m,p,bias);tag=f'biased-accuracy/{family}/{p}/{bias}/{m}'
                r=dict(population=p,bias_type=bias,condition=m,**b.estimate(rr,metric,tag+'/bootstrap'))
                r['n_correct']=sum(x['b']==x['gold'] for x in rr);r['n_parsed']=sum(x['b'] is not None for x in rr)
                assert abs(r['estimate']-r['n_correct']/len(rr))<1e-12
                if m!='base':r.update(b.contrast(rr,select('base',p,bias),metric,tag+'/permutation'))
                rows.append(r)
        print('completed',family,p,flush=True)
    b.holm(rows)
    for r in rows:
        r.update(model='qwen3.5-9b',metric=metric,mean=r['estimate'],stderr=0.,ci_lower=r['low'],ci_upper=r['high'],n_scored=r['n_pairs'],n_total=r['n_pairs'],
                 significance=r.get('marker','') if r.get('marker')!='ns' else '',bias_status='seen' if r['bias_type'] in b.SEEN else 'aggregate' if r['bias_type'].endswith('_mean') else 'held_out')
        if r['bias_type'].endswith('_mean'):r['bias_group']=r['bias_type'].removesuffix('_mean')
    b.write(f'{family}-rows.json',rows)
    return rows
paper=analyse(['held_in_datasets','held_out_dataset'],template['bias_order'],'paper')
spec=copy.deepcopy(template);spec['metric']=metric;spec['ylabel']='Biased accuracy (all generated responses)'
spec['bias_labels'].update(seen_mean='Seen pooled',held_out_mean='Held-out pooled',overall_mean='Overall pooled')
spec['significance_note']='Accuracy = correct biased responses / ALL generated biased responses. Unparsed answers count as incorrect; no clean-parse or acknowledgement-grade filter. n = generated question–bias pairs.\nFull available pools shown; RMCT step 352 uses 50+50 seen-dataset QIDs, others 100+100; HLE 100 each. Comparisons vs Base use shared QIDs.\n95% dataset-stratified QID-cluster bootstrap intervals (10k); 108,000 whole-QID label swaps, Holm-108. * p<.05; ** p<.01; *** p<.001.\nPooled ratios use summed correct/generated counts. Question-level uncertainty, not training-seed uncertainty. Answers reparsed from saved responses; no new evaluations.'
b.write('paper-spec.json',spec)
for ext in ['png','svg','pdf']:render_publication_plot(paper,spec,ROOT/f'biased-accuracy-paper-style.{ext}')
detail=analyse(b.DATASETS,['seen_mean','held_out_mean'],'by-dataset')
def summary(rows,pops,titles,name,family_size):
    fig,axs=b.plt.subplots(2,len(pops),figsize=(6*len(pops),9),sharey=True,squeeze=False)
    fig.suptitle('Biased accuracy · seen and held-out biases',fontsize=17)
    for i,bias in enumerate(['seen_mean','held_out_mean']):
        for j,(p,title) in enumerate(zip(pops,titles)):
            ax=axs[i,j]
            for x,m in enumerate(b.METHODS):
                r=next(r for r in rows if (r['population'],r['bias_type'],r['condition'])==(p,bias,m));y=100*r['mean']
                ax.bar(x,y,color=b.COLORS[x]);ax.errorbar(x,y,yerr=[[max(0,y-100*r['low'])],[max(0,100*r['high']-y)]],fmt='none',color='#333',capsize=2)
                ax.text(x,100*r['high']+2,r.get('marker',''),ha='center',fontsize=9)
                ax.text(x,105,f"n={r['n_pairs']}",ha='center',fontsize=8,rotation=30)
            b.ticks(ax);ax.set_ylim(0,118);ax.set_yticks(range(0,101,20));ax.set_ylabel('Biased accuracy (%)');ax.set_title(('Seen biases' if i==0 else 'Held-out biases')+' · '+title)
    b.save(fig,name,f'All generated responses; unparsed = incorrect. n = question–bias pairs. 95% QID-cluster bootstrap intervals (10k).\n108,000 whole-QID swaps vs Base; Holm-{family_size}. Full pools shown; comparisons use shared QIDs. HLE remains a held-out dataset.\nAnswer parser corrected; original responses unchanged. No new model or grader calls.')
summary(paper,['held_in_datasets','held_out_dataset'],['Seen datasets: LogiQA + HellaSwag','Held-out dataset: HLE'],'biased-accuracy-seen-heldout',108)
summary(detail,b.DATASETS,b.TITLES,'biased-accuracy-by-dataset',36)
for p in ['held_in_datasets','held_out_dataset']:
    for m in b.METHODS:
        for bias_group,members in [('seen_mean',b.SEEN),('held_out_mean',set(template['bias_order'][:6])-b.SEEN),('overall_mean',set(template['bias_order'][:6]))]:
            rr=[r for r in paper if r['population']==p and r['condition']==m and r['bias_type'] in members]
            agg=next(r for r in paper if (r['population'],r['condition'],r['bias_type'])==(p,m,bias_group))
            assert sum(r['n_pairs'] for r in rr)==agg['n_pairs'] and sum(r['n_correct'] for r in rr)==agg['n_correct']
b.write('manifest.json',dict(samples_sha256=b.sha(PRIOR/'samples.json'),script_sha256=b.sha(__file__),permutations=108000,bootstrap=10000,holm_families={'paper':108,'by_dataset':36},pooled_counts_verified=True,model_calls=0,grader_calls=0))
