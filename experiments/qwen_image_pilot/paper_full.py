"""Image-suite adapter for the existing main-paper renderer and paired tests."""
import argparse,json,sys,math,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from experiments.rmct_two_bias_eval.checkpoint_publication import _paired_label_swap,_significance_marker
from recovery import load_rows
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--template',type=Path,required=True);a=p.parse_args()
ROOT=a.root
rows=load_rows(ROOT)
lookup={(r['model'],r['qid'],r['condition']):r for r in rows}
methods=['base','act','attct','mlpct','bct','opct','rmct']
names=dict(zip(methods,['Base','ACT','AttCT','MLPCT','BCT','OPCT','RMCT']))
datasets=['logiqa','hellaswag','hle-text-mc']
biases=['suggested_answer','sampled_shape','spurious_few_shot_squares']
base_spec=json.loads(a.template.read_text())
out=a.output;out.mkdir(parents=True,exist_ok=False)
def count(r,metric):
    if r['error']:return (0,0)
    if metric=='accuracy':return (r['scores']['accuracy'],1)
    clean=lookup[r['model'],r['qid'],'clean']
    ok=not clean['error'] and clean['scores'].get('parsed') and r['scores'].get('parsed') and clean['answer']!=r['biased_option']
    return (int(r['answer']==r['biased_option']),1) if ok else (0,0)
for family in ['qwen','gemma']:
    ms=methods if family=='qwen' else ['base','rmct']
    for metric in ['accuracy','towards_bias_switch']:
        bs=['clean',*biases] if metric=='accuracy' else biases
        chart=[];tests=[]
        for dataset in datasets:
            for bias in bs:
                clusters={m:{r['qid']:count(r,metric) for r in rows if r['model']==family+'-'+m and r['dataset']==dataset and r['condition']==bias} for m in ms}
                for m in ms:
                    success=sum(v[0] for v in clusters[m].values());n=sum(v[1] for v in clusters[m].values());mean=success/n
                    z=1.959963984540054;den=1+z*z/n;mid=(mean+z*z/(2*n))/den;rad=z*math.sqrt(mean*(1-mean)/n+z*z/(4*n*n))/den
                    row=dict(condition=m,condition_label=names[m],method=m,model=family,population=dataset,bias_type=bias,bias_status='seen' if bias=='suggested_answer' else 'held_out' if bias!='clean' else 'aggregate',metric=metric,mean=mean,stderr=math.sqrt(mean*(1-mean)/n),ci_lower=mid-rad,ci_upper=mid+rad,n_scored=n,significance='')
                    if m!='base':
                        row.update(_paired_label_swap(clusters[m],clusters['base'],treatment_name=family+'-'+m,metric=metric,population=dataset,bias_type=bias,permutations=100000));tests.append(row)
                    chart.append(row)
        running=0
        for i,row in enumerate(sorted(tests,key=lambda r:r['p_value_raw'])):
            running=min(1,max(running,(len(tests)-i)*row['p_value_raw']));row.update(p_value=running,p_value_holm=running,significance=_significance_marker(running),holm_family_size=len(tests))
        spec=json.loads(json.dumps(base_spec));spec.update(metric=metric,condition_order=ms,condition_labels=names,model_order=[family],model_labels={family:'Qwen3.5-9B' if family=='qwen' else 'Gemma-4-12B'},bias_order=bs,bias_labels={'clean':'Clean','suggested_answer':'Suggested answer\n(seen cue, image input)','sampled_shape':'Sampled tick / circle\n(held-out visual cue)','spurious_few_shot_squares':'Black-square few-shot\n(text cue, image input)'},facet={'rows':['population']},facet_labels={'population':{'logiqa':'LogiQA · held-in dataset','hellaswag':'HellaSwag · held-in dataset','hle-text-mc':'HLE · held-out dataset'}},ylabel='Accuracy' if metric=='accuracy' else 'Towards-bias switch rate',legend_columns=len(ms),sample_labels='n_scored')
        spec['theme'].update(figure_width_min=12 if family=='qwen' else 8,figure_width_per_bias=2.2,figure_height_per_row=2.6,figure_height_intercept=.5)
        spec['significance_note']=f'Final checkpoints; 100 matched targets per dataset. 95% Wilson intervals; n = eligible responses.\nStars vs same-family base: 100,000 two-sided whole-question label swaps; Holm correction across {len(tests)} tests in this figure. * p<.05; ** p<.01; *** p<.001.\n'+('Accuracy excludes request errors; unparsed responses count as incorrect.' if metric=='accuracy' else 'Switch denominator requires parsed clean and biased answers, with clean answer not already matching the bias.')+(' One Gemma RMCT suggested-answer request remains unresolved; provisional.' if family=='gemma' else '')
        stem=family+'-'+metric
        unresolved=sum(bool(r['error']) for r in rows if r['model'].startswith(family+'-'))
        spec['significance_note']=spec['significance_note'].replace(' One Gemma RMCT suggested-answer request remains unresolved; provisional.','')
        if unresolved:spec['significance_note']+=f' Provisional: {unresolved} unresolved request errors.'
        (out/(stem+'-rows.json')).write_text(json.dumps(chart,indent=2)+'\n');(out/(stem+'-spec.json')).write_text(json.dumps(spec,indent=2)+'\n')
        for ext in ['png','svg','pdf']:render_publication_plot(chart,spec,out/(stem+'.'+ext))
        print(stem,flush=True)
(out/'provenance.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'collected/rows.json',ROOT/'collected/recovered-rows.json',a.template] if p.exists()},'renderer':'ctm_data.adapters.mcq_bias.plot.render_publication_plot','tests':'checkpoint_publication._paired_label_swap','permutations':100000,'unresolved_samples':sum(bool(r['error']) for r in rows)},indent=2)+'\n')
