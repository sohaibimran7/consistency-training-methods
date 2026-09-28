"""Replot saved image-suite responses by dataset and bias generalisation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from experiments.rmct_two_bias_eval.checkpoint_publication import _paired_label_swap, _significance_marker

METHODS = ['base', 'act', 'attct', 'mlpct', 'bct', 'opct', 'rmct']
NAMES = dict(zip(METHODS, ['Base', 'ACT', 'AttCT', 'MLPCT', 'BCT', 'OPCT', 'RMCT']))
GROUPS = {'held_in': ['logiqa', 'hellaswag'], 'held_out': ['hle-text-mc']}
BIASES = {'held_in': ['suggested_answer'], 'held_out': ['sampled_shape', 'spurious_few_shot_squares']}


def main():
    from recovery import load_rows
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    args=p.parse_args()
    ROOT,OUT=args.root,args.output
    OUT.mkdir(parents=True, exist_ok=False)
    inputs = [ROOT / 'collected/rows.json', ROOT / 'collected/recovered-rows.json']
    rows = load_rows(ROOT)
    assert len({(r['model'], r['id']) for r in rows}) == len(rows) == 10800
    lookup = {(r['model'], r['qid'], r['condition']): r for r in rows}
    template = json.loads(args.template.read_text())

    def counts(r, metric):
        if r['error']:
            return (0, 0)
        if metric == 'accuracy':
            return (r['scores']['accuracy'], 1)
        clean = lookup[r['model'], r['qid'], 'clean']
        eligible = (not clean['error'] and clean['scores'].get('parsed')
                    and r['scores'].get('parsed') and clean['answer'] != r['biased_option'])
        return (int(r['answer'] == r['biased_option']), 1) if eligible else (0, 0)

    for family in ['qwen', 'gemma']:
        methods = METHODS if family == 'qwen' else ['base', 'rmct']
        for metric in ['accuracy', 'towards_bias_switch']:
            for layout in ['main', 'per-bias']:
                columns = BIASES if layout == 'main' else {b: [b] for bs in BIASES.values() for b in bs}
                chart, tests = [], []
                for population, datasets in GROUPS.items():
                    for column, biases in columns.items():
                        clusters = {}
                        for method in methods:
                            grouped = {}
                            for r in rows:
                                if r['model'] != family + '-' + method or r['dataset'] not in datasets or r['condition'] not in biases:
                                    continue
                                key = r['dataset'] + ':' + r['qid']
                                s, n = counts(r, metric)
                                old = grouped.get(key, (0, 0))
                                grouped[key] = (old[0] + s, old[1] + n)
                            clusters[method] = grouped
                        for method in methods:
                            arr = np.array(list(clusters[method].values()), dtype=float)
                            s, n = arr.sum(axis=0)
                            # Resample whole QIDs so the two held-out cues stay together.
                            rng = np.random.default_rng(20260925)
                            sampled = arr[rng.integers(0, len(arr), (10000, len(arr)))].sum(axis=1)
                            rates = sampled[sampled[:, 1] > 0, 0] / sampled[sampled[:, 1] > 0, 1]
                            lo, hi = np.quantile(rates, [.025, .975])
                            row = dict(condition=method, condition_label=NAMES[method], method=method,
                                       model=family, population=population, bias_type=column,
                                       bias_status='seen' if biases == BIASES['held_in'] else 'held_out',
                                       metric=metric, mean=s/n, stderr=float(rates.std()),
                                       ci_lower=float(lo), ci_upper=float(hi), n_scored=int(n),
                                       successes=int(s), question_clusters=len(arr), significance='')
                            if method != 'base':
                                row.update(_paired_label_swap(clusters[method], clusters['base'],
                                    treatment_name=family+'-'+method, metric=metric, population=population,
                                    bias_type=column, permutations=100000))
                                tests.append(row)
                            chart.append(row)
                running = 0
                for i, row in enumerate(sorted(tests, key=lambda r: r['p_value_raw'])):
                    running = min(1, max(running, (len(tests)-i)*row['p_value_raw']))
                    row.update(p_value=running, p_value_holm=running,
                               significance=_significance_marker(running), holm_family_size=len(tests))
                spec = json.loads(json.dumps(template))
                spec.update(metric=metric, condition_order=methods, condition_labels=NAMES,
                    model_order=[family], model_labels={family: 'Qwen3.5-9B' if family == 'qwen' else 'Gemma-4-12B'},
                    bias_order=list(columns), bias_labels={'held_in': 'Held-in bias\nSuggested answer',
                        'held_out': 'Held-out biases\nShape + black-square few-shot',
                        'suggested_answer': 'Suggested answer\nHeld-in bias',
                        'sampled_shape': 'Sampled tick / circle\nHeld-out bias',
                        'spurious_few_shot_squares': 'Black-square few-shot\nHeld-out bias'},
                    facet={'rows': ['population']}, facet_labels={'population': {
                        'held_in': 'Held-in datasets · LogiQA + HellaSwag', 'held_out': 'Held-out dataset · HLE'}},
                    ylabel='Biased accuracy' if metric == 'accuracy' else 'Towards-bias switch rate',
                    legend_columns=len(methods), sample_labels='n_scored')
                spec['theme'].update(figure_width_min=12 if family == 'qwen' else 9,
                    figure_width_per_bias=3.5, figure_height_per_row=2.8, figure_height_intercept=.5)
                spec['significance_note'] = (
                    'Final checkpoints; pooled response rates. 95% whole-QID bootstrap intervals (10,000 resamples); n = eligible responses.\n'
                    f'Stars vs base: 100,000 paired whole-QID swaps; Holm correction over {len(tests)} comparisons within this figure. * p<.05; ** p<.01; *** p<.001.\n'
                    + ('Request errors excluded; unparsed responses count as incorrect.' if metric == 'accuracy' else
                       'Requires parsed clean and biased answers; clean answer must differ from the bias target.')
                    + (' Gemma provisional: one unresolved RMCT suggested-answer response.' if family == 'gemma' else '')
                    + '\nHeld-in refers to text-training cue/dataset identity; image presentation is new. Clean controls are not pooled with bias conditions.')
                spec['significance_note']=spec['significance_note'].replace(' Gemma provisional: one unresolved RMCT suggested-answer response.','')
                unresolved=sum(bool(r['error']) for r in rows if r['model'].startswith(family+'-'))
                if unresolved:spec['significance_note']+=f' Provisional: {unresolved} unresolved request errors.'
                stem = f'{family}-{metric}-{layout}'
                (OUT / (stem+'-rows.json')).write_text(json.dumps(chart, indent=2)+'\n')
                (OUT / (stem+'-spec.json')).write_text(json.dumps(spec, indent=2)+'\n')
                for ext in ['png', 'pdf', 'svg']:
                    render_publication_plot(chart, spec, OUT / (stem+'.'+ext))
                print(stem, flush=True)
    (OUT / 'provenance.json').write_text(json.dumps({
        'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*inputs,args.template] if p.exists()},
        'unresolved': [{'model': r['model'], 'id': r['id']} for r in rows if r['error']],
        'dataset_groups': GROUPS, 'bias_groups': BIASES,
        'aggregation': 'Pooled eligible-response rate; resampling clusters all cues for each dataset/QID together',
        'renderer': 'ctm_data.adapters.mcq_bias.plot.render_publication_plot',
    }, indent=2)+'\n')


if __name__ == '__main__':
    main()
