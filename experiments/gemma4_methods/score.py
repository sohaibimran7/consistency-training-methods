"""Merge completed eval shards, exclude truncation, score switches and grade."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path


def valid_completion(sample):
    if sample.error or not sample.output or not sample.output.choices:
        raise ValueError('Missing/errored sample is not a completed response')
    reason = sample.output.choices[0].stop_reason
    if reason not in {'stop', 'length', 'max_tokens'}:
        raise ValueError(f'Unknown termination: {reason}')
    return reason == 'stop'


def select_pairs(clean, biased):
    """Select jointly completed IDs, never treating missingness as no-switch."""
    return [qid for qid in biased if qid in clean]


def merge(root, output):
    from inspect_ai.log import read_eval_log, write_eval_log
    from experiments.gemma4_methods.evaluate import load, identity as short_identity
    from experiments.gemma4_methods import score_helpers as helpers
    record = load(root)
    output.mkdir(parents=True, exist_ok=False)
    samples, prototypes, sources = defaultdict(dict), {}, []
    for rank in range(record['workers']):
        receipt = json.loads((root / f'rank-{rank}/generation-complete.json').read_text())
        seen, truncated = set(), 0
        for item in receipt['logs']:
            assert short_identity(item['path']) == item
            log = read_eval_log(item['path'])
            assert log.status == 'success' and not log.error
            tag = log.eval.metadata['gemma_methods_eval']
            assert tag['rank'] == rank and tag['method'] == record['method']
            index = tag['task_index']
            prototypes.setdefault(index, log)
            for sample in log.samples or []:
                qid = str(sample.id)
                assert (index, qid) not in seen and qid not in samples[index]
                seen.add((index, qid))
                valid = valid_completion(sample)
                truncated += not valid
                samples[index][qid] = sample
            sources.append(item)
        expected = {(c['task_index'], q) for c in record['cells'] for q in c['question_ids'][rank::record['workers']]}
        assert seen == expected and len(seen) == receipt['generated'] and truncated == receipt['truncated']
    records, coverage, switch_logs, clean = [], [], {}, {}
    for cell in record['cells']:
        index = cell['task_index']
        assert set(samples[index]) == set(cell['question_ids'])
        completed = {q: samples[index][q] for q in cell['question_ids'] if valid_completion(samples[index][q])}
        def make_log(ids):
            log = prototypes[index].model_copy(deep=True)
            log.samples = [completed[q] for q in ids]
            log.status, log.error, log.results = 'success', None, None
            log.eval.dataset.samples = len(ids)
            log.eval.dataset.sample_ids = list(ids)
            log.eval.config.limit = len(ids)
            log.eval.metadata.update(condition=record['method'], question_ids_from=list(ids),
                 gemma_methods_merge_manifest=short_identity(root / 'manifest.json'))
            log.eval.task_args['question_ids_from'] = list(ids)
            return log
        path = output / f'task-{index:03d}.eval'
        log = make_log(list(completed))
        write_eval_log(log, str(path))
        cov = {'task_index': index, 'dataset': cell['dataset'], 'bias_type': cell['bias_type'],
               'raw_generated': len(samples[index]), 'completed': len(completed),
               'truncated': len(samples[index])-len(completed)}
        if cell['kind'] == 'unbiased':
            clean[cell['dataset']] = (completed, path)
        else:
            clean_samples, clean_path = clean[cell['dataset']]
            ids = select_pairs(clean_samples, completed)
            cov['jointly_completed_clean_biased'] = len(ids)
            selected = make_log(ids)
            if ids:
                selected = helpers.score_switch(selected, clean_path, ids)
            write_eval_log(selected, str(output / f'switch-{index:03d}.eval'))
            switch_logs[index] = selected
        records.append({**{k: cell[k] for k in ['task_index','kind','dataset','regime','population','bias_type']},
                        'sample_count': len(completed), 'log': helpers.identity(path)})
        coverage.append(cov)
    helpers.save(output / 'coverage.json', coverage)
    helpers.save(output / 'merge-complete.json', {'cells': records,
        'evaluation_manifest': short_identity(root / 'manifest.json'), 'raw_sources': sources,
        'implementations': [short_identity(__file__), short_identity(helpers.__file__)],
        'policy': 'Luna uses all completed biased outputs; switch uses jointly completed clean/biased IDs'})
    helpers.report(output, switch_logs, 'switch-summary.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['merge', 'grade'])
    p.add_argument('--root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'merge':
        merge(a.root, a.output)
    else:
        from experiments.gemma4_methods.score_helpers import grade
        grade(a.output)
