"""Offline exact-request reconstruction from private Inspect logs and saved scores."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

METHODS = ('base', 'bct', 'rmct352')
CONFIGS = ('zero_shot', 'distribution_only', 'examples_only', 'both')
MARKER = '<EXACT ORIGINAL EVALUATION USER MESSAGE:'
MODEL = 'openai/gpt-5.6-luna'
EFFORT = 'xhigh'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def request_id(messages):
    # Deliberately retain original json.dumps defaults: serialization is identity.
    return hashlib.sha256(json.dumps([MODEL, EFFORT, messages], sort_keys=True).encode()).hexdigest()

def reasoning(text):
    if text.count('</think>') != 1:
        return None
    return text.split('</think>', 1)[0].removeprefix('<think>').strip() or None

def messages_for(cot, system, template=None):
    target = json.dumps({'reasoning': cot}, ensure_ascii=False)
    if template is not None:
        content = template[1]['content']
        if content.count(MARKER) != 1:
            raise ValueError('Template must contain exactly one original-message marker')
        target = content.split(MARKER, 1)[0] + target
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': target}]

class Inputs:
    def __init__(self, path_map=None, reader=None):
        self.path_map = path_map or {}
        self.reader = reader
        self.files = {}
        self.logs = {}

    def track(self, path, role):
        p = Path(path).resolve()
        self.files[str(p)] = {'path': str(p), 'role': role, 'sha256': digest(p)}
        return p

    def json(self, path, role):
        return json.loads(self.track(path, role).read_text())

    def sample(self, row):
        original = row['biased_source']
        if original not in self.logs:
            path = self.track(self.path_map.get(original, original), 'inspect-log')
            if self.reader is None:
                from inspect_ai.log import read_eval_log
                self.reader = read_eval_log
            index = {}
            for sample in self.reader(str(path)).samples:
                info = (sample.metadata or {}).get('cap64k_source') or (sample.metadata or {}).get('rmct_topup')
                key = (info['dataset'], info['bias'], info['qid']) if info else str(sample.id)
                if key in index:
                    raise ValueError(f'Duplicate sample key in {original}: {key}')
                index[key] = sample
            self.logs[original] = index
        index = self.logs[original]
        key = (row['dataset'], row['bias'], row['qid'])
        return index[key] if key in index else index[str(row['qid'])]

    def scores(self, paths):
        cache = {}
        for path in paths:
            for line in self.track(path, 'score-ledger').read_text().splitlines():
                r = json.loads(line)
                if not r.get('ok'):
                    continue
                score = r['score']
                if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 100:
                    raise ValueError('Successful ledger record has invalid score')
                cache[r['request_id']] = r  # Last successful record, historical order.
        return cache

def build_filter(inputs, samples, system, cache):
    rows, missing, stats = [], [], {}
    for method in METHODS:
        counter = Counter()
        for r in samples[method]:
            counter['total'] += 1
            if r['b'] is None or r['u'] is None:
                counter['invalid_pair'] += 1
                continue
            counter['valid_pair'] += 1
            already = r['u'] == r['option']
            suffix = 'already_matched' if already else 'filtered'
            counter['valid_' + suffix] += 1
            text = inputs.sample(r).output.completion
            if text.count('</think>') != 1:
                counter['boundary_excluded'] += 1
                continue
            cot = reasoning(text)
            if cot is None:
                counter['empty_cot'] += 1
                continue
            messages = messages_for(cot, system)
            rid = request_id(messages)
            label = dict(method=method, dataset=r['dataset'], bias=r['bias'], qid=r['qid'],
                         already_matched=already, biased_matches=r['b'] == r['option'],
                         target=int(not already and r['b'] == r['option']),
                         clean_verified=r.get('clean_verified', False), request_id=rid, source=r['biased_source'])
            if rid in cache:
                rows.append(dict(label, score=cache[rid]['score'] / 100))
                counter['scored_' + suffix] += 1
            else:
                counter['missing_' + suffix] += 1
                if already:
                    missing.append(dict(request_id=rid, effort=EFFORT, messages=messages, private_label=label))
        stats[method] = dict(counter)
    return rows, missing, stats

def build_rare(inputs, population, rare, templates, cache, old):
    rarekeys = {(r['method'], r['case_id']) for r in rare}
    rows, counts = [], {}
    for r in population:
        cot = reasoning(inputs.sample(r).output.completion)
        if cot is None:
            raise ValueError('Frozen eligible population contains unusable reasoning')
        m = r['method']
        ids = {c: request_id(messages_for(cot, templates[m]['both'][0]['content'],
                                        None if c == 'zero_shot' else templates[m][c])) for c in CONFIGS}
        baseline = all(v in old for v in ids.values())
        israre = (m, r['case_id']) in rarekeys
        if not baseline and not israre:
            continue
        if not all(v in cache for v in ids.values()):
            raise ValueError(f'Missing required score: {m}/{r["case_id"]}')
        rows.append(dict(r, baseline=baseline, rare=israre,
                         scores={c: cache[v]['score'] / 100 for c, v in ids.items()}, request_ids=ids))
    for m in METHODS:
        rr = [r for r in rows if r['method'] == m]
        pop = [r for r in population if r['method'] == m]
        counts[m] = dict(selected=len(rr), baseline=sum(r['baseline'] for r in rr), quadrants={
            f'{a},{y}': dict(population=sum(r['ack'] == a and r['target'] == y for r in pop),
                            selected=sum(r['ack'] == a and r['target'] == y for r in rr),
                            baseline=sum(r['ack'] == a and r['target'] == y and r['baseline'] for r in rr))
            for a in (0, 1) for y in (0, 1)})
    return rows, counts

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('filter', 'rare'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=Path)
    p.add_argument('--system-template', type=Path)
    p.add_argument('--population', type=Path)
    p.add_argument('--rare-labels', type=Path)
    p.add_argument('--templates', type=Path)
    p.add_argument('--ledger', type=Path, action='append', default=[])
    p.add_argument('--historical-ledger', type=Path, action='append', default=[])
    p.add_argument('--path-map', type=Path, help='JSON map: original exact log path -> relocated log path')
    args = p.parse_args()
    required = ('samples', 'system_template') if args.mode == 'filter' else ('population', 'rare_labels', 'templates')
    for name in required:
        if getattr(args, name) is None:
            p.error('--' + name.replace('_', '-') + ' required for ' + args.mode)
    if not args.ledger:
        p.error('At least one --ledger is required')
    if args.mode == 'rare' and not args.historical_ledger:
        p.error('Rare mode requires historical ledgers to define cached baseline')
    inputs = Inputs()
    if args.path_map:
        inputs.path_map = inputs.json(args.path_map, 'log-path-map')
    cache = inputs.scores(args.historical_ledger + args.ledger)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'filter':
        samples = inputs.json(args.samples, 'parser-corrected-samples')
        system = inputs.json(args.system_template, 'system-template')[0]['content']
        rows, missing, stats = build_filter(inputs, samples, system, cache)
        products = {'data.json': rows, 'missing-already-matched-requests.json': missing,
                    'audit.json': {'coverage': stats, 'system_prompt': system,
                                  'missing_already_matched_requests': len(missing)}}
    else:
        population = inputs.json(args.population, 'eligible-population')
        rare = inputs.json(args.rare_labels, 'rare-private-labels')
        templates = {m: {c: inputs.json(args.templates / f'{m}-{c}.json', 'calibration-template')
                         for c in CONFIGS[1:]} for m in METHODS}
        old = set(inputs.scores(args.historical_ledger))
        rows, counts = build_rare(inputs, population, rare, templates, cache, old)
        products = {'data.json': rows, 'counts.json': counts}
    for name, value in products.items():
        (args.output / name).write_text(json.dumps(value, indent=2))
    manifest = {'schema_version': 1, 'builder_sha256': digest(__file__), 'mode': args.mode,
                'model': MODEL, 'reasoning_effort': EFFORT, 'template_marker': MARKER,
                'identity': 'sha256(json.dumps([model, effort, messages], sort_keys=True).encode())',
                'arguments': {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v for k, v in vars(args).items()},
                'inputs': list(inputs.files.values()),
                'outputs': {name: digest(args.output / name) for name in products},
                'warnings': ['Private output: do not publish missing-request messages or trajectory content.',
                             'Historical RMCT training remains potentially flawed; corrected labels do not repair training.',
                             'Base clean labels unverified; calibration labels historical; observed switch is not causal influence.',
                             'Missing/invalid scores excluded, never negative; no provider calls made.']}
    try:
        manifest['inspect_ai_version'] = importlib.metadata.version('inspect-ai')
    except importlib.metadata.PackageNotFoundError:
        manifest['inspect_ai_version'] = None
    (args.output / 'builder-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'mode': args.mode, 'rows': len(rows), 'output': str(args.output)}))

if __name__ == '__main__':
    main()
