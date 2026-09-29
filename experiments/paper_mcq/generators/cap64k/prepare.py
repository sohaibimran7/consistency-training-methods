"""Export only verified, previously unparsed 20k-cap responses, without inference."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'unparsed-diagnosis-20260917'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name, value):
    text = json.dumps(value, indent=2) + '\n'
    path = ROOT / name
    if path.exists():
        assert path.read_text() == text, path
    else:
        path.write_text(text)

rows = [r for r in json.loads((SOURCE/'diagnosis.json').read_text()) if r['reason']=='token_limit']
hashes = json.loads((SOURCE/'sources.json').read_text())
groups = defaultdict(list)
for row in rows:
    groups[row['source']].append(row)
methods = defaultdict(list)
settings = {}
for source, selected in sorted(groups.items()):
    assert sha(source) == hashes[source]
    log = read_eval_log(source)
    indexed = {str(s.id): s for s in log.samples}
    for row in selected:
        s = indexed[row['qid']]
        assert s.scores['mcq_bias_scorer'].value['answer_parsed'] == 0
        assert s.output.usage.output_tokens == 20480
        assert s.output.choices[0].stop_reason == 'max_tokens'
        assert log.eval.model_generate_config.max_tokens == 20480
        config = log.eval.model_generate_config.model_dump(exclude_none=True)
        config['max_tokens'] = 65536
        args = dict(log.eval.model_args)
        args['max_model_len'] = 98304
        setting = dict(model=log.eval.model, model_args=args, generation_config=config)
        method = row['method']
        if method in settings:
            assert settings[method] == setting
        settings[method] = setting
        identity = f"{row['dataset']}:{row['bias'] or 'clean'}:{row['qid']}"
        methods[method].append(dict(id=identity, input=s.model_dump(mode='json')['input'],
            target=s.target, metadata={**s.metadata, 'cap64k_source': row,
                                      'source_sha256': hashes[source]}))
allocations = dict(base=1, act=2, attct=3, mlpct=3, bct=3, opct=3, rmct352=1)
shards=[]
for method, count in allocations.items():
    samples=sorted(methods[method], key=lambda s:s['id'])
    assert len({s['id'] for s in samples}) == len(samples)
    for i in range(count):
        shard=dict(method=method, **settings[method], samples=samples[i::count])
        write(f'shard-{len(shards):02d}.json', shard)
        shards.append(dict(index=len(shards),method=method,n=len(shard['samples'])))
assert sum(s['n'] for s in shards)==564 and len(shards)==16
write('selection.json',dict(n=564,biased=387,clean=177,shards=shards,
    source_diagnosis_sha256=sha(SOURCE/'diagnosis.json'),
    max_tokens=65536,max_model_len=98304,
    note='Fresh generations from exact original prompts, not continuation of truncated text; original outputs retained.'))
print(json.dumps(shards,indent=2))
