"""Overlay the 64k reruns and RMCT IID top-up onto the parser-fixed samples.

Every new output is reparsed with the corrected parser. Existing Luna
acknowledgement receipts are reused; newly parseable biased outputs without a
receipt stay ungraded (ack=None). No model or grader calls.
"""
import copy, hashlib, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspect_ai.log import read_eval_log
from ctm_data.adapters.mcq_bias.parser_compat import install_extended_answer_parser
install_extended_answer_parser()
from mcq_bias.parsers import parse_answer

R = Path(__file__).resolve().parents[1]
FIXED = R / 'artifacts/parser-fixed-20260923'
REC = Path('/Users/work/.codex/worktrees/d6d6/consistency-training-methods/artifacts/recovered-publication-20260922')
OUT = R / 'artifacts/parser-fixed-64k-20260925'
OUT.mkdir(exist_ok=True)

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n, v): (OUT / n).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')

data = read(FIXED / 'samples.json')
maps = {m: {(r['dataset'], r['bias'], r['qid']): r for r in rows} for m, rows in data.items()}
originals = {m: set(rows) for m, rows in maps.items()}

# Same selection as recovered-publication-20260922/build.py merge().
expected_sources = read(REC / 'source-hashes.json')
updates = {}; sources = {}
for p in sorted((REC / 'inputs').glob('*/raw/shard-*/*.eval')):
    h = sha(p); assert expected_sources[str(p)] == h, p; sources[str(p)] = h
    log = read_eval_log(str(p))
    for s in log.samples or []:
        if s.error or not s.output or not s.output.choices or s.output.error: continue
        md = s.metadata
        if 'rmct_topup' in md: info = md['rmct_topup']; method = 'rmct352'; kind = 'topup'
        else: info = md['cap64k_source']; method = info['method']; kind = 'recovery'
        key = (method, info['dataset'], info['bias'], info['qid'])
        assert key not in updates, key
        updates[key] = (s, str(p), kind)
assert Counter(v[2] for v in updates.values()) == {'topup': 700, 'recovery': 552}

old_audit = {(a['method'], a['dataset'], a['bias'], a['qid']): a for a in read(REC / 'merge-audit.json')}
clean = {}; audit = []
for key, (s, path, kind) in updates.items():
    method, ds, bias, qid = key
    answer = parse_answer(s.output.completion)
    ack = None
    if bias is not None and answer is not None:
        identity = hashlib.sha256(json.dumps(dict(key=list(key), output=s.output.model_dump(mode='json')), sort_keys=True).encode()).hexdigest()
        receipt = REC / 'grades' / f'{identity}.json'
        if receipt.exists():
            grade = read(receipt); assert tuple(grade['key']) == key
            ack = grade['score']['value']['bias_acknowledged']; assert ack in (0, 1)
    audit.append(dict(method=method, dataset=ds, bias=bias, qid=qid, kind=kind, source=path,
                      parsed=answer is not None, graded=ack is not None,
                      parsed_old_parser=old_audit[key]['parsed'],
                      tokens=s.output.usage.output_tokens, stop=s.output.choices[0].stop_reason))
    if bias is None: clean[method, ds, qid] = (answer, path); continue
    k = (ds, bias, qid)
    if kind == 'topup':
        assert k not in maps[method]
        b = maps['base'][k]
        r = {f: b[f] for f in ('dataset', 'bias', 'qid', 'gold', 'option', 'prompt_hash')}
        assert s.target == r['gold'] and s.metadata['biased_option'] == r['option']
        r.update(b=answer, u=None, ack=ack, biased_source=path, clean_verified=False, source_kind='rmct_topup_20k')
    else:
        r = maps[method][k]; assert s.target == r['gold']
        r.update(pre_64k_b=r['b'], pre_64k_ack=r['ack'], b=answer, ack=ack, biased_source=path, source_kind='cap64k_rerun')
    maps[method][k] = r

for (method, ds, qid), (answer, path) in clean.items():
    hits = [r for (d, _, q), r in maps[method].items() if d == ds and q == qid]
    assert hits, (method, ds, qid)
    for r in hits:
        if r.get('source_kind') != 'rmct_topup_20k': r['pre_64k_u'] = r['u']
        r.update(u=answer, clean_verified=True, clean_source=path)

for m, rows in maps.items():
    assert len(rows) == 1800, (m, len(rows))
    if m == 'rmct352':
        assert all(r['u'] is not None or r['clean_verified'] for k, r in rows.items() if k not in originals[m])

missing = read(REC / 'missing-bct.json'); assert len(missing) == 12
for x in missing:
    rows = maps[x['method']]
    if x['bias'] is None:
        assert all(r['u'] is None for (d, _, q), r in rows.items() if d == x['dataset'] and q == x['qid']), x
    else:
        assert rows[x['dataset'], x['bias'], x['qid']]['b'] is None, x
print('missing BCT:', Counter('clean' if x['bias'] is None else 'biased' for x in missing))

summary = {}
for m, rows in maps.items():
    rs = list(rows.values())
    summary[m] = dict(total=len(rs), biased_parsed=sum(r['b'] is not None for r in rs),
                      parsed_graded=sum(r['b'] is not None and r['ack'] in (0, 1) for r in rs),
                      parsed_ungraded=sum(r['b'] is not None and r['ack'] is None for r in rs),
                      joint_parsed=sum(r['b'] is not None and r['u'] is not None for r in rs))
    print(m, summary[m], flush=True)
new_parse = [a for a in audit if a['parsed'] and not a['parsed_old_parser']]
lost_parse = [a for a in audit if not a['parsed'] and a['parsed_old_parser']]
print('newly parsed by fixed parser:', len(new_parse), 'lost:', len(lost_parse))
write('samples.json', {m: list(rows.values()) for m, rows in maps.items()})
write('merge-audit.json', audit); write('recovery-sources.json', sources)
write('missing-bct.json', missing); write('summary.json', summary)
write('provenance.json', dict(parser_fixed_samples_sha256=sha(FIXED / 'samples.json'),
                              recovered_build_sha256=sha(REC / 'build.py'),
                              parser_sha256=sha(R / 'ctm_data/adapters/mcq_bias/parser_compat.py'),
                              model_calls=0, grader_calls=0))
