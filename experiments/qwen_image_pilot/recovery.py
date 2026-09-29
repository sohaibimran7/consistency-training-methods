"""Validated successful-only retry overlays, independent of Inspect."""
import json
from pathlib import Path

def overlay(original, attempts):
    key = lambda r: (r['model'], r['id'])
    indexed = {key(r): r for r in original}
    if len(indexed) != len(original):
        raise ValueError('Duplicate original model/id')
    expected = {k for k,r in indexed.items() if r.get('error')}
    recovered, failures = {}, {}
    for row in attempts:
        k = key(row)
        if k not in expected:
            raise ValueError(f'Retry is not an original request error: {k}')
        for field in ('qid','dataset','condition','ground_truth','biased_option'):
            if row.get(field) != indexed[k].get(field):
                raise ValueError(f'Retry identity mismatch: {k}/{field}')
        if row.get('error'):
            failures.setdefault(k, []).append({'log':row.get('log'),'error':row['error']})
            continue
        if not row.get('scores') or row.get('stop_reason') is None:
            raise ValueError(f'Retry lacks output/score: {k}')
        if k in recovered:
            raise ValueError(f'Multiple successful retries: {k}; select explicitly')
        recovered[k] = row
    unresolved = [dict(model=k[0],id=k[1],reason='retry_failed' if k in failures else 'not_recovered',
                       original_log=indexed[k].get('log'),attempts=failures.get(k,[]))
                  for k in sorted(expected-recovered.keys())]
    return [recovered.get(key(r),r) for r in original], [recovered[k] for k in sorted(recovered)], unresolved

def load_rows(root):
    root = Path(root)
    original = json.loads((root/'collected/rows.json').read_text())
    retry = root/'collected/recovered-rows.json'
    return overlay(original,json.loads(retry.read_text()) if retry.exists() else [])[0]
