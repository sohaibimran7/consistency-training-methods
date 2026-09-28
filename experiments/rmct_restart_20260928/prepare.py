"""Prepare a fresh-start command, never submit or train.

The integration owner must supply an incorporated commit and a NEW runtime
attestation. This is a configuration contribution, not restart clearance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

BASELINE_SHA = 'b9ed4034be9ccb4d01006bc7d5a1864c2ed17951ea1ed3f7c99bce81ad54c42d'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
MANIFEST_SHA = 'eac0682fe0286126cc5928253e2e2ef4968eee50775a65feb21d061eec6853cc'


def build(baseline, *, repo, python, commit, run_name, data, manifest, attestation):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Exact incorporated commit required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_name):
        raise ValueError('Distinct safe run name required')
    for p in (repo, python, data, manifest, attestation):
        if not Path(p).is_absolute():
            raise ValueError('Explicit absolute deployment paths required')
    argv = list(baseline['argv'])
    def replace(flag, value):
        if argv.count(flag) != 1:
            raise ValueError(f'Expected unique {flag}')
        argv[argv.index(flag)+1] = value
    i = argv.index('--resume-from')
    del argv[i:i+2]
    argv.remove('--resume-with-optimizer')
    argv.remove('--resume-state-required')
    argv[:2] = [python, str(Path(repo)/'scripts/train_rlct.py')]
    replace('--run-name', run_name)
    replace('--experiment-name', 'rmct-restart-20260928')
    replace('--load-config', json.dumps(dict(cycle_segments=True, n_datapoints=32, segment_index=0)))
    replace('--setting-config', json.dumps(dict(data_path=data, manifest_path=manifest, expected_manifest_sha256=MANIFEST_SHA)))
    replace('--local-qwen35-rollout-parity-attestation', attestation)
    replace('--max-new-tokens', '20480')
    assert argv[argv.index('--model')+1].endswith('/'+REVISION)
    assert not any(x.startswith('--resume') for x in argv)
    return dict(schema='rmct-clean-restart-preparation-v1', ready_to_launch=False,
                incorporated_commit=commit, argv=argv, initial_model_revision=REVISION,
                optimizer='fresh', initial_segment=0, initial_optimizer_step=0,
                first_segment_end=16, first_validation_step=64,
                validation=dict(metric='tbsr', interval=64, patience=2, min_delta=0,
                                tie='earliest', history='new-campaign-only', thinking=True,
                                generated_tokens={'logiqa':20480, 'hellaswag':20480, 'other':65536}),
                required_before_optimizer=['incorporated commit verified against canonical branch',
                    'clean deployed tracked source and full source/config hashes',
                    'actual scheduled cwd, Python executable and imported module paths',
                    'dependency lock, installed distributions and tokenizer/model hashes',
                    'deployed parser/completion/thinking regressions pass',
                    'fresh integrated runtime preflight evidence and adapter checks',
                    'data/manifest hashes and fresh output directory verified',
                    'durable receipt saved before optimizer work'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','repo','python','commit','run-name','data','manifest','attestation','output'):
        p.add_argument('--'+name, required=True)
    a=p.parse_args()
    raw=Path(a.baseline).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BASELINE_SHA:
        raise ValueError('Historical recipe identity mismatch')
    result=build(json.loads(raw), repo=a.repo, python=a.python, commit=a.commit,
                 run_name=a.run_name, data=a.data, manifest=a.manifest, attestation=a.attestation)
    with Path(a.output).open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__': main()
