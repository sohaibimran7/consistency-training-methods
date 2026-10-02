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


def build(baseline, *, repo, python, python_prefix, commit, run_name, data, manifest, attestation):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Exact incorporated commit required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_name):
        raise ValueError('Distinct safe run name required')
    for p in (repo, python, python_prefix, data, manifest, attestation):
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
                incorporated_commit=commit, python_prefix=python_prefix, argv=argv, initial_model_revision=REVISION,
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


POOL_MANIFEST_SHA = 'e50396d8fb2188f5959f9ced378a8f813922b6015f1a1a73887cfbccc52b43d1'
ONE_BIAS_FACTORY = 'ctm_data.adapters.mcq_bias.shared_qid_one_bias:SharedQidOneBiasSetting'


def build_one_bias(baseline, *, one_bias_manifest, one_bias_manifest_sha256, **kwargs):
    """Fresh one-bias Qwen RMCT recipe (user-approved 2026-10-02).

    Same RMCT recipe, but the shared 7,680-QID pool (``data``/``manifest`` must be
    that pool), four distinct QIDs x one assigned cue per batch, one finite pass.
    Each child loads an absolute sampled-batch slice of the shared manifest.
    """
    if not Path(one_bias_manifest).is_absolute() or not re.fullmatch(r'[0-9a-f]{64}', one_bias_manifest_sha256):
        raise ValueError('Absolute frozen one-bias manifest and its identity required')
    plan = build(baseline, **kwargs)
    argv = plan['argv']
    def replace(flag, value):
        if argv.count(flag) != 1:
            raise ValueError(f'Expected unique {flag}')
        argv[argv.index(flag)+1] = value
    replace('--experiment-name', 'rmct-one-bias-20261002')
    replace('--setting-factory', ONE_BIAS_FACTORY)
    replace('--setting-config', json.dumps(dict(
        data_path=kwargs['data'], manifest_path=kwargs['manifest'], expected_manifest_sha256=POOL_MANIFEST_SHA,
        expected_qids_per_dataset=3840, expected_qids_per_dataset_per_segment=16,
        one_bias_manifest_path=one_bias_manifest, one_bias_manifest_sha256=one_bias_manifest_sha256)))
    replace('--load-config', json.dumps(dict(n_datapoints=16, attempt_offset=0)))
    replace('--n-datapoints', '16')
    replace('--batch-size', '4')
    plan.update(schema='rmct-one-bias-preparation-v1', one_bias_manifest_sha256=one_bias_manifest_sha256,
                first_validation_step=None, first_validation_encounters=256,
                exposure=dict(qids_per_attempted_batch=4, biases_per_qid=1, max_attempted_batches=1920,
                              no_update_batches_consume_encounters=True, pool_cycling=False))
    plan['validation'].update(interval=None, interval_encounters=256,
                              selection_contract='ctm-tbsr-selection-contract-v2-encounters')
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','repo','python','python-prefix','commit','run-name','data','manifest','attestation','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--one-bias-manifest'); p.add_argument('--one-bias-manifest-sha256')
    a=p.parse_args()
    raw=Path(a.baseline).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BASELINE_SHA:
        raise ValueError('Historical recipe identity mismatch')
    kwargs=dict(repo=a.repo, python=a.python, python_prefix=a.python_prefix, commit=a.commit,
                run_name=a.run_name, data=a.data, manifest=a.manifest, attestation=a.attestation)
    if a.one_bias_manifest:
        result=build_one_bias(json.loads(raw), one_bias_manifest=a.one_bias_manifest,
                              one_bias_manifest_sha256=a.one_bias_manifest_sha256, **kwargs)
    else:
        result=build(json.loads(raw), **kwargs)
    with Path(a.output).open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__': main()
