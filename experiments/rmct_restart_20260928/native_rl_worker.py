"""Disposable gate only: validate a full frozen segment, train its first batch.

The normal production CLI and setting remain unchanged. Never resume this run.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def select_first_batch(prepared):
    from ctm_data.adapters.mcq_bias.shared_qid_two_bias import SharedQidTwoBiasSetting
    from experiments.rmct_restart_20260928.gemma_production_setting import OneBiasGemmaSetting, QIDS_PER_UPDATE
    if isinstance(prepared.setting, OneBiasGemmaSetting):
        # One-bias campaign: validate a 16-QID slice, train its first 4-QID batch.
        if len(prepared.datapoints) != 4 * QIDS_PER_UPDATE:
            raise ValueError('Disposable gate requires a fully validated 16-QID one-bias slice')
        return replace(prepared, datapoints=prepared.datapoints[:QIDS_PER_UPDATE])
    if not isinstance(prepared.setting, SharedQidTwoBiasSetting) or len(prepared.datapoints) != 32:
        raise ValueError('Disposable gate requires a fully validated 32-QID shared segment')
    return replace(prepared, datapoints=prepared.datapoints[:2])


def main(argv=None):
    from scripts import train_rlct
    from experiments.rmct_restart_20260928.qwen_launch import write
    args = list(sys.argv[1:] if argv is None else argv)
    def value(flag):
        if args.count(flag) != 1:
            raise ValueError('Gate needs exactly one ' + flag)
        return args[args.index(flag)+1]
    if not os.environ.get('SLURM_JOB_ID') or any(x.startswith('--resume') for x in args):
        raise ValueError('Scheduled fresh disposable gate only')
    if value('--experiment-name') != 'disposable-native-rl-gate' or value('--run-name') != 'one-update':
        raise ValueError('Not a disposable experiment')
    load = json.loads(value('--load-config'))
    one_bias = 'attempt_offset' in load
    if value('--batch-size') != ('4' if one_bias else '2') or value('--n-epochs') != '1':
        raise ValueError('Exactly one four-QID (one-bias) or two-QID update required')
    if one_bias and load != {'n_datapoints': 16, 'attempt_offset': 0}:
        raise ValueError('Initial one-bias slice required')
    if not one_bias and (load.get('n_datapoints') != 32 or load.get('segment_index') != 0):
        raise ValueError('Full initial segment required')
    original = train_rlct.prepare_setting
    def bounded(*a, **kw):
        prepared = original(*a, **kw)
        subset = select_first_batch(prepared)
        write(Path.cwd()/'disposable-subset.json', {
            'loaded_datapoints': len(prepared.datapoints), 'selected_indices': list(range(len(subset.datapoints))),
            'load_config': load, 'production_resume_forbidden': True})
        return subset
    train_rlct.prepare_setting = bounded
    try:
        train_rlct.main(args)
    finally:
        train_rlct.prepare_setting = original


if __name__ == '__main__':
    main()
