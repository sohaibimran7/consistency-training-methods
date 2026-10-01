"""Compile a fresh Gemma RMCT first-segment argv; never execute or submit it.

The shared integrated controller must supply launch clearance and validation
scheduling. Qwen-specific native-worker receipts are not Gemma clearance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from experiments.gemma4_rmct.plan import segment_args, REVISION
from experiments.rmct_convergence.plan import DATA_SHA256, MANIFEST_SHA256
from experiments.rmct_restart_20260928.gemma_config import fresh_config


def build(*, repo, python, model, targets, data, manifest, commit, run_name,
          approval_reference, validation_sha256):
    for path in (repo, python, model, data, manifest):
        if not Path(path).is_absolute():
            raise ValueError('Absolute deployment paths required')
    if Path(model).name != REVISION:
        raise ValueError('Original pinned Gemma snapshot required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_name):
        raise ValueError('New safe run name required')
    policy = fresh_config(source_commit=commit, approval_reference=approval_reference,
                          validation_manifest_sha256=validation_sha256)
    # Reuse the actual Gemma recipe: it strips Qwen phase-sharing options and
    # supplies the historical one-trainer/three-worker native Gemma topology.
    args = segment_args(Path(repo), 0, model=model, target_modules=targets)
    args.pop('no_max_new_tokens', None)
    args.update(max_new_tokens=20480, experiment_name='gemma-rmct-restart-20260928',
        run_name=run_name, checkpoint_every=1,
        setting_factory='experiments.rmct_restart_20260928.gemma_production_setting:create_setting',
        setting_config={'data_path': data, 'manifest_path': manifest,
                        'expected_manifest_sha256': MANIFEST_SHA256},
        load_config={'n_datapoints': 32, 'segment_index': 0, 'batch_offset': 0})
    if any(k.startswith('resume') for k in args) or args.get('local_phase_shared'):
        raise RuntimeError('Fresh native Gemma topology violated')
    # Same scalar/dict CLI encoding as scripts.run_experiment, without its
    # unrelated YAML dependency during CPU-only plan preparation.
    argv = [python, str(Path(repo)/'scripts/train_rlct.py')]
    for key, value in args.items():
        if value is None or value is False:
            continue
        argv.append('--'+key.replace('_', '-'))
        if value is not True:
            if isinstance(value, list):
                raise ValueError('Unexpected list CLI option')
            argv.append(json.dumps(value, sort_keys=True, separators=(',', ':'))
                        if isinstance(value, dict) else str(value))
    return {'schema': 'gemma-rmct-fresh-command-v1', 'ready_to_launch': False,
            'incorporated_commit': commit, 'family': 'gemma', 'policy': policy,
            'argv': argv, 'argument_map': args,
            'data_sha256': DATA_SHA256, 'manifest_sha256': MANIFEST_SHA256,
            'initial_optimizer_step': 0, 'initial_batch_offset': 0,
            'environment': {'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
                            'CTM_EXCLUDE_LENGTH_TERMINATED': '1'},
            'required_gates': ['incorporated_source', 'cpu_provenance', 'gemma_native_gpu_parity',
                               'gemma_production_multiworker_rl_regression',
                               'validation_controller_integration', 'coordinator_clearance'],
            'controller_contract': {'actual_updates_per_validation': 64,
                'first_segment_max_attempted_batches': 16,
                'skips_do_not_increment_optimizer_steps': True,
                'resume_allowed_only_within_new_lineage_after_first_segment': True,
                'batch_offset_from_durable_global_step_not_optimizer_step': True,
                'validation_history': 'new', 'metric': 'tbsr', 'patience': 2,
                'min_delta': 0, 'improvement': 'strict_decrease'}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'python', 'model', 'data', 'manifest', 'targets',
                 'commit', 'run-name', 'approval-reference', 'validation-sha256', 'output'):
        p.add_argument('--'+name, required=True)
    a = vars(p.parse_args())
    output = Path(a.pop('output'))
    for key, expected in [('data', DATA_SHA256), ('manifest', MANIFEST_SHA256)]:
        if hashlib.sha256(Path(a[key]).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Original shared training pool identity mismatch: '+key)
    a['targets'] = json.loads(Path(a['targets']).read_text())
    result = build(**a)
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
