"""Fresh one-bias Qwen RMCT children over the shared 7,680-QID manifest.

Same strict-seal lineage as ``qwen_train_window`` (separate sampled-batch and
optimizer counters, exclusive start receipts, no replay), but each child loads
an absolute sampled-batch slice of the shared one-bias manifest: four distinct
QIDs x one assigned cue per batch, one finite pass. Boundaries are counted in
sampled batches (64 = 256 encountered QIDs), including no-update batches.
No submission, retries or historical resumes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.rmct_restart_20260928.qwen_checkpoint import seal_v2
from experiments.rmct_restart_20260928.qwen_progress import (
    ONE_BIAS_MAX_BATCHES, ONE_BIAS_VALIDATION_BATCHES, next_encounter_slice, progress)
from experiments.rmct_restart_20260928.qwen_validation_executor import write

QIDS_PER_BATCH = 4


def command(plan, before, selection, parent):
    argv = list(plan['argv'])
    if plan.get('schema') != 'rmct-one-bias-preparation-v1' or any(x.startswith('--resume') for x in argv):
        raise ValueError('Fresh one-bias recipe required')
    for flag, value in (('--batch-size', str(QIDS_PER_BATCH)), ('--n-epochs', '1'),
                        ('--gradient-accumulation-steps', '1'), ('--max-new-tokens', '20480')):
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != value:
            raise ValueError('Scientific recipe changed: ' + flag)
    if '--no-shuffle-datapoints' not in argv:
        raise ValueError('Frozen order required')
    if before != progress(before['sampled_batches'], before['optimizer_updates'],
                          no_progress_batches=before['no_progress_batches']):
        raise ValueError('Inconsistent cursor')
    if (selection['segment_index'], selection['batch_offset']) != (before['segment_index'], before['batch_offset']):
        raise ValueError('Slice replays or skips consumed data')
    count = selection['batch_count']
    if type(count) is not int or not 1 <= count <= 16 - before['batch_offset']:
        raise ValueError('Whole four-QID batches within one segment required')
    if parent is None and before != progress(0, 0):
        raise ValueError('Missing parent')
    if parent is not None and parent['progress'] != before:
        raise ValueError('Parent cursor differs')
    base = argv[argv.index('--run-name') + 1]
    argv[argv.index('--run-name') + 1] = f"{base}-b{before['sampled_batches']:06d}"
    # Absolute sampled-batch cursor into the shared manifest (no-update batches included).
    argv[argv.index('--load-config') + 1] = json.dumps(
        dict(n_datapoints=QIDS_PER_BATCH * count, attempt_offset=before['sampled_batches']), sort_keys=True)
    argv[argv.index('--n-datapoints') + 1] = str(QIDS_PER_BATCH * count)
    if parent is not None:
        argv += ['--resume-from', 'file://' + parent['checkpoint'], '--resume-with-optimizer', '--resume-state-required']
    return argv


def checkpoint_path(argv, repo):
    name = argv[argv.index('--run-name') + 1]
    experiment = argv[argv.index('--experiment-name') + 1]
    return Path(repo) / 'logs' / experiment / name / 'checkpoints' / f'{experiment}_{name}'


def slice_rows(argv):
    from ctm_data.adapters.mcq_bias.shared_qid_one_bias import SharedQidOneBiasSetting
    setting = SharedQidOneBiasSetting(**json.loads(argv[argv.index('--setting-config') + 1]))
    rows = setting.load_datapoints(**json.loads(argv[argv.index('--load-config') + 1]))
    return [r['question_id'] for r in rows], [r['bias'] for r in rows]


def boundary_for(before):
    return min((before['sampled_batches'] // ONE_BIAS_VALIDATION_BATCHES + 1) * ONE_BIAS_VALIDATION_BATCHES,
               ONE_BIAS_MAX_BATCHES)


def verify_lineage(receipt, plan, binding):
    """Reseal every node from bytes against the one-bias recipe; returns the verified head."""
    chain, node = [], receipt
    while node is not None:
        if node.get('binding') != binding or node.get('schema') != 'rmct-clean-checkpoint-v2':
            raise ValueError('Foreign binding or schema in one-bias lineage')
        chain.append(node)
        node = node.get('parent')
    parent = None
    for saved in reversed(chain):
        before = parent['progress'] if parent else progress(0, 0)
        argv = command(plan, before, saved['selection'], parent)
        current = seal_v2(checkpoint_path(argv, binding['source_root']), campaign_id=saved['campaign_id'],
                          before=before, selection=saved['selection'], command=argv, binding=binding, parent=parent)
        if current != saved:
            raise ValueError('Checkpoint seal, recipe or lineage changed')
        parent = current
    return parent


def train_to_encounters(plan, binding, root, target_batches, checked, parent, env):
    """Train children until ``target_batches`` sampled batches; never crosses a validation boundary."""
    repo = Path(binding['source_root'])
    campaign_id = plan['argv'][plan['argv'].index('--run-name') + 1]
    start = parent['progress'] if parent else progress(0, 0)
    if target_batches > boundary_for(start):
        raise ValueError('Target crosses an unvalidated 256-encounter boundary')
    while True:
        if parent is not None:
            verify_lineage(parent, plan, binding)
        before = parent['progress'] if parent else progress(0, 0)
        if before['sampled_batches'] >= target_batches:
            return parent
        selection = next_encounter_slice(before, boundary_for(before))
        selection['batch_count'] = min(selection['batch_count'], target_batches - before['sampled_batches'])
        argv = command(plan, before, selection, parent)
        qids, biases = slice_rows(argv)
        if len(qids) != QIDS_PER_BATCH * selection['batch_count']:
            raise ValueError('Runtime slice size differs')
        checkpoint = checkpoint_path(argv, repo)
        if checkpoint.parent.parent.exists():
            raise ValueError('Uncertain/existing production run; no replay')
        folder = Path(root) / 'batches' / str(before['sampled_batches'])
        write(folder / 'started.json', dict(argv=argv, binding=binding, gates=checked, before=before,
              selection=selection, target_batches=target_batches, consumed_question_ids=qids, biases=biases))
        subprocess.run(argv, cwd=repo, env=env, check=True)
        parent = seal_v2(checkpoint, campaign_id=campaign_id, before=before, selection=selection,
                         command=argv, binding=binding, parent=parent)
        write(folder / 'complete.json', parent)


def normalized_progress(seal, *, method='rmct', model, source_commit, campaign_id):
    """Shared ctm-training-progress-v1 record for the encounter selection controller."""
    state = seal['progress']
    return {'schema': 'ctm-training-progress-v1', 'campaign_id': campaign_id, 'method': method,
            'model': model, 'source_commit': source_commit,
            'actual_optimizer_step': state['optimizer_updates'], 'next_attempt_index': state['sampled_batches'],
            'sampled_batches': state['sampled_batches'], 'encounter_attempt': state['sampled_batches'],
            # No-update batches are inside the checkpoint's own global step.
            'encountered_qid_bias_examples': QIDS_PER_BATCH * state['sampled_batches'],
            'trailing_skip_files': [], 'checkpoint': seal['checkpoint'],
            'checkpoint_files': seal['files'], 'rmct_seal': seal}


def run(a):
    """One scheduled window: gates, encounter budget, then children up to the budget."""
    from experiments.rmct_restart_20260928 import validation_selection as selection
    from experiments.rmct_restart_20260928.qwen_checkpoint import identity
    from experiments.rmct_restart_20260928.qwen_one_bias_validation import Verifiers
    from experiments.rmct_restart_20260928.qwen_train_window import gates, plan_binding
    from experiments.rmct_restart_20260928.qwen_validation_executor import read
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Scheduled allocation required')
    plan = read(a.plan)
    if os.path.abspath(sys.executable) != os.path.abspath(plan['argv'][0]):
        raise ValueError('Wrong interpreter')
    checked = gates(plan, a.plan, a.gates / 'preflight/preflight-results.json',
                    a.gates / 'disposable-rl/receipt.json', a.gates / 'regressions/receipt.json',
                    Path(plan['argv'][1]).resolve().parents[1])
    binding = plan_binding(plan, a.plan)
    contract_record = identity(a.selection_contract)
    contract = selection.read_verified(contract_record)
    selection.check_contract(contract)
    if not selection.encounter_mode(contract) or contract['method'] != 'rmct':
        raise ValueError('RMCT v2 encounter selection contract required')
    if contract['source_commit'] != plan['incorporated_commit']:
        raise ValueError('Selection contract bound to different source')
    head = a.campaign / 'head.json'
    parent = verify_lineage(read(head), plan, binding) if head.exists() else None
    if parent is None:
        if any((a.campaign / 'batches').glob('*')):
            raise ValueError('Fresh start requested over existing children')
        budget = selection.interval_attempts()
    else:
        from transformers import AutoTokenizer
        args = dict(model=contract['model'], source_commit=contract['source_commit'], campaign_id=contract['campaign_id'])
        progress = normalized_progress(parent, **args)
        verifiers = Verifiers(manifest=a.validation_manifest, rmct_plan=plan, rmct_binding=binding,
                              tokenizer=AutoTokenizer.from_pretrained(contract['model'], local_files_only=True))
        verifiers.verify_checkpoint(progress, contract)
        state = selection.replay(selection.entries_from_folder(a.selection_folder, contract), contract_record,
                                 verify_checkpoint=verifiers.verify_checkpoint,
                                 verify_validation=verifiers.verify_validation)
        budget = selection.continuation_budget(progress, state, requested_updates=selection.interval_attempts())
    if not budget:
        return parent
    start = parent['progress']['sampled_batches'] if parent else 0
    env = dict(os.environ, PYTHONPATH=binding['source_root'], PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONHOME', None)
    parent = train_to_encounters(plan, binding, a.campaign, start + budget, checked, parent, env)
    seal = a.campaign / 'seals' / f"batches-{parent['progress']['sampled_batches']:06d}.json"
    write(seal, parent)
    os.replace(write_head(a.campaign, seal), head)
    return parent


def write_head(campaign, seal):
    temporary = Path(campaign) / f'.head-{os.getpid()}.json'
    temporary.write_text(Path(seal).read_text())
    return temporary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'gates', 'campaign', 'selection-contract', 'selection-folder', 'validation-manifest'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
