"""Qwen one-bias campaign validation on the shared 256-encounter TBSR grid.

Produces the shared ``ctm-training-progress-v1`` record from a sealed Qwen
checkpoint receipt, prepares/probes the evaluator adapter, scores a completed
600-response job and supplies the checkpoint/validation verifiers that
``validation_selection`` requires. It never submits jobs or advances training.
"""
import argparse
import os
from pathlib import Path

from experiments.rmct_restart_20260928 import qwen_validation as v
from experiments.rmct_restart_20260928 import validation_selection as selection
from experiments.rmct_restart_20260928.qwen_checkpoint import identity
from experiments.rmct_restart_20260928.qwen_validation_executor import check_contract, read, write
from experiments.rmct_restart_20260928.qwen_validation_producer import scheduler_complete

SCHEMA = 'qwen-one-bias-validation-v1'
FIVE_METHOD_FILES = ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'manifest.json')


def five_method_progress(run_dir, receipt_path, *, campaign_id, source_commit, model):
    """Normalize a sealed five-method receipt into shared progress (no inference)."""
    run_dir = Path(run_dir).resolve()
    receipt = read(receipt_path)
    exposure = receipt.get('exposure')
    if receipt.get('schema') != 'ctm-grouped-qid-resume-v1' or exposure is None:
        raise ValueError('One-bias five-method receipt required')
    checkpoint = (run_dir / receipt['checkpoint']).resolve()
    if not checkpoint.is_relative_to(run_dir / 'checkpoints') or checkpoint.is_symlink():
        raise ValueError('Checkpoint outside the method run directory')
    files = {}
    for name in FIVE_METHOD_FILES:
        record = identity(checkpoint / name)
        if receipt['checkpoint_files'].get(name) != record['sha256']:
            raise ValueError(f'Checkpoint bytes differ from receipt: {name}')
        files[name] = record
    manifest = read(checkpoint / 'manifest.json')
    if manifest['loop_state'].get('exposure') != exposure or manifest['loop_state']['convergence'] != receipt['convergence']:
        raise ValueError('Checkpoint loop state differs from its receipt')
    step = receipt['convergence']['step']
    if (exposure['actual_optimizer_step'], exposure['encounter_attempt']) != (step, step) or exposure['trailing_skip_files']:
        raise ValueError('Five-method one-bias checkpoints have no skipped attempts')
    return dict(schema='ctm-training-progress-v1', campaign_id=campaign_id, method=receipt['method'],
                model=model, source_commit=source_commit, actual_optimizer_step=step,
                next_attempt_index=step, sampled_batches=step, encounter_attempt=step,
                encountered_qid_bias_examples=exposure['encountered_qid_bias_examples'],
                one_bias_manifest_sha256=exposure['one_bias_manifest_sha256'],
                trailing_skip_files=[], receipt=identity(receipt_path),
                checkpoint=str(checkpoint), checkpoint_files=files)


def selection_contract(*, campaign_id, method, model, source_commit, population):
    """Shared v2 encounter selection contract for one Qwen method (never overwritten)."""
    return dict(schema='ctm-tbsr-selection-contract-v2-encounters', policy=selection.ENCOUNTER_POLICY,
                campaign_id=campaign_id, method=method, model=model, source_commit=source_commit,
                response_count=600, pair_count=400, settings=v.SETTINGS, population=identity(population))


def gated_budget(*, run_dir, step, requested, contract_path, folder, manifest, model_path):
    """Updates this job may train: first window fresh, later windows only past accepted validation."""
    if step == 0:
        return min(requested, selection.interval_attempts())
    from transformers import AutoTokenizer
    contract_record = identity(contract_path)
    contract = selection.read_verified(contract_record)
    selection.check_contract(contract)
    progress = five_method_progress(run_dir, Path(run_dir) / 'receipts' / f'step-{step:06d}.json',
                                    campaign_id=contract['campaign_id'], source_commit=contract['source_commit'],
                                    model=contract['model'])
    verifiers = Verifiers(manifest=manifest, tokenizer=AutoTokenizer.from_pretrained(model_path, local_files_only=True),
                          run_dir=run_dir)
    verifiers.verify_checkpoint(progress, contract)
    state = selection.replay(selection.entries_from_folder(folder, contract), contract_record,
                             verify_checkpoint=verifiers.verify_checkpoint,
                             verify_validation=verifiers.verify_validation)
    return selection.continuation_budget(progress, state, requested_updates=requested)


def boundary(progress):
    encounters = progress['encountered_qid_bias_examples']
    if type(encounters) is not int or encounters <= 0 or (encounters % 256 and encounters != 7680):
        raise ValueError('Validation requires a 256-encounter boundary')
    return encounters


def prepare(a):
    """GPU step: verify progress, translate and probe the adapter, write the contract."""
    from experiments.rmct_restart_20260928.qwen_launch import source_check
    from experiments.rmct_restart_20260928.qwen_validation_prepare import translate_and_probe
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Scheduled GPU allocation required')
    root = Path(__file__).resolve().parents[2]
    source_check(root, a.source_commit)
    scheduler_complete(a.training_job)
    v.population(a.manifest)
    contract = selection.read_verified(identity(a.selection_contract))
    selection.check_contract(contract)
    if not selection.encounter_mode(contract):
        raise ValueError('One-bias validation requires the encounter selection contract')
    progress = selection.check_progress(read(a.progress), contract)
    encounters = boundary(progress)
    out = a.folder.resolve()
    adapter, raw, _ = translate_and_probe(
        progress['checkpoint'], progress['checkpoint_files'], out, reference_report=a.reference_report,
        reference_sha=a.reference_sha, model=a.model, port=a.port,
        claim=dict(job_id=os.environ['SLURM_JOB_ID'], progress_sha256=v.sha(a.progress),
                   reference_sha256=a.reference_sha, source_commit=a.source_commit))
    value = dict(schema=SCHEMA, campaign_id=progress['campaign_id'], method=progress['method'],
                 step=progress['actual_optimizer_step'], encounter_attempt=progress['encounter_attempt'],
                 encountered_qid_bias_examples=encounters, source_commit=a.source_commit, model=a.model,
                 adapter=str(adapter), settings=v.SETTINGS, validation_sha256=v.PROMPT_SHA,
                 raw_adapter=raw, translated_adapter=identity(adapter / 'adapter_model.safetensors'),
                 activity_report=identity(out / 'activity-report.json'),
                 checkpoint_seal=identity(a.progress), progress=identity(a.progress),
                 checkpoint_files=progress['checkpoint_files'],
                 translation=identity(adapter / 'compatibility-manifest.json'),
                 raw_config=progress['checkpoint_files']['adapter_config.json'],
                 translated_config=identity(adapter / 'adapter_config.json'))
    check_contract(value, root)
    write(out / 'contract.json', value)


def score(folder, manifest, job):
    """After COMPLETED|0:0 only: reproduce TBSR and write the shared metrics record."""
    from transformers import AutoTokenizer
    folder = Path(folder).resolve()
    contract = read(folder / 'contract.json')
    if contract['schema'] != SCHEMA:
        raise ValueError('Not a one-bias validation folder')
    check_contract(contract, Path(__file__).resolve().parents[2])
    state = scheduler_complete(job)
    for rank in range(4):
        claim, done = read(folder / f'worker-{rank}-claim.json'), read(folder / f'worker-{rank}-complete.json')
        if claim['job_id'] != str(job) or done['count'] != 150 or done['contract_sha256'] != v.sha(folder / 'contract.json'):
            raise ValueError('Worker/job/contract mismatch')
    records = [read(p) for p in sorted((folder / 'responses').glob('*.json'))]
    metrics = v.score(manifest, records)
    row = dict(metrics, campaign_id=contract['campaign_id'], step=contract['step'],
               encountered_qid_bias_examples=contract['encountered_qid_bias_examples'],
               validation_sha256=v.PROMPT_SHA, settings_sha256=v.settings_sha(), verified=True,
               job_completed=True, response_count=len(records), contract_sha256=v.sha(folder / 'contract.json'),
               response_hashes={r['sample_id']: v.sha(folder / 'responses' / f"{r['sample_id']}.json") for r in records})
    write(folder / 'score.json', row)
    v.verify_saved(folder, manifest, AutoTokenizer.from_pretrained(contract['model'], local_files_only=True))
    write(folder / 'completed-job.json', dict(job_id=str(job), scheduler_state=state))
    metrics = selection_metrics(folder)
    write(folder / 'validation.json', metrics)
    return metrics


def selection_metrics(folder):
    folder = Path(folder).resolve()
    contract, row = read(folder / 'contract.json'), read(folder / 'score.json')
    return dict(step=row['step'], campaign_id=row['campaign_id'], response_count=row['response_count'],
                checkpoint_files=contract['checkpoint_files'], towards_switches=row['towards_switches'],
                eligible_pairs=row['eligible_pairs'], tbsr=row['tbsr'],
                encountered_qid_bias_examples=row['encountered_qid_bias_examples'],
                score_sha256=v.sha(folder / 'score.json'), contract_sha256=v.sha(folder / 'contract.json'))


class Verifiers:
    """Callbacks for validation_selection.replay/accept_validation."""

    def __init__(self, *, manifest, tokenizer, run_dir=None, rmct_plan=None, rmct_binding=None):
        self.manifest, self.tokenizer = manifest, tokenizer
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.rmct_plan, self.rmct_binding = rmct_plan, rmct_binding

    def verify_checkpoint(self, progress, contract):
        selection.check_progress(progress, contract)
        identity_args = dict(campaign_id=contract['campaign_id'], source_commit=contract['source_commit'],
                             model=contract['model'])
        if contract['method'] == 'rmct':
            from experiments.rmct_restart_20260928.qwen_one_bias_rmct import normalized_progress, verify_lineage
            # Reseal the whole one-bias RMCT lineage from bytes before trusting counters.
            head = verify_lineage(progress['rmct_seal'], self.rmct_plan, self.rmct_binding)
            fresh = normalized_progress(head, **identity_args)
        else:
            fresh = five_method_progress(self.run_dir, progress['receipt']['path'], **identity_args)
        if fresh != progress:
            raise ValueError('Progress differs from the sealed receipt')
        return progress

    def verify_validation(self, record, progress, contract):
        folder = Path(record['path']).resolve().parent
        saved = read(folder / 'contract.json')
        if saved['schema'] != SCHEMA or (saved['step'], saved['encountered_qid_bias_examples']) != (
                progress['actual_optimizer_step'], progress['encountered_qid_bias_examples']):
            raise ValueError('Validation contract mismatch')
        if saved['checkpoint_files'] != progress['checkpoint_files'] or saved['method'] != contract['method']:
            raise ValueError('Validation bound to a different checkpoint')
        scheduler_complete(read(folder / 'completed-job.json')['job_id'])
        v.verify_saved(folder, self.manifest, self.tokenizer)
        metrics = selection_metrics(folder)
        if read(record['path']) != metrics:
            raise ValueError('Stored selection metrics differ from saved responses')
        return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('progress', 'prepare', 'score', 'contract'))
    for name in ('folder', 'manifest', 'progress', 'reference-report', 'selection-contract', 'run-dir', 'receipt', 'output'):
        p.add_argument('--' + name, type=Path)
    for name in ('reference-sha', 'model', 'source-commit', 'training-job', 'campaign-id', 'job', 'method'):
        p.add_argument('--' + name)
    p.add_argument('--port', type=int, default=19789)
    a = p.parse_args()
    if a.mode == 'progress':
        write(a.output, five_method_progress(a.run_dir, a.receipt, campaign_id=a.campaign_id,
                                             source_commit=a.source_commit, model=a.model))
    elif a.mode == 'contract':
        write(a.output, selection_contract(campaign_id=a.campaign_id, method=a.method, model=a.model,
                                           source_commit=a.source_commit, population=a.manifest))
    elif a.mode == 'prepare':
        prepare(a)
    else:
        print(score(a.folder, a.manifest, a.job))
