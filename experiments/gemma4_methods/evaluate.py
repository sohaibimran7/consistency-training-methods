"""Four-worker terminal-checkpoint MCQ evaluation; no automatic cap retries."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

CAPS = {'logiqa': 20480, 'hellaswag': 20480, 'hle-text-mc': 65536}
WORKERS = 4
WORKER_CHOICES = (4,8,16,32)


def shard_question_ids(question_ids, rank, workers):
    if workers not in WORKER_CHOICES or type(rank) is not int or not 0 <= rank < workers:
        raise ValueError('Invalid reviewed evaluation shard topology')
    if len(question_ids) < workers or len(question_ids) != len(set(question_ids)):
        raise ValueError('Evaluation shard population must be unique and nonempty per worker')
    return question_ids[rank::workers]


def generation_for(dataset, *, bad_words):
    if dataset not in CAPS:
        raise ValueError('Unreviewed evaluation dataset: ' + str(dataset))
    return {'max_tokens': CAPS[dataset], 'temperature': 1.0, 'top_p': 0.95,
            'top_k': 20, 'max_connections': 32,
            'extra_body': {'top_k': 20, 'bad_words': list(bad_words),
                           'chat_template_kwargs': {'enable_thinking': True}}}


def evaluation_approval(root, approval_path):
    path = Path(approval_path) if approval_path else root / 'thinking-run-approval.json'
    document = json.loads(path.read_text())
    expected = {'run_root': str(root.resolve()), 'scope': 'mcq_evaluation',
                'enable_thinking': True, 'dataset_caps_including_reasoning': CAPS}
    if any(document.get(k) != v for k, v in expected.items()) or not document.get('user_approval_reference'):
        raise ValueError('Explicit dataset-scoped thinking-on evaluation approval required')
    return identity(path)


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def prepare(args):
    workers=getattr(args,'workers',WORKERS)
    if workers not in WORKER_CHOICES:
        raise ValueError('Unreviewed evaluation worker count')
    approval = evaluation_approval(args.output, getattr(args, 'approval', None))
    from experiments.gemma4_methods.reference import train, plan
    from infra.isambard import run_gemma4_12b_base_two_bias_evals_16gpu as base
    contract = args.train_root / 'contract.json'
    from experiments.gemma4_methods.selection_adapter import file_identity, load_native_hooks
    hooks = load_native_hooks(args.verifier_factory,args)
    selected = hooks.adapter.selected_manifest(args.selection_folder,file_identity(args.selection_contract))
    if selected['selection_status'] != 'terminal':
        raise RuntimeError('Final MCQ evaluation requires verified terminal selection')
    selected_progress = json.loads(Path(selected['progress']['path']).read_text())
    if selected_progress['method'] != args.method or selected_progress['model'] != args.model:
        raise ValueError('Selected checkpoint method/model mismatch')
    checkpoint = Path(selected['checkpoint'])
    state = {'step': selected['actual_optimizer_step'], 'selection_status': 'terminal'}
    training_prompt_mode = json.loads(contract.read_text()).get('contract', {}).get('prompt_mode')
    if training_prompt_mode is None or training_prompt_mode.get('enable_thinking') is not True:
        raise ValueError('Fresh thinking-enabled training contract required')
    specs = base._load_specs(args.deployment_manifest)
    cells = []
    for index, spec in enumerate(specs, 1):
        if spec.dataset not in CAPS:
            raise ValueError('Unreviewed evaluation dataset')
        cells.append({'task_index': index, 'kind': spec.kind, 'dataset': spec.dataset,
            'max_tokens': CAPS[spec.dataset],
            'regime': spec.regime, 'population': spec.population, 'bias_type': spec.bias_type,
            'frozen_file': identity(spec.frozen_file), 'question_ids': list(spec.question_ids[:50]),
            'source_identity_digest': spec.source_identity_digest})
    assert len(cells) == 21 and all(len(c['question_ids']) == 50 for c in cells)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'manifest.json', {'method': args.method, 'training_state': state,
        'thinking_run_approval': approval,
        'selection': selected,
        'training_contract': identity(contract), 'checkpoint': str(checkpoint),
        'checkpoint_files': train.checkpoint_identity(checkpoint), 'cells': cells,
        'deployment_manifest': identity(args.deployment_manifest), 'model': args.model,
        'model_config': identity(Path(args.model) / 'config.json'),
        'generation_config': identity(Path(args.model) / 'generation_config.json'),
        'prompt_mode': {'enable_thinking': True},
        'training_prompt_mode': training_prompt_mode,
        'dataset_caps_including_reasoning': CAPS, 'workers': workers,
        'length_policy': 'preserve raw output; exclude from estimates; no automatic retries',
        'implementation': identity(__file__)})


def load(root):
    record = json.loads((root / 'manifest.json').read_text())
    from experiments.gemma4_methods.reference.train import checkpoint_identity
    assert checkpoint_identity(Path(record['checkpoint'])) == record['checkpoint_files']
    from experiments.gemma4_methods.selection_adapter import read_verified
    selected = record['selection']
    if (selected['selection_status'] != 'terminal' or selected['checkpoint'] != record['checkpoint']
            or selected['actual_optimizer_step'] != record['training_state']['step']):
        raise ValueError('Evaluation selection identity differs')
    for key in ('contract','progress','validation'):
        read_verified(selected[key])
    for file in selected['checkpoint_files'].values():
        from experiments.gemma4_methods.selection_adapter import file_identity
        if file_identity(file['path']) != file:
            raise ValueError('Selected checkpoint bytes changed')
    if {k:v['sha256'] for k,v in selected['checkpoint_files'].items()} != record['checkpoint_files']:
        raise ValueError('Selected and evaluation checkpoint file maps differ')
    for item in [record['model_config'], record['generation_config'], record['deployment_manifest'],
                 record['training_contract'], record['implementation'], record['thinking_run_approval'],
                 *[c['frozen_file'] for c in record['cells']]]:
        assert identity(item['path']) == item, 'Evaluation evidence changed'
    assert record['dataset_caps_including_reasoning'] == CAPS and record['workers'] in WORKER_CHOICES
    assert all(c['max_tokens'] == CAPS[c['dataset']] for c in record['cells'])
    assert evaluation_approval(root, record['thinking_run_approval']['path']) == record['thinking_run_approval']
    return record


def tasks(root, rank, record, *, bad_words=()):
    from inspect_ai.model import GenerateConfig
    from infra.isambard import run_gemma4_12b_base_two_bias_evals_16gpu as base
    from experiments.stage2_ood_hle.tasks import stage2_ood_biased, stage2_ood_unbiased
    result = []
    for cell in record['cells']:
        ids = shard_question_ids(cell['question_ids'],rank,record['workers'])
        kwargs = dict(frozen_file=cell['frozen_file']['path'], dataset=cell['dataset'],
             regime=cell['regime'], population=cell['population'], question_ids_from=ids,
             source_identity_digest=cell['source_identity_digest'], prompt_style='none')
        if cell['kind'] == 'unbiased':
            task = stage2_ood_unbiased(**kwargs)
        else:
            task = stage2_ood_biased(**kwargs, bias_type=cell['bias_type'],
                  unbiased_log=base._GENERATION_ONLY_SWITCH_SENTINEL,
                  include_bias_acknowledged=False, grader_model=None)
            task = base._strip_live_switch_scorer(task)
        task.metadata['gemma_methods_eval'] = {'task_index': cell['task_index'],
            'method': record['method'], 'rank': rank, 'manifest': identity(root / 'manifest.json'),
            'effective_generation': generation_for(cell['dataset'], bad_words=bad_words)}
        task.config = GenerateConfig(**generation_for(cell['dataset'], bad_words=bad_words))
        result.append(task)
    return result


def run(root, rank, *, expected_workers=None):
    record = load(root)
    if expected_workers is not None and record['workers'] != expected_workers:
        raise ValueError('Scheduled worker count differs from immutable evaluation manifest')
    assert 0 <= rank < record['workers']
    if record.get('prompt_mode') != {'enable_thinking': True}:
        raise RuntimeError('Legacy manifest: preserve it; prepare a separate thinking-enabled evaluation')
    destination = root / f'rank-{rank}'
    destination.mkdir(exist_ok=False)
    from vllm import SamplingParams
    from vllm.tokenizers import get_tokenizer
    tokenizer = get_tokenizer(record['model'], local_files_only=True)
    from transformers import AutoProcessor
    from ctm.evals.local_model import Gemma4UnifiedTextProcessor
    from ctm.backends.gemma_thinking import attest_thinking
    processor = Gemma4UnifiedTextProcessor(AutoProcessor.from_pretrained(record['model'], local_files_only=True))
    save(destination / 'thinking-attestation.json', attest_thinking(processor))
    config = json.loads(Path(record['generation_config']['path']).read_text())
    suppressed = config.get('suppress_tokens', [])
    words = [tokenizer.decode([token], skip_special_tokens=False) for token in suppressed]
    params = SamplingParams(max_tokens=65536, bad_words=words)
    params.update_from_tokenizer(tokenizer)
    assert sorted(params.bad_words_token_ids or []) == sorted([[token] for token in suppressed])
    from inspect_ai import eval
    from inspect_ai.model import GenerateConfig, get_model
    generation = {d: generation_for(d, bad_words=words) for d in CAPS}
    save(destination / 'effective-generation.json', {'per_dataset': generation,
         'scope': 'requested per-task configs; live provider attestation required by deployment gate'})
    model = get_model('vllm/' + record['model'] + ':' + record['checkpoint'],
           config=GenerateConfig(**generation['hle-text-mc']), dtype='bfloat16', tensor_parallel_size=1,
           gpu_memory_utilization=0.85, max_num_seqs=32, max_num_batched_tokens=8192,
           enforce_eager=False, generation_config='auto', host='127.0.0.1')
    logs = eval(tasks(root, rank, record, bad_words=words), model=model, log_dir=str(destination / 'raw'),
                max_tasks=4, display='none')
    counts, seen = {'generated': 0, 'truncated': 0}, set()
    for log in logs:
        assert log.status == 'success' and not log.error
        index = log.eval.metadata['gemma_methods_eval']['task_index']
        for sample in log.samples or []:
            assert not sample.error and sample.output and sample.output.choices
            reason = sample.output.choices[0].stop_reason
            assert reason in {'stop', 'length', 'max_tokens'}
            key = (index, str(sample.id))
            assert key not in seen
            seen.add(key)
            counts['generated'] += 1
            counts['truncated'] += reason in {'length', 'max_tokens'}
    expected = {(c['task_index'], qid) for c in record['cells']
                for qid in shard_question_ids(c['question_ids'],rank,record['workers'])}
    assert seen == expected
    save(destination / 'generation-complete.json', {**counts, 'rank': rank,
        'logs': [identity(p) for p in sorted((destination / 'raw').glob('*.eval'))]})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--train-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deployment-manifest', type=Path)
    p.add_argument('--model')
    p.add_argument('--method')
    p.add_argument('--rank', type=int)
    p.add_argument('--workers',type=int,choices=WORKER_CHOICES,default=WORKERS)
    p.add_argument('--approval', type=Path)
    p.add_argument('--selection-contract', type=Path)
    p.add_argument('--selection-folder', type=Path)
    p.add_argument('--verifier-factory')
    a = p.parse_args()
    if a.action == 'prepare':
        if not all((a.selection_contract,a.selection_folder,a.verifier_factory)):
            p.error('Prepare requires shared selection inputs and integrated native verifiers')
        prepare(a)
    else:
        run(a.output,a.rank,expected_workers=a.workers)
