"""Gemma adaptation of the frozen Qwen grouped-QID training loop.

The reference supplies losses, convergence and checkpoint verification. This
adapter changes the model/renderer, vLLM execution and approved cap exclusion.
"""
from __future__ import annotations
import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from experiments.gemma4_methods.reference import plan as reference
from experiments.gemma4_methods.reference import train as helpers
from experiments.gemma4_methods import one_bias

MODEL = 'google/gemma-4-12B-it'
REVISION = '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
CAP = 20480
ORDER_SHA = '83898ab455412dd929e30e6865b25eb81af4f9b370e4e1a057964daaf028df3f'


def ordered_pool(root):
    data, manifest = root / reference.POOL, root / reference.MANIFEST
    assert reference.sha256(data) == reference.POOL_SHA
    assert reference.sha256(manifest) == reference.MANIFEST_SHA
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    by_id = {row['question_id']: row for row in rows}
    identity = json.loads(manifest.read_text())
    ids = [identity['datasets'][dataset]['permutation'][i]
           for i in range(3840) for dataset in ['logiqa', 'hellaswag']]
    assert len(by_id) == len(ids) == len(set(ids)) == 7680
    assert hashlib.sha256(reference.canonical(ids)).hexdigest() == ORDER_SHA
    return [by_id[qid] for qid in ids]


def exposure_manifest(root):
    return one_bias.manifest_for(ordered_pool(root), pool_sha256=reference.POOL_SHA,
                                 manifest_sha256=reference.MANIFEST_SHA, order_sha256=ORDER_SHA)


def recipe(manifest):
    contract = reference.contract()
    contract['exposure'] = one_bias.contract_block(manifest)
    contract['batch'] = {'qids_per_update': one_bias.QIDS_PER_UPDATE, 'biases_per_qid': one_bias.BIASES_PER_QID,
        'paired_rows_per_update': one_bias.QIDS_PER_UPDATE * one_bias.BIASES_PER_QID,
        'physical_rows_per_backward': 1, 'gradient_accumulations_per_update': one_bias.QIDS_PER_UPDATE,
        'ordering': 'frozen_rmct_interleaved_logiqa_hellaswag', 'shuffle': False,
        'repeat_after_pool_exhaustion': False}
    contract['bct']['reuse_identical_target_for_both_biases'] = False
    contract['bct']['supervised_bias'] = 'assigned_bias_only'
    contract['model'] = {'repo_id': MODEL, 'revision': REVISION}
    contract['prompt_mode'] = {'enable_thinking': True, 'fresh_training_required': True,
                             'legacy_thinking_off_resume_allowed': False}
    contract['convergence'] = {'owner': 'shared_validation_controller', 'metric': 'TBSR',
        'every_encountered_qid_bias_examples': 256, 'counts_no_update_batches': True,
        'patience': 2, 'min_delta': 0,
        'strict_decrease': True, 'loss_selects_or_stops': False,
        'adapter_status': 'ctm-tbsr-selection-contract-v2-encounters; native_hooks_required',
        'user_approval': '2026-10-02: validation/checkpoint every 256 encountered QIDs incl. no-update batches'}
    contract['generation'].update(output_token_cap=CAP, sampler='vllm',
        length_stop_policy='exclude_entire_incomplete_group_before_any_backward_no_resampling',
        user_approval='requires thinking-run-approval.json scoped to the new run; legacy approval not inherited')
    contract['act_scope'] = {'preflight': 'gemma_text_qv', 'difference': 'Separate Gemma Q/V; no Qwen DeltaNet fused QKV'}
    contract['lora']['act']['target_modules'] = ['q_proj', 'v_proj']
    contract['execution'].update(training_gpus={'opct': 4, 'bct': 4, 'act': 1, 'attct': 1, 'mlpct': 1},
        job_slice_updates=16, checkpoint_every_actual_update=True,
        online_topology='one trainer plus three independent vLLM workers',
        skip_policy='advance attempted QID group but not optimizer step or convergence window',
        hf_streaming_sampling=False)
    contract['evaluation'].update(samples_per_dataset_bias=50, execution_status='pending_training')
    return contract


def freeze(root, runtime):
    from ctm.backends.gemma_thinking import require_run_approval
    approval = require_run_approval(root, scope='training', cap=CAP)
    manifest_path, manifest = one_bias.freeze(root, ordered_pool(root), pool_sha256=reference.POOL_SHA,
        manifest_sha256=reference.MANIFEST_SHA, order_sha256=ORDER_SHA)
    files = sorted((runtime / 'experiments/gemma4_methods').rglob('*.py'))
    files += sorted((runtime / 'experiments/rmct_restart_20260928').rglob('*.py'))
    files += sorted((runtime / 'ctm').rglob('*.py'))
    files += sorted((runtime / 'ctm_data').rglob('*.py'))
    sources = {str(p.resolve()): reference.sha256(p) for p in files}
    value = {'contract': recipe(manifest), 'sources': sources, 'qid_order_sha256': ORDER_SHA,
             'one_bias_manifest': {'path': str(manifest_path), 'sha256': reference.sha256(manifest_path)},
             'thinking_run_approval': approval}
    reference.immutable_json(root / 'contract.json', value)


def verify(root):
    from ctm.backends.gemma_thinking import require_run_approval
    document = json.loads((root / 'contract.json').read_text())
    assert document['thinking_run_approval'] == require_run_approval(root, scope='training', cap=CAP)
    manifest = exposure_manifest(root)
    assert document['contract'] == recipe(manifest)
    assert reference.sha256(Path(document['one_bias_manifest']['path'])) == document['one_bias_manifest']['sha256']
    for name, expected in document['sources'].items():
        assert reference.sha256(Path(name)) == expected, f'Frozen source changed: {name}'
    return reference.sha256(root / 'contract.json')


class TruncatedGroup(Exception):
    def __init__(self, reason='incomplete_generation_group'):
        super().__init__(reason)
        self.reason = reason


INCOMPLETE = ('length', 'unclosed_reasoning')


def reasoning_ids(renderer):
    """Gemma reasoning open/close token ids from the renderer's tokenizer (None if unavailable)."""
    tokenizer = getattr(renderer, 'tokenizer', None)
    convert = getattr(tokenizer, 'convert_tokens_to_ids', None)
    if convert is None:
        return None
    opened, closed = convert('<|channel>'), convert('<channel|>')
    unknown = getattr(tokenizer, 'unk_token_id', None)
    if not all(isinstance(t, int) and t != unknown for t in (opened, closed)):
        raise ValueError('Gemma reasoning channel tokens missing from tokenizer')
    return opened, closed


def validate_completion(sequence, backend, *, renderer, max_tokens):
    """Portable equivalent of the Qwen helper, without a newer-engine import."""
    tokens = list(sequence.tokens)
    eos = [t for t in (renderer.get_stop_sequences() or []) if isinstance(t, int)]
    configured = getattr(getattr(backend.model, 'generation_config', None), 'eos_token_id', None)
    if isinstance(configured, int):
        configured = [configured]
    if isinstance(configured, (list, tuple)):
        eos.extend(t for t in configured if isinstance(t, int))
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError('Invalid output-token cap')
        if len(tokens) > max_tokens:
            raise ValueError('Completion exceeds the approved cap')
    if tokens and tokens[-1] in eos:
        # A native-EOS completion whose reasoning channel was opened but never
        # closed has no complete reasoning/answer; never train on it (user-approved
        # fix 2026-10-04, audit: 1 of 3,808 trained OPCT rollouts in one-bias c2).
        ids = reasoning_ids(renderer)
        if ids is not None:
            opened, closed = ids
            if opened in tokens and closed not in tokens[len(tokens) - 1 - tokens[::-1].index(opened):]:
                return tokens, 'unclosed_reasoning'
        return tokens, 'model_eos'
    if max_tokens is not None and len(tokens) == max_tokens:
        return tokens, 'length'
    raise ValueError('Unknown generation termination; refusing training')


def alignment_processor(processor):
    from ctm.evals.local_model import Gemma4UnifiedTextProcessor
    class Processor(Gemma4UnifiedTextProcessor):
        def apply_chat_template(self,*args,**kwargs):
            if kwargs.get('enable_thinking',True) is not True:
                raise ValueError('Fresh Gemma consistency alignment requires thinking enabled')
            kwargs['enable_thinking']=True
            return super().apply_chat_template(*args,**kwargs)
        def __call__(self, text=None, /, **kwargs):
            encoded = self.tokenizer(text, **kwargs)
            official = self.processor(text=text, add_special_tokens=kwargs.get('add_special_tokens', True),
                                      return_mm_token_type_ids=False)['input_ids']
            if official and isinstance(official[0], list):
                official = official[0]
            assert encoded['input_ids'] == official
            return encoded
    return Processor(processor)


def training_processors(official_processor):
    """Attest on/off controls without weakening thinking-on alignment."""
    from ctm.backends.gemma_thinking import attest_thinking
    from ctm.evals.local_model import Gemma4UnifiedTextProcessor
    evidence = attest_thinking(Gemma4UnifiedTextProcessor(official_processor))
    return alignment_processor(official_processor), evidence


async def run(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from ctm.backends.local.engine import LocalBackend
    from ctm.backends.local.rollout_workers import RolloutParallelBackend, resolve_rollout_gpus
    from ctm.backends.renderers import HuggingFaceChatTemplateRenderer
    from ctm.core.config import LoRAConfig, AdamConfig
    from ctm.training.consistency_data import build_consistency_datums_with_audit, require_full_reference_suffix_alignment
    from ctm.training.sft import METHOD_LOSS_FNS

    root, method = args.root.resolve(), args.method
    plan_hash = verify(root)
    assert Path(args.model).name == REVISION
    reference.OUTPUT_TOKEN_CAP = CAP  # Selected reference helpers receive only this run's approved cap.
    helpers.validate_completion = validate_completion
    run_dir = root / 'runs' / method
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from experiments.gemma4_methods.checkpoint import recover_publication
        recover_publication(run_dir,plan_hash,method,
                            one_bias.protocol.manifest_identity(exposure_manifest(root)))
        state, resume = helpers.load_resume(run_dir, plan_hash, method)
        if state['decision'] != 'continue':
            return
        from experiments.gemma4_methods.progress import record_update
        from experiments.gemma4_methods.selection_adapter import (
            file_identity, read_verified, load_native_hooks, verified_budget, verified_bootstrap)
        hooks = load_native_hooks(args.verifier_factory,args)
        selection_contract = file_identity(args.selection_contract)
        scientific_contract = read_verified(selection_contract)
        if scientific_contract['method'] != method or scientific_contract['model'] != args.model:
            raise ValueError('Selection/training method or model differs')
        if resume is None:
            # Native hook checks original weights, fresh optimizer, new lineage,
            # exact data/source/cap approvals and initialization gates.
            # Budgets are sampled-batch attempts (4 encounters each), incl. skips.
            budget = verified_bootstrap(selection_contract,hooks.start_record,
                                        args.updates,hooks.verify_start)
            # Before the first sealed update, earlier jobs may only have written
            # skip records; continue after them instead of replaying the same budget.
            prior = 0
            while (run_dir / 'skips' / f'attempt-{prior:07d}.json').exists():
                prior += 1
            from experiments.rmct_restart_20260928.validation_selection import interval_attempts
            end_attempt = min(prior + budget, interval_attempts())
        else:
            progress = hooks.normalized_progress(run_dir,scientific_contract)
            if (progress['actual_optimizer_step'],progress['next_attempt_index']) != (state['step'],state.get('attempts',state['step'])):
                raise ValueError('Native normalized cursor differs from saved state')
            budget = verified_budget(hooks.adapter,args.selection_folder,selection_contract,progress,args.updates)
            if not budget:
                return
            end_attempt = progress['encounter_attempt']+budget
        state.setdefault('attempts', state['step'])
        pool = ordered_pool(root)
        manifest = exposure_manifest(root)
        by_id = {row['question_id']: row for row in pool}
        # Fail template/control checks before starting any rollout workers.
        official_processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        tokenizer, evidence = training_processors(official_processor)
        online = method in {'bct', 'opct'}
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
                  local_files_only=True, **({'attn_implementation': 'eager'} if not online else {}))
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and 'language_model' in name.split('.')
                   and ((('self_attn' in name.split('.') or 'mlp' in name.split('.')) if online
                         else name.rsplit('.', 1)[-1] in {'q_proj', 'v_proj'}) )]
        assert targets
        lora = LoRAConfig(rank=8, alpha=16, dropout=0, train_mlp=online, train_attn=online,
                          train_unembed=False, seed=42, target_modules=targets)
        local = LocalBackend(device='cuda:0', dtype=torch.bfloat16, model_instance=model,
                    sampler='vllm' if online else 'hf', gradient_checkpointing=True,
                    consistency_loss_options=reference.LOSS_OPTIONS[method],
                    forward_microbatch_max_datums=1, forward_microbatch_max_tokens=40960,
                    target_logprob_chunk_size=2048)
        backend = local
        if online:
            gpus = resolve_rollout_gpus('1,2,3', cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                                        coordinator_device='cuda:0')
            backend = RolloutParallelBackend(local, gpus=gpus, status_dir=run_dir / 'workers',
                         worker_vllm_options={'dtype': 'bfloat16', 'gpu_memory_utilization': 0.85,
                          'max_num_seqs': 32, 'max_num_batched_tokens': 8192, 'enforce_eager': False,
                          'generation_config': 'vllm', 'seed': 42})
        backend.setup(model=args.model, lora=lora, resume_from=resume, resume_with_optimizer=resume is not None)
        reference.immutable_json(run_dir / 'attempts' / f'{os.environ.get("SLURM_JOB_ID", "local")}.json',
            {'plan_sha256': plan_hash, 'method': method, 'resume_from': resume,
             'starting_step': state['step'], 'starting_attempt': state['attempts'],
             'model': args.model, 'lora_targets': targets,
             'trainable_names': [n for n, p in backend.model.named_parameters() if p.requires_grad]})
        evidence_path = run_dir / 'thinking-attestation.json'
        if evidence_path.exists():
            assert json.loads(evidence_path.read_text()) == evidence
        else:
            reference.immutable_json(evidence_path, evidence)
        renderer = HuggingFaceChatTemplateRenderer(tokenizer, chat_template_kwargs={'enable_thinking': True})
        adam = AdamConfig(**reference.contract()['optimizer'])
        window_metrics = helpers.resume_window_metrics(run_dir, state)
        from experiments.gemma4_methods.checkpoint import seal_checkpoint, restore_coordinator_rng
        trainer = None
        sample_wall_seconds = 0.0
        if method == 'opct':
            from ctm.training.opct import OPCTTrainer, OPCTConfig, OPCTGenerationConfig
            trainer = OPCTTrainer(config=OPCTConfig(model=args.model, lora=lora, optimizer=adam,
                generation=OPCTGenerationConfig(rollouts_per_prompt=4, max_new_tokens=CAP, temperature=0.7),
                batch_size=1, gradient_accumulation_steps=4, shuffle_samples=False,
                kl_coef=2.0, kl_discount_factor=0.9, loss_fn='importance_sampling'), backend=backend)
            trainer.renderer, trainer.tokenizer = renderer, tokenizer
            trainer.sampling_client = backend.policy_sampler(name='opct-policy')
            trainer.reference_policy = backend.base_sampler()
            trainer.setup_done = True
            original_sample = trainer._sample_prepared_pairs
            async def sample_checked(prepared):
                nonlocal sample_wall_seconds
                sample_began=time.perf_counter()
                try:
                    sampled = await original_sample(prepared)
                finally:
                    sample_wall_seconds+=time.perf_counter()-sample_began
                records = []
                excluded = False
                unclosed = False
                for group in sampled:
                    for sample in group:
                        tokens, finish = helpers.validate_completion(sample, backend, renderer=renderer, max_tokens=CAP)
                        excluded |= finish in INCOMPLETE or sample.logprobs is None
                        unclosed |= finish == 'unclosed_reasoning'
                        records.append({'tokens': tokens, 'finish_reason': finish, 'usable': sample.logprobs is not None})
                helpers.append_json(run_dir / 'generations.jsonl', {'step': state['step'], 'attempt': state['attempts'], 'samples': records})
                if excluded:
                    raise TruncatedGroup('incomplete_reasoning_group' if unclosed else 'incomplete_generation_group')
                return sampled
            trainer._sample_prepared_pairs = sample_checked
        if resume is not None:
            restore_coordinator_rng(resume)
        starting_step = state['step']
        began = time.monotonic()
        try:
            while state['decision'] == 'continue' and state['attempts'] < end_attempt:
                if time.monotonic() - began > 9*3600:
                    break
                attempt = state['attempts']
                if attempt >= one_bias.max_attempts(manifest):
                    # Finite one-pass pool: report exhaustion, never cycle QIDs.
                    state['decision'] = 'exhausted'
                    print(json.dumps({'method': method, 'exhausted_at_attempt': attempt, **state}), flush=True)
                    break
                skip_path = run_dir / 'skips' / f'attempt-{attempt:07d}.json'
                if skip_path.exists():
                    skip = json.loads(skip_path.read_text())
                    assert skip['step'] == state['step'] and skip['plan_sha256'] == plan_hash
                    state['attempts'] += 1
                    continue
                random.seed(42 + state['step'])
                torch.manual_seed(42 + state['step'])
                qids, biases = one_bias.update_rows(manifest, by_id, attempt)
                pairs = reference.one_bias_pairs(qids, biases, method=method)
                update_began=time.perf_counter()
                sample_wall_seconds=0.0
                try:
                    if method == 'bct':
                        target_began=time.perf_counter()
                        for row in qids:
                            tokens = await helpers.bct_target(row, backend=backend, renderer=renderer,
                                cache_dir=run_dir / 'base-targets', plan_hash=plan_hash)
                            from ctm.backends.base import SampledSequence
                            _, finish = helpers.validate_completion(SampledSequence(tokens=tokens, logprobs=[]),
                                                backend, renderer=renderer, max_tokens=CAP)
                            if finish in INCOMPLETE:
                                raise TruncatedGroup('incomplete_reasoning_group' if finish == 'unclosed_reasoning'
                                                     else 'incomplete_generation_group')
                        sample_wall_seconds=time.perf_counter()-target_began
                        losses, detail = await helpers.supervised_update(method, qids, pairs,
                            backend=backend, renderer=renderer, tokenizer=tokenizer,
                            cache_dir=run_dir / 'base-targets', plan_hash=plan_hash, preflight_path=None)
                    elif method == 'opct':
                        losses, detail = await helpers.opct_update(trainer, pairs, backend=backend)
                    else:
                        datums, audit = build_consistency_datums_with_audit(tokenizer, pairs)
                        require_full_reference_suffix_alignment(audit)
                        assert len(datums) == len(pairs)
                        losses = []
                        for datum in datums:
                            pending = await backend.submit_forward_backward([datum], loss_fn=METHOD_LOSS_FNS[method])
                            output = await pending.result()
                            losses.append(float(output.metrics['loss']))
                    assert len(losses) == len(pairs) and all(math.isfinite(x) for x in losses)
                except TruncatedGroup as skipped:
                    assert local._gradient_accumulations == 0, 'Excluded group already changed gradients'
                    reference.immutable_json(skip_path, {'step': state['step'], 'attempt': attempt,
                        'plan_sha256': plan_hash, 'question_ids': [r['question_id'] for r in qids],
                        'biases': biases, 'reason': skipped.reason, 'optimizer_update': False,
                        'encounters_consumed': len(qids)})
                    state['attempts'] += 1
                    print(f'Skipped truncated group {attempt}; optimizer remains {state["step"]}', flush=True)
                    continue
                gradients = helpers.assert_gradients(backend)
                before_optimizer=time.perf_counter()
                pending = await backend.submit_optim_step(learning_rate=1e-4, adam=adam)
                await pending.result()
                optimizer_seconds=time.perf_counter()-before_optimizer
                prepare_score_backward_seconds=max(0.0,before_optimizer-update_began-sample_wall_seconds)
                state = record_update(state, attempt=attempt, loss=math.fsum(losses)/len(losses),
                                      question_ids=[r['question_id'] for r in qids])
                metric = {'step': state['step'], 'attempt': attempt, 'loss': math.fsum(losses)/len(losses),
                          'variant_metrics': losses, 'gradient_report': gradients,
                          'question_ids': [r['question_id'] for r in qids], 'biases': biases}
                window_metrics.append(metric)
                helpers.append_json(run_dir / 'metrics.jsonl', metric)
                checkpoint_began=time.perf_counter()
                sealed = await seal_checkpoint(backend, run_dir=run_dir, method=method,
                        state=state, plan_hash=plan_hash, window_metrics=window_metrics)
                checkpoint_seconds=time.perf_counter()-checkpoint_began
                helpers.append_json(run_dir/'performance.jsonl',{
                    'schema':'gemma-saved-update-timing-v1','run_root':str(root),
                    'method':method,'step':state['step'],'attempt':attempt,
                    'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
                    'checkpoint_saved':True,'checkpoint':sealed['checkpoint'],
                    'checkpoint_files':sealed['checkpoint_files'],
                    'stage_wall_seconds':{'sampling':sample_wall_seconds,
                        'prepare_score_backward':prepare_score_backward_seconds,
                        'optimizer':optimizer_seconds,'checkpoint':checkpoint_seconds},
                    'sampling_scope':'BCT includes cache reads; OPCT generation await only; internal methods zero',
                    'scope':'host wall time; excludes skipped attempts and startup; not whole-allocation throughput'})
                from experiments.gemma4_methods.checkpoint import make_progress
                # One builder for trainer and recovery so resumed jobs reproduce it exactly.
                reference.immutable_json(run_dir / 'progress' / f'step-{state["step"]:06d}.json',
                    make_progress(sealed, one_bias.protocol.manifest_identity(manifest)))
                if state['step'] % 16 == 0:
                    window_metrics = []
                if trainer is not None:
                    trainer.sampling_client = await backend.refresh_policy_sampler(name=f'opct-{state["step"]}')
                print(json.dumps({'method': method, **state}), flush=True)
        finally:
            backend.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'train'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--model')
    parser.add_argument('--method', choices=reference.METHODS)
    parser.add_argument('--updates', type=int, default=16)
    parser.add_argument('--selection-contract', type=Path)
    parser.add_argument('--selection-folder', type=Path)
    parser.add_argument('--verifier-factory')
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.root.resolve(), args.runtime.resolve())
    else:
        if not all((args.selection_contract,args.selection_folder,args.verifier_factory)):
            parser.error('Training requires shared selection contract/folder and integrated native verifier factory')
        asyncio.run(run(args))
