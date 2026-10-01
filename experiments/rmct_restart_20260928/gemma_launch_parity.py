"""Two-process native parity probe; synthetic nonzero adapter, no optimizer.

An explicitly approved one-token tail is unused; only fixed input tokens score.
This gate is not scientific launch clearance or full multiworker validation.
"""
import argparse
import hashlib
import json
import math
import importlib.metadata
from pathlib import Path
import subprocess
import sys

ENGINE_OPTIONS = {'dtype': 'bfloat16', 'gpu_memory_utilization': 0.85,
                  'max_lora_rank': 8, 'max_num_seqs': 32,
                  'max_num_batched_tokens': 8192, 'enforce_eager': False,
                  'generation_config': 'vllm', 'seed': 42}


def verify_module_origins(receipt, modules):
    root = Path(receipt['source_root']).resolve()
    attested = {Path(x['path']).resolve(): x['sha256'] for x in receipt['sources']}
    checked = []
    for path in modules:
        path = Path(path).resolve()
        if not path.is_relative_to(root) or path not in attested:
            raise RuntimeError('Imported module is not in CPU-attested source tree: ' + str(path))
        if hashlib.sha256(path.read_bytes()).hexdigest() != attested[path]:
            raise RuntimeError('Imported module changed since CPU preflight')
        checked.append(str(path))
    return checked


def save(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def score_records(sampler, records, *, use_base, termination):
    """Score every fixed completion token; discard the approved one-token tail."""
    if not records or any(not r['prompt'] or not r['completion'] for r in records):
        raise RuntimeError('Nonempty prompt/completion records required')
    sampler.wake_up()
    params = sampler._api.SamplingParams(n=1, max_tokens=1, temperature=0.0, prompt_logprobs=0)
    combined = [r['prompt'] + r['completion'] for r in records]
    outputs = sampler.engine.generate([sampler._api.TokensPrompt(prompt_token_ids=t) for t in combined],
        params, lora_request=None if use_base else sampler._policy_lora_request(), use_tqdm=False)
    if len(outputs) != len(records):
        raise RuntimeError('Parity batch incomplete')
    scores = []
    for record, ids, result in zip(records, combined, outputs):
        if (result.prompt_token_ids is None or list(result.prompt_token_ids) != ids
                or result.prompt_logprobs is None or len(result.prompt_logprobs) != len(ids)):
            raise RuntimeError('Prompt score token mismatch')
        if len(result.outputs) != 1:
            raise RuntimeError('Expected exactly one diagnostic tail')
        seq = result.outputs[0]
        termination.append({'base': use_base, 'finish_reason': seq.finish_reason,
                            'generated_tokens': len(seq.token_ids), 'cap': 1,
                            'generated_token_ids': list(seq.token_ids),
                            'used_for_scores_or_training': False})
        # Length termination is expected here, never accepted for a scientific
        # response. Scores below cover only supplied completion tokens.
        if (seq.finish_reason not in ('stop', 'length') or len(seq.token_ids) != 1
                or any(type(t) is not int or t < 0 for t in seq.token_ids)):
            raise RuntimeError('Unknown/invalid one-token diagnostic tail: parity fails closed')
        values = []
        for i in range(len(record['prompt']), len(ids)):
            entry = result.prompt_logprobs[i]
            if not entry or ids[i] not in entry or not math.isfinite(entry[ids[i]].logprob):
                raise RuntimeError('Missing/nonfinite teacher-forced score')
            values.append(entry[ids[i]].logprob)
        scores.append(values)
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['hf', 'vllm'])
    p.add_argument('--model', required=True)
    p.add_argument('--cpu-receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scoring-tail-approval-reference', required=True)
    a = p.parse_args()
    if not a.scoring_tail_approval_reference.strip():
        raise RuntimeError('Explicit scoped one-token diagnostic approval required')
    receipt = json.loads(a.cpu_receipt.read_text())
    if (receipt.get('schema'), receipt.get('status'), receipt.get('optimizer_work_authorized')) != (
            'rmct-restart-cpu-v1', 'cpu_checks_passed', False):
        raise RuntimeError('Current integrated CPU receipt required')
    if str(Path(sys.executable).absolute()) != str(Path(receipt['python']).absolute()):
        raise RuntimeError('Python environment differs from CPU preflight')
    root = Path(receipt['source_root']).resolve()
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(root), *argv], text=True).strip()
    if git('rev-parse', 'HEAD') != receipt['source_commit'] or git('status', '--porcelain'):
        raise RuntimeError('Incorporated clean source changed')
    for item in receipt['sources']:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('CPU-attested module changed')
    from ctm.backends import renderers, gemma_thinking
    from ctm.backends.local import vllm_sampler
    checked_modules = verify_module_origins(receipt, [__file__, renderers.__file__,
        gemma_thinking.__file__, vllm_sampler.__file__])
    pins = json.loads(Path(__file__).with_name('gemma_launch_pins.json').read_text())
    if Path(a.model).name != pins['revision']:
        raise RuntimeError('Wrong original snapshot')
    expected = pins['weights']['model.safetensors']
    matches = [x for x in receipt['model_files'] if Path(x['path']).name == 'model.safetensors']
    if len(matches) != 1 or any(matches[0].get(k) != expected[k] for k in ('sha256', 'bytes')):
        raise RuntimeError('Independent publisher weight identity mismatch')
    if Path(matches[0]['path']).resolve() != (Path(a.model) / 'model.safetensors').resolve():
        raise RuntimeError('Requested weights differ from CPU-attested weights')
    for name, version in receipt['dependencies'].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError('Dependency version changed since CPU preflight: ' + name)
    import numpy as np
    if a.stage == 'hf':
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText
        from ctm.backends.renderers import get_renderer_and_tokenizer
        a.output.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(42)
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16,
                                                          local_files_only=True)
        targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
                   and 'language_model' in n.split('.')
                   and ('self_attn' in n.split('.') or 'mlp' in n.split('.'))]
        if not targets:
            raise RuntimeError('No text LoRA targets')
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
                                               target_modules=targets)).to('cuda:0').eval()
        with torch.no_grad():
            for n, param in model.named_parameters():
                if 'lora_B' in n:
                    param.normal_(0, 0.01)
        model.save_pretrained(a.output / 'disposable-adapter')
        renderer, tokenizer = get_renderer_and_tokenizer(a.model, source='hf')
        prompt = renderer.build_generation_prompt(receipt['thinking']['probe_messages']).to_ints()
        if prompt != receipt['thinking']['prompt_token_ids']['true']:
            raise RuntimeError('Thinking renderer mismatch')
        records = []
        for letter in 'ABCD':
            completion = tokenizer.encode('The answer is ' + letter + '.', add_special_tokens=False)
            ids = torch.tensor([prompt + completion], device='cuda:0')
            def score():
                logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt)-1:-1].float()
                return logits.log_softmax(-1).gather(-1, ids[0, len(prompt):, None]).squeeze(-1).cpu().tolist()
            with torch.no_grad():
                policy = score()
                with model.disable_adapter():
                    base = score()
            records.append(dict(prompt=prompt, completion=completion, hf_base=base, hf_policy=policy))
        save(a.output / 'hf.json', records)
        save(a.output / 'provenance.json', {'cpu_receipt_sha256': hashlib.sha256(a.cpu_receipt.read_bytes()).hexdigest(),
             'source_commit': receipt['source_commit'], 'targets': targets,
             'python': sys.executable, 'checked_module_origins': checked_modules,
             'engine_options': ENGINE_OPTIONS,
             'context_limit': 'model configuration; no probe override',
             'sampling_temperature': 0.0,
             'diagnostic_generation': {'max_tokens': 1, 'tail_used_for_scores_or_training': False,
                 'length_termination': 'allowed only for unused diagnostic tail',
                 'approval_reference': a.scoring_tail_approval_reference},
             'production_equivalence': 'teacher-forced per-engine parity only; not rollout distribution/topology parity',
             'adapter': 'synthetic random nonzero LoRA, no optimizer, never a scientific parent'})
        return
    provenance = json.loads((a.output / 'provenance.json').read_text())
    if provenance['cpu_receipt_sha256'] != hashlib.sha256(a.cpu_receipt.read_bytes()).hexdigest():
        raise RuntimeError('HF and vLLM CPU provenance differ')
    if provenance.get('diagnostic_generation') != {
            'max_tokens': 1, 'tail_used_for_scores_or_training': False,
            'length_termination': 'allowed only for unused diagnostic tail',
            'approval_reference': a.scoring_tail_approval_reference}:
        raise RuntimeError('HF and vLLM scoped diagnostic approval differ')
    from ctm.backends.local.vllm_sampler import VLLMSampler
    sampler = VLLMSampler(a.model, **ENGINE_OPTIONS)
    sampler.advance_policy(str(a.output / 'disposable-adapter'))
    records = json.loads((a.output / 'hf.json').read_text())
    termination = []
    try:
        for use_base, key in [(True, 'vllm_base'), (False, 'vllm_policy')]:
            try:
                scores = score_records(sampler, records, use_base=use_base, termination=termination)
            except Exception:
                save(a.output / 'failed-termination.json', termination)
                raise
            for record, values in zip(records, scores):
                record[key] = values
    finally:
        sampler.shutdown()
    def flat(k):
        return np.array([v for r in records for v in r[k]], dtype=float)
    hb, hp, vb, vp = [flat(k) for k in ('hf_base', 'hf_policy', 'vllm_base', 'vllm_policy')]
    hd, vd = hp-hb, vp-vb
    hn, vn = np.linalg.norm(hd), np.linalg.norm(vd)
    metrics = dict(max_abs_raw_error=float(max(np.max(abs(hb-vb)), np.max(abs(hp-vp)))),
        raw_correlation=float(np.corrcoef(np.r_[hb,hp], np.r_[vb,vp])[0,1]),
        hf_effect_norm=float(hn), effect_cosine=float(hd@vd/(hn*vn)) if hn*vn else 0,
        effect_norm_ratio=float(vn/hn) if hn else 0,
        effect_relative_error=float(np.linalg.norm(hd-vd)/hn) if hn else 1)
    passed = (all(np.isfinite(v) for v in metrics.values()) and metrics['max_abs_raw_error'] <= .5
              and metrics['raw_correlation'] >= .999 and hn >= .5 and metrics['effect_cosine'] >= .9
              and .8 <= metrics['effect_norm_ratio'] <= 1.25 and metrics['effect_relative_error'] <= .25)
    save(a.output / 'scores.json', records)
    save(a.output / 'gpu-parity.json', {'status': 'passed' if passed else 'failed', 'metrics': metrics,
         'termination': termination, 'provenance': provenance, 'optimizer_work_authorized': False,
         'remaining_gates': ['production_multiworker_regression', 'validation_semantics', 'coordinator_clearance']})
    if not passed:
        raise SystemExit('Native GPU parity failed')


if __name__ == '__main__':
    main()
