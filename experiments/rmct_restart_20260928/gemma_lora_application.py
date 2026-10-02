"""Native proof that vLLM applies every intended Gemma LoRA tensor.

User decision 2026-10-02: the Gemma launch gate establishes adapter
*application* (vLLM loads and uses the adapter rather than ignoring it). It is
not an HF-vLLM numerical parity claim; numerical metrics remain recorded as
information. Known non-blocking residuals: vLLM applies the publisher
generation_config ``suppress_tokens`` (258883, 258882) to logits while HF
forward does not; vLLM stores LoRA weights in bf16 versus PEFT fp32; default
split-K LoRA accumulation is batch-composition dependent; bf16 logit rounding.
"""
import re

PACK = {'qkv_proj': ('q_proj', 'k_proj', 'v_proj'), 'gate_up_proj': ('gate_proj', 'up_proj'),
        'o_proj': ('o_proj',), 'down_proj': ('down_proj',)}
MODULE = re.compile(r'language_model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)$')
ACCEPTANCE = ('adapter application (user decision 2026-10-02): all intended LoRA tensors loaded with '
              'correct scaling/tied K->V, no unintended nonzero modules, nonzero adapter effect aligned '
              'with HF; NOT numerical HF-vLLM parity')
MIN_EFFECT_COSINE = 0.9  # retained component of the original gate


class LoRAInspectionExtension:
    """vLLM worker extension; compares the active slot with a reference file on the worker."""

    def verify_active_lora(self, adapter_id, reference_path, scaling):
        from safetensors.torch import load_file
        weights = load_file(reference_path)
        manager = self.model_runner.lora_manager._adapter_manager
        slot = manager.lora_index_to_id.index(adapter_id)
        checks, bad, extra, dtype = [], [], [], None
        for name, module in manager.modules.items():
            a_stack = getattr(module, 'lora_a_stacked', None)
            b_stack = getattr(module, 'lora_b_stacked', None)
            if a_stack is None or b_stack is None:
                bad.append([name, type(module).__name__])
                continue
            dtype = str(a_stack[0].dtype)
            match = MODULE.search(name)
            if not match or match.group(3) not in PACK:
                if any(a_stack[i][slot].abs().sum().item() or b_stack[i][slot].abs().sum().item()
                       for i in range(len(a_stack))):
                    extra.append(name)
                continue
            layer, block, kind = match.groups()
            if len(a_stack) != len(PACK[kind]):
                bad.append([name, f'{len(a_stack)} slices'])
                continue
            for i, sub in enumerate(PACK[kind]):
                key = f'base_model.model.model.language_model.layers.{layer}.{block}.{sub}'
                a_ref = weights[key + '.lora_A.weight'].to(a_stack[i].device)
                b_ref = (weights[key + '.lora_B.weight'].float() * scaling).to(b_stack[i].device)
                a = a_stack[i][slot, 0].float()
                b = b_stack[i][slot, 0].float()
                rank = a_ref.shape[0]
                checks.append({
                    'key': key, 'vllm_module': name, 'slice': i,
                    'a_max_abs_err': (a[:rank] - a_ref.to(a_stack[i].dtype).float()).abs().max().item(),
                    'b_max_abs_err_vs_scaled': (b[:, :rank] - b_ref.to(b_stack[i].dtype).float()).abs().max().item(),
                    'b_ref_max': b_ref.abs().max().item(), 'a_ref_abs_sum': a_ref.abs().sum().item(),
                    'pad_max': max(a[rank:].abs().max().item() if a.shape[0] > rank else 0.0,
                                   b[:, rank:].abs().max().item() if b.shape[1] > rank else 0.0)})
        return {'slot': slot, 'lora_dtype': dtype, 'checks': checks, 'bad': bad, 'extra_nonzero': extra}


def tensor_verdict(inspection, expected_targets, raw_keys):
    """Every expected target present exactly once and bf16-exact (B within one bf16 ulp scale)."""
    checks = inspection['checks']
    for check in checks:
        check['tied_v_from_k'] = check['key'] + '.lora_A.weight' not in raw_keys
        check['ok'] = (check['a_max_abs_err'] == 0.0 and check['pad_max'] == 0.0 and check['a_ref_abs_sum'] > 0
                       and check['b_max_abs_err_vs_scaled'] <= 2 ** -7 * check['b_ref_max'])
    covered = [c['key'] for c in checks]
    ok = (bool(checks) and all(c['ok'] for c in checks) and len(covered) == len(set(covered))
          and set(covered) == set(expected_targets) and not inspection['bad'] and not inspection['extra_nonzero'])
    return {'tensors_ok': ok, 'n_slice_checks': len(checks), 'n_expected_targets': len(set(expected_targets)),
            'n_tied_v_from_k': sum(c['tied_v_from_k'] for c in checks),
            'missing_targets': sorted(set(expected_targets) - set(covered)),
            'bad_modules': inspection['bad'], 'extra_nonzero_modules': inspection['extra_nonzero'],
            'failing_checks': [c for c in checks if not c['ok']][:20], 'vllm_lora_dtype': inspection['lora_dtype']}


def effect_verdict(hf_effect, vllm_effect, vllm_repeat_effect):
    import numpy as np
    hf, v1, v2 = (np.asarray(x, dtype=float) for x in (hf_effect, vllm_effect, vllm_repeat_effect))
    def cosine(x, y):
        n = np.linalg.norm(x) * np.linalg.norm(y)
        return float(x @ y / n) if n else 0.0
    out = {'vllm_effect_norm': float(np.linalg.norm(v1)), 'hf_effect_norm': float(np.linalg.norm(hf)),
           'cosine_vs_hf': cosine(v1, hf), 'informational_repeat_cosine': cosine(v1, v2),
           'informational_repeat_max_abs_diff': float(np.max(np.abs(v1 - v2)))}
    out['effect_ok'] = bool(np.all(np.isfinite(v1)) and out['vllm_effect_norm'] > 0
                            and out['cosine_vs_hf'] >= MIN_EFFECT_COSINE)
    return out
