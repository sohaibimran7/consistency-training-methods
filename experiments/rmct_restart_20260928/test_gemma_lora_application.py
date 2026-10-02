import pytest

from experiments.rmct_restart_20260928.gemma_lora_application import effect_verdict, tensor_verdict

K = 'base_model.model.model.language_model.layers.5.self_attn.'


def check(sub, **over):
    return {'key': K + sub, 'vllm_module': 'm', 'slice': 0, 'a_max_abs_err': 0.0,
            'b_max_abs_err_vs_scaled': 0.0, 'b_ref_max': 1.0, 'a_ref_abs_sum': 3.0, 'pad_max': 0.0, **over}


def inspection(checks, bad=(), extra=()):
    return {'checks': checks, 'bad': list(bad), 'extra_nonzero': list(extra), 'lora_dtype': 'torch.bfloat16'}


EXPECTED = {K + s for s in ('q_proj', 'k_proj', 'v_proj')}
RAW = {K + s + '.lora_A.weight' for s in ('q_proj', 'k_proj')}  # tied V comes from K


def test_complete_exact_application_passes_and_counts_tied_v():
    v = tensor_verdict(inspection([check('q_proj'), check('k_proj'), check('v_proj')]), EXPECTED, RAW)
    assert v['tensors_ok'] and v['n_tied_v_from_k'] == 1 and v['n_slice_checks'] == 3


@pytest.mark.parametrize('mutate', [
    lambda c: c.pop(),                                      # missing target
    lambda c: c.append(check('q_proj')),                    # duplicate
    lambda c: c[0].update(a_max_abs_err=1e-3),              # A not loaded exactly
    lambda c: c[0].update(b_max_abs_err_vs_scaled=0.1),     # wrong scaling
    lambda c: c[0].update(pad_max=0.5),                     # stale rank padding
    lambda c: c[0].update(a_ref_abs_sum=0.0),               # zero (ignored) adapter
])
def test_tensor_failures(mutate):
    checks = [check('q_proj'), check('k_proj'), check('v_proj')]
    mutate(checks)
    assert not tensor_verdict(inspection(checks), EXPECTED, RAW)['tensors_ok']


def test_unintended_or_unsupported_modules_fail():
    ok = [check('q_proj'), check('k_proj'), check('v_proj')]
    assert not tensor_verdict(inspection(ok, extra=['vision.x']), EXPECTED, RAW)['tensors_ok']
    assert not tensor_verdict(inspection(ok, bad=[['x', 'T']]), EXPECTED, RAW)['tensors_ok']


def test_effect_requires_nonzero_aligned_effect_and_ignores_repeat_noise():
    hf = [0.3, -0.2, 0.1]
    assert effect_verdict(hf, [0.25, -0.18, 0.12], [0.2, -0.2, 0.1])['effect_ok']
    assert not effect_verdict(hf, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])['effect_ok']   # adapter ignored
    assert not effect_verdict(hf, [-0.3, 0.2, -0.1], [-0.3, 0.2, -0.1])['effect_ok']  # wrong direction
