import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from ctm.backends.local.gemma4_vllm_compat import inference_view, tied_layers, MANIFEST


@pytest.fixture
def fixture(tmp_path):
    model = tmp_path / 'model'; model.mkdir()
    config = {'text_config': {'model_type': 'gemma4_unified_text', 'attention_k_eq_v': True,
        'layer_types': ['sliding_attention', 'full_attention', 'full_attention'],
        'num_hidden_layers': 3, 'num_kv_shared_layers': 1}}
    (model / 'config.json').write_text(json.dumps(config))
    raw = tmp_path / 'adapter'; raw.mkdir()
    target = 'language_model.model.layers.1.self_attn.k_proj'
    (raw / 'adapter_config.json').write_text(json.dumps({'target_modules': [target], 'r': 2, 'lora_alpha': 4}))
    tensors = {'base_model.model.' + target + '.lora_A.weight': torch.arange(6).reshape(2,3).float(),
               'base_model.model.' + target + '.lora_B.weight': torch.arange(8).reshape(4,2).float()}
    save_file(tensors, str(raw / 'adapter_model.safetensors'))
    return model, raw, tensors


def test_exact_tied_copy_and_original_unchanged(fixture):
    model, raw, tensors = fixture
    before = {p.name: p.read_bytes() for p in raw.iterdir()}
    view = Path(inference_view(raw, model=str(model), version=1))
    actual = load_file(str(view / 'adapter_model.safetensors'))
    assert len(actual) == 4
    for key, value in tensors.items():
        assert torch.equal(actual[key], value)
        assert torch.equal(actual[key.replace('.k_proj.', '.v_proj.')], value)
    assert before == {p.name: p.read_bytes() for p in raw.iterdir()}
    manifest = json.loads((view / MANIFEST).read_text())
    assert manifest['adapter_version'] == 1 and len(manifest['mapping']) == 2
    assert inference_view(raw, model=str(model), version=1) == str(view)


def test_concurrent_publication(fixture):
    model, raw, _ = fixture
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: inference_view(raw, model=str(model), version=1), range(4)))
    assert len(set(results)) == 1
    assert not list(raw.parent.glob('.gemma4-view-*'))


def test_corrupt_view_is_not_repaired(fixture):
    model, raw, _ = fixture
    view = Path(inference_view(raw, model=str(model), version=1))
    weights = view / 'adapter_model.safetensors'
    weights.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='view changed'):
        inference_view(raw, model=str(model), version=1)
    assert weights.read_bytes() == b'corrupt'


def test_raw_mutation_rejects_existing_view(fixture):
    model, raw, tensors = fixture
    inference_view(raw, model=str(model), version=1)
    save_file({k:v+1 for k,v in tensors.items()}, str(raw / 'adapter_model.safetensors'))
    with pytest.raises(ValueError, match='binding changed'):
        inference_view(raw, model=str(model), version=1)


def test_no_k_targets_returns_original(fixture):
    model, raw, tensors = fixture
    save_file({k.replace('.k_proj.', '.q_proj.'):v for k,v in tensors.items()},str(raw/'adapter_model.safetensors'))
    assert inference_view(str(raw), model=str(model), version=1) == str(raw)


def test_independent_v_conflict_fails(fixture):
    model, raw, tensors = fixture
    tensors.update({k.replace('.k_proj.', '.v_proj.'):v.clone() for k,v in list(tensors.items())})
    save_file(tensors,str(raw/'adapter_model.safetensors'))
    with pytest.raises(ValueError,match='Independent V'):
        inference_view(raw,model=str(model),version=1)


def test_missing_pair_fails(fixture):
    model,raw,tensors=fixture
    save_file({k:v for k,v in tensors.items() if 'lora_A' in k},str(raw/'adapter_model.safetensors'))
    with pytest.raises(ValueError,match='Incomplete'):
        inference_view(raw,model=str(model),version=1)


def test_only_nonshared_full_attention_layers(fixture):
    model,_,_=fixture
    assert tied_layers(json.loads((model/'config.json').read_text())) == {1}


@pytest.mark.parametrize('version',[0,-1,True,1.5])
def test_invalid_version(fixture,version):
    model,raw,_=fixture
    with pytest.raises(ValueError,match='Positive'):
        inference_view(raw,model=str(model),version=version)


def test_unaffected_model_does_not_read_adapter():
    assert inference_view('missing-adapter',model='unrelated-model',version=1)=='missing-adapter'


def test_nonexplicit_target_refused(fixture):
    model,raw,_=fixture
    (raw/'adapter_config.json').write_text(json.dumps({'target_modules':['k_proj']}))
    with pytest.raises(ValueError,match='explicit and unique'):
        inference_view(raw,model=str(model),version=1)


def test_sampler_uses_view_and_unique_versions(fixture):
    from types import SimpleNamespace
    from ctm.backends.local.vllm_sampler import VLLMSampler
    model,raw,_=fixture
    class Request:
        def __init__(self,name,version,path):self.path=path;self.version=version
    sampler=VLLMSampler(str(model),engine=object(),api=SimpleNamespace(LoRARequest=Request))
    sampler.advance_policy(str(raw),version=1)
    first=sampler._policy_lora_request()
    assert first.path!=str(raw) and Path(first.path,MANIFEST).is_file()
    sampler.advance_policy(str(raw),version=2)
    second=sampler._policy_lora_request()
    assert second.version==2 and second.path!=first.path
    with pytest.raises(ValueError,match='must increase'):
        sampler.advance_policy(str(raw),version=2)
