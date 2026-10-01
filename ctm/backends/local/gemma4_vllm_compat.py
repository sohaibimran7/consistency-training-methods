"""Immutable inference-only views preserving Gemma 4's tied K/V LoRA.

HF full-attention layers with ``attention_k_eq_v`` have no V projection:
the adapted K output is used for both K and V. The pinned vLLM runtime
duplicates the base K weights into its V slice, but does not duplicate LoRA.
Never modify the HF checkpoint; duplicate only the tied adapter tensors in a
hash-bound inference view. This transformation is not a native parity gate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

MANIFEST = 'ctm_gemma4_tied_kv_vllm.json'
SCHEMA = 'gemma4-tied-kv-vllm-v1'
KEY = re.compile(r'\.layers\.(\d+)\.self_attn\.k_proj\.lora_([AB])\.weight$')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_config_path(model):
    path = Path(model) / 'config.json'
    if path.is_file():
        return path.resolve()
    name = str(model).lower().replace('_', '-')
    if 'gemma-4' not in name and 'gemma4' not in name:
        return None
    from transformers.utils import cached_file
    return Path(cached_file(model, 'config.json', local_files_only=True)).resolve()


def tied_layers(config):
    text = config.get('text_config', config)
    if not str(text.get('model_type', '')).startswith('gemma4'):
        return set()
    if text.get('attention_k_eq_v') is not True:
        return set()
    kinds = text.get('layer_types')
    count = text.get('num_hidden_layers')
    shared = text.get('num_kv_shared_layers', 0)
    if (not isinstance(kinds, list) or type(count) is not int or count != len(kinds)
            or type(shared) is not int or not 0 <= shared <= count
            or any(t not in ('sliding_attention', 'full_attention') for t in kinds)):
        raise ValueError('Invalid Gemma tied-KV configuration')
    return {i for i, kind in enumerate(kinds[:count-shared]) if kind == 'full_attention'}


def inference_view(adapter_dir, *, model, version):
    """Return raw path for unaffected adapters, otherwise a verified sibling.

    The creation is atomic and safe for concurrent rollout workers. Existing
    views must validate; corruption is never silently repaired or overwritten.
    """
    if type(version) is not int or version < 1:
        raise ValueError('Positive adapter version required')
    config_path = model_config_path(model)
    if config_path is None:
        return str(adapter_dir)
    config_bytes = config_path.read_bytes()
    layers = tied_layers(json.loads(config_bytes))
    if not layers:
        return str(adapter_dir)
    source = Path(adapter_dir).resolve()
    source_config = source / 'adapter_config.json'
    source_weights = source / 'adapter_model.safetensors'
    if not source_config.is_file() or not source_weights.is_file():
        raise ValueError('Gemma inference view requires a complete safetensors adapter')
    identities = {'model_config': hashlib.sha256(config_bytes).hexdigest(),
                  'raw_config': digest(source_config), 'raw_weights': digest(source_weights)}
    from safetensors.torch import load_file, save_file
    import torch
    raw = load_file(str(source_weights))
    translated = dict(raw)
    mapping = {}
    by_layer = {}
    for name, value in raw.items():
        match = KEY.search(name)
        if not match or int(match[1]) not in layers:
            continue
        destination = name.replace('.self_attn.k_proj.', '.self_attn.v_proj.')
        if destination in raw:
            raise ValueError('Independent V adapter conflicts with HF tied-KV semantics')
        if value.ndim != 2 or not torch.isfinite(value).all():
            raise ValueError('Invalid tied-KV LoRA tensor')
        mapping[name] = destination
        by_layer.setdefault(int(match[1]), set()).add(match[2])
        translated[destination] = value.clone()
    if not mapping:
        return str(adapter_dir)
    if any(kinds != {'A', 'B'} for kinds in by_layer.values()):
        raise ValueError('Incomplete tied-KV A/B pair')
    config = json.loads(source_config.read_text())
    if (config.get('rank_pattern') or config.get('alpha_pattern')
            or config.get('use_dora') or config.get('use_rslora')):
        raise ValueError('Unreviewed per-module scaling/DoRA/RSLoRA for tied-KV view')
    targets = config.get('target_modules')
    if not isinstance(targets, list) or any(not isinstance(t, str) for t in targets):
        raise ValueError('Explicit target-module list required for tied-KV compatibility')
    additions = []
    for layer in sorted(by_layer):
        matches = [t for t in targets if t.endswith(f'.layers.{layer}.self_attn.k_proj')]
        if len(matches) != 1:
            raise ValueError('Tied K target must be explicit and unique')
        additions.append(matches[0].replace('.self_attn.k_proj', '.self_attn.v_proj'))
    config['target_modules'] = [*targets, *additions]
    if identities != {'model_config': digest(config_path), 'raw_config': digest(source_config),
                      'raw_weights': digest(source_weights)}:
        raise ValueError('Original adapter/config changed during loading')
    destination = source.with_name(source.name + f'.gemma4-vllm-v{version}')

    def verify(folder):
        manifest = json.loads((folder / MANIFEST).read_text())
        expected = {'schema': SCHEMA, 'source': str(source), 'model_config_path': str(config_path),
                    'adapter_version': version, 'identities': identities, 'mapping': mapping}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError('Gemma inference-view source/version binding changed')
        if (digest(folder / 'adapter_config.json') != manifest['view_config_sha256']
                or digest(folder / 'adapter_model.safetensors') != manifest['view_weights_sha256']
                or json.loads((folder / 'adapter_config.json').read_text()) != config):
            raise ValueError('Gemma inference view changed')
        actual = load_file(str(folder / 'adapter_model.safetensors'))
        if set(actual) != set(translated) or any(not torch.equal(actual[k], v) for k, v in translated.items()):
            raise ValueError('Gemma inference-view tensor content changed')
        return str(folder)

    if destination.exists():
        return verify(destination)
    temporary = Path(tempfile.mkdtemp(prefix='.gemma4-view-', dir=source.parent))
    try:
        (temporary / 'adapter_config.json').write_text(json.dumps(config, indent=2))
        save_file(translated, str(temporary / 'adapter_model.safetensors'))
        manifest = {'schema': SCHEMA, 'source': str(source), 'model_config_path': str(config_path),
                    'adapter_version': version, 'identities': identities, 'mapping': mapping,
                    'view_config_sha256': digest(temporary / 'adapter_config.json'),
                    'view_weights_sha256': digest(temporary / 'adapter_model.safetensors')}
        (temporary / MANIFEST).write_text(json.dumps(manifest, indent=2))
        verify(temporary)
        # Rebind original bytes after materialization, before publication.
        if identities != {'model_config': digest(config_path), 'raw_config': digest(source_config),
                          'raw_weights': digest(source_weights)}:
            raise ValueError('Original adapter/config changed during publication')
        try:
            os.rename(temporary, destination)
        except OSError:
            if not destination.exists():
                raise
        return verify(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
