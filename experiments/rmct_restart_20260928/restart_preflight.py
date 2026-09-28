"""Read-only CPU source, model and thinking checks; never authorizes training.

Run inside the scheduled deployment before native GPU parity. The immutable
receipt is evidence of these checks only, not a replacement for GPU gates.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

REVISIONS = {'qwen': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
             'gemma': '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'}
# Original cache content-addressed blob identities independently recorded by
# the checkpoint owner. Compare actual bytes, not merely snapshot directory names.
QWEN_SHARDS = {
    'db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e',
    '31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330',
    '7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec',
    'b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6',
}


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.absolute()), 'resolved': str(path.resolve()),
            'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def thinking_probe(renderer, tokenizer, family):
    messages = [{'role': 'user', 'content': 'What is 2 + 2?'}]
    if family == 'gemma':
        from ctm.backends.gemma_thinking import attest_thinking
        probe = attest_thinking(tokenizer)
        expected = probe['prompt_token_ids']['true']
        messages = probe['probe_messages']
    else:
        expected = tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=True)
        disabled = tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if expected == disabled:
            raise ValueError('Thinking toggle has no observable effect')
        text = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        if not text.rstrip().endswith('<think>'):
            raise ValueError('Qwen thinking-on generation boundary missing')
        if tokenizer.encode(text, add_special_tokens=False) != expected:
            raise ValueError('Native prompt text/token mismatch')
        probe = {'enable_thinking': True, 'prompt_token_ids':
                 {'true': expected, 'false': disabled}, 'probe_messages': messages}
    if renderer.build_generation_prompt(messages).to_ints() != expected:
        raise ValueError('Production renderer is not native thinking-on')
    return probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'model', 'validation-manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--family', choices=REVISIONS, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    def git(*argv):
        return subprocess.check_output(['git', '-C', str(root), *argv], text=True).strip()
    if git('rev-parse', 'HEAD') != args.source_commit or git('status', '--porcelain'):
        raise RuntimeError('Clean exact committed deployment required')
    if args.model.name != REVISIONS[args.family] or (args.model / 'adapter_config.json').exists():
        raise RuntimeError('Original pinned base snapshot required')
    from ctm.backends import renderers
    import ctm.training.rl as rl
    from ctm_data.adapters.mcq_bias import parser_compat
    for module in (renderers, rl, parser_compat):
        relative = Path(module.__file__).resolve().relative_to(root)
        git('ls-files', '--error-unmatch', str(relative))
    tracked = git('ls-files', '-z').split('\0')
    source_files = [identity(root / name) for name in tracked if name and
                    name.startswith(('ctm/', 'ctm_data/', 'scripts/', 'experiments/rmct_restart_20260928/'))]
    indices = list(args.model.glob('*.safetensors.index.json'))
    if len(indices) != 1:
        raise RuntimeError('Exactly one safetensors shard index required')
    index = json.loads(indices[0].read_text())
    shards = sorted(set(index['weight_map'].values()))
    if not shards or any(Path(s).name != s for s in shards):
        raise RuntimeError('Invalid shard index')
    model_files = [identity(args.model / shard) for shard in shards]
    if args.family == 'qwen' and {f['sha256'] for f in model_files} != QWEN_SHARDS:
        raise RuntimeError('Original Qwen weight content mismatch')
    model_files += [identity(path) for path in sorted(args.model.iterdir())
                    if path.is_file() and path.suffix in ('.json', '.jinja')]
    renderer, tokenizer = renderers.get_renderer_and_tokenizer(str(args.model), source='hf')
    receipt = {'schema': 'rmct-restart-cpu-v1', 'status': 'cpu_checks_passed',
        'optimizer_work_authorized': False, 'source_commit': args.source_commit,
        'source_root': str(root), 'python': sys.executable, 'cwd': str(Path.cwd()),
        'sources': source_files, 'model_files': model_files,
        'validation_manifest': identity(args.validation_manifest),
        'thinking': thinking_probe(renderer, tokenizer, args.family),
        'dependencies': {name: importlib.metadata.version(name) for name in
                         ('torch', 'transformers', 'vllm', 'peft')},
        'remaining_gates': ['native_gpu_parity', 'integrated_regressions',
                            'validation_semantics', 'incorporation_and_launch_clearance']
                           + (['gemma_independent_weight_identity'] if args.family == 'gemma' else [])}
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
