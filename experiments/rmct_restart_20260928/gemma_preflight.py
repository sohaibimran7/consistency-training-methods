"""CPU provenance/prompt gate; GPU regression and parity remain separate gates.

Run only from the incorporated clean source commit. Never loads model weights,
generates responses, resumes a checkpoint or submits a job.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

from experiments.rmct_restart_20260928.gemma_config import fresh_config, REVISION


def identity(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--validation-manifest', type=Path, required=True)
    p.add_argument('--approval-reference', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = a.source_root.resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != a.source_commit or git('status', '--porcelain'):
        raise RuntimeError('Preflight requires the clean incorporated source commit')
    if a.model.name != REVISION:
        raise RuntimeError('Pinned original Gemma snapshot required')
    if (a.model / 'adapter_config.json').exists():
        raise RuntimeError('Adapter initialization forbidden')
    from transformers import AutoProcessor
    from ctm.evals.local_model import Gemma4UnifiedTextProcessor
    from ctm.backends import gemma_thinking, renderers
    import ctm.evals.local_model as local_model
    sources = []
    for path in [__file__, Path(__file__).with_name('gemma_config.py'),
                 gemma_thinking.__file__, renderers.__file__, local_model.__file__]:
        resolved = Path(path).resolve()
        rel = resolved.relative_to(root)
        git('ls-files', '--error-unmatch', str(rel))
        sources.append(identity(resolved))
    probe = gemma_thinking.attest_thinking(Gemma4UnifiedTextProcessor(
        AutoProcessor.from_pretrained(str(a.model), local_files_only=True)))
    renderer, _ = renderers.get_renderer_and_tokenizer(str(a.model), source='hf')
    actual = renderer.build_generation_prompt(probe['probe_messages']).to_ints()
    if actual != probe['prompt_token_ids']['true']:
        raise RuntimeError('Production renderer differs from native thinking-on tokens')
    manifest = identity(a.validation_manifest)
    config = fresh_config(source_commit=a.source_commit, approval_reference=a.approval_reference,
                          validation_manifest_sha256=manifest['sha256'])
    receipt = {'status': 'cpu_prompt_provenance_passed', 'optimizer_work_authorized': False,
               'remaining_gates': ['integrated_regression', 'fresh_gpu_parity',
                                   'validation_manifest_semantic_verification', 'coordinator_clearance'],
               'config': config, 'sources': sources, 'validation_manifest': manifest,
               'model_files': [identity(a.model / n) for n in
                               ('config.json', 'generation_config.json', 'chat_template.jinja')],
               'dependencies': {n: importlib.metadata.version(n) for n in
                                ('torch', 'transformers', 'vllm', 'peft')},
               'thinking': probe}
    with a.output.open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
