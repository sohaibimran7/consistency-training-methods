"""Generate a selective static-import closure; never copy files or submit jobs.

Existing canonical modules are review dependencies, not overwrite candidates.
Dynamic string factories and runtime assets are explicitly listed separately.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

SEEDS = ['experiments/gemma4_methods/'+n for n in (
    'train.py', 'evaluate.py', 'score.py', 'train.sbatch', 'evaluate.sbatch', 'score.sbatch',
    'deployment_env.sh', 'launch_guard.py', 'native_prompt_probe.py',
    'native_method_probe.py','native_hooks.py','native_checkpoint_restore.py',
    'preflight.sbatch','online_preflight.sbatch')]
SEEDS += ['tests/test_gemma_methods_plan.py', 'tests/test_gemma_methods_generation.py',
          'tests/test_gemma_methods_score.py', 'tests/test_gemma_selection_adapter.py',
          'tests/test_gemma_launch_guard.py','tests/test_gemma_performance.py']


def inventory(source, canonical, additional_seeds=()):
    pending = [*SEEDS,*additional_seeds]
    seen = set()
    rows = []
    external = set()
    def resolve(module):
        stem = module.replace('.', '/')
        for name in (stem+'.py', stem+'/__init__.py'):
            if (source/name).is_file():
                return name
        return None
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = source/name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        other = canonical/name
        present = other.is_file()
        rows.append({'path': name, 'sha256': digest,
                     'canonical_sha256': hashlib.sha256(other.read_bytes()).hexdigest() if present else None,
                     'action': 'review_existing_do_not_overwrite' if present else 'review_missing_dependency'})
        if path.suffix != '.py':
            continue
        package = '.'.join(Path(name).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    base = importlib.util.resolve_name('.'*node.level+base, package)
                modules = [base] + [base+'.'+a.name for a in node.names if a.name != '*']
            for module in modules:
                target = resolve(module)
                if target:
                    pending.append(target)
                elif module.split('.')[0] not in {'ctm', 'ctm_data', 'experiments', 'infra', 'scripts', 'tests'}:
                    external.add(module.split('.')[0])
    return {'schema': 'gemma-methods-static-migration-inventory-v1', 'source': str(source),
            'canonical': str(canonical), 'files': sorted(rows, key=lambda r:r['path']),
            'external_import_roots': sorted(external), 'ready_to_launch': False,
            'limitations': ['Static imports only; CLI/string factories and runtime provider plugins need live checks',
                            'No existing canonical file is authorized for wholesale replacement'],
            'runtime_assets': ['frozen7680QIDpool+manifest', 'pinnedGemma12Bprocessor+weights',
                'newrunapproval', 'newvalidationmanifest', 'checkpointandtargetfreshnamespaces'],
            'launcher_dependencies': ['explicit tracked GEMMA_RUNTIME_ENV in canonical deployment',
                'reviewed gemma-reviewed-runtime-v1 manifest and SHA256',
                'configured grading credential environment (never copy secrets)']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--canonical', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--additional-seed',action='append',default=[])
    a = p.parse_args()
    result = inventory(a.source.resolve(), a.canonical.resolve(),a.additional_seed)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'files': len(result['files']), 'output': str(a.output)}))
