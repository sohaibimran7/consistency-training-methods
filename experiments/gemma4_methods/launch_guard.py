"""Validate explicit runtime pins and native import origins; never submit jobs."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def check_source(repository, commit):
    repository = Path(repository).resolve()
    def git(*args):
        return subprocess.check_output(['git','-C',str(repository),*args],text=True).strip()
    if git('rev-parse','HEAD') != commit or git('status','--porcelain','--untracked-files=all'):
        raise ValueError('Exact clean canonical deployment required')
    subprocess.run(['git','-C',str(repository),'merge-base','--is-ancestor',
                    '45f27c24855c82f3dc81019bd246a6f7641a13ec',commit],check=True)
    return repository


def verify(repository, commit, manifest, digest, *, verifier_factory=None, profile='generation'):
    repository = check_source(repository,commit)
    path = Path(manifest)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Reviewed runtime manifest bytes changed')
    pins = json.loads(raw)
    if pins.get('schema') != 'gemma-reviewed-runtime-v1':
        raise ValueError('Unknown reviewed runtime manifest')
    if Path(sys.executable).resolve() != Path(pins['python']).resolve():
        raise ValueError('Runtime interpreter differs from pinned interpreter')
    if sys.version != pins['python_version']:
        raise ValueError('Runtime Python build changed')
    if profile not in ('generation','grading'):
        raise ValueError('Unknown runtime profile')
    packages = {'torch','transformers','vllm','inspect_ai'} if profile=='generation' else {'inspect_ai','openai'}
    if pins.get('profile','generation') != profile:
        raise ValueError('Runtime profile differs from reviewed pins')
    if set(pins['packages']) != packages:
        raise ValueError('Complete runtime-profile package pins required')
    for package in sorted(packages):
        pin = pins['packages'][package]
        module = importlib.import_module(package)
        if (importlib.metadata.version(package) != pin['version']
                or Path(module.__file__).resolve() != Path(pin['module_file']).resolve()):
            raise ValueError('Runtime package identity differs: '+package)
    modules = ['ctm.backends.local.engine','ctm.backends.local.rollout_workers',
               'ctm.backends.renderers','experiments.gemma4_methods.train',
               'experiments.gemma4_methods.evaluate',
               'experiments.rmct_restart_20260928.validation_selection']
    if profile=='grading':
        modules=['experiments.gemma4_methods.score','experiments.gemma4_methods.score_helpers',
                 'experiments.gemma4_methods.evaluate',
                 'experiments.rmct_restart_20260928.validation_selection']
    if verifier_factory:
        factory_module, function = verifier_factory.split(':',1)
        modules.append(factory_module)
    for name in modules:
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(repository):
            raise ValueError('Repository import shadowed by overlay: '+name)
        subprocess.run(['git','-C',str(repository),'ls-files','--error-unmatch','--',
                        str(origin.relative_to(repository))],check=True,stdout=subprocess.DEVNULL)
    if verifier_factory and not callable(getattr(importlib.import_module(factory_module),function,None)):
        raise ValueError('Native verifier factory unavailable')
    return {'source_commit':commit,'runtime_manifest_sha256':digest,'checked_modules':modules}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--verifier-factory')
    p.add_argument('--profile',choices=['generation','grading'],default='generation')
    args=p.parse_args()
    print(json.dumps(verify(args.repository,args.commit,args.manifest,args.manifest_sha256,
                            verifier_factory=args.verifier_factory,profile=args.profile)))
