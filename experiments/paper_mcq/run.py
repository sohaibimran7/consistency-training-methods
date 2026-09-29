"""Offline, isolated replay of hash-pinned historical MCQ publication recipes.

All recipe code is bundled. Original input files are read-only. Statistical
functions, seeds, 10k bootstraps and 108k permutation settings are preserved.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BUNDLE = Path(__file__).resolve().parent
ENDPOINTS = {
    'switch': ('towards-bias-switch-standard/build.py', 'towards-bias-switch-standard/chart-rows.json'),
    'conditional': ('conditional-verbalisation-20260918/paper.py', 'conditional-verbalisation-20260918/main-paper-style-108000/chart-rows.json'),
    'verbalisation': ('bias-verbalisation-standard/build.py', 'bias-verbalisation-standard/chart-rows.json'),
    'accuracy': ('biased-accuracy-20260918/build.py', 'biased-accuracy-20260918/paper-rows.json'),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected exactly one historical anchor: {old!r}')
    return text.replace(old, new, 1)

def verify_code():
    lock = json.loads((BUNDLE/'source-lock.json').read_text())
    for relative, item in lock.items():
        if sha(BUNDLE/relative) != item['sha256']:
            raise ValueError(f'Bundled source changed: {relative}')
    return lock

def fresh_output(path, inputs):
    path = path.resolve()
    if path.exists():
        raise ValueError(f'Output must not exist: {path}')
    for source in inputs:
        source = source.resolve()
        if source == path or path in source.parents or source.is_dir() and source in path.parents:
            raise ValueError(f'Output overlaps input: {source}')
    path.mkdir(parents=True)
    return path

def stage_vendor(out):
    vendor = out/'vendor'
    shutil.copytree(BUNDLE/'vendor', vendor)
    for package in ['ctm_data','ctm_data/adapters','ctm_data/adapters/mcq_bias',
                    'experiments','experiments/rmct_two_bias_eval','mcq_bias']:
        (vendor/package/'__init__.py').write_text('"""Isolated historical replay namespace."""\n')
    return vendor

def environment(vendor, out):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(vendor), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MPLCONFIGDIR=str(out/'matplotlib-cache'), CTM_CONDITIONAL_PERMUTATIONS='108000')
    return env

def run_child(command, out, env, logname):
    with (out/logname).open('w') as log:
        result = subprocess.run(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Replay failed ({result.returncode}); inspect {out/logname}')

def figures(args):
    lock = verify_code()
    inputs = [args.samples, args.template] + ([args.sources] if args.sources else [])
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    data = json.loads(args.samples.read_text())
    if set(data) != {'base','act','attct','mlpct','bct','opct','rmct352'}:
        raise ValueError('Expected seven historical methods, including RMCT352')
    out = fresh_output(args.output, inputs+[BUNDLE])
    vendor = stage_vendor(out)
    stage = out/'figures'
    shutil.copytree(BUNDLE/'recipes', stage)
    prior = stage/'paper-behavioural-plots-20260917'
    shutil.copy2(args.samples, prior/'samples.json')
    if args.sources:
        shutil.copy2(args.sources, prior/'sources.json')
    else:
        write(prior/'sources.json', {})
    template = stage/'methods-verbalisation-all-seven-20260916/bias_acknowledged-vs-base/chart-spec.json'
    template.parent.mkdir(parents=True)
    shutil.copy2(args.template, template)
    # Only this module's SEEN_BIASES is needed by the saved-row endpoints.
    # The original raw-log extract() is intentionally not an exposed command.
    helper = prior/'build.py'
    text = replace_once(helper.read_text(),
        "REPO=Path('/Users/work/.codex/worktrees/d6d6/consistency-training-methods')",
        f'REPO=Path({str(vendor)!r})')
    text = replace_once(text, 'from experiments.rmct_two_bias_eval import checkpoint_publication as standard',
        'from experiments.rmct_two_bias_eval import contract as standard')
    helper.write_text(text)
    # Import-only probe loads statistical definitions without their driver.
    probe = """import ast, importlib.util
from pathlib import Path
helper=Path('figures/conditional-verbalisation-20260918/split.py').resolve()
tree=ast.parse(helper.read_text()); nodes=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rows' for t in node.targets): break
    nodes.append(node)
scope={'__file__':str(helper)}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(helper),'exec'),scope)
from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from ctm_data.adapters.mcq_bias.parser_compat import install_extended_answer_parser
install_extended_answer_parser()
from mcq_bias.parsers import parse_answer
assert parse_answer('<think>The answer is A.</think>The answer is B.')=='B'
assert scope['b'].SEEN=={'wrong_argument','suggested_answer'}
print('Imports, helper definitions, parser and seen-bias contract PASS; no estimates computed')
"""
    env = environment(vendor,out)
    run_child([sys.executable,'-c',probe],out,env,'import-check.log')
    receipt = dict(mode='import-only' if args.check_only else 'full-statistical-replay',
                   input_hashes={str(p.resolve()):sha(p) for p in inputs}, code=lock,
                   permutations=108000, bootstrap=10000, model_calls=0, grader_calls=0,
                   completed=[], comparisons={}, transformations=['REPO to isolated vendor',
                   'unused checkpoint_publication import narrowed to identical SEEN_BIASES contract'])
    write(out/'receipt.json',receipt)
    if args.check_only:
        print(out/'receipt.json')
        return
    for endpoint in args.only or ENDPOINTS:
        script, rows = ENDPOINTS[endpoint]
        run_child([sys.executable,str(stage/script)],out,env,endpoint+'.log')
        receipt['completed'].append(endpoint)
        if args.reference:
            expected = json.loads((args.reference/rows).read_text())
            actual = json.loads((stage/rows).read_text())
            if expected != actual:
                receipt['comparisons'][endpoint]='MISMATCH'
                write(out/'receipt.json',receipt)
                raise ValueError(f'Saved-stat mismatch: {endpoint}')
            receipt['comparisons'][endpoint]='EXACT JSON EQUALITY'
        write(out/'receipt.json',receipt)
    print(out/'receipt.json')

def merge(args):
    lock = verify_code()
    inputs = [args.fixed, args.recovery]
    for path in [args.fixed/'samples.json',args.recovery/'source-hashes.json',
                 args.recovery/'merge-audit.json',args.recovery/'missing-bct.json',args.recovery/'build.py']:
        if not path.is_file():
            raise FileNotFoundError(path)
    out = fresh_output(args.output,inputs+[BUNDLE])
    vendor = stage_vendor(out)
    text = (BUNDLE/'recipes/merge.py').read_text()
    text = replace_once(text, "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))",f'sys.path.insert(0, {str(vendor)!r})')
    text = replace_once(text, 'R = Path(__file__).resolve().parents[1]',f'R = Path({str(vendor)!r})')
    text = replace_once(text, "FIXED = R / 'artifacts/parser-fixed-20260923'",f'FIXED = Path({str(args.fixed.resolve())!r})')
    text = replace_once(text, "REC = Path('/Users/work/.codex/worktrees/d6d6/consistency-training-methods/artifacts/recovered-publication-20260922')",f'REC = Path({str(args.recovery.resolve())!r})')
    text = replace_once(text, "OUT = R / 'artifacts/parser-fixed-64k-20260925'",f'OUT = Path({str(out / "merged")!r})')
    # Retain the original hash verification when moving an input bundle.
    # Mapping is relative to /inputs/, never basename-only; collisions fail.
    text = replace_once(text, "expected_sources = read(REC / 'source-hashes.json')", """expected_sources = read(REC / 'source-hashes.json')
remapped = {}
for original, digest in expected_sources.items():
    prefix, separator, suffix = original.rpartition('/inputs/')
    assert separator and suffix and '..' not in Path(suffix).parts, original
    moved = str(REC / 'inputs' / suffix)
    assert moved not in remapped, moved
    remapped[moved] = digest
expected_sources = remapped""")
    script = out/'merge.py'
    script.write_text(text)
    env = environment(vendor,out)
    run_child([sys.executable,str(script)],out,env,'merge.log')
    receipt = dict(mode='raw-log-merge', code=lock, fixed_samples_sha256=sha(args.fixed/'samples.json'),
                   recovery_source_manifest_sha256=sha(args.recovery/'source-hashes.json'),
                   model_calls=0, grader_calls=0, original_inputs_modified=False,
                   source_paths_relocated=True)
    if args.reference:
        actual=json.loads((out/'merged/samples.json').read_text())
        expected=json.loads((args.reference/'samples.json').read_text())
        def semantic(data):
            return {m:[{k:v for k,v in row.items() if k not in {'biased_source','clean_source'}} for row in rows] for m,rows in data.items()}
        receipt['comparison']='EXACT EXCLUDING RELOCATED SOURCE PATHS' if semantic(actual)==semantic(expected) else 'MISMATCH'
        write(out/'receipt.json',receipt)
        if receipt['comparison']=='MISMATCH':
            raise ValueError('Merged sample mismatch; inspect receipt and outputs')
    write(out/'receipt.json',receipt)
    print(out/'receipt.json')

def reconcile(args):
    """Reparse archived clean/biased EvalLogs without touching saved scores."""
    lock=verify_code()
    inputs=[args.legacy_artifacts, BUNDLE]
    if args.path_map:
        inputs.append(args.path_map)
    mapping=json.loads(args.path_map.read_text()) if args.path_map else {}
    if not isinstance(mapping,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in mapping.items()):
        raise ValueError('Path map must be a JSON object of original prefix to relocated prefix')
    out=fresh_output(args.output,inputs)
    vendor=stage_vendor(out)
    text=(BUNDLE/'recipes/reconcile.py').read_text()
    text=replace_once(text,'sys.path.insert(0,str(Path(__file__).resolve().parents[1]))',f'sys.path.insert(0,{str(vendor)!r})')
    text=replace_once(text,'R=Path(__file__).resolve().parents[1]',f'R=Path({str(vendor)!r})')
    text=replace_once(text,"OLD=Path('/Users/work/.codex/worktrees/d6d6/consistency-training-methods/artifacts')",f'OLD=Path({str(args.legacy_artifacts.resolve())!r})')
    text=replace_once(text,"OUT=R/'artifacts/parser-fixed-20260923';OUT.mkdir(exist_ok=True)",f'OUT=Path({str(out/"reconciled")!r});OUT.mkdir(exist_ok=True)')
    if mapping:
        relocation=f'''PATH_MAP = {mapping!r}
def relocate(original):
 p=Path(original)
 for old,new in sorted(PATH_MAP.items(),key=lambda item:len(item[0]),reverse=True):
  try: relative=p.relative_to(old)
  except ValueError: continue
  return Path(new)/relative
 return p
'''
        text=replace_once(text,'def load(entry):',relocation+'def load(entry):')
        text=replace_once(text,"p=Path(entry['path']);h=hashlib.sha256(p.read_bytes()).hexdigest()","p=relocate(entry['path']);h=hashlib.sha256(p.read_bytes()).hexdigest()")
    script=out/'reconcile.py'
    script.write_text(text)
    run_child([sys.executable,str(script)],out,environment(vendor,out),'reconcile.log')
    receipt=dict(mode='raw-log-reconciliation',code=lock,path_map=mapping,model_calls=0,grader_calls=0,
                 original_inputs_modified=False,source_hash_checks_preserved=True)
    if args.reference:
        receipt['comparison']='EXACT JSON EQUALITY' if json.loads((out/'reconciled/samples.json').read_text())==json.loads((args.reference/'samples.json').read_text()) else 'MISMATCH'
        write(out/'receipt.json',receipt)
        if receipt['comparison']=='MISMATCH':
            raise ValueError('Reconciled sample mismatch')
    write(out/'receipt.json',receipt)
    print(out/'receipt.json')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    f=sub.add_parser('figures')
    f.add_argument('--samples',type=Path,required=True)
    f.add_argument('--template',type=Path,required=True)
    f.add_argument('--sources',type=Path)
    f.add_argument('--output',type=Path,required=True)
    f.add_argument('--reference',type=Path)
    f.add_argument('--check-only',action='store_true')
    f.add_argument('--only',nargs='+',choices=list(ENDPOINTS))
    f.set_defaults(func=figures)
    m=sub.add_parser('merge')
    m.add_argument('--fixed',type=Path,required=True)
    m.add_argument('--recovery',type=Path,required=True)
    m.add_argument('--output',type=Path,required=True)
    m.add_argument('--reference',type=Path)
    m.set_defaults(func=merge)
    r=sub.add_parser('reconcile')
    r.add_argument('--legacy-artifacts',type=Path,required=True)
    r.add_argument('--path-map',type=Path)
    r.add_argument('--output',type=Path,required=True)
    r.add_argument('--reference',type=Path)
    r.set_defaults(func=reconcile)
    args=parser.parse_args()
    args.func(args)

if __name__=='__main__':
    main()
