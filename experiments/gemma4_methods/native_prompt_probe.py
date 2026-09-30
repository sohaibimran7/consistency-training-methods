"""Reproduce40real training prompt checks; no weights, generation or optimizer.

Consumes the shared restart_preflight receipt, never promotes the older
gemma_preflight draft or fabricates successful GPU evidence.
"""
import argparse
import json
import os
import importlib.metadata
import sys
from pathlib import Path

from experiments.gemma4_methods.selection_adapter import file_identity, native_prompt
from experiments.gemma4_methods.launch_guard import check_source
from experiments.gemma4_methods.reference import plan


def probe(processor, qids):
    records=[]
    for method in plan.METHODS:
        for pair in plan.paired_rows(qids,method=method):
            for side in ('reference_messages','variant_messages'):
                messages=pair[side]
                enabled=native_prompt(processor,messages)
                disabled=processor.apply_chat_template(messages,tokenize=True,
                    return_dict=False,add_generation_prompt=True,enable_thinking=False)
                if hasattr(disabled,'tolist'):
                    disabled=disabled.tolist()
                if disabled and isinstance(disabled[0],list):
                    disabled=disabled[0]
                if enabled==disabled:
                    raise ValueError('Real prompt thinking toggle has no token effect')
                records.append({'method':method,'dataset':pair['source_dataset'],
                    'question_id':pair['question_id'],'bias':pair['bias'],'side':side,
                    'messages':messages,'enabled_tokens':enabled,'disabled_tokens':disabled})
    if len(records)!=40:
        raise ValueError('Expected40native prompt cases')
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repository','commit','model','data-root','cpu-receipt','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    root=check_source(a.repository,a.commit)
    receipt_path=Path(a.cpu_receipt)
    cpu=json.loads(receipt_path.read_text())
    if (cpu.get('schema'),cpu.get('status'),cpu.get('source_commit')) != (
            'rmct-restart-cpu-v1','cpu_checks_passed',a.commit):
        raise ValueError('Shared integrated restart_preflight receipt required')
    if Path(cpu['source_root']).resolve()!=root or cpu['optimizer_work_authorized'] is not False:
        raise ValueError('CPU preflight source/scope differs')
    if Path(cpu['python']).resolve()!=Path(sys.executable).resolve():
        raise ValueError('Probe interpreter differs from CPU receipt')
    for name,version in cpu['dependencies'].items():
        if importlib.metadata.version(name)!=version:
            raise ValueError('Probe dependency differs from CPU receipt: '+name)
    for file in (__file__,plan.__file__):
        relative=Path(file).resolve().relative_to(root)
        import subprocess
        subprocess.run(['git','-C',str(root),'ls-files','--error-unmatch','--',str(relative)],
                       check=True,stdout=subprocess.DEVNULL)
    model=Path(a.model)
    if model.name!='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7':
        raise ValueError('Pinned original Gemma required')
    for record in cpu['sources']+cpu['model_files']:
        actual=file_identity(record['path'])
        if any(actual[key]!=record[key] for key in ('sha256','bytes')):
            raise ValueError('CPU-attested source/model bytes changed')
    weights=[r for r in cpu['model_files'] if Path(r['path']).name=='model.safetensors']
    if len(weights)!=1 or Path(weights[0]['path']).resolve()!=(model/'model.safetensors').resolve():
        raise ValueError('Requested weights differ from CPU receipt')
    data=Path(a.data_root)/plan.POOL
    manifest=Path(a.data_root)/plan.MANIFEST
    if plan.sha256(data)!=plan.POOL_SHA or plan.sha256(manifest)!=plan.MANIFEST_SHA:
        raise ValueError('Frozen training data differs')
    rows={row['question_id']:row for row in map(json.loads,data.read_text().splitlines())}
    order=json.loads(manifest.read_text())['datasets']
    qids=[rows[order[d]['permutation'][0]] for d in ('logiqa','hellaswag')]
    from transformers import AutoProcessor
    from ctm.evals.local_model import Gemma4UnifiedTextProcessor
    processor=Gemma4UnifiedTextProcessor(AutoProcessor.from_pretrained(str(model),local_files_only=True))
    result={'schema':'gemma-native-prompt-probe-v1','source_commit':a.commit,
        'source_root':str(root),'implementation':file_identity(__file__),
        'reference_implementation':file_identity(plan.__file__),
        'cpu_receipt':file_identity(receipt_path),'data':file_identity(data),
        'data_manifest':file_identity(manifest),'model':str(model.resolve()),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'scope':'CPU prompt parity only; not GPU/generation/optimizer proof',
        'cases':probe(processor,qids),'generation_performed':False,'optimizer_updates':0}
    with Path(a.output).open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')


if __name__=='__main__':
    main()
