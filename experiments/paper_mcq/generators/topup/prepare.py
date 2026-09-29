"""Select only the missing RMCT step-352 QIDs from the reference 100-QID pools."""
import hashlib
import json
from pathlib import Path
from inspect_ai.log import read_eval_log
from experiments.rmct_two_bias_eval import checkpoint_publication as std

ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT.parent/'methods-complete-with-opct-20260916/towards_bias_switch-vs-base/manifest.json'
manifest=json.loads(MANIFEST.read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,value):
    path=ROOT/name; payload=json.dumps(value,indent=2)+'\n'
    if path.exists():assert path.read_text()==payload,path
    else:path.write_text(payload)
def load(record):
    assert sha(record['path'])==record['sha256'],record['path']
    return read_eval_log(record['path'])
def records(method):
    result={}
    for record in manifest['source_logs'][method]:
        log=load(record)
        result[std._dataset_from_log(log),std._bias_from_log(log)]=(record,log)
    for item in manifest['audit']['pairing'][method]:
        key=(item['dataset'],None)
        if key not in result:
            record=item['clean_log'];result[key]=(record,load(record))
    return result
reference=records('act')
existing=records('rmct352')
new=[];cells=[];settings=None
for (dataset,bias),(old_record,old) in existing.items():
    if dataset not in ('logiqa','hellaswag'):continue
    ref_record,ref=reference[dataset,bias]
    have={str(s.id):s for s in old.samples}
    target={str(s.id):s for s in ref.samples}
    assert len(have)==50 and len(target)==100 and have.keys()<=target.keys()
    for qid,s in have.items():
        # Message IDs are transport metadata, not prompt contents.
        def canonical(x):
            return [{k:v for k,v in m.items() if k!='id'} for m in x.model_dump(mode='json')['input']]
        assert canonical(s)==canonical(target[qid]),(dataset,bias,qid)
        assert s.target==target[qid].target
    missing=sorted(target.keys()-have.keys())
    cell=dict(dataset=dataset,bias=bias,retained=old_record,prompt_source=ref_record,
              retained_qids=sorted(have),new_qids=missing,expected_qids=sorted(target))
    cells.append(cell)
    current=dict(model=old.eval.model,model_args=old.eval.model_args,
                 generation_config=old.eval.model_generate_config.model_dump(exclude_none=True))
    assert current['generation_config']['max_tokens']==20480
    if settings is not None:assert current==settings
    settings=current
    for qid in missing:
        s=target[qid]
        new.append(dict(id=f'{dataset}:{bias or "clean"}:{qid}',
            input=s.model_dump(mode='json')['input'],target=s.target,
            metadata={**s.metadata,'rmct_topup':dict(dataset=dataset,bias=bias,qid=qid,
                       prompt_source=ref_record,retained_rmct_source=old_record)}))
assert len(new)==700 and len(cells)==14
assert len({s['id'] for s in new})==700
for dataset in ('logiqa','hellaswag'):
    selected=[c for c in cells if c['dataset']==dataset]
    assert len(selected)==7 and len({tuple(c['new_qids']) for c in selected})==1
new.sort(key=lambda s:s['id'])
for i in range(16):
    write(f'shard-{i:02d}.json',dict(method='rmct352',**settings,samples=new[i::16]))
write('selection.json',dict(checkpoint='rmct352',new_responses=700,new_clean=100,new_biased=600,
    retained_iid_responses=700,unchanged_hle_responses=700,cells=cells,
    settings=settings,manifest_sha256=sha(MANIFEST),
    merge_key=['dataset','bias','qid'],
    note='Union these new QIDs with retained RMCT responses. Keep 64k recovery as a separately versioned overlay.'))
print('Verified exact prompt/QID subset; exported 700 new responses in 16 shards.')
