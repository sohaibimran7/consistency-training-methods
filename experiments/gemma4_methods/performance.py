"""Read-only timing summaries; allocation counts are not measured speedups."""
import argparse
import json
import math
from pathlib import Path
import statistics
from experiments.gemma4_methods.reference.train import checkpoint_identity

STAGES=('sampling','prepare_score_backward','optimizer','checkpoint')


def summary(rows):
    seen=set()
    accepted=[]
    for row in rows:
        key=(row['run_root'],row['method'],row['step'])
        if key in seen:
            raise ValueError('Duplicate saved optimizer update in timing evidence')
        seen.add(key)
        values=row['stage_wall_seconds']
        if set(values)!=set(STAGES) or any(type(x) not in (int,float)
                or not math.isfinite(x) or x<0 for x in values.values()):
            raise ValueError('Invalid stage timing evidence')
        if row.get('checkpoint_saved') is not True:
            raise ValueError('Unsaved updates are not throughput evidence')
        run_dir=Path(row['run_root']).resolve()/'runs'/row['method']
        checkpoint=run_dir/row['checkpoint']
        if (checkpoint.is_symlink() or not checkpoint.resolve().is_relative_to(run_dir)
                or checkpoint_identity(checkpoint)!=row['checkpoint_files']):
            raise ValueError('Timing checkpoint bytes differ from saved evidence')
        accepted.append(row)
    if not accepted:
        raise ValueError('No saved-update timing evidence')
    totals={s:math.fsum(r['stage_wall_seconds'][s] for r in accepted) for s in STAGES}
    total=math.fsum(totals.values())
    return {'saved_updates':len(accepted),'total_measured_stage_seconds':total,
            'total_stage_seconds':totals,
            'median_stage_seconds':{s:statistics.median(r['stage_wall_seconds'][s] for r in accepted)
                                    for s in STAGES},
            'stage_fractions':{s:totals[s]/total if total else None for s in STAGES},
            'scope':'awaited host wall time; not GPU kernel time, total allocation cost or demonstrated scaling gain'}


def worker_benchmark_plan(method):
    if method not in ('bct','opct','rmct'):
        raise ValueError('Only online rollout stages use this benchmark')
    return {'method':method,'worker_counts':[1,2,3],
        'production_transport':'single-node local spawned workers',
        'concurrent_sequences_per_request':{'bct':1,'opct':16,'rmct':96}[method],
        'preserve':['original source/runtime/checkpoint','native prompts and token caps',
                    'effective batch, rollout counts, loss and consumed QIDs','selection policy'],
        'measured_speedup':None,'requires_native_gates':True,
        'notes':'BCT serial target calls route to worker0; extra workers cannot accelerate this existing path'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timing-jsonl',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(summary([json.loads(line) for line in a.timing_jsonl.read_text().splitlines()]),indent=2))
