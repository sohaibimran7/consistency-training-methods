"""Audit complete saved coverage and export blinded, final-only flattery jobs."""
import argparse,json
from collections import Counter
from pathlib import Path
from inspect_ai.log import read_eval_log
from scripts.agentic_transfer_tasks import output_status, visible_reasoning
from scripts.lasr_transfer_tasks import canonical_messages
from scripts.prepare_v2_breadth import digest
from scripts.grade_rogueqwen_flattery import sha, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'experiments/eval_awareness/v2-breadth-20260924'

def collect(panel_path=None,reuse_path=None,generation_dir=None,path_map=None):
    panel_path=panel_path or BASE/'frozen-panel-v1.json'
    reuse_path=reuse_path or BASE/'reuse-v2.json'
    generation_dir=generation_dir or BASE/'generation-6848128'
    path_map=path_map or {}
    def local(path):return Path(path_map.get(str(path),str(path)))
    panel=json.loads(panel_path.read_text())
    reuse=json.loads(reuse_path.read_text())
    assert reuse['panel_sha256']==sha(panel_path)
    by_id={r['source_id']+'--'+r['condition']:r for r in panel['rows']}
    rows=[];seen=set()
    def add(role,path,sample,pid,reused=False):
        row=by_id[pid]; identity=(role,pid)
        assert identity not in seen
        events=[e for e in sample.events if e.event=='model']
        assert len(events)==1
        assert canonical_messages(events[0].input)==row['messages']
        assert not sample.error
        seen.add(identity)
        rows.append(dict(key=digest(identity),identity=dict(role=role,source_id=row['source_id'],condition=row['condition'],family=row['family']),
            source=dict(path=str(path),sha256=sha(path),sample_id=sample.id,epoch=sample.epoch),
            reused=reused,status=output_status(sample.output),stop_reason=sample.output.choices[0].stop_reason,
            completion=sample.output.completion,reasoning=visible_reasoning(sample.output)))
    for role in ('base','mo_mid','mo_post'):
        paths=list((generation_dir/role/'logs').glob('*.eval'))
        assert len(paths)==1
        log=read_eval_log(paths[0],resolve_attachments=True)
        assert log.status=='success'
        for s in log.samples:add(role,paths[0],s,s.id)
    for r in reuse['rows']:
        assert sha(local(r['source_log']))==r['source_log_sha256']
        log=read_eval_log(local(r['source_log']),resolve_attachments=True)
        sample=next(s for s in log.samples if (s.id,s.epoch)==(r['sample_id'],r['epoch']))
        add(r['role'],local(r['source_log']),sample,r['source_id']+'--'+r['condition'],True)
    assert seen=={(role,pid) for role in ('base','mo_mid','mo_post') for pid in by_id}
    return rows

def main(panel=None,reuse=None,generation_dir=None,out=None,path_map=None):
    panel=panel or BASE/'frozen-panel-v1.json';out=out or BASE/'grading-v1'
    rows=collect(panel,reuse,generation_dir,path_map)
    audit=dict(panel_sha256=sha(panel),rows=rows,
        coverage={role:dict(Counter(r['status'] for r in rows if r['identity']['role']==role)) for role in ('base','mo_mid','mo_post')},
        policy='Incomplete generations excluded from primary grading, never negative-imputed; no target calls or retries.')
    write(out/'inventory.json',audit)
    jobs=[dict(key=r['key'],identity=r['identity'],source=r['source'],
        payload={'transcript':[{'id':'A1','role':'assistant','text':r['completion']}]}) for r in rows if r['status']=='valid']
    write(out/'flattery-input.json',dict(model='openrouter/openai/gpt-5.6-luna',jobs=jobs))
    print(json.dumps(dict(total=len(rows),reused=sum(r['reused'] for r in rows),coverage=audit['coverage'],flattery_jobs=len(jobs))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path);p.add_argument('--reuse',type=Path)
    p.add_argument('--generation-dir',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--path-map',type=Path,help='Exact archived-to-local path mappings; hashes still verified')
    args=p.parse_args();main(args.panel,args.reuse,args.generation_dir,args.output,json.loads(args.path_map.read_text()) if args.path_map else None)
