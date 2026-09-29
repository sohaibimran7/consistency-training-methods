"""Build a selective, hash-bound inventory; never copy credentials or call APIs."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ENTRYPOINTS=['run_v2_breadth','prepare_v2_breadth_grading','grade_v2_breadth_awareness','grade_v2_breadth_native',
    'grade_rogueqwen_flattery','repair_v2_awareness_basis','plot_v2_breadth_results','plot_v2_breadth_fit',
    'plot_vae_or_flattery','compare_v2_v3_v4_factors','analyze_conditional_misalignment','test_eval_gaming_portability',
    'prepare_eval_gaming_64k','test_eval_gaming_64k','inventory_eval_gaming_handoff']

def inventory():
    pending=[ROOT/'scripts'/f'{name}.py' for name in ENTRYPOINTS];selected=set()
    while pending:
        path=pending.pop()
        if path in selected:continue
        selected.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.Import):names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                names=([node.module] if node.module else [])
                if node.module=='scripts':names+=['scripts.'+n.name for n in node.names]
            for name in names:
                candidate=ROOT/(name.replace('.','/')+'.py') if name.startswith('scripts.') else ROOT/'scripts'/(name+'.py')
                if candidate.is_file() and candidate not in selected:pending.append(candidate)
    v2=ROOT/'experiments/eval_awareness/v2-breadth-20260924'
    old=ROOT/'experiments/eval_awareness/lasr_transfer/petri-pilots-20260914/expanded-r3-20260915/recovery-preparation-2249'
    artifacts={v2/p for p in ['frozen-panel-v1.json','reuse-v2.json','source-candidates.json','CONTROL-CONTRACT.md','EXECUTION-STATUS.md','analysis-v1/README.md']}
    artifacts.update(ROOT/'experiments/eval_awareness/paper-handoff-20260928'/p for p in ['README.md','future-65536-proposal.json'])
    for directory in [v2/'grading-v1',v2/'analysis-v1']:
        artifacts.update(p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.png','.svg'))
    artifacts.update(old/p for p in ['rogueqwen-flattery-v2-20260923/analysis.json','rogueqwen-flattery-v2-20260923/negative-audit-20260924/audited-results.json','analysis-completed-r3-v1/report.json','analysis-vae-or-flattery-20260924/figure-data.json'])
    def rows(paths):return [dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(paths)]
    versions={}
    for name in ('inspect-ai','numpy','matplotlib','pydantic','python-dotenv','openai','httpx'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return dict(entrypoints=ENTRYPOINTS,source_files=rows(selected),saved_analysis_and_grading_inputs=rows(artifacts),
        tested_local_dependencies=versions,
        exclusions=['Credentials, runtime api-key files, weights, all .env files, unrelated campaigns',
                    'Generation logs, external IE source/evidence and pinned native archive require separate acquisition from recorded manifest paths/hashes'],
        note='Static local import closure is a selective review list, not proof all dynamic upstream dependencies are bundled. Saved inputs contain synthetic transcripts/reasoning: review distribution rights before publication.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=inventory();args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(source_files=len(result['source_files']),artifacts=len(result['saved_analysis_and_grading_inputs']),output=str(args.output))))
