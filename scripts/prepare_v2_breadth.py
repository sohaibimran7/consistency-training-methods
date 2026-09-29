"""Export source-bound V2 breadth candidates without target or judge calls.

Select by source identity, never target responses. No factor acceptance is implied.
"""
import copy
import hashlib
import json
from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_ai._util.registry import registry_params
from inspect_ai.model import ChatMessageSystem
from scripts import lasr_suite_am_tasks as am
from scripts.lasr_transfer_tasks import canonical_messages, MODELS, TARGET_CONFIG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/eval_awareness/v2-breadth-20260924'
PIN = ROOT / 'experiments/eval_awareness/lasr_transfer/rescue-factor-inspection-20260913/runtime/pinned-source'
GOALS = ('america', 'global', 'safety', 'acceleration', 'ethical', 'pragmatic', 'individualism', 'collectivism')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    source = PIN / ('inspect_evals-' + am.FORK_COMMIT)
    proof = am.verify_source(source, PIN / 'source.zip')
    module = am._import_native(source)
    candidates = []
    for family in am.FAMILIES:
        for goal in GOALS:
            config = {**copy.deepcopy(am.NATIVE_CONFIG), 'goal_value': goal}
            task = module.agentic_misalignment(scenario=family, **config)
            sample = task.dataset[0]
            system = registry_params(list(task.solver)[0])['template']
            messages = canonical_messages([ChatMessageSystem(content=system), *sample.input])
            candidates.append(dict(id=f'am-{family}-{goal}', family=f'am-{family}',
                variation='goal configuration within one narrative, not an independent story',
                messages=messages, messages_sha256=digest(messages), native_config=config,
                source_proof=proof, target=sample.target, metadata=sample.metadata,
                native_scorer='pinned harmfulness_scorer; Luna judge adaptation',
                factors_status='unconstructed_unreviewed'))
    logs = list((ROOT/'artifacts/agentic-transfer-20260910/generation/base').glob('*transfer-instrumental-eval-O-*.eval'))
    assert len(logs) == 1
    path = logs[0]
    saved = read_eval_log(path)
    seen = {}
    for sample in saved.samples:
        # Deliberately never inspect messages/output/scores: only original task inputs.
        inp = sample.input
        messages = [{'role': 'user', 'content': inp}] if isinstance(inp, str) else canonical_messages(inp)
        record = dict(id=f'ie-{sample.id}', family='instrumentaleval',
            variation='distinct original source item; category recorded in metadata',
            messages=messages, messages_sha256=digest(messages), target=sample.target,
            metadata=sample.metadata, native_scorer='InstrumentalEval native rubric; Luna judge adaptation',
            source_log=str(path.relative_to(ROOT)), source_log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            factors_status='unconstructed_unreviewed')
        if record['id'] in seen:
            assert seen[record['id']] == record
        seen[record['id']] = record
    candidates.extend(seen.values())
    assert len({r['id'] for r in candidates}) == len(candidates)
    result = dict(status='source_candidates_only_not_accepted_factor_prompts',
        selection='All eight explicit-goal replacement configurations in three AM families; all 12 previously source-hash-selected InstrumentalEval pilot items; no response-based selection',
        repetitions_per_prompt=1, models=MODELS, target_config=TARGET_CONFIG,
        intended_conditions=['B']+[f'F{i}' for i in range(1,9)],
        candidate_sources=len(candidates), maximum_primary_trajectories=len(candidates)*9*3,
        additional_families=[dict(family='LASR executive/budget',status='inventory pending; not excluded'),
                             dict(family='LASR eval sabotage/SWE',status='requires confined sandbox readiness; no unconfined execution')],
        limitations=['AM goal configurations share three narratives', 'Factor effects must be matched within source and reported by family',
                     'Existing V2 rescue F2/F8 confounds retained explicitly; do not copy rescue-specific edits to unrelated tasks'],
        candidates=candidates)
    dest=OUT/'source-candidates.json'
    with dest.open('x') as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps({k:result[k] for k in ('status','candidate_sources','maximum_primary_trajectories')}))

if __name__ == '__main__':
    main()
