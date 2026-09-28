import itertools,json
from pathlib import Path
from common import artifact_directory
R=artifact_directory("compute-matched-selection-20260926")
D=json.loads((R/'selection.json').read_text())['ledger']
methods=['bct','opct','rmct']; ranked=[]
for rows in itertools.product(*(D[m] for m in methods)):
    costs=[r['allocation_estimate'] for r in rows]
    ranked.append({'steps':dict(zip(methods,[r['step'] for r in rows])),
                   'costs':dict(zip(methods,costs)),
                   'spread_over_max':(max(costs)-min(costs))/max(costs),
                   'budget':max(costs)})
ranked.sort(key=lambda r:(r['spread_over_max'],r['budget']))
budgets=[]
for b in [19.28,22.05,36.5,45.67,53.06]:
    picks={m:max((r for r in D[m] if r['allocation_bound_high']<=b),key=lambda r:r['step']) for m in methods}
    c=[p['allocation_estimate'] for p in picks.values()]
    budgets.append({'budget':b,'selections':picks,'spread_over_max':(max(c)-min(c))/max(c)})
result={'metric':'allocated GPU-hours, not algorithmic FLOPs','combinations':len(ranked),'ranked_combinations':ranked,'budget_sensitivity':budgets,
        'recommended':{m:next(r for r in D[m] if r['step']==ranked[0]['steps'][m]) for m in methods}}
(R/'restricted-search.json').write_text(json.dumps(result,indent=2)+'\n')
print('combinations',len(ranked),'best',ranked[0])
for b in budgets:print(b['budget'],{m:r['step'] for m,r in b['selections'].items()},b['spread_over_max'])
