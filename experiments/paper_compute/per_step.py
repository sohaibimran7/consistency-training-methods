"""Derive amortized per-optimizer-update costs from audited allocations."""
import json
from pathlib import Path

from common import artifact_directory
root=artifact_directory("compute-audit-20260917")
source=json.loads((root/'summary.json').read_text())
rows=[]
for method in ['act','attct','mlpct','bct','opct','rmct']:
    d=source[method]
    n=d['terminal_step']
    rows.append({'method':method,'updates':n,
                 'allocated_gpu_hours_per_update':d['gpu_hours_audited_lineage']/n,
                 'allocated_wall_seconds_per_update':sum(j['seconds'] for j in d['jobs'])/n,
                 'update_loop_seconds_per_update':d.get('committed_update_loop_hours',0)*3600/n if 'committed_update_loop_hours' in d else None})
base=next(r['allocated_gpu_hours_per_update'] for r in rows if r['method']=='mlpct')
for r in rows:r['relative_gpu_cost_per_update']=r['allocated_gpu_hours_per_update']/base
segments=[]
for i,j in enumerate(source['rmct']['jobs']):
    start=i*16+1;end=(i+1)*16
    if 'receipt' in j:
        assert j['receipt']['optimizer_step_start']==start
        assert j['receipt']['optimizer_step_end']==end
    segments.append({'start':start,'end':end,'job':j['job'],
                     'allocated_gpu_hours_per_update':j['gpu_hours']/16,
                     'allocated_wall_seconds_per_update':j['seconds']/16})
out={'methods':rows,'rmct_segments':segments,'definition':'Amortized allocation cost per committed optimizer update; not a measurement of each individual update.'}
(root/'per-step.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# Per-optimizer-step compute costs','',
       'Derived from the audited allocations in `summary.json`; no new model calls. All values are arithmetic means, not medians.', '',
       '| Method | Updates | Allocated GPU-hours/update | Allocated wall seconds/update | Logged update-loop seconds/update | GPU-cost ratio vs MLPCT |',
       '|---|---:|---:|---:|---:|---:|']
for r in rows:
    loop='Not available' if r['update_loop_seconds_per_update'] is None else f"{r['update_loop_seconds_per_update']:.2f}"
    lines.append(f"| {r['method']} | {r['updates']} | {r['allocated_gpu_hours_per_update']:.6f} | {r['allocated_wall_seconds_per_update']:.2f} | {loop} | {r['relative_gpu_cost_per_update']:.2f}× |")
lines += ['', 'Allocated wall time is summed running allocation time divided by committed updates; it excludes queue gaps. RMCT uses four GPUs; other methods use one, so wall-time and GPU-time ratios differ.', '',
          'Allocation totals include model startup/checkpointing and any replay within the audited jobs. BCT/OPCT include the audited initial timeout/replay; RMCT includes successful production jobs only. See README.md for full exclusions. Logged loop time is narrower, but includes generation/cache work where inside the update path; it is not pure backpropagation time.', '',
          'A step is an optimizer update, not one generated answer or one identical amount of work across methods. These are observed implementation costs under the actual training policies, not intrinsic algorithm costs. No currency price is assumed: multiply GPU-hours/update by an applicable GPU-hour tariff.', '',
          '## RMCT split', '',
          f"- Steps 1–176: {source['rmct']['initial_176_gpu_hours']/176:.6f} GPU-hours/update.",
          f"- Steps 177–352: {source['rmct']['continuation_gpu_hours']/176:.6f} GPU-hours/update.", '',
          '## RMCT 16-update segments', '',
          'Each row amortizes a measured allocation over its 16 committed updates; it does not assert that individual updates took equal time.', '',
          '| Updates | Job | GPU-hours/update | Wall minutes/update |', '|---|---|---:|---:|']
for r in segments:lines.append(f"| {r['start']}–{r['end']} | {r['job']} | {r['allocated_gpu_hours_per_update']:.4f} | {r['allocated_wall_seconds_per_update']/60:.2f} |")
(root/'PER_STEP.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:15]))
