"""Descriptive plots for the three-question image feasibility pilot."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ROOT=a.root
rows = json.loads((ROOT / 'scored-rows.json').read_text())
manifest = json.loads((ROOT / 'manifest.json').read_text())
ROOT=a.output;ROOT.mkdir(parents=True,exist_ok=False)
conditions = list(dict.fromkeys(c['condition'] for c in manifest['cases']))
models = ['base', 'rmct']
labels = []
for c in conditions:
    label = c.replace('__', ' · ').replace('_', ' ')
    label = label.replace('none', 'unmarked control').replace('no artifact', 'all unmarked')
    labels.append(label.capitalize())
metrics = [
    ('Correct answer', lambda r: r['scores']['accuracy'], 'Blues'),
    ('Designated bias answer¹', lambda r: r['scores']['bias_answer'], 'Oranges'),
    ('Unparsed answer', lambda r: 1-r['scores']['parsed'], 'Purples'),
    ('Output cap reached', lambda r: r['stop_reason'] in ('max_tokens', 'model_length'), 'Reds'),
]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titleweight': 'bold'})
fig, axes = plt.subplots(1, 4, figsize=(14, 15), sharey=True)
for ax, (title, fn, cmap) in zip(axes, metrics):
    values = np.array([[sum(fn(r) for r in rows if r['condition']==c and r['model'].split('/')[-1]==m)
                        for m in models] for c in conditions], dtype=float)
    if title.startswith('Designated'):
        values[conditions.index('are_you_sure'), :] = np.nan
    ax.imshow(values, vmin=0, vmax=3, cmap=cmap, aspect='auto')
    ax.set_xticks([0, 1], ['Base', 'RMCT'])
    ax.xaxis.tick_top()
    ax.set_title(title, pad=32, fontsize=12)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis='both', length=0)
    for i in range(len(conditions)):
        for j in range(2):
            v=values[i,j]
            ax.text(j, i, '—' if np.isnan(v) else f'{int(v)}/3', ha='center', va='center',
                    color='white' if v>=2 else '#222222', fontsize=10)
    for y in [0.5, 4.5, 9.5, 14.5, 22.5]:
        ax.axhline(y, color='white', linewidth=3)
    for spine in ax.spines.values(): spine.set_visible(False)
fig.suptitle('Image pilot: base Qwen3.5-9B vs RMCT', x=.60, y=.98, fontsize=19, weight='bold')
fig.text(.60, .952, '3 shared HLE questions × 31 conditions × 2 models; counts out of 3', ha='center')
fig.subplots_adjust(left=.36, right=.99, top=.90, bottom=.075, wspace=.20)
fig.text(.02,.039, '¹ Designated wrong-option selections, not switch rates. Unmarked controls retain the same designated option.\n'
         '“Are you sure” is non-directional (—). Unparsed responses remain in all denominators; truncation may overlap parsing.\n'
         'Length-selected feasibility sample: no significance tests or population-level conclusions.', fontsize=10, color='#444444')
for ext in ['png','pdf']: fig.savefig(ROOT / f'pilot-condition-results.{ext}', dpi=180)
plt.close(fig)

fig, axes=plt.subplots(1,2,figsize=(11,4.8))
colors=['#4279a8','#b65b38']
for j,m in enumerate(models):
    rs=[r for r in rows if r['model'].split('/')[-1]==m]
    counts=[sum(r['scores']['parsed'] and r['stop_reason'] not in ('max_tokens','model_length') for r in rs),
            sum(not r['scores']['parsed'] and r['stop_reason'] not in ('max_tokens','model_length') for r in rs),
            sum(r['stop_reason'] in ('max_tokens','model_length') for r in rs)]
    bottom=0
    for n,color,label in zip(counts,['#65a98e','#b6a1cd','#de866c'],['Parsed, not capped','Unparsed, not capped','Capped (any parse status)']):
        axes[0].bar(j,n,bottom=bottom,color=color,label=label if j==0 else None)
        if n: axes[0].text(j,bottom+n/2,str(n),ha='center',va='center')
        bottom+=n
    tok=np.array([r['output_tokens'] for r in rs])
    axes[1].scatter(np.full(len(tok),j)+np.linspace(-.13,.13,len(tok)),tok,s=16,alpha=.5,color=colors[j])
    axes[1].plot([j-.22,j+.22],[np.median(tok)]*2,color='black',linewidth=2)
axes[0].set_ylabel('Responses (93 per model)')
axes[0].set_title('Completion and parsing')
axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.13),fontsize=9)
axes[1].set_title('Output tokens per response')
axes[1].set_ylabel('Tokens, including reasoning')
axes[1].axhline(20480,color='#ad4b3b',linestyle='--',linewidth=1,label='20,480-token cap')
axes[1].legend(fontsize=9,loc='lower right')
for ax in axes:
    ax.set_xticks([0,1],['Base','RMCT'])
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Image pilot: generation diagnostics',weight='bold',fontsize=15)
fig.tight_layout(rect=(0,.035,1,.95))
for ext in ['png','pdf']: fig.savefig(ROOT / f'pilot-generation-diagnostics.{ext}',dpi=180,bbox_inches='tight')
print('Saved condition results and generation diagnostics (PNG + PDF).')
