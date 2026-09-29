"""Plot complete saved training histories; no fitted curves or model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

if not __debug__:
    raise RuntimeError('Do not disable scientific consistency assertions with Python -O')
CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]
sys.path.insert(0, str(REPO))
from ctm_data.adapters.mcq_bias.method_presentation import METHOD_COLORS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--histories', type=Path, required=True)
parser.add_argument('--remote-histories', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
lock = json.loads((CODE/'input-manifest.json').read_text())
for key, path in [('histories', args.histories), ('remote_histories', args.remote_histories)]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != lock[key]['sha256'] or len(payload) != lock[key]['bytes']:
        raise ValueError('Historical input identity mismatch: ' + key)
ROOT = args.output_dir.resolve()
if ROOT.exists():
    raise ValueError('Use a new output directory; historical outputs must stay immutable')
OLD = args.histories.resolve()
REMOTE = args.remote_histories.resolve()
old = json.loads(OLD.read_text())
remote = json.loads(REMOTE.read_text())
ROOT.mkdir(parents=True, exist_ok=False)
order = ['act', 'attct', 'mlpct', 'bct', 'opct', 'rmct']
names = {'act':'ACT', 'attct':'AttCT', 'mlpct':'MLPCT', 'bct':'BCT', 'opct':'OPCT', 'rmct':'RMCT'}
labels = {'act':'Activation mean-squared error', 'attct':'Attention Jensen–Shannon divergence',
          'mlpct':'MLP cosine distance', 'bct':'Training cross-entropy loss', 'opct':'Training reverse-KL estimate',
          'rmct':'Weighted absolute bias-rate gap'}
records = {name:{'terminal':old[name]['terminal'], 'rows':old[name]['rows'],
                 'convergence':old[name]['state']['convergence']} for name in order[:3]}
records.update({name:remote[name] for name in ('bct','opct')})
initial = [{'step':row.get('train/optimizer_step',row.get('step')),
            'gap_sum':sum(row[f'train/consistency_gap_abs_sum_{i}'] for i in (1,2)),
            'gap_count':sum(row[f'train/consistency_gap_abs_count_{i}'] for i in (1,2)),
            'loss':row['train/train/loss'], 'kl':row.get('train/kl_policy_base')} for row in old['rmct']['rows']]
records['rmct'] = {**remote['rmct'], 'rows':initial+remote['rmct']['rows']}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'figure.facecolor':'white'})
fig, axes = plt.subplots(2,3,figsize=(16,9))
summary = {}
for ax, name in zip(axes.flat, order):
    record = records[name]
    rows = record['rows']
    steps = np.array([row['step'] for row in rows])
    assert list(steps) == list(range(1,record['terminal']+1)), name
    ends = steps[15::16]
    color = METHOD_COLORS[name]
    if name == 'rmct':
        sums = np.array([row['gap_sum'] for row in rows])
        counts = np.array([row['gap_count'] for row in rows])
        assert np.all(counts > 0)
        values = sums/counts
        means = sums.reshape(-1,16).sum(1)/counts.reshape(-1,16).sum(1)
        decision = record['decisions'][-1]['record']
        assert decision['decision'] == 'converged' and decision['target']['optimizer_step_end'] == 352
        assert np.isclose(means[-1],decision['window']['current']['weighted_abs_gap'],rtol=0,atol=1e-12)
        state = decision['window']['state_after']
        best_step, best_value = state['best_optimizer_step'],state['best_gap']
        assert np.isclose(means[list(ends).index(best_step)],best_value,rtol=0,atol=1e-12)
        ax.axhline(.1,color='#777777',ls=':',lw=1)
        ax.axvline(176,color='#777777',ls=':',lw=1)
        annotation = f'Patience anchor: {best_step}\nFinal gap: {means[-1]:.5f}\nContinuation begins after 176'
        ax.set_ylim(bottom=0)
        summary[name] = {'terminal':352,'patience_anchor_step':best_step,'patience_anchor_gap':best_value,
                         'final_window_gap':float(means[-1]),'minimum_window_step':int(ends[np.argmin(means)]),
                         'stopping_rule':'8 eligible 16-update windows without >=0.01 improvement from anchor',
                         'decision_source':record['decisions'][-1]['source']}
    else:
        values = np.array([row['loss'] for row in rows])
        assert np.all(np.isfinite(values))
        means = values.reshape(-1,16).mean(1)
        state = record['convergence']
        assert state['step'] == record['terminal'] and state['decision'] == 'plateau'
        assert np.isclose(means[-1],state['window_mean'],rtol=0,atol=1e-10)
        best_step,best_value = state['best_step'],state['best']
        assert np.isclose(means[list(ends).index(best_step)],best_value,rtol=0,atol=1e-10)
        assert state['nonimproving'] == 8
        if np.all(values > 0):
            ax.set_yscale('log')
        else:
            ax.set_yscale('symlog',linthresh=1e-4)
        annotation = f'Best window: {best_step}\nBest: {best_value:.5g}  ·  Final: {means[-1]:.5g}'
        summary[name] = {**state,'terminal':record['terminal'],'scale':ax.get_yscale()}
    ax.plot(steps,values,color=color,alpha=.22,lw=.65)
    ax.plot(ends,means,color=color,lw=2.2,marker='o',ms=3)
    ax.axvspan(best_step,record['terminal'],color=color,alpha=.07)
    ax.scatter([best_step],[best_value],marker='*',s=140,color=color,edgecolors='#333333',linewidths=.5,zorder=5)
    ax.axvline(record['terminal'],color=color,ls='--',lw=1.2)
    ax.text(.98,.96,annotation,transform=ax.transAxes,ha='right',va='top',fontsize=9,
            bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':3})
    ax.set_title(f'{names[name]} · stopped at update {record["terminal"]}',loc='left',weight='bold',fontsize=12)
    ax.set_xlabel('Optimizer update')
    scale_note = ' (log scale)' if ax.get_yscale()=='log' else ' (symlog; linear near zero)' if ax.get_yscale()=='symlog' else ''
    ax.set_ylabel(labels[name]+scale_note)
    ax.set_xlim(0,record['terminal']*1.03)
    ax.grid(alpha=.15,which='major')
fig.suptitle('Historical training convergence · all six trained methods',fontsize=19,weight='bold',y=.98)
fig.text(.5,.934,'Separate objectives and axis ranges; loss magnitudes are not comparable across methods.',ha='center',fontsize=11)
handles = [Line2D([0],[0],color='#666666',alpha=.3,lw=1,label='Per update'),
           Line2D([0],[0],color='#444444',lw=2,marker='o',ms=3,label='Non-overlapping 16-update mean'),
           Line2D([0],[0],color='#444444',marker='*',linestyle='',ms=10,label='Best window / RMCT patience anchor'),
           Line2D([0],[0],color='#444444',linestyle='--',label='Stopping update')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.05),ncol=4,frameon=False,fontsize=9)
fig.text(.055,.025,'Shading: final eight-window patience period. RMCT averages gaps weighted by their counts; dotted lines mark gap 0.10 and update 176.\nHistorical flawed RMCT training; not corrected reruns or held-out performance. Base has no training-convergence curve.',fontsize=9,color='#444444')
fig.subplots_adjust(top=.875,bottom=.17,wspace=.3,hspace=.42)
for ext in ('png','svg','pdf'):
    fig.savefig(ROOT/f'convergence-all-methods.{ext}',dpi=200)
plt.close(fig)

rows = records['rmct']['rows']
fig, axes = plt.subplots(1,2,figsize=(12,4.5))
for ax,key,title in zip(axes,('loss','kl'),('Policy-gradient optimization loss','Policy/base KL estimate')):
    y = np.array([row[key] for row in rows],dtype=float)
    assert np.all(np.isfinite(y))
    ax.plot(np.arange(1,353),y,color=METHOD_COLORS['rmct'],alpha=.22,lw=.7)
    ax.plot(np.arange(16,353,16),y.reshape(-1,16).mean(1),color=METHOD_COLORS['rmct'],lw=2)
    ax.axvline(176,color='#777777',ls=':',lw=1)
    ax.set(xlabel='Optimizer update',ylabel=title,xlim=(0,360),title=title)
    ax.grid(alpha=.15)
fig.suptitle('RMCT optimization diagnostics · updates 1–352',weight='bold')
fig.text(.5,.025,'The stopping metric is the weighted absolute bias-rate gap shown in the main figure, not optimization loss.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.08,1,.94))
for ext in ('png','svg','pdf'):
    fig.savefig(ROOT/f'rmct-optimization-diagnostics.{ext}',dpi=180)
plt.close(fig)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
paths = [OLD,REMOTE,Path(__file__),REPO/'ctm_data/adapters/mcq_bias/method_presentation.py']
(ROOT/'manifest.json').write_text(json.dumps({'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'order':order,'palette':METHOD_COLORS,'complete_step_coverage':{m:records[m]['terminal'] for m in order},
    'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.suffix in ('.png','.svg','.pdf')},
    'model_or_training_calls':0, 'historical_flawed_rmct':True, 'selection':'historical own-objective stopping, not validation TBSR', 'runtime':{'matplotlib':matplotlib.__version__,'numpy':np.__version__}},indent=2)+'\n')
print(json.dumps({'completed':True,'steps':{m:records[m]['terminal'] for m in order}}))
