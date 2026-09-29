"""Portable, hash-pinned assembly of provisional manuscript figures.

Renders SAVED statistics and copies SAVED figures. Does not reproduce upstream
generation, parsing, grading, aggregation, training or hypothesis tests.
"""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

ROOT_NAMES = ('corrected_figures', 'monitor', 'ctm_artifacts', 'organism_artifacts')
LABELS = {'base': 'Base', 'act': 'ACT', 'attct': 'AttCT', 'mlpct': 'MLPCT',
          'bct': 'BCT', 'opct': 'OPCT', 'rmct352': 'RMCT'}
WARNING = 'PROVISIONAL — flawed RMCT training; historical checkpoints; not data-matched'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def child(root, relative):
    """Require root-relative paths, including after symlink resolution."""
    rel = Path(relative)
    path = (root / rel).resolve()
    if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root):
        raise ValueError(f'Path must remain inside its declared root: {relative}')
    return path


def load_sources(manifest_path, roots):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest['schema'] != 'ctm-manuscript-figure-sources-v1':
        raise ValueError('Unsupported source manifest schema')
    blobs, paths = {}, {}
    for key, source in manifest['sources'].items():
        path = child(roots[source['root']], source['path'])
        data = path.read_bytes()
        digest = sha(data)
        if digest != source['sha256']:
            raise ValueError(f'SHA256 mismatch for {key}: expected {source["sha256"]}, got {digest}')
        blobs[key], paths[key] = data, path
    return manifest, sha(manifest_bytes), blobs, paths


def plot_inputs(blobs, key):
    style = json.loads(blobs['style'])
    methods = style['condition_order']
    if methods != list(LABELS):
        raise ValueError('Unexpected method ordering; RMCT must remain last')
    rows = json.loads(blobs[key])
    selected = {}
    for row in rows:
        p, b, m = row['population'], row['bias_type'], row['condition']
        if p not in ('held_in_datasets', 'held_out_dataset') or b not in ('seen_mean', 'held_out_mean'):
            continue
        index = (p, b, m)
        if index in selected:
            raise ValueError(f'Duplicate plot cell: {index}')
        vals = [row[k] for k in ('ci_lower', 'mean', 'ci_upper')]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError(f'Undefined/nonfinite plot cell: {index}; never impute zero')
        if not 0 <= vals[0] <= vals[1] <= vals[2] <= 1:
            raise ValueError(f'Invalid estimate/interval: {index}')
        if not isinstance(row['n_scored'], int) or row['n_scored'] <= 0:
            raise ValueError(f'Invalid denominator: {index}')
        selected[index] = row
    expected = {(p,b,m) for p in ('held_in_datasets','held_out_dataset')
                for b in ('seen_mean','held_out_mean') for m in methods}
    if set(selected) != expected:
        raise ValueError(f'Expected exactly {len(expected)} pooled plot cells')
    return style, methods, selected


def render(blobs, key, destination):
    import matplotlib.pyplot as plt
    style, methods, rows = plot_inputs(blobs, key)
    ylabel = ('Towards-bias switch rate (%)' if key == 'main-switch'
              else 'Conditional bias verbalisation (%)')
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.6), sharey=True)
    for i, pop in enumerate(('held_in_datasets','held_out_dataset')):
        for j, bias in enumerate(('seen_mean','held_out_mean')):
            ax = axes[i,j]
            for x, method in enumerate(methods):
                r = rows[pop,bias,method]
                y, lo, hi = [100*r[k] for k in ('mean','ci_lower','ci_upper')]
                ax.bar(x,y,color=style['condition_styles'][method]['color'],width=.75)
                ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt='none',color='#333333',capsize=2,lw=1)
                ax.text(x,108,str(r['n_scored']),ha='center',fontsize=7,rotation=45)
                if r.get('significance'):
                    ax.text(x,hi+1,r['significance'],ha='center',fontsize=8)
            ax.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=35,ha='right',fontsize=8)
            ax.set_ylim(0,120); ax.set_yticks([0,25,50,75,100]); ax.tick_params(axis='y',labelsize=8)
            ax.set_title(('Seen' if j==0 else 'Held-out')+' biases',fontsize=10)
            if j==0:
                ax.set_ylabel(('Seen datasets' if i==0 else 'Held-out dataset')+'\n'+ylabel,fontsize=9)
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle(WARNING,fontsize=10,color='#9b2525')
    fig.tight_layout(rect=[0,0,1,.96])
    outputs = []
    for ext in ('pdf','png'):
        path = destination.with_suffix('.'+ext)
        metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {}
        fig.savefig(path,dpi=200,bbox_inches='tight',metadata=metadata)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for root in ROOT_NAMES:
        parser.add_argument('--'+root.replace('_','-')+'-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True,
                        help='Assembly root; figures/provisional and inventory are written beneath it')
    parser.add_argument('--manifest',type=Path,default=Path(__file__).with_name('figure-sources.json'))
    parser.add_argument('--check-only',action='store_true',help='Validate every source and main-panel row without writing outputs')
    args = parser.parse_args(argv)
    roots = {name:getattr(args,name+'_root').resolve() for name in ROOT_NAMES}
    output = args.output_dir.resolve()
    try:
        manifest,digest,blobs,paths = load_sources(args.manifest,roots)
        entries = manifest['entries']
        if len({e['id'] for e in entries}) != len(entries):
            raise ValueError('Duplicate inventory IDs')
        destinations = set()
        for entry in entries:
            if entry['kind'] not in ('render_saved_statistics','copy_saved_figure','source_only_manual_table'):
                raise ValueError(f'Unknown entry kind: {entry["kind"]}')
            if any(s not in blobs for s in entry['sources']):
                raise ValueError(f'Unknown source reference: {entry["id"]}')
            if entry['kind'] == 'render_saved_statistics':
                plot_inputs(blobs,entry['id'])
            if entry['output']:
                dest = child(output,entry['output'])
                targets = [dest, dest.with_suffix('.png')] if entry['kind']=='render_saved_statistics' else [dest]
                for target in targets:
                    if target in destinations or target in paths.values() or target == args.manifest.resolve():
                        raise ValueError(f'Output collision: {target}')
                    destinations.add(target)
        inventory_path = child(output,'figure-inventory.json')
        if inventory_path in paths.values() or inventory_path in destinations or inventory_path == args.manifest.resolve():
            raise ValueError('Inventory output collides with an input or figure')
    except (OSError,KeyError,ValueError,TypeError) as error:
        parser.exit(2,f'Preflight failed; no outputs written: {error}\n')
    if args.check_only:
        print(f'Verified {len(blobs)} source hashes and both 2x2 grids; no outputs written.')
        return
    import matplotlib
    matplotlib.use('Agg')
    records = []
    for entry in entries:
        generated = []
        if entry['output']:
            dest = child(output,entry['output']); dest.parent.mkdir(parents=True,exist_ok=True)
            if entry['kind'] == 'render_saved_statistics':
                generated = render(blobs,entry['id'],dest)
            elif entry['kind'] == 'copy_saved_figure':
                dest.write_bytes(blobs[entry['sources'][0]]); generated = [dest]
        records.append({**entry,'outputs':[{'path':str(p.relative_to(output)),'sha256':sha(p.read_bytes())} for p in generated]})
    result = {'schema':'ctm-manuscript-figure-assembly-v1','scope':manifest['scope'],
              'source_manifest_sha256':digest,'builder_sha256':sha(Path(__file__).read_bytes()),
              'runtime':{'python':platform.python_version(),'matplotlib':matplotlib.__version__},
              'sources':manifest['sources'],'entries':records,
              'model_calls':0,'grader_calls':0,'upstream_aggregation_performed':False,
              'copied_figures_require_manuscript_provisional_captions':True}
    output.mkdir(parents=True,exist_ok=True)
    inventory_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Assembled {sum(len(r["outputs"]) for r in records)} files; '
          f'{sum(r["kind"]=="source_only_manual_table" for r in records)} table sources verified, not generated.')


if __name__ == '__main__':
    main()
