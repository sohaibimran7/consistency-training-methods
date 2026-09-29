"""Expose existing immutable logs in one Inspect View folder, without merging."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.root
view=a.output;view.mkdir(parents=True,exist_ok=False)
for record in json.loads((root/'collected/logs.json').read_text()):
    source=Path(record['path']).resolve(strict=True);link=view/source.name
    if link.exists() and link.resolve()!=source:raise ValueError(f'Log basename collision: {source.name}')
    if not link.exists():link.symlink_to(source)
print(len(list(view.glob('*.eval'))),'logs linked')
