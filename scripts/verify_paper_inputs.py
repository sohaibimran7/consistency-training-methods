"""Verify historical paper input hashes; this is not figure regeneration.

Supply each artifact checkout with --root monitor=/path --root methods=/path
--root main=/path. Missing and changed inputs are failures, never silently skipped.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'docs/paper-results-manifest.json')
    parser.add_argument('--root', action='append', default=[], metavar='ALIAS=PATH')
    args = parser.parse_args()
    roots = dict(item.split('=', 1) for item in args.root)
    failures = 0
    for entry in json.loads(args.manifest.read_text())['results']:
        alias = entry['source_root']
        root = roots.get(alias)
        path = Path(root) / entry['source'] if root else None
        if path is None or not path.is_file():
            status = 'MISSING'
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry['source_sha256']:
            status = 'CHANGED'
        else:
            status = 'VERIFIED_HISTORICAL_INPUT'
        failures += status != 'VERIFIED_HISTORICAL_INPUT'
        print(f'{status}: {alias}:{entry["source"]}')
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
