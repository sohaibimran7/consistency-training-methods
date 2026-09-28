"""Verify input hashes and replay all six commands in an isolated scratch tree."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', required=True, type=Path)
    parser.add_argument('--report', type=Path, help='Optional verification receipt path.')
    args = parser.parse_args()
    source = args.artifact_root.resolve()
    manifest = json.loads((HERE / 'input-manifest.json').read_text())
    for record in manifest['inputs'] + manifest['baselines']:
        p = source / record['path']
        if p.stat().st_size != record['bytes'] or digest(p) != record['sha256']:
            raise ValueError(f'Input/baseline hash mismatch: {record["path"]}')
    checks = []
    with tempfile.TemporaryDirectory(prefix='paper-compute-') as temp:
        root = Path(temp)
        for record in manifest['inputs']:
            target = root / record['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source / record['path'])
        for command in ['summarize', 'per_step', 'analyze', 'restricted_search', 'data_analyze', 'data_report']:
            subprocess.run([sys.executable, str(HERE / (command + '.py')),
                            '--artifact-root', str(root)], check=True, capture_output=True, text=True)
        for record in manifest['baselines']:
            path = record['path']
            actual = (root / path).read_bytes()
            expected = (source / path).read_bytes()
            if path == 'data-matched-checkpoints-20260926/README.md':
                expected = expected.replace(b'python3 artifacts/data-matched-checkpoints-20260926/analyze.py',
                    b'python3 experiments/paper_compute/data_analyze.py --artifact-root /path/to/artifacts')
                expected = expected.replace(b'python3 artifacts/data-matched-checkpoints-20260926/report.py',
                    b'python3 experiments/paper_compute/data_report.py --artifact-root /path/to/artifacts')
            if actual != expected:
                raise ValueError(f'Output differs from archived baseline: {path}')
            checks.append({'path': path, 'sha256': hashlib.sha256(actual).hexdigest(), 'matched': True})
        # The data-report manifest intentionally reflects the files present in
        # this replay tree, not unrelated historical scripts/docs beside them.
        directory = root / 'data-matched-checkpoints-20260926'
        for name, sha in json.loads((directory / 'SHA256.json').read_text()).items():
            if digest(directory / name) != sha:
                raise ValueError(f'Generated manifest mismatch: {name}')
        # CLI must never silently fall back to the source worktree.
        missing = subprocess.run([sys.executable, str(HERE / 'summarize.py')], capture_output=True)
        if missing.returncode != 2 or b'--artifact-root' not in missing.stderr:
            raise ValueError('Required-root CLI check failed')
    result = {'passed': True, 'commands': 6, 'input_hashes_verified': len(manifest['inputs']),
              'outputs': checks, 'source_artifacts_modified': False,
              'note': 'Data README command links adapted; numerical/path outputs byte-identical. Historical snapshot only.'}
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
