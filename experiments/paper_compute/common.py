"""Explicit filesystem boundary for offline historical-paper analyses."""
import argparse
from pathlib import Path


def artifact_directory(campaign):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', required=True, type=Path,
                        help='Directory containing the dated artifact subdirectories; outputs are written there.')
    args = parser.parse_args()
    if not __debug__:
        parser.error('Do not use python -O: historical validation assertions must remain enabled.')
    root = args.artifact_root.expanduser().resolve()
    directory = root / campaign
    if not directory.is_dir():
        parser.error(f'Missing artifact directory: {directory}')
    return directory


def unique_match(directory, pattern):
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {pattern} in {directory}; found {len(matches)}')
    return matches[0]
