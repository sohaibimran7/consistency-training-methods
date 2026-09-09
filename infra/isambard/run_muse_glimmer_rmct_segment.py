#!/usr/bin/env python3
"""Run, seal, and score one exact Muse Glimmer RMCT training window."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.muse_glimmer_rmct_replication import plan  # noqa: E402
from infra.isambard import muse_glimmer_rmct_segment_contract as contract  # noqa: E402
from scripts.run_experiment import _argument_tokens  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    root = args.repo_root.resolve()
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if len(visible) != 4 or any(not value for value in visible) or len(set(visible)) != 4:
        raise contract.ContractError("Muse training requires exactly four visible GPUs")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise contract.ContractError("Muse training requires offline Hugging Face and Transformers resolution")

    guarded = contract.guard(root, args.segment_index)
    if guarded["action"] != "proceed":
        print(json.dumps(guarded, sort_keys=True), flush=True)
        return 0

    compiled = plan.segment_args(root, args.segment_index, model_snapshot=args.model_snapshot)
    # Preserve the venv spelling. ``bin/python`` is intentionally a symlink to
    # the uv-managed base interpreter; resolving it discards venv site-packages.
    runtime_python = Path(os.environ.get("CTM_MUSE_RUNTIME_PYTHON", ""))
    if not runtime_python.is_file() or not os.access(runtime_python, os.X_OK):
        raise contract.ContractError("Muse source-runtime Python is absent or not executable")
    command = [
        str(runtime_python),
        str(root / "scripts" / "train_rlct.py"),
        *_argument_tokens(compiled),
    ]
    contract.write_launch(
        root,
        args.segment_index,
        model_snapshot=args.model_snapshot,
        argv=command,
    )
    print("CTM_MUSE_SEGMENT_ARGV=" + json.dumps(command), flush=True)
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise contract.ContractError(
            f"Muse segment {args.segment_index} child failed with exit code {completed.returncode}; "
            "the residue is intentionally retained and cannot be silently resumed"
        )
    sealed = contract.seal(root, args.segment_index)
    convergence = contract.evaluate_convergence(root, args.segment_index)
    print("CTM_MUSE_SEGMENT_RECEIPT=" + str(sealed["receipt"]), flush=True)
    print("CTM_MUSE_CONVERGENCE=" + json.dumps(convergence, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
