#!/usr/bin/env python3
"""Archive a provably pre-training Muse RMCT startup failure for clean retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.muse_glimmer_rmct_replication import plan  # noqa: E402
from infra.isambard import muse_glimmer_rmct_segment_amendment as amendment  # noqa: E402
from infra.isambard import muse_glimmer_rmct_segment_contract as contract  # noqa: E402


SCHEMA = "muse-glimmer-rmct-failed-startup-archive-v1"
ARCHIVE_ROOT = Path("_archive/muse-glimmer-rmct-failed-startups")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise contract.ContractError(f"startup residue contains a linked/special entry: {path}")
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def archive(
    root_value: str | Path,
    segment_index: int,
    *,
    job_id: str,
    slurm_state: str,
    slurm_exit_code: str,
    slurm_output: str | Path,
) -> dict[str, Any]:
    root = Path(root_value).resolve()
    if isinstance(segment_index, bool) or not 0 < segment_index < plan.TOTAL_SEGMENTS:
        raise contract.ContractError("startup archive requires a non-base Muse segment index")
    if not job_id.isdigit() or slurm_state != "FAILED" or slurm_exit_code != "1:0":
        raise contract.ContractError("startup archive requires the exact failed Slurm identity")
    amendment.validate_attestation(root)
    amendment.install()
    contract.validate_receipt(root, segment_index - 1)
    contract.validate_convergence(root, segment_index - 1)

    control = contract.control_root(root, segment_index)
    run = contract.run_root(root, segment_index)
    if control.is_symlink() or not control.is_dir() or run.is_symlink() or not run.is_dir():
        raise contract.ContractError("failed startup does not have both regular residue directories")
    if contract.receipt_path(root, segment_index).exists() or contract.convergence_path(root, segment_index).exists():
        raise contract.ContractError("refusing to archive a sealed or convergence-scored segment")
    forbidden = [
        run / "metrics.jsonl",
        run / "rollouts",
        run / "checkpoints",
    ]
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise contract.ContractError("refusing to archive residue after training metrics/rollouts/checkpoints exist")

    launch = contract._json(control / "launch.json", label="failed startup launch")
    if launch.get("slurm_job_id") != job_id or launch.get("segment") != plan.segment_record(root, segment_index):
        raise contract.ContractError("failed startup launch identity differs from the incident")
    output = contract._regular_file(Path(slurm_output).resolve(), label="failed startup Slurm output")
    output_text = output.read_text(encoding="utf-8", errors="replace")
    required_markers = (
        "Segfault encountered",
        "worker 0 ready failed",
        "Engine core initialization failed",
        f"Muse segment {segment_index} child failed with exit code 1",
    )
    if any(marker not in output_text for marker in required_markers):
        raise contract.ContractError("Slurm output does not prove the pre-training vLLM startup failure")

    control_inventory = _inventory(control)
    run_inventory = _inventory(run)
    if not run_inventory or any(row["relative_path"].endswith("metrics.jsonl") for row in run_inventory):
        raise contract.ContractError("startup residue inventory is empty or contains trainer metrics")
    destination = root / ARCHIVE_ROOT / f"segment-{segment_index:03d}-job-{job_id}"
    if destination.exists() or destination.is_symlink():
        raise contract.ContractError(f"startup archive destination already exists: {destination}")
    destination.mkdir(parents=True)
    archived_control = destination / "control"
    archived_run = destination / "run"
    control.rename(archived_control)
    run.rename(archived_run)
    if _inventory(archived_control) != control_inventory or _inventory(archived_run) != run_inventory:
        raise contract.ContractError("archived startup residue differs after the recoverable move")

    document = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segment_index": segment_index,
        "incident": {
            "slurm_job_id": job_id,
            "state": slurm_state,
            "exit_code": slurm_exit_code,
            "reason": "pre-training-vllm-worker-0-segmentation-fault",
            "slurm_output": contract._identity(output, label="failed startup Slurm output"),
        },
        "pre_training_proof": {
            "metrics_absent": True,
            "rollouts_absent": True,
            "checkpoints_absent": True,
            "optimizer_updates": 0,
            "generation_requests": 0,
        },
        "archived_control": {
            "path": str(archived_control),
            "files": control_inventory,
        },
        "archived_run": {
            "path": str(archived_run),
            "files": run_inventory,
        },
        "parent_receipt": contract._identity(
            contract.receipt_path(root, segment_index - 1),
            label="parent Muse segment receipt",
        ),
        "boundary_amendment": contract._identity(
            amendment.amendment_path(root),
            label="Muse boundary amendment attestation",
        ),
        "archive_code": contract._identity(Path(__file__).resolve(), label="startup archive code"),
        "recovery": "retry-same-segment-from-unchanged-parent-checkpoint-and-rng-state",
        "output_token_cap": None,
    }
    receipt = destination / "recovery-receipt.json"
    with receipt.open("xb") as handle:
        handle.write(_canonical(document))
    return {"archive": str(destination), "receipt": contract._identity(receipt, label="startup archive receipt")}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--slurm-state", required=True)
    parser.add_argument("--slurm-exit-code", required=True)
    parser.add_argument("--slurm-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    result = archive(
        args.repo_root,
        args.segment_index,
        job_id=args.job_id,
        slurm_state=args.slurm_state,
        slurm_exit_code=args.slurm_exit_code,
        slurm_output=args.slurm_output,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
