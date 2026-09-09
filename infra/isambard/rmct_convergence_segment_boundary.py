#!/usr/bin/env python3
"""Small sealed-boundary helpers for the RMCT convergence Isambard launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctm.training.resume_state import load_strict_local_rl_resume_state
from experiments.rmct_convergence import controller


MARKER_SCHEMA = "rmct-convergence-training-started-v1"


class BoundaryError(ValueError):
    """A segment cannot safely be started, sealed, or replayed."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _identity(path: Path, *, label: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise BoundaryError(f"{label} must be a regular file: {path}")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _immutable(path: Path, document: Mapping[str, Any], *, label: str) -> str:
    payload = _canonical(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise BoundaryError(f"{label} parent must be a regular directory: {path.parent}")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise BoundaryError(f"refusing to overwrite different immutable {label}: {path}")
        return "resumed"
    return "written"


def _index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 32:
        raise BoundaryError("segment-index must be an integer in [0, 31]")
    return value


def seal(*, checkpoint: Path, segment_index: int, checkpoint_receipt: Path, completion_receipt: Path) -> dict[str, Any]:
    index = _index(segment_index)
    end = (index + 1) * controller.UPDATES_PER_SEGMENT
    # The local-RL loop state is the common sealed boundary.  A phase-shared
    # backend may add replica provenance beside it, but it cannot weaken this
    # rank-zero final/optimizer/RNG validation.
    state = load_strict_local_rl_resume_state(checkpoint)
    if state.global_step != end or state.optimizer_step != end:
        raise BoundaryError(
            f"final checkpoint does not seal segment {index}: global={state.global_step}, optimizer={state.optimizer_step}, expected={end}"
        )
    checkpoint_document = {
        "schema": controller.CHECKPOINT_RECEIPT_SCHEMA,
        "segment_index": index,
        "optimizer_step": end,
        "sealed": True,
    }
    completion_document = {
        "schema": controller.COMPLETION_RECEIPT_SCHEMA,
        "segment_index": index,
        "optimizer_step_start": index * controller.UPDATES_PER_SEGMENT + 1,
        "optimizer_step_end": end,
        "optimizer_steps": controller.UPDATES_PER_SEGMENT,
        "sealed": True,
    }
    return {
        "checkpoint_receipt": str(checkpoint_receipt.resolve()),
        "checkpoint_status": _immutable(checkpoint_receipt, checkpoint_document, label="checkpoint receipt"),
        "completion_receipt": str(completion_receipt.resolve()),
        "completion_status": _immutable(completion_receipt, completion_document, label="completion receipt"),
    }


def mark(*, path: Path, segment_index: int, command_attestation: Path, ready_receipt: Path) -> dict[str, Any]:
    index = _index(segment_index)
    document = {
        "schema": MARKER_SCHEMA,
        "segment_index": index,
        "command_attestation": _identity(command_attestation, label="command attestation"),
        "ready_receipt": _identity(ready_receipt, label="production readiness receipt"),
    }
    return {"path": str(path.resolve()), "status": _immutable(path, document, label="training-started marker")}


def decision(*, directory: Path, segment_index: int) -> dict[str, Any]:
    index = _index(segment_index)
    if directory.is_symlink() or not directory.is_dir():
        raise BoundaryError(f"decision directory must be a regular directory: {directory}")
    matches = sorted(directory.glob(f"checkpoint-window-decision-s{index:03d}-*.json"))
    if len(matches) != 1:
        raise BoundaryError(f"expected exactly one decision receipt for segment {index}, found {len(matches)}")
    receipt = controller.verify_decision_receipt(matches[0])
    return {"path": str(matches[0].resolve()), "decision": receipt["decision"], "afterok": receipt["afterok"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal_parser = commands.add_parser("seal", help="validate a final local checkpoint and write exact controller receipts")
    seal_parser.add_argument("--checkpoint", required=True, type=Path)
    seal_parser.add_argument("--segment-index", required=True, type=int)
    seal_parser.add_argument("--checkpoint-receipt", required=True, type=Path)
    seal_parser.add_argument("--completion-receipt", required=True, type=Path)
    mark_parser = commands.add_parser("mark", help="write a segment training-started marker exactly once")
    mark_parser.add_argument("--path", required=True, type=Path)
    mark_parser.add_argument("--segment-index", required=True, type=int)
    mark_parser.add_argument("--command-attestation", required=True, type=Path)
    mark_parser.add_argument("--ready-receipt", required=True, type=Path)
    decision_parser = commands.add_parser("decision", help="replay the one content-addressed decision for a segment")
    decision_parser.add_argument("--directory", required=True, type=Path)
    decision_parser.add_argument("--segment-index", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "seal":
            result = seal(
                checkpoint=args.checkpoint,
                segment_index=args.segment_index,
                checkpoint_receipt=args.checkpoint_receipt,
                completion_receipt=args.completion_receipt,
            )
        elif args.command == "mark":
            result = mark(
                path=args.path,
                segment_index=args.segment_index,
                command_attestation=args.command_attestation,
                ready_receipt=args.ready_receipt,
            )
        elif args.command == "decision":
            result = decision(directory=args.directory, segment_index=args.segment_index)
        else:  # pragma: no cover - argparse makes this unreachable
            raise AssertionError(args.command)
    except (BoundaryError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(_canonical(result).decode(), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
