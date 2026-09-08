#!/usr/bin/env python3
"""Run exactly one uncapped r5 RMCT window under the patience amendment."""

from __future__ import annotations

import argparse
import fcntl
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

from experiments.rmct_convergence import controller, patience  # noqa: E402
from experiments.rmct_convergence_r5_patience import plan  # noqa: E402
from infra.isambard import rmct_convergence_segment_boundary as boundary  # noqa: E402
from infra.isambard import verify_rmct_convergence_r4_recovery_production_ready as r4_ready  # noqa: E402


MARKER_SCHEMA = "rmct-convergence-uncapped-patience-training-started-v1"


class RunError(ValueError):
    """The requested r5 segment is not safe to run or resume."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path, *, label: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RunError(f"{label} must be a regular file: {path}")
    return {"content_sha256": _sha256(path), "path": str(path.resolve())}


def _json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RunError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RunError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RunError(f"{label} must contain one object")
    return value


def _immutable(path: Path, document: Mapping[str, Any], *, label: str) -> str:
    payload = _canonical(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RunError(f"refusing to overwrite different immutable {label}: {path}")
        return "resumed"
    return "written"


def _model_snapshot() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise RunError("HF_HOME is required for the pinned offline Qwen snapshot")
    snapshot = Path(hf_home).resolve() / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots" / plan.BASE_SNAPSHOT
    if snapshot.is_symlink() or not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise RunError(f"pinned offline Qwen snapshot is absent: {snapshot}")
    return snapshot


def _validate_four_gpus() -> None:
    tokens = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if len(tokens) != 4 or any(not value or value in {"-1", "NoDevFiles"} for value in tokens) or len(set(tokens)) != 4:
        raise RunError("r5 RMCT requires exactly four distinct Slurm-visible GPUs")


def _paths(root: Path, index: int) -> dict[str, Path]:
    run = plan.run_name(index)
    run_root = root / "logs" / plan.CONDITION_NAME / run
    segment = run_root / "segment"
    return {
        "run_root": run_root,
        "segment": segment,
        "checkpoint": plan.final_checkpoint_path(root, index),
        "command": segment / "training-command.json",
        "marker": segment / "training-started.json",
        "checkpoint_receipt": segment / "checkpoint-receipt.json",
        "completion_receipt": segment / "completion-receipt.json",
        "source_metrics": segment / "rmct-convergence-source-metrics.json",
        "decisions": run_root / "patience-decisions",
        "metrics": run_root / "metrics.jsonl",
        "lock": segment / "segment.lock",
    }


def _previous_receipt(root: Path, index: int) -> Path | None:
    if index == plan.START_SEGMENT_INDEX:
        return None
    previous = index - 1
    directory = _paths(root, previous)["decisions"]
    result = patience.decision_in_directory(root, directory, previous)
    if result["decision"] != "continue":
        return Path(result["path"])
    return Path(result["path"])


def _prior_terminal(root: Path, index: int) -> dict[str, Any] | None:
    """Return any earlier terminal decision, including across skipped no-op jobs."""

    for previous in range(plan.START_SEGMENT_INDEX, index):
        directory = _paths(root, previous)["decisions"]
        if not directory.exists():
            continue
        result = patience.decision_in_directory(root, directory, previous)
        if result["decision"] != "continue":
            return {"segment_index": previous, **result}
    return None


def _validate_parent(root: Path, index: int) -> None:
    previous = index - 1
    checkpoint = plan.parent_checkpoint_path(root) if index == plan.START_SEGMENT_INDEX else plan.final_checkpoint_path(root, previous)
    try:
        r4_ready._strict_checkpoint_identity(
            root,
            checkpoint,
            expected_segment_index=previous,
            label=f"r5 parent s{previous + 1:03d}",
        )
    except Exception as exc:
        raise RunError(f"r5 parent checkpoint is not strictly resumable: {exc}") from exc


def _validate_custody(root: Path, index: int, amendment: Path, snapshot: Path) -> dict[str, Any]:
    paths = _paths(root, index)
    command = _json(paths["command"], label="r5 training command")
    expected_command = plan.command_attestation(root, index, model_snapshot=snapshot)
    if command != expected_command:
        raise RunError("r5 training command differs from the uncapped frozen plan")
    marker = _json(paths["marker"], label="r5 training-started marker")
    expected_marker = {
        "schema": MARKER_SCHEMA,
        "segment_index": index,
        "command_attestation": _identity(paths["command"], label="r5 training command"),
        "patience_amendment": _identity(amendment, label="patience amendment"),
    }
    if marker != expected_marker:
        raise RunError("r5 training-started marker does not bind the exact command and amendment")
    try:
        checkpoint = r4_ready._strict_checkpoint_identity(
            root,
            paths["checkpoint"],
            expected_segment_index=index,
            label=f"r5 s{index + 1:03d}",
        )
    except Exception as exc:
        raise RunError(f"r5 checkpoint is not strictly resumable: {exc}") from exc
    return {"command": expected_marker["command_attestation"], "marker": marker, "checkpoint": checkpoint}


def run_segment(root_value: str | Path, index: int, *, amendment_value: str | Path, yes: bool) -> dict[str, Any]:
    if not yes:
        raise RunError("real r5 work requires --yes")
    root = Path(root_value).resolve()
    if root.is_symlink() or not root.is_dir():
        raise RunError("repository must be a regular directory")
    if not plan.START_SEGMENT_INDEX <= index < plan.TOTAL_SEGMENTS:
        raise RunError(f"segment index must be in [{plan.START_SEGMENT_INDEX}, {plan.TOTAL_SEGMENTS - 1}]")
    amendment = Path(amendment_value).resolve()
    patience.verify_amendment(root, amendment)
    _validate_four_gpus()
    snapshot = _model_snapshot()
    paths = _paths(root, index)

    if paths["decisions"].is_dir():
        try:
            existing = patience.decision_in_directory(root, paths["decisions"], index)
        except patience.PatienceError:
            existing = None
        if existing is not None:
            return {"status": "reused", **existing}

    terminal = _prior_terminal(root, index)
    if terminal is not None:
        return {
            "status": "no_op",
            "reason": "earlier_terminal_patience_decision",
            "terminal": terminal,
        }
    predecessor = _previous_receipt(root, index)
    if predecessor is not None:
        previous = patience.verify_decision_receipt(root, predecessor)
        if previous["decision"] != "continue":
            return {
                "status": "no_op",
                "reason": "terminal_patience_predecessor",
                "predecessor": str(predecessor),
            }
    _validate_parent(root, index)

    paths["segment"].mkdir(parents=True, exist_ok=True)
    with paths["lock"].open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunError(f"another process owns r5 segment {index}") from exc

        if paths["checkpoint"].exists():
            _validate_custody(root, index, amendment, snapshot)
        elif paths["marker"].exists() or paths["marker"].is_symlink():
            raise RunError("r5 has a training-started marker but no final checkpoint; refusing ambiguous replay")
        else:
            command = plan.command_attestation(root, index, model_snapshot=snapshot)
            plan.write_command(paths["command"], command)
            marker = {
                "schema": MARKER_SCHEMA,
                "segment_index": index,
                "command_attestation": _identity(paths["command"], label="r5 training command"),
                "patience_amendment": _identity(amendment, label="patience amendment"),
            }
            _immutable(paths["marker"], marker, label="r5 training-started marker")
            exit_code = plan.execute(
                root,
                index,
                model_snapshot=snapshot,
                command_output=paths["command"],
                yes=True,
            )
            if exit_code:
                raise RunError(f"r5 training process exited with status {exit_code}")
            _validate_custody(root, index, amendment, snapshot)

        boundary.seal(
            checkpoint=paths["checkpoint"],
            segment_index=index,
            checkpoint_receipt=paths["checkpoint_receipt"],
            completion_receipt=paths["completion_receipt"],
        )
        controller.extract_source_metrics(
            metrics_jsonl=paths["metrics"],
            segment_index=index,
            output=paths["source_metrics"],
        )
        decision = patience.guard_from_paths(
            root,
            amendment=amendment,
            source_metrics=paths["source_metrics"],
            checkpoint_receipt=paths["checkpoint_receipt"],
            completion_receipt=paths["completion_receipt"],
            output_directory=paths["decisions"],
            predecessor_receipt=predecessor,
        )
        return {"status": "sealed", "segment_index": index, **decision}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--segment-index", required=True, type=int)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--yes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_segment(args.repository, args.segment_index, amendment_value=args.amendment, yes=args.yes)
    except (OSError, RunError, patience.PatienceError, controller.ConvergenceError) as exc:
        parser.error(str(exc))
    print(_canonical(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
