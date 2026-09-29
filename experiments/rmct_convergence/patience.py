#!/usr/bin/env python3
"""Auditable best-so-far patience guard for the RMCT step-176 continuation.

The legacy step-176 receipt remains valid evidence under its original v1
controller.  A write-once amendment replays that complete receipt chain,
binds the sealed step-176 checkpoint and the capped continuation sources,
and seeds a new decision chain.  A meaningful lower weighted gap is a new best and
resets patience.  Eight consecutive eligible 16-update windows without a new
best are required before convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from experiments.rmct_convergence import controller
from experiments.rmct_convergence_r5_patience import plan as r5


AMENDMENT_SCHEMA = "rmct-convergence-capped-patience-amendment-v1"
DECISION_SCHEMA = "rmct-convergence-best-so-far-patience-decision-v1"
PATIENCE_WINDOWS = 8
MINIMUM_IMPROVEMENT = Decimal("0.01")
MINIMUM_COVERAGE = Decimal("0.85")
MAXIMUM_WEIGHTED_ABS_GAP = Decimal("0.10")
MAX_OPTIMIZER_STEPS = r5.HARD_CAP_OPTIMIZER_STEPS
AMENDMENT_RELATIVE = Path(
    "artifacts/rmct-capped-patience-20260910/amendment.json"
)
PROTECTED_RELATIVE_PATHS = (
    "experiments/rmct_convergence/patience.py",
    "experiments/rmct_convergence_r5_patience/plan.py",
    "infra/isambard/run_qwen35_rmct_convergence_r5_patience_segment.py",
    "infra/isambard/run_qwen35_rmct_convergence_r5_patience_segment.sbatch",
    "infra/isambard/submit_qwen35_rmct_convergence_r5_patience_chain.sh",
    "scripts/train_rlct.py",
    "scripts/run_experiment.py",
    "ctm/backends/base.py",
    "ctm/backends/local/vllm_sampler.py",
    "ctm/training/rl.py",
    "ctm_data/adapters/mcq_bias/shared_qid_two_bias.py",
    "infra/isambard/rmct_convergence_segment_boundary.py",
    "infra/isambard/verify_rmct_convergence_r4_recovery_production_ready.py",
)


class PatienceError(ValueError):
    """Patience evidence is incomplete, inconsistent, or ambiguous."""


@dataclass(frozen=True, slots=True)
class PatienceState:
    best_gap: Decimal
    best_segment_index: int
    best_optimizer_step: int
    consecutive_nonimproving_windows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_gap": float(self.best_gap),
            "best_optimizer_step": self.best_optimizer_step,
            "best_segment_index": self.best_segment_index,
            "consecutive_nonimproving_windows": self.consecutive_nonimproving_windows,
        }

    @classmethod
    def parse(cls, value: Any) -> "PatienceState":
        if not isinstance(value, Mapping) or set(value) != {
            "best_gap",
            "best_optimizer_step",
            "best_segment_index",
            "consecutive_nonimproving_windows",
        }:
            raise PatienceError("patience state has an unexpected envelope")
        gap = _decimal(value["best_gap"], label="state.best_gap")
        index = _integer(value["best_segment_index"], label="state.best_segment_index", minimum=0)
        step = _integer(value["best_optimizer_step"], label="state.best_optimizer_step", minimum=16)
        streak = _integer(
            value["consecutive_nonimproving_windows"],
            label="state.consecutive_nonimproving_windows",
            minimum=0,
        )
        if step != (index + 1) * controller.UPDATES_PER_SEGMENT:
            raise PatienceError("patience best optimizer step does not match its segment")
        return cls(gap, index, step, streak)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PatienceError(f"{label} must be an integer of at least {minimum}")
    return value


def _decimal(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise PatienceError(f"{label} must be a finite non-negative number")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise PatienceError(f"{label} must be a finite non-negative number")
    return result


def _root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if root.is_symlink() or not root.is_dir():
        raise PatienceError(f"repository must be a regular directory: {root}")
    return root


def _regular_file(path: str | Path, *, label: str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        raise PatienceError(f"{label} must be a regular file: {supplied}")
    resolved = supplied.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise PatienceError(f"{label} must resolve to a regular file: {supplied}")
    return resolved


def _identity(path: str | Path, *, label: str) -> dict[str, str]:
    resolved = _regular_file(path, label=label)
    return {"content_sha256": _sha256(resolved.read_bytes()), "path": str(resolved)}


def _validate_identity(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"content_sha256", "path"}:
        raise PatienceError(f"{label} must be an exact path/content_sha256 identity")
    path = _regular_file(value["path"], label=label)
    if value["content_sha256"] != _sha256(path.read_bytes()):
        raise PatienceError(f"{label} content changed after attestation: {path}")
    return path


def _policy() -> dict[str, Any]:
    return {
        "improvement_rule": "minimum_decrease_from_best_so_far_anchor",
        "minimum_improvement": float(MINIMUM_IMPROVEMENT),
        "minimum_coverage": float(MINIMUM_COVERAGE),
        "maximum_weighted_abs_gap": float(MAXIMUM_WEIGHTED_ABS_GAP),
        "patience_windows": PATIENCE_WINDOWS,
        "updates_per_window": controller.UPDATES_PER_SEGMENT,
        "patience_optimizer_steps": PATIENCE_WINDOWS * controller.UPDATES_PER_SEGMENT,
        "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
        "output_token_cap": r5.OUTPUT_TOKEN_CAP,
        "generation_termination": "legacy_stop_or_length",
    }


def _legacy_history(terminal_receipt: str | Path) -> tuple[list[dict[str, Any]], list[Path]]:
    terminal = _regular_file(terminal_receipt, label="legacy terminal receipt")
    documents: list[dict[str, Any]] = []
    paths: list[Path] = []
    seen: set[Path] = set()
    current: Path | None = terminal
    while current is not None:
        if current in seen:
            raise PatienceError("legacy receipt chain contains a cycle")
        seen.add(current)
        try:
            document = controller.verify_decision_receipt(current)
        except controller.ConvergenceError as exc:
            raise PatienceError(f"legacy receipt does not replay: {exc}") from exc
        documents.append(document)
        paths.append(current)
        predecessor = document.get("predecessor_receipt")
        current = None if predecessor is None else _validate_identity(predecessor, label="legacy predecessor")
    documents.reverse()
    paths.reverse()
    indices = [document["target"]["segment_index"] for document in documents]
    if indices != list(range(r5.PARENT_SEGMENT_INDEX + 1)):
        raise PatienceError(f"legacy history must contain segments 0..{r5.PARENT_SEGMENT_INDEX}: {indices}")
    last = documents[-1]
    if (
        last.get("decision") != "converged"
        or last.get("afterok") != {"permit_training": False, "successor_action": "no_op"}
        or last["target"].get("optimizer_step_end") != 176
    ):
        raise PatienceError("amendment accepts only the sealed legacy step-176 convergence receipt")
    return documents, paths


def _state_from_history(documents: Sequence[Mapping[str, Any]]) -> PatienceState:
    best_gap: Decimal | None = None
    best_index = -1
    streak = 0
    for document in documents:
        index = int(document["target"]["segment_index"])
        gap = _decimal(document["window"]["current"]["weighted_abs_gap"], label=f"legacy gap {index}")
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_index = index
            streak = 0
        else:
            streak += 1
    assert best_gap is not None
    return PatienceState(best_gap, best_index, (best_index + 1) * controller.UPDATES_PER_SEGMENT, streak)


def build_amendment(repository: str | Path, *, terminal_receipt: str | Path) -> dict[str, Any]:
    root = _root(repository)
    documents, paths = _legacy_history(terminal_receipt)
    protected = []
    for relative in PROTECTED_RELATIVE_PATHS:
        path = root / relative
        protected.append({"relative_path": relative, **_identity(path, label=f"protected source {relative}")})
    from infra.isambard import verify_rmct_convergence_r4_recovery_production_ready as r4_ready

    try:
        checkpoint = r4_ready._strict_checkpoint_identity(
            root,
            r5.parent_checkpoint_path(root),
            expected_segment_index=r5.PARENT_SEGMENT_INDEX,
            label="r5 sealed step-176 parent",
        )
    except Exception as exc:
        raise PatienceError(f"sealed r4 step-176 checkpoint is not resumable: {exc}") from exc
    history = [
        {
            "segment_index": document["target"]["segment_index"],
            "optimizer_step": document["target"]["optimizer_step_end"],
            "weighted_abs_gap": document["window"]["current"]["weighted_abs_gap"],
            "receipt": _identity(path, label="legacy decision receipt"),
        }
        for document, path in zip(documents, paths)
    ]
    return {
        "schema": AMENDMENT_SCHEMA,
        "scope": "post-step-176-generation-policy-and-convergence-controller-only",
        "authorization": {
            "requested_outcome": "train_until_no_improvement_for_a_long_time",
            "historical_max_nonimproving_windows": 4,
            "approved_patience_multiplier": 2,
            "approved_patience_windows": PATIENCE_WINDOWS,
            "output_token_cap_approved": True,
            "approved_output_token_cap": r5.OUTPUT_TOKEN_CAP,
        },
        "policy": _policy(),
        "legacy_terminal_receipt": _identity(paths[-1], label="legacy terminal receipt"),
        "legacy_history": history,
        "bootstrap_state": {
            **_state_from_history(documents).as_dict(),
            "consecutive_nonimproving_windows": 0,
        },
        "parent_checkpoint": checkpoint,
        "protected_sources": protected,
    }


def _immutable(path: Path, document: Mapping[str, Any], *, label: str) -> str:
    payload = _canonical(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PatienceError(f"{label} parent must be a regular directory")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise PatienceError(f"refusing to overwrite different immutable {label}: {path}")
        return "resumed"
    return "written"


def write_amendment(
    repository: str | Path, *, terminal_receipt: str | Path, output: str | Path
) -> dict[str, Any]:
    document = build_amendment(repository, terminal_receipt=terminal_receipt)
    path = Path(output).resolve()
    status = _immutable(path, document, label="patience amendment")
    return {"path": str(path), "status": status, "content_sha256": _sha256(path.read_bytes())}


def verify_amendment(repository: str | Path, amendment: str | Path) -> dict[str, Any]:
    path = _regular_file(amendment, label="patience amendment")
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PatienceError("patience amendment is not valid JSON") from exc
    if not isinstance(recorded, Mapping) or recorded.get("schema") != AMENDMENT_SCHEMA:
        raise PatienceError("patience amendment has an unexpected schema")
    terminal = _validate_identity(recorded.get("legacy_terminal_receipt"), label="legacy terminal receipt")
    expected = build_amendment(repository, terminal_receipt=terminal)
    if recorded != expected:
        raise PatienceError("patience amendment does not match the current protected evidence")
    return dict(recorded)


def _coverage_passes(summary: controller.SegmentSummary) -> bool:
    pooled = summary.pooled_coverage
    if pooled.parse_coverage < MINIMUM_COVERAGE or pooled.valid_coverage < MINIMUM_COVERAGE:
        return False
    return all(
        item.parse_coverage >= MINIMUM_COVERAGE and item.valid_coverage >= MINIMUM_COVERAGE
        for item in summary.coverage_by_bias.values()
    )


def _afterok(decision: str) -> dict[str, Any]:
    if decision == "continue":
        return {"permit_training": True, "successor_action": "launch"}
    return {"permit_training": False, "successor_action": "no_op"}


def _next_state(before: PatienceState, summary: controller.SegmentSummary) -> tuple[PatienceState, bool, bool]:
    eligible = _coverage_passes(summary)
    if not eligible:
        # Insufficient evidence breaks the streak: eight separated eligible
        # windows are not eight consecutive eligible windows.
        return PatienceState(before.best_gap, before.best_segment_index, before.best_optimizer_step, 0), False, False
    gap = summary.weighted_abs_gap
    if gap < before.best_gap and before.best_gap - gap >= MINIMUM_IMPROVEMENT:
        return PatienceState(gap, summary.identity.segment_index, summary.identity.optimizer_step_end, 0), True, True
    return PatienceState(
        before.best_gap,
        before.best_segment_index,
        before.best_optimizer_step,
        before.consecutive_nonimproving_windows + 1,
    ), False, True


def _build_decision(
    *,
    amendment_identity: Mapping[str, str],
    source: controller.ParsedSource,
    source_identity: Mapping[str, str],
    checkpoint_identity: Mapping[str, str],
    completion_identity: Mapping[str, str],
    predecessor_identity: Mapping[str, str] | None,
    state_before: PatienceState,
) -> dict[str, Any]:
    state_after, new_best, eligible = _next_state(state_before, source.summary)
    if eligible and state_after.consecutive_nonimproving_windows >= PATIENCE_WINDOWS and state_after.best_gap <= MAXIMUM_WEIGHTED_ABS_GAP:
        decision = "converged"
        reason = "best_so_far_patience_exhausted"
    else:
        decision = "continue"
        reason = "new_best_resets_patience" if new_best else ("patience_accumulating" if eligible else "coverage_ineligible")
    config = controller.WindowThresholds(
        minimum_coverage=MINIMUM_COVERAGE,
        maximum_weighted_abs_gap=MAXIMUM_WEIGHTED_ABS_GAP,
        maximum_abs_change=Decimal("0"),
        # This object formats coverage only; it does not control termination.
        max_optimizer_steps=source.identity.optimizer_step_end,
    )
    return {
        "schema": DECISION_SCHEMA,
        "decision": decision,
        "reason": reason,
        "afterok": _afterok(decision),
        "target": source.identity.as_dict(),
        "policy": _policy(),
        "amendment": dict(amendment_identity),
        "predecessor_patience_receipt": None if predecessor_identity is None else dict(predecessor_identity),
        "source_metrics": dict(source_identity),
        "checkpoint_receipt": dict(checkpoint_identity),
        "completion_receipt": dict(completion_identity),
        "window": {
            "current": source.summary.as_dict(config=config),
            "coverage_eligible": eligible,
            "new_global_best": new_best,
            "state_before": state_before.as_dict(),
            "state_after": state_after.as_dict(),
        },
    }


def _decision_filename(index: int, digest: str) -> str:
    return f"patience-decision-s{index:03d}-{digest}.json"


def _publish_decision(directory: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    target = controller.SegmentIdentity.parse(document["target"], label="patience target")
    payload = _canonical(document)
    digest = _sha256(payload)
    output = Path(directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or not output.is_dir():
        raise PatienceError("patience decision directory must be regular")
    prefix = f"patience-decision-s{target.segment_index:03d}-"
    expected_name = _decision_filename(target.segment_index, digest)
    competitors = [path for path in output.iterdir() if path.name.startswith(prefix) and path.name != expected_name]
    if competitors:
        raise PatienceError(f"competing patience decision exists: {competitors[0]}")
    path = output / expected_name
    status = _immutable(path, document, label="patience decision")
    return {"decision": document["decision"], "path": str(path), "status": status}


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PatienceError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PatienceError(f"{label} must contain one JSON object")
    return value


def _load_patience_receipt(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = _regular_file(path, label="patience decision receipt")
    document = _load_json(resolved, label="patience decision receipt")
    target = controller.SegmentIdentity.parse(document.get("target"), label="patience target")
    payload_digest = _sha256(resolved.read_bytes())
    if resolved.name != _decision_filename(target.segment_index, payload_digest):
        raise PatienceError("patience receipt filename is not content-addressed")
    return document, resolved


def _predecessor_state(
    repository: Path,
    amendment: Path,
    target: controller.SegmentIdentity,
    predecessor_receipt: str | Path | None,
) -> tuple[PatienceState, dict[str, str] | None]:
    amendment_document = verify_amendment(repository, amendment)
    if target.segment_index == r5.START_SEGMENT_INDEX:
        if predecessor_receipt is not None:
            raise PatienceError("first r5 patience window must not have a patience predecessor")
        return PatienceState.parse(amendment_document["bootstrap_state"]), None
    if predecessor_receipt is None:
        raise PatienceError("later r5 patience windows require their immediate predecessor")
    previous = verify_decision_receipt(repository, predecessor_receipt)
    previous_target = controller.SegmentIdentity.parse(previous["target"], label="previous patience target")
    if previous_target.segment_index + 1 != target.segment_index or previous_target.optimizer_step_end + 1 != target.optimizer_step_start:
        raise PatienceError("patience predecessor is not the immediately preceding window")
    if previous["decision"] != "continue":
        raise PatienceError("cannot continue after a terminal patience decision")
    return PatienceState.parse(previous["window"]["state_after"]), _identity(predecessor_receipt, label="patience predecessor")


def guard_from_paths(
    repository: str | Path,
    *,
    amendment: str | Path,
    source_metrics: str | Path,
    checkpoint_receipt: str | Path,
    completion_receipt: str | Path,
    output_directory: str | Path,
    predecessor_receipt: str | Path | None = None,
) -> dict[str, Any]:
    root = _root(repository)
    amendment_path = _regular_file(amendment, label="patience amendment")
    verify_amendment(root, amendment_path)
    try:
        source, source_identity = controller._load_source(source_metrics)
        checkpoint_identity = controller._validate_checkpoint_receipt(checkpoint_receipt, identity=source.identity)
        completion_identity = controller._validate_completion_receipt(completion_receipt, identity=source.identity)
    except controller.ConvergenceError as exc:
        raise PatienceError(str(exc)) from exc
    if source.identity.segment_index < r5.START_SEGMENT_INDEX:
        raise PatienceError("patience guard accepts only r5 continuation segments")
    before, predecessor_identity = _predecessor_state(
        root, amendment_path, source.identity, predecessor_receipt
    )
    document = _build_decision(
        amendment_identity=_identity(amendment_path, label="patience amendment"),
        source=source,
        source_identity=source_identity,
        checkpoint_identity=checkpoint_identity,
        completion_identity=completion_identity,
        predecessor_identity=predecessor_identity,
        state_before=before,
    )
    return _publish_decision(output_directory, document)


def verify_decision_receipt(
    repository: str | Path, receipt: str | Path, *, _seen: set[Path] | None = None
) -> dict[str, Any]:
    root = _root(repository)
    document, path = _load_patience_receipt(receipt)
    seen = set() if _seen is None else _seen
    if path in seen:
        raise PatienceError("patience predecessor chain contains a cycle")
    seen.add(path)
    try:
        if set(document) != {
            "schema", "decision", "reason", "afterok", "target", "policy", "amendment",
            "predecessor_patience_receipt", "source_metrics", "checkpoint_receipt", "completion_receipt", "window",
        } or document.get("schema") != DECISION_SCHEMA or document.get("policy") != _policy():
            raise PatienceError("patience decision has an unexpected envelope or policy")
        target = controller.SegmentIdentity.parse(document["target"], label="patience target")
        amendment_path = _validate_identity(document["amendment"], label="patience amendment")
        verify_amendment(root, amendment_path)
        source_path = _validate_identity(document["source_metrics"], label="source metrics")
        checkpoint_path = _validate_identity(document["checkpoint_receipt"], label="checkpoint receipt")
        completion_path = _validate_identity(document["completion_receipt"], label="completion receipt")
        try:
            source, source_identity = controller._load_source(source_path, error_type=controller.DecisionReceiptError)
            checkpoint_identity = controller._validate_checkpoint_receipt(checkpoint_path, identity=target)
            completion_identity = controller._validate_completion_receipt(completion_path, identity=target)
        except controller.ConvergenceError as exc:
            raise PatienceError(str(exc)) from exc
        if source.identity != target or source_identity != document["source_metrics"]:
            raise PatienceError("patience source metrics do not match target")
        predecessor_value = document["predecessor_patience_receipt"]
        if target.segment_index == r5.START_SEGMENT_INDEX:
            before, predecessor_identity = _predecessor_state(root, amendment_path, target, None)
            if predecessor_value is not None:
                raise PatienceError("first r5 decision cannot bind a patience predecessor")
        else:
            predecessor_path = _validate_identity(predecessor_value, label="patience predecessor")
            previous = verify_decision_receipt(root, predecessor_path, _seen=seen)
            previous_target = controller.SegmentIdentity.parse(previous["target"], label="previous patience target")
            if previous_target.segment_index + 1 != target.segment_index or previous["decision"] != "continue":
                raise PatienceError("patience predecessor is not the immediate continuing window")
            before = PatienceState.parse(previous["window"]["state_after"])
            predecessor_identity = _identity(predecessor_path, label="patience predecessor")
        expected = _build_decision(
            amendment_identity=_identity(amendment_path, label="patience amendment"),
            source=source,
            source_identity=source_identity,
            checkpoint_identity=checkpoint_identity,
            completion_identity=completion_identity,
            predecessor_identity=predecessor_identity,
            state_before=before,
        )
        if expected != document:
            raise PatienceError("patience decision does not replay from its evidence")
        return document
    finally:
        seen.remove(path)


def decision_in_directory(repository: str | Path, directory: str | Path, segment_index: int) -> dict[str, Any]:
    if isinstance(segment_index, bool) or not isinstance(segment_index, int):
        raise PatienceError("segment index must be an integer")
    target = Path(directory).resolve()
    if target.is_symlink() or not target.is_dir():
        raise PatienceError("patience decision directory must be regular")
    matches = sorted(target.glob(f"patience-decision-s{segment_index:03d}-*.json"))
    if len(matches) != 1:
        raise PatienceError(f"expected exactly one patience decision for segment {segment_index}, found {len(matches)}")
    document = verify_decision_receipt(repository, matches[0])
    return {"path": str(matches[0]), "decision": document["decision"], "afterok": document["afterok"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--repository", required=True, type=Path)
    bootstrap.add_argument("--terminal-receipt", required=True, type=Path)
    bootstrap.add_argument("--output", required=True, type=Path)
    verify_amend = commands.add_parser("verify-amendment")
    verify_amend.add_argument("--repository", required=True, type=Path)
    verify_amend.add_argument("--amendment", required=True, type=Path)
    guard = commands.add_parser("guard")
    guard.add_argument("--repository", required=True, type=Path)
    guard.add_argument("--amendment", required=True, type=Path)
    guard.add_argument("--source-metrics", required=True, type=Path)
    guard.add_argument("--checkpoint-receipt", required=True, type=Path)
    guard.add_argument("--completion-receipt", required=True, type=Path)
    guard.add_argument("--output-directory", required=True, type=Path)
    guard.add_argument("--predecessor-receipt", type=Path)
    verify = commands.add_parser("verify-receipt")
    verify.add_argument("--repository", required=True, type=Path)
    verify.add_argument("--receipt", required=True, type=Path)
    decision = commands.add_parser("decision")
    decision.add_argument("--repository", required=True, type=Path)
    decision.add_argument("--directory", required=True, type=Path)
    decision.add_argument("--segment-index", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "bootstrap":
            result = write_amendment(args.repository, terminal_receipt=args.terminal_receipt, output=args.output)
        elif args.command == "verify-amendment":
            result = verify_amendment(args.repository, args.amendment)
        elif args.command == "guard":
            result = guard_from_paths(
                args.repository,
                amendment=args.amendment,
                source_metrics=args.source_metrics,
                checkpoint_receipt=args.checkpoint_receipt,
                completion_receipt=args.completion_receipt,
                output_directory=args.output_directory,
                predecessor_receipt=args.predecessor_receipt,
            )
        elif args.command == "verify-receipt":
            result = verify_decision_receipt(args.repository, args.receipt)
        elif args.command == "decision":
            result = decision_in_directory(args.repository, args.directory, args.segment_index)
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (OSError, PatienceError, controller.ConvergenceError) as exc:
        parser.error(str(exc))
    print(_canonical(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AMENDMENT_RELATIVE",
    "AMENDMENT_SCHEMA",
    "DECISION_SCHEMA",
    "MAX_OPTIMIZER_STEPS",
    "PATIENCE_WINDOWS",
    "PatienceError",
    "PatienceState",
    "build_amendment",
    "decision_in_directory",
    "guard_from_paths",
    "verify_amendment",
    "verify_decision_receipt",
    "write_amendment",
]
