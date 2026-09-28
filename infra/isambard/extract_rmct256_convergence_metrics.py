#!/usr/bin/env python3
"""Extract one strict 16-update RMCT-256 convergence metric source receipt.

``train_rlct`` writes the authoritative per-update aggregates through Tinker's
local JSON logger.  This helper does not infer success counts or assume fixed
rollout denominators.  Instead it binds the completed segment receipt to the
raw logger file, selects exactly the 16 absolute global steps owned by that
segment, and re-emits the only three trainer metrics used by the plateau
controller:

* ``train/consistency_gap_abs_sum_1``
* ``train/consistency_gap_abs_count_1``
* ``train/consistency_gap_abs_mean_1``

Each update must cover all four questions in its batch.  Any parsed/resampled
shortfall is a failed convergence observation, not a numerical zero or an
invented 96-rollout denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.isambard import rmct256_convergence_segment_contract as contract  # noqa: E402


SOURCE_RECEIPT_SCHEMA = "rmct256-convergence-extracted-metrics-source-v1"
METRIC_KEYS = (
    "train/consistency_gap_abs_sum_1",
    "train/consistency_gap_abs_count_1",
    "train/consistency_gap_abs_mean_1",
)


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise contract.ContractError(f"{label} must be a regular file: {path}")


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _immutable_bytes(path: Path, payload: bytes, *, label: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise contract.ContractError(f"{label} parent must not be a symlink: {path.parent}")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        _regular_file(path, label=label)
        if path.read_bytes() != payload:
            raise contract.ContractError(f"refusing to overwrite different {label}: {path}")
        return "resumed"
    return "written"


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise contract.ContractError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise contract.ContractError(f"{label} must be finite")
    return number


def _exact_count(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise contract.ContractError(f"{label} must be an integer")
    if value != 4:
        raise contract.ContractError(f"{label} must be exactly 4 questions, got {value}")
    return value


def _load_metrics(path: Path, segment: contract.Segment) -> list[dict[str, Any]]:
    """Select and validate the one authoritative metrics record per update."""

    _regular_file(path, label="trainer metrics JSONL")
    expected_steps = list(
        range(
            segment.global_index * contract.UPDATES_PER_SEGMENT + 1,
            segment.checkpoint_step + 1,
        )
    )
    selected: dict[int, dict[str, Any]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            raise contract.ContractError(f"trainer metrics JSONL contains a blank line at {line_number}")
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise contract.ContractError(f"trainer metrics JSONL has invalid JSON at line {line_number}") from exc
        if not isinstance(record, Mapping):
            raise contract.ContractError(f"trainer metrics JSONL line {line_number} is not an object")
        present = [key for key in METRIC_KEYS if key in record]
        if not present:
            continue
        if len(present) != len(METRIC_KEYS):
            raise contract.ContractError(f"trainer metrics line {line_number} has a partial consistency-gap aggregate")
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise contract.ContractError(f"trainer metrics line {line_number} has no integer global step")
        if step not in expected_steps:
            raise contract.ContractError(
                f"trainer metrics line {line_number} has consistency data outside segment steps: {step}"
            )
        if step in selected:
            raise contract.ContractError(f"trainer metrics JSONL has duplicate consistency data for global step {step}")
        absolute_sum = _finite_float(record[METRIC_KEYS[0]], label=f"trainer metrics step {step} absolute sum")
        absolute_count = _exact_count(record[METRIC_KEYS[1]], label=f"trainer metrics step {step} absolute count")
        absolute_mean = _finite_float(record[METRIC_KEYS[2]], label=f"trainer metrics step {step} absolute mean")
        if not math.isclose(absolute_mean, absolute_sum / absolute_count, rel_tol=1e-12, abs_tol=1e-12):
            raise contract.ContractError(f"trainer metrics step {step} absolute mean does not equal sum/count")
        selected[step] = {
            "global_step": step,
            "metrics": {
                METRIC_KEYS[0]: absolute_sum,
                METRIC_KEYS[1]: absolute_count,
                METRIC_KEYS[2]: absolute_mean,
            },
        }
    missing = [step for step in expected_steps if step not in selected]
    if missing:
        raise contract.ContractError(f"trainer metrics JSONL is missing consistency aggregates for global steps: {missing}")
    if len(selected) != contract.UPDATES_PER_SEGMENT:
        raise contract.ContractError("trainer metrics JSONL does not contain exactly 16 consistency updates")
    return [selected[step] for step in expected_steps]


def _source_receipt_path(root: Path, segment: contract.Segment) -> Path:
    return contract.run_root(root, segment) / "segment" / "rmct256-convergence-training-gap-source-receipt.json"


def _normalized_metrics_path(root: Path, segment: contract.Segment) -> Path:
    return contract.run_root(root, segment) / "segment" / "rmct256-convergence-gap-metrics.jsonl"


def _compact_identity(path: Path, *, label: str) -> dict[str, str]:
    identity = contract.file_identity(path, label=label)
    return {"path": str(identity["path"]), "content_sha256": str(identity["sha256"])}


def _segment_question_ids(
    root: Path,
    plan_contract: Mapping[str, Any],
    segment: contract.Segment,
) -> list[str]:
    """Load the sealed ordered 64-question block from its segment manifest."""

    metadata = plan_contract.get("metadata")
    if not isinstance(metadata, Mapping):
        raise contract.ContractError("segment plan has no convergence metadata")
    raw_path = metadata.get("segment_manifest")
    expected_sha256 = metadata.get("segment_manifest_sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_sha256, str):
        raise contract.ContractError("segment plan has no sealed segment-manifest identity")
    manifest_path = contract._under_root(root, raw_path)
    identity = _compact_identity(manifest_path, label="segment manifest")
    if identity["content_sha256"] != expected_sha256:
        raise contract.ContractError("segment manifest content hash differs from the compiled segment metadata")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise contract.ContractError("segment manifest is not valid JSON") from exc
    if not isinstance(document, Mapping) or document.get("kind") != "rmct256_convergence_segments_manifest":
        raise contract.ContractError("segment manifest has an unexpected schema")
    segments = document.get("segments")
    if not isinstance(segments, list):
        raise contract.ContractError("segment manifest has no segments")
    matches = [entry for entry in segments if isinstance(entry, Mapping) and entry.get("index") == segment.segment_index]
    if len(matches) != 1:
        raise contract.ContractError("segment manifest has no unique block for this segment index")
    entry = matches[0]
    if entry.get("row_offset") != segment.row_offset or entry.get("row_count") != contract.ROWS_PER_SEGMENT:
        raise contract.ContractError("segment manifest block does not match the compiled row slice")
    question_ids = entry.get("question_ids")
    if (
        not isinstance(question_ids, list)
        or len(question_ids) != contract.ROWS_PER_SEGMENT
        or any(not isinstance(value, str) or not value for value in question_ids)
        or len(set(question_ids)) != contract.ROWS_PER_SEGMENT
    ):
        raise contract.ContractError("segment manifest must contain 64 ordered unique question IDs")
    return list(question_ids)


def extract(root: Path, plan: Path, segment: contract.Segment) -> dict[str, Any]:
    """Write/reuse a source-bound strict metric receipt for a sealed segment."""

    # A changed checkpoint, plan, base-model proof, target sidecar, or parent
    # invalidates the source observation before we read a single metric line.
    segment_receipt_path = contract.receipt_path(root, segment)
    segment_receipt = contract.validate_receipt(root, segment_receipt_path, expected_segment=segment)
    plan_contract = contract.validate_segment_plan(root, plan, segment)
    raw_metrics_path = contract.run_root(root, segment) / "metrics.jsonl"
    updates = _load_metrics(raw_metrics_path, segment)
    normalized_path = _normalized_metrics_path(root, segment)
    normalized_payload = b"".join(_canonical_json_line(update) for update in updates)
    normalized_status = _immutable_bytes(normalized_path, normalized_payload, label="normalized convergence metrics")
    raw_identity = _compact_identity(raw_metrics_path, label="trainer metrics JSONL")
    segment_identity = _compact_identity(segment_receipt_path, label="segment completion receipt")
    checkpoint = segment_receipt.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("checkpoint") != contract.checkpoint_uri(root, segment):
        raise contract.ContractError("segment completion receipt has no expected final checkpoint")
    checkpoint_sha256 = hashlib.sha256(
        json.dumps(dict(checkpoint), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = plan_contract.get("metadata")
    if not isinstance(metadata, Mapping):
        raise contract.ContractError("segment plan has no convergence metadata")
    selection_sha256 = metadata.get("selection_content_sha256")
    if not isinstance(selection_sha256, str) or len(selection_sha256) != 64:
        raise contract.ContractError("segment plan metadata has no selection content SHA256")
    document = {
        "schema": SOURCE_RECEIPT_SCHEMA,
        "selection": {"content_sha256": selection_sha256},
        "segment": {
            "pass_index": segment.pass_index,
            "segment_index": segment.segment_index,
            "checkpoint_step": segment.checkpoint_step,
            "updates": contract.UPDATES_PER_SEGMENT,
            "questions": contract.ROWS_PER_SEGMENT,
            "row_offset": segment.row_offset,
            "question_ids": _segment_question_ids(root, plan_contract, segment),
        },
        "checkpoint": {
            "path": contract.checkpoint_uri(root, segment),
            "sha256": checkpoint_sha256,
            "step": segment.checkpoint_step,
        },
        "raw_metrics_jsonl": raw_identity,
        "normalized_metrics_jsonl": _compact_identity(normalized_path, label="normalized convergence metrics"),
        "segment_completion_receipt": segment_identity,
    }
    receipt_path = _source_receipt_path(root, segment)
    receipt_payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    receipt_status = _immutable_bytes(receipt_path, receipt_payload, label="training-gap source receipt")
    return {
        "status": receipt_status,
        "normalized_metrics_status": normalized_status,
        "source_receipt": str(receipt_path),
        "normalized_metrics": str(normalized_path),
        "segment": segment.global_index,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--plan", type=Path, help="authored static convergence YAML")
    parser.add_argument("--segment-index", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = contract._absolute_root(args.repo_root)
        plan = contract.plan_path(root, args.plan)
        segment = contract.segment_for_index(args.segment_index)
        result = extract(root, plan, segment)
    except (contract.ContractError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
