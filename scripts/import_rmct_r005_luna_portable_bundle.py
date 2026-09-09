#!/usr/bin/env python3
"""Build a local Luna grading bundle from hash-verified r005 transfers.

This importer never opens an Isambard path.  It accepts the canonical raw
EvalLogs copied from the completed r005 evaluation plus the copied immutable
receipt set, verifies the original absolute-path/hash bindings, and emits the
portable bundle consumed by ``experiments.rmct_two_bias_eval.luna_bridge``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.rmct_two_bias_eval import luna_bridge as bridge
from experiments.rmct_two_bias_eval import raw_preflight
from experiments.rmct_two_bias_eval.contract import (
    EXPECTED_BIASED_TASKS,
    EXPECTED_TASKS,
    HELD_OUT_BIASES,
    SEEN_BIASES,
    bias_status,
)


PREFLIGHT_FILENAME = f"{bridge.R005_CONDITION}.json"


def _task_index(native_raw_root: Path, source: Mapping[str, Any]) -> int:
    raw_log = source.get("raw_log")
    sha256 = source.get("raw_log_sha256")
    if not isinstance(raw_log, str) or not bridge._is_sha256(sha256):
        raise bridge.LunaBridgeError("raw-preflight source has an incomplete canonical identity")
    return bridge._task_index_from_declared_raw_path(
        native_raw_root,
        raw_log,
        expected_sha256=str(sha256),
    )


def _evidence_file(root: Path, relative: str, *, label: str) -> Path:
    path = (root / relative).resolve()
    bridge._under_root(path, root, label=label)
    bridge._file_identity(path, label=label)
    return path


def _copy_custody(
    *,
    bundle_root: Path,
    local_path: Path,
    source_path: str,
    relative: str,
    expected_sha256: str | None = None,
    label: str,
) -> dict[str, Any]:
    identity = bridge._file_identity(local_path, label=label)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise bridge.LunaBridgeError(f"{label} differs from its receipt binding")
    destination = bridge._portable_destination(bundle_root, relative, label=label)
    bridge._copy_once(
        local_path,
        destination,
        expected_sha256=str(identity["sha256"]),
        label=label,
    )
    return {
        "source_path": source_path,
        "sha256": identity["sha256"],
        "size_bytes": identity["size_bytes"],
        "portable_path": relative,
    }


def import_bundle(raw_log_root: Path, evidence_root: Path, output_root: Path) -> Path:
    local_raw_root = bridge._regular_directory(raw_log_root, label="transferred r005 raw-log root")
    evidence = bridge._regular_directory(evidence_root, label="transferred r005 custody root")
    output = output_root.expanduser().resolve()
    if output == local_raw_root or output.is_relative_to(local_raw_root) or local_raw_root.is_relative_to(output):
        raise bridge.LunaBridgeError("portable output must be separate from the transferred raw-log root")
    if output == evidence or output.is_relative_to(evidence) or evidence.is_relative_to(output):
        raise bridge.LunaBridgeError("portable output must be separate from the transferred custody root")

    preflight_local = _evidence_file(evidence, PREFLIGHT_FILENAME, label="transferred raw-preflight report")
    evaluation_local = _evidence_file(evidence, "evaluation-receipt.json", label="transferred evaluation receipt")
    launch_local = _evidence_file(evidence, "launch-contract.json", label="transferred launch contract")
    completion_local = _evidence_file(evidence, "completion.json", label="transferred completion receipt")

    report = raw_preflight.validate_preflight_report(preflight_local)
    receipt = bridge._read_json_object(evaluation_local, label="transferred evaluation receipt")
    completion = bridge._read_json_object(completion_local, label="transferred completion receipt")
    if report.get("condition") != bridge.R005_CONDITION or receipt.get("condition") != bridge.R005_CONDITION:
        raise bridge.LunaBridgeError("transferred evidence is not the sealed r005 condition")
    native_raw_root = Path(str(report.get("raw_root", "")))
    if not native_raw_root.is_absolute() or native_raw_root.name != "raw" or native_raw_root.parent.name != "stage2":
        raise bridge.LunaBridgeError("raw-preflight report lost its canonical native r005 raw root")
    native_eval_root = native_raw_root.parent.parent
    native_evaluation_path = native_eval_root / "runtime" / "evaluation-receipt.json"
    native_preflight_path = native_raw_root.parent / "preflight" / PREFLIGHT_FILENAME
    native_launch_path = native_eval_root / "launch-contract.json"

    evaluation_identity = bridge._file_identity(evaluation_local, label="transferred evaluation receipt")
    launch_identity = bridge._file_identity(launch_local, label="transferred launch contract")
    if report.get("evaluation_receipt") != {
        "path": str(native_evaluation_path),
        "sha256": evaluation_identity["sha256"],
    }:
        raise bridge.LunaBridgeError("raw-preflight report is not bound to the transferred evaluation receipt")
    if report.get("contract", {}).get("runtime") != receipt.get("runtime"):
        raise bridge.LunaBridgeError("raw-preflight/evaluation runtime bindings differ")
    if completion.get("contract") != {
        "path": str(native_launch_path),
        "sha256": launch_identity["sha256"],
    }:
        raise bridge.LunaBridgeError("completion receipt is not bound to the transferred launch contract")
    completion_evaluation = completion.get("evaluation_receipt")
    if not isinstance(completion_evaluation, Mapping) or {
        "path": completion_evaluation.get("path"),
        "sha256": completion_evaluation.get("sha256"),
    } != {
        "path": str(native_evaluation_path),
        "sha256": evaluation_identity["sha256"],
    }:
        raise bridge.LunaBridgeError("completion receipt is not bound to the transferred evaluation receipt")

    sources = report.get("sources")
    if not isinstance(sources, list) or len(sources) != EXPECTED_TASKS:
        raise bridge.LunaBridgeError("raw-preflight report must contain exactly 21 sources")
    by_task: dict[int, Mapping[str, Any]] = {}
    local_logs: dict[int, Path] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise bridge.LunaBridgeError("raw-preflight source must be an object")
        index = _task_index(native_raw_root, source)
        if index in by_task:
            raise bridge.LunaBridgeError("raw-preflight report reuses a task index")
        sha256 = str(source["raw_log_sha256"])
        task_dir = local_raw_root / f"task-{index:03d}"
        candidates = list(task_dir.glob("*.eval"))
        if len(candidates) != 1 or candidates[0].name != f"{sha256}.eval":
            raise bridge.LunaBridgeError(f"transferred task-{index:03d} does not contain its one SHA-bound EvalLog")
        identity = bridge._file_identity(candidates[0], label=f"transferred task-{index:03d} EvalLog")
        if identity["sha256"] != sha256:
            raise bridge.LunaBridgeError(f"transferred task-{index:03d} EvalLog hash differs from preflight")
        by_task[index] = source
        local_logs[index] = candidates[0]
    if set(by_task) != set(range(1, EXPECTED_TASKS + 1)):
        raise bridge.LunaBridgeError("transferred raw logs must be exactly task-001 through task-021")
    if set(local_raw_root.glob("task-*")) != {local_raw_root / f"task-{index:03d}" for index in range(1, 22)}:
        raise bridge.LunaBridgeError("transferred raw root contains an unexpected task directory")

    bundle_path = bridge.portable_bundle_path(output)
    bundle_root = bundle_path.parent
    bundle_root.mkdir(parents=True, exist_ok=True)
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise bridge.LunaBridgeError("portable bundle root must be a regular directory")

    evaluation_record = _copy_custody(
        bundle_root=bundle_root,
        local_path=evaluation_local,
        source_path=str(native_evaluation_path),
        relative="custody/evaluation-receipt.json",
        expected_sha256=str(evaluation_identity["sha256"]),
        label="evaluation receipt",
    )
    preflight_record = _copy_custody(
        bundle_root=bundle_root,
        local_path=preflight_local,
        source_path=str(native_preflight_path),
        relative="custody/raw-preflight.json",
        label="raw-preflight report",
    )
    launch_record = _copy_custody(
        bundle_root=bundle_root,
        local_path=launch_local,
        source_path=str(native_launch_path),
        relative="custody/launch-contract.json",
        expected_sha256=str(launch_identity["sha256"]),
        label="launch contract",
    )

    task_receipts: list[dict[str, Any]] = []
    receipt_documents: dict[int, Mapping[str, Any]] = {}
    for index in range(1, EXPECTED_TASKS + 1):
        local_receipt = _evidence_file(evidence, f"receipts/task-{index:03d}.json", label=f"task-{index:03d} receipt")
        document = bridge._read_json_object(local_receipt, label=f"task-{index:03d} receipt")
        source = by_task[index]
        expected_canonical = {
            "path": source["raw_log"],
            "sha256": source["raw_log_sha256"],
            "size_bytes": local_logs[index].stat().st_size,
        }
        if (
            document.get("schema") != bridge.R005_TASK_RECEIPT_SCHEMA
            or document.get("task_index") != index
            or document.get("launch_contract_sha256") != launch_identity["sha256"]
            or document.get("evaluation_receipt_sha256") != evaluation_identity["sha256"]
            or document.get("canonical_log") != expected_canonical
        ):
            raise bridge.LunaBridgeError(f"task-{index:03d} receipt differs from transferred canonical evidence")
        attempt = document.get("attempt_log")
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("sha256") != expected_canonical["sha256"]
            or attempt.get("size_bytes") != expected_canonical["size_bytes"]
            or not isinstance(attempt.get("path"), str)
            or not Path(str(attempt["path"])).is_absolute()
        ):
            raise bridge.LunaBridgeError(f"task-{index:03d} receipt has an invalid preserved-attempt identity")
        native_receipt_path = native_raw_root.parent / "receipts" / f"task-{index:03d}.json"
        task_receipts.append(
            {
                "task_index": index,
                **_copy_custody(
                    bundle_root=bundle_root,
                    local_path=local_receipt,
                    source_path=str(native_receipt_path),
                    relative=f"custody/task-receipts/task-{index:03d}.json",
                    label=f"task-{index:03d} receipt",
                ),
            }
        )
        receipt_documents[index] = document

    portable_sources: list[dict[str, Any]] = []
    for index in range(4, EXPECTED_TASKS + 1):
        source = by_task[index]
        if source.get("kind") != "biased":
            raise bridge.LunaBridgeError(f"task-{index:03d} must be biased")
        bias_type = source.get("bias_type")
        if not isinstance(bias_type, str) or source.get("evaluation_bias_status") != bias_status(bias_type):
            raise bridge.LunaBridgeError(f"task-{index:03d} has an invalid scientific bias label")
        local_log = local_logs[index]
        sha256 = str(source["raw_log_sha256"])
        staged_relative = f"staged/task-{index:03d}/{sha256}.eval"
        staged = bridge._portable_destination(bundle_root, staged_relative, label=f"task-{index:03d} staged log")
        bridge._copy_once(local_log, staged, expected_sha256=sha256, label=f"task-{index:03d} raw log")
        staged_identity = bridge._file_identity(staged, label=f"task-{index:03d} staged log")
        receipt_path = evidence / "receipts" / f"task-{index:03d}.json"
        task_receipt_sha = bridge._sha256_file(receipt_path)
        staging_document = {
            "schema": bridge.STAGING_RECEIPT_SCHEMA,
            "bridge_schema": bridge.BRIDGE_SCHEMA,
            "source": {
                "task_index": index,
                "condition": bridge.R005_CONDITION,
                "regime": source["regime"],
                "population": source["population"],
                "dataset": source["dataset"],
                "bias_type": bias_type,
                "evaluation_bias_status": source["evaluation_bias_status"],
                "sample_count": source["sample_count"],
                "raw_log": {
                    "path": source["raw_log"],
                    "sha256": sha256,
                    "size_bytes": staged_identity["size_bytes"],
                },
                "paired_clean": source["paired_clean"],
            },
            "raw_preflight": {
                "path": str(native_preflight_path),
                "sha256": preflight_record["sha256"],
            },
            "evaluation_receipt": {
                "path": str(native_evaluation_path),
                "sha256": evaluation_identity["sha256"],
            },
            "task_receipt": {
                "path": str(native_raw_root.parent / "receipts" / f"task-{index:03d}.json"),
                "sha256": task_receipt_sha,
            },
            "staged_log": staged_identity,
            "scientific_labels": {
                "seen_biases": list(SEEN_BIASES),
                "held_out_biases": list(HELD_OUT_BIASES),
                "bias_status_source": "bias_type_not_legacy_substrate_regime",
            },
        }
        staging_path = staged.with_suffix(".staging.json")
        bridge._write_once(
            staging_path,
            bridge._canonical_json(staging_document),
            label=f"task-{index:03d} staging receipt",
        )
        staging_identity = bridge._file_identity(staging_path, label=f"task-{index:03d} staging receipt")
        portable_sources.append(
            {
                "task_index": index,
                "condition": bridge.R005_CONDITION,
                "regime": source["regime"],
                "population": source["population"],
                "dataset": source["dataset"],
                "bias_type": bias_type,
                "evaluation_bias_status": source["evaluation_bias_status"],
                "sample_count": source["sample_count"],
                "raw_log": {
                    "path": source["raw_log"],
                    "sha256": sha256,
                    "size_bytes": staged.stat().st_size,
                },
                "paired_clean": source["paired_clean"],
                "task_receipt": {
                    "source_path": str(native_raw_root.parent / "receipts" / f"task-{index:03d}.json"),
                    "sha256": task_receipt_sha,
                },
                "staged_log": {
                    "portable_path": staged_relative,
                    "sha256": sha256,
                    "size_bytes": staged.stat().st_size,
                },
                "staging_receipt": {
                    "portable_path": bridge._relative_bundle_path(bundle_root, staging_path, label="staging receipt"),
                    "sha256": staging_identity["sha256"],
                    "size_bytes": staging_identity["size_bytes"],
                },
            }
        )

    if len(portable_sources) != EXPECTED_BIASED_TASKS:
        raise bridge.LunaBridgeError("portable bundle must contain all 18 biased cells")
    document = {
        "schema": bridge.PORTABLE_BUNDLE_SCHEMA,
        "bridge_schema": bridge.BRIDGE_SCHEMA,
        "condition": bridge.R005_CONDITION,
        "source_policy": {
            "capture": "native_r005_receipt_and_preflight_reverified_before_copy",
            "grading": "local_bundle_and_staged_bytes_only_no_remote_absolute_path_reopen",
            "raw_task_count": EXPECTED_TASKS,
            "biased_task_count": EXPECTED_BIASED_TASKS,
        },
        "scientific_labels": {
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "bias_status_source": "bias_type_not_legacy_substrate_regime",
        },
        "native": {
            "evaluation_root": str(native_eval_root),
            "raw_log_root": str(native_raw_root),
            "evaluation_receipt": evaluation_record,
            "raw_preflight": preflight_record,
            "launch_contract": launch_record,
        },
        "task_receipts": task_receipts,
        "sources": portable_sources,
    }
    bridge._write_once(bundle_path, bridge._canonical_json(document), label="portable r005 Luna bundle")
    # Re-open the complete result through the grading-side validator before
    # returning it.  This proves that no source-host path is needed thereafter.
    bridge.portable_grade_inputs(bundle_path)
    return bundle_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-log-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = import_bundle(args.raw_log_root, args.evidence_root, args.output_root)
    print(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
