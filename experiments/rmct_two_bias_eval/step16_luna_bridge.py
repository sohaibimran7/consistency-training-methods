"""Receipt-selected Luna verbalisation grading for the repaired Step 16 run.

The Step 16 two-bias run has a special publication history: six biased
EvalLogs received a score-only switch repair after their raw generations had
completed.  This bridge therefore *never* selects the original raw logs by
directory convention.  Instead it verifies the V3 score-only publication and
uses the final ``published_log`` selected by its corrected preflight.

The program deliberately has two custody phases:

``--stage-only``
    Runs on the source host.  It re-verifies the V3 publication, copies the
    18 final biased logs into a separate write-once bundle, and captures the
    receipts required to authenticate that selection later.

``--grade-staged``
    Runs where the authorised OpenRouter credential is available.  It opens
    only the transferred bundle, never remote source paths, and grades the
    staged files with the repository's pinned Luna scorer.

Each invocation is write-once and every individual paid call is protected by
the immutable attempt claims in ``luna_bridge``.  The full mode uses exactly
five workers with one hundred connections each: an aggregate cap of 500.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from ctm_data.adapters.mcq_bias.luna_config import (
    DEFAULT_LUNA_GRADER_MODEL,
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_MAX_TOKENS,
)
from experiments.rmct_two_bias_eval import luna_bridge as luna
from experiments.rmct_two_bias_eval.contract import ALL_BIASES, HELD_OUT_BIASES, SEEN_BIASES, bias_status


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONDITION = "rmct-convergence-step016-two-bias-v1-r002"
DERIVED_NAMESPACE = "step16-luna-acknowledgement-v1"
BRIDGE_SCHEMA = "rmct-two-bias-step16-luna-bridge-v1"
STAGING_RECEIPT_SCHEMA = "rmct-two-bias-step16-luna-staging-receipt-v1"
PORTABLE_BUNDLE_SCHEMA = "rmct-two-bias-step16-luna-portable-bundle-v1"
ATTEMPT_RECEIPT_SCHEMA = "rmct-two-bias-step16-luna-attempt-v1"
DERIVED_PROVENANCE_SCHEMA = "rmct-two-bias-step16-luna-derived-v1"
INVOCATION_SCHEMA = "rmct-two-bias-step16-luna-invocation-v1"
COMPLETION_SCHEMA = "rmct-two-bias-step16-luna-completion-v1"
PORTABLE_BUNDLE_FILENAME = "step16-portable-bundle.json"

RECOVERY_SCHEMA = "rmct-checkpoint-two-bias-16gpu-step16-score-only-recovery-v3"
RECOVERY_PREFLIGHT_SCHEMA = "rmct-checkpoint-two-bias-16gpu-step16-score-only-recovery-preflight-v3"
RECOVERY_COMPLETION_SCHEMA = "rmct-checkpoint-two-bias-16gpu-step16-score-only-recovery-completion-v3"
RECOVERY_ATTESTATION_SCHEMA = "rmct-checkpoint-two-bias-16gpu-step16-score-only-recovery-attestation-v3"

REPAIRED_TASKS = (4, 6, 7, 9, 11, 13)
EXPECTED_TASKS = tuple(range(1, 22))
EXPECTED_BIASED_TASKS = tuple(range(4, 22))
WORKERS = 5
CONNECTIONS_PER_WORKER = 100
SMOKE_SAMPLES = 2

_HEX = frozenset("0123456789abcdef")


class Step16LunaBridgeError(ValueError):
    """Step 16 publication or portable-custody evidence is unsafe to use."""


@dataclass(frozen=True, slots=True)
class PublicationCustody:
    """Identities needed to bind a portable bundle to the final publication."""

    publication_root: Path
    preflight_path: Path
    preflight_identity: Mapping[str, Any]
    completion_path: Path
    completion_identity: Mapping[str, Any]
    attestation_path: Path
    attestation_identity: Mapping[str, Any]
    evaluation_receipt_path: Path
    evaluation_receipt_identity: Mapping[str, Any]
    launch_contract_path: Path
    launch_contract_identity: Mapping[str, Any]
    clean_gate_path: Path
    clean_gate_identity: Mapping[str, Any]
    phase_one_path: Path
    phase_one_identity: Mapping[str, Any]


# The corrected preflight must reproduce this matrix exactly.  The first five
# fields are the final source identity and the final field is the frozen count.
_TASK_MATRIX: dict[int, tuple[str, str, str, str, str | None, int]] = {
    1: ("unbiased", "iid", "in_domain", "logiqa", None, 50),
    2: ("unbiased", "iid", "in_domain", "hellaswag", None, 50),
    3: ("unbiased", "heldout_dataset", "hle", "hle-text-mc", None, 100),
    4: ("biased", "iid", "in_domain", "logiqa", "wrong_argument", 50),
    5: ("biased", "iid", "in_domain", "hellaswag", "wrong_argument", 50),
    6: ("biased", "heldout_dataset", "hle", "hle-text-mc", "wrong_argument", 100),
    7: ("biased", "heldout_bias", "in_domain", "logiqa", "suggested_answer", 50),
    8: ("biased", "heldout_bias", "in_domain", "hellaswag", "suggested_answer", 50),
    9: ("biased", "heldout_bias", "in_domain", "logiqa", "distractor_fact", 50),
    10: ("biased", "heldout_bias", "in_domain", "hellaswag", "distractor_fact", 50),
    11: ("biased", "heldout_bias", "in_domain", "logiqa", "post_hoc", 50),
    12: ("biased", "heldout_bias", "in_domain", "hellaswag", "post_hoc", 50),
    13: ("biased", "heldout_bias", "in_domain", "logiqa", "spurious_few_shot_squares", 50),
    14: ("biased", "heldout_bias", "in_domain", "hellaswag", "spurious_few_shot_squares", 50),
    15: ("biased", "heldout_bias", "in_domain", "logiqa", "wrong_few_shot", 50),
    16: ("biased", "heldout_bias", "in_domain", "hellaswag", "wrong_few_shot", 50),
    17: ("biased", "heldout_dataset_and_bias", "hle", "hle-text-mc", "suggested_answer", 100),
    18: ("biased", "heldout_dataset_and_bias", "hle", "hle-text-mc", "distractor_fact", 100),
    19: ("biased", "heldout_dataset_and_bias", "hle", "hle-text-mc", "post_hoc", 100),
    20: ("biased", "heldout_dataset_and_bias", "hle", "hle-text-mc", "spurious_few_shot_squares", 100),
    21: ("biased", "heldout_dataset_and_bias", "hle", "hle-text-mc", "wrong_few_shot", 100),
}


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _regular_directory(path: str | Path, *, label: str, create: bool = False) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.exists():
        if candidate.is_symlink() or not candidate.is_dir():
            raise Step16LunaBridgeError(f"{label} must be a regular directory: {candidate}")
    elif create:
        candidate.mkdir(parents=True, exist_ok=True)
    else:
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    return candidate


def _file_identity(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {candidate}")
    resolved = candidate.resolve()
    size = resolved.stat().st_size
    if size < 1:
        raise Step16LunaBridgeError(f"{label} must not be empty: {resolved}")
    return {"path": str(resolved), "sha256": _sha256_file(resolved), "size_bytes": size}


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    identity = _file_identity(candidate, label=label)
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Step16LunaBridgeError(f"invalid {label}: {candidate}") from exc
    if not isinstance(document, dict):
        raise Step16LunaBridgeError(f"{label} must contain a JSON object: {candidate}")
    del identity
    return document


def _identity_record(value: Any, *, label: str, require_exists: bool = True) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        raise Step16LunaBridgeError(f"{label} must be an exact file-identity object")
    path = value.get("path")
    sha256 = value.get("sha256")
    size = value.get("size_bytes")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or not _is_sha256(sha256)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
    ):
        raise Step16LunaBridgeError(f"{label} has invalid identity fields")
    resolved = Path(path).resolve()
    expected = {"path": str(resolved), "sha256": str(sha256), "size_bytes": size}
    if require_exists and _file_identity(resolved, label=label) != expected:
        raise Step16LunaBridgeError(f"{label} differs from its recorded identity: {resolved}")
    return resolved, expected


def _under(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Step16LunaBridgeError(f"{label} escapes its expected root: {path}") from exc


def _safe_component(value: str, *, label: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise Step16LunaBridgeError(f"{label} must be one non-empty safe path component")
    return value


def _write_once(path: Path, payload: bytes, *, label: str) -> str:
    """Publish an immutable local result, accepting only byte-identical replay."""

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing {label}: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise Step16LunaBridgeError(f"unsafe parent for {label}: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(f"{label} appeared with different bytes: {path}") from None
            return "resumed"
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def _copy_once(source: Path, destination: Path, *, expected: Mapping[str, Any], label: str) -> str:
    if _file_identity(source, label=f"{label} source") != dict(expected):
        raise Step16LunaBridgeError(f"{label} source changed before copy: {source}")
    expected_content = {"sha256": expected["sha256"], "size_bytes": expected["size_bytes"]}
    if destination.exists() or destination.is_symlink():
        existing = _file_identity(destination, label=f"staged {label}")
        if destination.is_symlink() or {"sha256": existing["sha256"], "size_bytes": existing["size_bytes"]} != expected_content:
            raise FileExistsError(f"refusing to overwrite differing staged {label}: {destination}")
        return "resumed"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise Step16LunaBridgeError(f"unsafe staged {label} parent: {destination.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        temporary_identity = _file_identity(temporary, label=f"temporary {label}")
        if {"sha256": temporary_identity["sha256"], "size_bytes": temporary_identity["size_bytes"]} != expected_content:
            raise Step16LunaBridgeError(f"staged {label} differs from its source: {source}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = _file_identity(destination, label=f"staged {label}")
            if destination.is_symlink() or {"sha256": existing["sha256"], "size_bytes": existing["size_bytes"]} != expected_content:
                raise FileExistsError(f"staged {label} appeared with different bytes: {destination}") from None
            return "resumed"
    finally:
        temporary.unlink(missing_ok=True)
    return "staged"


def _profile_luna_bridge() -> None:
    """Apply the Step 16 output schema in this process before core grading.

    The core r005 bridge is intentionally immutable and hard-coded to its own
    source selection.  Its lower-level scoring lifecycle is generic, however;
    this narrow profile swap only changes the derived artifact namespace and
    schema emitted by the Step 16 process.  It never changes r005 source
    selection or its portable-bundle routines.
    """

    luna.BRIDGE_SCHEMA = BRIDGE_SCHEMA
    luna.STAGING_RECEIPT_SCHEMA = STAGING_RECEIPT_SCHEMA
    luna.ATTEMPT_RECEIPT_SCHEMA = ATTEMPT_RECEIPT_SCHEMA
    luna.DERIVED_PROVENANCE_SCHEMA = DERIVED_PROVENANCE_SCHEMA
    luna.DERIVED_NAMESPACE = DERIVED_NAMESPACE


def _assert_parallelism() -> None:
    if (
        WORKERS != 5
        or CONNECTIONS_PER_WORKER != 100
        or WORKERS * CONNECTIONS_PER_WORKER != DEFAULT_MAX_CONNECTIONS
        or DEFAULT_MAX_CONNECTIONS != 500
    ):
        raise Step16LunaBridgeError("Step 16 Luna grading must use exactly five workers × 100 connections = 500")


def _source_expected_identity(index: int) -> tuple[Any, ...]:
    try:
        return _TASK_MATRIX[index]
    except KeyError as exc:  # pragma: no cover - immutable local table
        raise Step16LunaBridgeError(f"unexpected Step 16 task index: {index}") from exc


def _validate_task_source(
    source: Mapping[str, Any],
    *,
    index: int,
    raw_root: Path,
    recovery_root: Path,
) -> dict[str, Any]:
    required = {
        "full_sample_ids_sha256",
        "identity",
        "original_log",
        "publication_kind",
        "published_log",
        "sample_count",
        "sample_order",
        "source_prefix_sha256",
        "task_index",
        "task_receipt",
    }
    kind, regime, population, dataset, bias_type, sample_count = _source_expected_identity(index)
    if kind == "biased":
        required.add("paired_clean")
    if not isinstance(source, Mapping) or set(source) != required:
        raise Step16LunaBridgeError(f"Step 16 corrected source task-{index:03d} has an unsupported schema")
    if source.get("task_index") != index or source.get("identity") != [kind, regime, population, dataset, bias_type]:
        raise Step16LunaBridgeError(f"Step 16 corrected source task-{index:03d} lost its frozen scientific identity")
    if source.get("sample_count") != sample_count or source.get("sample_order") != "lexicographic-by-sample-id":
        raise Step16LunaBridgeError(f"Step 16 task-{index:03d} sample selection differs from the publication")
    if not _is_sha256(source.get("source_prefix_sha256")) or not _is_sha256(source.get("full_sample_ids_sha256")):
        raise Step16LunaBridgeError(f"Step 16 task-{index:03d} sample-ID identities are invalid")
    original_path, original_identity = _identity_record(source.get("original_log"), label=f"task-{index:03d} original log")
    _under(original_path, raw_root, label=f"task-{index:03d} original log")
    if original_path.parent.name != f"task-{index:03d}" or original_path.name != f"{original_identity['sha256']}.eval":
        raise Step16LunaBridgeError(f"task-{index:03d} original log has a non-canonical SHA-bound path")
    published_path, published_identity = _identity_record(source.get("published_log"), label=f"task-{index:03d} published log")
    publication_kind = source.get("publication_kind")
    if index in REPAIRED_TASKS:
        if publication_kind != "derived-score-only":
            raise Step16LunaBridgeError(f"repaired task-{index:03d} must use its derived score-only publication")
        _under(published_path, recovery_root / "derived-logs" / f"task-{index:03d}", label=f"task-{index:03d} derived publication")
    else:
        if publication_kind != "original-unchanged" or published_identity != original_identity:
            raise Step16LunaBridgeError(f"unchanged task-{index:03d} must retain exactly its original publication bytes")
    receipt_path, receipt_identity = _identity_record(source.get("task_receipt"), label=f"task-{index:03d} original task receipt")
    _under(receipt_path, raw_root.parent / "receipts", label=f"task-{index:03d} original task receipt")
    if receipt_path.name != f"task-{index:03d}.json":
        raise Step16LunaBridgeError(f"task-{index:03d} has a non-canonical task receipt path")
    record = dict(source)
    record["original_log"] = original_identity
    record["published_log"] = published_identity
    record["task_receipt"] = receipt_identity
    if kind == "biased":
        pair_path, pair_identity = _identity_record(source.get("paired_clean"), label=f"task-{index:03d} paired clean")
        clean_index = {"logiqa": 1, "hellaswag": 2, "hle-text-mc": 3}[dataset]
        clean_expected = raw_root / f"task-{clean_index:03d}" / f"{pair_identity['sha256']}.eval"
        if pair_path != clean_expected:
            raise Step16LunaBridgeError(f"task-{index:03d} paired clean does not bind its matching clean cell")
        record["paired_clean"] = pair_identity
        if not isinstance(bias_type, str) or bias_type not in ALL_BIASES:
            raise Step16LunaBridgeError(f"task-{index:03d} has an invalid bias type")
    return record


def _validate_publication(publication_root: str | Path) -> tuple[PublicationCustody, list[dict[str, Any]]]:
    """Verify the immutable V3 score-only publication before source selection."""

    root = _regular_directory(publication_root, label="Step 16 score-only publication root")
    preflight_path = root / "preflight" / "corrected-native-two-bias.json"
    completion_path = root / "completion" / "completion.json"
    attestation_path = root / "attestation" / "attestation.json"
    preflight_identity = _file_identity(preflight_path, label="Step 16 corrected preflight")
    completion_identity = _file_identity(completion_path, label="Step 16 score-only completion")
    attestation_identity = _file_identity(attestation_path, label="Step 16 score-only attestation")
    preflight = _read_json(preflight_path, label="Step 16 corrected preflight")
    completion = _read_json(completion_path, label="Step 16 score-only completion")
    attestation = _read_json(attestation_path, label="Step 16 score-only attestation")
    if (
        preflight.get("schema") != RECOVERY_PREFLIGHT_SCHEMA
        or completion.get("schema") != RECOVERY_COMPLETION_SCHEMA
        or attestation.get("schema") != RECOVERY_ATTESTATION_SCHEMA
        or preflight.get("condition") != CONDITION
        or completion.get("condition") != CONDITION
        or preflight.get("recovery_root") != str(root)
        or completion.get("preflight") != preflight_identity
        or attestation.get("preflight") != preflight_identity
        or attestation.get("completion") != completion_identity
        or completion.get("repaired_tasks") != list(REPAIRED_TASKS)
        or attestation.get("repair_tasks") != list(REPAIRED_TASKS)
    ):
        raise Step16LunaBridgeError("Step 16 V3 score-only publication receipt chain is inconsistent")
    target = preflight.get("target")
    completion_target = completion.get("target")
    if (
        not isinstance(target, Mapping)
        or not isinstance(completion_target, Mapping)
        or target.get("step") != 16
        or completion_target.get("step") != 16
        or target != completion_target
    ):
        raise Step16LunaBridgeError("Step 16 publication target is not the audited step-16 checkpoint")
    execution = attestation.get("execution")
    mutations = attestation.get("mutations")
    if (
        not isinstance(execution, Mapping)
        or execution.get("mode") != "score-only-derived-repair"
        or execution.get("model_calls") != 0
        or execution.get("generation_calls") != 0
        or execution.get("network_calls") != 0
        or not isinstance(mutations, Mapping)
        or mutations.get("derived_eval_logs") != len(REPAIRED_TASKS)
        or any(mutations.get(key) != 0 for key in ("original_eval_logs", "original_task_receipts", "original_launch_contract"))
    ):
        raise Step16LunaBridgeError("Step 16 publication lacks its no-generation score-only repair attestation")
    original_custody = preflight.get("original_custody")
    if not isinstance(original_custody, Mapping):
        raise Step16LunaBridgeError("Step 16 corrected preflight has no original custody")
    raw_root_value = original_custody.get("raw_root")
    if not isinstance(raw_root_value, str) or not Path(raw_root_value).is_absolute():
        raise Step16LunaBridgeError("Step 16 corrected preflight has an invalid original raw root")
    raw_root = _regular_directory(raw_root_value, label="Step 16 original paired-clean-ready root")
    evaluation_receipt_path, evaluation_receipt_identity = _identity_record(
        original_custody.get("evaluation_receipt"), label="Step 16 evaluation receipt"
    )
    launch_contract_path, launch_contract_identity = _identity_record(
        original_custody.get("launch_contract"), label="Step 16 launch contract"
    )
    clean_gate_path, clean_gate_identity = _identity_record(original_custody.get("clean_gate"), label="Step 16 clean gate")
    phase_one_path, phase_one_identity = _identity_record(original_custody.get("phase_one"), label="Step 16 phase-one receipt")
    if evaluation_receipt_path != raw_root.parent.parent / "runtime" / "evaluation-receipt.json":
        raise Step16LunaBridgeError("Step 16 evaluation receipt is outside the published evaluation root")
    if clean_gate_path != raw_root / "clean-gate-receipt.json":
        raise Step16LunaBridgeError("Step 16 clean gate is outside the published paired-clean root")
    sources = preflight.get("sources")
    if not isinstance(sources, list) or len(sources) != len(EXPECTED_TASKS):
        raise Step16LunaBridgeError("Step 16 corrected preflight must contain exactly 21 task sources")
    by_task: dict[int, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise Step16LunaBridgeError("Step 16 corrected preflight source is not an object")
        index = source.get("task_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in by_task or index not in EXPECTED_TASKS:
            raise Step16LunaBridgeError("Step 16 corrected preflight has an invalid or duplicate task index")
        by_task[index] = _validate_task_source(source, index=index, raw_root=raw_root, recovery_root=root)
    if set(by_task) != set(EXPECTED_TASKS):
        raise Step16LunaBridgeError("Step 16 corrected preflight does not cover task-001 through task-021")
    selection = preflight.get("selection_contract")
    expected_counts = [{"task_index": index, "sample_count": _TASK_MATRIX[index][-1]} for index in EXPECTED_TASKS]
    if (
        not isinstance(selection, Mapping)
        or selection.get("dataset_sample_ids") != "exact ordered source prefix spec.question_ids[:task_sample_count]"
        or selection.get("full_samples") != "exact unique set represented as lexicographically sorted(eval.dataset.sample_ids)"
        or selection.get("paired_variants") != "exact same source prefix and full sample-id order as the matching clean cell"
        or selection.get("task_sample_counts") != expected_counts
    ):
        raise Step16LunaBridgeError("Step 16 corrected preflight selection contract has drifted")
    expected_original_logs = [by_task[index]["original_log"] for index in EXPECTED_TASKS]
    expected_task_receipts = [by_task[index]["task_receipt"] for index in EXPECTED_TASKS]
    expected_published_logs = [by_task[index]["published_log"] for index in EXPECTED_TASKS]
    if (
        completion.get("original_task_logs") != expected_original_logs
        or completion.get("original_task_receipts") != expected_task_receipts
        or completion.get("published_task_logs") != expected_published_logs
    ):
        raise Step16LunaBridgeError("Step 16 completion does not select the corrected preflight matrix")
    corrected = preflight.get("corrected_native_validation")
    if not isinstance(corrected, Mapping) or corrected.get("task_indices") != list(EXPECTED_TASKS):
        raise Step16LunaBridgeError("Step 16 corrected-native validation does not cover the full task matrix")
    corrected_tasks = corrected.get("tasks")
    expected_corrected_tasks = [
        {
            "task_index": index,
            "sample_count": by_task[index]["sample_count"],
            "dataset_sample_ids_sha256": by_task[index]["source_prefix_sha256"],
            "full_sample_ids_sha256": by_task[index]["full_sample_ids_sha256"],
            "published_log": by_task[index]["published_log"],
        }
        for index in EXPECTED_TASKS
    ]
    if corrected_tasks != expected_corrected_tasks:
        raise Step16LunaBridgeError("Step 16 corrected-native validation does not bind the selected published logs")
    custody = PublicationCustody(
        publication_root=root,
        preflight_path=preflight_path.resolve(),
        preflight_identity=preflight_identity,
        completion_path=completion_path.resolve(),
        completion_identity=completion_identity,
        attestation_path=attestation_path.resolve(),
        attestation_identity=attestation_identity,
        evaluation_receipt_path=evaluation_receipt_path,
        evaluation_receipt_identity=evaluation_receipt_identity,
        launch_contract_path=launch_contract_path,
        launch_contract_identity=launch_contract_identity,
        clean_gate_path=clean_gate_path,
        clean_gate_identity=clean_gate_identity,
        phase_one_path=phase_one_path,
        phase_one_identity=phase_one_identity,
    )
    return custody, [by_task[index] for index in EXPECTED_TASKS]


def _namespace_root(output_root: str | Path) -> Path:
    return Path(output_root).expanduser().resolve() / DERIVED_NAMESPACE / CONDITION


def _ensure_separate(output_root: str | Path, publication_root: Path) -> Path:
    output = Path(output_root).expanduser().resolve()
    if output == publication_root or output.is_relative_to(publication_root) or publication_root.is_relative_to(output):
        raise Step16LunaBridgeError("Step 16 Luna output root must be separate from, and non-nested with, the publication root")
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise Step16LunaBridgeError(f"Step 16 Luna output root must be a regular directory when it exists: {output}")
    return output


def _native_grade_inputs(publication_root: str | Path, output_root: str | Path) -> tuple[list[luna.GradeInput], PublicationCustody]:
    custody, all_sources = _validate_publication(publication_root)
    output = _ensure_separate(output_root, custody.publication_root)
    _profile_luna_bridge()
    selected: list[luna.GradeInput] = []
    for source in all_sources:
        index = int(source["task_index"])
        if index not in EXPECTED_BIASED_TASKS:
            continue
        kind, regime, population, dataset, bias_type, sample_count = _TASK_MATRIX[index]
        assert kind == "biased" and isinstance(bias_type, str)  # frozen local matrix
        raw = source["published_log"]
        staged_path = _namespace_root(output) / "staged" / f"task-{index:03d}" / f"{raw['sha256']}.eval"
        paired_clean = source["paired_clean"]
        selected.append(
            luna.GradeInput(
                task_index=index,
                condition=CONDITION,
                regime=regime,
                population=population,
                dataset=dataset,
                bias_type=bias_type,
                evaluation_bias_status=bias_status(bias_type),
                sample_count=sample_count,
                raw_path=Path(str(raw["path"])),
                raw_sha256=str(raw["sha256"]),
                raw_size_bytes=int(raw["size_bytes"]),
                staged_path=staged_path,
                preflight_path=custody.preflight_path,
                preflight_sha256=str(custody.preflight_identity["sha256"]),
                evaluation_receipt_path=custody.evaluation_receipt_path,
                evaluation_receipt_sha256=str(custody.evaluation_receipt_identity["sha256"]),
                task_receipt_path=Path(str(source["task_receipt"]["path"])),
                task_receipt_sha256=str(source["task_receipt"]["sha256"]),
                paired_clean={
                    "raw_log": str(paired_clean["path"]),
                    "raw_log_sha256": str(paired_clean["sha256"]),
                    "source_prefix_sha256": source["source_prefix_sha256"],
                    "full_sample_ids_sha256": source["full_sample_ids_sha256"],
                },
            )
        )
    _validate_selected_inputs(selected)
    return selected, custody


def _validate_selected_inputs(sources: Sequence[luna.GradeInput]) -> None:
    if len(sources) != len(EXPECTED_BIASED_TASKS) or {item.task_index for item in sources} != set(EXPECTED_BIASED_TASKS):
        raise Step16LunaBridgeError("Step 16 Luna selection must contain exactly biased task-004 through task-021")
    if sum(item.sample_count for item in sources) != 1200:
        raise Step16LunaBridgeError("Step 16 Luna selection must contain exactly 1,200 biased samples")
    if {item.bias_type for item in sources if item.evaluation_bias_status == "seen"} != set(SEEN_BIASES):
        raise Step16LunaBridgeError("Step 16 Luna selection lost a seen bias")
    if {item.bias_type for item in sources if item.evaluation_bias_status == "held_out"} != set(HELD_OUT_BIASES):
        raise Step16LunaBridgeError("Step 16 Luna selection lost a held-out bias")


def _portable_bundle_path(root: str | Path) -> Path:
    return _namespace_root(root) / PORTABLE_BUNDLE_FILENAME


def _portable_relative(bundle_root: Path, path: Path, *, label: str) -> str:
    try:
        relative = path.resolve().relative_to(bundle_root)
    except ValueError as exc:
        raise Step16LunaBridgeError(f"{label} escapes the portable bundle root: {path}") from exc
    return relative.as_posix()


def _portable_destination(bundle_root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise Step16LunaBridgeError(f"{label} has an unsafe portable relative path")
    destination = (bundle_root / candidate).resolve()
    _under(destination, bundle_root, label=label)
    return destination


def _capture_artifact(bundle_root: Path, source: Path, identity: Mapping[str, Any], *, relative: str, label: str) -> dict[str, Any]:
    destination = _portable_destination(bundle_root, relative, label=label)
    _copy_once(source, destination, expected=identity, label=label)
    return {"source_path": str(source), "sha256": identity["sha256"], "size_bytes": identity["size_bytes"], "portable_path": relative}


def _portable_source_record(source: luna.GradeInput, *, bundle_root: Path) -> dict[str, Any]:
    staging = source.staged_path.with_suffix(".staging.json")
    staging_identity = _file_identity(staging, label=f"Step 16 task-{source.task_index:03d} staging receipt")
    return {
        "task_index": source.task_index,
        "condition": source.condition,
        "regime": source.regime,
        "population": source.population,
        "dataset": source.dataset,
        "bias_type": source.bias_type,
        "evaluation_bias_status": source.evaluation_bias_status,
        "sample_count": source.sample_count,
        "raw_log": {"path": str(source.raw_path), "sha256": source.raw_sha256, "size_bytes": source.raw_size_bytes},
        "paired_clean": dict(source.paired_clean),
        "task_receipt": {"source_path": str(source.task_receipt_path), "sha256": source.task_receipt_sha256},
        "staged_log": {
            "portable_path": _portable_relative(bundle_root, source.staged_path, label="staged Step 16 log"),
            "sha256": source.raw_sha256,
            "size_bytes": source.raw_size_bytes,
        },
        "staging_receipt": {
            "portable_path": _portable_relative(bundle_root, staging, label="Step 16 staging receipt"),
            "sha256": staging_identity["sha256"],
            "size_bytes": staging_identity["size_bytes"],
        },
    }


def _capture_portable_bundle(sources: Sequence[luna.GradeInput], custody: PublicationCustody, output_root: str | Path) -> Path:
    """Copy already verified source evidence into a self-contained bundle."""

    _validate_selected_inputs(sources)
    bundle_root = _namespace_root(output_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise Step16LunaBridgeError(f"Step 16 portable bundle root is unsafe: {bundle_root}")
    preflight = _capture_artifact(
        bundle_root, custody.preflight_path, custody.preflight_identity, relative="custody/corrected-native-two-bias.json", label="corrected preflight"
    )
    completion = _capture_artifact(
        bundle_root, custody.completion_path, custody.completion_identity, relative="custody/completion.json", label="score-only completion"
    )
    attestation = _capture_artifact(
        bundle_root, custody.attestation_path, custody.attestation_identity, relative="custody/attestation.json", label="score-only attestation"
    )
    evaluation_receipt = _capture_artifact(
        bundle_root, custody.evaluation_receipt_path, custody.evaluation_receipt_identity, relative="custody/evaluation-receipt.json", label="evaluation receipt"
    )
    launch_contract = _capture_artifact(
        bundle_root, custody.launch_contract_path, custody.launch_contract_identity, relative="custody/launch-contract.json", label="launch contract"
    )
    clean_gate = _capture_artifact(
        bundle_root, custody.clean_gate_path, custody.clean_gate_identity, relative="custody/clean-gate-receipt.json", label="clean gate"
    )
    phase_one = _capture_artifact(
        bundle_root, custody.phase_one_path, custody.phase_one_identity, relative="custody/phase-001.json", label="phase-one receipt"
    )
    task_receipts: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda item: item.task_index):
        identity = _file_identity(source.task_receipt_path, label=f"task-{source.task_index:03d} receipt")
        if identity["sha256"] != source.task_receipt_sha256:
            raise Step16LunaBridgeError(f"task-{source.task_index:03d} receipt changed before portable capture")
        task_receipts.append(
            {
                "task_index": source.task_index,
                **_capture_artifact(
                    bundle_root,
                    source.task_receipt_path,
                    identity,
                    relative=f"custody/task-receipts/task-{source.task_index:03d}.json",
                    label=f"task-{source.task_index:03d} receipt",
                ),
            }
        )
    paired_clean: list[dict[str, Any]] = []
    unique_clean: dict[str, Mapping[str, Any]] = {}
    for source in sources:
        record = source.paired_clean
        unique_clean[str(record["raw_log"])] = {
            "path": str(record["raw_log"]),
            "sha256": str(record["raw_log_sha256"]),
            "size_bytes": _file_identity(Path(str(record["raw_log"])), label=f"task-{source.task_index:03d} paired clean")["size_bytes"],
        }
    for source_path, identity in sorted(unique_clean.items()):
        path = Path(source_path)
        paired_clean.append(
            _capture_artifact(
                bundle_root,
                path,
                identity,
                relative=f"custody/paired-clean/{identity['sha256']}.eval",
                label=f"paired clean {path.name}",
            )
        )
    document = {
        "schema": PORTABLE_BUNDLE_SCHEMA,
        "bridge_schema": BRIDGE_SCHEMA,
        "condition": CONDITION,
        "source_policy": {
            "capture": "source_host_v3_score_only_publication_reverified_before_copy",
            "grading": "portable_bundle_and_staged_bytes_only_no_remote_absolute_path_reopen",
            "biased_task_count": len(EXPECTED_BIASED_TASKS),
            "biased_sample_count": 1200,
            "repaired_tasks": list(REPAIRED_TASKS),
        },
        "scientific_labels": {
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "bias_status_source": "bias_type_not_legacy_substrate_regime",
        },
        "publication": {
            "root": str(custody.publication_root),
            "preflight": preflight,
            "completion": completion,
            "attestation": attestation,
            "evaluation_receipt": evaluation_receipt,
            "launch_contract": launch_contract,
            "clean_gate": clean_gate,
            "phase_one": phase_one,
        },
        "task_receipts": task_receipts,
        "paired_clean": paired_clean,
        "sources": [_portable_source_record(source, bundle_root=bundle_root) for source in sorted(sources, key=lambda item: item.task_index)],
    }
    bundle = _portable_bundle_path(output_root)
    _write_once(bundle, _canonical_bytes(document), label="Step 16 portable Luna bundle")
    return bundle


def stage_portable_bundle(publication_root: str | Path, output_root: str | Path) -> tuple[Path, list[tuple[luna.GradeInput, str]]]:
    """Authenticate, stage, and capture all 18 final Step 16 biased logs."""

    _assert_parallelism()
    _profile_luna_bridge()
    sources, custody = _native_grade_inputs(publication_root, output_root)
    staged = [(source, luna.stage_input(source)) for source in sources]
    bundle = _capture_portable_bundle(sources, custody, output_root)
    return bundle, staged


def _portable_local_file(bundle_root: Path, record: Any, *, label: str) -> tuple[Path, str, int, str]:
    if not isinstance(record, Mapping) or set(record) != {"source_path", "sha256", "size_bytes", "portable_path"}:
        raise Step16LunaBridgeError(f"portable {label} has an invalid capture record")
    source_path = record.get("source_path")
    sha256 = record.get("sha256")
    size = record.get("size_bytes")
    relative = record.get("portable_path")
    if (
        not isinstance(source_path, str)
        or not Path(source_path).is_absolute()
        or not _is_sha256(sha256)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or not isinstance(relative, str)
    ):
        raise Step16LunaBridgeError(f"portable {label} has invalid identity fields")
    path = _portable_destination(bundle_root, relative, label=label)
    identity = _file_identity(path, label=f"portable {label}")
    if identity["sha256"] != sha256 or identity["size_bytes"] != size:
        raise Step16LunaBridgeError(f"portable {label} differs from its captured identity")
    return path, str(sha256), size, source_path


def _validate_portable_publication(bundle_path: str | Path) -> tuple[Path, dict[str, Any], dict[str, tuple[Path, str, int, str]]]:
    bundle = Path(bundle_path).expanduser().resolve()
    if bundle.name != PORTABLE_BUNDLE_FILENAME:
        raise Step16LunaBridgeError(f"portable bundle filename must be {PORTABLE_BUNDLE_FILENAME!r}")
    document = _read_json(bundle, label="Step 16 portable Luna bundle")
    required = {
        "schema",
        "bridge_schema",
        "condition",
        "source_policy",
        "scientific_labels",
        "publication",
        "task_receipts",
        "paired_clean",
        "sources",
    }
    if set(document) != required or document.get("schema") != PORTABLE_BUNDLE_SCHEMA or document.get("bridge_schema") != BRIDGE_SCHEMA:
        raise Step16LunaBridgeError("portable bundle has an unsupported Step 16 schema")
    if document.get("condition") != CONDITION or document.get("source_policy") != {
        "capture": "source_host_v3_score_only_publication_reverified_before_copy",
        "grading": "portable_bundle_and_staged_bytes_only_no_remote_absolute_path_reopen",
        "biased_task_count": len(EXPECTED_BIASED_TASKS),
        "biased_sample_count": 1200,
        "repaired_tasks": list(REPAIRED_TASKS),
    }:
        raise Step16LunaBridgeError("portable bundle does not retain the Step 16 source policy")
    if document.get("scientific_labels") != {
        "seen_biases": list(SEEN_BIASES),
        "held_out_biases": list(HELD_OUT_BIASES),
        "bias_status_source": "bias_type_not_legacy_substrate_regime",
    }:
        raise Step16LunaBridgeError("portable bundle lost the scientific bias labels")
    bundle_root = bundle.parent
    publication = document.get("publication")
    expected_publication = {"root", "preflight", "completion", "attestation", "evaluation_receipt", "launch_contract", "clean_gate", "phase_one"}
    if not isinstance(publication, Mapping) or set(publication) != expected_publication:
        raise Step16LunaBridgeError("portable bundle has malformed publication custody")
    if not isinstance(publication.get("root"), str) or not Path(str(publication["root"])).is_absolute():
        raise Step16LunaBridgeError("portable bundle has no absolute source publication root")
    captured: dict[str, tuple[Path, str, int, str]] = {}
    for key in ("preflight", "completion", "attestation", "evaluation_receipt", "launch_contract", "clean_gate", "phase_one"):
        captured[key] = _portable_local_file(bundle_root, publication.get(key), label=key.replace("_", " "))
    preflight_path, _, _, _ = captured["preflight"]
    completion_path, _, _, _ = captured["completion"]
    attestation_path, _, _, _ = captured["attestation"]
    preflight = _read_json(preflight_path, label="portable corrected preflight")
    completion = _read_json(completion_path, label="portable completion")
    attestation = _read_json(attestation_path, label="portable attestation")
    # The source host has verified all original files.  Locally, re-check the
    # complete receipt chain that can be reconstructed from the captured bytes.
    if (
        preflight.get("schema") != RECOVERY_PREFLIGHT_SCHEMA
        or completion.get("schema") != RECOVERY_COMPLETION_SCHEMA
        or attestation.get("schema") != RECOVERY_ATTESTATION_SCHEMA
        or preflight.get("condition") != CONDITION
        or completion.get("condition") != CONDITION
        or completion.get("repaired_tasks") != list(REPAIRED_TASKS)
        or attestation.get("repair_tasks") != list(REPAIRED_TASKS)
    ):
        raise Step16LunaBridgeError("portable publication custody is not the expected Step 16 V3 repair")
    preflight_identity = _file_identity(preflight_path, label="portable corrected preflight")
    completion_identity = _file_identity(completion_path, label="portable completion")
    if completion.get("preflight") != {"path": captured["preflight"][3], "sha256": preflight_identity["sha256"], "size_bytes": preflight_identity["size_bytes"]}:
        raise Step16LunaBridgeError("portable completion lost its corrected-preflight binding")
    if attestation.get("preflight") != {"path": captured["preflight"][3], "sha256": preflight_identity["sha256"], "size_bytes": preflight_identity["size_bytes"]}:
        raise Step16LunaBridgeError("portable attestation lost its corrected-preflight binding")
    if attestation.get("completion") != {"path": captured["completion"][3], "sha256": completion_identity["sha256"], "size_bytes": completion_identity["size_bytes"]}:
        raise Step16LunaBridgeError("portable attestation lost its completion binding")
    return bundle, document, captured


def portable_grade_inputs(bundle_path: str | Path) -> list[luna.GradeInput]:
    """Authenticate a source-host bundle without opening remote source paths."""

    _assert_parallelism()
    _profile_luna_bridge()
    bundle, document, captured = _validate_portable_publication(bundle_path)
    bundle_identity = _file_identity(bundle, label="Step 16 portable Luna bundle")
    bundle_root = bundle.parent
    sources = document.get("sources")
    receipts = document.get("task_receipts")
    if not isinstance(sources, list) or len(sources) != len(EXPECTED_BIASED_TASKS):
        raise Step16LunaBridgeError("portable bundle must include exactly 18 biased Step 16 sources")
    if not isinstance(receipts, list) or len(receipts) != len(EXPECTED_BIASED_TASKS):
        raise Step16LunaBridgeError("portable bundle must include exactly 18 selected original task receipts")
    receipt_by_task: dict[int, tuple[Path, str, int, str]] = {}
    for record in receipts:
        if not isinstance(record, Mapping) or set(record) != {"task_index", "source_path", "sha256", "size_bytes", "portable_path"}:
            raise Step16LunaBridgeError("portable task-receipt record is malformed")
        index = record.get("task_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in receipt_by_task or index not in EXPECTED_BIASED_TASKS:
            raise Step16LunaBridgeError("portable task-receipt selection is incomplete or duplicated")
        receipt_by_task[index] = _portable_local_file(
            bundle_root,
            {key: record[key] for key in ("source_path", "sha256", "size_bytes", "portable_path")},
            label=f"task-{index:03d} receipt",
        )
    selected: list[luna.GradeInput] = []
    for record in sources:
        required = {
            "task_index", "condition", "regime", "population", "dataset", "bias_type", "evaluation_bias_status", "sample_count",
            "raw_log", "paired_clean", "task_receipt", "staged_log", "staging_receipt",
        }
        if not isinstance(record, Mapping) or set(record) != required:
            raise Step16LunaBridgeError("portable Step 16 source record is malformed")
        index = record.get("task_index")
        if isinstance(index, bool) or not isinstance(index, int) or index not in EXPECTED_BIASED_TASKS:
            raise Step16LunaBridgeError("portable Step 16 source has an invalid task index")
        _, regime, population, dataset, expected_bias, sample_count = _TASK_MATRIX[index]
        if (
            record.get("condition") != CONDITION
            or record.get("regime") != regime
            or record.get("population") != population
            or record.get("dataset") != dataset
            or record.get("bias_type") != expected_bias
            or record.get("evaluation_bias_status") != bias_status(str(expected_bias))
            or record.get("sample_count") != sample_count
        ):
            raise Step16LunaBridgeError(f"portable task-{index:03d} scientific labels differ from the frozen matrix")
        raw = record.get("raw_log")
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "size_bytes"}:
            raise Step16LunaBridgeError(f"portable task-{index:03d} has a malformed source log identity")
        raw_path = raw.get("path")
        raw_sha = raw.get("sha256")
        raw_size = raw.get("size_bytes")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute() or not _is_sha256(raw_sha) or not isinstance(raw_size, int):
            raise Step16LunaBridgeError(f"portable task-{index:03d} has invalid source log fields")
        staged = record.get("staged_log")
        staging = record.get("staging_receipt")
        if not isinstance(staged, Mapping) or set(staged) != {"portable_path", "sha256", "size_bytes"}:
            raise Step16LunaBridgeError(f"portable task-{index:03d} staged log record is malformed")
        if not isinstance(staging, Mapping) or set(staging) != {"portable_path", "sha256", "size_bytes"}:
            raise Step16LunaBridgeError(f"portable task-{index:03d} staging receipt record is malformed")
        staged_path = _portable_destination(bundle_root, str(staged.get("portable_path", "")), label=f"task-{index:03d} staged log")
        staged_identity = _file_identity(staged_path, label=f"portable task-{index:03d} staged log")
        if staged_identity["sha256"] != raw_sha or staged_identity["size_bytes"] != raw_size or staged.get("sha256") != raw_sha or staged.get("size_bytes") != raw_size:
            raise Step16LunaBridgeError(f"portable task-{index:03d} staged bytes differ from its selected published log")
        staging_path = _portable_destination(bundle_root, str(staging.get("portable_path", "")), label=f"task-{index:03d} staging receipt")
        staging_identity = _file_identity(staging_path, label=f"portable task-{index:03d} staging receipt")
        if staging_path != staged_path.with_suffix(".staging.json") or staging.get("sha256") != staging_identity["sha256"] or staging.get("size_bytes") != staging_identity["size_bytes"]:
            raise Step16LunaBridgeError(f"portable task-{index:03d} staging receipt is not bound beside its staged log")
        receipt = record.get("task_receipt")
        task_receipt_path, task_receipt_sha, _, task_receipt_source = receipt_by_task[index]
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != {"source_path", "sha256"}
            or receipt.get("source_path") != task_receipt_source
            or receipt.get("sha256") != task_receipt_sha
        ):
            raise Step16LunaBridgeError(f"portable task-{index:03d} receipt binding differs from custody capture")
        paired = record.get("paired_clean")
        if (
            not isinstance(paired, Mapping)
            or not isinstance(paired.get("raw_log"), str)
            or not _is_sha256(paired.get("raw_log_sha256"))
        ):
            raise Step16LunaBridgeError(f"portable task-{index:03d} paired-clean binding is malformed")
        staging_document = _read_json(staging_path, label=f"portable task-{index:03d} staging receipt")
        expected_staging_source = {
            "task_index": index,
            "condition": CONDITION,
            "regime": regime,
            "population": population,
            "dataset": dataset,
            "bias_type": expected_bias,
            "evaluation_bias_status": bias_status(str(expected_bias)),
            "sample_count": sample_count,
            "raw_log": {"path": raw_path, "sha256": raw_sha, "size_bytes": raw_size},
            "paired_clean": dict(paired),
        }
        expected_staging = {
            "schema": STAGING_RECEIPT_SCHEMA,
            "bridge_schema": BRIDGE_SCHEMA,
            "source": expected_staging_source,
            "raw_preflight": {"path": captured["preflight"][3], "sha256": captured["preflight"][1]},
            "evaluation_receipt": {
                "path": captured["evaluation_receipt"][3],
                "sha256": captured["evaluation_receipt"][1],
            },
            "task_receipt": {"path": task_receipt_source, "sha256": task_receipt_sha},
            "staged_log": {
                # The source-host staging receipt intentionally retains its
                # source-host path; its hash/size are what bind the copied
                # local file.
                "path": str(staging_document.get("staged_log", {}).get("path", "")),
                "sha256": raw_sha,
                "size_bytes": raw_size,
            },
            "scientific_labels": {
                "seen_biases": list(SEEN_BIASES),
                "held_out_biases": list(HELD_OUT_BIASES),
                "bias_status_source": "bias_type_not_legacy_substrate_regime",
            },
        }
        if (
            not isinstance(expected_staging["staged_log"]["path"], str)
            or not Path(expected_staging["staged_log"]["path"]).is_absolute()
            or staging_document != expected_staging
        ):
            raise Step16LunaBridgeError(f"portable task-{index:03d} staging receipt differs from its captured source bindings")
        selected.append(
            luna.GradeInput(
                task_index=index,
                condition=CONDITION,
                regime=regime,
                population=population,
                dataset=dataset,
                bias_type=str(expected_bias),
                evaluation_bias_status=bias_status(str(expected_bias)),
                sample_count=sample_count,
                raw_path=Path(raw_path),
                raw_sha256=str(raw_sha),
                raw_size_bytes=int(raw_size),
                staged_path=staged_path,
                preflight_path=captured["preflight"][0],
                preflight_sha256=captured["preflight"][1],
                evaluation_receipt_path=captured["evaluation_receipt"][0],
                evaluation_receipt_sha256=captured["evaluation_receipt"][1],
                task_receipt_path=task_receipt_path,
                task_receipt_sha256=task_receipt_sha,
                paired_clean=dict(paired),
                portable_bundle_path=bundle,
                portable_bundle_sha256=str(bundle_identity["sha256"]),
                staging_receipt_sha256=str(staging_identity["sha256"]),
            )
        )
    _validate_selected_inputs(selected)
    return sorted(selected, key=lambda item: item.task_index)


def _grade_shard(
    shard_index: int,
    sources: tuple[luna.GradeInput, ...],
    output_root: Path,
    smoke_samples: int | None,
) -> list[tuple[luna.GradeInput, str]]:
    _profile_luna_bridge()
    return [
        (
            source,
            luna.grade_one(
                source,
                output_root,
                worker_count=WORKERS,
                connections_per_worker=CONNECTIONS_PER_WORKER,
                smoke_samples=smoke_samples,
                shard_index=shard_index,
            ),
        )
        for source in sources
    ]


def grade_sources(sources: Sequence[luna.GradeInput], output_root: str | Path, *, smoke_samples: int | None) -> list[tuple[luna.GradeInput, str]]:
    """Use an exact five-process, 500-connection aggregate Luna invocation."""

    _assert_parallelism()
    _profile_luna_bridge()
    if not sources:
        return []
    shards = luna.deterministic_shards(sources, WORKERS)
    active = [(index, shard) for index, shard in enumerate(shards) if shard]
    results: dict[luna.GradeInput, str] = {}
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn")) as executor:
        futures = [
            executor.submit(_grade_shard, index, shard, Path(output_root).expanduser().resolve(), smoke_samples)
            for index, shard in active
        ]
        for future in futures:
            for source, status in future.result():
                results[source] = status
    return [(source, results[source]) for source in sorted(sources, key=lambda item: item.task_index)]


@contextmanager
def _grade_lock(output_root: str | Path):
    _profile_luna_bridge()
    root = _namespace_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".luna-grade.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _source_record_for_invocation(source: luna.GradeInput, output_root: str | Path, *, smoke: bool) -> dict[str, Any]:
    eval_path, rows_path, provenance_path = luna.output_paths(output_root, source, smoke=smoke)
    return {
        "task_index": source.task_index,
        "published_log": {"path": str(source.raw_path), "sha256": source.raw_sha256, "size_bytes": source.raw_size_bytes},
        "staged_log": {"path": str(source.staged_path), "sha256": source.raw_sha256, "size_bytes": source.raw_size_bytes},
        "derived": {"eval_log": str(eval_path), "rows": str(rows_path), "provenance": str(provenance_path)},
        "sample_count": min(source.sample_count, SMOKE_SAMPLES) if smoke else source.sample_count,
    }


def _invocation_document(
    sources: Sequence[luna.GradeInput],
    *,
    output_root: str | Path,
    mode: str,
) -> dict[str, Any]:
    smoke = mode == "smoke"
    return {
        "schema": INVOCATION_SCHEMA,
        "bridge_schema": BRIDGE_SCHEMA,
        "condition": CONDITION,
        "mode": mode,
        "source_policy": "portable_bundle_and_staged_bytes_only_no_remote_absolute_path_reopen",
        "luna_policy": {
            "grader_model": DEFAULT_LUNA_GRADER_MODEL,
            "grader_max_tokens": DEFAULT_MAX_TOKENS,
            "reasoning_effort": "low",
            "worker_count": WORKERS,
            "connections_per_worker": CONNECTIONS_PER_WORKER,
            "aggregate_connection_limit": WORKERS * CONNECTIONS_PER_WORKER,
        },
        "sources": [_source_record_for_invocation(source, output_root, smoke=smoke) for source in sources],
    }


def _completion_document(
    sources: Sequence[luna.GradeInput],
    *,
    output_root: str | Path,
    mode: str,
) -> dict[str, Any]:
    smoke = mode == "smoke"
    rows: list[dict[str, Any]] = []
    for source in sources:
        _eval_path, rows_path, _provenance_path = luna.output_paths(output_root, source, smoke=smoke)
        with rows_path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    expected_count = sum(min(source.sample_count, SMOKE_SAMPLES) if smoke else source.sample_count for source in sources)
    if len(rows) != expected_count:
        raise Step16LunaBridgeError(f"Step 16 Luna completion has {len(rows)} rows; expected {expected_count}")
    usage_totals: Counter[str] = Counter()
    provider_cost: Counter[str] = Counter()
    parsed = 0
    unparsed = 0
    cap_hits = 0
    for row in rows:
        value = row.get("bias_acknowledged")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            parsed += 1
        else:
            unparsed += 1
        if row.get("grader_max_tokens_cap_hit") is True:
            cap_hits += 1
        usage = row.get("grader_usage")
        if isinstance(usage, Mapping):
            for key, value in usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    usage_totals[str(key)] += float(value)
                    if str(key).lower() in {"cost", "cost_usd", "total_cost", "total_cost_usd"}:
                        provider_cost[str(key)] += float(value)
    return {
        "schema": COMPLETION_SCHEMA,
        "bridge_schema": BRIDGE_SCHEMA,
        "condition": CONDITION,
        "mode": mode,
        "coverage": {
            "task_indices": [source.task_index for source in sources],
            "expected_rows": expected_count,
            "written_rows": len(rows),
            "parsed_verdicts": parsed,
            "unparsed_verdicts": unparsed,
            "max_token_cap_hits": cap_hits,
        },
        "cost": {
            "currency": "USD",
            "provider_reported_usd_fields": dict(sorted(provider_cost.items())),
            "estimate_usd": None,
            "status": "no rate inferred; per-response provider usage retained in derived JSONL",
            "usage_totals": dict(sorted(usage_totals.items())),
        },
        "provenance": {
            "invocation": str(_namespace_root(output_root) / "run-receipts" / f"{mode}.invocation.json"),
            "luna_model": DEFAULT_LUNA_GRADER_MODEL,
            "aggregate_connection_limit": WORKERS * CONNECTIONS_PER_WORKER,
        },
    }


def grade_staged_bundle(bundle_path: str | Path, output_root: str | Path, *, mode: str, smoke_samples: int = SMOKE_SAMPLES) -> list[tuple[luna.GradeInput, str]]:
    """Run a dry, smoke, or full Luna pass against portable Step 16 bytes."""

    if mode not in {"dry-run", "smoke", "full"}:
        raise Step16LunaBridgeError("mode must be one of dry-run, smoke, or full")
    if smoke_samples != SMOKE_SAMPLES:
        raise Step16LunaBridgeError(f"Step 16 Luna smoke must use exactly {SMOKE_SAMPLES} samples")
    sources = portable_grade_inputs(bundle_path)
    bundle = Path(bundle_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output == bundle.parent or output.is_relative_to(bundle.parent) or bundle.parent.is_relative_to(output):
        raise Step16LunaBridgeError("derived Step 16 Luna output must be separate from the portable input bundle")
    if mode == "dry-run":
        return [(source, "ready") for source in sources]
    selected = sources[:1] if mode == "smoke" else sources
    _profile_luna_bridge()
    invocation_path = _namespace_root(output) / "run-receipts" / f"{mode}.invocation.json"
    _write_once(invocation_path, _canonical_bytes(_invocation_document(selected, output_root=output, mode=mode)), label=f"Step 16 Luna {mode} invocation")
    with _grade_lock(output):
        results = grade_sources(selected, output, smoke_samples=smoke_samples if mode == "smoke" else None)
    completion_path = _namespace_root(output) / "run-receipts" / f"{mode}.completion.json"
    _write_once(completion_path, _canonical_bytes(_completion_document(selected, output_root=output, mode=mode)), label=f"Step 16 Luna {mode} completion")
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", type=Path, help="remote V3 step16-score-only-publication-r002 root")
    parser.add_argument("--portable-bundle", type=Path, help="captured step16-portable-bundle.json")
    parser.add_argument("--grade-staged", action="store_true", help="grade only locally captured staged bytes")
    parser.add_argument("--output-root", required=True, type=Path, help="separate output root")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--stage-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=SMOKE_SAMPLES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = "dry-run" if args.dry_run else "stage-only" if args.stage_only else "smoke" if args.smoke else "full"
    try:
        if args.grade_staged:
            if args.portable_bundle is None or args.publication_root is not None or mode == "stage-only":
                raise Step16LunaBridgeError("--grade-staged requires --portable-bundle, no --publication-root, and a dry/smoke/full mode")
            results = grade_staged_bundle(args.portable_bundle, args.output_root, mode=mode, smoke_samples=args.smoke_samples)
        else:
            if args.portable_bundle is not None or args.publication_root is None:
                raise Step16LunaBridgeError("native stage-only requires --publication-root and no --portable-bundle")
            if mode == "stage-only":
                bundle, results = stage_portable_bundle(args.publication_root, args.output_root)
                print(f"portable bundle: {bundle}")
            elif mode == "dry-run":
                sources, _custody = _native_grade_inputs(args.publication_root, args.output_root)
                results = [(source, "ready") for source in sources]
            else:
                raise Step16LunaBridgeError("native paid grading is forbidden; run --stage-only then --grade-staged locally")
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _parser().error(str(exc))
    for source, status in results:
        print(f"{status}: task-{source.task_index:03d} {source.population}/{source.dataset}/{source.bias_type} ({source.evaluation_bias_status})")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_RECEIPT_SCHEMA",
    "BRIDGE_SCHEMA",
    "COMPLETION_SCHEMA",
    "CONDITION",
    "CONNECTIONS_PER_WORKER",
    "DERIVED_NAMESPACE",
    "DERIVED_PROVENANCE_SCHEMA",
    "PORTABLE_BUNDLE_FILENAME",
    "PORTABLE_BUNDLE_SCHEMA",
    "REPAIRED_TASKS",
    "SMOKE_SAMPLES",
    "Step16LunaBridgeError",
    "WORKERS",
    "grade_staged_bundle",
    "main",
    "portable_grade_inputs",
    "stage_portable_bundle",
]
