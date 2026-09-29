#!/usr/bin/env python3
"""Receipt-bound, score-only publication recovery for the sealed r002 step-16 logs.

This is deliberately a *derived* recovery.  It never replaces or alters an
original r002 EvalLog, task receipt, launch contract, phase receipt, or the
frozen r002 evaluator source.  The only permitted semantic change is to the
six sealed ``switch_scorer`` failures produced when five LogiQA cells and one
HLE cell resolved the HellaSwag clean EvalLog during the original concurrent
launch.

The recovery is intentionally conservative:

* every original task receipt and canonical EvalLog is replayed first;
* the pre-existing phase-one paired-clean gate is replayed read-only; a
  missing or divergent gate aborts rather than being sealed by this program;
* ``eval.dataset.sample_ids`` must retain the ordered source-prefix selected
  by r002, while Inspect's full ``log.samples`` must be the deterministic
  lexicographic ordering of that exact set;
* all non-repaired cells (including task 15, all HellaSwag cells, and the
  remaining HLE cells) must already have a correct switch-score binding and
  are left as the original bytes;
* repaired logs are deep copies written into a new immutable recovery
  namespace.  Only ``switch_scorer`` sample values/metadata and the aggregate
  rows derived from those values may differ;
* no solver, model, generation, or network path is invoked.  The attestation
  binds the scorer source bytes and records zero model/generation calls.

It is a local-only implementation aid until independently reviewed.  The
command refuses to write unless ``--write`` is supplied, and it never calls
Slurm clients.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# The frozen launcher and its scorer imports live in the original r002
# checkout.  Do not let ordinary Python import caching create ``__pycache__``
# evidence there while this derived-only program is merely validating it.
sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECOVERY_SCHEMA = "rmct-checkpoint-two-bias-16gpu-step16-score-only-recovery-v3"
PREFLIGHT_SCHEMA = "rmct-checkpoint-two-bias-16gpu-step16-score-only-recovery-preflight-v3"
COMPLETION_SCHEMA = "rmct-checkpoint-two-bias-16gpu-step16-score-only-recovery-completion-v3"
ATTESTATION_SCHEMA = "rmct-checkpoint-two-bias-16gpu-step16-score-only-recovery-attestation-v3"
RECOVERY_NAMESPACE = "step16-score-only-publication-r002"

# Every repair must reproduce the exact, independently audited corruption
# profile: all switch values unresolved against task 2, then recompute only
# against this task's receipt-selected clean counterpart.
REPAIR_CLEAN_TASK_BY_TASK = {
    4: 1,
    6: 3,
    7: 1,
    9: 1,
    11: 1,
    13: 1,
}
REPAIRED_TASKS = tuple(REPAIR_CLEAN_TASK_BY_TASK)
KNOWN_CORRUPT_WRONG_CLEAN_TASK = 2
REPAIR_SAMPLE_COUNT_BY_TASK = {
    4: 50,
    6: 100,
    7: 50,
    9: 50,
    11: 50,
    13: 50,
}
EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK = {
    4: {"unbiased_log": 50, "unbiased_answer": 50, "note_removed": 50},
    6: {"unbiased_log": 100, "unbiased_answer": 75, "note_removed": 100},
    7: {"unbiased_log": 50, "unbiased_answer": 50, "note_removed": 50},
    9: {"unbiased_log": 50, "unbiased_answer": 50, "note_removed": 50},
    11: {"unbiased_log": 50, "unbiased_answer": 50, "note_removed": 50},
    13: {"unbiased_log": 50, "unbiased_answer": 50, "note_removed": 50},
}
if set(REPAIR_SAMPLE_COUNT_BY_TASK) != set(REPAIRED_TASKS) or set(EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK) != set(
    REPAIRED_TASKS
):
    raise RuntimeError("score-only repair constants do not cover the exact repaired task set")
if any(
    set(expected) != {"unbiased_log", "unbiased_answer", "note_removed"}
    or expected["unbiased_log"] != REPAIR_SAMPLE_COUNT_BY_TASK[task_index]
    or expected["note_removed"] != REPAIR_SAMPLE_COUNT_BY_TASK[task_index]
    or not isinstance(expected["unbiased_answer"], int)
    or not 0 <= expected["unbiased_answer"] <= REPAIR_SAMPLE_COUNT_BY_TASK[task_index]
    for task_index, expected in EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK.items()
):
    raise RuntimeError("score-only repair metadata-delta constants have an invalid exact shape")
UNCHANGED_IID_LOGIQA_TASK = 15
VERIFIED_UNCHANGED_TASKS = {
    "task15_logiqa": [UNCHANGED_IID_LOGIQA_TASK],
    "hellaswag": [2, 5, 8, 10, 12, 14, 16],
    "hle": [3, 17, 18, 19, 20, 21],
}
SWITCH_SCORE_NAME = "switch_scorer"
ANSWER_SCORE_NAME = "mcq_bias_scorer"
SWITCH_KEYS = (
    "unbiased_matches_bias",
    "towards_bias_switch",
    "away_from_bias_switch",
    "net_switch",
    "abs_switch",
)
KNOWN_CORRUPT_SWITCH_NOTE = (
    "sample id not present in the unbiased log — it was not evaluated there "
    "(e.g. --limit truncation or mismatched question sets)"
)
KNOWN_SWITCH_METRICS = ("mcq_bias/nanmean", "mcq_bias/nanstderr")


class RecoveryError(ValueError):
    """The original r002 evidence cannot safely support this derived repair."""


@dataclass(frozen=True)
class LoadedCell:
    """One receipt-selected original EvalLog and its frozen task specification."""

    task_index: int
    receipt: Mapping[str, Any]
    original_path: Path
    log: Any
    spec: Any
    source_prefix: tuple[str, ...]
    sample_ids: tuple[str, ...]


def _attr(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _set_attr(value: Any, name: str, replacement: Any) -> None:
    """Set a field on the mutable Inspect model (or its test-double mapping)."""

    if isinstance(value, dict):
        value[name] = replacement
    else:
        setattr(value, name, replacement)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _equivalent(left: Any, right: Any) -> bool:
    """Structural equality that treats two NaNs as the same score sentinel."""

    if _is_nan(left) and _is_nan(right):
        return True
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_equivalent(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equivalent(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _normalise(value: Any) -> Any:
    """Produce a stable JSON-compatible snapshot without losing NaN semantics."""

    if _is_nan(value):
        return {"$float": "nan"}
    if isinstance(value, float) and math.isinf(value):
        return {"$float": "inf" if value > 0 else "-inf"}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _normalise(model_dump(mode="python"))
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(_normalise(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_identity(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("path"), str)
        and Path(value["path"]).is_absolute()
        and _is_sha256(value.get("sha256"))
        and isinstance(value.get("size_bytes"), int)
        and not isinstance(value.get("size_bytes"), bool)
        and value["size_bytes"] > 0
    )


def _require_regular_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise RecoveryError(f"{label} must be a non-linked regular file: {candidate}")
    return candidate.resolve()


def _require_regular_directory(path: str | Path, *, label: str, create: bool = False) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        if candidate.is_symlink() or not candidate.is_dir():
            raise RecoveryError(f"{label} must be a non-linked regular directory: {candidate}")
        return candidate.resolve()
    if not create:
        raise RecoveryError(f"{label} is absent: {candidate}")
    parent = candidate.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RecoveryError(f"{label} has an unsafe ancestor: {parent}")
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate.resolve()


def _identity(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = _require_regular_file(path, label=label)
    return {"path": str(resolved), "sha256": _sha256_file(resolved), "size_bytes": resolved.stat().st_size}


def _require_offline_environment() -> None:
    """Fail closed unless the caller has disabled all hub/model network paths."""

    expected = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    observed = {name: os.environ.get(name) for name in expected}
    if observed != expected:
        raise RecoveryError("score-only recovery requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")


def _write_immutable_bytes(path: Path, payload: bytes, *, label: str) -> str:
    """Publish new recovery evidence once, or replay byte-identical evidence."""

    parent = _require_regular_directory(path.parent, label=f"{label} parent", create=True)
    target = parent / path.name
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise RecoveryError(f"{label} target is not a regular file: {target}")
        if target.read_bytes() != payload:
            raise RecoveryError(f"{label} already exists with different bytes: {target}")
        return "resumed"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
                raise RecoveryError(f"{label} appeared with different bytes: {target}") from None
            return "resumed"
    finally:
        if temporary.exists():
            temporary.unlink()
    return "written"


def _write_immutable_json(path: Path, document: Mapping[str, Any], *, label: str) -> str:
    return _write_immutable_bytes(path, _canonical_bytes(document) + b"\n", label=label)


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    source = _require_regular_file(path, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"{label} is not valid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must contain a JSON object: {source}")
    return value


def _load_frozen_launcher(path: str | Path) -> Any:
    source = _require_regular_file(path, label="frozen r002 launcher")
    spec = importlib.util.spec_from_file_location("rmct_step16_score_only_frozen_r002", source)
    if spec is None or spec.loader is None:
        raise RecoveryError(f"could not load frozen r002 launcher: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_frozen_sources(frozen: Any, launch: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    critical = launch.get("critical_sources")
    if not isinstance(critical, Mapping):
        raise RecoveryError("r002 launch contract has no frozen critical-source identities")
    observed: dict[str, dict[str, Any]] = {}
    for relative, expected in critical.items():
        if not isinstance(relative, str) or not isinstance(expected, Mapping):
            raise RecoveryError("r002 launch contract has malformed critical-source identities")
        candidate = Path(frozen.PROJECT_ROOT) / relative
        identity = frozen._identity(candidate, label=f"frozen r002 source {relative}")
        if identity != expected:
            raise RecoveryError(f"frozen r002 source differs from its launch contract: {relative}")
        observed[relative] = identity
    return observed


def _validate_existing_clean_gate_read_only(
    *,
    frozen: Any,
    paths: Any,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
) -> dict[str, Any]:
    """Replay the sealed clean gate without ever calling its writer.

    The frozen evaluator's ``_seal_clean_gate`` is intentionally
    write-or-replay.  It is unsafe in a derived-only repair: an absent gate
    must stop this recovery, rather than be manufactured from otherwise
    valid receipts.  This function therefore requires the existing file and
    compares its exact document to the receipt-derived gate payload.
    """

    gate_path = _require_regular_file(paths.clean_gate_receipt, label="r002 paired-clean gate receipt")
    rows: list[dict[str, Any]] = []
    for task_index in frozen.CLEAN_TASK_INDICES:
        receipt = frozen._load_task_receipt(
            paths,
            task_index=task_index,
            launch_contract_sha256=launch_contract_sha256,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
        )
        if receipt is None:
            raise RecoveryError(f"r002 paired-clean gate has no sealed clean task-{task_index} receipt")
        canonical = receipt.get("canonical_log")
        if not isinstance(canonical, Mapping):
            raise RecoveryError(f"r002 paired-clean gate task-{task_index} receipt has no canonical EvalLog")
        rows.append(
            {
                "task_index": task_index,
                "sample_count": _task_sample_count(frozen, task_index),
                "task_receipt": _identity(paths.receipts / f"task-{task_index:03d}.json", label=f"r002 clean task-{task_index} receipt"),
                "clean_log": dict(canonical),
            }
        )
    expected = {
        "schema": frozen.CLEAN_GATE_RECEIPT_SCHEMA,
        "raw_root": str(paths.raw),
        "launch_contract_sha256": launch_contract_sha256,
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
        "clean": rows,
        "publication_order": "task_receipt_before_clean_eval_log",
    }
    document = _load_json(gate_path, label="r002 paired-clean gate receipt")
    if document != expected:
        raise RecoveryError("r002 paired-clean gate receipt differs from its sealed clean task receipts")
    return _identity(gate_path, label="r002 paired-clean gate receipt")


def _validate_phase_one_read_only(
    *,
    frozen: Any,
    campaign: Path,
    phase_one: Path,
    paths: Any,
    target: Any,
) -> dict[str, Any]:
    """Replay phase one and its gate using read-only frozen primitives only."""

    # This must be first: recover() is prohibited from synthesizing a missing
    # gate, and must fail before it reaches any potentially mutable path.
    _require_regular_file(paths.clean_gate_receipt, label="r002 paired-clean gate receipt")
    document = _load_json(phase_one, label="r002 phase-one receipt")
    expected_plan = frozen.PHASES.get(1)
    if expected_plan != {target.step: tuple(range(1, 15))}:
        raise RecoveryError("frozen r002 phase-one plan is not the expected step-16 task-1..14 plan")
    expected_fields = {"schema", "campaign_root", "phase", "previous_phase", "targets"}
    if (
        set(document) != expected_fields
        or document.get("schema") != frozen.PHASE_RECEIPT_SCHEMA
        or document.get("campaign_root") != str(campaign)
        or document.get("phase") != 1
        or document.get("previous_phase") is not None
    ):
        raise RecoveryError("r002 phase-one receipt has an unsupported custody schema")
    targets = document.get("targets")
    if not isinstance(targets, list) or len(targets) != 1 or not isinstance(targets[0], Mapping):
        raise RecoveryError("r002 phase-one receipt has no exact target record")
    record = targets[0]
    indices = tuple(expected_plan[target.step])
    if (
        record.get("step") != target.step
        or record.get("condition") != target.condition
        or record.get("output_root") != str(paths.root)
        or record.get("task_indices") != list(indices)
        or not isinstance(record.get("launch_contract"), Mapping)
        or not isinstance(record.get("evaluation_receipt"), Mapping)
        or not isinstance(record.get("task_receipts"), list)
        or not isinstance(record.get("checkpoint_custody"), Mapping)
    ):
        raise RecoveryError("r002 phase-one target custody differs from the frozen plan")
    launch, launch_sha256 = frozen._load_launch(paths, target=target)
    _evaluation, evaluation_sha256 = frozen._load_evaluation_receipt(
        paths,
        target=target,
        launch=launch,
        launch_sha256=launch_sha256,
    )
    if record["launch_contract"] != _identity(paths.contract, label="r002 phase-one launch contract"):
        raise RecoveryError("r002 phase-one launch-contract identity changed")
    if record["evaluation_receipt"] != _identity(paths.evaluation_receipt, label="r002 phase-one evaluation receipt"):
        raise RecoveryError("r002 phase-one evaluation-receipt identity changed")
    if record["checkpoint_custody"] != launch.get("checkpoint_custody"):
        raise RecoveryError("r002 phase-one checkpoint custody differs from its launch contract")
    expected_receipts: list[dict[str, Any]] = []
    for task_index in indices:
        receipt = frozen._load_task_receipt(
            paths,
            task_index=task_index,
            launch_contract_sha256=launch_sha256,
            evaluation_receipt_sha256=evaluation_sha256,
        )
        if receipt is None:
            raise RecoveryError(f"r002 phase-one names an unsealed task-{task_index}")
        expected_receipts.append(
            {
                "task_index": task_index,
                "sample_count": _task_sample_count(frozen, task_index),
                "receipt": _identity(paths.receipts / f"task-{task_index:03d}.json", label=f"r002 phase-one task-{task_index} receipt"),
            }
        )
    if record["task_receipts"] != expected_receipts:
        raise RecoveryError("r002 phase-one task receipt identities changed")
    clean_gate = _validate_existing_clean_gate_read_only(
        frozen=frozen,
        paths=paths,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    )
    if record.get("paired_clean_gate") != clean_gate:
        raise RecoveryError("r002 phase-one paired-clean gate identity changed")
    return {"phase_one": _identity(phase_one, label="r002 phase-one receipt"), "clean_gate": clean_gate}


def _score_snapshot_without_switch(sample: Any) -> dict[str, Any]:
    """Every sample field other than switch score value/metadata must survive unchanged."""

    model_dump = getattr(sample, "model_dump", None)
    if not callable(model_dump):
        raise RecoveryError("Inspect sample has no model_dump representation")
    payload = model_dump(mode="python")
    if not isinstance(payload, dict):
        raise RecoveryError("Inspect sample model_dump is not an object")
    scores = payload.get("scores")
    if not isinstance(scores, Mapping):
        raise RecoveryError("Inspect sample has no score mapping")
    mutable_scores = dict(scores)
    switch = mutable_scores.pop(SWITCH_SCORE_NAME, None)
    if not isinstance(switch, Mapping):
        raise RecoveryError("Inspect sample lacks switch_scorer")
    # Keep the immutable parts of the switch score in the snapshot too.
    switch_immutable = dict(switch)
    switch_immutable.pop("value", None)
    switch_immutable.pop("metadata", None)
    payload["scores"] = mutable_scores
    payload["switch_scorer_immutable"] = switch_immutable
    return _normalise(payload)


def _log_header_snapshot(log: Any) -> dict[str, Any]:
    """The serialized EvalLog state that must never change in this repair.

    Inspect adds a location while reading a file, so it is deliberately not
    part of the snapshot.  Everything else outside samples/results is frozen.
    """

    model_dump = getattr(log, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="python")
        if not isinstance(payload, Mapping):
            raise RecoveryError("Inspect EvalLog model_dump is not an object")
        frozen = dict(payload)
        frozen.pop("samples", None)
        frozen.pop("results", None)
        # ``location`` is a read-time attribute in some Inspect versions, not
        # part of the serialized log contract.
        frozen.pop("location", None)
        return _normalise(frozen)
    return {
        "status": _normalise(_attr(log, "status")),
        "eval": _normalise(_attr(log, "eval")),
        "plan": _normalise(_attr(log, "plan")),
        "stats": _normalise(_attr(log, "stats")),
        "error": _normalise(_attr(log, "error")),
        "reductions": _normalise(_attr(log, "reductions")),
    }


def _log_semantic_snapshot(log: Any) -> dict[str, Any]:
    """Stable complete payload used to verify a re-opened derived EvalLog."""

    samples = list(_attr(log, "samples", []) or [])
    return {
        "header": _log_header_snapshot(log),
        "samples": [_normalise(sample) for sample in samples],
        "results": _normalise(_attr(log, "results")),
    }


def _answer(sample: Any) -> str | None:
    scores = _mapping(_attr(sample, "scores", {}))
    score = scores.get(ANSWER_SCORE_NAME)
    if score is None:
        raise RecoveryError(f"sample {_attr(sample, 'id', '')!r} lacks {ANSWER_SCORE_NAME}")
    answer = _attr(score, "answer")
    if answer is not None and not isinstance(answer, str):
        raise RecoveryError(f"sample {_attr(sample, 'id', '')!r} has an invalid parsed answer")
    return answer


def _switch_score(sample: Any) -> Any:
    scores = _mapping(_attr(sample, "scores", {}))
    score = scores.get(SWITCH_SCORE_NAME)
    if score is None:
        raise RecoveryError(f"sample {_attr(sample, 'id', '')!r} lacks {SWITCH_SCORE_NAME}")
    return score


def _switch_value(score: Any) -> dict[str, Any]:
    value = _attr(score, "value")
    if not isinstance(value, Mapping) or set(value) != set(SWITCH_KEYS):
        raise RecoveryError("switch_scorer has an incomplete score mapping")
    return dict(value)


def _switch_metadata(score: Any) -> dict[str, Any]:
    metadata = _attr(score, "metadata", {})
    if not isinstance(metadata, Mapping):
        raise RecoveryError("switch_scorer has non-mapping metadata")
    return dict(metadata)


def _task_sample_count(frozen: Any, task_index: int) -> int:
    return int(frozen._sample_count_for_task(task_index))


def _source_prefix_from_log(log: Any, *, spec: Any, cap: int, task_index: int) -> tuple[str, ...]:
    dataset = _attr(_attr(log, "eval"), "dataset")
    sample_ids = _attr(dataset, "sample_ids")
    if not isinstance(sample_ids, list) or any(not isinstance(value, str) or not value for value in sample_ids):
        raise RecoveryError(f"task-{task_index} EvalLog has no valid eval.dataset.sample_ids")
    expected = tuple(getattr(spec, "question_ids", ()))[:cap]
    if len(expected) != cap or tuple(sample_ids) != expected:
        raise RecoveryError(f"task-{task_index} eval.dataset.sample_ids does not retain the exact ordered r002 source prefix")
    if _attr(dataset, "shuffled", False) is not False:
        raise RecoveryError(f"task-{task_index} EvalLog unexpectedly records a shuffled dataset")
    return tuple(sample_ids)


def _sample_ids(log: Any, *, task_index: int) -> tuple[str, ...]:
    samples = list(_attr(log, "samples", []) or [])
    values = tuple(_attr(sample, "id", "") for sample in samples)
    if any(not isinstance(value, str) or not value for value in values) or len(values) != len(set(values)):
        raise RecoveryError(f"task-{task_index} EvalLog has missing or duplicate sample IDs")
    return values


def _validate_sorted_sample_semantics(cell: LoadedCell) -> None:
    expected_sorted = tuple(sorted(cell.source_prefix))
    if cell.sample_ids != expected_sorted:
        raise RecoveryError(
            f"task-{cell.task_index} full EvalLog samples are not the deterministic lexicographic ordering "
            "of eval.dataset.sample_ids"
        )


def _expected_clean_index(specs: Sequence[Any], task_index: int) -> int:
    spec = specs[task_index - 1]
    matches = [
        index
        for index, candidate in enumerate(specs, start=1)
        if getattr(candidate, "kind", None) == "unbiased"
        and getattr(candidate, "population", None) == getattr(spec, "population", None)
        and getattr(candidate, "dataset", None) == getattr(spec, "dataset", None)
    ]
    if len(matches) != 1:
        raise RecoveryError(f"task-{task_index} has no unique matching clean cell")
    return matches[0]


def _clean_answers(cell: LoadedCell) -> dict[str, str | None]:
    answers: dict[str, str | None] = {}
    for sample in _attr(cell.log, "samples", []) or []:
        sample_id = _attr(sample, "id", "")
        if sample_id in answers:
            raise RecoveryError(f"clean task-{cell.task_index} has duplicate sample ID {sample_id!r}")
        answers[sample_id] = _answer(sample)
    if tuple(sorted(answers)) != cell.sample_ids:
        raise RecoveryError(f"clean task-{cell.task_index} answer map does not retain its sample IDs")
    return answers


def _expected_switch_metadata(*, clean_log: Path, clean_answer: str | None) -> dict[str, Any]:
    return {"unbiased_log": str(clean_log), "unbiased_answer": clean_answer}


def _validate_correct_switch_binding(
    cell: LoadedCell,
    *,
    clean: LoadedCell,
    switch_values: Any,
) -> None:
    clean_answers = _clean_answers(clean)
    for sample in _attr(cell.log, "samples", []) or []:
        sample_id = _attr(sample, "id", "")
        metadata = _mapping(_attr(sample, "metadata", {}))
        biased_option = metadata.get("biased_option")
        if not isinstance(biased_option, str) or not biased_option:
            raise RecoveryError(f"task-{cell.task_index} sample {sample_id!r} has no frozen biased option")
        if sample_id not in clean_answers:
            raise RecoveryError(f"task-{cell.task_index} sample {sample_id!r} is absent from its clean answer map")
        score = _switch_score(sample)
        expected_value = dict(switch_values(_answer(sample), clean_answers[sample_id], biased_option))
        if not _equivalent(_switch_value(score), expected_value):
            raise RecoveryError(f"task-{cell.task_index} sample {sample_id!r} has an incorrect switch-score value")
        expected_metadata = _expected_switch_metadata(clean_log=clean.original_path, clean_answer=clean_answers[sample_id])
        if _switch_metadata(score) != expected_metadata:
            raise RecoveryError(f"task-{cell.task_index} sample {sample_id!r} resolves the wrong clean EvalLog")


def _validate_known_bad_binding(
    cell: LoadedCell,
    *,
    clean: LoadedCell,
    wrong_clean: LoadedCell,
) -> None:
    """Accept only the byte-for-byte semantic r002 switch-corruption profile.

    The repair removes the legacy unresolved-note metadata.  Therefore it
    must not accept a merely similar score: every original switch payload has
    to be precisely the observed task-2-path / ``None`` / fixed-note profile,
    with no additional metadata that could otherwise be silently discarded.
    """

    expected_clean_task = REPAIR_CLEAN_TASK_BY_TASK.get(cell.task_index)
    expected_metadata_changes = EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK.get(cell.task_index)
    if (
        expected_clean_task is None
        or expected_metadata_changes is None
        or clean.task_index != expected_clean_task
        or wrong_clean.task_index != KNOWN_CORRUPT_WRONG_CLEAN_TASK
    ):
        raise RecoveryError("score-only repair must use the exact task-specific clean/task-2 known-corruption profile")
    samples = list(_attr(cell.log, "samples", []) or [])
    if len(samples) != expected_metadata_changes["unbiased_log"]:
        raise RecoveryError(f"task-{cell.task_index} does not have the exact known-corrupt sample count")
    expected_metadata = {
        "unbiased_log": str(wrong_clean.original_path),
        "unbiased_answer": None,
        "note": KNOWN_CORRUPT_SWITCH_NOTE,
    }
    for sample in samples:
        score = _switch_score(sample)
        values = _switch_value(score)
        if not all(_is_nan(values[key]) for key in SWITCH_KEYS):
            raise RecoveryError(f"task-{cell.task_index} is not the known all-unresolved switch-score failure")
        metadata = _switch_metadata(score)
        if metadata != expected_metadata:
            raise RecoveryError(
                f"task-{cell.task_index} does not exhibit the exact known corrupt switch metadata profile"
            )


def _validate_known_bad_aggregate_switch_results(log: Any) -> str:
    """Require the original five aggregate rows before changing any of them.

    A score-only recovery is admissible only for the observed all-unresolved
    profile: exactly five ``switch_scorer`` rows in the fixed metric order,
    both aggregate metrics NaN, and every sample unscored.  Any alternative
    aggregate state may contain unreviewed evidence and must fail closed.
    """

    total = len(_attr(log, "samples", []) or [])
    if total <= 0:
        raise RecoveryError("known corrupt switch aggregate has no samples")
    results = _attr(log, "results")
    records = [record for record in list(_attr(results, "scores", []) or []) if _attr(record, "scorer") == SWITCH_SCORE_NAME]
    if len(records) != len(SWITCH_KEYS) or [_attr(record, "name") for record in records] != list(SWITCH_KEYS):
        raise RecoveryError("known corrupt switch aggregate does not have the exact five switch rows")
    snapshot: list[dict[str, Any]] = []
    for key, record in zip(SWITCH_KEYS, records, strict=True):
        metrics = _mapping(_attr(record, "metrics", {}))
        if set(metrics) != set(KNOWN_SWITCH_METRICS):
            raise RecoveryError(f"known corrupt switch aggregate row {key!r} has an unexpected metric profile")
        mean = _attr(metrics[KNOWN_SWITCH_METRICS[0]], "value")
        stderr = _attr(metrics[KNOWN_SWITCH_METRICS[1]], "value")
        scored = _attr(record, "scored_samples")
        unscored = _attr(record, "unscored_samples")
        if (
            not _is_nan(mean)
            or not _is_nan(stderr)
            or not isinstance(scored, int)
            or not isinstance(unscored, int)
            or isinstance(scored, bool)
            or isinstance(unscored, bool)
            or scored != 0
            or unscored != total
        ):
            raise RecoveryError(f"known corrupt switch aggregate row {key!r} is not all-NaN/unscored")
        snapshot.append(
            {
                "name": key,
                "metrics": {KNOWN_SWITCH_METRICS[0]: mean, KNOWN_SWITCH_METRICS[1]: stderr},
                "scored_samples": scored,
                "unscored_samples": unscored,
            }
        )
    return _digest(snapshot)


def _nan_stats(values: Sequence[Any]) -> tuple[float, float, int]:
    finite = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))]
    if not finite:
        return math.nan, math.nan, 0
    mean = sum(finite) / len(finite)
    if len(finite) <= 1:
        return mean, 0.0, len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return mean, math.sqrt(variance / len(finite)), len(finite)


def _repair_aggregate_switch_results(log: Any) -> dict[str, dict[str, Any]]:
    """Update only aggregate rows mechanically implied by corrected sample scores."""

    values_by_key = {key: [] for key in SWITCH_KEYS}
    for sample in _attr(log, "samples", []) or []:
        value = _switch_value(_switch_score(sample))
        for key in SWITCH_KEYS:
            values_by_key[key].append(value[key])
    results = _attr(log, "results")
    result_scores = list(_attr(results, "scores", []) or [])
    changed: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    total = len(_attr(log, "samples", []) or [])
    for record in result_scores:
        if _attr(record, "scorer") != SWITCH_SCORE_NAME:
            continue
        key = _attr(record, "name")
        if key not in values_by_key or key in seen:
            raise RecoveryError("derived EvalLog has malformed switch aggregate rows")
        seen.add(key)
        mean, stderr, scored = _nan_stats(values_by_key[key])
        metrics = _mapping(_attr(record, "metrics", {}))
        if set(metrics) != set(KNOWN_SWITCH_METRICS):
            raise RecoveryError(f"switch aggregate row {key!r} has an unexpected metric shape")
        mean_metric = metrics[KNOWN_SWITCH_METRICS[0]]
        stderr_metric = metrics[KNOWN_SWITCH_METRICS[1]]
        before = {
            "mean": _attr(mean_metric, "value"),
            "stderr": _attr(stderr_metric, "value"),
            "scored_samples": _attr(record, "scored_samples"),
            "unscored_samples": _attr(record, "unscored_samples"),
        }
        after = {
            "mean": mean,
            "stderr": stderr,
            "scored_samples": scored,
            "unscored_samples": total - scored,
        }
        _set_attr(mean_metric, "value", mean)
        _set_attr(stderr_metric, "value", stderr)
        _set_attr(record, "scored_samples", scored)
        _set_attr(record, "unscored_samples", total - scored)
        changed[key] = {
            "scored_samples_before": before["scored_samples"],
            "unscored_samples_before": before["unscored_samples"],
            "scored_samples_after": scored,
            "unscored_samples_after": total - scored,
            "before_sha256": _digest(before),
            "after_sha256": _digest(after),
            "aggregate_changed": int(not _equivalent(before, after)),
        }
    if seen != set(SWITCH_KEYS):
        raise RecoveryError("derived EvalLog does not expose exactly five switch aggregate rows")
    return changed


def _repair_one_log(
    cell: LoadedCell,
    *,
    clean: LoadedCell,
    wrong_clean: LoadedCell,
    switch_values: Any,
) -> tuple[Any, dict[str, Any]]:
    """Deep-copy one log, changing only its bad switch scoring evidence."""

    _validate_known_bad_binding(cell, clean=clean, wrong_clean=wrong_clean)
    expected_metadata_changes = EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK[cell.task_index]
    original_samples = list(_attr(cell.log, "samples", []) or [])
    known_bad_aggregate_rows_sha256 = _validate_known_bad_aggregate_switch_results(cell.log)
    clean_answers = _clean_answers(clean)
    derived = copy.deepcopy(cell.log)
    derived_samples = list(_attr(derived, "samples", []) or [])
    if len(original_samples) != len(derived_samples):  # pragma: no cover - deepcopy guard
        raise RecoveryError(f"task-{cell.task_index} copy lost samples")
    changed_by_key = {key: 0 for key in SWITCH_KEYS}
    changed_metadata_fields: dict[str, int] = {"unbiased_log": 0, "unbiased_answer": 0, "note_removed": 0}
    changed_samples = 0
    preserved_rows: list[dict[str, Any]] = []
    switch_rows_before: list[dict[str, Any]] = []
    switch_rows_after: list[dict[str, Any]] = []
    switch_inputs: list[dict[str, Any]] = []
    for original, replacement in zip(original_samples, derived_samples, strict=True):
        original_id = _attr(original, "id", "")
        if _attr(replacement, "id", "") != original_id:
            raise RecoveryError(f"task-{cell.task_index} copy changed a sample ID")
        if _score_snapshot_without_switch(original) != _score_snapshot_without_switch(replacement):
            raise RecoveryError(f"task-{cell.task_index} copy changed a non-switch sample field")
        original_metadata = _mapping(_attr(original, "metadata", {}))
        replacement_metadata = _mapping(_attr(replacement, "metadata", {}))
        original_answer = _answer(original)
        if _answer(replacement) != original_answer or replacement_metadata.get("biased_option") != original_metadata.get("biased_option"):
            raise RecoveryError(f"task-{cell.task_index} copy changed a biased answer or target")
        biased_option = original_metadata.get("biased_option")
        if not isinstance(biased_option, str) or not biased_option or original_id not in clean_answers:
            raise RecoveryError(f"task-{cell.task_index} cannot bind sample {original_id!r} to clean evidence")
        source_score = _switch_score(original)
        target_score = _switch_score(replacement)
        expected_value = dict(switch_values(original_answer, clean_answers[original_id], biased_option))
        expected_metadata = _expected_switch_metadata(clean_log=clean.original_path, clean_answer=clean_answers[original_id])
        before_value = _switch_value(source_score)
        before_metadata = _switch_metadata(source_score)
        for key in SWITCH_KEYS:
            if not _equivalent(before_value[key], expected_value[key]):
                changed_by_key[key] += 1
        if before_metadata.get("unbiased_log") != expected_metadata["unbiased_log"]:
            changed_metadata_fields["unbiased_log"] += 1
        if not _equivalent(before_metadata.get("unbiased_answer"), expected_metadata["unbiased_answer"]):
            changed_metadata_fields["unbiased_answer"] += 1
        if "note" in before_metadata:
            changed_metadata_fields["note_removed"] += 1
        if not _equivalent(before_value, expected_value) or before_metadata != expected_metadata:
            changed_samples += 1
        _set_attr(target_score, "value", expected_value)
        _set_attr(target_score, "metadata", expected_metadata)
        if _score_snapshot_without_switch(original) != _score_snapshot_without_switch(replacement):
            raise RecoveryError(f"task-{cell.task_index} repair altered a non-switch sample field")
        preserved_rows.append(
            {
                "sample_id": original_id,
                "biased_answer": original_answer,
                "biased_option": biased_option,
            }
        )
        switch_rows_before.append({"sample_id": original_id, "value": before_value, "metadata": before_metadata})
        switch_rows_after.append({"sample_id": original_id, "value": expected_value, "metadata": expected_metadata})
        switch_inputs.append(
            {
                "sample_id": original_id,
                "biased_answer": original_answer,
                "unbiased_answer": clean_answers[original_id],
                "biased_option": biased_option,
            }
        )
    if changed_metadata_fields != expected_metadata_changes:
        raise RecoveryError(
            f"task-{cell.task_index} does not retain the exact known switch-metadata delta profile"
        )
    aggregate_changes = _repair_aggregate_switch_results(derived)
    # Recheck all derived score values/bindings before any file is published.
    temporary = LoadedCell(
        task_index=cell.task_index,
        receipt=cell.receipt,
        original_path=cell.original_path,
        log=derived,
        spec=cell.spec,
        source_prefix=cell.source_prefix,
        sample_ids=_sample_ids(derived, task_index=cell.task_index),
    )
    _validate_sorted_sample_semantics(temporary)
    _validate_correct_switch_binding(temporary, clean=clean, switch_values=switch_values)
    if changed_samples != len(original_samples):
        raise RecoveryError(f"task-{cell.task_index} score-only repair did not replace every known-bad switch score")
    return derived, {
        "task_index": cell.task_index,
        "sample_count": len(original_samples),
        "changed_sample_count": changed_samples,
        "changed_score_keys": {key: changed_by_key[key] for key in SWITCH_KEYS},
        "changed_metadata_fields": changed_metadata_fields,
        "aggregate_changes": aggregate_changes,
        "known_bad_aggregate_rows_sha256": known_bad_aggregate_rows_sha256,
        "clean_task_index": clean.task_index,
        "clean_log": _identity(clean.original_path, label=f"task-{cell.task_index} clean scoring source"),
        "switch_inputs_sha256": _digest(switch_inputs),
        "switch_scores_before_sha256": _digest(switch_rows_before),
        "switch_scores_after_sha256": _digest(switch_rows_after),
        "sample_ids_sha256": _digest([row["sample_id"] for row in preserved_rows]),
        "biased_answers_sha256": _digest([row["biased_answer"] for row in preserved_rows]),
        "biased_options_sha256": _digest([row["biased_option"] for row in preserved_rows]),
        "non_switch_sample_payload_sha256": _digest([_score_snapshot_without_switch(sample) for sample in original_samples]),
        "derived_non_switch_sample_payload_sha256": _digest([_score_snapshot_without_switch(sample) for sample in derived_samples]),
    }


def _write_immutable_eval_log(log: Any, path: Path) -> str:
    """Write a derived EvalLog once without ever opening an original log for write."""

    parent = _require_regular_directory(path.parent, label="derived EvalLog parent", create=True)
    target = parent / path.name
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise RecoveryError(f"derived EvalLog target is unsafe: {target}")
        return "resumed"
    try:
        from inspect_ai.log import write_eval_log
    except ImportError as exc:  # pragma: no cover - requires the frozen evaluator venv
        raise RecoveryError("Inspect AI is required to write the score-only derived EvalLog") from exc
    temporary = parent / f".{target.stem}.{uuid.uuid4().hex}.eval"
    try:
        write_eval_log(log, temporary, format="eval")
        if temporary.is_symlink() or not temporary.is_file():
            raise RecoveryError(f"Inspect did not write a regular derived EvalLog: {temporary}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file():
                raise RecoveryError(f"derived EvalLog target appeared unsafe: {target}") from None
            return "resumed"
    finally:
        if temporary.exists():
            temporary.unlink()
    return "written"


def _scorer_identity() -> tuple[Any, dict[str, Any]]:
    """Load only pure score code and bind both upstream and compatibility bytes."""

    try:
        import mcq_bias.scorers as scorer_module
        from ctm_data.adapters.mcq_bias.scorer_compat import install_conditional_nan_compat
    except ImportError as exc:  # pragma: no cover - requires frozen evaluator venv
        raise RecoveryError("the frozen mcq-bias scorer is unavailable") from exc
    source_path = _require_regular_file(inspect.getsourcefile(scorer_module.switch_values) or "", label="mcq-bias switch scorer source")
    compat_path = _require_regular_file(inspect.getsourcefile(install_conditional_nan_compat) or "", label="switch scorer compatibility source")
    source_text = inspect.getsource(scorer_module.switch_values)
    install_conditional_nan_compat()
    return scorer_module.switch_values, {
        "switch_values_source": _identity(source_path, label="mcq-bias switch scorer source"),
        "switch_values_function_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "nan_compat_source": _identity(compat_path, label="switch scorer compatibility source"),
    }


def _load_cells(
    *,
    frozen: Any,
    paths: Any,
    launch: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    launch_sha256: str,
    evaluation_sha256: str,
) -> tuple[list[LoadedCell], list[Any]]:
    try:
        from inspect_ai.log import read_eval_log
        from experiments.stage2_ood_hle import raw_preflight as mechanical
        from experiments.stage2_ood_hle.tasks import ood_task_specs
    except ImportError as exc:  # pragma: no cover - requires frozen evaluator venv
        raise RecoveryError("frozen Stage-2 preflight dependencies are unavailable") from exc
    deployment = launch.get("deployment_manifest")
    runtime = evaluation.get("runtime")
    if not isinstance(deployment, Mapping) or not isinstance(runtime, Mapping):
        raise RecoveryError("r002 launch/evaluation receipt has no deployment/runtime evidence")
    manifest = deployment.get("path")
    if not isinstance(manifest, str) or not Path(manifest).is_absolute():
        raise RecoveryError("r002 launch receipt has no absolute deployment manifest")
    specs = list(ood_task_specs(manifest))
    frozen._validate_r002_sampling_matrix(specs)
    snapshot_runtime = frozen._early_snapshot_vllm_runtime(runtime)
    cells: list[LoadedCell] = []
    for task_index, spec in enumerate(specs, start=1):
        receipt = frozen._load_task_receipt(
            paths,
            task_index=task_index,
            launch_contract_sha256=launch_sha256,
            evaluation_receipt_sha256=evaluation_sha256,
        )
        if receipt is None:
            raise RecoveryError(f"r002 task-{task_index} receipt is absent")
        canonical = receipt.get("canonical_log")
        if not isinstance(canonical, Mapping):
            raise RecoveryError(f"r002 task-{task_index} receipt has no canonical log")
        path = _require_regular_file(canonical.get("path", ""), label=f"r002 task-{task_index} canonical EvalLog")
        if dict(canonical) != frozen._identity(path, label=f"r002 task-{task_index} canonical EvalLog"):
            raise RecoveryError(f"r002 task-{task_index} canonical EvalLog changed after receipt publication")
        header = read_eval_log(str(path), header_only=True)
        try:
            mechanical._validate_header(header, path=path, spec=spec, raw_root=paths.raw)
            frozen._assert_early_snapshot_vllm_runtime(header, path=path, runtime=snapshot_runtime)
        except Exception as exc:
            raise RecoveryError(f"r002 task-{task_index} header/runtime replay failed: {exc}") from exc
        log = read_eval_log(str(path), header_only=False)
        prefix = _source_prefix_from_log(log, spec=spec, cap=_task_sample_count(frozen, task_index), task_index=task_index)
        ids = _sample_ids(log, task_index=task_index)
        cell = LoadedCell(task_index, receipt, path, log, spec, prefix, ids)
        _validate_sorted_sample_semantics(cell)
        cells.append(cell)
    return cells, specs


def _verify_pair_orders(cells: Sequence[LoadedCell], specs: Sequence[Any]) -> None:
    by_index = {cell.task_index: cell for cell in cells}
    for task_index, cell in by_index.items():
        if getattr(cell.spec, "kind", None) != "biased":
            continue
        clean = by_index[_expected_clean_index(specs, task_index)]
        if cell.source_prefix != clean.source_prefix or cell.sample_ids != clean.sample_ids:
            raise RecoveryError(f"task-{task_index} does not retain the exact matching clean source/sample ordering")


def _validate_corrected_native_selection(
    *,
    frozen: Any,
    paths: Any,
    launch: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    specs: Sequence[Any],
    published: Mapping[int, Mapping[str, Any]],
    switch_values: Any,
) -> dict[str, Any]:
    """Run the corrected native preflight over original+derived selected logs.

    The frozen r002 finalizer's only incorrect assumption was that Inspect
    preserved source order in ``log.samples``.  This replays its header,
    runtime, source-prefix, and Stage-2 switch checks while treating the full
    sample list as the canonical sorted representation of the exact selected
    dataset ID set.
    """

    try:
        from inspect_ai.log import read_eval_log
        from experiments.stage2_ood_hle import raw_preflight as mechanical
    except ImportError as exc:  # pragma: no cover - requires frozen evaluator venv
        raise RecoveryError("corrected native preflight dependencies are unavailable") from exc
    runtime = evaluation.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RecoveryError("r002 evaluation receipt has no runtime for corrected native preflight")
    snapshot_runtime = frozen._early_snapshot_vllm_runtime(runtime)
    selected: dict[int, LoadedCell] = {}
    loaded: dict[int, Any] = {}
    full_logs: dict[int, Any] = {}
    task_records: list[dict[str, Any]] = []
    for task_index, spec in enumerate(specs, start=1):
        expected_identity = published.get(task_index)
        if not isinstance(expected_identity, Mapping):
            raise RecoveryError(f"corrected native preflight has no published identity for task-{task_index}")
        path = _require_regular_file(expected_identity.get("path", ""), label=f"corrected native task-{task_index} EvalLog")
        observed_identity = _identity(path, label=f"corrected native task-{task_index} EvalLog")
        if dict(expected_identity) != observed_identity:
            raise RecoveryError(f"corrected native task-{task_index} EvalLog differs from its selected identity")
        try:
            header = read_eval_log(str(path), header_only=True)
            created = mechanical._validate_header(header, path=path, spec=spec, raw_root=paths.raw)
            model, observed_runtime = frozen._assert_early_snapshot_vllm_runtime(
                header,
                path=path,
                runtime=snapshot_runtime,
            )
            # This reuses r002's independent limit/full-source/prefix check.
            frozen._validate_promotable_eval_log(path=path, launch=launch, task_index=task_index)
            full_log = read_eval_log(str(path), header_only=False)
        except Exception as exc:
            raise RecoveryError(f"corrected native preflight could not validate task-{task_index}: {exc}") from exc
        prefix = _source_prefix_from_log(
            full_log,
            spec=spec,
            cap=_task_sample_count(frozen, task_index),
            task_index=task_index,
        )
        sample_ids = _sample_ids(full_log, task_index=task_index)
        selected_cell = LoadedCell(
            task_index=task_index,
            receipt={},
            original_path=path,
            log=full_log,
            spec=spec,
            source_prefix=prefix,
            sample_ids=sample_ids,
        )
        _validate_sorted_sample_semantics(selected_cell)
        selected[task_index] = selected_cell
        full_logs[task_index] = full_log
        loaded[task_index] = mechanical.LoadedTaskLog(
            frozen._subset_spec_for_task(spec, task_index=task_index),
            path,
            created,
            header,
            model,
            observed_runtime,
        )
    _verify_pair_orders(list(selected.values()), specs)
    clean_paths = {
        (loaded[index].spec.population, loaded[index].spec.dataset): loaded[index].path
        for index in frozen.CLEAN_TASK_INDICES
    }
    if len(clean_paths) != len(frozen.CLEAN_TASK_INDICES):
        raise RecoveryError("corrected native preflight has no unique matching clean reference")
    for task_index in range(1, frozen.TASK_COUNT + 1):
        try:
            sample_count = mechanical._validate_samples(
                full_logs[task_index],
                loaded=loaded[task_index],
                clean_paths=clean_paths,
            )
        except Exception as exc:
            raise RecoveryError(f"corrected native switch/sample preflight failed for task-{task_index}: {exc}") from exc
        if sample_count != _task_sample_count(frozen, task_index):
            raise RecoveryError(f"corrected native preflight saw an invalid sample count for task-{task_index}")
        if getattr(selected[task_index].spec, "kind", None) == "biased":
            clean_index = _expected_clean_index(specs, task_index)
            _validate_correct_switch_binding(
                selected[task_index],
                clean=selected[clean_index],
                switch_values=switch_values,
            )
        task_records.append(
            {
                "task_index": task_index,
                "published_log": _identity(
                    loaded[task_index].path,
                    label=f"corrected native task-{task_index} identity",
                ),
                "sample_count": sample_count,
                "dataset_sample_ids_sha256": _digest(selected[task_index].source_prefix),
                "full_sample_ids_sha256": _digest(selected[task_index].sample_ids),
            }
        )
    return {
        "validator": "frozen-r002-prefix-plus-stage2-switch-preflight",
        "task_indices": list(range(1, frozen.TASK_COUNT + 1)),
        "switch_bindings": "all-validated",
        "dataset_sample_ids": "all tasks match their exact ordered source prefix",
        "full_samples": "all tasks are the exact source-ID set in deterministic lexicographic sample-id order",
        "tasks": task_records,
    }


def _derive_artifact_name(*, original: Mapping[str, Any], repair: Mapping[str, Any], scorer: Mapping[str, Any]) -> str:
    key = _digest({"original": original, "repair": repair, "scorer": scorer})
    return f"{key}.eval"


def _source_record(
    *,
    cell: LoadedCell,
    published: Mapping[str, Any],
    original: Mapping[str, Any],
    clean: LoadedCell | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_index": cell.task_index,
        "identity": [
            getattr(cell.spec, "kind", None),
            getattr(cell.spec, "regime", None),
            getattr(cell.spec, "population", None),
            getattr(cell.spec, "dataset", None),
            getattr(cell.spec, "bias_type", None),
        ],
        # Filled by _build_preflight from the frozen receipt location.  Do not
        # infer it from an EvalLog path: receipts are the authoritative selector.
        "task_receipt": None,
        "original_log": dict(original),
        "published_log": dict(published),
        "publication_kind": "derived-score-only" if cell.task_index in REPAIRED_TASKS else "original-unchanged",
        "source_prefix_sha256": _digest(cell.source_prefix),
        "full_sample_ids_sha256": _digest(cell.sample_ids),
        "sample_count": len(cell.sample_ids),
        "sample_order": "lexicographic-by-sample-id",
    }
    if clean is not None:
        record["paired_clean"] = _identity(clean.original_path, label=f"task-{cell.task_index} matching clean EvalLog")
    return record


def _build_preflight(
    *,
    frozen: Any,
    target: Any,
    paths: Any,
    campaign: Path,
    recovery_root: Path,
    launch: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    cells: Sequence[LoadedCell],
    specs: Sequence[Any],
    originals: Mapping[int, Mapping[str, Any]],
    published: Mapping[int, Mapping[str, Any]],
    repair_records: Sequence[Mapping[str, Any]],
    scorer_identity: Mapping[str, Any],
    frozen_sources: Mapping[str, Any],
    corrected_native_validation: Mapping[str, Any],
    clean_gate_identity: Mapping[str, Any],
) -> dict[str, Any]:
    by_index = {cell.task_index: cell for cell in cells}
    source_records = []
    for cell in cells:
        clean = None if getattr(cell.spec, "kind", None) == "unbiased" else by_index[_expected_clean_index(specs, cell.task_index)]
        record = _source_record(cell=cell, published=published[cell.task_index], original=originals[cell.task_index], clean=clean)
        receipt_path = Path(paths.receipts) / f"task-{cell.task_index:03d}.json"
        record["task_receipt"] = _identity(receipt_path, label=f"task-{cell.task_index} receipt")
        source_records.append(record)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "condition": target.condition,
        "target": {"step": target.step, "run_name": target.run_name, "segment_index": target.segment_index},
        "recovery_root": str(recovery_root),
        "original_custody": {
            "campaign_root": str(campaign),
            "phase_one": _identity(campaign / "phases" / "phase-001.json", label="r002 phase-one receipt"),
            "clean_gate": dict(clean_gate_identity),
            "launch_contract": _identity(paths.contract, label="r002 launch contract"),
            "evaluation_receipt": _identity(paths.evaluation_receipt, label="r002 evaluation receipt"),
            "raw_root": str(paths.raw),
        },
        "frozen_r002_sources": dict(frozen_sources),
        "recovery_implementation": {
            "program": _identity(Path(__file__), label="score-only recovery program"),
            "frozen_launcher": _identity(Path(frozen.__file__), label="frozen r002 launcher"),
            "scorer": dict(scorer_identity),
            "execution": {
                "mode": "score-only-derived-repair",
                "model_calls": 0,
                "generation_calls": 0,
                "network_calls": 0,
                "allowed_operations": ["read_eval_log", "pure_switch_values", "write_eval_log"],
                "no_model_call_proof": {
                    "program_identity": _identity(Path(__file__), label="score-only recovery program"),
                    "model_or_generation_entrypoints": "not imported or invoked",
                    "offline_environment": {
                        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                    },
                },
            },
        },
        "selection_contract": {
            "dataset_sample_ids": "exact ordered source prefix spec.question_ids[:task_sample_count]",
            "full_samples": "exact unique set represented as lexicographically sorted(eval.dataset.sample_ids)",
            "paired_variants": "exact same source prefix and full sample-id order as the matching clean cell",
            "task_sample_counts": [{"task_index": cell.task_index, "sample_count": len(cell.sample_ids)} for cell in cells],
        },
        "corrected_native_validation": dict(corrected_native_validation),
        "repairs": [dict(record) for record in repair_records],
        "unchanged_task_indices": [index for index in range(1, len(cells) + 1) if index not in REPAIRED_TASKS],
        "verified_unchanged": dict(VERIFIED_UNCHANGED_TASKS),
        "sources": source_records,
    }


def validate_corrected_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the self-contained recovery report before a publication uses it."""

    report = dict(value)
    expected = {
        "schema",
        "condition",
        "target",
        "recovery_root",
        "original_custody",
        "frozen_r002_sources",
        "recovery_implementation",
        "selection_contract",
        "corrected_native_validation",
        "repairs",
        "unchanged_task_indices",
        "verified_unchanged",
        "sources",
    }
    if set(report) != expected or report.get("schema") != PREFLIGHT_SCHEMA:
        raise RecoveryError("score-only recovery preflight has an unsupported schema")
    recovery_root = report.get("recovery_root")
    if (
        not isinstance(recovery_root, str)
        or not Path(recovery_root).is_absolute()
        or Path(recovery_root).name != RECOVERY_NAMESPACE
    ):
        raise RecoveryError("score-only recovery preflight does not bind the fresh immutable recovery namespace")
    original_custody = report.get("original_custody")
    if (
        not isinstance(original_custody, Mapping)
        or set(original_custody) != {
            "campaign_root",
            "phase_one",
            "clean_gate",
            "launch_contract",
            "evaluation_receipt",
            "raw_root",
        }
        or not isinstance(original_custody.get("campaign_root"), str)
        or not Path(original_custody["campaign_root"]).is_absolute()
        or not isinstance(original_custody.get("raw_root"), str)
        or not Path(original_custody["raw_root"]).is_absolute()
        or not all(
            _is_identity(original_custody.get(field))
            for field in ("phase_one", "clean_gate", "launch_contract", "evaluation_receipt")
        )
    ):
        raise RecoveryError("score-only recovery preflight has incomplete original clean-gate custody")
    repairs = report.get("repairs")
    if not isinstance(repairs, list) or len(repairs) != len(REPAIRED_TASKS) or [entry.get("task_index") for entry in repairs if isinstance(entry, Mapping)] != list(REPAIRED_TASKS):
        raise RecoveryError("score-only recovery preflight has the wrong repair task list")
    for repair in repairs:
        expected_repair_fields = {
            "task_index",
            "sample_count",
            "changed_sample_count",
            "changed_score_keys",
            "changed_metadata_fields",
            "aggregate_changes",
            "known_bad_aggregate_rows_sha256",
            "clean_task_index",
            "clean_log",
            "switch_inputs_sha256",
            "switch_scores_before_sha256",
            "switch_scores_after_sha256",
            "sample_ids_sha256",
            "biased_answers_sha256",
            "biased_options_sha256",
            "non_switch_sample_payload_sha256",
            "derived_non_switch_sample_payload_sha256",
        }
        if (
            not isinstance(repair, Mapping)
            or set(repair) != expected_repair_fields
            or not isinstance(repair.get("sample_count"), int)
            or repair["sample_count"] <= 0
            or repair.get("changed_sample_count") != repair["sample_count"]
        ):
            raise RecoveryError("score-only recovery preflight has an incomplete score repair")
        changed = repair.get("changed_score_keys")
        if (
            not isinstance(changed, Mapping)
            or set(changed) != set(SWITCH_KEYS)
            or any(not isinstance(count, int) or count < 0 or count > repair["sample_count"] for count in changed.values())
        ):
            raise RecoveryError("score-only recovery preflight has invalid changed switch-score counts")
        expected_metadata_changes = EXPECTED_METADATA_CHANGE_COUNTS_BY_TASK.get(repair.get("task_index"))
        metadata_changed = repair.get("changed_metadata_fields")
        if (
            not isinstance(metadata_changed, Mapping)
            or repair.get("sample_count") != REPAIR_SAMPLE_COUNT_BY_TASK.get(repair.get("task_index"))
            or metadata_changed != expected_metadata_changes
        ):
            raise RecoveryError("score-only recovery preflight has an incomplete exact switch-metadata repair")
        aggregate = repair.get("aggregate_changes")
        if not isinstance(aggregate, Mapping) or set(aggregate) != set(SWITCH_KEYS):
            raise RecoveryError("score-only recovery preflight lacks complete switch aggregate evidence")
        for aggregate_change in aggregate.values():
            if (
                not isinstance(aggregate_change, Mapping)
                or not isinstance(aggregate_change.get("scored_samples_after"), int)
                or not isinstance(aggregate_change.get("unscored_samples_after"), int)
                or aggregate_change["scored_samples_after"] + aggregate_change["unscored_samples_after"] != repair["sample_count"]
                or not _is_sha256(aggregate_change.get("before_sha256"))
                or not _is_sha256(aggregate_change.get("after_sha256"))
            ):
                raise RecoveryError("score-only recovery preflight has invalid switch aggregate evidence")
        expected_clean_task = REPAIR_CLEAN_TASK_BY_TASK.get(repair.get("task_index"))
        if repair.get("clean_task_index") != expected_clean_task or not _is_identity(repair.get("clean_log")):
            raise RecoveryError("score-only recovery preflight does not bind its exact clean scoring evidence")
        if repair.get("non_switch_sample_payload_sha256") != repair.get("derived_non_switch_sample_payload_sha256"):
            raise RecoveryError("score-only recovery preflight changed a non-switch sample payload")
        for field in (
            "known_bad_aggregate_rows_sha256",
            "switch_inputs_sha256",
            "switch_scores_before_sha256",
            "switch_scores_after_sha256",
            "sample_ids_sha256",
            "biased_answers_sha256",
            "biased_options_sha256",
            "non_switch_sample_payload_sha256",
        ):
            if not _is_sha256(repair.get(field)):
                raise RecoveryError(f"score-only recovery preflight has invalid {field}")
        if repair["switch_scores_before_sha256"] == repair["switch_scores_after_sha256"]:
            raise RecoveryError("score-only recovery preflight does not bind an actual switch-score repair")
    sources = report.get("sources")
    if not isinstance(sources, list) or len(sources) != 21:
        raise RecoveryError("score-only recovery preflight must contain all 21 r002 task sources")
    source_by_task_index: dict[int, Mapping[str, Any]] = {}
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, Mapping) or source.get("task_index") != index:
            raise RecoveryError("score-only recovery preflight source ordering drifted")
        if source.get("publication_kind") != ("derived-score-only" if index in REPAIRED_TASKS else "original-unchanged"):
            raise RecoveryError("score-only recovery preflight source publication kind drifted")
        for field in ("original_log", "published_log", "task_receipt"):
            candidate = source.get(field)
            if not _is_identity(candidate):
                raise RecoveryError(f"score-only recovery preflight has invalid {field}")
        if source.get("sample_order") != "lexicographic-by-sample-id":
            raise RecoveryError("score-only recovery preflight has an unsupported sample-order rule")
        if not _is_sha256(source.get("source_prefix_sha256")) or not _is_sha256(source.get("full_sample_ids_sha256")):
            raise RecoveryError("score-only recovery preflight has malformed ordered sample evidence")
        if not isinstance(source.get("sample_count"), int) or source["sample_count"] <= 0:
            raise RecoveryError("score-only recovery preflight has invalid sample count evidence")
        if index in REPAIRED_TASKS:
            if source["original_log"].get("sha256") == source["published_log"].get("sha256"):
                raise RecoveryError("score-only recovery preflight did not publish a distinct repaired EvalLog")
        elif source["original_log"] != source["published_log"]:
            raise RecoveryError("score-only recovery preflight changed an unrepaired EvalLog")
        source_by_task_index[index] = source
    for repair in repairs:
        expected_clean_task = REPAIR_CLEAN_TASK_BY_TASK[repair["task_index"]]
        if repair["clean_log"] != source_by_task_index[expected_clean_task]["original_log"]:
            raise RecoveryError("score-only recovery preflight does not bind the receipt-selected clean EvalLog")
        if repair["sample_count"] != source_by_task_index[repair["task_index"]]["sample_count"]:
            raise RecoveryError("score-only recovery preflight repair count differs from its receipt-selected EvalLog")
    if report.get("unchanged_task_indices") != [index for index in range(1, 22) if index not in REPAIRED_TASKS]:
        raise RecoveryError("score-only recovery preflight has an incomplete unchanged-task proof")
    if report.get("verified_unchanged") != VERIFIED_UNCHANGED_TASKS:
        raise RecoveryError("score-only recovery preflight has an incomplete task15/HellaSwag/HLE unchanged proof")
    native = report.get("corrected_native_validation")
    native_tasks = _mapping(native).get("tasks")
    if (
        not isinstance(native, Mapping)
        or native.get("task_indices") != list(range(1, 22))
        or native.get("switch_bindings") != "all-validated"
        or not isinstance(native_tasks, list)
        or len(native_tasks) != 21
        or [entry.get("task_index") for entry in native_tasks if isinstance(entry, Mapping)] != list(range(1, 22))
    ):
        raise RecoveryError("score-only recovery preflight lacks a corrected native validation result")
    for source, native_task in zip(sources, native_tasks, strict=True):
        if (
            not isinstance(native_task, Mapping)
            or native_task.get("published_log") != source.get("published_log")
            or native_task.get("sample_count") != source.get("sample_count")
            or native_task.get("dataset_sample_ids_sha256") != source.get("source_prefix_sha256")
            or native_task.get("full_sample_ids_sha256") != source.get("full_sample_ids_sha256")
        ):
            raise RecoveryError("corrected native preflight drifted from its bound selected EvalLogs")
    execution = _mapping(_mapping(report.get("recovery_implementation")).get("execution"))
    proof = _mapping(execution.get("no_model_call_proof"))
    if (
        execution.get("model_calls") != 0
        or execution.get("generation_calls") != 0
        or execution.get("network_calls") != 0
        or execution.get("allowed_operations") != ["read_eval_log", "pure_switch_values", "write_eval_log"]
        or proof.get("model_or_generation_entrypoints") != "not imported or invoked"
        or not _is_identity(proof.get("program_identity"))
        or _mapping(proof.get("offline_environment")) != {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    ):
        raise RecoveryError("score-only recovery preflight does not prove a no-model-call execution")
    return report


def validate_completion(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable, derived-only completion receipt."""

    receipt = dict(value)
    expected = {
        "schema",
        "condition",
        "target",
        "preflight",
        "original_task_receipts",
        "original_task_logs",
        "published_task_logs",
        "repaired_tasks",
    }
    if set(receipt) != expected or receipt.get("schema") != COMPLETION_SCHEMA:
        raise RecoveryError("score-only recovery completion has an unsupported schema")
    if not _is_identity(receipt.get("preflight")):
        raise RecoveryError("score-only recovery completion lacks a preflight identity")
    for field in ("original_task_receipts", "original_task_logs", "published_task_logs"):
        records = receipt.get(field)
        if not isinstance(records, list) or len(records) != 21 or not all(_is_identity(record) for record in records):
            raise RecoveryError(f"score-only recovery completion has malformed {field}")
    if receipt.get("repaired_tasks") != list(REPAIRED_TASKS):
        raise RecoveryError("score-only recovery completion has the wrong repair task list")
    for index, (original, published) in enumerate(zip(receipt["original_task_logs"], receipt["published_task_logs"], strict=True), start=1):
        if index not in REPAIRED_TASKS and original != published:
            raise RecoveryError("score-only recovery completion changed an unrepaired EvalLog")
        if index in REPAIRED_TASKS and original.get("sha256") == published.get("sha256"):
            raise RecoveryError("score-only recovery completion did not bind a derived repaired EvalLog")
    return receipt


def validate_attestation(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the final no-mutation recovery attestation."""

    attestation = dict(value)
    expected = {"schema", "completion", "preflight", "original_custody", "mutations", "execution", "repair_tasks"}
    if set(attestation) != expected or attestation.get("schema") != ATTESTATION_SCHEMA:
        raise RecoveryError("score-only recovery attestation has an unsupported schema")
    if not _is_identity(attestation.get("completion")) or not _is_identity(attestation.get("preflight")):
        raise RecoveryError("score-only recovery attestation lacks immutable receipt identities")
    original_custody = _mapping(attestation.get("original_custody"))
    if set(original_custody) != {"before", "after"}:
        raise RecoveryError("score-only recovery attestation lacks original clean-gate custody")
    before_custody = _mapping(original_custody.get("before"))
    after_custody = _mapping(original_custody.get("after"))
    if (
        set(before_custody) != {"phase_one", "clean_gate"}
        or set(after_custody) != {"phase_one", "clean_gate"}
        or not all(
            _is_identity(custody.get(field))
            for custody in (before_custody, after_custody)
            for field in ("phase_one", "clean_gate")
        )
        or before_custody != after_custody
    ):
        raise RecoveryError("score-only recovery attestation has changed original clean-gate custody")
    mutations = _mapping(attestation.get("mutations"))
    if mutations != {
        "original_eval_logs": 0,
        "original_task_receipts": 0,
        "original_launch_contract": 0,
        "original_phase_receipts": 0,
        "original_clean_gate_receipt": 0,
        "step64_artifacts": 0,
        "derived_eval_logs": len(REPAIRED_TASKS),
    }:
        raise RecoveryError("score-only recovery attestation has an unsupported mutation record")
    if attestation.get("repair_tasks") != list(REPAIRED_TASKS):
        raise RecoveryError("score-only recovery attestation has the wrong repair task list")
    execution = _mapping(attestation.get("execution"))
    if execution.get("model_calls") != 0 or execution.get("generation_calls") != 0 or execution.get("network_calls") != 0:
        raise RecoveryError("score-only recovery attestation does not preserve its no-model-call proof")
    return attestation


def recover(
    *,
    step16_root: str | Path,
    campaign_root: str | Path,
    recovery_root: str | Path,
    frozen_launcher: str | Path,
    write: bool,
) -> dict[str, Any]:
    """Revalidate r002 and optionally publish only immutable derived score repairs."""

    _require_offline_environment()
    frozen = _load_frozen_launcher(frozen_launcher)
    target = frozen._target(16)
    paths = frozen._paths(step16_root, target=target)
    campaign = _require_regular_directory(campaign_root, label="r002 campaign root")
    recovery = Path(recovery_root).expanduser()
    if not recovery.is_absolute():
        raise RecoveryError("score-only recovery root must be an absolute path")
    if recovery.exists() and recovery.is_symlink():
        raise RecoveryError("score-only recovery root cannot be a symlink")
    recovery = recovery.resolve()
    if recovery.name != RECOVERY_NAMESPACE:
        raise RecoveryError(f"score-only recovery must use the fresh immutable namespace: {RECOVERY_NAMESPACE}")
    recovery_namespace_path = campaign / "recoveries"
    if recovery_namespace_path.exists() and (
        recovery_namespace_path.is_symlink() or not recovery_namespace_path.is_dir()
    ):
        raise RecoveryError("score-only campaign recoveries namespace must be a non-linked regular directory")
    recovery_namespace = recovery_namespace_path.resolve()
    try:
        recovery.relative_to(recovery_namespace)
    except ValueError as exc:
        raise RecoveryError("score-only recovery root must live under the separate campaign recoveries namespace") from exc
    if recovery == recovery_namespace or recovery.is_relative_to(paths.root):
        raise RecoveryError("score-only recovery root overlaps a frozen r002 output namespace")
    phase_one = campaign / "phases" / "phase-001.json"
    phase_two = campaign / "phases" / "phase-002.json"
    if phase_two.exists() or phase_two.is_symlink():
        raise RecoveryError("official r002 phase-two receipt exists; score-only step16 recovery is no longer admissible")
    phase_custody = _validate_phase_one_read_only(
        frozen=frozen,
        campaign=campaign,
        phase_one=phase_one,
        paths=paths,
        target=target,
    )
    if paths.native_preflight.exists() or paths.native_preflight.is_symlink() or paths.completion.exists() or paths.completion.is_symlink():
        raise RecoveryError("original r002 step16 already has native preflight/completion; refusing a competing recovery")
    launch, launch_sha256 = frozen._load_launch(paths, target=target)
    evaluation, evaluation_sha256 = frozen._load_evaluation_receipt(
        paths,
        target=target,
        launch=launch,
        launch_sha256=launch_sha256,
    )
    frozen._validate_canonical_raw_custody(
        paths=paths,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    )
    frozen_sources = _verify_frozen_sources(frozen, launch)
    immutable_custody_before = {
        "phase_one": phase_custody["phase_one"],
        "clean_gate": phase_custody["clean_gate"],
        "launch_contract": _identity(paths.contract, label="r002 launch contract"),
        "evaluation_receipt": _identity(paths.evaluation_receipt, label="r002 evaluation receipt"),
        "task_receipts": {
            task_index: _identity(paths.receipts / f"task-{task_index:03d}.json", label=f"r002 task-{task_index} receipt")
            for task_index in range(1, frozen.TASK_COUNT + 1)
        },
    }
    originals_before = {
        task_index: _identity(Path(str(frozen._load_task_receipt(paths, task_index=task_index, launch_contract_sha256=launch_sha256, evaluation_receipt_sha256=evaluation_sha256)["canonical_log"]["path"])), label=f"original task-{task_index} EvalLog")
        for task_index in range(1, frozen.TASK_COUNT + 1)
    }
    cells, specs = _load_cells(
        frozen=frozen,
        paths=paths,
        launch=launch,
        evaluation=evaluation,
        launch_sha256=launch_sha256,
        evaluation_sha256=evaluation_sha256,
    )
    _verify_pair_orders(cells, specs)
    by_index = {cell.task_index: cell for cell in cells}
    switch_values, scorer_identity = _scorer_identity()
    for task_index, cell in by_index.items():
        if getattr(cell.spec, "kind", None) != "biased":
            continue
        clean = by_index[_expected_clean_index(specs, task_index)]
        if task_index in REPAIRED_TASKS:
            _validate_known_bad_binding(cell, clean=clean, wrong_clean=by_index[2])
        else:
            _validate_correct_switch_binding(cell, clean=clean, switch_values=switch_values)
    if UNCHANGED_IID_LOGIQA_TASK in REPAIRED_TASKS:  # pragma: no cover - fixed constant guard
        raise RecoveryError("the known-good LogiQA task must never be repaired")
    derived_logs: dict[int, Any] = {}
    repair_records: list[dict[str, Any]] = []
    for task_index in REPAIRED_TASKS:
        cell = by_index[task_index]
        clean = by_index[_expected_clean_index(specs, task_index)]
        derived, record = _repair_one_log(
            cell,
            clean=clean,
            wrong_clean=by_index[2],
            switch_values=switch_values,
        )
        derived_logs[task_index] = derived
        repair_records.append(record)
    # An all-read run exposes the exact repair proof but cannot publish any bytes.
    if not write:
        return {
            "status": "validated-no-write",
            "condition": target.condition,
            "repaired_tasks": list(REPAIRED_TASKS),
            "repair_records": repair_records,
            "original_logs": originals_before,
            "scorer": scorer_identity,
        }
    recovery = _require_regular_directory(recovery, label="score-only recovery root", create=True)
    published: dict[int, dict[str, Any]] = dict(originals_before)
    for task_index, derived in derived_logs.items():
        record = next(item for item in repair_records if item["task_index"] == task_index)
        name = _derive_artifact_name(original=originals_before[task_index], repair=record, scorer=scorer_identity)
        destination = recovery / "derived-logs" / f"task-{task_index:03d}" / name
        _write_immutable_eval_log(derived, destination)
        # Re-open/revalidate the published derived byte stream, never trusting the in-memory copy.
        try:
            from inspect_ai.log import read_eval_log
        except ImportError as exc:  # pragma: no cover
            raise RecoveryError("Inspect AI is required to re-open derived logs") from exc
        reread = read_eval_log(str(destination), header_only=False)
        if _log_semantic_snapshot(reread) != _log_semantic_snapshot(derived):
            raise RecoveryError(f"re-opened derived task-{task_index} EvalLog differs from the score-only payload written")
        reread_cell = LoadedCell(
            task_index=task_index,
            receipt=by_index[task_index].receipt,
            original_path=destination.resolve(),
            log=reread,
            spec=by_index[task_index].spec,
            source_prefix=_source_prefix_from_log(reread, spec=by_index[task_index].spec, cap=_task_sample_count(frozen, task_index), task_index=task_index),
            sample_ids=_sample_ids(reread, task_index=task_index),
        )
        _validate_sorted_sample_semantics(reread_cell)
        _validate_correct_switch_binding(reread_cell, clean=by_index[_expected_clean_index(specs, task_index)], switch_values=switch_values)
        published[task_index] = _identity(destination, label=f"derived task-{task_index} score-only EvalLog")
    # Original evidence must remain byte-identical after the only permitted writes.
    immutable_custody_after = {
        "phase_one": _identity(phase_one, label="r002 phase-one receipt after recovery"),
        "clean_gate": _validate_existing_clean_gate_read_only(
            frozen=frozen,
            paths=paths,
            launch_contract_sha256=launch_sha256,
            evaluation_receipt_sha256=evaluation_sha256,
        ),
        "launch_contract": _identity(paths.contract, label="r002 launch contract after recovery"),
        "evaluation_receipt": _identity(paths.evaluation_receipt, label="r002 evaluation receipt after recovery"),
        "task_receipts": {
            task_index: _identity(
                paths.receipts / f"task-{task_index:03d}.json",
                label=f"r002 task-{task_index} receipt after recovery",
            )
            for task_index in range(1, frozen.TASK_COUNT + 1)
        },
    }
    if immutable_custody_after != immutable_custody_before:
        raise RecoveryError("immutable original r002 custody changed during score-only recovery")
    for task_index, before in originals_before.items():
        after = _identity(Path(before["path"]), label=f"original task-{task_index} EvalLog after recovery")
        if after != before:
            raise RecoveryError(f"original task-{task_index} EvalLog changed during score-only recovery")
    for relative, before in frozen_sources.items():
        after = frozen._identity(Path(frozen.PROJECT_ROOT) / relative, label=f"frozen r002 source {relative} after recovery")
        if after != before:
            raise RecoveryError(f"frozen r002 source changed during score-only recovery: {relative}")
    # The typed before/after snapshots above intentionally include the paired
    # clean-gate receipt.  Keep the phase/gate subset for the final attestation.
    attested_original_custody = {
        label: {
            "phase_one": immutable_custody_before["phase_one"] if label == "before" else immutable_custody_after["phase_one"],
            "clean_gate": immutable_custody_before["clean_gate"] if label == "before" else immutable_custody_after["clean_gate"],
        }
        for label in ("before", "after")
    }
    if attested_original_custody["before"] != attested_original_custody["after"]:
        raise RecoveryError("r002 phase-one/clean-gate custody changed during score-only recovery")
    corrected_native_validation = _validate_corrected_native_selection(
        frozen=frozen,
        paths=paths,
        launch=launch,
        evaluation=evaluation,
        specs=specs,
        published=published,
        switch_values=switch_values,
    )
    report = _build_preflight(
        frozen=frozen,
        target=target,
        paths=paths,
        campaign=campaign,
        recovery_root=recovery,
        launch=launch,
        evaluation=evaluation,
        cells=cells,
        specs=specs,
        originals=originals_before,
        published=published,
        repair_records=repair_records,
        scorer_identity=scorer_identity,
        frozen_sources=frozen_sources,
        corrected_native_validation=corrected_native_validation,
        clean_gate_identity=immutable_custody_before["clean_gate"],
    )
    validate_corrected_preflight(report)
    preflight_path = recovery / "preflight" / "corrected-native-two-bias.json"
    _write_immutable_json(preflight_path, report, label="score-only corrected native preflight")
    persisted_preflight = _load_json(preflight_path, label="score-only corrected native preflight")
    if validate_corrected_preflight(persisted_preflight) != report:
        raise RecoveryError("persisted score-only corrected native preflight differs from the validated report")
    completion = {
        "schema": COMPLETION_SCHEMA,
        "condition": target.condition,
        "target": {"step": target.step, "run_name": target.run_name, "segment_index": target.segment_index},
        "preflight": _identity(preflight_path, label="score-only corrected native preflight"),
        "original_task_receipts": [
            immutable_custody_before["task_receipts"][index] for index in range(1, frozen.TASK_COUNT + 1)
        ],
        "original_task_logs": [originals_before[index] for index in range(1, frozen.TASK_COUNT + 1)],
        "published_task_logs": [published[index] for index in range(1, frozen.TASK_COUNT + 1)],
        "repaired_tasks": list(REPAIRED_TASKS),
    }
    validate_completion(completion)
    completion_path = recovery / "completion" / "completion.json"
    _write_immutable_json(completion_path, completion, label="score-only recovery completion")
    persisted_completion = _load_json(completion_path, label="score-only recovery completion")
    if validate_completion(persisted_completion) != completion:
        raise RecoveryError("persisted score-only recovery completion differs from the validated receipt")
    attestation = {
        "schema": ATTESTATION_SCHEMA,
        "completion": _identity(completion_path, label="score-only recovery completion"),
        "preflight": _identity(preflight_path, label="score-only corrected native preflight"),
        "original_custody": attested_original_custody,
        "mutations": {
            "original_eval_logs": 0,
            "original_task_receipts": 0,
            "original_launch_contract": 0,
            "original_phase_receipts": 0,
            "original_clean_gate_receipt": 0,
            "step64_artifacts": 0,
            "derived_eval_logs": len(REPAIRED_TASKS),
        },
        "execution": report["recovery_implementation"]["execution"],
        "repair_tasks": list(REPAIRED_TASKS),
    }
    validate_attestation(attestation)
    attestation_path = recovery / "attestation" / "attestation.json"
    _write_immutable_json(attestation_path, attestation, label="score-only recovery attestation")
    persisted_attestation = _load_json(attestation_path, label="score-only recovery attestation")
    if validate_attestation(persisted_attestation) != attestation:
        raise RecoveryError("persisted score-only recovery attestation differs from the validated receipt")
    return {
        "status": "written",
        "condition": target.condition,
        "preflight": _identity(preflight_path, label="score-only corrected native preflight"),
        "completion": _identity(completion_path, label="score-only recovery completion"),
        "attestation": _identity(attestation_path, label="score-only recovery attestation"),
        "repaired_tasks": list(REPAIRED_TASKS),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step16-root", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--frozen-launcher", required=True)
    parser.add_argument("--write", action="store_true", help="publish only new immutable recovery artifacts")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = recover(
            step16_root=args.step16_root,
            campaign_root=args.campaign_root,
            recovery_root=args.recovery_root,
            frozen_launcher=args.frozen_launcher,
            write=args.write,
        )
    except (RecoveryError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
