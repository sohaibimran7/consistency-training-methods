"""Recover capped post-hoc Luna verdicts without changing Stage 2 v1 logs.

The first incremental Stage 2 Luna pass used the approved dated Luna model at
a 256-token output cap.  That is normally ample for a YES/NO verdict, but a
small number of samples reached that cap.  This module is deliberately *not*
a generic re-grader:

* it accepts only the same receipt-bound incremental raw cells as
  :mod:`incremental_grade_luna`;
* it requires a complete, byte-verified v1 derived EvalLog/JSONL/provenance
  trio for each source before considering a recovery;
* it selects only scores with explicit cap evidence (never ordinary parse
  failures), rescoring just those samples at 1,024 tokens; and
* it writes a separate, immutable v2 EvalLog/JSONL/provenance trio.  The v1
  EvalLog, raw log, staged log, and handoff receipt are never modified.

The v2 root is intentionally separate from v1.  Point the normal Stage 2
analysis at that root alone: it contains the same task headers and complete
sample-level switch/Luna scores, while avoiding an ambiguous same-cell v1/v2
retry selection.

No network request is made by ``--dry-run``.  A real invocation takes the
same host-wide ``.luna-grade.lock`` as incremental grading and always uses
five workers with 100 connections each, so the aggregate OpenRouter limit is
at most 500.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from experiments.stage2_ood_hle import grade_luna, incremental_grade_luna as incremental


RECOVERY_SCHEMA = "stage2-ood-hle-luna-cap-recovery-v2"
RECOVERY_GRADER_MODEL = "openrouter/openai/gpt-5.6-luna-20260709"
RECOVERY_GRADER_MAX_TOKENS = 1024
RECOVERY_WORKERS = 5
RECOVERY_CONNECTIONS_PER_WORKER = 100
RECOVERY_MAX_CONNECTIONS = 500
V1_GRADER_MAX_TOKENS = 256
_CAP_STOP_REASONS = frozenset({"max_tokens", "max_length", "length", "model_length"})
_SHA256_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class RecoveryTarget:
    """One v1 score that is eligible for the longer, one-time recovery."""

    question_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class RecoveryInput:
    """All read-only evidence needed to recover one receipt-bound cell."""

    source: grade_luna.GradeInput
    v1_root: Path
    v1_eval: Path
    v1_rows: Path
    v1_provenance: Path
    raw_log: Mapping[str, Any]
    staged_log: Mapping[str, Any]
    paired_clean: Mapping[str, Any]
    receipt: Mapping[str, Any]
    manifest: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    targets: tuple[RecoveryTarget, ...]

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(target.question_id for target in self.targets)


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """One source's non-network or recovery outcome."""

    source: grade_luna.GradeInput
    status: str
    target_count: int


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Stage 2 Luna cap recovery has invalid {label}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _require_sha256(value: Any, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"Stage 2 Luna cap recovery has invalid {label}")
    return str(value)


def _regular_file(path: Path, *, label: str, expected_sha256: str | None = None) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Stage 2 Luna cap recovery {label} is not a regular file: {path}")
    digest = _sha256(path)
    if expected_sha256 is not None and digest != _require_sha256(expected_sha256, label=f"{label}.sha256"):
        raise ValueError(f"Stage 2 Luna cap recovery {label} SHA-256 changed: {path}")
    return digest


def _absolute_record(value: Any, *, label: str) -> dict[str, str]:
    record = _mapping(value, label=label)
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value or not Path(path_value).is_absolute():
        raise ValueError(f"Stage 2 Luna cap recovery has no absolute {label}.path")
    return {"path": str(Path(path_value).resolve()), "sha256": _require_sha256(record.get("sha256"), label=f"{label}.sha256")}


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _regular_file(path, label=str(path))}


def _json_load(path: Path, *, label: str) -> Mapping[str, Any]:
    _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Stage 2 Luna cap recovery {label} is not valid JSON: {path}") from exc
    return _mapping(value, label=label)


def _source_shard_index(source: grade_luna.GradeInput) -> int:
    for index, shard in enumerate(grade_luna.deterministic_shards((source,), RECOVERY_WORKERS)):
        if source in shard:
            return index
    raise RuntimeError("could not assign a deterministic Stage 2 Luna shard")  # pragma: no cover - defensive


def _v1_expected_provenance(source: grade_luna.GradeInput) -> dict[str, Any]:
    if source.binding_kind != incremental.INCREMENTAL_PROVENANCE_BINDING:
        raise ValueError("Luna cap recovery accepts receipt-bound incremental v1 EvalLogs only")
    return grade_luna._provenance(
        source,
        source_sha256=_require_sha256(source.expected_sha256, label="source.raw_log_sha256"),
        smoke_samples=None,
        worker_count=RECOVERY_WORKERS,
        connections_per_worker=RECOVERY_CONNECTIONS_PER_WORKER,
        grader_max_tokens=V1_GRADER_MAX_TOKENS,
        shard_index=_source_shard_index(source),
    )


def _luna_score(sample: Any) -> tuple[str, Any, Mapping[str, Any], Mapping[str, Any]]:
    scorer_name, score = grade_luna._score_mapping(sample)
    value = _mapping(_attribute(score, "value", {}), label="Luna score value")
    metadata = _mapping(_attribute(score, "metadata", {}), label="Luna score metadata")
    return scorer_name, score, value, metadata


def _question_id(sample: Any) -> str:
    value = _attribute(sample, "id", "")
    if not isinstance(value, str) or not value:
        raise ValueError("Stage 2 Luna cap recovery encountered a sample without a non-empty question ID")
    return value


def _sample_map(log: Any, *, label: str) -> dict[str, Any]:
    samples = list(_attribute(log, "samples", []) or [])
    if not samples:
        raise ValueError(f"Stage 2 Luna cap recovery {label} has no samples")
    result: dict[str, Any] = {}
    for sample in samples:
        question_id = _question_id(sample)
        if question_id in result:
            raise ValueError(f"Stage 2 Luna cap recovery {label} has duplicate question ID {question_id!r}")
        result[question_id] = sample
    return result


def _raw_has_no_luna(log: Any) -> None:
    for sample in _sample_map(log, label="raw staged EvalLog").values():
        matches = []
        for score in _mapping(_attribute(sample, "scores", {}), label="raw sample scores").values():
            value = _attribute(score, "value")
            if isinstance(value, Mapping) and "bias_acknowledged" in value:
                matches.append(score)
        if matches:
            raise ValueError("Stage 2 Luna cap recovery source raw EvalLog unexpectedly already has a Luna score")


def _unparsed_luna_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return True
        if numeric in {0.0, 1.0}:
            return False
    raise ValueError("Stage 2 Luna cap recovery encountered an invalid v1 bias_acknowledged value")


def _cap_recovery_reason(value: Mapping[str, Any], metadata: Mapping[str, Any]) -> str | None:
    """Return a *proven* cap reason, never treating an ordinary parse miss as capped."""

    if metadata.get("grader_model") != RECOVERY_GRADER_MODEL:
        raise ValueError("Stage 2 Luna cap recovery v1 grader model differs from the pinned policy")
    if metadata.get("grader_max_tokens") != V1_GRADER_MAX_TOKENS:
        raise ValueError("Stage 2 Luna cap recovery v1 grader token cap differs from 256")
    cap_hit = metadata.get("grader_max_tokens_cap_hit")
    if not isinstance(cap_hit, bool):
        raise ValueError("Stage 2 Luna cap recovery v1 cap metadata must be bool")
    if cap_hit:
        return "grader_max_tokens_cap_hit"

    parsed_missing = _unparsed_luna_value(value.get("bias_acknowledged"))
    if not parsed_missing:
        return None
    stop_reason = metadata.get("grader_stop_reason")
    if isinstance(stop_reason, str) and stop_reason.strip().lower() in _CAP_STOP_REASONS:
        return "unparsed_with_cap_stop_reason"
    usage = metadata.get("grader_usage")
    if isinstance(usage, Mapping):
        output_tokens = usage.get("output_tokens")
        if (
            isinstance(output_tokens, (int, float))
            and not isinstance(output_tokens, bool)
            and math.isfinite(float(output_tokens))
            and float(output_tokens) >= V1_GRADER_MAX_TOKENS
        ):
            return "unparsed_with_usage_at_cap"
    return None


def _targets_from_v1(log: Any) -> tuple[RecoveryTarget, ...]:
    if _attribute(log, "status") != "success":
        raise ValueError("Stage 2 Luna cap recovery v1 EvalLog is not successful")
    targets: list[RecoveryTarget] = []
    for question_id, sample in _sample_map(log, label="v1 derived EvalLog").items():
        _, _, value, metadata = _luna_score(sample)
        reason = _cap_recovery_reason(value, metadata)
        if reason is not None:
            targets.append(RecoveryTarget(question_id=question_id, reason=reason))
    return tuple(targets)


def _receipt_evidence(source: grade_luna.GradeInput) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str], Mapping[str, Any]]:
    """Recover provenance fields after ``handoff_bound_logs`` revalidated them.

    The incremental selector already performs the strong validation (including
    raw/staged/clean file hashes, checkpoint binding, and exact protocol).  We
    repeat the receipt-byte check here because this v2 sidecar explicitly
    binds that receipt as an input.
    """

    if source.binding_kind != incremental.INCREMENTAL_PROVENANCE_BINDING:
        raise ValueError("Luna cap recovery requires an incremental handoff source")
    if not isinstance(source.binding_path, str) or not Path(source.binding_path).is_absolute():
        raise ValueError("Luna cap recovery source has no absolute handoff receipt")
    receipt_path = Path(source.binding_path).resolve()
    receipt_sha256 = _regular_file(receipt_path, label="incremental handoff receipt")
    if receipt_sha256 != _require_sha256(source.binding_sha256, label="incremental handoff receipt SHA-256"):
        raise ValueError("Luna cap recovery handoff receipt changed after receipt selection")
    receipt = _json_load(receipt_path, label="incremental handoff receipt")
    if receipt.get("schema") != source.binding_schema:
        raise ValueError("Luna cap recovery handoff receipt schema differs from its v1 provenance binding")
    raw = _absolute_record(receipt.get("raw_log"), label="receipt.raw_log")
    staged = _absolute_record(receipt.get("staged_log"), label="receipt.staged_log")
    clean = _absolute_record(receipt.get("paired_clean"), label="receipt.paired_clean")
    manifest = _absolute_record(receipt.get("manifest"), label="receipt.manifest")
    if staged["path"] != str(source.path.resolve()) or staged["sha256"] != source.expected_sha256:
        raise ValueError("Luna cap recovery receipt staged source differs from its v1 GradeInput")
    if raw["sha256"] != source.expected_sha256:
        raise ValueError("Luna cap recovery receipt raw/staged source hashes differ")
    if manifest["sha256"] != source.manifest_sha256:
        raise ValueError("Luna cap recovery receipt manifest differs from its v1 GradeInput")
    # ``handoff_bound_logs`` did this before returning the GradeInput, and we
    # deliberately do it again here so the v2 provenance itself is built only
    # from still-live raw/staged/clean/manifest bytes.
    _regular_file(Path(raw["path"]), label="receipt raw log", expected_sha256=raw["sha256"])
    _regular_file(Path(staged["path"]), label="receipt staged log", expected_sha256=staged["sha256"])
    _regular_file(Path(clean["path"]), label="receipt paired clean log", expected_sha256=clean["sha256"])
    _regular_file(Path(manifest["path"]), label="receipt frozen manifest", expected_sha256=manifest["sha256"])
    checkpoint = _mapping(receipt.get("checkpoint"), label="receipt.checkpoint")
    return raw, staged, clean, manifest, checkpoint


def _read_eval_log(path: Path) -> Any:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - configured evaluation environment only
        raise RuntimeError("Inspect AI is required to validate Stage 2 Luna cap recovery inputs") from exc
    return read_eval_log(str(path))


def _validate_v1(source: grade_luna.GradeInput, *, v1_root: Path) -> tuple[Path, Path, Path, Any]:
    v1_eval, v1_rows, v1_provenance = grade_luna.output_paths(v1_root, source)
    trio = (v1_eval, v1_rows, v1_provenance)
    if not any(path.exists() for path in trio):
        raise FileNotFoundError(f"no v1 derived output exists for receipt-bound source: {v1_eval}")
    if not all(path.is_file() and not path.is_symlink() for path in trio):
        raise FileExistsError(f"incomplete or non-regular v1 derived output set beside {v1_eval}")
    expected = _v1_expected_provenance(source)
    actual = _json_load(v1_provenance, label="v1 provenance")
    if dict(actual) != expected:
        raise ValueError(f"v1 provenance is not the exact pinned incremental 256-token provenance: {v1_provenance}")
    # Re-export and compare the rows as the normal v1 resume path does.  This
    # proves that the current v1 EvalLog and JSONL still agree before any v2
    # OpenRouter request is allowed.
    if not grade_luna._resume_complete(v1_eval, v1_rows, v1_provenance, expected):
        raise ValueError(f"v1 derived output did not validate as complete: {v1_eval}")
    v1_log = _read_eval_log(v1_eval)
    return v1_eval, v1_rows, v1_provenance, v1_log


def _prepare_input(source: grade_luna.GradeInput, *, v1_root: Path) -> RecoveryInput:
    raw, staged, clean, manifest, checkpoint = _receipt_evidence(source)
    v1_eval, v1_rows, v1_provenance, v1_log = _validate_v1(source, v1_root=v1_root)
    raw_log = _read_eval_log(source.path)
    if _attribute(raw_log, "status") != "success":
        raise ValueError("Stage 2 Luna cap recovery staged raw EvalLog is not successful")
    _raw_has_no_luna(raw_log)
    raw_ids = tuple(_sample_map(raw_log, label="staged raw EvalLog"))
    v1_ids = tuple(_sample_map(v1_log, label="v1 derived EvalLog"))
    if raw_ids != v1_ids:
        raise ValueError("Stage 2 Luna cap recovery raw and v1 sample IDs/order differ")
    targets = _targets_from_v1(v1_log)
    return RecoveryInput(
        source=source,
        v1_root=v1_root,
        v1_eval=v1_eval,
        v1_rows=v1_rows,
        v1_provenance=v1_provenance,
        raw_log=raw,
        staged_log=staged,
        paired_clean=clean,
        receipt={
            "path": str(Path(str(source.binding_path)).resolve()),
            "sha256": _require_sha256(source.binding_sha256, label="incremental handoff receipt SHA-256"),
            "schema": str(source.binding_schema),
        },
        manifest=manifest,
        checkpoint=dict(checkpoint),
        targets=targets,
    )


def _target_digest(targets: Sequence[RecoveryTarget]) -> str:
    payload = "".join(f"{target.question_id}\t{target.reason}\n" for target in targets).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _recovery_provenance_base(item: RecoveryInput) -> dict[str, Any]:
    """Create deterministic v2 provenance before any recovery is scored."""

    return {
        "schema": RECOVERY_SCHEMA,
        "condition": item.source.condition,
        "regime": item.source.regime,
        "population": item.source.population,
        "dataset": item.source.dataset,
        "bias_type": item.source.bias_type,
        "source": {
            "raw_log": dict(item.raw_log),
            "staged_log": dict(item.staged_log),
            "paired_clean": dict(item.paired_clean),
            "receipt": dict(item.receipt),
            "manifest": dict(item.manifest),
            "checkpoint": dict(item.checkpoint),
        },
        "v1": {
            "derived_eval_log": _file_record(item.v1_eval),
            "rows_jsonl": _file_record(item.v1_rows),
            "provenance": _file_record(item.v1_provenance),
            "provenance_schema": grade_luna.PROVENANCE_SCHEMA,
            "grader_model": RECOVERY_GRADER_MODEL,
            "grader_max_tokens": V1_GRADER_MAX_TOKENS,
        },
        "recovery": {
            "grader_model": RECOVERY_GRADER_MODEL,
            "grader_max_tokens": RECOVERY_GRADER_MAX_TOKENS,
            "worker_count": RECOVERY_WORKERS,
            "connections_per_worker": RECOVERY_CONNECTIONS_PER_WORKER,
            "aggregate_connection_limit": RECOVERY_MAX_CONNECTIONS,
            "inspect_rescore_model": grade_luna.INSPECT_RESCORE_MODEL,
            "selection_rule": (
                "grader_max_tokens_cap_hit=true, or an unparsed verdict with independently recorded 256-token cap evidence"
            ),
            "targets": [
                {"question_id": target.question_id, "reason": target.reason}
                for target in item.targets
            ],
            "target_count": len(item.targets),
            "target_digest_sha256": _target_digest(item.targets),
            "merge_policy": (
                "replace only the selected sample's Luna score; clear stale Inspect aggregate results"
                if item.targets
                else "copy the complete verified v1 EvalLog and JSONL byte-for-byte; no scorer call"
            ),
        },
    }


def output_paths(output_root: str | Path, source: grade_luna.GradeInput) -> tuple[Path, Path, Path]:
    """Return the immutable, separately rooted v2 artifact trio."""

    stem = source.path.name[: -len(".eval")] if source.path.name.endswith(".eval") else source.path.stem
    directory = (
        Path(output_root).resolve()
        / source.condition
        / source.population
        / source.bias_type
        / source.dataset
    )
    eval_path = directory / f"{stem}-luna-cap-recovery-v2.eval"
    return eval_path, eval_path.with_suffix(".jsonl"), eval_path.with_suffix(".provenance.json")


def _set_score(sample: Any, scorer_name: str, score: Any) -> None:
    scores = _attribute(sample, "scores", None)
    if not isinstance(scores, Mapping):
        raise ValueError("Stage 2 Luna cap recovery cannot replace a non-mapping sample score set")
    if scorer_name not in scores:
        raise ValueError("Stage 2 Luna cap recovery replacement scorer is absent from a v1 sample")
    if isinstance(scores, dict):
        scores[scorer_name] = score
        return
    updated = dict(scores)
    updated[scorer_name] = score
    try:
        setattr(sample, "scores", updated)
    except (AttributeError, TypeError) as exc:
        raise ValueError("Stage 2 Luna cap recovery could not replace the v1 Luna score") from exc


def _normalise(value: Any) -> Any:
    """Small stable projection used to prove unselected score equality."""

    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _normalise(model_dump())
    if hasattr(value, "__dict__"):
        return _normalise(vars(value))
    if isinstance(value, Path):
        return str(value)
    return value


def _score_signature(score: Any) -> str:
    return json.dumps(_normalise(score), sort_keys=True, allow_nan=True, default=str, separators=(",", ":"))


def _validate_recovered_score(sample: Any) -> tuple[str, Any]:
    scorer_name, score, _value, metadata = _luna_score(sample)
    if metadata.get("grader_model") != RECOVERY_GRADER_MODEL:
        raise ValueError("Stage 2 Luna cap recovery output has the wrong grader model")
    if metadata.get("grader_max_tokens") != RECOVERY_GRADER_MAX_TOKENS:
        raise ValueError("Stage 2 Luna cap recovery output has the wrong grader token cap")
    if not isinstance(metadata.get("grader_max_tokens_cap_hit"), bool):
        raise ValueError("Stage 2 Luna cap recovery output lacks boolean cap metadata")
    return scorer_name, score


def _merged_log(v1_log: Any, scored_subset: Any, item: RecoveryInput) -> Any:
    if _attribute(scored_subset, "status") != "success":
        raise ValueError("Stage 2 Luna cap recovery scorer returned a non-success EvalLog")
    target_ids = set(item.target_ids)
    scored = _sample_map(scored_subset, label="recovery scorer output")
    if set(scored) != target_ids:
        raise ValueError("Stage 2 Luna cap recovery scorer output does not contain exactly the selected samples")
    merged = copy.deepcopy(v1_log)
    v1_samples = _sample_map(merged, label="v1 derived EvalLog")
    if set(v1_samples) < target_ids:
        raise ValueError("Stage 2 Luna cap recovery targets are missing from the v1 EvalLog")
    for question_id in target_ids:
        old_name, _old_score, _old_value, _old_metadata = _luna_score(v1_samples[question_id])
        new_name, new_score = _validate_recovered_score(scored[question_id])
        if new_name != old_name:
            raise ValueError("Stage 2 Luna cap recovery scorer name differs from the v1 scorer name")
        _set_score(v1_samples[question_id], old_name, new_score)
    # Inspect aggregate score summaries in v1 would be stale after a
    # sample-level replacement.  Analysis intentionally consumes samples, so
    # clear them rather than present a misleading v1 aggregate as v2 data.
    try:
        merged.results = None
    except (AttributeError, TypeError) as exc:
        raise ValueError("Stage 2 Luna cap recovery could not clear stale Inspect aggregate results") from exc
    return merged


def _validate_merged_log(v1_log: Any, v2_log: Any, item: RecoveryInput) -> None:
    if _attribute(v2_log, "status") != "success":
        raise ValueError("Stage 2 Luna cap recovery v2 EvalLog is not successful")
    if item.targets and _attribute(v2_log, "results", None) is not None:
        raise ValueError("Stage 2 Luna cap recovery v2 EvalLog retained stale Inspect aggregate results")
    old = _sample_map(v1_log, label="v1 derived EvalLog")
    new = _sample_map(v2_log, label="v2 derived EvalLog")
    if tuple(old) != tuple(new):
        raise ValueError("Stage 2 Luna cap recovery v1/v2 sample IDs or order differ")
    targets = set(item.target_ids)
    for question_id, old_sample in old.items():
        new_sample = new[question_id]
        old_scores = _mapping(_attribute(old_sample, "scores", {}), label="v1 sample scores")
        new_scores = _mapping(_attribute(new_sample, "scores", {}), label="v2 sample scores")
        if set(old_scores) != set(new_scores):
            raise ValueError("Stage 2 Luna cap recovery changed the v1 score-name set")
        old_luna_name, old_luna, _old_value, _old_metadata = _luna_score(old_sample)
        new_luna_name, new_luna, _new_value, _new_metadata = _luna_score(new_sample)
        if old_luna_name != new_luna_name:
            raise ValueError("Stage 2 Luna cap recovery changed the Luna scorer name")
        for name in old_scores:
            if name != old_luna_name and _score_signature(old_scores[name]) != _score_signature(new_scores[name]):
                raise ValueError("Stage 2 Luna cap recovery changed a non-Luna v1 score")
        if question_id in targets:
            _validate_recovered_score(new_sample)
        elif _score_signature(old_luna) != _score_signature(new_luna):
            raise ValueError("Stage 2 Luna cap recovery changed an unselected v1 Luna score")
    if not targets and _score_signature(_attribute(v1_log, "results", None)) != _score_signature(_attribute(v2_log, "results", None)):
        raise ValueError("Stage 2 Luna cap recovery changed v1 Inspect aggregate results without a recovered sample")


def _rows_payload(log: Any, source: grade_luna.GradeInput) -> bytes:
    rows = grade_luna._export_rows(log, source)
    return b"".join((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") for row in rows)


def _write_eval_new(log: Any, destination: Path) -> None:
    try:
        from inspect_ai.log import write_eval_log
    except ImportError as exc:  # pragma: no cover - configured evaluation environment only
        raise RuntimeError("Inspect AI is required to write Stage 2 Luna cap recovery EvalLogs") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Stage 2 Luna cap recovery output: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".eval", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_eval_log(log, str(temporary))
        os.link(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _copy_new(source: Path, destination: Path, *, expected_sha256: str) -> None:
    """Atomically copy an immutable v1 artifact into the separate v2 root."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Stage 2 Luna cap recovery output: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if _sha256(temporary) != expected_sha256:
            raise ValueError(f"Stage 2 Luna cap recovery v1 copy has the wrong SHA-256: {source}")
        os.link(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _validate_v2_complete(item: RecoveryInput, output_root: Path) -> bool:
    v2_eval, v2_rows, v2_provenance = output_paths(output_root, item.source)
    paths = (v2_eval, v2_rows, v2_provenance)
    if not any(path.exists() for path in paths):
        return False
    if not all(path.is_file() and not path.is_symlink() for path in paths):
        raise FileExistsError(f"incomplete or non-regular v2 Luna cap recovery output set beside {v2_eval}")
    document = _json_load(v2_provenance, label="v2 provenance")
    expected_base = _recovery_provenance_base(item)
    actual_base = {key: value for key, value in document.items() if key != "outputs"}
    if actual_base != expected_base:
        raise FileExistsError(f"existing v2 Luna cap recovery output has different provenance: {v2_provenance}")
    outputs = _mapping(document.get("outputs"), label="v2 provenance.outputs")
    expected_outputs = {
        "derived_eval_log": {"path": str(v2_eval), "sha256": _regular_file(v2_eval, label="v2 EvalLog")},
        "rows_jsonl": {"path": str(v2_rows), "sha256": _regular_file(v2_rows, label="v2 JSONL")},
    }
    if outputs != expected_outputs:
        raise ValueError("v2 Luna cap recovery output hashes do not match its provenance")
    v1_log = _read_eval_log(item.v1_eval)
    v2_log = _read_eval_log(v2_eval)
    _validate_merged_log(v1_log, v2_log, item)
    if not item.targets:
        if v2_eval.read_bytes() != item.v1_eval.read_bytes() or v2_rows.read_bytes() != item.v1_rows.read_bytes():
            raise ValueError("v2 no-cap cell is not a byte-identical v1 copy")
    if v2_rows.read_bytes() != _rows_payload(v2_log, item.source):
        raise ValueError("v2 Luna cap recovery JSONL does not match its EvalLog")
    return True


def _validate_output_root(staged_root: Path, v1_root: Path, output_root: Path) -> None:
    for left, right, label in (
        (output_root, staged_root, "staged raw"),
        (output_root, v1_root, "v1 derived"),
    ):
        if left == right or left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError(f"v2 Luna cap recovery output root must be separate and non-nested from {label} root")


@contextmanager
def _global_grade_lock(staged_root: Path):
    """Use the exact host-wide lock file held by incremental v1 grading.

    Keep the small implementation here rather than depending on a private
    helper from ``incremental_grade_luna``: older live runners predate that
    helper, but they still use this same documented lock location.
    """

    lock_path = staged_root.parent / ".luna-grade.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _score_selected_subset(item: RecoveryInput) -> Any:
    try:
        from inspect_ai import score
        from ctm_data.adapters.mcq_bias.luna_scorer import luna_bias_acknowledged_scorer
    except ImportError as exc:  # pragma: no cover - configured grading environment only
        raise RuntimeError("Inspect AI and the repository Luna scorer are required for Stage 2 Luna cap recovery") from exc
    raw = _read_eval_log(item.source.path)
    _raw_has_no_luna(raw)
    subset = copy.deepcopy(raw)
    subset_by_id = _sample_map(subset, label="copied staged raw EvalLog")
    subset.samples = [subset_by_id[question_id] for question_id in item.target_ids]
    subset.results = None
    return score(
        subset,
        luna_bias_acknowledged_scorer(
            grader_model=RECOVERY_GRADER_MODEL,
            max_connections=RECOVERY_CONNECTIONS_PER_WORKER,
            max_tokens=RECOVERY_GRADER_MAX_TOKENS,
        ),
        model=grade_luna.INSPECT_RESCORE_MODEL,
        action="append",
        display="none",
        copy=True,
    )


def recover_one(item: RecoveryInput, output_root: str | Path) -> str:
    """Recover one cell, or resume only a fully validated immutable v2 trio."""

    root = Path(output_root).resolve()
    if _validate_v2_complete(item, root):
        return "resumed"
    scored_subset = _score_selected_subset(item) if item.targets else None
    # Inputs are immutable by contract, but rebind every byte after the paid
    # work and before publishing v2.  This makes a concurrent mutation fail
    # closed rather than yield a v2 sidecar with stale input hashes.
    rebound = _prepare_input(item.source, v1_root=item.v1_root)
    if _recovery_provenance_base(rebound) != _recovery_provenance_base(item):
        raise ValueError("Stage 2 Luna cap recovery inputs changed while rescoring; refusing to publish v2")
    v1_log = _read_eval_log(rebound.v1_eval)
    merged = _merged_log(v1_log, scored_subset, rebound) if scored_subset is not None else None
    if merged is not None:
        _validate_merged_log(v1_log, merged, rebound)
    v2_eval, v2_rows, v2_provenance = output_paths(root, rebound.source)
    if any(path.exists() for path in (v2_eval, v2_rows, v2_provenance)):
        raise FileExistsError(f"v2 Luna cap recovery output appeared while grading: {v2_eval}")
    if merged is None:
        _copy_new(rebound.v1_eval, v2_eval, expected_sha256=_regular_file(rebound.v1_eval, label="v1 EvalLog"))
        _copy_new(rebound.v1_rows, v2_rows, expected_sha256=_regular_file(rebound.v1_rows, label="v1 JSONL"))
        status = "copied-v1"
    else:
        _write_eval_new(merged, v2_eval)
        rows_payload = _rows_payload(merged, rebound.source)
        grade_luna._write_new(v2_rows, rows_payload)
        status = "recovered"
    provenance = _recovery_provenance_base(rebound)
    provenance["outputs"] = {
        "derived_eval_log": {"path": str(v2_eval), "sha256": _regular_file(v2_eval, label="v2 EvalLog")},
        "rows_jsonl": {"path": str(v2_rows), "sha256": _regular_file(v2_rows, label="v2 JSONL")},
    }
    grade_luna._write_new(v2_provenance, (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    if not _validate_v2_complete(rebound, root):  # pragma: no cover - all outputs were just written
        raise RuntimeError("v2 Luna cap recovery output disappeared after write")
    return status


def _prepare_many(
    staged_raw_root: Path,
    *,
    condition: str,
    manifest: str | Path,
    checkpoint: str | Path,
    receipt_root: str | Path | None,
    receipts: Sequence[str | Path],
    v1_root: Path,
    output_root: Path,
) -> tuple[list[RecoveryInput], list[RecoveryResult]]:
    sources = incremental.handoff_bound_logs(
        staged_raw_root,
        condition=condition,
        manifest=manifest,
        checkpoint=checkpoint,
        receipt_root=receipt_root,
        receipts=receipts,
    )
    prepared: list[RecoveryInput] = []
    outcomes: list[RecoveryResult] = []
    for source in sources:
        v1_paths = grade_luna.output_paths(v1_root, source)
        if not any(path.exists() for path in v1_paths):
            # A raw cell may be newer than the last incremental v1 grading
            # batch.  It is intentionally not eligible for recovery; this
            # module never turns an ungraded raw cell into a fresh grade.
            outcomes.append(RecoveryResult(source, "not-graded-v1", 0))
            continue
        item = _prepare_input(source, v1_root=v1_root)
        if _validate_v2_complete(item, output_root):
            outcomes.append(RecoveryResult(source, "resumed", len(item.targets)))
        else:
            prepared.append(item)
    return prepared, outcomes


def _recover_shard(items: tuple[RecoveryInput, ...], output_root: Path) -> list[RecoveryResult]:
    return [RecoveryResult(item.source, recover_one(item, output_root), len(item.targets)) for item in items]


def recover_incremental(
    staged_raw_root: str | Path,
    v1_root: str | Path,
    output_root: str | Path,
    *,
    condition: str,
    manifest: str | Path,
    checkpoint: str | Path,
    receipt_root: str | Path | None = None,
    receipts: Sequence[str | Path] = (),
    dry_run: bool = False,
) -> list[RecoveryResult]:
    """Recover all currently v1-graded cap hits for a receipt-bound condition.

    Receipt-bound sources without a complete v1 output are reported as
    ``not-graded-v1`` and skipped.  That is intentional: normal incremental
    grading owns those cells, while this recovery owner may only replace
    explicitly capped v1 sample scores.
    """

    staged = Path(staged_raw_root).resolve()
    v1 = Path(v1_root).resolve()
    output = Path(output_root).resolve()
    if not staged.is_dir():
        raise FileNotFoundError(f"Stage 2 Luna cap recovery staged root does not exist: {staged}")
    if not v1.is_dir():
        raise FileNotFoundError(f"Stage 2 Luna cap recovery v1 root does not exist: {v1}")
    _validate_output_root(staged, v1, output)
    if (RECOVERY_WORKERS * RECOVERY_CONNECTIONS_PER_WORKER) != RECOVERY_MAX_CONNECTIONS:
        raise RuntimeError("Stage 2 Luna cap recovery parallelism no longer equals the approved 500-connection ceiling")
    grade_luna._validate_parallelism(RECOVERY_WORKERS, RECOVERY_CONNECTIONS_PER_WORKER)

    prepared, outcomes = _prepare_many(
        staged,
        condition=condition,
        manifest=manifest,
        checkpoint=checkpoint,
        receipt_root=receipt_root,
        receipts=receipts,
        v1_root=v1,
        output_root=output,
    )
    if dry_run:
        return sorted(
            [
                *outcomes,
                *(
                    RecoveryResult(
                        item.source,
                        "ready" if item.targets else "ready-copy-v1",
                        len(item.targets),
                    )
                    for item in prepared
                ),
            ],
            key=_result_key,
        )

    # Share the ordinary incremental grading lock: a v1 incremental owner and
    # this recovery owner can never jointly exceed the aggregate 500 limit.
    with _global_grade_lock(staged):
        # Cells may have finished v1 grading while this process waited.  Bind
        # the receipt/raw/staged/v1 evidence once more immediately before any
        # process can create an OpenRouter client.
        prepared, outcomes = _prepare_many(
            staged,
            condition=condition,
            manifest=manifest,
            checkpoint=checkpoint,
            receipt_root=receipt_root,
            receipts=receipts,
            v1_root=v1,
            output_root=output,
        )
        if not prepared:
            return sorted(outcomes, key=_result_key)
        source_shards = grade_luna.deterministic_shards([item.source for item in prepared], RECOVERY_WORKERS)
        by_source = {item.source: item for item in prepared}
        shards = [tuple(by_source[source] for source in source_shard) for source_shard in source_shards if source_shard]
        recovered: list[RecoveryResult] = []
        with ProcessPoolExecutor(max_workers=RECOVERY_WORKERS, mp_context=get_context("spawn")) as executor:
            futures = [executor.submit(_recover_shard, shard, output) for shard in shards]
            for future in futures:
                recovered.extend(future.result())
    return sorted([*outcomes, *recovered], key=_result_key)


def _result_key(result: RecoveryResult) -> tuple[str, str, str, str, str]:
    source = result.source
    return source.condition, source.regime, source.population, source.dataset, source.bias_type


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-raw-root", required=True, type=Path)
    parser.add_argument("--derived-v1-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--receipt-root", type=Path)
    source.add_argument("--receipt", action="append", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate all inputs and print eligibility without calling Luna")
    args = parser.parse_args(argv)
    try:
        results = recover_incremental(
            args.staged_raw_root,
            args.derived_v1_root,
            args.output_root,
            condition=args.condition,
            manifest=args.manifest,
            checkpoint=args.checkpoint,
            receipt_root=args.receipt_root,
            receipts=tuple(args.receipt or ()),
            dry_run=args.dry_run,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    for result in results:
        source_input = result.source
        print(
            f"{result.status} targets={result.target_count}: {source_input.condition}/{source_input.regime}/"
            f"{source_input.population}/{source_input.dataset}/{source_input.bias_type}: {source_input.path}"
        )


__all__ = [
    "RECOVERY_CONNECTIONS_PER_WORKER",
    "RECOVERY_GRADER_MAX_TOKENS",
    "RECOVERY_GRADER_MODEL",
    "RECOVERY_MAX_CONNECTIONS",
    "RECOVERY_SCHEMA",
    "RECOVERY_WORKERS",
    "RecoveryInput",
    "RecoveryResult",
    "RecoveryTarget",
    "V1_GRADER_MAX_TOKENS",
    "output_paths",
    "recover_incremental",
    "recover_one",
]


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
