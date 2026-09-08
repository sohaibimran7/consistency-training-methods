"""Hash-bound posthoc GPT-5.6 Luna grading for Stage 2 OOD biased logs.

Raw Stage 2 generation is deliberately grader-free.  This module only accepts
the eighteen biased logs named and SHA-256-bound by
``stage2_ood_hle.raw_preflight``.  It appends the repository-local Luna scorer
to immutable copies, leaving both clean and raw generation logs untouched.

For transfer between hosts, stage each report-listed raw log at::

    <staged-raw-root>/<condition>/<population>/<bias_type>/<dataset>/<filename>.eval

The file bytes must still match the generation-host raw-preflight report.  The
resulting derived ``.eval`` files retain the original Inspect task arguments,
so ``experiments.stage2_ood_hle.analyze --run CONDITION=<derived-root>`` can
consume them directly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from experiments.stage2_ood_hle.raw_preflight import PREFLIGHT_SCHEMA, validate_preflight_report


INSPECT_RESCORE_MODEL = "mockllm/model"
PROVENANCE_SCHEMA = "stage2-ood-hle-luna-grade-v1"
# This is deliberately the same dated pin and cap used by the repository's
# existing Luna scorer.  The scorer enforces ``max_connections <= 500`` too;
# process sharding additionally ensures the aggregate stays within that cap.
DEFAULT_LUNA_GRADER_MODEL = "openrouter/openai/gpt-5.6-luna-20260709"
DEFAULT_MAX_CONNECTIONS = 500
DEFAULT_MAX_TOKENS = 256
DEFAULT_WORKERS = 5
DEFAULT_CONNECTIONS_PER_WORKER = 100
_SHA256_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class GradeInput:
    """One staged raw biased cell, immutably bound to its preflight report."""

    path: Path
    condition: str
    regime: str
    population: str
    dataset: str
    bias_type: str
    created: str
    expected_sha256: str
    preflight_report_sha256: str
    manifest_sha256: str
    # Source path/cell/hash already uniquely identify a grade input.  Keep the
    # nested provenance mapping out of dataclass hashing so inputs can key the
    # deterministic result map even though JSON objects are mutable/unhashable.
    paired_clean: Mapping[str, Any] = field(compare=False, hash=False)
    # The normal completed-matrix path is bound to one raw-preflight report.
    # Native-HF resume workers may additionally make a *single*, already
    # full-paired-validated biased cell available before the 21-cell matrix is
    # complete.  Keep that provenance distinct rather than pretending that a
    # per-cell handoff is a completed raw-preflight report.
    binding_kind: str = "raw_preflight"
    binding_sha256: str | None = None
    binding_path: str | None = None
    binding_schema: str | None = None


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _require_sha256(value: Any, *, field: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"Stage 2 Luna handoff has invalid {field}")
    return str(value)


def _safe_component(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Stage 2 Luna handoff has invalid {field}")
    return value


def _staged_path(root: Path, *, condition: str, source: Mapping[str, Any]) -> Path:
    population = _safe_component(source.get("population"), field="population")
    bias_type = _safe_component(source.get("bias_type"), field="bias_type")
    dataset = _safe_component(source.get("dataset"), field="dataset")
    raw_log = source.get("raw_log")
    if not isinstance(raw_log, str) or not raw_log or not Path(raw_log).is_absolute():
        raise ValueError("Stage 2 Luna handoff source has no absolute raw_log")
    filename = Path(raw_log).name
    if not filename or filename in {".", ".."} or not filename.endswith(".eval"):
        raise ValueError("Stage 2 Luna handoff source raw_log is not an .eval file")
    result = (root / condition / population / bias_type / dataset / filename).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:  # pragma: no cover - component checks above make this defensive
        raise ValueError("Stage 2 Luna staged source escapes the supplied raw root") from exc
    return result


def preflight_bound_logs(raw_root: str | Path, preflight_report: str | Path) -> list[GradeInput]:
    """Select exactly the eighteen hash-bound staged biased raw EvalLogs."""

    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Stage 2 Luna staged raw root does not exist: {root}")
    report_path = Path(preflight_report).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Stage 2 raw preflight report does not exist: {report_path}")
    report = validate_preflight_report(report_path)
    condition = _safe_component(report.get("condition"), field="condition")
    report_digest = _sha256(report_path)
    manifest_digest = _require_sha256(report.get("manifest_sha256"), field="manifest_sha256")
    sources = report.get("sources")
    assert isinstance(sources, list)  # guaranteed by validate_preflight_report
    selected: list[GradeInput] = []
    seen_paths: set[Path] = set()
    seen_cells: set[tuple[str, str, str, str]] = set()
    for source in sources:
        assert isinstance(source, Mapping)  # guaranteed by validate_preflight_report
        if source["kind"] != "biased":
            continue
        cell = (
            str(source["regime"]),
            str(source["population"]),
            str(source["dataset"]),
            str(source["bias_type"]),
        )
        if cell in seen_cells:
            raise ValueError(f"Stage 2 raw preflight report has duplicate biased cell: {cell!r}")
        seen_cells.add(cell)
        staged = _staged_path(root, condition=condition, source=source)
        if not staged.is_file():
            raise FileNotFoundError(f"missing hash-bound staged Stage 2 raw log for {cell}: {staged}")
        expected_sha = _require_sha256(source.get("raw_log_sha256"), field="raw_log_sha256")
        if _sha256(staged) != expected_sha:
            raise ValueError(f"staged Stage 2 raw log SHA-256 does not match its preflight report for {cell}")
        if staged in seen_paths:
            raise ValueError(f"Stage 2 raw preflight maps multiple cells to one staged log: {staged}")
        seen_paths.add(staged)
        paired_clean = source.get("paired_clean")
        assert isinstance(paired_clean, Mapping)  # guaranteed by validate_preflight_report
        selected.append(
            GradeInput(
                path=staged,
                condition=condition,
                regime=str(source["regime"]),
                population=str(source["population"]),
                dataset=str(source["dataset"]),
                bias_type=str(source["bias_type"]),
                created=str(source["created"]),
                expected_sha256=expected_sha,
                preflight_report_sha256=report_digest,
                manifest_sha256=manifest_digest,
                paired_clean=dict(paired_clean),
            )
        )
    if len(selected) != 18:
        raise ValueError(f"Stage 2 raw preflight report must stage exactly 18 biased logs, got {len(selected)}")
    return sorted(selected, key=lambda item: (item.condition, item.regime, item.population, item.dataset, item.bias_type))


def _score_mapping(sample: Any) -> tuple[str, Any]:
    matches: list[tuple[str, Any]] = []
    for name, score in _mapping(_attribute(sample, "scores", {})).items():
        value = _attribute(score, "value")
        if isinstance(value, Mapping) and "bias_acknowledged" in value:
            matches.append((str(name), score))
    if len(matches) != 1:
        raise ValueError(
            f"sample {_attribute(sample, 'id', '<unknown>')!r} must have exactly one Luna acknowledgement score"
        )
    return matches[0]


def _json_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _export_rows(log: Any, source: GradeInput) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in list(_attribute(log, "samples", []) or []):
        scorer_name, score = _score_mapping(sample)
        value = _mapping(_attribute(score, "value", {})).get("bias_acknowledged")
        metadata = dict(_mapping(_attribute(score, "metadata", {})))
        rows.append(
            {
                "condition": source.condition,
                "regime": source.regime,
                "population": source.population,
                "dataset": source.dataset,
                "bias_type": source.bias_type,
                "question_id": str(_attribute(sample, "id", "")),
                "scorer": scorer_name,
                "bias_acknowledged": _json_number(value),
                "grader_model": metadata.get("grader_model"),
                "grader_response": metadata.get("grader_response"),
                "grader_usage": metadata.get("grader_usage"),
                "grader_stop_reason": metadata.get("grader_stop_reason"),
                "grader_max_tokens_cap_hit": metadata.get("grader_max_tokens_cap_hit", False),
            }
        )
    return rows


def output_paths(output_root: str | Path, source: GradeInput, *, smoke: bool = False) -> tuple[Path, Path, Path]:
    suffix = "-luna-smoke" if smoke else "-luna"
    stem = source.path.name[: -len(".eval")] if source.path.name.endswith(".eval") else source.path.stem
    directory = (
        Path(output_root).resolve()
        / source.condition
        / source.population
        / source.bias_type
        / source.dataset
    )
    eval_path = directory / f"{stem}{suffix}.eval"
    return eval_path, eval_path.with_suffix(".jsonl"), eval_path.with_suffix(".provenance.json")


def _provenance(
    source: GradeInput,
    *,
    source_sha256: str,
    smoke_samples: int | None,
    worker_count: int,
    connections_per_worker: int,
    grader_max_tokens: int,
    shard_index: int,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "source_log": str(source.path),
        "source_sha256": source_sha256,
        "condition": source.condition,
        "regime": source.regime,
        "population": source.population,
        "dataset": source.dataset,
        "bias_type": source.bias_type,
        "grader_model": DEFAULT_LUNA_GRADER_MODEL,
        "worker_count": worker_count,
        "connections_per_worker": connections_per_worker,
        "aggregate_connection_limit": worker_count * connections_per_worker,
        "deterministic_shard_index": shard_index,
        "grader_max_tokens": grader_max_tokens,
        "inspect_rescore_model": INSPECT_RESCORE_MODEL,
        "smoke_samples": smoke_samples,
    }
    if source.binding_kind == "raw_preflight":
        if source.binding_sha256 is not None or source.binding_path is not None or source.binding_schema is not None:
            raise ValueError("raw-preflight Luna input has unexpected incremental binding metadata")
        document["raw_preflight"] = {
            "schema": PREFLIGHT_SCHEMA,
            "report_sha256": source.preflight_report_sha256,
            "source_sha256": source.expected_sha256,
            "manifest_sha256": source.manifest_sha256,
            "paired_clean": dict(source.paired_clean),
        }
    elif source.binding_kind == "incremental_handoff":
        receipt_sha256 = _require_sha256(source.binding_sha256, field="incremental_handoff.receipt_sha256")
        if not isinstance(source.binding_path, str) or not Path(source.binding_path).is_absolute():
            raise ValueError("incremental Luna handoff has no absolute receipt path")
        if not isinstance(source.binding_schema, str) or not source.binding_schema:
            raise ValueError("incremental Luna handoff has no receipt schema")
        document["incremental_handoff"] = {
            "schema": source.binding_schema,
            "receipt_path": source.binding_path,
            "receipt_sha256": receipt_sha256,
            "source_sha256": source.expected_sha256,
            "manifest_sha256": source.manifest_sha256,
            "paired_clean": dict(source.paired_clean),
        }
    else:
        raise ValueError(f"unsupported Stage 2 Luna source binding kind: {source.binding_kind!r}")
    return document


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing Stage 2 Luna artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _incremental_handoff_compatible_with_preflight(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Accept an already-graded receipt-bound cell in the final matrix pass.

    The native-HF resume owner stages its successful cells below a different
    raw-root before the full 21-cell report exists.  The final all-cell
    preflight stages byte-identical copies below the canonical root.  A strict
    byte-for-byte provenance comparison would therefore trigger a needless
    second paid grader call even though the scientific source is the same.

    This compatibility check intentionally permits *only* that binding-kind
    difference.  It rechecks the incremental receipt bytes and its original
    staged source, and requires every analysis-relevant identity, paired-clean
    SHA, grader pin, connection/token cap, deterministic shard, derived log,
    and exported rows to remain common.  It never rewrites the incremental
    provenance sidecar: the receipt remains the truthful record of how the
    existing derived EvalLog entered the canonical output tree.
    """

    expected_preflight = expected.get("raw_preflight")
    actual_handoff = actual.get("incremental_handoff")
    if not isinstance(expected_preflight, Mapping) or not isinstance(actual_handoff, Mapping):
        return False
    if "incremental_handoff" in expected or "raw_preflight" in actual:
        return False
    exact_fields = (
        "schema",
        "source_sha256",
        "condition",
        "regime",
        "population",
        "dataset",
        "bias_type",
        "grader_model",
        "worker_count",
        "connections_per_worker",
        "aggregate_connection_limit",
        "deterministic_shard_index",
        "grader_max_tokens",
        "inspect_rescore_model",
        "smoke_samples",
    )
    if any(actual.get(field) != expected.get(field) for field in exact_fields):
        return False
    if actual_handoff.get("source_sha256") != expected_preflight.get("source_sha256"):
        return False
    if actual_handoff.get("manifest_sha256") != expected_preflight.get("manifest_sha256"):
        return False
    actual_clean = actual_handoff.get("paired_clean")
    expected_clean = expected_preflight.get("paired_clean")
    if not isinstance(actual_clean, Mapping) or not isinstance(expected_clean, Mapping):
        return False
    if actual_clean.get("raw_log_sha256") != expected_clean.get("raw_log_sha256"):
        return False
    # The source path need not be equal: full staging has its own canonical
    # root.  Both paths must, however, still prove the same bytes.
    try:
        incremental_source = Path(str(actual.get("source_log"))).resolve()
        if not incremental_source.is_file() or _sha256(incremental_source) != actual.get("source_sha256"):
            return False
        receipt_path = Path(str(actual_handoff.get("receipt_path"))).resolve()
        receipt_sha256 = actual_handoff.get("receipt_sha256")
        if not _is_sha256(receipt_sha256) or not receipt_path.is_file() or _sha256(receipt_path) != receipt_sha256:
            return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def _incremental_derived_log_matches_policy(log: Any, expected: Mapping[str, Any]) -> bool:
    """Check grader metadata embedded in the receipt-bound derived EvalLog."""

    grader_model = expected.get("grader_model")
    grader_max_tokens = expected.get("grader_max_tokens")
    try:
        samples = list(_attribute(log, "samples", []) or [])
        if not samples:
            return False
        for sample in samples:
            _, score = _score_mapping(sample)
            metadata = _mapping(_attribute(score, "metadata", {}))
            if metadata.get("grader_model") != grader_model:
                return False
            if metadata.get("grader_max_tokens") != grader_max_tokens:
                return False
    except (TypeError, ValueError):
        return False
    return True


def _resume_complete(eval_path: Path, rows_path: Path, provenance_path: Path, expected: Mapping[str, Any]) -> bool:
    if not any(path.exists() for path in (eval_path, rows_path, provenance_path)):
        return False
    if not all(path.is_file() for path in (eval_path, rows_path, provenance_path)):
        raise FileExistsError(f"incomplete prior Stage 2 Luna output set beside {eval_path}; archive it before retrying")
    actual = json.loads(provenance_path.read_text(encoding="utf-8"))
    incremental_compatible = actual != expected and _incremental_handoff_compatible_with_preflight(actual, expected)
    if actual != expected and not incremental_compatible:
        raise FileExistsError(f"existing Stage 2 Luna output has different provenance: {provenance_path}")
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - configured evaluation environment only
        raise RuntimeError("Inspect AI is required to verify Stage 2 Luna output") from exc
    log = read_eval_log(str(eval_path))
    if _attribute(log, "status") != "success":
        raise ValueError(f"existing Stage 2 Luna output is not successful: {eval_path}")
    if incremental_compatible and not _incremental_derived_log_matches_policy(log, expected):
        raise ValueError(f"incremental Stage 2 Luna output has the wrong embedded grader policy: {eval_path}")
    raw_preflight = expected.get("raw_preflight")
    incremental_handoff = expected.get("incremental_handoff")
    if isinstance(raw_preflight, Mapping) and incremental_handoff is None:
        source = GradeInput(
            path=eval_path,
            condition=str(expected["condition"]),
            regime=str(expected["regime"]),
            population=str(expected["population"]),
            dataset=str(expected["dataset"]),
            bias_type=str(expected["bias_type"]),
            created="",
            expected_sha256=str(expected["source_sha256"]),
            preflight_report_sha256=str(raw_preflight["report_sha256"]),
            manifest_sha256=str(raw_preflight["manifest_sha256"]),
            paired_clean=dict(raw_preflight["paired_clean"]),
        )
    elif isinstance(incremental_handoff, Mapping) and raw_preflight is None:
        source = GradeInput(
            path=eval_path,
            condition=str(expected["condition"]),
            regime=str(expected["regime"]),
            population=str(expected["population"]),
            dataset=str(expected["dataset"]),
            bias_type=str(expected["bias_type"]),
            created="",
            expected_sha256=str(expected["source_sha256"]),
            # The field is unused for incremental output provenance, but
            # retaining a SHA here preserves one shared GradeInput shape.
            preflight_report_sha256=str(incremental_handoff["receipt_sha256"]),
            manifest_sha256=str(incremental_handoff["manifest_sha256"]),
            paired_clean=dict(incremental_handoff["paired_clean"]),
            binding_kind="incremental_handoff",
            binding_sha256=str(incremental_handoff["receipt_sha256"]),
            binding_path=str(incremental_handoff["receipt_path"]),
            binding_schema=str(incremental_handoff["schema"]),
        )
    else:
        raise ValueError(f"Stage 2 Luna provenance has an invalid source binding: {provenance_path}")
    rows = _export_rows(log, source)
    payload = b"".join((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode() for row in rows)
    if rows_path.read_bytes() != payload:
        raise ValueError(f"row export does not match its derived Stage 2 Luna EvalLog: {rows_path}")
    return True


def _validate_parallelism(worker_count: int, connections_per_worker: int) -> None:
    for value, name in ((worker_count, "workers"), (connections_per_worker, "connections_per_worker")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    aggregate = worker_count * connections_per_worker
    if aggregate > DEFAULT_MAX_CONNECTIONS:
        raise ValueError(f"workers * connections_per_worker must be <= {DEFAULT_MAX_CONNECTIONS}; got {aggregate}")


def _validate_grader_max_tokens(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("grader_max_tokens must be a positive integer")


def grade_one(
    source: GradeInput,
    output_root: str | Path,
    *,
    smoke_samples: int | None = None,
    worker_count: int = DEFAULT_WORKERS,
    connections_per_worker: int = DEFAULT_CONNECTIONS_PER_WORKER,
    grader_max_tokens: int = DEFAULT_MAX_TOKENS,
    shard_index: int = 0,
) -> str:
    """Append Luna to one hash-bound raw biased EvalLog, returning status."""

    if smoke_samples is not None and (isinstance(smoke_samples, bool) or not isinstance(smoke_samples, int) or smoke_samples < 1):
        raise ValueError("smoke_samples must be a positive integer")
    _validate_parallelism(worker_count, connections_per_worker)
    _validate_grader_max_tokens(grader_max_tokens)
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or not 0 <= shard_index < worker_count:
        raise ValueError("shard_index must be an integer in [0, worker_count)")
    source_digest = _sha256(source.path)
    if source_digest != _require_sha256(source.expected_sha256, field="raw_log_sha256"):
        raise ValueError(f"staged Stage 2 raw log changed after raw preflight: {source.path}")
    eval_path, rows_path, provenance_path = output_paths(output_root, source, smoke=smoke_samples is not None)
    provenance = _provenance(
        source,
        source_sha256=source_digest,
        smoke_samples=smoke_samples,
        worker_count=worker_count,
        connections_per_worker=connections_per_worker,
        grader_max_tokens=grader_max_tokens,
        shard_index=shard_index,
    )
    if _resume_complete(eval_path, rows_path, provenance_path, provenance):
        return "resumed"

    try:
        from inspect_ai import score
        from inspect_ai.log import read_eval_log, write_eval_log
        from ctm_data.adapters.mcq_bias.luna_scorer import luna_bias_acknowledged_scorer
    except ImportError as exc:  # pragma: no cover - configured grading environment only
        raise RuntimeError("Inspect AI and the repository Luna scorer are required for Stage 2 Luna grading") from exc
    raw = read_eval_log(str(source.path))
    if _attribute(raw, "status") != "success":
        raise ValueError(f"refusing to grade non-success Stage 2 raw log: {source.path}")
    if smoke_samples is not None:
        raw = copy.deepcopy(raw)
        samples = list(_attribute(raw, "samples", []) or [])[:smoke_samples]
        if not samples:
            raise ValueError(f"Stage 2 Luna smoke input has no samples: {source.path}")
        raw.samples = samples
        raw.results = None

    scorer_kwargs: dict[str, int] = {"max_connections": connections_per_worker}
    if grader_max_tokens != DEFAULT_MAX_TOKENS:
        scorer_kwargs["max_tokens"] = grader_max_tokens
    scored = score(
        raw,
        luna_bias_acknowledged_scorer(**scorer_kwargs),
        model=INSPECT_RESCORE_MODEL,
        action="append",
        display="none",
        copy=True,
    )
    rows = _export_rows(scored, source)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in (eval_path, rows_path, provenance_path)):
        raise FileExistsError(f"derived Stage 2 Luna output appeared while grading: {eval_path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{eval_path.name}.", suffix=".eval", dir=eval_path.parent)
    os.close(descriptor)
    temporary_eval = Path(temporary_name)
    try:
        write_eval_log(scored, str(temporary_eval))
        os.link(temporary_eval, eval_path)
    finally:
        try:
            os.unlink(temporary_eval)
        except FileNotFoundError:
            pass
    rows_payload = b"".join((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode() for row in rows)
    _write_new(rows_path, rows_payload)
    _write_new(provenance_path, (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode())
    return "graded"


def deterministic_shards(sources: Sequence[GradeInput], worker_count: int) -> list[tuple[GradeInput, ...]]:
    """Assign cells by stable identity while allowing new conditions later."""

    _validate_parallelism(worker_count, 1)
    ordered = sorted(sources, key=lambda item: (item.condition, item.regime, item.population, item.dataset, item.bias_type))
    shards: list[list[GradeInput]] = [[] for _ in range(worker_count)]
    for source in ordered:
        identity = "\0".join((source.condition, source.regime, source.population, source.dataset, source.bias_type)).encode()
        index = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % worker_count
        shards[index].append(source)
    return [tuple(shard) for shard in shards]


def _grade_shard(
    shard_index: int,
    sources: tuple[GradeInput, ...],
    output_root: Path,
    worker_count: int,
    connections_per_worker: int,
    grader_max_tokens: int,
    smoke_samples: int | None,
) -> list[tuple[GradeInput, str]]:
    return [
        (
            source,
            grade_one(
                source,
                output_root,
                smoke_samples=smoke_samples,
                worker_count=worker_count,
                connections_per_worker=connections_per_worker,
                grader_max_tokens=grader_max_tokens,
                shard_index=shard_index,
            ),
        )
        for source in sources
    ]


def grade_all(
    staged_raw_root: str | Path,
    output_root: str | Path,
    *,
    preflight_report: str | Path,
    smoke: bool = False,
    smoke_samples: int = 2,
    workers: int = DEFAULT_WORKERS,
    connections_per_worker: int = DEFAULT_CONNECTIONS_PER_WORKER,
    grader_max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[tuple[GradeInput, str]]:
    """Grade one report-bound condition, at <=500 connections per invocation."""

    return grade_many(
        staged_raw_root,
        output_root,
        preflight_reports=(preflight_report,),
        smoke=smoke,
        smoke_samples=smoke_samples,
        workers=workers,
        connections_per_worker=connections_per_worker,
        grader_max_tokens=grader_max_tokens,
    )


def grade_many(
    staged_raw_root: str | Path,
    output_root: str | Path,
    *,
    preflight_reports: Sequence[str | Path],
    smoke: bool = False,
    smoke_samples: int = 2,
    workers: int = DEFAULT_WORKERS,
    connections_per_worker: int = DEFAULT_CONNECTIONS_PER_WORKER,
    grader_max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[tuple[GradeInput, str]]:
    """Grade one or more condition reports under one global connection cap.

    Use this entrypoint for simultaneous completed conditions.  Unlike running
    several ``grade_all`` processes, its single worker pool guarantees that
    ``workers * connections_per_worker`` is the aggregate OpenRouter cap over
    every supplied condition.
    """

    raw_path = Path(staged_raw_root).resolve()
    output_path = Path(output_root).resolve()
    if output_path == raw_path or output_path.is_relative_to(raw_path) or raw_path.is_relative_to(output_path):
        raise ValueError("staged raw and derived Stage 2 Luna output roots must be separate, non-nested directories")
    _validate_parallelism(workers, connections_per_worker)
    _validate_grader_max_tokens(grader_max_tokens)
    if not preflight_reports:
        raise ValueError("at least one Stage 2 raw preflight report is required")
    selected: list[GradeInput] = []
    conditions: set[str] = set()
    for report in preflight_reports:
        bound = preflight_bound_logs(raw_path, report)
        condition = bound[0].condition
        if condition in conditions:
            raise ValueError(f"duplicate Stage 2 Luna preflight condition: {condition!r}")
        conditions.add(condition)
        selected.extend(bound)
    if smoke:
        selected = selected[:1]
    if not selected:
        return []
    shards = deterministic_shards(selected, workers)
    active = [(index, shard) for index, shard in enumerate(shards) if shard]
    by_source: dict[GradeInput, str] = {}
    # Spawn keeps OpenRouter clients and event-loop state process-local.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        futures = [
            executor.submit(
                _grade_shard,
                index,
                shard,
                output_path,
                workers,
                connections_per_worker,
                grader_max_tokens,
                smoke_samples if smoke else None,
            )
            for index, shard in active
        ]
        for future in futures:
            for source, status in future.result():
                by_source[source] = status
    ordered = [source for shard in shards for source in shard]
    ordered.sort(key=lambda item: (item.condition, item.regime, item.population, item.dataset, item.bias_type))
    return [(source, by_source[source]) for source in ordered]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--preflight-report",
        required=True,
        action="append",
        type=Path,
        help="repeat for multiple completed conditions; all share one global connection cap",
    )
    parser.add_argument("--smoke", action="store_true", help="grade only the first two samples of one hash-bound biased log")
    parser.add_argument("--smoke-samples", type=int, default=2)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--connections-per-worker", type=int, default=DEFAULT_CONNECTIONS_PER_WORKER)
    parser.add_argument("--grader-max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args(argv)
    try:
        results = grade_many(
            args.staged_raw_root,
            args.output_root,
            preflight_reports=args.preflight_report,
            smoke=args.smoke,
            smoke_samples=args.smoke_samples,
            workers=args.workers,
            connections_per_worker=args.connections_per_worker,
            grader_max_tokens=args.grader_max_tokens,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    for source, status in results:
        print(
            f"{status}: {source.condition}/{source.regime}/{source.population}/"
            f"{source.dataset}/{source.bias_type}: {source.path}"
        )


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
