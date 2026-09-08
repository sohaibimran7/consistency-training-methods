"""Posthoc GPT-5.6 Luna grading for completed Stage 1 IID logs.

Only successful biased tasks are selected.  Inspect performs the rescore with
``mockllm/model`` as the inert primary model; the scorer itself calls the dated
Luna pin.  Source logs are read-only and every derived artifact is written
under a separate condition/split directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

SPLITS = ("train_eval", "heldout_in_domain")
DATASETS = ("logiqa", "hellaswag")
INSPECT_RESCORE_MODEL = "mockllm/model"
PROVENANCE_SCHEMA = "stage1-iid-luna-grade-v2"
# Mirrored from ``ctm_data.adapters.mcq_bias.luna_config`` so discovery and
# offline analysis stay importable before the optional mcq-bias/Inspect stack
# is installed. Tests assert the scorer and pipeline pins stay identical.
DEFAULT_LUNA_GRADER_MODEL = "openrouter/openai/gpt-5.6-luna-20260709"
DEFAULT_MAX_CONNECTIONS = 500
DEFAULT_MAX_TOKENS = 256
DEFAULT_WORKERS = 5
DEFAULT_CONNECTIONS_PER_WORKER = 100
RAW_PREFLIGHT_SCHEMA = "stage1-iid-raw-preflight-v1"
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class GradeInput:
    path: Path
    condition: str
    split: str
    dataset: str
    created: str
    expected_sha256: str | None = None
    preflight_report_sha256: str | None = None


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _task_basename(value: Any) -> str:
    return str(value or "").rsplit("@", 1)[-1].rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def _local_log_path(value: Any) -> Path:
    """Normalize Inspect local log names across plain-path and file-URI releases."""

    raw = str(value if isinstance(value, (str, os.PathLike)) else _attribute(value, "name", value))
    parsed = urlsplit(raw)
    if parsed.scheme:
        if parsed.scheme.lower() != "file":
            raise ValueError(f"Stage 1 Luna grading requires local logs, got URI scheme {parsed.scheme!r}: {raw}")
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"remote file URI authorities are not supported: {raw}")
        if parsed.query or parsed.fragment:
            raise ValueError(f"file log URI must not contain a query or fragment: {raw}")
        decoded = unquote(parsed.path)
        if not decoded:
            raise ValueError(f"file log URI has no path: {raw}")
        path = Path(decoded)
        if not path.is_absolute():
            raise ValueError(f"file log URI must contain an absolute path: {raw}")
        return path.resolve()
    return Path(raw).resolve()


def _layout(path: Path, raw_root: Path) -> tuple[str, str]:
    try:
        relative = path.resolve().relative_to(raw_root.resolve())
    except ValueError as exc:
        raise ValueError(f"log is outside raw root: {path}") from exc
    parts = relative.parts
    if len(parts) != 3 or parts[1] not in SPLITS:
        raise ValueError(
            f"expected raw log layout <condition>/<split>/<file>.eval under {raw_root}, got {relative}"
        )
    condition, split = parts[:2]
    if not condition or condition in SPLITS:
        raise ValueError(f"invalid condition directory in {relative}")
    return condition, split


def discover_biased_logs(raw_root: str | Path) -> list[GradeInput]:
    """Select the latest successful biased retry for every condition/split/dataset."""

    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw log root does not exist: {root}")
    try:
        from inspect_ai.log import list_eval_logs, read_eval_log
    except ImportError as exc:  # pragma: no cover - configured evaluation environment only
        raise RuntimeError("Inspect AI is required for Luna grading") from exc

    candidates: dict[tuple[str, str, str], GradeInput] = {}
    infos = list_eval_logs(str(root), formats=["eval"], recursive=True)
    for info in infos:
        path = _local_log_path(info)
        condition, directory_split = _layout(path, root)
        try:
            log = read_eval_log(str(path), header_only=True)
        except Exception:
            continue
        evaluation = _attribute(log, "eval")
        args = _mapping(_attribute(evaluation, "task_args", {}))
        split = str(args.get("split", ""))
        dataset = str(args.get("dataset", args.get("source_dataset", "")))
        bias_type = args.get("bias_type")
        if (
            _attribute(log, "status") != "success"
            or _task_basename(_attribute(evaluation, "task")) != "stage1_iid_biased"
            or not bias_type
        ):
            continue
        if split != directory_split:
            raise ValueError(f"split header/directory mismatch for {path}: {split!r} != {directory_split!r}")
        if dataset not in DATASETS:
            raise ValueError(f"unexpected Stage 1 IID dataset in {path}: {dataset!r}")
        created = str(_attribute(evaluation, "created", ""))
        item = GradeInput(path, condition, split, dataset, created)
        key = (condition, split, dataset)
        previous = candidates.get(key)
        if previous is not None and previous.created == created:
            raise ValueError(f"ambiguous successful retries for {key}: {previous.path} and {path}")
        if previous is None or created > previous.created:
            candidates[key] = item
    return sorted(candidates.values(), key=lambda item: (item.condition, item.split, item.dataset))


def _preflight_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
        raise ValueError(f"raw preflight report has invalid {field}")
    return value


def preflight_bound_logs(raw_root: str | Path, preflight_report: str | Path) -> list[GradeInput]:
    """Select only hash-bound logs staged in the canonical Luna layout.

    This deliberately does not call Inspect discovery: a raw-preflight v1
    report names the four eligible source cells, and each locally staged copy
    must still match the recorded content hash immediately before scoring.
    """

    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw log root does not exist: {root}")
    report_path = Path(preflight_report).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"raw preflight report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid raw preflight JSON report: {report_path}") from exc
    if not isinstance(report, Mapping) or report.get("schema") != RAW_PREFLIGHT_SCHEMA:
        raise ValueError(f"unsupported raw preflight report schema: {report_path}")

    condition = report.get("condition")
    if (
        not isinstance(condition, str)
        or not condition
        or Path(condition).name != condition
        or condition in {".", ".."}
    ):
        raise ValueError("raw preflight report has an invalid condition")
    raw_root_value = report.get("raw_root")
    if not isinstance(raw_root_value, str) or not Path(raw_root_value).is_absolute():
        raise ValueError("raw preflight report has no absolute raw_root")
    recorded_root = Path(raw_root_value).resolve()
    source_rows = report.get("sources")
    if not isinstance(source_rows, list):
        raise ValueError("raw preflight report sources must be a list")

    expected_cells = {(split, dataset) for split in SPLITS for dataset in DATASETS}
    selected: list[GradeInput] = []
    seen_cells: set[tuple[str, str]] = set()
    seen_paths: set[Path] = set()
    report_digest = _sha256(report_path)
    for source in source_rows:
        if not isinstance(source, Mapping):
            raise ValueError("raw preflight report source must be an object")
        split = source.get("split")
        dataset = source.get("dataset")
        if split not in SPLITS or dataset not in DATASETS:
            raise ValueError("raw preflight report source has an invalid split or dataset")
        cell = (split, dataset)
        if cell in seen_cells:
            raise ValueError(f"raw preflight report has duplicate source cell: {cell}")
        seen_cells.add(cell)
        raw_log = source.get("raw_log")
        if not isinstance(raw_log, str) or not Path(raw_log).is_absolute():
            raise ValueError(f"raw preflight report source has no absolute raw_log for {cell}")
        recorded_log = Path(raw_log).resolve()
        try:
            recorded_log.relative_to(recorded_root)
        except ValueError as exc:
            raise ValueError(f"raw preflight report raw_log is outside raw_root for {cell}") from exc
        if recorded_log.suffix != ".eval":
            raise ValueError(f"raw preflight report source is not an .eval log for {cell}")
        expected_sha256 = _preflight_sha256(source.get("raw_log_sha256"), field="raw_log_sha256")
        created = source.get("created")
        if not isinstance(created, str):
            raise ValueError(f"raw preflight report source has invalid created value for {cell}")

        # The copied staging layout is intentionally independent of the
        # generation host's absolute path, but preserves the report's filename.
        staged_path = (root / condition / split / recorded_log.name).resolve()
        try:
            staged_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"staged raw log escapes raw root for {cell}") from exc
        if not staged_path.is_file():
            raise FileNotFoundError(f"missing preflight-bound staged raw log for {cell}: {staged_path}")
        if _sha256(staged_path) != expected_sha256:
            raise ValueError(f"staged raw log SHA-256 does not match raw preflight report for {cell}")
        if staged_path in seen_paths:
            raise ValueError(f"raw preflight report maps multiple cells to one staged log: {staged_path}")
        seen_paths.add(staged_path)
        selected.append(
            GradeInput(
                staged_path,
                condition,
                split,
                dataset,
                created,
                expected_sha256=expected_sha256,
                preflight_report_sha256=report_digest,
            )
        )
    if seen_cells != expected_cells:
        missing = sorted(expected_cells - seen_cells)
        unexpected = sorted(seen_cells - expected_cells)
        raise ValueError(f"raw preflight report sources are incomplete or unexpected; missing={missing}, unexpected={unexpected}")
    return sorted(selected, key=lambda item: (item.condition, item.split, item.dataset))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
                "split": source.split,
                "dataset": source.dataset,
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
    directory = Path(output_root).resolve() / source.condition / source.split
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
    provenance: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "source_log": str(source.path),
        "source_sha256": source_sha256,
        "condition": source.condition,
        "split": source.split,
        "dataset": source.dataset,
        "grader_model": DEFAULT_LUNA_GRADER_MODEL,
        "worker_count": worker_count,
        "connections_per_worker": connections_per_worker,
        "aggregate_connection_limit": worker_count * connections_per_worker,
        "deterministic_shard_index": shard_index,
        "grader_max_tokens": grader_max_tokens,
        "inspect_rescore_model": INSPECT_RESCORE_MODEL,
        "smoke_samples": smoke_samples,
    }
    if source.expected_sha256 is not None or source.preflight_report_sha256 is not None:
        if source.expected_sha256 is None or source.preflight_report_sha256 is None:
            raise ValueError("preflight-bound GradeInput must include both source and report SHA-256 values")
        provenance["raw_preflight"] = {
            "schema": RAW_PREFLIGHT_SCHEMA,
            "report_sha256": _preflight_sha256(
                source.preflight_report_sha256, field="preflight_report_sha256"
            ),
            "source_sha256": _preflight_sha256(source.expected_sha256, field="raw_log_sha256"),
        }
    return provenance


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing derived artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


def _resume_complete(eval_path: Path, rows_path: Path, provenance_path: Path, expected: Mapping[str, Any]) -> bool:
    if not any(path.exists() for path in (eval_path, rows_path, provenance_path)):
        return False
    if not all(path.is_file() for path in (eval_path, rows_path, provenance_path)):
        raise FileExistsError(f"incomplete prior Luna output set beside {eval_path}; archive it before retrying")
    actual = json.loads(provenance_path.read_text())
    if actual != expected:
        raise FileExistsError(f"existing Luna output has different provenance: {provenance_path}")
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(eval_path))
    if _attribute(log, "status") != "success":
        raise ValueError(f"existing Luna output is not successful: {eval_path}")
    rows = _export_rows(
        log,
        GradeInput(eval_path, str(expected["condition"]), str(expected["split"]), str(expected["dataset"]), ""),
    )
    expected_rows = b"".join(
        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode() for row in rows
    )
    if rows_path.read_bytes() != expected_rows:
        raise ValueError(f"row export does not match its derived EvalLog: {rows_path}")
    return True


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
    """Grade one log, returning ``graded`` or ``resumed``."""

    if smoke_samples is not None and (isinstance(smoke_samples, bool) or smoke_samples < 1):
        raise ValueError("smoke_samples must be a positive integer")
    _validate_parallelism(worker_count, connections_per_worker)
    _validate_grader_max_tokens(grader_max_tokens)
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or not 0 <= shard_index < worker_count:
        raise ValueError("shard_index must be an integer in [0, worker_count)")
    eval_path, rows_path, provenance_path = output_paths(output_root, source, smoke=smoke_samples is not None)
    source_digest = _sha256(source.path)
    if source.expected_sha256 is not None:
        expected_sha256 = _preflight_sha256(source.expected_sha256, field="raw_log_sha256")
        if source_digest != expected_sha256:
            raise ValueError(
                f"staged raw log SHA-256 no longer matches raw preflight report: {source.path}"
            )
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

    from inspect_ai import score
    from inspect_ai.log import read_eval_log, write_eval_log
    from ctm_data.adapters.mcq_bias.luna_scorer import luna_bias_acknowledged_scorer

    raw = read_eval_log(str(source.path))
    if _attribute(raw, "status") != "success":
        raise ValueError(f"refusing to grade non-success log: {source.path}")
    if smoke_samples is not None:
        raw = copy.deepcopy(raw)
        samples = list(_attribute(raw, "samples", []) or [])[:smoke_samples]
        if not samples:
            raise ValueError(f"smoke input has no samples: {source.path}")
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
        raise FileExistsError(f"derived output appeared while grading; refusing overwrite: {eval_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{eval_path.name}.", suffix=".eval", dir=eval_path.parent
    )
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


def _validate_parallelism(worker_count: int, connections_per_worker: int) -> None:
    for value, name in (
        (worker_count, "workers"),
        (connections_per_worker, "connections_per_worker"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    aggregate = worker_count * connections_per_worker
    if aggregate > DEFAULT_MAX_CONNECTIONS:
        raise ValueError(
            f"workers * connections_per_worker must be <= {DEFAULT_MAX_CONNECTIONS}; got {aggregate}"
        )


def _validate_grader_max_tokens(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("grader_max_tokens must be a positive integer")


def deterministic_shards(sources: Sequence[GradeInput], worker_count: int) -> list[tuple[GradeInput, ...]]:
    """Assign inputs by a stable cell hash that survives incremental discovery."""

    _validate_parallelism(worker_count, 1)
    ordered = sorted(sources, key=lambda item: (item.condition, item.split, item.dataset, str(item.path)))
    shards: list[list[GradeInput]] = [[] for _ in range(worker_count)]
    for source in ordered:
        identity = f"{source.condition}\0{source.split}\0{source.dataset}".encode()
        shard_index = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % worker_count
        shards[shard_index].append(source)
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
    """Run one deterministic shard serially inside one worker process."""

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
    raw_root: str | Path,
    output_root: str | Path,
    *,
    preflight_report: str | Path | None = None,
    smoke: bool = False,
    smoke_samples: int = 2,
    workers: int = DEFAULT_WORKERS,
    connections_per_worker: int = DEFAULT_CONNECTIONS_PER_WORKER,
    grader_max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[tuple[GradeInput, str]]:
    """Grade fixed process shards while keeping aggregate provider connections <=500."""

    raw_path = Path(raw_root).resolve()
    output_path = Path(output_root).resolve()
    if output_path == raw_path or output_path.is_relative_to(raw_path) or raw_path.is_relative_to(output_path):
        raise ValueError("raw and derived output roots must be separate, non-nested directories")
    _validate_parallelism(workers, connections_per_worker)
    _validate_grader_max_tokens(grader_max_tokens)
    selected = (
        discover_biased_logs(raw_path)
        if preflight_report is None
        else preflight_bound_logs(raw_path, preflight_report)
    )
    if smoke:
        selected = selected[:1]
    if not selected:
        return []
    shards = deterministic_shards(selected, workers)
    active = [(index, shard) for index, shard in enumerate(shards) if shard]
    by_source: dict[GradeInput, str] = {}
    # Spawn keeps provider clients and event-loop state process-local even on
    # platforms whose multiprocessing default is fork.
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        futures = [
            executor.submit(
                _grade_shard,
                shard_index,
                shard,
                output_path,
                workers,
                connections_per_worker,
                grader_max_tokens,
                smoke_samples if smoke else None,
            )
            for shard_index, shard in active
        ]
        for future in futures:
            for source, status in future.result():
                by_source[source] = status
    ordered = [source for shard in shards for source in shard]
    ordered.sort(key=lambda item: (item.condition, item.split, item.dataset, str(item.path)))
    return [(source, by_source[source]) for source in ordered]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-log-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--preflight-report",
        type=Path,
        help="raw-preflight v1 report; grade only its hash-bound locally staged source logs",
    )
    parser.add_argument("--smoke", action="store_true", help="grade only the first two samples from one biased log")
    parser.add_argument("--smoke-samples", type=int, default=2)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--connections-per-worker", type=int, default=DEFAULT_CONNECTIONS_PER_WORKER)
    parser.add_argument(
        "--grader-max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="maximum Luna completion tokens; recorded in derived provenance",
    )
    args = parser.parse_args(argv)
    try:
        results = grade_all(
            args.raw_log_root,
            args.output_root,
            preflight_report=args.preflight_report,
            smoke=args.smoke,
            smoke_samples=args.smoke_samples,
            workers=args.workers,
            connections_per_worker=args.connections_per_worker,
            grader_max_tokens=args.grader_max_tokens,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not results:
        raise SystemExit("no successful biased Stage 1 IID logs found")
    for source, status in results:
        print(f"{status}: {source.condition}/{source.split}/{source.dataset}: {source.path}")


if __name__ == "__main__":
    main()
