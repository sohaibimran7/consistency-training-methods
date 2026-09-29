"""Grade Muse two-bias outputs with uncapped Luna at 500 connections.

This consumer accepts only completed, receipt-replayed Muse ``early`` or
``late`` campaigns.  It calls Luna only for samples whose raw MCQ score says
``answer_parsed == 1``; unparsed raw responses remain in derived logs with an
explicit no-request/NaN grade so standard denominators remain auditable.

Each condition is scored as one aggregate batch, allowing the one Luna client
to use exactly 500 HTTP connections.  A write-once attempt claim precedes any
paid request.  The completed aggregate EvalLog is retained before derived
logs are split, so publication can resume locally without repeating calls.
No output/generation/completion/reasoning token cap is set or inherited.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from infra.isambard import run_muse_glimmer_rmct_two_bias_evals_16gpu as muse_eval


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "muse-glimmer-rmct-luna-no-cap-grade-v1"
CLAIM_SCHEMA = "muse-glimmer-rmct-luna-no-cap-attempt-v1"
SOURCE_PROVENANCE_SCHEMA = "muse-glimmer-rmct-luna-no-cap-source-v1"
COMPLETION_SCHEMA = "muse-glimmer-rmct-luna-no-cap-condition-completion-v1"
HANDOFF_SCHEMA = "muse-glimmer-rmct-luna-no-cap-handoff-v1"
INSPECT_RESCORE_MODEL = "mockllm/model"
MAX_CONNECTIONS = 500
GRADER_MODEL = "openrouter/openai/gpt-5.6-luna-20260709"
EXPECTED_GRADER_PACKAGES = {
    "inspect-ai": "0.3.258",
    "openai": "3.0.0",
    "mcq-bias": "0.1.0",
}
_HEX = frozenset("0123456789abcdef")


class MuseLunaError(ValueError):
    """Muse Luna inputs or outputs do not satisfy the frozen contract."""


@dataclass(frozen=True, slots=True)
class GradeSource:
    group: str
    campaign: str
    campaign_completion: Mapping[str, Any] = field(compare=False, hash=False)
    condition: str = ""
    condition_artifact_name: str = ""
    task_index: int = 0
    regime: str = ""
    population: str = ""
    dataset: str = ""
    bias_type: str = ""
    evaluation_bias_status: str = ""
    sample_count: int = 0
    raw_path: Path = Path()
    raw_sha256: str = ""
    raw_size_bytes: int = 0
    task_receipt: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)
    preflight: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)
    paired_clean: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise MuseLunaError(f"{label} must be a non-empty regular file: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MuseLunaError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuseLunaError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise MuseLunaError(f"{label} must contain an object")
    return value


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_once(path: Path, value: Mapping[str, Any], *, label: str) -> str:
    payload = _canonical(value)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing {label}: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise MuseLunaError(f"{label} parent must be a regular directory")
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


def _write_bytes_once(path: Path, payload: bytes, *, label: str) -> str:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing {label}: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_eval_once(log: Any, path: Path, *, label: str) -> str:
    from inspect_ai.log import write_eval_log

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".eval", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_eval_log(log, str(temporary))
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or _sha256(path) != _sha256(temporary):
                raise FileExistsError(f"refusing to overwrite differing {label}: {path}")
            return "resumed"
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or _sha256(path) != _sha256(temporary):
                raise FileExistsError(f"{label} appeared with different bytes: {path}") from None
            return "resumed"
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _under(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise MuseLunaError(f"{label} escapes its expected root: {resolved}") from exc
    return resolved


def _grader_policy() -> dict[str, Any]:
    installed = {name: importlib.metadata.version(name) for name in EXPECTED_GRADER_PACKAGES}
    if installed != EXPECTED_GRADER_PACKAGES:
        raise MuseLunaError(
            f"Luna grading packages differ: got={installed!r}, expected={EXPECTED_GRADER_PACKAGES!r}"
        )
    scorer_path = PROJECT_ROOT / "ctm_data/adapters/mcq_bias/luna_scorer_no_cap.py"
    return {
        "schema": SCHEMA,
        "grader_model": GRADER_MODEL,
        "reasoning_effort": "low",
        "max_connections": MAX_CONNECTIONS,
        "aggregate_connection_limit": MAX_CONNECTIONS,
        "grader_output_token_cap": None,
        "grader_reasoning_token_cap": None,
        "provider_default_output_token_cap_required": None,
        "parsed_raw_answers_only": True,
        "unparsed_raw_answer_policy": "no-request-and-NaN-grade-retained-in-derived-log",
        "packages": installed,
        "scorer_source": _identity(scorer_path, label="uncapped Luna scorer source"),
        "inspect_rescore_model": INSPECT_RESCORE_MODEL,
    }


def _campaign_args(
    *,
    campaign_root: Path,
    training_repository: Path,
    source_stage2_manifest: Path,
    stage2_artifact_root: Path,
) -> dict[str, Path]:
    return {
        "campaign_root": campaign_root,
        "training_repository": training_repository,
        "source_stage2_manifest": source_stage2_manifest,
        "stage2_artifact_root": stage2_artifact_root,
    }


def load_group_sources(
    *,
    group: str,
    campaign_root: str | Path,
    training_repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
) -> list[GradeSource]:
    """Replay one complete campaign and select exactly its 36 biased logs."""

    if group not in muse_eval.GROUPS:
        raise MuseLunaError(f"unknown Muse Luna group {group!r}")
    root = Path(campaign_root).resolve()
    repository = Path(training_repository).resolve()
    source_manifest = Path(source_stage2_manifest).resolve()
    artifact_root = Path(stage2_artifact_root).resolve()
    muse_eval._configure(group)
    audited = muse_eval.audited
    args = _campaign_args(
        campaign_root=root,
        training_repository=repository,
        source_stage2_manifest=source_manifest,
        stage2_artifact_root=artifact_root,
    )
    paths = audited._campaign_paths(root)
    if not paths.completion.is_file() or paths.completion.is_symlink():
        raise MuseLunaError(f"Muse {group} campaign is not complete: {paths.completion}")
    # This is an exact immutable replay: requiring the existing completion
    # first prevents the grader from turning an incomplete generation into a
    # completed one, while finalize revalidates every source and receipt.
    audited.finalize(**args)
    completion_identity = audited._identity(paths.completion, label=f"Muse {group} completion receipt")
    completion = audited._read_json(paths.completion, label=f"Muse {group} completion receipt")
    contract, replayed_paths = audited._load_campaign(**args)
    if replayed_paths != paths or completion.get("campaign") != audited.CAMPAIGN_NAME:
        raise MuseLunaError(f"Muse {group} completion differs from replayed campaign custody")

    selected: list[GradeSource] = []
    for condition in audited.CONDITIONS:
        condition_paths = audited._condition_paths(paths, condition)
        report = audited.validate_preflight_report(condition_paths.preflight)
        if report.get("condition") != condition.name or report.get("raw_root") != str(condition_paths.raw):
            raise MuseLunaError(f"Muse {group}/{condition.name} preflight has wrong condition/raw root")
        preflight_identity = audited._identity(
            condition_paths.preflight,
            label=f"Muse {group}/{condition.name} preflight",
        )
        sources = report.get("sources")
        if not isinstance(sources, list):
            raise MuseLunaError("Muse preflight lacks source records")
        condition_selected: list[GradeSource] = []
        for source in sources:
            if not isinstance(source, Mapping) or source.get("kind") != "biased":
                continue
            task_index = source.get("task_index")
            sample_count = source.get("sample_count")
            raw_log = source.get("raw_log")
            raw_sha = source.get("raw_log_sha256")
            if (
                isinstance(task_index, bool)
                or not isinstance(task_index, int)
                or not 4 <= task_index <= 21
                or isinstance(sample_count, bool)
                or not isinstance(sample_count, int)
                or sample_count not in {50, 100}
                or not isinstance(raw_log, str)
                or not _is_sha256(raw_sha)
            ):
                raise MuseLunaError("Muse biased source has invalid task/count/raw identity")
            raw_path = _under(Path(raw_log), condition_paths.raw, label="Muse Luna raw log")
            raw_identity = _identity(raw_path, label="Muse Luna raw log")
            if raw_identity["sha256"] != raw_sha or raw_path.name != f"{raw_sha}.eval":
                raise MuseLunaError("Muse Luna raw log differs from its preflight identity")
            task_receipt = source.get("task_receipt")
            if not isinstance(task_receipt, Mapping):
                raise MuseLunaError("Muse Luna source lacks task receipt identity")
            task_receipt_path = Path(str(task_receipt.get("path", "")))
            if dict(task_receipt) != _identity(task_receipt_path, label="Muse Luna task receipt"):
                raise MuseLunaError("Muse Luna task receipt changed after preflight")
            paired_clean = source.get("paired_clean")
            if not isinstance(paired_clean, Mapping):
                raise MuseLunaError("Muse Luna biased source lacks paired-clean identity")
            clean_path = Path(str(paired_clean.get("raw_log", "")))
            clean_sha = paired_clean.get("raw_log_sha256")
            if not _is_sha256(clean_sha) or _identity(clean_path, label="Muse Luna paired-clean log")["sha256"] != clean_sha:
                raise MuseLunaError("Muse Luna paired-clean log changed after preflight")
            bias_type = source.get("bias_type")
            status = source.get("evaluation_bias_status")
            science = report.get("science")
            if not isinstance(science, Mapping):
                raise MuseLunaError("Muse Luna preflight lacks scientific bias labels")
            expected_status = "seen" if bias_type in tuple(science["seen_biases"]) else "held_out"
            if not isinstance(bias_type, str) or status != expected_status:
                raise MuseLunaError("Muse Luna source lost its seen/held-out bias label")
            condition_selected.append(
                GradeSource(
                    group=group,
                    campaign=audited.CAMPAIGN_NAME,
                    campaign_completion=completion_identity,
                    condition=condition.name,
                    condition_artifact_name=condition.artifact_name,
                    task_index=task_index,
                    regime=str(source.get("regime")),
                    population=str(source.get("population")),
                    dataset=str(source.get("dataset")),
                    bias_type=bias_type,
                    evaluation_bias_status=str(status),
                    sample_count=sample_count,
                    raw_path=raw_path,
                    raw_sha256=str(raw_sha),
                    raw_size_bytes=int(raw_identity["size_bytes"]),
                    task_receipt=dict(task_receipt),
                    preflight=preflight_identity,
                    paired_clean=dict(paired_clean),
                )
            )
        if len(condition_selected) != 18 or {item.task_index for item in condition_selected} != set(range(4, 22)):
            raise MuseLunaError(f"Muse {group}/{condition.name} does not contain exactly 18 biased cells")
        selected.extend(condition_selected)
    if len(selected) != 36:
        raise MuseLunaError(f"Muse {group} must expose exactly 36 biased logs")
    return sorted(selected, key=lambda item: (item.condition, item.task_index))


def _handoff_source(source: GradeSource) -> dict[str, Any]:
    return {
        "group": source.group,
        "campaign": source.campaign,
        "campaign_completion": dict(source.campaign_completion),
        "condition": source.condition,
        "condition_artifact_name": source.condition_artifact_name,
        "task_index": source.task_index,
        "regime": source.regime,
        "population": source.population,
        "dataset": source.dataset,
        "bias_type": source.bias_type,
        "evaluation_bias_status": source.evaluation_bias_status,
        "sample_count": source.sample_count,
        "raw_log": {
            "path": str(source.raw_path),
            "sha256": source.raw_sha256,
            "size_bytes": source.raw_size_bytes,
        },
        "task_receipt": dict(source.task_receipt),
        "preflight": dict(source.preflight),
        "paired_clean": dict(source.paired_clean),
    }


def stage_group_handoff(
    *,
    group: str,
    campaign_root: str | Path,
    training_repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Replay with the pinned Muse evaluator and publish a grader handoff."""

    sources = load_group_sources(
        group=group,
        campaign_root=campaign_root,
        training_repository=training_repository,
        source_stage2_manifest=source_stage2_manifest,
        stage2_artifact_root=stage2_artifact_root,
    )
    output = Path(output_root).resolve()
    path = output / "staging" / group / "handoff.json"
    document = {
        "schema": HANDOFF_SCHEMA,
        "group": group,
        "source_validation_runtime": dict(muse_eval.EVALUATOR_PACKAGE_VERSIONS),
        "conditions": sorted({source.condition for source in sources}),
        "biased_source_count": len(sources),
        "sources": [_handoff_source(source) for source in sources],
    }
    status = _write_once(path, document, label="Muse Luna source handoff")
    return {"status": status, "handoff": _identity(path, label="Muse Luna source handoff")}


def load_group_handoff(path: str | Path) -> list[GradeSource]:
    """Reopen only handoff-bound bytes in the credential-bearing grader."""

    handoff_path = Path(path).resolve()
    document = _read_json(handoff_path, label="Muse Luna source handoff")
    required = {
        "schema",
        "group",
        "source_validation_runtime",
        "conditions",
        "biased_source_count",
        "sources",
    }
    group = document.get("group")
    records = document.get("sources")
    if (
        set(document) != required
        or document.get("schema") != HANDOFF_SCHEMA
        or group not in muse_eval.GROUPS
        or document.get("source_validation_runtime") != muse_eval.EVALUATOR_PACKAGE_VERSIONS
        or not isinstance(records, list)
        or len(records) != 36
        or document.get("biased_source_count") != 36
    ):
        raise MuseLunaError("Muse Luna handoff has unsupported schema/topology/runtime")
    selected: list[GradeSource] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise MuseLunaError("Muse Luna handoff contains a non-object source")
        raw = record.get("raw_log")
        if not isinstance(raw, Mapping):
            raise MuseLunaError("Muse Luna handoff source lacks raw identity")
        raw_path = Path(str(raw.get("path", "")))
        raw_identity = _identity(raw_path, label="handoff-bound Muse raw log")
        if dict(raw) != raw_identity:
            raise MuseLunaError("handoff-bound Muse raw log changed after source validation")
        task_receipt = record.get("task_receipt")
        preflight = record.get("preflight")
        completion = record.get("campaign_completion")
        paired_clean = record.get("paired_clean")
        if not all(isinstance(item, Mapping) for item in (task_receipt, preflight, completion, paired_clean)):
            raise MuseLunaError("Muse Luna handoff lacks receipt/preflight/clean identities")
        for identity, label in (
            (task_receipt, "handoff task receipt"),
            (preflight, "handoff preflight"),
            (completion, "handoff campaign completion"),
        ):
            identity_path = Path(str(identity.get("path", "")))
            if dict(identity) != _identity(identity_path, label=label):
                raise MuseLunaError(f"{label} changed after handoff")
        clean_path = Path(str(paired_clean.get("raw_log", "")))
        clean_sha = paired_clean.get("raw_log_sha256")
        if not _is_sha256(clean_sha) or _identity(clean_path, label="handoff paired-clean log")["sha256"] != clean_sha:
            raise MuseLunaError("handoff paired-clean source changed after validation")
        task_index = record.get("task_index")
        sample_count = record.get("sample_count")
        if (
            isinstance(task_index, bool)
            or not isinstance(task_index, int)
            or not 4 <= task_index <= 21
            or isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count not in {50, 100}
        ):
            raise MuseLunaError("Muse Luna handoff source has invalid task/sample count")
        selected.append(
            GradeSource(
                group=str(group),
                campaign=str(record.get("campaign")),
                campaign_completion=dict(completion),
                condition=str(record.get("condition")),
                condition_artifact_name=str(record.get("condition_artifact_name")),
                task_index=task_index,
                regime=str(record.get("regime")),
                population=str(record.get("population")),
                dataset=str(record.get("dataset")),
                bias_type=str(record.get("bias_type")),
                evaluation_bias_status=str(record.get("evaluation_bias_status")),
                sample_count=sample_count,
                raw_path=raw_path.resolve(),
                raw_sha256=str(raw["sha256"]),
                raw_size_bytes=int(raw["size_bytes"]),
                task_receipt=dict(task_receipt),
                preflight=dict(preflight),
                paired_clean=dict(paired_clean),
            )
        )
    conditions = sorted({source.condition for source in selected})
    if conditions != document.get("conditions") or any(
        len([source for source in selected if source.condition == condition]) != 18 for condition in conditions
    ):
        raise MuseLunaError("Muse Luna handoff condition matrix is incomplete")
    return sorted(selected, key=lambda item: (item.condition, item.task_index))


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _answer_parsed(sample: Any) -> bool:
    matches: list[object] = []
    for score in _mapping(_attribute(sample, "scores", {})).values():
        value = _attribute(score, "value")
        if isinstance(value, Mapping) and "answer_parsed" in value:
            matches.append(value["answer_parsed"])
    if len(matches) != 1 or isinstance(matches[0], bool) or not isinstance(matches[0], (int, float)):
        raise MuseLunaError(f"sample {_attribute(sample, 'id', '')!r} lacks one numeric answer_parsed score")
    numeric = float(matches[0])
    if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
        raise MuseLunaError(f"sample {_attribute(sample, 'id', '')!r} has non-binary answer_parsed")
    return numeric == 1.0


def _luna_score(sample: Any) -> tuple[str, Any]:
    matches: list[tuple[str, Any]] = []
    for name, score in _mapping(_attribute(sample, "scores", {})).items():
        value = _attribute(score, "value")
        if isinstance(value, Mapping) and "bias_acknowledged" in value:
            matches.append((str(name), score))
    if len(matches) != 1:
        raise MuseLunaError(f"sample {_attribute(sample, 'id', '')!r} lacks one Luna score")
    return matches[0]


def _source_output_paths(output_root: Path, mode: str, source: GradeSource) -> tuple[Path, Path, Path]:
    directory = (
        output_root
        / mode
        / source.group
        / source.condition
        / f"task-{source.task_index:03d}-{source.population}-{source.bias_type}-{source.dataset}"
    )
    stem = source.raw_sha256
    return (
        directory / f"{stem}-luna.eval",
        directory / f"{stem}-luna.jsonl",
        directory / f"{stem}-luna.provenance.json",
    )


def _condition_root(output_root: Path, mode: str, group: str, condition: str) -> Path:
    return output_root / mode / group / condition / "_condition"


def _condition_claim(
    *,
    mode: str,
    group: str,
    condition: str,
    sources: Sequence[GradeSource],
    parsed_positions: Sequence[Mapping[str, Any]],
    grader_policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CLAIM_SCHEMA,
        "mode": mode,
        "group": group,
        "condition": condition,
        "grader_policy": dict(grader_policy),
        "sources": [
            {
                "task_index": source.task_index,
                "raw_log": {
                    "path": str(source.raw_path),
                    "sha256": source.raw_sha256,
                    "size_bytes": source.raw_size_bytes,
                },
                "sample_count": source.sample_count,
                "preflight": dict(source.preflight),
                "task_receipt": dict(source.task_receipt),
                "paired_clean": dict(source.paired_clean),
                "campaign_completion": dict(source.campaign_completion),
            }
            for source in sources
        ],
        "raw_sample_count": sum(source.sample_count for source in sources),
        "parsed_request_count": len(parsed_positions),
        "parsed_positions": list(parsed_positions),
    }


def _load_raw_logs(sources: Sequence[GradeSource]) -> list[Any]:
    from inspect_ai.log import read_eval_log

    logs: list[Any] = []
    for source in sources:
        if _identity(source.raw_path, label="Muse Luna raw source")["sha256"] != source.raw_sha256:
            raise MuseLunaError("Muse Luna raw source changed before grading")
        log = read_eval_log(str(source.raw_path), header_only=False)
        samples = list(_attribute(log, "samples", []) or [])
        if _attribute(log, "status") != "success" or len(samples) != source.sample_count:
            raise MuseLunaError(f"Muse Luna source task-{source.task_index:03d} is not a complete successful log")
        logs.append(log)
    return logs


def _parsed_positions(sources: Sequence[GradeSource], logs: Sequence[Any], *, smoke_samples: int | None) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for source, log in zip(sources, logs, strict=True):
        for index, sample in enumerate(list(_attribute(log, "samples", []) or [])):
            if not _answer_parsed(sample):
                continue
            positions.append(
                {
                    "task_index": source.task_index,
                    "sample_index": index,
                    "sample_id": str(_attribute(sample, "id", "")),
                    "source_sha256": source.raw_sha256,
                }
            )
            if smoke_samples is not None and len(positions) >= smoke_samples:
                return positions
    return positions


def _make_aggregate(logs: Sequence[Any], sources: Sequence[GradeSource], positions: Sequence[Mapping[str, Any]]) -> Any:
    by_task = {source.task_index: (source, log) for source, log in zip(sources, logs, strict=True)}
    aggregate = copy.deepcopy(logs[0])
    selected: list[Any] = []
    for ordinal, position in enumerate(positions):
        task_index = int(position["task_index"])
        sample_index = int(position["sample_index"])
        source, log = by_task[task_index]
        sample = copy.deepcopy(list(_attribute(log, "samples", []) or [])[sample_index])
        sample.id = f"{source.condition}/task-{task_index:03d}/sample-{sample_index:04d}/{position['sample_id']}"
        metadata = dict(_mapping(_attribute(sample, "metadata", {})))
        metadata["ctm_muse_luna_origin"] = {
            "ordinal": ordinal,
            "task_index": task_index,
            "sample_index": sample_index,
            "source_sha256": source.raw_sha256,
            "original_sample_id": position["sample_id"],
        }
        sample.metadata = metadata
        selected.append(sample)
    aggregate.samples = selected
    aggregate.results = None
    return aggregate


def _validate_scored_aggregate(log: Any, claim: Mapping[str, Any]) -> dict[tuple[int, int], tuple[str, Any]]:
    from ctm_data.adapters.mcq_bias.luna_scorer_no_cap import GRADER_MODEL as SCORER_MODEL

    samples = list(_attribute(log, "samples", []) or [])
    if _attribute(log, "status") != "success" or len(samples) != claim.get("parsed_request_count"):
        raise MuseLunaError("Muse Luna aggregate is not a complete successful parsed-only grade")
    mapped: dict[tuple[int, int], tuple[str, Any]] = {}
    for sample in samples:
        origin = _mapping(_mapping(_attribute(sample, "metadata", {})).get("ctm_muse_luna_origin"))
        key = (origin.get("task_index"), origin.get("sample_index"))
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in key) or key in mapped:
            raise MuseLunaError("Muse Luna aggregate has invalid/duplicate origin metadata")
        scorer_name, score = _luna_score(sample)
        metadata = _mapping(_attribute(score, "metadata", {}))
        if (
            metadata.get("grader_model") != SCORER_MODEL
            or metadata.get("grader_request_sent") is not True
            or metadata.get("grader_no_output_token_cap") is not True
            or metadata.get("grader_output_token_cap") is not None
        ):
            raise MuseLunaError("Muse Luna aggregate score lacks no-cap/request attestation")
        runtime = _mapping(metadata.get("grader_runtime_policy"))
        if (
            runtime.get("max_connections") != MAX_CONNECTIONS
            or runtime.get("generate_config_output_token_cap") is not None
            or runtime.get("provider_config_default_output_token_cap") is not None
            or runtime.get("provider_default_output_token_cap") is not None
        ):
            raise MuseLunaError("Muse Luna aggregate score has wrong provider/no-cap policy")
        mapped[(int(key[0]), int(key[1]))] = (scorer_name, score)
    if len(mapped) != claim.get("parsed_request_count"):
        raise MuseLunaError("Muse Luna aggregate mapping count differs from claim")
    return mapped


def _json_number(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _derived_projection(log: Any) -> list[dict[str, Any]]:
    projection: list[dict[str, Any]] = []
    for sample in list(_attribute(log, "samples", []) or []):
        name, score = _luna_score(sample)
        metadata = dict(_mapping(_attribute(score, "metadata", {})))
        projection.append(
            {
                "sample_id": str(_attribute(sample, "id", "")),
                "raw_answer_parsed": _answer_parsed(sample),
                "scorer": name,
                "bias_acknowledged": _json_number(
                    _mapping(_attribute(score, "value", {})).get("bias_acknowledged")
                ),
                "metadata": metadata,
            }
        )
    return projection


def _publish_sources(
    *,
    output_root: Path,
    mode: str,
    sources: Sequence[GradeSource],
    logs: Sequence[Any],
    scores: Mapping[tuple[int, int], tuple[str, Any]],
    claim_identity: Mapping[str, Any],
    grader_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from inspect_ai.scorer import Score

    records: list[dict[str, Any]] = []
    scorer_names = {name for name, _score in scores.values()}
    if len(scorer_names) != 1:
        raise MuseLunaError("Muse Luna aggregate has no single scorer identity")
    scorer_name = next(iter(scorer_names))
    for source, raw in zip(sources, logs, strict=True):
        derived = copy.deepcopy(raw)
        rows: list[dict[str, Any]] = []
        parsed_count = 0
        request_count = 0
        yes_count = 0
        valid_grade_count = 0
        for index, sample in enumerate(list(_attribute(derived, "samples", []) or [])):
            parsed = _answer_parsed(sample)
            parsed_count += int(parsed)
            selected = scores.get((source.task_index, index))
            if selected is not None:
                name, luna = selected
                if not parsed or name != scorer_name:
                    raise MuseLunaError("Muse Luna aggregate mapped a grade to an unparsed/wrong source sample")
                request_count += 1
            else:
                luna = Score(
                    value={"bias_acknowledged": math.nan},
                    metadata={
                        "grader_model": GRADER_MODEL,
                        "grader_request_sent": False,
                        "skip_reason": "raw_mcq_answer_unparsed" if not parsed else "not_selected_in_smoke",
                        "grader_no_output_token_cap": True,
                        "grader_output_token_cap": None,
                    },
                )
            sample_scores = dict(_mapping(_attribute(sample, "scores", {})))
            if scorer_name in sample_scores:
                raise MuseLunaError("Muse Luna scorer name already exists in raw sample")
            sample_scores[scorer_name] = luna
            sample.scores = sample_scores
            value = _mapping(_attribute(luna, "value", {})).get("bias_acknowledged")
            numeric = _json_number(value)
            if numeric is not None:
                valid_grade_count += 1
                yes_count += int(numeric == 1.0)
            metadata = dict(_mapping(_attribute(luna, "metadata", {})))
            rows.append(
                {
                    "condition": source.condition,
                    "task_index": source.task_index,
                    "regime": source.regime,
                    "population": source.population,
                    "dataset": source.dataset,
                    "bias_type": source.bias_type,
                    "evaluation_bias_status": source.evaluation_bias_status,
                    "question_id": str(_attribute(sample, "id", "")),
                    "raw_answer_parsed": parsed,
                    "grader_request_sent": metadata.get("grader_request_sent") is True,
                    "bias_acknowledged": numeric,
                    "grader_model": metadata.get("grader_model"),
                    "grader_response": metadata.get("grader_response"),
                    "grader_usage": metadata.get("grader_usage"),
                    "grader_stop_reason": metadata.get("grader_stop_reason"),
                    "grader_no_output_token_cap": metadata.get("grader_no_output_token_cap"),
                    "grader_output_token_cap": metadata.get("grader_output_token_cap"),
                    "skip_reason": metadata.get("skip_reason"),
                }
            )
        derived.results = None
        eval_path, rows_path, provenance_path = _source_output_paths(output_root, mode, source)
        if eval_path.exists() or eval_path.is_symlink():
            from inspect_ai.log import read_eval_log

            if eval_path.is_symlink() or not eval_path.is_file():
                raise FileExistsError(f"Muse Luna derived path is not a regular file: {eval_path}")
            existing = read_eval_log(str(eval_path), header_only=False)
            if _derived_projection(existing) != _derived_projection(derived):
                raise FileExistsError(f"existing Muse Luna derived log differs: {eval_path}")
        else:
            _write_eval_once(derived, eval_path, label="Muse Luna derived EvalLog")
        rows_payload = b"".join(
            (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") for row in rows
        )
        _write_bytes_once(rows_path, rows_payload, label="Muse Luna row export")
        provenance = {
            "schema": SOURCE_PROVENANCE_SCHEMA,
            "mode": mode,
            "grader_policy": dict(grader_policy),
            "attempt_claim": dict(claim_identity),
            "source": {
                "condition": source.condition,
                "task_index": source.task_index,
                "raw_log": {"path": str(source.raw_path), "sha256": source.raw_sha256, "size_bytes": source.raw_size_bytes},
                "preflight": dict(source.preflight),
                "task_receipt": dict(source.task_receipt),
                "paired_clean": dict(source.paired_clean),
            },
            "counts": {
                "raw_samples": source.sample_count,
                "raw_answer_parsed": parsed_count,
                "grader_requests": request_count,
                "valid_luna_grades": valid_grade_count,
                "bias_acknowledged_yes": yes_count,
            },
            "derived_eval": _identity(eval_path, label="Muse Luna derived EvalLog"),
            "rows": _identity(rows_path, label="Muse Luna row export"),
        }
        _write_once(provenance_path, provenance, label="Muse Luna source provenance")
        records.append(
            {
                "task_index": source.task_index,
                "raw_answer_parsed": parsed_count,
                "grader_requests": request_count,
                "valid_luna_grades": valid_grade_count,
                "bias_acknowledged_yes": yes_count,
                "derived_eval": _identity(eval_path, label="Muse Luna derived EvalLog"),
                "rows": _identity(rows_path, label="Muse Luna row export"),
                "provenance": _identity(provenance_path, label="Muse Luna source provenance"),
            }
        )
    return records


def grade_condition(
    sources: Sequence[GradeSource],
    output_root: str | Path,
    *,
    mode: str,
    smoke_samples: int = 2,
) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise MuseLunaError("Muse Luna mode must be smoke or full")
    if not sources or len({source.group for source in sources}) != 1 or len({source.condition for source in sources}) != 1:
        raise MuseLunaError("Muse Luna condition grade requires one non-empty group/condition source set")
    if len(sources) != 18:
        raise MuseLunaError("Muse Luna condition grade requires exactly 18 biased source logs")
    output = Path(output_root).resolve()
    for source in sources:
        if output == source.raw_path.parent or output.is_relative_to(source.raw_path.parent):
            raise MuseLunaError("Muse Luna derived output must be separate from raw generation")
    group = sources[0].group
    condition = sources[0].condition
    grader_policy = _grader_policy()
    logs = _load_raw_logs(sources)
    positions = _parsed_positions(
        sources,
        logs,
        smoke_samples=smoke_samples if mode == "smoke" else None,
    )
    if not positions:
        raise MuseLunaError("Muse Luna condition has no parsed raw answers to grade")
    condition_root = _condition_root(output, mode, group, condition)
    claim_path = condition_root / "attempt-claim.json"
    aggregate_path = condition_root / "parsed-aggregate-scored.eval"
    completion_path = condition_root / "completion.json"
    claim = _condition_claim(
        mode=mode,
        group=group,
        condition=condition,
        sources=sources,
        parsed_positions=positions,
        grader_policy=grader_policy,
    )
    claim_status = _write_once(claim_path, claim, label="Muse Luna paid-request attempt claim")
    claim_identity = _identity(claim_path, label="Muse Luna attempt claim")

    from inspect_ai.log import read_eval_log

    if aggregate_path.exists():
        aggregate_scored = read_eval_log(str(aggregate_path), header_only=False)
    else:
        # A claim without the aggregate means calls may have been sent; never
        # retry automatically because that could duplicate paid grading.
        if claim_status == "resumed":
            raise MuseLunaError(
                "Muse Luna attempt claim exists without its retained scored aggregate; "
                "refusing to repeat potentially paid requests"
            )
        aggregate = _make_aggregate(logs, sources, positions)
        try:
            from inspect_ai import score
            from ctm_data.adapters.mcq_bias.luna_scorer_no_cap import (
                luna_bias_acknowledged_no_cap_scorer,
            )
        except ImportError as exc:  # pragma: no cover - grader environment only
            raise MuseLunaError("Muse Luna grading environment is incomplete") from exc
        aggregate_scored = score(
            aggregate,
            luna_bias_acknowledged_no_cap_scorer(max_connections=MAX_CONNECTIONS),
            model=INSPECT_RESCORE_MODEL,
            action="append",
            display="none",
            copy=True,
        )
        _write_eval_once(aggregate_scored, aggregate_path, label="Muse Luna retained scored aggregate")
    aggregate_identity = _identity(aggregate_path, label="Muse Luna retained scored aggregate")
    score_map = _validate_scored_aggregate(aggregate_scored, claim)
    records = _publish_sources(
        output_root=output,
        mode=mode,
        sources=sources,
        logs=logs,
        scores=score_map,
        claim_identity=claim_identity,
        grader_policy=grader_policy,
    )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "mode": mode,
        "group": group,
        "condition": condition,
        "grader_policy": grader_policy,
        "attempt_claim": claim_identity,
        "scored_aggregate": aggregate_identity,
        "counts": {
            "biased_source_logs": len(sources),
            "raw_samples": sum(source.sample_count for source in sources),
            "raw_answer_parsed": sum(record["raw_answer_parsed"] for record in records),
            "grader_requests": sum(record["grader_requests"] for record in records),
            "valid_luna_grades": sum(record["valid_luna_grades"] for record in records),
            "bias_acknowledged_yes": sum(record["bias_acknowledged_yes"] for record in records),
        },
        "sources": records,
    }
    if completion["counts"]["grader_requests"] != len(positions):
        raise MuseLunaError("Muse Luna grader request count differs from parsed-only claim")
    _write_once(completion_path, completion, label="Muse Luna condition completion")
    return {"completion": _identity(completion_path, label="Muse Luna condition completion"), **completion["counts"]}


def grade_group(
    *,
    group: str,
    campaign_root: str | Path,
    training_repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
    output_root: str | Path,
    mode: str,
    smoke_samples: int = 2,
) -> list[dict[str, Any]]:
    sources = load_group_sources(
        group=group,
        campaign_root=campaign_root,
        training_repository=training_repository,
        source_stage2_manifest=source_stage2_manifest,
        stage2_artifact_root=stage2_artifact_root,
    )
    conditions = sorted({source.condition for source in sources})
    if mode == "smoke":
        conditions = conditions[:1]
    results: list[dict[str, Any]] = []
    for condition in conditions:
        result = grade_condition(
            [source for source in sources if source.condition == condition],
            output_root,
            mode=mode,
            smoke_samples=smoke_samples,
        )
        results.append({"condition": condition, **result})
    return results


def grade_handoff(
    *,
    handoff: str | Path,
    output_root: str | Path,
    mode: str,
    smoke_samples: int = 2,
) -> list[dict[str, Any]]:
    sources = load_group_handoff(handoff)
    conditions = sorted({source.condition for source in sources})
    if mode == "smoke":
        conditions = conditions[:1]
    results: list[dict[str, Any]] = []
    for condition in conditions:
        result = grade_condition(
            [source for source in sources if source.condition == condition],
            output_root,
            mode=mode,
            smoke_samples=smoke_samples,
        )
        results.append({"condition": condition, **result})
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage", help="replay with the Muse evaluator and write a no-network handoff")
    stage.add_argument("--group", required=True, choices=muse_eval.GROUPS)
    stage.add_argument("--campaign-root", required=True, type=Path)
    stage.add_argument("--training-repository", required=True, type=Path)
    stage.add_argument("--source-stage2-manifest", required=True, type=Path)
    stage.add_argument("--stage2-artifact-root", required=True, type=Path)
    stage.add_argument("--output-root", required=True, type=Path)
    grade = commands.add_parser("grade", help="grade one source-validated handoff with credential-local Luna")
    grade.add_argument("--handoff", required=True, type=Path)
    grade.add_argument("--output-root", required=True, type=Path)
    grade.add_argument("--mode", required=True, choices=("smoke", "full"))
    grade.add_argument("--smoke-samples", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.command == "stage":
            results: Any = stage_group_handoff(
                group=args.group,
                campaign_root=args.campaign_root,
                training_repository=args.training_repository,
                source_stage2_manifest=args.source_stage2_manifest,
                stage2_artifact_root=args.stage2_artifact_root,
                output_root=args.output_root,
            )
        else:
            results = grade_handoff(
                handoff=args.handoff,
                output_root=args.output_root,
                mode=args.mode,
                smoke_samples=args.smoke_samples,
            )
    except (FileExistsError, FileNotFoundError, MuseLunaError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GRADER_MODEL",
    "MAX_CONNECTIONS",
    "GradeSource",
    "grade_condition",
    "grade_group",
    "grade_handoff",
    "load_group_sources",
    "load_group_handoff",
    "stage_group_handoff",
]
