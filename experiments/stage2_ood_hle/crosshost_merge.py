"""Safely finalize a split-host Stage 2 raw-EvalLog matrix locally.

This module deliberately has no SSH, SCP, Vast, or scheduler integration.  A
caller first brings completed source EvalLogs into one or more *local* handoff
directories, then this helper performs the only safe local merge:

* the canonical condition root already owns all three clean EvalLogs at the
  exact paths used by the paired-switch scorer;
* every incoming biased EvalLog is checked against the frozen Stage-2 task
  identity, runtime contract, and SHA-256 of its source clean EvalLog; and
* missing biased cells are copied into a new, immutable transaction directory
  under the canonical root.  Existing files are never overwritten.

The final normal ``raw_preflight`` is still authoritative.  Its immutable
report, plus a separate handoff receipt, are written only after the complete
21-cell matrix passes preflight.  If anything fails, any already copied files
remain in their transaction directory for inspection and a later invocation
may safely resume exact copies.

The helper intentionally does not relocate clean logs.  A biased EvalLog's
switch score records the absolute path of the clean log it used.  Moving a
clean log into an arbitrary handoff directory would make that evidence false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.stage2_ood_hle import raw_preflight
from experiments.stage2_ood_hle.tasks import OODTaskSpec, ood_task_specs

HANDOFF_SCHEMA = "stage2-ood-hle-crosshost-handoff-v1"
TRANSACTION_DIRECTORY = "_crosshost_handoff"
_HANDOFF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
TaskIdentity = tuple[str, str, str, str, str | None]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One successful, header-validated Stage-2 EvalLog in a local tree."""

    identity: TaskIdentity
    spec: OODTaskSpec
    root: Path
    path: Path
    created: str
    sha256: str
    model: str
    runtime: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PairEvidence:
    """The source clean file proving a transferred biased score's pairing."""

    source_clean: Path
    canonical_clean: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class ImportPlan:
    candidate: Candidate
    destination: Path
    pair_evidence: PairEvidence


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Published paths and resumable-copy outcomes from one finalization."""

    preflight_report: Path
    preflight_status: str
    handoff_report: Path
    handoff_status: str
    copy_statuses: tuple[tuple[Path, str], ...]


def _sha256_file(path: Path) -> str:
    return raw_preflight._sha256_file(path)


def _safe_component(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _HANDOFF_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty safe path component")
    return value


def _validate_condition(value: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("condition must be a non-empty single path component")
    return value


def _as_root(value: str | Path, *, field: str) -> Path:
    root = Path(value).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{field} does not exist or is not a directory: {root}")
    return root


def _report_destination(value: str | Path, *, field: str) -> Path:
    supplied = Path(value)
    if supplied.exists() and supplied.is_symlink():
        raise ValueError(f"{field} must not be a symbolic link: {supplied}")
    return supplied.resolve()


def _non_nested_incoming_roots(canonical_root: Path, values: Sequence[str | Path]) -> tuple[Path, ...]:
    if not values:
        raise ValueError("at least one --incoming-log-root is required")
    roots = tuple(sorted((_as_root(value, field="incoming log root") for value in values), key=str))
    if len(set(roots)) != len(roots):
        raise ValueError("incoming log roots must be distinct")
    for root in roots:
        if root == canonical_root or root.is_relative_to(canonical_root) or canonical_root.is_relative_to(root):
            raise ValueError("incoming log roots must be separate and non-nested from --raw-log-root")
    return roots


def _identity_label(identity: TaskIdentity) -> str:
    return raw_preflight._display_identity(identity)


def _identity_payload(identity: TaskIdentity) -> dict[str, str | None]:
    kind, regime, population, dataset, bias_type = identity
    return {
        "kind": kind,
        "regime": regime,
        "population": population,
        "dataset": dataset,
        "bias_type": bias_type,
    }


def _identity_digest(identity: TaskIdentity) -> str:
    payload = json.dumps(_identity_payload(identity), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scan_successful_stage2_logs(
    root: Path,
    *,
    canonical_root: Path,
    expected: Mapping[TaskIdentity, OODTaskSpec],
    runtime: Mapping[str, Any],
) -> list[Candidate]:
    """Read successful Stage-2 headers in one local root without selecting retries."""

    observed: list[Candidate] = []
    for candidate_path in raw_preflight._discover_eval_log_paths(root):
        path = Path(candidate_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:  # defensive: the production discovery helper already checks this
            raise ValueError(f"Stage 2 EvalLog discovery escaped its supplied root: {path}") from exc
        try:
            header = raw_preflight._read_eval_log(path, header_only=True)
        except Exception as exc:
            raise ValueError(f"could not read Stage 2 raw EvalLog header: {path}") from exc
        if raw_preflight._attribute(header, "status") != "success":
            continue
        evaluation = raw_preflight._attribute(header, "eval")
        task_name = raw_preflight._task_basename(raw_preflight._attribute(evaluation, "task"))
        if task_name not in {raw_preflight.TASK_UNBIASED, raw_preflight.TASK_BIASED}:
            continue
        identity = raw_preflight._parse_candidate_identity(evaluation, task_name=task_name, path=path)
        spec = expected.get(identity)
        if spec is None:
            raise ValueError(f"unexpected successful Stage 2 raw task cell: {_identity_label(identity)} at {path}")
        created = raw_preflight._validate_header(header, path=path, spec=spec, raw_root=canonical_root)
        model, observed_runtime = raw_preflight._assert_runtime(path, runtime=runtime)
        observed.append(
            Candidate(
                identity=identity,
                spec=spec,
                root=root,
                path=path,
                created=created,
                sha256=_sha256_file(path),
                model=model,
                runtime=dict(observed_runtime),
            )
        )
    return observed


def _by_identity(candidates: Sequence[Candidate]) -> dict[TaskIdentity, list[Candidate]]:
    grouped: dict[TaskIdentity, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.identity].append(candidate)
    for values in grouped.values():
        values.sort(key=lambda item: (item.created, str(item.path)))
    return dict(grouped)


def _select_existing(identity: TaskIdentity, candidates: Sequence[Candidate]) -> Candidate:
    """Mirror raw_preflight's timestamp rule, including its tie rejection."""

    if not candidates:
        raise ValueError(f"internal error: no candidates for {_identity_label(identity)}")
    latest = max(item.created for item in candidates)
    tied = [item for item in candidates if item.created == latest]
    if len(tied) != 1:
        paths = ", ".join(str(item.path) for item in tied)
        raise ValueError(f"ambiguous successful Stage 2 raw retries for {_identity_label(identity)}: {paths}")
    return tied[0]


def _select_incoming(identity: TaskIdentity, candidates: Sequence[Candidate]) -> Candidate:
    """Allow byte-identical aliases, but never choose between differing imports."""

    if not candidates:
        raise ValueError(f"internal error: no incoming candidates for {_identity_label(identity)}")
    hashes = {item.sha256 for item in candidates}
    if len(hashes) != 1:
        locations = ", ".join(f"{item.path} ({item.sha256})" for item in candidates)
        raise ValueError(
            f"conflicting incoming successful Stage 2 EvalLogs for {_identity_label(identity)}; "
            f"curate one exact retry before merging: {locations}"
        )
    return min(candidates, key=lambda item: (str(item.root), str(item.path)))


def _canonical_clean_selection(
    canonical: Mapping[TaskIdentity, Sequence[Candidate]],
    expected: Mapping[TaskIdentity, OODTaskSpec],
) -> dict[tuple[str, str], Candidate]:
    """Require all clean cells at their real canonical paths before import."""

    selected: dict[tuple[str, str], Candidate] = {}
    missing: list[str] = []
    for identity, spec in expected.items():
        if spec.kind != "unbiased":
            continue
        values = canonical.get(identity, ())
        if not values:
            missing.append(_identity_label(identity))
            continue
        item = _select_existing(identity, values)
        selected[(spec.population, spec.dataset)] = item
    if missing:
        raise ValueError(
            "cross-host Stage 2 merge requires all three clean EvalLogs already in the canonical raw root; "
            "it will not relocate clean files whose absolute paired paths would change. "
            f"missing={sorted(missing)}"
        )
    if len(selected) != raw_preflight.EXPECTED_CLEAN_TASKS:  # pragma: no cover - task topology guard
        raise RuntimeError("internal Stage 2 clean-cell topology is inconsistent")
    return selected


def _source_clean_evidence(candidate: Candidate, *, canonical_root: Path, canonical_clean: Candidate) -> PairEvidence:
    """Prove a transferred biased log used the same bytes as the canonical clean log.

    The source root must contain the clean EvalLog at the same *relative* path
    as the canonical root.  This lets a locally staged copy of a remote raw
    directory provide hash evidence without rewriting any EvalLog paths.
    """

    try:
        full_log = raw_preflight._read_eval_log(candidate.path, header_only=False)
    except Exception as exc:
        raise ValueError(f"could not read full incoming Stage 2 EvalLog: {candidate.path}") from exc
    resolved_paths: set[Path] = set()
    for sample in list(raw_preflight._attribute(full_log, "samples", []) or []):
        _switch, score_metadata = raw_preflight._switch_score(sample, path=candidate.path)
        resolved_paths.add(
            raw_preflight._normalize_resolved_clean_path(score_metadata.get("unbiased_log"), path=candidate.path)
        )
    if resolved_paths != {canonical_clean.path}:
        rendered = ", ".join(str(path) for path in sorted(resolved_paths)) or "<none>"
        raise ValueError(
            f"incoming biased EvalLog does not resolve the selected canonical clean log for "
            f"{_identity_label(candidate.identity)}: {rendered} != {canonical_clean.path}"
        )
    try:
        relative_clean = canonical_clean.path.relative_to(canonical_root)
    except ValueError as exc:  # canonical scan should already make this impossible
        raise ValueError(f"selected canonical clean log lies outside raw root: {canonical_clean.path}") from exc
    source_clean = (candidate.root / relative_clean).resolve()
    try:
        source_clean.relative_to(candidate.root)
    except ValueError as exc:
        raise ValueError(f"source clean path escaped incoming root: {source_clean}") from exc
    if not source_clean.is_file():
        raise FileNotFoundError(
            f"incoming root has no clean EvalLog matching the paired source path for "
            f"{_identity_label(candidate.identity)}: {source_clean}"
        )
    expected_sha256 = canonical_clean.sha256
    if _sha256_file(source_clean) != expected_sha256:
        raise ValueError(
            f"incoming paired clean EvalLog bytes differ from the canonical clean EvalLog for "
            f"{_identity_label(candidate.identity)}: {source_clean} != {canonical_clean.path}"
        )
    return PairEvidence(source_clean=source_clean, canonical_clean=canonical_clean.path, sha256=expected_sha256)


def _transaction_root(canonical_root: Path, handoff_id: str) -> Path:
    parent = canonical_root / TRANSACTION_DIRECTORY
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise ValueError(f"cross-host transaction parent is not a real directory: {parent}")
    if not parent.exists():
        parent.mkdir()
    transaction = parent / handoff_id
    if transaction.exists() and (transaction.is_symlink() or not transaction.is_dir()):
        raise ValueError(f"cross-host transaction path is not a real directory: {transaction}")
    if not transaction.exists():
        transaction.mkdir()
    return transaction


def _ensure_transaction_destination_parent(transaction: Path, destination: Path) -> None:
    """Create only real child directories below our transaction directory."""

    try:
        relative = destination.relative_to(transaction)
    except ValueError as exc:  # defensive: destinations are constructed locally below
        raise ValueError(f"cross-host destination escaped its transaction directory: {destination}") from exc
    if len(relative.parts) < 2 or relative.name != destination.name:
        raise ValueError(f"cross-host destination has an invalid transaction-relative path: {destination}")
    parent = transaction
    for component in relative.parts[:-1]:
        if component in {"", ".", ".."}:  # pragma: no cover - pathlib construction guard
            raise ValueError(f"cross-host destination has an unsafe path component: {destination}")
        parent = parent / component
        if parent.exists():
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError(f"cross-host destination parent is not a real directory: {parent}")
        else:
            parent.mkdir()


def _copy_new(source: Path, destination: Path, *, expected_sha256: str) -> str:
    """Copy exactly one immutable EvalLog; never replace an existing path."""

    if _sha256_file(source) != expected_sha256:
        raise ValueError(f"incoming Stage 2 EvalLog changed after it was validated: {source}")
    if destination.exists():
        if destination.is_symlink() or not destination.is_file() or _sha256_file(destination) != expected_sha256:
            raise FileExistsError(f"refusing to overwrite differing cross-host Stage 2 EvalLog: {destination}")
        return "resumed"
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ValueError(f"cross-host destination parent is not a real directory: {destination.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if _sha256_file(temporary) != expected_sha256:
            raise ValueError(f"copied cross-host Stage 2 EvalLog has the wrong SHA-256: {source}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or _sha256_file(destination) != expected_sha256:
                raise FileExistsError(f"cross-host Stage 2 destination appeared and differs: {destination}")
            return "resumed"
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return "copied"


def _write_json_new(path: str | Path, document: Mapping[str, Any]) -> str:
    """Atomically publish one JSON receipt, allowing only byte-identical resume."""

    destination = Path(path).resolve()
    payload = (json.dumps(dict(document), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_symlink() and destination.is_file() and destination.read_bytes() == payload:
            return "resumed"
        raise FileExistsError(f"refusing to overwrite differing Stage 2 cross-host handoff receipt: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
                raise FileExistsError(f"Stage 2 cross-host handoff receipt appeared and differs: {destination}")
            return "resumed"
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return "written"


def _absolute_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"cross-host handoff receipt has no absolute {field}")
    return Path(value).resolve()


def validate_handoff_report(report: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Validate the portable immutable receipt before treating a merge as complete."""

    if isinstance(report, Mapping):
        document = dict(report)
    else:
        path = Path(report)
        if not path.is_file():
            raise FileNotFoundError(f"Stage 2 cross-host handoff receipt does not exist: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Stage 2 cross-host handoff JSON receipt: {path}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("Stage 2 cross-host handoff receipt must be a JSON object")
        document = dict(value)
    if document.get("schema") != HANDOFF_SCHEMA:
        raise ValueError("unsupported Stage 2 cross-host handoff receipt schema")
    _validate_condition(document.get("condition"))
    _safe_component(document.get("handoff_id"), field="handoff_id")
    raw_root = _absolute_path(document.get("raw_root"), field="raw_root")
    _absolute_path(document.get("manifest"), field="manifest")
    if not raw_preflight._is_sha256(document.get("manifest_sha256")):
        raise ValueError("cross-host handoff receipt has an invalid manifest_sha256")
    incoming_roots = document.get("incoming_roots")
    if not isinstance(incoming_roots, list) or not incoming_roots:
        raise ValueError("cross-host handoff receipt has no incoming_roots")
    normalized_roots = [_absolute_path(value, field="incoming_root") for value in incoming_roots]
    if len(set(normalized_roots)) != len(normalized_roots) or normalized_roots != sorted(normalized_roots, key=str):
        raise ValueError("cross-host handoff receipt incoming_roots must be unique and sorted")
    for root in normalized_roots:
        if root == raw_root or root.is_relative_to(raw_root) or raw_root.is_relative_to(root):
            raise ValueError("cross-host handoff receipt has nested incoming/raw roots")
    preflight = document.get("preflight")
    if not isinstance(preflight, Mapping) or set(preflight) != {"path", "sha256", "schema"}:
        raise ValueError("cross-host handoff receipt has an invalid preflight binding")
    _absolute_path(preflight.get("path"), field="preflight path")
    if preflight.get("schema") != raw_preflight.PREFLIGHT_SCHEMA or not raw_preflight._is_sha256(
        preflight.get("sha256")
    ):
        raise ValueError("cross-host handoff receipt has an invalid preflight binding")
    sources = document.get("incoming_sources")
    if not isinstance(sources, list):
        raise ValueError("cross-host handoff receipt has no incoming_sources list")
    seen: set[TaskIdentity] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("cross-host handoff incoming source must be an object")
        required = {
            "identity",
            "source_root",
            "source_raw_log",
            "source_raw_log_sha256",
            "canonical_raw_log",
            "canonical_raw_log_sha256",
        }
        identity_raw = source.get("identity")
        if not isinstance(identity_raw, Mapping) or set(identity_raw) != {
            "kind",
            "regime",
            "population",
            "dataset",
            "bias_type",
        }:
            raise ValueError("cross-host handoff incoming source has an invalid identity")
        identity = (
            identity_raw["kind"],
            identity_raw["regime"],
            identity_raw["population"],
            identity_raw["dataset"],
            identity_raw["bias_type"],
        )
        if identity not in raw_preflight._expected_static_cells() or identity in seen:
            raise ValueError("cross-host handoff incoming source has an invalid or duplicate task cell")
        seen.add(identity)
        if identity[0] == "biased":
            required.add("source_paired_clean")
        if set(source) != required:
            raise ValueError("cross-host handoff incoming source has invalid fields")
        source_root = _absolute_path(source.get("source_root"), field="source_root")
        if source_root not in normalized_roots:
            raise ValueError("cross-host handoff incoming source refers to an undeclared source_root")
        source_path = _absolute_path(source.get("source_raw_log"), field="source_raw_log")
        canonical_path = _absolute_path(source.get("canonical_raw_log"), field="canonical_raw_log")
        try:
            source_path.relative_to(source_root)
            canonical_path.relative_to(raw_root)
        except ValueError as exc:
            raise ValueError("cross-host handoff source path lies outside its declared root") from exc
        if not raw_preflight._is_sha256(source.get("source_raw_log_sha256")) or not raw_preflight._is_sha256(
            source.get("canonical_raw_log_sha256")
        ):
            raise ValueError("cross-host handoff incoming source has invalid SHA-256 metadata")
        if identity[0] == "biased":
            paired = source.get("source_paired_clean")
            if not isinstance(paired, Mapping) or set(paired) != {"source_raw_log", "canonical_raw_log", "sha256"}:
                raise ValueError("cross-host handoff biased source has invalid clean-pair evidence")
            paired_source = _absolute_path(paired.get("source_raw_log"), field="source paired clean")
            paired_canonical = _absolute_path(paired.get("canonical_raw_log"), field="canonical paired clean")
            try:
                paired_source.relative_to(source_root)
                paired_canonical.relative_to(raw_root)
            except ValueError as exc:
                raise ValueError("cross-host handoff clean-pair path lies outside its declared root") from exc
            if not raw_preflight._is_sha256(paired.get("sha256")):
                raise ValueError("cross-host handoff biased source has invalid paired-clean SHA-256")
    return document


def write_handoff_report(path: str | Path, report: Mapping[str, Any]) -> str:
    """Validate then atomically create a receipt, or resume only exact bytes."""

    validate_handoff_report(report)
    return _write_json_new(path, report)


def _incoming_receipt_sources(
    incoming_selected: Mapping[TaskIdentity, Candidate],
    *,
    report_sources: Mapping[TaskIdentity, Mapping[str, Any]],
    pair_evidence: Mapping[TaskIdentity, PairEvidence],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for identity in sorted(incoming_selected):
        candidate = incoming_selected[identity]
        selected = report_sources.get(identity)
        if selected is None:
            raise ValueError(f"post-merge preflight omitted incoming task cell: {_identity_label(identity)}")
        if selected["raw_log_sha256"] != candidate.sha256:
            raise ValueError(
                f"post-merge preflight selected different bytes than the validated incoming EvalLog for "
                f"{_identity_label(identity)}"
            )
        record: dict[str, Any] = {
            "identity": _identity_payload(identity),
            "source_root": str(candidate.root),
            "source_raw_log": str(candidate.path),
            "source_raw_log_sha256": candidate.sha256,
            "canonical_raw_log": str(selected["raw_log"]),
            "canonical_raw_log_sha256": str(selected["raw_log_sha256"]),
        }
        if candidate.spec.kind == "biased":
            evidence = pair_evidence.get(identity)
            if evidence is None:
                raise RuntimeError(f"missing clean-pair evidence for incoming task cell: {_identity_label(identity)}")
            record["source_paired_clean"] = {
                "source_raw_log": str(evidence.source_clean),
                "canonical_raw_log": str(evidence.canonical_clean),
                "sha256": evidence.sha256,
            }
        records.append(record)
    return records


def merge_crosshost_raw_logs(
    raw_log_root: str | Path,
    incoming_log_roots: Sequence[str | Path],
    manifest: str | Path,
    *,
    condition: str,
    runtime_profile: str,
    handoff_id: str,
    preflight_output: str | Path,
    handoff_output: str | Path,
    expected_base_model: str = raw_preflight.BASE_MODEL,
    expected_checkpoint: str | None = None,
    expected_max_connections: int | None = None,
) -> MergeResult:
    """Merge a complete locally staged split-host matrix, then run raw preflight.

    This operation only creates new transaction files and immutable JSON
    reports.  It never contacts a remote host, alters source files, rewrites
    EvalLogs, or overwrites a pre-existing destination.
    """

    condition = _validate_condition(condition)
    handoff_id = _safe_component(handoff_id, field="handoff_id")
    canonical_root = _as_root(raw_log_root, field="canonical raw-log root")
    incoming_roots = _non_nested_incoming_roots(canonical_root, incoming_log_roots)
    preflight_path = _report_destination(preflight_output, field="preflight output")
    handoff_path = _report_destination(handoff_output, field="handoff output")
    if preflight_path == handoff_path:
        raise ValueError("preflight_output and handoff_output must be different files")
    for output_path in (preflight_path, handoff_path):
        if any(output_path.is_relative_to(root) for root in incoming_roots):
            raise ValueError("cross-host reports must not be written inside an incoming handoff root")
    manifest_path = Path(manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage 2 OOD manifest does not exist: {manifest_path}")
    runtime = raw_preflight._validate_runtime_contract(
        runtime_profile=runtime_profile,
        expected_base_model=expected_base_model,
        expected_checkpoint=expected_checkpoint,
        expected_max_connections=expected_max_connections,
        require_vllm_adapter_attestation=True,
    )
    raw_preflight.validate_manifest(manifest_path)
    expected = raw_preflight._expected_cell_specs(ood_task_specs(manifest_path))

    canonical_candidates = _scan_successful_stage2_logs(
        canonical_root, canonical_root=canonical_root, expected=expected, runtime=runtime
    )
    incoming_candidates = [
        candidate
        for root in incoming_roots
        for candidate in _scan_successful_stage2_logs(
            root, canonical_root=canonical_root, expected=expected, runtime=runtime
        )
    ]
    canonical_by_identity = _by_identity(canonical_candidates)
    incoming_by_identity = _by_identity(incoming_candidates)
    canonical_clean = _canonical_clean_selection(canonical_by_identity, expected)

    incoming_selected = {
        identity: _select_incoming(identity, values) for identity, values in incoming_by_identity.items()
    }
    plans: list[ImportPlan] = []
    pair_evidence: dict[TaskIdentity, PairEvidence] = {}
    missing: list[str] = []
    for identity, spec in expected.items():
        canonical_values = canonical_by_identity.get(identity, ())
        incoming = incoming_selected.get(identity)
        if canonical_values:
            canonical_selected = _select_existing(identity, canonical_values)
            if incoming is not None and incoming.sha256 != canonical_selected.sha256:
                raise ValueError(
                    f"incoming EvalLog conflicts with the selected canonical task cell for "
                    f"{_identity_label(identity)}; it will not replace a retry"
                )
            if incoming is not None and spec.kind == "biased":
                clean = canonical_clean[(spec.population, spec.dataset)]
                pair_evidence[identity] = _source_clean_evidence(
                    incoming, canonical_root=canonical_root, canonical_clean=clean
                )
            continue
        if incoming is None:
            missing.append(_identity_label(identity))
            continue
        if spec.kind == "unbiased":
            # The clean precondition above normally catches this first.  Keep
            # an explicit branch so future task topologies cannot accidentally
            # start copying clean logs into a different absolute location.
            raise ValueError(f"refusing to relocate incoming clean Stage 2 EvalLog: {_identity_label(identity)}")
        clean = canonical_clean[(spec.population, spec.dataset)]
        evidence = _source_clean_evidence(incoming, canonical_root=canonical_root, canonical_clean=clean)
        pair_evidence[identity] = evidence
        # A hash-derived filename prevents source filenames from controlling a
        # destination path and makes an interrupted copy safely resumable.
        destination = (
            canonical_root / TRANSACTION_DIRECTORY / handoff_id / _identity_digest(identity) / f"{incoming.sha256}.eval"
        )
        plans.append(ImportPlan(candidate=incoming, destination=destination, pair_evidence=evidence))
    if missing:
        raise ValueError(f"Stage 2 cross-host source union is incomplete; missing={sorted(missing)}")

    # Rehash all non-copied pair evidence before creating any destination.  A
    # source staging directory changing during handoff is a hard failure, not
    # a reason to silently use a newer local retry.
    for evidence in pair_evidence.values():
        if _sha256_file(evidence.source_clean) != evidence.sha256:
            raise ValueError(f"incoming paired clean EvalLog changed after validation: {evidence.source_clean}")
        if _sha256_file(evidence.canonical_clean) != evidence.sha256:
            raise ValueError(f"canonical paired clean EvalLog changed after validation: {evidence.canonical_clean}")

    copy_statuses: list[tuple[Path, str]] = []
    transaction: Path | None = None
    if plans:
        transaction = _transaction_root(canonical_root, handoff_id)
    for plan in sorted(plans, key=lambda item: (str(item.destination), str(item.candidate.path))):
        assert transaction is not None  # plans is non-empty
        _ensure_transaction_destination_parent(transaction, plan.destination)
        copy_statuses.append(
            (plan.destination, _copy_new(plan.candidate.path, plan.destination, expected_sha256=plan.candidate.sha256))
        )

    report = raw_preflight.preflight_raw_logs(
        canonical_root,
        manifest_path,
        condition=condition,
        runtime_profile=runtime_profile,
        expected_base_model=expected_base_model,
        expected_checkpoint=expected_checkpoint,
        expected_max_connections=expected_max_connections,
    )
    preflight_status = raw_preflight.write_report(preflight_path, report)
    report_sources = {
        (
            source["kind"],
            source["regime"],
            source["population"],
            source["dataset"],
            source["bias_type"],
        ): source
        for source in report["sources"]
    }
    handoff = {
        "schema": HANDOFF_SCHEMA,
        "condition": condition,
        "handoff_id": handoff_id,
        "raw_root": str(canonical_root),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "incoming_roots": [str(root) for root in incoming_roots],
        "preflight": {
            "path": str(preflight_path),
            "sha256": _sha256_file(preflight_path),
            "schema": raw_preflight.PREFLIGHT_SCHEMA,
        },
        "incoming_sources": _incoming_receipt_sources(
            incoming_selected, report_sources=report_sources, pair_evidence=pair_evidence
        ),
    }
    handoff_status = write_handoff_report(handoff_path, handoff)
    return MergeResult(
        preflight_report=preflight_path,
        preflight_status=preflight_status,
        handoff_report=handoff_path,
        handoff_status=handoff_status,
        copy_statuses=tuple(copy_statuses),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-log-root", required=True, type=Path)
    parser.add_argument("--incoming-log-root", required=True, action="append", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--runtime-profile", required=True, choices=sorted(raw_preflight.RUNTIME_PROFILES))
    parser.add_argument("--expected-base-model", default=raw_preflight.BASE_MODEL)
    parser.add_argument("--expected-checkpoint")
    parser.add_argument("--expected-max-connections", type=int)
    parser.add_argument("--handoff-id", required=True)
    parser.add_argument("--preflight-output", required=True, type=Path)
    parser.add_argument("--handoff-output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = merge_crosshost_raw_logs(
            args.raw_log_root,
            args.incoming_log_root,
            args.manifest,
            condition=args.condition,
            runtime_profile=args.runtime_profile,
            handoff_id=args.handoff_id,
            preflight_output=args.preflight_output,
            handoff_output=args.handoff_output,
            expected_base_model=args.expected_base_model,
            expected_checkpoint=args.expected_checkpoint,
            expected_max_connections=args.expected_max_connections,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    counts = {
        status: sum(1 for _, observed in result.copy_statuses if observed == status)
        for status in {item[1] for item in result.copy_statuses}
    }
    print(
        f"Stage 2 cross-host merge complete: preflight={result.preflight_status} {result.preflight_report}; "
        f"handoff={result.handoff_status} {result.handoff_report}; copies={counts}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
