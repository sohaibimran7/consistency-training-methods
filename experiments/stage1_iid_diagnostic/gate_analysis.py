"""Deterministic raw-log gate for the ACT native/canonical prompt repair.

This module intentionally reads only the local paired-switch scores already
stored in successful Inspect ``.eval`` logs.  It never instantiates a model,
calls an acknowledgement grader, or imports the Luna grading pipeline.

The two input roots are deliberately separate.  Each must contain exactly one
successful ``stage1_iid_biased`` log for each of the two frozen diagnostic
splits and two source datasets.  A biased log already contains the matched
clean/biased switch score, so clean-only logs are not inputs to this gate.

For example::

    python -m experiments.stage1_iid_diagnostic.gate_analysis \
      --native-root logs/act-gate/native \
      --canonical-root logs/act-gate/canonical \
      --output artifacts/stage1-iid-act-gate/analysis.json

The result is content-addressed in practice: an identical rerun resumes, and
the command refuses to overwrite a different report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

ANALYSIS_SCHEMA = "stage1-iid-act-gate-analysis-v1"
VARIANTS = ("native", "canonical")
SPLITS = ("train_eval", "heldout_in_domain")
DATASETS = ("logiqa", "hellaswag")
BIASED_TASK = "stage1_iid_biased"
BIAS_TYPE = "wrong_argument"
SWITCH_METRICS = (
    "unbiased_matches_bias",
    "towards_bias_switch",
    "away_from_bias_switch",
    "net_switch",
    "abs_switch",
)
_MISSING = object()


@dataclass(frozen=True, slots=True)
class Observation:
    """One local clean/biased outcome from a raw biased Inspect sample."""

    prompt_variant: str
    split: str
    dataset: str
    question_id: str
    joint_parse: bool
    clean_matches_bias: int | None
    toward_bias_switch: int | None


@dataclass(frozen=True, slots=True)
class LogHeader:
    """Pinned provenance needed to prove one raw gate cell is comparable."""

    prompt_variant: str
    split: str
    dataset: str
    question_ids: tuple[str, ...]
    variant_file: str | None
    unbiased_log: str
    prompt_style: str
    source_identity_digest: str | None
    created: str


@dataclass(frozen=True, slots=True)
class LoadedLog:
    header: LogHeader
    path: Path
    sha256: str
    rows: tuple[Observation, ...]


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _task_basename(value: Any) -> str:
    return str(value or "").rsplit("@", 1)[-1].rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    payload = json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _local_log_path(value: Any) -> Path:
    """Normalize local Inspect log names across path and file-URI releases."""

    raw = str(value if isinstance(value, (str, os.PathLike)) else _attribute(value, "name", value))
    parsed = urlsplit(raw)
    if parsed.scheme:
        if parsed.scheme.lower() != "file":
            raise ValueError(f"ACT gate accepts only local Inspect logs, got URI scheme {parsed.scheme!r}: {raw}")
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"ACT gate does not support remote file authorities: {raw}")
        if parsed.query or parsed.fragment:
            raise ValueError(f"ACT gate log URI must not contain a query or fragment: {raw}")
        decoded = unquote(parsed.path)
        if not decoded:
            raise ValueError(f"ACT gate log URI has no path: {raw}")
        path = Path(decoded)
        if not path.is_absolute():
            raise ValueError(f"ACT gate file URI must contain an absolute path: {raw}")
        return path.resolve()
    return Path(raw).resolve()


def _header_value(
    evaluation: Any,
    *names: str,
    field: str,
    required: bool = True,
) -> Any:
    """Read a value duplicated between Inspect task args and task metadata.

    The diagnostic task records most provenance in both locations.  Accepting
    either preserves compatibility across Inspect releases; accepting a
    disagreement would make native/canonical comparisons unauditable.
    """

    task_args = _mapping(_attribute(evaluation, "task_args", {}))
    metadata = _mapping(_attribute(evaluation, "metadata", {}))
    values: list[Any] = []
    for mapping in (task_args, metadata):
        for name in names:
            if name in mapping:
                values.append(mapping[name])
    if not values:
        if required:
            joined = "/".join(names)
            raise ValueError(f"Inspect header is missing required {field} ({joined})")
        return _MISSING
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"Inspect header has conflicting {field}: {values!r}")
    return first


def _header(log: Any, *, prompt_variant: str, path: Path) -> LogHeader:
    if prompt_variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant: {prompt_variant!r}")
    if _attribute(log, "status") != "success":
        raise ValueError(f"ACT gate log is not successful: {path}")
    evaluation = _attribute(log, "eval")
    if _task_basename(_attribute(evaluation, "task")) != BIASED_TASK:
        raise ValueError(f"ACT gate log is not a {BIASED_TASK!r} task: {path}")

    dataset = _header_value(evaluation, "dataset", "source_dataset", field="dataset")
    split = _header_value(evaluation, "split", field="split")
    bias_type = _header_value(evaluation, "bias_type", field="bias_type")
    question_ids = _header_value(evaluation, "question_ids_from", field="question_ids_from")
    # Inspect omits a task's default-valued keyword from some historical native
    # EvalLog headers.  Its absence is therefore equivalent to the native
    # default (null), while a canonical comparison must record a concrete
    # alternate file explicitly.
    variant_file = _header_value(evaluation, "variant_file", field="variant_file", required=False)
    unbiased_log = _header_value(evaluation, "unbiased_log", field="unbiased_log")
    prompt_style = _header_value(evaluation, "prompt_style", field="prompt_style")
    source_identity = _header_value(
        evaluation,
        "source_identity_digest",
        "source_sha256",
        field="source identity digest",
        required=False,
    )
    acknowledgement = _header_value(
        evaluation,
        "include_bias_acknowledged",
        field="include_bias_acknowledged",
        required=False,
    )
    grader_model = _header_value(evaluation, "grader_model", field="grader_model", required=False)

    if dataset not in DATASETS or split not in SPLITS:
        raise ValueError(f"ACT gate header has invalid dataset/split in {path}: {dataset!r}/{split!r}")
    if bias_type != BIAS_TYPE:
        raise ValueError(f"ACT gate header must use bias_type={BIAS_TYPE!r} in {path}")
    if acknowledgement is not _MISSING and acknowledgement is not False:
        raise ValueError(f"ACT gate raw log unexpectedly enables acknowledgement grading: {path}")
    if grader_model is not _MISSING and grader_model is not None:
        raise ValueError(f"ACT gate raw log unexpectedly pins a grader model: {path}")
    if not isinstance(unbiased_log, str) or not unbiased_log:
        raise ValueError(f"ACT gate header has no clean-pair log reference: {path}")
    if not isinstance(prompt_style, str) or not prompt_style:
        raise ValueError(f"ACT gate header has no prompt_style: {path}")
    if source_identity is not _MISSING and source_identity is not None and not isinstance(source_identity, str):
        raise ValueError(f"ACT gate source identity is not a string in {path}")
    if (
        not isinstance(question_ids, list)
        or not question_ids
        or any(not isinstance(item, str) or not item for item in question_ids)
    ):
        raise ValueError(f"ACT gate header has invalid question_ids_from in {path}")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"ACT gate header has duplicate question_ids_from in {path}")

    if prompt_variant == "native":
        if variant_file is not _MISSING and variant_file is not None:
            raise ValueError(f"native ACT gate log must have variant_file=null: {path}")
        normalized_variant_file = None
    else:
        if not isinstance(variant_file, str) or not variant_file:
            raise ValueError(f"canonical ACT gate log must have a non-empty variant_file: {path}")
        normalized_variant_file = variant_file

    return LogHeader(
        prompt_variant=prompt_variant,
        split=split,
        dataset=dataset,
        question_ids=tuple(question_ids),
        variant_file=normalized_variant_file,
        unbiased_log=unbiased_log,
        prompt_style=prompt_style,
        source_identity_digest=source_identity if isinstance(source_identity, str) else None,
        created=str(_attribute(evaluation, "created", "") or ""),
    )


def _missing_number(value: Any) -> Any:
    """Normalize Inspect's NaN null sentinel and legacy ``None`` to one value."""

    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
        return None
    return value


def _binary(value: Any, *, field: str, allow_missing: bool = False) -> int | None:
    value = _missing_number(value)
    if value is None and allow_missing:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) in {0.0, 1.0}:
        return int(value)
    suffix = " or missing" if allow_missing else ""
    raise ValueError(f"{field} must be binary{suffix}; got {value!r}")


def _switch_mapping(sample: Any) -> Mapping[str, Any]:
    """Return exactly one complete paired-switch score mapping for a sample."""

    candidates: list[Mapping[str, Any]] = []
    for score in _mapping(_attribute(sample, "scores", {})).values():
        value = _attribute(score, "value")
        if isinstance(value, Mapping) and any(metric in value for metric in SWITCH_METRICS):
            candidates.append(value)
    sample_id = _attribute(sample, "id", "<unknown>")
    if len(candidates) != 1:
        raise ValueError(f"sample {sample_id!r} must contain exactly one paired-switch score mapping")
    values = candidates[0]
    missing = [metric for metric in SWITCH_METRICS if metric not in values]
    if missing:
        raise ValueError(f"sample {sample_id!r} paired-switch score is missing {missing}")
    return values


def _paired_outcome(values: Mapping[str, Any], *, sample_id: str) -> tuple[bool, int | None, int | None]:
    """Validate the conditional switch representation and return gate fields."""

    clean_raw = _missing_number(values["unbiased_matches_bias"])
    toward_raw = _missing_number(values["towards_bias_switch"])
    away_raw = _missing_number(values["away_from_bias_switch"])
    net_raw = _missing_number(values["net_switch"])
    total_raw = _missing_number(values["abs_switch"])
    if net_raw is None or total_raw is None:
        if any(value is not None for value in (clean_raw, toward_raw, away_raw, net_raw, total_raw)):
            raise ValueError(f"sample {sample_id!r} has a partial paired parse failure")
        return False, None, None

    clean = _binary(clean_raw, field="unbiased_matches_bias")
    assert clean is not None
    if clean == 0:
        toward = _binary(toward_raw, field="towards_bias_switch")
        away = 0 if away_raw is None else _binary(away_raw, field="away_from_bias_switch")
    else:
        toward = 0 if toward_raw is None else _binary(toward_raw, field="towards_bias_switch")
        away = _binary(away_raw, field="away_from_bias_switch")
    total = _binary(total_raw, field="abs_switch")
    if (
        isinstance(net_raw, bool)
        or not isinstance(net_raw, (int, float))
        or not math.isfinite(float(net_raw))
        or float(net_raw) not in {-1.0, 0.0, 1.0}
    ):
        raise ValueError(f"sample {sample_id!r} has invalid net_switch {net_raw!r}")
    assert toward is not None and away is not None and total is not None
    if int(net_raw) != toward - away or total != toward + away:
        raise ValueError(f"sample {sample_id!r} has incoherent paired switch scores")
    if (toward and clean) or (away and not clean):
        raise ValueError(f"sample {sample_id!r} violates conditional switch eligibility")
    return True, clean, toward


def observations_from_raw_log(log: Any, header: LogHeader) -> tuple[Observation, ...]:
    """Extract local outcomes and prove the log covers its pinned sample IDs."""

    if _attribute(log, "status") != "success":
        raise ValueError("ACT gate full EvalLog is not successful")
    samples = list(_attribute(log, "samples", []) or [])
    if not samples:
        raise ValueError("ACT gate EvalLog has no samples")
    rows: list[Observation] = []
    seen: set[str] = set()
    for sample in samples:
        sample_id = _attribute(sample, "id", "")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"ACT gate log has missing or duplicate question_id {sample_id!r}")
        seen.add(sample_id)
        metadata = _mapping(_attribute(sample, "metadata", {}))
        if metadata.get("variant") != "biased":
            raise ValueError(f"sample {sample_id!r} is not a biased prompt")
        if metadata.get("source_dataset") != header.dataset:
            raise ValueError(f"sample {sample_id!r} dataset conflicts with its header")
        if metadata.get("bias_type") != BIAS_TYPE:
            raise ValueError(f"sample {sample_id!r} bias type conflicts with its header")
        if metadata.get("prompt_style") != header.prompt_style:
            raise ValueError(f"sample {sample_id!r} prompt style conflicts with its header")
        joint_parse, clean_matches_bias, toward_bias_switch = _paired_outcome(
            _switch_mapping(sample), sample_id=sample_id
        )
        rows.append(
            Observation(
                prompt_variant=header.prompt_variant,
                split=header.split,
                dataset=header.dataset,
                question_id=sample_id,
                joint_parse=joint_parse,
                clean_matches_bias=clean_matches_bias,
                toward_bias_switch=toward_bias_switch,
            )
        )
    expected = set(header.question_ids)
    if seen != expected:
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        raise ValueError(
            f"ACT gate log sample IDs do not match question_ids_from; "
            f"missing={missing[:3]!r} ({len(missing)}), unexpected={unexpected[:3]!r} ({len(unexpected)})"
        )
    return tuple(rows)


def _discover_paths(root: str | Path) -> list[Path]:
    directory = Path(root).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"ACT gate raw-log root does not exist: {directory}")
    try:
        from inspect_ai.log import list_eval_logs
    except ImportError as exc:  # pragma: no cover - only the configured evaluation environment needs Inspect
        raise RuntimeError("Inspect AI is required to read raw ACT gate logs") from exc
    paths = {_local_log_path(item) for item in list_eval_logs(str(directory), formats=["eval"], recursive=True)}
    if not paths:
        raise FileNotFoundError(f"no Inspect .eval logs found under {directory}")
    return sorted(paths)


def scan_variant_logs(root: str | Path, *, prompt_variant: str) -> dict[tuple[str, str], LoadedLog]:
    """Select exactly one successful biased log per split/dataset gate cell."""

    if prompt_variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant: {prompt_variant!r}")
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - only the configured evaluation environment needs Inspect
        raise RuntimeError("Inspect AI is required to read raw ACT gate logs") from exc

    found: dict[tuple[str, str], LoadedLog] = {}
    for path in _discover_paths(root):
        try:
            raw_header = read_eval_log(str(path), header_only=True)
        except Exception as exc:
            raise ValueError(f"could not read Inspect header for ACT gate log {path}") from exc
        evaluation = _attribute(raw_header, "eval")
        if (
            _task_basename(_attribute(evaluation, "task")) != BIASED_TASK
            or _attribute(raw_header, "status") != "success"
        ):
            # Clean logs and failed retries are not evidence for this gate.
            continue
        header = _header(raw_header, prompt_variant=prompt_variant, path=path)
        key = (header.split, header.dataset)
        if key in found:
            raise ValueError(
                f"duplicate successful ACT gate log for {prompt_variant}/{header.split}/{header.dataset}: "
                f"{found[key].path} and {path}"
            )
        try:
            full_log = read_eval_log(str(path))
        except Exception as exc:
            raise ValueError(f"could not read full Inspect ACT gate log {path}") from exc
        rows = observations_from_raw_log(full_log, header)
        found[key] = LoadedLog(header=header, path=path, sha256=_sha256(path), rows=rows)

    expected = {(split, dataset) for split in SPLITS for dataset in DATASETS}
    missing = sorted(expected - set(found))
    unexpected = sorted(set(found) - expected)
    if missing or unexpected:
        raise ValueError(f"{prompt_variant} ACT gate logs are incomplete; missing={missing}, unexpected={unexpected}")
    return found


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def summarize(rows: Sequence[Observation]) -> dict[str, Any]:
    """Compute parser coverage, eligibility coverage, and conditional TBSR."""

    parsed = [row for row in rows if row.joint_parse]
    eligible = [row for row in parsed if row.clean_matches_bias == 0]
    ineligible = [row for row in parsed if row.clean_matches_bias == 1]
    toward = [row for row in eligible if row.toward_bias_switch == 1]
    return {
        "counts": {
            "samples": len(rows),
            "joint_parsed": len(parsed),
            "joint_parse_failures": len(rows) - len(parsed),
            "eligible_clean_answer_not_bias_answer": len(eligible),
            "ineligible_clean_answer_equals_bias_answer": len(ineligible),
            "toward_bias_switches": len(toward),
        },
        "rates": {
            "parser_coverage": _rate(len(parsed), len(rows)),
            "eligible_coverage": _rate(len(eligible), len(parsed)),
            "tbsr": _rate(len(toward), len(eligible)),
        },
    }


def _validate_comparability(loaded: Mapping[str, Mapping[tuple[str, str], LoadedLog]]) -> None:
    """Reject prompt variants that do not represent the same frozen comparison."""

    prompt_styles: set[str] = set()
    source_identities: set[str] = set()
    for split in SPLITS:
        for dataset in DATASETS:
            native = loaded["native"][(split, dataset)]
            canonical = loaded["canonical"][(split, dataset)]
            if native.header.question_ids != canonical.header.question_ids:
                raise ValueError(f"native/canonical question_ids_from differ for {split}/{dataset}")
            prompt_styles.update((native.header.prompt_style, canonical.header.prompt_style))
            for item in (native, canonical):
                if item.header.source_identity_digest is not None:
                    source_identities.add(item.header.source_identity_digest)

    if len(prompt_styles) != 1:
        raise ValueError(f"ACT gate prompt_style differs across cells: {sorted(prompt_styles)!r}")
    if len(source_identities) > 1:
        raise ValueError("ACT gate source identity digest differs across native/canonical cells")
    for prompt_variant in VARIANTS:
        train_ids = set().union(
            *(set(loaded[prompt_variant][("train_eval", dataset)].header.question_ids) for dataset in DATASETS)
        )
        heldout_ids = set().union(
            *(set(loaded[prompt_variant][("heldout_in_domain", dataset)].header.question_ids) for dataset in DATASETS)
        )
        overlap = sorted(train_ids & heldout_ids)
        if overlap:
            raise ValueError(f"{prompt_variant} ACT gate training and held-out samples overlap, first={overlap[0]!r}")


def build_report(
    native_root: str | Path,
    canonical_root: str | Path,
    *,
    condition: str = "act",
) -> dict[str, Any]:
    """Build a raw, local-only ACT prompt-repair decision report."""

    if not isinstance(condition, str) or not condition:
        raise ValueError("condition must be a non-empty string")
    loaded = {
        "native": scan_variant_logs(native_root, prompt_variant="native"),
        "canonical": scan_variant_logs(canonical_root, prompt_variant="canonical"),
    }
    _validate_comparability(loaded)

    cells: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    for prompt_variant in VARIANTS:
        for split in SPLITS:
            pooled_rows: list[Observation] = []
            per_dataset: dict[str, Any] = {}
            for dataset in DATASETS:
                item = loaded[prompt_variant][(split, dataset)]
                pooled_rows.extend(item.rows)
                per_dataset[dataset] = summarize(item.rows)
                sources.append(
                    {
                        "prompt_variant": prompt_variant,
                        "split": split,
                        "dataset": dataset,
                        "raw_log": str(item.path),
                        "raw_log_sha256": item.sha256,
                        "samples": len(item.rows),
                        "question_ids_sha256": _ids_sha256(item.header.question_ids),
                        "unbiased_log": item.header.unbiased_log,
                        "variant_file": item.header.variant_file,
                        "prompt_style": item.header.prompt_style,
                        "source_identity_digest": item.header.source_identity_digest,
                        "created": item.header.created,
                    }
                )
            cells[f"{prompt_variant}/{split}"] = {
                "condition": condition,
                "prompt_variant": prompt_variant,
                "split": split,
                "pooled": summarize(pooled_rows),
                "per_dataset": per_dataset,
            }

    return {
        "schema": ANALYSIS_SCHEMA,
        "condition": condition,
        "analysis_mode": "raw_local_paired_switch_scores_only",
        "metric_definitions": {
            "parser_coverage": "P(clean and biased answers jointly parsed)",
            "eligible_coverage": "P(clean answer != bias answer | jointly parsed)",
            "tbsr": "P(biased answer = bias answer | clean answer != bias answer, jointly parsed)",
        },
        "sources": sources,
        "cells": cells,
    }


def write_report(path: str | Path, report: Mapping[str, Any]) -> str:
    """Atomically publish an immutable JSON report, or resume byte-identically."""

    destination = Path(path).resolve()
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == payload:
            return "resumed"
        raise FileExistsError(f"refusing to overwrite differing ACT gate report: {destination}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise FileExistsError(f"ACT gate report appeared and differs: {destination}")
            return "resumed"
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return "written"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", required=True, type=Path)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--condition", default="act")
    args = parser.parse_args(argv)
    try:
        report = build_report(args.native_root, args.canonical_root, condition=args.condition)
        status = write_report(args.output, report)
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{status}: {args.output.resolve()}")


if __name__ == "__main__":
    main()
