"""Compute descriptive conditional towards-bias switch rates from local Inspect logs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.rmct_tbsr.constants import (
    HLE_BIASES,
    HLE_DATASET,
    HLE_FILE_SHA256,
    HLE_FILES,
    HLE_PAPER_HELD_OUT_BIASES,
    HLE_ROWS,
    HLE_SPLIT,
    MODELS,
    TRAINING_COUNTS,
    TRAINING_DATASETS,
    TRAINING_SPLIT,
)
from experiments.rmct_tbsr.prepare import TrainingArtifact, validate_training_manifest
from experiments.switch_gate.analyze import Observation, observations_from_log, validate_hle_clusters

SCHEMA_VERSION = "rmct-tbsr-analysis-v1"


@dataclass(frozen=True, slots=True)
class CellSpec:
    scope: str
    split: str
    dataset: str
    bias_type: str
    expected_samples: int


CELL_SPECS = (
    *(
        CellSpec("train", TRAINING_SPLIT, dataset, "wrong_argument", TRAINING_COUNTS[dataset])
        for dataset in TRAINING_DATASETS
    ),
    *(CellSpec("test", HLE_SPLIT, HLE_DATASET, bias, HLE_ROWS) for bias in HLE_BIASES),
)
CELL_BY_HEADER = {(spec.split, spec.dataset, spec.bias_type): spec for spec in CELL_SPECS}

CellKey = tuple[str, str, str, str]


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _model_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("name", value.get("model", "")))
    return str(_attribute(value, "name", value) or "")


def model_matches(expected: str, actual: str) -> bool:
    """Accept an exact model or the same exact identity with a provider prefix."""

    if actual == expected:
        return True
    suffix = f"/{expected}"
    return actual.endswith(suffix) and bool(actual[: -len(suffix)].strip("/"))


def _json_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _file_identity(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    payload = path.read_bytes()
    identity: dict[str, Any] = {
        "path": str(path.resolve()),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    if rows is not None:
        identity["rows"] = rows
    return identity


def _discover_log_paths(location: str) -> list[Path]:
    path = Path(location)
    if path.is_file():
        return [path.resolve()]
    if not path.exists() and glob.has_magic(location):
        matches = [Path(match).resolve() for match in glob.glob(location, recursive=True) if Path(match).is_file()]
        if matches:
            return sorted(set(matches))
    try:
        from inspect_ai.log import list_eval_logs
    except ImportError as exc:  # pragma: no cover - requires configured eval environment
        raise RuntimeError("Inspect AI is required to read .eval logs") from exc
    infos = list_eval_logs(location)
    paths = [Path(str(_attribute(info, "name", info))).resolve() for info in infos]
    if not paths:
        raise FileNotFoundError(f"no Inspect logs found at {location!r}")
    if any(not candidate.is_file() for candidate in paths):
        raise ValueError("RMCT TBSR analysis requires local filesystem Inspect logs")
    return sorted(set(paths))


def _task_basename(task_name: str) -> str:
    return task_name.rsplit("@", 1)[-1].rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def load_inspect_cells(
    model_locations: Mapping[str, str],
    *,
    expected_files: Mapping[tuple[str, str, str], Path],
) -> tuple[dict[CellKey, list[Observation]], list[dict[str, Any]]]:
    """Select the latest successful exact biased task for every requested cell."""

    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - requires configured eval environment
        raise RuntimeError("Inspect AI is required to read .eval logs") from exc

    identities: list[dict[str, Any]] = []
    selected: dict[CellKey, tuple[str, Path, dict[str, Any]]] = {}
    for model, location in model_locations.items():
        for path in _discover_log_paths(location):
            file_identity = _file_identity(path)
            log = read_eval_log(str(path), header_only=True)
            evaluation = _attribute(log, "eval")
            args = _mapping(_attribute(evaluation, "task_args", {}))
            actual_model = _model_name(_attribute(evaluation, "model", ""))
            if not model_matches(model, actual_model):
                continue
            identity = {
                "model_label": model,
                "log_model": actual_model,
                **file_identity,
                "status": str(_attribute(log, "status", "")),
                "created": str(_attribute(evaluation, "created", "")),
                "task": str(_attribute(evaluation, "task", "")),
                "task_args": dict(sorted((str(key), value) for key, value in args.items() if _json_scalar(value))),
                "selected": False,
                "selection_reason": "not_a_successful_exact_rmct_biased_cell",
            }
            identities.append(identity)
            if identity["status"] != "success" or _task_basename(identity["task"]) != "switch_gate_biased":
                continue
            dataset = str(args.get("source_dataset", args.get("dataset", "")))
            bias_type = str(args.get("bias_type", ""))
            split = str(args.get("split", ""))
            prompt_style = str(args.get("prompt_style", "none"))
            spec = CELL_BY_HEADER.get((split, dataset, bias_type))
            if spec is None or prompt_style != "none":
                continue
            expected_file = expected_files.get((split, dataset, bias_type))
            if expected_file is None:
                raise ValueError(f"no exact frozen-file identity configured for {(split, dataset, bias_type)}")
            declared_files = [args[name] for name in ("frozen_file", "dataset_file") if args.get(name)]
            if not declared_files or any(
                not isinstance(declared, str) or Path(declared).resolve() != expected_file.resolve()
                for declared in declared_files
            ):
                identity["selection_reason"] = "frozen_file_does_not_match_exact_cell_artifact"
                continue
            created = identity["created"]
            if not created:
                raise ValueError(f"successful candidate log has no created timestamp: {path}")
            key = (model, split, dataset, bias_type)
            previous = selected.get(key)
            if previous is not None and created == previous[0]:
                raise ValueError(f"ambiguous successful logs with the same timestamp for {key}")
            identity["selection_reason"] = "candidate"
            if previous is None or created > previous[0]:
                if previous is not None:
                    previous[2]["selection_reason"] = "superseded_by_later_success"
                selected[key] = (created, path, identity)
            else:
                identity["selection_reason"] = "superseded_by_later_success"

    records: dict[CellKey, list[Observation]] = {}
    for key, (_, path, identity) in selected.items():
        identity["selected"] = True
        identity["selection_reason"] = "latest_successful_exact_cell"
        log = read_eval_log(str(path))
        rows = observations_from_log(log, expected_model=key[0])
        if any(
            row.source_dataset != key[2]
            or row.bias_type != key[3]
            or row.prompt_style != "none"
            or row.variant != "biased"
            for row in rows
        ):
            raise ValueError(f"selected log samples conflict with selected cell {key}")
        records[key] = rows

    model_order = {model: index for index, model in enumerate(MODELS)}
    identities.sort(key=lambda item: (model_order[item["model_label"]], item["path"]))
    return records, identities


def load_expected_hle(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[set[str], dict[str, Any]]:
    """Verify the exact common clean HLE n=100 population and return its IDs."""

    expected_path = Path(path)
    if not expected_path.is_file():
        raise FileNotFoundError(f"expected HLE file does not exist: {expected_path}")
    payload = expected_path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    required_hash = HLE_FILE_SHA256["unbiased"] if expected_sha256 is None else expected_sha256
    if actual_hash != required_hash:
        raise ValueError(f"{expected_path}: SHA-256 does not match the exact RMCT clean HLE input")
    ids: list[str] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{expected_path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"{expected_path}:{line_number}: row must be a JSON object")
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{expected_path}:{line_number}: question_id must be non-empty")
        if row.get("source_dataset") != HLE_DATASET or row.get("prompt_style") != "none":
            raise ValueError(f"{expected_path}:{line_number}: unexpected HLE dataset or prompt style")
        ids.append(question_id)
    if len(ids) != HLE_ROWS or len(set(ids)) != HLE_ROWS:
        raise ValueError(f"{expected_path}: expected exactly {HLE_ROWS} unique HLE question IDs")
    return set(ids), {**_file_identity(expected_path, rows=HLE_ROWS), "role": "expected_clean_hle"}


def load_expected_hle_bias_files(clean_path: str | Path) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Verify all six biased HLE artifacts adjacent to the exact clean file."""

    root = Path(clean_path).resolve().parent
    paths: dict[str, Path] = {}
    identities: list[dict[str, Any]] = []
    for bias in HLE_BIASES:
        path = root / HLE_FILES[bias]
        if not path.is_file():
            raise FileNotFoundError(f"exact RMCT HLE {bias!r} file does not exist: {path}")
        identity = _file_identity(path, rows=HLE_ROWS)
        if identity["content_sha256"] != HLE_FILE_SHA256[bias]:
            raise ValueError(f"{path}: SHA-256 does not match the exact RMCT HLE {bias!r} input")
        paths[bias] = path
        identities.append({"role": f"expected_hle_{bias}", **identity})
    return paths, identities


def _expected_training_ids(artifact: TrainingArtifact) -> dict[str, set[str]]:
    return {
        dataset: {row["question_id"] for row in artifact.rows if row["source_dataset"] == dataset}
        for dataset in TRAINING_DATASETS
    }


def _count_outcomes(records: Sequence[Observation]) -> dict[str, int | float | None]:
    parsed = [record for record in records if record.joint_parse]
    clean_target = sum(record.clean_target == 1 for record in parsed)
    clean_non_target = sum(record.clean_target == 0 for record in parsed)
    toward = sum(int(record.toward or 0) for record in parsed)
    away = sum(int(record.away or 0) for record in parsed)
    net = sum(int(record.net_switch or 0) for record in parsed)
    absolute = sum(int(record.abs_switch or 0) for record in parsed)
    if clean_target + clean_non_target != len(parsed):
        raise AssertionError("joint parses must partition into clean-target and clean-non-target outcomes")
    if net != toward - away or absolute != toward + away:
        raise AssertionError("switch score contract failed after strict extraction")
    return {
        "observed_samples": len(records),
        "joint_parse_numerator": len(parsed),
        "joint_parse_total": len(records),
        "joint_parse_rate": len(parsed) / len(records) if records else None,
        "missing_joint_parse": len(records) - len(parsed),
        "clean_target": clean_target,
        "clean_non_target": clean_non_target,
        "toward": toward,
        "away": away,
        "net_switch_sum": net,
        "target_status_switch": absolute,
        "tbsr_numerator_toward": toward,
        "tbsr_denominator_clean_non_target": clean_non_target,
        "tbsr": toward / clean_non_target if clean_non_target else None,
    }


def _cell_row(
    model: str,
    spec: CellSpec,
    records: Sequence[Observation],
    expected_ids: set[str],
    selected_log: Mapping[str, Any] | None,
) -> dict[str, Any]:
    actual_ids = [record.question_id for record in records]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError(f"duplicate observed question ID for {model}/{spec.dataset}/{spec.bias_type}")
    extra = sorted(set(actual_ids) - expected_ids)
    if extra:
        raise ValueError(
            f"{model}/{spec.dataset}/{spec.bias_type} has IDs outside the exact frozen population: {extra[:5]}"
        )
    missing = sorted(expected_ids - set(actual_ids))
    if not records:
        data_status = "missing"
    elif missing:
        data_status = "partial"
    else:
        data_status = "complete"
    counts = _count_outcomes(records)
    return {
        "model": model,
        "scope": spec.scope,
        "split": spec.split,
        "dataset": spec.dataset,
        "bias_type": spec.bias_type,
        "data_status": data_status,
        "expected_samples": spec.expected_samples,
        "observed_samples": counts["observed_samples"],
        "id_validation": {
            "exact": not missing and not extra,
            "missing_count": len(missing),
            "missing_examples": missing[:10],
            "extra_count": 0,
        },
        "tbsr": {
            "numerator_toward": counts["tbsr_numerator_toward"],
            "eligible_clean_non_target": counts["tbsr_denominator_clean_non_target"],
            "rate": counts["tbsr"],
        },
        "joint_parse": {
            "numerator": counts["joint_parse_numerator"],
            "total": counts["joint_parse_total"],
            "rate": counts["joint_parse_rate"],
        },
        "counts": {
            "missing_joint_parse": counts["missing_joint_parse"],
            "clean_target": counts["clean_target"],
            "clean_non_target": counts["clean_non_target"],
            "toward": counts["toward"],
            "away": counts["away"],
            "net_switch_sum": counts["net_switch_sum"],
            "target_status_switch": counts["target_status_switch"],
        },
        "selected_log": (
            {key: selected_log[key] for key in ("path", "content_sha256", "bytes", "created", "log_model", "task")}
            if selected_log is not None
            else None
        ),
    }


def _micro_summary(
    model: str,
    label: str,
    rows: Sequence[dict[str, Any]],
    records: Sequence[Observation],
    *,
    included_biases: Sequence[str],
) -> dict[str, Any]:
    statuses = [row["data_status"] for row in rows]
    if statuses and all(status == "complete" for status in statuses):
        data_status = "complete"
    elif records:
        data_status = "partial"
    else:
        data_status = "missing"
    return {
        "model": model,
        "summary": label,
        "pooling": "sample-count micro pool over biased-question observations",
        "descriptive_only": True,
        "data_status": data_status,
        "included_biases": list(included_biases),
        "included_cells": [
            {"scope": row["scope"], "dataset": row["dataset"], "bias_type": row["bias_type"]} for row in rows
        ],
        "expected_samples": sum(int(row["expected_samples"]) for row in rows),
        **_count_outcomes(records),
    }


def build_report(
    records_by_cell: Mapping[CellKey, Sequence[Observation]],
    *,
    expected_training: Mapping[str, set[str]],
    expected_hle: set[str],
    input_logs: Sequence[Mapping[str, Any]] = (),
    input_artifacts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the complete 24-row descriptive matrix, including missing cells."""

    expected_counts = {dataset: len(expected_training.get(dataset, set())) for dataset in TRAINING_DATASETS}
    if expected_counts != TRAINING_COUNTS:
        raise ValueError(f"expected training IDs must have exact counts {TRAINING_COUNTS}, got {expected_counts}")
    if len(expected_hle) != HLE_ROWS:
        raise ValueError(f"expected HLE IDs must contain exactly {HLE_ROWS} IDs")
    known_keys = {(model, spec.split, spec.dataset, spec.bias_type) for model in MODELS for spec in CELL_SPECS}
    unknown = sorted(key for key, records in records_by_cell.items() if records and key not in known_keys)
    if unknown:
        raise ValueError(f"observations contain unknown RMCT TBSR cells: {unknown[:5]}")

    selected_logs = {
        (
            str(identity["model_label"]),
            str(_mapping(identity.get("task_args", {})).get("split", "")),
            str(
                _mapping(identity.get("task_args", {})).get(
                    "source_dataset", _mapping(identity.get("task_args", {})).get("dataset", "")
                )
            ),
            str(_mapping(identity.get("task_args", {})).get("bias_type", "")),
        ): identity
        for identity in input_logs
        if identity.get("selected")
    }

    rows: list[dict[str, Any]] = []
    records_by_model: dict[str, list[Observation]] = {model: [] for model in MODELS}
    for model in MODELS:
        for spec in CELL_SPECS:
            key = (model, spec.split, spec.dataset, spec.bias_type)
            records = list(records_by_cell.get(key, ()))
            if any(record.model != model for record in records):
                raise ValueError(f"observation model conflicts with cell key {key}")
            expected_ids = expected_training[spec.dataset] if spec.scope == "train" else expected_hle
            rows.append(_cell_row(model, spec, records, expected_ids, selected_logs.get(key)))
            records_by_model[model].extend(records)
        validate_hle_clusters([record for record in records_by_model[model] if record.source_dataset == HLE_DATASET])

    summaries: list[dict[str, Any]] = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        training_rows = [row for row in model_rows if row["scope"] == "train"]
        training_records = [
            record
            for spec in CELL_SPECS
            if spec.scope == "train"
            for record in records_by_cell.get((model, spec.split, spec.dataset, spec.bias_type), ())
        ]
        summaries.append(
            _micro_summary(
                model,
                "pooled_training_wrong_argument",
                training_rows,
                training_records,
                included_biases=("wrong_argument",),
            )
        )
        held_out_rows = [
            row for row in model_rows if row["scope"] == "test" and row["bias_type"] in HLE_PAPER_HELD_OUT_BIASES
        ]
        held_out_records = [
            record
            for spec in CELL_SPECS
            if spec.scope == "test" and spec.bias_type in HLE_PAPER_HELD_OUT_BIASES
            for record in records_by_cell.get((model, spec.split, spec.dataset, spec.bias_type), ())
        ]
        summaries.append(
            _micro_summary(
                model,
                "paper_held_out_hle_excluding_wrong_argument",
                held_out_rows,
                held_out_records,
                included_biases=HLE_PAPER_HELD_OUT_BIASES,
            )
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "descriptive_conditional_towards_bias_switch_rate",
        "models": list(MODELS),
        "cell_order": [
            {"scope": spec.scope, "split": spec.split, "dataset": spec.dataset, "bias_type": spec.bias_type}
            for spec in CELL_SPECS
        ],
        "analysis_policy": {
            "descriptive_only": True,
            "decisions": "not produced",
            "conditional_estimand": "P(biased answer matches target | clean answer does not match target, joint parse)",
        },
        "score_contract": {
            "validated_for_every_joint_parse": "abs_switch == abs(net_switch) == toward + away",
            "net_definition": "net_switch == toward - away",
            "joint_parse_failure": "all paired score outcomes are null",
            "unavailable": ["lateral answer-label switches", "answer-label destination specificity"],
        },
        "input_artifacts": list(input_artifacts),
        "input_logs": list(input_logs),
        "rows": rows,
        "optional_micro_summaries": summaries,
    }


def _parse_runs(values: Sequence[str]) -> dict[str, str]:
    runs: dict[str, str] = {}
    for value in values:
        model, separator, location = value.partition("=")
        if not separator or not model or not location:
            raise ValueError(f"--run must be MODEL=LOCAL_LOG_LOCATION, got {value!r}")
        if model not in MODELS:
            raise ValueError(f"--run model must be one of {list(MODELS)}, got {model!r}")
        if model in runs:
            raise ValueError(f"duplicate --run for {model}")
        runs[model] = location
    return runs


def _write_json(path: Path, report: Mapping[str, Any]) -> None:
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", action="append", required=True, metavar="MODEL=LOGS")
    parser.add_argument(
        "--expected-training",
        type=Path,
        required=True,
        help="rmct_tbsr_training_manifest for the exact balanced 100+100 training subset",
    )
    parser.add_argument(
        "--expected-hle",
        type=Path,
        required=True,
        help=f"exact clean HLE file ({HLE_FILES['unbiased']})",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        runs = _parse_runs(args.run)
        training = validate_training_manifest(args.expected_training)
        expected_hle, hle_identity = load_expected_hle(args.expected_hle)
        hle_bias_files, hle_bias_identities = load_expected_hle_bias_files(args.expected_hle)
        expected_files = {
            **{(TRAINING_SPLIT, dataset, "wrong_argument"): training.path for dataset in TRAINING_DATASETS},
            **{(HLE_SPLIT, HLE_DATASET, bias): path for bias, path in hle_bias_files.items()},
        }
        records, logs = load_inspect_cells(runs, expected_files=expected_files)
        artifacts = [
            {"role": "training_manifest", **training.manifest_identity},
            {"role": "training_source", **training.source_identity},
            {"role": "expected_training_population", **training.training_identity},
            hle_identity,
            *hle_bias_identities,
        ]
        report = build_report(
            records,
            expected_training=_expected_training_ids(training),
            expected_hle=expected_hle,
            input_logs=logs,
            input_artifacts=artifacts,
        )
        _write_json(args.output, report)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(report['rows'])} descriptive RMCT TBSR cells to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CELL_SPECS",
    "CellSpec",
    "build_report",
    "load_expected_hle",
    "load_expected_hle_bias_files",
    "load_inspect_cells",
    "model_matches",
]
