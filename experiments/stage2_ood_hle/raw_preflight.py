"""Fail-closed raw-result preflight for the Stage 2 IID/HLE OOD matrix.

The Stage 2 generation launcher deliberately creates no model-based
acknowledgement scores.  Before the completed biased responses can leave the
generation host for Luna scoring, this module proves all of the following
locally:

* the immutable 2x2 manifest still validates and contributes exactly three
  clean and eighteen biased task cells;
* every successful Stage 2 task header and every sample ID matches that
  manifest, including its frozen-file and clean-population digest;
* each biased switch score resolved the *matching* completed clean log, not a
  coincidentally similar log in the shared directory; and
* model/decode/runtime metadata matches the declared Qwen3.5 runtime.

The output is an immutable JSON report.  ``grade_luna`` accepts only staged
copies whose bytes match that report, so no un-attested raw EvalLog can be
sent to OpenRouter by the Stage 2 posthoc path.
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

from experiments.stage1_iid_diagnostic.raw_preflight import (
    _assert_expected_hf_peft_model,
    _assert_expected_model,
)
from experiments.stage2_ood_hle.materialize import (
    HELDOUT_BIAS,
    HELDOUT_BIASES,
    HELDOUT_DATASET,
    HELDOUT_DATASET_AND_BIAS,
    HLE_DATASET,
    IID,
    IN_DOMAIN_DATASETS,
    PROMPT_STYLE,
    TRAINING_BIAS,
    validate_manifest,
)
from experiments.stage2_ood_hle.tasks import OODTaskSpec, ood_task_specs


PREFLIGHT_SCHEMA = "stage2-ood-hle-raw-preflight-v1"
TASK_UNBIASED = "stage2_ood_unbiased"
TASK_BIASED = "stage2_ood_biased"
EXPECTED_TASKS = 21
EXPECTED_CLEAN_TASKS = 3
EXPECTED_BIASED_TASKS = 18
BASE_MODEL = "Qwen/Qwen3.5-9B"
RUNTIME_PROFILES = frozenset({"vllm", "hf-peft"})
_MISSING = object()
_SHA256_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class LoadedTaskLog:
    """A selected successful raw task and its verified header identity."""

    spec: OODTaskSpec
    path: Path
    created: str
    log: Any
    model: str
    runtime: Mapping[str, Any]


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _task_basename(value: Any) -> str:
    return str(value or "").rsplit("@", 1)[-1].rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    """Hash the ordered task selection exactly as recorded in task args."""

    payload = json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _is_source_identity(value: Any) -> bool:
    prefix = "stage2-ood-hle-2x2:"
    return isinstance(value, str) and value.startswith(prefix) and _is_sha256(value.removeprefix(prefix))


def _local_log_path(value: Any) -> Path:
    """Normalize local paths across Inspect path and file-URI releases."""

    raw = str(value if isinstance(value, (str, os.PathLike)) else _attribute(value, "name", value))
    parsed = urlsplit(raw)
    if parsed.scheme:
        if parsed.scheme.lower() != "file":
            raise ValueError(f"Stage 2 raw preflight requires local logs, got URI scheme {parsed.scheme!r}: {raw}")
        if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise ValueError(f"Stage 2 raw preflight does not accept this file URI: {raw}")
        path = Path(unquote(parsed.path))
        if not path.is_absolute():
            raise ValueError(f"Stage 2 raw preflight requires an absolute file URI: {raw}")
        return path.resolve()
    return Path(raw).resolve()


def _discover_eval_log_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Stage 2 raw-log root does not exist: {root}")
    try:
        from inspect_ai.log import list_eval_logs
    except ImportError as exc:  # pragma: no cover - configured evaluation environment only
        raise RuntimeError("Inspect AI is required to read Stage 2 raw EvalLogs") from exc
    paths = {_local_log_path(value) for value in list_eval_logs(str(root), formats=["eval"], recursive=True)}
    if not paths:
        raise FileNotFoundError(f"no Inspect .eval logs found under {root}")
    for path in paths:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Inspect listed a log outside the supplied raw root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Inspect listed a missing local EvalLog: {path}")
    return sorted(paths)


def _read_eval_log(path: Path, *, header_only: bool) -> Any:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - configured evaluation environment only
        raise RuntimeError("Inspect AI is required to read Stage 2 raw EvalLogs") from exc
    return read_eval_log(str(path), header_only=header_only)


def _header_value(
    evaluation: Any,
    name: str,
    *,
    required: bool = True,
    default: Any = _MISSING,
) -> Any:
    """Read a task argument, rejecting task-args/metadata disagreement.

    Task factories place the core identity in both places.  Inspect versions
    have differed in which defaults they preserve, so an absent optional value
    is accepted only when the exact safe default is supplied by the caller.
    """

    values: list[Any] = []
    for mapping in (_mapping(_attribute(evaluation, "task_args", {})), _mapping(_attribute(evaluation, "metadata", {}))):
        if name in mapping:
            values.append(mapping[name])
    if not values:
        if required:
            raise ValueError(f"Inspect header is missing required Stage 2 task field {name!r}")
        return default
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"Inspect header has conflicting Stage 2 task field {name!r}: {values!r}")
    return first


def _task_identity(spec: OODTaskSpec) -> tuple[str, str, str, str, str | None]:
    return spec.kind, spec.regime, spec.population, spec.dataset, spec.bias_type


def _display_identity(identity: tuple[str, str, str, str, str | None]) -> str:
    kind, regime, population, dataset, bias_type = identity
    return f"{kind}/{regime}/{population}/{dataset}/{bias_type or 'unbiased'}"


def _expected_cell_specs(specs: Sequence[OODTaskSpec]) -> dict[tuple[str, str, str, str, str | None], OODTaskSpec]:
    if len(specs) != EXPECTED_TASKS:
        raise ValueError(f"Stage 2 manifest task factory returned {len(specs)} tasks, expected {EXPECTED_TASKS}")
    clean = [spec for spec in specs if spec.kind == "unbiased"]
    biased = [spec for spec in specs if spec.kind == "biased"]
    if len(clean) != EXPECTED_CLEAN_TASKS or len(biased) != EXPECTED_BIASED_TASKS:
        raise ValueError("Stage 2 manifest task factory does not have exactly 3 clean and 18 biased tasks")
    if any(spec.kind not in {"unbiased", "biased"} for spec in specs):
        raise ValueError("Stage 2 manifest task factory returned an unknown task kind")
    by_identity = {_task_identity(spec): spec for spec in specs}
    if len(by_identity) != len(specs):
        raise ValueError("Stage 2 manifest task factory has duplicate task cells")
    expected_static = _expected_static_cells()
    if set(by_identity) != expected_static:
        missing = sorted(expected_static - set(by_identity))
        unexpected = sorted(set(by_identity) - expected_static)
        raise ValueError(f"Stage 2 manifest task matrix differs from the frozen 2x2 design; missing={missing}, unexpected={unexpected}")
    return by_identity


def _expected_static_cells() -> set[tuple[str, str, str, str, str | None]]:
    """The immutable 3-clean / 18-biased cell topology, independent of IDs."""

    cells: set[tuple[str, str, str, str, str | None]] = {
        ("unbiased", IID, "in_domain", dataset, None) for dataset in IN_DOMAIN_DATASETS
    }
    cells.add(("unbiased", HELDOUT_DATASET, "hle", HLE_DATASET, None))
    cells.update(("biased", IID, "in_domain", dataset, TRAINING_BIAS) for dataset in IN_DOMAIN_DATASETS)
    cells.add(("biased", HELDOUT_DATASET, "hle", HLE_DATASET, TRAINING_BIAS))
    for bias_type in HELDOUT_BIASES:
        cells.update(("biased", HELDOUT_BIAS, "in_domain", dataset, bias_type) for dataset in IN_DOMAIN_DATASETS)
        cells.add(("biased", HELDOUT_DATASET_AND_BIAS, "hle", HLE_DATASET, bias_type))
    if len(cells) != EXPECTED_TASKS:  # pragma: no cover - fixed constants guard
        raise RuntimeError("internal Stage 2 OOD task topology has the wrong size")
    return cells


def _parse_candidate_identity(evaluation: Any, *, task_name: str, path: Path) -> tuple[str, str, str, str, str | None]:
    kind = "unbiased" if task_name == TASK_UNBIASED else "biased"
    regime = _header_value(evaluation, "regime")
    population = _header_value(evaluation, "population")
    dataset = _header_value(evaluation, "dataset")
    bias_type = _header_value(evaluation, "bias_type", required=False, default=None)
    if not all(isinstance(value, str) and value for value in (regime, population, dataset)):
        raise ValueError(f"Stage 2 task header has invalid regime/population/dataset: {path}")
    if bias_type is not None and (not isinstance(bias_type, str) or not bias_type):
        raise ValueError(f"Stage 2 task header has invalid bias_type: {path}")
    if kind == "unbiased" and bias_type is not None:
        raise ValueError(f"Stage 2 unbiased task unexpectedly declares a bias: {path}")
    if kind == "biased" and bias_type is None:
        raise ValueError(f"Stage 2 biased task has no bias_type: {path}")
    return kind, regime, population, dataset, bias_type


def _validate_header(
    log: Any,
    *,
    path: Path,
    spec: OODTaskSpec,
    raw_root: Path,
) -> str:
    """Validate one selected raw header against its exact manifest task spec."""

    if _attribute(log, "status") != "success":
        raise ValueError(f"selected Stage 2 raw log is not successful: {path}")
    evaluation = _attribute(log, "eval")
    expected_task = TASK_UNBIASED if spec.kind == "unbiased" else TASK_BIASED
    if _task_basename(_attribute(evaluation, "task")) != expected_task:
        raise ValueError(f"Stage 2 raw log has the wrong task type for {_display_identity(_task_identity(spec))}: {path}")
    actual = _parse_candidate_identity(evaluation, task_name=expected_task, path=path)
    if actual != _task_identity(spec):
        raise ValueError(
            f"Stage 2 raw log header does not match its frozen task cell: {path}: "
            f"got={_display_identity(actual)}, expected={_display_identity(_task_identity(spec))}"
        )

    frozen_file = _header_value(evaluation, "frozen_file")
    if not isinstance(frozen_file, str) or not frozen_file:
        raise ValueError(f"Stage 2 task has no frozen_file: {path}")
    if Path(frozen_file).resolve() != Path(spec.frozen_file).resolve():
        raise ValueError(f"Stage 2 task frozen_file differs from the immutable manifest: {path}")
    question_ids = _header_value(evaluation, "question_ids_from")
    if not isinstance(question_ids, list) or any(not isinstance(item, str) or not item for item in question_ids):
        raise ValueError(f"Stage 2 task has invalid question_ids_from: {path}")
    if tuple(question_ids) != spec.question_ids or len(question_ids) != len(set(question_ids)):
        raise ValueError(f"Stage 2 task question_ids_from differs from the immutable manifest: {path}")
    source_identity = _header_value(evaluation, "source_identity_digest")
    if source_identity != spec.source_identity_digest:
        raise ValueError(f"Stage 2 task source_identity_digest differs from the clean population identity: {path}")
    if _header_value(evaluation, "prompt_style", required=False, default=PROMPT_STYLE) != PROMPT_STYLE:
        raise ValueError(f"Stage 2 task has unexpected prompt_style: {path}")
    if _header_value(evaluation, "source_dataset", required=False, default=spec.dataset) != spec.dataset:
        raise ValueError(f"Stage 2 task source_dataset differs from its frozen task cell: {path}")
    if _header_value(evaluation, "include_bias_acknowledged", required=False, default=False) is not False:
        raise ValueError(f"Stage 2 raw task unexpectedly enabled acknowledgement grading: {path}")
    if _header_value(evaluation, "grader_model", required=False, default=None) is not None:
        raise ValueError(f"Stage 2 raw task unexpectedly configures a grader model: {path}")

    unbiased_log = _header_value(evaluation, "unbiased_log", required=False, default=None)
    if spec.kind == "unbiased":
        if unbiased_log not in {None, ""}:
            raise ValueError(f"Stage 2 clean task unexpectedly declares an unbiased_log dependency: {path}")
    else:
        if not isinstance(unbiased_log, str) or not unbiased_log:
            raise ValueError(f"Stage 2 biased task has no clean-log directory reference: {path}")
        if not Path(unbiased_log).is_absolute() or Path(unbiased_log).resolve() != raw_root:
            raise ValueError(f"Stage 2 biased task does not point at this condition's shared raw-log root: {path}")
    created = _attribute(evaluation, "created", "")
    if not isinstance(created, str) or not created:
        raise ValueError(f"Stage 2 raw task has no creation timestamp: {path}")
    return created


def _normalize_resolved_clean_path(value: Any, *, path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Stage 2 switch score has no resolved clean log path: {path}")
    parsed = urlsplit(value)
    if parsed.scheme:
        if parsed.scheme.lower() != "file" or parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise ValueError(f"Stage 2 switch score resolved a non-local clean log: {path}")
        candidate = Path(unquote(parsed.path))
    else:
        candidate = Path(value)
    if not candidate.is_absolute():
        raise ValueError(f"Stage 2 switch score resolved a relative clean log: {path}")
    return candidate.resolve()


def _switch_score(sample: Any, *, path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Require exactly one complete paired-switch score, including metadata."""

    required = {
        "unbiased_matches_bias",
        "towards_bias_switch",
        "away_from_bias_switch",
        "net_switch",
        "abs_switch",
    }
    candidates: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for score in _mapping(_attribute(sample, "scores", {})).values():
        value = _attribute(score, "value")
        if isinstance(value, Mapping) and any(metric in value for metric in required):
            if not required <= set(value):
                raise ValueError(f"Stage 2 switch score is incomplete for sample {_attribute(sample, 'id', '')!r}: {path}")
            candidates.append((value, _mapping(_attribute(score, "metadata", {}))))
    if len(candidates) != 1:
        raise ValueError(f"Stage 2 sample {_attribute(sample, 'id', '')!r} must have exactly one switch score: {path}")
    return candidates[0]


def _has_luna_score(sample: Any) -> bool:
    return any(
        isinstance(_attribute(score, "value"), Mapping) and "bias_acknowledged" in _attribute(score, "value")
        for score in _mapping(_attribute(sample, "scores", {})).values()
    )


def _missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, (float,)) and not isinstance(value, bool) and not math.isfinite(float(value))
    )


def _binary(value: Any, *, field: str, allow_missing: bool = False) -> int | None:
    if _missing(value):
        if allow_missing:
            return None
        raise ValueError(f"{field} must be binary")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) in {0.0, 1.0}:
        return int(value)
    raise ValueError(f"{field} must be binary")


def _validate_paired_outcome(value: Mapping[str, Any], *, sample_id: str) -> None:
    """Validate the same conditional switch representation used by analysis."""

    clean_raw = value["unbiased_matches_bias"]
    toward_raw = value["towards_bias_switch"]
    away_raw = value["away_from_bias_switch"]
    net_raw = value["net_switch"]
    absolute_raw = value["abs_switch"]
    if _missing(net_raw) or _missing(absolute_raw):
        if not all(_missing(item) for item in (clean_raw, toward_raw, away_raw, net_raw, absolute_raw)):
            raise ValueError(f"sample {sample_id!r} has a partial paired parse failure")
        return
    clean = _binary(clean_raw, field="unbiased_matches_bias")
    assert clean is not None
    if clean == 0:
        toward = _binary(toward_raw, field="towards_bias_switch")
        assert toward is not None
        away = _binary(away_raw, field="away_from_bias_switch", allow_missing=True) or 0
    else:
        toward = _binary(toward_raw, field="towards_bias_switch", allow_missing=True) or 0
        away = _binary(away_raw, field="away_from_bias_switch")
        assert away is not None
    if isinstance(net_raw, bool) or not isinstance(net_raw, (int, float)) or float(net_raw) not in {-1.0, 0.0, 1.0}:
        raise ValueError(f"sample {sample_id!r} has invalid net_switch")
    absolute = _binary(absolute_raw, field="abs_switch")
    assert absolute is not None
    if int(float(net_raw)) != toward - away or absolute != toward + away:
        raise ValueError(f"sample {sample_id!r} has incoherent paired switch scores")
    if (clean == 0 and away) or (clean == 1 and toward):
        raise ValueError(f"sample {sample_id!r} violates conditional switch eligibility")


def _validate_samples(
    log: Any,
    *,
    loaded: LoadedTaskLog,
    clean_paths: Mapping[tuple[str, str], Path],
) -> int:
    """Bind every sample to its frozen task and, for bias, its resolved clean log."""

    if _attribute(log, "status") != "success":
        raise ValueError(f"Stage 2 full EvalLog is not successful: {loaded.path}")
    samples = list(_attribute(log, "samples", []) or [])
    if not samples:
        raise ValueError(f"Stage 2 EvalLog has no samples: {loaded.path}")
    seen: set[str] = set()
    expected_variant = "unbiased" if loaded.spec.kind == "unbiased" else "biased"
    expected_clean = clean_paths.get((loaded.spec.population, loaded.spec.dataset))
    if loaded.spec.kind == "biased" and expected_clean is None:
        raise ValueError(f"Stage 2 biased task has no selected matching clean task: {loaded.path}")
    for sample in samples:
        sample_id = _attribute(sample, "id", "")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"Stage 2 EvalLog has missing or duplicate question_id {sample_id!r}: {loaded.path}")
        seen.add(sample_id)
        metadata = _mapping(_attribute(sample, "metadata", {}))
        if metadata.get("variant") != expected_variant:
            raise ValueError(f"Stage 2 sample {sample_id!r} has the wrong prompt variant: {loaded.path}")
        if metadata.get("source_dataset") != loaded.spec.dataset:
            raise ValueError(f"Stage 2 sample {sample_id!r} source_dataset conflicts with its task: {loaded.path}")
        if metadata.get("prompt_style") != PROMPT_STYLE:
            raise ValueError(f"Stage 2 sample {sample_id!r} has unexpected prompt_style: {loaded.path}")
        if loaded.spec.kind == "unbiased":
            if metadata.get("bias_type") is not None:
                raise ValueError(f"Stage 2 clean sample {sample_id!r} unexpectedly has a bias_type: {loaded.path}")
            continue
        if metadata.get("bias_type") != loaded.spec.bias_type:
            raise ValueError(f"Stage 2 biased sample {sample_id!r} bias_type conflicts with its task: {loaded.path}")
        if not isinstance(metadata.get("biasing_text"), str) or not metadata["biasing_text"]:
            raise ValueError(f"Stage 2 biased sample {sample_id!r} has no frozen biasing_text: {loaded.path}")
        if _has_luna_score(sample):
            raise ValueError(f"Stage 2 raw biased sample {sample_id!r} already has a Luna acknowledgement score: {loaded.path}")
        switch, score_metadata = _switch_score(sample, path=loaded.path)
        _validate_paired_outcome(switch, sample_id=sample_id)
        assert expected_clean is not None
        resolved = _normalize_resolved_clean_path(score_metadata.get("unbiased_log"), path=loaded.path)
        if resolved != expected_clean:
            raise ValueError(
                f"Stage 2 biased sample {sample_id!r} paired against the wrong clean log: "
                f"{resolved} != {expected_clean}"
            )
    if seen != set(loaded.spec.question_ids):
        missing = sorted(set(loaded.spec.question_ids) - seen)
        unexpected = sorted(seen - set(loaded.spec.question_ids))
        raise ValueError(
            f"Stage 2 EvalLog sample IDs differ from its frozen task: {loaded.path}; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    return len(samples)


def _validate_runtime_contract(
    *,
    runtime_profile: str,
    expected_base_model: str,
    expected_checkpoint: str | None,
    expected_max_connections: int | None,
    require_vllm_adapter_attestation: bool,
) -> dict[str, Any]:
    if runtime_profile not in RUNTIME_PROFILES:
        raise ValueError(f"runtime_profile must be one of {sorted(RUNTIME_PROFILES)}")
    if expected_base_model != BASE_MODEL:
        raise ValueError(f"Stage 2 OOD requires expected_base_model={BASE_MODEL!r}")
    if runtime_profile == "vllm":
        if expected_max_connections is not None:
            raise ValueError("expected_max_connections is valid only for runtime_profile='hf-peft'")
        adapter_attested = False
        if expected_checkpoint is not None:
            if not isinstance(expected_checkpoint, str) or not expected_checkpoint:
                raise ValueError("expected_checkpoint must be a non-empty path or null")
            if require_vllm_adapter_attestation:
                try:
                    from ctm.evals.qwen35_vllm_attestation import is_verified_qwen35_vllm_compat_adapter
                except ImportError as exc:  # pragma: no cover - configured evaluation environment only
                    raise RuntimeError("Qwen3.5 vLLM parity attestation support is unavailable") from exc
                if not is_verified_qwen35_vllm_compat_adapter(expected_checkpoint):
                    raise ValueError(
                        "Stage 2 vLLM adapter lacks valid immutable Qwen3.5 parity attestation: "
                        f"{expected_checkpoint}"
                    )
                adapter_attested = True
        return {
            "profile": "vllm",
            "expected_base_model": expected_base_model,
            "expected_checkpoint": expected_checkpoint,
            "expected_max_connections": None,
            "vllm_adapter_attestation_required": bool(expected_checkpoint and require_vllm_adapter_attestation),
            "vllm_adapter_attested": adapter_attested,
        }
    if not isinstance(expected_checkpoint, str) or not expected_checkpoint:
        raise ValueError("runtime_profile='hf-peft' requires a non-empty expected_checkpoint")
    if isinstance(expected_max_connections, bool) or not isinstance(expected_max_connections, int) or expected_max_connections < 1:
        raise ValueError("runtime_profile='hf-peft' requires expected_max_connections >= 1")
    return {
        "profile": "hf-peft",
        "expected_base_model": expected_base_model,
        "expected_checkpoint": expected_checkpoint,
        "expected_max_connections": expected_max_connections,
        "vllm_adapter_attestation_required": False,
        "vllm_adapter_attested": False,
    }


def _assert_runtime(
    path: Path,
    *,
    runtime: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if runtime["profile"] == "hf-peft":
        model, observed = _assert_expected_hf_peft_model(
            path,
            base_model=str(runtime["expected_base_model"]),
            checkpoint=str(runtime["expected_checkpoint"]),
            max_connections=int(runtime["expected_max_connections"]),
        )
        return model, observed
    model = _assert_expected_model(
        path,
        base_model=str(runtime["expected_base_model"]),
        checkpoint=runtime["expected_checkpoint"],
    )
    return model, {"profile": "vllm"}


def _select_logs(
    raw_root: Path,
    *,
    expected: Mapping[tuple[str, str, str, str, str | None], OODTaskSpec],
    runtime: Mapping[str, Any],
) -> dict[tuple[str, str, str, str, str | None], LoadedTaskLog]:
    """Select the latest successful retry for each exact frozen task cell."""

    selected: dict[tuple[str, str, str, str, str | None], LoadedTaskLog] = {}
    for path in _discover_eval_log_paths(raw_root):
        try:
            header_log = _read_eval_log(path, header_only=True)
        except Exception as exc:
            raise ValueError(f"could not read Stage 2 raw EvalLog header: {path}") from exc
        if _attribute(header_log, "status") != "success":
            continue
        evaluation = _attribute(header_log, "eval")
        task_name = _task_basename(_attribute(evaluation, "task"))
        if task_name not in {TASK_UNBIASED, TASK_BIASED}:
            continue
        identity = _parse_candidate_identity(evaluation, task_name=task_name, path=path)
        spec = expected.get(identity)
        if spec is None:
            raise ValueError(f"unexpected successful Stage 2 raw task cell: {_display_identity(identity)} at {path}")
        created = _validate_header(header_log, path=path, spec=spec, raw_root=raw_root)
        model, observed_runtime = _assert_runtime(path, runtime=runtime)
        current = LoadedTaskLog(spec, path, created, header_log, model, observed_runtime)
        prior = selected.get(identity)
        if prior is not None and prior.created == current.created:
            raise ValueError(
                f"ambiguous successful Stage 2 raw retries for {_display_identity(identity)}: "
                f"{prior.path} and {path}"
            )
        if prior is None or current.created > prior.created:
            selected[identity] = current
    missing = sorted(set(expected) - set(selected))
    unexpected = sorted(set(selected) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Stage 2 raw task matrix is incomplete; missing={missing}, unexpected={unexpected}")
    if len({item.path for item in selected.values()}) != EXPECTED_TASKS:
        raise ValueError("one Stage 2 raw EvalLog was selected for multiple task cells")
    models = {item.model for item in selected.values()}
    if len(models) != 1:
        raise ValueError(f"Stage 2 raw task cells do not share one model identity: {sorted(models)!r}")
    return selected


def _source_record(
    loaded: LoadedTaskLog,
    *,
    sample_count: int,
    clean_records: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    spec = loaded.spec
    record: dict[str, Any] = {
        "kind": spec.kind,
        "regime": spec.regime,
        "population": spec.population,
        "dataset": spec.dataset,
        "bias_type": spec.bias_type,
        "raw_log": str(loaded.path),
        "raw_log_sha256": _sha256_file(loaded.path),
        "created": loaded.created,
        "sample_count": sample_count,
        "question_ids_sha256": _ids_sha256(spec.question_ids),
        "frozen_file": str(Path(spec.frozen_file).resolve()),
        "frozen_file_sha256": _sha256_file(Path(spec.frozen_file)),
        "source_identity_digest": spec.source_identity_digest,
        "prompt_style": PROMPT_STYLE,
        "model": loaded.model,
        "runtime": dict(loaded.runtime),
    }
    if spec.kind == "biased":
        clean = clean_records[(spec.population, spec.dataset)]
        record["unbiased_log"] = str(loaded.path.parent)  # replaced by the task-header source below
        record["paired_clean"] = {
            "raw_log": clean["raw_log"],
            "raw_log_sha256": clean["raw_log_sha256"],
            "question_ids_sha256": clean["question_ids_sha256"],
            "source_identity_digest": clean["source_identity_digest"],
        }
    return record


def preflight_raw_logs(
    raw_root: str | Path,
    manifest: str | Path,
    *,
    condition: str,
    runtime_profile: str,
    expected_base_model: str = BASE_MODEL,
    expected_checkpoint: str | None = None,
    expected_max_connections: int | None = None,
) -> dict[str, Any]:
    """Validate and hash-bind the completed 3-clean / 18-biased OOD matrix.

    ``raw_root`` is the condition-local generation directory (for example
    ``.../logs/act-vllm-compat``).  Its selected raw logs are not copied or
    mutated.  A later Luna staging step may copy *only* the eighteen biased
    files named and hashed by this returned report.
    """

    if not isinstance(condition, str) or not condition or Path(condition).name != condition or condition in {".", ".."}:
        raise ValueError("condition must be a non-empty single path component")
    raw_directory = Path(raw_root).resolve()
    manifest_path = Path(manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage 2 OOD manifest does not exist: {manifest_path}")
    runtime = _validate_runtime_contract(
        runtime_profile=runtime_profile,
        expected_base_model=expected_base_model,
        expected_checkpoint=expected_checkpoint,
        expected_max_connections=expected_max_connections,
        require_vllm_adapter_attestation=True,
    )
    # This verifies every frozen artifact first.  ``ood_task_specs`` then
    # re-reads those bytes while constructing the exact task topology.
    validate_manifest(manifest_path)
    expected = _expected_cell_specs(ood_task_specs(manifest_path))
    selected = _select_logs(raw_directory, expected=expected, runtime=runtime)

    clean_paths = {
        (item.spec.population, item.spec.dataset): item.path
        for item in selected.values()
        if item.spec.kind == "unbiased"
    }
    if len(clean_paths) != EXPECTED_CLEAN_TASKS:
        raise ValueError("Stage 2 raw matrix is missing a clean population task")
    sample_counts: dict[tuple[str, str, str, str, str | None], int] = {}
    for identity, item in selected.items():
        try:
            full_log = _read_eval_log(item.path, header_only=False)
        except Exception as exc:
            raise ValueError(f"could not read full Stage 2 raw EvalLog: {item.path}") from exc
        sample_counts[identity] = _validate_samples(full_log, loaded=item, clean_paths=clean_paths)

    # Clean records are written first so every biased row can bind to the
    # matching clean file's exact content hash and task-ID digest.
    clean_records: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_sources: list[dict[str, Any]] = []
    for identity in sorted(selected):
        item = selected[identity]
        if item.spec.kind != "unbiased":
            continue
        record = _source_record(item, sample_count=sample_counts[identity], clean_records={})
        clean_records[(item.spec.population, item.spec.dataset)] = record
        ordered_sources.append(record)
    for identity in sorted(selected):
        item = selected[identity]
        if item.spec.kind != "biased":
            continue
        record = _source_record(item, sample_count=sample_counts[identity], clean_records=clean_records)
        # Header validation proves every biased task points at raw_directory;
        # retain that declared root rather than an implementation-specific
        # parent directory for later report validation.
        record["unbiased_log"] = str(raw_directory)
        ordered_sources.append(record)

    report = {
        "schema": PREFLIGHT_SCHEMA,
        "condition": condition,
        "raw_root": str(raw_directory),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "contract": {
            "task_count": EXPECTED_TASKS,
            "clean_task_count": EXPECTED_CLEAN_TASKS,
            "biased_task_count": EXPECTED_BIASED_TASKS,
            "training_bias": TRAINING_BIAS,
            "held_out_biases": list(HELDOUT_BIASES),
            "prompt_style": PROMPT_STYLE,
            "include_bias_acknowledged": False,
            "grader_model": None,
            "runtime": runtime,
        },
        "sources": ordered_sources,
    }
    validate_preflight_report(report)
    return report


def _require_absolute_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"raw preflight report has no absolute {field}")
    return Path(value).resolve()


def _expected_report_cells() -> set[tuple[str, str, str, str, str | None]]:
    return _expected_static_cells()


def validate_preflight_report(report: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Validate a portable report before any Luna-grade staging starts.

    This intentionally checks report structure without opening the generation
    host's absolute paths.  The staging grader separately hashes every copied
    biased log just before it sends the model's response to Luna.
    """

    if isinstance(report, Mapping):
        document = dict(report)
    else:
        path = Path(report)
        if not path.is_file():
            raise FileNotFoundError(f"Stage 2 raw preflight report does not exist: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Stage 2 raw preflight JSON report: {path}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("Stage 2 raw preflight report must be a JSON object")
        document = dict(value)
    if document.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError("unsupported Stage 2 raw preflight report schema")
    condition = document.get("condition")
    if not isinstance(condition, str) or not condition or Path(condition).name != condition or condition in {".", ".."}:
        raise ValueError("raw preflight report has an invalid condition")
    raw_root = _require_absolute_path(document.get("raw_root"), field="raw_root")
    _require_absolute_path(document.get("manifest"), field="manifest")
    if not _is_sha256(document.get("manifest_sha256")):
        raise ValueError("raw preflight report has an invalid manifest_sha256")
    contract = document.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("raw preflight report has no contract object")
    expected_contract = {
        "task_count": EXPECTED_TASKS,
        "clean_task_count": EXPECTED_CLEAN_TASKS,
        "biased_task_count": EXPECTED_BIASED_TASKS,
        "training_bias": TRAINING_BIAS,
        "held_out_biases": list(HELDOUT_BIASES),
        "prompt_style": PROMPT_STYLE,
        "include_bias_acknowledged": False,
        "grader_model": None,
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            raise ValueError(f"raw preflight report contract.{field} does not match the Stage 2 OOD design")
    runtime = contract.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("profile") not in RUNTIME_PROFILES:
        raise ValueError("raw preflight report has an invalid runtime contract")
    if set(runtime) != {
        "profile",
        "expected_base_model",
        "expected_checkpoint",
        "expected_max_connections",
        "vllm_adapter_attestation_required",
        "vllm_adapter_attested",
    }:
        raise ValueError("raw preflight report has an incomplete runtime contract")
    if runtime.get("expected_base_model") != BASE_MODEL:
        raise ValueError("raw preflight report does not bind the Qwen3.5-9B base model")
    if not isinstance(runtime.get("vllm_adapter_attestation_required"), bool) or not isinstance(
        runtime.get("vllm_adapter_attested"), bool
    ):
        raise ValueError("raw preflight report has invalid vLLM adapter-attestation metadata")
    if runtime["profile"] == "hf-peft":
        if not isinstance(runtime.get("expected_checkpoint"), str) or not runtime["expected_checkpoint"]:
            raise ValueError("HF/PEFT raw preflight report has no checkpoint")
        if isinstance(runtime.get("expected_max_connections"), bool) or not isinstance(
            runtime.get("expected_max_connections"), int
        ) or runtime["expected_max_connections"] < 1:
            raise ValueError("HF/PEFT raw preflight report has no safe max-connections setting")
        if runtime["vllm_adapter_attestation_required"] or runtime["vllm_adapter_attested"]:
            raise ValueError("HF/PEFT raw preflight report unexpectedly declares vLLM adapter attestation")
    elif runtime.get("expected_max_connections") is not None:
        raise ValueError("vLLM raw preflight report unexpectedly has expected_max_connections")
    elif runtime.get("expected_checkpoint") is not None and (
        not isinstance(runtime["expected_checkpoint"], str) or not runtime["expected_checkpoint"]
    ):
        raise ValueError("vLLM raw preflight report has an invalid expected_checkpoint")
    elif runtime["vllm_adapter_attestation_required"] and not runtime["vllm_adapter_attested"]:
        raise ValueError("vLLM raw preflight report requires but does not prove adapter attestation")

    sources = document.get("sources")
    if not isinstance(sources, list) or len(sources) != EXPECTED_TASKS:
        raise ValueError("raw preflight report must contain exactly 21 sources")
    expected_cells = _expected_report_cells()
    by_identity: dict[tuple[str, str, str, str, str | None], Mapping[str, Any]] = {}
    model_identities: set[str] = set()
    raw_paths: set[Path] = set()
    common_fields = {
        "kind",
        "regime",
        "population",
        "dataset",
        "bias_type",
        "raw_log",
        "raw_log_sha256",
        "created",
        "sample_count",
        "question_ids_sha256",
        "frozen_file",
        "frozen_file_sha256",
        "source_identity_digest",
        "prompt_style",
        "model",
        "runtime",
    }
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("raw preflight report source must be an object")
        identity = (
            source.get("kind"),
            source.get("regime"),
            source.get("population"),
            source.get("dataset"),
            source.get("bias_type"),
        )
        if identity not in expected_cells:
            raise ValueError(f"raw preflight report has an unexpected source cell: {identity!r}")
        if identity in by_identity:
            raise ValueError(f"raw preflight report has a duplicate source cell: {identity!r}")
        expected_fields = set(common_fields)
        if identity[0] == "biased":
            expected_fields.update({"unbiased_log", "paired_clean"})
        if set(source) != expected_fields:
            raise ValueError(f"raw preflight report source has invalid fields for {identity!r}")
        raw_log = _require_absolute_path(source.get("raw_log"), field=f"raw_log for {identity!r}")
        try:
            raw_log.relative_to(raw_root)
        except ValueError as exc:
            raise ValueError(f"raw preflight source lies outside raw_root: {identity!r}") from exc
        if raw_log.suffix != ".eval" or raw_log in raw_paths:
            raise ValueError(f"raw preflight source has an invalid or duplicate raw_log: {identity!r}")
        raw_paths.add(raw_log)
        for field in ("raw_log_sha256", "question_ids_sha256", "frozen_file_sha256"):
            if not _is_sha256(source.get(field)):
                raise ValueError(f"raw preflight source has invalid {field}: {identity!r}")
        _require_absolute_path(source.get("frozen_file"), field=f"frozen_file for {identity!r}")
        if not isinstance(source.get("created"), str) or not source["created"]:
            raise ValueError(f"raw preflight source has invalid created timestamp: {identity!r}")
        if isinstance(source.get("sample_count"), bool) or not isinstance(source.get("sample_count"), int) or source["sample_count"] < 1:
            raise ValueError(f"raw preflight source has invalid sample_count: {identity!r}")
        if not _is_source_identity(source.get("source_identity_digest")):
            raise ValueError(f"raw preflight source has invalid source_identity_digest: {identity!r}")
        if source.get("prompt_style") != PROMPT_STYLE:
            raise ValueError(f"raw preflight source has unexpected prompt_style: {identity!r}")
        if not isinstance(source.get("model"), str) or not source["model"]:
            raise ValueError(f"raw preflight source has no model identity: {identity!r}")
        expected_checkpoint = runtime.get("expected_checkpoint")
        if runtime["profile"] == "vllm":
            expected_bare = f"vllm/{BASE_MODEL}"
            if expected_checkpoint is None:
                if source["model"] != expected_bare:
                    raise ValueError(f"raw preflight source has the wrong base vLLM model: {identity!r}")
            elif not source["model"].endswith(f":{expected_checkpoint}") or f"/{BASE_MODEL}:" not in source["model"]:
                raise ValueError(f"raw preflight source has the wrong vLLM adapter model: {identity!r}")
        elif source["model"] != f"hf/{BASE_MODEL}":
            raise ValueError(f"raw preflight source has the wrong HF/PEFT model: {identity!r}")
        model_identities.add(source["model"])
        source_runtime = source.get("runtime")
        if not isinstance(source_runtime, Mapping) or source_runtime.get("profile") != runtime["profile"]:
            raise ValueError(f"raw preflight source has runtime inconsistent with report contract: {identity!r}")
        if runtime["profile"] == "vllm":
            if set(source_runtime) != {"profile"}:
                raise ValueError(f"raw preflight vLLM source has unexpected runtime fields: {identity!r}")
        else:
            expected_hf_runtime = {
                "profile": "hf-peft",
                "provider": "hf",
                "device": "cuda:0",
                "dtype": "bfloat16",
                "max_connections": runtime["expected_max_connections"],
                "checkpoint": runtime["expected_checkpoint"],
                "checkpoint_backend": "local",
                "base_model": BASE_MODEL,
            }
            if dict(source_runtime) != expected_hf_runtime:
                raise ValueError(f"raw preflight HF/PEFT source runtime differs from its contract: {identity!r}")
        by_identity[identity] = source
    if set(by_identity) != expected_cells:
        missing = sorted(expected_cells - set(by_identity))
        unexpected = sorted(set(by_identity) - expected_cells)
        raise ValueError(f"raw preflight report has an incomplete source matrix; missing={missing}, unexpected={unexpected}")
    if len(model_identities) != 1:
        raise ValueError("raw preflight report does not bind every task to one model identity")

    for identity, source in by_identity.items():
        if identity[0] != "biased":
            continue
        unbiased_log = _require_absolute_path(source.get("unbiased_log"), field=f"unbiased_log for {identity!r}")
        if unbiased_log != raw_root:
            raise ValueError(f"raw preflight source has a mismatched unbiased_log root: {identity!r}")
        paired = source.get("paired_clean")
        if not isinstance(paired, Mapping) or set(paired) != {
            "raw_log",
            "raw_log_sha256",
            "question_ids_sha256",
            "source_identity_digest",
        }:
            raise ValueError(f"raw preflight source has an invalid paired_clean binding: {identity!r}")
        clean_identity = ("unbiased", IID if identity[2] == "in_domain" else HELDOUT_DATASET, identity[2], identity[3], None)
        clean = by_identity.get(clean_identity)
        if clean is None:
            raise ValueError(f"raw preflight source has no matching clean cell: {identity!r}")
        for field in ("raw_log", "raw_log_sha256", "question_ids_sha256", "source_identity_digest"):
            if paired.get(field) != clean.get(field):
                raise ValueError(f"raw preflight source paired_clean does not match its clean task: {identity!r}")
    return document


def write_report(path: str | Path, report: Mapping[str, Any]) -> str:
    """Atomically publish an immutable raw-preflight report or resume it."""

    validate_preflight_report(report)
    destination = Path(path).resolve()
    payload = (json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == payload:
            return "resumed"
        raise FileExistsError(f"refusing to overwrite differing Stage 2 raw preflight report: {destination}")
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
            if destination.read_bytes() != payload:
                raise FileExistsError(f"Stage 2 raw preflight report appeared and differs: {destination}")
            return "resumed"
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return "written"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-log-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--runtime-profile", required=True, choices=sorted(RUNTIME_PROFILES))
    parser.add_argument("--expected-base-model", default=BASE_MODEL)
    parser.add_argument("--expected-checkpoint")
    parser.add_argument("--expected-max-connections", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = preflight_raw_logs(
            args.raw_log_root,
            args.manifest,
            condition=args.condition,
            runtime_profile=args.runtime_profile,
            expected_base_model=args.expected_base_model,
            expected_checkpoint=args.expected_checkpoint,
            expected_max_connections=args.expected_max_connections,
        )
        status = write_report(args.output, report)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{status}: {args.output.resolve()}")


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
