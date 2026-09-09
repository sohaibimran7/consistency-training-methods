"""Custody-safe Luna grading and standard publication for Gemma base evals.

This consumer only accepts the completed 21-cell, 50-question-per-cell Gemma
base campaign.  It leaves the canonical merged EvalLogs unchanged, sends Luna
requests only for parsed *biased* responses, and writes all derived evidence
in a separate namespace.

The campaign is intentionally base-only.  Its charts therefore carry Wilson
intervals and sample labels but no significance stars: there is no within-model
Gemma checkpoint comparator, and no cross-model pairing is asserted here.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ctm_data.adapters.mcq_bias.analysis import (
    aggregate_logs,
    append_bias_group_summaries,
    append_binomial_wilson_intervals,
)
from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from ctm_data.adapters.mcq_bias.plot_registry import load_presentation_registry, registry_labels


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_NAME = "gemma4-12b-base-two-bias-50x21-16gpu-v1"
MODEL_ID = "google/gemma-4-12B-it"
MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
MERGE_RECEIPT_SCHEMA = "gemma4-12b-base-two-bias-merge-receipt-v1"
LAUNCH_SCHEMA = "gemma4-12b-base-two-bias-launch-v1"
PREFLIGHT_SCHEMA = "gemma4-12b-base-screen-preflight-v1"
ATTEMPT_SCHEMA = "gemma4-12b-base-screen-luna-attempt-v1"
DERIVED_SCHEMA = "gemma4-12b-base-screen-luna-derived-v1"
COMPLETION_SCHEMA = "gemma4-12b-base-screen-luna-completion-v1"
PUBLICATION_SCHEMA = "gemma4-12b-base-screen-publication-v1"

DATASETS = ("logiqa", "hellaswag", "hle-text-mc")
SEEN_BIASES = ("wrong_argument", "suggested_answer")
HELD_OUT_BIASES = (
    "distractor_fact",
    "post_hoc",
    "spurious_few_shot_squares",
    "wrong_few_shot",
)
ALL_BIASES = (*SEEN_BIASES, *HELD_OUT_BIASES)
BIAS_GROUPS = {
    "seen_mean": SEEN_BIASES,
    "held_out_mean": HELD_OUT_BIASES,
    "overall_mean": ALL_BIASES,
}
BIAS_ORDER = (*ALL_BIASES, *BIAS_GROUPS)
POPULATIONS = {
    "held_in_datasets": ("logiqa", "hellaswag"),
    "held_out_dataset": ("hle-text-mc",),
}
POPULATION_ORDER = tuple(POPULATIONS)
QUESTIONS_PER_CELL = 50
TASK_COUNT = 21
DERIVED_NAMESPACE = "derived-luna-no-cap-v1"
INSPECT_RESCORE_MODEL = "mockllm/model"
LUNA_GRADER_MODEL = "openrouter/openai/gpt-5.6-luna-20260709"
NO_SIGNIFICANCE_REASON = "base_only_screen_has_no_within_model_checkpoint_comparator"
GENERATION_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "max_connections": 8,
    "termination": "model_eos_only",
}


class GemmaBasePostprocessError(ValueError):
    """The campaign or a derived artifact is unsuitable for publication."""


@dataclass(frozen=True, slots=True)
class GradeSource:
    """One receipt-bound canonical 50-question merged cell."""

    task_index: int
    kind: str
    regime: str
    population: str
    dataset: str
    bias_type: str | None
    sample_count: int
    raw_path: Path
    raw_sha256: str
    raw_size_bytes: int
    receipt_path: Path
    receipt_sha256: str
    full_question_ids_sha256: str

    @property
    def is_biased(self) -> bool:
        return self.kind == "biased"


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            candidate = method()
            if isinstance(candidate, Mapping):
                return dict(candidate)
    return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size < 1:
        raise GemmaBasePostprocessError(f"{label} must be a non-empty regular file: {candidate}")
    resolved = candidate.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved), "size_bytes": resolved.stat().st_size}


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GemmaBasePostprocessError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise GemmaBasePostprocessError(f"{label} must contain an object: {path}")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_bytes_once(path: Path, payload: bytes, *, label: str) -> str:
    """Publish a derived file once, accepting only an identical resume."""

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing {label}: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise GemmaBasePostprocessError(f"{label} parent must be a regular directory: {path.parent}")
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


def _write_json_once(path: Path, value: Mapping[str, Any], *, label: str) -> str:
    return _write_bytes_once(path, _canonical_json(value), label=label)


def _write_eval_once(log: Any, path: Path, *, label: str) -> str:
    """Persist one derived EvalLog without replacing a raw or prior log."""

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


def _under(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise GemmaBasePostprocessError(f"{label} escapes its expected root: {resolved}") from exc
    return resolved


def _regular_directory(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise FileNotFoundError(f"{label} must be a regular directory: {candidate}")
    return candidate.resolve()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _identity_from_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GemmaBasePostprocessError(f"{label} must be an identity object")
    path, digest, size = value.get("path"), value.get("sha256"), value.get("size_bytes")
    if (
        not isinstance(path, str)
        or not path
        or not _is_sha256(digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
    ):
        raise GemmaBasePostprocessError(f"{label} is malformed")
    return {"path": path, "sha256": digest, "size_bytes": size}


def _resolve_roots(
    *, campaign_root: str | Path | None, merged_root: str | Path | None
) -> tuple[Path, Path]:
    if campaign_root is None and merged_root is None:
        raise GemmaBasePostprocessError("provide --campaign-root or --merged-root")
    merged = _regular_directory(merged_root, label="merged root") if merged_root is not None else None
    campaign = _regular_directory(campaign_root, label="campaign root") if campaign_root is not None else None
    if campaign is None:
        assert merged is not None
        campaign = merged.parent
    if merged is None:
        merged = _regular_directory(campaign / "merged", label="campaign merged root")
    _under(merged, campaign, label="merged root")
    return campaign, merged


def _load_launch_contract(campaign_root: Path) -> dict[str, Any]:
    path = campaign_root / "launch-contract.json"
    document = _read_json(path, label="Gemma launch contract")
    if document.get("schema") != LAUNCH_SCHEMA or document.get("campaign") != CAMPAIGN_NAME:
        raise GemmaBasePostprocessError("launch contract has the wrong Gemma campaign identity")
    model = _mapping(_mapping(document.get("model_snapshot")))
    if model.get("model_id") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise GemmaBasePostprocessError("launch contract does not bind the pinned Gemma 4 12B base model")
    sampling = _mapping(document.get("sampling"))
    if sampling != GENERATION_SAMPLING:
        raise GemmaBasePostprocessError("launch contract differs from the EOS-only no-token-cap sampling policy")
    return document


def _receipt_paths(merged_root: Path) -> dict[int, Path]:
    receipt_root = _regular_directory(merged_root / "receipts", label="Gemma merged receipt root")
    expected = {f"task-{index:03d}.json" for index in range(1, TASK_COUNT + 1)}
    found = {path.name for path in receipt_root.iterdir() if path.is_file() or path.is_symlink()}
    if found != expected:
        raise GemmaBasePostprocessError(
            f"Gemma merged receipts must be exactly task-001 through task-021; found={sorted(found)}"
        )
    paths: dict[int, Path] = {}
    for index in range(1, TASK_COUNT + 1):
        path = receipt_root / f"task-{index:03d}.json"
        if path.is_symlink():
            raise GemmaBasePostprocessError(f"Gemma merged receipt must not be linked: {path}")
        paths[index] = path.resolve()
    return paths


def _read_eval(path: Path) -> Any:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - evaluation environment boundary
        raise RuntimeError("Inspect AI is required to read Gemma EvalLogs") from exc
    return read_eval_log(str(path), header_only=False)


def _task_args(log: Any) -> dict[str, Any]:
    return _mapping(_attribute(_attribute(log, "eval"), "task_args", {}))


def _sample_ids(log: Any) -> list[str]:
    output = [str(_attribute(sample, "id", "")) for sample in (_attribute(log, "samples", []) or [])]
    if not output or any(not sample_id for sample_id in output) or len(output) != len(set(output)):
        raise GemmaBasePostprocessError("canonical Gemma EvalLog samples lack unique non-empty question IDs")
    return output


def _validate_shard_evidence(receipt: Mapping[str, Any], *, campaign_root: Path) -> None:
    shards = receipt.get("shard_logs")
    if not isinstance(shards, list) or len(shards) != 16:
        raise GemmaBasePostprocessError("Gemma merge receipt must bind exactly 16 source shards")
    ranks: set[int] = set()
    for record in shards:
        if not isinstance(record, Mapping):
            raise GemmaBasePostprocessError("Gemma merge receipt has a malformed shard record")
        rank = record.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < 16 or rank in ranks:
            raise GemmaBasePostprocessError("Gemma merge receipt has invalid or duplicate shard ranks")
        ranks.add(rank)
        identity = _identity_from_record(record.get("raw_log"), label="Gemma shard identity")
        shard_path = Path(str(identity["path"])).expanduser()
        _under(shard_path, campaign_root / "live-shards", label="Gemma shard evidence")
        if _identity(shard_path, label="Gemma shard evidence") != identity:
            raise GemmaBasePostprocessError("Gemma shard evidence differs from its merge receipt")
    if ranks != set(range(16)):
        raise GemmaBasePostprocessError("Gemma merge receipt does not cover ranks 0 through 15")


def _source_from_receipt(
    *, task_index: int, receipt_path: Path, campaign_root: Path, merged_root: Path
) -> GradeSource:
    receipt = _read_json(receipt_path, label=f"Gemma task-{task_index:03d} merge receipt")
    if receipt.get("schema") != MERGE_RECEIPT_SCHEMA or receipt.get("campaign") != CAMPAIGN_NAME:
        raise GemmaBasePostprocessError(f"Gemma task-{task_index:03d} merge receipt has the wrong identity")
    if receipt.get("task_index") != task_index:
        raise GemmaBasePostprocessError(f"Gemma task-{task_index:03d} merge receipt has the wrong task index")
    kind, dataset, bias = receipt.get("kind"), receipt.get("dataset"), receipt.get("bias_type")
    population, regime = receipt.get("population"), receipt.get("regime")
    sample_count = receipt.get("sample_count")
    if kind not in {"unbiased", "biased"} or dataset not in DATASETS:
        raise GemmaBasePostprocessError(f"Gemma task-{task_index:03d} has unsupported scientific labels")
    if not isinstance(population, str) or not population or not isinstance(regime, str) or not regime:
        raise GemmaBasePostprocessError(f"Gemma task-{task_index:03d} lacks population/regime labels")
    if sample_count != QUESTIONS_PER_CELL or receipt.get("shard_count") != 16:
        raise GemmaBasePostprocessError(f"Gemma task-{task_index:03d} does not retain the fixed 50×16 protocol")
    full_ids = receipt.get("full_question_ids_sha256")
    if not _is_sha256(full_ids):
        raise GemmaBasePostprocessError(f"Gemma task-{task_index:03d} lacks its frozen question-ID digest")
    if kind == "unbiased":
        if bias is not None:
            raise GemmaBasePostprocessError("unbiased Gemma receipt unexpectedly names a bias")
    elif not isinstance(bias, str) or bias not in ALL_BIASES:
        raise GemmaBasePostprocessError(f"Gemma task-{task_index:03d} has an unsupported bias")
    _validate_shard_evidence(receipt, campaign_root=campaign_root)
    raw_record = _identity_from_record(receipt.get("raw_log"), label="Gemma canonical merged log identity")
    raw_path = Path(str(raw_record["path"])).expanduser()
    _under(raw_path, merged_root, label="Gemma canonical merged log")
    observed = _identity(raw_path, label="Gemma canonical merged log")
    if observed != raw_record:
        raise GemmaBasePostprocessError(f"Gemma task-{task_index:03d} canonical EvalLog differs from its receipt")
    expected_parent = merged_root / f"task-{task_index:03d}"
    _under(raw_path, expected_parent, label="Gemma canonical task path")
    if raw_path.name != f"{raw_record['sha256']}.eval":
        raise GemmaBasePostprocessError("Gemma canonical merged log must use its content hash as filename")
    log = _read_eval(raw_path)
    if _attribute(log, "status") != "success" or len(_sample_ids(log)) != QUESTIONS_PER_CELL:
        raise GemmaBasePostprocessError(f"Gemma task-{task_index:03d} is not a successful 50-sample EvalLog")
    args = _task_args(log)
    if args.get("dataset") != dataset or args.get("bias_type") != bias:
        raise GemmaBasePostprocessError(f"Gemma task-{task_index:03d} EvalLog labels differ from its receipt")
    ids = args.get("question_ids_from")
    if not isinstance(ids, list) or len(ids) != QUESTIONS_PER_CELL or any(not isinstance(item, str) or not item for item in ids):
        raise GemmaBasePostprocessError(f"Gemma task-{task_index:03d} lacks its 50 frozen question IDs")
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    if digest != full_ids or set(ids) != set(_sample_ids(log)):
        raise GemmaBasePostprocessError(f"Gemma task-{task_index:03d} question IDs do not match merged evidence")
    _validate_source_samples(log, kind=kind, bias_type=bias if isinstance(bias, str) else None)
    return GradeSource(
        task_index=task_index,
        kind=kind,
        regime=regime,
        population=population,
        dataset=dataset,
        bias_type=bias if isinstance(bias, str) else None,
        sample_count=QUESTIONS_PER_CELL,
        raw_path=raw_path.resolve(),
        raw_sha256=str(observed["sha256"]),
        raw_size_bytes=int(observed["size_bytes"]),
        receipt_path=receipt_path.resolve(),
        receipt_sha256=_sha256(receipt_path),
        full_question_ids_sha256=str(full_ids),
    )


def load_campaign(
    *, campaign_root: str | Path | None = None, merged_root: str | Path | None = None
) -> tuple[Path, Path, list[GradeSource]]:
    """Open and validate the fixed completed campaign without changing it."""

    campaign, merged = _resolve_roots(campaign_root=campaign_root, merged_root=merged_root)
    _load_launch_contract(campaign)
    sources = [
        _source_from_receipt(
            task_index=index,
            receipt_path=path,
            campaign_root=campaign,
            merged_root=merged,
        )
        for index, path in sorted(_receipt_paths(merged).items())
    ]
    if len(sources) != TASK_COUNT:
        raise GemmaBasePostprocessError("Gemma campaign does not have 21 merged cells")
    observed: set[tuple[str, str | None]] = {(source.dataset, source.bias_type) for source in sources}
    expected = {(dataset, None) for dataset in DATASETS} | {
        (dataset, bias) for dataset in DATASETS for bias in ALL_BIASES
    }
    if observed != expected or len(observed) != len(sources):
        raise GemmaBasePostprocessError("Gemma campaign does not contain exactly 3 clean plus 18 biased cells")
    for dataset in DATASETS:
        digests = {source.full_question_ids_sha256 for source in sources if source.dataset == dataset}
        if len(digests) != 1:
            raise GemmaBasePostprocessError(
                f"Gemma campaign variants for {dataset!r} do not retain one shared 50-question pool"
            )
    return campaign, merged, sources


def _source_record(source: GradeSource) -> dict[str, Any]:
    return {
        "task_index": source.task_index,
        "kind": source.kind,
        "regime": source.regime,
        "population": source.population,
        "dataset": source.dataset,
        "bias_type": source.bias_type,
        "sample_count": source.sample_count,
        "full_question_ids_sha256": source.full_question_ids_sha256,
        "raw_log": {
            "path": str(source.raw_path),
            "sha256": source.raw_sha256,
            "size_bytes": source.raw_size_bytes,
        },
        "merge_receipt": {
            "path": str(source.receipt_path),
            "sha256": source.receipt_sha256,
            "size_bytes": source.receipt_path.stat().st_size,
        },
    }


def preflight_campaign(
    *,
    campaign_root: str | Path | None = None,
    merged_root: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Publish an immutable validation receipt for the 21 canonical logs."""

    campaign, merged, sources = load_campaign(campaign_root=campaign_root, merged_root=merged_root)
    output_path = Path(output).expanduser().resolve() if output is not None else campaign / "preflight.json"
    _under(output_path, campaign, label="Gemma preflight output")
    document = {
        "schema": PREFLIGHT_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "campaign_root": str(campaign),
        "merged_root": str(merged),
        "launch_contract": _identity(campaign / "launch-contract.json", label="Gemma launch contract"),
        "counts": {
            "cells": len(sources),
            "clean_cells": sum(not source.is_biased for source in sources),
            "biased_cells": sum(source.is_biased for source in sources),
            "questions_per_cell": QUESTIONS_PER_CELL,
            "total_generations": sum(source.sample_count for source in sources),
        },
        "scientific_scope": {
            "condition": "base",
            "within_model_significance": "not_applicable_base_only",
            "cross_model_significance": "not_claimed_without_proven_matched_question_pools",
        },
        "sources": [_source_record(source) for source in sources],
    }
    status = _write_json_once(output_path, document, label="Gemma base-screen preflight")
    return {
        "status": status,
        "preflight": _identity(output_path, label="Gemma base-screen preflight"),
        **document["counts"],
    }


def _grader_policy() -> dict[str, Any]:
    scorer_path = PROJECT_ROOT / "ctm_data" / "adapters" / "mcq_bias" / "luna_scorer_no_cap.py"
    return {
        "grader_model": LUNA_GRADER_MODEL,
        "max_connections": 500,
        "aggregate_connection_limit": 500,
        "output_token_cap": None,
        "reasoning_token_cap": None,
        "completion_token_cap": None,
        "termination": "provider_model_eos_or_natural_stop",
        "parsed_biased_outputs_only": True,
        "unparsed_output_policy": "no_request_and_nan_grade_retained",
        "scorer_source": _identity(scorer_path, label="uncapped Luna scorer source"),
    }


def _answer_parsed(sample: Any) -> bool:
    matches: list[Any] = []
    for score in _mapping(_attribute(sample, "scores", {})).values():
        value = _attribute(score, "value")
        if isinstance(value, Mapping) and "answer_parsed" in value:
            matches.append(value["answer_parsed"])
    if len(matches) != 1 or isinstance(matches[0], bool) or not isinstance(matches[0], (int, float)):
        raise GemmaBasePostprocessError(f"sample {_attribute(sample, 'id', '')!r} lacks one numeric answer_parsed score")
    result = float(matches[0])
    if not math.isfinite(result) or result not in {0.0, 1.0}:
        raise GemmaBasePostprocessError(f"sample {_attribute(sample, 'id', '')!r} has non-binary answer_parsed")
    return result == 1.0


def _switch_value(sample: Any) -> float | None:
    matches: list[Any] = []
    for score in _mapping(_attribute(sample, "scores", {})).values():
        value = _attribute(score, "value")
        if isinstance(value, Mapping) and "towards_bias_switch" in value:
            matches.append(value["towards_bias_switch"])
    if len(matches) != 1:
        raise GemmaBasePostprocessError(
            f"sample {_attribute(sample, 'id', '')!r} lacks one towards_bias_switch score"
        )
    value = matches[0]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GemmaBasePostprocessError(
            f"sample {_attribute(sample, 'id', '')!r} has an invalid towards_bias_switch score"
        )
    # The standard pinned switch scorer uses NaN as Inspect's canonical
    # conditional-unscored sentinel.  These rows are already biased in the
    # clean response and must be excluded from the towards-switch denominator,
    # rather than making a completed canonical cell fail preflight.
    if math.isnan(float(value)):
        return None
    if not math.isfinite(float(value)):
        raise GemmaBasePostprocessError(
            f"sample {_attribute(sample, 'id', '')!r} has an invalid towards_bias_switch score"
        )
    if float(value) not in {0.0, 1.0}:
        raise GemmaBasePostprocessError(
            f"sample {_attribute(sample, 'id', '')!r} has a non-binary towards_bias_switch score"
        )
    return float(value)


def _validate_source_samples(log: Any, *, kind: str, bias_type: str | None) -> None:
    expected_variant = "biased" if kind == "biased" else "unbiased"
    for sample in _attribute(log, "samples", []) or []:
        metadata = _mapping(_attribute(sample, "metadata", {}))
        if metadata.get("variant") != expected_variant:
            raise GemmaBasePostprocessError(
                f"sample {_attribute(sample, 'id', '')!r} has no {expected_variant!r} variant evidence"
            )
        if metadata.get("bias_type") != bias_type:
            raise GemmaBasePostprocessError(
                f"sample {_attribute(sample, 'id', '')!r} bias label conflicts with its canonical cell"
            )
        if kind == "biased" and (
            not isinstance(metadata.get("biasing_text"), str) or not metadata["biasing_text"]
        ):
            raise GemmaBasePostprocessError(
                f"biased sample {_attribute(sample, 'id', '')!r} lacks the frozen biasing text required by Luna"
            )
        _answer_parsed(sample)
        if kind == "biased":
            _switch_value(sample)


def _luna_score(sample: Any) -> tuple[str, Any]:
    matches: list[tuple[str, Any]] = []
    for name, score in _mapping(_attribute(sample, "scores", {})).items():
        value = _attribute(score, "value")
        if isinstance(value, Mapping) and "bias_acknowledged" in value:
            matches.append((str(name), score))
    if len(matches) != 1:
        raise GemmaBasePostprocessError(f"sample {_attribute(sample, 'id', '')!r} lacks one Luna score")
    return matches[0]


def _load_raw_logs(sources: Sequence[GradeSource]) -> list[Any]:
    logs: list[Any] = []
    for source in sources:
        observed = _identity(source.raw_path, label="Gemma source EvalLog")
        if observed["sha256"] != source.raw_sha256 or observed["size_bytes"] != source.raw_size_bytes:
            raise GemmaBasePostprocessError("a receipt-bound Gemma source EvalLog changed before grading")
        log = _read_eval(source.raw_path)
        if _attribute(log, "status") != "success" or len(_sample_ids(log)) != source.sample_count:
            raise GemmaBasePostprocessError(f"Gemma task-{source.task_index:03d} is no longer complete")
        logs.append(log)
    return logs


def _parsed_positions(sources: Sequence[GradeSource], logs: Sequence[Any]) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for source, log in zip(sources, logs, strict=True):
        if not source.is_biased:
            raise GemmaBasePostprocessError("Luna source selection must contain biased cells only")
        for sample_index, sample in enumerate(_attribute(log, "samples", []) or []):
            if _answer_parsed(sample):
                positions.append(
                    {
                        "task_index": source.task_index,
                        "sample_index": sample_index,
                        "sample_id": str(_attribute(sample, "id", "")),
                        "source_sha256": source.raw_sha256,
                    }
                )
    return positions


def _make_aggregate(
    logs: Sequence[Any], sources: Sequence[GradeSource], positions: Sequence[Mapping[str, Any]]
) -> Any:
    if not logs:
        raise GemmaBasePostprocessError("cannot form a Luna aggregate without biased source logs")
    by_task = {source.task_index: (source, log) for source, log in zip(sources, logs, strict=True)}
    aggregate = copy.deepcopy(logs[0])
    selected: list[Any] = []
    for ordinal, position in enumerate(positions):
        task_index = position.get("task_index")
        sample_index = position.get("sample_index")
        if isinstance(task_index, bool) or not isinstance(task_index, int) or isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise GemmaBasePostprocessError("Luna parsed-position record is malformed")
        source, raw = by_task[task_index]
        raw_samples = list(_attribute(raw, "samples", []) or [])
        if not 0 <= sample_index < len(raw_samples):
            raise GemmaBasePostprocessError("Luna parsed-position record points outside its raw EvalLog")
        sample = copy.deepcopy(raw_samples[sample_index])
        original_sample_id = str(_attribute(sample, "id", ""))
        sample.id = f"base/task-{task_index:03d}/sample-{sample_index:04d}/{original_sample_id}"
        metadata = _mapping(_attribute(sample, "metadata", {}))
        metadata["ctm_gemma4_12b_base_luna_origin"] = {
            "ordinal": ordinal,
            "task_index": task_index,
            "sample_index": sample_index,
            "source_sha256": source.raw_sha256,
            "original_sample_id": original_sample_id,
        }
        sample.metadata = metadata
        selected.append(sample)
    aggregate.samples = selected
    aggregate.results = None
    return aggregate


def _validate_scored_aggregate(log: Any, claim: Mapping[str, Any]) -> dict[tuple[int, int], tuple[str, Any]]:
    samples = list(_attribute(log, "samples", []) or [])
    if _attribute(log, "status") != "success" or len(samples) != claim.get("parsed_request_count"):
        raise GemmaBasePostprocessError("Luna parsed aggregate is incomplete")
    scores: dict[tuple[int, int], tuple[str, Any]] = {}
    for sample in samples:
        origin = _mapping(_mapping(_attribute(sample, "metadata", {})).get("ctm_gemma4_12b_base_luna_origin"))
        task_index, sample_index = origin.get("task_index"), origin.get("sample_index")
        if (
            isinstance(task_index, bool)
            or not isinstance(task_index, int)
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or (task_index, sample_index) in scores
        ):
            raise GemmaBasePostprocessError("Luna aggregate origin metadata is invalid or duplicated")
        scorer_name, score = _luna_score(sample)
        metadata = _mapping(_attribute(score, "metadata", {}))
        runtime = _mapping(metadata.get("grader_runtime_policy"))
        if (
            metadata.get("grader_model") != LUNA_GRADER_MODEL
            or metadata.get("grader_request_sent") is not True
            or metadata.get("grader_no_output_token_cap") is not True
            or metadata.get("grader_output_token_cap") is not None
            or runtime.get("max_connections") != 500
            or runtime.get("generate_config_output_token_cap") is not None
            or runtime.get("provider_config_default_output_token_cap") is not None
            or runtime.get("provider_default_output_token_cap") is not None
        ):
            raise GemmaBasePostprocessError("Luna aggregate lacks the required uncapped 500-connection attestation")
        scores[(task_index, sample_index)] = (scorer_name, score)
    if len(scores) != claim.get("parsed_request_count"):
        raise GemmaBasePostprocessError("Luna aggregate score count differs from its paid-request claim")
    claimed = claim.get("parsed_positions")
    if not isinstance(claimed, list):
        raise GemmaBasePostprocessError("Luna paid-request claim lacks parsed positions")
    expected = {
        (position.get("task_index"), position.get("sample_index"))
        for position in claimed
        if isinstance(position, Mapping)
    }
    if expected != set(scores) or len(expected) != len(claimed):
        raise GemmaBasePostprocessError("Luna aggregate positions differ from the paid-request claim")
    return scores


def _json_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _derived_projection(log: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in _attribute(log, "samples", []) or []:
        scorer_name, score = _luna_score(sample)
        rows.append(
            {
                "sample_id": str(_attribute(sample, "id", "")),
                "raw_answer_parsed": _answer_parsed(sample),
                "scorer": scorer_name,
                "bias_acknowledged": _json_number(_mapping(_attribute(score, "value", {})).get("bias_acknowledged")),
                "metadata": _mapping(_attribute(score, "metadata", {})),
            }
        )
    return rows


def _source_paths(output_root: Path, source: GradeSource) -> tuple[Path, Path, Path]:
    bias = source.bias_type or "unbiased"
    directory = output_root / "cells" / f"task-{source.task_index:03d}-{source.dataset}-{bias}"
    stem = source.raw_sha256
    return directory / f"{stem}-luna.eval", directory / f"{stem}-luna.jsonl", directory / f"{stem}-luna.provenance.json"


def _publish_sources(
    *,
    output_root: Path,
    sources: Sequence[GradeSource],
    logs: Sequence[Any],
    scores: Mapping[tuple[int, int], tuple[str, Any]],
    attempt_identity: Mapping[str, Any],
    grader_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from inspect_ai.log import read_eval_log
    from inspect_ai.scorer import Score

    scorer_names = {name for name, _score in scores.values()}
    if len(scorer_names) != 1:
        raise GemmaBasePostprocessError("Luna scored aggregate has no single scorer identity")
    scorer_name = next(iter(scorer_names))
    grader_model = grader_policy.get("grader_model")
    if grader_model != LUNA_GRADER_MODEL:
        raise GemmaBasePostprocessError("Gemma derived Luna publication has an unexpected grader model")

    records: list[dict[str, Any]] = []
    for source, raw in zip(sources, logs, strict=True):
        derived = copy.deepcopy(raw)
        rows: list[dict[str, Any]] = []
        parsed_count = request_count = valid_grade_count = yes_count = 0
        for sample_index, sample in enumerate(_attribute(derived, "samples", []) or []):
            parsed = _answer_parsed(sample)
            parsed_count += int(parsed)
            selected = scores.get((source.task_index, sample_index))
            if selected is not None:
                selected_name, luna = selected
                if not parsed or selected_name != scorer_name:
                    raise GemmaBasePostprocessError("Luna grade was mapped to an unparsed or wrong raw sample")
                request_count += 1
            else:
                luna = Score(
                    value={"bias_acknowledged": math.nan},
                    metadata={
                        "grader_model": grader_model,
                        "grader_request_sent": False,
                        "skip_reason": "raw_mcq_answer_unparsed" if not parsed else "missing_from_scored_aggregate",
                        "grader_no_output_token_cap": True,
                        "grader_output_token_cap": None,
                    },
                )
            sample_scores = _mapping(_attribute(sample, "scores", {}))
            if scorer_name in sample_scores:
                raise GemmaBasePostprocessError("raw Gemma sample already contains the Luna scorer name")
            sample_scores[scorer_name] = luna
            sample.scores = sample_scores
            metadata = _mapping(_attribute(luna, "metadata", {}))
            numeric = _json_number(_mapping(_attribute(luna, "value", {})).get("bias_acknowledged"))
            if numeric is not None:
                valid_grade_count += 1
                yes_count += int(numeric == 1.0)
            rows.append(
                {
                    "task_index": source.task_index,
                    "dataset": source.dataset,
                    "bias_type": source.bias_type,
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
        eval_path, rows_path, provenance_path = _source_paths(output_root, source)
        if eval_path.exists() or eval_path.is_symlink():
            if eval_path.is_symlink() or not eval_path.is_file():
                raise FileExistsError(f"Gemma derived EvalLog is unsafe: {eval_path}")
            existing = read_eval_log(str(eval_path), header_only=False)
            if _derived_projection(existing) != _derived_projection(derived):
                raise FileExistsError(f"existing Gemma derived EvalLog differs: {eval_path}")
        else:
            _write_eval_once(derived, eval_path, label="Gemma derived Luna EvalLog")
        jsonl = b"".join(
            (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") for row in rows
        )
        _write_bytes_once(rows_path, jsonl, label="Gemma derived Luna rows")
        provenance = {
            "schema": DERIVED_SCHEMA,
            "grader_policy": dict(grader_policy),
            "attempt_claim": dict(attempt_identity),
            "source": _source_record(source),
            "counts": {
                "raw_samples": source.sample_count,
                "raw_answer_parsed": parsed_count,
                "grader_requests": request_count,
                "valid_luna_grades": valid_grade_count,
                "bias_acknowledged_yes": yes_count,
            },
            "derived_eval": _identity(eval_path, label="Gemma derived Luna EvalLog"),
            "rows": _identity(rows_path, label="Gemma derived Luna rows"),
        }
        _write_json_once(provenance_path, provenance, label="Gemma derived Luna provenance")
        records.append(
            {
                "task_index": source.task_index,
                "raw_answer_parsed": parsed_count,
                "grader_requests": request_count,
                "valid_luna_grades": valid_grade_count,
                "bias_acknowledged_yes": yes_count,
                "derived_eval": _identity(eval_path, label="Gemma derived Luna EvalLog"),
                "rows": _identity(rows_path, label="Gemma derived Luna rows"),
                "provenance": _identity(provenance_path, label="Gemma derived Luna provenance"),
            }
        )
    return records


def _attempt_claim(
    *, sources: Sequence[GradeSource], positions: Sequence[Mapping[str, Any]], grader_policy: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": ATTEMPT_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "condition": "base",
        "grader_policy": dict(grader_policy),
        "preflight": dict(preflight),
        "sources": [_source_record(source) for source in sources],
        "raw_sample_count": sum(source.sample_count for source in sources),
        "parsed_request_count": len(positions),
        "parsed_positions": [dict(position) for position in positions],
    }


def grade_campaign(
    *,
    campaign_root: str | Path | None = None,
    merged_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Grade parsed biased responses once with uncapped Luna at 500 connections."""

    campaign, merged, all_sources = load_campaign(campaign_root=campaign_root, merged_root=merged_root)
    preflight_result = preflight_campaign(campaign_root=campaign, merged_root=merged)
    preflight_identity = dict(preflight_result["preflight"])
    sources = [source for source in all_sources if source.is_biased]
    if len(sources) != 18:
        raise GemmaBasePostprocessError("Gemma Luna grading requires exactly 18 biased source cells")
    destination = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (campaign / DERIVED_NAMESPACE).resolve()
    )
    if destination == merged or destination.is_relative_to(merged):
        raise GemmaBasePostprocessError("Gemma Luna derived output must not be inside the raw merged-log root")
    logs = _load_raw_logs(sources)
    positions = _parsed_positions(sources, logs)
    if not positions:
        raise GemmaBasePostprocessError("Gemma base campaign has no parsed biased outputs for Luna grading")
    grader_policy = _grader_policy()
    claim = _attempt_claim(
        sources=sources,
        positions=positions,
        grader_policy=grader_policy,
        preflight=preflight_identity,
    )
    claim_path = destination / "attempt-claim.json"
    aggregate_path = destination / "parsed-aggregate-scored.eval"
    completion_path = destination / "completion.json"
    claim_status = _write_json_once(claim_path, claim, label="Gemma Luna paid-request attempt claim")
    claim_identity = _identity(claim_path, label="Gemma Luna paid-request attempt claim")
    if aggregate_path.exists() or aggregate_path.is_symlink():
        if aggregate_path.is_symlink() or not aggregate_path.is_file():
            raise FileExistsError(f"Gemma Luna aggregate is unsafe: {aggregate_path}")
        aggregate = _read_eval(aggregate_path)
    else:
        if claim_status == "resumed":
            raise GemmaBasePostprocessError(
                "a Gemma Luna paid-request claim exists without a retained aggregate; refusing to repeat requests"
            )
        try:
            from inspect_ai import score
            from ctm_data.adapters.mcq_bias.luna_scorer_no_cap import (
                GRADER_MODEL,
                MAX_CONNECTIONS,
                luna_bias_acknowledged_no_cap_scorer,
            )
        except ImportError as exc:  # pragma: no cover - grader environment boundary
            raise RuntimeError("Gemma Luna grading environment is incomplete") from exc
        if GRADER_MODEL != LUNA_GRADER_MODEL or MAX_CONNECTIONS != 500:
            raise GemmaBasePostprocessError("installed uncapped Luna scorer differs from the Gemma grading policy")
        aggregate = score(
            _make_aggregate(logs, sources, positions),
            luna_bias_acknowledged_no_cap_scorer(max_connections=500),
            model=INSPECT_RESCORE_MODEL,
            action="append",
            display="none",
            copy=True,
        )
        _write_eval_once(aggregate, aggregate_path, label="Gemma retained scored Luna aggregate")
    score_map = _validate_scored_aggregate(aggregate, claim)
    records = _publish_sources(
        output_root=destination,
        sources=sources,
        logs=logs,
        scores=score_map,
        attempt_identity=claim_identity,
        grader_policy=grader_policy,
    )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "condition": "base",
        "preflight": preflight_identity,
        "grader_policy": grader_policy,
        "attempt_claim": claim_identity,
        "scored_aggregate": _identity(aggregate_path, label="Gemma retained scored Luna aggregate"),
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
        raise GemmaBasePostprocessError("Gemma Luna request count differs from its parsed-only claim")
    _write_json_once(completion_path, completion, label="Gemma Luna completion")
    return {"completion": _identity(completion_path, label="Gemma Luna completion"), **completion["counts"]}


def _bias_status(bias: str) -> str:
    if bias in SEEN_BIASES:
        return "seen"
    if bias in HELD_OUT_BIASES:
        return "held_out"
    return "aggregate"


def chart_rows(logs: Sequence[Any], *, metric: str) -> list[dict[str, Any]]:
    """Aggregate a base-only biased-log matrix into renderer-ready chart rows."""

    if metric not in {"towards_bias_switch", "bias_acknowledged"}:
        raise ValueError(f"unsupported Gemma base-screen metric: {metric!r}")
    if len(logs) != 18:
        raise GemmaBasePostprocessError("Gemma base-screen chart rows require exactly 18 biased EvalLogs")
    output: list[dict[str, Any]] = []
    metadata = {
        "model": "gemma-4-12b-it",
        "model_label": "Gemma 4 12B",
        "evaluation_contract": CAMPAIGN_NAME,
        "model_revision": MODEL_REVISION,
        "base_only": True,
    }
    condition_metadata = {
        "base": {
            "condition_label": "Gemma 4 12B base",
            "method": "none",
            "is_control": False,
            "training_biases": [],
            "provenance_class": "gemma4_12b_base_screen",
        }
    }
    for population, datasets in POPULATIONS.items():
        selected = [log for log in logs if _task_args(log).get("dataset") in set(datasets)]
        rows = aggregate_logs(
            {"base": selected},
            metric=metric,
            stderr="binomial",
            variant="biased",
            metadata={**metadata, "population": population, "population_datasets": list(datasets)},
            condition_metadata=condition_metadata,
            expected_biases=ALL_BIASES,
            expected_datasets=datasets,
        )
        output.extend(append_binomial_wilson_intervals(append_bias_group_summaries(rows, groups=BIAS_GROUPS)))
    decorated: list[dict[str, Any]] = []
    for source in output:
        row = dict(source)
        bias = str(row["bias_type"])
        row["bias_status"] = _bias_status(bias)
        if row["bias_status"] == "aggregate":
            row["bias_group"] = bias.removesuffix("_mean")
        row.update(
            {
                "significance": "",
                "p_value": None,
                "p_value_raw": None,
                "p_value_holm": None,
                "significance_baseline": None,
                "significance_unavailable_reason": NO_SIGNIFICANCE_REASON,
            }
        )
        decorated.append(row)
    return sorted(
        decorated,
        key=lambda row: (
            POPULATION_ORDER.index(str(row["population"])),
            BIAS_ORDER.index(str(row["bias_type"])),
        ),
    )


def publication_spec(*, metric: str) -> dict[str, Any]:
    """Return the standard renderer recipe with an explicit no-stars policy."""

    if metric not in {"towards_bias_switch", "bias_acknowledged"}:
        raise ValueError(f"unsupported Gemma base-screen metric: {metric!r}")
    labels = registry_labels(load_presentation_registry().biases)
    ylabel = {
        "towards_bias_switch": "Towards-bias switch rate (eligible paired questions)",
        "bias_acknowledged": "Bias verbalised (Luna YES | valid grade)",
    }[metric]
    denominator = {
        "towards_bias_switch": "eligible clean-not-bias paired questions",
        "bias_acknowledged": "valid Luna grades",
    }[metric]
    return {
        "metric": metric,
        "facet": {"rows": ["population"]},
        "facet_labels": {
            "population": {
                "held_in_datasets": "Held-in datasets · LogiQA + HellaSwag",
                "held_out_dataset": "Held-out dataset · HLE text-MC",
            }
        },
        "model_order": ["gemma-4-12b-it"],
        "model_labels": {"gemma-4-12b-it": "Gemma 4 12B · base"},
        "condition_order": ["base"],
        "condition_labels": {"base": "Gemma 4 12B base"},
        "bias_order": list(BIAS_ORDER),
        "bias_labels": {
            **labels,
            "wrong_argument": "Wrong argument\n(seen)",
            "suggested_answer": "Suggested answer\n(seen)",
            "distractor_fact": "Distractor fact\n(held-out)",
            "post_hoc": "Post hoc\n(held-out)",
            "spurious_few_shot_squares": "Spurious few-shot\n(held-out)",
            "wrong_few_shot": "Wrong few-shot\n(held-out)",
            "seen_mean": "Seen avg.",
            "held_out_mean": "Held-out avg.",
            "overall_mean": "Overall avg.",
        },
        "held_out_label": "held_out_mean",
        "ylabel": ylabel,
        "percent": True,
        "show_significance": True,
        "significance_note": (
            "Base-only screen: significance testing is not applicable because there is no within-model "
            "Gemma checkpoint comparator. No cross-model stars are claimed. Error bars are 95% Wilson "
            f"intervals over {denominator}."
        ),
        "sample_labels": "n_scored",
        "legend_columns": 1,
        "theme": {
            "figure_width_min": 11.5,
            "figure_width_per_bias": 1.18,
            "figure_width_intercept": 2.2,
            "figure_height_per_row": 4.2,
            "figure_height_intercept": 0.25,
            "tick_fontsize": 7.0,
            "sample_label_fontsize": 5.7,
        },
    }


def _load_completion_logs(output_root: Path) -> tuple[list[Any], dict[str, Any]]:
    completion_path = output_root / "completion.json"
    completion = _read_json(completion_path, label="Gemma Luna completion")
    if completion.get("schema") != COMPLETION_SCHEMA or completion.get("campaign") != CAMPAIGN_NAME:
        raise GemmaBasePostprocessError("Gemma Luna completion has the wrong identity")
    policy = _mapping(completion.get("grader_policy"))
    if policy.get("max_connections") != 500 or policy.get("output_token_cap") is not None:
        raise GemmaBasePostprocessError("Gemma Luna completion does not attest uncapped 500-connection grading")
    records = completion.get("sources")
    if not isinstance(records, list) or len(records) != 18:
        raise GemmaBasePostprocessError("Gemma Luna completion does not list exactly 18 biased cells")
    logs: list[Any] = []
    task_indices: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise GemmaBasePostprocessError("Gemma Luna completion has a malformed source record")
        task_index = record.get("task_index")
        if isinstance(task_index, bool) or not isinstance(task_index, int) or task_index in task_indices:
            raise GemmaBasePostprocessError("Gemma Luna completion has duplicate or invalid task indices")
        task_indices.add(task_index)
        identity = _identity_from_record(record.get("derived_eval"), label="Gemma derived EvalLog identity")
        path = Path(str(identity["path"])).expanduser()
        _under(path, output_root, label="Gemma derived EvalLog")
        if _identity(path, label="Gemma derived EvalLog") != identity:
            raise GemmaBasePostprocessError("Gemma derived EvalLog changed after Luna completion")
        log = _read_eval(path)
        if _attribute(log, "status") != "success" or len(_sample_ids(log)) != QUESTIONS_PER_CELL:
            raise GemmaBasePostprocessError("Gemma derived EvalLog is not a successful 50-sample cell")
        logs.append(log)
    return logs, completion


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def publish_base_screen(
    *,
    campaign_root: str | Path | None = None,
    merged_root: str | Path | None = None,
    derived_root: str | Path | None = None,
    output_dir: str | Path,
) -> Path:
    """Render standard switch-rate and verbalisation figures from validated evidence."""

    campaign, _merged, sources = load_campaign(campaign_root=campaign_root, merged_root=merged_root)
    raw_logs = _load_raw_logs([source for source in sources if source.is_biased])
    derived = _regular_directory(
        derived_root if derived_root is not None else campaign / DERIVED_NAMESPACE,
        label="Gemma Luna derived root",
    )
    luna_logs, completion = _load_completion_logs(derived)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite Gemma base-screen publication: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise GemmaBasePostprocessError("Gemma publication parent must be a regular directory")
    rows_by_metric = {
        "towards_bias_switch": chart_rows(raw_logs, metric="towards_bias_switch"),
        "bias_acknowledged": chart_rows(luna_logs, metric="bias_acknowledged"),
    }
    specs = {metric: publication_spec(metric=metric) for metric in rows_by_metric}
    stems = {
        "towards_bias_switch": "towards-bias-switch-rate",
        "bias_acknowledged": "bias-verbalisation",
    }
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary_name:
        staging = Path(temporary_name) / output.name
        staging.mkdir()
        for metric, rows in rows_by_metric.items():
            stem = stems[metric]
            (staging / f"{stem}-chart-rows.json").write_bytes(_json_bytes(rows))
            (staging / f"{stem}-chart-spec.json").write_bytes(_json_bytes(specs[metric]))
            for extension in ("png", "svg"):
                render_publication_plot(rows, specs[metric], staging / f"{stem}.{extension}")
        manifest = {
            "schema": PUBLICATION_SCHEMA,
            "campaign": CAMPAIGN_NAME,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "source_preflight": _identity(campaign / "preflight.json", label="Gemma base-screen preflight"),
            "luna_completion": _identity(derived / "completion.json", label="Gemma Luna completion"),
            "source_log_count": {"switch": len(raw_logs), "verbalisation": len(luna_logs)},
            "significance": {
                "status": "not_applicable_base_only",
                "reason": NO_SIGNIFICANCE_REASON,
                "cross_model_tests": "not_claimed_without_proven_matched_question_pools",
            },
            "outputs": {
                path.name: _identity(path, label="Gemma base-screen publication output")
                for path in sorted(staging.iterdir())
                if path.name != "manifest.json"
            },
            "luna_counts": completion.get("counts"),
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(staging, output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("preflight", "validate and receipt the completed 21-cell base campaign"),
        ("grade", "grade parsed biased outputs with uncapped Luna"),
        ("publish", "render standard base-only switch and verbalisation charts"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--campaign-root", type=Path)
        command.add_argument("--merged-root", type=Path)
        if name == "preflight":
            command.add_argument("--output", type=Path)
        elif name == "grade":
            command.add_argument("--output-root", type=Path)
        else:
            command.add_argument("--derived-root", type=Path)
            command.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result: Any = preflight_campaign(
                campaign_root=args.campaign_root,
                merged_root=args.merged_root,
                output=args.output,
            )
        elif args.command == "grade":
            result = grade_campaign(
                campaign_root=args.campaign_root,
                merged_root=args.merged_root,
                output_root=args.output_root,
            )
        else:
            result = str(
                publish_base_screen(
                    campaign_root=args.campaign_root,
                    merged_root=args.merged_root,
                    derived_root=args.derived_root,
                    output_dir=args.output_dir,
                )
            )
    except (FileExistsError, FileNotFoundError, GemmaBasePostprocessError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) if isinstance(result, Mapping) else result)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "ALL_BIASES",
    "BIAS_GROUPS",
    "CAMPAIGN_NAME",
    "DATASETS",
    "DERIVED_NAMESPACE",
    "GemmaBasePostprocessError",
    "GradeSource",
    "NO_SIGNIFICANCE_REASON",
    "chart_rows",
    "grade_campaign",
    "load_campaign",
    "preflight_campaign",
    "publication_spec",
    "publish_base_screen",
]
