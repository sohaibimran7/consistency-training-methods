"""Standard, receipt-bound plots for Base versus RMCT checkpoints 16 and 176.

This is intentionally separate from :mod:`experiments.rmct_two_bias_eval.publication`.
That module is the sealed two-condition r005 adapter; this adapter compares the
receipt-selected step-16 repair publication with the archived Base and r005
step-176 references.  Bars retain each condition's available question pool;
paired tests use the exact comparison-specific overlap recorded in the
publication manifest.

The source-of-truth selection is the step-16 score-only recovery V3 preflight.
For switch rate it selects the preflight's published EvalLogs directly (six
derived repaired logs and twelve unchanged originals).  For verbalisation it
accepts a separately Luna-graded, provenance-bound version of that same
selection.  Neither mode modifies a source EvalLog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ctm_data.adapters.mcq_bias.analysis import (
    aggregate_logs,
    append_bias_group_summaries,
    append_binomial_wilson_intervals,
)
from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from ctm_data.adapters.mcq_bias.plot_registry import load_presentation_registry, registry_labels
from experiments.rmct_two_bias_eval.contract import ALL_BIASES, HELD_OUT_BIASES, SEEN_BIASES

# The fixed report order is deliberately scientific rather than filesystem
# order.  The first condition is the sole inferential baseline.
BASELINE = "base_archived"
STEP16 = "rmct_step16"
STEP176 = "rmct_step176"
CONDITIONS = (BASELINE, STEP16, STEP176)
TREATMENTS = (STEP16, STEP176)

DATASETS = ("logiqa", "hellaswag", "hle-text-mc")
POPULATION_DATASETS = {
    "held_in_datasets": ("logiqa", "hellaswag"),
    "held_out_dataset": ("hle-text-mc",),
}
POPULATION_ORDER = tuple(POPULATION_DATASETS)
BIAS_GROUPS = {
    "seen_mean": SEEN_BIASES,
    "held_out_mean": HELD_OUT_BIASES,
    "overall_mean": ALL_BIASES,
}
BIAS_ORDER = (*ALL_BIASES, *BIAS_GROUPS)

# These values are independent membership fingerprints, obtained from the
# receipt-selected Step-16 clean cells.  They deliberately use a newline
# representation so the publication manifest is easy to audit without the
# recovery implementation's private canonical-JSON helper.
STEP16_SELECTION = {
    "logiqa": {
        "count": 50,
        "newline_sorted_ids_sha256": "695cf7fe1be73c02e86b95e11d95b2e59eacb01bb36692edab2051f5af341ebf",
    },
    "hellaswag": {
        "count": 50,
        "newline_sorted_ids_sha256": "fde4076dcbcf973a9d44a8caaaaa015927c5e324460511f4ac796aa22e7ef68e",
    },
    "hle-text-mc": {
        "count": 100,
        "newline_sorted_ids_sha256": "4a3856f4dc2bb79010dcab79829d658fece822df13da85fc4caf80081e42d433",
    },
}

# Base and step-176 retain their complete 100-question pool in every dataset
# for the plotted estimates.  Only the step-16-versus-Base significance test
# contracts to the receipt-selected 50/50 IID overlap.
FULL_PAIRWISE_SELECTION_COUNTS = {
    "logiqa": 100,
    "hellaswag": 100,
    "hle-text-mc": 100,
}

# Prefer the durable local portable bundle over the ephemeral recovery scratch
# tree.  Its task directories contain the *published* bytes, named by their
# published SHA-256 (which differs from the remote filename for repaired
# logs).  The two roots are intentionally the same: ``publication_kind`` is
# still checked against the preflight and each selected byte is re-hashed.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP16_PORTABLE_ROOT = PROJECT_ROOT / (
    "artifacts/rmct-step16-luna-portable-20260821/" "rmct-convergence-step016-two-bias-v1-r002"
)
DEFAULT_STEP16_PREFLIGHT = DEFAULT_STEP16_PORTABLE_ROOT / "custody" / "corrected-native-two-bias.json"
DEFAULT_STEP16_ORIGINAL_ROOT = DEFAULT_STEP16_PORTABLE_ROOT / "staged"
DEFAULT_STEP16_DERIVED_ROOT = DEFAULT_STEP16_PORTABLE_ROOT / "staged"
STEP16_PREFLIGHT_SHA256 = "2883d83c823b44c3eb27af3c0ecbf7d8dc0f5c35f19da15021793f90570ec299"

CONDITION_METADATA: dict[str, dict[str, Any]] = {
    BASELINE: {
        "condition_label": "Archived base reference",
        "method": "none",
        "is_control": False,
        "training_biases": [],
        "provenance_class": "historical_same_task_reference",
    },
    STEP16: {
        "condition_label": "RMCT step 16",
        "method": "rate_matching",
        "is_control": False,
        "training_biases": list(SEEN_BIASES),
        "provenance_class": "receipt_selected_step16_score_recovery_r002",
    },
    STEP176: {
        "condition_label": "RMCT step 176",
        "method": "rate_matching",
        "is_control": False,
        "training_biases": list(SEEN_BIASES),
        "provenance_class": "sealed_r005_step176",
    },
}

SUPPORTED_METRICS = ("bias_acknowledged", "towards_bias_switch")
OUTPUT_STEMS = {
    "bias_acknowledged": "bias-verbalisation",
    "towards_bias_switch": "towards-bias-switch-rate",
}
# The first revision applied the Step-16 membership to every bar.  This
# revision deliberately keeps the Base and Step-176 estimates on their full
# 100-question IID pools while retaining comparison-specific paired tests.
OUTPUT_SCHEMA = "rmct-checkpoint-standard-publication-v2"
SIGNIFICANCE_METHOD = "paired_question_cluster_label_swap_randomization"
SIGNIFICANCE_PERMUTATIONS = 10_000
SIGNIFICANCE_BASE_SEED = 2_026_082_100
REPAIRED_STEP16_TASKS = (4, 6, 7, 9, 11, 13)
STEP16_LUNA_DERIVED_SCHEMA = "rmct-two-bias-step16-luna-derived-v1"


class CheckpointPublicationError(ValueError):
    """A source, selection, or chart contract is unsuitable for publication."""


@dataclass(frozen=True)
class Step16Inputs:
    """Receipt-bound Step-16 biased logs and their shared question-ID selection."""

    logs: tuple[Any, ...]
    question_ids: Mapping[str, frozenset[str]]
    source_manifest: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: str | Path, *, label: str) -> Path:
    """Resolve a regular non-symlink input only after rejecting links."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {candidate}")
    return candidate.resolve()


def _regular_directory(path: str | Path, *, label: str) -> Path:
    """Resolve a regular non-symlink input directory only after checking it."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise FileNotFoundError(f"{label} must be a regular directory: {candidate}")
    return candidate.resolve()


def _identity(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _regular_file(path, label=label)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointPublicationError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CheckpointPublicationError(f"{label} must contain a JSON object: {path}")
    return value


def _verified_step16_preflight(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Return only the immutable copied V3 selection authority."""

    identity = _identity(Path(path), label="Step-16 recovery preflight")
    if identity["sha256"] != STEP16_PREFLIGHT_SHA256:
        raise CheckpointPublicationError(
            "Step-16 recovery preflight SHA-256 differs from the immutable V3 selection authority"
        )
    return Path(str(identity["path"])), identity


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _output_identity(path: Path, *, output_dir: Path) -> dict[str, Any]:
    identity = _identity(path, label="staged publication output")
    identity["path"] = str((output_dir / path.name).resolve())
    return identity


def _normalise_dataset(value: Any) -> str:
    dataset = str(value)
    return "hle-text-mc" if dataset == "hle" else dataset


def _task_args(log: Any) -> Mapping[str, Any]:
    evaluation = getattr(log, "eval", None)
    args = getattr(evaluation, "task_args", None)
    if not isinstance(args, Mapping):
        raise CheckpointPublicationError("EvalLog has no mapping eval.task_args")
    return args


def _dataset_from_log(log: Any) -> str:
    args = _task_args(log)
    return _normalise_dataset(args.get("source_dataset", args.get("dataset", "")))


def _bias_from_log(log: Any) -> str:
    bias = _task_args(log).get("bias_type")
    if not isinstance(bias, str) or bias not in ALL_BIASES:
        raise CheckpointPublicationError(f"EvalLog has invalid biased task label: {bias!r}")
    return bias


def _sample_ids(log: Any, *, label: str) -> frozenset[str]:
    samples = list(getattr(log, "samples", None) or [])
    identifiers = [str(getattr(sample, "id", "")) for sample in samples]
    if not identifiers or any(not identifier for identifier in identifiers):
        raise CheckpointPublicationError(f"{label} has an empty sample ID")
    if len(set(identifiers)) != len(identifiers):
        raise CheckpointPublicationError(f"{label} has duplicate sample IDs")
    return frozenset(identifiers)


def _newline_selection_digest(ids: Sequence[str] | frozenset[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _canonical_id_digest(ids: Sequence[str] | frozenset[str]) -> str:
    # Matches the recovery's canonical JSON digest for its lexicographically
    # sorted source sample IDs.
    payload = json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selection_manifest(question_ids: Mapping[str, frozenset[str]]) -> dict[str, Any]:
    expected_datasets = set(STEP16_SELECTION)
    if set(question_ids) != expected_datasets:
        raise CheckpointPublicationError(
            f"Step-16 selection datasets differ; expected={sorted(expected_datasets)}, got={sorted(question_ids)}"
        )
    manifest: dict[str, Any] = {}
    for dataset, expected in STEP16_SELECTION.items():
        ids = question_ids[dataset]
        if len(ids) != expected["count"]:
            raise CheckpointPublicationError(
                f"Step-16 selection has {len(ids)} {dataset} IDs; expected {expected['count']}"
            )
        digest = _newline_selection_digest(ids)
        if digest != expected["newline_sorted_ids_sha256"]:
            raise CheckpointPublicationError(
                f"Step-16 selection digest differs for {dataset}; refusing a non-receipt-selected subset"
            )
        manifest[dataset] = {
            "count": len(ids),
            "newline_sorted_ids_sha256": digest,
            "canonical_sorted_ids_sha256": _canonical_id_digest(ids),
        }
    return manifest


def _validate_step16_preflight(document: Mapping[str, Any]) -> dict[str, Any]:
    """Use the recovery's own strict validator when it is available."""

    try:
        from infra.isambard.rmct_checkpoint_step16_score_only_publication_recovery import (
            validate_corrected_preflight,
        )
    except ImportError as exc:  # pragma: no cover - configured publication environment
        raise RuntimeError("the Step-16 score-only recovery validator is unavailable") from exc
    try:
        return validate_corrected_preflight(document)
    except Exception as exc:
        raise CheckpointPublicationError(f"Step-16 recovery preflight did not validate: {exc}") from exc


def _validated_step16_sources(preflight: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Perform adapter-local invariants after the recovery validator."""

    sources = preflight.get("sources")
    if not isinstance(sources, list) or len(sources) != 21:
        raise CheckpointPublicationError("Step-16 recovery preflight must expose all 21 selected tasks")
    output: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, Mapping) or source.get("task_index") != index:
            raise CheckpointPublicationError("Step-16 recovery preflight task ordering drifted")
        identity = source.get("identity")
        if not isinstance(identity, Sequence) or isinstance(identity, (str, bytes)) or len(identity) != 5:
            raise CheckpointPublicationError(f"Step-16 task-{index:03d} has no complete identity")
        kind, _regime, _population, dataset, bias = identity
        if kind not in {"unbiased", "biased"} or _normalise_dataset(dataset) not in DATASETS:
            raise CheckpointPublicationError(f"Step-16 task-{index:03d} has an invalid task identity")
        if kind == "biased" and bias not in ALL_BIASES:
            raise CheckpointPublicationError(f"Step-16 task-{index:03d} has an invalid bias identity")
        if kind == "unbiased" and bias is not None:
            raise CheckpointPublicationError(f"Step-16 clean task-{index:03d} unexpectedly has a bias")
        expected_kind = "derived-score-only" if index in REPAIRED_STEP16_TASKS else "original-unchanged"
        if source.get("publication_kind") != expected_kind:
            raise CheckpointPublicationError(f"Step-16 task-{index:03d} does not match its repair selection")
        published = source.get("published_log")
        if not isinstance(published, Mapping):
            raise CheckpointPublicationError(f"Step-16 task-{index:03d} has no selected published log")
        if not isinstance(published.get("path"), str) or not isinstance(published.get("sha256"), str):
            raise CheckpointPublicationError(f"Step-16 task-{index:03d} has malformed published-log identity")
        if isinstance(published.get("size_bytes"), bool) or not isinstance(published.get("size_bytes"), int):
            raise CheckpointPublicationError(f"Step-16 task-{index:03d} has malformed published-log size")
        output.append(dict(source))
    biased = [source for source in output if source["identity"][0] == "biased"]
    if len(biased) != 18:
        raise CheckpointPublicationError("Step-16 recovery preflight must select exactly 18 biased logs")
    expected_keys = {(dataset, bias) for dataset in DATASETS for bias in ALL_BIASES}
    observed_keys = {(_normalise_dataset(source["identity"][3]), str(source["identity"][4])) for source in biased}
    if observed_keys != expected_keys:
        raise CheckpointPublicationError("Step-16 recovery biased source matrix is incomplete or duplicated")
    return output


def _source_key(source: Mapping[str, Any]) -> tuple[str, str]:
    identity = source["identity"]
    return _normalise_dataset(identity[3]), str(identity[4])


def _published_local_path(
    source: Mapping[str, Any],
    *,
    original_root: Path,
    derived_root: Path,
) -> Path:
    task_index = int(source["task_index"])
    published = source["published_log"]
    remote_filename = Path(str(published["path"])).name
    published_sha256 = str(published["sha256"])
    if not remote_filename or remote_filename in {".", ".."}:
        raise CheckpointPublicationError(f"Step-16 task-{task_index:03d} has an unsafe published filename")
    root = derived_root if source["publication_kind"] == "derived-score-only" else original_root
    task_dir = root / f"task-{task_index:03d}"
    # The durable portable bundle names every source by selected content SHA,
    # whereas remote recovery-derived logs retain their legacy filename.  Use
    # the durable content-addressed candidate first, while preserving the old
    # filename as a compatibility fallback.  The caller verifies both hash and
    # size before a log is read.
    filenames = tuple(dict.fromkeys((f"{published_sha256}.eval", remote_filename)))
    for filename in filenames:
        candidate = task_dir / filename
        if candidate.exists() or candidate.is_symlink():
            try:
                candidate.resolve().relative_to(root)
            except ValueError as exc:
                raise CheckpointPublicationError(
                    f"Step-16 task-{task_index:03d} local source escapes its selected root"
                ) from exc
            return candidate
    raise FileNotFoundError(
        f"Step-16 task-{task_index:03d} selected EvalLog is absent under {task_dir}; "
        f"expected one of: {', '.join(filenames)}"
    )


def _verify_local_published_log(source: Mapping[str, Any], path: Path) -> dict[str, Any]:
    task_index = int(source["task_index"])
    actual = _identity(path, label=f"Step-16 task-{task_index:03d} selected EvalLog")
    expected = source["published_log"]
    if actual["sha256"] != expected["sha256"] or actual["size_bytes"] != expected["size_bytes"]:
        raise CheckpointPublicationError(
            f"Step-16 task-{task_index:03d} local EvalLog differs from preflight selection"
        )
    return actual


def _read_eval_log_default(path: Path) -> Any:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - configured publication environment
        raise RuntimeError("Inspect AI is required to read checkpoint publication inputs") from exc
    return read_eval_log(str(path))


def _validate_log_matches_source(log: Any, source: Mapping[str, Any], *, label: str) -> frozenset[str]:
    if getattr(log, "status", None) != "success":
        raise CheckpointPublicationError(f"{label} is not a successful EvalLog")
    dataset, bias = _source_key(source)
    if _dataset_from_log(log) != dataset or _bias_from_log(log) != bias:
        raise CheckpointPublicationError(f"{label} task identity differs from its Step-16 preflight source")
    ids = _sample_ids(log, label=label)
    if len(ids) != source["sample_count"]:
        raise CheckpointPublicationError(f"{label} sample count differs from its Step-16 preflight source")
    expected_digest = source.get("full_sample_ids_sha256")
    if isinstance(expected_digest, str) and _canonical_id_digest(ids) != expected_digest:
        raise CheckpointPublicationError(f"{label} question IDs differ from its Step-16 preflight source")
    return ids


def _verify_step16_selection(
    logs_by_source: Mapping[tuple[str, str], Any],
    sources: Sequence[Mapping[str, Any]],
) -> Mapping[str, frozenset[str]]:
    question_ids: dict[str, frozenset[str]] = {}
    for source in sources:
        if source["identity"][0] != "biased":
            continue
        key = _source_key(source)
        log = logs_by_source.get(key)
        if log is None:
            raise CheckpointPublicationError(f"Step-16 selected matrix lacks {key}")
        ids = _validate_log_matches_source(log, source, label=f"Step-16 {key[0]}/{key[1]}")
        previous = question_ids.get(key[0])
        if previous is not None and previous != ids:
            raise CheckpointPublicationError(f"Step-16 biased cells do not share exact IDs for {key[0]}")
        question_ids[key[0]] = ids
    _selection_manifest(question_ids)
    return question_ids


def _matrix_from_logs(logs: Sequence[Any], *, condition: str) -> dict[tuple[str, str], Any]:
    output: dict[tuple[str, str], Any] = {}
    for log in logs:
        if getattr(log, "status", None) != "success":
            raise CheckpointPublicationError(f"{condition} contains a non-success EvalLog")
        key = (_dataset_from_log(log), _bias_from_log(log))
        if key in output:
            raise CheckpointPublicationError(f"{condition} has duplicate biased cell {key}")
        output[key] = log
    expected = {(dataset, bias) for dataset in DATASETS for bias in ALL_BIASES}
    if set(output) != expected:
        raise CheckpointPublicationError(
            f"{condition} biased matrix is incomplete; missing={sorted(expected - set(output))}, "
            f"extra={sorted(set(output) - expected)}"
        )
    return output


def _question_id_manifest(question_ids: Mapping[str, frozenset[str]]) -> dict[str, Any]:
    """Return portable membership fingerprints without recording raw IDs."""

    if set(question_ids) != set(DATASETS):
        raise CheckpointPublicationError(
            f"question-ID manifest datasets differ; expected={list(DATASETS)}, got={sorted(question_ids)}"
        )
    return {
        dataset: {
            "count": len(question_ids[dataset]),
            "newline_sorted_ids_sha256": _newline_selection_digest(question_ids[dataset]),
            "canonical_sorted_ids_sha256": _canonical_id_digest(question_ids[dataset]),
        }
        for dataset in DATASETS
    }


def _condition_question_ids(
    logs: Sequence[Any],
    *,
    condition: str,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, frozenset[str]]:
    """Require all bias cells in one condition to share each dataset's IDs."""

    if expected_counts is not None and set(expected_counts) != set(DATASETS):
        raise CheckpointPublicationError("expected full-pool counts must cover exactly the three publication datasets")
    matrix = _matrix_from_logs(logs, condition=condition)
    output: dict[str, frozenset[str]] = {}
    for dataset in DATASETS:
        cell_ids = [_sample_ids(matrix[(dataset, bias)], label=f"{condition}/{dataset}/{bias}") for bias in ALL_BIASES]
        first = cell_ids[0]
        if any(ids != first for ids in cell_ids[1:]):
            raise CheckpointPublicationError(f"{condition} biased cells do not share exact IDs for {dataset}")
        if expected_counts is not None and len(first) != expected_counts[dataset]:
            raise CheckpointPublicationError(
                f"{condition} has {len(first)} {dataset} question IDs; expected {expected_counts[dataset]}"
            )
        output[dataset] = first
    return output


def _filtered_log_view(log: Any, *, ids: frozenset[str], condition: str, selection_label: str) -> Any:
    """Return an analysis-only view without mutating the read EvalLog."""

    selected = [sample for sample in (getattr(log, "samples", None) or []) if str(getattr(sample, "id", "")) in ids]
    observed = {str(getattr(sample, "id", "")) for sample in selected}
    if observed != set(ids) or len(selected) != len(ids):
        dataset, bias = _dataset_from_log(log), _bias_from_log(log)
        raise CheckpointPublicationError(
            f"{condition}/{dataset}/{bias} does not cover the exact {selection_label} question IDs"
        )
    source_eval = getattr(log, "eval", None)
    source_args = _task_args(log)
    args = dict(source_args)
    # Older archived HLE logs may report ``hle`` and/or ``source_dataset``;
    # normalize only the analysis view so generic aggregation sees one matrix.
    # The count is part of the standard aggregation task identity.  The three
    # bars intentionally have different available pools (Base/Step-176: 100
    # IID questions; Step-16: receipt-selected 50), so normalize the analysis
    # identity while retaining the actual samples for the estimate and its
    # denominator.  This is only an adapter view; source EvalLogs are never
    # mutated.
    args["dataset"] = _dataset_from_log(log)
    args["n_questions"] = None
    evaluation = SimpleNamespace(
        created=getattr(source_eval, "created", ""),
        task_args=args,
    )
    return SimpleNamespace(status=getattr(log, "status", None), eval=evaluation, samples=selected)


def _filter_to_question_ids(
    logs: Sequence[Any],
    *,
    question_ids: Mapping[str, frozenset[str]],
    condition: str,
    selection_label: str,
) -> list[Any]:
    """Filter a complete biased matrix to an exact comparison membership."""

    matrix = _matrix_from_logs(logs, condition=condition)
    if set(question_ids) != set(DATASETS):
        raise CheckpointPublicationError("filtering requires exactly the three publication dataset selections")
    return [
        _filtered_log_view(
            matrix[(dataset, bias)],
            ids=question_ids[dataset],
            condition=condition,
            selection_label=selection_label,
        )
        for dataset in DATASETS
        for bias in ALL_BIASES
    ]


def filter_to_step16_question_ids(
    logs: Sequence[Any],
    *,
    question_ids: Mapping[str, frozenset[str]],
    condition: str,
) -> list[Any]:
    """Filter a complete biased matrix to the exact Step-16 receipt selection."""

    return _filter_to_question_ids(
        logs,
        question_ids=question_ids,
        condition=condition,
        selection_label="receipt-selected Step-16",
    )


def _prepare_checkpoint_comparison(
    logs_by_condition: Mapping[str, Sequence[Any]],
    *,
    selection: Mapping[str, frozenset[str]],
) -> tuple[dict[str, list[Any]], dict[str, dict[str, list[Any]]], dict[str, Any]]:
    """Build condition-specific estimate and paired-test analysis views.

    The plotted estimates intentionally retain each checkpoint's available
    question pool: Base and Step-176 use the full 100/100/100 pool while
    Step-16 uses its immutable receipt-selected 50/50/100 pool.  Inference is
    comparison-specific: Step-16 versus Base is restricted to the receipt
    selection, while Step-176 versus Base uses their complete paired pool.
    """

    if set(logs_by_condition) != set(CONDITIONS):
        raise CheckpointPublicationError(f"checkpoint publication conditions must be exactly {list(CONDITIONS)}")
    receipt_selection = _selection_manifest(selection)
    step16_question_ids = _condition_question_ids(logs_by_condition[STEP16], condition=STEP16)
    if step16_question_ids != dict(selection):
        raise CheckpointPublicationError("Step-16 logs do not retain the exact receipt-selected question-ID membership")
    base_question_ids = _condition_question_ids(
        logs_by_condition[BASELINE],
        condition=BASELINE,
        expected_counts=FULL_PAIRWISE_SELECTION_COUNTS,
    )
    step176_question_ids = _condition_question_ids(
        logs_by_condition[STEP176],
        condition=STEP176,
        expected_counts=FULL_PAIRWISE_SELECTION_COUNTS,
    )
    if base_question_ids != step176_question_ids:
        raise CheckpointPublicationError(
            "Base and Step-176 full pools do not share exact question-ID membership; refusing an unpaired comparison"
        )

    # Filter even full pools to produce normalized, immutable analysis views.
    # This validates every selected ID and makes the generic renderer's task
    # identity independent of the source `n_questions` declaration.
    base_full = _filter_to_question_ids(
        logs_by_condition[BASELINE],
        question_ids=base_question_ids,
        condition=BASELINE,
        selection_label="full available Base pool",
    )
    step16_selected = _filter_to_question_ids(
        logs_by_condition[STEP16],
        question_ids=selection,
        condition=STEP16,
        selection_label="receipt-selected Step-16 pool",
    )
    step176_full = _filter_to_question_ids(
        logs_by_condition[STEP176],
        question_ids=step176_question_ids,
        condition=STEP176,
        selection_label="full available Step-176 pool",
    )
    base_step16_overlap = _filter_to_question_ids(
        logs_by_condition[BASELINE],
        question_ids=selection,
        condition=BASELINE,
        selection_label="receipt-selected Step-16 comparison overlap",
    )

    full_selection = _question_id_manifest(base_question_ids)
    bar_logs = {
        BASELINE: base_full,
        STEP16: step16_selected,
        STEP176: step176_full,
    }
    comparison_logs_by_treatment = {
        STEP16: {
            BASELINE: base_step16_overlap,
            STEP16: step16_selected,
        },
        STEP176: {
            BASELINE: base_full,
            STEP176: step176_full,
        },
    }
    membership = {
        "bar_estimates": {
            BASELINE: {
                "description": (
                    "full available Base pool: 100 questions in each of LogiQA, HellaSwag, and HLE text-MC"
                ),
                "question_ids": full_selection,
            },
            STEP16: {
                "description": (
                    "exact receipt-selected Step-16 pool: 50 LogiQA, 50 HellaSwag, and 100 HLE text-MC questions"
                ),
                "question_ids": receipt_selection,
            },
            STEP176: {
                "description": (
                    "full available Step-176 pool: 100 questions in each of LogiQA, HellaSwag, and HLE text-MC"
                ),
                "question_ids": _question_id_manifest(step176_question_ids),
            },
        },
        "pairwise_significance": {
            STEP16: {
                "baseline": BASELINE,
                "treatment": STEP16,
                "description": (
                    "The Base bar retains its full available pool; this Step-16-versus-Base test uses only the exact "
                    "receipt-selected 50 LogiQA, 50 HellaSwag, and 100 HLE text-MC question overlap."
                ),
                "question_ids": receipt_selection,
                "n_questions_total": sum(item["count"] for item in receipt_selection.values()),
            },
            STEP176: {
                "baseline": BASELINE,
                "treatment": STEP176,
                "description": (
                    "This Step-176-versus-Base test uses their full exact paired pool: 100 questions in each "
                    "of LogiQA, HellaSwag, and HLE text-MC."
                ),
                "question_ids": full_selection,
                "n_questions_total": sum(item["count"] for item in full_selection.values()),
            },
        },
    }
    return bar_logs, comparison_logs_by_treatment, membership


def load_step16_switch_inputs(
    *,
    preflight_path: str | Path,
    original_root: str | Path,
    derived_root: str | Path,
    read_eval_log: Callable[[Path], Any] | None = None,
) -> Step16Inputs:
    """Load the mixed original/derived, receipt-selected Step-16 switch logs."""

    preflight_file, preflight_identity = _verified_step16_preflight(preflight_path)
    original = _regular_directory(original_root, label="Step-16 original root")
    derived = _regular_directory(derived_root, label="Step-16 derived root")
    preflight = _validate_step16_preflight(_read_json_object(preflight_file, label="Step-16 recovery preflight"))
    sources = _validated_step16_sources(preflight)
    reader = read_eval_log or _read_eval_log_default
    logs: list[Any] = []
    source_entries: list[dict[str, Any]] = []
    for source in sources:
        if source["identity"][0] != "biased":
            continue
        path = _published_local_path(source, original_root=original, derived_root=derived)
        local_identity = _verify_local_published_log(source, path)
        logs.append(reader(path))
        source_entries.append(
            {
                "task_index": source["task_index"],
                "publication_kind": source["publication_kind"],
                "published_log": dict(source["published_log"]),
                "local_input": local_identity,
            }
        )
    matrix = _matrix_from_logs(logs, condition=STEP16)
    question_ids = _verify_step16_selection(matrix, sources)
    return Step16Inputs(
        logs=tuple(logs),
        question_ids=question_ids,
        source_manifest={
            "selection_preflight": preflight_identity,
            "receipt_selected_biased_logs": source_entries,
        },
    )


def _luna_provenance_source(path: Path) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Read the exact Step-16 Luna sidecar binding for one derived log."""

    provenance_path = path.with_suffix(".provenance.json")
    document = _read_json_object(provenance_path, label=f"Step-16 Luna provenance for {path.name}")
    if document.get("schema") != STEP16_LUNA_DERIVED_SCHEMA:
        raise CheckpointPublicationError(f"Step-16 Luna provenance has an unsupported schema: {provenance_path}")
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise CheckpointPublicationError(f"Step-16 Luna provenance has no source binding: {provenance_path}")
    raw = source.get("raw_log")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("sha256"), str):
        raise CheckpointPublicationError(
            f"Step-16 Luna provenance lacks its raw selected-source SHA-256: {provenance_path}"
        )
    return document, source, raw


def _validate_luna_source_binding(
    *,
    source: Mapping[str, Any],
    raw: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_condition: str,
    provenance_path: Path,
) -> None:
    """Fail closed unless the Luna sidecar retains the selected source identity."""

    expected_dataset, expected_bias = _source_key(expected)
    published = expected["published_log"]
    task_index = expected["task_index"]
    if (
        source.get("task_index") != task_index
        or source.get("condition") != expected_condition
        or _normalise_dataset(source.get("dataset", "")) != expected_dataset
        or source.get("bias_type") != expected_bias
        or source.get("sample_count") != expected["sample_count"]
        or raw.get("sha256") != published["sha256"]
        or raw.get("size_bytes") != published["size_bytes"]
    ):
        raise CheckpointPublicationError(
            f"Step-16 Luna provenance does not bind the receipt-selected task-{int(task_index):03d}: {provenance_path}"
        )


def load_step16_verbalisation_inputs(
    *,
    preflight_path: str | Path,
    luna_root: str | Path,
    read_eval_log: Callable[[Path], Any] | None = None,
) -> Step16Inputs:
    """Load Luna-derived Step-16 logs only when each binds the selected byte."""

    preflight_file, preflight_identity = _verified_step16_preflight(preflight_path)
    root = _regular_directory(luna_root, label="Step-16 Luna root")
    preflight = _validate_step16_preflight(_read_json_object(preflight_file, label="Step-16 recovery preflight"))
    sources = _validated_step16_sources(preflight)
    preflight_condition = preflight.get("condition")
    if not isinstance(preflight_condition, str) or not preflight_condition:
        raise CheckpointPublicationError("Step-16 recovery preflight has no condition identity")
    expected_by_key = {_source_key(source): source for source in sources if source["identity"][0] == "biased"}
    expected_by_sha = {str(source["published_log"]["sha256"]): source for source in expected_by_key.values()}
    paths = sorted(path for path in root.rglob("*-luna.eval") if not path.name.endswith("-luna-smoke.eval"))
    if len(paths) != 18:
        raise CheckpointPublicationError(
            f"Step-16 Luna root requires exactly 18 full derived EvalLogs, got {len(paths)}"
        )
    reader = read_eval_log or _read_eval_log_default
    logs_by_key: dict[tuple[str, str], Any] = {}
    source_entries: list[dict[str, Any]] = []
    for path in paths:
        identity = _identity(path, label="Step-16 Luna EvalLog")
        _document, provenance_source, raw_binding = _luna_provenance_source(path)
        bound_sha = str(raw_binding["sha256"])
        source = expected_by_sha.get(bound_sha)
        if source is None:
            raise CheckpointPublicationError(f"Step-16 Luna provenance selects an unapproved source: {path}")
        _validate_luna_source_binding(
            source=provenance_source,
            raw=raw_binding,
            expected=source,
            expected_condition=preflight_condition,
            provenance_path=path.with_suffix(".provenance.json"),
        )
        log = reader(path)
        key = (_dataset_from_log(log), _bias_from_log(log))
        if key != _source_key(source) or key in logs_by_key:
            raise CheckpointPublicationError(f"Step-16 Luna EvalLog task matrix mismatches its selected source: {path}")
        logs_by_key[key] = log
        source_entries.append(
            {
                "task_index": source["task_index"],
                "publication_kind": source["publication_kind"],
                "published_log": dict(source["published_log"]),
                "luna_eval_log": identity,
                "luna_provenance": _identity(path.with_suffix(".provenance.json"), label="Step-16 Luna provenance"),
            }
        )
    question_ids = _verify_step16_selection(logs_by_key, sources)
    return Step16Inputs(
        logs=tuple(logs_by_key[(dataset, bias)] for dataset in DATASETS for bias in ALL_BIASES),
        question_ids=question_ids,
        source_manifest={
            "selection_preflight": preflight_identity,
            "receipt_selected_biased_luna_logs": sorted(source_entries, key=lambda entry: int(entry["task_index"])),
        },
    )


def _load_logs_from_root(
    root: str | Path,
    *,
    metric: str,
    read_eval_log: Callable[[Path], Any] | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    directory = _regular_directory(root, label="publication input root")
    if metric == "bias_acknowledged":
        paths = sorted(path for path in directory.rglob("*-luna.eval") if not path.name.endswith("-luna-smoke.eval"))
    elif metric == "towards_bias_switch":
        paths = sorted(
            path for path in directory.rglob("*.eval") if not path.name.endswith(("-luna.eval", "-luna-smoke.eval"))
        )
    else:
        raise CheckpointPublicationError(f"unsupported metric: {metric!r}")
    if len(paths) != 18:
        raise CheckpointPublicationError(
            f"publication input root requires exactly 18 EvalLogs, got {len(paths)}: {directory}"
        )
    reader = read_eval_log or _read_eval_log_default
    logs = [reader(path) for path in paths]
    _matrix_from_logs(logs, condition=str(directory))
    return logs, [_identity(path, label="publication input EvalLog") for path in paths]


def _sample_metric(sample: Any, metric: str) -> float | None:
    matches: list[float] = []
    for score in (getattr(sample, "scores", None) or {}).values():
        value = getattr(score, "value", None)
        if not isinstance(value, Mapping) or metric not in value:
            continue
        candidate = value[metric]
        if candidate is None or isinstance(candidate, bool):
            continue
        try:
            parsed = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            matches.append(parsed)
    if len(matches) > 1:
        raise CheckpointPublicationError(
            f"sample {getattr(sample, 'id', '<unknown>')!r} has duplicate {metric!r} scores"
        )
    return matches[0] if matches else None


def _cluster_counts(
    logs: Sequence[Any],
    *,
    datasets: Sequence[str],
    biases: Sequence[str],
    metric: str,
) -> dict[str, tuple[int, int]]:
    dataset_set, bias_set = set(datasets), set(biases)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    seen: set[tuple[str, str, str]] = set()
    for log in logs:
        dataset, bias = _dataset_from_log(log), _bias_from_log(log)
        if dataset not in dataset_set or bias not in bias_set:
            continue
        for sample in getattr(log, "samples", None) or []:
            question_id = str(getattr(sample, "id", ""))
            if not question_id:
                raise CheckpointPublicationError("paired significance requires non-empty sample IDs")
            observation = (dataset, bias, question_id)
            if observation in seen:
                raise CheckpointPublicationError(
                    f"paired significance encountered duplicate observation: {observation}"
                )
            seen.add(observation)
            cluster = f"{dataset}:{question_id}"
            counts[cluster]  # retain zero-denominator question clusters
            value = _sample_metric(sample, metric)
            if value is None:
                continue
            if value not in {0.0, 1.0}:
                raise CheckpointPublicationError(f"paired significance requires a binary metric, got {value}")
            counts[cluster][0] += int(value)
            counts[cluster][1] += 1
    if not counts:
        raise CheckpointPublicationError("paired significance selected no question clusters")
    return {key: (value[0], value[1]) for key, value in counts.items()}


def _derived_significance_seed(*, treatment: str, metric: str, population: str, bias_type: str) -> int:
    payload = f"{SIGNIFICANCE_BASE_SEED}|{BASELINE}|{treatment}|{metric}|{population}|{bias_type}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _paired_label_swap(
    treatment: Mapping[str, tuple[int, int]],
    baseline: Mapping[str, tuple[int, int]],
    *,
    treatment_name: str,
    metric: str,
    population: str,
    bias_type: str,
    permutations: int,
) -> dict[str, Any]:
    if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 1:
        raise CheckpointPublicationError("significance permutations must be a positive integer")
    if set(treatment) != set(baseline):
        raise CheckpointPublicationError(
            f"paired significance has misaligned question clusters for {treatment_name}/{population}/{bias_type}"
        )
    question_ids = sorted(treatment)
    treatment_counts = np.asarray([treatment[key] for key in question_ids], dtype=np.int64)
    baseline_counts = np.asarray([baseline[key] for key in question_ids], dtype=np.int64)
    treatment_total, baseline_total = treatment_counts.sum(axis=0), baseline_counts.sum(axis=0)
    if treatment_total[1] <= 0 or baseline_total[1] <= 0:
        raise CheckpointPublicationError(
            f"paired significance has a zero denominator for {treatment_name}/{population}/{bias_type}"
        )
    observed = float(treatment_total[0] / treatment_total[1] - baseline_total[0] / baseline_total[1])
    seed = _derived_significance_seed(
        treatment=treatment_name,
        metric=metric,
        population=population,
        bias_type=bias_type,
    )
    rng = np.random.default_rng(seed)
    total_counts = treatment_counts + baseline_counts
    combined_total = total_counts.sum(axis=0)
    extreme, valid, drawn = 0, 0, 0
    observed_abs = abs(observed)
    tolerance = np.finfo(np.float64).eps * max(1.0, observed_abs) * 8.0
    while valid < permutations:
        batch = min(1_000, permutations - valid)
        swaps = rng.integers(0, 2, size=(batch, len(question_ids)), dtype=np.int8).astype(bool)
        randomized_treatment = np.where(swaps[..., np.newaxis], baseline_counts, treatment_counts).sum(axis=1)
        randomized_baseline = combined_total - randomized_treatment
        usable = (randomized_treatment[:, 1] > 0) & (randomized_baseline[:, 1] > 0)
        statistics = (
            randomized_treatment[usable, 0] / randomized_treatment[usable, 1]
            - randomized_baseline[usable, 0] / randomized_baseline[usable, 1]
        )
        statistics = statistics[: permutations - valid]
        extreme += int(np.count_nonzero(np.abs(statistics) >= observed_abs - tolerance))
        valid += len(statistics)
        drawn += batch
        if drawn > permutations * 100:
            raise CheckpointPublicationError("paired significance could not draw enough valid swaps")
    return {
        "significance_method": SIGNIFICANCE_METHOD,
        "significance_baseline": BASELINE,
        "significance_resampling_unit": "question_id",
        "significance_sidedness": "two_sided",
        "significance_statistic": "difference_in_rates",
        "question_clusters": len(question_ids),
        "observed_difference": observed,
        "permutations_requested": permutations,
        "permutations_valid": valid,
        "permutations_drawn": drawn,
        "significance_seed": seed,
        "p_value_raw": (extreme + 1) / (valid + 1),
    }


def _significance_marker(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _append_paired_significance(
    rows: Sequence[Mapping[str, Any]],
    *,
    significance_logs_by_treatment: Mapping[str, Mapping[str, Sequence[Any]]],
    significance_metadata_by_treatment: Mapping[str, Mapping[str, Any]] | None,
    metric: str,
    permutations: int,
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    treatment_rows: list[dict[str, Any]] = []
    for row in output:
        condition = str(row["condition"])
        row["significance_baseline"] = BASELINE
        if condition == BASELINE:
            row.update(
                {
                    "p_value": None,
                    "p_value_raw": None,
                    "p_value_holm": None,
                    "significance": "",
                    "significance_unavailable_reason": "baseline_cell",
                }
            )
            continue
        if condition not in TREATMENTS:
            raise CheckpointPublicationError(f"unexpected non-baseline condition in chart rows: {condition}")
        comparison_logs = significance_logs_by_treatment.get(condition)
        if not isinstance(comparison_logs, Mapping) or set(comparison_logs) != {BASELINE, condition}:
            raise CheckpointPublicationError(
                f"paired significance requires exactly Base and {condition} logs for the comparison"
            )
        metadata = (
            significance_metadata_by_treatment.get(condition, {})
            if significance_metadata_by_treatment is not None
            else {}
        )
        if not isinstance(metadata, Mapping):
            raise CheckpointPublicationError(f"significance metadata for {condition} must be a mapping")
        biases = row.get("component_biases") or [row["bias_type"]]
        population = str(row["population"])
        datasets = POPULATION_DATASETS[population]
        treatment_counts = _cluster_counts(comparison_logs[condition], datasets=datasets, biases=biases, metric=metric)
        baseline_counts = _cluster_counts(comparison_logs[BASELINE], datasets=datasets, biases=biases, metric=metric)
        question_counts_by_dataset = {
            dataset: sum(cluster.startswith(f"{dataset}:") for cluster in treatment_counts) for dataset in datasets
        }
        row.update(
            _paired_label_swap(
                treatment_counts,
                baseline_counts,
                treatment_name=condition,
                metric=metric,
                population=population,
                bias_type=str(row["bias_type"]),
                permutations=permutations,
            )
        )
        row["significance_analysis_membership"] = str(metadata.get("membership", "condition_input_membership"))
        row["significance_analysis_question_counts_by_dataset"] = question_counts_by_dataset
        row["significance_analysis_n_questions"] = sum(question_counts_by_dataset.values())
        treatment_rows.append(row)

    # Both checkpoint-vs-Base families are corrected together: 2 checkpoints
    # x 2 population panels x 9 displayed bias cells = 36 hypotheses.
    ranked = sorted(
        treatment_rows,
        key=lambda row: (
            float(row["p_value_raw"]),
            CONDITIONS.index(str(row["condition"])),
            POPULATION_ORDER.index(str(row["population"])),
            BIAS_ORDER.index(str(row["bias_type"])),
        ),
    )
    family_size, running = len(ranked), 0.0
    if family_size != 36:
        raise CheckpointPublicationError(
            f"checkpoint publication must Holm-correct 36 treatment cells, got {family_size}"
        )
    for rank, row in enumerate(ranked):
        adjusted = min(1.0, max(running, (family_size - rank) * float(row["p_value_raw"])))
        running = adjusted
        row.update(
            {
                "p_value": adjusted,
                "p_value_holm": adjusted,
                "significance": _significance_marker(adjusted),
                "significance_multiplicity": "holm_across_36_displayed_checkpoint_vs_base_cells",
                "holm_family_size": family_size,
            }
        )
    return output


def chart_rows(
    logs_by_condition: Mapping[str, Sequence[Any]],
    *,
    metric: str = "towards_bias_switch",
    significance_permutations: int = SIGNIFICANCE_PERMUTATIONS,
    significance_logs_by_treatment: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
    significance_metadata_by_treatment: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate checkpoint bars and add paired, whole-question significance.

    ``logs_by_condition`` controls the plotted estimates.  Optional
    ``significance_logs_by_treatment`` independently supplies the exact
    paired membership for each checkpoint-versus-Base test.
    """

    if set(logs_by_condition) != set(CONDITIONS):
        raise CheckpointPublicationError(f"checkpoint publication conditions must be exactly {list(CONDITIONS)}")
    if metric not in SUPPORTED_METRICS:
        raise CheckpointPublicationError(f"unsupported checkpoint publication metric: {metric!r}")
    rows: list[dict[str, Any]] = []
    for population, datasets in POPULATION_DATASETS.items():
        dataset_set = set(datasets)
        population_logs = {
            condition: [log for log in logs if _dataset_from_log(log) in dataset_set]
            for condition, logs in logs_by_condition.items()
        }
        population_rows = aggregate_logs(
            population_logs,
            metric=metric,
            stderr="binomial",
            variant="biased",
            metadata={
                "model": "qwen3.5-9b",
                "model_label": "Qwen 3.5 9B",
                "evaluation_contract": "condition_specific_checkpoint_estimates_with_paired_membership_tests",
                "population": population,
                "population_datasets": list(datasets),
            },
            condition_metadata=CONDITION_METADATA,
            expected_biases=ALL_BIASES,
            expected_datasets=datasets,
        )
        rows.extend(append_binomial_wilson_intervals(append_bias_group_summaries(population_rows, groups=BIAS_GROUPS)))
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        bias = str(row["bias_type"])
        if bias in SEEN_BIASES:
            row["bias_status"] = "seen"
        elif bias in HELD_OUT_BIASES:
            row["bias_status"] = "held_out"
        else:
            row["bias_status"] = "aggregate"
            row["bias_group"] = bias.removesuffix("_mean")
        output.append(row)
    ordered = sorted(
        output,
        key=lambda row: (
            POPULATION_ORDER.index(str(row["population"])),
            CONDITIONS.index(str(row["condition"])),
            BIAS_ORDER.index(str(row["bias_type"])),
        ),
    )
    comparison_logs = significance_logs_by_treatment or {
        treatment: {
            BASELINE: logs_by_condition[BASELINE],
            treatment: logs_by_condition[treatment],
        }
        for treatment in TREATMENTS
    }
    return _append_paired_significance(
        ordered,
        significance_logs_by_treatment=comparison_logs,
        significance_metadata_by_treatment=significance_metadata_by_treatment,
        metric=metric,
        permutations=significance_permutations,
    )


def publication_spec(*, metric: str = "towards_bias_switch") -> dict[str, Any]:
    """Return the standard renderer recipe with distinct checkpoint styles."""

    if metric not in SUPPORTED_METRICS:
        raise CheckpointPublicationError(f"unsupported checkpoint publication metric: {metric!r}")
    labels = registry_labels(load_presentation_registry().biases)
    ylabel = {
        "bias_acknowledged": "Bias verbalised (Luna YES | valid grade)",
        "towards_bias_switch": "Towards-bias switch rate (eligible paired questions)",
    }[metric]
    uncertainty_note = {
        "bias_acknowledged": "valid Luna grades",
        "towards_bias_switch": "eligible clean-not-bias paired questions",
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
        "model_order": ["qwen3.5-9b"],
        "model_labels": {"qwen3.5-9b": "Qwen 3.5 9B · biased prompts"},
        "condition_order": list(CONDITIONS),
        "condition_labels": {
            condition: str(CONDITION_METADATA[condition]["condition_label"]) for condition in CONDITIONS
        },
        "condition_styles": {
            BASELINE: {"color": "#9aa0a6"},
            STEP16: {"color": "#6fa8dc"},
            STEP176: {"color": "#8cc39a"},
        },
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
            "Archived base is a same-task historical reference without the sealed checkpoint runtime identity; "
            "comparisons are descriptive. "
            f"Error bars are 95% Wilson intervals over {uncertainty_note}. "
            "Bars retain each condition's available question pool; Step-16-versus-base tests use its receipt-selected "
            "overlap, while Step-176-versus-base tests use their full paired pool. "
            "Stars compare each RMCT checkpoint with base using two-sided paired whole-question label-swap "
            "tests with Holm correction across the 36 displayed checkpoint-vs-base treatment cells. "
            "Key: * adjusted p<0.05; ** p<0.01; *** p<0.001; no star means adjusted p≥0.05."
        ),
        "sample_labels": "n_scored",
        "legend_columns": 3,
        "theme": {
            "figure_width_min": 12.0,
            "figure_width_per_bias": 1.18,
            "figure_width_intercept": 2.2,
            "figure_height_per_row": 4.2,
            "figure_height_intercept": 0.25,
            "tick_fontsize": 7.0,
            "sample_label_fontsize": 5.7,
        },
    }


def render_checkpoint_comparison(
    *,
    logs_by_condition: Mapping[str, Sequence[Any]],
    selection: Mapping[str, frozenset[str]],
    source_manifest: Mapping[str, Any],
    output_dir: str | Path,
    metric: str = "towards_bias_switch",
    significance_permutations: int = SIGNIFICANCE_PERMUTATIONS,
) -> Path:
    """Atomically write standard rows, spec, provenance manifest, PNG, and SVG."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint publication output: {output}")
    selection_document = _selection_manifest(selection)
    bar_logs, comparison_logs, analysis_membership = _prepare_checkpoint_comparison(
        logs_by_condition,
        selection=selection,
    )
    rows = chart_rows(
        bar_logs,
        metric=metric,
        significance_permutations=significance_permutations,
        significance_logs_by_treatment=comparison_logs,
        significance_metadata_by_treatment={
            STEP16: {"membership": "exact_receipt_selected_step16_base_overlap"},
            STEP176: {"membership": "full_exact_base_step176_paired_pool"},
        },
    )
    spec = publication_spec(metric=metric)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        (staging / "chart-rows.json").write_bytes(_json_bytes(rows))
        (staging / "chart-spec.json").write_bytes(_json_bytes(spec))
        output_stem = OUTPUT_STEMS[metric]
        for extension in ("png", "svg"):
            render_publication_plot(rows, spec, staging / f"{output_stem}.{extension}")
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "metric": metric,
            "conditions": CONDITION_METADATA,
            "selection": {
                "description": "immutable exact Step-16 receipt-selected question-ID authority",
                "question_ids": selection_document,
            },
            "analysis_membership": analysis_membership,
            "bias_groups": {name: list(values) for name, values in BIAS_GROUPS.items()},
            "dataset_populations": {name: list(values) for name, values in POPULATION_DATASETS.items()},
            "significance": {
                "method": SIGNIFICANCE_METHOD,
                "baseline": BASELINE,
                "comparisons": {condition: BASELINE for condition in TREATMENTS},
                "permutations": significance_permutations,
                "multiplicity": "holm_across_36_displayed_checkpoint_vs_base_cells",
                "holm_family_size": 36,
                "comparison_specific_question_membership": analysis_membership["pairwise_significance"],
            },
            "source_inputs": source_manifest,
            "comparison_caveat": (
                "Archived base is a same-task historical reference without the sealed checkpoint runtime identity; "
                "comparisons are descriptive."
            ),
            "outputs": {
                path.name: _output_identity(path, output_dir=output)
                for path in sorted(staging.iterdir())
                if path.name != "manifest.json"
            },
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(staging, output)
    return output


def _assemble_cli_inputs(
    *,
    metric: str,
    base_logs: Path,
    step176_logs: Path,
    step16_preflight: Path,
    step16_original_root: Path | None,
    step16_derived_root: Path | None,
    step16_logs: Path | None,
) -> tuple[dict[str, list[Any]], Mapping[str, frozenset[str]], dict[str, Any]]:
    base, base_identities = _load_logs_from_root(base_logs, metric=metric)
    step176, step176_identities = _load_logs_from_root(step176_logs, metric=metric)
    if metric == "towards_bias_switch":
        if step16_original_root is None or step16_derived_root is None:
            raise CheckpointPublicationError(
                "switch-rate publication requires --step16-original-root and --step16-derived-root"
            )
        step16 = load_step16_switch_inputs(
            preflight_path=step16_preflight,
            original_root=step16_original_root,
            derived_root=step16_derived_root,
        )
    else:
        if step16_logs is None:
            raise CheckpointPublicationError(
                "verbalisation publication requires --step16-logs with 18 provenance-bound Luna EvalLogs"
            )
        step16 = load_step16_verbalisation_inputs(
            preflight_path=step16_preflight,
            luna_root=step16_logs,
        )
    return (
        {BASELINE: base, STEP16: list(step16.logs), STEP176: step176},
        step16.question_ids,
        {
            BASELINE: base_identities,
            STEP16: dict(step16.source_manifest),
            STEP176: step176_identities,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-logs", required=True, type=Path, help="18 Base raw or Luna EvalLogs for the selected metric"
    )
    parser.add_argument(
        "--step176-logs", required=True, type=Path, help="18 Step-176 raw or Luna EvalLogs for the selected metric"
    )
    parser.add_argument(
        "--step16-preflight",
        type=Path,
        default=DEFAULT_STEP16_PREFLIGHT,
        help="authoritative V3 Step-16 score-only recovery preflight",
    )
    parser.add_argument(
        "--step16-original-root",
        type=Path,
        default=DEFAULT_STEP16_ORIGINAL_ROOT,
        help="task-NNN root containing original-published Step-16 biased EvalLogs; may equal the portable staged root",
    )
    parser.add_argument(
        "--step16-derived-root",
        type=Path,
        default=DEFAULT_STEP16_DERIVED_ROOT,
        help="task-NNN root containing repair-published Step-16 EvalLogs; may equal the portable staged root",
    )
    parser.add_argument(
        "--step16-logs",
        type=Path,
        help="18 provenance-bound Step-16 Luna EvalLogs; required for bias_acknowledged",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metric", choices=SUPPORTED_METRICS, default="towards_bias_switch")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        logs, selection, sources = _assemble_cli_inputs(
            metric=args.metric,
            base_logs=args.base_logs,
            step176_logs=args.step176_logs,
            step16_preflight=args.step16_preflight,
            step16_original_root=args.step16_original_root,
            step16_derived_root=args.step16_derived_root,
            step16_logs=args.step16_logs,
        )
        output = render_checkpoint_comparison(
            logs_by_condition=logs,
            selection=selection,
            source_manifest=sources,
            output_dir=args.output_dir,
            metric=args.metric,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _parser().error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "BASELINE",
    "BIAS_GROUPS",
    "CONDITIONS",
    "DEFAULT_STEP16_DERIVED_ROOT",
    "DEFAULT_STEP16_ORIGINAL_ROOT",
    "DEFAULT_STEP16_PREFLIGHT",
    "DEFAULT_STEP16_PORTABLE_ROOT",
    "FULL_PAIRWISE_SELECTION_COUNTS",
    "REPAIRED_STEP16_TASKS",
    "SIGNIFICANCE_PERMUTATIONS",
    "STEP16",
    "STEP16_PREFLIGHT_SHA256",
    "STEP16_SELECTION",
    "STEP176",
    "CheckpointPublicationError",
    "Step16Inputs",
    "chart_rows",
    "filter_to_step16_question_ids",
    "load_step16_switch_inputs",
    "load_step16_verbalisation_inputs",
    "publication_spec",
    "render_checkpoint_comparison",
]
