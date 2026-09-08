"""Standard significance-marked Muse Glimmer RMCT checkpoint plots.

The four conditions are the pinned base snapshot, global/data steps 16 and 64
(with separately custody-bound realized optimizer counts), and the
trajectory's receipt-selected final checkpoint.  All estimates and
paired tests use the same frozen 50 LogiQA, 50 HellaSwag, and 100 HLE text-MC
question membership.  Switch inputs come only from completed two-bias
campaign preflights; verbalisation inputs come only from completed full,
parsed-only, no-cap Luna publications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ctm_data.adapters.mcq_bias.analysis import (
    aggregate_logs,
    append_bias_group_summaries,
    append_binomial_wilson_intervals,
)
from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from ctm_data.adapters.mcq_bias.plot_registry import load_presentation_registry, registry_labels
from experiments.muse_glimmer_rmct_replication import luna_grade_no_cap
from experiments.rmct_two_bias_eval import checkpoint_publication as standard
from experiments.rmct_two_bias_eval.contract import ALL_BIASES, HELD_OUT_BIASES, SEEN_BIASES
from infra.isambard import run_muse_glimmer_rmct_two_bias_evals_16gpu as muse_eval


BASELINE = "base"
STEP16 = "step016"
STEP64 = "step064"
FINAL = "final"
CONDITIONS = (BASELINE, STEP16, STEP64, FINAL)
TREATMENTS = (STEP16, STEP64, FINAL)
GROUP_CONDITIONS = {"early": (BASELINE, STEP16), "late": (STEP64, FINAL)}
DATASETS = ("logiqa", "hellaswag", "hle-text-mc")
EXPECTED_QUESTION_COUNTS = {"logiqa": 50, "hellaswag": 50, "hle-text-mc": 100}
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
CONDITION_METADATA: dict[str, dict[str, Any]] = {
    BASELINE: {
        "condition_label": "Muse Glimmer base",
        "method": "none",
        "is_control": False,
        "training_biases": [],
        "provenance_class": "pinned_base_snapshot",
    },
    STEP16: {
        "condition_label": "Muse Glimmer RMCT global/data step 16",
        "method": "rate_matching",
        "is_control": False,
        "training_biases": list(SEEN_BIASES),
        "provenance_class": "sealed_raw_training_checkpoint_step16",
    },
    STEP64: {
        "condition_label": "Muse Glimmer RMCT global/data step 64",
        "method": "rate_matching",
        "is_control": False,
        "training_biases": list(SEEN_BIASES),
        "provenance_class": "sealed_raw_training_checkpoint_step64",
    },
    FINAL: {
        "condition_label": "Muse Glimmer RMCT final",
        "method": "rate_matching",
        "is_control": False,
        "training_biases": list(SEEN_BIASES),
        "provenance_class": "receipt_selected_final_checkpoint",
    },
}
SUPPORTED_METRICS = ("bias_acknowledged", "towards_bias_switch")
OUTPUT_STEMS = {
    "bias_acknowledged": "bias-verbalisation",
    "towards_bias_switch": "towards-bias-switch-rate",
}
OUTPUT_SCHEMA = "muse-glimmer-rmct-standard-checkpoint-publication-v1"
SIGNIFICANCE_METHOD = "paired_question_cluster_label_swap_randomization"
SIGNIFICANCE_PERMUTATIONS = 10_000
SIGNIFICANCE_BASE_SEED = 2_026_082_400
HOLM_FAMILY_SIZE = len(TREATMENTS) * len(POPULATION_DATASETS) * len(BIAS_ORDER)


class MusePublicationError(ValueError):
    """The supplied Muse results cannot support the frozen publication."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size < 1:
        raise MusePublicationError(f"{label} must be a non-empty regular file: {candidate}")
    resolved = candidate.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved), "size_bytes": resolved.stat().st_size}


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    _identity(candidate, label=label)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MusePublicationError(f"invalid {label}: {candidate}") from exc
    if not isinstance(value, dict):
        raise MusePublicationError(f"{label} must contain an object")
    return value


def _read_eval(path: Path) -> Any:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - runtime boundary
        raise MusePublicationError("Inspect AI is required to read Muse EvalLogs") from exc
    return read_eval_log(str(path), header_only=False)


def _validate_matrix(logs: Sequence[Any], *, condition: str) -> dict[str, frozenset[str]]:
    try:
        standard._matrix_from_logs(logs, condition=condition)
        ids = standard._condition_question_ids(
            logs,
            condition=condition,
            expected_counts=EXPECTED_QUESTION_COUNTS,
        )
    except (TypeError, ValueError) as exc:
        raise MusePublicationError(str(exc)) from exc
    return ids


def _raw_source_path(source: Mapping[str, Any], *, condition: str) -> tuple[Path, dict[str, Any]]:
    path_value = source.get("raw_log")
    expected_sha = source.get("raw_log_sha256")
    if not isinstance(path_value, str) or not isinstance(expected_sha, str):
        raise MusePublicationError(f"{condition} preflight source lacks its raw-log identity")
    identity = _identity(path_value, label=f"{condition} raw EvalLog")
    if identity["sha256"] != expected_sha:
        raise MusePublicationError(f"{condition} raw EvalLog changed after preflight")
    return Path(identity["path"]), identity


def load_switch_group(group: str, campaign_root: str | Path) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Load two receipt-selected raw conditions from a completed campaign."""

    if group not in GROUP_CONDITIONS:
        raise MusePublicationError(f"unknown Muse publication group: {group!r}")
    muse_eval._configure(group)
    audited = muse_eval.audited
    try:
        paths = audited._campaign_paths(campaign_root)
        completion = audited._read_json(paths.completion, label=f"Muse {group} campaign completion")
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise MusePublicationError(str(exc)) from exc
    if (
        completion.get("schema") != audited.COMPLETION_SCHEMA
        or completion.get("campaign") != audited.CAMPAIGN_NAME
        or completion.get("total_generations") != 2 * audited.TOTAL_SAMPLES_PER_CONDITION
    ):
        raise MusePublicationError(f"Muse {group} campaign completion has the wrong schema or generation count")
    completion_rows = completion.get("conditions")
    if not isinstance(completion_rows, list):
        raise MusePublicationError(f"Muse {group} campaign completion lacks condition receipts")
    rows_by_name = {
        str(row.get("name")): row for row in completion_rows if isinstance(row, Mapping)
    }
    if set(rows_by_name) != set(GROUP_CONDITIONS[group]):
        raise MusePublicationError(f"Muse {group} completion contains the wrong conditions")

    logs_by_condition: dict[str, list[Any]] = {}
    sources_by_condition: dict[str, Any] = {}
    for condition_name in GROUP_CONDITIONS[group]:
        condition = audited._condition(condition_name)
        condition_paths = audited._condition_paths(paths, condition)
        expected_preflight_identity = audited._identity(
            condition_paths.preflight,
            label=f"Muse {group}/{condition_name} preflight",
        )
        if rows_by_name[condition_name].get("preflight") != expected_preflight_identity:
            raise MusePublicationError(f"Muse {group}/{condition_name} completion does not bind its preflight")
        try:
            report = audited.validate_preflight_report(condition_paths.preflight)
        except (OSError, TypeError, ValueError) as exc:
            raise MusePublicationError(str(exc)) from exc
        selected = [source for source in report["sources"] if source.get("kind") == "biased"]
        if len(selected) != 18 or {source.get("task_index") for source in selected} != set(range(4, 22)):
            raise MusePublicationError(f"Muse {group}/{condition_name} preflight lacks exactly 18 biased sources")
        paths_and_identities = [_raw_source_path(source, condition=condition_name) for source in selected]
        logs = [_read_eval(path) for path, _identity_record in paths_and_identities]
        _validate_matrix(logs, condition=condition_name)
        logs_by_condition[condition_name] = logs
        sources_by_condition[condition_name] = {
            "campaign_completion": _identity(paths.completion, label=f"Muse {group} completion"),
            "preflight": expected_preflight_identity,
            "raw_eval_logs": [identity for _path, identity in paths_and_identities],
        }
    return logs_by_condition, sources_by_condition


def _require_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MusePublicationError(f"{label} lacks a file identity")
    identity = _identity(str(value.get("path", "")), label=label)
    if dict(value) != identity:
        raise MusePublicationError(f"{label} changed after publication")
    return identity


def _validate_luna_policy(policy: Any) -> None:
    if not isinstance(policy, Mapping):
        raise MusePublicationError("Muse Luna completion lacks grader policy")
    if (
        policy.get("grader_model") != luna_grade_no_cap.GRADER_MODEL
        or policy.get("max_connections") != 500
        or policy.get("aggregate_connection_limit") != 500
        or policy.get("grader_output_token_cap") is not None
        or policy.get("grader_reasoning_token_cap") is not None
        or policy.get("provider_default_output_token_cap_required") is not None
        or policy.get("parsed_raw_answers_only") is not True
    ):
        raise MusePublicationError("Muse Luna completion differs from parsed-only 500-connection no-cap policy")


def load_luna_group(group: str, output_root: str | Path) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Load two full, parsed-only, no-cap Luna condition publications."""

    if group not in GROUP_CONDITIONS:
        raise MusePublicationError(f"unknown Muse Luna publication group: {group!r}")
    root = Path(output_root).expanduser().resolve()
    logs_by_condition: dict[str, list[Any]] = {}
    sources_by_condition: dict[str, Any] = {}
    for condition in GROUP_CONDITIONS[group]:
        completion_path = root / "full" / group / condition / "_condition" / "completion.json"
        completion = _read_json(completion_path, label=f"Muse Luna {group}/{condition} completion")
        if (
            completion.get("schema") != luna_grade_no_cap.COMPLETION_SCHEMA
            or completion.get("mode") != "full"
            or completion.get("group") != group
            or completion.get("condition") != condition
        ):
            raise MusePublicationError(f"Muse Luna {group}/{condition} completion has the wrong identity")
        _validate_luna_policy(completion.get("grader_policy"))
        records = completion.get("sources")
        counts = completion.get("counts")
        if not isinstance(records, list) or len(records) != 18 or not isinstance(counts, Mapping):
            raise MusePublicationError(f"Muse Luna {group}/{condition} completion is not an 18-cell matrix")
        if (
            counts.get("biased_source_logs") != 18
            or counts.get("grader_requests") != counts.get("raw_answer_parsed")
            or counts.get("valid_luna_grades", 0) > counts.get("grader_requests", -1)
        ):
            raise MusePublicationError(f"Muse Luna {group}/{condition} is not a full parsed-only grade")
        task_indices = {record.get("task_index") for record in records if isinstance(record, Mapping)}
        if task_indices != set(range(4, 22)):
            raise MusePublicationError(f"Muse Luna {group}/{condition} lacks task indices 4..21")
        logs: list[Any] = []
        identities: list[dict[str, Any]] = []
        for record in sorted(records, key=lambda item: int(item["task_index"])):
            derived_identity = _require_identity(record.get("derived_eval"), label="Muse Luna derived EvalLog")
            provenance_identity = _require_identity(record.get("provenance"), label="Muse Luna provenance")
            provenance = _read_json(provenance_identity["path"], label="Muse Luna provenance")
            if (
                provenance.get("schema") != luna_grade_no_cap.SOURCE_PROVENANCE_SCHEMA
                or provenance.get("mode") != "full"
                or provenance.get("derived_eval") != derived_identity
                or provenance.get("grader_policy") != completion.get("grader_policy")
            ):
                raise MusePublicationError("Muse Luna derived source is not bound to its full no-cap completion")
            source = provenance.get("source")
            if (
                not isinstance(source, Mapping)
                or source.get("condition") != condition
                or source.get("task_index") != record.get("task_index")
            ):
                raise MusePublicationError("Muse Luna provenance has the wrong condition/task source")
            logs.append(_read_eval(Path(derived_identity["path"])))
            identities.append({"derived_eval": derived_identity, "provenance": provenance_identity})
        _validate_matrix(logs, condition=condition)
        logs_by_condition[condition] = logs
        sources_by_condition[condition] = {
            "condition_completion": _identity(completion_path, label=f"Muse Luna {group}/{condition} completion"),
            "derived_sources": identities,
        }
    return logs_by_condition, sources_by_condition


def load_inputs(
    *,
    metric: str,
    early_campaign_root: str | Path,
    late_campaign_root: str | Path,
    luna_output_root: str | Path | None = None,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    if metric not in SUPPORTED_METRICS:
        raise MusePublicationError(f"unsupported Muse publication metric: {metric!r}")
    if metric == "towards_bias_switch":
        early, early_sources = load_switch_group("early", early_campaign_root)
        late, late_sources = load_switch_group("late", late_campaign_root)
    else:
        if luna_output_root is None:
            raise MusePublicationError("bias verbalisation publication requires --luna-output-root")
        early, early_sources = load_luna_group("early", luna_output_root)
        late, late_sources = load_luna_group("late", luna_output_root)
    logs = {**early, **late}
    if tuple(logs) != CONDITIONS:
        raise MusePublicationError(f"Muse publication conditions must be ordered exactly {list(CONDITIONS)}")
    ids_by_condition = {condition: _validate_matrix(values, condition=condition) for condition, values in logs.items()}
    baseline_ids = ids_by_condition[BASELINE]
    if any(ids != baseline_ids for ids in ids_by_condition.values()):
        raise MusePublicationError("Muse conditions do not share exact frozen question-ID membership")
    return logs, {
        "early": early_sources,
        "late": late_sources,
        "shared_question_membership": standard._question_id_manifest(baseline_ids),
    }


def _significance_seed(*, treatment: str, metric: str, population: str, bias_type: str) -> int:
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
        raise MusePublicationError("significance permutations must be a positive integer")
    if set(treatment) != set(baseline):
        raise MusePublicationError(f"paired significance membership differs for {treatment_name}/{population}/{bias_type}")
    clusters = sorted(treatment)
    treatment_counts = np.asarray([treatment[key] for key in clusters], dtype=np.int64)
    baseline_counts = np.asarray([baseline[key] for key in clusters], dtype=np.int64)
    treatment_total, baseline_total = treatment_counts.sum(axis=0), baseline_counts.sum(axis=0)
    if treatment_total[1] <= 0 or baseline_total[1] <= 0:
        raise MusePublicationError(f"paired significance has a zero denominator for {treatment_name}/{population}/{bias_type}")
    observed = float(treatment_total[0] / treatment_total[1] - baseline_total[0] / baseline_total[1])
    seed = _significance_seed(
        treatment=treatment_name,
        metric=metric,
        population=population,
        bias_type=bias_type,
    )
    rng = np.random.default_rng(seed)
    combined_total = (treatment_counts + baseline_counts).sum(axis=0)
    extreme = valid = drawn = 0
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 8.0
    while valid < permutations:
        batch = min(1_000, permutations - valid)
        swaps = rng.integers(0, 2, size=(batch, len(clusters)), dtype=np.int8).astype(bool)
        randomized_treatment = np.where(swaps[..., np.newaxis], baseline_counts, treatment_counts).sum(axis=1)
        randomized_baseline = combined_total - randomized_treatment
        usable = (randomized_treatment[:, 1] > 0) & (randomized_baseline[:, 1] > 0)
        statistics = (
            randomized_treatment[usable, 0] / randomized_treatment[usable, 1]
            - randomized_baseline[usable, 0] / randomized_baseline[usable, 1]
        )[: permutations - valid]
        extreme += int(np.count_nonzero(np.abs(statistics) >= abs(observed) - tolerance))
        valid += len(statistics)
        drawn += batch
        if drawn > permutations * 100:
            raise MusePublicationError("paired significance could not draw enough valid swaps")
    return {
        "significance_method": SIGNIFICANCE_METHOD,
        "significance_baseline": BASELINE,
        "significance_resampling_unit": "question_id",
        "significance_sidedness": "two_sided",
        "significance_statistic": "difference_in_rates",
        "question_clusters": len(clusters),
        "observed_difference": observed,
        "permutations_requested": permutations,
        "permutations_valid": valid,
        "permutations_drawn": drawn,
        "significance_seed": seed,
        "p_value_raw": (extreme + 1) / (valid + 1),
    }


def _append_significance(
    rows: Sequence[Mapping[str, Any]],
    logs_by_condition: Mapping[str, Sequence[Any]],
    *,
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
            raise MusePublicationError(f"unexpected Muse treatment: {condition!r}")
        biases = row.get("component_biases") or [row["bias_type"]]
        population = str(row["population"])
        datasets = POPULATION_DATASETS[population]
        treatment_counts = standard._cluster_counts(
            logs_by_condition[condition], datasets=datasets, biases=biases, metric=metric
        )
        baseline_counts = standard._cluster_counts(
            logs_by_condition[BASELINE], datasets=datasets, biases=biases, metric=metric
        )
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
        row["significance_analysis_membership"] = "full_exact_shared_frozen_pool"
        row["significance_analysis_question_counts_by_dataset"] = {
            dataset: sum(cluster.startswith(f"{dataset}:") for cluster in treatment_counts) for dataset in datasets
        }
        row["significance_analysis_n_questions"] = sum(
            row["significance_analysis_question_counts_by_dataset"].values()
        )
        treatment_rows.append(row)

    ranked = sorted(
        treatment_rows,
        key=lambda row: (
            float(row["p_value_raw"]),
            CONDITIONS.index(str(row["condition"])),
            POPULATION_ORDER.index(str(row["population"])),
            BIAS_ORDER.index(str(row["bias_type"])),
        ),
    )
    if len(ranked) != HOLM_FAMILY_SIZE:
        raise MusePublicationError(
            f"Muse publication must Holm-correct {HOLM_FAMILY_SIZE} treatment cells, got {len(ranked)}"
        )
    running = 0.0
    for rank, row in enumerate(ranked):
        adjusted = min(1.0, max(running, (HOLM_FAMILY_SIZE - rank) * float(row["p_value_raw"])))
        running = adjusted
        row.update(
            {
                "p_value": adjusted,
                "p_value_holm": adjusted,
                "significance": standard._significance_marker(adjusted),
                "significance_multiplicity": f"holm_across_{HOLM_FAMILY_SIZE}_displayed_checkpoint_vs_base_cells",
                "holm_family_size": HOLM_FAMILY_SIZE,
            }
        )
    return output


def chart_rows(
    logs_by_condition: Mapping[str, Sequence[Any]],
    *,
    metric: str = "towards_bias_switch",
    significance_permutations: int = SIGNIFICANCE_PERMUTATIONS,
) -> list[dict[str, Any]]:
    if tuple(logs_by_condition) != CONDITIONS:
        raise MusePublicationError(f"Muse publication conditions must be ordered exactly {list(CONDITIONS)}")
    if metric not in SUPPORTED_METRICS:
        raise MusePublicationError(f"unsupported Muse publication metric: {metric!r}")
    ids_by_condition = {
        condition: _validate_matrix(logs, condition=condition) for condition, logs in logs_by_condition.items()
    }
    if any(ids != ids_by_condition[BASELINE] for ids in ids_by_condition.values()):
        raise MusePublicationError("Muse chart inputs do not share exact paired question membership")
    rows: list[dict[str, Any]] = []
    for population, datasets in POPULATION_DATASETS.items():
        population_logs = {
            condition: [log for log in logs if standard._dataset_from_log(log) in set(datasets)]
            for condition, logs in logs_by_condition.items()
        }
        population_rows = aggregate_logs(
            population_logs,
            metric=metric,
            stderr="binomial",
            variant="biased",
            metadata={
                "model": "muse-glimmer-30b",
                "model_label": "Muse Glimmer 30B",
                "evaluation_contract": "exact_shared_frozen_pool_checkpoint_comparison",
                "population": population,
                "population_datasets": list(datasets),
            },
            condition_metadata=CONDITION_METADATA,
            expected_biases=ALL_BIASES,
            expected_datasets=datasets,
        )
        rows.extend(
            append_binomial_wilson_intervals(
                append_bias_group_summaries(population_rows, groups=BIAS_GROUPS)
            )
        )
    prepared: list[dict[str, Any]] = []
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
        prepared.append(row)
    ordered = sorted(
        prepared,
        key=lambda row: (
            POPULATION_ORDER.index(str(row["population"])),
            CONDITIONS.index(str(row["condition"])),
            BIAS_ORDER.index(str(row["bias_type"])),
        ),
    )
    return _append_significance(
        ordered,
        logs_by_condition,
        metric=metric,
        permutations=significance_permutations,
    )


def publication_spec(*, metric: str = "towards_bias_switch") -> dict[str, Any]:
    if metric not in SUPPORTED_METRICS:
        raise MusePublicationError(f"unsupported Muse publication metric: {metric!r}")
    labels = registry_labels(load_presentation_registry().biases)
    ylabel = {
        "bias_acknowledged": "Bias verbalised (Luna YES | valid grade)",
        "towards_bias_switch": "Towards-bias switch rate (eligible paired questions)",
    }[metric]
    denominator = {
        "bias_acknowledged": "valid no-cap Luna grades of parsed raw answers",
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
        "model_order": ["muse-glimmer-30b"],
        "model_labels": {"muse-glimmer-30b": "Muse Glimmer 30B · biased prompts"},
        "condition_order": list(CONDITIONS),
        "condition_labels": {
            condition: str(CONDITION_METADATA[condition]["condition_label"]) for condition in CONDITIONS
        },
        "condition_styles": {
            BASELINE: {"color": "#9aa0a6"},
            STEP16: {"color": "#6fa8dc"},
            STEP64: {"color": "#d99a6c"},
            FINAL: {"color": "#8cc39a"},
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
            f"Error bars are 95% Wilson intervals over {denominator}. "
            "Every condition uses the same frozen 50 LogiQA, 50 HellaSwag, and 100 HLE text-MC questions. "
            "Stars compare each RMCT checkpoint with the pinned base using two-sided paired whole-question "
            f"label-swap tests with Holm correction across the {HOLM_FAMILY_SIZE} displayed checkpoint-vs-base "
            "treatment cells. Key: * adjusted p<0.05; ** p<0.01; *** p<0.001; no star means adjusted p≥0.05."
        ),
        "sample_labels": "n_scored",
        "legend_columns": 4,
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


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def render_checkpoint_comparison(
    *,
    logs_by_condition: Mapping[str, Sequence[Any]],
    source_manifest: Mapping[str, Any],
    output_dir: str | Path,
    metric: str = "towards_bias_switch",
    significance_permutations: int = SIGNIFICANCE_PERMUTATIONS,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite Muse publication output: {output}")
    rows = chart_rows(
        logs_by_condition,
        metric=metric,
        significance_permutations=significance_permutations,
    )
    spec = publication_spec(metric=metric)
    membership = standard._question_id_manifest(
        _validate_matrix(logs_by_condition[BASELINE], condition=BASELINE)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        (staging / "chart-rows.json").write_bytes(_json_bytes(rows))
        (staging / "chart-spec.json").write_bytes(_json_bytes(spec))
        stem = OUTPUT_STEMS[metric]
        for extension in ("png", "svg"):
            render_publication_plot(rows, spec, staging / f"{stem}.{extension}")
        outputs = {
            path.name: {
                "path": str(output / path.name),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.iterdir())
        }
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "metric": metric,
            "model": "meta-models/Muse-Glimmer-30B",
            "conditions": CONDITION_METADATA,
            "question_membership": membership,
            "bias_groups": {name: list(values) for name, values in BIAS_GROUPS.items()},
            "dataset_populations": {name: list(values) for name, values in POPULATION_DATASETS.items()},
            "significance": {
                "method": SIGNIFICANCE_METHOD,
                "baseline": BASELINE,
                "comparisons": {condition: BASELINE for condition in TREATMENTS},
                "permutations": significance_permutations,
                "multiplicity": f"holm_across_{HOLM_FAMILY_SIZE}_displayed_checkpoint_vs_base_cells",
                "holm_family_size": HOLM_FAMILY_SIZE,
                "membership": "full_exact_shared_frozen_pool",
            },
            "source_inputs": dict(source_manifest),
            "outputs": outputs,
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(staging, output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=SUPPORTED_METRICS, default="towards_bias_switch")
    parser.add_argument("--early-campaign-root", required=True, type=Path)
    parser.add_argument("--late-campaign-root", required=True, type=Path)
    parser.add_argument("--luna-output-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        logs, sources = load_inputs(
            metric=args.metric,
            early_campaign_root=args.early_campaign_root,
            late_campaign_root=args.late_campaign_root,
            luna_output_root=args.luna_output_root,
        )
        result = render_checkpoint_comparison(
            logs_by_condition=logs,
            source_manifest=sources,
            output_dir=args.output_dir,
            metric=args.metric,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "BASELINE",
    "BIAS_GROUPS",
    "CONDITIONS",
    "EXPECTED_QUESTION_COUNTS",
    "FINAL",
    "HOLM_FAMILY_SIZE",
    "SIGNIFICANCE_PERMUTATIONS",
    "STEP16",
    "STEP64",
    "chart_rows",
    "load_inputs",
    "publication_spec",
    "render_checkpoint_comparison",
]
