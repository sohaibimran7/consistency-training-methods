"""Standard publication adapter for the sealed r005 Luna comparison.

The generic mcq-bias analysis layer owns aggregation, arbitrary bias-group
pooling, coverage, and Wilson intervals.  This module supplies only the r005
scientific labels and condition provenance, then delegates all drawing to the
standard :func:`ctm_data.adapters.mcq_bias.plot.render_publication_plot`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
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
from experiments.rmct_two_bias_eval.contract import ALL_BIASES, HELD_OUT_BIASES, SEEN_BIASES


CONDITIONS = ("untrained", "rate-matching")
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
CONDITION_METADATA: dict[str, dict[str, Any]] = {
    "untrained": {
        "condition_label": "Archived base reference",
        "method": "none",
        "is_control": False,
        "training_biases": [],
        "provenance_class": "historical_same_task_reference",
    },
    "rate-matching": {
        "condition_label": "RMCT step 176",
        "method": "rate_matching",
        "is_control": False,
        "training_biases": list(SEEN_BIASES),
        "provenance_class": "sealed_r005_step176",
    },
}
OUTPUT_SCHEMA = "rmct-two-bias-r005-standard-publication-v2"
SUPPORTED_METRICS = ("bias_acknowledged", "towards_bias_switch")
OUTPUT_STEMS = {
    "bias_acknowledged": "bias-verbalisation",
    "towards_bias_switch": "towards-bias-switch-rate",
}
SIGNIFICANCE_METHOD = "paired_question_cluster_label_swap_randomization"
SIGNIFICANCE_PERMUTATIONS = 10_000
SIGNIFICANCE_BASE_SEED = 2_026_082_100


def chart_rows(
    logs_by_condition: Mapping[str, Sequence[Any]],
    *,
    metric: str = "bias_acknowledged",
    significance_permutations: int = SIGNIFICANCE_PERMUTATIONS,
) -> list[dict[str, Any]]:
    """Build standard chart rows with bias status supplied as data."""

    if set(logs_by_condition) != set(CONDITIONS):
        raise ValueError(f"r005 publication conditions must be exactly {list(CONDITIONS)}")
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"unsupported r005 publication metric: {metric!r}")
    rows: list[dict[str, Any]] = []
    for population, datasets in POPULATION_DATASETS.items():
        dataset_set = set(datasets)
        population_logs = {
            condition: [
                log
                for log in logs
                if str(log.eval.task_args.get("dataset", "")) in dataset_set
            ]
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
                "evaluation_contract": "rmct-convergence-r4-s011-two-bias-v1-r005",
                "population": population,
                "population_datasets": list(datasets),
            },
            condition_metadata=CONDITION_METADATA,
            expected_biases=ALL_BIASES,
            expected_datasets=datasets,
        )
        population_rows = append_bias_group_summaries(population_rows, groups=BIAS_GROUPS)
        rows.extend(append_binomial_wilson_intervals(population_rows))
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
    return _append_paired_significance(
        ordered,
        logs_by_condition=logs_by_condition,
        metric=metric,
        permutations=significance_permutations,
    )


def _sample_metric(sample: Any, metric: str) -> float | None:
    for score in (getattr(sample, "scores", None) or {}).values():
        value = getattr(score, "value", None)
        if isinstance(value, Mapping) and metric in value:
            candidate = value[metric]
            if candidate is None or isinstance(candidate, bool):
                return None
            try:
                parsed = float(candidate)
            except (TypeError, ValueError):
                return None
            return parsed if math.isfinite(parsed) else None
    return None


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
        args = log.eval.task_args
        dataset, bias = str(args.get("dataset", "")), str(args.get("bias_type", ""))
        if dataset not in dataset_set or bias not in bias_set:
            continue
        for sample in log.samples or []:
            question_id = str(getattr(sample, "id", ""))
            if not question_id:
                raise ValueError("paired significance requires non-empty sample IDs")
            observation = (dataset, bias, question_id)
            if observation in seen:
                raise ValueError(f"paired significance encountered duplicate observation: {observation}")
            seen.add(observation)
            cluster = f"{dataset}:{question_id}"
            counts[cluster]  # retain zero-denominator clusters for paired label swaps
            value = _sample_metric(sample, metric)
            if value is None:
                continue
            if value not in {0.0, 1.0}:
                raise ValueError(f"paired significance requires a binary metric, got {value}")
            counts[cluster][0] += int(value)
            counts[cluster][1] += 1
    if not counts:
        raise ValueError("paired significance selected no question clusters")
    return {key: (value[0], value[1]) for key, value in counts.items()}


def _derived_significance_seed(*, metric: str, population: str, bias_type: str) -> int:
    payload = f"{SIGNIFICANCE_BASE_SEED}|{metric}|{population}|{bias_type}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _paired_label_swap(
    treatment: Mapping[str, tuple[int, int]],
    baseline: Mapping[str, tuple[int, int]],
    *,
    metric: str,
    population: str,
    bias_type: str,
    permutations: int,
) -> dict[str, Any]:
    if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 1:
        raise ValueError("significance permutations must be a positive integer")
    if set(treatment) != set(baseline):
        raise ValueError(
            f"paired significance has misaligned question clusters for {population}/{bias_type}"
        )
    question_ids = sorted(treatment)
    treatment_counts = np.asarray([treatment[key] for key in question_ids], dtype=np.int64)
    baseline_counts = np.asarray([baseline[key] for key in question_ids], dtype=np.int64)
    treatment_total, baseline_total = treatment_counts.sum(axis=0), baseline_counts.sum(axis=0)
    if treatment_total[1] <= 0 or baseline_total[1] <= 0:
        raise ValueError(f"paired significance has a zero denominator for {population}/{bias_type}")
    observed = float(
        treatment_total[0] / treatment_total[1] - baseline_total[0] / baseline_total[1]
    )
    seed = _derived_significance_seed(metric=metric, population=population, bias_type=bias_type)
    rng = np.random.default_rng(seed)
    total_counts = treatment_counts + baseline_counts
    combined_total = total_counts.sum(axis=0)
    extreme, valid, drawn = 0, 0, 0
    observed_abs = abs(observed)
    tolerance = np.finfo(np.float64).eps * max(1.0, observed_abs) * 8.0
    while valid < permutations:
        batch = min(1_000, permutations - valid)
        swaps = rng.integers(0, 2, size=(batch, len(question_ids)), dtype=np.int8).astype(bool)
        randomized_treatment = np.where(
            swaps[..., np.newaxis], baseline_counts, treatment_counts
        ).sum(axis=1)
        randomized_baseline = combined_total - randomized_treatment
        usable = (randomized_treatment[:, 1] > 0) & (randomized_baseline[:, 1] > 0)
        statistics = (
            randomized_treatment[usable, 0] / randomized_treatment[usable, 1]
            - randomized_baseline[usable, 0] / randomized_baseline[usable, 1]
        )
        remaining = permutations - valid
        statistics = statistics[:remaining]
        extreme += int(np.count_nonzero(np.abs(statistics) >= observed_abs - tolerance))
        valid += len(statistics)
        drawn += batch
        if drawn > permutations * 100:
            raise ValueError(f"paired significance could not draw enough valid swaps for {population}/{bias_type}")
    return {
        "significance_method": SIGNIFICANCE_METHOD,
        "significance_baseline": "untrained",
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
    logs_by_condition: Mapping[str, Sequence[Any]],
    metric: str,
    permutations: int,
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    treatment_rows: list[dict[str, Any]] = []
    for row in output:
        row["significance_baseline"] = "untrained"
        if row["condition"] == "untrained":
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
        biases = row.get("component_biases") or [row["bias_type"]]
        population = str(row["population"])
        datasets = POPULATION_DATASETS[population]
        treatment = _cluster_counts(
            logs_by_condition["rate-matching"],
            datasets=datasets,
            biases=biases,
            metric=metric,
        )
        baseline = _cluster_counts(
            logs_by_condition["untrained"],
            datasets=datasets,
            biases=biases,
            metric=metric,
        )
        row.update(
            _paired_label_swap(
                treatment,
                baseline,
                metric=metric,
                population=population,
                bias_type=str(row["bias_type"]),
                permutations=permutations,
            )
        )
        treatment_rows.append(row)

    ranked = sorted(treatment_rows, key=lambda row: (float(row["p_value_raw"]), str(row["population"]), str(row["bias_type"])))
    family_size, running = len(ranked), 0.0
    for rank, row in enumerate(ranked):
        adjusted = min(1.0, max(running, (family_size - rank) * float(row["p_value_raw"])))
        running = adjusted
        row.update(
            {
                "p_value": adjusted,
                "p_value_holm": adjusted,
                "significance": _significance_marker(adjusted),
                "significance_multiplicity": "holm_across_displayed_treatment_cells",
                "holm_family_size": family_size,
            }
        )
    return output


def publication_spec(*, metric: str = "bias_acknowledged") -> dict[str, Any]:
    """Return the declarative standard-renderer recipe."""

    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"unsupported r005 publication metric: {metric!r}")
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
            "Archived base is a same-task historical reference without the sealed r005 runtime identity; "
            f"comparisons are descriptive. Error bars are 95% Wilson intervals over {uncertainty_note}. "
            "Stars compare RMCT with base using two-sided paired whole-question label-swap tests "
            "with Holm correction across the 18 displayed treatment cells. "
            "Key: * adjusted p<0.05; ** p<0.01; *** p<0.001; no star means adjusted p≥0.05."
        ),
        "sample_labels": "n_scored",
        "legend_columns": 2,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"standard publication input must be a regular file: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _output_identity(path: Path, *, output_dir: Path) -> dict[str, Any]:
    """Fingerprint a staged output while recording its final durable path."""

    identity = _identity(path)
    identity["path"] = str((output_dir / path.name).resolve())
    return identity


def _load_logs(root: str | Path, *, metric: str) -> tuple[list[Any], list[dict[str, Any]]]:
    directory = Path(root).expanduser().resolve()
    if directory.is_symlink() or not directory.is_dir():
        raise FileNotFoundError(f"standard publication log root must be a regular directory: {directory}")
    if metric == "bias_acknowledged":
        paths = sorted(
            path
            for path in directory.rglob("*-luna.eval")
            if not path.name.endswith("-luna-smoke.eval")
        )
    elif metric == "towards_bias_switch":
        paths = sorted(
            path
            for path in directory.rglob("*.eval")
            if not path.name.endswith(("-luna.eval", "-luna-smoke.eval"))
        )
    else:
        raise ValueError(f"unsupported r005 publication metric: {metric!r}")
    if len(paths) != 18:
        raise ValueError(f"standard r005 comparison requires exactly 18 biased EvalLogs under {directory}, got {len(paths)}")
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - plotting environment boundary
        raise RuntimeError("Inspect AI is required to read standard publication inputs") from exc
    logs = [read_eval_log(str(path)) for path in paths]
    if any(getattr(log, "status", None) != "success" for log in logs):
        raise ValueError(f"standard publication input contains a non-success EvalLog: {directory}")
    return logs, [_identity(path) for path in paths]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def render_standard_comparison(
    *,
    rmct_logs: str | Path,
    base_logs: str | Path,
    output_dir: str | Path,
    metric: str = "bias_acknowledged",
) -> Path:
    """Write rows, spec, manifest, PNG, and SVG through the standard pipeline."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite standard publication output: {output}")
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"unsupported r005 publication metric: {metric!r}")
    rmct, rmct_identities = _load_logs(rmct_logs, metric=metric)
    base, base_identities = _load_logs(base_logs, metric=metric)
    rows = chart_rows({"untrained": base, "rate-matching": rmct}, metric=metric)
    spec = publication_spec(metric=metric)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        rows_path = staging / "chart-rows.json"
        spec_path = staging / "chart-spec.json"
        rows_path.write_bytes(_json_bytes(rows))
        spec_path.write_bytes(_json_bytes(spec))
        output_stem = OUTPUT_STEMS[metric]
        for extension in ("png", "svg"):
            render_publication_plot(rows, spec, staging / f"{output_stem}.{extension}")
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "metric": metric,
            "bias_groups": {label: list(members) for label, members in BIAS_GROUPS.items()},
            "dataset_populations": {
                label: list(datasets) for label, datasets in POPULATION_DATASETS.items()
            },
            "condition_provenance": CONDITION_METADATA,
            "source_logs": {"untrained": base_identities, "rate-matching": rmct_identities},
            "outputs": {
                path.name: _output_identity(path, output_dir=output)
                for path in sorted(staging.iterdir())
                if path.name != "manifest.json"
            },
            "comparison_caveat": (
                "untrained is an archived same-task reference without the sealed r005 runtime identity; "
                "descriptive comparison only"
            ),
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(staging, output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmct-logs", required=True, type=Path)
    parser.add_argument("--base-logs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metric", choices=SUPPORTED_METRICS, default="bias_acknowledged")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = render_standard_comparison(
        rmct_logs=args.rmct_logs,
        base_logs=args.base_logs,
        output_dir=args.output_dir,
        metric=args.metric,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BIAS_GROUPS",
    "BIAS_ORDER",
    "CONDITION_METADATA",
    "POPULATION_DATASETS",
    "POPULATION_ORDER",
    "SUPPORTED_METRICS",
    "chart_rows",
    "publication_spec",
    "render_standard_comparison",
]
