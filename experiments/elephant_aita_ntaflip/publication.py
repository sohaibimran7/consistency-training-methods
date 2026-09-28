"""Paired, final-answer-only publication for AITA-NTA-FLIP conditions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ctm_data.adapters.mcq_bias.method_presentation import base_first, method_color


REPORT_SCHEMA = "elephant-aita-nta-flip-paired-publication-v1"
PREFLIGHT_SCHEMA = "elephant-aita-nta-flip-preflight-v5-r005"
PARSER_SCHEMA = "elephant-aita-nta-flip-parser-v2-final-answer-only"
EXPECTED_PAIRS = 1591
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2_026_082_400


class PublicationError(ValueError):
    """A supplied preflight cannot support the paired publication."""


@dataclass(frozen=True)
class ConditionData:
    label: str
    path: Path
    identity: Mapping[str, Any]
    manifest: Mapping[str, Any]
    pair_ids: tuple[str, ...]
    outcomes: np.ndarray
    parsed_pairs: int
    parsed_responses: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {candidate}")
    return candidate.resolve()


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicationError(f"{label} must be an object")
    return value


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PublicationError(f"{label} must be an integer")
    return value


def load_condition(label: str, path: str | Path, *, expected_pairs: int = EXPECTED_PAIRS) -> ConditionData:
    if not label.strip():
        raise PublicationError("condition label must not be empty")
    candidate = _regular_file(path, label=f"{label} preflight")
    try:
        preflight = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"invalid {label} preflight JSON: {candidate}") from exc
    preflight = _mapping(preflight, label=f"{label} preflight")
    if preflight.get("schema") != PREFLIGHT_SCHEMA or preflight.get("benchmark") != "elephant-aita-nta-flip":
        raise PublicationError(f"{label} preflight has the wrong benchmark/schema")
    manifest = _mapping(preflight.get("manifest"), label=f"{label} manifest")
    if _integer(manifest.get("pair_count"), label=f"{label} manifest pair_count") != expected_pairs:
        raise PublicationError(f"{label} does not contain exactly {expected_pairs} pairs")

    metrics = _mapping(preflight.get("metrics"), label=f"{label} metrics")
    final = _mapping(metrics.get("final_answer_only"), label=f"{label} final-answer metrics")
    if final.get("parser_schema") != PARSER_SCHEMA:
        raise PublicationError(f"{label} was not scored by the final-answer-only parser")
    parsed_pair_record = _mapping(final.get("parsed_pair_coverage"), label=f"{label} parsed pair coverage")
    parsed_response_record = _mapping(
        final.get("parsed_response_coverage"), label=f"{label} parsed response coverage"
    )
    parsed_pairs = _integer(parsed_pair_record.get("count"), label=f"{label} parsed pair count")
    parsed_responses = _integer(parsed_response_record.get("count"), label=f"{label} parsed response count")

    records = preflight.get("pair_records")
    if not isinstance(records, list) or len(records) != expected_pairs:
        raise PublicationError(f"{label} must contain {expected_pairs} pair records")
    by_id: dict[str, int] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, label=f"{label} pair record {index}")
        pair_id = record.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in by_id:
            raise PublicationError(f"{label} has an empty or duplicate pair ID")
        indicators = _mapping(record.get("final_indicators"), label=f"{label} pair {pair_id} indicators")
        outcome = _integer(indicators.get("nta_nta"), label=f"{label} pair {pair_id} NTA/NTA indicator")
        if outcome not in (0, 1):
            raise PublicationError(f"{label} pair {pair_id} has a non-binary NTA/NTA indicator")
        by_id[pair_id] = outcome

    pair_ids = tuple(sorted(by_id))
    outcomes = np.fromiter((by_id[pair_id] for pair_id in pair_ids), dtype=np.int8, count=expected_pairs)
    reported = _mapping(final.get("outcomes"), label=f"{label} final outcomes")
    reported_nta = _mapping(reported.get("nta_nta"), label=f"{label} reported NTA/NTA")
    if _integer(reported_nta.get("count"), label=f"{label} reported NTA/NTA count") != int(outcomes.sum()):
        raise PublicationError(f"{label} pair records disagree with its reported NTA/NTA count")
    return ConditionData(
        label=label,
        path=candidate,
        identity=_identity(candidate),
        manifest=dict(manifest),
        pair_ids=pair_ids,
        outcomes=outcomes,
        parsed_pairs=parsed_pairs,
        parsed_responses=parsed_responses,
    )


def _validate_shared_custody(conditions: Sequence[ConditionData]) -> None:
    if len(conditions) < 2:
        raise PublicationError("publication requires a baseline and at least one comparison")
    labels = [condition.label for condition in conditions]
    if len(set(labels)) != len(labels):
        raise PublicationError("condition labels must be unique")
    reference = conditions[0]
    custody_keys = ("schema", "pair_count", "pair_ids_sha256", "pair_artifact_sha256", "sha256")
    for condition in conditions[1:]:
        if condition.pair_ids != reference.pair_ids:
            raise PublicationError(f"{condition.label} pair IDs differ from {reference.label}")
        for key in custody_keys:
            if condition.manifest.get(key) != reference.manifest.get(key):
                raise PublicationError(f"{condition.label} manifest {key} differs from {reference.label}")


def _exact_mcnemar(baseline: np.ndarray, treatment: np.ndarray) -> tuple[int, int, float]:
    baseline_only = int(np.sum((baseline == 1) & (treatment == 0)))
    treatment_only = int(np.sum((baseline == 0) & (treatment == 1)))
    discordant = baseline_only + treatment_only
    if discordant == 0:
        return baseline_only, treatment_only, 1.0
    lower = min(baseline_only, treatment_only)
    numerator = sum(math.comb(discordant, k) for k in range(lower + 1))
    p_value = min(1.0, 2.0 * numerator / (2**discordant))
    return baseline_only, treatment_only, float(p_value)


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _stars(adjusted_p: float) -> str:
    if adjusted_p < 0.001:
        return "***"
    if adjusted_p < 0.01:
        return "**"
    if adjusted_p < 0.05:
        return "*"
    return "ns"


def _bootstrap(
    conditions: Sequence[ConditionData], *, resamples: int, seed: int
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    if resamples < 100:
        raise PublicationError("bootstrap resamples must be at least 100")
    rng = np.random.default_rng(seed)
    matrix = np.stack([condition.outcomes for condition in conditions], axis=1).astype(np.float64)
    n_pairs = matrix.shape[0]
    estimates: list[np.ndarray] = []
    remaining = resamples
    while remaining:
        batch = min(remaining, 512)
        indices = rng.integers(0, n_pairs, size=(batch, n_pairs), endpoint=False)
        estimates.append(matrix[indices].mean(axis=1))
        remaining -= batch
    samples = np.concatenate(estimates, axis=0)
    rate_intervals: dict[str, tuple[float, float]] = {}
    difference_intervals: dict[str, tuple[float, float]] = {}
    for index, condition in enumerate(conditions):
        low, high = np.quantile(samples[:, index], (0.025, 0.975), method="linear")
        rate_intervals[condition.label] = (float(low), float(high))
        if index:
            differences = samples[:, index] - samples[:, 0]
            low, high = np.quantile(differences, (0.025, 0.975), method="linear")
            difference_intervals[condition.label] = (float(low), float(high))
    return rate_intervals, difference_intervals


def build_report(
    conditions: Sequence[ConditionData], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    _validate_shared_custody(conditions)
    rate_intervals, difference_intervals = _bootstrap(conditions, resamples=resamples, seed=seed)
    baseline = conditions[0]
    comparisons: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for treatment in conditions[1:]:
        baseline_only, treatment_only, raw_p = _exact_mcnemar(baseline.outcomes, treatment.outcomes)
        raw_p_values.append(raw_p)
        low, high = difference_intervals[treatment.label]
        comparisons.append(
            {
                "baseline": baseline.label,
                "treatment": treatment.label,
                "difference_treatment_minus_baseline": float(treatment.outcomes.mean() - baseline.outcomes.mean()),
                "difference_bootstrap_95_ci": {"low": low, "high": high},
                "discordant": {
                    "baseline_nta_nta_treatment_not": baseline_only,
                    "baseline_not_treatment_nta_nta": treatment_only,
                    "total": baseline_only + treatment_only,
                },
                "p_value_raw": raw_p,
            }
        )
    adjusted = _holm_adjust(raw_p_values)
    for comparison, adjusted_p in zip(comparisons, adjusted):
        comparison["p_value_holm"] = adjusted_p
        comparison["significance"] = _stars(adjusted_p)

    rows = []
    for condition in conditions:
        low, high = rate_intervals[condition.label]
        rows.append(
            {
                "label": condition.label,
                "preflight": dict(condition.identity),
                "n_pairs": len(condition.pair_ids),
                "nta_nta_count": int(condition.outcomes.sum()),
                "nta_nta_rate": float(condition.outcomes.mean()),
                "bootstrap_95_ci": {"low": low, "high": high},
                "parsed_pairs": condition.parsed_pairs,
                "parsed_responses": condition.parsed_responses,
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "benchmark": "elephant-aita-nta-flip",
        "metric": {
            "name": "final_answer_only_both_nta",
            "label": "final-answer-only NTA/NTA",
            "direction": "lower_is_better",
            "parser_schema": PARSER_SCHEMA,
        },
        "baseline": baseline.label,
        "manifest": dict(baseline.manifest),
        "bootstrap": {"method": "paired_nonparametric_pair_resampling", "resamples": resamples, "seed": seed},
        "significance": {
            "test": "exact_two_sided_mcnemar",
            "multiplicity": "holm_across_all_treatments_vs_baseline",
            "family_size": len(comparisons),
            "markers": {"*": "p_holm < .05", "**": "p_holm < .01", "***": "p_holm < .001"},
        },
        "conditions": base_first(rows, key=lambda row: row["label"]),
        "comparisons": base_first(comparisons, key=lambda row: row["treatment"]),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _render(report: Mapping[str, Any], png: Path, pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = base_first(report["conditions"], key=lambda row: row["label"])
    comparisons = {row["treatment"]: row for row in report["comparisons"]}
    labels = [row["label"] for row in rows]
    values = np.array([row["nta_nta_rate"] for row in rows])
    lows = np.array([row["bootstrap_95_ci"]["low"] for row in rows])
    highs = np.array([row["bootstrap_95_ci"]["high"] for row in rows])
    errors = np.vstack((values - lows, highs - values))
    palette = ["#6B7280", "#2563EB", "#60A5FA", "#93C5FD", "#D97706", "#059669", "#7C3AED", "#DB2777", "#0891B2"]
    colors = [method_color(row["label"], palette[index % len(palette)]) for index, row in enumerate(rows)]

    width = max(10.0, 1.15 * len(rows))
    fig, axis = plt.subplots(figsize=(width, 6.6), constrained_layout=True)
    positions = np.arange(len(rows))
    bars = axis.bar(positions, values, yerr=errors, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
    baseline_index = labels.index(report["baseline"])
    axis.axhline(values[baseline_index], color=colors[baseline_index], linewidth=1.2, linestyle="--", alpha=0.75, label=f"{report['baseline']} rate")
    axis.set_ylabel("Both-NTA rate")
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylim(0.0, min(1.0, max(0.8, float(highs.max()) + 0.09)))
    axis.yaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "AITA-NTA-FLIP cross-task generalisation",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    axis.set_title(
        "Final-answer-only both-NTA (lower is better) • 95% paired-bootstrap CI\n"
        f"Markers vs {report['baseline']} • exact McNemar + Holm",
        fontsize=9.5,
        color="#4B5563",
        loc="left",
        pad=12,
    )
    for index, (bar, row) in enumerate(zip(bars, rows)):
        marker = "" if row["label"] == report["baseline"] else comparisons[row["label"]]["significance"]
        marker = "" if marker == "ns" else marker
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            highs[index] + 0.012,
            f"{values[index]:.1%}" + (f" {marker}" if marker else ""),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if marker else "normal",
        )
    axis.text(
        1.0,
        -0.22,
        f"n={rows[0]['n_pairs']:,} paired examples per condition.  * p<.05, ** p<.01, *** p<.001 after Holm correction.",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def publish(
    condition_specs: Sequence[tuple[str, str | Path]],
    *,
    output_dir: str | Path,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    expected_pairs: int = EXPECTED_PAIRS,
) -> dict[str, Any]:
    conditions = [load_condition(label, path, expected_pairs=expected_pairs) for label, path in condition_specs]
    report = build_report(conditions, resamples=resamples, seed=seed)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and (destination.is_symlink() or not destination.is_dir()):
        raise PublicationError(f"output directory is linked or not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    # Use a unique, non-recursively-cleaned staging directory.  A failed run
    # deliberately leaves its evidence in place; a successful run removes
    # only the now-empty directory after atomically promoting each file.
    staging = Path(tempfile.mkdtemp(prefix=".aita-publication-", dir=destination))
    report_path = staging / "aita-nta-flip-paired-report.json"
    csv_path = staging / "aita-nta-flip-paired-results.csv"
    png_path = staging / "aita-nta-flip-both-nta-rate.png"
    pdf_path = staging / "aita-nta-flip-both-nta-rate.pdf"
    _write_json(report_path, report)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("label", "n_pairs", "nta_nta_count", "nta_nta_rate", "ci_low", "ci_high", "p_holm", "marker"),
        )
        writer.writeheader()
        comparisons = {row["treatment"]: row for row in report["comparisons"]}
        for row in report["conditions"]:
            comparison = comparisons.get(row["label"], {})
            writer.writerow(
                {
                    "label": row["label"],
                    "n_pairs": row["n_pairs"],
                    "nta_nta_count": row["nta_nta_count"],
                    "nta_nta_rate": row["nta_nta_rate"],
                    "ci_low": row["bootstrap_95_ci"]["low"],
                    "ci_high": row["bootstrap_95_ci"]["high"],
                    "p_holm": comparison.get("p_value_holm", ""),
                    "marker": comparison.get("significance", ""),
                }
            )
    _render(report, png_path, pdf_path)
    for path in (report_path, csv_path, png_path, pdf_path):
        os.replace(path, destination / path.name)
    staging.rmdir()
    return report


def _condition_spec(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("condition must be LABEL=/path/to/preflight.json")
    return label.strip(), Path(raw_path).expanduser()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", type=_condition_spec, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--expected-pairs", type=int, default=EXPECTED_PAIRS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = publish(
        args.condition,
        output_dir=args.output_dir,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
        expected_pairs=args.expected_pairs,
    )
    print(json.dumps({"schema": report["schema"], "output_dir": str(args.output_dir), "conditions": len(report["conditions"])}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
