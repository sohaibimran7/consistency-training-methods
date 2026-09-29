"""Descriptive base-model comparison across historical Qwen, Muse, and Gemma.

This module deliberately operates at the chart-row boundary.  It takes the
standard publication rows produced by each model's own pipeline, keeps their
original estimates, Wilson intervals, and denominators, and only normalises
the base-condition labels so that the standard renderer can draw the three
models side by side.  It does not claim shared question membership or run any
cross-model significance test.

The historical Qwen verbalisation result used a 256-token Luna output cap.
Gemma's new verbalisation result is uncapped.  That incompatibility is made
visible in the verbalisation figure and retained in the output manifest rather
than being silently papered over by a pooled statistic.
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
from typing import Any, TypeAlias

from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from ctm_data.adapters.mcq_bias.plot_registry import load_presentation_registry, registry_labels
from experiments.gemma4_12b_base_eval import postprocess as gemma_postprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_METRICS = ("towards_bias_switch", "bias_acknowledged")
OUTPUT_STEMS = {
    "towards_bias_switch": "towards-bias-switch-rate",
    "bias_acknowledged": "bias-verbalisation",
}
OUTPUT_SCHEMA = "gemma4-12b-base-descriptive-model-comparison-v1"
COMPARISON_STATUS = "descriptive_only_no_cross_model_significance"
NO_CROSS_MODEL_SIGNIFICANCE_REASON = (
    "source publications do not establish one shared, paired question pool and identical runtime protocol across "
    "Qwen, Muse, and Gemma"
)
QWEN_HISTORICAL_LUNA_OUTPUT_TOKEN_CAP = 256
QWEN_HISTORICAL_GENERATION_TOKEN_CAP = 20_480
QWEN_HISTORICAL_GENERATION_CAP_HITS = 54
QWEN_HISTORICAL_GENERATION_BIASED_OUTPUTS = 1_800
QWEN_HISTORICAL_LUNA_CAP_HITS = 9
QWEN_HISTORICAL_LUNA_VALID_GRADES = 1_791

BIAS_ORDER = tuple(gemma_postprocess.BIAS_ORDER)
POPULATION_ORDER = tuple(gemma_postprocess.POPULATION_ORDER)


class CombinedBaseComparisonError(ValueError):
    """A source chart publication cannot support the descriptive comparison."""


RowsInput: TypeAlias = str | Path | Mapping[str, str | Path]


@dataclass(frozen=True, slots=True)
class _SourceProfile:
    key: str
    source_models: tuple[str, ...]
    source_base_conditions: tuple[str, ...]
    combined_model: str
    combined_condition: str
    combined_label: str
    combined_model_label: str
    color: str
    provenance_class: str


@dataclass(frozen=True, slots=True)
class _LoadedRows:
    path: Path
    identity: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    sidecars: dict[str, dict[str, Any]]


_PROFILES: dict[str, _SourceProfile] = {
    "qwen": _SourceProfile(
        key="qwen",
        source_models=("qwen3.5-9b",),
        source_base_conditions=("base_archived", "untrained", "base"),
        combined_model="qwen3.5-9b",
        combined_condition="qwen_base",
        combined_label="Qwen 3.5 9B base (historical)",
        combined_model_label="Qwen 3.5 9B · historical base",
        color="#9aa0a6",
        provenance_class="historical_qwen_base_publication",
    ),
    "muse": _SourceProfile(
        key="muse",
        source_models=("muse-glimmer-30b",),
        source_base_conditions=("base", "base_archived", "untrained"),
        combined_model="muse-glimmer-30b",
        combined_condition="muse_base",
        combined_label="Muse Glimmer 30B base (historical)",
        combined_model_label="Muse Glimmer 30B · historical base",
        color="#6fa8dc",
        provenance_class="historical_muse_base_publication",
    ),
    "gemma": _SourceProfile(
        key="gemma",
        source_models=("gemma-4-12b-it",),
        source_base_conditions=("base",),
        combined_model="gemma-4-12b-it",
        combined_condition="gemma_base",
        combined_label="Gemma 4 12B base",
        combined_model_label="Gemma 4 12B · base",
        color="#8cc39a",
        provenance_class="gemma4_12b_base_screen",
    ),
}
PROFILE_ORDER = ("qwen", "muse", "gemma")
REQUIRED_PROFILE_KEYS = ("qwen", "gemma")
CONDITION_ORDER = tuple(_PROFILES[key].combined_condition for key in PROFILE_ORDER)

# These are the current sealed Qwen consistency-gap publications.  The
# semantic fallback below is intentionally strict: if a future artifact root
# contains more than one suitable candidate, callers must pass its path rather
# than have the adapter guess which historical result they meant.
_KNOWN_HISTORICAL_ROWS: dict[str, dict[str, tuple[str, ...]]] = {
    "qwen": {
        "towards_bias_switch": (
            "rmct-step16-step176-standard-switch-rate-by-dataset-significance-key-r003-20260821/chart-rows.json",
            "rmct-step16-step176-standard-switch-rate-by-dataset-significance-key-20260821/chart-rows.json",
            "rmct-r005-standard-switch-rate-by-dataset-significance-key-20260821/chart-rows.json",
        ),
        "bias_acknowledged": (
            "rmct-step16-step176-standard-bias-verbalisation-by-dataset-significance-key-r003-20260821/chart-rows.json",
            "rmct-step16-step176-standard-bias-verbalisation-by-dataset-significance-key-20260821/chart-rows.json",
            "rmct-r005-standard-bias-verbalisation-by-dataset-significance-20260821/chart-rows.json",
        ),
    },
    "muse": {"towards_bias_switch": (), "bias_acknowledged": ()},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size < 1:
        raise CombinedBaseComparisonError(f"{label} must be a non-empty regular file: {candidate}")
    resolved = candidate.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved), "size_bytes": resolved.stat().st_size}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _metric_filename(metric: str) -> str:
    if metric not in OUTPUT_STEMS:
        raise CombinedBaseComparisonError(f"unsupported combined-comparison metric: {metric!r}")
    return f"{OUTPUT_STEMS[metric]}-chart-rows.json"


def _resolve_rows_path(source: RowsInput, *, metric: str, label: str) -> Path:
    """Resolve either a chart-row file, publication directory, or metric map."""

    if isinstance(source, Mapping):
        if metric not in source:
            raise CombinedBaseComparisonError(f"{label} input map has no path for {metric!r}")
        value: str | Path = source[metric]
    else:
        value = source
    path = Path(value).expanduser()
    if path.is_symlink():
        raise CombinedBaseComparisonError(f"{label} source must not be a symlink: {path}")
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} source is neither a chart-row file nor a publication directory: {path}")
    candidates = [path / _metric_filename(metric), path / "chart-rows.json"]
    existing = [candidate for candidate in candidates if candidate.exists() or candidate.is_symlink()]
    if not existing:
        raise FileNotFoundError(f"{label} publication lacks rows for {metric!r}: {path}")
    for candidate in existing:
        if candidate.is_symlink() or not candidate.is_file():
            raise CombinedBaseComparisonError(f"{label} chart rows must be a regular file: {candidate}")
    # Metric-specific files take precedence. The subsequent row validation also
    # rejects accidentally pointing a one-metric directory at the other metric.
    return existing[0].resolve()


def _read_rows(source: RowsInput, *, metric: str, label: str) -> _LoadedRows:
    path = _resolve_rows_path(source, metric=metric, label=label)
    identity = _identity(path, label=f"{label} chart rows")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CombinedBaseComparisonError(f"invalid {label} chart rows: {path}") from exc
    if not isinstance(document, list) or not document or not all(isinstance(row, Mapping) for row in document):
        raise CombinedBaseComparisonError(f"{label} chart rows must be a non-empty JSON array of objects: {path}")

    sidecars: dict[str, dict[str, Any]] = {}
    spec_name = path.name.replace("chart-rows.json", "chart-spec.json")
    for key, candidate in (("chart_spec", path.with_name(spec_name)), ("manifest", path.with_name("manifest.json"))):
        if candidate.exists() or candidate.is_symlink():
            sidecars[key] = _identity(candidate, label=f"{label} {key.replace('_', ' ')}")
    return _LoadedRows(
        path=path,
        identity=identity,
        rows=tuple(dict(row) for row in document),
        sidecars=sidecars,
    )


def _finite_number(value: Any, *, field: str, label: str) -> float:
    if isinstance(value, bool):
        raise CombinedBaseComparisonError(f"{label} has a non-numeric {field}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CombinedBaseComparisonError(f"{label} has a non-numeric {field}") from exc
    if not math.isfinite(numeric):
        raise CombinedBaseComparisonError(f"{label} has a non-finite {field}")
    return numeric


def _validate_base_matrix(
    rows: Sequence[Mapping[str, Any]], *, profile: _SourceProfile, metric: str, label: str
) -> tuple[list[dict[str, Any]], str]:
    eligible = [
        dict(row)
        for row in rows
        if str(row.get("metric", "")) == metric
        and str(row.get("model", "")) in profile.source_models
        and str(row.get("condition", "")) in profile.source_base_conditions
    ]
    if not eligible:
        raise CombinedBaseComparisonError(
            f"{label} contains no {profile.key} base rows for {metric!r}; expected model "
            f"{list(profile.source_models)} and base condition {list(profile.source_base_conditions)}"
        )
    source_conditions = {str(row["condition"]) for row in eligible}
    if len(source_conditions) != 1:
        raise CombinedBaseComparisonError(
            f"{label} contains ambiguous {profile.key} base conditions for {metric!r}: {sorted(source_conditions)}"
        )

    expected = {(population, bias) for population in POPULATION_ORDER for bias in BIAS_ORDER}
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in eligible:
        population, bias = str(row.get("population", "")), str(row.get("bias_type", ""))
        cell = (population, bias)
        if cell in observed:
            raise CombinedBaseComparisonError(f"{label} has duplicate {profile.key} base cell: {cell}")
        if row.get("ci_method") != "wilson":
            raise CombinedBaseComparisonError(f"{label} {cell} does not carry a Wilson confidence interval")
        mean = _finite_number(row.get("mean"), field="mean", label=f"{label} {cell}")
        lower = _finite_number(row.get("ci_lower"), field="ci_lower", label=f"{label} {cell}")
        upper = _finite_number(row.get("ci_upper"), field="ci_upper", label=f"{label} {cell}")
        _finite_number(row.get("stderr"), field="stderr", label=f"{label} {cell}")
        if not lower <= mean <= upper:
            raise CombinedBaseComparisonError(f"{label} {cell} has an invalid Wilson interval")
        n_scored = row.get("n_scored")
        if isinstance(n_scored, bool) or not isinstance(n_scored, int) or n_scored < 1:
            raise CombinedBaseComparisonError(f"{label} {cell} has an invalid n_scored")
        datasets = row.get("datasets")
        if not isinstance(datasets, Sequence) or isinstance(datasets, (str, bytes)) or not all(
            isinstance(dataset, str) and dataset for dataset in datasets
        ):
            raise CombinedBaseComparisonError(f"{label} {cell} has no valid dataset membership")
        observed[cell] = row
    if set(observed) != expected:
        missing, extra = sorted(expected - set(observed)), sorted(set(observed) - expected)
        raise CombinedBaseComparisonError(
            f"{label} must contain the standard two-population, nine-bias base matrix; "
            f"missing={missing}, extra={extra}"
        )
    return [observed[(population, bias)] for population in POPULATION_ORDER for bias in BIAS_ORDER], next(iter(source_conditions))


def _normalise_base_rows(
    source: _LoadedRows, *, profile: _SourceProfile, metric: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows, source_condition = _validate_base_matrix(
        source.rows,
        profile=profile,
        metric=metric,
        label=f"{profile.key} source {source.path}",
    )
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = dict(raw)
        source_condition_label = row.get("condition_label")
        row.update(
            {
                "model": profile.combined_model,
                "model_label": profile.combined_model_label,
                "condition": profile.combined_condition,
                "condition_label": profile.combined_label,
                "method": "none",
                "is_control": False,
                "training_biases": [],
                "provenance_class": profile.provenance_class,
                "source_model": raw.get("model"),
                "source_condition": source_condition,
                "source_condition_label": source_condition_label,
                "comparison_status": COMPARISON_STATUS,
                "significance": "",
                "p_value": None,
                "p_value_raw": None,
                "p_value_holm": None,
                "significance_baseline": None,
                "significance_unavailable_reason": NO_CROSS_MODEL_SIGNIFICANCE_REASON,
            }
        )
        rows.append(row)
    source_manifest = {
        "chart_rows": source.identity,
        "sidecars": source.sidecars,
        "source_model": list(profile.source_models),
        "source_base_condition": source_condition,
        "selected_rows": len(rows),
    }
    return rows, source_manifest


def _included_profile_keys(*, muse_rows: RowsInput | None) -> tuple[str, ...]:
    """Return the displayed model order, keeping Muse genuinely optional."""

    included = tuple(key for key in PROFILE_ORDER if key != "muse" or muse_rows is not None)
    if not set(REQUIRED_PROFILE_KEYS).issubset(included):  # pragma: no cover - defensive invariant
        raise CombinedBaseComparisonError("combined comparison must include both historical Qwen and Gemma")
    return included


def _comparison_title(included: Sequence[str]) -> str:
    labels = {
        "qwen": "Qwen 3.5 9B",
        "muse": "Muse Glimmer 30B",
        "gemma": "Gemma 4 12B",
    }
    names = [labels[key] for key in included]
    if len(names) == 2:
        rendered_names = f"{names[0]} and {names[1]}"
    else:
        rendered_names = ", ".join(names[:-1]) + f", and {names[-1]}"
    return f"Descriptive base-model comparison · {rendered_names}"


def _protocol_caveat() -> dict[str, Any]:
    """Return the fixed audit finding without asserting protocol parity."""

    return {
        "status": "not_protocol_harmonised_descriptive_only",
        "qwen_historical_generation": {
            "max_tokens": QWEN_HISTORICAL_GENERATION_TOKEN_CAP,
            "biased_outputs": QWEN_HISTORICAL_GENERATION_BIASED_OUTPUTS,
            "cap_hits": QWEN_HISTORICAL_GENERATION_CAP_HITS,
            "description": (
                "Historical Qwen base generation used max_tokens=20480; 54 of 1800 biased outputs hit that cap."
            ),
        },
        "qwen_historical_luna": {
            "output_token_cap": QWEN_HISTORICAL_LUNA_OUTPUT_TOKEN_CAP,
            "grades": QWEN_HISTORICAL_GENERATION_BIASED_OUTPUTS,
            "cap_hits": QWEN_HISTORICAL_LUNA_CAP_HITS,
            "valid_grades": QWEN_HISTORICAL_LUNA_VALID_GRADES,
            "description": "Historical Qwen Luna used a 256-token cap; 9 of 1800 grades hit it, leaving 1791 valid.",
        },
        "gemma": {
            "generation_token_cap": None,
            "luna_output_token_cap": None,
            "description": "Gemma generation and Luna grading are uncapped.",
        },
        "consequence": (
            "The combined rows preserve each source estimate for descriptive context; they do not imply generation "
            "or grader-protocol parity and receive no cross-model significance test."
        ),
    }


def _source_rows_for_metric(
    source: RowsInput | None,
    *,
    profile: _SourceProfile,
    metric: str,
    artifact_root: Path,
) -> RowsInput:
    if source is not None:
        return source
    return discover_historical_rows(profile.key, metric=metric, artifact_root=artifact_root)


def _chart_rows_and_sources(
    *,
    metric: str,
    qwen_rows: RowsInput | None,
    muse_rows: RowsInput | None,
    gemma_rows: RowsInput,
    artifact_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if metric not in SUPPORTED_METRICS:
        raise CombinedBaseComparisonError(f"unsupported combined-comparison metric: {metric!r}")
    included = _included_profile_keys(muse_rows=muse_rows)
    inputs: dict[str, RowsInput] = {
        "qwen": _source_rows_for_metric(
            qwen_rows, profile=_PROFILES["qwen"], metric=metric, artifact_root=artifact_root
        ),
        "gemma": gemma_rows,
    }
    if muse_rows is not None:
        inputs["muse"] = muse_rows
    output: list[dict[str, Any]] = []
    source_manifest: dict[str, Any] = {"included_models": list(included), "comparison_title": _comparison_title(included)}
    for key in included:
        profile = _PROFILES[key]
        loaded = _read_rows(inputs[key], metric=metric, label=f"{key} publication")
        rows, evidence = _normalise_base_rows(loaded, profile=profile, metric=metric)
        output.extend(rows)
        source_manifest[key] = evidence
    return output, source_manifest


def chart_rows(
    *,
    metric: str,
    qwen_rows: RowsInput | None = None,
    muse_rows: RowsInput | None = None,
    gemma_rows: RowsInput,
    artifact_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load and normalise the three base row matrices for one standard plot."""

    root = Path(artifact_root).expanduser().resolve() if artifact_root is not None else PROJECT_ROOT / "artifacts"
    rows, _sources = _chart_rows_and_sources(
        metric=metric,
        qwen_rows=qwen_rows,
        muse_rows=muse_rows,
        gemma_rows=gemma_rows,
        artifact_root=root,
    )
    return rows


def publication_spec(*, metric: str, included_models: Sequence[str] = REQUIRED_PROFILE_KEYS) -> dict[str, Any]:
    """Return the common renderer recipe, with an explicit descriptive footer."""

    if metric not in SUPPORTED_METRICS:
        raise CombinedBaseComparisonError(f"unsupported combined-comparison metric: {metric!r}")
    included = tuple(included_models)
    if not included or any(key not in _PROFILES for key in included):
        raise CombinedBaseComparisonError(f"unknown included model keys: {list(included)}")
    if tuple(key for key in PROFILE_ORDER if key in included) != included:
        raise CombinedBaseComparisonError(f"included models must follow canonical order {list(PROFILE_ORDER)}")
    if not set(REQUIRED_PROFILE_KEYS).issubset(included):
        raise CombinedBaseComparisonError("combined comparison must include both historical Qwen and Gemma")
    profiles = [_PROFILES[key] for key in included]
    title = _comparison_title(included)
    labels = registry_labels(load_presentation_registry().biases)
    if metric == "towards_bias_switch":
        ylabel = "Towards-bias switch rate (eligible paired questions)"
        footer = (
            f"{title}. Descriptive comparison only: no cross-model paired significance tests or stars are claimed. "
            "Bars retain each model's original question pools; error bars are 95% Wilson intervals over eligible "
            "clean-not-bias paired questions. Historical Qwen generation used max_tokens=20480 (54/1800 biased "
            "outputs hit it); Gemma generation is uncapped. Do not infer protocol parity."
        )
    else:
        ylabel = "Bias verbalised (Luna YES | valid grade)"
        footer = (
            f"{title}. Descriptive comparison only: no cross-model paired significance tests or stars are claimed. "
            "Historical Qwen generation used max_tokens=20480 (54/1800 biased outputs hit it), and Qwen Luna used "
            "a 256-token cap (9/1800 grades hit it; 1791 valid); Gemma generation and Luna are uncapped. "
            "Verbalisation bars are not protocol-harmonised. Error bars are 95% Wilson intervals over each source's "
            "valid Luna grades."
        )
    return {
        "title": title,
        "included_models": list(included),
        "protocol_caveat": _protocol_caveat(),
        "metric": metric,
        "facet": {"rows": ["population"]},
        "facet_labels": {
            "population": {
                "held_in_datasets": "Held-in datasets · LogiQA + HellaSwag",
                "held_out_dataset": "Held-out dataset · HLE text-MC",
            }
        },
        "model_order": [profile.combined_model for profile in profiles],
        "model_labels": {profile.combined_model: profile.combined_model_label for profile in profiles},
        "condition_order": [profile.combined_condition for profile in profiles],
        "condition_labels": {profile.combined_condition: profile.combined_label for profile in profiles},
        "condition_styles": {profile.combined_condition: {"color": profile.color} for profile in profiles},
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
        # Keep this enabled solely to render the transparent inferential
        # caveat below. Every output row has an empty marker.
        "show_significance": True,
        "significance_note": footer,
        "sample_labels": "n_scored",
        "legend_columns": 3,
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


def discover_historical_rows(
    model: str,
    *,
    metric: str,
    artifact_root: str | Path = PROJECT_ROOT / "artifacts",
) -> Path:
    """Find one exact historical base publication without choosing among ties.

    Known sealed Qwen consistency-gap paths are preferred.  For Muse (and for
    future artifact layouts), a semantic search is allowed only when it yields
    exactly one complete base matrix.  This makes an accidental stale-artifact
    selection a visible error rather than a silent plotting choice.
    """

    if model not in {"qwen", "muse"}:
        raise CombinedBaseComparisonError(f"only historical Qwen or Muse rows can be discovered, not {model!r}")
    if metric not in SUPPORTED_METRICS:
        raise CombinedBaseComparisonError(f"unsupported combined-comparison metric: {metric!r}")
    root = Path(artifact_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"historical artifact root must be a regular directory: {root}")
    profile = _PROFILES[model]
    for relative_path in _KNOWN_HISTORICAL_ROWS[model][metric]:
        candidate = root / relative_path
        if not candidate.exists() and not candidate.is_symlink():
            continue
        loaded = _read_rows(candidate, metric=metric, label=f"known {model} publication")
        _normalise_base_rows(loaded, profile=profile, metric=metric)
        return candidate.resolve()

    candidates: list[Path] = []
    for candidate in sorted(root.rglob("chart-rows.json")):
        try:
            loaded = _read_rows(candidate, metric=metric, label=f"candidate {model} publication")
            _normalise_base_rows(loaded, profile=profile, metric=metric)
        except (CombinedBaseComparisonError, FileNotFoundError, OSError):
            continue
        candidates.append(candidate.resolve())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"could not find a complete {model} base chart-row matrix for {metric!r} under {root}; pass an explicit path"
        )
    raise CombinedBaseComparisonError(
        f"found multiple complete {model} base chart-row matrices for {metric!r}: "
        f"{[str(path) for path in candidates]}; pass an explicit path"
    )


def _published_identity(staged_path: Path, *, published_path: Path, label: str) -> dict[str, Any]:
    identity = _identity(staged_path, label=label)
    identity["path"] = str(published_path)
    return identity


def render_combined_base_comparison(
    *,
    gemma_rows: RowsInput,
    output_dir: str | Path,
    qwen_rows: RowsInput | None = None,
    muse_rows: RowsInput | None = None,
    artifact_root: str | Path | None = None,
) -> Path:
    """Render standard switch and verbalisation figures from three base sources."""

    root = Path(artifact_root).expanduser().resolve() if artifact_root is not None else PROJECT_ROOT / "artifacts"
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite combined base-model publication: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise CombinedBaseComparisonError("combined publication parent must be a regular directory")

    rows_by_metric: dict[str, list[dict[str, Any]]] = {}
    sources_by_metric: dict[str, dict[str, Any]] = {}
    included = _included_profile_keys(muse_rows=muse_rows)
    for metric in SUPPORTED_METRICS:
        rows, sources = _chart_rows_and_sources(
            metric=metric,
            qwen_rows=qwen_rows,
            muse_rows=muse_rows,
            gemma_rows=gemma_rows,
            artifact_root=root,
        )
        rows_by_metric[metric] = rows
        sources_by_metric[metric] = sources
    specs = {metric: publication_spec(metric=metric, included_models=included) for metric in SUPPORTED_METRICS}

    # Unlike a temporary-directory context manager, this staging directory is
    # deliberately left intact on a failed render for audit and recovery.  It
    # is renamed only after every byte has been produced successfully.
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for metric in SUPPORTED_METRICS:
            stem = OUTPUT_STEMS[metric]
            (staging / f"{stem}-chart-rows.json").write_bytes(_json_bytes(rows_by_metric[metric]))
            (staging / f"{stem}-chart-spec.json").write_bytes(_json_bytes(specs[metric]))
            for extension in ("png", "svg"):
                render_publication_plot(rows_by_metric[metric], specs[metric], staging / f"{stem}.{extension}")
        outputs = {
            path.name: _published_identity(
                path,
                published_path=output / path.name,
                label="combined base-model publication output",
            )
            for path in sorted(staging.iterdir())
        }
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "title": _comparison_title(included),
            "included_models": list(included),
            "comparison_status": COMPARISON_STATUS,
            "cross_model_significance": {
                "claimed": False,
                "reason": NO_CROSS_MODEL_SIGNIFICANCE_REASON,
                "markers": "all empty; no cross-model stars are rendered",
            },
            "metrics": list(SUPPORTED_METRICS),
            "models": {
                key: {
                    "combined_model": profile.combined_model,
                    "combined_condition": profile.combined_condition,
                    "label": profile.combined_label,
                }
                for key, profile in _PROFILES.items()
                if key in included
            },
            "source_inputs": sources_by_metric,
            "verbalisation_protocol_compatibility": {
                "status": "not_harmonised_descriptive_only",
                "qwen": {
                    "luna_output_token_cap": QWEN_HISTORICAL_LUNA_OUTPUT_TOKEN_CAP,
                    "description": "Historical Qwen Luna verbalisation grading used a 256-token output cap.",
                },
                "gemma": {
                    "luna_output_token_cap": None,
                    "description": "Gemma base Luna verbalisation grading is uncapped.",
                },
                "consequence": (
                    "Qwen and Gemma verbalisation estimates are shown side by side for descriptive context only; "
                    "they are not protocol-harmonised and receive no cross-model significance test."
                ),
            },
            "protocol_caveat": _protocol_caveat(),
            "outputs": outputs,
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        # Recheck immediately before the move so a concurrently-created output
        # is never intentionally replaced.
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite combined base-model publication: {output}")
        os.rename(staging, output)
    except BaseException:
        # Preserve the unique staging directory rather than deleting evidence
        # with a recursive cleanup operation.
        raise
    return output


def _cli_rows_input(
    *,
    common: Path | None,
    switch: Path | None,
    verbalisation: Path | None,
) -> RowsInput | None:
    if common is None and switch is None and verbalisation is None:
        return None
    source: dict[str, str | Path] = {}
    if common is not None:
        source = {metric: common for metric in SUPPORTED_METRICS}
    if switch is not None:
        source["towards_bias_switch"] = switch
    if verbalisation is not None:
        source["bias_acknowledged"] = verbalisation
    return source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT / "artifacts")
    for prefix, help_prefix in (
        ("gemma", "Gemma base publication rows (required; a directory may contain both metrics)"),
        ("qwen", "historical Qwen base rows (optional; otherwise discover the sealed current artifacts)"),
        ("muse", "historical Muse base rows (optional; otherwise discover exactly one complete publication)"),
    ):
        parser.add_argument(f"--{prefix}-rows", type=Path, help=help_prefix)
        parser.add_argument(f"--{prefix}-switch-rows", type=Path)
        parser.add_argument(f"--{prefix}-verbalisation-rows", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    gemma_rows = _cli_rows_input(
        common=args.gemma_rows,
        switch=args.gemma_switch_rows,
        verbalisation=args.gemma_verbalisation_rows,
    )
    if gemma_rows is None:
        parser.error("provide --gemma-rows or both metric-specific Gemma row paths")
    try:
        result = render_combined_base_comparison(
            gemma_rows=gemma_rows,
            qwen_rows=_cli_rows_input(
                common=args.qwen_rows,
                switch=args.qwen_switch_rows,
                verbalisation=args.qwen_verbalisation_rows,
            ),
            muse_rows=_cli_rows_input(
                common=args.muse_rows,
                switch=args.muse_switch_rows,
                verbalisation=args.muse_verbalisation_rows,
            ),
            artifact_root=args.artifact_root,
            output_dir=args.output_dir,
        )
    except (CombinedBaseComparisonError, FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "COMPARISON_STATUS",
    "CombinedBaseComparisonError",
    "NO_CROSS_MODEL_SIGNIFICANCE_REASON",
    "OUTPUT_SCHEMA",
    "QWEN_HISTORICAL_LUNA_OUTPUT_TOKEN_CAP",
    "chart_rows",
    "discover_historical_rows",
    "publication_spec",
    "render_combined_base_comparison",
]
