#!/usr/bin/env python3
"""Render the strict EvalAwareBench Figure 6 reproduction heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm_data.adapters.eval_awareness.figure6_analysis import (
    EXPECTED_CELL_COUNT,
    EXPECTED_MODEL_JUDGMENT_COUNT,
    MODEL_KEY_ORDER,
    OPENROUTER_DEEPSEEK_V32_ALTERNATIVE_LABEL,
    OPENROUTER_DEEPSEEK_V32_JUDGE_MODEL,
    OPENROUTER_DEEPSEEK_V32_PROFILE,
    OPENROUTER_MUSE_ALTERNATIVE_LABEL,
    OPENROUTER_MUSE_JUDGE_MODEL,
    QWEN_MODEL_KEY_ORDER,
    STRICT_SUBSET_ALTERNATIVE_RESULT_LABEL,
    STRICT_SUBSET_RESULT_LABEL,
    should_annotate_delta,
)
from ctm_data.adapters.eval_awareness.figure6_judge import DEFAULT_JUDGE_MODEL
from ctm_data.adapters.eval_awareness.figure6_spec import (
    FIGURE6_CONDITIONS,
    FIGURE6_TASK_COUNT,
    FIGURE6_VALENCES,
    MODEL_SPECS,
)

MODEL_DISPLAY_ORDER = tuple(model.display_name for model in MODEL_SPECS.values())
CONFIG_LABELS = {"baseline": "BL", **{f"F{index}": f"F{index}" for index in range(1, 9)}}
EXPECTED_DENOMINATOR = EXPECTED_CELL_COUNT
POSITIVE_DELTA_COLOR = "#c62828"
NEGATIVE_DELTA_COLOR = "#217a3c"
DEFAULT_TITLE = "EvalAwareBench factor reproduction"
DEFAULT_SOURCE_NOTE = "Computed from supplied generation and judge artifacts; not paper result data."
FIXTURE_TITLE = "EvalAwareBench rendering fixture"
FIXTURE_SOURCE_NOTE = "Synthetic offline fixture for renderer verification; not experimental result data."
FULL_RESULT_LABELS = frozenset(
    {
        "complete_reproduction",
        "complete_reproduction_user_pinned_alternative_judge",
    }
)
SUBSET_RESULT_LABELS = frozenset(
    {
        STRICT_SUBSET_RESULT_LABEL,
        STRICT_SUBSET_ALTERNATIVE_RESULT_LABEL,
    }
)
ALTERNATIVE_RESULT_LABELS = frozenset(
    {
        "complete_reproduction_user_pinned_alternative_judge",
        STRICT_SUBSET_ALTERNATIVE_RESULT_LABEL,
    }
)


def _bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
        return value.strip().casefold() == "true"
    raise ValueError(f"{name} must be a boolean")


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return result


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _summary_value(summary: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if summary is None:
        return None
    value = summary.get("summary", summary)
    return value if isinstance(value, Mapping) else None


def _expected_model_keys(
    summary: Mapping[str, Any] | None,
    *,
    fixture_mode: bool,
) -> tuple[str, ...]:
    if fixture_mode or summary is None:
        return MODEL_KEY_ORDER
    value = _summary_value(summary)
    if value is None:
        raise ValueError("analysis summary must contain an object")
    expected = value.get("expected")
    if expected is None:
        # Compatibility for strict seven-model summaries written before the
        # ordered model scope was added to the schema.
        return MODEL_KEY_ORDER
    if not isinstance(expected, Mapping):
        raise ValueError("analysis summary expected scope must be an object")
    raw_model_keys = expected.get("model_keys")
    if (
        not isinstance(raw_model_keys, Sequence)
        or isinstance(raw_model_keys, (str, bytes))
        or not raw_model_keys
        or any(not isinstance(key, str) or not key for key in raw_model_keys)
    ):
        raise ValueError("analysis summary must contain a non-empty ordered model_keys list")
    model_keys = tuple(raw_model_keys)
    if len(set(model_keys)) != len(model_keys):
        raise ValueError("analysis summary model_keys must be unique")
    unregistered = [key for key in model_keys if key not in MODEL_SPECS]
    if unregistered:
        raise ValueError(f"analysis summary contains unregistered model_keys: {unregistered}")
    expected_displays = [MODEL_SPECS[key].display_name for key in model_keys]
    if expected.get("model_displays") != expected_displays:
        raise ValueError("analysis summary model_displays do not match its ordered pinned model_keys")
    return model_keys


def _alternative_plot_label(summary: Mapping[str, Any] | None) -> str | None:
    value = _summary_value(summary)
    if value is None or value.get("result_label") not in ALTERNATIVE_RESULT_LABELS:
        return None
    label = value.get("plot_label")
    if not isinstance(label, str) or not label:
        raise ValueError("alternative-judge analysis summary must include a non-empty plot_label")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("alternative-judge analysis summary must include provenance")
    if provenance.get("expected_judge_model") == OPENROUTER_MUSE_JUDGE_MODEL:
        if provenance.get("judge_provider") != "OpenRouter" or label != OPENROUTER_MUSE_ALTERNATIVE_LABEL:
            raise ValueError("OpenRouter Muse alternative summary has an invalid provider or plot label")
    if provenance.get("expected_judge_model") == OPENROUTER_DEEPSEEK_V32_JUDGE_MODEL:
        if (
            provenance.get("judge_provider") != "OpenRouter"
            or provenance.get("judge_profile") != OPENROUTER_DEEPSEEK_V32_PROFILE
            or label != OPENROUTER_DEEPSEEK_V32_ALTERNATIVE_LABEL
        ):
            raise ValueError("OpenRouter DeepSeek alternative summary has an invalid profile, provider, or label")
    source_note = value.get("source_note")
    if not isinstance(source_note, str) or label not in source_note:
        raise ValueError("alternative-judge analysis source note must include its plot label")
    return label


def _subset_plot_label(summary: Mapping[str, Any] | None) -> str | None:
    value = _summary_value(summary)
    if value is None or value.get("result_label") not in SUBSET_RESULT_LABELS:
        return None
    if value.get("scope_kind") != "registered_model_subset":
        raise ValueError("strict subset analysis summary must identify a registered-model subset scope")
    label = value.get("scope_label")
    if not isinstance(label, str) or not label:
        raise ValueError("strict subset analysis summary must include a non-empty scope_label")
    model_keys = _expected_model_keys(summary, fixture_mode=False)
    subset_name = "Qwen-only subset" if set(model_keys).issubset(QWEN_MODEL_KEY_ORDER) else "Registered-model subset"
    expected_label = (
        f"{subset_name} ({len(model_keys)} of {len(MODEL_KEY_ORDER)} models; "
        f"{len(model_keys) * EXPECTED_MODEL_JUDGMENT_COUNT:,} strict judgments)"
    )
    if label != expected_label:
        raise ValueError(f"strict subset analysis scope_label must be exactly {expected_label!r}")
    source_note = value.get("source_note")
    if not isinstance(source_note, str) or label not in source_note:
        raise ValueError("strict subset analysis source note must include its scope label")
    return label


def validate_plot_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any] | None = None,
    fixture_mode: bool = False,
) -> list[dict[str, Any]]:
    """Require the summary-scoped complete strict aggregation matrix."""

    if not rows:
        raise ValueError("plot rows must not be empty")
    model_keys = _expected_model_keys(summary, fixture_mode=fixture_mode)
    required = {
        "model_key",
        "model_display",
        "model_id",
        "model_revision",
        "valence",
        "config_name",
        "n",
        "n_tasks",
        "n_replicates",
        "matched_awareness_percent",
        "performance_delta_pp",
        "annotate_performance_delta",
        "publication_complete",
    }
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, source in enumerate(rows, start=1):
        if not isinstance(source, Mapping):
            raise TypeError(f"plot row {index} must be an object")
        missing = sorted(required - source.keys())
        if missing:
            raise ValueError(f"plot row {index} is missing fields: {missing}")
        row = dict(source)
        model_key = row["model_key"]
        if model_key not in MODEL_SPECS:
            raise ValueError(f"plot row {index} has unexpected model_key {model_key!r}")
        model = MODEL_SPECS[model_key]
        identity = (row["model_display"], row["model_id"], row["model_revision"])
        expected_identity = (model.display_name, model.model_id, model.revision)
        if identity != expected_identity:
            raise ValueError(f"plot row {index} does not match the pinned identity for {model_key!r}")
        if row["valence"] not in FIGURE6_VALENCES:
            raise ValueError(f"plot row {index} has unexpected valence {row['valence']!r}")
        if row["config_name"] not in FIGURE6_CONDITIONS:
            raise ValueError(f"plot row {index} has unexpected config_name {row['config_name']!r}")
        key = (model_key, row["valence"], row["config_name"])
        if key in seen:
            raise ValueError(f"duplicate plot cell {key}")
        seen.add(key)
        if _integer(row["n"], name=f"row {index}.n") != EXPECTED_DENOMINATOR:
            raise ValueError(f"plot row {index} must have the exact denominator {EXPECTED_DENOMINATOR}")
        if _integer(row["n_tasks"], name=f"row {index}.n_tasks") != FIGURE6_TASK_COUNT:
            raise ValueError(f"plot row {index} must contain {FIGURE6_TASK_COUNT} tasks")
        if _integer(row["n_replicates"], name=f"row {index}.n_replicates") != 3:
            raise ValueError(f"plot row {index} must contain three replicates")
        publication_complete = _bool(row["publication_complete"], name=f"row {index}.publication_complete")
        is_fixture = _bool(row.get("fixture_mode", False), name=f"row {index}.fixture_mode")
        if fixture_mode and not is_fixture:
            raise ValueError("fixture mode requires every row to be explicitly marked as a fixture")
        if not fixture_mode and not publication_complete:
            raise ValueError("incomplete diagnostic rows cannot be rendered as the publication reproduction")
        if not fixture_mode and is_fixture:
            raise ValueError("fixture rows require explicit fixture mode")
        awareness = _finite(row["matched_awareness_percent"], name=f"row {index}.matched_awareness_percent")
        if not 0.0 <= awareness <= 100.0:
            raise ValueError(f"plot row {index}.matched_awareness_percent must be in [0, 100]")
        delta = _finite(row["performance_delta_pp"], name=f"row {index}.performance_delta_pp")
        annotation = _bool(
            row["annotate_performance_delta"],
            name=f"row {index}.annotate_performance_delta",
        )
        expected_annotation = row["config_name"] != "baseline" and should_annotate_delta(delta)
        if annotation != expected_annotation:
            raise ValueError(f"plot row {index} has an inconsistent performance annotation threshold")
        if row["config_name"] == "baseline" and not math.isclose(delta, 0.0, abs_tol=1e-12):
            raise ValueError(f"plot row {index} baseline performance delta must be zero")
        row.update(
            {
                "n": EXPECTED_DENOMINATOR,
                "n_tasks": FIGURE6_TASK_COUNT,
                "n_replicates": 3,
                "matched_awareness_percent": awareness,
                "performance_delta_pp": delta,
                "annotate_performance_delta": annotation,
                "publication_complete": publication_complete,
                "fixture_mode": is_fixture,
            }
        )
        normalized.append(row)

    expected = {
        (model_key, valence, config)
        for model_key in model_keys
        for valence in FIGURE6_VALENCES
        for config in FIGURE6_CONDITIONS
    }
    if seen != expected:
        missing = sorted(expected - seen)[:8]
        extra = sorted(seen - expected)[:8]
        raise ValueError(f"plot matrix is incomplete or has extra cells; missing={missing}, extra={extra}")
    records_by_model = {
        model_key: sum(row["n"] for row in normalized if row["model_key"] == model_key) for model_key in model_keys
    }
    if any(count != EXPECTED_MODEL_JUDGMENT_COUNT for count in records_by_model.values()):
        raise ValueError(
            f"every plotted model must represent exactly {EXPECTED_MODEL_JUDGMENT_COUNT:,} strict judgments; "
            f"observed={records_by_model}"
        )
    if summary is not None and not fixture_mode:
        summary_value = _summary_value(summary)
        if not isinstance(summary_value, Mapping) or summary_value.get("publication_complete") is not True:
            raise ValueError("analysis summary does not mark this reproduction complete")
        expected_total = len(model_keys) * EXPECTED_MODEL_JUDGMENT_COUNT
        observed = summary_value.get("observed")
        if not isinstance(observed, Mapping) or observed.get("valid_unique_records") != expected_total:
            raise ValueError(
                f"analysis summary does not report exactly {expected_total:,} valid unique judgments "
                f"for its {len(model_keys)}-model scope"
            )
        result_label = summary_value.get("result_label")
        is_subset = set(model_keys) != set(MODEL_KEY_ORDER)
        supported_labels = SUBSET_RESULT_LABELS if is_subset else FULL_RESULT_LABELS
        if result_label not in supported_labels:
            raise ValueError("analysis summary has an unsupported complete-result label")
        expected_scope = summary_value.get("expected")
        if expected_scope is not None:
            if not isinstance(expected_scope, Mapping):
                raise ValueError("analysis summary expected scope must be an object")
            exact_expected_values = {
                "valences": list(FIGURE6_VALENCES),
                "configs": list(FIGURE6_CONDITIONS),
                "task_count": FIGURE6_TASK_COUNT,
                "replicates": [1, 2, 3],
                "judgment_count": expected_total,
                "cell_denominator": EXPECTED_DENOMINATOR,
            }
            mismatched_expected = {
                name: {"expected": expected_value, "observed": expected_scope.get(name)}
                for name, expected_value in exact_expected_values.items()
                if expected_scope.get(name) != expected_value
            }
            if mismatched_expected:
                raise ValueError(f"analysis summary has an incompatible strict matrix scope: {mismatched_expected}")
            if (
                "judgments_per_model" in expected_scope
                and expected_scope.get("judgments_per_model") != EXPECTED_MODEL_JUDGMENT_COUNT
            ):
                raise ValueError("analysis summary must require exactly 5,400 judgments per model")
            expected_observed_by_model = {model_key: EXPECTED_MODEL_JUDGMENT_COUNT for model_key in model_keys}
            observed_by_model = observed.get("valid_unique_records_by_model")
            if observed_by_model is not None and observed_by_model != expected_observed_by_model:
                raise ValueError(
                    "analysis summary does not report exactly 5,400 valid unique judgments for every model"
                )
            if is_subset and observed_by_model is None:
                raise ValueError("strict subset summaries must report valid unique judgments for every model")
            expected_aggregate_rows = len(model_keys) * len(FIGURE6_VALENCES) * len(FIGURE6_CONDITIONS)
            if observed.get("aggregate_rows") != expected_aggregate_rows:
                raise ValueError(f"analysis summary does not report exactly {expected_aggregate_rows} aggregate rows")
            if (is_subset or "scope_complete" in summary_value) and summary_value.get("scope_complete") is not True:
                raise ValueError("analysis summary does not mark its strict model scope complete")
        provenance = summary_value.get("provenance")
        if result_label in {"complete_reproduction", STRICT_SUBSET_RESULT_LABEL} and (
            not isinstance(provenance, Mapping)
            or provenance.get("expected_judge_model", provenance.get("judge_model")) != DEFAULT_JUDGE_MODEL
        ):
            raise ValueError("paper-judge plot summary does not identify the default GPT-5 judge")
        _alternative_plot_label(summary)
        _subset_plot_label(summary)
    return normalized


def performance_delta_annotation(delta_pp: float, *, config_name: str) -> tuple[str, str] | None:
    """Return paper-style signed text/color only beyond the strict threshold."""

    if config_name == "baseline" or not should_annotate_delta(delta_pp):
        return None
    color = POSITIVE_DELTA_COLOR if delta_pp > 0 else NEGATIVE_DELTA_COLOR
    return f"{delta_pp:+.0f}", color


def _lazy_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render Figure 6") from exc
    return matplotlib, plt


def render_figure6(
    rows: Sequence[Mapping[str, Any]],
    *,
    png_path: str | Path,
    pdf_path: str | Path,
    summary: Mapping[str, Any] | None = None,
    title: str = DEFAULT_TITLE,
    source_note: str = DEFAULT_SOURCE_NOTE,
    fixture_mode: bool = False,
) -> tuple[Path, Path]:
    """Render complete strict rows to both PNG and PDF."""

    normalized = validate_plot_rows(rows, summary=summary, fixture_mode=fixture_mode)
    model_keys = _expected_model_keys(summary, fixture_mode=fixture_mode)
    model_displays = tuple(MODEL_SPECS[key].display_name for key in model_keys)
    subset_label = _subset_plot_label(summary) if not fixture_mode else None
    alternative_label = _alternative_plot_label(summary) if not fixture_mode else None
    if fixture_mode and title == DEFAULT_TITLE:
        title = FIXTURE_TITLE
    if fixture_mode and source_note == DEFAULT_SOURCE_NOTE:
        source_note = FIXTURE_SOURCE_NOTE
    required_labels = [label for label in (subset_label, alternative_label) if label is not None]
    if required_labels:
        if title == DEFAULT_TITLE:
            title = f"{DEFAULT_TITLE} — {' — '.join(required_labels)}"
        if source_note == DEFAULT_SOURCE_NOTE:
            summary_value = _summary_value(summary)
            summary_source_note = summary_value.get("source_note") if summary_value is not None else None
            source_note = (
                summary_source_note
                if subset_label is not None and isinstance(summary_source_note, str) and summary_source_note
                else f"{DEFAULT_SOURCE_NOTE} {'; '.join(required_labels)}."
            )
        missing_labels = [label for label in required_labels if label not in title or label not in source_note]
        if missing_labels:
            raise ValueError(
                "strict subset and alternative-judge plots must render every exact label in both the title "
                f"and source note; missing={missing_labels}"
            )
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    if not isinstance(source_note, str) or not source_note.strip():
        raise ValueError("source_note must be a non-empty string")
    png_target = Path(png_path)
    pdf_target = Path(pdf_path)
    if png_target.suffix.lower() != ".png" or pdf_target.suffix.lower() != ".pdf":
        raise ValueError("png_path must end in .png and pdf_path must end in .pdf")
    for target in (png_target, pdf_target):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing figure: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    _, plt = _lazy_matplotlib()
    lookup = {(row["model_key"], row["valence"], row["config_name"]): row for row in normalized}
    rcparams = {
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
    }
    with plt.rc_context(rcparams):
        figure, axes = plt.subplots(1, 2, figsize=(14.4, 5.0), sharey=True, constrained_layout=True)
        figure.suptitle(title, fontsize=12, fontweight="semibold")
        all_awareness_values = [row["matched_awareness_percent"] for row in normalized]
        shared_vmax = max(all_awareness_values)
        if shared_vmax <= 0:
            shared_vmax = 1.0
        for panel_index, (axis, valence) in enumerate(zip(axes, FIGURE6_VALENCES, strict=True)):
            matrix = [
                [lookup[(model_key, valence, config)]["matched_awareness_percent"] for config in FIGURE6_CONDITIONS]
                for model_key in model_keys
            ]
            axis.imshow(
                matrix,
                cmap="Blues",
                vmin=0,
                vmax=shared_vmax,
                aspect="auto",
                interpolation="nearest",
            )
            axis.set_title(f"{valence.title()} Tasks", pad=10)
            axis.set_xticks(range(len(FIGURE6_CONDITIONS)), [CONFIG_LABELS[name] for name in FIGURE6_CONDITIONS])
            axis.set_yticks(range(len(model_keys)), model_displays)
            axis.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True, length=0, pad=5)
            axis.tick_params(axis="y", length=0, pad=6)
            if panel_index:
                axis.tick_params(axis="y", labelleft=False)
            axis.set_xticks([value - 0.5 for value in range(1, len(FIGURE6_CONDITIONS))], minor=True)
            axis.set_yticks([value - 0.5 for value in range(1, len(model_keys))], minor=True)
            axis.grid(which="minor", color="white", linewidth=1.2)
            axis.tick_params(which="minor", bottom=False, left=False)
            for row_index, model_key in enumerate(model_keys):
                for column_index, config in enumerate(FIGURE6_CONDITIONS):
                    row = lookup[(model_key, valence, config)]
                    awareness = row["matched_awareness_percent"]
                    axis.text(
                        column_index,
                        row_index,
                        f"{awareness:.0f}%",
                        ha="center",
                        va="center",
                        color="white" if awareness >= shared_vmax * 0.52 else "#14243a",
                        fontsize=8.2,
                        fontweight="medium",
                    )
                    annotation = performance_delta_annotation(
                        row["performance_delta_pp"],
                        config_name=config,
                    )
                    if annotation is not None:
                        annotation_text, annotation_color = annotation
                        axis.text(
                            column_index + 0.42,
                            row_index - 0.36,
                            annotation_text,
                            ha="right",
                            va="top",
                            color=annotation_color,
                            fontsize=6.3,
                            fontweight="bold",
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.45},
                        )
            for spine in axis.spines.values():
                spine.set_visible(False)

        figure.text(0.5, -0.015, source_note, ha="center", va="top", fontsize=7.5, color="#4f5660")
        metadata_text = f"{title}. {source_note}"
        figure.savefig(
            png_target,
            format="png",
            metadata={"Title": title, "Description": metadata_text, "Software": "consistency-training-methods"},
            bbox_inches="tight",
            pad_inches=0.08,
        )
        figure.savefig(
            pdf_target,
            format="pdf",
            metadata={"Title": title, "Subject": metadata_text, "Creator": "consistency-training-methods"},
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close(figure)
    return png_target, pdf_target


def fixture_rows() -> list[dict[str, Any]]:
    """Generate a benign complete matrix for offline renderer smoke tests."""

    rows: list[dict[str, Any]] = []
    for model_index, (model_key, model) in enumerate(MODEL_SPECS.items()):
        for valence_index, valence in enumerate(FIGURE6_VALENCES):
            baseline_performance = 54.0 + model_index - valence_index * 2
            for config_index, config_name in enumerate(FIGURE6_CONDITIONS):
                awareness = min(94.0, 8.0 + 7.0 * config_index + 2.0 * model_index + 3.0 * valence_index)
                delta = 0.0 if config_name == "baseline" else ((config_index % 5) - 2) * 3.2
                performance = baseline_performance + delta
                rows.append(
                    {
                        "model_key": model_key,
                        "model_display": model.display_name,
                        "model_id": model.model_id,
                        "model_revision": model.revision,
                        "valence": valence,
                        "config_name": config_name,
                        "n": EXPECTED_DENOMINATOR,
                        "n_tasks": FIGURE6_TASK_COUNT,
                        "n_replicates": 3,
                        "matched_awareness_percent": awareness,
                        "performance_yes_percent": performance,
                        "performance_delta_pp": delta,
                        "annotate_performance_delta": config_name != "baseline" and should_annotate_delta(delta),
                        "publication_complete": False,
                        "fixture_mode": True,
                    }
                )
    return rows


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _outputs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[Path, Path]:
    if args.output_prefix is not None:
        if args.png is not None or args.pdf is not None:
            parser.error("--output-prefix cannot be combined with --png or --pdf")
        return args.output_prefix.with_suffix(".png"), args.output_prefix.with_suffix(".pdf")
    if args.png is None or args.pdf is None:
        parser.error("provide --output-prefix or both --png and --pdf")
    return args.png, args.pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", type=Path)
    source.add_argument("--fixture", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--source-note", default=DEFAULT_SOURCE_NOTE)
    args = parser.parse_args(argv)
    if not args.fixture and args.summary_json is None:
        parser.error("--summary-json is required with --input-csv")
    png_path, pdf_path = _outputs(args, parser)
    rows = fixture_rows() if args.fixture else _read_csv(args.input_csv)
    summary = json.loads(args.summary_json.read_text(encoding="utf-8")) if args.summary_json is not None else None
    render_figure6(
        rows,
        png_path=png_path,
        pdf_path=pdf_path,
        summary=summary,
        title=FIXTURE_TITLE if args.fixture and args.title == DEFAULT_TITLE else args.title,
        source_note=(
            FIXTURE_SOURCE_NOTE if args.fixture and args.source_note == DEFAULT_SOURCE_NOTE else args.source_note
        ),
        fixture_mode=args.fixture,
    )
    print(json.dumps({"png": str(png_path), "pdf": str(pdf_path), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
