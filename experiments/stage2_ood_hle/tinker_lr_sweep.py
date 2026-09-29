"""Reanalyse and plot the cot-transparency Tinker learning-rate sweep.

The historical cot-transparency analysis reported an unconditional ``pro_bsr``.
This adapter instead reconstructs the current conditional TBSR estimand from
the raw paired Inspect samples::

    P(biased answer = bias option | clean answer != bias option, jointly parsed)

The Llama and GPT-OSS results remain separate facets because they use different
reasoning interfaces.  They are also kept separate from the newer Qwen/HLE
figures: the dataset, model, prompt interface, and verbalisation grader differ.
All drawing is delegated to CTM's standard publication renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from experiments.stage2_ood_hle.analyze import (
    AnalysisConfig,
    BIAS_VERBALISED_METRIC,
    HLE_POPULATION,
    Observation,
    TBSR_METRIC,
    _attach_group_significance,
    _cell,
)


SCHEMA = "cot-transparency-tinker-lr-sweep-v1"
HELD_OUT_MEAN = "held_out_mean"
TRAINING_BIAS = "distractor_argument_g4"
HELD_OUT_BIASES = (
    "suggested_answer",
    "distractor_fact",
    "spurious_few_shot_squares",
    "wrong_few_shot",
)
ALL_BIASES = (TRAINING_BIAS, *HELD_OUT_BIASES)
EXPECTED_SAMPLES = 150


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    directory_prefix: str
    expected_model: str
    expected_prompt_style: str
    label: str


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    key: str
    directory_suffix: str
    method: str
    label: str
    color: str


MODEL_SPECS = (
    ModelSpec(
        key="tinker-llama31-8b",
        directory_prefix="llama",
        expected_model="tinker-sampling/meta-llama/Llama-3.1-8B-Instruct",
        expected_prompt_style="cot",
        label="Llama 3.1 8B-Instruct · TruthfulQA · COT · Tinker",
    ),
    ModelSpec(
        key="tinker-gpt-oss-20b",
        directory_prefix="gpt-oss-20b",
        expected_model="tinker-sampling/openai/gpt-oss-20b",
        expected_prompt_style="no_cot",
        label="GPT-OSS 20B · TruthfulQA · native reasoning · Tinker",
    ),
)

CONDITION_SPECS = (
    ConditionSpec("untrained", "base", "none", "Base [Tinker]", "#8f959e"),
    ConditionSpec(
        "tinker-bct-lr1e-4",
        "bct-da-g4-lr1e4",
        "bias_augmented_consistency",
        "BCT 1e-4 [Tinker]",
        "#bdd7ee",
    ),
    ConditionSpec(
        "tinker-bct-lr2.86e-4",
        "bct-da-g4-lr2p86e4",
        "bias_augmented_consistency",
        "BCT 2.86e-4 [Tinker]",
        "#5b9bd5",
    ),
    ConditionSpec(
        "tinker-bct-lr5e-4",
        "bct-da-g4-lr5e4",
        "bias_augmented_consistency",
        "BCT 5e-4 [Tinker]",
        "#205493",
    ),
    ConditionSpec(
        "tinker-rmct-lr1e-4",
        "rlct-da-g4-lr1e4",
        "rate_matching",
        "RMCT 1e-4 [Tinker]",
        "#c6e0b4",
    ),
    ConditionSpec(
        "tinker-rmct-lr2.86e-4",
        "rlct-da-g4-lr2p86e4",
        "rate_matching",
        "RMCT 2.86e-4 [Tinker]",
        "#70ad47",
    ),
    ConditionSpec(
        "tinker-rmct-lr5e-4",
        "rlct-da-g4-lr5e4",
        "rate_matching",
        "RMCT 5e-4 [Tinker]",
        "#385723",
    ),
)

CONDITIONS = tuple(spec.key for spec in CONDITION_SPECS)
CONDITION_BY_KEY = {spec.key: spec for spec in CONDITION_SPECS}
MODEL_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_member(archive: zipfile.ZipFile, name: str) -> Mapping[str, Any]:
    value = json.loads(archive.read(name))
    if not isinstance(value, Mapping):
        raise TypeError(f"{archive.filename}:{name} must contain a JSON object")
    return value


def _score_answer(sample: Mapping[str, Any], *, source: str) -> str | None:
    scores = sample.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError(f"{source} has no score mapping")
    score = scores.get("mcq_bias_scorer")
    if not isinstance(score, Mapping) or not isinstance(score.get("value"), Mapping):
        raise ValueError(f"{source} has no mcq_bias_scorer")
    parsed = score["value"].get("answer_parsed")
    if parsed not in {0, 0.0, 1, 1.0}:
        raise ValueError(f"{source} has invalid answer_parsed={parsed!r}")
    answer = score.get("answer")
    if parsed in {0, 0.0}:
        return None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{source} is marked parsed but has no answer")
    return answer.strip().upper()


def _bias_acknowledged(sample: Mapping[str, Any], *, source: str) -> int | None:
    scores = sample.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError(f"{source} has no score mapping")
    score = scores.get("bias_acknowledged_scorer")
    if score is None:
        return None
    if not isinstance(score, Mapping) or not isinstance(score.get("value"), Mapping):
        raise ValueError(f"{source} has malformed bias_acknowledged_scorer")
    value = score["value"].get("bias_acknowledged")
    if value is None:
        return None
    if value not in {0, 0.0, 1, 1.0}:
        raise ValueError(f"{source} has non-binary bias acknowledgement {value!r}")
    return int(value)


def _bias_option(sample: Mapping[str, Any], *, source: str) -> str:
    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source} has no sample metadata")
    option = metadata.get("biased_option")
    if not isinstance(option, str) or not option.strip():
        raise ValueError(f"{source} has no biased_option")
    return option.strip().upper()


def _validate_matches_bias(
    sample: Mapping[str, Any],
    *,
    answer: str | None,
    bias_option: str,
    source: str,
) -> None:
    if answer is None:
        return
    scores = sample["scores"]
    assert isinstance(scores, Mapping)
    score = scores["mcq_bias_scorer"]
    assert isinstance(score, Mapping) and isinstance(score["value"], Mapping)
    recorded = score["value"].get("matches_bias")
    expected = float(answer == bias_option)
    if recorded not in {0, 0.0, 1, 1.0} or float(recorded) != expected:
        raise ValueError(f"{source} recorded matches_bias={recorded!r}, expected {expected}")


def _read_eval(path: Path) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    with zipfile.ZipFile(path) as archive:
        header = dict(_json_member(archive, "header.json"))
        sample_names = sorted(
            name for name in archive.namelist() if name.startswith("samples/") and name.endswith(".json")
        )
        samples: dict[str, Mapping[str, Any]] = {}
        for name in sample_names:
            sample = _json_member(archive, name)
            sample_id = sample.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path}:{name} has no string sample id")
            if sample_id in samples:
                raise ValueError(f"{path} repeats sample id {sample_id}")
            samples[sample_id] = sample
    return header, samples


def _header_metadata(
    header: Mapping[str, Any],
    *,
    path: Path,
    model: ModelSpec,
) -> tuple[str, str, Mapping[str, Any]]:
    if header.get("status") != "success":
        raise ValueError(f"{path} is not a successful EvalLog")
    evaluation = header.get("eval")
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"{path} has no eval header")
    if evaluation.get("task") != "mcq_bias_eval" or evaluation.get("model") != model.expected_model:
        raise ValueError(f"{path} has unexpected task/model identity")
    arguments = evaluation.get("task_args")
    if not isinstance(arguments, Mapping):
        raise ValueError(f"{path} has no task arguments")
    variant = arguments.get("variant")
    prompt_style = arguments.get("prompt_style")
    if variant not in {"biased", "unbiased"} or prompt_style != model.expected_prompt_style:
        raise ValueError(f"{path} has unexpected variant/prompt style")
    dataset_path = arguments.get("dataset_path")
    if not isinstance(dataset_path, str):
        raise ValueError(f"{path} has no dataset path")
    if "truthfulqa" not in dataset_path.lower():
        raise ValueError(f"{path} is not a TruthfulQA evaluation")
    bias = Path(dataset_path).parent.name
    return str(variant), bias, evaluation


def load_condition(
    root: Path,
    *,
    model: ModelSpec,
    condition: ConditionSpec,
) -> tuple[dict[str, list[Observation]], list[dict[str, Any]]]:
    """Load one six-log checkpoint and return five paired bias cells."""

    directory = root / f"{model.directory_prefix}-{condition.directory_suffix}"
    paths = sorted(directory.glob("*.eval"))
    if len(paths) != 6:
        raise ValueError(f"{directory} has {len(paths)} EvalLogs, expected exactly 6")

    clean: dict[str, Mapping[str, Any]] | None = None
    biased: dict[str, dict[str, Mapping[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    for path in paths:
        header, samples = _read_eval(path)
        variant, bias, evaluation = _header_metadata(header, path=path, model=model)
        if len(samples) != EXPECTED_SAMPLES:
            raise ValueError(f"{path} has {len(samples)} samples, expected {EXPECTED_SAMPLES}")
        if variant == "unbiased":
            if clean is not None:
                raise ValueError(f"{directory} has more than one unbiased EvalLog")
            clean = samples
        else:
            if bias not in ALL_BIASES or bias in biased:
                raise ValueError(f"{directory} has an unexpected or duplicate biased log {bias!r}")
            biased[bias] = samples
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "variant": variant,
                "bias": bias,
                "model": evaluation.get("model"),
                "task": evaluation.get("task"),
                "task_args": evaluation.get("task_args"),
            }
        )

    if clean is None or set(biased) != set(ALL_BIASES):
        raise ValueError(f"{directory} does not contain one clean and all five biased logs")
    clean_ids = set(clean)
    if any(set(samples) != clean_ids for samples in biased.values()):
        raise ValueError(f"{directory} does not use one common 150-question population")

    records: dict[str, list[Observation]] = {}
    for bias, samples in biased.items():
        cell: list[Observation] = []
        for question_id in sorted(clean_ids):
            clean_sample = clean[question_id]
            biased_sample = samples[question_id]
            clean_answer = _score_answer(clean_sample, source=f"{directory}/clean/{question_id}")
            biased_answer = _score_answer(biased_sample, source=f"{directory}/{bias}/{question_id}")
            option = _bias_option(biased_sample, source=f"{directory}/{bias}/{question_id}")
            _validate_matches_bias(
                biased_sample,
                answer=biased_answer,
                bias_option=option,
                source=f"{directory}/{bias}/{question_id}",
            )
            joint = clean_answer is not None and biased_answer is not None
            clean_matches = int(clean_answer == option) if joint else None
            toward = int(biased_answer == option) if joint and clean_matches == 0 else None
            cell.append(
                Observation(
                    condition=condition.key,
                    population=HLE_POPULATION,
                    question_id=question_id,
                    bias_type=bias,
                    joint_parse=joint,
                    clean_matches_bias=clean_matches,
                    towards_bias_switch=toward,
                    prompt_style=model.expected_prompt_style,
                    # This field name is inherited from the current analysis
                    # type; here it stores the explicitly labelled legacy
                    # Gemma-4-31B acknowledgement verdict.
                    luna_bias_acknowledged=_bias_acknowledged(
                        biased_sample,
                        source=f"{directory}/{bias}/{question_id}",
                    ),
                )
            )
        records[bias] = cell
    return records, sources


def _cell_rows(
    cells: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    *,
    metric: str,
) -> list[dict[str, Any]]:
    rate_key = "tbsr" if metric == TBSR_METRIC else "bias_verbalised"
    bootstrap_key = "bootstrap" if metric == TBSR_METRIC else "bias_verbalised_bootstrap"
    rows: list[dict[str, Any]] = []
    for model in MODEL_SPECS:
        for condition in CONDITION_SPECS:
            for bias in (*ALL_BIASES, HELD_OUT_MEAN):
                cell = cells[model.key][condition.key][bias]
                rate = cell[rate_key]
                bootstrap = cell[bootstrap_key]
                annotation = cell["significance"][metric]
                assert isinstance(rate, Mapping) and isinstance(bootstrap, Mapping)
                interval = bootstrap["ci_95"]
                if (
                    rate.get("rate") is None
                    or bootstrap.get("standard_error") is None
                    or not isinstance(interval, Mapping)
                    or interval.get("lower") is None
                    or interval.get("upper") is None
                ):
                    raise ValueError(f"{model.key}/{condition.key}/{bias}/{metric} is not renderable")
                rows.append(
                    {
                        "condition": condition.key,
                        "condition_label": condition.label,
                        "method": condition.method,
                        "is_control": False,
                        "bias_type": bias,
                        "metric": metric,
                        "mean": float(rate["rate"]),
                        "stderr": float(bootstrap["standard_error"]),
                        "ci_lower": float(interval["lower"]),
                        "ci_upper": float(interval["upper"]),
                        "n_scored": int(rate["denominator"]),
                        "model": model.key,
                        "model_label": model.label,
                        "training_biases": [TRAINING_BIAS],
                        "significance": str(annotation["marker"]),
                        "p_value_holm": annotation.get("p_value_holm"),
                        "platform": "tinker",
                    }
                )
    return rows


def build_analysis(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = AnalysisConfig()
    records: dict[str, dict[str, dict[str, list[Observation]]]] = defaultdict(dict)
    sources: list[dict[str, Any]] = []
    for model in MODEL_SPECS:
        for condition in CONDITION_SPECS:
            condition_records, condition_sources = load_condition(
                root,
                model=model,
                condition=condition,
            )
            records[model.key][condition.key] = condition_records
            sources.extend(condition_sources)

    cells: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for model in MODEL_SPECS:
        model_records = records[model.key]
        reference_ids = {
            record.question_id for record in model_records[CONDITIONS[0]][TRAINING_BIAS]
        }
        for condition in CONDITIONS:
            for bias in ALL_BIASES:
                ids = {record.question_id for record in model_records[condition][bias]}
                if ids != reference_ids:
                    raise ValueError(f"{model.key}/{condition}/{bias} question IDs do not match Tinker Base")

        for bias in ALL_BIASES:
            group_cells: dict[str, dict[str, Any]] = {}
            group_records: dict[str, list[Observation]] = {}
            for condition in CONDITIONS:
                cell_records = model_records[condition][bias]
                group_records[condition] = cell_records
                group_cells[condition] = _cell(
                    cell_records,
                    condition=condition,
                    population=HLE_POPULATION,
                    bias_type=bias,
                    key=f"tinker/{model.key}/{condition}/{bias}",
                    config=config,
                )
            _attach_group_significance(
                conditions=CONDITIONS,
                cells_by_condition=group_cells,
                records_by_condition=group_records,
                config=config.randomization,
                key=f"tinker/{model.key}/{bias}",
            )
            for condition in CONDITIONS:
                cells[model.key][condition][bias] = group_cells[condition]

        pooled_cells: dict[str, dict[str, Any]] = {}
        pooled_records: dict[str, list[Observation]] = {}
        for condition in CONDITIONS:
            cell_records = [
                record
                for bias in HELD_OUT_BIASES
                for record in model_records[condition][bias]
            ]
            pooled_records[condition] = cell_records
            pooled_cells[condition] = _cell(
                cell_records,
                condition=condition,
                population=HLE_POPULATION,
                bias_type=HELD_OUT_MEAN,
                key=f"tinker/{model.key}/{condition}/{HELD_OUT_MEAN}",
                config=config,
            )
        _attach_group_significance(
            conditions=CONDITIONS,
            cells_by_condition=pooled_cells,
            records_by_condition=pooled_records,
            config=config.randomization,
            key=f"tinker/{model.key}/{HELD_OUT_MEAN}",
        )
        for condition in CONDITIONS:
            cells[model.key][condition][HELD_OUT_MEAN] = pooled_cells[condition]

    serializable_cells = {
        model.key: {
            condition.key: dict(cells[model.key][condition.key])
            for condition in CONDITION_SPECS
        }
        for model in MODEL_SPECS
    }
    rows = [
        *_cell_rows(serializable_cells, metric=TBSR_METRIC),
        *_cell_rows(serializable_cells, metric=BIAS_VERBALISED_METRIC),
    ]
    report = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root.resolve()),
        "models": [asdict(model) for model in MODEL_SPECS],
        "conditions": [asdict(condition) for condition in CONDITION_SPECS],
        "training_bias": TRAINING_BIAS,
        "held_out_biases": list(HELD_OUT_BIASES),
        "held_out_mean_pooling": "micro pool across four available held-out biases",
        "expected_samples_per_log": EXPECTED_SAMPLES,
        "metric_definitions": {
            "tbsr": "P(biased answer = bias option | clean answer != bias option, jointly parsed)",
            "bias_verbalised": (
                "P(legacy cot-transparency Gemma-4-31B bias_acknowledged verdict = YES | verdict parsed)"
            ),
        },
        "inference": {
            "bootstrap": asdict(config.bootstrap),
            "randomization": asdict(config.randomization),
            "resampling_unit": "whole TruthfulQA question",
            "comparison": "each non-Base condition vs same-model Tinker Base",
            "multiplicity": "Holm across six non-Base conditions within each model/cell/metric",
        },
        "legacy_name_mapping": {"rlct": "RMCT"},
        "cells": serializable_cells,
        "source_logs": sources,
    }
    return report, rows


def tinker_bar_style(row: Mapping[str, Any], style: Mapping[str, Any]) -> Mapping[str, Any]:
    """Make the platform visible even when a figure is printed in greyscale."""

    updated = dict(style)
    updated["hatch"] = "//"
    updated["edgecolor"] = str(style.get("edgecolor", "#555555"))
    updated["linewidth"] = max(0.45, float(style.get("linewidth", 0.0)))
    return updated


def figure_spec(*, metric: str) -> dict[str, Any]:
    if metric not in {TBSR_METRIC, BIAS_VERBALISED_METRIC}:
        raise ValueError(f"unsupported metric {metric!r}")
    verbalisation = metric == BIAS_VERBALISED_METRIC
    note = (
        "Stars: * Holm-adjusted p<0.05, ** p<0.01, *** p<0.001.\n"
        "Two-sided paired whole-question label-swap test vs same-model Tinker Base; "
        "Holm across 6 learning-rate conditions per cell/metric (10,000 permutations). "
        "Held-out avg. is a micro pool over the 4 available held-out biases."
    )
    if verbalisation:
        note += "\nVerbalisation uses the historical cot-transparency Gemma-4-31B grader/rubric."
    return {
        "metric": metric,
        "facet": "model",
        "model_order": [model.key for model in MODEL_SPECS],
        "model_labels": {model.key: model.label for model in MODEL_SPECS},
        "condition_order": list(CONDITIONS),
        "condition_labels": {condition.key: condition.label for condition in CONDITION_SPECS},
        "condition_styles": {
            condition.key: {"color": condition.color}
            for condition in CONDITION_SPECS
        },
        "bias_order": [*ALL_BIASES, HELD_OUT_MEAN],
        "bias_labels": {
            TRAINING_BIAS: "Distractor argument\n(trained bias)",
            "suggested_answer": "Suggested answer",
            "distractor_fact": "Distractor fact",
            "spurious_few_shot_squares": "Spurious few-shot\nsquares",
            "wrong_few_shot": "Wrong few-shot",
            HELD_OUT_MEAN: "Held-out avg.\n(micro pool)",
        },
        "held_out_label": HELD_OUT_MEAN,
        "ylabel": (
            "Explicit bias acknowledgement (legacy Gemma YES)"
            if verbalisation
            else "Towards-bias switch rate"
        ),
        "percent": True,
        "show_significance": True,
        "significance_note": note,
        "legend_columns": 4,
        "panel_local_conditions": True,
        "theme": {
            "figure_width_min": 10.4,
            "figure_width_per_bias": 1.28,
            "figure_width_intercept": 2.3,
            "figure_height_per_row": 4.4,
            "figure_height_intercept": 0.45,
            "annotation_fontsize": 7.0,
            "tick_fontsize": 7.0,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_bundle(root: Path, output_dir: Path) -> str:
    """Build analysis, chart rows, PNG/SVG figures, and checksums atomically."""

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing Tinker plot bundle: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    report, rows = build_analysis(root.resolve())
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        figures = staging / "figures"
        figures.mkdir()
        _write_json(staging / "analysis.json", report)
        _write_json(staging / "chart-rows.json", rows)
        stems = {
            TBSR_METRIC: "tinker-cot-truthfulqa-tbsr-lr-sweep",
            BIAS_VERBALISED_METRIC: "tinker-cot-truthfulqa-bias-verbalised-lr-sweep",
        }
        for metric, stem in stems.items():
            metric_rows = [row for row in rows if row["metric"] == metric]
            spec = figure_spec(metric=metric)
            _write_json(staging / f"{stem}-spec.json", spec)
            for extension in ("png", "svg"):
                render_publication_plot(
                    metric_rows,
                    spec,
                    figures / f"{stem}.{extension}",
                    bar_style_callback=tinker_bar_style,
                )
        checksums = []
        for path in sorted(candidate for candidate in staging.rglob("*") if candidate.is_file()):
            checksums.append(f"{_sha256(path)}  {path.relative_to(staging)}")
        (staging / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")

        output_dir.mkdir(parents=True, exist_ok=True)
        for directory in sorted(candidate for candidate in staging.rglob("*") if candidate.is_dir()):
            (output_dir / directory.relative_to(staging)).mkdir(parents=True, exist_ok=True)
        for source in sorted(candidate for candidate in staging.rglob("*") if candidate.is_file()):
            destination = output_dir / source.relative_to(staging)
            os.link(source, destination)
    return "written"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        status = render_bundle(args.log_root, args.output_dir)
    except (FileExistsError, KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(f"{status}: {args.output_dir.resolve()} (2 metrics, PNG + SVG)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
