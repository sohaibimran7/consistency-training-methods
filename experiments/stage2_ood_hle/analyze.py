"""Analyze the four-column Stage 2 OOD diagnostic without model/API calls.

The diagnostic holds the Qwen Stage-1 population fixed where possible and
separates two generalisation axes:

1. IID held-out questions under the training bias (``wrong_argument``);
2. canonical HLE questions under that same bias;
3. IID held-out questions under five biases held out from training; and
4. canonical HLE questions under those five held-out biases.

The latter two headline estimates are *micro* pools: their numerator is the
sum of towards-bias switches and their denominator is the sum of jointly
parsed clean-non-bias pairs.  They are not averages of five bias rates.
Because all five biased prompts share a clean answer for a question, error
bars come from a non-parametric bootstrap of whole question clusters rather
than an independent-binomial approximation.

This module is deliberately independent of ``stage1_iid_diagnostic.plot`` so
the historic two-column figures remain immutable.  It can consume a small
normalised JSONL record file or read completed local Inspect logs; neither
path creates model requests or grader calls.
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

import numpy as np

from experiments.stage2_ood_hle.prepare import HELDOUT_BIASES, TRAINING_BIAS, load_bias_contract


ANALYSIS_SCHEMA = "stage2-ood-hle-analysis-v1"
OBSERVATION_BUNDLE_SCHEMA = "stage2-ood-hle-observations-v1"
IID_POPULATION = "iid"
HLE_POPULATION = "hle"
POPULATIONS = (IID_POPULATION, HLE_POPULATION)
HEADLINE_COLUMNS = (
    "iid",
    "held_out_dataset",
    "held_out_bias",
    "held_out_dataset_and_bias",
)
HEADLINE_COLUMN_METADATA = {
    "iid": {
        "population": IID_POPULATION,
        "kind": "training_bias",
        "label": "IID",
        "subtitle": "heldout_in_domain n=200; wrong_argument",
    },
    "held_out_dataset": {
        "population": HLE_POPULATION,
        "kind": "training_bias",
        "label": "Held-out dataset",
        "subtitle": "canonical HLE n=100; wrong_argument",
    },
    "held_out_bias": {
        "population": IID_POPULATION,
        "kind": "held_out_bias_pool",
        "label": "Held-out bias",
        "subtitle": "heldout_in_domain n=200; five biases pooled",
    },
    "held_out_dataset_and_bias": {
        "population": HLE_POPULATION,
        "kind": "held_out_bias_pool",
        "label": "Held-out dataset + bias",
        "subtitle": "canonical HLE n=100; five biases pooled",
    },
}
DEFAULT_CONDITION_ORDER = (
    "untrained",
    "bct",
    "bct-control",
    "opct",
    "rmct",
    "rmct-control",
    "act",
    "attct",
    "mlpct",
)
SWITCH_METRICS = (
    "unbiased_matches_bias",
    "towards_bias_switch",
    "away_from_bias_switch",
    "net_switch",
    "abs_switch",
)
TBSR_METRIC = "tbsr"
BIAS_VERBALISED_METRIC = "bias_verbalised"
SIGNIFICANCE_BASELINE_CONDITION = "untrained"
SIGNIFICANCE_METHOD = "paired_question_cluster_label_swap_randomization"
SIGNIFICANCE_STATISTIC = "treatment_minus_base_micro_pooled_rate"
SIGNIFICANCE_MULTIPLICITY = "holm"
SIGNIFICANCE_SIDEDNESS = "two-sided"
SIGNIFICANCE_NOTE = (
    "Each non-Base condition is compared with Base using a two-sided paired whole-question "
    "label-swap randomization test. Stars use Holm-adjusted p-values across the non-Base "
    "conditions within each displayed cell and metric."
)

# Stage 2 analysis only consumes sample identity, task metadata, and scores.
# Reasoning-model generations can be tens of thousands of tokens per sample,
# so loading messages/output/events for all 162 logs needlessly retains many
# gigabytes. ``epoch``, ``input``, and ``target`` are required by Inspect's
# EvalSample schema and therefore cannot be excluded even though we do not use
# them after validation.
COMPACT_EVAL_EXCLUDE_FIELDS = {
    "choices",
    "sandbox",
    "files",
    "setup",
    "messages",
    "output",
    "store",
    "events",
    "timelines",
    "model_usage",
    "role_usage",
    "model_fallbacks",
    "started_at",
    "completed_at",
    "total_time",
    "working_time",
    "uuid",
    "invalidation",
    "error",
    "error_retries",
    "attachments",
    "events_data",
    "limit",
}


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Deterministic question-cluster bootstrap configuration."""

    replicates: int = 10_000
    seed: int = 20260802
    chunk_size: int = 512

    def __post_init__(self) -> None:
        for value, name, minimum in (
            (self.replicates, "replicates", 2),
            (self.seed, "seed", 0),
            (self.chunk_size, "chunk_size", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                comparator = ">= 2" if name == "replicates" else ">= 0" if name == "seed" else ">= 1"
                raise ValueError(f"bootstrap {name} must be an integer {comparator}")


@dataclass(frozen=True, slots=True)
class RandomizationConfig:
    """Deterministic paired whole-question label-swap configuration."""

    permutations: int = 10_000
    seed: int = 20260805
    chunk_size: int = 512

    def __post_init__(self) -> None:
        for value, name, minimum in (
            (self.permutations, "permutations", 10_000),
            (self.seed, "seed", 0),
            (self.chunk_size, "chunk_size", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                comparator = ">= 10,000" if name == "permutations" else ">= 0" if name == "seed" else ">= 1"
                raise ValueError(f"randomization {name} must be an integer {comparator}")


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Population and inference contract for one complete OOD matrix."""

    iid_questions: int = 200
    hle_questions: int = 100
    training_bias: str = TRAINING_BIAS
    held_out_biases: tuple[str, ...] = HELDOUT_BIASES
    expected_prompt_style: str = "none"
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)

    def __post_init__(self) -> None:
        for value, name in ((self.iid_questions, "iid_questions"), (self.hle_questions, "hle_questions")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.training_bias, str) or not self.training_bias:
            raise ValueError("training_bias must be a non-empty string")
        if (
            not self.held_out_biases
            or any(not isinstance(bias, str) or not bias for bias in self.held_out_biases)
            or len(self.held_out_biases) != len(set(self.held_out_biases))
            or self.training_bias in self.held_out_biases
        ):
            raise ValueError("held_out_biases must be unique, non-empty, and exclude training_bias")
        if not isinstance(self.expected_prompt_style, str) or not self.expected_prompt_style:
            raise ValueError("expected_prompt_style must be a non-empty string")

    @property
    def expected_questions(self) -> dict[str, int]:
        return {IID_POPULATION: self.iid_questions, HLE_POPULATION: self.hle_questions}


@dataclass(frozen=True, slots=True)
class Observation:
    """One paired clean/biased result retained for the OOD estimand."""

    condition: str
    population: Literal["iid", "hle"]
    question_id: str
    bias_type: str
    joint_parse: bool
    clean_matches_bias: int | None
    towards_bias_switch: int | None
    prompt_style: str = "none"
    luna_bias_acknowledged: int | None = None
    grader_max_tokens_cap_hit: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.condition, str) or not self.condition:
            raise ValueError("observation condition must be non-empty")
        if self.population not in POPULATIONS:
            raise ValueError(f"observation population must be one of {POPULATIONS}")
        if not isinstance(self.question_id, str) or not self.question_id:
            raise ValueError("observation question_id must be non-empty")
        if not isinstance(self.bias_type, str) or not self.bias_type:
            raise ValueError("observation bias_type must be non-empty")
        if not isinstance(self.prompt_style, str) or not self.prompt_style:
            raise ValueError("observation prompt_style must be non-empty")
        if self.luna_bias_acknowledged not in {None, 0, 1}:
            raise ValueError("observation luna_bias_acknowledged must be binary or null")
        if not isinstance(self.grader_max_tokens_cap_hit, bool):
            raise ValueError("observation grader_max_tokens_cap_hit must be bool")
        if not isinstance(self.joint_parse, bool):
            raise ValueError("observation joint_parse must be bool")
        if not self.joint_parse:
            if self.clean_matches_bias is not None or self.towards_bias_switch is not None:
                raise ValueError("joint-parse failures must have null paired outcomes")
            return
        if self.clean_matches_bias not in {0, 1}:
            raise ValueError("jointly parsed clean_matches_bias must be binary")
        if self.clean_matches_bias == 0 and self.towards_bias_switch not in {0, 1}:
            raise ValueError("eligible jointly parsed observations need a binary towards_bias_switch")
        if self.clean_matches_bias == 1 and self.towards_bias_switch is not None:
            raise ValueError("clean-bias matches must have null towards_bias_switch")

    @property
    def eligible(self) -> bool:
        return self.joint_parse and self.clean_matches_bias == 0


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, (float, np.floating)) and not isinstance(value, bool) and not math.isfinite(float(value))
    )


def _binary(value: Any, *, field: str, allow_missing: bool = False) -> int | None:
    if _missing(value):
        if allow_missing:
            return None
        raise ValueError(f"{field} must be binary")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return int(numeric)
    raise ValueError(f"{field} must be binary")


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be bool")
    return value


def _integer_switch(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field} must be an integer switch value")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric not in {-1.0, 0.0, 1.0}:
        raise ValueError(f"{field} must be -1, 0, or 1")
    return int(numeric)


def _paired_outcome(values: Mapping[str, Any], *, sample_id: str) -> tuple[bool, int | None, int | None]:
    """Validate the shared switch-scorer contract and retain TBSR fields."""

    missing_metrics = [metric for metric in SWITCH_METRICS if metric not in values]
    if missing_metrics:
        raise ValueError(f"sample {sample_id!r} has no complete switch score mapping: {missing_metrics}")
    clean_raw = values["unbiased_matches_bias"]
    toward_raw = values["towards_bias_switch"]
    away_raw = values["away_from_bias_switch"]
    net_raw = values["net_switch"]
    absolute_raw = values["abs_switch"]
    if _missing(net_raw) or _missing(absolute_raw):
        if not all(_missing(value) for value in (clean_raw, toward_raw, away_raw, net_raw, absolute_raw)):
            raise ValueError(f"sample {sample_id!r} has a partial paired parse failure")
        return False, None, None

    clean = _binary(clean_raw, field="unbiased_matches_bias")
    assert clean is not None
    if clean == 0:
        toward = _binary(toward_raw, field="towards_bias_switch")
        assert toward is not None
        away = _binary(away_raw, field="away_from_bias_switch", allow_missing=True)
        away = 0 if away is None else away
    else:
        toward_value = _binary(toward_raw, field="towards_bias_switch", allow_missing=True)
        toward = 0 if toward_value is None else toward_value
        away = _binary(away_raw, field="away_from_bias_switch")
        assert away is not None
    net = _integer_switch(net_raw, field="net_switch")
    absolute = _binary(absolute_raw, field="abs_switch")
    assert absolute is not None
    if net != toward - away or absolute != toward + away:
        raise ValueError(f"sample {sample_id!r} has an incoherent paired switch score")
    if (clean == 0 and away) or (clean == 1 and toward):
        raise ValueError(f"sample {sample_id!r} violates conditional switch eligibility")
    return True, clean, toward if clean == 0 else None


def _switch_score(sample: Any) -> Mapping[str, Any]:
    matches: list[Mapping[str, Any]] = []
    for score in _mapping(_attribute(sample, "scores", {})).values():
        value = _attribute(score, "value")
        if isinstance(value, Mapping) and all(metric in value for metric in SWITCH_METRICS):
            matches.append(value)
    if len(matches) != 1:
        raise ValueError(f"sample {_attribute(sample, 'id', '<unknown>')!r} must contain exactly one switch score")
    return matches[0]


def _luna_matches(sample: Any) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Return acknowledgement-score mappings without treating absence as NO."""

    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for score in _mapping(_attribute(sample, "scores", {})).values():
        value = _attribute(score, "value")
        if isinstance(value, Mapping) and "bias_acknowledged" in value:
            matches.append((value, _mapping(_attribute(score, "metadata", {}))))
    return matches


def _luna_outcome(sample: Any, *, required: bool = True) -> tuple[int | None, bool]:
    """Read the posthoc Luna acknowledgement verdict from one graded sample.

    Stage 2 generation deliberately omits the paid scorer.  This analysis
    therefore accepts only its later appended score and never silently turns
    an ungraded raw response into a "no" verbalisation verdict.
    """

    matches = _luna_matches(sample)
    if not matches and not required:
        return None, False
    if len(matches) != 1:
        raise ValueError(
            f"sample {_attribute(sample, 'id', '<unknown>')!r} must contain exactly one Luna acknowledgement score"
        )
    values, metadata = matches[0]
    return (
        _binary(values["bias_acknowledged"], field="bias_acknowledged", allow_missing=True),
        _boolean(metadata.get("grader_max_tokens_cap_hit", False), field="grader_max_tokens_cap_hit"),
    )


def _log_has_complete_luna_score(log: Any) -> bool:
    """Distinguish a raw retry from a fully posthoc-graded retry locally."""

    # Avoid duplicating references to the potentially large EvalLog sample
    # graph. Streaming callers release the graph immediately after this check.
    samples = _attribute(log, "samples", []) or ()
    if not samples:
        raise ValueError("selected Stage 2 OOD EvalLog has no samples")
    counts = [len(_luna_matches(sample)) for sample in samples]
    if all(count == 0 for count in counts):
        return False
    if all(count == 1 for count in counts):
        return True
    raise ValueError("Stage 2 OOD EvalLog has a partial or ambiguous Luna acknowledgement score")


def _task_basename(value: Any) -> str:
    return str(value or "").rsplit("@", 1)[-1].rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _population_from_task_args(args: Mapping[str, Any]) -> str | None:
    population_aliases = {
        IID_POPULATION: IID_POPULATION,
        "in_domain": IID_POPULATION,
        "heldout_in_domain": IID_POPULATION,
        HLE_POPULATION: HLE_POPULATION,
        "canonical_hle": HLE_POPULATION,
        "stage2_ood_hle": HLE_POPULATION,
    }
    for explicit in (args.get("stage2_population"), args.get("population")):
        if explicit in population_aliases:
            return population_aliases[str(explicit)]
    split = str(args.get("split", ""))
    dataset = str(args.get("source_dataset", args.get("dataset", ""))).lower()
    if split == "heldout_in_domain":
        return IID_POPULATION
    if split in {"hle", "canonical_hle", "stage2_ood_hle"} or dataset.startswith("hle"):
        return HLE_POPULATION
    return None


def _local_log_path(value: Any) -> Path:
    raw = str(value if isinstance(value, (str, os.PathLike)) else _attribute(value, "name", value))
    parsed = urlsplit(raw)
    if parsed.scheme:
        if parsed.scheme.lower() != "file" or parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise ValueError(f"Stage 2 OOD analysis requires a local EvalLog path, got {raw!r}")
        path = Path(unquote(parsed.path))
    else:
        path = Path(raw)
    if not path.is_absolute():
        path = path.resolve()
    return path.resolve()


def _discover_eval_logs(location: str | Path) -> list[Path]:
    path = Path(location)
    if path.is_file():
        return [path.resolve()]
    if not path.exists() and glob.has_magic(str(location)):
        return sorted({Path(item).resolve() for item in glob.glob(str(location), recursive=True) if Path(item).is_file()})
    try:
        from inspect_ai.log import list_eval_logs
    except ImportError as exc:  # pragma: no cover - requires configured Inspect environment
        raise RuntimeError("Inspect AI is required to read Stage 2 OOD EvalLogs") from exc
    infos = list_eval_logs(str(location), formats=["eval"], recursive=True)
    paths = sorted({_local_log_path(info) for info in infos})
    if not paths:
        raise FileNotFoundError(f"no Inspect EvalLogs found at {location!r}")
    if any(not candidate.is_file() for candidate in paths):
        raise ValueError("Stage 2 OOD analysis requires local EvalLog files")
    return paths


def _discard_eval_log(log: Any) -> None:
    """Drop a consumed Inspect EvalLog's bulky sample graph promptly."""

    samples = _attribute(log, "samples", None)
    if isinstance(samples, list):
        samples.clear()
    del samples
    # Inspect/Pydantic objects can contain reference cycles.  This call is
    # intentionally on the one-log-at-a-time extraction boundary, not inside
    # the statistical loops.
    gc.collect()


def _observations_from_eval_log(
    log: Any,
    *,
    condition: str,
    population: str,
    bias_type: str,
    expected_prompt_style: str,
    require_luna: bool = True,
) -> list[Observation]:
    if _attribute(log, "status") != "success":
        raise ValueError("selected Stage 2 OOD EvalLog is not successful")
    evaluation = _attribute(log, "eval")
    task_args = _mapping(_attribute(evaluation, "task_args", {}))
    header_bias = task_args.get("bias_type")
    if header_bias != bias_type:
        raise ValueError("selected EvalLog bias_type conflicts with its header")
    header_style = str(task_args.get("prompt_style", expected_prompt_style))
    if header_style != expected_prompt_style:
        raise ValueError(f"selected EvalLog has unexpected prompt_style {header_style!r}")
    header_dataset = task_args.get("source_dataset", task_args.get("dataset"))
    samples = _attribute(log, "samples", []) or ()
    if not samples:
        raise ValueError("selected Stage 2 OOD EvalLog has no samples")
    seen: set[str] = set()
    observations: list[Observation] = []
    for sample in samples:
        question_id = str(_attribute(sample, "id", ""))
        if not question_id or question_id in seen:
            raise ValueError(f"missing or duplicate question_id {question_id!r} in a Stage 2 EvalLog")
        seen.add(question_id)
        metadata = _mapping(_attribute(sample, "metadata", {}))
        if metadata.get("variant") != "biased":
            raise ValueError(f"sample {question_id!r} is not a biased prompt")
        if metadata.get("bias_type") != bias_type:
            raise ValueError(f"sample {question_id!r} bias_type conflicts with its log header")
        if metadata.get("prompt_style", expected_prompt_style) != expected_prompt_style:
            raise ValueError(f"sample {question_id!r} has unexpected prompt_style")
        if header_dataset is not None and metadata.get("source_dataset") != header_dataset:
            raise ValueError(f"sample {question_id!r} source dataset conflicts with its log header")
        joint_parse, clean, toward = _paired_outcome(_switch_score(sample), sample_id=question_id)
        luna, grader_cap_hit = _luna_outcome(sample, required=require_luna)
        observations.append(
            Observation(
                condition=condition,
                population=population,  # type: ignore[arg-type] -- checked at discovery boundary
                question_id=question_id,
                bias_type=bias_type,
                joint_parse=joint_parse,
                clean_matches_bias=clean,
                towards_bias_switch=toward,
                prompt_style=expected_prompt_style,
                luna_bias_acknowledged=luna,
                grader_max_tokens_cap_hit=grader_cap_hit,
            )
        )
    return observations


def parse_runs(values: Sequence[str]) -> dict[str, str]:
    runs: dict[str, str] = {}
    for value in values:
        condition, separator, location = value.partition("=")
        if not separator or not condition or not location:
            raise ValueError(f"--run must use CONDITION=LOCAL_LOG_LOCATION, got {value!r}")
        if condition in runs:
            raise ValueError(f"duplicate --run condition {condition!r}")
        runs[condition] = location
    return runs


def load_inspect_runs(
    runs: Mapping[str, str],
    *,
    expected_prompt_style: str = "none",
) -> tuple[list[Observation], list[dict[str, Any]]]:
    """Read the latest complete biased log for every condition/population/bias/dataset.

    Retry selection happens at the log level, while population validation is
    done later over sample IDs.  This permits the IID population to arrive as
    two 100-question dataset logs and HLE as one 100-question log.
    """

    if not runs:
        raise ValueError("at least one Stage 2 OOD run is required")
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - requires configured Inspect environment
        raise RuntimeError("Inspect AI is required to read Stage 2 OOD EvalLogs") from exc

    # Prefer a complete posthoc-Luna retry over its raw generation source,
    # even if Inspect preserved the same creation timestamp while rescoring.
    selected: dict[
        tuple[str, str, str, str],
        tuple[bool, str, Path, list[Observation]],
    ] = {}
    for condition, location in runs.items():
        if not condition:
            raise ValueError("Stage 2 OOD condition names must be non-empty")
        for path in _discover_eval_logs(location):
            log = read_eval_log(str(path), header_only=True)
            complete_log: Any | None = None
            try:
                if _attribute(log, "status") != "success":
                    continue
                evaluation = _attribute(log, "eval")
                args = _mapping(_attribute(evaluation, "task_args", {}))
                bias = args.get("bias_type")
                population = _population_from_task_args(args)
                dataset = args.get("source_dataset", args.get("dataset"))
                if (
                    not isinstance(bias, str)
                    or not bias
                    or population not in POPULATIONS
                    or not isinstance(dataset, str)
                    or not dataset
                    or "biased" not in _task_basename(_attribute(evaluation, "task"))
                ):
                    continue
                if str(args.get("prompt_style", expected_prompt_style)) != expected_prompt_style:
                    continue
                created = str(_attribute(evaluation, "created", ""))
                if not created:
                    raise ValueError(f"successful Stage 2 OOD candidate has no creation timestamp: {path}")
                complete_log = read_eval_log(str(path), exclude_fields=COMPACT_EVAL_EXCLUDE_FIELDS)
                has_luna = _log_has_complete_luna_score(complete_log)
                rows = _observations_from_eval_log(
                    complete_log,
                    condition=condition,
                    population=population,
                    bias_type=bias,
                    expected_prompt_style=expected_prompt_style,
                    require_luna=has_luna,
                )
                key = (condition, population, bias, dataset)
                prior = selected.get(key)
                candidate_rank = (int(has_luna), created)
                if prior is not None and candidate_rank == (int(prior[0]), prior[1]):
                    raise ValueError(f"ambiguous Stage 2 OOD retries for {key}: {prior[2]} and {path}")
                if prior is None or candidate_rank > (int(prior[0]), prior[1]):
                    if prior is not None:
                        prior[3].clear()
                    selected[key] = (has_luna, created, path, rows)
                else:
                    rows.clear()
            finally:
                if complete_log is not None:
                    _discard_eval_log(complete_log)
                    del complete_log
                _discard_eval_log(log)
                del log

    observations: list[Observation] = []
    sources: list[dict[str, Any]] = []
    for (condition, population, bias, dataset), (has_luna, created, path, rows) in sorted(selected.items()):
        observations.extend(rows)
        sources.append(
            {
                "condition": condition,
                "population": population,
                "bias_type": bias,
                "source_dataset": dataset,
                "created": created,
                "path": str(path),
                "sha256": _sha256(path),
                "samples": len(rows),
                "luna_grading_present": has_luna,
            }
        )
    if not observations:
        raise FileNotFoundError("no successful biased Stage 2 OOD EvalLogs were selected")
    return observations, sources


def _observation_sources_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.sources.json")


def extract_inspect_runs_to_jsonl(
    runs: Mapping[str, str],
    output: str | Path,
    *,
    expected_prompt_style: str = "none",
) -> dict[str, Any]:
    """Extract one EvalLog at a time into compact observations plus provenance.

    ``load_inspect_runs`` deliberately keeps only normalized :class:`Observation`
    rows while each loaded EvalLog (including its samples) is discarded and
    collected before the next path.  The companion sidecar binds the compact
    JSONL to every immutable selected EvalLog so analysis can proceed without
    reloading Inspect objects.
    """

    destination = Path(output).resolve()
    sources_path = _observation_sources_path(destination)
    if destination.exists() or sources_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing Stage 2 OOD observation bundle: {destination} or {sources_path}"
        )
    observations, sources = load_inspect_runs(runs, expected_prompt_style=expected_prompt_style)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for observation in observations:
                handle.write(json.dumps(asdict(observation), sort_keys=True, allow_nan=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        observations_sha256 = _sha256(temporary)
        sidecar: dict[str, Any] = {
            "schema": OBSERVATION_BUNDLE_SCHEMA,
            "observations": {
                "path": str(destination),
                "sha256": observations_sha256,
                "rows": len(observations),
            },
            "input_sources": sources,
        }
        sidecar_payload = (json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        sidecar_descriptor, temporary_sources_name = tempfile.mkstemp(
            prefix=f".{sources_path.name}.", suffix=".tmp", dir=sources_path.parent
        )
        temporary_sources = Path(temporary_sources_name)
        try:
            with os.fdopen(sidecar_descriptor, "wb") as handle:
                handle.write(sidecar_payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, destination)
            os.link(temporary_sources, sources_path)
        finally:
            temporary_sources.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return sidecar


def load_observation_sources(
    path: str | Path,
    *,
    observations_path: str | Path,
    observations_count: int,
) -> list[dict[str, Any]]:
    """Load raw-log provenance only when it cryptographically binds the JSONL."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Stage 2 OOD observation source sidecar does not exist: {source}")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Stage 2 OOD observation source sidecar is invalid JSON: {source}") from exc
    if not isinstance(document, Mapping) or document.get("schema") != OBSERVATION_BUNDLE_SCHEMA:
        raise ValueError("Stage 2 OOD observation source sidecar has an unexpected schema")
    observation_binding = document.get("observations")
    if not isinstance(observation_binding, Mapping):
        raise ValueError("Stage 2 OOD observation source sidecar lacks an observation binding")
    observations = Path(observations_path).resolve()
    if observation_binding.get("sha256") != _sha256(observations):
        raise ValueError("Stage 2 OOD observation source sidecar does not match the observation JSONL")
    if observation_binding.get("rows") != observations_count:
        raise ValueError("Stage 2 OOD observation source sidecar has the wrong observation count")
    sources = document.get("input_sources")
    if not isinstance(sources, list) or not sources or any(not isinstance(item, Mapping) for item in sources):
        raise ValueError("Stage 2 OOD observation source sidecar has invalid input sources")
    return [dict(item) for item in sources]


def observation_from_mapping(value: Mapping[str, Any]) -> Observation:
    """Parse one normalised JSONL row, accepting scorer-name aliases."""

    if not isinstance(value, Mapping):
        raise ValueError("Stage 2 OOD observation row must be an object")
    population = value.get("population", value.get("domain"))
    clean = value.get("clean_matches_bias", value.get("unbiased_matches_bias"))
    toward = value.get("towards_bias_switch")
    luna = value.get(
        "luna_bias_acknowledged",
        value.get("bias_acknowledged", value.get("bias_verbalised", value.get("luna"))),
    )
    joint = value.get("joint_parse")
    if not isinstance(joint, bool):
        raise ValueError("Stage 2 OOD JSONL observation has no boolean joint_parse")
    clean_value = _binary(clean, field="clean_matches_bias", allow_missing=True)
    toward_value = _binary(toward, field="towards_bias_switch", allow_missing=True)
    luna_value = _binary(luna, field="bias_acknowledged", allow_missing=True)
    if joint and clean_value == 1:
        # Normalised exports may retain the scorer's structural zero here;
        # the OOD observation contract represents it as inapplicable.
        if toward_value not in {None, 0}:
            raise ValueError("clean-bias matches cannot be towards-bias switches")
        toward_value = None
    return Observation(
        condition=str(value.get("condition", "")),
        population=str(population),  # type: ignore[arg-type] -- dataclass validates runtime values
        question_id=str(value.get("question_id", "")),
        bias_type=str(value.get("bias_type", "")),
        joint_parse=joint,
        clean_matches_bias=clean_value,
        towards_bias_switch=toward_value,
        prompt_style=str(value.get("prompt_style", "none")),
        luna_bias_acknowledged=luna_value,
        grader_max_tokens_cap_hit=_boolean(
            value.get("grader_max_tokens_cap_hit", value.get("grader_cap_hit", False)),
            field="grader_max_tokens_cap_hit",
        ),
    )


def load_observations_jsonl(path: str | Path) -> list[Observation]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Stage 2 OOD observation JSONL does not exist: {source}")
    observations: list[Observation] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"{source}:{line_number}: blank observation rows are not allowed")
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid observation JSON") from exc
        try:
            observations.append(observation_from_mapping(row))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if not observations:
        raise ValueError(f"{source}: observation JSONL is empty")
    return observations


def _counts(records: Sequence[Observation]) -> dict[str, int]:
    joint = [record for record in records if record.joint_parse]
    eligible = [record for record in joint if record.clean_matches_bias == 0]
    luna = [record for record in records if record.luna_bias_acknowledged is not None]
    return {
        "attempted_pairs": len(records),
        "joint_parsed_pairs": len(joint),
        "joint_parse_failures": len(records) - len(joint),
        "eligible_clean_not_bias_pairs": len(eligible),
        "toward_bias_switches": sum(int(record.towards_bias_switch or 0) for record in eligible),
        "luna_parsed": len(luna),
        "luna_parse_failures": len(records) - len(luna),
        "bias_verbalised_yes": sum(int(record.luna_bias_acknowledged or 0) for record in luna),
        "grader_max_token_cap_hits": sum(record.grader_max_tokens_cap_hit for record in records),
    }


def _rate(counts: Mapping[str, int], *, numerator_key: str, denominator_key: str) -> dict[str, int | float | None]:
    numerator = counts[numerator_key]
    denominator = counts[denominator_key]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _tbsr_rate(counts: Mapping[str, int]) -> dict[str, int | float | None]:
    return _rate(
        counts,
        numerator_key="toward_bias_switches",
        denominator_key="eligible_clean_not_bias_pairs",
    )


def _bias_verbalised_rate(counts: Mapping[str, int]) -> dict[str, int | float | None]:
    return _rate(
        counts,
        numerator_key="bias_verbalised_yes",
        denominator_key="luna_parsed",
    )


def _derived_seed(config: BootstrapConfig | RandomizationConfig, key: str) -> int:
    payload = f"{config.seed}\0{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _metric_counts(record: Observation, *, metric: str) -> tuple[int, int]:
    """Return one observation's numerator/denominator contribution."""

    if metric == TBSR_METRIC:
        return (
            int(record.towards_bias_switch or 0) if record.eligible else 0,
            int(record.eligible),
        )
    if metric == BIAS_VERBALISED_METRIC:
        return int(record.luna_bias_acknowledged or 0), int(record.luna_bias_acknowledged is not None)
    raise ValueError(f"unsupported question-cluster bootstrap metric {metric!r}")


def question_cluster_bootstrap_rates(
    records: Sequence[Observation],
    *,
    config: BootstrapConfig,
    key: str,
    metric: str = TBSR_METRIC,
) -> np.ndarray:
    """Return one metric estimate per whole-question bootstrap resample.

    A cluster contains every bias observation for a question.  In particular,
    a held-out-bias pool never resamples one bias response independently of the
    other four responses sharing the same clean answer.
    """

    if metric not in {TBSR_METRIC, BIAS_VERBALISED_METRIC}:
        raise ValueError(f"unsupported question-cluster bootstrap metric {metric!r}")

    grouped: dict[str, list[Observation]] = defaultdict(list)
    for record in records:
        grouped[record.question_id].append(record)
    if not grouped:
        return np.array([], dtype=np.float64)
    cluster_counts = np.array(
        [
            (
                sum(_metric_counts(record, metric=metric)[0] for record in rows),
                sum(_metric_counts(record, metric=metric)[1] for record in rows),
            )
            for _, rows in sorted(grouped.items())
        ],
        dtype=np.int64,
    )
    n_clusters = len(cluster_counts)
    rng = np.random.default_rng(_derived_seed(config, key))
    rates: list[np.ndarray] = []
    remaining = config.replicates
    while remaining:
        batch = min(remaining, config.chunk_size)
        sample_index = rng.integers(0, n_clusters, size=(batch, n_clusters), endpoint=False)
        sampled = cluster_counts[sample_index].sum(axis=1)
        numerator = sampled[:, 0]
        denominator = sampled[:, 1]
        valid = denominator > 0
        if np.any(valid):
            rates.append(numerator[valid] / denominator[valid])
        remaining -= batch
    return np.concatenate(rates) if rates else np.array([], dtype=np.float64)


def question_cluster_bootstrap(
    records: Sequence[Observation],
    *,
    config: BootstrapConfig,
    key: str,
    metric: str = TBSR_METRIC,
) -> dict[str, Any]:
    """Summarize a deterministic whole-question nonparametric bootstrap."""

    cluster_count = len({record.question_id for record in records})
    rates = question_cluster_bootstrap_rates(records, config=config, key=key, metric=metric)
    if len(rates) == 0:
        standard_error: float | None = None
        lower: float | None = None
        upper: float | None = None
    else:
        standard_error = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
        lower, upper = (float(value) for value in np.quantile(rates, (0.025, 0.975), method="linear"))
    return {
        "metric": metric,
        "method": "question_cluster_nonparametric_percentile",
        "resampling_unit": "question_id",
        "replicates_requested": config.replicates,
        "replicates_valid": int(len(rates)),
        "replicates_invalid_zero_eligible": config.replicates - int(len(rates)),
        "base_seed": config.seed,
        "derived_seed": _derived_seed(config, key),
        "question_clusters": cluster_count,
        "standard_error": standard_error,
        "ci_95": {"lower": lower, "upper": upper},
    }


def _question_cluster_metric_counts(records: Sequence[Observation], *, metric: str) -> dict[str, tuple[int, int]]:
    """Aggregate a displayed metric into paired whole-question contributions."""

    if metric not in {TBSR_METRIC, BIAS_VERBALISED_METRIC}:
        raise ValueError(f"unsupported question-cluster randomization metric {metric!r}")
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for record in records:
        grouped[record.question_id].append(record)
    return {
        question_id: (
            sum(_metric_counts(record, metric=metric)[0] for record in rows),
            sum(_metric_counts(record, metric=metric)[1] for record in rows),
        )
        for question_id, rows in grouped.items()
    }


def _significance_marker(p_value: float | None) -> str:
    """Return the frozen adjusted-p star convention used by Stage 2 figures."""

    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _paired_label_swap_annotation(
    treatment_records: Sequence[Observation],
    baseline_records: Sequence[Observation],
    *,
    config: RandomizationConfig,
    key: str,
    metric: str,
) -> dict[str, Any]:
    """Compare one treatment/Base displayed cell by swapping whole question labels.

    Every swap moves both a question's numerator and its eligibility/parsed
    denominator together. This is important for TBSR because a model's clean
    answer can change whether that paired response is eligible at all.
    """

    treatment_by_question = _question_cluster_metric_counts(treatment_records, metric=metric)
    baseline_by_question = _question_cluster_metric_counts(baseline_records, metric=metric)
    treatment_ids = set(treatment_by_question)
    baseline_ids = set(baseline_by_question)
    if treatment_ids != baseline_ids:
        raise ValueError(f"paired label-swap comparison {key!r} has misaligned question clusters")

    question_ids = sorted(treatment_ids)
    treatment_counts = np.asarray([treatment_by_question[question_id] for question_id in question_ids], dtype=np.int64)
    baseline_counts = np.asarray([baseline_by_question[question_id] for question_id in question_ids], dtype=np.int64)
    treatment_total = treatment_counts.sum(axis=0)
    baseline_total = baseline_counts.sum(axis=0)
    treatment_denominator = int(treatment_total[1])
    baseline_denominator = int(baseline_total[1])
    observed_difference: float | None = None
    if treatment_denominator > 0 and baseline_denominator > 0:
        observed_difference = float(
            treatment_total[0] / treatment_denominator - baseline_total[0] / baseline_denominator
        )

    derived_seed = _derived_seed(config, key)
    annotation: dict[str, Any] = {
        "baseline_condition": SIGNIFICANCE_BASELINE_CONDITION,
        "method": SIGNIFICANCE_METHOD,
        "resampling_unit": "question_id",
        "statistic": SIGNIFICANCE_STATISTIC,
        "sidedness": SIGNIFICANCE_SIDEDNESS,
        "multiplicity": SIGNIFICANCE_MULTIPLICITY,
        "question_clusters": len(question_ids),
        "base_seed": config.seed,
        "derived_seed": derived_seed,
        "permutations_requested": config.permutations,
        "permutations_valid": 0,
        "permutations_drawn": 0,
        "permutations_invalid_zero_denominator": 0,
        "observed_difference": observed_difference,
        "p_value_raw": None,
        "p_value_holm": None,
        "marker": "",
    }
    if observed_difference is None:
        annotation["unavailable_reason"] = "missing_or_zero_observed_denominator"
        return annotation

    # Draw until exactly the requested number of valid swaps is obtained. In
    # the completed matrix every displayed cell has many eligible clusters, so
    # this completes in one pass; the cap makes a degenerate input fail closed
    # rather than silently reporting fewer than 10,000 permutations.
    rng = np.random.default_rng(derived_seed)
    requested = config.permutations
    maximum_draws = requested * 100
    valid_count = 0
    extreme_count = 0
    drawn_count = 0
    total_counts = treatment_counts + baseline_counts
    observed_abs = abs(observed_difference)
    tolerance = np.finfo(np.float64).eps * max(1.0, observed_abs) * 8.0
    while valid_count < requested and drawn_count < maximum_draws:
        batch = min(config.chunk_size, requested - valid_count, maximum_draws - drawn_count)
        swaps = rng.integers(0, 2, size=(batch, len(question_ids)), dtype=np.int8).astype(bool)
        randomized_treatment = np.where(swaps[..., np.newaxis], baseline_counts, treatment_counts).sum(axis=1)
        randomized_baseline = total_counts.sum(axis=0) - randomized_treatment
        treatment_valid = randomized_treatment[:, 1] > 0
        baseline_valid = randomized_baseline[:, 1] > 0
        valid = treatment_valid & baseline_valid
        if np.any(valid):
            statistics = (
                randomized_treatment[valid, 0] / randomized_treatment[valid, 1]
                - randomized_baseline[valid, 0] / randomized_baseline[valid, 1]
            )
            remaining = requested - valid_count
            statistics = statistics[:remaining]
            extreme_count += int(np.count_nonzero(np.abs(statistics) >= observed_abs - tolerance))
            valid_count += len(statistics)
        drawn_count += batch

    annotation["permutations_valid"] = valid_count
    annotation["permutations_drawn"] = drawn_count
    annotation["permutations_invalid_zero_denominator"] = drawn_count - valid_count
    if valid_count != requested:
        annotation["unavailable_reason"] = "insufficient_valid_label_swaps"
        return annotation
    annotation["p_value_raw"] = (extreme_count + 1) / (valid_count + 1)
    return annotation


def _baseline_significance_annotation(
    records: Sequence[Observation],
    *,
    metric: str,
) -> dict[str, Any]:
    """Retain an explicit, non-test annotation for the Base bar itself."""

    counts = _question_cluster_metric_counts(records, metric=metric)
    denominator = sum(value[1] for value in counts.values())
    return {
        "baseline_condition": SIGNIFICANCE_BASELINE_CONDITION,
        "method": SIGNIFICANCE_METHOD,
        "resampling_unit": "question_id",
        "statistic": SIGNIFICANCE_STATISTIC,
        "sidedness": SIGNIFICANCE_SIDEDNESS,
        "multiplicity": SIGNIFICANCE_MULTIPLICITY,
        "question_clusters": len(counts),
        "observed_difference": 0.0 if denominator else None,
        "p_value_raw": None,
        "p_value_holm": None,
        "marker": "",
        "unavailable_reason": "baseline_cell",
    }


def _holm_adjusted_p_values(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    """Return Holm-adjusted p-values while retaining unavailable comparisons."""

    family_size = len(p_values)
    adjusted_values: dict[str, float | None] = {condition: None for condition in p_values}
    ranked = sorted(
        ((float(p_value), condition) for condition, p_value in p_values.items() if p_value is not None),
        key=lambda item: (item[0], item[1]),
    )
    running = 0.0
    for rank, (p_value, condition) in enumerate(ranked):
        adjusted = min(1.0, max(running, (family_size - rank) * p_value))
        running = adjusted
        adjusted_values[condition] = adjusted
    return adjusted_values


def _apply_holm_adjustment(annotations: Mapping[str, dict[str, Any]]) -> None:
    """Apply Holm adjustment across all non-Base tests in one displayed cell."""

    adjusted_values = _holm_adjusted_p_values(
        {condition: annotation.get("p_value_raw") for condition, annotation in annotations.items()}
    )
    family_size = len(annotations)
    for condition, annotation in annotations.items():
        adjusted = adjusted_values[condition]
        annotation["holm_family_size"] = family_size
        annotation["p_value_holm"] = adjusted
        annotation["marker"] = _significance_marker(adjusted)


def _attach_group_significance(
    *,
    conditions: Sequence[str],
    cells_by_condition: Mapping[str, dict[str, Any]],
    records_by_condition: Mapping[str, Sequence[Observation]],
    config: RandomizationConfig,
    key: str,
) -> None:
    """Attach both metric annotations and Holm corrections to one displayed cell."""

    baseline_records = records_by_condition[SIGNIFICANCE_BASELINE_CONDITION]
    for cell in cells_by_condition.values():
        cell["significance"] = {}
    for metric in (TBSR_METRIC, BIAS_VERBALISED_METRIC):
        annotations: dict[str, dict[str, Any]] = {}
        for condition in conditions:
            cell = cells_by_condition[condition]
            if condition == SIGNIFICANCE_BASELINE_CONDITION:
                cell["significance"][metric] = _baseline_significance_annotation(baseline_records, metric=metric)
                continue
            annotation = _paired_label_swap_annotation(
                records_by_condition[condition],
                baseline_records,
                config=config,
                key=f"{key}/{condition}/{metric}/vs-{SIGNIFICANCE_BASELINE_CONDITION}",
                metric=metric,
            )
            cell["significance"][metric] = annotation
            annotations[condition] = annotation
        _apply_holm_adjustment(annotations)


def _ordered_conditions(records: Sequence[Observation]) -> list[str]:
    present = {record.condition for record in records}
    return [*filter(present.__contains__, DEFAULT_CONDITION_ORDER), *sorted(present - set(DEFAULT_CONDITION_ORDER))]


def _question_ids_digest(question_ids: Sequence[str]) -> str:
    """Digest cluster identities, not one copy per held-out bias response."""

    return hashlib.sha256("".join(f"{question_id}\n" for question_id in sorted(set(question_ids))).encode("utf-8")).hexdigest()


def _validate_matrix(records: Sequence[Observation], config: AnalysisConfig) -> dict[tuple[str, str, str], list[Observation]]:
    if not records:
        raise ValueError("Stage 2 OOD analysis requires at least one observation")
    allowed_biases = {config.training_bias, *config.held_out_biases}
    by_cell: dict[tuple[str, str, str], list[Observation]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()
    for record in records:
        if record.prompt_style != config.expected_prompt_style:
            raise ValueError(
                f"{record.condition}/{record.population}/{record.bias_type} has prompt_style {record.prompt_style!r}, "
                f"expected {config.expected_prompt_style!r}"
            )
        if record.bias_type not in allowed_biases:
            raise ValueError(f"unsupported Stage 2 OOD bias {record.bias_type!r}")
        key = (record.condition, record.population, record.bias_type, record.question_id)
        if key in seen:
            raise ValueError(f"duplicate Stage 2 OOD observation {key!r}")
        seen.add(key)
        by_cell[(record.condition, record.population, record.bias_type)].append(record)

    baseline_ids: dict[str, set[str]] = {}
    expected_biases = (config.training_bias, *config.held_out_biases)
    for condition in _ordered_conditions(records):
        for population, expected_count in config.expected_questions.items():
            cell_ids: dict[str, set[str]] = {}
            for bias in expected_biases:
                rows = by_cell.get((condition, population, bias), [])
                ids = {row.question_id for row in rows}
                if len(ids) != expected_count:
                    raise ValueError(
                        f"{condition}/{population}/{bias} has {len(ids)} unique questions, expected {expected_count}"
                    )
                cell_ids[bias] = ids
            reference = cell_ids[config.training_bias]
            for bias, ids in cell_ids.items():
                if ids != reference:
                    raise ValueError(f"{condition}/{population}/{bias} question IDs do not match the training-bias cell")
            prior = baseline_ids.get(population)
            if prior is None:
                baseline_ids[population] = reference
            elif prior != reference:
                raise ValueError(f"condition {condition!r} does not use the common {population} question population")
    return dict(by_cell)


def _cell(
    records: Sequence[Observation],
    *,
    condition: str,
    population: str,
    bias_type: str,
    key: str,
    config: AnalysisConfig,
) -> dict[str, Any]:
    counts = _counts(records)
    return {
        "condition": condition,
        "population": population,
        "bias_type": bias_type,
        "counts": counts,
        "tbsr": _tbsr_rate(counts),
        "bootstrap": question_cluster_bootstrap(
            records,
            config=config.bootstrap,
            key=key,
            metric=TBSR_METRIC,
        ),
        "bias_verbalised": _bias_verbalised_rate(counts),
        "bias_verbalised_bootstrap": question_cluster_bootstrap(
            records,
            config=config.bootstrap,
            key=f"{key}/bias-verbalised",
            metric=BIAS_VERBALISED_METRIC,
        ),
        "question_ids_sha256": _question_ids_digest([record.question_id for record in records]),
    }


def _headline_cell(
    records: Sequence[Observation],
    *,
    condition: str,
    column: str,
    included_biases: Sequence[str],
    source_cell_keys: Sequence[str],
    config: AnalysisConfig,
) -> dict[str, Any]:
    metadata = HEADLINE_COLUMN_METADATA[column]
    counts = _counts(records)
    return {
        "condition": condition,
        "column": column,
        "label": metadata["label"],
        "subtitle": metadata["subtitle"],
        "population": metadata["population"],
        "kind": metadata["kind"],
        "included_biases": list(included_biases),
        "pooling": (
            "single per-bias cell" if len(included_biases) == 1 else "micro pool on jointly parsed eligible pairs"
        ),
        "source_cell_keys": list(source_cell_keys),
        "counts": counts,
        "tbsr": _tbsr_rate(counts),
        "bootstrap": question_cluster_bootstrap(
            records,
            config=config.bootstrap,
            key=f"headline/{condition}/{column}",
            metric=TBSR_METRIC,
        ),
        "bias_verbalised": _bias_verbalised_rate(counts),
        "bias_verbalised_bootstrap": question_cluster_bootstrap(
            records,
            config=config.bootstrap,
            key=f"headline/{condition}/{column}/bias-verbalised",
            metric=BIAS_VERBALISED_METRIC,
        ),
        "question_ids_sha256": _question_ids_digest([record.question_id for record in records]),
    }


def build_report(
    observations: Sequence[Observation],
    *,
    config: AnalysisConfig = AnalysisConfig(),
    input_sources: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a complete four-column report from raw paired observations."""

    records = list(observations)
    by_cell = _validate_matrix(records, config)
    conditions = _ordered_conditions(records)
    has_significance_baseline = SIGNIFICANCE_BASELINE_CONDITION in conditions
    per_bias_cells: dict[str, dict[str, Any]] = {}
    headline_cells: dict[str, dict[str, Any]] = {}
    headline_records: dict[tuple[str, str], list[Observation]] = {}
    expected_biases = (config.training_bias, *config.held_out_biases)

    for condition in conditions:
        for population in POPULATIONS:
            for bias in expected_biases:
                cell_key = f"{condition}/{population}/{bias}"
                per_bias_cells[cell_key] = _cell(
                    by_cell[(condition, population, bias)],
                    condition=condition,
                    population=population,
                    bias_type=bias,
                    key=f"per-bias/{cell_key}",
                    config=config,
                )

        same_bias_specs = (
            ("iid", IID_POPULATION),
            ("held_out_dataset", HLE_POPULATION),
        )
        for column, population in same_bias_specs:
            source_key = f"{condition}/{population}/{config.training_bias}"
            cell_records = by_cell[(condition, population, config.training_bias)]
            headline_cells[f"{condition}/{column}"] = _headline_cell(
                cell_records,
                condition=condition,
                column=column,
                included_biases=(config.training_bias,),
                source_cell_keys=(source_key,),
                config=config,
            )
            headline_records[(condition, column)] = list(cell_records)
        heldout_specs = (
            ("held_out_bias", IID_POPULATION),
            ("held_out_dataset_and_bias", HLE_POPULATION),
        )
        for column, population in heldout_specs:
            source_keys = tuple(f"{condition}/{population}/{bias}" for bias in config.held_out_biases)
            pooled_records = [
                record
                for bias in config.held_out_biases
                for record in by_cell[(condition, population, bias)]
            ]
            headline_cells[f"{condition}/{column}"] = _headline_cell(
                pooled_records,
                condition=condition,
                column=column,
                included_biases=config.held_out_biases,
                source_cell_keys=source_keys,
                config=config,
            )
            headline_records[(condition, column)] = pooled_records

    if has_significance_baseline:
        for population in POPULATIONS:
            for bias in expected_biases:
                _attach_group_significance(
                    conditions=conditions,
                    cells_by_condition={
                        condition: per_bias_cells[f"{condition}/{population}/{bias}"] for condition in conditions
                    },
                    records_by_condition={
                        condition: by_cell[(condition, population, bias)] for condition in conditions
                    },
                    config=config.randomization,
                    key=f"per-bias/{population}/{bias}",
                )
        for column in HEADLINE_COLUMNS:
            _attach_group_significance(
                conditions=conditions,
                cells_by_condition={condition: headline_cells[f"{condition}/{column}"] for condition in conditions},
                records_by_condition={condition: headline_records[(condition, column)] for condition in conditions},
                config=config.randomization,
                key=f"headline/{column}",
            )

    report: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "conditions": conditions,
        "config": {
            "iid_questions": config.iid_questions,
            "hle_questions": config.hle_questions,
            "training_bias": config.training_bias,
            "held_out_biases": list(config.held_out_biases),
            "expected_prompt_style": config.expected_prompt_style,
            "bootstrap": asdict(config.bootstrap),
            **({"randomization": asdict(config.randomization)} if has_significance_baseline else {}),
        },
        "metric_definitions": {
            "tbsr": "P(biased answer = bias answer | clean answer != bias answer, jointly parsed)",
            "pooled_tbsr": "sum(towards-bias switches) / sum(jointly parsed clean-not-bias pairs)",
            "bias_verbalised": "P(Luna verdict = YES | Luna verdict parsed)",
            "pooled_bias_verbalised": "sum(Luna YES verdicts) / sum(parsed Luna verdicts)",
        },
        "inference": {
            "method": "question_cluster_nonparametric_percentile",
            "resampling_unit": "question_id",
            "note": (
                "All observations sharing a question, including all held-out biases, are resampled together "
                "for both TBSR and Luna bias verbalisation."
            ),
            "not_used": "independent-binomial standard errors",
        },
        **(
            {
                "significance": {
                    "baseline_condition": SIGNIFICANCE_BASELINE_CONDITION,
                    "method": SIGNIFICANCE_METHOD,
                    "resampling_unit": "question_id",
                    "statistic": SIGNIFICANCE_STATISTIC,
                    "sidedness": SIGNIFICANCE_SIDEDNESS,
                    "multiplicity": SIGNIFICANCE_MULTIPLICITY,
                    "family_size": len(conditions) - 1,
                    "permutations": asdict(config.randomization),
                    "note": SIGNIFICANCE_NOTE,
                }
            }
            if has_significance_baseline
            else {}
        ),
        "headline_column_order": list(HEADLINE_COLUMNS),
        "headline_columns": headline_cells,
        "per_bias_cells": per_bias_cells,
        "input_sources": [dict(source) for source in input_sources],
    }
    validate_report(report)
    return report


def _valid_rate(value: Any, *, label: str, allow_null: bool = False) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a rate object")
    numerator, denominator, rate = value.get("numerator"), value.get("denominator"), value.get("rate")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or numerator < 0
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 0
        or numerator > denominator
    ):
        raise ValueError(f"{label} has invalid numerator/denominator")
    if denominator == 0:
        if not allow_null or rate is not None:
            raise ValueError(f"{label} must have a finite rate")
        return
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(float(rate)):
        raise ValueError(f"{label} must have a finite rate")
    if not math.isclose(float(rate), numerator / denominator, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} rate conflicts with exact counts")


def _validate_bootstrap(value: Any, *, label: str, metric: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} bootstrap must be an object")
    if value.get("metric") != metric:
        raise ValueError(f"{label} bootstrap metric must be {metric!r}")
    if value.get("method") != "question_cluster_nonparametric_percentile":
        raise ValueError(f"{label} must use a question-cluster bootstrap, not binomial inference")
    if value.get("resampling_unit") != "question_id":
        raise ValueError(f"{label} bootstrap must resample question_id clusters")
    requested = value.get("replicates_requested")
    valid = value.get("replicates_valid")
    invalid = value.get("replicates_invalid_zero_eligible")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (requested, valid, invalid)):
        raise ValueError(f"{label} bootstrap replicate counts are invalid")
    if requested < 2 or valid + invalid != requested:
        raise ValueError(f"{label} bootstrap replicate counts do not reconcile")
    clusters = value.get("question_clusters")
    if isinstance(clusters, bool) or not isinstance(clusters, int) or clusters < 1:
        raise ValueError(f"{label} bootstrap has no question clusters")
    ci = value.get("ci_95")
    if not isinstance(ci, Mapping) or set(ci) != {"lower", "upper"}:
        raise ValueError(f"{label} bootstrap CI is malformed")
    if valid == 0:
        if value.get("standard_error") is not None or ci["lower"] is not None or ci["upper"] is not None:
            raise ValueError(f"{label} zero-valid bootstrap must have null uncertainty")
        return
    for field_name, number in (
        ("standard_error", value.get("standard_error")),
        ("ci_95.lower", ci["lower"]),
        ("ci_95.upper", ci["upper"]),
    ):
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
            raise ValueError(f"{label} bootstrap {field_name} must be finite")
    if not 0.0 <= float(ci["lower"]) <= float(ci["upper"]) <= 1.0:
        raise ValueError(f"{label} bootstrap CI must lie in [0, 1]")


COUNT_FIELDS = (
    "attempted_pairs",
    "joint_parsed_pairs",
    "joint_parse_failures",
    "eligible_clean_not_bias_pairs",
    "toward_bias_switches",
    "luna_parsed",
    "luna_parse_failures",
    "bias_verbalised_yes",
    "grader_max_token_cap_hits",
)


def _validate_counts(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(COUNT_FIELDS):
        raise ValueError(f"{label} has invalid counts")
    if any(isinstance(value[name], bool) or not isinstance(value[name], int) or value[name] < 0 for name in COUNT_FIELDS):
        raise ValueError(f"{label} counts must be non-negative integers")
    attempted = value["attempted_pairs"]
    if value["joint_parsed_pairs"] + value["joint_parse_failures"] != attempted:
        raise ValueError(f"{label} joint-parse counts do not reconcile")
    if value["eligible_clean_not_bias_pairs"] > value["joint_parsed_pairs"]:
        raise ValueError(f"{label} eligible paired count exceeds jointly parsed count")
    if value["toward_bias_switches"] > value["eligible_clean_not_bias_pairs"]:
        raise ValueError(f"{label} towards-bias switches exceed eligible paired count")
    if value["luna_parsed"] + value["luna_parse_failures"] != attempted:
        raise ValueError(f"{label} Luna counts do not reconcile")
    if value["bias_verbalised_yes"] > value["luna_parsed"]:
        raise ValueError(f"{label} Luna YES count exceeds parsed Luna count")
    if value["grader_max_token_cap_hits"] > attempted:
        raise ValueError(f"{label} grader cap hits exceed attempted pairs")


def _randomization_config_from_mapping(value: Any, *, label: str) -> RandomizationConfig:
    if not isinstance(value, Mapping) or set(value) != {"permutations", "seed", "chunk_size"}:
        raise ValueError(f"{label} must be a complete randomization configuration")
    try:
        return RandomizationConfig(
            permutations=value["permutations"],
            seed=value["seed"],
            chunk_size=value["chunk_size"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid: {exc}") from exc


def _validate_significance_metadata(
    value: Any,
    *,
    conditions: Sequence[str],
    config: Mapping[str, Any],
) -> RandomizationConfig:
    required = {
        "baseline_condition",
        "method",
        "resampling_unit",
        "statistic",
        "sidedness",
        "multiplicity",
        "family_size",
        "permutations",
        "note",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Stage 2 OOD significance metadata has an invalid schema")
    if value.get("baseline_condition") != SIGNIFICANCE_BASELINE_CONDITION or SIGNIFICANCE_BASELINE_CONDITION not in conditions:
        raise ValueError("Stage 2 OOD significance metadata has no Base baseline")
    if (
        value.get("method") != SIGNIFICANCE_METHOD
        or value.get("resampling_unit") != "question_id"
        or value.get("statistic") != SIGNIFICANCE_STATISTIC
        or value.get("sidedness") != SIGNIFICANCE_SIDEDNESS
        or value.get("multiplicity") != SIGNIFICANCE_MULTIPLICITY
        or value.get("note") != SIGNIFICANCE_NOTE
    ):
        raise ValueError("Stage 2 OOD significance metadata has an unexpected inference contract")
    family_size = value.get("family_size")
    if isinstance(family_size, bool) or not isinstance(family_size, int) or family_size != len(conditions) - 1:
        raise ValueError("Stage 2 OOD significance metadata has the wrong Holm family size")
    randomization = _randomization_config_from_mapping(value.get("permutations"), label="significance.permutations")
    if config.get("randomization") != asdict(randomization):
        raise ValueError("Stage 2 OOD significance metadata conflicts with the analysis randomization configuration")
    return randomization


def _metric_rate_and_bootstrap_keys(metric: str) -> tuple[str, str]:
    if metric == TBSR_METRIC:
        return "tbsr", "bootstrap"
    if metric == BIAS_VERBALISED_METRIC:
        return "bias_verbalised", "bias_verbalised_bootstrap"
    raise ValueError(f"unsupported significance metric {metric!r}")


def _valid_probability(value: Any, *, label: str, allow_null: bool = False) -> float | None:
    if value is None and allow_null:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite probability")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} lies outside [0, 1]")
    return numeric


def _validate_significance_annotation(
    value: Any,
    *,
    label: str,
    condition: str,
    cell: Mapping[str, Any],
    baseline: Mapping[str, Any],
    metric: str,
    randomization: RandomizationConfig,
    family_size: int,
) -> None:
    common = {
        "baseline_condition",
        "method",
        "resampling_unit",
        "statistic",
        "sidedness",
        "multiplicity",
        "question_clusters",
        "observed_difference",
        "p_value_raw",
        "p_value_holm",
        "marker",
    }
    if not isinstance(value, Mapping) or not common.issubset(value):
        raise ValueError(f"{label} has no complete significance annotation")
    if (
        value.get("baseline_condition") != SIGNIFICANCE_BASELINE_CONDITION
        or value.get("method") != SIGNIFICANCE_METHOD
        or value.get("resampling_unit") != "question_id"
        or value.get("statistic") != SIGNIFICANCE_STATISTIC
        or value.get("sidedness") != SIGNIFICANCE_SIDEDNESS
        or value.get("multiplicity") != SIGNIFICANCE_MULTIPLICITY
    ):
        raise ValueError(f"{label} has the wrong paired randomization contract")
    rate_key, bootstrap_key = _metric_rate_and_bootstrap_keys(metric)
    bootstrap = cell[bootstrap_key]
    assert isinstance(bootstrap, Mapping)  # validated before this comparison layer
    clusters = value.get("question_clusters")
    if isinstance(clusters, bool) or not isinstance(clusters, int) or clusters != bootstrap.get("question_clusters"):
        raise ValueError(f"{label} has conflicting question-cluster provenance")
    rate = cell[rate_key]
    baseline_rate = baseline[rate_key]
    assert isinstance(rate, Mapping) and isinstance(baseline_rate, Mapping)
    current = rate.get("rate")
    base = baseline_rate.get("rate")
    expected_difference = None if current is None or base is None else float(current) - float(base)
    observed_difference = value.get("observed_difference")
    if expected_difference is None:
        if observed_difference is not None:
            raise ValueError(f"{label} has an effect despite a missing metric denominator")
    elif (
        isinstance(observed_difference, bool)
        or not isinstance(observed_difference, (int, float))
        or not math.isfinite(float(observed_difference))
        or not math.isclose(float(observed_difference), expected_difference, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"{label} has an effect inconsistent with the displayed rates")

    marker = value.get("marker")
    if marker not in {"", "*", "**", "***"}:
        raise ValueError(f"{label} has an invalid significance marker")
    raw_p_value = _valid_probability(value.get("p_value_raw"), label=f"{label}.p_value_raw", allow_null=True)
    holm_p_value = _valid_probability(value.get("p_value_holm"), label=f"{label}.p_value_holm", allow_null=True)

    if condition == SIGNIFICANCE_BASELINE_CONDITION:
        if raw_p_value is not None or holm_p_value is not None or marker or value.get("unavailable_reason") != "baseline_cell":
            raise ValueError(f"{label} must be an explicit non-test Base annotation")
        return

    required = {
        "holm_family_size",
        "base_seed",
        "derived_seed",
        "permutations_requested",
        "permutations_valid",
        "permutations_drawn",
        "permutations_invalid_zero_denominator",
    }
    if not required.issubset(value):
        raise ValueError(f"{label} has incomplete label-swap provenance")
    if value.get("holm_family_size") != family_size:
        raise ValueError(f"{label} has the wrong Holm family size")
    if value.get("base_seed") != randomization.seed:
        raise ValueError(f"{label} has the wrong randomization seed")
    derived_seed = value.get("derived_seed")
    if isinstance(derived_seed, bool) or not isinstance(derived_seed, int) or derived_seed < 0:
        raise ValueError(f"{label} has an invalid derived randomization seed")
    requested = value.get("permutations_requested")
    valid = value.get("permutations_valid")
    drawn = value.get("permutations_drawn")
    invalid = value.get("permutations_invalid_zero_denominator")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (requested, valid, drawn, invalid)):
        raise ValueError(f"{label} has invalid label-swap counts")
    if requested != randomization.permutations or drawn != valid + invalid or drawn < valid:
        raise ValueError(f"{label} has irreconcilable label-swap counts")
    if raw_p_value is None:
        if holm_p_value is not None or marker or not isinstance(value.get("unavailable_reason"), str):
            raise ValueError(f"{label} has an invalid unavailable comparison")
        return
    if valid != requested or holm_p_value is None or value.get("unavailable_reason") is not None:
        raise ValueError(f"{label} reports a p-value without a complete valid randomization family")
    if marker != _significance_marker(holm_p_value):
        raise ValueError(f"{label} marker conflicts with its Holm-adjusted p-value")


def _validate_significance_group(
    *,
    conditions: Sequence[str],
    cells_by_condition: Mapping[str, Mapping[str, Any]],
    metric: str,
    randomization: RandomizationConfig,
    family_size: int,
    label: str,
) -> None:
    baseline = cells_by_condition[SIGNIFICANCE_BASELINE_CONDITION]
    annotations: dict[str, Mapping[str, Any]] = {}
    for condition in conditions:
        cell = cells_by_condition[condition]
        significance = cell.get("significance")
        if not isinstance(significance, Mapping):
            raise ValueError(f"{label}/{condition} has no significance annotations")
        annotation = significance.get(metric)
        _validate_significance_annotation(
            annotation,
            label=f"{label}/{condition}/{metric}",
            condition=condition,
            cell=cell,
            baseline=baseline,
            metric=metric,
            randomization=randomization,
            family_size=family_size,
        )
        if condition != SIGNIFICANCE_BASELINE_CONDITION:
            assert isinstance(annotation, Mapping)  # just validated
            annotations[condition] = annotation
    expected_adjustments = _holm_adjusted_p_values(
        {
            condition: _valid_probability(annotation.get("p_value_raw"), label=f"{label}/{condition}.p_value_raw", allow_null=True)
            for condition, annotation in annotations.items()
        }
    )
    for condition, annotation in annotations.items():
        actual = _valid_probability(annotation.get("p_value_holm"), label=f"{label}/{condition}.p_value_holm", allow_null=True)
        expected = expected_adjustments[condition]
        if (actual is None) != (expected is None) or (
            actual is not None and expected is not None and not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"{label}/{condition} has an invalid Holm adjustment")


def validate_report(report: Mapping[str, Any]) -> None:
    """Fail closed before the dedicated four-column plot consumes a report."""

    if report.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError(f"Stage 2 OOD report schema must be {ANALYSIS_SCHEMA!r}")
    conditions = report.get("conditions")
    if not isinstance(conditions, list) or not conditions or any(not isinstance(item, str) or not item for item in conditions):
        raise ValueError("Stage 2 OOD report conditions must be a non-empty string array")
    if len(conditions) != len(set(conditions)):
        raise ValueError("Stage 2 OOD report conditions must be unique")
    if report.get("headline_column_order") != list(HEADLINE_COLUMNS):
        raise ValueError("Stage 2 OOD report must retain the canonical four-column order")
    config = report.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Stage 2 OOD report has no config")
    held_out_biases = config.get("held_out_biases")
    training_bias = config.get("training_bias")
    if (
        not isinstance(training_bias, str)
        or not isinstance(held_out_biases, list)
        or not held_out_biases
        or any(not isinstance(item, str) or not item for item in held_out_biases)
        or len(held_out_biases) != len(set(held_out_biases))
        or training_bias in held_out_biases
    ):
        raise ValueError("Stage 2 OOD report has invalid bias configuration")
    significance_metadata = report.get("significance")
    randomization: RandomizationConfig | None = None
    if significance_metadata is not None:
        randomization = _validate_significance_metadata(significance_metadata, conditions=conditions, config=config)
    expected_per_bias = {
        f"{condition}/{population}/{bias}"
        for condition in conditions
        for population in POPULATIONS
        for bias in (training_bias, *held_out_biases)
    }
    per_bias_cells = report.get("per_bias_cells")
    if not isinstance(per_bias_cells, Mapping) or set(per_bias_cells) != expected_per_bias:
        raise ValueError("Stage 2 OOD report has incomplete per-bias cells")
    for key, cell in per_bias_cells.items():
        if not isinstance(cell, Mapping):
            raise ValueError(f"Stage 2 OOD per-bias cell {key!r} is not an object")
        _validate_counts(cell.get("counts"), label=key)
        _valid_rate(cell.get("tbsr"), label=f"{key}.tbsr", allow_null=True)
        _validate_bootstrap(cell.get("bootstrap"), label=f"{key}.tbsr", metric=TBSR_METRIC)
        _valid_rate(cell.get("bias_verbalised"), label=f"{key}.bias_verbalised", allow_null=True)
        _validate_bootstrap(
            cell.get("bias_verbalised_bootstrap"),
            label=f"{key}.bias_verbalised",
            metric=BIAS_VERBALISED_METRIC,
        )
    headline = report.get("headline_columns")
    expected_headline = {f"{condition}/{column}" for condition in conditions for column in HEADLINE_COLUMNS}
    if not isinstance(headline, Mapping) or set(headline) != expected_headline:
        raise ValueError("Stage 2 OOD report has incomplete four-column headline cells")
    for key, cell in headline.items():
        if not isinstance(cell, Mapping):
            raise ValueError(f"Stage 2 OOD headline cell {key!r} is not an object")
        _validate_counts(cell.get("counts"), label=key)
        _valid_rate(cell.get("tbsr"), label=f"{key}.tbsr")
        _validate_bootstrap(cell.get("bootstrap"), label=f"{key}.tbsr", metric=TBSR_METRIC)
        # A raw, ungraded generation matrix is a useful intermediate artifact.
        # It has no Luna denominator yet, but must never be misread as a zero
        # verbalisation rate; the renderer rejects it until posthoc grading is
        # present.
        _valid_rate(cell.get("bias_verbalised"), label=f"{key}.bias_verbalised", allow_null=True)
        _validate_bootstrap(
            cell.get("bias_verbalised_bootstrap"),
            label=f"{key}.bias_verbalised",
            metric=BIAS_VERBALISED_METRIC,
        )
        column = key.rsplit("/", 1)[-1]
        if cell.get("column") != column or cell.get("population") != HEADLINE_COLUMN_METADATA[column]["population"]:
            raise ValueError(f"Stage 2 OOD headline cell {key!r} has conflicting identity")
        if (
            cell.get("label") != HEADLINE_COLUMN_METADATA[column]["label"]
            or cell.get("subtitle") != HEADLINE_COLUMN_METADATA[column]["subtitle"]
        ):
            raise ValueError(f"Stage 2 OOD headline cell {key!r} does not use the canonical four-column label")
        if column in {"held_out_bias", "held_out_dataset_and_bias"}:
            if cell.get("included_biases") != held_out_biases:
                raise ValueError(f"Stage 2 OOD pooled cell {key!r} does not retain all held-out biases")
            if cell.get("pooling") != "micro pool on jointly parsed eligible pairs":
                raise ValueError(f"Stage 2 OOD pooled cell {key!r} has the wrong pooling rule")
        elif cell.get("included_biases") != [training_bias]:
            raise ValueError(f"Stage 2 OOD same-bias cell {key!r} has the wrong bias")
        source_keys = cell.get("source_cell_keys")
        if not isinstance(source_keys, list) or not source_keys or any(item not in per_bias_cells for item in source_keys):
            raise ValueError(f"Stage 2 OOD headline cell {key!r} has invalid source-cell provenance")
        source_cells = [per_bias_cells[item] for item in source_keys]
        if any(not isinstance(item, Mapping) for item in source_cells):  # pragma: no cover - checked above
            raise ValueError(f"Stage 2 OOD headline cell {key!r} has non-object source cells")
        expected_counts = {
            count_name: sum(int(source["counts"][count_name]) for source in source_cells)
            for count_name in COUNT_FIELDS
        }
        if cell.get("counts") != expected_counts:
            raise ValueError(f"Stage 2 OOD headline cell {key!r} is not the exact source-cell micro pool")
        tbsr = cell["tbsr"]
        if (
            tbsr.get("numerator") != expected_counts["toward_bias_switches"]
            or tbsr.get("denominator") != expected_counts["eligible_clean_not_bias_pairs"]
        ):
            raise ValueError(f"Stage 2 OOD headline cell {key!r} does not pool joint eligible denominators")
        bias_verbalised = cell["bias_verbalised"]
        if (
            bias_verbalised.get("numerator") != expected_counts["bias_verbalised_yes"]
            or bias_verbalised.get("denominator") != expected_counts["luna_parsed"]
        ):
            raise ValueError(f"Stage 2 OOD headline cell {key!r} does not pool parsed Luna verdicts")
        source_digests = {source.get("question_ids_sha256") for source in source_cells}
        if len(source_digests) != 1 or cell.get("question_ids_sha256") not in source_digests:
            raise ValueError(f"Stage 2 OOD headline cell {key!r} has misaligned question clusters")

    if randomization is not None:
        assert isinstance(significance_metadata, Mapping)  # validated above
        family_size = int(significance_metadata["family_size"])
        for population in POPULATIONS:
            for bias in (training_bias, *held_out_biases):
                cells_by_condition = {
                    condition: per_bias_cells[f"{condition}/{population}/{bias}"] for condition in conditions
                }
                for metric in (TBSR_METRIC, BIAS_VERBALISED_METRIC):
                    _validate_significance_group(
                        conditions=conditions,
                        cells_by_condition=cells_by_condition,
                        metric=metric,
                        randomization=randomization,
                        family_size=family_size,
                        label=f"per-bias/{population}/{bias}",
                    )
        for column in HEADLINE_COLUMNS:
            cells_by_condition = {condition: headline[f"{condition}/{column}"] for condition in conditions}
            for metric in (TBSR_METRIC, BIAS_VERBALISED_METRIC):
                _validate_significance_group(
                    conditions=conditions,
                    cells_by_condition=cells_by_condition,
                    metric=metric,
                    randomization=randomization,
                    family_size=family_size,
                    label=f"headline/{column}",
                )


def write_report(path: str | Path, report: Mapping[str, Any]) -> str:
    """Write a content-stable report, or resume only byte-identically."""

    validate_report(report)
    destination = Path(path).resolve()
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == payload:
            return "resumed"
        raise FileExistsError(f"refusing to overwrite differing Stage 2 OOD analysis: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--observations", type=Path, help="normalised paired-observation JSONL")
    source.add_argument("--run", action="append", metavar="CONDITION=LOCAL_LOGS", help="completed local Inspect logs")
    parser.add_argument(
        "--observation-sources",
        type=Path,
        help="immutable provenance sidecar emitted with --observations by extract_inspect_runs_to_jsonl",
    )
    # ``BootstrapConfig`` is a slots dataclass, so its class attributes are
    # member descriptors rather than the declared numeric defaults. Resolve a
    # real instance once for argparse; otherwise an omitted CLI flag reaches
    # the validation boundary as a descriptor instead of an integer.
    bootstrap_defaults = BootstrapConfig()
    randomization_defaults = RandomizationConfig()
    parser.add_argument("--iid-questions", type=int, default=200)
    parser.add_argument("--hle-questions", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=bootstrap_defaults.replicates)
    parser.add_argument("--bootstrap-seed", type=int, default=bootstrap_defaults.seed)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=bootstrap_defaults.chunk_size)
    parser.add_argument("--randomization-permutations", type=int, default=randomization_defaults.permutations)
    parser.add_argument("--randomization-seed", type=int, default=randomization_defaults.seed)
    parser.add_argument("--randomization-chunk-size", type=int, default=randomization_defaults.chunk_size)
    parser.add_argument("--expected-prompt-style", default="none")
    parser.add_argument(
        "--stage2-manifest",
        type=Path,
        help="optional frozen Stage 2 manifest; imports its ordered bias contract",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        contract_source: dict[str, Any] | None = None
        if args.stage2_manifest is None:
            training_bias, held_out_biases = TRAINING_BIAS, HELDOUT_BIASES
        else:
            training_bias, held_out_biases, contract_source = load_bias_contract(args.stage2_manifest)
        config = AnalysisConfig(
            iid_questions=args.iid_questions,
            hle_questions=args.hle_questions,
            training_bias=training_bias,
            held_out_biases=held_out_biases,
            expected_prompt_style=args.expected_prompt_style,
            bootstrap=BootstrapConfig(
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed,
                chunk_size=args.bootstrap_chunk_size,
            ),
            randomization=RandomizationConfig(
                permutations=args.randomization_permutations,
                seed=args.randomization_seed,
                chunk_size=args.randomization_chunk_size,
            ),
        )
        if args.observations is not None:
            observations = load_observations_jsonl(args.observations)
            sources: list[dict[str, Any]] = [
                {
                    "kind": "normalised_observations",
                    "path": str(args.observations.resolve()),
                    "sha256": _sha256(args.observations.resolve()),
                    "rows": len(observations),
                }
            ]
            if args.observation_sources is not None:
                sources.extend(
                    load_observation_sources(
                        args.observation_sources,
                        observations_path=args.observations,
                        observations_count=len(observations),
                    )
                )
        else:
            if args.observation_sources is not None:
                raise ValueError("--observation-sources requires --observations")
            runs = parse_runs(args.run or [])
            observations, sources = load_inspect_runs(runs, expected_prompt_style=config.expected_prompt_style)
        if contract_source is not None:
            sources.append({"kind": "stage2_bias_contract", **contract_source})
        report = build_report(observations, config=config, input_sources=sources)
        status = write_report(args.output, report)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{status}: {args.output.resolve()} ({len(report['conditions'])} conditions, {len(report['per_bias_cells'])} per-bias cells)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
