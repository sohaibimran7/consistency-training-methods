#!/usr/bin/env python3
"""Estimate Figure 6 runtime, GPU exposure, NHR, and cache capacity.

Queue wait is deliberately outside this estimator.  Scheduler start estimates
are volatile observations, not runtime guarantees.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_GENERATIONS_PER_MODEL = 5_400
NHR_GPU_HOURS = 4.0
REQUIRED_CACHE_CAPACITY_GB = 1_300.0


@dataclass(frozen=True, slots=True)
class PilotObservation:
    """Successful append-only generation records and their wall time."""

    model_key: str
    successful_generations: int
    completion_tokens: int
    tokenized_generations: int
    elapsed_seconds: float
    elapsed_source: str
    request_elapsed_seconds: float
    timed_generations: int

    @property
    def generations_per_second(self) -> float:
        return observed_rate(self.successful_generations, self.elapsed_seconds)

    @property
    def completion_tokens_per_second(self) -> float | None:
        if self.completion_tokens <= 0:
            return None
        return observed_rate(self.completion_tokens, self.elapsed_seconds)

    @property
    def average_completion_tokens(self) -> float | None:
        if self.tokenized_generations <= 0:
            return None
        return self.completion_tokens / self.tokenized_generations

    @property
    def average_request_seconds(self) -> float | None:
        if self.timed_generations <= 0:
            return None
        return self.request_elapsed_seconds / self.timed_generations


@dataclass(frozen=True, slots=True)
class RuntimeEstimate:
    """Projected compute consumption for one model's full generation target."""

    model_key: str
    target_generations: int
    completed_generations: int
    remaining_generations: int
    tensor_parallel_size: int
    generations_per_second: float
    completion_tokens_per_second: float | None
    average_completion_tokens: float | None
    projected_total_seconds: float
    remaining_seconds: float
    projected_total_gpu_hours: float
    remaining_gpu_hours: float
    projected_total_nhr: float
    remaining_nhr: float


def observed_rate(amount: int | float, elapsed_seconds: int | float) -> float:
    """Return a positive amount-per-second observation."""

    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise TypeError("amount must be numeric")
    if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float)):
        raise TypeError("elapsed_seconds must be numeric")
    if not math.isfinite(float(amount)) or amount <= 0:
        raise ValueError("amount must be finite and > 0")
    if not math.isfinite(float(elapsed_seconds)) or elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be finite and > 0")
    return float(amount) / float(elapsed_seconds)


def estimate_runtime(
    *,
    model_key: str,
    target_generations: int,
    completed_generations: int = 0,
    tensor_parallel_size: int,
    generations_per_second: float,
    completion_tokens_per_second: float | None = None,
    average_completion_tokens: float | None = None,
) -> RuntimeEstimate:
    """Project runtime and Isambard NHR from a measured generation rate."""

    if not model_key:
        raise ValueError("model_key must not be empty")
    if isinstance(target_generations, bool) or not isinstance(target_generations, int) or target_generations < 1:
        raise ValueError("target_generations must be an integer >= 1")
    if (
        isinstance(completed_generations, bool)
        or not isinstance(completed_generations, int)
        or not 0 <= completed_generations <= target_generations
    ):
        raise ValueError("completed_generations must be an integer in [0, target_generations]")
    if isinstance(tensor_parallel_size, bool) or not isinstance(tensor_parallel_size, int) or tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be an integer >= 1")
    rate = observed_rate(generations_per_second, 1.0)
    if completion_tokens_per_second is not None:
        completion_tokens_per_second = observed_rate(completion_tokens_per_second, 1.0)
    if average_completion_tokens is not None:
        average_completion_tokens = observed_rate(average_completion_tokens, 1.0)
    remaining_generations = target_generations - completed_generations
    projected_total_seconds = target_generations / rate
    remaining_seconds = remaining_generations / rate
    projected_total_gpu_hours = projected_total_seconds / 3600.0 * tensor_parallel_size
    remaining_gpu_hours = remaining_seconds / 3600.0 * tensor_parallel_size
    return RuntimeEstimate(
        model_key=model_key,
        target_generations=target_generations,
        completed_generations=completed_generations,
        remaining_generations=remaining_generations,
        tensor_parallel_size=tensor_parallel_size,
        generations_per_second=rate,
        completion_tokens_per_second=completion_tokens_per_second,
        average_completion_tokens=average_completion_tokens,
        projected_total_seconds=projected_total_seconds,
        remaining_seconds=remaining_seconds,
        projected_total_gpu_hours=projected_total_gpu_hours,
        remaining_gpu_hours=remaining_gpu_hours,
        projected_total_nhr=projected_total_gpu_hours / NHR_GPU_HOURS,
        remaining_nhr=remaining_gpu_hours / NHR_GPU_HOURS,
    )


def storage_summary(
    weights_gb: Mapping[str, int | float],
    *,
    required_capacity_gb: float = REQUIRED_CACHE_CAPACITY_GB,
    available_gb: float | None = None,
) -> dict[str, Any]:
    """Summarize decimal-GB model weights and the required cache buffer."""

    if not weights_gb:
        raise ValueError("weights_gb must not be empty")
    normalized: dict[str, float] = {}
    for key, value in weights_gb.items():
        if not key:
            raise ValueError("weight keys must not be empty")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"weight size for {key} must be numeric")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"weight size for {key} must be finite and >= 0")
        normalized[key] = float(value)
    if isinstance(required_capacity_gb, bool) or not isinstance(required_capacity_gb, (int, float)):
        raise TypeError("required_capacity_gb must be numeric")
    if not math.isfinite(required_capacity_gb) or required_capacity_gb <= 0:
        raise ValueError("required_capacity_gb must be finite and > 0")
    if available_gb is not None:
        if isinstance(available_gb, bool) or not isinstance(available_gb, (int, float)):
            raise TypeError("available_gb must be numeric")
        if not math.isfinite(available_gb) or available_gb < 0:
            raise ValueError("available_gb must be finite and >= 0")
    total = sum(normalized.values())
    result: dict[str, Any] = {
        "weights_by_model_gb": normalized,
        "weights_total_gb": total,
        "weights_total_tb": total / 1000.0,
        "required_cache_capacity_gb": float(required_capacity_gb),
        "required_cache_capacity_tb": float(required_capacity_gb) / 1000.0,
        "buffer_above_weights_gb": float(required_capacity_gb) - total,
    }
    if available_gb is not None:
        result.update(
            {
                "available_gb": float(available_gb),
                "margin_above_requirement_gb": float(available_gb) - float(required_capacity_gb),
                "meets_requirement": available_gb >= required_capacity_gb,
            }
        )
    return result


def read_generation_jsonl(paths: Sequence[str | Path], *, expected_model_key: str) -> list[dict[str, Any]]:
    """Read generation JSONL and reject records for a different model."""

    records: list[dict[str, Any]] = []
    for path_like in paths:
        path = Path(path_like)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: each generation record must be an object")
                if record.get("model_key") != expected_model_key:
                    raise ValueError(
                        f"{path}:{line_number}: expected model_key {expected_model_key!r}, "
                        f"got {record.get('model_key')!r}"
                    )
                records.append(record)
    if not records:
        raise ValueError(f"pilot JSONL for {expected_model_key} contained no records")
    return records


def summarize_pilot_records(
    model_key: str,
    records: Sequence[Mapping[str, Any]],
    *,
    elapsed_seconds: float | None = None,
) -> PilotObservation:
    """Summarize unique successful logical generations from append-only rows."""

    elapsed_source = "cli_wall_seconds"
    if elapsed_seconds is None:
        elapsed_seconds = infer_elapsed_seconds(records)
        elapsed_source = "record_timestamps"
    observed_rate(1, elapsed_seconds)
    successes: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        if record.get("model_key") != model_key:
            raise ValueError(f"pilot record {index} does not belong to {model_key}")
        if record.get("status") != "success":
            continue
        generation_key = record.get("generation_key")
        if not isinstance(generation_key, str) or not generation_key:
            raise ValueError(f"successful pilot record {index} has no generation_key")
        if generation_key in successes:
            raise ValueError(f"pilot contains duplicate successful generation_key {generation_key}")
        successes[generation_key] = record
    if not successes:
        raise ValueError(f"pilot for {model_key} has no successful generations")
    completion_tokens = 0
    tokenized_generations = 0
    request_elapsed_seconds = 0.0
    timed_generations = 0
    for record in successes.values():
        tokens = record.get("completion_tokens")
        if tokens is None:
            continue
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("completion_tokens must be a non-negative integer or null")
        completion_tokens += tokens
        tokenized_generations += 1
    for record in successes.values():
        request_elapsed = record.get("elapsed_seconds")
        if request_elapsed is None:
            continue
        if (
            isinstance(request_elapsed, bool)
            or not isinstance(request_elapsed, (int, float))
            or not math.isfinite(float(request_elapsed))
            or request_elapsed < 0
        ):
            raise ValueError("elapsed_seconds must be a finite non-negative number or null")
        request_elapsed_seconds += float(request_elapsed)
        timed_generations += 1
    return PilotObservation(
        model_key=model_key,
        successful_generations=len(successes),
        completion_tokens=completion_tokens,
        tokenized_generations=tokenized_generations,
        elapsed_seconds=float(elapsed_seconds),
        elapsed_source=elapsed_source,
        request_elapsed_seconds=request_elapsed_seconds,
        timed_generations=timed_generations,
    )


def _timestamp(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def infer_elapsed_seconds(records: Sequence[Mapping[str, Any]]) -> float:
    """Infer pilot wall time from explicit per-record start/completion timestamps."""

    start_fields = ("started_at", "request_started_at", "started_at_utc")
    completion_fields = ("completed_at", "request_completed_at", "completed_at_utc")
    starts: list[dt.datetime] = []
    completions: list[dt.datetime] = []
    for record in records:
        start = None
        completion = None
        for field in start_fields:
            start = _timestamp(record.get(field))
            if start is not None:
                break
        for field in completion_fields:
            completion = _timestamp(record.get(field))
            if completion is not None:
                break
        if start is not None:
            starts.append(start)
        if completion is not None:
            completions.append(completion)
    if starts and completions:
        elapsed = (max(completions) - min(starts)).total_seconds()
        if elapsed > 0:
            return elapsed
    raise ValueError(
        "pilot rows do not contain a usable start/completion timestamp span; "
        "supply --pilot-elapsed-seconds MODEL=SECONDS (sacct ElapsedRaw is preferred)"
    )


def _key_value(value: str, *, option: str) -> tuple[str, str]:
    key, separator, raw = value.partition("=")
    if not separator or not key or not raw:
        raise ValueError(f"{option} values must have the form MODEL=VALUE")
    return key, raw


def _group_paths(values: Sequence[str], *, option: str) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        key, raw = _key_value(value, option=option)
        result[key].append(Path(raw))
    return dict(result)


def _float_map(values: Sequence[str], *, option: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        key, raw = _key_value(value, option=option)
        if key in result:
            raise ValueError(f"duplicate {option} value for {key}")
        try:
            parsed = float(raw)
        except ValueError as exc:
            raise ValueError(f"{option} value for {key} must be numeric") from exc
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"{option} value for {key} must be finite and > 0")
        result[key] = parsed
    return result


def _format_number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-config",
        type=Path,
        default=Path("experiments/eval_awareness/figure6/models.yaml"),
    )
    parser.add_argument(
        "--pilot",
        action="append",
        default=[],
        metavar="MODEL=JSONL",
        help="Append-only pilot generation JSONL; repeat to combine files for a model.",
    )
    parser.add_argument(
        "--pilot-elapsed-seconds",
        action="append",
        default=[],
        metavar="MODEL=SECONDS",
        help="Measured wall time override for all pilot files; sacct ElapsedRaw includes model startup.",
    )
    parser.add_argument(
        "--generations-per-second",
        action="append",
        default=[],
        metavar="MODEL=RATE",
        help="Direct observed generation rate, usable without pilot JSONL.",
    )
    parser.add_argument(
        "--completion-tokens-per-second",
        action="append",
        default=[],
        metavar="MODEL=RATE",
        help="Direct observed completion-token throughput.",
    )
    parser.add_argument(
        "--average-completion-tokens",
        action="append",
        default=[],
        metavar="MODEL=TOKENS",
        help="Mean completion length; combines with token throughput when no generation rate is supplied.",
    )
    parser.add_argument("--target-generations", type=int, default=DEFAULT_GENERATIONS_PER_MODEL)
    availability = parser.add_mutually_exclusive_group()
    availability.add_argument("--available-storage-gb", type=float)
    availability.add_argument("--cache-path", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.target_generations < 1:
        parser.error("--target-generations must be >= 1")
    try:
        with args.models_config.open(encoding="utf-8") as handle:
            models = yaml.safe_load(handle)["models"]
        pilots = _group_paths(args.pilot, option="--pilot")
        elapsed_by_model = _float_map(args.pilot_elapsed_seconds, option="--pilot-elapsed-seconds")
        generation_rates = _float_map(args.generations_per_second, option="--generations-per-second")
        token_rates = _float_map(args.completion_tokens_per_second, option="--completion-tokens-per-second")
        average_tokens = _float_map(args.average_completion_tokens, option="--average-completion-tokens")
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    supplied_keys = set(pilots) | set(elapsed_by_model) | set(generation_rates) | set(token_rates) | set(average_tokens)
    unknown = sorted(supplied_keys - set(models))
    if unknown:
        parser.error(f"unknown model key(s): {unknown}")
    if not set(elapsed_by_model).issubset(pilots):
        parser.error("--pilot-elapsed-seconds may name only models supplied with --pilot")

    observations: dict[str, PilotObservation] = {}
    for key, paths in pilots.items():
        try:
            records = read_generation_jsonl(paths, expected_model_key=key)
            observations[key] = summarize_pilot_records(key, records, elapsed_seconds=elapsed_by_model.get(key))
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))

    estimates: dict[str, RuntimeEstimate] = {}
    for key, model in models.items():
        observation = observations.get(key)
        generation_rate = generation_rates.get(key)
        token_rate = token_rates.get(key)
        average = average_tokens.get(key)
        if observation is not None:
            generation_rate = generation_rate or observation.generations_per_second
            token_rate = token_rate or observation.completion_tokens_per_second
            average = average or observation.average_completion_tokens
        if generation_rate is None and token_rate is not None and average is not None:
            generation_rate = token_rate / average
        if generation_rate is None:
            continue
        estimates[key] = estimate_runtime(
            model_key=key,
            target_generations=args.target_generations,
            completed_generations=observation.successful_generations if observation is not None else 0,
            tensor_parallel_size=int(model["tensor_parallel_size"]),
            generations_per_second=generation_rate,
            completion_tokens_per_second=token_rate,
            average_completion_tokens=average,
        )

    available_gb = args.available_storage_gb
    availability_kind = "declared"
    if args.cache_path is not None:
        try:
            available_gb = shutil.disk_usage(args.cache_path).free / 1_000_000_000
        except OSError as exc:
            parser.error(str(exc))
        availability_kind = "filesystem_free_not_user_quota"
    try:
        storage = storage_summary(
            {key: float(model["approximate_weights_gb"]) for key, model in models.items()},
            available_gb=available_gb,
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if available_gb is not None:
        storage["availability_kind"] = availability_kind

    observation_report = {
        key: {
            **asdict(value),
            "generations_per_second": value.generations_per_second,
            "completion_tokens_per_second": value.completion_tokens_per_second,
            "average_completion_tokens": value.average_completion_tokens,
            "average_request_seconds": value.average_request_seconds,
        }
        for key, value in observations.items()
    }
    report = {
        "target_generations_per_model": args.target_generations,
        "observations": observation_report,
        "estimates": {key: asdict(value) for key, value in estimates.items()},
        "storage": storage,
        "queue_wait_included": False,
        "scheduler_estimate_note": "Queue wait is external; live scheduler estimates are volatile and not guarantees.",
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("model                 done/target   gen/s    tok/s   avg tok   req s   ETA h   GPU h     NHR")
    print("--------------------  -----------  -------  -------  --------  ------  ------  ------  ------")
    for key in models:
        observation = observations.get(key)
        estimate = estimates.get(key)
        print(
            f"{key:20}  "
            f"{((str(observation.successful_generations) + '/' + str(args.target_generations)) if observation else ('0/' + str(args.target_generations))):>11}  "
            f"{_format_number(estimate.generations_per_second if estimate else None, 4):>7}  "
            f"{_format_number(estimate.completion_tokens_per_second if estimate else None, 2):>7}  "
            f"{_format_number(estimate.average_completion_tokens if estimate else None, 1):>8}  "
            f"{_format_number(observation.average_request_seconds if observation else None, 2):>6}  "
            f"{_format_number(estimate.remaining_seconds / 3600 if estimate else None, 2):>6}  "
            f"{_format_number(estimate.remaining_gpu_hours if estimate else None, 2):>6}  "
            f"{_format_number(estimate.remaining_nhr if estimate else None, 2):>6}"
        )
    print()
    print(
        f"Model weights: {storage['weights_total_gb']:.1f} GB ({storage['weights_total_tb']:.4f} TB); "
        f"required shared-cache capacity: {storage['required_cache_capacity_gb']:.0f} GB; "
        f"buffer above listed weights: {storage['buffer_above_weights_gb']:.1f} GB."
    )
    if available_gb is not None:
        print(
            f"Reported available storage: {available_gb:.1f} GB; "
            f"margin to requirement: {storage['margin_above_requirement_gb']:.1f} GB; "
            f"meets requirement: {storage['meets_requirement']}."
        )
        if availability_kind != "declared":
            print("Filesystem free space is not proof of an individual project/scratch quota.")
    print("Queue wait is external; live scheduler estimates are volatile and not guarantees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
