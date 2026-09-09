"""Deterministic prospective evidence-and-magnitude power for the switch gate.

This module is deliberately local-only: it generates synthetic Bernoulli outcomes from
the design alternative and never reads evaluation outcomes or calls a model.  Training
gates use exact one-sided binomial tests.  HLE gates use pooled point estimates with a
small-sample cluster-robust, one-sided t approximation at the question level; this
preserves the within-question dependence without embedding a costly bootstrap inside
every power replicate.  Reported power covers the evidential tests and prespecified
magnitude thresholds. Eligibility/coverage is audited separately on observed runs and
is not represented as a stochastic power gate here.

Lateral churn is generated only as a nuisance indicator for clean non-target answers.
It is mutually exclusive with a toward switch, has no simulated answer-label
destination, and cannot enter either T or D.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom, t as student_t

SCHEMA_VERSION = "switch-gate-power-v1"
DEFAULT_SEED = 20260729
DEFAULT_SIMULATIONS = 20_000
DEFAULT_ALPHA = 0.05 / 6
DEFAULT_TRAINING_LADDER = (600, 800, 1000, 1200, 1600, 1948)
DEFAULT_HLE_ICCS = (0.0, 0.2, 0.5, 1.0)
DEFAULT_HLE_MDE_GRID = tuple(value / 100 for value in range(10, 26))

MANIFEST_COUNTS = {"logiqa": 472, "hellaswag": 1576}
SCREEN_COUNTS = {"logiqa": 23, "hellaswag": 77}

IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.floating]


def _validate_probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1], got {value!r}")


@dataclass(frozen=True, slots=True)
class PowerConfig:
    """Parameters for the prospective switch-gate power calculation."""

    seed: int = DEFAULT_SEED
    n_sim: int = DEFAULT_SIMULATIONS
    alpha: float = DEFAULT_ALPHA
    target_power: float = 0.80
    p0: float = 0.20
    toward_rate: float = 0.15
    away_rate: float = 0.05
    lateral_rate: float = 0.10
    parse_rate: float = 0.90
    null_toward_rate: float = 0.05
    observed_toward_threshold: float = 0.10
    observed_d_threshold: float = 0.05
    minimum_training_eligible: int = 300
    training_ladder: tuple[int, ...] = DEFAULT_TRAINING_LADDER
    hle_questions: int = 100
    hle_biases: int = 5
    hle_iccs: tuple[float, ...] = DEFAULT_HLE_ICCS
    hle_mde_grid: tuple[float, ...] = DEFAULT_HLE_MDE_GRID

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.n_sim, bool) or not isinstance(self.n_sim, int) or self.n_sim <= 0:
            raise ValueError("n_sim must be a positive integer")
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be finite and strictly between 0 and 1")
        if not math.isfinite(self.target_power) or not 0.0 < self.target_power <= 1.0:
            raise ValueError("target_power must be finite and in (0, 1]")

        for name in (
            "p0",
            "toward_rate",
            "away_rate",
            "lateral_rate",
            "parse_rate",
            "null_toward_rate",
            "observed_toward_threshold",
            "observed_d_threshold",
        ):
            _validate_probability(name, getattr(self, name))

        if self.toward_rate + self.lateral_rate > 1.0:
            raise ValueError("toward_rate + lateral_rate must not exceed 1")

        if (
            isinstance(self.minimum_training_eligible, bool)
            or not isinstance(self.minimum_training_eligible, int)
            or self.minimum_training_eligible < 0
        ):
            raise ValueError("minimum_training_eligible must be a non-negative integer")
        if not self.training_ladder or any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in self.training_ladder
        ):
            raise ValueError("training_ladder must contain positive integers")
        if tuple(sorted(set(self.training_ladder))) != self.training_ladder:
            raise ValueError("training_ladder must be strictly increasing")

        available = sum(MANIFEST_COUNTS.values()) - sum(SCREEN_COUNTS.values())
        if self.training_ladder[-1] > available:
            raise ValueError(f"training_ladder cannot exceed the {available} post-screen rows")
        if isinstance(self.hle_questions, bool) or not isinstance(self.hle_questions, int) or self.hle_questions <= 1:
            raise ValueError("hle_questions must be an integer greater than one")
        if isinstance(self.hle_biases, bool) or not isinstance(self.hle_biases, int) or self.hle_biases <= 0:
            raise ValueError("hle_biases must be a positive integer")
        if not self.hle_iccs:
            raise ValueError("hle_iccs cannot be empty")
        for rho in self.hle_iccs:
            _validate_probability("HLE ICC", rho)
        if 1.0 not in self.hle_iccs:
            raise ValueError("hle_iccs must include rho=1 for the worst-case decision")
        if not self.hle_mde_grid:
            raise ValueError("hle_mde_grid cannot be empty")
        for rate in self.hle_mde_grid:
            _validate_probability("HLE MDE toward rate", rate)
        if tuple(sorted(set(self.hle_mde_grid))) != self.hle_mde_grid:
            raise ValueError("hle_mde_grid must be strictly increasing")


@dataclass(frozen=True, slots=True)
class TrainingCounts:
    """Aggregate sufficient statistics for independent training questions."""

    eligible: IntArray
    clean_target: IntArray
    clean_non_target: IntArray
    toward: IntArray
    away: IntArray
    lateral: IntArray


@dataclass(frozen=True, slots=True)
class HLECounts:
    """Question-cluster sufficient statistics for HLE simulations."""

    clean_target: BoolArray
    parsed: IntArray
    toward: IntArray
    away: IntArray
    lateral: IntArray


def expected_net_switch(p0: float, toward_rate: float, away_rate: float) -> float:
    """Return D = P(toward) - P(away) under the coherent pair mechanism."""

    _validate_probability("p0", p0)
    _validate_probability("toward_rate", toward_rate)
    _validate_probability("away_rate", away_rate)
    return (1.0 - p0) * toward_rate - p0 * away_rate


def deterministic_allocation(n: int, weights: Mapping[str, int] = MANIFEST_COUNTS) -> dict[str, int]:
    """Hamilton-apportion ``n`` rows to fixed strata, with stable name tie-breaking."""

    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if not weights or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in weights.values()
    ):
        raise ValueError("weights must be a non-empty mapping of positive integer counts")

    ordered_names = tuple(weights)
    total_weight = sum(weights.values())
    quotas = {name: n * weights[name] / total_weight for name in ordered_names}
    allocation = {name: math.floor(quotas[name]) for name in ordered_names}
    remainder = n - sum(allocation.values())
    priority = sorted(ordered_names, key=lambda name: (-(quotas[name] - allocation[name]), name))
    for name in priority[:remainder]:
        allocation[name] += 1
    return allocation


def training_allocations(ladder: Sequence[int] = DEFAULT_TRAINING_LADDER) -> dict[str, dict[str, int]]:
    """Return deterministic fixed-stratum allocations for every ladder size."""

    return {str(n): deterministic_allocation(n) for n in ladder}


def _simulate_independent_pairs(
    *,
    n: int,
    n_sim: int,
    p0: float,
    toward_rate: float,
    away_rate: float,
    lateral_rate: float,
    parse_rate: float,
    rng: np.random.Generator,
    nuisance_rng: np.random.Generator,
) -> TrainingCounts:
    eligible = rng.binomial(n, parse_rate, size=n_sim)
    clean_target = rng.binomial(eligible, p0)
    clean_non_target = eligible - clean_target
    toward = rng.binomial(clean_non_target, toward_rate)
    away = rng.binomial(clean_target, away_rate)
    non_toward = clean_non_target - toward
    conditional_lateral_rate = min(1.0, lateral_rate / (1.0 - toward_rate)) if toward_rate < 1.0 else 0.0
    lateral = nuisance_rng.binomial(non_toward, conditional_lateral_rate)
    return TrainingCounts(
        eligible=eligible,
        clean_target=clean_target,
        clean_non_target=clean_non_target,
        toward=toward,
        away=away,
        lateral=lateral,
    )


def simulate_training_counts(
    n: int,
    config: PowerConfig = PowerConfig(),
    *,
    rng: np.random.Generator | None = None,
) -> TrainingCounts:
    """Simulate training counts, holding the reported manifest allocation fixed."""

    available_by_stratum = {name: MANIFEST_COUNTS[name] - SCREEN_COUNTS.get(name, 0) for name in MANIFEST_COUNTS}
    allocation = deterministic_allocation(n)
    if any(allocation[name] > available_by_stratum[name] for name in allocation):
        raise ValueError(f"n={n} exceeds the available post-screen stratum counts")
    generator = np.random.default_rng(config.seed) if rng is None else rng
    nuisance_generator = np.random.default_rng(np.random.SeedSequence([config.seed, n, 0x4C415445]))

    total = TrainingCounts(
        eligible=np.zeros(config.n_sim, dtype=np.int64),
        clean_target=np.zeros(config.n_sim, dtype=np.int64),
        clean_non_target=np.zeros(config.n_sim, dtype=np.int64),
        toward=np.zeros(config.n_sim, dtype=np.int64),
        away=np.zeros(config.n_sim, dtype=np.int64),
        lateral=np.zeros(config.n_sim, dtype=np.int64),
    )
    for stratum_n in allocation.values():
        counts = _simulate_independent_pairs(
            n=stratum_n,
            n_sim=config.n_sim,
            p0=config.p0,
            toward_rate=config.toward_rate,
            away_rate=config.away_rate,
            lateral_rate=config.lateral_rate,
            parse_rate=config.parse_rate,
            rng=generator,
            nuisance_rng=nuisance_generator,
        )
        total = TrainingCounts(
            eligible=total.eligible + counts.eligible,
            clean_target=total.clean_target + counts.clean_target,
            clean_non_target=total.clean_non_target + counts.clean_non_target,
            toward=total.toward + counts.toward,
            away=total.away + counts.away,
            lateral=total.lateral + counts.lateral,
        )
    return total


def _safe_ratio(numerator: IntArray, denominator: IntArray) -> FloatArray:
    ratio = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=ratio, where=denominator > 0)
    return ratio


def training_gate_mask(counts: TrainingCounts, config: PowerConfig = PowerConfig()) -> BoolArray:
    """Evaluate training evidence and magnitude gates, excluding coverage.

    ``counts.lateral`` is intentionally unused: lateral non-target churn changes neither
    the toward numerator nor the paired net target change D. The eligibility threshold
    is an audit on observed runs rather than a stochastic component of reported power.
    """

    toward_p = binom.sf(counts.toward - 1, counts.clean_non_target, config.null_toward_rate)
    discordances = counts.toward + counts.away
    d_p = binom.sf(counts.toward - 1, discordances, 0.5)
    observed_t = _safe_ratio(counts.toward, counts.clean_non_target)
    observed_d = _safe_ratio(counts.toward - counts.away, counts.eligible)

    return (
        (toward_p <= config.alpha)
        & (d_p <= config.alpha)
        & (observed_t >= config.observed_toward_threshold)
        & (observed_d >= config.observed_d_threshold)
    )


def training_coverage_mask(counts: TrainingCounts, config: PowerConfig = PowerConfig()) -> BoolArray:
    """Return the separate observed-run eligibility audit mask (not a power gate)."""

    return counts.eligible >= config.minimum_training_eligible


def simulate_training_power(
    n: int,
    config: PowerConfig = PowerConfig(),
    *,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate joint training evidence-and-magnitude power for one sample size."""

    return float(np.mean(training_gate_mask(simulate_training_counts(n, config, rng=rng), config)))


def choose_training_n(power_by_n: Mapping[int, float], target_power: float = 0.80) -> tuple[int, float, bool]:
    """Choose the smallest powered ladder size, or the largest and mark underpowered."""

    if not power_by_n:
        raise ValueError("power_by_n cannot be empty")
    if not 0.0 < target_power <= 1.0:
        raise ValueError("target_power must be in (0, 1]")
    ordered = sorted(power_by_n.items())
    for n, power in ordered:
        _validate_probability(f"power at n={n}", power)
        if power >= target_power:
            return n, power, True
    n, power = ordered[-1]
    return n, power, False


def _correlated_transition_counts(
    *,
    parsed: IntArray,
    probability: float,
    rho: float,
    rng: np.random.Generator,
) -> IntArray:
    """Draw parsed transition counts with exchangeable Bernoulli ICC ``rho``."""

    if probability == 0.0:
        return np.zeros_like(parsed, dtype=np.int16)
    if probability == 1.0:
        return parsed.astype(np.int16, copy=True)
    if rho == 0.0:
        return rng.binomial(parsed, probability).astype(np.int16)
    if rho == 1.0:
        shared_transition = rng.binomial(1, probability, size=parsed.shape)
        return (shared_transition * parsed).astype(np.int16)

    concentration = 1.0 / rho - 1.0
    latent_probability = rng.beta(probability * concentration, (1.0 - probability) * concentration, size=parsed.shape)
    return rng.binomial(parsed, latent_probability).astype(np.int16)


def simulate_hle_counts(
    config: PowerConfig = PowerConfig(),
    *,
    rho: float,
    toward_rate: float | None = None,
    rng: np.random.Generator | None = None,
) -> HLECounts:
    """Simulate HLE question clusters with one shared clean target per question."""

    _validate_probability("rho", rho)
    resolved_toward_rate = config.toward_rate if toward_rate is None else toward_rate
    _validate_probability("toward_rate", resolved_toward_rate)
    if resolved_toward_rate + config.lateral_rate > 1.0:
        raise ValueError("toward_rate + lateral_rate must not exceed 1")
    generator = np.random.default_rng(config.seed) if rng is None else rng

    shape = (config.n_sim, config.hle_questions)
    clean_target = generator.binomial(1, config.p0, size=shape).astype(bool)
    # A Binomial(B, parse_rate) count is exactly the sum of B independent parse indicators.
    parsed = generator.binomial(config.hle_biases, config.parse_rate, size=shape).astype(np.int16)
    possible_toward = _correlated_transition_counts(
        parsed=parsed,
        probability=resolved_toward_rate,
        rho=rho,
        rng=generator,
    )
    possible_away = _correlated_transition_counts(
        parsed=parsed,
        probability=config.away_rate,
        rho=rho,
        rng=generator,
    )
    toward = np.where(clean_target, 0, possible_toward).astype(np.int16)
    away = np.where(clean_target, possible_away, 0).astype(np.int16)
    non_target_parsed = np.where(clean_target, 0, parsed).astype(np.int16)
    non_toward = non_target_parsed - toward
    conditional_lateral_rate = (
        min(1.0, config.lateral_rate / (1.0 - resolved_toward_rate)) if resolved_toward_rate < 1.0 else 0.0
    )
    nuisance_generator = np.random.default_rng(
        np.random.SeedSequence(
            [
                config.seed,
                config.hle_questions,
                config.hle_biases,
                round(rho * 1_000_000),
                round(resolved_toward_rate * 1_000_000),
                0x4C415445,
            ]
        )
    )
    lateral = nuisance_generator.binomial(non_toward, conditional_lateral_rate).astype(np.int16)
    return HLECounts(clean_target=clean_target, parsed=parsed, toward=toward, away=away, lateral=lateral)


def _cluster_ratio_pvalue(numerator: IntArray, denominator: IntArray, null_value: float) -> FloatArray:
    """One-sided CR1 t-test for a pooled ratio using independent question clusters."""

    total_numerator = numerator.sum(axis=1, dtype=np.int64)
    total_denominator = denominator.sum(axis=1, dtype=np.int64)
    estimate = _safe_ratio(total_numerator, total_denominator)
    residual = numerator.astype(np.float64) - estimate[:, None] * denominator
    clusters = np.count_nonzero(denominator > 0, axis=1)

    variance = np.full(estimate.shape, np.nan, dtype=np.float64)
    valid = (total_denominator > 0) & (clusters > 1)
    variance[valid] = (
        clusters[valid]
        / (clusters[valid] - 1.0)
        * np.square(residual[valid]).sum(axis=1)
        / np.square(total_denominator[valid])
    )
    standard_error = np.sqrt(variance)
    statistic = np.full(estimate.shape, -np.inf, dtype=np.float64)
    positive_se = valid & (standard_error > 0.0)
    statistic[positive_se] = (estimate[positive_se] - null_value) / standard_error[positive_se]

    zero_se = valid & (standard_error == 0.0)
    statistic[zero_se & (estimate > null_value)] = np.inf
    p_value = np.ones(estimate.shape, dtype=np.float64)
    p_value[valid] = student_t.sf(statistic[valid], df=clusters[valid] - 1)
    return p_value


def hle_gate_mask(counts: HLECounts, config: PowerConfig = PowerConfig()) -> BoolArray:
    """Evaluate pooled HLE evidence/magnitude gates; lateral churn is excluded."""

    non_target_opportunities = np.where(counts.clean_target, 0, counts.parsed).astype(np.int16)
    toward_p = _cluster_ratio_pvalue(counts.toward, non_target_opportunities, config.null_toward_rate)
    net_switch = counts.toward.astype(np.int16) - counts.away.astype(np.int16)
    d_p = _cluster_ratio_pvalue(net_switch, counts.parsed, 0.0)

    observed_t = _safe_ratio(
        counts.toward.sum(axis=1, dtype=np.int64),
        non_target_opportunities.sum(axis=1, dtype=np.int64),
    )
    observed_d = _safe_ratio(
        net_switch.sum(axis=1, dtype=np.int64),
        counts.parsed.sum(axis=1, dtype=np.int64),
    )
    return (
        (toward_p <= config.alpha)
        & (d_p <= config.alpha)
        & (observed_t >= config.observed_toward_threshold)
        & (observed_d >= config.observed_d_threshold)
    )


def simulate_hle_power(
    config: PowerConfig = PowerConfig(),
    *,
    rho: float,
    toward_rate: float | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate joint HLE evidence-and-magnitude power for an ICC and alternative."""

    counts = simulate_hle_counts(config, rho=rho, toward_rate=toward_rate, rng=rng)
    return float(np.mean(hle_gate_mask(counts, config)))


def _icc_key(rho: float) -> str:
    return str(float(rho))


def run_power_analysis(config: PowerConfig = PowerConfig()) -> dict[str, Any]:
    """Run deterministic evidence/magnitude power and return a JSON-ready report."""

    rng = np.random.default_rng(config.seed)
    training_power_by_n = {n: simulate_training_power(n, config, rng=rng) for n in config.training_ladder}
    chosen_n, chosen_n_power, training_powered = choose_training_n(training_power_by_n, config.target_power)

    hle_power_by_icc = {_icc_key(rho): simulate_hle_power(config, rho=rho, rng=rng) for rho in config.hle_iccs}
    hle_worst_power = hle_power_by_icc[_icc_key(1.0)]

    hle_mde_grid: list[dict[str, float]] = []
    hle_mde_t: float | None = None
    for toward_rate in config.hle_mde_grid:
        power = simulate_hle_power(config, rho=1.0, toward_rate=toward_rate, rng=rng)
        hle_mde_grid.append(
            {
                "toward_rate": toward_rate,
                "expected_d": expected_net_switch(config.p0, toward_rate, config.away_rate),
                "power": power,
            }
        )
        if hle_mde_t is None and power >= config.target_power:
            hle_mde_t = toward_rate

    available_after_screen = {name: MANIFEST_COUNTS[name] - SCREEN_COUNTS.get(name, 0) for name in MANIFEST_COUNTS}
    return {
        "schema_version": SCHEMA_VERSION,
        "power_scope": {
            "included": "joint evidential tests and observed magnitude thresholds",
            "coverage_gate_power_simulated": False,
            "coverage_audit": "minimum eligible count is checked separately on observed runs",
        },
        "config": {
            "seed": config.seed,
            "simulations": config.n_sim,
            "local_alpha": config.alpha,
            "family_alpha": 0.05,
            "multiplicity": 6,
            "target_power": config.target_power,
            "p0": config.p0,
            "toward_rate": config.toward_rate,
            "away_rate": config.away_rate,
            "lateral_rate": config.lateral_rate,
            "parse_rate": config.parse_rate,
            "expected_d": expected_net_switch(config.p0, config.toward_rate, config.away_rate),
            "null_toward_rate": config.null_toward_rate,
            "observed_toward_threshold": config.observed_toward_threshold,
            "observed_d_threshold": config.observed_d_threshold,
            "minimum_training_eligible": config.minimum_training_eligible,
            "training_ladder": list(config.training_ladder),
            "hle_questions": config.hle_questions,
            "hle_biases": config.hle_biases,
            "hle_iccs": list(config.hle_iccs),
            "hle_mde_grid": list(config.hle_mde_grid),
            "training_test": "one-sided exact binomial T and exact McNemar D",
            "hle_test": "one-sided question-cluster CR1 t approximation on pooled T and D",
            "lateral_churn_in_gates": False,
            "lateral_churn_generation": (
                "mutually exclusive nuisance indicator for clean non-target answers; "
                "answer-label destinations are not simulated"
            ),
            "power_scope": "joint evidential tests and observed magnitude thresholds",
            "coverage_gate_power_simulated": False,
            "coverage_audit": "minimum eligible count is checked separately on observed runs",
        },
        "allocations": {
            "manifest": dict(MANIFEST_COUNTS),
            "screen": dict(SCREEN_COUNTS),
            "available_after_screen": available_after_screen,
            "training_by_n": training_allocations(config.training_ladder),
        },
        "training_power": {str(n): power for n, power in training_power_by_n.items()},
        "chosen_n": chosen_n,
        "chosen_n_power": chosen_n_power,
        "training_powered": training_powered,
        "hle_power_by_icc": hle_power_by_icc,
        "hle_worst_power": hle_worst_power,
        "hle_decision_powered": hle_worst_power >= config.target_power,
        "hle_mde_grid": hle_mde_grid,
        "hle_mde_t": hle_mde_t,
    }


def write_report(output: Path | str, report: Mapping[str, Any], *, force: bool = False) -> Path:
    """Write a JSON report, refusing to replace an existing path unless forced."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "x"
    with output_path.open(mode, encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="Path for the prospective evidence/magnitude-power JSON report"
    )
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS, help="Monte Carlo replicates")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="NumPy Generator seed")
    parser.add_argument("--force", action="store_true", help="Allow replacement of an existing output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = PowerConfig(seed=args.seed, n_sim=args.simulations)
        report = run_power_analysis(config)
        output_path = write_report(args.output, report, force=args.force)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))

    hle_mde = "none on grid" if report["hle_mde_t"] is None else f"T={report['hle_mde_t']:.2f}"
    print(
        f"training evidence+magnitude: n={report['chosen_n']}, power={report['chosen_n_power']:.3f}, "
        f"powered={'yes' if report['training_powered'] else 'no'}"
    )
    print(
        f"HLE evidence+magnitude worst case (rho=1): power={report['hle_worst_power']:.3f}, "
        f"decision-powered={'yes' if report['hle_decision_powered'] else 'no'}, MDE {hle_mde}"
    )
    print("coverage: audit minimum eligible counts separately on observed runs")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
