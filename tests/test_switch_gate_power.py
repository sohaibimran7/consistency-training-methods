import json

import numpy as np
import pytest

from experiments.switch_gate.power import (
    DEFAULT_TRAINING_LADDER,
    PowerConfig,
    TrainingCounts,
    choose_training_n,
    deterministic_allocation,
    expected_net_switch,
    main,
    run_power_analysis,
    simulate_hle_counts,
    simulate_training_counts,
    training_coverage_mask,
    training_gate_mask,
    write_report,
)


def _small_config(**overrides):
    values = {
        "n_sim": 64,
        "training_ladder": (600, 800),
        "hle_iccs": (0.0, 1.0),
        "hle_mde_grid": (0.10, 0.20),
    }
    values.update(overrides)
    return PowerConfig(**values)


def test_power_report_is_reproducible():
    config = _small_config()

    assert run_power_analysis(config) == run_power_analysis(config)


def test_expected_d_uses_coherent_toward_and_away_identity():
    assert expected_net_switch(p0=0.20, toward_rate=0.15, away_rate=0.05) == pytest.approx(0.11)


def test_lateral_churn_and_coverage_are_excluded_from_power_gate():
    counts = TrainingCounts(
        eligible=np.array([299]),
        clean_target=np.array([49]),
        clean_non_target=np.array([250]),
        toward=np.array([50]),
        away=np.array([0]),
        lateral=np.array([100]),
    )

    assert training_gate_mask(counts).item()
    assert not training_coverage_mask(counts).item()

    without_lateral = TrainingCounts(
        eligible=counts.eligible,
        clean_target=counts.clean_target,
        clean_non_target=counts.clean_non_target,
        toward=counts.toward,
        away=counts.away,
        lateral=np.array([0]),
    )
    assert np.array_equal(training_gate_mask(counts), training_gate_mask(without_lateral))


def test_lateral_rate_cannot_perturb_t_or_d_draws():
    no_lateral = _small_config(lateral_rate=0.0)
    with_lateral = _small_config(lateral_rate=0.10)

    training_without = simulate_training_counts(600, no_lateral)
    training_with = simulate_training_counts(600, with_lateral)
    for field in ("eligible", "clean_target", "clean_non_target", "toward", "away"):
        assert np.array_equal(getattr(training_without, field), getattr(training_with, field))
    assert training_without.lateral.sum() == 0
    assert training_with.lateral.sum() / training_with.clean_non_target.sum() == pytest.approx(0.10, abs=0.01)

    hle_without = simulate_hle_counts(no_lateral, rho=0.5)
    hle_with = simulate_hle_counts(with_lateral, rho=0.5)
    for field in ("clean_target", "parsed", "toward", "away"):
        assert np.array_equal(getattr(hle_without, field), getattr(hle_with, field))


def test_manifest_allocations_are_deterministic_and_exhaust_n():
    expected = {
        600: {"logiqa": 138, "hellaswag": 462},
        800: {"logiqa": 184, "hellaswag": 616},
        1000: {"logiqa": 230, "hellaswag": 770},
        1200: {"logiqa": 277, "hellaswag": 923},
        1600: {"logiqa": 369, "hellaswag": 1231},
        1948: {"logiqa": 449, "hellaswag": 1499},
    }

    assert {n: deterministic_allocation(n) for n in DEFAULT_TRAINING_LADDER} == expected
    assert all(sum(allocation.values()) == n for n, allocation in expected.items())


def test_ladder_choice_selects_first_powered_n_or_flags_maximum():
    assert choose_training_n({600: 0.79, 800: 0.80, 1000: 0.91}) == (800, 0.80, True)
    assert choose_training_n({600: 0.40, 1948: 0.79}) == (1948, 0.79, False)


def test_rho_one_shares_each_transition_and_drives_worst_case_output():
    config = _small_config(n_sim=128, hle_questions=20)
    counts = simulate_hle_counts(config, rho=1.0)
    non_target = ~counts.clean_target
    target = counts.clean_target

    assert np.all(counts.away[non_target] == 0)
    assert np.all(counts.toward[target] == 0)
    assert np.all(counts.lateral[target] == 0)
    assert np.all((counts.toward[non_target] == 0) | (counts.toward[non_target] == counts.parsed[non_target]))
    assert np.all((counts.away[target] == 0) | (counts.away[target] == counts.parsed[target]))
    assert np.all(counts.toward[non_target] + counts.lateral[non_target] <= counts.parsed[non_target])

    report = run_power_analysis(config)
    assert set(report["hle_power_by_icc"]) == {"0.0", "1.0"}
    assert report["hle_worst_power"] == report["hle_power_by_icc"]["1.0"]
    assert report["hle_decision_powered"] == (report["hle_worst_power"] >= config.target_power)
    assert report["config"]["lateral_rate"] == 0.10
    assert not report["config"]["lateral_churn_in_gates"]
    assert not report["power_scope"]["coverage_gate_power_simulated"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_sim": 0}, "n_sim"),
        ({"parse_rate": 1.01}, "parse_rate"),
        ({"lateral_rate": 1.01}, "lateral_rate"),
        ({"toward_rate": 0.95, "lateral_rate": 0.10}, "must not exceed"),
        ({"hle_iccs": (0.0, 0.2)}, "rho=1"),
        ({"training_ladder": (800, 600)}, "strictly increasing"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PowerConfig(**kwargs)


def test_cli_writes_schema_and_write_refuses_overwrite(tmp_path, capsys):
    output = tmp_path / "power.json"

    assert main(["--output", str(output), "--simulations", "16"]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "switch-gate-power-v1"
    assert "training evidence+magnitude:" in capsys.readouterr().out

    with pytest.raises(FileExistsError):
        write_report(output, {"replacement": True})
    write_report(output, {"replacement": True}, force=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {"replacement": True}
