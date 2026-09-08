import argparse
from argparse import Namespace
from types import SimpleNamespace

import pytest

from ctm.core.config import resolve_lora_config
from ctm.training.rl import RLConfig, TrainingLoopConfig
from scripts import train_rlct
from scripts.train_rlct import (
    _adam_config_from_args,
    _add_adam_optimizer_args,
    _add_kl_discount_factor_arg,
    _validate_numeric_args,
)


def test_rl_cli_can_select_explicit_eos_only_generation(monkeypatch):
    captured = []
    original_generation = train_rlct.GenerationConfig

    def capture_generation(**kwargs):
        captured.append(kwargs)
        return original_generation(**kwargs)

    prepared = SimpleNamespace(
        datapoints=[],
        perturbations=[],
        training_indices=[1],
        setting=SimpleNamespace(name="unit-setting"),
        trait_classifier=None,
        answer_parser=None,
    )
    monkeypatch.setattr(train_rlct, "prepare_setting", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(train_rlct, "setting_run_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train_rlct, "GenerationConfig", capture_generation)

    train_rlct.main(
        [
            "--setting-factory",
            "unit:factory",
            "--experiment-name",
            "unit",
            "--run-name",
            "eos-only",
            "--no-max-new-tokens",
            "--dry-run",
        ]
    )

    assert captured[-1]["max_new_tokens"] is None


def test_rl_lora_config_can_match_rmct_module_selection():
    config = resolve_lora_config(
        {
            "rank": 8,
            "alpha": 16,
            "train_mlp": True,
            "train_attn": True,
            "train_unembed": False,
        }
    )

    assert config.rank == 8
    assert config.resolved_alpha == 16
    assert config.train_mlp is True
    assert config.train_attn is True
    assert config.train_unembed is False


def test_rl_scalar_flags_override_nested_lora_values():
    config = resolve_lora_config({"rank": 16, "seed": 1}, rank=4, seed=2)

    assert config.rank == 4
    assert config.seed == 2


def test_rl_datapoint_order_cli_defaults_to_epoch_shuffle_and_can_be_disabled(monkeypatch):
    captured: list[bool] = []

    original_loop = train_rlct.TrainingLoopConfig

    def capture_loop(**kwargs):
        captured.append(kwargs["shuffle_datapoints"])
        return original_loop(**kwargs)

    prepared = SimpleNamespace(
        datapoints=[],
        perturbations=[],
        training_indices=[1],
        setting=SimpleNamespace(name="unit-setting"),
        trait_classifier=None,
        answer_parser=None,
    )
    monkeypatch.setattr(train_rlct, "prepare_setting", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(train_rlct, "setting_run_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train_rlct, "TrainingLoopConfig", capture_loop)

    common = [
        "--setting-factory",
        "unit:factory",
        "--experiment-name",
        "unit",
        "--run-name",
        "datapoint-order",
        "--dry-run",
    ]
    train_rlct.main(common)
    train_rlct.main([*common, "--no-shuffle-datapoints"])

    assert captured == [True, False]
    assert TrainingLoopConfig().shuffle_datapoints is True


def _numeric_args(
    *,
    anchor_weight: float,
    n_ref_rollouts: int = 96,
    n_anchor_rollouts: int = 96,
) -> Namespace:
    return Namespace(
        batch_size=1,
        gradient_accumulation_steps=4,
        refresh_every=1,
        n_epochs=1,
        checkpoint_every=8,
        lora_rank=8,
        n_ref_rollouts=n_ref_rollouts,
        n_train_rollouts=96,
        max_new_tokens=20480,
        max_resample_attempts=1,
        n_consistency_rollouts=96,
        n_anchor_rollouts=n_anchor_rollouts,
        lr=1e-4,
        lr_schedule="linear",
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        temperature=1.0,
        kl_coef=0.05,
        kl_discount_factor=0.0,
        anchor_weight=anchor_weight,
        snr_z=2.0,
    )


def test_rl_allows_inert_anchor_selection_to_exceed_reference_pool():
    _validate_numeric_args(_numeric_args(anchor_weight=0.0))


def test_rl_rejects_active_anchor_selection_larger_than_reference_pool():
    with pytest.raises(ValueError, match="--n-anchor-rollouts cannot exceed --n-ref-rollouts"):
        _validate_numeric_args(
            _numeric_args(anchor_weight=0.5, n_ref_rollouts=96, n_anchor_rollouts=128)
        )


def test_rl_adam_cli_flags_preserve_existing_defaults_and_build_explicit_config():
    parser = argparse.ArgumentParser()
    _add_adam_optimizer_args(parser)

    parsed = parser.parse_args([])
    assert vars(parsed) == {
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
    }

    config = _adam_config_from_args(
        Namespace(lr=1e-4, lr_schedule="linear", **vars(parsed))
    )
    assert config.model_dump() == {
        "learning_rate": 1e-4,
        "lr_schedule": "linear",
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
    }


def test_rl_adam_cli_flags_propagate_explicit_optimizer_values():
    parser = argparse.ArgumentParser()
    _add_adam_optimizer_args(parser)
    parsed = parser.parse_args(
        [
            "--beta1",
            "0.8",
            "--beta2",
            "0.9",
            "--eps",
            "1e-7",
            "--weight-decay",
            "0.1",
            "--grad-clip-norm",
            "0.5",
        ]
    )

    config = _adam_config_from_args(
        Namespace(lr=2e-4, lr_schedule="cosine", **vars(parsed))
    )
    assert config.model_dump() == {
        "learning_rate": 2e-4,
        "lr_schedule": "cosine",
        "beta1": 0.8,
        "beta2": 0.9,
        "eps": 1e-7,
        "weight_decay": 0.1,
        "grad_clip_norm": 0.5,
    }


def test_rl_kl_discount_cli_flag_preserves_the_token_local_default_and_config_wiring():
    parser = argparse.ArgumentParser()
    _add_kl_discount_factor_arg(parser)

    assert parser.parse_args([]).kl_discount_factor == 0.0
    parsed = parser.parse_args(["--kl-discount-factor", "0.6"])
    assert RLConfig(kl_discount_factor=parsed.kl_discount_factor).kl_discount_factor == 0.6


def test_rl_cli_wires_frozen_kl_and_local_ppo_values_into_runtime_config(monkeypatch):
    captured = {}

    class FakeTrainer:
        def __init__(self, *, config, backend, **_kwargs):
            captured["config"] = config
            captured["backend"] = backend

        def setup(self):
            return None

        async def train(self, **_kwargs):
            return "unit-checkpoint"

    prepared = SimpleNamespace(
        datapoints=[],
        perturbations=[],
        training_indices=[1],
        setting=SimpleNamespace(name="unit-setting"),
        trait_classifier=None,
        answer_parser=None,
    )
    backend = object()
    monkeypatch.setattr(train_rlct, "prepare_setting", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(train_rlct, "setting_run_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train_rlct, "build_backend", lambda _args: backend)
    monkeypatch.setattr(train_rlct, "RLTrainer", FakeTrainer)

    train_rlct.main(
        [
            "--setting-factory",
            "unit:factory",
            "--experiment-name",
            "unit",
            "--run-name",
            "kl-ppo-wiring",
            "--backend",
            "local",
            "--local-device",
            "cpu",
            "--local-sampler",
            "hf",
            "--kl-discount-factor",
            "0.0",
            "--local-ppo-clip-epsilon",
            "0.2",
            "--yes",
        ]
    )

    assert captured["backend"] is backend
    assert captured["config"].kl_discount_factor == 0.0
    assert captured["config"].run_metadata["local_backend"] == {"ppo_clip_epsilon": 0.2}


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_rl_config_rejects_invalid_kl_discount_factors(value):
    with pytest.raises(ValueError, match="kl_discount_factor must be a finite number in \\[0, 1\\]"):
        RLConfig(kl_discount_factor=value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("beta1", -0.01, "--beta1 must be finite and in \\[0, 1\\)"),
        ("beta1", 1.0, "--beta1 must be finite and in \\[0, 1\\)"),
        ("beta1", float("nan"), "--beta1 must be finite and in \\[0, 1\\)"),
        ("beta2", float("inf"), "--beta2 must be finite and in \\[0, 1\\)"),
        ("eps", 0.0, "--eps must be finite and positive"),
        ("eps", float("nan"), "--eps must be finite and positive"),
        ("weight_decay", -0.01, "--weight-decay must be finite and non-negative"),
        ("weight_decay", float("inf"), "--weight-decay must be finite and non-negative"),
        ("grad_clip_norm", -0.01, "--grad-clip-norm must be finite and non-negative"),
        ("grad_clip_norm", float("nan"), "--grad-clip-norm must be finite and non-negative"),
        ("kl_discount_factor", -0.01, "--kl-discount-factor must be finite and in \\[0, 1\\]"),
        ("kl_discount_factor", 1.01, "--kl-discount-factor must be finite and in \\[0, 1\\]"),
        ("kl_discount_factor", float("nan"), "--kl-discount-factor must be finite and in \\[0, 1\\]"),
    ],
)
def test_rl_rejects_invalid_optimizer_and_kl_cli_values(field, value, message):
    args = _numeric_args(anchor_weight=0.5)
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        _validate_numeric_args(args)
