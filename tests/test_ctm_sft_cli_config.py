import json
import sys

import pytest

from scripts import train_bct
from scripts.train_bct import resolve_lora_config, resolve_optimizer_config


def test_dry_run_loads_bct_rows_without_initializing_backend(monkeypatch, tmp_path, capsys):
    data = tmp_path / "bct.jsonl"
    data.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Question?"},
                    {"role": "assistant", "content": "Answer."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_bct, "build_backend", lambda *_args, **_kwargs: pytest.fail("backend initialized"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_bct.py",
            "--backend",
            "local",
            "--model",
            "test-model",
            "--method",
            "bct",
            "--data",
            str(data),
            "--dry-run",
        ],
    )

    train_bct.main()

    assert "Dry run complete; no backend was initialized." in capsys.readouterr().out


def test_consistency_methods_reject_identical_variant_and_reference_fields(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_bct.py",
            "--backend",
            "local",
            "--model",
            "test-model",
            "--method",
            "act",
            "--data",
            "rows.jsonl",
            "--reference-messages-field",
            "unbiased_messages",
            "--variant-messages-field",
            "unbiased_messages",
        ],
    )
    with pytest.raises(SystemExit):
        train_bct.main()
    assert "identically zero" in capsys.readouterr().err


def test_nested_training_configs_expose_portable_backend_parameters():
    lora = resolve_lora_config(
        {
            "rank": 16,
            "train_mlp": True,
            "train_attn": False,
            "train_unembed": False,
            "seed": 7,
        },
        rank=None,
        seed=None,
    )
    optimizer = resolve_optimizer_config(
        {
            "learning_rate": 2e-4,
            "lr_schedule": "cosine",
            "beta1": 0.8,
            "beta2": 0.9,
            "eps": 1e-7,
            "weight_decay": 0.1,
            "grad_clip_norm": 0.5,
        },
        learning_rate=None,
        lr_schedule=None,
    )

    assert lora.model_dump() == {
        "rank": 16,
        "alpha": None,
        "dropout": 0.0,
        "target_modules": None,
        "train_mlp": True,
        "train_attn": False,
        "train_unembed": False,
        "seed": 7,
    }
    assert optimizer.model_dump() == {
        "learning_rate": 2e-4,
        "lr_schedule": "cosine",
        "beta1": 0.8,
        "beta2": 0.9,
        "eps": 1e-7,
        "weight_decay": 0.1,
        "grad_clip_norm": 0.5,
    }


def test_exact_lora_parameters_are_preserved():
    lora = resolve_lora_config(
        {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        rank=None,
        seed=None,
    )

    assert lora.resolved_alpha == 16
    assert lora.dropout == 0.05
    assert lora.target_modules == ["q_proj", "v_proj"]


def test_scalar_training_flags_override_nested_values():
    lora = resolve_lora_config({"rank": 16, "seed": 1}, rank=4, seed=2)
    optimizer = resolve_optimizer_config(
        {"learning_rate": 2e-4, "lr_schedule": "cosine"},
        learning_rate=1e-4,
        lr_schedule="linear",
    )

    assert lora.rank == 4
    assert lora.seed == 2
    assert optimizer.learning_rate == 1e-4
    assert optimizer.lr_schedule == "linear"


@pytest.mark.parametrize(
    ("resolver", "raw", "message"),
    [
        (lambda raw: resolve_lora_config(raw, rank=None, seed=None), {"rnak": 8}, "unknown"),
        (
            lambda raw: resolve_optimizer_config(raw, learning_rate=None, lr_schedule=None),
            {"bet1": 0.9},
            "unknown",
        ),
    ],
)
def test_nested_training_configs_reject_typos(resolver, raw, message):
    with pytest.raises(ValueError, match=message):
        resolver(raw)
