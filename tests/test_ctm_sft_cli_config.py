import pytest

from scripts.train_bct import resolve_lora_config, resolve_optimizer_config


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
