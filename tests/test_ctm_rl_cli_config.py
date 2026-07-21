from ctm.core.config import resolve_lora_config


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
