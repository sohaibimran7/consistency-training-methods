"""CPU-only contracts for the non-production phase-shared Qwen3.5 harness."""

from __future__ import annotations

import asyncio
import ast
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "infra" / "vastai" / "preflight_qwen35_phase_shared.py"
WRAPPER = ROOT / "infra" / "vastai" / "preflight_qwen35_phase_shared.sh"
COMPAT = ROOT / "ctm" / "backends" / "local" / "qwen35_vllm_compat.py"


def _module():
    spec = importlib.util.spec_from_file_location("qwen35_phase_shared_preflight_test_module", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclass inspects sys.modules while evaluating postponed annotations.
    # Normal imports install this automatically; direct test loading must do
    # the equivalent before exec_module().
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _compat_module():
    spec = importlib.util.spec_from_file_location("qwen35_vllm_compat_shape_test_module", COMPAT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(module, output: Path, *extra: str):
    return module.parse_args(["--output-dir", str(output), *extra])


def _write_safetensors_stub(path: Path, tensor_names: list[str]) -> None:
    """Write a header-valid tiny safetensors file for cache-readiness tests."""

    header = {
        name: {"dtype": "F32", "shape": [1], "data_offsets": [index * 4, (index + 1) * 4]}
        for index, name in enumerate(tensor_names)
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, byteorder="little") + encoded + b"\0" * (4 * len(tensor_names)))


def test_pinned_hf_snapshot_prefetch_is_single_canonical_and_validates_all_indexed_shards(tmp_path):
    module = _module()
    cache_root = tmp_path / "hf-cache" / "models--Qwen--Qwen3.5-9B"
    snapshot = cache_root / "snapshots" / module.MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"qwen3_5"}\n', encoding="utf-8")
    (snapshot / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    _write_safetensors_stub(snapshot / first, ["layer.0.weight", "layer.1.weight"])
    _write_safetensors_stub(snapshot / second, ["lm_head.weight"])
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "layer.0.weight": first,
                    "layer.1.weight": first,
                    "lm_head.weight": second,
                },
            }
        ),
        encoding="utf-8",
    )

    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    class Config:
        model_type = "qwen3_5"

    class Tokenizer:
        pass

    class AutoConfig:
        @staticmethod
        def from_pretrained(path, *, local_files_only):
            assert path == str(snapshot.resolve())
            assert local_files_only is True
            return Config()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, *, local_files_only):
            assert path == str(snapshot.resolve())
            assert local_files_only is True
            return Tokenizer()

    receipt = module._prepare_pinned_hf_snapshot(
        repo_id=module.MODEL,
        revision=module.MODEL_REVISION,
        AutoConfig=AutoConfig,
        AutoTokenizer=AutoTokenizer,
        snapshot_download=snapshot_download,
    )

    assert calls == [
        {
            "repo_id": module.MODEL,
            "revision": module.MODEL_REVISION,
            "local_files_only": False,
        }
    ]
    assert receipt["repo_id"] == module.MODEL
    assert receipt["requested_revision"] == receipt["resolved_commit"] == module.MODEL_REVISION
    assert receipt["resolved_snapshot_path"] == str(snapshot.resolve())
    assert receipt["runtime_model_argument"] == str(snapshot.resolve())
    assert receipt["safetensors_index"]["indexed_tensor_count"] == 3
    assert receipt["safetensors_index"]["indexed_shard_count"] == 2
    assert [entry["relative_path"] for entry in receipt["safetensors_index"]["indexed_shards"]] == [first, second]
    assert [entry["indexed_tensor_count"] for entry in receipt["safetensors_index"]["indexed_shards"]] == [2, 1]
    assert all(entry["sha256"] for entry in receipt["safetensors_index"]["indexed_shards"])
    assert receipt["local_consumer_validation"] == {
        "config_class": "Config",
        "tokenizer_class": "Tokenizer",
        "config_model_type": "qwen3_5",
        "local_files_only": True,
    }


def test_pinned_hf_snapshot_rejects_an_indexed_shard_missing_its_tensor_header(tmp_path):
    module = _module()
    snapshot = tmp_path / "hf-cache" / "models--Qwen--Qwen3.5-9B" / "snapshots" / module.MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    shard = "model-00001-of-00001.safetensors"
    _write_safetensors_stub(snapshot / shard, ["present.weight"])
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"missing.weight": shard}}), encoding="utf-8"
    )

    class AutoConfig:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise AssertionError("consumer validation must not run after shard validation fails")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise AssertionError("consumer validation must not run after shard validation fails")

    with pytest.raises(RuntimeError, match="missing 1 indexed tensor keys"):
        module._prepare_pinned_hf_snapshot(
            repo_id=module.MODEL,
            revision=module.MODEL_REVISION,
            AutoConfig=AutoConfig,
            AutoTokenizer=AutoTokenizer,
            snapshot_download=lambda **_kwargs: str(snapshot),
        )


@pytest.mark.parametrize(
    ("snapshot_path", "message"),
    [
        ("wrong-commit", "does not match the requested immutable Qwen commit"),
        ("outside-canonical-cache", "canonical cache snapshot location"),
    ],
)
def test_pinned_hf_snapshot_rejects_a_wrong_commit_or_noncanonical_cache_path(tmp_path, snapshot_path, message):
    module = _module()
    if snapshot_path == "wrong-commit":
        snapshot = tmp_path / "hf-cache" / "models--Qwen--Qwen3.5-9B" / "snapshots" / ("0" * len(module.MODEL_REVISION))
    else:
        snapshot = tmp_path / "not-the-hf-cache" / module.MODEL_REVISION
    snapshot.mkdir(parents=True)

    class UnexpectedConsumer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise AssertionError("consumer validation must not run for an invalid snapshot path")

    with pytest.raises(RuntimeError, match=message):
        module._prepare_pinned_hf_snapshot(
            repo_id=module.MODEL,
            revision=module.MODEL_REVISION,
            AutoConfig=UnexpectedConsumer,
            AutoTokenizer=UnexpectedConsumer,
            snapshot_download=lambda **_kwargs: str(snapshot),
        )


def test_runtime_model_identity_keeps_canonical_repo_separate_from_runtime_path():
    module = _module()
    runtime_path = "/scratch/hf/models--Qwen--Qwen3.5-9B/snapshots/" + module.MODEL_REVISION

    identity = module._runtime_model_identity(
        canonical_repo_id=module.MODEL,
        snapshot_readiness={
            "requested_revision": module.MODEL_REVISION,
            "resolved_commit": module.MODEL_REVISION,
            "runtime_model_argument": runtime_path,
        },
    )

    assert identity["canonical_repo_id"] == module.MODEL
    assert identity["runtime_model_argument"] == runtime_path
    assert identity["worker_attestation_expected_model"] == runtime_path
    assert identity["worker_attestation_portability"] == "runtime_path_bound_nonportable_topology_generic"


def test_run_preflight_forwards_the_truthful_local_runtime_model_to_every_model_consumer():
    """Guard the exact arguments whose drift would invalidate the attestation.

    The test reads syntax rather than constructing CUDA workers: it verifies
    the actual preflight control flow forwards one local snapshot string to
    the tokenizer, one-rank reference, replicated/outer worker setup, writer,
    and its exact-model attestation revalidation.
    """

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_preflight"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]

    def has_keyword_call(function_name: str, keyword: str) -> bool:
        return any(
            isinstance(call.func, ast.Name)
            and call.func.id == function_name
            and any(
                item.arg == keyword and isinstance(item.value, ast.Name) and item.value.id == "runtime_model"
                for item in call.keywords
            )
            for call in calls
        )

    assert any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "from_pretrained"
        and call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "runtime_model"
        for call in calls
    )
    assert has_keyword_call("_single_rank_reference_update", "runtime_model")
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "backend"
        and call.func.attr == "setup"
        and any(
            item.arg == "model" and isinstance(item.value, ast.Name) and item.value.id == "runtime_model"
            for item in call.keywords
        )
        for call in calls
    )
    assert has_keyword_call("write_qwen35_rollout_worker_parity_attestation", "model")
    assert has_keyword_call("validate_qwen35_rollout_worker_parity_attestation", "expected_model")


@pytest.mark.parametrize("gpu_count", [2, 4, 8])
def test_preflight_contract_supports_arbitrary_2_4_and_8_gpu_phase_shared_allocations(tmp_path, gpu_count: int):
    module = _module()
    args = _args(module, tmp_path / f"out-{gpu_count}")
    contract = module.resolve_preflight_contract(
        args,
        cuda_visible_devices=",".join(f"GPU-{index}" for index in range(gpu_count)),
    )

    assert contract.world_size == gpu_count
    assert contract.coordinator_device == "cuda:0"
    assert [gpu.logical_index for gpu in contract.topology.train_gpus] == list(range(gpu_count))
    assert [gpu.logical_index for gpu in contract.topology.rollout_gpus] == list(range(gpu_count))
    sweep = {entry["world_size"]: entry["status"] for entry in contract.topology_contract_sweep}
    for size in (2, 4, 8):
        expected = (
            "resolved_against_explicit_cuda_visible_devices"
            if size <= gpu_count
            else "not_available_in_this_allocation"
        )
        assert sweep[size] == expected


def test_preflight_contract_preserves_explicit_logical_gpu_order_and_non_numeric_device_tokens(tmp_path):
    module = _module()
    args = _args(
        module,
        tmp_path / "out",
        "--training-gpus",
        "3,1",
        "--rollout-gpus",
        "3,1",
    )
    contract = module.resolve_preflight_contract(
        args,
        cuda_visible_devices="GPU-a,GPU-b,GPU-c,GPU-d",
    )

    assert contract.coordinator_device == "cuda:3"
    assert [(gpu.logical_index, gpu.device_token) for gpu in contract.topology.train_gpus] == [
        (3, "GPU-d"),
        (1, "GPU-b"),
    ]
    assert contract.as_dict()["rollout_gpus"] == [
        {"logical_index": 3, "device_token": "GPU-d"},
        {"logical_index": 1, "device_token": "GPU-b"},
    ]


def test_preflight_requires_identical_trainer_and_rollout_gpu_sets(tmp_path):
    module = _module()
    args = _args(
        module,
        tmp_path / "out",
        "--training-gpus",
        "0,1",
        "--rollout-gpus",
        "1,0",
    )

    with pytest.raises(ValueError, match="identical ordered"):
        module.resolve_preflight_contract(args, cuda_visible_devices="GPU-a,GPU-b")


def test_preflight_requires_at_least_two_real_trainer_gpus(tmp_path):
    module = _module()
    args = _args(module, tmp_path / "out", "--training-gpus", "0")

    with pytest.raises(ValueError, match="at least two real trainer GPUs"):
        module.resolve_preflight_contract(args, cuda_visible_devices="GPU-a,GPU-b")


def test_preflight_can_require_an_eight_gpu_topology_contract_sweep(tmp_path):
    module = _module()
    args = _args(module, tmp_path / "out", "--require-full-topology-contract-sweep")

    with pytest.raises(ValueError, match=r"unavailable=\[4, 8\]"):
        module.resolve_preflight_contract(args, cuda_visible_devices="GPU-a,GPU-b")


def test_preflight_requires_all_three_candidate_packing_budgets(tmp_path):
    module = _module()

    with pytest.raises(SystemExit) as exc_info:
        _args(module, tmp_path / "out", "--packing-budgets", "20480", "40960")

    assert exc_info.value.code == 2


def test_preflight_records_configurable_checkpoint_layer_scope(tmp_path):
    module = _module()
    args = _args(module, tmp_path / "out", "--gradient-checkpoint-layers", "3")
    contract = module.resolve_preflight_contract(args, cuda_visible_devices="GPU-a,GPU-b")

    assert contract.checkpoint_layers == 3
    assert contract.as_dict()["gradient_checkpoint_layers"] == 3


def test_preflight_derives_a_parameter_parity_tolerance_from_the_first_step_learning_rate(tmp_path):
    module = _module()

    args = _args(module, tmp_path / "out", "--learning-rate", "1e-4")

    assert args.parameter_parity_atol == pytest.approx(2.05e-4)
    assert args.parameter_parity_atol_source == "derived_2.05x_learning_rate_first_adamw_step"
    thresholds = module._contract_config(args)["thresholds"]
    assert thresholds["parameter_parity_atol"] == pytest.approx(2.05e-4)
    assert thresholds["parameter_parity_delta_cosine_min"] == pytest.approx(0.9995)
    assert thresholds["parameter_parity_delta_relative_l2_max"] == pytest.approx(0.005)
    assert thresholds["parameter_parity_delta_max_abs"] == pytest.approx(2.5e-5)
    assert thresholds["parameter_parity_delta_max_abs_source"] == "derived_0.25x_learning_rate_first_adamw_step"
    assert thresholds["parameter_parity_envelope_relative_l2_max"] == pytest.approx(0.035)
    assert thresholds["parameter_parity_b_sign_mismatch_max"] == pytest.approx(4e-4)
    assert thresholds["parameter_parity_hard_safety_semantics"] == (
        "exact_common_initial_adapter_finite_nonzero_updates_and_final_coordinate_bound"
    )
    assert thresholds["parameter_parity_monitoring_semantics"] == "non_gating_pending_cross_allocation_calibration"
    assert thresholds["gradient_parity_cosine_min"] == pytest.approx(0.999)
    assert thresholds["gradient_parity_norm_ratio_min"] == pytest.approx(0.95)
    assert thresholds["gradient_parity_norm_ratio_max"] == pytest.approx(1.05)
    assert thresholds["gradient_parity_relative_l2_max"] == pytest.approx(0.05)
    assert thresholds["gradient_parity_magnitude_weighted_sign_mismatch_max"] == pytest.approx(0.01)
    assert thresholds["effect_norm_ratio_min"] == pytest.approx(0.80)
    assert thresholds["effect_norm_ratio_max"] == pytest.approx(1.25)
    assert thresholds["effect_relative_l2_max"] == pytest.approx(0.25)


def test_preflight_preserves_an_explicit_parameter_parity_tolerance(tmp_path):
    module = _module()

    args = _args(
        module,
        tmp_path / "out",
        "--learning-rate",
        "1e-4",
        "--parameter-parity-atol",
        "3e-4",
        "--parameter-parity-delta-max-abs",
        "1e-5",
    )

    assert args.parameter_parity_atol == pytest.approx(3e-4)
    assert args.parameter_parity_atol_source == "explicit"
    assert args.parameter_parity_delta_max_abs == pytest.approx(1e-5)
    assert args.parameter_parity_delta_max_abs_source == "explicit"


def test_reference_microbatch_matches_one_replicated_local_shard(tmp_path):
    module = _module()

    default_args = _args(module, tmp_path / "default")
    one_datum_args = _args(
        module,
        tmp_path / "one-datum",
        "--forward-microbatch-max-datums",
        "1",
        "--parity-datums-per-rank",
        "2",
    )

    assert module._parity_reference_microbatch_max_datums(default_args) == 2
    assert module._parity_reference_microbatch_max_datums(one_datum_args) == 1


def test_update_aligned_probe_uses_repeated_cross_entropy_suffixes_and_records_its_recipe():
    module = _module()

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert "Solve the question carefully" in text
            assert add_special_tokens is True
            return [11, 17, 23]

    class ModelInput:
        def __init__(self, tokens):
            self._tokens = list(tokens)

        @classmethod
        def from_ints(cls, *, tokens):
            return cls(tokens)

        def to_ints(self):
            return list(self._tokens)

    Types = type("Types", (), {"ModelInput": ModelInput})

    tokenizer = Tokenizer()
    prompts, completions, receipt = module._make_update_aligned_probe_candidates(
        tokenizer,
        Types,
        count=4,
        sequence_length=12,
        completion_tokens=5,
    )
    shorter_prompts, shorter_completions, shorter_receipt = module._make_update_aligned_probe_candidates(
        tokenizer,
        Types,
        count=2,
        sequence_length=12,
        completion_tokens=5,
    )

    assert receipt["kind"] == "repeated-cross-entropy-sequence-suffix-score-completions-v1"
    assert receipt["sequence_token_count"] == 12
    assert receipt["prompt_token_count"] == 7
    assert receipt["completion_token_count_per_candidate"] == 5
    assert receipt["candidate_rotation_offsets"] == [0, 1, 2, 3]
    assert "token_ids" not in json.dumps(receipt)
    assert receipt["cross_entropy_seed_token_sha256"] == shorter_receipt["cross_entropy_seed_token_sha256"]

    for index, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
        expected = module._repeated_tokens([11, 17, 23], 12, offset=index)
        assert prompt.to_ints() + completion == expected
        assert len(prompt.to_ints()) == 7
        assert len(completion) == 5
    # Candidate construction does not depend on a particular world size: a
    # larger topology merely consumes the same fixed candidate recipe.
    assert [prompt.to_ints() for prompt in prompts[:2]] == [prompt.to_ints() for prompt in shorter_prompts]
    assert completions[:2] == shorter_completions

    score_prompts, score_completions, score_grid = module._all_worker_update_probe_score_grid(
        prompts,
        completions,
        worker_count=3,
    )
    assert len(score_prompts) == len(score_completions) == 12
    assert [(row["candidate_index"], row["worker_index"]) for row in score_grid] == [
        (candidate_index, worker_index) for candidate_index in range(4) for worker_index in range(3)
    ]
    assert all(row["score_row"] % 3 == row["worker_index"] for row in score_grid)


def test_update_aligned_probe_completion_must_leave_a_nonempty_prefix(tmp_path):
    module = _module()

    with pytest.raises(SystemExit) as exc_info:
        _args(
            module,
            tmp_path / "out",
            "--parity-sequence-tokens",
            "12",
            "--effect-probe-completion-tokens",
            "12",
        )

    assert exc_info.value.code == 2


@pytest.mark.parametrize("gpu_count", [2, 4, 8])
def test_reference_uses_the_replicated_backend_lpt_shard_order(tmp_path, gpu_count: int):
    module = _module()

    class ModelInput:
        def __init__(self, token_count: int):
            self._tokens = [0] * token_count

        def to_ints(self):
            return list(self._tokens)

    class Datum:
        def __init__(self, original_index: int):
            self.original_index = original_index
            self.model_input = ModelInput(512)

    args = _args(module, tmp_path / "out")
    contract = module.resolve_preflight_contract(
        args,
        cuda_visible_devices=",".join(f"GPU-{index}" for index in range(gpu_count)),
    )
    datums = tuple(Datum(index) for index in range(2 * gpu_count))

    ordered, shards = module._parity_reference_datums_in_lpt_shard_order(datums, contract=contract)

    expected_shards = tuple(tuple(range(rank, 2 * gpu_count, gpu_count)) for rank in range(gpu_count))
    assert shards == expected_shards
    assert [datum.original_index for datum in ordered] == [index for shard in expected_shards for index in shard]


def test_loss_parity_defers_a_small_bf16_relative_discrepancy_until_adapter_state_parity():
    module = _module()

    report = module._fixed_update_loss_parity(
        single_rank_loss=0.5790573358535767,
        replicated_loss=0.5788222104310989,
        atol=1e-4,
        rtol=5e-4,
    )

    assert report["abs_difference"] == pytest.approx(0.00023512542247772217)
    assert 3e-4 < report["relative_difference"] < 5e-4
    assert report["absolute_passed"] is False
    assert report["relative_passed"] is True
    assert report["candidate_passed"] is True
    assert report["passed"] is False
    assert report["requires_adapter_state_parity"] is True
    assert report["acceptance_basis"] is None


def test_loss_parity_gross_difference_fails_before_adapter_publication():
    module = _module()

    report = module._fixed_update_loss_parity(
        single_rank_loss=0.58,
        replicated_loss=0.579,
        atol=1e-4,
        rtol=5e-4,
    )

    assert report["absolute_passed"] is False
    assert report["relative_passed"] is False
    assert report["candidate_passed"] is False
    assert report["passed"] is False


def test_adapter_update_receipt_marks_opposite_first_step_diagnostics_without_hiding_raw_gradient_gate():
    import torch

    module = _module()
    initial = {
        "lora_A": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "lora_B": torch.tensor([0.0, 0.0], dtype=torch.float32),
    }
    reference_final = {
        "lora_A": torch.tensor([5e-5, -5e-5], dtype=torch.float32),
        "lora_B": torch.tensor([2.5e-5, -2.5e-5], dtype=torch.float32),
    }
    close_replicated_final = {name: value * 0.999 for name, value in reference_final.items()}
    close_report = module._adapter_update_delta_parity(
        reference_initial=initial,
        reference_final=reference_final,
        replicated_initial={name: value.clone() for name, value in initial.items()},
        replicated_final=close_replicated_final,
        atol=2e-4,
        delta_cosine_min=0.9999,
        delta_relative_l2_max=0.005,
        delta_max_abs_atol=2.5e-5,
    )

    assert close_report["initial_state_exact"] is True
    assert close_report["final_absolute_passed"] is True
    assert close_report["delta_cosine_passed"] is True
    assert close_report["delta_relative_l2_passed"] is True
    assert close_report["passed"] is True

    opposite_replicated_final = {name: -value for name, value in reference_final.items()}
    opposite_report = module._adapter_update_delta_parity(
        reference_initial=initial,
        reference_final=reference_final,
        replicated_initial={name: value.clone() for name, value in initial.items()},
        replicated_final=opposite_replicated_final,
        # The final states differ by at most 1e-4, so an absolute-only 2e-4
        # check would accept this materially wrong, opposite update.
        atol=2e-4,
        delta_cosine_min=0.9999,
        delta_relative_l2_max=0.005,
        delta_max_abs_atol=2.5e-5,
    )

    assert opposite_report["final_absolute_passed"] is True
    assert opposite_report["delta_cosine_similarity"] == pytest.approx(-1.0)
    assert opposite_report["delta_cosine_diagnostic_passed"] is False
    assert opposite_report["delta_relative_l2_diagnostic_passed"] is False
    assert opposite_report["delta_max_abs_diagnostic_passed"] is False
    assert opposite_report["lora_b_sign_mismatch_diagnostic_passed"] is False
    # Post-Adam signals are retained as calibration diagnostics.  The hard
    # rejection happens earlier from the raw pre-optimizer gradient receipt.
    assert opposite_report["post_adam_diagnostics_are_non_gating"]
    assert opposite_report["passed"] is True


def test_pre_optimizer_gradient_parity_passes_tiny_near_zero_sign_flips_and_reports_tensor_diagnostics():
    import torch

    module = _module()
    reference = {
        "lora_A": torch.tensor([1.0, -2.0, 3.0, 1e-9, -1e-9], dtype=torch.float32),
        "lora_B": torch.tensor([4.0, -5.0], dtype=torch.float32),
    }
    replicated = {
        "lora_A": torch.tensor([1.0, -2.0, 3.0, -1e-9, 1e-9], dtype=torch.float32),
        "lora_B": torch.tensor([4.0, -5.0], dtype=torch.float32),
    }

    report = module._gradient_vector_parity(reference=reference, replicated=replicated)
    gate = module._gradient_parity_gate_summary(
        report,
        cosine_min=0.999,
        norm_ratio_min=0.95,
        norm_ratio_max=1.05,
        relative_l2_max=0.05,
        magnitude_weighted_sign_mismatch_max=0.01,
    )

    assert report["sign_mismatch_fraction"] > 0
    assert report["magnitude_weighted_sign_disagreement"] < 1e-8
    assert report["top_offenders"][0]["name"] == "lora_A"
    assert len(report["per_tensor"]) == 2
    assert gate["passed"] is True


@pytest.mark.parametrize(
    ("kind", "replicated"),
    [
        ("gross-opposite", [-1.0, 2.0, -3.0, -4.0]),
        ("dropped", [0.0, 0.0, 0.0, 0.0]),
        ("wrong-scale", [1.2, -2.4, 3.6, 4.8]),
    ],
)
def test_pre_optimizer_gradient_parity_rejects_gross_drop_and_wrong_scale(kind, replicated):
    import torch

    module = _module()
    reference = {"lora_B": torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=torch.float32)}
    report = module._gradient_vector_parity(
        reference=reference,
        replicated={"lora_B": torch.tensor(replicated, dtype=torch.float32)},
    )
    gate = module._gradient_parity_gate_summary(
        report,
        cosine_min=0.999,
        norm_ratio_min=0.95,
        norm_ratio_max=1.05,
        relative_l2_max=0.05,
        magnitude_weighted_sign_mismatch_max=0.01,
    )

    assert gate["passed"] is False, kind
    if kind == "gross-opposite":
        assert gate["cosine_passed"] is False
        assert gate["magnitude_weighted_sign_mismatch_passed"] is False
    if kind == "dropped":
        assert gate["replicated_gradient_nonzero"] is False
    if kind == "wrong-scale":
        assert gate["cosine_passed"] is True
        assert gate["norm_ratio_passed"] is False


def test_rank_zero_gradient_capture_wraps_existing_reducer_and_records_pre_sum_summary(tmp_path):
    import torch
    from safetensors.torch import load_file

    module = _module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_B = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

    class Backend:
        def __init__(self):
            self.model = Model()
            self.model.lora_B.grad = torch.tensor([1.0, -2.0], dtype=torch.float32)
            self._gradient_accumulations = 1
            self.reducer_calls = 0

            def original(parameters):
                self.reducer_calls += 1
                for parameter in parameters:
                    parameter.grad.mul_(2.0)

            self._gradient_reducer = original

        def _require_model(self):
            return self.model

        def set_gradient_reducer(self, reducer):
            self._gradient_reducer = reducer

    backend = Backend()
    diagnostics = []
    capture = module._install_rank_zero_gradient_capture(
        backend,
        output_path=tmp_path / "replicated_pre_optimizer_gradients.safetensors",
        diagnostic_sink=lambda receipt: diagnostics.append(copy.deepcopy(receipt)),
    )
    backend._gradient_reducer([backend.model.lora_B])

    assert backend.reducer_calls == 1
    assert capture["capture_count"] == 1
    assert capture["gradient_accumulations_before_reduce"] == 1
    assert capture["pre_sum_rank_zero_gradient_summary"]["l1_norm"] == pytest.approx(3.0)
    saved = load_file(capture["receipt"]["path"], device="cpu")
    assert torch.equal(saved["lora_B"], torch.tensor([2.0, -4.0], dtype=torch.float32))
    assert diagnostics[-1]["status"] == "captured"
    assert diagnostics[-1]["receipt"]["sha256"] == capture["receipt"]["sha256"]
    with pytest.raises(RuntimeError, match="invoked more than once"):
        backend._gradient_reducer([backend.model.lora_B])
    assert backend.reducer_calls == 1

    class FailingReducerBackend(Backend):
        def __init__(self):
            super().__init__()

            def original(_parameters):
                self.reducer_calls += 1
                raise RuntimeError("synthetic reducer failure")

            self._gradient_reducer = original

    failing_backend = FailingReducerBackend()
    module._install_rank_zero_gradient_capture(
        failing_backend,
        output_path=tmp_path / "failing_reducer_pre_optimizer_gradients.safetensors",
    )
    with pytest.raises(RuntimeError, match="synthetic reducer failure"):
        failing_backend._gradient_reducer([failing_backend.model.lora_B])
    with pytest.raises(RuntimeError, match="invoked more than once"):
        failing_backend._gradient_reducer([failing_backend.model.lora_B])
    assert failing_backend.reducer_calls == 1


def test_single_rank_reference_failure_retains_prior_immutable_artifact_receipt(tmp_path):
    """A reference F/B failure must not fall back to the minimal early receipt."""

    module = _module()

    class FakeCuda:
        @staticmethod
        def synchronize(_device):
            return None

        @staticmethod
        def empty_cache():
            return None

    class FakeTorch:
        bfloat16 = "bf16"
        cuda = FakeCuda()

        @staticmethod
        def manual_seed(_seed):
            return None

    class Model:
        def save_pretrained(self, directory):
            target = Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            (target / "adapter_model.safetensors").write_bytes(b"immutable-initial-adapter")

    class Backend:
        latest = None

        def __init__(self, **_kwargs):
            self.model = Model()
            self.shutdown_called = False
            self.setup_kwargs = None
            type(self).latest = self

        def setup(self, **_kwargs):
            self.setup_kwargs = dict(_kwargs)
            return None

        def _require_model(self):
            return self.model

        async def submit_forward_backward(self, _datums, _loss_name):
            raise RuntimeError("synthetic reference forward failure")

        def shutdown(self):
            self.shutdown_called = True

    args = SimpleNamespace(
        forward_microbatch_max_datums=2,
        parity_datums_per_rank=2,
        lora_seed=42,
        lora_rank=8,
        lora_alpha=16,
        packing_budgets=[20_480],
        target_logprob_chunk_size=2_048,
        model="Qwen/Qwen3.5-9B",
        learning_rate=1e-4,
    )
    contract = SimpleNamespace(coordinator_device="cuda:0", checkpoint_layers=None)
    diagnostics = []
    runtime_model = "/scratch/hf/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

    with pytest.raises(RuntimeError, match="synthetic reference forward failure"):
        asyncio.run(
            module._single_rank_reference_update(
                args=args,
                contract=contract,
                datums=(),
                planned_original_indices=((),),
                output_dir=tmp_path / "evidence",
                torch=FakeTorch(),
                LocalBackend=Backend,
                LoRAConfig=lambda **kwargs: kwargs,
                AdamConfig=lambda **kwargs: kwargs,
                trainable_state_hash=lambda _backend: "reference-state-before",
                runtime_model=runtime_model,
                diagnostic_sink=lambda receipt: diagnostics.append(copy.deepcopy(receipt)),
            )
        )

    receipt = diagnostics[-1]
    assert receipt["status"] == "failed"
    assert receipt["failure"]["type"] == "RuntimeError"
    assert receipt["initial_adapter_dir"].endswith("single_rank_reference/initial_adapter")
    assert receipt["initial_adapter_model_sha256"]
    assert receipt["state_hash_before"] == "reference-state-before"
    assert receipt["canonical_model_repo"] == args.model
    assert receipt["runtime_model_argument"] == runtime_model
    assert Backend.latest is not None
    assert Backend.latest.setup_kwargs is not None
    assert Backend.latest.setup_kwargs["model"] == runtime_model
    assert "lora" in Backend.latest.setup_kwargs
    assert Backend.latest.shutdown_called is True
    # This is the exact payload shape the outer failure handler serializes.
    json.dumps({"single_rank_reference": receipt}, sort_keys=True)


def test_packing_sweep_sink_preserves_prior_successes_and_failing_budget(monkeypatch):
    """The caller can serialize partial packing results if a later budget fails."""

    module = _module()

    class Datum:
        model_input = SimpleNamespace(to_ints=lambda: [1, 2, 3])

    class Model:
        def zero_grad(self, *, set_to_none):
            assert set_to_none is True

    class Output:
        metrics = {"loss": 0.25}

    class RankZero:
        def __init__(self):
            self.forward_microbatch_max_tokens = 1
            self.forward_microbatch_max_datums = 1
            self._gradient_accumulations = 0
            self.calls = 0
            self.model = Model()

        def _require_model(self):
            return self.model

        def _forward_microbatches(self, _counts):
            return [object()]

        async def submit_forward_backward(self, _datums, _loss_name):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic packing failure")
            return Output()

    class FakeCuda:
        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def reset_peak_memory_stats(_device):
            return None

        @staticmethod
        def synchronize(_device):
            return None

        @staticmethod
        def max_memory_allocated(_device):
            return 123

        @staticmethod
        def max_memory_reserved(_device):
            return 456

    monkeypatch.setattr(
        module, "_make_cross_entropy_datums", lambda **kwargs: [Datum() for _ in range(kwargs["count"])]
    )
    args = SimpleNamespace(packing_budgets=[10, 20], packing_probe_datums=2)
    snapshots = []

    with pytest.raises(RuntimeError, match="packing probe failed at budget 20"):
        asyncio.run(
            module._packing_sweep(
                args=args,
                rank_zero=RankZero(),
                tokenizer=None,
                torch=SimpleNamespace(cuda=FakeCuda()),
                types=None,
                datum_from_model_input_weights=None,
                trainable_state_hash=lambda _rank_zero: "unchanged",
                coordinator_device="cuda:0",
                diagnostic_sink=lambda partial: snapshots.append(copy.deepcopy(partial)),
            )
        )

    assert snapshots[0] == []
    partial = snapshots[-1]
    assert [entry["packing_budget"] for entry in partial] == [10, 20]
    assert partial[0]["passed"] is True
    assert partial[1]["passed"] is False
    assert "synthetic packing failure" in partial[1]["failure"]


def test_success_marker_is_exclusive_and_written_after_the_final_event(tmp_path):
    module = _module()
    marker = tmp_path / "SUCCESS"

    module._write_success_marker(marker)

    assert marker.read_text(encoding="utf-8") == "phase-shared preflight passed\n"
    assert list(tmp_path.glob(".SUCCESS.prepared-*"))
    with pytest.raises(FileExistsError):
        module._write_success_marker(marker)

    source = SCRIPT.read_text(encoding="utf-8")
    success_block = source[
        source.index('result["status"] = "passed"') : source.index(
            "    except BaseException as exc:", source.index('result["status"] = "passed"')
        )
    ]
    assert success_block.index('_append_event(events, "preflight_passed")') < success_block.index(
        '_write_success_marker(output_dir / "SUCCESS")'
    )
    assert success_block.index('_write_success_marker(output_dir / "SUCCESS")') < success_block.index("return result")


def test_preflight_omits_the_gdn_prefill_override_by_default(tmp_path):
    module = _module()
    args = _args(module, tmp_path / "out")

    assert args.worker_gdn_prefill_backend is None
    assert "gdn_prefill_backend" not in module._rank_zero_vllm_options(args)
    assert "gdn_prefill_backend" not in module._worker_vllm_options(args)
    assert module._contract_config(args)["worker_gdn_prefill_backend"] is None


@pytest.mark.parametrize("backend", ["flashinfer", "triton"])
def test_preflight_propagates_an_explicit_gdn_prefill_backend_to_every_vllm_path(tmp_path, backend: str):
    module = _module()
    args = _args(module, tmp_path / "out", "--worker-gdn-prefill-backend", backend)

    assert module._rank_zero_vllm_options(args)["gdn_prefill_backend"] == backend
    assert module._worker_vllm_options(args)["gdn_prefill_backend"] == backend
    assert module._contract_config(args)["worker_gdn_prefill_backend"] == backend


def test_formal_worker_effect_summary_has_hash_bound_attestation_shape():
    module = _module()
    summary = module._formal_worker_effect_parity(
        worker_policy=[[2.0, 3.0]],
        worker_base=[[1.0, 1.0]],
        hf_policy=[[2.0, 3.0]],
        hf_base=[[1.0, 1.0]],
    )

    assert summary["worker_v2_minus_base"]["max_abs_difference"] == 2.0
    assert summary["coordinator_updated_minus_base"]["max_abs_difference"] == 2.0
    assert summary["worker_minus_coordinator_effect"]["max_abs_difference"] == 0.0
    assert summary["cosine_similarity"] == pytest.approx(1.0)


def test_effect_fidelity_gate_rejects_same_direction_but_mis_scaled_and_opposite_effects():
    module = _module()

    def summary(worker_effect, hf_effect):
        return module._effect_summary(
            worker_policy=[worker_effect],
            worker_base=[[0.0 for _ in worker_effect]],
            hf_policy=[hf_effect],
            hf_base=[[0.0 for _ in hf_effect]],
        )

    exact = module._effect_gate_summary(
        summary([1.0, -1.0], [1.0, -1.0]),
        min_effect=0.01,
        cosine_min=0.9,
        norm_ratio_min=0.8,
        norm_ratio_max=1.25,
        relative_l2_max=0.25,
    )
    tiny_same_direction = module._effect_gate_summary(
        summary([0.01, -0.01], [1.0, -1.0]),
        min_effect=0.005,
        cosine_min=0.9,
        norm_ratio_min=0.8,
        norm_ratio_max=1.25,
        relative_l2_max=0.25,
    )
    opposite = module._effect_gate_summary(
        summary([-1.0, 1.0], [1.0, -1.0]),
        min_effect=0.01,
        cosine_min=0.9,
        norm_ratio_min=0.8,
        norm_ratio_max=1.25,
        relative_l2_max=0.25,
    )

    assert exact["passed"] is True
    assert tiny_same_direction["cosine_passed"] is True
    assert tiny_same_direction["norm_ratio_passed"] is False
    assert tiny_same_direction["relative_l2_passed"] is False
    assert tiny_same_direction["passed"] is False
    assert opposite["norm_ratio_passed"] is True
    assert opposite["cosine_passed"] is False
    assert opposite["relative_l2_passed"] is False
    assert opposite["passed"] is False


def test_post_update_effect_sink_retains_full_diagnostics_before_a_strict_cosine_failure():
    module = _module()

    class ModelInput:
        def __init__(self, tokens):
            self._tokens = list(tokens)

        def to_ints(self):
            return list(self._tokens)

    class ScoreSampler:
        def __init__(self, rows):
            self.rows = rows

        async def score_completions(self, prompts, completions):
            assert len(prompts) == len(completions) == 4
            return copy.deepcopy(self.rows)

    class Backend:
        def policy_sampler(self, reason):
            assert reason == "phase_shared_preflight"
            return ScoreSampler([[-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02]])

        def base_sampler(self):
            return ScoreSampler([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    class RankZero:
        def _score_completions(self, prompts, completions, *, use_base):
            if len(prompts) == 2:
                # Candidate selection sees a clear nonzero HF effect.
                return [[0.0, 0.0], [0.0, 0.0]] if use_base else [[0.02, -0.02], [0.02, -0.02]]
            assert len(prompts) == len(completions) == 4
            return (
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
                if use_base
                else [[0.02, -0.02], [0.02, -0.02], [0.02, -0.02], [0.02, -0.02]]
            )

    class GPU:
        def __init__(self, index):
            self.logical_index = index
            self.device_token = f"GPU-{index}"

        def as_dict(self):
            return {"logical_index": self.logical_index, "device_token": self.device_token}

    reports = []
    with pytest.raises(
        RuntimeError, match="aggregate post-update worker effect: worker and HF adapter effects disagree"
    ):
        asyncio.run(
            module._post_update_worker_effect(
                backend=Backend(),
                rank_zero=RankZero(),
                candidates=[ModelInput([1, 2, 3]), ModelInput([2, 3, 4])],
                candidate_completions=[[4, 5], [5, 6]],
                fixed_token_probe_contract={
                    "schema": "phase-shared-update-aligned-fixed-token-probe-v1",
                    "kind": "repeated-cross-entropy-sequence-suffix-score-completions-v1",
                    "completion_token_count_per_candidate": 2,
                },
                worker_count=2,
                worker_gpus=[GPU(0), GPU(1)],
                min_effect=0.01,
                cosine_min=0.9,
                norm_ratio_min=0.8,
                norm_ratio_max=1.25,
                relative_l2_max=0.25,
                diagnostic_sink=lambda report: reports.append(copy.deepcopy(report)),
            )
        )

    # The last sink call happens before the aggregate gate, after every worker
    # was measured, and therefore is exactly what the outer failure handler
    # serializes into failure.json.
    diagnostic = reports[-1]
    assert diagnostic["status"] == "scored"
    assert diagnostic["aggregate"]["worker_effect_l2_norm"] == pytest.approx(math.sqrt(8) * 0.02)
    assert diagnostic["aggregate"]["hf_effect_l2_norm"] == pytest.approx(math.sqrt(8) * 0.02)
    assert diagnostic["aggregate"]["effect_difference_l2_norm"] == pytest.approx(math.sqrt(8) * 0.04)
    assert diagnostic["aggregate"]["effect_difference_relative_l2_error"] == pytest.approx(2.0)
    assert diagnostic["aggregate"]["cosine_similarity"] == pytest.approx(-1.0)
    assert diagnostic["aggregate_gate"]["cosine_passed"] is False
    assert len(diagnostic["per_worker"]) == 2
    assert all(worker["gate"]["passed"] is False for worker in diagnostic["per_worker"])
    assert diagnostic["per_worker"][0]["candidate_indices"] == [0, 1]
    assert diagnostic["per_worker"][1]["candidate_indices"] == [0, 1]
    assert diagnostic["score_grid"] == {
        "assignment_policy": "candidate_major_round_robin_all_workers-v1",
        "score_row_count": 4,
        "candidate_count": 2,
        "rows_per_worker": 2,
        "all_workers_receive_all_candidates": True,
    }
    assert diagnostic["fixed_token_probe"]["score_rows"] == 4
    assert diagnostic["fixed_token_probe"]["score_inputs_sha256"]
    assert set(diagnostic["fixed_token_probe"]) == {
        "kind",
        "score_rows",
        "completion_token_count",
        "score_inputs_sha256",
    }
    assert diagnostic["fixed_token_probe"]["kind"] == "post-update-worker-score-completions-v1"
    assert diagnostic["update_aligned_fixed_token_probe"]["kind"] == (
        "repeated-cross-entropy-sequence-suffix-score-completions-v1"
    )
    # Exercise the same narrow validator used by the attestation writer.
    compat = _compat_module()
    assert compat._fixed_token_probe(diagnostic["fixed_token_probe"]) == diagnostic["fixed_token_probe"]

    failure_payload = json.loads(
        json.dumps({"phase_shared": {"post_update_worker_lora_effect": diagnostic}, "status": "failed"})
    )
    serialized = json.dumps(failure_payload, sort_keys=True)
    assert "worker_policy" not in serialized
    assert "worker_base" not in serialized
    assert "hf_policy" not in serialized
    assert "hf_base" not in serialized
    assert failure_payload["phase_shared"]["post_update_worker_lora_effect"]["per_worker"][1]["logical_gpu"] == 1


def test_dry_run_is_cpu_only_and_contains_the_non_production_contract(monkeypatch, tmp_path, capsys):
    module = _module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")

    code = module.main(["--dry-run", "--output-dir", str(tmp_path / "evidence")])

    assert code == 0
    assert not (tmp_path / "evidence").exists()
    payload = capsys.readouterr().out
    assert '"dry_run": true' in payload
    assert '"non_production": true' in payload
    assert "single-rank versus replicated fixed-update loss and adapter parity" in payload
    assert "49152" in payload


def test_wrapper_delegates_without_creating_or_selecting_gpu_resources():
    source = WRAPPER.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES" in source
    assert "preflight_qwen35_phase_shared.py" in source
    assert "mkdir" not in source
    assert "vastai" not in source.lower().replace("infra/vastai", "")
    assert "rm -rf" not in source


def test_wrapper_runs_a_short_lived_all_visible_gpu_health_probe_before_the_main_harness():
    source = WRAPPER.read_text(encoding="utf-8")

    probe_start = source.index("\"$python_bin\" - <<'PY'")
    main_exec = source.index('exec "$python_bin" infra/vastai/preflight_qwen35_phase_shared.py')
    assert probe_start < main_exec
    assert "torch.cuda.device_count()" in source
    assert "for logical_index in range(visible_count):" in source
    assert 'torch.empty((1,), device=f"cuda:{logical_index}", dtype=torch.uint8)' in source
    assert "torch.cuda.synchronize(logical_index)" in source
    assert "CTM_PHASE_SHARED_GPU_HEALTH=passed" in source
    assert "The subprocess" in source
    assert "exit releases all of these probe contexts" in source
