"""Static contract tests for the direct Qwen3.5 ACT repair gate."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import types
from pathlib import Path

import pytest

# The normal experiment image installs the pinned mcq_bias package. This
# offline contract test only exercises frozen-row transforms, so provide the
# two adapter-import symbols only when a lightweight local test environment
# genuinely omits that optional package. Checking ``sys.modules`` alone is
# order-dependent: a later scorer test would otherwise inherit this stub even
# when the real package is installed.
try:
    import mcq_bias.parsers  # noqa: F401
except ModuleNotFoundError:
    _mcq_bias = types.ModuleType("mcq_bias")
    _mcq_bias.__path__ = []
    _parsers = types.ModuleType("mcq_bias.parsers")
    _parsers.parse_answer = lambda response: response
    _scorers = types.ModuleType("mcq_bias.scorers")
    _scorers.matches_bias = lambda answer, target: float(answer == target)
    _mcq_bias.parsers = _parsers
    _mcq_bias.scorers = _scorers
    sys.modules["mcq_bias"] = _mcq_bias
    sys.modules["mcq_bias.parsers"] = _parsers
    sys.modules["mcq_bias.scorers"] = _scorers

from scripts import run_experiment as experiment
from experiments.act_repair_gate import prepare as gate_prepare
from experiments.act_repair_gate.runtime_parity import (
    DEFAULT_MAX_LOGPROBS,
    GDN_PREFILL_BACKENDS,
    RESULT_VARIANTS,
    VLLM_LOGPROBS_MODE,
    VLLM_PARITY_SAMPLING,
    VLLM_SCORE_TRANSPORT,
    _parallel_isolated_vllm_plan,
    _parallel_isolated_vllm_report_metadata,
    _runtime_adapter_loads,
    _vllm_server_command,
    _vllm_scores,
)
from experiments.act_repair_gate import runtime_parity
from experiments.stage1_iid_diagnostic import prepare as iid_prepare


ROOT = Path(__file__).parent.parent
PLAN = ROOT / "experiments" / "act_repair_gate" / "experiment.yaml"
MODEL = "Qwen/Qwen3.5-9B"
GENERATION = {
    "max_tokens": 20480,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "extra_body": {"top_k": 20},
}
ATTENTION_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]


def _by_name(entries: list[dict]) -> dict[str, dict]:
    return {entry["name"]: entry for entry in entries}


def _synthetic_row(index: int) -> dict:
    dataset = "logiqa" if index % 2 == 0 else "hellaswag"
    clean = f"Question {index}: select the correct option."
    return {
        "question": clean,
        "question_id": f"q-{index:04d}",
        "source_dataset": dataset,
        "prompt_style": "encourage_cot",
        "unbiased_messages": [{"role": "user", "content": clean}],
        "biased_messages": [{"role": "user", "content": f"native bias {index}"}],
        "bias_type": "wrong_argument",
        "ground_truth": "A",
        "biased_option": "B",
        "biasing_text": f"wrong argument {index}",
    }


def _write_synthetic_source(path: Path) -> str:
    payload = b"".join(
        (json.dumps(_synthetic_row(index), sort_keys=True) + "\n").encode()
        for index in range(iid_prepare.SOURCE_ROWS)
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_direct_plan_rebuilds_remote_local_iid_artifacts_before_canonicalization():
    source = experiment.load_experiment_source(PLAN)
    config = experiment.load_experiment(PLAN)

    assert "experiment_factory" not in source
    assert config["name"] == "act_repair_gate_qwen3_5_9b_20260731"
    assert source["variables"]["frozen_iid_source"].endswith(
        "/source/distractor-argument-pairs.jsonl"
    )

    preparation = _by_name(config["data_preparation"])
    assert list(preparation) == ["prepare_remote_local_gate_data"]
    assert all(entry["target"] == "data" and entry["resource"] == "cpu" for entry in preparation.values())

    localize = preparation["prepare_remote_local_gate_data"]
    assert localize["command"][-1] == "experiments.act_repair_gate.prepare"
    assert localize["args"]["output_dir"] == "${artifact_root}/data"
    assert localize["args"] == {
        "source": "${frozen_iid_source}",
        "output_dir": "${artifact_root}/data",
    }


def test_gate_prep_creates_remote_local_artifacts_and_resumes_byte_identically(tmp_path, monkeypatch):
    source = tmp_path / "source" / "distractor-argument-pairs.jsonl"
    source.parent.mkdir()
    source_hash = _write_synthetic_source(source)
    monkeypatch.setattr(iid_prepare, "SOURCE_SHA256", source_hash)
    output = tmp_path / "gate-data"

    first = gate_prepare.prepare_act_repair_gate(source, output)
    snapshot = {path.relative_to(output): path.read_bytes() for path in output.iterdir()}
    second = gate_prepare.prepare_act_repair_gate(source, output)

    assert first["iid_splits"] == "written"
    assert first["canonical_splits"] == {
        "train_eval": "written",
        "heldout_in_domain": "written",
    }
    assert second["iid_splits"] == "resumed"
    assert second["canonical_splits"] == {
        "train_eval": "resumed",
        "heldout_in_domain": "resumed",
    }
    assert {path.relative_to(output): path.read_bytes() for path in output.iterdir()} == snapshot

    document = json.loads((output / "manifest.json").read_text())
    assert Path(document["source"]["path"]).resolve() == source.resolve()
    assert Path(document["splits"]["train_eval"]["path"]).resolve() == (
        output / "train-eval-n200.jsonl"
    ).resolve()
    assert Path(document["splits"]["heldout_in_domain"]["path"]).resolve() == (
        output / "heldout-in-domain-n200.jsonl"
    ).resolve()


def test_gate_prep_rejects_partial_split_artifacts(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    source_hash = _write_synthetic_source(source)
    monkeypatch.setattr(iid_prepare, "SOURCE_SHA256", source_hash)
    output = tmp_path / "gate-data"
    output.mkdir()
    (output / "train-eval-n200.jsonl").write_text("partial\n")

    with pytest.raises(FileExistsError, match="incomplete remote-local IID split artifact set"):
        gate_prepare.prepare_act_repair_gate(source, output)


def test_runtime_parity_registers_the_evaluator_path_under_its_exact_identity():
    checkpoint = "/immutable/compat-adapter"
    loads = _runtime_adapter_loads(
        {
            "full": "full",
            "linear_only": "linear_only",
            "self_attn_only": "self_attn_only",
            "evaluator_path": checkpoint,
        },
        {
            "full": "/variants/full",
            "linear_only": "/variants/linear_only",
            "self_attn_only": "/variants/self_attn_only",
            "evaluator_path": checkpoint,
        },
    )

    assert loads == {
        "full": "/variants/full",
        "linear_only": "/variants/linear_only",
        "self_attn_only": "/variants/self_attn_only",
        checkpoint: checkpoint,
    }


@pytest.mark.parametrize("backend", GDN_PREFILL_BACKENDS)
def test_runtime_parity_passes_an_explicit_gdn_prefill_backend_to_vllm(backend: str):
    command = _vllm_server_command(
        executable="vllm",
        model=MODEL,
        port=8789,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        max_loras=4,
        enforce_eager=False,
        gdn_prefill_backend=backend,
    )

    assert command[-2:] == ["--gdn-prefill-backend", backend]
    assert command[command.index("--max-logprobs") + 1] == str(DEFAULT_MAX_LOGPROBS)
    assert command[command.index("--logprobs-mode") + 1] == VLLM_LOGPROBS_MODE


def test_runtime_parity_omits_gdn_prefill_backend_by_default_and_rejects_auto():
    command = _vllm_server_command(
        executable="vllm",
        model=MODEL,
        port=8789,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        max_loras=4,
        enforce_eager=True,
        gdn_prefill_backend=None,
    )

    assert "--gdn-prefill-backend" not in command
    assert command[command.index("--max-logprobs") + 1] == str(DEFAULT_MAX_LOGPROBS)
    assert command[command.index("--logprobs-mode") + 1] == VLLM_LOGPROBS_MODE
    assert command[-1] == "--enforce-eager"
    with pytest.raises(ValueError, match="flashinfer.*triton.*cutedsl"):
        _vllm_server_command(
            executable="vllm",
            model=MODEL,
            port=8789,
            max_model_len=32768,
            gpu_memory_utilization=0.9,
            max_loras=4,
            enforce_eager=False,
            gdn_prefill_backend="auto",
        )
    with pytest.raises(ValueError, match="max_logprobs must be exactly 29"):
        _vllm_server_command(
            executable="vllm",
            model=MODEL,
            port=8789,
            max_model_len=32768,
            gpu_memory_utilization=0.9,
            max_loras=4,
            enforce_eager=False,
            gdn_prefill_backend="triton",
            max_logprobs=20,
        )


def test_runtime_parity_parallel_isolation_requires_the_complete_four_variant_attestation():
    plan = _parallel_isolated_vllm_plan(
        requested_result_variants=RESULT_VARIANTS,
        device_tokens=("0", "1", "2", "3"),
        vllm_port=8789,
        isolate_vllm_variants=True,
        parallel_isolated_vllm_variants=True,
        enforce_eager=True,
    )

    assert plan == {
        "full": {
            "device_token": "0",
            "port": 8789,
            "server_log": "vllm-server-full.log",
            "max_logprobs": DEFAULT_MAX_LOGPROBS,
            "logprobs_mode": VLLM_LOGPROBS_MODE,
        },
        "linear_only": {
            "device_token": "1",
            "port": 8790,
            "server_log": "vllm-server-linear_only.log",
            "max_logprobs": DEFAULT_MAX_LOGPROBS,
            "logprobs_mode": VLLM_LOGPROBS_MODE,
        },
        "self_attn_only": {
            "device_token": "2",
            "port": 8791,
            "server_log": "vllm-server-self_attn_only.log",
            "max_logprobs": DEFAULT_MAX_LOGPROBS,
            "logprobs_mode": VLLM_LOGPROBS_MODE,
        },
        "evaluator_path": {
            "device_token": "3",
            "port": 8792,
            "server_log": "vllm-server-evaluator_path.log",
            "max_logprobs": DEFAULT_MAX_LOGPROBS,
            "logprobs_mode": VLLM_LOGPROBS_MODE,
        },
    }

    with pytest.raises(ValueError, match="four unique device tokens"):
        _parallel_isolated_vllm_plan(
            requested_result_variants=RESULT_VARIANTS,
            device_tokens=("0", "1", "1", "3"),
            vllm_port=8789,
            isolate_vllm_variants=True,
            parallel_isolated_vllm_variants=True,
            enforce_eager=True,
        )
    with pytest.raises(ValueError, match="complete attestation set"):
        _parallel_isolated_vllm_plan(
            requested_result_variants=RESULT_VARIANTS[:-1],
            device_tokens=("0", "1", "2", "3"),
            vllm_port=8789,
            isolate_vllm_variants=True,
            parallel_isolated_vllm_variants=True,
            enforce_eager=True,
        )
    with pytest.raises(ValueError, match="requires isolate_vllm_variants"):
        _parallel_isolated_vllm_plan(
            requested_result_variants=RESULT_VARIANTS,
            device_tokens=("0", "1", "2", "3"),
            vllm_port=8789,
            isolate_vllm_variants=False,
            parallel_isolated_vllm_variants=True,
            enforce_eager=True,
        )
    with pytest.raises(ValueError, match="requires enforce_eager=True"):
        _parallel_isolated_vllm_plan(
            requested_result_variants=RESULT_VARIANTS,
            device_tokens=("0", "1", "2", "3"),
            vllm_port=8789,
            isolate_vllm_variants=True,
            parallel_isolated_vllm_variants=True,
            enforce_eager=False,
        )
    with pytest.raises(ValueError, match="requires parallel_isolated_vllm_variants"):
        _parallel_isolated_vllm_plan(
            requested_result_variants=RESULT_VARIANTS,
            device_tokens=("0", "1", "2", "3"),
            vllm_port=8789,
            isolate_vllm_variants=True,
            parallel_isolated_vllm_variants=False,
            enforce_eager=True,
        )


def test_runtime_parity_parallel_isolation_records_deterministic_unique_servers(tmp_path: Path):
    plan = _parallel_isolated_vllm_plan(
        requested_result_variants=RESULT_VARIANTS,
        device_tokens=("GPU-a", "GPU-b", "GPU-c", "GPU-d"),
        vllm_port=9000,
        isolate_vllm_variants=True,
        parallel_isolated_vllm_variants=True,
        enforce_eager=True,
    )
    assert plan is not None

    metadata = _parallel_isolated_vllm_report_metadata(root=tmp_path, plan=plan)

    assert list(metadata) == list(RESULT_VARIANTS)
    assert [metadata[variant]["device_token"] for variant in RESULT_VARIANTS] == [
        "GPU-a",
        "GPU-b",
        "GPU-c",
        "GPU-d",
    ]
    assert [metadata[variant]["port"] for variant in RESULT_VARIANTS] == [9000, 9001, 9002, 9003]
    assert len({metadata[variant]["server_log"] for variant in RESULT_VARIANTS}) == 4
    assert {metadata[variant]["max_loras"] for variant in RESULT_VARIANTS} == {1}
    assert {metadata[variant]["max_logprobs"] for variant in RESULT_VARIANTS} == {DEFAULT_MAX_LOGPROBS}
    assert {metadata[variant]["logprobs_mode"] for variant in RESULT_VARIANTS} == {VLLM_LOGPROBS_MODE}
    cache_paths = [
        path
        for variant in RESULT_VARIANTS
        for path in metadata[variant]["cache_directories"].values()
    ]
    assert len(cache_paths) == len(set(cache_paths)) == 12
    assert all(str(path).startswith(str(tmp_path)) for path in cache_paths)


def test_runtime_parity_parallel_servers_are_concurrent_and_return_in_canonical_order(monkeypatch, tmp_path: Path):
    plan = _parallel_isolated_vllm_plan(
        requested_result_variants=RESULT_VARIANTS,
        device_tokens=("0", "1", "2", "3"),
        vllm_port=9100,
        isolate_vllm_variants=True,
        parallel_isolated_vllm_variants=True,
        enforce_eager=True,
    )
    assert plan is not None
    started: list[dict] = []
    stopped: list[str] = []
    observed_urls: list[str] = []
    barrier = threading.Barrier(len(RESULT_VARIANTS), timeout=5)

    class FakeProcess:
        def __init__(self, token: str):
            self.token = token

    def fake_start(**kwargs):
        started.append(kwargs)
        return FakeProcess(kwargs["device_token"])

    def fake_stop(process):
        stopped.append(process.token)

    def fake_probe(**kwargs):
        # A sequential implementation times out here; reaching the barrier
        # proves the four fresh servers were serviced concurrently.
        barrier.wait()
        observed_urls.append(kwargs["base_url"])
        runtime_parity._stop_process(kwargs["process"])
        variant = kwargs["variant"]
        return {
            "base_name": f"base-{variant}",
            "model_ids": [f"base-{variant}", kwargs["runtime_adapter_name"]],
            "base_scores": [{1: 0.0}],
            "scores": [{1: 1.0}],
            "variant": variant,
        }

    monkeypatch.setattr(runtime_parity, "_start_vllm", fake_start)
    monkeypatch.setattr(runtime_parity, "_stop_process", fake_stop)
    monkeypatch.setattr(runtime_parity, "_probe_isolated_vllm_server", fake_probe)
    outcomes = runtime_parity._run_parallel_isolated_vllm_variants(
        model_name=MODEL,
        root=tmp_path,
        plan=plan,
        runtime_adapter_names={variant: f"adapter-{variant}" for variant in RESULT_VARIANTS},
        runtime_adapter_paths={variant: f"/adapters/{variant}" for variant in RESULT_VARIANTS},
        prompt_token_ids=[[1]],
        token_ids=[[2]],
        max_model_len=32768,
        vllm_memory_utilization=0.9,
        enforce_eager=True,
        gdn_prefill_backend="triton",
    )

    assert list(outcomes) == list(RESULT_VARIANTS)
    assert [entry["device_token"] for entry in started] == ["0", "1", "2", "3"]
    assert [entry["port"] for entry in started] == [9100, 9101, 9102, 9103]
    assert {entry["max_loras"] for entry in started} == {1}
    assert {entry["max_logprobs"] for entry in started} == {DEFAULT_MAX_LOGPROBS}
    assert len({entry["log_path"] for entry in started}) == 4
    cache_directories = [path for entry in started for path in entry["cache_directories"].values()]
    assert len(cache_directories) == len(set(cache_directories)) == 12
    assert all(path.is_dir() and not path.is_symlink() for path in cache_directories)
    assert set(observed_urls) == {
        "http://127.0.0.1:9100/v1",
        "http://127.0.0.1:9101/v1",
        "http://127.0.0.1:9102/v1",
        "http://127.0.0.1:9103/v1",
    }
    assert set(stopped) == {"0", "1", "2", "3"}


def test_runtime_parity_parallel_server_process_receives_one_device_and_its_cache_dirs(monkeypatch, tmp_path: Path):
    captured: dict = {}

    class FakeProcess:
        pass

    def fake_popen(command, *, stdout, stderr, env):
        captured["command"] = command
        captured["environment"] = env
        return FakeProcess()

    cache_directories = {}
    for environment_name, directory_name in runtime_parity.PARALLEL_VLLM_CACHE_ENVIRONMENT:
        path = tmp_path / directory_name
        path.mkdir()
        cache_directories[environment_name] = path
    monkeypatch.setattr(runtime_parity.shutil, "which", lambda _: "/opt/venv/bin/vllm")
    monkeypatch.setattr(runtime_parity.subprocess, "Popen", fake_popen)
    process = runtime_parity._start_vllm(
        model=MODEL,
        port=9300,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        max_loras=1,
        log_path=tmp_path / "server.log",
        enforce_eager=True,
        gdn_prefill_backend="triton",
        device_token="GPU-unique-token",
        cache_directories=cache_directories,
    )

    assert captured["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-unique-token"
    assert {name: captured["environment"][name] for name in cache_directories} == {
        name: str(path) for name, path in cache_directories.items()
    }
    assert "--max-loras" in captured["command"]
    assert captured["command"][captured["command"].index("--max-loras") + 1] == "1"
    assert captured["command"][captured["command"].index("--max-logprobs") + 1] == str(DEFAULT_MAX_LOGPROBS)
    assert captured["command"][captured["command"].index("--logprobs-mode") + 1] == VLLM_LOGPROBS_MODE
    getattr(process, "_ctm_log_handle").close()


def test_runtime_parity_scores_with_exact_allowed_token_transport(monkeypatch):
    requests: list[dict] = []

    def fake_http(method, url, payload=None, *, timeout=60.0):
        assert method == "POST"
        assert url == "http://127.0.0.1:8789/v1/completions"
        assert timeout == 60.0
        assert payload is not None
        requests.append(payload)
        return {
            "choices": [
                {
                    "logprobs": {
                        "top_logprobs": [
                            {"token_id:7": -0.25, "token_id:11": -1.5}
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(runtime_parity, "_http_json", fake_http)
    scores = _vllm_scores(
        base_url="http://127.0.0.1:8789/v1",
        model_name="parity-adapter",
        prompt_token_ids=[[1, 2, 3]],
        token_ids=[[7, 11, 7]],
    )

    assert scores == [{7: -0.25, 11: -1.5}]
    assert requests == [
        {
            "model": "parity-adapter",
            "prompt": [1, 2, 3],
            **VLLM_PARITY_SAMPLING,
            "logprobs": 2,
            "allowed_token_ids": [7, 11],
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
        }
    ]
    assert "logprob_token_ids" not in requests[0]


def test_runtime_parity_rejects_vllm_logprobs_outside_the_allowed_token_set(monkeypatch):
    def fake_http(*_args, **_kwargs):
        return {
            "choices": [
                {
                    "logprobs": {
                        "top_logprobs": [
                            {"token_id:7": -0.25, "token_id:11": -1.5, "token_id:99": -4.0}
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(runtime_parity, "_http_json", fake_http)
    with pytest.raises(RuntimeError, match=r"exactly the requested logprobs.*unexpected=\[99\]"):
        _vllm_scores(
            base_url="http://127.0.0.1:8789/v1",
            model_name="parity-adapter",
            prompt_token_ids=[[1, 2, 3]],
            token_ids=[[7, 11]],
        )


def test_runtime_parity_score_transport_contract_is_frozen():
    assert VLLM_SCORE_TRANSPORT == "allowed_token_ids_restricted_softmax"
    assert VLLM_PARITY_SAMPLING == {
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_tokens": 0,
        "ignore_eos": True,
    }


def test_runtime_parity_parallel_failure_signals_all_sibling_servers(monkeypatch, tmp_path: Path):
    plan = _parallel_isolated_vllm_plan(
        requested_result_variants=RESULT_VARIANTS,
        device_tokens=("0", "1", "2", "3"),
        vllm_port=9200,
        isolate_vllm_variants=True,
        parallel_isolated_vllm_variants=True,
        enforce_eager=True,
    )
    assert plan is not None
    stopped: list[str] = []
    signalled: list[str] = []
    barrier = threading.Barrier(len(RESULT_VARIANTS), timeout=5)

    class FakeProcess:
        def __init__(self, token: str):
            self.token = token

    def fake_start(**kwargs):
        return FakeProcess(kwargs["device_token"])

    def fake_stop(process):
        stopped.append(process.token)

    def fake_signal(process):
        signalled.append(process.token)

    def failing_probe(**kwargs):
        barrier.wait()
        try:
            if kwargs["variant"] == "full":
                raise RuntimeError("deliberate isolated-server failure")
            return {
                "base_name": "base",
                "model_ids": ["base", kwargs["runtime_adapter_name"]],
                "base_scores": [{1: 0.0}],
                "scores": [{1: 1.0}],
                "variant": kwargs["variant"],
            }
        finally:
            runtime_parity._stop_process(kwargs["process"])

    monkeypatch.setattr(runtime_parity, "_start_vllm", fake_start)
    monkeypatch.setattr(runtime_parity, "_stop_process", fake_stop)
    monkeypatch.setattr(runtime_parity, "_signal_process_termination", fake_signal)
    monkeypatch.setattr(runtime_parity, "_probe_isolated_vllm_server", failing_probe)
    with pytest.raises(RuntimeError, match="deliberate isolated-server failure"):
        runtime_parity._run_parallel_isolated_vllm_variants(
            model_name=MODEL,
            root=tmp_path,
            plan=plan,
            runtime_adapter_names={variant: f"adapter-{variant}" for variant in RESULT_VARIANTS},
            runtime_adapter_paths={variant: f"/adapters/{variant}" for variant in RESULT_VARIANTS},
            prompt_token_ids=[[1]],
            token_ids=[[2]],
            max_model_len=32768,
            vllm_memory_utilization=0.9,
            enforce_eager=True,
            gdn_prefill_backend="triton",
        )

    assert set(stopped) == {"0", "1", "2", "3"}
    assert set(signalled) == {"0", "1", "2", "3"}


def test_runtime_parity_cli_passes_explicit_parallel_device_tokens(monkeypatch, tmp_path: Path):
    captured: dict = {}

    def fake_run_probe(**kwargs):
        captured.update(kwargs)
        return {
            "results": {
                variant: {"verdict": "hf_vllm_effects_agree"}
                for variant in RESULT_VARIANTS
            }
        }

    monkeypatch.setattr(runtime_parity, "run_probe", fake_run_probe)
    runtime_parity.main(
        [
            "--model",
            MODEL,
            "--adapter",
            str(tmp_path / "adapter"),
            "--data",
            str(tmp_path / "data.jsonl"),
            "--output-dir",
            str(tmp_path / "output"),
            "--isolate-vllm-variants",
            "--parallel-isolated-vllm-variants",
            "--vllm-device-tokens",
            "0",
            "1",
            "2",
            "3",
        ]
    )

    assert captured["isolate_vllm_variants"] is True
    assert captured["parallel_isolated_vllm_variants"] is True
    assert captured["vllm_device_tokens"] == ["0", "1", "2", "3"]
    assert captured["max_logprobs"] == DEFAULT_MAX_LOGPROBS


def test_act_repair_training_is_paper_like_and_has_an_actual_4000_step_budget():
    config = experiment.load_experiment(PLAN)
    training = config["training"]

    assert len(training) == 1
    run = training[0]
    args = run["args"]
    assert run["name"] == "act_repair"
    assert run["target"] == "act"
    assert run["resource"] == "gpu"
    assert run["gpu_count"] == 1
    assert args["backend"] == "local"
    assert args["local_sampler"] == "hf"
    assert args["local_gradient_checkpointing"] is True
    assert args["local_forward_microbatch_max_datums"] == 1
    assert args["local_forward_microbatch_max_tokens"] == 20480
    assert args["model"] == "${model}"
    assert args["method"] == "act"
    assert args["data"] == ["${artifact_root}/data/canonical-train-eval-n200.jsonl:200"]
    assert args["data_manifest"] == [
        "${artifact_root}/data/canonical-train-eval-n200.manifest.json"
    ]
    assert args["reference_messages_field"] == "unbiased_messages"
    assert args["variant_messages_field"] == "biased_messages"
    assert "alignment_text_field" not in args
    assert args["qwen35_consistency_preflight"] is True
    assert args["method_config"] == {
        "weight": 0.00005,
        "layer_selection": "all",
        "normalize": False,
    }
    assert args["lora_config"] == {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ATTENTION_TARGETS,
        "train_mlp": False,
        "train_attn": False,
        "train_unembed": False,
        "seed": 42,
    }
    assert args["optimizer_config"] == {
        "learning_rate": 0.0001,
        "lr_schedule": "constant",
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 0.00000001,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
    }
    assert args["batch_size"] == 1
    assert args["gradient_accumulation_steps"] == 1
    assert args["epochs"] == 20
    assert args["minimum_optimizer_steps"] == 4000
    assert 200 * args["epochs"] // args["gradient_accumulation_steps"] == 4000


def test_gate_has_four_prompt_split_cells_per_checkpoint_with_shared_clean_logs():
    config = experiment.load_experiment(PLAN)
    evaluations = _by_name(config["evaluation"])
    assert list(evaluations) == [
        "untrained_canonical_train_eval",
        "untrained_native_train_eval",
        "untrained_canonical_heldout_in_domain",
        "untrained_native_heldout_in_domain",
        "act_hf_tiny_behavior_report",
        "attest_act_hf_tiny_behavior",
        "act_canonical_train_eval",
        "act_native_train_eval",
        "act_canonical_heldout_in_domain",
        "act_native_heldout_in_domain",
    ]

    for condition, target in (("untrained", "preflight"), ("act", "act")):
        for split, canonical_file in (
            ("train_eval", "canonical-train-eval-n200.jsonl"),
            ("heldout_in_domain", "canonical-heldout-in-domain-n200.jsonl"),
        ):
            canonical = evaluations[f"{condition}_canonical_{split}"]
            native = evaluations[f"{condition}_native_{split}"]
            canonical_args = canonical["args"]
            native_args = native["args"]
            canonical_log = f"logs/evals/${{experiment}}/{condition}/canonical/{split}"

            assert canonical["target"] == native["target"] == target
            assert canonical["resource"] == native["resource"] == "gpu"
            assert canonical["gpu_count"] == native["gpu_count"] == 1
            assert canonical_args["task_factory"].endswith(":diagnostic_tasks")
            assert native_args["task_factory"].endswith(":diagnostic_biased_tasks")
            assert canonical_args["task_args"] == {
                "manifest": "${artifact_root}/data/manifest.json",
                "split": split,
                "unbiased_log": canonical_log,
                "variant_file": f"${{artifact_root}}/data/{canonical_file}",
                "include_bias_acknowledged": False,
            }
            assert native_args["task_args"] == {
                "manifest": "${artifact_root}/data/manifest.json",
                "split": split,
                "unbiased_log": canonical_log,
                "include_bias_acknowledged": False,
            }
            assert canonical_args["log_dir"] == canonical_log
            assert native_args["log_dir"] == f"logs/evals/${{experiment}}/{condition}/native/{split}"
            assert canonical_args["generation_config"] == native_args["generation_config"] == GENERATION
            for args in (canonical_args, native_args):
                assert args["max_tasks"] == 1
                assert args["isolate_tasks"] is True
                assert args["persistent_vllm_server"] is True

    untrained = evaluations["untrained_canonical_train_eval"]["args"]
    trained = evaluations["act_canonical_train_eval"]["args"]
    assert untrained["model"] == "vllm/${model}"
    assert "local_checkpoint" not in untrained
    assert trained["local_checkpoint"] == "${checkpoint}"
    assert trained["base_model"] == "${model}"
    assert trained["model_args"]["provider"] == "vllm"

    tiny_report = evaluations["act_hf_tiny_behavior_report"]
    assert tiny_report == {
        "name": "act_hf_tiny_behavior_report",
        "target": "act",
        "resource": "gpu",
        "gpu_count": 1,
        "command": ["${python}", "-m", "experiments.act_repair_gate.direct_answer"],
        "args": {
            "model": "${model}",
            "adapter": "${checkpoint}",
            "train_data": "${artifact_root}/data/canonical-train-eval-n200.jsonl",
            "heldout_data": "${artifact_root}/data/canonical-heldout-in-domain-n200.jsonl",
            "output_dir": "${artifact_root}/tiny-hf-behavioral-gate/direct-answer",
            "condition_name": "act",
            "limit_per_dataset": 4,
        },
    }
    tiny_attestation = evaluations["attest_act_hf_tiny_behavior"]
    assert tiny_attestation == {
        "name": "attest_act_hf_tiny_behavior",
        "target": "act",
        "resource": "cpu",
        "command": ["${python}", "-m", "experiments.act_repair_gate.behavioral_gate"],
        "args": {
            "report": "${artifact_root}/tiny-hf-behavioral-gate/direct-answer/report.json",
            "adapter": "${checkpoint}",
            "train_data": "${artifact_root}/data/canonical-train-eval-n200.jsonl",
            "heldout_data": "${artifact_root}/data/canonical-heldout-in-domain-n200.jsonl",
            "output": "${artifact_root}/tiny-hf-behavioral-gate/attestation.json",
            "gate_split": "train_eval",
            "min_base_switches": 1,
            "expected_limit_per_dataset": 4,
            "require_pass": True,
        },
    }

    analyses = _by_name(config["analysis"])
    assert analyses == {
        "analyze_untrained_raw_gate": {
            "name": "analyze_untrained_raw_gate",
            "target": "preflight",
            "resource": "cpu",
            "command": [
                "${python}",
                "-m",
                "experiments.stage1_iid_diagnostic.gate_analysis",
            ],
            "args": {
                "native_root": "logs/evals/${experiment}/untrained/native",
                "canonical_root": "logs/evals/${experiment}/untrained/canonical",
                "condition": "untrained",
                "output": "${artifact_root}/analysis/untrained-raw.json",
            },
        },
        "analyze_act_raw_gate": {
            "name": "analyze_act_raw_gate",
            "target": "act",
            "resource": "cpu",
            "command": [
                "${python}",
                "-m",
                "experiments.stage1_iid_diagnostic.gate_analysis",
            ],
            "args": {
                "native_root": "logs/evals/${experiment}/act/native",
                "canonical_root": "logs/evals/${experiment}/act/canonical",
                "condition": "act",
                "output": "${artifact_root}/analysis/act-raw.json",
            },
        },
    }

    renderings = _by_name(config["rendering"])
    assert renderings == {
        "render_publication_style_tbsr": {
            "name": "render_publication_style_tbsr",
            "target": "act",
            "resource": "cpu",
            "command": ["${python}", "-m", "experiments.act_repair_gate.plot"],
            "args": {
                "untrained_report": "${artifact_root}/analysis/untrained-raw.json",
                "act_report": "${artifact_root}/analysis/act-raw.json",
                "output_dir": "${artifact_root}/figures",
            },
        }
    }


def test_all_commands_render_strictly_with_one_explicit_checkpoint():
    config = experiment.load_experiment(PLAN)
    context = experiment.initial_context(config, checkpoint="file:///checkpoints/act-repair")
    planned = experiment.planned_commands(
        config,
        ["data_preparation", "training", "evaluation", "analysis", "rendering"],
        context,
        strict=True,
    )

    assert len(planned) == 15
    assert planned[0][0:2] == ("data_preparation", "prepare_remote_local_gate_data")
    assert planned[1][0:2] == ("training", "act_repair")
    assert planned[6][0:2] == ("evaluation", "act_hf_tiny_behavior_report")
    assert planned[7][0:2] == ("evaluation", "attest_act_hf_tiny_behavior")
    assert planned[-3][0:2] == ("analysis", "analyze_untrained_raw_gate")
    assert planned[-2][0:2] == ("analysis", "analyze_act_raw_gate")
    assert planned[-1][0:2] == ("rendering", "render_publication_style_tbsr")
    assert all("${" not in token for _, _, argv in planned for token in argv)
    assert planned[3][2][0] == sys.executable
    assert "file:///checkpoints/act-repair" in planned[6][2]
    assert "file:///checkpoints/act-repair" in planned[7][2]
