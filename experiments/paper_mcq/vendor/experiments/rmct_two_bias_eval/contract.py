"""Fail-closed provenance contract for evaluating the converged RMCT r4 run.

This module intentionally *reads* the existing immutable Stage 2 OOD-HLE
substrate.  It never materializes, copies, rewrites, or relabels those files.
The historical substrate was organised around one training bias; this run was
trained with two.  Therefore its legacy ``regime`` value is retained as a
file-layout/provenance field only, while scientific seen-vs-held-out status is
derived afresh from the named bias type in one place below.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

# This is deliberately the exact offline snapshot recorded by the sealed r4
# checkpoint, not the mutable Hugging Face model alias. A deployment receipt
# must reject a superficially compatible adapter trained against another
# snapshot or revision.
BASE_MODEL = (
    "/lus/lfs1aip2/scratch/a5v/sohaib.a5v/ctm/huggingface/hub/"
    "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
MODEL_ALIAS = "Qwen/Qwen3.5-9B"
TASK_FACTORY = "experiments.stage2_ood_hle.tasks:ood_tasks"

SUBSTRATE_SCHEMA = "rmct-two-bias-stage2-substrate-v1"
EVALUATION_RECEIPT_SCHEMA = "rmct-two-bias-evaluation-receipt-v1"

# Keep the fixed Stage 2 order for reproducible reporting.  The first two are
# exactly the interventions used by the converged r4 training run.
SEEN_BIASES = ("wrong_argument", "suggested_answer")
HELD_OUT_BIASES = (
    "distractor_fact",
    "post_hoc",
    "spurious_few_shot_squares",
    "wrong_few_shot",
)
ALL_BIASES = (*SEEN_BIASES, *HELD_OUT_BIASES)

EXPECTED_TASKS = 21
EXPECTED_CLEAN_TASKS = 3
EXPECTED_BIASED_TASKS = 18
EXPECTED_BIASED_TASKS_PER_BIAS = 3
PROMPT_STYLE = "none"

R4_RUN_PREFIX = "rmct-convergence-gcall-r2-mb40960-r4"
R4_FINAL_SEGMENT_INDEX = 10
R4_FINAL_OPTIMIZER_STEP = 176
R4_FINAL_CHECKPOINT_NAME = f"rmct-convergence_{R4_RUN_PREFIX}-s011"

# The vLLM route is allowed only as the parity-attested translated-adapter
# route.  Keep these knobs in the receipt as well as in the launcher so a raw
# log cannot be reinterpreted as an equivalent run with a different serving
# configuration.
VLLM_GENERATION_CONFIG = {
    "max_tokens": 20480,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "extra_body": {"top_k": 20},
}
VLLM_GDN_PREFILL_BACKEND = "triton"
VLLM_MODEL_ARGS = {
    "provider": "vllm",
    "gpu_memory_utilization": 0.9,
    "max_model_len": 32768,
    "language_model_only": True,
    "max_num_seqs": 256,
    "gdn_prefill_backend": VLLM_GDN_PREFILL_BACKEND,
}
VLLM_VERSION = "0.21.0"
VLLM_PARITY_TOP_TOKEN_COUNT = 16
VLLM_PARITY_MAX_LOGPROBS = 29
VLLM_PARITY_LOGPROBS_MODE = "processed_logprobs"
VLLM_PARITY_SCORE_TRANSPORT = "allowed_token_ids_restricted_softmax"
VLLM_PARITY_SAMPLING = {
    "max_tokens": 1,
    "min_tokens": 0,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "ignore_eos": True,
}
VLLM_SAMPLER_ENVIRONMENT = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
VLLM_SAMPLER_RUNTIME = {
    "vllm_version": VLLM_VERSION,
    "environment": dict(VLLM_SAMPLER_ENVIRONMENT),
    "implementation": "pytorch_native",
    "parity_gdn_prefill_backend": VLLM_GDN_PREFILL_BACKEND,
    "parity_top_token_count": VLLM_PARITY_TOP_TOKEN_COUNT,
    "parity_max_logprobs": VLLM_PARITY_MAX_LOGPROBS,
    "parity_logprobs_mode": VLLM_PARITY_LOGPROBS_MODE,
    "parity_score_transport": VLLM_PARITY_SCORE_TRANSPORT,
    "parity_sampling": dict(VLLM_PARITY_SAMPLING),
    "persistent_gdn_prefill_backend": VLLM_GDN_PREFILL_BACKEND,
}
_VLLM_COMPATIBILITY_MANIFEST = "compatibility-manifest.json"
_VLLM_PARITY_ATTESTATION = "vllm-parity-attestation.json"
_VLLM_SOURCE_PREFIX = "base_model.model.model.layers."
_VLLM_DESTINATION_PREFIX = "base_model.model.model.language_model.layers."
_R003_PARITY_REPORT_SCHEMA = "qwen35-lora-runtime-parity-v1"
_R003_PARITY_VARIANTS = ("full", "linear_only", "self_attn_only", "evaluator_path")
_R003_PARITY_CACHE_ENVIRONMENT = (
    "XDG_CACHE_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
)
_R004_PARITY_SERVER_KEYS = frozenset(
    {
        "device_token",
        "port",
        "server_log",
        "max_loras",
        "max_logprobs",
        "cache_directories",
    }
)
_R005_PARITY_SERVER_KEYS = _R004_PARITY_SERVER_KEYS | {"logprobs_mode"}

_HEX = frozenset("0123456789abcdef")


class EvaluationContractError(ValueError):
    """The requested evaluation cannot be tied to the converged r4 state."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationContractError(f"{label} must contain a JSON object: {path}")
    return value


def _local_path(value: str | Path, *, label: str) -> Path:
    raw = str(value)
    parsed = urlsplit(raw)
    if parsed.scheme:
        if parsed.scheme.lower() != "file" or parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise EvaluationContractError(f"{label} must be a local path or file URI")
        raw = unquote(parsed.path)
    if not raw:
        raise EvaluationContractError(f"{label} must be non-empty")
    return Path(raw).expanduser().resolve()


def _safe_component(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise EvaluationContractError(f"{label} must be one non-empty path component")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationContractError(f"{label} must be a positive integer")
    return value


def bias_status(bias_type: str) -> str:
    """Return the scientific status of one fixed Stage 2 intervention."""

    if bias_type in SEEN_BIASES:
        return "seen"
    if bias_type in HELD_OUT_BIASES:
        return "held_out"
    raise EvaluationContractError(f"unexpected Stage 2 bias type {bias_type!r}")


def _spec_digest(specs: Sequence[Any]) -> str:
    """Hash the exact task selection without introducing a new data artifact."""

    rows = [
        {
            "kind": spec.kind,
            "regime": spec.regime,
            "population": spec.population,
            "dataset": spec.dataset,
            "bias_type": spec.bias_type,
            "frozen_file": str(Path(spec.frozen_file).resolve()),
            "question_ids": list(spec.question_ids),
            "source_identity_digest": spec.source_identity_digest,
        }
        for spec in specs
    ]
    return hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_stage2_substrate(manifest: str | Path) -> dict[str, Any]:
    """Validate and describe the unchanged frozen 3-clean / 18-biased suite.

    ``experiments.stage2_ood_hle`` remains the sole owner of the frozen data
    and task factory.  Calling this function only verifies those bytes and
    produces a read-only description with the corrected two-bias science
    labels.
    """

    manifest_path = _local_path(manifest, label="Stage 2 manifest")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"Stage 2 manifest must be a regular file: {manifest_path}")
    try:
        from experiments.rmct_two_bias_eval.deployment import validate_deployment_manifest
        from experiments.stage2_ood_hle.materialize import validate_manifest
        from experiments.stage2_ood_hle.tasks import ood_task_specs
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise RuntimeError("the frozen Stage 2 task substrate is unavailable") from exc

    # The materializer validates every frozen artifact (including hashes and
    # shared IDs); the task factory re-reads the selected artifacts afterwards.
    validate_manifest(manifest_path)
    deployment_provenance = validate_deployment_manifest(manifest_path)
    specs = list(ood_task_specs(manifest_path))
    clean = [spec for spec in specs if spec.kind == "unbiased"]
    biased = [spec for spec in specs if spec.kind == "biased"]
    if (len(specs), len(clean), len(biased)) != (EXPECTED_TASKS, EXPECTED_CLEAN_TASKS, EXPECTED_BIASED_TASKS):
        raise EvaluationContractError("Stage 2 substrate must resolve to exactly 3 clean plus 18 biased tasks")
    if any(spec.kind not in {"unbiased", "biased"} for spec in specs):
        raise EvaluationContractError("Stage 2 substrate contains an unsupported task kind")
    if any(spec.bias_type is not None for spec in clean):
        raise EvaluationContractError("Stage 2 clean tasks unexpectedly declare a bias")

    by_bias: dict[str, list[Any]] = defaultdict(list)
    for spec in biased:
        if not isinstance(spec.bias_type, str):
            raise EvaluationContractError("Stage 2 biased task has no named bias")
        if spec.bias_type not in ALL_BIASES:
            raise EvaluationContractError(f"Stage 2 substrate has an unsupported bias {spec.bias_type!r}")
        by_bias[spec.bias_type].append(spec)
    if tuple(by_bias) != ALL_BIASES:
        # ``ood_task_specs`` has a meaningful stable execution order.  Refuse
        # a reordered/substituted substrate instead of normalising it here.
        raise EvaluationContractError(
            f"Stage 2 bias order differs from the frozen six-bias contract: {tuple(by_bias)!r}"
        )
    if any(len(by_bias[bias]) != EXPECTED_BIASED_TASKS_PER_BIAS for bias in ALL_BIASES):
        raise EvaluationContractError("every Stage 2 bias must retain its three frozen task cells")

    clean_populations = {(spec.population, spec.dataset) for spec in clean}
    if len(clean_populations) != EXPECTED_CLEAN_TASKS:
        raise EvaluationContractError("Stage 2 substrate must retain three distinct clean reference tasks")
    for bias, bias_specs in by_bias.items():
        paired_populations = {(spec.population, spec.dataset) for spec in bias_specs}
        if paired_populations != clean_populations:
            raise EvaluationContractError(f"Stage 2 bias {bias!r} does not use every shared clean population")

    bias_records = [
        {
            "bias_type": bias,
            "evaluation_bias_status": bias_status(bias),
            "biased_task_count": len(by_bias[bias]),
            # Retain legacy layout labels as provenance only.  Downstream
            # analysis must classify by ``evaluation_bias_status`` instead.
            "substrate_regimes": [spec.regime for spec in by_bias[bias]],
        }
        for bias in ALL_BIASES
    ]
    return {
        "schema": SUBSTRATE_SCHEMA,
        "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
        "task_factory": TASK_FACTORY,
        "task_count": EXPECTED_TASKS,
        "clean_task_count": EXPECTED_CLEAN_TASKS,
        "biased_task_count": EXPECTED_BIASED_TASKS,
        "prompt_style": PROMPT_STYLE,
        "all_biases": list(ALL_BIASES),
        "seen_biases": list(SEEN_BIASES),
        "held_out_biases": list(HELD_OUT_BIASES),
        "biases": bias_records,
        "clean_before_biased_task_order": True,
        "task_selection_sha256": _spec_digest(specs),
        "legacy_regime_policy": "provenance_only_not_scientific_bias_status",
        "frozen_data_policy": "validate_read_only_no_materialization_or_mutation",
        "deployment_provenance": deployment_provenance,
    }


def _required_regular_file(directory: Path, name: str) -> Path:
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise FileNotFoundError(f"converged checkpoint requires a non-empty regular {name}: {path}")
    return path


def checkpoint_identity(checkpoint: str | Path) -> dict[str, Any]:
    """Read and hash-bind all artifacts needed to identify a resumable r4 checkpoint."""

    directory = _local_path(checkpoint, label="checkpoint")
    if directory.is_symlink() or not directory.is_dir():
        raise FileNotFoundError(f"checkpoint must be a regular directory: {directory}")
    try:
        from ctm.evals.local_model import read_local_checkpoint
        from ctm.training.resume_state import load_strict_local_rl_resume_state
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise RuntimeError("local RMCT checkpoint support is unavailable") from exc

    resolved, manifest = read_local_checkpoint(directory)
    if resolved != directory or manifest.get("backend") != "local" or manifest.get("lora") is not True:
        raise EvaluationContractError("evaluation requires a local LoRA checkpoint")
    if manifest.get("model") != BASE_MODEL or manifest.get("kind") != "both":
        raise EvaluationContractError("evaluation requires a local kind='both' checkpoint for the sealed Qwen3.5 snapshot")
    state = load_strict_local_rl_resume_state(directory)
    if state.checkpoint_dir != directory:
        raise EvaluationContractError("checkpoint resume-state path differs from requested checkpoint")

    files = {
        "adapter_config_sha256": _required_regular_file(directory, "adapter_config.json"),
        "adapter_model_sha256": _required_regular_file(directory, "adapter_model.safetensors"),
        "optimizer_sha256": _required_regular_file(directory, "optimizer.pt"),
        "manifest_sha256": _required_regular_file(directory, "manifest.json"),
        "replicated_training_manifest_sha256": _required_regular_file(directory, "replicated_training_manifest.json"),
        "replicated_training_rng_sha256": _required_regular_file(directory, "replicated_training_rng.pt"),
    }
    adapter_config = _read_json_object(files["adapter_config_sha256"], label="checkpoint adapter configuration")
    if adapter_config.get("base_model_name_or_path") != BASE_MODEL:
        raise EvaluationContractError("checkpoint adapter configuration does not bind the sealed Qwen3.5 snapshot")
    _read_json_object(files["replicated_training_manifest_sha256"], label="replicated-training manifest")
    return {
        "path": str(directory),
        "uri": f"file://{directory}",
        "checkpoint_name": directory.name,
        "backend": "local",
        "lora": True,
        "base_model": BASE_MODEL,
        "kind": "both",
        "global_step": state.global_step,
        "optimizer_step": state.optimizer_step,
        "completed_epochs": state.completed_epochs,
        "resume_state_required": True,
        "replicated_rng_state_required": True,
        **{field: _sha256_file(path) for field, path in files.items()},
    }


def _decision_identity(path: str | Path) -> dict[str, Any]:
    """Replay and bind the final converged controller receipt and its chain."""

    receipt_path = _local_path(path, label="convergence decision receipt")
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise FileNotFoundError(f"convergence decision receipt must be a regular file: {receipt_path}")
    try:
        from experiments.rmct_convergence.controller import verify_decision_receipt
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise RuntimeError("RMCT convergence-controller verification is unavailable") from exc
    document = verify_decision_receipt(receipt_path)
    target = document.get("target")
    afterok = document.get("afterok")
    if not isinstance(target, Mapping) or not isinstance(afterok, Mapping):
        raise EvaluationContractError("verified convergence receipt lacks target/afterok")
    if document.get("decision") != "converged" or dict(afterok) != {"permit_training": False, "successor_action": "no_op"}:
        raise EvaluationContractError("evaluation requires the terminal converged RMCT controller receipt")
    if (
        target.get("segment_index") != R4_FINAL_SEGMENT_INDEX
        or target.get("optimizer_step_end") != R4_FINAL_OPTIMIZER_STEP
        or target.get("optimizer_step_start") != R4_FINAL_OPTIMIZER_STEP - 15
        or target.get("optimizer_steps") != 16
    ):
        raise EvaluationContractError("convergence receipt is not the sealed r4 s011 / step-176 decision")
    window = document.get("window")
    if not isinstance(window, Mapping):
        raise EvaluationContractError("convergence receipt lacks its convergence window")
    return {
        "path": str(receipt_path),
        "sha256": _sha256_file(receipt_path),
        "decision": "converged",
        "afterok": {"permit_training": False, "successor_action": "no_op"},
        "segment_index": R4_FINAL_SEGMENT_INDEX,
        "optimizer_step": R4_FINAL_OPTIMIZER_STEP,
        "window": dict(window),
    }


def _absolute_file_identity(value: Any, *, label: str) -> dict[str, Any]:
    """Resolve and hash one attestation-referenced regular file."""

    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise EvaluationContractError(f"{label} path must be an absolute path")
    path = Path(value).resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
        raise FileNotFoundError(f"{label} must be a non-empty regular file: {path}")
    return {"path": str(path), "sha256": _sha256_file(path)}


def validate_r004_parallel_parity_report(
    report: str | Path,
    *,
    compatibility_adapter: str | Path,
) -> dict[str, Any]:
    """Validate r004's four-server native-vLLM parity topology.

    The normal Qwen3.5 attestation verifier establishes that an adapter and
    parity report are mutually hash-bound.  r003 additionally needs proof
    that all four diagnostic variants were run in independent one-LoRA vLLM
    servers.  This validator is intentionally report-schema based, so the
    established attestation schema remains unchanged.  The r004 retry adds an
    exact 29-entry logprob ceiling after r003 stopped before numerical
    comparison at vLLM's default limit of 20.  Twenty-nine is the fail-closed
    frozen request bound: 16 top-token entries plus the pinned tokenizer's 13
    unique A/B/C/D candidate token IDs (the newline token is shared).
    """

    report_path = _local_path(report, label="r004 vLLM parity report")
    document = _read_json_object(report_path, label="r004 vLLM parity report")
    adapter = _local_path(compatibility_adapter, label="vLLM compatibility adapter")
    if document.get("schema") != _R003_PARITY_REPORT_SCHEMA or document.get("model") != BASE_MODEL:
        raise EvaluationContractError("r004 vLLM parity report does not bind the sealed runtime schema/model")
    report_adapter = document.get("adapter")
    if not isinstance(report_adapter, Mapping) or report_adapter.get("path") != str(adapter):
        raise EvaluationContractError("r004 vLLM parity report does not name the served compatibility adapter")
    protocol = document.get("token_protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("requested_result_variants") != list(_R003_PARITY_VARIANTS)
        or protocol.get("top_token_count") != VLLM_PARITY_TOP_TOKEN_COUNT
    ):
        raise EvaluationContractError("r004 parity report lacks the canonical four-variant protocol")
    results = document.get("results")
    if not isinstance(results, Mapping) or set(results) != set(_R003_PARITY_VARIANTS):
        raise EvaluationContractError("r004 parity report lacks exactly four result variants")

    backends = document.get("backends")
    vllm = backends.get("vllm") if isinstance(backends, Mapping) else None
    if (
        not isinstance(vllm, Mapping)
        or vllm.get("isolate_vllm_variants") is not True
        or vllm.get("parallel_isolated_vllm_variants") is not True
        or vllm.get("enforce_eager") is not True
        or vllm.get("gdn_prefill_backend") != VLLM_GDN_PREFILL_BACKEND
        or vllm.get("max_logprobs") != VLLM_PARITY_MAX_LOGPROBS
    ):
        raise EvaluationContractError("r004 parity report lacks the pinned isolated native-vLLM runtime")

    plan = vllm.get("parallel_isolated_server_plan")
    names = vllm.get("runtime_adapter_names")
    model_ids = vllm.get("model_ids_after_load")
    if not isinstance(plan, Mapping) or set(plan) != set(_R003_PARITY_VARIANTS):
        raise EvaluationContractError("r004 parity report lacks the canonical four-server plan")
    if not isinstance(names, Mapping) or set(names) != set(_R003_PARITY_VARIANTS):
        raise EvaluationContractError("r004 parity report lacks all four runtime adapter identities")
    if not isinstance(model_ids, Mapping) or set(model_ids) != set(_R003_PARITY_VARIANTS):
        raise EvaluationContractError("r004 parity report lacks per-server model identities")

    expected_names = {
        "full": "full",
        "linear_only": "linear_only",
        "self_attn_only": "self_attn_only",
        "evaluator_path": str(adapter),
    }
    if dict(names) != expected_names:
        raise EvaluationContractError("r004 parity report runtime adapter identities differ from the exact four probes")

    root = report_path.parent.resolve()
    device_tokens: set[str] = set()
    ports: set[int] = set()
    logs: set[Path] = set()
    cache_directories: set[Path] = set()
    for variant in _R003_PARITY_VARIANTS:
        server = plan[variant]
        if not isinstance(server, Mapping) or frozenset(server) not in {
            _R004_PARITY_SERVER_KEYS,
            _R005_PARITY_SERVER_KEYS,
        }:
            raise EvaluationContractError(f"r004 parity server plan is malformed for {variant!r}")
        token = server.get("device_token")
        port = server.get("port")
        raw_log = server.get("server_log")
        if (
            not isinstance(token, str)
            or not token
            or token != token.strip()
            or any(character.isspace() for character in token)
            or "," in token
            or isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
            or server.get("max_loras") != 1
            or server.get("max_logprobs") != VLLM_PARITY_MAX_LOGPROBS
            or not isinstance(raw_log, str)
            or not raw_log
            or not Path(raw_log).is_absolute()
        ):
            raise EvaluationContractError(f"r004 parity server plan has invalid isolation settings for {variant!r}")
        log = Path(raw_log).resolve()
        try:
            log.relative_to(root)
        except ValueError as exc:
            raise EvaluationContractError(f"r004 parity server log escapes its report root: {variant!r}") from exc
        if log.is_symlink() or not log.is_file() or log.stat().st_size < 1:
            raise EvaluationContractError(f"r004 parity server log is not a non-empty regular file: {variant!r}")
        cache = server.get("cache_directories")
        if not isinstance(cache, Mapping) or set(cache) != set(_R003_PARITY_CACHE_ENVIRONMENT):
            raise EvaluationContractError(f"r004 parity server has an incomplete cache-directory plan: {variant!r}")
        for name in _R003_PARITY_CACHE_ENVIRONMENT:
            raw_cache = cache[name]
            if not isinstance(raw_cache, str) or not raw_cache or not Path(raw_cache).is_absolute():
                raise EvaluationContractError(f"r004 parity server cache path is invalid: {variant!r}/{name}")
            cache_path = Path(raw_cache).resolve()
            try:
                cache_path.relative_to(root)
            except ValueError as exc:
                raise EvaluationContractError(
                    f"r004 parity server cache directory escapes its report root: {variant!r}/{name}"
                ) from exc
            if cache_path.is_symlink() or not cache_path.is_dir():
                raise EvaluationContractError(
                    f"r004 parity server cache directory is not a regular directory: {variant!r}/{name}"
                )
            cache_directories.add(cache_path)
        device_tokens.add(token)
        ports.add(port)
        logs.add(log)
        loaded = model_ids[variant]
        if (
            not isinstance(loaded, list)
            or not loaded
            or any(not isinstance(value, str) or not value for value in loaded)
            or expected_names[variant] not in loaded
        ):
            raise EvaluationContractError(f"r004 parity server did not expose its exact adapter identity: {variant!r}")
    if len(device_tokens) != len(_R003_PARITY_VARIANTS):
        raise EvaluationContractError("r004 parity server plan reuses a device token")
    if len(ports) != len(_R003_PARITY_VARIANTS):
        raise EvaluationContractError("r004 parity server plan reuses a port")
    if len(logs) != len(_R003_PARITY_VARIANTS):
        raise EvaluationContractError("r004 parity server plan reuses a server log")
    if len(cache_directories) != len(_R003_PARITY_VARIANTS) * len(_R003_PARITY_CACHE_ENVIRONMENT):
        raise EvaluationContractError("r004 parity server plan reuses a cache directory")
    return document


def _contains_mapping_key(value: Any, key: str) -> bool:
    """Return whether a nested JSON value contains a forbidden field name."""

    if isinstance(value, Mapping):
        return key in value or any(_contains_mapping_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_mapping_key(item, key) for item in value)
    return False


def validate_r005_parallel_parity_report(
    report: str | Path,
    *,
    compatibility_adapter: str | Path,
) -> dict[str, Any]:
    """Validate r005's restricted-softmax processed-logprob protocol.

    r005 preserves r004's four isolated one-LoRA servers and exact 29-token
    ceiling, but replaces the unsupported ``logprob_token_ids`` request with
    vLLM's real ``allowed_token_ids`` restriction.  Neutral sampling leaves
    the processed distribution as a softmax over exactly that requested set;
    the subset normalizer then cancels in each adapter-minus-base relative
    score.
    """

    document = validate_r004_parallel_parity_report(
        report,
        compatibility_adapter=compatibility_adapter,
    )
    protocol = document["token_protocol"]
    if (
        protocol.get("vllm_score_transport") != VLLM_PARITY_SCORE_TRANSPORT
        or protocol.get("vllm_allowed_token_ids") != "requested_token_ids"
        or protocol.get("vllm_response_token_ids") != "exactly_requested_token_ids"
        or _contains_mapping_key(document, "logprob_token_ids")
    ):
        raise EvaluationContractError("r005 parity report lacks the exact allowed-token score protocol")

    requested = protocol.get("requested_token_ids")
    references = protocol.get("reference_token_ids")
    if (
        not isinstance(requested, list)
        or not requested
        or not isinstance(references, list)
        or len(references) != len(requested)
    ):
        raise EvaluationContractError("r005 parity report lacks aligned requested/reference token IDs")
    for index, (row, reference) in enumerate(zip(requested, references, strict=True)):
        if (
            not isinstance(row, list)
            or not 1 <= len(row) <= VLLM_PARITY_MAX_LOGPROBS
            or any(isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in row)
            or len(set(row)) != len(row)
            or isinstance(reference, bool)
            or not isinstance(reference, int)
            or reference not in row
        ):
            raise EvaluationContractError(f"r005 parity report has an invalid requested token set at row {index}")

    backends = document["backends"]
    vllm = backends["vllm"]
    if (
        vllm.get("logprobs_mode") != VLLM_PARITY_LOGPROBS_MODE
        or vllm.get("score_transport") != VLLM_PARITY_SCORE_TRANSPORT
        or vllm.get("parity_sampling") != VLLM_PARITY_SAMPLING
    ):
        raise EvaluationContractError("r005 parity report lacks the exact processed-logprob runtime protocol")
    plan = vllm["parallel_isolated_server_plan"]
    for variant in _R003_PARITY_VARIANTS:
        server = plan[variant]
        if set(server) != _R005_PARITY_SERVER_KEYS or server.get("logprobs_mode") != VLLM_PARITY_LOGPROBS_MODE:
            raise EvaluationContractError(
                f"r005 parity server does not bind processed logprobs for {variant!r}"
            )
    return document


def validate_r003_parallel_parity_report(
    report: str | Path,
    *,
    compatibility_adapter: str | Path,
) -> dict[str, Any]:
    """Compatibility spelling for callers migrating from the failed r003 attempt."""

    return validate_r004_parallel_parity_report(report, compatibility_adapter=compatibility_adapter)


def _vllm_parity_report_identities(attestation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every parity report authenticated by a verified attestation."""

    try:
        from ctm.evals.qwen35_vllm_attestation import (
            COMPOSITE_PARITY_ATTESTATION_SCHEMA_V1,
            PARITY_ATTESTATION_SCHEMA_V1,
        )
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise RuntimeError("Qwen3.5 vLLM attestation support is unavailable") from exc

    from ctm.evals.qwen35_vllm_scope import ATTESTATION_SCHEMA as SCOPE_ATTESTATION_SCHEMA

    schema = attestation.get("schema")
    records: list[dict[str, Any]] = []
    if schema in (PARITY_ATTESTATION_SCHEMA_V1, SCOPE_ATTESTATION_SCHEMA):
        identity = _absolute_file_identity(attestation.get("report_path"), label="vLLM parity report")
        if attestation.get("report_sha256") != identity["sha256"]:
            raise EvaluationContractError("vLLM parity attestation report hash does not match its report")
        records.append(identity)
    elif schema == COMPOSITE_PARITY_ATTESTATION_SCHEMA_V1:
        for name in ("primary", "amplified_self_attn"):
            stage = attestation.get(name)
            report = stage.get("report") if isinstance(stage, Mapping) else None
            if not isinstance(report, Mapping):
                raise EvaluationContractError(f"composite vLLM attestation lacks its {name} report")
            identity = _absolute_file_identity(report.get("path"), label=f"vLLM {name} parity report")
            if report.get("sha256") != identity["sha256"]:
                raise EvaluationContractError(f"composite vLLM {name} parity report hash does not match its report")
            records.append(identity)
    else:
        raise EvaluationContractError("vLLM parity attestation has an unsupported schema")
    for record in records:
        report = _read_json_object(Path(record["path"]), label="vLLM parity report")
        if report.get("model") != BASE_MODEL:
            raise EvaluationContractError("vLLM parity report does not bind the sealed Qwen3.5 snapshot")
    if not records or len({record["path"] for record in records}) != len(records):
        raise EvaluationContractError("vLLM parity attestation has ambiguous report evidence")
    return records


def vllm_compatibility_identity(
    compatibility_adapter: str | Path,
    *,
    raw_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the sole admissible vLLM adapter route for the sealed r4 LoRA.

    The compatibility directory is not merely named in the receipt: its
    translated weights, source/destination translation manifest, attestation,
    and every report referenced by that attestation are re-opened and
    hash-bound.  ``is_verified_qwen35_vllm_compat_adapter`` follows the full
    parity chain; the explicit records returned here make that chain portable
    and auditable with the evaluation result.
    """

    directory = _local_path(compatibility_adapter, label="vLLM compatibility adapter")
    if directory.is_symlink() or not directory.is_dir():
        raise FileNotFoundError(f"vLLM compatibility adapter must be a regular directory: {directory}")
    try:
        from ctm.evals.qwen35_vllm_attestation import (
            COMPATIBILITY_MANIFEST_SCHEMA,
            is_verified_qwen35_vllm_compat_adapter,
        )
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise RuntimeError("Qwen3.5 vLLM attestation support is unavailable") from exc

    files = {
        "adapter_config_sha256": _required_regular_file(directory, "adapter_config.json"),
        "adapter_model_sha256": _required_regular_file(directory, "adapter_model.safetensors"),
        "compatibility_manifest_sha256": _required_regular_file(directory, _VLLM_COMPATIBILITY_MANIFEST),
        "parity_attestation_sha256": _required_regular_file(directory, _VLLM_PARITY_ATTESTATION),
    }
    adapter_config = _read_json_object(files["adapter_config_sha256"], label="vLLM compatibility adapter configuration")
    from ctm.evals.qwen35_vllm_scope import same_model_snapshot

    if not same_model_snapshot(adapter_config.get("base_model_name_or_path"), BASE_MODEL):
        raise EvaluationContractError("vLLM compatibility adapter does not bind the sealed Qwen3.5 snapshot")
    if _sha256_file(files["adapter_config_sha256"]) != raw_checkpoint.get("adapter_config_sha256"):
        raise EvaluationContractError(
            "vLLM compatibility adapter configuration differs from the sealed raw r4 checkpoint"
        )
    if not is_verified_qwen35_vllm_compat_adapter(directory):
        raise EvaluationContractError("vLLM compatibility adapter lacks a valid immutable Qwen3.5 parity attestation")

    manifest = _read_json_object(files["compatibility_manifest_sha256"], label="vLLM compatibility manifest")
    source = manifest.get("source")
    destination = manifest.get("destination")
    translation = manifest.get("translation")
    if (
        manifest.get("schema") != COMPATIBILITY_MANIFEST_SCHEMA
        or not isinstance(source, Mapping)
        or not isinstance(destination, Mapping)
        or not isinstance(translation, Mapping)
        or source.get("path") != raw_checkpoint.get("path")
        or source.get("adapter_model_sha256") != raw_checkpoint.get("adapter_model_sha256")
        or destination.get("path") != str(directory)
        or destination.get("adapter_model_sha256") != _sha256_file(files["adapter_model_sha256"])
        or translation.get("source_prefix") != _VLLM_SOURCE_PREFIX
        or translation.get("destination_prefix") != _VLLM_DESTINATION_PREFIX
    ):
        raise EvaluationContractError("vLLM compatibility manifest does not bind the exact r4 adapter translation")
    tensor_count = _positive_int(translation.get("tensor_count"), label="vLLM compatibility tensor_count")
    if translation.get("translated_tensor_count") != tensor_count or not _is_sha256(
        translation.get("translated_tensor_names_sha256")
    ):
        raise EvaluationContractError("vLLM compatibility manifest has incomplete translated-tensor identity")

    attestation = _read_json_object(files["parity_attestation_sha256"], label="vLLM parity attestation")
    try:
        from ctm.evals.qwen35_vllm_attestation import COMPOSITE_PARITY_ATTESTATION_SCHEMA_V1
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise RuntimeError("Qwen3.5 vLLM attestation support is unavailable") from exc
    if attestation.get("schema") == COMPOSITE_PARITY_ATTESTATION_SCHEMA_V1 and attestation.get("model") != BASE_MODEL:
        raise EvaluationContractError("composite vLLM parity attestation does not bind the sealed Qwen3.5 snapshot")
    reports = _vllm_parity_report_identities(attestation)
    # The first report is the direct/full primary report for both supported
    # attestation forms. The optional amplified composite sidecar remains
    # separately hash-bound above, but cannot substitute for r005's complete
    # four-server restricted-softmax parity proof of the served adapter.
    validate_r005_parallel_parity_report(reports[0]["path"], compatibility_adapter=directory)
    return {
        "path": str(directory),
        "adapter_model_sha256": _sha256_file(files["adapter_model_sha256"]),
        "adapter_config_sha256": _sha256_file(files["adapter_config_sha256"]),
        "compatibility_manifest": {
            "path": str(files["compatibility_manifest_sha256"]),
            "sha256": _sha256_file(files["compatibility_manifest_sha256"]),
        },
        "parity_attestation": {
            "path": str(files["parity_attestation_sha256"]),
            "sha256": _sha256_file(files["parity_attestation_sha256"]),
            "schema": attestation["schema"],
        },
        "parity_reports": reports,
        "source_raw_checkpoint": str(raw_checkpoint["path"]),
        "source_raw_adapter_model_sha256": str(raw_checkpoint["adapter_model_sha256"]),
        "base_model": BASE_MODEL,
    }


def build_evaluation_receipt(
    *,
    manifest: str | Path,
    checkpoint: str | Path,
    convergence_receipt: str | Path,
    condition: str,
    raw_log_root: str | Path | None = None,
    runtime_profile: str = "hf-peft",
    max_connections: int | None = None,
    vllm_compat_adapter: str | Path | None = None,
) -> dict[str, Any]:
    """Build a complete, CPU-only receipt before any evaluation generation.

    Native HF/PEFT is supported directly.  vLLM is supported only through a
    translated Qwen3.5 adapter that has a current immutable parity
    attestation.  The returned receipt can be written once with
    :func:`write_evaluation_receipt`, verified again just before raw
    generation, and carried into raw-log preflight/analysis.
    """

    condition = _safe_component(condition, label="condition")
    if runtime_profile not in {"hf-peft", "vllm"}:
        raise EvaluationContractError("runtime_profile must be 'hf-peft' or parity-attested 'vllm'")
    if runtime_profile == "hf-peft":
        max_connections = 4 if max_connections is None else _positive_int(max_connections, label="max_connections")
        if vllm_compat_adapter is not None:
            raise EvaluationContractError("vllm_compat_adapter is valid only with runtime_profile='vllm'")
    elif max_connections is not None:
        raise EvaluationContractError("max_connections is a native-HF-only setting; omit it for runtime_profile='vllm'")
    raw_root = None if raw_log_root is None else str(_local_path(raw_log_root, label="raw_log_root"))
    substrate = validate_stage2_substrate(manifest)
    if substrate["deployment_provenance"] is None:
        raise EvaluationContractError(
            "the production r4 evaluation requires a deployment-rebased Stage 2 manifest with source provenance"
        )
    checkpoint_record = checkpoint_identity(checkpoint)
    if checkpoint_record["checkpoint_name"] != R4_FINAL_CHECKPOINT_NAME:
        raise EvaluationContractError(
            f"evaluation requires the converged r4 checkpoint {R4_FINAL_CHECKPOINT_NAME!r}, "
            f"got {checkpoint_record['checkpoint_name']!r}"
        )
    convergence = _decision_identity(convergence_receipt)
    if (
        checkpoint_record["global_step"] != convergence["optimizer_step"]
        or checkpoint_record["optimizer_step"] != convergence["optimizer_step"]
    ):
        raise EvaluationContractError("checkpoint global/optimizer step does not match the converged controller receipt")

    if runtime_profile == "hf-peft":
        runtime: dict[str, Any] = {
            "profile": "hf-peft",
            "base_model": BASE_MODEL,
            "checkpoint": checkpoint_record["path"],
            "max_connections": max_connections,
            "provider": "hf",
            "device": "cuda:0",
            "dtype": "bfloat16",
        }
    else:
        if vllm_compat_adapter is None:
            raise EvaluationContractError("runtime_profile='vllm' requires vllm_compat_adapter")
        compatibility = vllm_compatibility_identity(vllm_compat_adapter, raw_checkpoint=checkpoint_record)
        runtime = {
            "profile": "vllm",
            "base_model": BASE_MODEL,
            "checkpoint": compatibility["path"],
            "raw_checkpoint": checkpoint_record["path"],
            "provider": "vllm",
            "generation": dict(VLLM_GENERATION_CONFIG),
            "model_args": dict(VLLM_MODEL_ARGS),
            "sampler": dict(VLLM_SAMPLER_RUNTIME),
            "compatibility_adapter": compatibility,
        }

    return {
        "schema": EVALUATION_RECEIPT_SCHEMA,
        "condition": condition,
        "checkpoint": checkpoint_record,
        "convergence": convergence,
        "stage2_substrate": substrate,
        "runtime": runtime,
        "raw_generation": {
            "raw_log_root": raw_root,
            "task_factory": TASK_FACTORY,
            "task_count": EXPECTED_TASKS,
            "clean_task_count": EXPECTED_CLEAN_TASKS,
            "biased_task_count": EXPECTED_BIASED_TASKS,
            "clean_before_biased": True,
            "prompt_style": PROMPT_STYLE,
            "include_bias_acknowledged": False,
            "grader_model": None,
        },
        "science": {
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "bias_status_source": "bias_type_not_legacy_substrate_regime",
            "preserve_per_bias_results": True,
        },
    }


def _load_receipt(value: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, Mapping):
        return dict(value), None
    path = _local_path(value, label="evaluation receipt")
    return _read_json_object(path, label="evaluation receipt"), path


def validate_evaluation_receipt(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Validate portable receipt structure without reopening remote-side files."""

    document, _ = _load_receipt(value)
    required = {"schema", "condition", "checkpoint", "convergence", "stage2_substrate", "runtime", "raw_generation", "science"}
    if set(document) != required or document.get("schema") != EVALUATION_RECEIPT_SCHEMA:
        raise EvaluationContractError("unsupported or incomplete two-bias evaluation receipt")
    _safe_component(document.get("condition"), label="receipt condition")

    checkpoint = document.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise EvaluationContractError("evaluation receipt has no checkpoint identity")
    required_checkpoint = {
        "path",
        "uri",
        "checkpoint_name",
        "backend",
        "lora",
        "base_model",
        "kind",
        "global_step",
        "optimizer_step",
        "completed_epochs",
        "resume_state_required",
        "replicated_rng_state_required",
        "adapter_config_sha256",
        "adapter_model_sha256",
        "optimizer_sha256",
        "manifest_sha256",
        "replicated_training_manifest_sha256",
        "replicated_training_rng_sha256",
    }
    if set(checkpoint) != required_checkpoint:
        raise EvaluationContractError("evaluation receipt checkpoint identity is incomplete")
    if (
        checkpoint.get("checkpoint_name") != R4_FINAL_CHECKPOINT_NAME
        or checkpoint.get("backend") != "local"
        or checkpoint.get("lora") is not True
        or checkpoint.get("base_model") != BASE_MODEL
        or checkpoint.get("kind") != "both"
        or checkpoint.get("global_step") != R4_FINAL_OPTIMIZER_STEP
        or checkpoint.get("optimizer_step") != R4_FINAL_OPTIMIZER_STEP
        or checkpoint.get("resume_state_required") is not True
        or checkpoint.get("replicated_rng_state_required") is not True
    ):
        raise EvaluationContractError("evaluation receipt does not bind the converged r4 checkpoint")
    if not isinstance(checkpoint.get("path"), str) or not Path(checkpoint["path"]).is_absolute():
        raise EvaluationContractError("evaluation receipt checkpoint path must be absolute")
    if checkpoint.get("uri") != f"file://{checkpoint['path']}":
        raise EvaluationContractError("evaluation receipt checkpoint URI differs from its path")
    for field in required_checkpoint:
        if field.endswith("_sha256") and not _is_sha256(checkpoint.get(field)):
            raise EvaluationContractError(f"evaluation receipt checkpoint has invalid {field}")

    convergence = document.get("convergence")
    if not isinstance(convergence, Mapping) or set(convergence) != {
        "path",
        "sha256",
        "decision",
        "afterok",
        "segment_index",
        "optimizer_step",
        "window",
    }:
        raise EvaluationContractError("evaluation receipt has an invalid convergence identity")
    if (
        not isinstance(convergence.get("path"), str)
        or not Path(convergence["path"]).is_absolute()
        or not _is_sha256(convergence.get("sha256"))
        or convergence.get("decision") != "converged"
        or convergence.get("afterok") != {"permit_training": False, "successor_action": "no_op"}
        or convergence.get("segment_index") != R4_FINAL_SEGMENT_INDEX
        or convergence.get("optimizer_step") != R4_FINAL_OPTIMIZER_STEP
        or not isinstance(convergence.get("window"), Mapping)
    ):
        raise EvaluationContractError("evaluation receipt convergence identity is not the terminal r4 decision")

    substrate = document.get("stage2_substrate")
    if not isinstance(substrate, Mapping):
        raise EvaluationContractError("evaluation receipt has no Stage 2 substrate")
    expected_substrate = {
        "schema",
        "manifest",
        "task_factory",
        "task_count",
        "clean_task_count",
        "biased_task_count",
        "prompt_style",
        "all_biases",
        "seen_biases",
        "held_out_biases",
        "biases",
        "clean_before_biased_task_order",
        "task_selection_sha256",
        "legacy_regime_policy",
        "frozen_data_policy",
        "deployment_provenance",
    }
    if set(substrate) != expected_substrate:
        raise EvaluationContractError("evaluation receipt has an incomplete Stage 2 substrate identity")
    manifest = substrate.get("manifest")
    if (
        substrate.get("schema") != SUBSTRATE_SCHEMA
        or substrate.get("task_factory") != TASK_FACTORY
        or (substrate.get("task_count"), substrate.get("clean_task_count"), substrate.get("biased_task_count"))
        != (EXPECTED_TASKS, EXPECTED_CLEAN_TASKS, EXPECTED_BIASED_TASKS)
        or substrate.get("prompt_style") != PROMPT_STYLE
        or substrate.get("all_biases") != list(ALL_BIASES)
        or substrate.get("seen_biases") != list(SEEN_BIASES)
        or substrate.get("held_out_biases") != list(HELD_OUT_BIASES)
        or substrate.get("clean_before_biased_task_order") is not True
        or substrate.get("legacy_regime_policy") != "provenance_only_not_scientific_bias_status"
        or substrate.get("frozen_data_policy") != "validate_read_only_no_materialization_or_mutation"
        or not _is_sha256(substrate.get("task_selection_sha256"))
        or not isinstance(manifest, Mapping)
        or set(manifest) != {"path", "sha256"}
        or not isinstance(manifest.get("path"), str)
        or not Path(manifest["path"]).is_absolute()
        or not _is_sha256(manifest.get("sha256"))
    ):
        raise EvaluationContractError("evaluation receipt Stage 2 substrate differs from the two-bias contract")
    biases = substrate.get("biases")
    expected_biases = [
        {
            "bias_type": bias,
            "evaluation_bias_status": bias_status(bias),
            "biased_task_count": EXPECTED_BIASED_TASKS_PER_BIAS,
        }
        for bias in ALL_BIASES
    ]
    if not isinstance(biases, list) or len(biases) != len(expected_biases):
        raise EvaluationContractError("evaluation receipt lacks exact per-bias substrate records")
    for actual, expected in zip(biases, expected_biases, strict=True):
        if not isinstance(actual, Mapping) or set(actual) != {"bias_type", "evaluation_bias_status", "biased_task_count", "substrate_regimes"}:
            raise EvaluationContractError("evaluation receipt has malformed per-bias substrate provenance")
        if any(actual.get(field) != value for field, value in expected.items()) or not isinstance(
            actual.get("substrate_regimes"), list
        ):
            raise EvaluationContractError("evaluation receipt has a misclassified Stage 2 bias")
    deployment = substrate.get("deployment_provenance")
    if not isinstance(deployment, Mapping) or set(deployment) != {
        "schema",
        "source_manifest",
        "artifact_root",
        "copied_artifacts",
        "path_rewrite_policy",
    }:
        raise EvaluationContractError("production evaluation receipt lacks deployment-manifest provenance")
    source_manifest = deployment.get("source_manifest")
    copied_artifacts = deployment.get("copied_artifacts")
    if (
        deployment.get("schema") != "rmct-two-bias-stage2-deployment-v1"
        or not isinstance(source_manifest, Mapping)
        or set(source_manifest) != {"path", "sha256"}
        or not isinstance(source_manifest.get("path"), str)
        or not Path(source_manifest["path"]).is_absolute()
        or not _is_sha256(source_manifest.get("sha256"))
        or not isinstance(deployment.get("artifact_root"), str)
        or not Path(deployment["artifact_root"]).is_absolute()
        or deployment.get("path_rewrite_policy")
        != "only_population_artifact_paths_rewritten_jsonl_bytes_unchanged"
        or not isinstance(copied_artifacts, Mapping)
        or len(copied_artifacts) != 14
    ):
        raise EvaluationContractError("production evaluation receipt has invalid deployment-manifest provenance")
    for key, artifact in copied_artifacts.items():
        if not isinstance(key, str) or not isinstance(artifact, Mapping) or set(artifact) != {
            "source_path",
            "deployed_path",
            "sha256",
            "byte_count",
        }:
            raise EvaluationContractError("production evaluation receipt has malformed deployed artifact provenance")
        if (
            not isinstance(artifact.get("source_path"), str)
            or not Path(artifact["source_path"]).is_absolute()
            or not isinstance(artifact.get("deployed_path"), str)
            or not Path(artifact["deployed_path"]).is_absolute()
            or not _is_sha256(artifact.get("sha256"))
            or isinstance(artifact.get("byte_count"), bool)
            or not isinstance(artifact.get("byte_count"), int)
            or artifact["byte_count"] < 1
        ):
            raise EvaluationContractError("production evaluation receipt has invalid deployed artifact identity")

    runtime = document.get("runtime")
    if not isinstance(runtime, Mapping):
        raise EvaluationContractError("evaluation receipt has no runtime contract")
    if runtime.get("profile") == "hf-peft":
        if dict(runtime) != {
            "profile": "hf-peft",
            "base_model": BASE_MODEL,
            "checkpoint": checkpoint["path"],
            "max_connections": runtime.get("max_connections"),
            "provider": "hf",
            "device": "cuda:0",
            "dtype": "bfloat16",
        }:
            raise EvaluationContractError("evaluation receipt has an invalid native-HF runtime contract")
        _positive_int(runtime.get("max_connections"), label="receipt max_connections")
    elif runtime.get("profile") == "vllm":
        expected_keys = {
            "profile",
            "base_model",
            "checkpoint",
            "raw_checkpoint",
            "provider",
            "generation",
            "model_args",
            "sampler",
            "compatibility_adapter",
        }
        compatibility = runtime.get("compatibility_adapter")
        if (
            set(runtime) != expected_keys
            or runtime.get("base_model") != BASE_MODEL
            or runtime.get("raw_checkpoint") != checkpoint["path"]
            or runtime.get("provider") != "vllm"
            or dict(runtime.get("generation", {})) != VLLM_GENERATION_CONFIG
            or dict(runtime.get("model_args", {})) != VLLM_MODEL_ARGS
            or dict(runtime.get("sampler", {})) != VLLM_SAMPLER_RUNTIME
            or not isinstance(runtime.get("checkpoint"), str)
            or not Path(runtime["checkpoint"]).is_absolute()
            or not isinstance(compatibility, Mapping)
        ):
            raise EvaluationContractError("evaluation receipt has an invalid vLLM runtime contract")
        compatibility_keys = {
            "path",
            "adapter_model_sha256",
            "adapter_config_sha256",
            "compatibility_manifest",
            "parity_attestation",
            "parity_reports",
            "source_raw_checkpoint",
            "source_raw_adapter_model_sha256",
            "base_model",
        }
        if (
            set(compatibility) != compatibility_keys
            or compatibility.get("path") != runtime["checkpoint"]
            or compatibility.get("source_raw_checkpoint") != checkpoint["path"]
            or compatibility.get("source_raw_adapter_model_sha256") != checkpoint["adapter_model_sha256"]
            or compatibility.get("base_model") != BASE_MODEL
            or not isinstance(compatibility.get("path"), str)
            or not Path(compatibility["path"]).is_absolute()
            or any(not _is_sha256(compatibility.get(field)) for field in ("adapter_model_sha256", "adapter_config_sha256", "source_raw_adapter_model_sha256"))
        ):
            raise EvaluationContractError("evaluation receipt has incomplete vLLM compatibility-adapter identity")
        manifest_identity = compatibility.get("compatibility_manifest")
        attestation_identity = compatibility.get("parity_attestation")
        reports = compatibility.get("parity_reports")
        if (
            not isinstance(manifest_identity, Mapping)
            or set(manifest_identity) != {"path", "sha256"}
            or not isinstance(manifest_identity.get("path"), str)
            or not Path(manifest_identity["path"]).is_absolute()
            or not _is_sha256(manifest_identity.get("sha256"))
            or not isinstance(attestation_identity, Mapping)
            or set(attestation_identity) != {"path", "sha256", "schema"}
            or not isinstance(attestation_identity.get("path"), str)
            or not Path(attestation_identity["path"]).is_absolute()
            or not _is_sha256(attestation_identity.get("sha256"))
            or not isinstance(attestation_identity.get("schema"), str)
            or not isinstance(reports, list)
            or not reports
            or any(
                not isinstance(report, Mapping)
                or set(report) != {"path", "sha256"}
                or not isinstance(report.get("path"), str)
                or not Path(report["path"]).is_absolute()
                or not _is_sha256(report.get("sha256"))
                for report in reports
            )
            or len({report["path"] for report in reports if isinstance(report, Mapping)}) != len(reports)
        ):
            raise EvaluationContractError("evaluation receipt has incomplete vLLM parity evidence")
        try:
            from ctm.evals.qwen35_vllm_attestation import (
                COMPOSITE_PARITY_ATTESTATION_SCHEMA_V1,
                PARITY_ATTESTATION_SCHEMA_V1,
            )
        except ImportError as exc:  # pragma: no cover - configured evaluation environment
            raise RuntimeError("Qwen3.5 vLLM attestation support is unavailable") from exc
        from ctm.evals.qwen35_vllm_scope import ATTESTATION_SCHEMA as SCOPE_ATTESTATION_SCHEMA

        expected_report_count = {
            PARITY_ATTESTATION_SCHEMA_V1: 1,
            COMPOSITE_PARITY_ATTESTATION_SCHEMA_V1: 2,
            SCOPE_ATTESTATION_SCHEMA: 1,
        }.get(attestation_identity["schema"])
        if expected_report_count is None or len(reports) != expected_report_count:
            raise EvaluationContractError("evaluation receipt has an unsupported or incomplete vLLM parity attestation")
    else:
        raise EvaluationContractError("evaluation receipt runtime profile is unsupported")

    raw = document.get("raw_generation")
    if not isinstance(raw, Mapping) or set(raw) != {
        "raw_log_root",
        "task_factory",
        "task_count",
        "clean_task_count",
        "biased_task_count",
        "clean_before_biased",
        "prompt_style",
        "include_bias_acknowledged",
        "grader_model",
    }:
        raise EvaluationContractError("evaluation receipt has an incomplete raw-generation contract")
    if (
        raw.get("task_factory") != TASK_FACTORY
        or (raw.get("task_count"), raw.get("clean_task_count"), raw.get("biased_task_count"))
        != (EXPECTED_TASKS, EXPECTED_CLEAN_TASKS, EXPECTED_BIASED_TASKS)
        or raw.get("clean_before_biased") is not True
        or raw.get("prompt_style") != PROMPT_STYLE
        or raw.get("include_bias_acknowledged") is not False
        or raw.get("grader_model") is not None
        or (raw.get("raw_log_root") is not None and (not isinstance(raw["raw_log_root"], str) or not Path(raw["raw_log_root"]).is_absolute()))
    ):
        raise EvaluationContractError("evaluation receipt raw-generation contract is invalid")
    science = document.get("science")
    if not isinstance(science, Mapping) or dict(science) != {
        "seen_biases": list(SEEN_BIASES),
        "held_out_biases": list(HELD_OUT_BIASES),
        "bias_status_source": "bias_type_not_legacy_substrate_regime",
        "preserve_per_bias_results": True,
    }:
        raise EvaluationContractError("evaluation receipt science labels are invalid or ambiguous")
    return document


def verify_evaluation_receipt(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Re-open all local evidence and reject post-receipt provenance drift."""

    document = validate_evaluation_receipt(value)
    runtime = document["runtime"]
    rebuilt = build_evaluation_receipt(
        manifest=document["stage2_substrate"]["manifest"]["path"],
        checkpoint=document["checkpoint"]["path"],
        convergence_receipt=document["convergence"]["path"],
        condition=document["condition"],
        raw_log_root=document["raw_generation"]["raw_log_root"],
        runtime_profile=runtime["profile"],
        max_connections=runtime.get("max_connections"),
        vllm_compat_adapter=(
            runtime["compatibility_adapter"]["path"] if runtime["profile"] == "vllm" else None
        ),
    )
    if rebuilt != document:
        raise EvaluationContractError("evaluation receipt differs from the currently verified provenance")
    return document


def write_evaluation_receipt(path: str | Path, receipt: Mapping[str, Any]) -> str:
    """Publish a receipt once, or resume only when it is byte-identical."""

    document = validate_evaluation_receipt(receipt)
    destination = _local_path(path, label="evaluation receipt output")
    payload = _canonical_json(document)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing evaluation receipt: {destination}")
        return "resumed"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
                raise FileExistsError(f"evaluation receipt appeared and differs: {destination}") from None
            return "resumed"
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


__all__ = [
    "ALL_BIASES",
    "BASE_MODEL",
    "EVALUATION_RECEIPT_SCHEMA",
    "EXPECTED_BIASED_TASKS",
    "EXPECTED_CLEAN_TASKS",
    "EXPECTED_TASKS",
    "HELD_OUT_BIASES",
    "MODEL_ALIAS",
    "PROMPT_STYLE",
    "R4_FINAL_CHECKPOINT_NAME",
    "R4_FINAL_OPTIMIZER_STEP",
    "R4_FINAL_SEGMENT_INDEX",
    "R4_RUN_PREFIX",
    "SEEN_BIASES",
    "SUBSTRATE_SCHEMA",
    "TASK_FACTORY",
    "VLLM_GDN_PREFILL_BACKEND",
    "VLLM_GENERATION_CONFIG",
    "VLLM_MODEL_ARGS",
    "VLLM_PARITY_LOGPROBS_MODE",
    "VLLM_PARITY_MAX_LOGPROBS",
    "VLLM_PARITY_SAMPLING",
    "VLLM_PARITY_SCORE_TRANSPORT",
    "VLLM_PARITY_TOP_TOKEN_COUNT",
    "VLLM_SAMPLER_ENVIRONMENT",
    "VLLM_SAMPLER_RUNTIME",
    "VLLM_VERSION",
    "EvaluationContractError",
    "bias_status",
    "build_evaluation_receipt",
    "checkpoint_identity",
    "validate_evaluation_receipt",
    "validate_r003_parallel_parity_report",
    "validate_r004_parallel_parity_report",
    "validate_r005_parallel_parity_report",
    "validate_stage2_substrate",
    "verify_evaluation_receipt",
    "vllm_compatibility_identity",
    "write_evaluation_receipt",
]
