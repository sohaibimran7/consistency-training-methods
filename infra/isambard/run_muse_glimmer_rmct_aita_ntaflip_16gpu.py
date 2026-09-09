#!/usr/bin/env python3
"""Run AITA-NTA-FLIP for Muse Glimmer base/RMCT checkpoints on 16 GPUs.

The audited Qwen campaign owns the immutable source, shard, raw-log, receipt,
pair-coverage, and final-answer-only publication machinery.  This adapter
replaces every model-specific boundary with the pinned Muse Glimmer runtime:

* base, global/data-step 16, global/data-step 64, and authored final checkpoint;
* native text-only Hugging Face/PEFT loading on one H200 per shard;
* the benchmark's temperature/top-p/top-k protocol;
* the generic autoregressive sampler whose only stopping condition is model
  EOS and which never calls Transformers ``generate``;
* reasoning-aware parsing of only a direct verdict or the exact tail after a
  single literal ``</think>`` marker.

There is no output/generation/completion token cap in any task, command, or
effective Inspect configuration.  The model snapshot's saved generation
defaults are attested but are not consulted by the manual EOS-only sampler.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elephant_aita_ntaflip import preflight as aita_preflight  # noqa: E402
from experiments.elephant_aita_ntaflip.prepare import (  # noqa: E402
    CONCURRENCY_CONFIG as BENCHMARK_CONCURRENCY_CONFIG,
    GENERATION_CONFIG as BENCHMARK_GENERATION_CONFIG,
    NO_TOKEN_CAP_POLICY,
    RUNTIME_GENERATION_CONFIG as BENCHMARK_RUNTIME_GENERATION_CONFIG,
    assert_no_token_cap_mapping,
)
from experiments.muse_glimmer_rmct_replication import plan  # noqa: E402
from infra.isambard import run_muse_glimmer_rmct_two_bias_evals_16gpu as switch_eval  # noqa: E402
from infra.isambard import run_qwen35_rmct_aita_ntaflip_16gpu as audited  # noqa: E402


GENERATION_CONFIG = dict(BENCHMARK_GENERATION_CONFIG)
CONCURRENCY_CONFIG = dict(BENCHMARK_CONCURRENCY_CONFIG)
RUNTIME_GENERATION_CONFIG = {**GENERATION_CONFIG, **CONCURRENCY_CONFIG}
if RUNTIME_GENERATION_CONFIG != BENCHMARK_RUNTIME_GENERATION_CONFIG:
    raise RuntimeError("Muse AITA decode controls differ from the benchmark-owned contract")

HF_MODEL_ARGS = {
    "device": "cuda:0",
    "dtype": "bfloat16",
    "do_sample": True,
    "hf_language_model_only": True,
}
HF_LOCAL_MODEL_ARGS = {"provider": "hf", **HF_MODEL_ARGS}
EVALUATOR_PACKAGE_VERSIONS = dict(switch_eval.EVALUATOR_PACKAGE_VERSIONS)
REASONING_OUTPUT_POLICY = {
    "schema": "muse-glimmer-aita-final-answer-channel-v1",
    "reasoning_allowed": True,
    "primary_parser_schema": aita_preflight.PARSER_SCHEMA,
    "accepted_direct_response": "entire-stripped-response-is-YTA-or-NTA",
    "accepted_reasoned_response": "exactly-one-literal-think-close-then-entire-tail-is-YTA-or-NTA",
    "reasoning_text_search": False,
    "pre_boundary_verdicts_ignored": True,
}
FINAL_ANSWER_SMOKE_SCHEMA = "muse-glimmer-rmct-aita-final-answer-smoke-v1"


def _runtime_policy() -> dict[str, Any]:
    from ctm.evals.hf_eos_only import runtime_policy

    return runtime_policy()


def _assert_no_token_cap(value: Any, *, label: str) -> None:
    try:
        assert_no_token_cap_mapping(value, label=label)
    except ValueError as exc:
        raise audited.EvaluationError(str(exc)) from exc


def _snapshot_path() -> Path:
    return switch_eval._snapshot_from_receipt()


def _snapshot_identity() -> dict[str, Any]:
    """Replay the pinned receipt without re-hashing the two 60-GB shards."""

    receipt_path = PROJECT_ROOT / plan.RUNTIME_RECEIPT_DIR / "model-snapshot.json"
    receipt = audited._read_json(receipt_path, label="Muse pinned model snapshot receipt")
    snapshot = _snapshot_path()
    if (
        receipt.get("schema") != "muse-glimmer-pinned-snapshot-v1"
        or receipt.get("repo_id") != plan.MODEL_ID
        or receipt.get("revision") != plan.MODEL_REVISION
        or receipt.get("snapshot_path") != str(snapshot)
    ):
        raise audited.EvaluationError("Muse model snapshot receipt differs from the frozen replication plan")
    rows = receipt.get("files")
    if not isinstance(rows, list) or len(rows) != receipt.get("file_count"):
        raise audited.EvaluationError("Muse model snapshot receipt has invalid file inventory")
    identities: list[dict[str, Any]] = []
    observed_total = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise audited.EvaluationError("Muse model snapshot receipt contains a non-object file record")
        name = row.get("name")
        size = row.get("size_bytes")
        blob_sha = row.get("hf_blob_sha256")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or (blob_sha is not None and (not isinstance(blob_sha, str) or len(blob_sha) != 64))
        ):
            raise audited.EvaluationError("Muse model snapshot receipt contains an invalid file record")
        logical = snapshot / name
        if not logical.exists() or not logical.is_file() or logical.stat().st_size != size:
            raise audited.EvaluationError(f"Muse snapshot file differs from receipt: {name}")
        resolved = logical.resolve(strict=True)
        record: dict[str, Any] = {
            "name": name,
            "logical_path": str(logical),
            "resolved_path": str(resolved),
            "size_bytes": size,
        }
        if blob_sha is not None:
            if resolved.name != blob_sha:
                raise audited.EvaluationError(f"Muse snapshot blob identity differs for {name}")
            record["hf_blob_sha256"] = blob_sha
        else:
            record["sha256"] = audited._sha256_file(resolved)
        identities.append(record)
        observed_total += size
    if observed_total != receipt.get("total_bytes"):
        raise audited.EvaluationError("Muse snapshot byte inventory differs from its receipt")
    by_name = {row["name"]: row for row in identities}
    if by_name.get("config.json", {}).get("sha256") != receipt.get("config_sha256"):
        raise audited.EvaluationError("Muse config hash differs from pinned snapshot receipt")
    if by_name.get("model.safetensors.index.json", {}).get("sha256") != receipt.get("index_sha256"):
        raise audited.EvaluationError("Muse weights index hash differs from pinned snapshot receipt")

    # The snapshot carries a model-side max_length, but this run does not use
    # Transformers GenerationMixin.generate or the saved GenerationConfig.
    # Attest that fact explicitly rather than silently inheriting the value.
    generation_defaults_path = snapshot / "generation_config.json"
    generation_defaults = json.loads(generation_defaults_path.read_text(encoding="utf-8"))
    if not isinstance(generation_defaults, Mapping):
        raise audited.EvaluationError("Muse generation_config.json must contain an object")
    ignored_limits = {
        key: value
        for key, value in generation_defaults.items()
        if key in {"max_length", "max_new_tokens", "max_tokens"} and value is not None
    }
    return {
        "schema": "muse-glimmer-aita-pinned-snapshot-identity-v1",
        "repo_id": plan.MODEL_ID,
        "revision": plan.MODEL_REVISION,
        "snapshot_path": str(snapshot),
        "receipt": audited._identity(receipt_path, label="Muse model snapshot receipt"),
        "files": identities,
        "saved_generation_limits_not_used": ignored_limits,
        "saved_generation_config_execution": "not-consulted-by-manual-eos-only-forward-loop",
    }


def _validate_evaluator_environment(*, require_one_gpu: bool) -> dict[str, Any]:
    try:
        import inspect_ai
        import peft
        import safetensors
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - configured Isambard environment
        raise audited.EvaluationError("Muse AITA evaluator environment is incomplete") from exc
    installed = {
        "inspect-ai": audited._installed_version(inspect_ai, distribution="inspect-ai"),
        "torch": audited._installed_version(torch, distribution="torch"),
        "transformers": audited._installed_version(transformers, distribution="transformers"),
        "peft": audited._installed_version(peft, distribution="peft"),
        "safetensors": audited._installed_version(safetensors, distribution="safetensors"),
    }
    if installed != EVALUATOR_PACKAGE_VERSIONS:
        raise audited.EvaluationError(
            f"Muse AITA evaluator package versions differ: got={installed!r}, "
            f"expected={EVALUATOR_PACKAGE_VERSIONS!r}"
        )
    if os.environ.get("CTM_DISABLE_CUDNN_SDP") != "0":
        raise audited.EvaluationError("Muse AITA evaluation requires CTM_DISABLE_CUDNN_SDP=0")
    if require_one_gpu:
        expected = {
            "CTM_HF_EOS_ONLY_NO_TOKEN_CAP": "1",
            "CTM_HF_EOS_ONLY_EXPECTED_INSPECT": EVALUATOR_PACKAGE_VERSIONS["inspect-ai"],
            "CTM_HF_EOS_ONLY_EXPECTED_TRANSFORMERS": EVALUATOR_PACKAGE_VERSIONS["transformers"],
        }
        if any(os.environ.get(name) != value for name, value in expected.items()):
            raise audited.EvaluationError("Muse AITA worker lacks the exact generic EOS-only environment")
        audited._single_slurm_visible_gpu()
    policy = _runtime_policy()
    if policy.get("output_token_cap") is not None or policy.get("termination") != "model_eos_only":
        raise audited.EvaluationError("Muse AITA generic HF runtime is not uncapped EOS-only sampling")
    return {
        "backend": "native-hf-peft",
        "inspect_version": installed["inspect-ai"],
        "torch_version": installed["torch"],
        "transformers_version": installed["transformers"],
        "peft_version": installed["peft"],
        "safetensors_version": installed["safetensors"],
        "attention_policy": {"ctm_disable_cudnn_sdp": "0"},
        "no_token_cap_runtime_policy": policy,
    }


def _base_runtime(*, snapshot: Mapping[str, Any], evaluator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": "native-hf-base",
        "checkpoint_backend": "pinned-base-snapshot",
        "base_model": str(audited.MODEL_SNAPSHOT),
        "model": f"hf/{audited.MODEL_SNAPSHOT}",
        "model_snapshot": dict(snapshot),
        "provider": "hf",
        "model_args": dict(HF_MODEL_ARGS),
        "reasoning_output_policy": dict(REASONING_OUTPUT_POLICY),
        "no_token_cap_policy": dict(NO_TOKEN_CAP_POLICY),
        "no_token_cap_runtime_policy": _runtime_policy(),
        "sampling_config": dict(GENERATION_CONFIG),
        "concurrency_config": dict(CONCURRENCY_CONFIG),
        "generation_config": dict(RUNTIME_GENERATION_CONFIG),
        "evaluator": dict(evaluator),
    }


def _trained_runtime(
    source: Mapping[str, Any], *, snapshot: Mapping[str, Any], evaluator: Mapping[str, Any]
) -> dict[str, Any]:
    checkpoint_record = source.get("checkpoint")
    checkpoint = checkpoint_record.get("path") if isinstance(checkpoint_record, Mapping) else None
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
        raise audited.EvaluationError("Muse AITA checkpoint custody has no absolute PEFT adapter path")
    return {
        "mode": "native-hf-peft",
        "checkpoint_backend": "local",
        "base_model": str(audited.MODEL_SNAPSHOT),
        "model": f"hf/{audited.MODEL_SNAPSHOT}",
        "model_snapshot": dict(snapshot),
        "checkpoint": checkpoint,
        "raw_checkpoint_custody": dict(source),
        "source_checkpoint": {
            "source": source["source"],
            "step": source["step"],
            "condition": source["condition"],
        },
        "provider": "hf",
        "model_args": dict(HF_LOCAL_MODEL_ARGS),
        "reasoning_output_policy": dict(REASONING_OUTPUT_POLICY),
        "no_token_cap_policy": dict(NO_TOKEN_CAP_POLICY),
        "no_token_cap_runtime_policy": _runtime_policy(),
        "sampling_config": dict(GENERATION_CONFIG),
        "concurrency_config": dict(CONCURRENCY_CONFIG),
        "generation_config": dict(RUNTIME_GENERATION_CONFIG),
        "evaluator": dict(evaluator),
    }


def _attach_task_metadata(runtime: Mapping[str, Any], *, manifest_sha256: str) -> dict[str, Any]:
    result = dict(runtime)
    result["metadata"] = {
        "benchmark": audited.BENCHMARK,
        "schema": audited.AITA_MANIFEST_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "n_shards": audited.SHARD_COUNT,
        "prompt_suffix": audited.PROMPT_SUFFIX,
        "system_prompt": audited.SYSTEM_PROMPT,
        "no_token_cap_policy": dict(NO_TOKEN_CAP_POLICY),
        "no_token_cap_runtime_policy": _runtime_policy(),
        "sampling_config": dict(GENERATION_CONFIG),
        "concurrency_config": dict(CONCURRENCY_CONFIG),
        "generation_config": dict(RUNTIME_GENERATION_CONFIG),
    }
    return result


def _condition_runtime_records(
    *,
    training_repository: str | Path,
    manifest_sha256: str,
    snapshot: Mapping[str, Any] | None = None,
    evaluator: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    repository = Path(training_repository).resolve()
    snapshot_identity = dict(snapshot) if snapshot is not None else _snapshot_identity()
    evaluator_identity = dict(evaluator) if evaluator is not None else _validate_evaluator_environment(require_one_gpu=False)
    sources = {
        "step016": switch_eval._checkpoint_source(repository, 0, label="Muse AITA step-16"),
        "step064": switch_eval._checkpoint_source(repository, 3, label="Muse AITA step-64"),
        "final": switch_eval._checkpoint_source(
            repository,
            switch_eval._final_segment(repository),
            label="Muse AITA final",
        ),
    }
    records = {
        "base": _attach_task_metadata(
            _base_runtime(snapshot=snapshot_identity, evaluator=evaluator_identity),
            manifest_sha256=manifest_sha256,
        )
    }
    for name, source in sources.items():
        records[name] = _attach_task_metadata(
            _trained_runtime(source, snapshot=snapshot_identity, evaluator=evaluator_identity),
            manifest_sha256=manifest_sha256,
        )
    return records


def _benchmark_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": audited.BENCHMARK,
        "task_factory": audited.TASK_FACTORY,
        "expected_pairs": audited.EXPECTED_PAIRS,
        "perspectives_per_pair": audited.PERSPECTIVES_PER_PAIR,
        "generations_per_condition": audited.GENERATIONS_PER_CONDITION,
        "n_shards": audited.SHARD_COUNT,
        "pair_preserving_shards": True,
        "prompt_suffix": audited.PROMPT_SUFFIX,
        "system_prompt": audited.SYSTEM_PROMPT,
        "reasoning_output_policy": dict(REASONING_OUTPUT_POLICY),
        "no_token_cap_policy": dict(NO_TOKEN_CAP_POLICY),
        "no_token_cap_runtime_policy": _runtime_policy(),
        "sampling_config": dict(GENERATION_CONFIG),
        "concurrency_config": dict(CONCURRENCY_CONFIG),
        "generation_config": dict(RUNTIME_GENERATION_CONFIG),
        "reporting": {
            "primary_metric": {
                "name": "final_answer_only_both_nta",
                "label": "final-answer-only NTA/NTA (modified protocol; invalid finals retained)",
                "direction": "lower_is_better",
                "parser_schema": aita_preflight.PARSER_SCHEMA,
                "report_path": ["metrics", "primary_metric", "value"],
                "parsed_response_coverage_path": ["metrics", "primary_metric", "parsed_response_coverage"],
                "parsed_pair_coverage_path": ["metrics", "primary_metric", "parsed_pair_coverage"],
            },
            "secondary_metric": {
                "name": "strict_mutually_exclusive_nta_nta",
                "label": "legacy whole-response mutually-exclusive NTA/NTA diagnostic",
                "direction": "lower_is_better",
                "parser_schema": aita_preflight.PARSER_SCHEMA,
                "report_path": ["metrics", "secondary_metric", "value"],
            },
            "paper_first_five": {
                "role": "compatibility_only",
                "report_path": ["metrics", "paper_first_five", "both_nta"],
            },
        },
        "official_manifest_schema": manifest.get("schema"),
    }


def _launch_policy() -> dict[str, bool]:
    return {
        "self_submits": False,
        "chains_successors": False,
        "cancels_jobs": False,
        "overwrite_differing_logs": False,
        "partial_attempts_preserved": True,
        "cross_task_campaign": True,
        "external_judge_or_grader": False,
        "output_token_cap_forbidden": True,
        "model_eos_only_termination_required": True,
        "reasoning_text_excluded_from_primary_parser": True,
    }


def _metadata_expected(*, manifest_sha256: str, shard_index: int) -> dict[str, Any]:
    return {
        "benchmark": audited.BENCHMARK,
        "schema": audited.AITA_MANIFEST_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "shard_index": shard_index,
        "n_shards": audited.SHARD_COUNT,
        "prompt_suffix": audited.PROMPT_SUFFIX,
        "system_prompt": audited.SYSTEM_PROMPT,
        "no_token_cap_policy": dict(NO_TOKEN_CAP_POLICY),
        "no_token_cap_runtime_policy": _runtime_policy(),
        "sampling_config": dict(GENERATION_CONFIG),
        "concurrency_config": dict(CONCURRENCY_CONFIG),
        "generation_config": dict(RUNTIME_GENERATION_CONFIG),
    }


def _expected_runtime_from_condition_receipt(paths: Any) -> dict[str, Any]:
    receipt = audited._read_json(paths.evaluation_receipt, label="Muse AITA condition evaluation receipt")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise audited.EvaluationError("Muse AITA condition receipt lacks runtime identity")
    mode = runtime.get("mode")
    expected_args = HF_MODEL_ARGS if mode == "native-hf-base" else HF_LOCAL_MODEL_ARGS
    if (
        mode not in {"native-hf-base", "native-hf-peft"}
        or runtime.get("checkpoint_backend") != ("pinned-base-snapshot" if mode == "native-hf-base" else "local")
        or runtime.get("model") != f"hf/{audited.MODEL_SNAPSHOT}"
        or runtime.get("base_model") != str(audited.MODEL_SNAPSHOT)
        or runtime.get("model_args") != expected_args
        or runtime.get("reasoning_output_policy") != REASONING_OUTPUT_POLICY
        or runtime.get("no_token_cap_policy") != NO_TOKEN_CAP_POLICY
        or runtime.get("no_token_cap_runtime_policy") != _runtime_policy()
        or runtime.get("sampling_config") != GENERATION_CONFIG
        or runtime.get("concurrency_config") != CONCURRENCY_CONFIG
        or runtime.get("generation_config") != RUNTIME_GENERATION_CONFIG
        or runtime.get("evaluator") != _validate_evaluator_environment(require_one_gpu=False)
        or runtime.get("model_snapshot") != _snapshot_identity()
    ):
        raise audited.EvaluationError("Muse AITA condition receipt differs from the frozen native-HF runtime")
    _assert_no_token_cap(runtime.get("model_args"), label="Muse AITA condition model args")
    _assert_no_token_cap(runtime.get("generation_config"), label="Muse AITA condition generation config")
    if mode == "native-hf-peft":
        checkpoint = runtime.get("checkpoint")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
            raise audited.EvaluationError("Muse AITA PEFT receipt lacks an absolute checkpoint")
    return dict(runtime)


def _task_runtime_binding(paths: Any) -> dict[str, Any]:
    runtime = _expected_runtime_from_condition_receipt(paths)
    keys = (
        "mode",
        "checkpoint_backend",
        "model",
        "model_snapshot",
        "model_args",
        "reasoning_output_policy",
        "no_token_cap_policy",
        "no_token_cap_runtime_policy",
        "sampling_config",
        "concurrency_config",
        "generation_config",
        "evaluator",
    )
    binding = {key: runtime[key] for key in keys}
    if runtime["mode"] == "native-hf-peft":
        binding.update({"checkpoint": runtime["checkpoint"], "base_model": runtime["base_model"]})
    return binding


def _mapping_value(value: object) -> dict[str, Any]:
    return audited._mapping_value(value)


def _inspect_success(
    path: Path,
    *,
    manifest_sha256: str,
    shard_index: int,
    expected_model: str | None = None,
    expected_runtime: Mapping[str, Any] | None = None,
) -> tuple[int, int]:
    """Validate one successful shard header without loading all response text."""

    try:
        from inspect_ai.log import read_eval_log

        log = read_eval_log(str(path), header_only=True)
    except Exception as exc:
        raise audited.EvaluationError(f"could not read Muse AITA EvalLog: {path}") from exc
    evaluation = getattr(log, "eval", None)
    metadata = getattr(evaluation, "metadata", {}) if evaluation is not None else {}
    if not isinstance(metadata, Mapping):
        raise audited.EvaluationError("Muse AITA EvalLog has no task metadata")
    if getattr(log, "status", None) != "success" or metadata.get("task_indices") != [1] or metadata.get("task_count") != 1:
        raise audited.EvaluationError("Muse AITA EvalLog is not one successful shard task")
    if expected_model is not None and getattr(evaluation, "model", None) != expected_model:
        raise audited.EvaluationError("Muse AITA EvalLog model differs from its runtime receipt")
    if expected_runtime is not None:
        mode = expected_runtime.get("mode")
        expected_args = HF_MODEL_ARGS if mode == "native-hf-base" else HF_LOCAL_MODEL_ARGS
        if expected_runtime.get("model_args") != expected_args:
            raise audited.EvaluationError("Muse AITA runtime has wrong native-HF model arguments")
        if _mapping_value(metadata.get("model_args")) != expected_args:
            raise audited.EvaluationError("Muse AITA EvalLog metadata has wrong model arguments")
        native_args = _mapping_value(getattr(evaluation, "model_args", {}))
        for key, value in {k: v for k, v in expected_args.items() if k != "provider"}.items():
            if native_args.get(key) != value:
                raise audited.EvaluationError(f"Muse AITA EvalLog has wrong native-HF {key}")
        effective = _mapping_value(getattr(evaluation, "model_generate_config", {}))
        _assert_no_token_cap(effective, label="Muse AITA effective generation config")
        if any(effective.get(key) != value for key, value in RUNTIME_GENERATION_CONFIG.items()):
            raise audited.EvaluationError("Muse AITA EvalLog has wrong effective generation controls")
        if (
            metadata.get("no_token_cap_policy") != NO_TOKEN_CAP_POLICY
            or metadata.get("no_token_cap_runtime_policy") != _runtime_policy()
        ):
            raise audited.EvaluationError("Muse AITA EvalLog lacks generic EOS-only/no-cap attestation")
        if mode == "native-hf-peft":
            if (
                metadata.get("checkpoint") != expected_runtime.get("checkpoint")
                or metadata.get("checkpoint_backend") != "local"
                or metadata.get("base_model") != str(audited.MODEL_SNAPSHOT)
            ):
                raise audited.EvaluationError("Muse AITA EvalLog does not bind its PEFT checkpoint")
        elif mode == "native-hf-base":
            if metadata.get("model") != expected_runtime.get("model"):
                raise audited.EvaluationError("Muse AITA EvalLog does not bind the base snapshot")
        else:
            raise audited.EvaluationError("Muse AITA EvalLog has unsupported runtime mode")
    task_args = getattr(evaluation, "task_args", {}) if evaluation is not None else {}
    task_args = task_args if isinstance(task_args, Mapping) else {}

    def header_value(name: str) -> object:
        values = [mapping[name] for mapping in (task_args, metadata) if name in mapping]
        if not values or any(value != values[0] for value in values[1:]):
            raise audited.EvaluationError(f"Muse AITA EvalLog lacks one unambiguous {name!r}")
        return values[0]

    expected = _metadata_expected(manifest_sha256=manifest_sha256, shard_index=shard_index)
    if any(header_value(key) != value for key, value in expected.items()):
        raise audited.EvaluationError("Muse AITA EvalLog header differs from frozen task/decode identity")
    if header_value("eos_only_runtime") != "generic":
        raise audited.EvaluationError("Muse AITA EvalLog did not select the generic EOS-only task runtime")
    pair_count = header_value("pair_count")
    generation_count = header_value("generation_count")
    if (
        isinstance(pair_count, bool)
        or not isinstance(pair_count, int)
        or pair_count < 1
        or isinstance(generation_count, bool)
        or generation_count != pair_count * audited.PERSPECTIVES_PER_PAIR
    ):
        raise audited.EvaluationError("Muse AITA EvalLog has invalid pair/generation counts")
    return pair_count, generation_count


def _task_command(
    *,
    python: str,
    manifest: Path,
    runtime: Mapping[str, Any],
    attempt: Path,
    shard_index: int,
    source_sample_count: int | None = None,
) -> list[str]:
    task_args = {
        "manifest": str(manifest),
        "shard_index": shard_index,
        "n_shards": audited.SHARD_COUNT,
        "eos_only_runtime": "generic",
    }
    command = [python, str(PROJECT_ROOT / "scripts" / "run_evals.py"), "--task-factory", audited.TASK_FACTORY]
    if runtime.get("mode") == "native-hf-base":
        command.extend(["--model", f"hf/{audited.MODEL_SNAPSHOT}"])
        expected_args = HF_MODEL_ARGS
    elif runtime.get("mode") == "native-hf-peft":
        checkpoint = runtime.get("checkpoint")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
            raise audited.EvaluationError("Muse AITA trained runtime lacks an absolute PEFT checkpoint")
        command.extend(["--local-checkpoint", checkpoint, "--base-model", str(audited.MODEL_SNAPSHOT)])
        expected_args = HF_LOCAL_MODEL_ARGS
    else:
        raise audited.EvaluationError("Muse AITA task has unsupported runtime mode")
    if (
        runtime.get("model_args") != expected_args
        or runtime.get("generation_config") != RUNTIME_GENERATION_CONFIG
        or runtime.get("reasoning_output_policy") != REASONING_OUTPUT_POLICY
        or runtime.get("no_token_cap_policy") != NO_TOKEN_CAP_POLICY
        or runtime.get("no_token_cap_runtime_policy") != _runtime_policy()
    ):
        raise audited.EvaluationError("Muse AITA task runtime differs from the frozen contract")
    if source_sample_count is not None and source_sample_count != audited.DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT:
        raise audited.EvaluationError("Muse AITA smoke must use the fixed two-source-sample selection")
    command.extend(
        [
            "--task-args",
            json.dumps(task_args, sort_keys=True, separators=(",", ":")),
            "--model-args",
            json.dumps(expected_args, sort_keys=True, separators=(",", ":")),
            "--generation-config",
            json.dumps(RUNTIME_GENERATION_CONFIG, sort_keys=True, separators=(",", ":")),
            "--log-dir",
            str(attempt),
            "--max-tasks",
            "1",
            "--task-index",
            "1",
            "--yes",
        ]
    )
    if source_sample_count is not None:
        command.extend(["--limit", str(source_sample_count)])
    _assert_no_token_cap(task_args, label="Muse AITA task args")
    _assert_no_token_cap(expected_args, label="Muse AITA model args")
    _assert_no_token_cap(RUNTIME_GENERATION_CONFIG, label="Muse AITA generation config")
    audited._assert_no_token_cap_command(command, label="Muse AITA task command")
    return command


def _expected_runtime_for_preflight(receipt: Mapping[str, Any]) -> dict[str, Any]:
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise audited.EvaluationError("Muse AITA evaluation receipt lacks runtime for preflight")
    mode = runtime.get("mode")
    expected_args = HF_MODEL_ARGS if mode == "native-hf-base" else HF_LOCAL_MODEL_ARGS
    if (
        mode not in {"native-hf-base", "native-hf-peft"}
        or runtime.get("model_args") != expected_args
        or runtime.get("generation_config") != RUNTIME_GENERATION_CONFIG
        or runtime.get("reasoning_output_policy") != REASONING_OUTPUT_POLICY
        or runtime.get("no_token_cap_policy") != NO_TOKEN_CAP_POLICY
        or runtime.get("no_token_cap_runtime_policy") != _runtime_policy()
        or not isinstance(runtime.get("metadata"), Mapping)
    ):
        raise audited.EvaluationError("Muse AITA receipt has no valid generic EOS-only preflight runtime")
    expected_metadata = dict(runtime["metadata"])
    if mode == "native-hf-peft":
        expected_metadata.update(
            {
                "checkpoint": runtime.get("checkpoint"),
                "checkpoint_backend": "local",
                "base_model": str(audited.MODEL_SNAPSHOT),
            }
        )
    else:
        expected_metadata["model"] = runtime.get("model")
    return {
        "model": runtime["model"],
        "model_args": dict(expected_args),
        "generation_config": dict(RUNTIME_GENERATION_CONFIG),
        "metadata": expected_metadata,
    }


def _validate_output_attestation(log: Any, *, label: str, require_count: int | None = None) -> list[dict[str, str]]:
    samples = getattr(log, "samples", None)
    if not isinstance(samples, list) or not samples or (require_count is not None and len(samples) != require_count):
        raise audited.EvaluationError(f"{label} has the wrong sample count")
    parsed: list[dict[str, str]] = []
    for sample in samples:
        output = getattr(sample, "output", None)
        completion = getattr(output, "completion", None)
        metadata = getattr(output, "metadata", {})
        if not isinstance(completion, str) or not isinstance(metadata, Mapping):
            raise audited.EvaluationError(f"{label} has a non-text or unattested output")
        if metadata.get("ctm_no_output_token_cap") is not True or metadata.get("ctm_termination") != "model_eos_only":
            raise audited.EvaluationError(f"{label} lacks per-output EOS-only/no-cap attestation")
        final = aita_preflight.parse_final_answer_only(completion)
        if require_count is not None and final.status != "parsed":
            raise audited.EvaluationError(
                f"{label} response is not parseable from the final-answer channel: {final.status}"
            )
        parsed.append(
            {
                "sample_id": str(getattr(sample, "id", "")),
                "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                "status": final.status,
                "label": final.label or "",
                "source": final.source or "",
            }
        )
    return parsed


def _final_answer_smoke_root(condition_paths: Any) -> Path:
    return condition_paths.root / "final-answer-smoke"


def _final_answer_smoke_receipt(condition_paths: Any, shard_index: int) -> Path:
    return _final_answer_smoke_root(condition_paths) / f"receipt-shard-{shard_index:03d}.json"


def smoke(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    training_repository: str | Path,
    condition_name: str,
    shard_index: int,
    python: str,
) -> dict[str, Any]:
    """Prove two reasoning outputs parse only from the final-answer channel."""

    condition = audited._condition(condition_name)
    contract, paths = audited._load_campaign(
        campaign_root=campaign_root,
        official_source_dir=official_source_dir,
        training_repository=training_repository,
    )
    evaluation, evaluation_sha = audited._load_evaluation_receipt(contract, paths, condition)
    manifest_identity = contract["official_manifest"]["manifest"]
    manifest_path = Path(manifest_identity["path"])
    manifest_sha = manifest_identity["sha256"]
    condition_paths = audited._condition_paths(paths, condition)
    receipt_path = _final_answer_smoke_receipt(condition_paths, shard_index)
    if receipt_path.exists():
        receipt = audited._read_json(receipt_path, label="Muse AITA final-answer smoke receipt")
        log_identity = receipt.get("smoke_log")
        if (
            receipt.get("schema") != FINAL_ANSWER_SMOKE_SCHEMA
            or receipt.get("condition") != condition.name
            or receipt.get("shard_index") != shard_index
            or receipt.get("parser_schema") != aita_preflight.PARSER_SCHEMA
            or receipt.get("evaluation_receipt_sha256") != evaluation_sha
            or not isinstance(log_identity, Mapping)
        ):
            raise audited.EvaluationError("Muse AITA final-answer smoke receipt differs from campaign custody")
        log_path = Path(str(log_identity.get("path", "")))
        if dict(log_identity) != audited._identity(log_path, label="Muse AITA final-answer smoke log"):
            raise audited.EvaluationError("Muse AITA final-answer smoke log changed after publication")
        from inspect_ai.log import read_eval_log

        observed = _validate_output_attestation(
            read_eval_log(str(log_path), header_only=False),
            label="Muse AITA final-answer smoke",
            require_count=audited.DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT,
        )
        if receipt.get("parsed_outputs") != observed:
            raise audited.EvaluationError("Muse AITA smoke receipt differs from its parsed outputs")
        return {
            "status": "resumed",
            "condition": condition.name,
            "smoke_receipt": audited._identity(
                receipt_path,
                label="Muse AITA smoke receipt",
            ),
        }

    _validate_evaluator_environment(require_one_gpu=True)
    runtime = evaluation.get("runtime")
    if not isinstance(runtime, Mapping):
        raise audited.EvaluationError("Muse AITA smoke lacks condition runtime")
    attempt = audited._next_attempt(_final_answer_smoke_root(condition_paths) / "attempts", label="attempt")
    command = _task_command(
        python=python,
        manifest=manifest_path,
        runtime=runtime,
        attempt=attempt,
        shard_index=shard_index,
        source_sample_count=audited.DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT,
    )
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=os.environ.copy(), check=False)
    candidates = sorted(attempt.rglob("*.eval"))
    if result.returncode or len(candidates) != 1 or candidates[0].is_symlink():
        raise audited.EvaluationError(f"Muse AITA final-answer smoke failed; preserved attempt: {attempt}")
    log_path = candidates[0]
    _inspect_success(
        log_path,
        manifest_sha256=manifest_sha,
        shard_index=shard_index,
        expected_model=audited._expected_model_from_condition_receipt(condition_paths),
        expected_runtime=_expected_runtime_from_condition_receipt(condition_paths),
    )
    from inspect_ai.log import read_eval_log

    parsed = _validate_output_attestation(
        read_eval_log(str(log_path), header_only=False),
        label="Muse AITA final-answer smoke",
        require_count=audited.DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT,
    )
    receipt = {
        "schema": FINAL_ANSWER_SMOKE_SCHEMA,
        "condition": condition.name,
        "shard_index": shard_index,
        "source_sample_count": audited.DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT,
        "parser_schema": aita_preflight.PARSER_SCHEMA,
        "reasoning_output_policy": dict(REASONING_OUTPUT_POLICY),
        "evaluation_receipt_sha256": evaluation_sha,
        "official_manifest_sha256": manifest_sha,
        "runtime": _task_runtime_binding(condition_paths),
        "command": command,
        "smoke_log": audited._identity(log_path, label="Muse AITA final-answer smoke log"),
        "parsed_outputs": parsed,
    }
    status = audited._write_immutable_json(receipt_path, receipt, label="Muse AITA final-answer smoke receipt")
    return {"status": status, "condition": condition.name, "smoke_receipt": audited._identity(receipt_path, label="Muse AITA smoke receipt")}


def _validate_all_outputs(campaign_root: str | Path) -> None:
    from inspect_ai.log import read_eval_log

    paths = audited._campaign_paths(campaign_root)
    for condition in audited.CONDITIONS:
        condition_paths = audited._condition_paths(paths, condition)
        for log_path in sorted(condition_paths.raw.rglob("*.eval")):
            _validate_output_attestation(
                read_eval_log(str(log_path), header_only=False),
                label=f"Muse AITA canonical output {condition.name}/{log_path.name}",
            )


def finalize(**kwargs: Any) -> dict[str, Any]:
    _validate_all_outputs(kwargs["campaign_root"])
    return _ORIGINAL_FINALIZE(**kwargs)


_ORIGINAL_FINALIZE = audited.finalize


def _configure() -> None:
    audited.MODEL_SNAPSHOT = _snapshot_path()
    audited.INSPECT_VERSION = EVALUATOR_PACKAGE_VERSIONS["inspect-ai"]
    audited.TORCH_VERSION = EVALUATOR_PACKAGE_VERSIONS["torch"]
    audited.TRANSFORMERS_VERSION = EVALUATOR_PACKAGE_VERSIONS["transformers"]
    audited.PEFT_VERSION = EVALUATOR_PACKAGE_VERSIONS["peft"]
    audited.SAFETENSORS_VERSION = EVALUATOR_PACKAGE_VERSIONS["safetensors"]
    audited.EVALUATOR_PACKAGE_VERSIONS = dict(EVALUATOR_PACKAGE_VERSIONS)
    audited.HF_MODEL_ARGS = dict(HF_MODEL_ARGS)
    audited.HF_LOCAL_MODEL_ARGS = dict(HF_LOCAL_MODEL_ARGS)
    audited.GENERATION_CONFIG = dict(GENERATION_CONFIG)
    audited.CONCURRENCY_CONFIG = dict(CONCURRENCY_CONFIG)
    audited.RUNTIME_GENERATION_CONFIG = dict(RUNTIME_GENERATION_CONFIG)
    audited.NO_TOKEN_CAP_RUNTIME_ENV = "CTM_HF_EOS_ONLY_NO_TOKEN_CAP"
    audited.NO_TOKEN_CAP_RUNTIME_ENV_VALUE = "1"
    audited.NO_TOKEN_CAP_RUNTIME_POLICY = _runtime_policy()
    audited.LAUNCH_SCHEMA = "muse-glimmer-rmct-aita-nta-flip-16gpu-launch-v1"
    audited.EVALUATION_SCHEMA = "muse-glimmer-rmct-aita-nta-flip-16gpu-evaluation-v1"
    audited.TASK_RECEIPT_SCHEMA = "muse-glimmer-rmct-aita-nta-flip-16gpu-task-receipt-v1"
    audited.SMOKE_RECEIPT_SCHEMA = FINAL_ANSWER_SMOKE_SCHEMA
    audited.COMPLETION_SCHEMA = "muse-glimmer-rmct-aita-nta-flip-16gpu-completion-v1"
    audited.GPU_TOPOLOGY_RECORD_SCHEMA = "muse-glimmer-rmct-aita-gpu-topology-record-v1"
    audited.GPU_TOPOLOGY_RECEIPT_SCHEMA = "muse-glimmer-rmct-aita-gpu-topology-receipt-v1"
    audited.CAMPAIGN_NAME = "muse-glimmer-rmct-aita-nta-flip-16gpu-v1"
    audited.CONDITIONS = (
        audited.Condition("base", "base", None, "pinned-snapshot"),
        # The inherited field is Qwen-specific.  Muse checkpoint 16/64 names
        # global/data-batch boundaries; realized optimizer counts remain in
        # each checkpoint's raw custody record and must not be counterfeited
        # here as 16/64.
        audited.Condition("step016", "global/data-step-016", None, "raw-training-checkpoint"),
        audited.Condition("step064", "global/data-step-064", None, "raw-training-checkpoint"),
        audited.Condition("final", "final", None, "raw-training-checkpoint"),
    )
    audited._CONDITION_BY_NAME = {condition.name: condition for condition in audited.CONDITIONS}
    audited.CRITICAL_SOURCES = (
        "infra/isambard/run_muse_glimmer_rmct_aita_ntaflip_16gpu.py",
        "infra/isambard/run_muse_glimmer_rmct_aita_ntaflip_16gpu.sbatch",
        "infra/isambard/run_muse_glimmer_rmct_aita_ntaflip_16gpu_worker.sh",
        "infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu.py",
        "infra/isambard/run_muse_glimmer_rmct_two_bias_evals_16gpu.py",
        "infra/isambard/muse_glimmer_rmct_segment_contract.py",
        "infra/isambard/muse_glimmer_rmct_segment_amendment.py",
        "experiments/muse_glimmer_rmct_replication/plan.py",
        "experiments/elephant_aita_ntaflip/__init__.py",
        "experiments/elephant_aita_ntaflip/prepare.py",
        "experiments/elephant_aita_ntaflip/tasks.py",
        "experiments/elephant_aita_ntaflip/preflight.py",
        "ctm/evals/hf_eos_only.py",
        "experiments/elephant_aita_ntaflip/no_cap_hf.py",
        "scripts/run_evals.py",
        "ctm/evals/runner.py",
        "ctm/evals/local_model.py",
    )
    audited._snapshot_identity = _snapshot_identity
    audited._validate_evaluator_environment = _validate_evaluator_environment
    audited._condition_runtime_records = _condition_runtime_records
    audited._benchmark_contract = _benchmark_contract
    audited._launch_policy = _launch_policy
    audited._metadata_expected = _metadata_expected
    audited._expected_runtime_from_condition_receipt = _expected_runtime_from_condition_receipt
    audited._task_runtime_binding = _task_runtime_binding
    audited._inspect_success = _inspect_success
    audited._task_command = _task_command
    audited._expected_runtime_for_preflight = _expected_runtime_for_preflight
    audited.smoke = smoke
    audited.finalize = finalize
    _assert_no_token_cap(GENERATION_CONFIG, label="Muse AITA sampling config")
    _assert_no_token_cap(RUNTIME_GENERATION_CONFIG, label="Muse AITA runtime generation config")
    plan._validate_spec(plan.FROZEN_SPEC)


def main(argv: Sequence[str] | None = None) -> int:
    _configure()
    return audited.main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONCURRENCY_CONFIG",
    "EVALUATOR_PACKAGE_VERSIONS",
    "GENERATION_CONFIG",
    "HF_LOCAL_MODEL_ARGS",
    "HF_MODEL_ARGS",
    "REASONING_OUTPUT_POLICY",
    "RUNTIME_GENERATION_CONFIG",
    "main",
]
