"""Build a fail-closed evaluation contract for fresh Aug-03 Qwen3.5 runs.

This launcher is deliberately a *planner*, not an executor.  It validates a
fresh target-scoped training publication, the exact raw LocalBackend LoRA
checkpoint it names, and the frozen Stage 1 and Stage 2 evaluation manifests.
It then writes a deterministic, write-once JSON contract containing the exact
generation, raw-preflight, staging, Luna, and analysis commands.

It never starts CUDA, vLLM, Inspect, or a model-based grader.  A person or a
separate scheduler must run the emitted commands in order after reviewing the
contract.  This boundary prevents a training-only Aug-03 plan from silently
falling back to a historical checkpoint, stale compatibility adapter, or
unattested Qwen3.5 vLLM path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctm.evals.local_model import read_local_checkpoint
from ctm.evals.qwen35_vllm_attestation import (
    COMPATIBILITY_MANIFEST_SCHEMA,
    PARITY_ATTESTATION_SCHEMA_V1,
    is_verified_qwen35_vllm_compat_adapter,
)
from scripts import run_experiment


LAUNCH_SCHEMA = "qwen35-fresh-onpolicy-evaluation-launch-v1"
MODEL = "Qwen/Qwen3.5-9B"
STAGE1_GRADER_MAX_TOKENS = 1024
STAGE2_GRADER_MAX_TOKENS = 256
DEFAULT_HF_MAX_CONNECTIONS = 8
_SOURCE_PREFIX = "base_model.model.model.layers."
_DESTINATION_PREFIX = "base_model.model.model.language_model.layers."
_RUNTIME_PARITY_REPORT_SCHEMA = "qwen35-lora-runtime-parity-v1"


@dataclass(frozen=True, slots=True)
class CurrentTarget:
    """One approved fresh Aug-03 training target and its only eval runtime."""

    target: str
    plan_relative_path: str
    experiment: str
    training_command: str
    training_script: str
    runtime_profile: str


CURRENT_TARGETS: dict[str, CurrentTarget] = {
    "rmct-main": CurrentTarget(
        target="rmct-main",
        plan_relative_path="experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct_paper_fidelity_isambard_phase2_4gpu_20260803.yaml",
        experiment="rmct_paper_isambard_phase2_qwen3_5_9b_rng_repair_4gpu_20260803",
        training_command="rate_matching_lr1",
        training_script="scripts/train_rlct.py",
        runtime_profile="hf-peft",
    ),
    "rmct-control": CurrentTarget(
        target="rmct-control",
        plan_relative_path="experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct_paper_fidelity_isambard_phase2_4gpu_20260803.yaml",
        experiment="rmct_paper_isambard_phase2_qwen3_5_9b_rng_repair_4gpu_20260803",
        training_command="rate_matching_control_lr1",
        training_script="scripts/train_rlct.py",
        runtime_profile="hf-peft",
    ),
    "opct": CurrentTarget(
        target="opct",
        plan_relative_path="experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_opct_recovery_20260803.yaml",
        experiment="rmct_paper_vast_dense_qwen3_5_9b_opct_recovery_20260803",
        training_command="opct_lr1",
        training_script="scripts/train_opct.py",
        runtime_profile="vllm",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_path(value: str | Path) -> Path:
    """Resolve a local path or a local ``file://`` URI without guessing."""

    raw = str(value)
    parsed = urlparse(raw)
    if parsed.scheme:
        if parsed.scheme.lower() != "file" or parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise ValueError(f"expected a local path or file URI, got {raw!r}")
        raw = unquote(parsed.path)
    if not raw:
        raise ValueError("path must be non-empty")
    return Path(raw).expanduser().resolve()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return document


def _safe_component(value: str, *, label: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{label} must be one safe path component")
    return value


def _positive_int(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256_hex(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def target_spec(target: str) -> CurrentTarget:
    """Return one current target, rejecting historical or unscoped names."""

    try:
        return CURRENT_TARGETS[target]
    except KeyError as exc:
        raise ValueError(f"unsupported fresh Qwen3.5 target {target!r}; choose from {sorted(CURRENT_TARGETS)}") from exc


def _current_plan(target: CurrentTarget, plan: str | Path | None) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    """Compile and constrain a supplied plan to the one approved Aug-03 target."""

    plan_path = _local_path(plan) if plan is not None else (PROJECT_ROOT / target.plan_relative_path).resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"fresh on-policy plan does not exist: {plan_path}")
    source = run_experiment.load_experiment_source(plan_path)
    if source.get("name") != target.experiment:
        raise ValueError(f"plan does not name the current {target.target!r} experiment: {plan_path}")
    authored_spec = source.get("spec")
    if not isinstance(authored_spec, Mapping):
        raise ValueError("current fresh on-policy plan must be factory-authored with a spec object")
    if authored_spec.get("model") != MODEL or authored_spec.get("training_only") is not True:
        raise ValueError("fresh on-policy evaluation requires the current training-only Qwen3.5 plan")

    compiled = run_experiment.compile_experiment(source)
    if "evaluation" in compiled or "analysis" in compiled or "rendering" in compiled:
        raise ValueError("the current fresh on-policy plan must remain training-only; evaluation belongs to this handoff")
    training_entries = [entry for entry in compiled.get("training", []) if entry.get("target") == target.target]
    if len(training_entries) != 1:
        raise ValueError(f"current plan does not compile exactly one training command for target {target.target!r}")
    entry = training_entries[0]
    if entry.get("name") != target.training_command:
        raise ValueError(f"current plan training command for {target.target!r} drifted from its target contract")
    command = entry.get("command")
    args = entry.get("args")
    if not isinstance(command, list) or not command or command[-1] != target.training_script or not isinstance(args, Mapping):
        raise ValueError(f"current plan training command for {target.target!r} has an unexpected executable contract")
    if args.get("model") != MODEL or args.get("backend") != "local":
        raise ValueError(f"current plan training command for {target.target!r} is not a local {MODEL} run")

    resolved_target_text = run_experiment.resolved_plan_text(compiled, target=target.target)
    return plan_path, source, compiled, resolved_target_text


def _local_lora_checkpoint_identity(checkpoint: str | Path) -> dict[str, Any]:
    """Validate one LocalBackend Qwen3.5 PEFT directory and record immutable bytes."""

    directory, manifest = read_local_checkpoint(checkpoint)
    if manifest.get("model") != MODEL or manifest.get("lora") is not True:
        raise ValueError(f"checkpoint is not a local Qwen3.5 LoRA checkpoint: {directory}")
    files = {
        "adapter_model_sha256": directory / "adapter_model.safetensors",
        "adapter_config_sha256": directory / "adapter_config.json",
        "checkpoint_manifest_sha256": directory / "manifest.json",
    }
    for path in files.values():
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"checkpoint requires a non-symlink regular artifact: {path}")
    config = _read_json_object(files["adapter_config_sha256"], label="PEFT adapter configuration")
    if config.get("base_model_name_or_path") != MODEL:
        raise ValueError(f"PEFT adapter config does not bind {MODEL}: {files['adapter_config_sha256']}")
    return {
        "path": str(directory),
        "uri": f"file://{directory}",
        "model": MODEL,
        "backend": "local",
        "lora": True,
        **{name: _sha256(path) for name, path in files.items()},
    }


def validate_target_training_output(
    *,
    target: str,
    checkpoint: str | Path,
    target_output_state: str | Path,
    target_resolved_plan: str | Path | None = None,
    plan: str | Path | None = None,
) -> dict[str, Any]:
    """Bind an explicit checkpoint to its one target-scoped training publication.

    The output state alone records only a checkpoint URI.  Its neighboring
    resolved target plan supplies the immutable command identity.  Rebuilding
    the current plan and comparing bytes prevents an arbitrary outputs.json
    with a familiar experiment name from entering the evaluation path.
    """

    target_definition = target_spec(target)
    plan_path, source, _compiled, expected_target_plan = _current_plan(target_definition, plan)
    state_path = _local_path(target_output_state)
    if not state_path.is_file():
        raise FileNotFoundError(f"target-scoped training outputs.json does not exist: {state_path}")
    state = _read_json_object(state_path, label="target-scoped training output state")
    if state.get("schema_version") != 1:
        raise ValueError("target-scoped training output state has an unsupported schema_version")
    if state.get("experiment") != target_definition.experiment or state.get("execution_target") != target_definition.target:
        raise ValueError("target-scoped training output state does not belong to the requested current target")
    checkpoints = state.get("training_checkpoints")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {target_definition.training_command}:
        raise ValueError("target-scoped training output state must publish exactly its current training command")
    published_checkpoint = checkpoints[target_definition.training_command]
    if not isinstance(published_checkpoint, str) or not published_checkpoint:
        raise ValueError("target-scoped training output state has no checkpoint URI")
    checkpoint_path = _local_path(checkpoint)
    if _local_path(published_checkpoint) != checkpoint_path:
        raise ValueError("explicit checkpoint does not match the target-scoped training output state")

    resolved_plan_path = _local_path(target_resolved_plan) if target_resolved_plan is not None else state_path.with_name("resolved-plan.yaml")
    if not resolved_plan_path.is_file():
        raise FileNotFoundError(f"target-scoped resolved plan does not exist: {resolved_plan_path}")
    if resolved_plan_path.read_text(encoding="utf-8") != expected_target_plan:
        raise ValueError("target-scoped resolved plan does not byte-match the current Aug-03 training plan")

    checkpoint_identity = _local_lora_checkpoint_identity(checkpoint_path)
    return {
        "target": target_definition.target,
        "experiment": target_definition.experiment,
        "training_command": target_definition.training_command,
        "training_plan": {
            "path": str(plan_path),
            "sha256": _sha256(plan_path),
            "resolved_target_plan": {
                "path": str(resolved_plan_path),
                "sha256": _sha256(resolved_plan_path),
            },
        },
        "target_output_state": {
            "path": str(state_path),
            "sha256": _sha256(state_path),
            "published_checkpoint": published_checkpoint,
        },
        "checkpoint": checkpoint_identity,
        "authored_plan_name": source["name"],
    }


def validate_stage1_manifest(manifest: str | Path) -> dict[str, Any]:
    """Revalidate the frozen no-CoT Stage 1 train/IID split contract locally."""

    from experiments.stage1_iid_diagnostic_none.prepare import validate_manifest

    manifest_path = _local_path(manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"frozen Stage 1 manifest does not exist: {manifest_path}")
    document = validate_manifest(manifest_path, verify_source=True)
    splits = document.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train_eval", "heldout_in_domain"}:
        raise ValueError("frozen Stage 1 manifest must contain train_eval and heldout_in_domain only")
    frozen_splits: dict[str, dict[str, Any]] = {}
    for split in ("train_eval", "heldout_in_domain"):
        entry = splits[split]
        if not isinstance(entry, Mapping):
            raise ValueError(f"frozen Stage 1 manifest has no {split} object")
        raw_path = entry.get("path")
        expected_sha = entry.get("content_sha256")
        if not isinstance(raw_path, str) or not raw_path or not isinstance(expected_sha, str):
            raise ValueError(f"frozen Stage 1 manifest has incomplete {split} identity")
        path = _local_path(raw_path)
        if not path.is_file() or _sha256(path) != expected_sha:
            raise ValueError(f"frozen Stage 1 {split} bytes do not match their manifest")
        frozen_splits[split] = {
            "path": str(path),
            "sha256": expected_sha,
            "row_count": entry.get("row_count"),
            "question_ids_sha256": entry.get("question_ids_sha256"),
        }
    return {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "prompt_style": "none",
        "splits": frozen_splits,
        "raw_task_count": 8,
        "metrics": ["conditional_tbsr", "luna_bias_acknowledgement"],
    }


def validate_stage2_manifest(manifest: str | Path) -> dict[str, Any]:
    """Revalidate the frozen held-out-IID/HLE Stage 2 matrix locally."""

    from experiments.stage2_ood_hle.materialize import PROMPT_STYLE, validate_manifest
    from experiments.stage2_ood_hle.tasks import ood_task_specs

    manifest_path = _local_path(manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"frozen Stage 2 manifest does not exist: {manifest_path}")
    document = validate_manifest(manifest_path)
    specs = ood_task_specs(manifest_path)
    clean_count = sum(spec.kind == "unbiased" for spec in specs)
    biased_count = sum(spec.kind == "biased" for spec in specs)
    populations = {spec.population for spec in specs}
    if (len(specs), clean_count, biased_count) != (21, 3, 18):
        raise ValueError("frozen Stage 2 manifest must resolve to 3 clean plus 18 biased task cells")
    if populations != {"in_domain", "hle"}:
        raise ValueError("frozen Stage 2 manifest must cover both held-out-IID and HLE populations")
    return {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "schema": document.get("schema"),
        "prompt_style": PROMPT_STYLE,
        "raw_task_count": len(specs),
        "clean_task_count": clean_count,
        "biased_task_count": biased_count,
        "populations": ["in_domain", "hle"],
        "metrics": ["conditional_tbsr", "luna_bias_acknowledgement"],
    }


def _runtime_parity_report_identity(
    report_path: Path,
    *,
    raw_checkpoint: Mapping[str, Any],
    compatibility: Mapping[str, Any],
) -> dict[str, str]:
    """Require a parity probe to have compared this raw/served adapter pair.

    A passing report for the served compatibility copy alone is insufficient:
    it could have compared against a different HF adapter.  The report's
    adapter fields are the missing identity edge between the fresh checkpoint
    and the HF side of the fixed-token probe.
    """

    report = _read_json_object(report_path, label="OPCT HF/vLLM runtime parity report")
    if report.get("schema") != _RUNTIME_PARITY_REPORT_SCHEMA or report.get("model") != MODEL:
        raise ValueError("OPCT HF/vLLM runtime parity report has an unsupported model/schema")
    adapter = report.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError("OPCT HF/vLLM runtime parity report has no adapter identity")
    for field, expected_path, expected_hash in (
        ("path", _local_path(str(compatibility["path"])), compatibility["adapter_model_sha256"]),
        ("hf_path", _local_path(str(raw_checkpoint["path"])), raw_checkpoint["adapter_model_sha256"]),
    ):
        observed_path = adapter.get(field)
        hash_field = "adapter_model_sha256" if field == "path" else "hf_adapter_model_sha256"
        if not isinstance(observed_path, str) or _local_path(observed_path) != expected_path:
            raise ValueError(f"OPCT HF/vLLM runtime parity report {field} does not name the fresh adapter pair")
        if adapter.get(hash_field) != expected_hash:
            raise ValueError(f"OPCT HF/vLLM runtime parity report {hash_field} does not match the fresh adapter pair")
    return {"path": str(report_path), "sha256": _sha256(report_path)}


def validate_opct_vllm_compatibility(
    *,
    raw_checkpoint: Mapping[str, Any],
    vllm_compat_adapter: str | Path,
) -> dict[str, Any]:
    """Prove OPCT's served vLLM adapter is a translated, parity-attested copy.

    Qwen3.5 cannot safely send its raw PEFT tensor namespace to vLLM.  This
    validates both edges of the necessary provenance chain: raw fresh adapter
    -> translated compatibility copy -> immutable HF/vLLM runtime parity.
    """

    raw_path = _local_path(str(raw_checkpoint["path"]))
    compatibility = _local_lora_checkpoint_identity(vllm_compat_adapter)
    compatibility_path = _local_path(str(compatibility["path"]))
    manifest_path = compatibility_path / "compatibility-manifest.json"
    attestation_path = compatibility_path / "vllm-parity-attestation.json"
    for path in (manifest_path, attestation_path):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"OPCT vLLM compatibility adapter is missing immutable evidence: {path}")
    if not is_verified_qwen35_vllm_compat_adapter(compatibility_path):
        raise ValueError("OPCT vLLM compatibility adapter lacks a valid HF/vLLM runtime parity attestation")

    translation = _read_json_object(manifest_path, label="OPCT vLLM compatibility manifest")
    if translation.get("schema") != COMPATIBILITY_MANIFEST_SCHEMA:
        raise ValueError("OPCT vLLM compatibility adapter has an unsupported translation-manifest schema")
    source = translation.get("source")
    destination = translation.get("destination")
    details = translation.get("translation")
    if not isinstance(source, Mapping) or not isinstance(destination, Mapping) or not isinstance(details, Mapping):
        raise ValueError("OPCT vLLM compatibility manifest has incomplete source/destination/translation evidence")
    source_path = source.get("path")
    destination_path = destination.get("path")
    if not isinstance(source_path, str) or _local_path(source_path) != raw_path:
        raise ValueError("OPCT vLLM compatibility adapter was not translated from this fresh raw checkpoint")
    if not isinstance(destination_path, str) or _local_path(destination_path) != compatibility_path:
        raise ValueError("OPCT vLLM compatibility manifest does not name the supplied served adapter")
    if source.get("adapter_model_sha256") != raw_checkpoint["adapter_model_sha256"]:
        raise ValueError("OPCT vLLM compatibility source adapter SHA-256 does not match the fresh raw checkpoint")
    if destination.get("adapter_model_sha256") != compatibility["adapter_model_sha256"]:
        raise ValueError("OPCT vLLM compatibility destination adapter SHA-256 does not match its adapter bytes")
    if details.get("source_prefix") != _SOURCE_PREFIX or details.get("destination_prefix") != _DESTINATION_PREFIX:
        raise ValueError("OPCT vLLM compatibility manifest has an unexpected Qwen3.5 key translation")
    tensor_count = details.get("tensor_count")
    translated_tensor_count = details.get("translated_tensor_count")
    if (
        isinstance(tensor_count, bool)
        or not isinstance(tensor_count, int)
        or tensor_count < 1
        or tensor_count != translated_tensor_count
    ):
        raise ValueError("OPCT vLLM compatibility manifest has invalid tensor translation counts")
    _sha256_hex(details.get("translated_tensor_names_sha256"), label="OPCT translated tensor-name SHA-256")

    attestation = _read_json_object(attestation_path, label="OPCT vLLM parity attestation")
    parity: dict[str, Any] = {
        "path": str(attestation_path),
        "sha256": _sha256(attestation_path),
        "schema": attestation.get("schema"),
    }
    if attestation.get("schema") == PARITY_ATTESTATION_SCHEMA_V1:
        report_value = attestation.get("report_path")
        report_sha = attestation.get("report_sha256")
        if not isinstance(report_value, str) or not report_value:
            raise ValueError("OPCT vLLM parity attestation has no runtime report path")
        report_path = _local_path(report_value)
        if not report_path.is_file() or report_sha != _sha256(report_path):
            raise ValueError("OPCT vLLM parity attestation runtime report does not match its recorded SHA-256")
        parity["runtime_reports"] = [
            _runtime_parity_report_identity(
                report_path,
                raw_checkpoint=raw_checkpoint,
                compatibility=compatibility,
            )
        ]
    else:
        # The shared validator fully revalidates the narrowly permitted
        # composite schema.  Bind its primary HF adapter to this raw
        # checkpoint as well; otherwise a valid composite proof for another
        # adapter could be paired with a translation manifest by path alone.
        primary = attestation.get("primary")
        if not isinstance(primary, Mapping):
            raise ValueError("OPCT composite parity attestation has no primary evidence")
        hf_adapter = primary.get("hf_adapter")
        report_entry = primary.get("report")
        if not isinstance(hf_adapter, Mapping) or not isinstance(report_entry, Mapping):
            raise ValueError("OPCT composite parity attestation has incomplete primary identity")
        hf_path = hf_adapter.get("path")
        if not isinstance(hf_path, str) or _local_path(hf_path) != raw_path:
            raise ValueError("OPCT composite parity HF adapter does not name the fresh raw checkpoint")
        if hf_adapter.get("adapter_model_sha256") != raw_checkpoint["adapter_model_sha256"]:
            raise ValueError("OPCT composite parity HF adapter hash does not match the fresh raw checkpoint")
        report_value = report_entry.get("path")
        report_sha = report_entry.get("sha256")
        if not isinstance(report_value, str) or not isinstance(report_sha, str):
            raise ValueError("OPCT composite parity attestation has no primary runtime report identity")
        report_path = _local_path(report_value)
        if not report_path.is_file() or report_sha != _sha256(report_path):
            raise ValueError("OPCT composite parity primary runtime report does not match its recorded SHA-256")
        parity["runtime_reports"] = [
            _runtime_parity_report_identity(
                report_path,
                raw_checkpoint=raw_checkpoint,
                compatibility=compatibility,
            )
        ]
    return {
        "profile": "vllm",
        "served_compatibility_adapter": compatibility,
        "translation_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "runtime_parity": parity,
    }


def _json_argument(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def _command(name: str, argv: Sequence[str], *, kind: str) -> dict[str, Any]:
    rendered = [str(token) for token in argv]
    return {
        "name": name,
        "kind": kind,
        "cwd": str(PROJECT_ROOT),
        "argv": rendered,
        "display": shlex.join(rendered),
    }


def _runtime_settings(
    *,
    profile: str,
    raw_checkpoint: Mapping[str, Any],
    vllm_compatibility: Mapping[str, Any] | None,
    hf_max_connections: int,
) -> dict[str, Any]:
    """Return one evaluator runtime; target policy makes the choice non-optional."""

    generation = {"max_tokens": 20480, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
    if profile == "hf-peft":
        connections = _positive_int(hf_max_connections, label="hf_max_connections")
        return {
            "profile": profile,
            "checkpoint": raw_checkpoint["path"],
            "model_args": {"provider": "hf", "device": "cuda:0", "dtype": "bfloat16"},
            "generation_config": {**generation, "max_connections": connections},
            "execution": {"max_tasks": 1, "isolate_tasks": True, "persistent_vllm_server": False},
        }
    if profile == "vllm":
        if vllm_compatibility is None:
            raise ValueError("OPCT vLLM evaluation requires a translated, parity-attested compatibility adapter")
        served = vllm_compatibility["served_compatibility_adapter"]
        return {
            "profile": profile,
            "checkpoint": served["path"],
            "model_args": {
                "provider": "vllm",
                "gpu_memory_utilization": 0.9,
                "max_model_len": 32768,
                "language_model_only": True,
                "max_num_seqs": 256,
            },
            "generation_config": {**generation, "extra_body": {"top_k": 20}},
            "execution": {"max_tasks": 1, "isolate_tasks": True, "persistent_vllm_server": True},
        }
    raise AssertionError(f"unsupported evaluator profile: {profile}")


def _raw_eval_command(
    *,
    name: str,
    task_factory: str,
    task_args: Mapping[str, Any],
    log_dir: Path,
    runtime: Mapping[str, Any],
    python: str,
    limit: int,
) -> dict[str, Any]:
    argv = [
        python,
        str(PROJECT_ROOT / "scripts" / "run_evals.py"),
        "--task-factory",
        task_factory,
        "--local-checkpoint",
        str(runtime["checkpoint"]),
        "--base-model",
        MODEL,
        "--task-args",
        _json_argument(task_args),
        "--model-args",
        _json_argument(runtime["model_args"]),
        "--generation-config",
        _json_argument(runtime["generation_config"]),
        "--log-dir",
        str(log_dir),
        "--limit",
        str(limit),
        "--max-tasks",
        "1",
        "--isolate-tasks",
    ]
    if runtime["execution"]["persistent_vllm_server"]:
        argv.append("--persistent-vllm-server")
    argv.append("--yes")
    return _command(name, argv, kind="gpu_generation")


def _output_paths(output_root: str | Path, condition: str) -> dict[str, Path]:
    root = _local_path(output_root)
    condition = _safe_component(condition, label="condition")
    return {
        "root": root,
        "stage1_raw": root / "stage1" / "raw",
        "stage1_preflight": root / "stage1" / "preflight" / f"{condition}.json",
        "stage1_staged_raw": root / "stage1" / "staged-raw",
        "stage1_luna": root / "stage1" / "luna",
        "stage1_analysis": root / "stage1" / "analysis" / f"{condition}.json",
        "stage2_raw": root / "stage2" / "raw",
        "stage2_preflight": root / "stage2" / "preflight" / f"{condition}.json",
        "stage2_staged_raw": root / "stage2" / "staged-raw",
        "stage2_luna": root / "stage2" / "luna",
        "stage2_analysis": root / "stage2" / "analysis" / f"{condition}.json",
    }


def build_launch_contract(
    *,
    target: str,
    checkpoint: str | Path,
    target_output_state: str | Path,
    stage1_manifest: str | Path,
    stage2_manifest: str | Path,
    output_root: str | Path,
    plan: str | Path | None = None,
    target_resolved_plan: str | Path | None = None,
    vllm_compat_adapter: str | Path | None = None,
    condition: str | None = None,
    runtime_profile: str | None = None,
    hf_max_connections: int = DEFAULT_HF_MAX_CONNECTIONS,
    python: str = sys.executable,
    contract_output: str | Path | None = None,
) -> dict[str, Any]:
    """Validate local evidence and return an exact, non-executing eval handoff.

    RMCT main/control are deliberately native HF/PEFT only.  OPCT is
    deliberately vLLM only, but receives a translated adapter that has already
    passed fresh HF/vLLM parity checks tied to this checkpoint's tensor hash.
    """

    definition = target_spec(target)
    requested_runtime = runtime_profile or definition.runtime_profile
    if requested_runtime != definition.runtime_profile:
        raise ValueError(
            f"{definition.target} must use runtime_profile={definition.runtime_profile!r}; "
            "the fresh-checkpoint handoff does not permit a backend substitution"
        )
    if not isinstance(python, str) or not python:
        raise ValueError("python must be a non-empty command path")
    condition = _safe_component(condition or definition.target, label="condition")
    training = validate_target_training_output(
        target=target,
        checkpoint=checkpoint,
        target_output_state=target_output_state,
        target_resolved_plan=target_resolved_plan,
        plan=plan,
    )
    stage1 = validate_stage1_manifest(stage1_manifest)
    stage2 = validate_stage2_manifest(stage2_manifest)
    vllm_compatibility: dict[str, Any] | None = None
    if definition.runtime_profile == "vllm":
        if vllm_compat_adapter is None:
            raise ValueError("OPCT requires --vllm-compat-adapter before any vLLM evaluation command is emitted")
        vllm_compatibility = validate_opct_vllm_compatibility(
            raw_checkpoint=training["checkpoint"],
            vllm_compat_adapter=vllm_compat_adapter,
        )
    elif vllm_compat_adapter is not None:
        raise ValueError("--vllm-compat-adapter is only valid for the OPCT target")

    runtime = _runtime_settings(
        profile=definition.runtime_profile,
        raw_checkpoint=training["checkpoint"],
        vllm_compatibility=vllm_compatibility,
        hf_max_connections=hf_max_connections,
    )
    paths = _output_paths(output_root, condition)
    contract_path = _local_path(contract_output) if contract_output is not None else None
    stage1_task_args = {
        "manifest": stage1["path"],
        "unbiased_log": str(paths["stage1_raw"]),
        "prompt_style": "none",
        "include_bias_acknowledged": False,
    }
    stage2_task_args = {
        "manifest": stage2["path"],
        "unbiased_log": str(paths["stage2_raw"]),
        "prompt_style": "none",
        "include_bias_acknowledged": False,
    }
    commands: list[dict[str, Any]] = [
        _raw_eval_command(
            name="stage1_raw_generation",
            task_factory="experiments.stage1_iid_diagnostic_none.tasks:diagnostic_matrix_tasks",
            task_args=stage1_task_args,
            log_dir=paths["stage1_raw"],
            runtime=runtime,
            python=python,
            limit=100,
        ),
        _command(
            "stage1_raw_preflight",
            [
                python,
                "-m",
                "experiments.stage1_iid_diagnostic_none.raw_preflight",
                "--raw-log-root",
                str(paths["stage1_raw"]),
                "--manifest",
                stage1["path"],
                "--split-file",
                f"train_eval={stage1['splits']['train_eval']['path']}",
                "--split-file",
                f"heldout_in_domain={stage1['splits']['heldout_in_domain']['path']}",
                "--condition",
                condition,
                "--expected-base-model",
                MODEL,
                "--expected-checkpoint",
                str(runtime["checkpoint"]),
                "--runtime-profile",
                str(runtime["profile"]),
                *(
                    ["--expected-max-connections", str(runtime["generation_config"]["max_connections"])]
                    if runtime["profile"] == "hf-peft"
                    else []
                ),
                *(
                    [
                        "--expected-adapter-model-sha256",
                        training["checkpoint"]["adapter_model_sha256"],
                        "--expected-adapter-config-sha256",
                        training["checkpoint"]["adapter_config_sha256"],
                        "--expected-checkpoint-manifest-sha256",
                        training["checkpoint"]["checkpoint_manifest_sha256"],
                    ]
                    if runtime["profile"] == "hf-peft"
                    else []
                ),
                "--output",
                str(paths["stage1_preflight"]),
            ],
            kind="cpu_preflight",
        ),
        _command(
            "stage1_stage_hash_bound_raw_logs",
            [
                python,
                "-m",
                "experiments.stage1_iid_diagnostic_none.stage_luna",
                "--preflight-report",
                str(paths["stage1_preflight"]),
                "--output-root",
                str(paths["stage1_staged_raw"]),
            ],
            kind="cpu_staging",
        ),
        _command(
            "stage1_luna_verbalisation",
            [
                python,
                "-m",
                "experiments.stage1_iid_diagnostic_none.grade_luna",
                "--raw-log-root",
                str(paths["stage1_staged_raw"]),
                "--preflight-report",
                str(paths["stage1_preflight"]),
                "--output-root",
                str(paths["stage1_luna"]),
                "--workers",
                "5",
                "--connections-per-worker",
                "100",
                "--grader-max-tokens",
                str(STAGE1_GRADER_MAX_TOKENS),
            ],
            kind="paid_luna_grading",
        ),
        _command(
            "stage1_tbsr_and_luna_analysis",
            [
                python,
                "-m",
                "experiments.stage1_iid_diagnostic.analyze",
                "--graded-root",
                str(paths["stage1_luna"]),
                "--manifest",
                stage1["path"],
                "--grader-max-tokens",
                str(STAGE1_GRADER_MAX_TOKENS),
                "--output",
                str(paths["stage1_analysis"]),
            ],
            kind="cpu_analysis",
        ),
        _raw_eval_command(
            name="stage2_raw_generation",
            task_factory="experiments.stage2_ood_hle.tasks:ood_tasks",
            task_args=stage2_task_args,
            log_dir=paths["stage2_raw"],
            runtime=runtime,
            python=python,
            limit=200,
        ),
        _command(
            "stage2_raw_preflight",
            [
                python,
                "-m",
                "experiments.stage2_ood_hle.raw_preflight",
                "--raw-log-root",
                str(paths["stage2_raw"]),
                "--manifest",
                stage2["path"],
                "--condition",
                condition,
                "--runtime-profile",
                str(runtime["profile"]),
                "--expected-base-model",
                MODEL,
                "--expected-checkpoint",
                str(runtime["checkpoint"]),
                *(
                    ["--expected-max-connections", str(runtime["generation_config"]["max_connections"])]
                    if runtime["profile"] == "hf-peft"
                    else []
                ),
                "--output",
                str(paths["stage2_preflight"]),
            ],
            kind="cpu_preflight",
        ),
        _command(
            "stage2_stage_hash_bound_raw_logs",
            [
                python,
                "-m",
                "experiments.stage2_ood_hle.stage_luna",
                "--preflight-report",
                str(paths["stage2_preflight"]),
                "--output-root",
                str(paths["stage2_staged_raw"]),
            ],
            kind="cpu_staging",
        ),
        _command(
            "stage2_luna_verbalisation",
            [
                python,
                "-m",
                "experiments.stage2_ood_hle.grade_luna",
                "--staged-raw-root",
                str(paths["stage2_staged_raw"]),
                "--preflight-report",
                str(paths["stage2_preflight"]),
                "--output-root",
                str(paths["stage2_luna"]),
                "--workers",
                "5",
                "--connections-per-worker",
                "100",
                "--grader-max-tokens",
                str(STAGE2_GRADER_MAX_TOKENS),
            ],
            kind="paid_luna_grading",
        ),
        _command(
            "stage2_tbsr_and_luna_analysis",
            [
                python,
                "-m",
                "experiments.stage2_ood_hle.analyze",
                "--run",
                f"{condition}={paths['stage2_luna'] / condition}",
                "--stage2-manifest",
                stage2["path"],
                "--expected-prompt-style",
                "none",
                "--output",
                str(paths["stage2_analysis"]),
            ],
            kind="cpu_analysis",
        ),
    ]
    if contract_path is not None:
        commands.insert(
            0,
            _command(
                "revalidate_fresh_contract",
                [
                    python,
                    str(PROJECT_ROOT / "scripts" / "launch_fresh_onpolicy_evals.py"),
                    "--verify-contract",
                    str(contract_path),
                ],
                kind="cpu_contract_guard",
            ),
        )
    return {
        "schema": LAUNCH_SCHEMA,
        **({"contract_path": str(contract_path), "launcher_python": python} if contract_path is not None else {}),
        "target": definition.target,
        "condition": condition,
        "model": MODEL,
        "training": training,
        "runtime": {
            **runtime,
            **({"vllm_compatibility": vllm_compatibility} if vllm_compatibility is not None else {}),
        },
        "evaluation": {
            "stage1": {
                **stage1,
                "grader_max_tokens": STAGE1_GRADER_MAX_TOKENS,
                "raw_generation_has_luna": False,
                "required_splits": ["train_eval", "heldout_in_domain"],
            },
            "stage2": {
                **stage2,
                "grader_max_tokens": STAGE2_GRADER_MAX_TOKENS,
                "raw_generation_has_luna": False,
                "required_populations": ["in_domain", "hle"],
            },
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "execution_policy": {
            "launcher_executes_commands": False,
            "required_order": [command["name"] for command in commands],
            "first_command_revalidates_contract": contract_path is not None,
            "luna_credentials": "must be configured by the operator; no credential is recorded in this contract",
        },
        "commands": commands,
    }


def _serialized_contract(contract: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(contract), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def validate_new_output_layout(contract: Mapping[str, Any], output: str | Path) -> None:
    """Refuse to seed a fresh contract inside an already-used evaluation root."""

    outputs = contract.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get("root"), str):
        raise ValueError("launch contract has no output root")
    root = _local_path(outputs["root"])
    destination = _local_path(output)
    if destination == root:
        raise ValueError("launch contract output must be a JSON file, not the evaluation output root")
    for name, value in outputs.items():
        if name == "root" or not isinstance(value, str):
            continue
        generated_path = _local_path(value)
        if destination == generated_path or destination.is_relative_to(generated_path):
            raise ValueError(f"launch contract output must not be nested inside generated {name} evidence")
    if destination.exists():
        return
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f"evaluation output root is not a directory: {root}")
        if any(root.iterdir()):
            raise FileExistsError(
                f"refusing to seed a fresh launch contract in non-empty evaluation output root: {root}"
            )


def write_launch_contract(path: str | Path, contract: Mapping[str, Any]) -> str:
    """Write one exact contract once, or resume only byte-identical evidence."""

    destination = _local_path(path)
    payload = _serialized_contract(contract)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing fresh on-policy evaluation contract: {destination}")
        return "resumed"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(payload.decode("utf-8"))
            handle.flush()
    except FileExistsError:
        if not destination.is_file() or destination.read_bytes() != payload:
            raise FileExistsError(f"fresh on-policy evaluation contract appeared and differs: {destination}")
        return "resumed"
    return "written"


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"fresh on-policy evaluation contract has no {label} object")
    return value


def _required_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"fresh on-policy evaluation contract has no {label} path")
    return value


def validate_launch_contract(path: str | Path) -> dict[str, Any]:
    """Rebuild a stored contract and reject any changed input before execution.

    This is a local, CPU-only command.  It is intentionally the first emitted
    command for contracts written through the CLI, so a checkpoint, frozen
    manifest, target output state, or OPCT parity artifact cannot change after
    planning but before generation without being detected.
    """

    contract_path = _local_path(path)
    if not contract_path.is_file():
        raise FileNotFoundError(f"fresh on-policy evaluation contract does not exist: {contract_path}")
    stored = _read_json_object(contract_path, label="fresh on-policy evaluation contract")
    if stored.get("schema") != LAUNCH_SCHEMA:
        raise ValueError("fresh on-policy evaluation contract has an unsupported schema")
    if stored.get("contract_path") != str(contract_path):
        raise ValueError("fresh on-policy evaluation contract does not bind its own path")
    target = stored.get("target")
    condition = stored.get("condition")
    python = stored.get("launcher_python")
    if not isinstance(target, str) or not isinstance(condition, str) or not isinstance(python, str) or not python:
        raise ValueError("fresh on-policy evaluation contract has incomplete target/condition/python identity")
    training = _required_mapping(stored.get("training"), label="training")
    checkpoint = _required_mapping(training.get("checkpoint"), label="training.checkpoint")
    output_state = _required_mapping(training.get("target_output_state"), label="training.target_output_state")
    training_plan = _required_mapping(training.get("training_plan"), label="training.training_plan")
    resolved_target_plan = _required_mapping(training_plan.get("resolved_target_plan"), label="training.training_plan.resolved_target_plan")
    evaluation = _required_mapping(stored.get("evaluation"), label="evaluation")
    stage1 = _required_mapping(evaluation.get("stage1"), label="evaluation.stage1")
    stage2 = _required_mapping(evaluation.get("stage2"), label="evaluation.stage2")
    outputs = _required_mapping(stored.get("outputs"), label="outputs")
    runtime = _required_mapping(stored.get("runtime"), label="runtime")
    runtime_profile = runtime.get("profile")
    if runtime_profile not in {"hf-peft", "vllm"}:
        raise ValueError("fresh on-policy evaluation contract has an invalid runtime profile")
    if runtime_profile == "hf-peft":
        generation = _required_mapping(runtime.get("generation_config"), label="runtime.generation_config")
        hf_max_connections = _positive_int(generation.get("max_connections"), label="contract hf max_connections")
        vllm_compat_adapter: str | None = None
    else:
        hf_max_connections = DEFAULT_HF_MAX_CONNECTIONS
        vllm = _required_mapping(runtime.get("vllm_compatibility"), label="runtime.vllm_compatibility")
        served = _required_mapping(vllm.get("served_compatibility_adapter"), label="runtime.vllm_compatibility.served_compatibility_adapter")
        vllm_compat_adapter = _required_path(served.get("path"), label="served vLLM compatibility adapter")

    rebuilt = build_launch_contract(
        target=target,
        checkpoint=_required_path(checkpoint.get("path"), label="fresh checkpoint"),
        target_output_state=_required_path(output_state.get("path"), label="target output state"),
        target_resolved_plan=_required_path(resolved_target_plan.get("path"), label="target resolved plan"),
        plan=_required_path(training_plan.get("path"), label="training plan"),
        stage1_manifest=_required_path(stage1.get("path"), label="Stage 1 manifest"),
        stage2_manifest=_required_path(stage2.get("path"), label="Stage 2 manifest"),
        output_root=_required_path(outputs.get("root"), label="output root"),
        vllm_compat_adapter=vllm_compat_adapter,
        condition=condition,
        runtime_profile=str(runtime_profile),
        hf_max_connections=hf_max_connections,
        python=python,
        contract_output=contract_path,
    )
    if stored != rebuilt:
        raise ValueError("fresh on-policy evaluation contract does not exactly revalidate its current inputs")
    return {"path": str(contract_path), "sha256": _sha256(contract_path), "contract": rebuilt}


def _print_commands(contract: Mapping[str, Any]) -> None:
    print(f"Validated {contract['target']} as condition {contract['condition']}.")
    print("This launcher has not started a model, GPU process, or Luna request.")
    for command in contract["commands"]:
        print(f"\n[{command['name']} | {command['kind']}] cwd={command['cwd']}")
        print(f"  {command['display']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--target", required=True, choices=sorted(CURRENT_TARGETS))
    parser.add_argument("--checkpoint", required=True, type=Path, help="explicit fresh raw LocalBackend checkpoint")
    parser.add_argument(
        "--target-output-state",
        required=True,
        type=Path,
        help="target-scoped training outputs.json published by the selected target",
    )
    parser.add_argument(
        "--target-resolved-plan",
        type=Path,
        help="target-scoped resolved-plan.yaml; defaults to the sibling of --target-output-state",
    )
    parser.add_argument("--plan", type=Path, help="current Aug-03 source plan; defaults to the target's approved plan")
    parser.add_argument("--stage1-manifest", required=True, type=Path, help="frozen no-CoT train/IID manifest")
    parser.add_argument("--stage2-manifest", required=True, type=Path, help="frozen held-out-IID/HLE manifest")
    parser.add_argument("--output-root", required=True, type=Path, help="new root for this condition's generated evidence")
    parser.add_argument("--output", required=True, type=Path, help="write-once JSON launch contract")
    parser.add_argument("--condition", help="safe condition label; defaults to --target")
    parser.add_argument("--runtime-profile", choices=("hf-peft", "vllm"), help="must equal the target's required runtime")
    parser.add_argument(
        "--vllm-compat-adapter",
        type=Path,
        help="OPCT-only translated adapter with fresh HF/vLLM runtime-parity attestation",
    )
    parser.add_argument("--hf-max-connections", type=int, default=DEFAULT_HF_MAX_CONNECTIONS)
    parser.add_argument("--python", default=sys.executable, help="interpreter placed in emitted commands")
    parser.add_argument("--dry-run", action="store_true", help="validate and print without writing the launch contract")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    requested = list(sys.argv[1:] if argv is None else argv)
    if requested[:1] == ["--verify-contract"]:
        verifier = argparse.ArgumentParser(description="Revalidate one fresh on-policy evaluation contract locally")
        verifier.add_argument("--verify-contract", required=True, type=Path)
        verify_args = verifier.parse_args(requested)
        try:
            result = validate_launch_contract(verify_args.verify_contract)
        except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            verifier.error(str(exc))
            return
        print(f"validated: {result['path']} (sha256:{result['sha256']})")
        return
    parser = _parser()
    args = parser.parse_args(requested)
    try:
        contract = build_launch_contract(
            target=args.target,
            checkpoint=args.checkpoint,
            target_output_state=args.target_output_state,
            target_resolved_plan=args.target_resolved_plan,
            plan=args.plan,
            stage1_manifest=args.stage1_manifest,
            stage2_manifest=args.stage2_manifest,
            output_root=args.output_root,
            vllm_compat_adapter=args.vllm_compat_adapter,
            condition=args.condition,
            runtime_profile=args.runtime_profile,
            hf_max_connections=args.hf_max_connections,
            python=args.python,
            contract_output=args.output,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
        return
    if args.dry_run:
        _print_commands(contract)
        return
    try:
        validate_new_output_layout(contract, args.output)
        status = write_launch_contract(args.output, contract)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return
    print(f"{status}: {_local_path(args.output)}")
    _print_commands(contract)


__all__ = [
    "CURRENT_TARGETS",
    "DEFAULT_HF_MAX_CONNECTIONS",
    "LAUNCH_SCHEMA",
    "MODEL",
    "STAGE1_GRADER_MAX_TOKENS",
    "STAGE2_GRADER_MAX_TOKENS",
    "CurrentTarget",
    "build_launch_contract",
    "target_spec",
    "validate_new_output_layout",
    "validate_launch_contract",
    "validate_opct_vllm_compatibility",
    "validate_stage1_manifest",
    "validate_stage2_manifest",
    "validate_target_training_output",
    "write_launch_contract",
]


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
