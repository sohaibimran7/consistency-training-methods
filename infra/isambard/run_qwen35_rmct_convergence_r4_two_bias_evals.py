#!/usr/bin/env python3
"""Run the sealed r4 step-176 Stage-2 two-bias evaluation safely.

This is deliberately a narrow Isambard execution boundary.  It accepts only
the terminal r4 checkpoint, makes an immutable Qwen3.5 vLLM compatibility
copy, proves that copy against the untouched HF adapter, and then executes the
frozen 21-cell Stage-2 substrate.  There is no native-HF fallback in this
condition: a missing or failed parity attestation stops the run before raw
evaluation logs are created.

The evaluator is resumable without treating a partial Inspect log as a result:

* each task is generated in a fresh, preserved attempt directory;
* only a successful, task-index-bound ``.eval`` file is hard-linked/captured
  into the canonical raw tree and receives a write-once receipt;
* biased task groups cannot begin until the three canonical clean receipts are
  present and inspect-readable; and
* the new two-bias preflight is the sole authority for the completed 21-cell
  matrix and its seen/held-out bias labels.

It never submits Slurm work, never chains successors, and never overwrites a
different log, receipt, compatibility copy, parity proof, or deployment
manifest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA = "rmct-convergence-r4-step176-two-bias-eval-launch-v1"
TASK_RECEIPT_SCHEMA = "rmct-convergence-r4-step176-two-bias-task-receipt-v2"
RUNTIME_RECEIPT_SCHEMA = "rmct-convergence-r4-step176-two-bias-runtime-receipt-v1"
FINAL_SEGMENT_INDEX = 10
FINAL_OPTIMIZER_STEP = 176
RUN_PREFIX = "rmct-convergence-gcall-r2-mb40960-r4"
RUN_NAME = f"{RUN_PREFIX}-s011"
# r005 is deliberately a fresh immutable condition.  The r001--r004 roots
# remain custody evidence for their failed/obsolete parity protocols.
CONDITION = "rmct-convergence-r4-s011-two-bias-v1-r005"
MODEL_SNAPSHOT = Path(
    "/lus/lfs1aip2/scratch/a5v/sohaib.a5v/ctm/huggingface/hub/"
    "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
STAGE1_PARITY_MANIFEST_SHA256 = "8cdd4da0575a125b01b2b9b62d9c0e07176142eec6194e72e8abd04401ca2fec"
STAGE1_PARITY_MANIFEST_KIND = "stage1_iid_diagnostic_none_manifest"
STAGE1_PARITY_MANIFEST_SCHEMA_VERSION = 1
STAGE1_PARITY_TRAIN_EVAL_FILENAME = "train-eval-n200.jsonl"
STAGE1_PARITY_TRAIN_EVAL_ROWS = 200
STAGE2_TASK_FACTORY = "experiments.stage2_ood_hle.tasks:ood_tasks"
TASK_COUNT = 21
CLEAN_TASK_INDICES = (1, 2, 3)
BIASED_TASK_INDICES = tuple(range(4, 22))
SEEN_BIASES = ("wrong_argument", "suggested_answer")
HELD_OUT_BIASES = ("distractor_fact", "post_hoc", "spurious_few_shot_squares", "wrong_few_shot")
GENERATION_CONFIG = {
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
# vLLM 0.21.0 enables FlashInfer top-k/top-p sampling by default on CUDA.
# The installed Isambard build has no nvcc, so that default tries to JIT a
# FlashInfer sampler and fails before parity.  This is an explicit frozen
# runtime choice, rather than a best-effort fallback: r005 requires native
# PyTorch sampling in both the direct parity probes and persistent evaluators.
VLLM_VERSION = "0.21.0"
VLLM_SAMPLER_ENVIRONMENT = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
VLLM_SAMPLER_IMPLEMENTATION = "pytorch_native"
# The fast r005 parity path remains a strict transport diagnostic: HF runs
# once, then each of these exact variants runs in a fresh, one-LoRA vLLM
# server on its own Slurm-visible GPU token.  Keep this local copy aligned
# with ``runtime_parity.RESULT_VARIANTS`` rather than accepting a subset.
PARITY_RESULT_VARIANTS = ("full", "linear_only", "self_attn_only", "evaluator_path")
PARITY_SERVER_COUNT = len(PARITY_RESULT_VARIANTS)
DEFAULT_PARITY_DEVICE_TOKENS = ("0", "1", "2", "3")
# The frozen parity protocol requests at most 29 unique next-token scores per
# one-token completion (16 top tokens plus 13 unique candidate IDs; a newline
# candidate token is shared; observed maximum 29). vLLM defaults to 20, so
# pin the server admission cap to this
# exact protocol bound. This changes neither prompts, requested IDs, nor any
# compared score.
PARITY_TOP_TOKEN_COUNT = 16
VLLM_PARITY_MAX_LOGPROBS = 29
# r005 asks vLLM for processed logprobs over exactly the requested token IDs.
# Every score-affecting sampling control is explicitly neutral so an API or
# server-default change cannot silently alter the parity measurement.
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
CRITICAL_SOURCES = (
    "infra/isambard/run_qwen35_rmct_convergence_r4_two_bias_evals.py",
    "infra/isambard/run_qwen35_rmct_convergence_r4_two_bias_evals.sbatch",
    "scripts/run_evals.py",
    "ctm/evals/runner.py",
    "ctm/evals/local_model.py",
    "ctm/evals/qwen35_vllm_attestation.py",
    "ctm/evals/qwen35_vllm_scope.py",
    "ctm/backends/local/packed_lora.py",
    "experiments/act_repair_gate/runtime_parity.py",
    "experiments/act_repair_gate/vllm_compat_adapter.py",
    "experiments/stage1_iid_diagnostic_none/prepare.py",
    "experiments/stage2_ood_hle/tasks.py",
    "experiments/stage2_ood_hle/raw_preflight.py",
    "experiments/stage2_ood_hle/materialize.py",
    "experiments/rmct_two_bias_eval/__init__.py",
    "experiments/rmct_two_bias_eval/contract.py",
    "experiments/rmct_two_bias_eval/deployment.py",
    "experiments/rmct_two_bias_eval/raw_preflight.py",
    "infra/isambard/verify_rmct_convergence_r4_recovery_production_ready.py",
    "experiments/rmct_convergence_r4_recovery/plan.py",
    "experiments/rmct_convergence/controller.py",
)
_SHA256_CHARS = frozenset("0123456789abcdef")


class EvaluationError(ValueError):
    """The terminal checkpoint or evaluation evidence is unsafe to use."""


@dataclass(frozen=True)
class LaunchPaths:
    root: Path
    contract: Path
    deployment_manifest: Path
    raw: Path
    attempts: Path
    receipts: Path
    runtime: Path
    evaluation_receipt: Path
    completion: Path


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationError(f"{label} must be a regular file: {path}")
    size = path.stat().st_size
    if size < 1:
        raise EvaluationError(f"{label} must not be empty: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256_file(path), "size_bytes": size}


def _critical_source_identities() -> dict[str, dict[str, Any]]:
    """Hash every implementation boundary that affects this exact eval run."""

    records: dict[str, dict[str, Any]] = {}
    for relative in CRITICAL_SOURCES:
        path = _under_root(PROJECT_ROOT / relative, PROJECT_ROOT, label=f"critical source {relative}")
        records[relative] = _identity(path, label=f"critical source {relative}")
    return records


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be a JSON object: {path}")
    return value


def _safe_component(value: str, *, label: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise EvaluationError(f"{label} must be one non-empty safe path component")
    return value


def _under_root(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvaluationError(f"{label} escapes its expected root: {resolved}") from exc
    return resolved


def _write_immutable_json(path: Path, value: Mapping[str, Any], *, label: str) -> str:
    """Create a JSON receipt once, accepting only byte-identical resume."""

    payload = _canonical_json(value)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing {label}: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise EvaluationError(f"{label} parent must be a regular directory: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(f"{label} appeared with different bytes: {path}") from None
            return "resumed"
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def _launch_paths(output_root: str | Path) -> LaunchPaths:
    root = Path(output_root).expanduser().resolve()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise EvaluationError(f"evaluation output root must be a regular directory: {root}")
    return LaunchPaths(
        root=root,
        contract=root / "launch-contract.json",
        deployment_manifest=root / "input" / "stage2-deployment-manifest.json",
        raw=root / "stage2" / "raw",
        attempts=root / "stage2" / "attempts",
        receipts=root / "stage2" / "receipts",
        runtime=root / "runtime",
        evaluation_receipt=root / "runtime" / "evaluation-receipt.json",
        completion=root / "stage2" / "completion.json",
    )


def _expected_checkpoint(repository: Path) -> Path:
    return (
        repository
        / "logs"
        / "rmct-convergence"
        / RUN_NAME
        / "checkpoints"
        / f"rmct-convergence_{RUN_NAME}"
    )


def _validate_model_snapshot() -> dict[str, Any]:
    # Hugging Face snapshots are usually directories whose files are symbolic
    # links into that model cache's content-addressed ``blobs/`` directory.
    # Accept that one documented layout without relaxing the generic artifact
    # identity rule used for checkpoints, logs, or receipts.
    snapshot = MODEL_SNAPSHOT
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.parent.name != "snapshots":
        raise EvaluationError(f"pinned Qwen3.5 snapshot is absent or incomplete: {snapshot}")
    logical_config = snapshot / "config.json"
    if not logical_config.exists():
        raise EvaluationError(f"pinned Qwen3.5 snapshot is absent or incomplete: {snapshot}")
    if logical_config.is_symlink():
        model_cache = snapshot.parent.parent
        blob_tree = model_cache / "blobs"
        if blob_tree.is_symlink() or not blob_tree.is_dir():
            raise EvaluationError(f"pinned model config has no regular Hugging Face blob tree: {logical_config}")
        resolved_config = logical_config.resolve()
        if resolved_config.is_symlink() or not resolved_config.is_file() or resolved_config.stat().st_size < 1:
            raise EvaluationError(f"pinned model config must resolve to a non-empty regular file: {logical_config}")
        try:
            resolved_config.relative_to(blob_tree.resolve())
        except ValueError as exc:
            raise EvaluationError(
                f"pinned model config must resolve under the same Hugging Face model blob tree: {logical_config}"
            ) from exc
    else:
        # Preserve the strict ordinary-file guard for non-cache layouts.
        _identity(logical_config, label="pinned model config")
        resolved_config = logical_config.resolve()
    return {
        "path": str(snapshot),
        "config": {
            "logical_path": str(logical_config),
            "resolved_path": str(resolved_config),
            "sha256": _sha256_file(resolved_config),
            "size_bytes": resolved_config.stat().st_size,
        },
    }


def validate_final_checkpoint(repository: str | Path, checkpoint: str | Path | None = None) -> dict[str, Any]:
    """Require the one terminal r4 step-176 checkpoint and its convergence receipt."""

    root = Path(repository).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise EvaluationError(f"repository must be a regular directory: {root}")
    selected = Path(checkpoint).expanduser().resolve() if checkpoint is not None else _expected_checkpoint(root).resolve()
    if selected != _expected_checkpoint(root).resolve():
        raise EvaluationError(f"evaluation accepts only the sealed r4 step-{FINAL_OPTIMIZER_STEP} checkpoint: {_expected_checkpoint(root)}")
    _under_root(selected, root, label="terminal checkpoint")

    # Reuse the established strict all-rank final-state verifier rather than
    # accepting a weights-only adapter directory.
    try:
        from infra.isambard import verify_rmct_convergence_r4_recovery_production_ready as ready
    except ImportError as exc:  # pragma: no cover - deployment error
        raise EvaluationError("r4 strict checkpoint custody verifier is unavailable") from exc
    try:
        custody = ready._strict_checkpoint_identity(
            root,
            selected,
            expected_segment_index=FINAL_SEGMENT_INDEX,
            label="terminal r4 evaluation checkpoint",
        )
    except Exception as exc:  # keep the launch boundary independent of verifier internals
        raise EvaluationError(f"terminal r4 checkpoint fails strict resumability validation: {exc}") from exc

    manifest = _read_json(selected / "manifest.json", label="terminal checkpoint manifest")
    adapter = _read_json(selected / "adapter_config.json", label="terminal adapter configuration")
    if manifest.get("model") != str(MODEL_SNAPSHOT) or adapter.get("base_model_name_or_path") != str(MODEL_SNAPSHOT):
        raise EvaluationError("terminal checkpoint and adapter configuration must bind the exact pinned Qwen3.5 snapshot")

    decisions = selected.parent.parent / "decisions"
    if decisions.is_symlink() or not decisions.is_dir():
        raise EvaluationError(f"terminal r4 decision directory is absent or linked: {decisions}")
    candidates = sorted(decisions.glob(f"checkpoint-window-decision-s{FINAL_SEGMENT_INDEX:03d}-*.json"))
    if len(candidates) != 1:
        raise EvaluationError(f"expected exactly one terminal r4 decision receipt, found {len(candidates)}")
    try:
        from experiments.rmct_convergence import controller

        decision = controller.verify_decision_receipt(candidates[0])
    except Exception as exc:
        raise EvaluationError(f"terminal r4 convergence decision cannot be replayed: {exc}") from exc
    target = decision.get("target")
    if (
        decision.get("decision") != "converged"
        or decision.get("afterok") != {"permit_training": False, "successor_action": "no_op"}
        or not isinstance(target, Mapping)
        or target.get("segment_index") != FINAL_SEGMENT_INDEX
        or target.get("optimizer_step_end") != FINAL_OPTIMIZER_STEP
    ):
        raise EvaluationError("terminal receipt is not the converged r4 step-176 no-successor decision")
    return {
        "path": str(selected),
        "custody": custody,
        "manifest": _identity(selected / "manifest.json", label="terminal checkpoint manifest"),
        "adapter_config": _identity(selected / "adapter_config.json", label="terminal adapter configuration"),
        "adapter_model": _identity(selected / "adapter_model.safetensors", label="terminal adapter weights"),
        "decision_receipt": _identity(candidates[0], label="terminal convergence decision"),
    }


def _two_bias_contract_module():
    try:
        from experiments.rmct_two_bias_eval import contract
    except ImportError as exc:  # pragma: no cover - deployment error
        raise EvaluationError("the r005 two-bias evaluation contract package is unavailable") from exc
    return contract


def validate_two_bias_substrate(manifest: str | Path) -> dict[str, Any]:
    """Bind the new two-bias semantics before emitting any model command."""

    contract = _two_bias_contract_module()
    path = Path(manifest).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise EvaluationError(f"two-bias Stage-2 manifest must be a regular file: {path}")
    try:
        validated = contract.validate_stage2_substrate(path)
    except Exception as exc:
        raise EvaluationError(f"two-bias Stage-2 substrate validation failed: {exc}") from exc
    if not isinstance(validated, Mapping):
        raise EvaluationError("two-bias Stage-2 substrate validator returned no structured contract")
    if tuple(getattr(contract, "SEEN_BIASES", ())) != SEEN_BIASES:
        raise EvaluationError("two-bias contract does not classify both wrong_argument and suggested_answer as seen")
    if tuple(getattr(contract, "HELD_OUT_BIASES", ())) != HELD_OUT_BIASES:
        raise EvaluationError("two-bias contract has an unexpected held-out bias set")
    # The frozen task factory remains the established 3-clean + 18-biased
    # substrate.  Check its topology independently of any report labels.
    try:
        from experiments.stage2_ood_hle.tasks import ood_task_specs

        specs = ood_task_specs(path)
    except Exception as exc:
        raise EvaluationError(f"frozen Stage-2 task matrix is unavailable: {exc}") from exc
    biases = {spec.bias_type for spec in specs if spec.kind == "biased"}
    if (
        len(specs) != TASK_COUNT
        or sum(spec.kind == "unbiased" for spec in specs) != len(CLEAN_TASK_INDICES)
        or sum(spec.kind == "biased" for spec in specs) != len(BIASED_TASK_INDICES)
        or biases != set((*SEEN_BIASES, *HELD_OUT_BIASES))
    ):
        raise EvaluationError("frozen Stage-2 task matrix is not exactly 3 clean plus 18 two-bias-aware biased cells")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "task_count": len(specs),
        "clean_task_count": len(CLEAN_TASK_INDICES),
        "biased_task_count": len(BIASED_TASK_INDICES),
        "seen_biases": list(SEEN_BIASES),
        "held_out_biases": list(HELD_OUT_BIASES),
        "contract": dict(validated),
    }


def materialize_deployment_manifest(
    source_manifest: str | Path,
    *,
    artifact_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Use the two-bias portable-manifest bridge; never edit JSONL payloads."""

    try:
        from experiments.rmct_two_bias_eval import deployment
    except ImportError as exc:  # pragma: no cover - deployment error
        raise EvaluationError("the r005 two-bias deployment-manifest bridge is unavailable") from exc
    try:
        record = deployment.materialize_deployment_manifest(
            source_manifest,
            artifact_root,
            output,
        )
    except Exception as exc:
        raise EvaluationError(f"Stage-2 deployment manifest could not be materialized safely: {exc}") from exc
    if not isinstance(record, Mapping):
        raise EvaluationError("deployment-manifest bridge returned no structured provenance")
    destination = Path(record.get("manifest_path", "")).expanduser().resolve()
    if destination != Path(output).expanduser().resolve():
        raise EvaluationError("deployment-manifest bridge did not bind the requested output path")
    return dict(record)


def _raw_checkpoint_identity(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """The minimal local-LoRA identity used to bind translation and parity."""

    path = Path(str(checkpoint["path"])).resolve()
    return {
        "path": str(path),
        "adapter_model_sha256": str(checkpoint["adapter_model"]["sha256"]),
        "adapter_config_sha256": str(checkpoint["adapter_config"]["sha256"]),
        "checkpoint_manifest_sha256": str(checkpoint["manifest"]["sha256"]),
    }


def _validate_translation_only(raw: Mapping[str, Any], compatibility: Path) -> None:
    """Validate a pre-attestation conversion without mistaking it for usable vLLM."""

    manifest_path = compatibility / "compatibility-manifest.json"
    weights = compatibility / "adapter_model.safetensors"
    config = compatibility / "adapter_config.json"
    for path, label in ((manifest_path, "compatibility manifest"), (weights, "compatibility weights"), (config, "compatibility config")):
        _identity(path, label=label)
    document = _read_json(manifest_path, label="compatibility manifest")
    source = document.get("source")
    destination = document.get("destination")
    translation = document.get("translation")
    if not isinstance(source, Mapping) or not isinstance(destination, Mapping) or not isinstance(translation, Mapping):
        raise EvaluationError("compatibility manifest has incomplete source/destination/translation identity")
    if (
        document.get("schema") != "qwen35-vllm-compat-adapter-v1"
        or source.get("path") != raw["path"]
        or source.get("adapter_model_sha256") != raw["adapter_model_sha256"]
        or destination.get("path") != str(compatibility)
        or destination.get("adapter_model_sha256") != _sha256_file(weights)
        or translation.get("source_prefix") != "base_model.model.model.layers."
        or translation.get("destination_prefix") != "base_model.model.model.language_model.layers."
    ):
        raise EvaluationError("compatibility copy does not prove the exact raw-adapter Qwen3.5 key translation")


def _next_attempt(root: Path, *, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise EvaluationError(f"attempt root must be a regular directory: {root}")
    for index in range(1, 10_000):
        candidate = root / f"{label}-{index:04d}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise EvaluationError(f"exhausted preserved attempt slots under {root}")


def _require_parity_device_tokens(
    values: Sequence[object], *, label: str
) -> tuple[str, str, str, str]:
    """Require four exact opaque CUDA_VISIBLE_DEVICES tokens.

    Slurm may expose logical indices, physical indices, or GPU UUIDs.  They
    must therefore remain strings end-to-end; converting them to integers
    would silently select a different device on UUID-based allocations.
    """

    if isinstance(values, (str, bytes)):
        raise EvaluationError(f"{label} must contain four separate CUDA device tokens")
    tokens = tuple(values)
    if (
        len(tokens) != PARITY_SERVER_COUNT
        or any(
            not isinstance(token, str)
            or not token
            or token != token.strip()
            or any(character.isspace() for character in token)
            or "," in token
            or token in {"-1", "NoDevFiles"}
            for token in tokens
        )
        or len(set(tokens)) != PARITY_SERVER_COUNT
    ):
        raise EvaluationError(
            f"{label} must contain exactly four distinct non-empty CUDA_VISIBLE_DEVICES tokens"
        )
    return tokens  # type: ignore[return-value]


def _parse_gpu_tokens(value: str, *, label: str) -> tuple[str, str, str, str]:
    if not isinstance(value, str):
        raise EvaluationError(f"{label} must be a comma-separated CUDA device-token string")
    return _require_parity_device_tokens(tuple(value.split(",")), label=label)


def _vllm_sampler_runtime(
    *, vllm_device_tokens: Sequence[object] = DEFAULT_PARITY_DEVICE_TOKENS
) -> dict[str, Any]:
    """Return r005's immutable native-sampler and parity-topology profile."""

    tokens = _require_parity_device_tokens(vllm_device_tokens, label="r005 parity device tokens")
    return {
        "vllm_version": VLLM_VERSION,
        "environment": dict(VLLM_SAMPLER_ENVIRONMENT),
        "implementation": VLLM_SAMPLER_IMPLEMENTATION,
        # ``runtime_parity`` has a CLI-specific spelling; persistent eval
        # uses the model-args spelling above.  Bind both so neither quietly
        # follows a changed vLLM default.
        "parity_gdn_prefill_backend": VLLM_GDN_PREFILL_BACKEND,
        "parity_top_token_count": PARITY_TOP_TOKEN_COUNT,
        "parity_max_logprobs": VLLM_PARITY_MAX_LOGPROBS,
        "parity_logprobs_mode": VLLM_PARITY_LOGPROBS_MODE,
        "parity_score_transport": VLLM_PARITY_SCORE_TRANSPORT,
        "parity_sampling": dict(VLLM_PARITY_SAMPLING),
        "persistent_gdn_prefill_backend": VLLM_MODEL_ARGS["gdn_prefill_backend"],
        # Bind the fast path itself.  A single multi-LoRA server or a
        # sequential one-GPU retry is a different numerical diagnostic and
        # cannot reuse this r005 contract.
        "isolate_vllm_variants": True,
        "parallel_isolated_vllm_variants": True,
        "enforce_eager": True,
        "vllm_device_tokens": list(tokens),
        "vllm_device_count": PARITY_SERVER_COUNT,
        "one_adapter_per_server": True,
        "max_loras_per_server": 1,
        "parity_result_variants": list(PARITY_RESULT_VARIANTS),
    }


def _validate_vllm_sampler_runtime(launch: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless this process exactly matches r005's sampler ABI."""

    launch_runtime = launch.get("runtime")
    sampler = launch_runtime.get("sampler") if isinstance(launch_runtime, Mapping) else None
    try:
        tokens = _require_parity_device_tokens(
            sampler.get("vllm_device_tokens", ()) if isinstance(sampler, Mapping) else (),
            label="launch-contract vLLM parity device tokens",
        )
    except EvaluationError as exc:
        raise EvaluationError("launch contract does not bind four exact r005 parity device tokens") from exc
    expected = _vllm_sampler_runtime(vllm_device_tokens=tokens)
    if (
        not isinstance(launch_runtime, Mapping)
        or sampler != expected
        or launch_runtime.get("model_args") != VLLM_MODEL_ARGS
    ):
        raise EvaluationError("launch contract does not bind the required vLLM 0.21 native sampler runtime")
    for name, value in VLLM_SAMPLER_ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise EvaluationError(
                f"{name} must be exactly {value!r} for the sealed r005 vLLM sampler runtime"
            )
    try:
        import vllm
    except ImportError as exc:  # pragma: no cover - deployment error
        raise EvaluationError("vLLM is unavailable for the sealed r005 sampler runtime") from exc
    version = getattr(vllm, "__version__", None)
    if version != VLLM_VERSION:
        raise EvaluationError(
            f"r005 requires vLLM {VLLM_VERSION} for its pinned sampler runtime, got {version!r}"
        )
    return expected


def _validate_parity_report_runtime(report: Mapping[str, Any], sampler: Mapping[str, Any]) -> None:
    """Check launch-bound r005 runtime and score-transport settings."""

    backends = report.get("backends")
    vllm = backends.get("vllm") if isinstance(backends, Mapping) else None
    expected_prefill = sampler.get("parity_gdn_prefill_backend")
    expected_top_token_count = sampler.get("parity_top_token_count")
    expected_max_logprobs = sampler.get("parity_max_logprobs")
    expected_logprobs_mode = sampler.get("parity_logprobs_mode")
    expected_score_transport = sampler.get("parity_score_transport")
    expected_sampling = sampler.get("parity_sampling")
    if (
        not isinstance(vllm, Mapping)
        or expected_top_token_count != PARITY_TOP_TOKEN_COUNT
        or expected_max_logprobs != VLLM_PARITY_MAX_LOGPROBS
        or expected_logprobs_mode != VLLM_PARITY_LOGPROBS_MODE
        or expected_score_transport != VLLM_PARITY_SCORE_TRANSPORT
        or expected_sampling != VLLM_PARITY_SAMPLING
        or vllm.get("gdn_prefill_backend") != expected_prefill
        or vllm.get("max_logprobs") != expected_max_logprobs
        or vllm.get("logprobs_mode") != expected_logprobs_mode
        or vllm.get("score_transport") != expected_score_transport
        or vllm.get("parity_sampling") != expected_sampling
        or vllm.get("isolate_vllm_variants") is not True
        or vllm.get("parallel_isolated_vllm_variants") is not True
        or vllm.get("enforce_eager") is not True
    ):
        raise EvaluationError(
            "runtime parity report does not bind the r005 eager, isolated triton processed-logprob runtime"
        )
    protocol = report.get("token_protocol")
    plan = vllm.get("parallel_isolated_server_plan")
    tokens = sampler.get("vllm_device_tokens")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("requested_result_variants") != list(PARITY_RESULT_VARIANTS)
        or protocol.get("top_token_count") != expected_top_token_count
        or protocol.get("vllm_score_transport") != expected_score_transport
        or protocol.get("vllm_allowed_token_ids") != "requested_token_ids"
        or protocol.get("vllm_response_token_ids") != "exactly_requested_token_ids"
        or not isinstance(plan, Mapping)
        or set(plan) != set(PARITY_RESULT_VARIANTS)
        or not isinstance(tokens, list)
    ):
        raise EvaluationError(
            "runtime parity report does not bind the canonical r005 allowed-token four-server topology"
        )
    actual_tokens: list[str] = []
    for variant in PARITY_RESULT_VARIANTS:
        server = plan.get(variant)
        if (
            not isinstance(server, Mapping)
            or not isinstance(server.get("device_token"), str)
            or server.get("max_logprobs") != expected_max_logprobs
            or server.get("logprobs_mode") != expected_logprobs_mode
        ):
            raise EvaluationError(
                "runtime parity report has an invalid r005 server device token, processed-logprob mode, or logprob cap"
            )
        actual_tokens.append(server["device_token"])
    if actual_tokens != tokens:
        raise EvaluationError(
            "runtime parity report device-token order differs from the immutable r005 launch contract"
        )


def _validate_r005_parity_report(
    report_path: Path,
    *,
    compatibility_adapter: Path,
    sampler: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the local binding checks and the public r005 report validator."""

    document = _read_json(report_path, label="runtime parity report")
    _validate_parity_report_runtime(document, sampler)
    contract = _two_bias_contract_module()
    try:
        validated = contract.validate_r005_parallel_parity_report(
            report_path,
            compatibility_adapter=compatibility_adapter,
        )
    except Exception as exc:
        raise EvaluationError(f"runtime parity report fails the strict r005 four-server validator: {exc}") from exc
    if not isinstance(validated, Mapping):
        raise EvaluationError("strict r005 parity validator returned no report")
    return dict(validated)


def _attested_r005_compatibility(
    *, raw: Mapping[str, Any], compatibility_adapter: Path, sampler: Mapping[str, Any]
) -> dict[str, Any]:
    """Reopen an existing attestation and reject non-r005 parity proof reuse."""

    contract = _two_bias_contract_module()
    try:
        identity = contract.vllm_compatibility_identity(
            compatibility_adapter,
            raw_checkpoint=raw,
        )
    except Exception as exc:
        raise EvaluationError(f"compatibility adapter lacks strict r005 parity custody: {exc}") from exc
    reports = identity.get("parity_reports") if isinstance(identity, Mapping) else None
    if not isinstance(reports, list) or not reports or not isinstance(reports[0], Mapping):
        raise EvaluationError("compatibility adapter has no primary r005 parity report identity")
    report_path = reports[0].get("path")
    if not isinstance(report_path, str) or not Path(report_path).is_absolute():
        raise EvaluationError("compatibility adapter has an invalid primary r005 parity report path")
    _validate_r005_parity_report(
        Path(report_path),
        compatibility_adapter=compatibility_adapter,
        sampler=sampler,
    )
    return dict(identity)


def _validate_inherited_cuda_tokens(expected: Sequence[object]) -> tuple[str, str, str, str]:
    """Require the launch contract to preserve Slurm's four visible tokens."""

    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        raise EvaluationError("r005 parallel parity requires inherited CUDA_VISIBLE_DEVICES")
    actual = _parse_gpu_tokens(inherited, label="inherited CUDA_VISIBLE_DEVICES")
    bound = _require_parity_device_tokens(expected, label="launch-contract vLLM parity device tokens")
    if actual != bound:
        raise EvaluationError(
            "launch-contract parity device tokens differ from inherited CUDA_VISIBLE_DEVICES; refusing to remap Slurm GPUs"
        )
    return bound


def _parity_command(
    *, python: str, selected: Path, raw: Mapping[str, Any], launch: Mapping[str, Any], attempt: Path, sampler: Mapping[str, Any]
) -> list[str]:
    tokens = _require_parity_device_tokens(
        sampler.get("vllm_device_tokens", ()) if isinstance(sampler, Mapping) else (),
        label="launch-contract vLLM parity device tokens",
    )
    scope_args: list[str] = []
    if launch.get("adapter_scope") is not None:
        from ctm.evals.qwen35_vllm_scope import SCOPE

        if launch["adapter_scope"] != SCOPE:
            raise EvaluationError("unsupported explicit adapter scope")
        scope_args = ["--adapter-scope", SCOPE]
    return [
        python,
        "-m",
        "experiments.act_repair_gate.runtime_parity",
        "--model",
        str(MODEL_SNAPSHOT),
        "--adapter",
        str(selected),
        "--hf-adapter",
        str(raw["path"]),
        "--data",
        str(launch["parity_data"]["path"]),
        "--output-dir",
        str(attempt),
        "--enforce-eager",
        "--isolate-vllm-variants",
        "--parallel-isolated-vllm-variants",
        "--vllm-device-tokens",
        *tokens,
        "--top-token-count",
        str(sampler["parity_top_token_count"]),
        "--max-logprobs",
        str(sampler["parity_max_logprobs"]),
        "--gdn-prefill-backend",
        str(sampler["parity_gdn_prefill_backend"]),
        *scope_args,
    ]


def ensure_attested_vllm_runtime(
    *,
    launch: Mapping[str, Any],
    paths: LaunchPaths,
    python: str,
) -> dict[str, Any]:
    """Translate, probe, and attest one exact compatibility copy or stop.

    A failed parity attempt stays in ``runtime/parity-attempts`` for custody;
    the next invocation may make a new probe attempt, but never switches to
    native HF inside this immutable vLLM evaluation condition.
    """

    sampler_runtime = _validate_vllm_sampler_runtime(launch)
    _validate_inherited_cuda_tokens(sampler_runtime["vllm_device_tokens"])
    raw = _raw_checkpoint_identity(launch["checkpoint"])
    runtime_root = paths.runtime
    compatibility_root = runtime_root / "compatibility"
    parity_attempts = runtime_root / "parity-attempts"
    compatibility_root.mkdir(parents=True, exist_ok=True)
    if compatibility_root.is_symlink() or not compatibility_root.is_dir():
        raise EvaluationError(f"compatibility root must be a regular directory: {compatibility_root}")

    selected: Path | None = None
    unattested_candidate: Path | None = None
    for candidate in sorted(item for item in compatibility_root.iterdir() if item.is_dir() and not item.is_symlink()):
        required_translation_files = (
            candidate / "adapter_config.json",
            candidate / "adapter_model.safetensors",
            candidate / "compatibility-manifest.json",
        )
        # A conversion can be interrupted between its individual writes.  It
        # remains preserved forensic evidence, but cannot prevent a later
        # fresh immutable conversion attempt from proceeding.
        if any(path.is_symlink() or not path.is_file() for path in required_translation_files):
            continue
        # Once all translation artifacts are present, malformed/mismatched
        # provenance is a conflict rather than a recoverable partial write.
        _validate_translation_only(raw, candidate)
        attestation = candidate / "vllm-parity-attestation.json"
        if attestation.exists():
            # This is intentionally delegated to the established complete
            # validator, which also checks the raw-HF adapter link in its
            # parity report.
            try:
                from ctm.evals.qwen35_vllm_attestation import is_verified_qwen35_vllm_compat_adapter
            except ImportError as exc:  # pragma: no cover - deployment error
                raise EvaluationError("Qwen3.5 vLLM attestation verifier is unavailable") from exc
            if not is_verified_qwen35_vllm_compat_adapter(candidate):
                raise EvaluationError(f"existing compatibility adapter has an invalid parity attestation: {candidate}")
            # A generic v1 attestation may have been made by the earlier
            # sequential/multi-LoRA mode.  Preserve it, but do not let it
            # substitute for r005's exact four-device protocol.
            try:
                _attested_r005_compatibility(
                    raw=raw,
                    compatibility_adapter=candidate,
                    sampler=sampler_runtime,
                )
            except EvaluationError:
                continue
            selected = candidate
            break
        if unattested_candidate is None:
            unattested_candidate = candidate
    if selected is None:
        selected = unattested_candidate
        if selected is None:
            selected = _next_attempt(compatibility_root, label="adapter")
            try:
                from experiments.act_repair_gate.vllm_compat_adapter import make_compat_adapter

                make_compat_adapter(raw["path"], selected)
            except Exception as exc:
                raise EvaluationError(f"could not create a new immutable vLLM compatibility adapter: {exc}") from exc
            _validate_translation_only(raw, selected)

    attestation = selected / "vllm-parity-attestation.json"
    if not attestation.exists():
        report_paths = sorted(
            item / "report.json"
            for item in parity_attempts.iterdir()
            if item.is_dir() and not item.is_symlink() and (item / "report.json").is_file()
        ) if parity_attempts.is_dir() else []
        report: Path | None = None
        for candidate in report_paths:
            try:
                document = _read_json(candidate, label="runtime parity report")
                adapter = document.get("adapter")
                if isinstance(adapter, Mapping) and adapter.get("path") == str(selected) and adapter.get("hf_path") == raw["path"]:
                    _validate_r005_parity_report(
                        candidate,
                        compatibility_adapter=selected,
                        sampler=sampler_runtime,
                    )
                    report = candidate
                    break
            except EvaluationError:
                # An old, partial, or weaker parity attempt remains evidence
                # only.  A fresh immutable report is required before this
                # compatibility directory may be attested for r005.
                continue
        if report is None:
            attempt = _next_attempt(parity_attempts, label="parity")
            command = _parity_command(
                python=python,
                selected=selected,
                raw=raw,
                launch=launch,
                attempt=attempt,
                sampler=sampler_runtime,
            )
            environment = os.environ.copy()
            environment.update(sampler_runtime["environment"])
            for name in ("VLLM_BASE_URL", "VLLM_API_KEY", "CTM_PERSISTENT_VLLM_SERVER_METADATA"):
                environment.pop(name, None)
            completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=environment, check=False)
            if completed.returncode:
                raise EvaluationError(
                    f"HF/vLLM parity attempt exited with {completed.returncode}; preserved evidence: {attempt}"
                )
            report = attempt / "report.json"
            _identity(report, label="runtime parity report")
            _validate_r005_parity_report(
                report,
                compatibility_adapter=selected,
                sampler=sampler_runtime,
            )
        try:
            from experiments.act_repair_gate.vllm_compat_adapter import attest_compat_adapter

            attest_compat_adapter(selected, report)
        except Exception as exc:
            raise EvaluationError(f"runtime parity did not support a direct immutable attestation: {exc}") from exc

    compatibility_identity = _attested_r005_compatibility(
        raw=raw,
        compatibility_adapter=selected,
        sampler=sampler_runtime,
    )
    report = _read_json(attestation, label="vLLM parity attestation")
    runtime = {
        "schema": RUNTIME_RECEIPT_SCHEMA,
        "profile": "vllm",
        "raw_checkpoint": raw,
        "compatibility_adapter": {
            "path": str(selected),
            "adapter_model_sha256": _sha256_file(selected / "adapter_model.safetensors"),
            "translation_manifest": _identity(selected / "compatibility-manifest.json", label="compatibility manifest"),
            "attestation": _identity(attestation, label="vLLM parity attestation"),
            "attestation_schema": report.get("schema"),
            "parity_reports": compatibility_identity.get("parity_reports"),
        },
        "model_snapshot": str(MODEL_SNAPSHOT),
        "sampler": sampler_runtime,
    }
    _write_immutable_json(paths.runtime / "runtime-receipt.json", runtime, label="vLLM runtime receipt")
    return runtime


def build_vllm_evaluation_receipt(
    *,
    launch: Mapping[str, Any],
    runtime: Mapping[str, Any],
    paths: LaunchPaths,
) -> dict[str, Any]:
    """Write and re-verify the two-bias receipt after direct vLLM parity.

    The static launch contract intentionally cannot name a compatibility
    adapter which has not yet existed.  This second, write-once receipt binds
    every raw EvalLog to the selected raw checkpoint *and* the attested
    translation/parity chain before any task is allowed to run.
    """

    try:
        from experiments.rmct_two_bias_eval import contract
    except ImportError as exc:  # pragma: no cover - deployment error
        raise EvaluationError("the r005 two-bias evaluation-receipt contract is unavailable") from exc
    convergence = launch["checkpoint"].get("decision_receipt")
    if not isinstance(convergence, Mapping) or not isinstance(convergence.get("path"), str):
        raise EvaluationError("launch contract lacks the terminal convergence decision identity")
    compatibility = runtime.get("compatibility_adapter")
    if not isinstance(compatibility, Mapping) or not isinstance(compatibility.get("path"), str):
        raise EvaluationError("attested vLLM runtime lacks its compatibility-adapter identity")
    try:
        receipt = contract.build_evaluation_receipt(
            manifest=launch["deployment_manifest"]["path"],
            checkpoint=launch["checkpoint"]["path"],
            convergence_receipt=convergence["path"],
            condition=CONDITION,
            raw_log_root=paths.raw,
            runtime_profile="vllm",
            vllm_compat_adapter=compatibility["path"],
        )
        status = contract.write_evaluation_receipt(paths.evaluation_receipt, receipt)
        verified = contract.verify_evaluation_receipt(paths.evaluation_receipt)
    except Exception as exc:
        raise EvaluationError(f"two-bias vLLM evaluation receipt could not be built and reverified: {exc}") from exc
    if not isinstance(verified, Mapping) or verified.get("runtime", {}).get("profile") != "vllm":
        raise EvaluationError("two-bias evaluation receipt does not bind the vLLM runtime")
    return {
        "path": str(paths.evaluation_receipt),
        "sha256": _sha256_file(paths.evaluation_receipt),
        "status": status,
        "document": dict(verified),
    }


def _task_receipt_path(paths: LaunchPaths, task_index: int) -> Path:
    return paths.receipts / f"task-{task_index:03d}.json"


def _inspect_success(path: Path, *, task_index: int) -> None:
    """Check just enough Inspect metadata to make the clean barrier real."""

    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - evaluation environment only
        raise EvaluationError("Inspect AI is required to validate raw task receipts") from exc
    try:
        log = read_eval_log(str(path), header_only=True)
    except Exception as exc:
        raise EvaluationError(f"could not read canonical Inspect log: {path}") from exc
    status = getattr(log, "status", None)
    evaluation = getattr(log, "eval", None)
    metadata = getattr(evaluation, "metadata", {}) if evaluation is not None else {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    if status != "success" or metadata.get("task_indices") != [task_index] or metadata.get("task_count") != TASK_COUNT:
        raise EvaluationError(f"canonical raw log is not the exact successful Stage-2 task {task_index}: {path}")


def _load_task_receipt(
    paths: LaunchPaths,
    *,
    task_index: int,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
) -> dict[str, Any] | None:
    receipt_path = _task_receipt_path(paths, task_index)
    if not receipt_path.exists() and not receipt_path.is_symlink():
        return None
    document = _read_json(receipt_path, label=f"task-{task_index} receipt")
    required = {
        "schema",
        "task_index",
        "launch_contract_sha256",
        "evaluation_receipt_sha256",
        "canonical_log",
        "attempt_log",
    }
    if set(document) != required or document.get("schema") != TASK_RECEIPT_SCHEMA:
        raise EvaluationError(f"task-{task_index} receipt has an unsupported schema")
    if (
        document.get("task_index") != task_index
        or document.get("launch_contract_sha256") != launch_contract_sha256
        or document.get("evaluation_receipt_sha256") != evaluation_receipt_sha256
    ):
        raise EvaluationError(f"task-{task_index} receipt does not bind this immutable launch and vLLM receipt")
    attempt = document.get("attempt_log")
    if not isinstance(attempt, Mapping):
        raise EvaluationError(f"task-{task_index} receipt has no attempt log identity")
    attempt_path = Path(str(attempt.get("path", ""))).resolve()
    _under_root(attempt_path, paths.attempts, label=f"task-{task_index} attempt log")
    if attempt != _identity(attempt_path, label=f"task-{task_index} attempt log"):
        raise EvaluationError(f"task-{task_index} attempt log changed after receipt publication")
    canonical = document.get("canonical_log")
    if not isinstance(canonical, Mapping):
        raise EvaluationError(f"task-{task_index} receipt has no canonical log identity")
    path = Path(str(canonical.get("path", ""))).resolve()
    _under_root(path, paths.raw, label=f"task-{task_index} canonical log")
    if canonical != _identity(path, label=f"task-{task_index} canonical log"):
        raise EvaluationError(f"task-{task_index} canonical log changed after receipt publication")
    _inspect_success(path, task_index=task_index)
    return document


def _copy_immutable_file(source: Path, destination: Path, *, label: str) -> None:
    """Publish a file with link-first atomic no-overwrite semantics."""

    _identity(source, label=label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or _sha256_file(destination) != _sha256_file(source):
            raise FileExistsError(f"refusing to overwrite differing {label}: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle, source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or _sha256_file(destination) != _sha256_file(source):
                raise FileExistsError(f"{label} appeared with different bytes: {destination}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _promote_attempt(
    *,
    attempt: Path,
    task_indices: Sequence[int],
    paths: LaunchPaths,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
) -> list[int]:
    """Capture only exact successful logs; leave every partial attempt intact."""

    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - evaluation environment only
        raise EvaluationError("Inspect AI is required to capture raw task attempts") from exc
    wanted = set(task_indices)
    selected: dict[int, Path] = {}
    for candidate in sorted(attempt.rglob("*.eval")):
        if candidate.is_symlink() or not candidate.is_file():
            raise EvaluationError(f"evaluation attempt contains a linked/non-file EvalLog: {candidate}")
        try:
            log = read_eval_log(str(candidate), header_only=True)
        except Exception:
            # An incomplete/corrupt log remains in this preserved attempt;
            # it is never promoted into canonical raw evidence.
            continue
        evaluation = getattr(log, "eval", None)
        metadata = getattr(evaluation, "metadata", {}) if evaluation is not None else {}
        metadata = metadata if isinstance(metadata, Mapping) else {}
        indices = metadata.get("task_indices")
        if getattr(log, "status", None) != "success" or not isinstance(indices, list) or len(indices) != 1:
            continue
        index = indices[0]
        if index not in wanted or metadata.get("task_count") != TASK_COUNT:
            raise EvaluationError(f"attempt produced an unexpected successful task identity: {candidate}")
        if index in selected:
            raise EvaluationError(f"attempt produced duplicate successful logs for task {index}: {selected[index]} and {candidate}")
        selected[index] = candidate
    promoted: list[int] = []
    for index, source in sorted(selected.items()):
        if _load_task_receipt(
            paths,
            task_index=index,
            launch_contract_sha256=launch_contract_sha256,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
        ) is not None:
            continue
        digest = _sha256_file(source)
        canonical = paths.raw / f"task-{index:03d}" / f"{digest}.eval"
        _copy_immutable_file(source, canonical, label=f"task-{index} canonical raw log")
        _inspect_success(canonical, task_index=index)
        receipt = {
            "schema": TASK_RECEIPT_SCHEMA,
            "task_index": index,
            "launch_contract_sha256": launch_contract_sha256,
            "evaluation_receipt_sha256": evaluation_receipt_sha256,
            "attempt_log": _identity(source, label=f"task-{index} attempt log"),
            "canonical_log": _identity(canonical, label=f"task-{index} canonical raw log"),
        }
        _write_immutable_json(_task_receipt_path(paths, index), receipt, label=f"task-{index} receipt")
        promoted.append(index)
    return promoted


def _promote_prior_attempts(
    *,
    phase_root: Path,
    attempt_label: str,
    task_indices: Sequence[int],
    paths: LaunchPaths,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
) -> list[int]:
    """Recover successful cells from every preserved prior attempt first.

    A scheduler timeout can interrupt a persistent server after it has written
    one or more complete logs.  Those logs are safe to promote on the next
    window; rerunning them would be unnecessary sampling, not a resume.
    """

    if not phase_root.exists():
        return []
    if phase_root.is_symlink() or not phase_root.is_dir():
        raise EvaluationError(f"phase attempt root must be a regular directory: {phase_root}")
    promoted: list[int] = []
    prefix = f"{_safe_component(attempt_label, label='attempt label')}-"
    for prior in sorted(
        path
        for path in phase_root.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.startswith(prefix)
    ):
        promoted.extend(
            _promote_attempt(
                attempt=prior,
                task_indices=task_indices,
                paths=paths,
                launch_contract_sha256=launch_contract_sha256,
                evaluation_receipt_sha256=evaluation_receipt_sha256,
            )
        )
    return promoted


def _task_command(
    *,
    launch: Mapping[str, Any],
    runtime: Mapping[str, Any],
    paths: LaunchPaths,
    attempt: Path,
    task_indices: Sequence[int],
    python: str,
) -> list[str]:
    if not task_indices or any(index < 1 or index > TASK_COUNT for index in task_indices):
        raise EvaluationError("task group has an invalid Stage-2 task index")
    task_args = {
        "manifest": str(launch["deployment_manifest"]["path"]),
        "unbiased_log": str(paths.raw),
        "prompt_style": "none",
        "include_bias_acknowledged": False,
    }
    return [
        python,
        str(PROJECT_ROOT / "scripts" / "run_evals.py"),
        "--task-factory",
        STAGE2_TASK_FACTORY,
        "--local-checkpoint",
        str(runtime["compatibility_adapter"]["path"]),
        "--base-model",
        str(MODEL_SNAPSHOT),
        "--task-args",
        json.dumps(task_args, sort_keys=True, separators=(",", ":")),
        "--model-args",
        json.dumps(VLLM_MODEL_ARGS, sort_keys=True, separators=(",", ":")),
        "--generation-config",
        json.dumps(GENERATION_CONFIG, sort_keys=True, separators=(",", ":")),
        "--log-dir",
        str(attempt),
        "--limit",
        "100",
        "--max-tasks",
        "1",
        "--isolate-tasks",
        "--persistent-vllm-server",
        "--yes",
        *[item for index in task_indices for item in ("--task-index", str(index))],
    ]


def _run_group(
    *,
    gpu: str,
    task_indices: Sequence[int],
    launch: Mapping[str, Any],
    runtime: Mapping[str, Any],
    paths: LaunchPaths,
    python: str,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
    phase: str,
) -> dict[str, Any]:
    # A Slurm-visible CUDA token may be a UUID or a MIG token containing
    # punctuation.  Never turn it into a path component; the stable digest is
    # enough to scope retries to the same assigned device identity.
    gpu_label = hashlib.sha256(gpu.encode("utf-8")).hexdigest()[:16]
    attempt_label = f"gpu-{gpu_label}-tasks-{'-'.join(str(index) for index in task_indices)}"
    missing = [
        index
        for index in task_indices
        if _load_task_receipt(
            paths,
            task_index=index,
            launch_contract_sha256=launch_contract_sha256,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
        ) is None
    ]
    if missing:
        _promote_prior_attempts(
            phase_root=paths.attempts / phase,
            attempt_label=attempt_label,
            # A prior own-group attempt can contain one already-promoted cell
            # and one that was left unreceipted by a timeout.  Scan the full
            # stable group; _promote_attempt skips the receipted cell.
            task_indices=task_indices,
            paths=paths,
            launch_contract_sha256=launch_contract_sha256,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
        )
        missing = [
            index
            for index in task_indices
            if _load_task_receipt(
                paths,
                task_index=index,
                launch_contract_sha256=launch_contract_sha256,
                evaluation_receipt_sha256=evaluation_receipt_sha256,
            ) is None
        ]
    if not missing:
        return {"gpu": gpu, "task_indices": list(task_indices), "status": "resumed", "promoted": []}
    attempt = _next_attempt(paths.attempts / phase, label=attempt_label)
    command = _task_command(
        launch=launch,
        runtime=runtime,
        paths=paths,
        attempt=attempt,
        task_indices=missing,
        python=python,
    )
    runtime_sampler = runtime.get("sampler")
    expected_sampler = _vllm_sampler_runtime(
        vllm_device_tokens=(
            runtime_sampler.get("vllm_device_tokens", ())
            if isinstance(runtime_sampler, Mapping)
            else ()
        )
    )
    if runtime.get("sampler") != expected_sampler:
        raise EvaluationError("persistent-vLLM group lacks the r005 pinned sampler runtime")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.update(expected_sampler["environment"])
    for name in ("VLLM_BASE_URL", "VLLM_API_KEY", "CTM_PERSISTENT_VLLM_SERVER_METADATA"):
        environment.pop(name, None)
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=environment, check=False)
    promoted = _promote_attempt(
        attempt=attempt,
        task_indices=missing,
        paths=paths,
        launch_contract_sha256=launch_contract_sha256,
        evaluation_receipt_sha256=evaluation_receipt_sha256,
    )
    if completed.returncode:
        raise EvaluationError(
            f"{phase} persistent-vLLM group on gpu {gpu} exited with {completed.returncode}; "
            f"successful tasks were retained={promoted}, attempt preserved={attempt}"
        )
    unresolved = [
        index
        for index in missing
        if _load_task_receipt(
            paths,
            task_index=index,
            launch_contract_sha256=launch_contract_sha256,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
        ) is None
    ]
    if unresolved:
        raise EvaluationError(f"{phase} group returned success but did not emit exact successful logs for tasks {unresolved}: {attempt}")
    return {"gpu": gpu, "task_indices": list(task_indices), "status": "completed", "promoted": promoted, "attempt": str(attempt)}


def _run_parallel_groups(
    groups: Sequence[tuple[str, Sequence[int]]],
    *,
    launch: Mapping[str, Any],
    runtime: Mapping[str, Any],
    paths: LaunchPaths,
    python: str,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
    phase: str,
) -> list[dict[str, Any]]:
    if not groups:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = [
            executor.submit(
                _run_group,
                gpu=gpu,
                task_indices=indices,
                launch=launch,
                runtime=runtime,
                paths=paths,
                python=python,
                launch_contract_sha256=launch_contract_sha256,
                evaluation_receipt_sha256=evaluation_receipt_sha256,
                phase=phase,
            )
            for gpu, indices in groups
        ]
        return [future.result() for future in futures]


def _require_clean_barrier(
    paths: LaunchPaths,
    *,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
) -> None:
    missing = [
        index
        for index in CLEAN_TASK_INDICES
        if _load_task_receipt(
            paths,
            task_index=index,
            launch_contract_sha256=launch_contract_sha256,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
        ) is None
    ]
    if missing:
        raise EvaluationError(f"biased tasks are blocked until every clean task is exact and successful; missing={missing}")


def _validate_canonical_raw_custody(
    paths: LaunchPaths,
    *,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
) -> None:
    """Reject stray or duplicated canonical logs before semantic preflight.

    The Stage-2 preflight intentionally scans a directory.  Limit that scan to
    exactly the 21 receipt-selected artifacts so a preserved retry cannot be
    mistaken for the current immutable vLLM condition.
    """

    expected: set[Path] = set()
    for index in range(1, TASK_COUNT + 1):
        receipt = _load_task_receipt(
            paths,
            task_index=index,
            launch_contract_sha256=launch_contract_sha256,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
        )
        if receipt is None:  # caller reports the friendlier incomplete-matrix error
            return
        canonical = receipt["canonical_log"]
        assert isinstance(canonical, Mapping)  # validated by _load_task_receipt
        expected.add(Path(str(canonical["path"])).resolve())
    found: set[Path] = set()
    for path in paths.raw.rglob("*.eval"):
        if path.is_symlink() or not path.is_file():
            raise EvaluationError(f"canonical raw tree contains a linked/non-file EvalLog: {path}")
        found.add(path.resolve())
    if found != expected:
        raise EvaluationError(
            "canonical raw tree contains unreceipted or missing EvalLogs; "
            f"unexpected={sorted(str(path) for path in found - expected)}, "
            f"missing={sorted(str(path) for path in expected - found)}"
        )


def _preflight_complete_matrix(
    *,
    evaluation_receipt: Mapping[str, Any],
    paths: LaunchPaths,
) -> dict[str, Any]:
    """Delegate semantic identity and label verification to the new contract."""

    try:
        from experiments.rmct_two_bias_eval import raw_preflight
    except ImportError as exc:  # pragma: no cover - deployment error
        raise EvaluationError("the r005 two-bias raw preflight is unavailable") from exc
    try:
        report = raw_preflight.preflight_raw_logs(
            paths.raw,
            evaluation_receipt["path"],
        )
        status = raw_preflight.write_preflight_report(paths.root / "stage2" / "preflight" / f"{CONDITION}.json", report)
    except Exception as exc:
        raise EvaluationError(f"two-bias raw preflight failed: {exc}") from exc
    if not isinstance(report, Mapping):
        raise EvaluationError("two-bias raw preflight returned no structured report")
    contract = report.get("contract")
    if not isinstance(contract, Mapping) or tuple(contract.get("seen_biases", ())) != SEEN_BIASES:
        raise EvaluationError("two-bias raw preflight did not bind suggested_answer as a seen training bias")
    if tuple(contract.get("held_out_biases", ())) != HELD_OUT_BIASES:
        raise EvaluationError("two-bias raw preflight has an unexpected held-out-bias classification")
    return {"report": report, "status": status}


def _completed_replay(
    *,
    paths: LaunchPaths,
    launch_contract_sha256: str,
    evaluation_receipt: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a prior terminal completion only when its immutable bindings match.

    Completion timestamps and worker-attempt summaries are deliberately
    observational.  Rebuilding them on a clean replay would change bytes and
    turn a completed evaluation into a false overwrite conflict.
    """

    if not paths.completion.exists() and not paths.completion.is_symlink():
        return None
    document = _read_json(paths.completion, label="r005 two-bias evaluation completion receipt")
    if set(document) != {"schema", "contract", "evaluation_receipt", "completed_at", "runtime", "clean", "biased", "preflight"}:
        raise EvaluationError("completion receipt has an unsupported schema")
    contract = document.get("contract")
    receipt = document.get("evaluation_receipt")
    preflight = document.get("preflight")
    if (
        document.get("schema") != SCHEMA
        or not isinstance(contract, Mapping)
        or contract != {"path": str(paths.contract), "sha256": launch_contract_sha256}
        or not isinstance(receipt, Mapping)
        or receipt.get("path") != evaluation_receipt.get("path")
        or receipt.get("sha256") != evaluation_receipt.get("sha256")
        or not isinstance(preflight, Mapping)
        or preflight.get("path") != str(paths.root / "stage2" / "preflight" / f"{CONDITION}.json")
    ):
        raise EvaluationError("completion receipt conflicts with the current immutable evaluation evidence")
    return document


def _parse_gpus(value: str) -> tuple[str, str, str, str]:
    """Parse the exact Slurm-provided CUDA tokens passed by the wrapper."""

    return _parse_gpu_tokens(value, label="--gpus")


def _required_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _SHA256_CHARS for character in value):
        raise EvaluationError(f"{label} must be a lowercase SHA-256 string")
    return value


def _parity_data(parity_data: str | Path, parity_manifest: str | Path) -> dict[str, Any]:
    """Bind runtime parity to the deployed frozen Stage-1 train/eval artifact.

    The canonical Stage-1 manifest is intentionally allowed to retain its
    original local absolute paths.  We only parse the attested ``train_eval``
    entry, compare its basename/hash/byte count with the separately staged
    deployment file, and never dereference that stale source path.
    """

    manifest_path = Path(parity_manifest).expanduser()
    manifest_identity = _identity(manifest_path, label="canonical Stage-1 parity manifest")
    if manifest_identity["sha256"] != STAGE1_PARITY_MANIFEST_SHA256:
        raise EvaluationError(
            "canonical Stage-1 parity manifest differs from the pinned no-CoT train/eval manifest"
        )
    document = _read_json(manifest_path, label="canonical Stage-1 parity manifest")
    if (
        document.get("schema_version") != STAGE1_PARITY_MANIFEST_SCHEMA_VERSION
        or document.get("kind") != STAGE1_PARITY_MANIFEST_KIND
    ):
        raise EvaluationError("canonical Stage-1 parity manifest has an unsupported schema")
    splits = document.get("splits")
    if not isinstance(splits, Mapping):
        raise EvaluationError("canonical Stage-1 parity manifest has no splits object")
    entry = splits.get("train_eval")
    if not isinstance(entry, Mapping):
        raise EvaluationError("canonical Stage-1 parity manifest has no train_eval entry")
    declared_path = entry.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        raise EvaluationError("canonical Stage-1 parity manifest train_eval.path is invalid")
    if Path(declared_path).name != STAGE1_PARITY_TRAIN_EVAL_FILENAME:
        raise EvaluationError(
            "canonical Stage-1 parity manifest train_eval.path does not name the frozen train-eval file"
        )
    expected_sha256 = _required_sha256(
        entry.get("content_sha256"),
        label="canonical Stage-1 parity manifest train_eval.content_sha256",
    )
    expected_bytes = entry.get("byte_count")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 1:
        raise EvaluationError("canonical Stage-1 parity manifest train_eval.byte_count is invalid")
    if entry.get("row_count") != STAGE1_PARITY_TRAIN_EVAL_ROWS:
        raise EvaluationError("canonical Stage-1 parity manifest train_eval.row_count is invalid")

    data_path = Path(parity_data).expanduser()
    if data_path.name != STAGE1_PARITY_TRAIN_EVAL_FILENAME:
        raise EvaluationError(
            "deployed Stage-1 parity data must be named "
            f"{STAGE1_PARITY_TRAIN_EVAL_FILENAME!r}"
        )
    identity = _identity(data_path, label="deployed Stage-1 parity train/eval data")
    if identity["sha256"] != expected_sha256:
        raise EvaluationError("deployed Stage-1 parity data SHA-256 differs from the canonical train_eval entry")
    if identity["size_bytes"] != expected_bytes:
        raise EvaluationError("deployed Stage-1 parity data byte count differs from the canonical train_eval entry")
    return {
        **identity,
        "canonical_manifest": manifest_identity,
        "manifest_train_eval": {
            "declared_path": declared_path,
            "filename": STAGE1_PARITY_TRAIN_EVAL_FILENAME,
            "content_sha256": expected_sha256,
            "byte_count": expected_bytes,
            "row_count": STAGE1_PARITY_TRAIN_EVAL_ROWS,
        },
    }


def build_launch_contract(
    *,
    repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
    parity_data: str | Path,
    parity_manifest: str | Path,
    output_root: str | Path,
    gpus: Sequence[object],
) -> dict[str, Any]:
    """Validate immutable inputs and return the exact non-submitting launch plan."""

    repository_path = Path(repository).expanduser().resolve()
    paths = _launch_paths(output_root)
    checkpoint = validate_final_checkpoint(repository_path)
    snapshot = _validate_model_snapshot()
    deployment = materialize_deployment_manifest(
        source_stage2_manifest,
        artifact_root=stage2_artifact_root,
        output=paths.deployment_manifest,
    )
    # ``written`` versus ``resumed`` is operational telemetry, not immutable
    # input provenance.  Keeping it in the launch receipt would make a safe
    # replay differ byte-for-byte from its first invocation.
    deployment.pop("status", None)
    substrate = validate_two_bias_substrate(paths.deployment_manifest)
    parity_identity = _parity_data(parity_data, parity_manifest)
    parity_tokens = _require_parity_device_tokens(gpus, label="launch-contract parity device tokens")
    return {
        "schema": SCHEMA,
        "condition": CONDITION,
        "repository": str(repository_path),
        "checkpoint": checkpoint,
        "model_snapshot": snapshot,
        "parity_data": parity_identity,
        "critical_sources": _critical_source_identities(),
        "source_stage2_manifest": _identity(Path(source_stage2_manifest).expanduser().resolve(), label="source Stage-2 manifest"),
        "stage2_artifact_root": str(Path(stage2_artifact_root).expanduser().resolve()),
        "deployment_manifest": {"path": str(paths.deployment_manifest), "provenance": deployment, "substrate": substrate},
        "runtime": {
            "profile": "vllm",
            "sampler": _vllm_sampler_runtime(vllm_device_tokens=parity_tokens),
            "hf_fallback_permitted": False,
            "requires_translated_parity_attested_adapter": True,
            "persistent_vllm_server": True,
            "model_args": VLLM_MODEL_ARGS,
            "generation_config": GENERATION_CONFIG,
        },
        "matrix": {
            "task_factory": STAGE2_TASK_FACTORY,
            "task_count": TASK_COUNT,
            "clean_task_indices": list(CLEAN_TASK_INDICES),
            "biased_task_indices": list(BIASED_TASK_INDICES),
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "clean_before_biased": True,
        },
        "outputs": {
            "root": str(paths.root),
            "raw": str(paths.raw),
            "attempts": str(paths.attempts),
            "receipts": str(paths.receipts),
            "runtime": str(paths.runtime),
            "completion": str(paths.completion),
        },
        "policy": {
            "self_submits": False,
            "chains_successors": False,
            "overwrite_differing_logs": False,
            "partial_attempts_preserved": True,
        },
    }


def prepare_launch_contract(**kwargs: Any) -> tuple[dict[str, Any], LaunchPaths, str]:
    """Create/replay the immutable plan before any conversion or GPU work."""

    paths = _launch_paths(kwargs["output_root"])
    if paths.root.exists() and not paths.contract.exists() and any(paths.root.iterdir()):
        raise FileExistsError(f"refusing to seed an r005 evaluation contract in a non-empty output root: {paths.root}")
    contract = build_launch_contract(**kwargs)
    _write_immutable_json(paths.contract, contract, label="r005 two-bias launch contract")
    return contract, paths, _sha256_file(paths.contract)


def dry_run_summary(
    *,
    repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
    parity_data: str | Path,
    parity_manifest: str | Path,
    output_root: str | Path,
    gpus: Sequence[str],
) -> dict[str, Any]:
    contract, paths, contract_sha256 = prepare_launch_contract(
        repository=repository,
        source_stage2_manifest=source_stage2_manifest,
        stage2_artifact_root=stage2_artifact_root,
        parity_data=parity_data,
        parity_manifest=parity_manifest,
        output_root=output_root,
        gpus=gpus,
    )
    return {
        "schema": SCHEMA,
        "contract": {"path": str(paths.contract), "sha256": contract_sha256},
        "runtime": contract["runtime"],
        "clean_groups": [[gpus[index], task] for index, task in enumerate(CLEAN_TASK_INDICES)],
        "biased_groups": [
            [gpus[0], [4, 8, 12, 16, 20]],
            [gpus[1], [5, 9, 13, 17, 21]],
            [gpus[2], [6, 10, 14, 18]],
            [gpus[3], [7, 11, 15, 19]],
        ],
        "actions": ["validate-terminal-custody", "materialize-deployment-manifest", "translate-and-attest-vllm", "clean", "biased", "two-bias-preflight"],
    }


def execute(
    *,
    repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
    parity_data: str | Path,
    parity_manifest: str | Path,
    output_root: str | Path,
    gpus: Sequence[str],
    python: str,
) -> dict[str, Any]:
    """Run missing cells only, preserving retry evidence and requiring parity."""

    parity_tokens = _require_parity_device_tokens(gpus, label="execution GPU tokens")
    contract, paths, contract_sha256 = prepare_launch_contract(
        repository=repository,
        source_stage2_manifest=source_stage2_manifest,
        stage2_artifact_root=stage2_artifact_root,
        parity_data=parity_data,
        parity_manifest=parity_manifest,
        output_root=output_root,
        gpus=parity_tokens,
    )
    runtime = ensure_attested_vllm_runtime(
        launch=contract,
        paths=paths,
        python=python,
    )
    evaluation_receipt = build_vllm_evaluation_receipt(
        launch=contract,
        runtime=runtime,
        paths=paths,
    )
    evaluation_receipt_sha256 = str(evaluation_receipt["sha256"])

    clean_groups = [(parity_tokens[index], (task_index,)) for index, task_index in enumerate(CLEAN_TASK_INDICES)]
    clean = _run_parallel_groups(
        clean_groups,
        launch=contract,
        runtime=runtime,
        paths=paths,
        python=python,
        launch_contract_sha256=contract_sha256,
        evaluation_receipt_sha256=evaluation_receipt_sha256,
        phase="clean",
    )
    _require_clean_barrier(
        paths,
        launch_contract_sha256=contract_sha256,
        evaluation_receipt_sha256=evaluation_receipt_sha256,
    )

    biased_groups = (
        (parity_tokens[0], (4, 8, 12, 16, 20)),
        (parity_tokens[1], (5, 9, 13, 17, 21)),
        (parity_tokens[2], (6, 10, 14, 18)),
        (parity_tokens[3], (7, 11, 15, 19)),
    )
    biased = _run_parallel_groups(
        biased_groups,
        launch=contract,
        runtime=runtime,
        paths=paths,
        python=python,
        launch_contract_sha256=contract_sha256,
        evaluation_receipt_sha256=evaluation_receipt_sha256,
        phase="biased",
    )
    missing = [
        index
        for index in range(1, TASK_COUNT + 1)
        if _load_task_receipt(
            paths,
            task_index=index,
            launch_contract_sha256=contract_sha256,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
        ) is None
    ]
    if missing:
        raise EvaluationError(f"raw Stage-2 matrix remains incomplete after successful workers: {missing}")
    _validate_canonical_raw_custody(
        paths,
        launch_contract_sha256=contract_sha256,
        evaluation_receipt_sha256=evaluation_receipt_sha256,
    )
    preflight = _preflight_complete_matrix(evaluation_receipt=evaluation_receipt, paths=paths)
    prior_completion = _completed_replay(
        paths=paths,
        launch_contract_sha256=contract_sha256,
        evaluation_receipt=evaluation_receipt,
    )
    if prior_completion is not None:
        return prior_completion
    completion = {
        "schema": SCHEMA,
        "contract": {"path": str(paths.contract), "sha256": contract_sha256},
        "evaluation_receipt": {
            "path": str(evaluation_receipt["path"]),
            "sha256": evaluation_receipt_sha256,
            "status": evaluation_receipt["status"],
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "clean": clean,
        "biased": biased,
        "preflight": {"status": preflight["status"], "path": str(paths.root / "stage2" / "preflight" / f"{CONDITION}.json")},
    }
    _write_immutable_json(paths.completion, completion, label="r005 two-bias evaluation completion receipt")
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path, help="durable Isambard checkout containing the sealed r4 run")
    parser.add_argument("--source-stage2-manifest", required=True, type=Path, help="canonical Stage-2 source manifest before deployment rebasing")
    parser.add_argument("--stage2-artifact-root", required=True, type=Path, help="deployment copy of the immutable Stage-2 artifact tree")
    parser.add_argument(
        "--parity-data",
        required=True,
        type=Path,
        help="deployed frozen Stage-1 train/eval JSONL used only for HF/vLLM parity",
    )
    parser.add_argument(
        "--parity-manifest",
        required=True,
        type=Path,
        help="canonical pinned Stage-1 manifest that attests the deployed parity JSONL",
    )
    parser.add_argument("--output-root", required=True, type=Path, help="fresh/replayable condition-local evidence root")
    parser.add_argument("--gpus", default="0,1,2,3", help="four visible logical GPU indices")
    parser.add_argument("--python", default=sys.executable, help="evaluation Python interpreter")
    parser.add_argument("--dry-run", action="store_true", help="validate and write/replay the plan without model work")
    parser.add_argument("--yes", action="store_true", help="run parity and missing raw evaluation work")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.yes:
        parser.error("--dry-run and --yes are mutually exclusive")
    if not args.dry_run and not args.yes:
        parser.error("--yes is required to start parity or evaluation work; use --dry-run to inspect")
    try:
        gpus = _parse_gpus(args.gpus)
        values = {
            "repository": args.repository,
            "source_stage2_manifest": args.source_stage2_manifest,
            "stage2_artifact_root": args.stage2_artifact_root,
            "parity_data": args.parity_data,
            "parity_manifest": args.parity_manifest,
            "output_root": args.output_root,
        }
        if args.dry_run:
            result = dry_run_summary(**values, gpus=gpus)
        else:
            result = execute(**values, gpus=gpus, python=args.python)
    except (EvaluationError, FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
