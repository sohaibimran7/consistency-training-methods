#!/usr/bin/env python3
"""Run the ELEPHANT AITA-NTA-FLIP cross-task evaluation on 16 GPUs.

This is intentionally a fresh, self-contained evaluation campaign.  It does
not write into the Stage-2 two-bias trees and it never submits, cancels, or
chains Slurm jobs.  The companion sbatch file owns one allocation only:

* base Qwen3.5 snapshot;
* the raw, sealed step-16 PEFT checkpoint;
* the raw, sealed step-64 PEFT checkpoint; and
* the raw, sealed step-176 PEFT checkpoint.

Each condition is split by *pair*, not by individual AITA perspective, into
four deterministic shards.  The 16 resulting cells each use one GPU and the
native Hugging Face/PEFT backend while Inspect evaluates a single shard.  A
task receipt is written only after a successful, identity-bound EvalLog is
copied into the canonical raw tree.  Failed attempts remain where they were
written.

Every trained condition is deliberately fail-closed on *training* custody,
rather than on a prior behavioral-evaluation campaign.  The launcher replays
the complete resumable checkpoint bundle (adapter configuration and weights,
optimizer, manifests, replicated-training manifest and RNG state), plus the
applicable sealed receipt/decision evidence.  Thus a weights-only or
partly-completed checkpoint can never accidentally become an AITA source.

The pinned base snapshot uses the controlled offline Hugging Face cache.  Its
large weight files are attested by their direct SHA-256-named blob target and
size rather than re-hashed by all sixteen workers; that deliberately trusts
the cache's immutable content-addressed-blob invariant.  Small configuration,
tokenizer, template, and index files are fully hashed on every replay.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elephant_aita_ntaflip.no_cap_hf import (
    QWEN_THINKING_POLICY as BENCHMARK_QWEN_THINKING_POLICY,
    RUNTIME_ENV as NO_TOKEN_CAP_RUNTIME_ENV,
    RUNTIME_ENV_VALUE as NO_TOKEN_CAP_RUNTIME_ENV_VALUE,
    RUNTIME_POLICY as NO_TOKEN_CAP_RUNTIME_POLICY,
)
from experiments.elephant_aita_ntaflip.prepare import (
    GENERATION_CONFIG as BENCHMARK_GENERATION_CONFIG,
    MANIFEST_SCHEMA as BENCHMARK_MANIFEST_SCHEMA,
    NO_TOKEN_CAP_POLICY,
    RUNTIME_GENERATION_CONFIG as BENCHMARK_RUNTIME_GENERATION_CONFIG,
    TOKEN_CAP_FIELD_NAMES,
    assert_no_token_cap_mapping,
)
from experiments.elephant_aita_ntaflip.preflight import (
    PARSER_SCHEMA as FINAL_ANSWER_PARSER_SCHEMA,
)


LAUNCH_SCHEMA = "rmct-elephant-aita-nta-flip-16gpu-launch-v6-r006"
EVALUATION_SCHEMA = "rmct-elephant-aita-nta-flip-16gpu-evaluation-v6-r006"
TASK_RECEIPT_SCHEMA = "rmct-elephant-aita-nta-flip-16gpu-task-receipt-v6-r006"
SMOKE_RECEIPT_SCHEMA = "rmct-elephant-aita-nta-flip-16gpu-direct-verdict-smoke-v2-r006"
COMPLETION_SCHEMA = "rmct-elephant-aita-nta-flip-16gpu-completion-v7-r006"
GPU_TOPOLOGY_RECORD_SCHEMA = "rmct-elephant-aita-nta-flip-16gpu-gpu-topology-record-v1-r006"
GPU_TOPOLOGY_RECEIPT_SCHEMA = "rmct-elephant-aita-nta-flip-16gpu-gpu-topology-receipt-v1-r006"
BENCHMARK = "elephant-aita-nta-flip"
CAMPAIGN_NAME = "elephant-aita-nta-flip-qwen35-rmct-16gpu-v3-r006"
TASK_FACTORY = "experiments.elephant_aita_ntaflip.tasks:aita_nta_flip_shard"

# On this Isambard deployment, ``--gpus-per-task=1`` and ``single:1`` alone
# did not reduce a singleton step's CUDA-visible list.  These exact step-level
# requests and the documented per-task bind were proven in a live 16-rank
# allocation: every rank reported one CUDA device and all sixteen physical
# UUIDs were distinct (four per node in rank blocks).
GPU_BINDING = "verbose,per_task:1"
GPU_DISTRIBUTION = "block:block"
SMOKE_STEP_GPU_COUNT = 1
FULL_STEP_GPU_COUNT = 16
GPU_TOPOLOGY_TASK_COUNT = 16
GPU_TOPOLOGY_PROBE_DIRNAME = "gpu-topology-probe"

# The paper's AITA-NTA-FLIP split contains 1,591 original/flipped pairs.
# Every condition generates one constrained YTA/NTA response for both sides
# of every pair, hence 3,182 generations per condition.
EXPECTED_PAIRS = 1591
PERSPECTIVES_PER_PAIR = 2
GENERATIONS_PER_CONDITION = EXPECTED_PAIRS * PERSPECTIVES_PER_PAIR
SHARD_COUNT = 4
AITA_MANIFEST_SCHEMA = BENCHMARK_MANIFEST_SCHEMA
OFFICIAL_MANIFEST_FILENAME = "aita-nta-flip.manifest.json"
# This limits only the number of source rows exercised by the pre-launch
# direct-verdict check.  It is not a generation/output/completion limit: each
# selected row still samples until EOS through the frozen r005 runtime hook.
DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT = 2

# Preserve the paper's public decode protocol.  The benchmark task supplies
# the constrained "Output only YTA or NTA" prompt and *no* system prompt.
# ``top_k`` is explicit because the official HF implementation's effective
# default is 50; no provider is allowed to inject a hidden sampling default.
GENERATION_CONFIG = {
    **BENCHMARK_GENERATION_CONFIG,
}
PROMPT_SUFFIX = "\nOutput only YTA or NTA."
SYSTEM_PROMPT: None = None

MODEL_SNAPSHOT = Path(
    "/lus/lfs1aip2/scratch/a5v/sohaib.a5v/ctm/huggingface/hub/"
    "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
INSPECT_VERSION = "0.3.258"
TORCH_VERSION = "2.11.0+cu129"
TRANSFORMERS_VERSION = "5.5.4"
PEFT_VERSION = "0.20.0"
SAFETENSORS_VERSION = "0.8.0"
EVALUATOR_PACKAGE_VERSIONS = {
    "inspect-ai": INSPECT_VERSION,
    "torch": TORCH_VERSION,
    "transformers": TRANSFORMERS_VERSION,
    "peft": PEFT_VERSION,
    "safetensors": SAFETENSORS_VERSION,
}
HF_MODEL_ARGS = {
    "device": "cuda:0",
    "dtype": "bfloat16",
    # Inspect's HuggingFace provider otherwise relies on its mutable default.
    # The paper decode is stochastic, so make that branch explicit for every
    # condition (including the raw-PEFT local-checkpoint path).
    "do_sample": True,
    # Inspect 0.3.258 passes this model argument directly into Qwen's chat
    # template as ``enable_thinking``.  It must be explicit on both base and
    # local-PEFT native-HF paths, otherwise the template emits a reasoning
    # block before the requested verdict.
    "enable_thinking": False,
}
HF_LOCAL_MODEL_ARGS = {"provider": "hf", **HF_MODEL_ARGS}
CONCURRENCY_CONFIG = {"max_connections": 4}
RUNTIME_GENERATION_CONFIG = {**GENERATION_CONFIG, **CONCURRENCY_CONFIG}
if RUNTIME_GENERATION_CONFIG != BENCHMARK_RUNTIME_GENERATION_CONFIG:
    raise RuntimeError("AITA r006 launcher generation configuration differs from benchmark-owned contract")
assert_no_token_cap_mapping(GENERATION_CONFIG, label="AITA r006 launcher sampling_config")
assert_no_token_cap_mapping(RUNTIME_GENERATION_CONFIG, label="AITA r006 launcher generation_config")
QWEN_THINKING_POLICY = dict(BENCHMARK_QWEN_THINKING_POLICY)

# These are the fixed-model files consumed by AutoConfig/AutoTokenizer.  They
# are mandatory; do not silently let a changed snapshot fall back to a
# Hub/default asset.  Sampling is supplied independently by the frozen
# ``RUNTIME_GENERATION_CONFIG``, so a model-side generation config is only
# attested when its pinned snapshot actually carries one.
SNAPSHOT_REQUIRED_CONFIGURATION_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
# Some revisions express tokenizer special tokens or templates in one or more
# of these separate assets.  When present, they are execution-critical too.
SNAPSHOT_OPTIONAL_CONFIGURATION_FILES = (
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "processor_config.json",
    "preprocessor_config.json",
    "image_processor_config.json",
    "video_processor_config.json",
    "video_preprocessor_config.json",
    "audio_processor_config.json",
    "audio_preprocessor_config.json",
    "feature_extractor_config.json",
)
SNAPSHOT_WEIGHT_IDENTITY_POLICY = {
    "mode": "hf_content_address_and_size",
    "trust_boundary": "controlled_immutable_hf_blob_store",
    "full_shard_rehash_on_worker_replay": False,
}

R005_CONDITION = "rmct-convergence-r4-s011-two-bias-v1-r005"

CRITICAL_SOURCES = (
    "infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu.py",
    "infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu.sbatch",
    "infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu_worker.sh",
    "scripts/run_evals.py",
    "ctm/evals/runner.py",
    "ctm/evals/local_model.py",
    "ctm/training/resume_state.py",
    "experiments/elephant_aita_ntaflip/__init__.py",
    "experiments/elephant_aita_ntaflip/prepare.py",
    "experiments/elephant_aita_ntaflip/tasks.py",
    "experiments/elephant_aita_ntaflip/no_cap_hf.py",
    "experiments/elephant_aita_ntaflip/preflight.py",
    "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.py",
    "infra/isambard/run_qwen35_rmct_convergence_r4_two_bias_evals.py",
    "infra/isambard/verify_rmct_convergence_r4_recovery_production_ready.py",
    "experiments/rmct_convergence/controller.py",
)


class EvaluationError(ValueError):
    """A requested cross-task run lacks immutable, replayable custody."""


@dataclass(frozen=True)
class Condition:
    name: str
    label: str
    optimizer_step: int | None
    source_kind: str


CONDITIONS: tuple[Condition, ...] = (
    Condition("base", "base", None, "pinned-snapshot"),
    Condition("step016", "step-016", 16, "raw-training-checkpoint"),
    Condition("step064", "step-064", 64, "raw-training-checkpoint"),
    Condition("step176", "step-176", 176, "raw-training-checkpoint"),
)
_CONDITION_BY_NAME = {condition.name: condition for condition in CONDITIONS}


@dataclass(frozen=True)
class CampaignPaths:
    root: Path
    input: Path
    manifest: Path
    contract: Path
    conditions: Path
    completion: Path


@dataclass(frozen=True)
class ConditionPaths:
    root: Path
    evaluation_receipt: Path
    raw: Path
    attempts: Path
    receipts: Path
    preflight: Path


def _condition(condition: str) -> Condition:
    try:
        return _CONDITION_BY_NAME[condition]
    except KeyError as exc:
        raise EvaluationError(f"unknown AITA condition {condition!r}") from exc


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise EvaluationError(f"{label} must be a regular file: {candidate}")
    size = candidate.stat().st_size
    if size < 1:
        raise EvaluationError(f"{label} must not be empty: {candidate}")
    return {"path": str(candidate.resolve()), "sha256": _sha256_file(candidate), "size_bytes": size}


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise EvaluationError(f"{label} must be a regular file: {candidate}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid {label}: {candidate}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must contain a JSON object: {candidate}")
    return value


def _under_root(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise EvaluationError(f"{label} escapes expected root: {resolved}") from exc
    return resolved


def _resolve_unlinked(value: str | Path, *, label: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise EvaluationError(f"{label} must not be a symlink: {candidate}")
    return candidate.resolve()


def _write_immutable_json(path: Path, value: Mapping[str, Any], *, label: str) -> str:
    """Write a receipt once, or accept only an exact byte-for-byte replay."""

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


def _assert_no_token_cap_command(command: Sequence[str], *, label: str) -> None:
    """Reject output-length flags or JSON payloads in an executable receipt."""

    for token in command:
        if not isinstance(token, str):
            raise EvaluationError(f"{label} must contain only strings")
        option = token.split("=", 1)[0].lstrip("-").lower().replace("-", "_")
        if option in TOKEN_CAP_FIELD_NAMES:
            raise EvaluationError(f"{label} must not contain output-token cap flag {token!r}")
        try:
            decoded = json.loads(token)
        except json.JSONDecodeError:
            continue
        try:
            assert_no_token_cap_mapping(decoded, label=label)
        except ValueError as exc:
            raise EvaluationError(str(exc)) from exc


def _copy_immutable_file(source: Path, destination: Path, *, label: str) -> None:
    """Atomically publish a raw log, never replacing a differing result."""

    _identity(source, label=f"source {label}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or _sha256_file(destination) != _sha256_file(source):
            raise FileExistsError(f"refusing to overwrite differing {label}: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, output)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or _sha256_file(destination) != _sha256_file(source):
                raise FileExistsError(f"{label} appeared with different bytes: {destination}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _campaign_paths(campaign_root: str | Path) -> CampaignPaths:
    requested = Path(campaign_root).expanduser()
    if requested.is_symlink() or (requested.exists() and not requested.is_dir()):
        raise EvaluationError(f"campaign root must be a regular directory: {requested}")
    root = requested.resolve()
    if root.name != CAMPAIGN_NAME:
        raise EvaluationError(f"campaign root must be named exactly {CAMPAIGN_NAME!r}; got {root.name!r}")
    return CampaignPaths(
        root=root,
        input=root / "input",
        manifest=root / "input" / OFFICIAL_MANIFEST_FILENAME,
        contract=root / "launch-contract.json",
        conditions=root / "conditions",
        completion=root / "completion.json",
    )


def _condition_paths(paths: CampaignPaths, condition: Condition) -> ConditionPaths:
    root = paths.conditions / condition.name
    return ConditionPaths(
        root=root,
        evaluation_receipt=root / "evaluation-receipt.json",
        raw=root / "raw",
        attempts=root / "attempts",
        receipts=root / "receipts",
        preflight=root / "preflight.json",
    )


def _critical_source_identities() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in CRITICAL_SOURCES:
        candidate = _under_root(PROJECT_ROOT / relative, PROJECT_ROOT, label=f"critical source {relative}")
        records[relative] = _identity(candidate, label=f"critical source {relative}")
    return records


def _snapshot_blobs_root(snapshot: Path) -> Path:
    """Return the one regular HF blob store associated with ``snapshot``."""

    blobs = snapshot.parent.parent / "blobs"
    if blobs.is_symlink() or not blobs.is_dir():
        raise EvaluationError("pinned Qwen3.5 snapshot has no regular Hugging Face blob tree")
    return blobs.resolve()


def _snapshot_resolved_file(snapshot: Path, logical: Path, *, label: str) -> tuple[Path, bool]:
    """Resolve a direct snapshot asset without permitting cache-root escapes.

    Hugging Face snapshots normally contain logical symlinks directly to one
    content-addressed file under ``../blobs``.  We permit exactly that shape;
    a logical symlink to another snapshot asset, a nested link, or an external
    path cannot silently alter the model selected by a sealed launch contract.
    """

    if logical.parent != snapshot:
        raise EvaluationError(f"pinned Qwen3.5 {label} is not a top-level snapshot asset")
    if not logical.exists():
        raise EvaluationError(f"pinned Qwen3.5 snapshot lacks {label}: {logical}")
    if logical.is_symlink():
        blobs = _snapshot_blobs_root(snapshot)
        try:
            resolved = logical.resolve(strict=True)
        except OSError as exc:
            raise EvaluationError(f"pinned Qwen3.5 {label} has a broken blob link: {logical}") from exc
        if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size < 1:
            raise EvaluationError(f"pinned Qwen3.5 {label} resolves to an invalid blob")
        _under_root(resolved, blobs, label=f"pinned Qwen3.5 {label} blob")
        if resolved.parent != blobs:
            raise EvaluationError(f"pinned Qwen3.5 {label} does not resolve directly to a blob")
        return resolved, True
    try:
        resolved = logical.resolve(strict=True)
    except OSError as exc:
        raise EvaluationError(f"pinned Qwen3.5 {label} is unreadable: {logical}") from exc
    if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size < 1:
        raise EvaluationError(f"pinned Qwen3.5 {label} is not a non-empty regular file")
    _under_root(resolved, snapshot, label=f"pinned Qwen3.5 {label}")
    return resolved, False


def _snapshot_logical_identity(snapshot: Path, logical: Path, *, label: str) -> dict[str, Any]:
    """Hash a small snapshot asset and retain its exact resolved target."""

    resolved, from_blob = _snapshot_resolved_file(snapshot, logical, label=label)
    identity: dict[str, Any] = {
        "logical_path": str(logical),
        "resolved_path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if from_blob:
        # Retain the blob spelling as well as its full digest.  This makes an
        # attempted link swap visible even before comparing content bytes.
        identity["blob_content_address"] = resolved.name
    return identity


def _snapshot_weight_record(snapshot: Path, logical: Path, *, label: str) -> dict[str, Any]:
    """Bind a model shard by direct HF blob identity, without rehash storms.

    Re-hashing all 9B-model shards independently from sixteen ranks would
    create a multi-terabyte shared-filesystem storm.  Standard Hugging Face
    snapshot links target a SHA-256-named immutable blob.  For the controlled
    cache named in ``SNAPSHOT_WEIGHT_IDENTITY_POLICY``, the direct blob content
    address plus size is the replayable shard identity; a non-symlink snapshot
    shard is hashed in full instead.  A same-path mutation beneath an existing
    content address violates that explicit cache trust boundary.
    """

    resolved, from_blob = _snapshot_resolved_file(snapshot, logical, label=label)
    record: dict[str, Any] = {
        "logical_path": str(logical),
        "resolved_path": str(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if from_blob:
        if not _valid_sha256(resolved.name):
            raise EvaluationError(f"pinned Qwen3.5 {label} blob is not SHA-256 content-addressed: {resolved.name!r}")
        record["content_address"] = resolved.name
    else:
        record["sha256"] = _sha256_file(resolved)
    return record


def _safe_snapshot_weight_name(value: object) -> str:
    """Accept only a direct top-level safetensors filename from an index."""

    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or Path(value).name != value
        or value.startswith(".")
        or not value.endswith(".safetensors")
    ):
        raise EvaluationError(f"pinned Qwen3.5 safetensors index contains an unsafe weight filename: {value!r}")
    return value


def _validate_snapshot_no_token_caps(snapshot: Path, files: Mapping[str, Mapping[str, Any]]) -> None:
    """Reject any generation-length setting carried by the pinned snapshot.

    The r006 launcher uses the frozen r005 sampler, which never calls
    ``transformers.GenerationMixin.generate``.
    Nevertheless, accepting a snapshot generation config that carries an
    explicit token bound would make the run ambiguous and could be accidentally
    reintroduced by a later runtime change, so reject it during every custody
    replay.
    """

    for name in ("config.json", "generation_config.json"):
        record = files.get(name)
        if not isinstance(record, Mapping):
            continue
        resolved = record.get("resolved_path")
        if not isinstance(resolved, str):
            raise EvaluationError(f"pinned Qwen3.5 {name} has no resolved identity")
        try:
            document = json.loads(Path(resolved).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"pinned Qwen3.5 {name} is not valid JSON") from exc
        if not isinstance(document, Mapping):
            raise EvaluationError(f"pinned Qwen3.5 {name} must contain a JSON object")
        try:
            assert_no_token_cap_mapping(document, label=f"pinned Qwen3.5 {name}")
        except ValueError as exc:
            raise EvaluationError(str(exc)) from exc


def _snapshot_identity() -> dict[str, Any]:
    """Identity-bind every model, tokenizer, and generation-critical asset.

    The campaign contract is replayed by each worker before a GPU is used.
    Binding only ``config.json`` would let changed weights or tokenizer assets
    pass that replay, so this records the safetensors index, every referenced
    (and any additional top-level) model shard, plus all Qwen tokenizer and
    generation files that can affect native-HF execution.
    """

    snapshot = MODEL_SNAPSHOT
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.parent.name != "snapshots":
        raise EvaluationError(f"pinned Qwen3.5 snapshot is absent or linked: {snapshot}")

    files: dict[str, dict[str, Any]] = {}
    for name in SNAPSHOT_REQUIRED_CONFIGURATION_FILES:
        files[name] = _snapshot_logical_identity(snapshot, snapshot / name, label=name)
    # Hugging Face revisions use both ``chat_template.json`` and
    # ``chat_template.jinja``.  Capture either spelling and any additional
    # top-level tokenizer-template or processor asset rather than accidentally
    # changing prompt serialization or processor initialization on a replay.
    auxiliary_assets = {
        candidate.name
        for candidate in snapshot.iterdir()
        if (
            any(
                token in candidate.name.lower()
                for token in ("template", "processor", "preprocessor", "feature_extractor")
            )
            and (candidate.is_file() or candidate.is_symlink())
        )
    }
    for name in sorted(set(SNAPSHOT_OPTIONAL_CONFIGURATION_FILES) | auxiliary_assets):
        logical = snapshot / name
        if logical.exists() or logical.is_symlink():
            files[name] = _snapshot_logical_identity(snapshot, logical, label=name)
    _validate_snapshot_no_token_caps(snapshot, files)

    indices = sorted(snapshot.glob("*.safetensors.index.json"))
    if len(indices) > 1:
        raise EvaluationError("pinned Qwen3.5 snapshot has ambiguous safetensors indices")
    indexed_weights: set[str] = set()
    safetensors_index: dict[str, Any] | None = None
    if indices:
        index = indices[0]
        relative = index.relative_to(snapshot).as_posix()
        safetensors_index = _snapshot_logical_identity(snapshot, index, label=relative)
        try:
            document = json.loads(Path(safetensors_index["resolved_path"]).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"pinned Qwen3.5 has an invalid safetensors index: {index}") from exc
        # Detect a mutable index changing between hashing and parsing before
        # its names are used to bind model shards.
        if _snapshot_logical_identity(snapshot, index, label=relative) != safetensors_index:
            raise EvaluationError("pinned Qwen3.5 safetensors index changed during custody replay")
        weight_map = document.get("weight_map") if isinstance(document, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise EvaluationError(f"pinned Qwen3.5 safetensors index has no weight map: {index}")
        for tensor_name, filename in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise EvaluationError("pinned Qwen3.5 safetensors index has an invalid tensor name")
            indexed_weights.add(_safe_snapshot_weight_name(filename))

    # Retain all model shard logical files as well as the explicitly indexed
    # ones.  The union is fail-closed against a snapshot gaining an alternate
    # top-level shard while preserving compatibility with unindexed snapshots.
    top_level_weights = {path.name for path in snapshot.glob("*.safetensors")}
    weight_names = indexed_weights | top_level_weights
    if not weight_names:
        raise EvaluationError("pinned Qwen3.5 snapshot has no safetensors model shards")
    weight_files = {
        filename: _snapshot_weight_record(snapshot, snapshot / filename, label=filename)
        for filename in sorted(weight_names)
    }
    return {
        "path": str(snapshot),
        "configuration_files": files,
        "safetensors_index": safetensors_index,
        "indexed_weight_files": sorted(indexed_weights),
        "weight_files": weight_files,
        "weight_file_count": len(weight_files),
        "weight_identity_policy": dict(SNAPSHOT_WEIGHT_IDENTITY_POLICY),
    }


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _installed_version(module: Any, *, distribution: str) -> str | None:
    """Read a package version without trusting a mutable environment string."""

    declared = getattr(module, "__version__", None)
    if isinstance(declared, str) and declared:
        return declared
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _single_slurm_visible_gpu() -> str:
    """Require the one-device task view established by r006's GPU binding."""

    tokens = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if (
        len(tokens) != 1
        or tokens[0] in {"-1", "NoDevFiles"}
        or any(any(character.isspace() for character in token) for token in tokens)
    ):
        raise EvaluationError(
            "each AITA r006 task requires exactly one Slurm-visible GPU; "
            f"got CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    return tokens[0]


def _validate_evaluator_environment(*, require_one_gpu: bool) -> dict[str, Any]:
    """Pin the exact native-HF evaluator and one-GPU worker boundary."""

    try:
        import inspect_ai
        import peft
        import safetensors
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - configured Isambard runtime
        raise EvaluationError("AITA native-HF evaluation environment is incomplete") from exc
    installed = {
        "inspect-ai": _installed_version(inspect_ai, distribution="inspect-ai"),
        "torch": _installed_version(torch, distribution="torch"),
        "transformers": _installed_version(transformers, distribution="transformers"),
        "peft": _installed_version(peft, distribution="peft"),
        "safetensors": _installed_version(safetensors, distribution="safetensors"),
    }
    if installed != EVALUATOR_PACKAGE_VERSIONS:
        raise EvaluationError(
            "AITA evaluation package versions differ from the frozen native-HF runtime: "
            f"got={installed!r}, expected={EVALUATOR_PACKAGE_VERSIONS!r}"
        )
    # The direct base-HF route and local PEFT route must share an explicit
    # attention policy.  Do not accept an inherited nonzero setting or rely on
    # either backend's ambient default.
    if os.environ.get("CTM_DISABLE_CUDNN_SDP") != "0":
        raise EvaluationError("AITA evaluation requires CTM_DISABLE_CUDNN_SDP to be exactly '0'")
    if require_one_gpu:
        if os.environ.get(NO_TOKEN_CAP_RUNTIME_ENV) != NO_TOKEN_CAP_RUNTIME_ENV_VALUE:
            raise EvaluationError(
                "AITA r006 worker requires the frozen r005 EOS-only sampler marker "
                f"{NO_TOKEN_CAP_RUNTIME_ENV}={NO_TOKEN_CAP_RUNTIME_ENV_VALUE!r}"
            )
        _single_slurm_visible_gpu()
    return {
        "backend": "native-hf-peft",
        "inspect_version": installed["inspect-ai"],
        "torch_version": installed["torch"],
        "transformers_version": installed["transformers"],
        "peft_version": installed["peft"],
        "safetensors_version": installed["safetensors"],
        "attention_policy": {"ctm_disable_cudnn_sdp": "0"},
        "no_token_cap_runtime_policy": dict(NO_TOKEN_CAP_RUNTIME_POLICY),
    }


def _evaluator_runtime() -> dict[str, Any]:
    """The immutable native-HF evaluator identity stored in every r006 receipt."""

    return _validate_evaluator_environment(require_one_gpu=False)


def _required_slurm_int(name: str, *, minimum: int, maximum: int) -> int:
    """Read a bounded integer from the active topology-probe Slurm step."""

    value = os.environ.get(name)
    if value is None or not value.isdecimal():
        raise EvaluationError(f"AITA r006 GPU-topology probe requires integer {name}; got {value!r}")
    result = int(value)
    if result < minimum or result > maximum:
        raise EvaluationError(
            f"AITA r006 GPU-topology probe requires {name} in [{minimum}, {maximum}], got {result}"
        )
    return result


def _gpu_topology_probe_root(paths: CampaignPaths) -> Path:
    root = paths.root / GPU_TOPOLOGY_PROBE_DIRNAME
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise EvaluationError(f"AITA r006 GPU-topology probe root must be a regular directory: {root}")
    return root


def _gpu_topology_record_path(paths: CampaignPaths, *, rank: int) -> Path:
    if rank < 0 or rank >= GPU_TOPOLOGY_TASK_COUNT:
        raise EvaluationError("AITA r006 GPU-topology probe rank is outside [0, 15]")
    return _gpu_topology_probe_root(paths) / f"rank-{rank:03d}.json"


def _gpu_topology_receipt_path(paths: CampaignPaths) -> Path:
    return _gpu_topology_probe_root(paths) / "topology-receipt.json"


def _visible_gpu_uuid() -> str:
    """Read the one physical UUID permitted to this Slurm task.

    This exact ``nvidia-smi`` query was empirically proven on Isambard for the
    r006 ``--gpus=16 --gpus-per-task=1 --gpu-bind=verbose,per_task:1`` step.
    CUDA logical IDs are remapped to ``0`` on every isolated rank, so they
    cannot establish whether ranks received distinct physical devices.
    """

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise EvaluationError(
            "AITA r006 GPU-topology probe could not query nvidia-smi UUIDs: "
            f"exit={result.returncode}, stderr={result.stderr.strip()!r}"
        )
    uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if (
        len(uuids) != 1
        or not uuids[0].startswith("GPU-")
        or any(character.isspace() for character in uuids[0])
    ):
        raise EvaluationError(
            "AITA r006 GPU-topology probe requires one physical GPU UUID from nvidia-smi; "
            f"got {uuids!r}"
        )
    return uuids[0]


def gpu_topology_probe(*, campaign_root: str | Path) -> dict[str, Any]:
    """Publish one immutable rank record for the bounded 16-rank GPU probe.

    The enclosing ``srun`` is intentionally the same topology as the later
    full evaluation but runs only this visibility/UUID check.  It performs no
    task construction or model generation.  A coordinator seals the sixteen
    rank records before ``prepare`` is permitted to write campaign receipts.
    """

    paths = _campaign_paths(campaign_root)
    if paths.contract.exists() or paths.contract.is_symlink():
        raise EvaluationError("AITA r006 GPU-topology probe must run before the campaign launch contract exists")
    rank = _required_slurm_int("SLURM_PROCID", minimum=0, maximum=GPU_TOPOLOGY_TASK_COUNT - 1)
    local_rank = _required_slurm_int("SLURM_LOCALID", minimum=0, maximum=3)
    node_id = _required_slurm_int("SLURM_NODEID", minimum=0, maximum=3)
    task_count = _required_slurm_int(
        "SLURM_NTASKS", minimum=GPU_TOPOLOGY_TASK_COUNT, maximum=GPU_TOPOLOGY_TASK_COUNT
    )
    node_count = _required_slurm_int("SLURM_NNODES", minimum=4, maximum=4)
    if node_id != rank // 4 or local_rank != rank % 4:
        raise EvaluationError(
            "AITA r006 GPU-topology probe requires --distribution=block:block; "
            f"got rank={rank}, node_id={node_id}, local_rank={local_rank}"
        )
    job_id = os.environ.get("SLURM_JOB_ID")
    step_id = os.environ.get("SLURM_STEP_ID")
    if not isinstance(job_id, str) or not job_id or not isinstance(step_id, str) or not step_id:
        raise EvaluationError("AITA r006 GPU-topology probe requires non-empty SLURM_JOB_ID and SLURM_STEP_ID")
    visible_gpu = _single_slurm_visible_gpu()
    runtime = _validate_evaluator_environment(require_one_gpu=True)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - covered by runtime validation on the evaluator
        raise EvaluationError("AITA r006 GPU-topology probe cannot import torch") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise EvaluationError(
            "AITA r006 GPU-topology probe requires torch to see exactly one CUDA device; "
            f"available={torch.cuda.is_available()}, count={torch.cuda.device_count()}"
        )
    record = {
        "schema": GPU_TOPOLOGY_RECORD_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "gpu_bind": f"--gpu-bind={GPU_BINDING}",
        "distribution": f"--distribution={GPU_DISTRIBUTION}",
        "step_gpu_count": FULL_STEP_GPU_COUNT,
        "task_count": task_count,
        "node_count": node_count,
        "rank": rank,
        "local_rank": local_rank,
        "node_id": node_id,
        "hostname": socket.gethostname(),
        "cuda_visible_devices": visible_gpu,
        "torch_cuda_device_count": 1,
        "nvidia_smi_uuid": _visible_gpu_uuid(),
        "slurm": {
            "job_id": job_id,
            "step_id": step_id,
            "step_gpus": os.environ.get("SLURM_STEP_GPUS"),
        },
        "runtime": runtime,
    }
    path = _gpu_topology_record_path(paths, rank=rank)
    status = _write_immutable_json(path, record, label=f"AITA r006 GPU-topology rank-{rank:03d} record")
    return {
        "status": status,
        "rank": rank,
        "record": _identity(path, label=f"AITA r006 GPU-topology rank-{rank:03d} record"),
    }


def _read_gpu_topology_record(paths: CampaignPaths, *, rank: int) -> dict[str, Any]:
    path = _gpu_topology_record_path(paths, rank=rank)
    record = _read_json(path, label=f"AITA r006 GPU-topology rank-{rank:03d} record")
    expected_keys = {
        "schema",
        "campaign",
        "gpu_bind",
        "distribution",
        "step_gpu_count",
        "task_count",
        "node_count",
        "rank",
        "local_rank",
        "node_id",
        "hostname",
        "cuda_visible_devices",
        "torch_cuda_device_count",
        "nvidia_smi_uuid",
        "slurm",
        "runtime",
    }
    if set(record) != expected_keys:
        raise EvaluationError(f"AITA r006 GPU-topology rank-{rank:03d} record has an unsupported schema")
    if (
        record.get("schema") != GPU_TOPOLOGY_RECORD_SCHEMA
        or record.get("campaign") != CAMPAIGN_NAME
        or record.get("gpu_bind") != f"--gpu-bind={GPU_BINDING}"
        or record.get("distribution") != f"--distribution={GPU_DISTRIBUTION}"
        or record.get("step_gpu_count") != FULL_STEP_GPU_COUNT
        or record.get("task_count") != GPU_TOPOLOGY_TASK_COUNT
        or record.get("node_count") != 4
        or record.get("rank") != rank
        or record.get("node_id") != rank // 4
        or record.get("local_rank") != rank % 4
        or not isinstance(record.get("hostname"), str)
        or not record["hostname"]
        or not isinstance(record.get("cuda_visible_devices"), str)
        or not record["cuda_visible_devices"]
        or "," in record["cuda_visible_devices"]
        or record.get("torch_cuda_device_count") != 1
        or not isinstance(record.get("nvidia_smi_uuid"), str)
        or not record["nvidia_smi_uuid"].startswith("GPU-")
        or not isinstance(record.get("slurm"), Mapping)
        or not isinstance(record["slurm"].get("job_id"), str)
        or not record["slurm"]["job_id"]
        or not isinstance(record["slurm"].get("step_id"), str)
        or not record["slurm"]["step_id"]
        or not isinstance(record.get("runtime"), Mapping)
    ):
        raise EvaluationError(f"AITA r006 GPU-topology rank-{rank:03d} record violates the binding contract")
    return record


def _expected_gpu_topology_receipt(paths: CampaignPaths) -> dict[str, Any]:
    root = _gpu_topology_probe_root(paths)
    if root.is_symlink() or not root.is_dir():
        raise EvaluationError("AITA r006 GPU-topology probe records are absent")
    expected_record_names = {f"rank-{rank:03d}.json" for rank in range(GPU_TOPOLOGY_TASK_COUNT)}
    record_names = {item.name for item in root.iterdir() if item.is_file() and not item.is_symlink()}
    allowed_names = expected_record_names | {"topology-receipt.json"}
    if record_names - allowed_names or expected_record_names - record_names:
        raise EvaluationError("AITA r006 GPU-topology probe has missing or unexpected rank-record files")
    if any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise EvaluationError("AITA r006 GPU-topology probe must contain only regular receipt files")
    records = [_read_gpu_topology_record(paths, rank=rank) for rank in range(GPU_TOPOLOGY_TASK_COUNT)]
    job_ids = {str(record["slurm"]["job_id"]) for record in records}
    step_ids = {str(record["slurm"]["step_id"]) for record in records}
    runtimes = [record["runtime"] for record in records]
    if len(job_ids) != 1 or len(step_ids) != 1 or any(runtime != runtimes[0] for runtime in runtimes[1:]):
        raise EvaluationError("AITA r006 GPU-topology probe records do not describe one consistent Slurm step")
    topology: list[dict[str, Any]] = []
    all_uuids: set[str] = set()
    hostnames: set[str] = set()
    for node_id in range(4):
        node_records = [record for record in records if record["node_id"] == node_id]
        node_uuids = [str(record["nvidia_smi_uuid"]) for record in node_records]
        node_hosts = {str(record["hostname"]) for record in node_records}
        if len(node_records) != 4 or len(set(node_uuids)) != 4 or len(node_hosts) != 1:
            raise EvaluationError(f"AITA r006 GPU-topology probe node {node_id} is not a four-distinct-GPU rank block")
        all_uuids.update(node_uuids)
        hostnames.update(node_hosts)
        topology.append(
            {
                "node_id": node_id,
                "hostname": next(iter(node_hosts)),
                "ranks": [record["rank"] for record in node_records],
                "gpu_uuids": node_uuids,
            }
        )
    if len(all_uuids) != GPU_TOPOLOGY_TASK_COUNT or len(hostnames) != 4:
        raise EvaluationError("AITA r006 GPU-topology probe does not prove sixteen distinct GPUs across four nodes")
    return {
        "schema": GPU_TOPOLOGY_RECEIPT_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "gpu_bind": f"--gpu-bind={GPU_BINDING}",
        "distribution": f"--distribution={GPU_DISTRIBUTION}",
        "step_gpu_count": FULL_STEP_GPU_COUNT,
        "task_count": GPU_TOPOLOGY_TASK_COUNT,
        "job_id": next(iter(job_ids)),
        "step_id": next(iter(step_ids)),
        "rank_records": [
            _identity(_gpu_topology_record_path(paths, rank=rank), label=f"AITA r006 GPU-topology rank-{rank:03d} record")
            for rank in range(GPU_TOPOLOGY_TASK_COUNT)
        ],
        "topology": topology,
        "global_gpu_uuid_count": len(all_uuids),
        "runtime": dict(runtimes[0]),
    }


def seal_gpu_topology_probe(*, campaign_root: str | Path) -> dict[str, Any]:
    """Seal the successful 16-rank binding probe before campaign preparation."""

    paths = _campaign_paths(campaign_root)
    if paths.contract.exists() or paths.contract.is_symlink():
        raise EvaluationError("AITA r006 GPU-topology probe must seal before the campaign launch contract exists")
    receipt = _expected_gpu_topology_receipt(paths)
    receipt_path = _gpu_topology_receipt_path(paths)
    status = _write_immutable_json(receipt_path, receipt, label="AITA r006 GPU-topology receipt")
    return {
        "status": status,
        "topology_receipt": _identity(receipt_path, label="AITA r006 GPU-topology receipt"),
    }


def _sealed_gpu_topology_receipt(paths: CampaignPaths) -> dict[str, Any]:
    receipt_path = _gpu_topology_receipt_path(paths)
    receipt = _read_json(receipt_path, label="AITA r006 GPU-topology receipt")
    expected = _expected_gpu_topology_receipt(paths)
    if receipt != expected:
        raise EvaluationError("AITA r006 GPU-topology receipt differs from its immutable rank records")
    return receipt


def _next_attempt(root: Path, *, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise EvaluationError(f"attempt root must be a regular directory: {root}")
    for index in range(1, 10_000):
        candidate = root / f"{label}-{index:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            if candidate.is_symlink() or not candidate.is_dir():
                raise EvaluationError(f"attempt name is occupied by a non-directory: {candidate}")
            continue
        return candidate
    raise EvaluationError(f"exhausted preserved attempt slots under {root}")


def _core_prepare_module():
    try:
        from experiments.elephant_aita_ntaflip import prepare
    except ImportError as exc:  # pragma: no cover - package is deployed with this launcher
        raise EvaluationError("ELEPHANT AITA-NTA-FLIP preparation package is unavailable") from exc
    return prepare


def _core_preflight_module():
    try:
        from experiments.elephant_aita_ntaflip import preflight
    except ImportError as exc:  # pragma: no cover - package is deployed with this launcher
        raise EvaluationError("ELEPHANT AITA-NTA-FLIP preflight package is unavailable") from exc
    return preflight


def _validate_official_manifest(path: Path) -> dict[str, Any]:
    """Delegate source/hash/pair validation to the benchmark owner, then bind it."""

    core = _core_prepare_module()
    try:
        document = core.validate_manifest(path)
    except Exception as exc:
        raise EvaluationError(f"official AITA-NTA-FLIP manifest failed validation: {exc}") from exc
    if not isinstance(document, Mapping):
        raise EvaluationError("official AITA-NTA-FLIP manifest validator returned no object")
    if document.get("schema") != AITA_MANIFEST_SCHEMA:
        raise EvaluationError("AITA r006 launcher refuses legacy or differently-schemaed manifests")
    if document.get("no_token_cap_policy") != NO_TOKEN_CAP_POLICY:
        raise EvaluationError("AITA r006 requires the frozen r005 no-token-cap manifest policy")
    try:
        assert_no_token_cap_mapping(document.get("sampling_config"), label="AITA r006 manifest sampling_config")
        assert_no_token_cap_mapping(document.get("generation_config"), label="AITA r006 manifest generation_config")
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc
    expected_pairs = document.get("expected_pairs", document.get("pair_count"))
    if expected_pairs != EXPECTED_PAIRS:
        raise EvaluationError(f"official AITA-NTA-FLIP manifest must bind exactly {EXPECTED_PAIRS} pairs")
    return dict(document)


def materialize_official_manifest(*, official_source_dir: str | Path, paths: CampaignPaths) -> dict[str, Any]:
    """Build/replay the benchmark-owned source manifest under this fresh campaign."""

    source = _resolve_unlinked(official_source_dir, label="official AITA-NTA-FLIP source directory")
    if not source.is_dir():
        raise EvaluationError("official AITA-NTA-FLIP source must be a regular directory")
    if paths.input.exists() and (paths.input.is_symlink() or not paths.input.is_dir()):
        raise EvaluationError("campaign input root must be a regular directory")
    if not paths.manifest.exists():
        if paths.input.exists() and any(paths.input.iterdir()):
            raise FileExistsError(
                f"refusing to create an official manifest in a non-empty campaign input directory: {paths.input}"
            )
        core = _core_prepare_module()
        try:
            result = core.build_manifest(source, paths.input)
        except Exception as exc:
            raise EvaluationError(f"could not build official AITA-NTA-FLIP manifest: {exc}") from exc
        if not isinstance(result, Mapping):
            raise EvaluationError("official AITA-NTA-FLIP manifest builder returned no object")
    validated = _validate_official_manifest(paths.manifest)
    return {
        "manifest": _identity(paths.manifest, label="official AITA-NTA-FLIP manifest"),
        "validated": validated,
        "source_directory": str(source),
    }


def _r002_module():
    try:
        from infra.isambard import run_qwen35_rmct_checkpoint_two_bias_evals_16gpu as r002
    except ImportError as exc:  # pragma: no cover - deployment error
        raise EvaluationError("the sealed r002 early-checkpoint evaluator is unavailable") from exc
    return r002


def _identity_record(value: object, *, label: str, root: Path | None = None) -> dict[str, Any]:
    """Require a portable file identity returned by a raw-custody verifier."""

    if not isinstance(value, Mapping):
        raise EvaluationError(f"{label} is absent from raw checkpoint custody")
    path = value.get("path")
    digest = value.get("sha256")
    size = value.get("size_bytes")
    if (
        not isinstance(path, str)
        or not _valid_sha256(digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
    ):
        raise EvaluationError(f"{label} has an incomplete immutable identity")
    candidate = Path(path)
    if not candidate.is_absolute():
        if root is None:
            raise EvaluationError(f"{label} must use an absolute path")
        candidate = _under_root(root / candidate, root, label=label)
    actual = _identity(candidate, label=label)
    if actual["sha256"] != digest or actual["size_bytes"] != size:
        raise EvaluationError(f"{label} changed after raw-custody validation")
    return actual


def _revalidate_early_checkpoint_source(*, training_repository: str | Path, step: int) -> dict[str, Any]:
    """Replay an approved raw step-16/64 checkpoint, not a prior eval run."""

    r002 = _r002_module()
    try:
        custody = r002.validate_approved_target(training_repository, step=step)
    except Exception as exc:
        raise EvaluationError(f"raw step-{step} training checkpoint failed sealed-custody replay: {exc}") from exc
    if not isinstance(custody, Mapping):
        raise EvaluationError(f"raw step-{step} custody verifier returned no object")
    target = custody.get("target")
    checkpoint = custody.get("checkpoint")
    if (
        not isinstance(target, Mapping)
        or target.get("step") != step
        or not isinstance(target.get("condition"), str)
        or not isinstance(checkpoint, Mapping)
        or not isinstance(checkpoint.get("path"), str)
        or not Path(checkpoint["path"]).is_absolute()
        or checkpoint.get("optimizer_step") != step
        or checkpoint.get("full_resumability_required") is not True
    ):
        raise EvaluationError(f"raw step-{step} custody lacks the exact resumable checkpoint identity")
    files = checkpoint.get("files")
    if not isinstance(files, Mapping):
        raise EvaluationError(f"raw step-{step} custody lacks its complete checkpoint file set")
    required_files = {
        "adapter_config",
        "adapter_model",
        "optimizer",
        "manifest",
        "replicated_training_manifest",
        "replicated_training_rng",
    }
    if set(files) != required_files:
        raise EvaluationError(f"raw step-{step} custody has an incomplete resumability file set")
    for name in sorted(required_files):
        _identity_record(files[name], label=f"raw step-{step} checkpoint {name}")
    return {
        "source": "raw-training-checkpoint",
        "step": step,
        "condition": target["condition"],
        "checkpoint": dict(checkpoint),
        "checkpoint_receipt": _identity_record(custody.get("checkpoint_receipt"), label=f"raw step-{step} checkpoint receipt"),
        "completion_receipt": _identity_record(custody.get("completion_receipt"), label=f"raw step-{step} completion receipt"),
        "decision_receipt": _identity_record(custody.get("decision_receipt"), label=f"raw step-{step} continuation decision"),
    }


def _r005_module():
    try:
        from infra.isambard import run_qwen35_rmct_convergence_r4_two_bias_evals as r005
    except ImportError as exc:  # pragma: no cover - deployment error
        raise EvaluationError("the sealed r005 terminal evaluator is unavailable") from exc
    return r005


def _revalidate_terminal_checkpoint_source(*, training_repository: str | Path) -> dict[str, Any]:
    """Replay the converged raw r4 checkpoint without touching its eval tree."""

    r005 = _r005_module()
    repository = _resolve_unlinked(training_repository, label="terminal training repository")
    try:
        custody = r005.validate_final_checkpoint(repository)
    except Exception as exc:
        raise EvaluationError(f"raw terminal step-176 training checkpoint failed sealed-custody replay: {exc}") from exc
    if not isinstance(custody, Mapping):
        raise EvaluationError("terminal raw checkpoint custody verifier returned no object")
    checkpoint_path = custody.get("path")
    strict_custody = custody.get("custody")
    if not isinstance(checkpoint_path, str) or not Path(checkpoint_path).is_absolute() or not isinstance(strict_custody, Mapping):
        raise EvaluationError("terminal raw checkpoint custody lacks its strict resumability identity")
    strict_files = strict_custody.get("files")
    required_files = {
        "adapter_config",
        "adapter_model",
        "optimizer",
        "manifest",
        "replicated_training_manifest",
        "replicated_training_rng",
    }
    if not isinstance(strict_files, Mapping) or set(strict_files) != required_files:
        raise EvaluationError("terminal raw checkpoint custody has an incomplete resumability file set")
    files = {
        name: _identity_record(strict_files[name], label=f"terminal checkpoint {name}", root=repository)
        for name in sorted(required_files)
    }
    if (
        files["adapter_config"] != _identity_record(custody.get("adapter_config"), label="terminal checkpoint adapter config")
        or files["adapter_model"] != _identity_record(custody.get("adapter_model"), label="terminal checkpoint adapter weights")
        or files["manifest"] != _identity_record(custody.get("manifest"), label="terminal checkpoint manifest")
    ):
        raise EvaluationError("terminal raw checkpoint custody disagrees with its strict file identities")
    decision = _identity_record(custody.get("decision_receipt"), label="terminal convergence decision")
    # ``validate_final_checkpoint`` delegates to the production strict
    # verifier, which binds optimizer, replicated manifest, and RNG.  Retain
    # the complete returned object in the condition receipt for replay.
    return {
        "source": "raw-training-checkpoint",
        "step": 176,
        "condition": R005_CONDITION,
        "checkpoint": {
            "path": checkpoint_path,
            "optimizer_step": 176,
            "full_resumability_required": True,
            "files": files,
            "strict_custody": dict(strict_custody),
        },
        "decision_receipt": decision,
    }


def _base_runtime(
    *,
    snapshot: Mapping[str, Any] | None = None,
    evaluator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot_identity = dict(snapshot) if snapshot is not None else _snapshot_identity()
    evaluator_identity = dict(evaluator) if evaluator is not None else _evaluator_runtime()
    return {
        "mode": "native-hf-base",
        "checkpoint_backend": "pinned-base-snapshot",
        "base_model": str(MODEL_SNAPSHOT),
        "model": f"hf/{MODEL_SNAPSHOT}",
        "model_snapshot": snapshot_identity,
        "provider": "hf",
        "model_args": dict(HF_MODEL_ARGS),
        "qwen_thinking_policy": dict(QWEN_THINKING_POLICY),
        "no_token_cap_policy": dict(NO_TOKEN_CAP_POLICY),
        "no_token_cap_runtime_policy": dict(NO_TOKEN_CAP_RUNTIME_POLICY),
        "sampling_config": dict(GENERATION_CONFIG),
        "concurrency_config": dict(CONCURRENCY_CONFIG),
        "generation_config": dict(RUNTIME_GENERATION_CONFIG),
        "evaluator": evaluator_identity,
    }


def _trained_runtime(
    source: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any] | None = None,
    evaluator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_custody = source.get("checkpoint")
    checkpoint = checkpoint_custody.get("path") if isinstance(checkpoint_custody, Mapping) else None
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
        raise EvaluationError("raw checkpoint custody has no PEFT adapter path")
    snapshot_identity = dict(snapshot) if snapshot is not None else _snapshot_identity()
    evaluator_identity = dict(evaluator) if evaluator is not None else _evaluator_runtime()
    return {
        "mode": "native-hf-peft",
        "checkpoint_backend": "local",
        "base_model": str(MODEL_SNAPSHOT),
        "model": f"hf/{MODEL_SNAPSHOT}",
        "model_snapshot": snapshot_identity,
        "checkpoint": checkpoint,
        "raw_checkpoint_custody": dict(source),
        "source_checkpoint": {
            "source": source["source"],
            "step": source["step"],
            "condition": source["condition"],
        },
        "provider": "hf",
        "model_args": dict(HF_LOCAL_MODEL_ARGS),
        "qwen_thinking_policy": dict(QWEN_THINKING_POLICY),
        "no_token_cap_policy": dict(NO_TOKEN_CAP_POLICY),
        "no_token_cap_runtime_policy": dict(NO_TOKEN_CAP_RUNTIME_POLICY),
        "sampling_config": dict(GENERATION_CONFIG),
        "concurrency_config": dict(CONCURRENCY_CONFIG),
        "generation_config": dict(RUNTIME_GENERATION_CONFIG),
        "evaluator": evaluator_identity,
    }


def _attach_task_metadata(runtime: Mapping[str, Any], *, manifest_sha256: str) -> dict[str, Any]:
    """Add the condition-invariant Inspect metadata expected on every shard."""

    if not _valid_sha256(manifest_sha256):
        raise EvaluationError("official AITA manifest has an invalid SHA-256")
    result = dict(runtime)
    result["metadata"] = {
        "benchmark": BENCHMARK,
        "schema": AITA_MANIFEST_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "n_shards": SHARD_COUNT,
        "prompt_suffix": PROMPT_SUFFIX,
        "system_prompt": SYSTEM_PROMPT,
        "qwen_thinking_policy": dict(QWEN_THINKING_POLICY),
        "no_token_cap_policy": dict(NO_TOKEN_CAP_POLICY),
        "no_token_cap_runtime_policy": dict(NO_TOKEN_CAP_RUNTIME_POLICY),
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
    snapshot_identity = dict(snapshot) if snapshot is not None else _snapshot_identity()
    evaluator_identity = dict(evaluator) if evaluator is not None else _evaluator_runtime()
    step16 = _revalidate_early_checkpoint_source(training_repository=training_repository, step=16)
    step64 = _revalidate_early_checkpoint_source(training_repository=training_repository, step=64)
    step176 = _revalidate_terminal_checkpoint_source(training_repository=training_repository)
    return {
        "base": _attach_task_metadata(
            _base_runtime(snapshot=snapshot_identity, evaluator=evaluator_identity),
            manifest_sha256=manifest_sha256,
        ),
        "step016": _attach_task_metadata(
            _trained_runtime(step16, snapshot=snapshot_identity, evaluator=evaluator_identity),
            manifest_sha256=manifest_sha256,
        ),
        "step064": _attach_task_metadata(
            _trained_runtime(step64, snapshot=snapshot_identity, evaluator=evaluator_identity),
            manifest_sha256=manifest_sha256,
        ),
        "step176": _attach_task_metadata(
            _trained_runtime(step176, snapshot=snapshot_identity, evaluator=evaluator_identity),
            manifest_sha256=manifest_sha256,
        ),
    }


def _benchmark_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK,
        "task_factory": TASK_FACTORY,
        "expected_pairs": EXPECTED_PAIRS,
        "perspectives_per_pair": PERSPECTIVES_PER_PAIR,
        "generations_per_condition": GENERATIONS_PER_CONDITION,
        "n_shards": SHARD_COUNT,
        "pair_preserving_shards": True,
        "prompt_suffix": PROMPT_SUFFIX,
        "system_prompt": SYSTEM_PROMPT,
        "qwen_thinking_policy": dict(QWEN_THINKING_POLICY),
        "no_token_cap_policy": dict(NO_TOKEN_CAP_POLICY),
        "no_token_cap_runtime_policy": dict(NO_TOKEN_CAP_RUNTIME_POLICY),
        "sampling_config": dict(GENERATION_CONFIG),
        "concurrency_config": dict(CONCURRENCY_CONFIG),
        "generation_config": dict(RUNTIME_GENERATION_CONFIG),
        "reporting": {
            "primary_metric": {
                "name": "final_answer_only_both_nta",
                "label": "final-answer-only NTA/NTA (modified protocol; invalid finals retained)",
                "direction": "lower_is_better",
                "parser_schema": FINAL_ANSWER_PARSER_SCHEMA,
                "report_path": ["metrics", "primary_metric", "value"],
                "parsed_response_coverage_path": [
                    "metrics",
                    "primary_metric",
                    "parsed_response_coverage",
                ],
                "parsed_pair_coverage_path": [
                    "metrics",
                    "primary_metric",
                    "parsed_pair_coverage",
                ],
            },
            "secondary_metric": {
                "name": "strict_mutually_exclusive_nta_nta",
                "label": "legacy whole-response mutually-exclusive NTA/NTA diagnostic",
                "direction": "lower_is_better",
                "parser_schema": FINAL_ANSWER_PARSER_SCHEMA,
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
    }


def build_launch_contract(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    training_repository: str | Path,
) -> tuple[dict[str, Any], CampaignPaths]:
    """Build one deterministic plan after replaying raw checkpoint custody."""

    paths = _campaign_paths(campaign_root)
    gpu_topology_probe = _sealed_gpu_topology_receipt(paths)
    evaluator = _validate_evaluator_environment(require_one_gpu=False)
    repository = _resolve_unlinked(training_repository, label="sealed training repository")
    if not repository.is_dir():
        raise EvaluationError("sealed training repository must be a regular directory")
    official = materialize_official_manifest(official_source_dir=official_source_dir, paths=paths)
    snapshot = _snapshot_identity()
    runtime = _condition_runtime_records(
        training_repository=repository,
        manifest_sha256=official["manifest"]["sha256"],
        snapshot=snapshot,
        evaluator=evaluator,
    )
    contract = {
        "schema": LAUNCH_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "training_repository": str(repository),
        "model_snapshot": snapshot,
        "official_manifest": official,
        "gpu_topology_probe": _identity(
            _gpu_topology_receipt_path(paths), label="AITA r006 GPU-topology receipt"
        ),
        "benchmark": _benchmark_contract(official["validated"]),
        "evaluator": evaluator,
        "conditions": [{"name": item.name, "label": item.label, "optimizer_step": item.optimizer_step, "source_kind": item.source_kind, "runtime": runtime[item.name]} for item in CONDITIONS],
        "topology": {
            "nodes": 4,
            "gpus_per_node": 4,
            "workers": 16,
            "one_gpu_per_worker": True,
            "gpu_binding": {
                "gpu_bind": f"--gpu-bind={GPU_BINDING}",
                "distribution": f"--distribution={GPU_DISTRIBUTION}",
                "full_step_gpu_count": FULL_STEP_GPU_COUNT,
                "smoke_step_gpu_count": SMOKE_STEP_GPU_COUNT,
                "topology_probe_schema": gpu_topology_probe["schema"],
                "topology_probe_sha256": _identity(
                    _gpu_topology_receipt_path(paths), label="AITA r006 GPU-topology receipt"
                )["sha256"],
            },
            "native_hf_peft_workers": 16,
            "cells": [{"condition": item.name, "shard_index": shard} for item in CONDITIONS for shard in range(SHARD_COUNT)],
        },
        "critical_sources": _critical_source_identities(),
        "outputs": {
            "root": str(paths.root),
            "input": str(paths.input),
            "conditions": str(paths.conditions),
            "completion": str(paths.completion),
        },
        "policy": _launch_policy(),
    }
    return contract, paths


def _write_condition_evaluation_receipt(
    *, paths: CampaignPaths, contract: Mapping[str, Any], condition: Condition
) -> str:
    condition_paths = _condition_paths(paths, condition)
    rows = contract.get("conditions")
    if not isinstance(rows, list):
        raise EvaluationError("campaign contract has no conditions")
    matched = [row for row in rows if isinstance(row, Mapping) and row.get("name") == condition.name]
    if len(matched) != 1:
        raise EvaluationError(f"campaign contract has ambiguous runtime for {condition.name}")
    runtime = matched[0].get("runtime")
    if not isinstance(runtime, Mapping):
        raise EvaluationError(f"campaign contract has no runtime for {condition.name}")
    receipt = {
        "schema": EVALUATION_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "condition": {"name": condition.name, "label": condition.label, "optimizer_step": condition.optimizer_step},
        "launch_contract": _identity(paths.contract, label="AITA campaign launch contract"),
        "official_manifest": contract["official_manifest"]["manifest"],
        "benchmark": contract["benchmark"],
        "runtime": dict(runtime),
        "outputs": {"root": str(condition_paths.root), "raw": str(condition_paths.raw), "attempts": str(condition_paths.attempts)},
    }
    return _write_immutable_json(condition_paths.evaluation_receipt, receipt, label=f"{condition.name} AITA evaluation receipt")


def prepare(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    training_repository: str | Path,
) -> dict[str, Any]:
    """Create/replay campaign receipts, but deliberately perform no generation."""

    paths = _campaign_paths(campaign_root)
    _sealed_gpu_topology_receipt(paths)
    if paths.root.exists() and not paths.contract.exists() and {
        item.name for item in paths.root.iterdir()
    } != {GPU_TOPOLOGY_PROBE_DIRNAME}:
        raise FileExistsError(f"refusing to seed an AITA contract in a non-empty campaign root: {paths.root}")
    contract, paths = build_launch_contract(
        campaign_root=campaign_root,
        official_source_dir=official_source_dir,
        training_repository=training_repository,
    )
    status = _write_immutable_json(paths.contract, contract, label="AITA campaign launch contract")
    receipts = {
        condition.name: _write_condition_evaluation_receipt(paths=paths, contract=contract, condition=condition)
        for condition in CONDITIONS
    }
    return {
        "campaign": CAMPAIGN_NAME,
        "launch_contract": _identity(paths.contract, label="AITA campaign launch contract"),
        "launch_status": status,
        "evaluation_receipt_statuses": receipts,
        "cells": 16,
        "generations_per_condition": GENERATIONS_PER_CONDITION,
    }


def _load_campaign(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    training_repository: str | Path,
) -> tuple[dict[str, Any], CampaignPaths]:
    paths = _campaign_paths(campaign_root)
    stored = _read_json(paths.contract, label="AITA campaign launch contract")
    rebuilt, rebuilt_paths = build_launch_contract(
        campaign_root=campaign_root,
        official_source_dir=official_source_dir,
        training_repository=training_repository,
    )
    if paths != rebuilt_paths or stored != rebuilt:
        raise EvaluationError("AITA campaign launch contract differs from current replayed source/checkpoint/runtime custody")
    return rebuilt, paths


def _condition_row(contract: Mapping[str, Any], condition: Condition) -> Mapping[str, Any]:
    rows = contract.get("conditions")
    if not isinstance(rows, list):
        raise EvaluationError("campaign contract lacks condition records")
    matched = [row for row in rows if isinstance(row, Mapping) and row.get("name") == condition.name]
    if len(matched) != 1:
        raise EvaluationError(f"campaign contract has ambiguous condition record for {condition.name}")
    return matched[0]


def _expected_evaluation_receipt(contract: Mapping[str, Any], paths: CampaignPaths, condition: Condition) -> dict[str, Any]:
    condition_paths = _condition_paths(paths, condition)
    row = _condition_row(contract, condition)
    runtime = row.get("runtime")
    if not isinstance(runtime, Mapping):
        raise EvaluationError("campaign condition record lacks runtime")
    return {
        "schema": EVALUATION_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "condition": {"name": condition.name, "label": condition.label, "optimizer_step": condition.optimizer_step},
        "launch_contract": _identity(paths.contract, label="AITA campaign launch contract"),
        "official_manifest": contract["official_manifest"]["manifest"],
        "benchmark": contract["benchmark"],
        "runtime": dict(runtime),
        "outputs": {"root": str(condition_paths.root), "raw": str(condition_paths.raw), "attempts": str(condition_paths.attempts)},
    }


def _load_evaluation_receipt(contract: Mapping[str, Any], paths: CampaignPaths, condition: Condition) -> tuple[dict[str, Any], str]:
    candidate = _condition_paths(paths, condition).evaluation_receipt
    receipt = _read_json(candidate, label=f"{condition.name} AITA evaluation receipt")
    expected = _expected_evaluation_receipt(contract, paths, condition)
    if receipt != expected:
        raise EvaluationError(f"{condition.name} AITA evaluation receipt differs from replayed campaign custody")
    return expected, _sha256_file(candidate)


def _task_receipt_path(paths: ConditionPaths, shard_index: int) -> Path:
    if shard_index < 0 or shard_index >= SHARD_COUNT:
        raise EvaluationError("AITA shard index is outside [0, 3]")
    return paths.receipts / f"shard-{shard_index:03d}.json"


def _metadata_expected(*, manifest_sha256: str, shard_index: int) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK,
        "schema": AITA_MANIFEST_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "shard_index": shard_index,
        "n_shards": SHARD_COUNT,
        "prompt_suffix": PROMPT_SUFFIX,
        "system_prompt": SYSTEM_PROMPT,
        "qwen_thinking_policy": dict(QWEN_THINKING_POLICY),
        "no_token_cap_policy": dict(NO_TOKEN_CAP_POLICY),
        "no_token_cap_runtime_policy": dict(NO_TOKEN_CAP_RUNTIME_POLICY),
        "sampling_config": dict(GENERATION_CONFIG),
        "concurrency_config": dict(CONCURRENCY_CONFIG),
        "generation_config": dict(RUNTIME_GENERATION_CONFIG),
    }


def _expected_model_from_condition_receipt(paths: ConditionPaths) -> str:
    """Extract the one model spelling that a promoted EvalLog may claim."""

    receipt = _read_json(paths.evaluation_receipt, label="AITA condition evaluation receipt")
    runtime = receipt.get("runtime")
    model = runtime.get("model") if isinstance(runtime, Mapping) else None
    if not isinstance(model, str) or not model:
        raise EvaluationError("AITA condition evaluation receipt lacks its exact Inspect model identity")
    return model


def _expected_runtime_from_condition_receipt(paths: ConditionPaths) -> dict[str, Any]:
    receipt = _read_json(paths.evaluation_receipt, label="AITA condition evaluation receipt")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise EvaluationError("AITA condition evaluation receipt lacks runtime identity")
    mode = runtime.get("mode")
    backend = runtime.get("checkpoint_backend")
    model_args = runtime.get("model_args")
    if (
        mode not in {"native-hf-base", "native-hf-peft"}
        or backend not in {"pinned-base-snapshot", "local"}
        or (mode == "native-hf-base") != (backend == "pinned-base-snapshot")
        or not isinstance(model_args, Mapping)
        or dict(model_args) != (HF_MODEL_ARGS if mode == "native-hf-base" else HF_LOCAL_MODEL_ARGS)
        or runtime.get("qwen_thinking_policy") != QWEN_THINKING_POLICY
        or runtime.get("no_token_cap_policy") != NO_TOKEN_CAP_POLICY
        or runtime.get("no_token_cap_runtime_policy") != NO_TOKEN_CAP_RUNTIME_POLICY
        or runtime.get("sampling_config") != GENERATION_CONFIG
        or runtime.get("concurrency_config") != CONCURRENCY_CONFIG
        or runtime.get("generation_config") != RUNTIME_GENERATION_CONFIG
        or runtime.get("evaluator") != _evaluator_runtime()
        or runtime.get("model_snapshot") != _snapshot_identity()
    ):
        raise EvaluationError("AITA condition evaluation receipt lacks the frozen native-HF identity")
    try:
        assert_no_token_cap_mapping(model_args, label="AITA condition model_args")
        assert_no_token_cap_mapping(runtime.get("sampling_config"), label="AITA condition sampling_config")
        assert_no_token_cap_mapping(runtime.get("generation_config"), label="AITA condition generation_config")
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc
    if mode == "native-hf-peft":
        checkpoint = runtime.get("checkpoint")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
            raise EvaluationError("AITA PEFT condition receipt lacks its raw checkpoint path")
    return dict(runtime)


def _mapping_value(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    for name in ("model_dump", "dict"):
        method = getattr(value, name, None)
        if callable(method):
            candidate = method()
            if isinstance(candidate, Mapping):
                return dict(candidate)
    return {}


def _inspect_success(
    path: Path,
    *,
    manifest_sha256: str,
    shard_index: int,
    expected_model: str | None = None,
    expected_runtime: Mapping[str, Any] | None = None,
) -> tuple[int, int]:
    """Require the core task's exact header identity before publication."""

    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - configured Isambard runtime
        raise EvaluationError("Inspect AI is required to validate AITA EvalLogs") from exc
    try:
        log = read_eval_log(str(path), header_only=True)
    except Exception as exc:
        raise EvaluationError(f"could not read AITA Inspect EvalLog: {path}") from exc
    evaluation = getattr(log, "eval", None)
    metadata = getattr(evaluation, "metadata", {}) if evaluation is not None else {}
    if not isinstance(metadata, Mapping):
        raise EvaluationError("AITA EvalLog has no task metadata")
    if getattr(log, "status", None) != "success" or metadata.get("task_indices") != [1] or metadata.get("task_count") != 1:
        raise EvaluationError("AITA EvalLog is not a successful one-shard task")
    if expected_model is not None and getattr(evaluation, "model", None) != expected_model:
        raise EvaluationError("AITA EvalLog model differs from the condition's sealed runtime receipt")
    if expected_runtime is not None:
        model_args = expected_runtime.get("model_args")
        expected_model_args = (
            HF_MODEL_ARGS if expected_runtime.get("mode") == "native-hf-base" else HF_LOCAL_MODEL_ARGS
        )
        if not isinstance(model_args, Mapping) or dict(model_args) != expected_model_args:
            raise EvaluationError("AITA condition runtime has invalid native-HF model arguments")
        metadata_model_args = _mapping_value(metadata.get("model_args"))
        if metadata_model_args != dict(model_args):
            raise EvaluationError("AITA EvalLog metadata has different native-HF model arguments")
        native_model_args = _mapping_value(getattr(evaluation, "model_args", {}))
        expected_native_args = dict(model_args)
        expected_native_args.pop("provider", None)
        if any(native_model_args.get(key) != value for key, value in expected_native_args.items()):
            raise EvaluationError("AITA EvalLog model arguments do not bind the native-HF provider")
        logged_generate_config = _mapping_value(getattr(evaluation, "model_generate_config", {}))
        if not logged_generate_config:
            raise EvaluationError("AITA EvalLog has no inspectable effective generation configuration")
        try:
            assert_no_token_cap_mapping(
                logged_generate_config,
                label="AITA EvalLog effective generation configuration",
            )
        except ValueError as exc:
            raise EvaluationError(str(exc)) from exc
        if any(logged_generate_config.get(key) != value for key, value in RUNTIME_GENERATION_CONFIG.items()):
            raise EvaluationError("AITA EvalLog effective generation configuration differs from r006")
        if (
            metadata.get("qwen_thinking_policy") != QWEN_THINKING_POLICY
            or metadata.get("no_token_cap_policy") != NO_TOKEN_CAP_POLICY
            or metadata.get("no_token_cap_runtime_policy") != NO_TOKEN_CAP_RUNTIME_POLICY
        ):
            raise EvaluationError("AITA EvalLog lacks the r006 launch thinking/no-token-cap attestations")
        if expected_runtime.get("mode") == "native-hf-peft":
            if (
                metadata.get("checkpoint") != expected_runtime.get("checkpoint")
                or metadata.get("checkpoint_backend") != "local"
                or metadata.get("base_model") != str(MODEL_SNAPSHOT)
            ):
                raise EvaluationError("AITA EvalLog does not bind the condition's raw PEFT checkpoint")
        elif expected_runtime.get("mode") == "native-hf-base":
            if metadata.get("model") != expected_runtime.get("model"):
                raise EvaluationError("AITA EvalLog does not bind the pinned native-HF base snapshot")
        else:
            raise EvaluationError("AITA condition runtime does not name a supported native-HF mode")
    task_args = getattr(evaluation, "task_args", {}) if evaluation is not None else {}
    task_args = task_args if isinstance(task_args, Mapping) else {}

    def header_value(name: str) -> object:
        values = [mapping[name] for mapping in (task_args, metadata) if name in mapping]
        if not values:
            raise EvaluationError(f"AITA EvalLog is missing task/header field {name!r}")
        if any(value != values[0] for value in values[1:]):
            raise EvaluationError(f"AITA EvalLog has disagreeing task/header values for {name!r}")
        return values[0]

    expected = _metadata_expected(manifest_sha256=manifest_sha256, shard_index=shard_index)
    if any(header_value(key) != value for key, value in expected.items()):
        raise EvaluationError("AITA EvalLog header differs from the frozen task/decode identity")
    pair_count = header_value("pair_count")
    generation_count = header_value("generation_count")
    if (
        isinstance(pair_count, bool)
        or not isinstance(pair_count, int)
        or pair_count < 1
        or isinstance(generation_count, bool)
        or generation_count != pair_count * PERSPECTIVES_PER_PAIR
    ):
        raise EvaluationError("AITA EvalLog has invalid pair/generation counts")
    return pair_count, generation_count


def _task_runtime_binding(paths: ConditionPaths) -> dict[str, Any]:
    runtime = _expected_runtime_from_condition_receipt(paths)
    binding: dict[str, Any] = {
        "mode": runtime["mode"],
        "checkpoint_backend": runtime["checkpoint_backend"],
        "model": runtime["model"],
        "model_snapshot": dict(runtime["model_snapshot"]),
        "model_args": dict(runtime["model_args"]),
        "qwen_thinking_policy": dict(runtime["qwen_thinking_policy"]),
        "no_token_cap_policy": dict(runtime["no_token_cap_policy"]),
        "no_token_cap_runtime_policy": dict(runtime["no_token_cap_runtime_policy"]),
        "sampling_config": dict(runtime["sampling_config"]),
        "concurrency_config": dict(runtime["concurrency_config"]),
        "generation_config": dict(runtime["generation_config"]),
        "evaluator": dict(runtime["evaluator"]),
    }
    if runtime["mode"] == "native-hf-peft":
        binding["checkpoint"] = runtime["checkpoint"]
        binding["base_model"] = runtime["base_model"]
    return binding


def _load_task_receipt(
    *,
    condition_paths: ConditionPaths,
    condition: Condition,
    shard_index: int,
    launch_sha256: str,
    evaluation_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any] | None:
    path = _task_receipt_path(condition_paths, shard_index)
    if not path.exists() and not path.is_symlink():
        return None
    receipt = _read_json(path, label=f"{condition.name} shard-{shard_index} receipt")
    required = {
        "schema",
        "condition",
        "shard_index",
        "pair_count",
        "generation_count",
        "launch_contract_sha256",
        "evaluation_receipt_sha256",
        "official_manifest_sha256",
        "runtime",
        "attempt_log",
        "canonical_log",
    }
    if set(receipt) != required or receipt.get("schema") != TASK_RECEIPT_SCHEMA:
        raise EvaluationError(f"{condition.name} shard-{shard_index} has an unsupported receipt schema")
    if (
        receipt.get("condition") != condition.name
        or receipt.get("shard_index") != shard_index
        or receipt.get("launch_contract_sha256") != launch_sha256
        or receipt.get("evaluation_receipt_sha256") != evaluation_sha256
        or receipt.get("official_manifest_sha256") != manifest_sha256
        or receipt.get("runtime") != _task_runtime_binding(condition_paths)
    ):
        raise EvaluationError(f"{condition.name} shard-{shard_index} receipt binds different campaign evidence")
    attempt = receipt.get("attempt_log")
    canonical = receipt.get("canonical_log")
    if not isinstance(attempt, Mapping) or not isinstance(canonical, Mapping):
        raise EvaluationError(f"{condition.name} shard-{shard_index} receipt lacks log identities")
    attempt_path = Path(str(attempt.get("path", "")))
    canonical_path = Path(str(canonical.get("path", "")))
    if attempt_path.is_symlink() or canonical_path.is_symlink():
        raise EvaluationError(f"{condition.name} shard-{shard_index} receipt names a linked log")
    attempt_path = _under_root(attempt_path, condition_paths.attempts, label="AITA attempt log")
    canonical_path = _under_root(canonical_path, condition_paths.raw, label="AITA canonical log")
    if attempt != _identity(attempt_path, label="AITA attempt log") or canonical != _identity(canonical_path, label="AITA canonical log"):
        raise EvaluationError(f"{condition.name} shard-{shard_index} receipt log changed after publication")
    pairs, generations = _inspect_success(
        canonical_path,
        manifest_sha256=manifest_sha256,
        shard_index=shard_index,
        expected_model=_expected_model_from_condition_receipt(condition_paths),
        expected_runtime=_expected_runtime_from_condition_receipt(condition_paths),
    )
    if receipt.get("pair_count") != pairs or receipt.get("generation_count") != generations:
        raise EvaluationError(f"{condition.name} shard-{shard_index} receipt count differs from canonical EvalLog")
    return receipt


def _validate_canonical_raw_custody(
    *,
    condition_paths: ConditionPaths,
    condition: Condition,
    launch_sha256: str,
    evaluation_sha256: str,
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    """Require the raw tree to be exactly the four receipt-attested logs.

    Core preflight rejects stray EvalLogs too, but this receipt-level guard
    makes a wrong-task or unclaimed ``.eval`` fail before any scoring/report
    publication can occur.
    """

    raw = condition_paths.raw
    if raw.is_symlink() or not raw.is_dir():
        raise EvaluationError(f"{condition.name} canonical raw root must be a regular directory")
    receipts: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    for shard_index in range(SHARD_COUNT):
        receipt = _load_task_receipt(
            condition_paths=condition_paths,
            condition=condition,
            shard_index=shard_index,
            launch_sha256=launch_sha256,
            evaluation_sha256=evaluation_sha256,
            manifest_sha256=manifest_sha256,
        )
        if receipt is None:
            raise EvaluationError(f"{condition.name} AITA shard-{shard_index} is not sealed")
        canonical = receipt["canonical_log"]
        canonical_path = Path(str(canonical["path"])).resolve()
        _under_root(canonical_path, raw, label="AITA canonical raw log")
        expected_paths.add(canonical_path)
        receipts.append(receipt)
    if len(expected_paths) != SHARD_COUNT:
        raise EvaluationError(f"{condition.name} canonical receipts do not name four distinct logs")

    found_paths: set[Path] = set()
    for candidate in raw.rglob("*"):
        if candidate.is_symlink():
            raise EvaluationError(f"{condition.name} canonical raw tree contains a symlink: {candidate}")
        if candidate.is_file() and candidate.suffix == ".eval":
            found_paths.add(_under_root(candidate.resolve(), raw, label="AITA raw EvalLog"))
    if found_paths != expected_paths:
        unexpected = sorted(str(path) for path in found_paths - expected_paths)
        missing = sorted(str(path) for path in expected_paths - found_paths)
        raise EvaluationError(
            f"{condition.name} canonical raw tree differs from its four task receipts; "
            f"unexpected={unexpected}, missing={missing}"
        )
    return receipts


def _promote_attempt(
    *,
    attempt: Path,
    condition_paths: ConditionPaths,
    condition: Condition,
    shard_index: int,
    launch_sha256: str,
    evaluation_sha256: str,
    manifest_sha256: str,
) -> bool:
    if attempt.is_symlink() or not attempt.is_dir():
        raise EvaluationError(f"AITA attempt must be a regular directory: {attempt}")
    expected_model = _expected_model_from_condition_receipt(condition_paths)
    expected_runtime = _expected_runtime_from_condition_receipt(condition_paths)
    selected: Path | None = None
    for candidate in sorted(attempt.rglob("*.eval")):
        if candidate.is_symlink() or not candidate.is_file():
            raise EvaluationError(f"AITA attempt contains a linked/non-file EvalLog: {candidate}")
        try:
            _inspect_success(
                candidate,
                manifest_sha256=manifest_sha256,
                shard_index=shard_index,
                expected_model=expected_model,
                expected_runtime=expected_runtime,
            )
        except EvaluationError:
            continue  # incomplete/corrupt/wrong identity evidence remains preserved
        if selected is not None:
            raise EvaluationError(f"AITA attempt produced ambiguous successful EvalLogs: {selected} and {candidate}")
        selected = candidate
    if selected is None:
        return False
    if _load_task_receipt(
        condition_paths=condition_paths,
        condition=condition,
        shard_index=shard_index,
        launch_sha256=launch_sha256,
        evaluation_sha256=evaluation_sha256,
        manifest_sha256=manifest_sha256,
    ) is not None:
        return False
    pairs, generations = _inspect_success(
        selected,
        manifest_sha256=manifest_sha256,
        shard_index=shard_index,
        expected_model=expected_model,
        expected_runtime=expected_runtime,
    )
    digest = _sha256_file(selected)
    canonical = condition_paths.raw / f"shard-{shard_index:03d}" / f"{digest}.eval"
    _copy_immutable_file(selected, canonical, label=f"{condition.name} shard-{shard_index} canonical AITA log")
    _inspect_success(
        canonical,
        manifest_sha256=manifest_sha256,
        shard_index=shard_index,
        expected_model=expected_model,
        expected_runtime=expected_runtime,
    )
    receipt = {
        "schema": TASK_RECEIPT_SCHEMA,
        "condition": condition.name,
        "shard_index": shard_index,
        "pair_count": pairs,
        "generation_count": generations,
        "launch_contract_sha256": launch_sha256,
        "evaluation_receipt_sha256": evaluation_sha256,
        "official_manifest_sha256": manifest_sha256,
        "runtime": _task_runtime_binding(condition_paths),
        "attempt_log": _identity(selected, label="AITA attempt EvalLog"),
        "canonical_log": _identity(canonical, label="AITA canonical EvalLog"),
    }
    _write_immutable_json(_task_receipt_path(condition_paths, shard_index), receipt, label=f"{condition.name} shard-{shard_index} receipt")
    if _load_task_receipt(
        condition_paths=condition_paths,
        condition=condition,
        shard_index=shard_index,
        launch_sha256=launch_sha256,
        evaluation_sha256=evaluation_sha256,
        manifest_sha256=manifest_sha256,
    ) is None:
        raise EvaluationError("AITA task receipt was not readable after publication")
    return True


def _promote_prior_attempts(
    *,
    condition_paths: ConditionPaths,
    condition: Condition,
    shard_index: int,
    launch_sha256: str,
    evaluation_sha256: str,
    manifest_sha256: str,
) -> bool:
    root = condition_paths.attempts / f"shard-{shard_index:03d}"
    if not root.exists():
        return False
    if root.is_symlink() or not root.is_dir():
        raise EvaluationError(f"AITA attempt root must be a regular directory: {root}")
    promoted = False
    for attempt in sorted(item for item in root.iterdir() if item.is_dir() and not item.is_symlink()):
        promoted = _promote_attempt(
            attempt=attempt,
            condition_paths=condition_paths,
            condition=condition,
            shard_index=shard_index,
            launch_sha256=launch_sha256,
            evaluation_sha256=evaluation_sha256,
            manifest_sha256=manifest_sha256,
        ) or promoted
    return promoted


def _task_command(
    *,
    python: str,
    manifest: Path,
    runtime: Mapping[str, Any],
    attempt: Path,
    shard_index: int,
    source_sample_count: int | None = None,
) -> list[str]:
    task_args = {"manifest": str(manifest), "shard_index": shard_index, "n_shards": SHARD_COUNT}
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "run_evals.py"),
        "--task-factory",
        TASK_FACTORY,
    ]
    if runtime.get("mode") == "native-hf-base":
        model = runtime.get("model")
        if not isinstance(model, str) or not model.startswith("hf/"):
            raise EvaluationError("base AITA runtime lacks model identity")
        command.extend(["--model", model])
    elif runtime.get("mode") == "native-hf-peft":
        checkpoint = runtime.get("checkpoint")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
            raise EvaluationError("trained AITA runtime lacks raw PEFT checkpoint")
        command.extend(["--local-checkpoint", checkpoint, "--base-model", str(MODEL_SNAPSHOT)])
    else:
        raise EvaluationError("AITA runtime mode is unsupported")
    model_args = runtime.get("model_args")
    expected_model_args = HF_MODEL_ARGS if runtime.get("mode") == "native-hf-base" else HF_LOCAL_MODEL_ARGS
    generation_config = runtime.get("generation_config")
    if not isinstance(model_args, Mapping) or dict(model_args) != expected_model_args:
        raise EvaluationError("AITA runtime must use the frozen native-HF model arguments")
    if not isinstance(generation_config, Mapping) or dict(generation_config) != RUNTIME_GENERATION_CONFIG:
        raise EvaluationError("AITA runtime must use the frozen effective HF generation configuration")
    if (
        runtime.get("qwen_thinking_policy") != QWEN_THINKING_POLICY
        or runtime.get("no_token_cap_policy") != NO_TOKEN_CAP_POLICY
        or runtime.get("no_token_cap_runtime_policy") != NO_TOKEN_CAP_RUNTIME_POLICY
    ):
        raise EvaluationError("AITA runtime lacks the frozen r006 launch thinking/no-token-cap policies")
    try:
        assert_no_token_cap_mapping(model_args, label="AITA task command model_args")
        assert_no_token_cap_mapping(generation_config, label="AITA task command generation_config")
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc
    if source_sample_count is not None and source_sample_count != DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT:
        raise EvaluationError("AITA direct-verdict smoke must use its fixed source sample count")
    command.extend(
        [
            "--task-args",
            json.dumps(task_args, sort_keys=True, separators=(",", ":")),
            "--model-args",
            json.dumps(dict(model_args), sort_keys=True, separators=(",", ":")),
            "--generation-config",
            json.dumps(dict(generation_config), sort_keys=True, separators=(",", ":")),
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
    _assert_no_token_cap_command(command, label="AITA task command")
    return command


def worker(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    training_repository: str | Path,
    condition_name: str,
    shard_index: int,
    python: str,
) -> dict[str, Any]:
    """Run one resumable condition/shard cell on exactly one visible GPU."""

    condition = _condition(condition_name)
    if shard_index < 0 or shard_index >= SHARD_COUNT:
        raise EvaluationError("AITA worker shard index must be in [0, 3]")
    contract, paths = _load_campaign(
        campaign_root=campaign_root,
        official_source_dir=official_source_dir,
        training_repository=training_repository,
    )
    receipt, evaluation_sha256 = _load_evaluation_receipt(contract, paths, condition)
    launch_sha256 = _sha256_file(paths.contract)
    manifest_record = contract.get("official_manifest")
    if not isinstance(manifest_record, Mapping) or not isinstance(manifest_record.get("manifest"), Mapping):
        raise EvaluationError("AITA campaign contract lacks official manifest identity")
    manifest_identity = manifest_record["manifest"]
    manifest_path = Path(str(manifest_identity.get("path", "")))
    manifest_sha256 = manifest_identity.get("sha256")
    if not isinstance(manifest_sha256, str) or not _valid_sha256(manifest_sha256):
        raise EvaluationError("AITA campaign manifest identity has an invalid SHA-256")
    if _identity(manifest_path, label="official AITA-NTA-FLIP manifest") != manifest_identity:
        raise EvaluationError("official AITA-NTA-FLIP manifest changed after campaign contract publication")
    _validate_official_manifest(manifest_path)
    condition_paths = _condition_paths(paths, condition)
    if _load_task_receipt(
        condition_paths=condition_paths,
        condition=condition,
        shard_index=shard_index,
        launch_sha256=launch_sha256,
        evaluation_sha256=evaluation_sha256,
        manifest_sha256=manifest_sha256,
    ) is not None:
        return {"condition": condition.name, "shard_index": shard_index, "status": "resumed", "promoted": False}
    promoted = _promote_prior_attempts(
        condition_paths=condition_paths,
        condition=condition,
        shard_index=shard_index,
        launch_sha256=launch_sha256,
        evaluation_sha256=evaluation_sha256,
        manifest_sha256=manifest_sha256,
    )
    if _load_task_receipt(
        condition_paths=condition_paths,
        condition=condition,
        shard_index=shard_index,
        launch_sha256=launch_sha256,
        evaluation_sha256=evaluation_sha256,
        manifest_sha256=manifest_sha256,
    ) is not None:
        return {"condition": condition.name, "shard_index": shard_index, "status": "promoted-prior", "promoted": promoted}
    _validate_evaluator_environment(require_one_gpu=True)
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise EvaluationError("AITA evaluation receipt lacks runtime")
    attempt = _next_attempt(condition_paths.attempts / f"shard-{shard_index:03d}", label="attempt")
    command = _task_command(
        python=python,
        manifest=manifest_path,
        runtime=runtime,
        attempt=attempt,
        shard_index=shard_index,
    )
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=os.environ.copy(), check=False)
    promoted_now = _promote_attempt(
        attempt=attempt,
        condition_paths=condition_paths,
        condition=condition,
        shard_index=shard_index,
        launch_sha256=launch_sha256,
        evaluation_sha256=evaluation_sha256,
        manifest_sha256=manifest_sha256,
    )
    if result.returncode:
        raise EvaluationError(
            f"{condition.name} shard-{shard_index} evaluator exited {result.returncode}; preserved attempt: {attempt}"
        )
    if not promoted_now:
        raise EvaluationError(f"{condition.name} shard-{shard_index} exited successfully without a promotable EvalLog")
    return {"condition": condition.name, "shard_index": shard_index, "status": "generated", "promoted": True}


def _direct_verdict_smoke_root(paths: ConditionPaths) -> Path:
    root = paths.root / "direct-verdict-smoke"
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise EvaluationError(f"AITA direct-verdict smoke root must be a regular directory: {root}")
    return root


def _direct_verdict_smoke_receipt_path(paths: ConditionPaths, *, shard_index: int) -> Path:
    if shard_index < 0 or shard_index >= SHARD_COUNT:
        raise EvaluationError("AITA direct-verdict smoke shard index is outside [0, 3]")
    return _direct_verdict_smoke_root(paths) / f"receipt-shard-{shard_index:03d}.json"


def _direct_verdict_smoke_log(
    path: Path,
    *,
    manifest_sha256: str,
    shard_index: int,
    expected_model: str,
    expected_runtime: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Require small pre-launch samples to be literal direct verdicts.

    This reads only the smoke EvalLog.  It never moves that evidence into the
    full-evaluation raw tree, so the source-sample check cannot contaminate
    pair coverage or cross-task scoring.
    """

    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - configured Isambard runtime
        raise EvaluationError("Inspect AI is required to validate AITA smoke EvalLogs") from exc
    _inspect_success(
        path,
        manifest_sha256=manifest_sha256,
        shard_index=shard_index,
        expected_model=expected_model,
        expected_runtime=expected_runtime,
    )
    try:
        log = read_eval_log(str(path), exclude_fields={"events", "messages", "store"})
    except Exception as exc:
        raise EvaluationError(f"could not read direct-verdict smoke EvalLog: {path}") from exc
    samples = getattr(log, "samples", None)
    if not isinstance(samples, list) or len(samples) != DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT:
        raise EvaluationError(
            "AITA direct-verdict smoke did not contain exactly "
            f"{DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT} source samples"
        )
    verdicts: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for sample in samples:
        sample_id = getattr(sample, "id", None)
        if not isinstance(sample_id, (str, int)):
            raise EvaluationError("AITA direct-verdict smoke has an invalid sample ID")
        sample_id_text = str(sample_id)
        if sample_id_text in seen_ids:
            raise EvaluationError("AITA direct-verdict smoke duplicated a sample ID")
        seen_ids.add(sample_id_text)
        sample_metadata = getattr(sample, "metadata", {})
        if not isinstance(sample_metadata, Mapping) or sample_metadata.get("shard_index") != shard_index:
            raise EvaluationError("AITA direct-verdict smoke sample has the wrong shard identity")
        output = getattr(sample, "output", None)
        completion = getattr(output, "completion", None)
        if not isinstance(completion, str) or completion.strip() not in {"YTA", "NTA"}:
            raise EvaluationError(
                "AITA direct-verdict smoke requires every response to be a literal YTA or NTA after whitespace"
            )
        output_metadata = getattr(output, "metadata", {})
        if not isinstance(output_metadata, Mapping) or (
            output_metadata.get("r005_no_token_cap") is not True
            or output_metadata.get("r005_termination") != "model_eos_only"
            or output_metadata.get("qwen_enable_thinking") is not False
        ):
            raise EvaluationError("AITA direct-verdict smoke lacks EOS-only/no-thinking output attestation")
        verdicts.append(
            {
                "sample_id": sample_id_text,
                "verdict": completion.strip(),
                "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
            }
        )
    return verdicts


def _read_direct_verdict_smoke_receipt(
    *,
    paths: ConditionPaths,
    condition: Condition,
    shard_index: int,
    launch_sha256: str,
    evaluation_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any] | None:
    receipt_path = _direct_verdict_smoke_receipt_path(paths, shard_index=shard_index)
    if not receipt_path.exists() and not receipt_path.is_symlink():
        return None
    receipt = _read_json(receipt_path, label=f"{condition.name} direct-verdict smoke receipt")
    required = {
        "schema",
        "condition",
        "shard_index",
        "source_sample_count",
        "output_contract",
        "launch_contract_sha256",
        "evaluation_receipt_sha256",
        "official_manifest_sha256",
        "runtime",
        "command",
        "smoke_log",
        "verdicts",
    }
    if set(receipt) != required or receipt.get("schema") != SMOKE_RECEIPT_SCHEMA:
        raise EvaluationError(f"{condition.name} direct-verdict smoke receipt has an unsupported schema")
    if (
        receipt.get("condition") != condition.name
        or receipt.get("shard_index") != shard_index
        or receipt.get("source_sample_count") != DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT
        or receipt.get("output_contract") != "stripped-response-exactly-YTA-or-NTA"
        or receipt.get("launch_contract_sha256") != launch_sha256
        or receipt.get("evaluation_receipt_sha256") != evaluation_sha256
        or receipt.get("official_manifest_sha256") != manifest_sha256
        or receipt.get("runtime") != _task_runtime_binding(paths)
    ):
        raise EvaluationError(f"{condition.name} direct-verdict smoke receipt binds different campaign evidence")
    command = receipt.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise EvaluationError(f"{condition.name} direct-verdict smoke receipt has an invalid command")
    if command.count("--limit") != 1:
        raise EvaluationError(f"{condition.name} direct-verdict smoke receipt lacks its fixed source selection")
    limit_index = command.index("--limit")
    if limit_index + 1 >= len(command) or command[limit_index + 1] != str(DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT):
        raise EvaluationError(f"{condition.name} direct-verdict smoke receipt has the wrong source selection")
    try:
        _assert_no_token_cap_command(command, label="AITA direct-verdict smoke command")
    except (TypeError, ValueError) as exc:  # ValueError retained for custom JSON implementations.
        raise EvaluationError(str(exc)) from exc
    smoke_log = receipt.get("smoke_log")
    if not isinstance(smoke_log, Mapping):
        raise EvaluationError(f"{condition.name} direct-verdict smoke receipt lacks log identity")
    log_path = _under_root(
        Path(str(smoke_log.get("path", ""))),
        _direct_verdict_smoke_root(paths),
        label="AITA direct-verdict smoke log",
    )
    if smoke_log != _identity(log_path, label="AITA direct-verdict smoke log"):
        raise EvaluationError(f"{condition.name} direct-verdict smoke log changed after publication")
    observed = _direct_verdict_smoke_log(
        log_path,
        manifest_sha256=manifest_sha256,
        shard_index=shard_index,
        expected_model=_expected_model_from_condition_receipt(paths),
        expected_runtime=_expected_runtime_from_condition_receipt(paths),
    )
    if receipt.get("verdicts") != observed:
        raise EvaluationError(f"{condition.name} direct-verdict smoke receipt differs from its EvalLog")
    return receipt


def smoke(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    training_repository: str | Path,
    condition_name: str,
    shard_index: int,
    python: str,
) -> dict[str, Any]:
    """Run a tiny EOS-only direct-verdict check before the full 16-GPU step."""

    condition = _condition(condition_name)
    if shard_index < 0 or shard_index >= SHARD_COUNT:
        raise EvaluationError("AITA smoke shard index must be in [0, 3]")
    contract, paths = _load_campaign(
        campaign_root=campaign_root,
        official_source_dir=official_source_dir,
        training_repository=training_repository,
    )
    receipt, evaluation_sha256 = _load_evaluation_receipt(contract, paths, condition)
    launch_sha256 = _sha256_file(paths.contract)
    manifest_identity = contract.get("official_manifest", {}).get("manifest") if isinstance(contract.get("official_manifest"), Mapping) else None
    if not isinstance(manifest_identity, Mapping):
        raise EvaluationError("AITA campaign contract lacks official manifest custody")
    manifest_path = Path(str(manifest_identity.get("path", "")))
    manifest_sha256 = manifest_identity.get("sha256")
    if not isinstance(manifest_sha256, str) or not _valid_sha256(manifest_sha256):
        raise EvaluationError("AITA campaign manifest identity has an invalid SHA-256")
    if _identity(manifest_path, label="official AITA-NTA-FLIP manifest") != dict(manifest_identity):
        raise EvaluationError("official AITA-NTA-FLIP manifest changed before runtime smoke")
    _validate_official_manifest(manifest_path)
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise EvaluationError("AITA condition receipt lacks runtime for smoke")
    condition_paths = _condition_paths(paths, condition)
    existing = _read_direct_verdict_smoke_receipt(
        paths=condition_paths,
        condition=condition,
        shard_index=shard_index,
        launch_sha256=launch_sha256,
        evaluation_sha256=evaluation_sha256,
        manifest_sha256=manifest_sha256,
    )
    if existing is not None:
        return {
            "status": "resumed",
            "condition": condition.name,
            "shard_index": shard_index,
            "smoke_receipt": _identity(
                _direct_verdict_smoke_receipt_path(condition_paths, shard_index=shard_index),
                label="AITA direct-verdict smoke receipt",
            ),
        }
    _validate_evaluator_environment(require_one_gpu=True)
    smoke_root = _direct_verdict_smoke_root(condition_paths)
    attempt = _next_attempt(smoke_root / "attempts", label=f"shard-{shard_index:03d}")
    command = _task_command(
        python=python,
        manifest=manifest_path,
        runtime=runtime,
        attempt=attempt,
        shard_index=shard_index,
        source_sample_count=DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT,
    )
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=os.environ.copy(), check=False)
    all_candidates = sorted(attempt.rglob("*.eval"))
    if any(candidate.is_symlink() or not candidate.is_file() for candidate in all_candidates):
        raise EvaluationError(f"{condition.name} direct-verdict smoke attempt contains a linked/non-file EvalLog")
    candidates = all_candidates
    if result.returncode:
        raise EvaluationError(
            f"{condition.name} direct-verdict smoke exited {result.returncode}; preserved attempt: {attempt}"
        )
    if len(candidates) != 1:
        raise EvaluationError(f"{condition.name} direct-verdict smoke produced {len(candidates)} EvalLogs, expected one")
    log_path = candidates[0]
    verdicts = _direct_verdict_smoke_log(
        log_path,
        manifest_sha256=manifest_sha256,
        shard_index=shard_index,
        expected_model=_expected_model_from_condition_receipt(condition_paths),
        expected_runtime=_expected_runtime_from_condition_receipt(condition_paths),
    )
    smoke_receipt = {
        "schema": SMOKE_RECEIPT_SCHEMA,
        "condition": condition.name,
        "shard_index": shard_index,
        "source_sample_count": DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT,
        "output_contract": "stripped-response-exactly-YTA-or-NTA",
        "launch_contract_sha256": launch_sha256,
        "evaluation_receipt_sha256": evaluation_sha256,
        "official_manifest_sha256": manifest_sha256,
        "runtime": _task_runtime_binding(condition_paths),
        "command": command,
        "smoke_log": _identity(log_path, label="AITA direct-verdict smoke log"),
        "verdicts": verdicts,
    }
    status = _write_immutable_json(
        _direct_verdict_smoke_receipt_path(condition_paths, shard_index=shard_index),
        smoke_receipt,
        label=f"{condition.name} direct-verdict smoke receipt",
    )
    return {
        "status": status,
        "condition": condition.name,
        "shard_index": shard_index,
        "smoke_receipt": _identity(
            _direct_verdict_smoke_receipt_path(condition_paths, shard_index=shard_index),
            label="AITA direct-verdict smoke receipt",
        ),
        "verdicts": verdicts,
    }


def _expected_runtime_for_preflight(receipt: Mapping[str, Any]) -> dict[str, Any]:
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise EvaluationError("AITA evaluation receipt lacks runtime for preflight")
    frozen = dict(runtime)
    # The core validator deliberately consumes only native Inspect header
    # fields.  Supply the complete condition-specific subset it can observe;
    # raw checkpoint custody remains bound in this immutable receipt and in
    # each task receipt's explicit runtime binding.
    model = frozen.get("model")
    model_args = frozen.get("model_args")
    generation_config = frozen.get("generation_config")
    metadata = frozen.get("metadata")
    expected_model_args = HF_MODEL_ARGS if frozen.get("mode") == "native-hf-base" else HF_LOCAL_MODEL_ARGS
    if (
        not isinstance(model, str)
        or not isinstance(model_args, Mapping)
        or dict(model_args) != expected_model_args
        or not isinstance(generation_config, Mapping)
        or dict(generation_config) != RUNTIME_GENERATION_CONFIG
        or frozen.get("qwen_thinking_policy") != QWEN_THINKING_POLICY
        or frozen.get("no_token_cap_policy") != NO_TOKEN_CAP_POLICY
        or frozen.get("no_token_cap_runtime_policy") != NO_TOKEN_CAP_RUNTIME_POLICY
        or frozen.get("sampling_config") != GENERATION_CONFIG
        or frozen.get("concurrency_config") != CONCURRENCY_CONFIG
        or frozen.get("model_snapshot") != _snapshot_identity()
        or frozen.get("evaluator") != _evaluator_runtime()
        or not isinstance(metadata, Mapping)
    ):
        raise EvaluationError("AITA evaluation receipt has no valid native-HF preflight runtime")
    try:
        assert_no_token_cap_mapping(model_args, label="AITA preflight model_args")
        assert_no_token_cap_mapping(generation_config, label="AITA preflight generation_config")
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc
    expected_metadata = dict(metadata)
    if frozen.get("mode") == "native-hf-peft":
        expected_metadata.update(
            {
                "checkpoint": frozen.get("checkpoint"),
                "checkpoint_backend": "local",
                "base_model": str(MODEL_SNAPSHOT),
            }
        )
    elif frozen.get("mode") == "native-hf-base":
        expected_metadata["model"] = model
    else:
        raise EvaluationError("AITA evaluation receipt has an unsupported native-HF mode")
    return {
        "model": model,
        "model_args": dict(model_args),
        "generation_config": dict(generation_config),
        "metadata": expected_metadata,
    }


def _preflight_headline(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract the final-answer-only primary and strict secondary diagnostic.

    The core report already recomputes both metrics from exact pair records.
    This launcher only names and carries their immutable descriptors forward;
    it never performs a second parser or substitutes a stricter score for the
    final-answer-only headline.  The paper's first-five calculation remains a
    separately named compatibility-only metric and is never used as the r006
    headline.
    """

    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise EvaluationError("AITA preflight report lacks metrics")
    final_answer_only = metrics.get("final_answer_only")
    paper = metrics.get("paper_first_five")
    strict = metrics.get("strict")
    primary = metrics.get("primary_metric")
    secondary = metrics.get("secondary_metric")
    expected_primary = {
        "name": "final_answer_only_both_nta",
        "label": "final-answer-only NTA/NTA (modified protocol; invalid finals retained)",
        "direction": "lower_is_better",
        "parser_schema": FINAL_ANSWER_PARSER_SCHEMA,
    }
    expected_secondary = {
        "name": "strict_mutually_exclusive_nta_nta",
        "label": "legacy whole-response mutually-exclusive NTA/NTA diagnostic",
        "direction": "lower_is_better",
        "parser_schema": FINAL_ANSWER_PARSER_SCHEMA,
    }
    final_outcomes = final_answer_only.get("outcomes") if isinstance(final_answer_only, Mapping) else None
    conditional = (
        final_answer_only.get("conditional_outcomes_when_both_final_parsed")
        if isinstance(final_answer_only, Mapping)
        else None
    )
    conditional_outcomes = conditional.get("outcomes") if isinstance(conditional, Mapping) else None
    primary_response_coverage = primary.get("parsed_response_coverage") if isinstance(primary, Mapping) else None
    primary_pair_coverage = primary.get("parsed_pair_coverage") if isinstance(primary, Mapping) else None
    primary_conditional = (
        primary.get("conditional_value_when_both_final_parsed") if isinstance(primary, Mapping) else None
    )
    if (
        not isinstance(final_answer_only, Mapping)
        or not isinstance(paper, Mapping)
        or not isinstance(strict, Mapping)
        or not isinstance(primary, Mapping)
        or not isinstance(secondary, Mapping)
        or isinstance(metrics.get("denominator_pairs"), bool)
        or not isinstance(metrics.get("denominator_pairs"), int)
        or metrics["denominator_pairs"] < 1
        or not isinstance(final_outcomes, Mapping)
        or not isinstance(conditional_outcomes, Mapping)
        or not isinstance(primary_response_coverage, Mapping)
        or not isinstance(primary_pair_coverage, Mapping)
        or not isinstance(primary_conditional, Mapping)
        or {key: primary.get(key) for key in expected_primary} != expected_primary
        or {key: secondary.get(key) for key in expected_secondary} != expected_secondary
        or paper.get("role") != "compatibility_only"
        or final_answer_only.get("parser_schema") != FINAL_ANSWER_PARSER_SCHEMA
        or strict.get("parser_schema") != FINAL_ANSWER_PARSER_SCHEMA
        or conditional.get("denominator_pairs") != primary_pair_coverage.get("count")
        or primary.get("value") != final_outcomes.get("nta_nta")
        or primary_response_coverage != final_answer_only.get("parsed_response_coverage")
        or primary_pair_coverage != final_answer_only.get("parsed_pair_coverage")
        or primary_conditional != conditional_outcomes.get("nta_nta")
        or secondary.get("value") != strict.get("strict_nta_nta")
    ):
        raise EvaluationError("AITA preflight report does not expose the frozen final-answer-primary metric contract")

    denominator_pairs = metrics["denominator_pairs"]
    assert isinstance(denominator_pairs, int) and not isinstance(denominator_pairs, bool)

    def require_count_rate(
        value: Any,
        *,
        label: str,
        denominator: int,
        includes_denominator: bool,
        allow_undefined_rate: bool = False,
    ) -> None:
        expected_keys = {"count", "rate"}
        if includes_denominator:
            expected_keys.add("denominator")
        if (
            not isinstance(value, Mapping)
            or set(value) != expected_keys
            or isinstance(value.get("count"), bool)
            or not isinstance(value.get("count"), int)
            or value["count"] < 0
        ):
            raise EvaluationError(f"AITA {label} metric has an invalid count/rate schema")
        if includes_denominator and value.get("denominator") != denominator:
            raise EvaluationError(f"AITA {label} metric has a different denominator")
        if value["count"] > denominator:
            raise EvaluationError(f"AITA {label} metric count exceeds its denominator")
        rate = value["rate"]
        if denominator == 0:
            if not allow_undefined_rate or rate is not None:
                raise EvaluationError(f"AITA {label} metric has an invalid zero-denominator rate")
            return
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not 0 <= float(rate) <= 1
            or float(rate) != value["count"] / denominator
        ):
            raise EvaluationError(f"AITA {label} metric has an invalid count/rate")

    require_count_rate(
        primary["value"],
        label="final-answer-primary",
        denominator=denominator_pairs,
        includes_denominator=False,
    )
    require_count_rate(
        primary["parsed_response_coverage"],
        label="final-answer-response-coverage",
        denominator=denominator_pairs * PERSPECTIVES_PER_PAIR,
        includes_denominator=True,
    )
    require_count_rate(
        primary["parsed_pair_coverage"],
        label="final-answer-pair-coverage",
        denominator=denominator_pairs,
        includes_denominator=True,
    )
    conditional_denominator = conditional["denominator_pairs"]
    if isinstance(conditional_denominator, bool) or not isinstance(conditional_denominator, int) or conditional_denominator < 0:
        raise EvaluationError("AITA final-answer conditional metric has an invalid denominator")
    require_count_rate(
        primary["conditional_value_when_both_final_parsed"],
        label="final-answer-conditional",
        denominator=conditional_denominator,
        includes_denominator=True,
        allow_undefined_rate=True,
    )
    require_count_rate(
        secondary["value"],
        label="strict-secondary",
        denominator=denominator_pairs,
        includes_denominator=False,
    )
    return {"primary_metric": dict(primary), "secondary_metric": dict(secondary)}


def finalize(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    training_repository: str | Path,
) -> dict[str, Any]:
    """Seal all four conditions only after pair-coverage preflight succeeds."""

    contract, paths = _load_campaign(
        campaign_root=campaign_root,
        official_source_dir=official_source_dir,
        training_repository=training_repository,
    )
    manifest_identity = contract["official_manifest"]["manifest"]
    manifest_path = Path(manifest_identity["path"])
    manifest_sha256 = manifest_identity["sha256"]
    if _identity(manifest_path, label="official AITA-NTA-FLIP manifest") != manifest_identity:
        raise EvaluationError("official AITA-NTA-FLIP manifest changed before finalization")
    _validate_official_manifest(manifest_path)
    launch_sha256 = _sha256_file(paths.contract)
    preflight_module = _core_preflight_module()
    conditions: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_paths = _condition_paths(paths, condition)
        evaluation, evaluation_sha256 = _load_evaluation_receipt(contract, paths, condition)
        sealed_receipts = _validate_canonical_raw_custody(
            condition_paths=condition_paths,
            condition=condition,
            launch_sha256=launch_sha256,
            evaluation_sha256=evaluation_sha256,
            manifest_sha256=manifest_sha256,
        )
        task_receipts = [
            _identity(_task_receipt_path(condition_paths, shard_index), label="AITA shard receipt")
            for shard_index in range(SHARD_COUNT)
        ]
        expected_runtime = _expected_runtime_for_preflight(evaluation)
        try:
            report = preflight_module.preflight_raw_logs(
                raw_root=condition_paths.raw,
                manifest=manifest_path,
                expected_runtime=expected_runtime,
            )
        except Exception as exc:
            raise EvaluationError(f"{condition.name} AITA pair-coverage preflight failed: {exc}") from exc
        if not isinstance(report, Mapping):
            raise EvaluationError(f"{condition.name} AITA pair-coverage preflight returned no object")
        try:
            validated_report = preflight_module.validate_preflight_report(
                dict(report),
                manifest=manifest_path,
                raw_root=condition_paths.raw,
                expected_runtime=expected_runtime,
            )
            written = preflight_module.write_preflight_report(
                dict(validated_report),
                condition_paths.preflight,
                manifest=manifest_path,
                raw_root=condition_paths.raw,
                expected_runtime=expected_runtime,
            )
        except Exception as exc:
            raise EvaluationError(f"{condition.name} AITA preflight could not be published immutably: {exc}") from exc
        if Path(written).resolve() != condition_paths.preflight.resolve():
            raise EvaluationError(f"{condition.name} AITA preflight writer returned an unexpected path")
        stored_report = _read_json(condition_paths.preflight, label="AITA pair-coverage preflight")
        try:
            replayed_report = preflight_module.validate_preflight_report(
                stored_report,
                manifest=manifest_path,
                raw_root=condition_paths.raw,
                expected_runtime=expected_runtime,
            )
        except Exception as exc:
            raise EvaluationError(f"{condition.name} AITA stored preflight report failed custody replay: {exc}") from exc
        if dict(replayed_report) != dict(validated_report) or dict(stored_report) != dict(validated_report):
            raise EvaluationError(f"{condition.name} AITA stored preflight report differs from its exact replay")
        headline = _preflight_headline(validated_report)
        total_generations = sum(int(receipt["generation_count"]) for receipt in sealed_receipts)
        if total_generations != GENERATIONS_PER_CONDITION:
            raise EvaluationError(
                f"{condition.name} AITA receipts contain {total_generations} generations, expected {GENERATIONS_PER_CONDITION}"
            )
        conditions.append(
            {
                "name": condition.name,
                "evaluation_receipt": _identity(condition_paths.evaluation_receipt, label="AITA evaluation receipt"),
                "task_receipts": task_receipts,
                "preflight": _identity(condition_paths.preflight, label="AITA pair-coverage preflight"),
                "headline": headline,
                "generations": total_generations,
            }
        )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "launch_contract": _identity(paths.contract, label="AITA campaign launch contract"),
        "official_manifest": manifest_identity,
        "benchmark": contract["benchmark"],
        "conditions": conditions,
    }
    status = _write_immutable_json(paths.completion, completion, label="AITA campaign completion receipt")
    return {
        "completion": _identity(paths.completion, label="AITA campaign completion receipt"),
        "status": status,
        "primary_metrics": [
            {"condition": item["name"], **item["headline"]["primary_metric"]}
            for item in conditions
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def campaign_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--campaign-root", required=True, type=Path)
        command.add_argument("--official-source-dir", required=True, type=Path)
        command.add_argument("--training-repository", required=True, type=Path)

    topology_probe_parser = commands.add_parser(
        "probe-gpu-topology",
        help="write one rank record for the bounded 16-rank r006 GPU-topology probe",
    )
    topology_probe_parser.add_argument("--campaign-root", required=True, type=Path)
    topology_seal_parser = commands.add_parser(
        "seal-gpu-topology-probe",
        help="validate and immutably seal the 16-rank r006 GPU-topology receipt",
    )
    topology_seal_parser.add_argument("--campaign-root", required=True, type=Path)

    prepare_parser = commands.add_parser("prepare", help="replay sealed sources and write campaign receipts")
    campaign_inputs(prepare_parser)
    prepare_parser.add_argument("--yes", action="store_true")

    worker_parser = commands.add_parser("worker", help="run one condition/shard on one Slurm-visible GPU")
    campaign_inputs(worker_parser)
    worker_parser.add_argument("--condition", choices=[condition.name for condition in CONDITIONS], required=True)
    worker_parser.add_argument("--shard-index", type=int, required=True)
    worker_parser.add_argument("--python", required=True)

    smoke_parser = commands.add_parser("smoke", help="run a small direct-verdict check before full evaluation")
    campaign_inputs(smoke_parser)
    smoke_parser.add_argument("--condition", choices=[condition.name for condition in CONDITIONS], required=True)
    smoke_parser.add_argument("--shard-index", type=int, required=True)
    smoke_parser.add_argument("--python", required=True)

    finalize_parser = commands.add_parser("finalize", help="preflight and seal all 16 completed cells")
    campaign_inputs(finalize_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    values: dict[str, Any] = {}
    if args.command not in {"probe-gpu-topology", "seal-gpu-topology-probe"}:
        values = {
            "campaign_root": args.campaign_root,
            "official_source_dir": args.official_source_dir,
            "training_repository": args.training_repository,
        }
    try:
        if args.command == "probe-gpu-topology":
            result = gpu_topology_probe(campaign_root=args.campaign_root)
        elif args.command == "seal-gpu-topology-probe":
            result = seal_gpu_topology_probe(campaign_root=args.campaign_root)
        elif args.command == "prepare":
            if not args.yes:
                parser.error("prepare requires --yes after reviewing the sealed source roots")
            result = prepare(**values)
        elif args.command == "worker":
            result = worker(
                **values,
                condition_name=args.condition,
                shard_index=args.shard_index,
                python=args.python,
            )
        elif args.command == "smoke":
            result = smoke(
                **values,
                condition_name=args.condition,
                shard_index=args.shard_index,
                python=args.python,
            )
        elif args.command == "finalize":
            result = finalize(**values)
        else:  # pragma: no cover - argparse guarantees a valid command
            parser.error("unsupported command")
            return 2
    except (EvaluationError, FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
