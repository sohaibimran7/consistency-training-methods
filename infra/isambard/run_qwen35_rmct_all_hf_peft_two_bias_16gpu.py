#!/usr/bin/env python3
"""Run a matched native-HF/PEFT two-bias campaign for base and RMCT steps.

This is an intentionally new custody boundary.  It evaluates the pinned base
snapshot plus the sealed optimizer-step 16, 64, and 176 PEFT checkpoints with
the same frozen Stage-2 two-bias matrix:

* 14 IID cells are limited to the first 50 frozen question IDs;
* 7 HLE cells retain their full 100-question population; and
* each condition therefore contributes exactly 1,400 generations.

The accompanying sbatch wrapper owns one 4-node / 16-GPU allocation.  It
first generates all twelve clean cells in one wave, seals a clean-gate receipt
for each condition, and only then admits the 72 biased cells in five balanced
global waves.  This deliberately avoids asking the installed switch scorer to
wait for a clean EvalLog and so avoids its frozen 3,600-second polling timeout
race.

No scheduler client is called here.  Failed or incomplete attempts are
preserved; this module never retries, deletes, overwrites a differing result,
or changes a model/training parameter.  A later operator invocation can resume
only through the same immutable contract and receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LAUNCH_SCHEMA = "rmct-all-hf-peft-two-bias-16gpu-launch-v2"
RUNTIME_RECEIPT_SCHEMA = "rmct-all-hf-peft-two-bias-16gpu-runtime-v2"
EVALUATION_RECEIPT_SCHEMA = "rmct-all-hf-peft-two-bias-16gpu-evaluation-v2"
TASK_RECEIPT_SCHEMA = "rmct-all-hf-peft-two-bias-16gpu-task-receipt-v2"
CLEAN_GATE_RECEIPT_SCHEMA = "rmct-all-hf-peft-two-bias-16gpu-clean-gate-v2"
PREFLIGHT_SCHEMA = "rmct-all-hf-peft-two-bias-16gpu-preflight-v2"
COMPLETION_SCHEMA = "rmct-all-hf-peft-two-bias-16gpu-completion-v2"

CAMPAIGN_NAME = "rmct-convergence-all-hf-peft-two-bias-16gpu-v1-r002"
TRAINING_CONDITION = "rmct-convergence"
MODEL_SNAPSHOT = Path(
    "/lus/lfs1aip2/scratch/a5v/sohaib.a5v/ctm/huggingface/hub/"
    "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
INSPECT_VERSION = "0.3.258"
EVALUATOR_PACKAGE_VERSIONS = {
    "inspect-ai": INSPECT_VERSION,
    "torch": "2.11.0+cu129",
    "transformers": "5.5.4",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
}
TASK_FACTORY = "experiments.stage2_ood_hle.tasks:ood_tasks"
TASK_COUNT = 21
CLEAN_TASK_INDICES = (1, 2, 3)
BIASED_TASK_INDICES = tuple(range(4, TASK_COUNT + 1))
IID_CLEAN_TASK_INDICES = (1, 2)
HLE_CLEAN_TASK_INDICES = (3,)
IID_BIASED_TASK_INDICES = (4, 5, *range(7, 17))
HLE_BIASED_TASK_INDICES = (6, *range(17, 22))
IID_TASK_INDICES = (*IID_CLEAN_TASK_INDICES, *IID_BIASED_TASK_INDICES)
HLE_TASK_INDICES = (*HLE_CLEAN_TASK_INDICES, *HLE_BIASED_TASK_INDICES)
TASK_SAMPLE_COUNTS = {
    **{task_index: 50 for task_index in IID_TASK_INDICES},
    **{task_index: 100 for task_index in HLE_TASK_INDICES},
}
TOTAL_SAMPLES_PER_CONDITION = sum(TASK_SAMPLE_COUNTS.values())
SEEN_BIASES = ("wrong_argument", "suggested_answer")
HELD_OUT_BIASES = ("distractor_fact", "post_hoc", "spurious_few_shot_squares", "wrong_few_shot")
ALL_BIASES = (*SEEN_BIASES, *HELD_OUT_BIASES)

# This deliberately has no seed.  The frozen Stage-2 protocol sampled at
# temperature 1.0 without adding a new seed post hoc.  Recording an invented
# seed now would create a different comparison protocol.
GENERATION_CONFIG = {
    "max_tokens": 20480,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "max_connections": 8,
}
# ``get_model("hf/...")`` selects Inspect's HF provider from the model
# spelling, while the LocalBackend PEFT bridge needs an explicit provider.
# They are the same native-HF backend and decode controls, but keeping this
# API-level distinction explicit avoids passing an unsupported ``provider``
# keyword to the direct base-model constructor.
HF_MODEL_ARGS = {"device": "cuda:0", "dtype": "bfloat16"}
HF_LOCAL_MODEL_ARGS = {"provider": "hf", **HF_MODEL_ARGS}

# These are the native-HF model/tokenizer assets consumed by every condition.
# Decode is supplied explicitly in ``GENERATION_CONFIG``, so a snapshot may
# legitimately omit ``generation_config.json``; when present it is still
# custody-bound.  Processor assets can alter multimodal serialization and are
# likewise bound whenever the pinned revision exposes them.
SNAPSHOT_REQUIRED_CONFIGURATION_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
SNAPSHOT_OPTIONAL_CONFIGURATION_FILES = (
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "image_processor_config.json",
    "image_preprocessor_config.json",
    "video_processor_config.json",
    "video_preprocessor_config.json",
    "audio_processor_config.json",
    "audio_preprocessor_config.json",
    "feature_extractor_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)

# Run the twelve clean cells in a single 16-rank wave.  The final four ranks
# are intentionally idle: making duplicate clean generations to occupy them
# would corrupt the one-clean-reference protocol.
CLEAN_WAVE: dict[int, tuple[str, int]] = {
    0: ("base", 1),
    1: ("base", 2),
    2: ("base", 3),
    3: ("step016", 1),
    4: ("step016", 2),
    5: ("step016", 3),
    6: ("step064", 1),
    7: ("step064", 2),
    8: ("step064", 3),
    9: ("step176", 1),
    10: ("step176", 2),
    11: ("step176", 3),
}

# Seventy-two biased cells are spread over five 16-rank waves (16, 16, 16,
# 16, 8 cells).  Across all waves every rank receives exactly 300 source
# samples: ranks 0..7 run one HLE and four IID cells; ranks 8..15 run two HLE
# and two IID cells.  That is the balanced solution for 24 HLE×100 plus
# 48 IID×50 frozen cells.  A condition/task pair may occur only once.
BIASED_WAVES: dict[int, dict[int, tuple[str, int]]] = {
    1: {
        0: ("base", 6), 1: ("base", 17), 2: ("step016", 6), 3: ("step016", 17),
        4: ("step064", 6), 5: ("step064", 17), 6: ("step176", 6), 7: ("step176", 17),
        8: ("base", 18), 9: ("base", 19), 10: ("base", 20), 11: ("base", 21),
        12: ("step064", 18), 13: ("step064", 19), 14: ("step064", 20), 15: ("step064", 21),
    },
    2: {
        0: ("base", 4), 1: ("base", 5), 2: ("step016", 4), 3: ("step016", 5),
        4: ("step064", 4), 5: ("step064", 5), 6: ("step176", 4), 7: ("step176", 5),
        8: ("step016", 18), 9: ("step016", 19), 10: ("step016", 20), 11: ("step016", 21),
        12: ("step176", 18), 13: ("step176", 19), 14: ("step176", 20), 15: ("step176", 21),
    },
    3: {
        0: ("base", 7), 1: ("base", 8), 2: ("step016", 7), 3: ("step016", 8),
        4: ("step064", 7), 5: ("step064", 8), 6: ("step176", 7), 7: ("step176", 8),
        8: ("base", 13), 9: ("base", 15), 10: ("step016", 13), 11: ("step016", 15),
        12: ("step064", 13), 13: ("step064", 15), 14: ("step176", 13), 15: ("step176", 15),
    },
    4: {
        0: ("base", 9), 1: ("base", 10), 2: ("step016", 9), 3: ("step016", 10),
        4: ("step064", 9), 5: ("step064", 10), 6: ("step176", 9), 7: ("step176", 10),
        8: ("base", 14), 9: ("base", 16), 10: ("step016", 14), 11: ("step016", 16),
        12: ("step064", 14), 13: ("step064", 16), 14: ("step176", 14), 15: ("step176", 16),
    },
    5: {
        0: ("base", 11), 1: ("base", 12), 2: ("step016", 11), 3: ("step016", 12),
        4: ("step064", 11), 5: ("step064", 12), 6: ("step176", 11), 7: ("step176", 12),
    },
}

CRITICAL_SOURCES = (
    "infra/isambard/run_qwen35_rmct_all_hf_peft_two_bias_16gpu.py",
    "infra/isambard/run_qwen35_rmct_all_hf_peft_two_bias_16gpu.sbatch",
    "infra/isambard/run_qwen35_rmct_all_hf_peft_two_bias_16gpu_worker.sh",
    "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.py",
    "infra/isambard/run_qwen35_rmct_convergence_r4_two_bias_evals.py",
    "infra/isambard/verify_rmct_convergence_r4_recovery_production_ready.py",
    "scripts/run_evals.py",
    "ctm/evals/runner.py",
    "ctm/evals/local_model.py",
    "ctm/training/resume_state.py",
    "experiments/rmct_convergence/controller.py",
    "experiments/stage2_ood_hle/tasks.py",
    "experiments/stage2_ood_hle/materialize.py",
    "experiments/stage2_ood_hle/prepare.py",
    "experiments/stage2_ood_hle/raw_preflight.py",
    "experiments/rmct_two_bias_eval/contract.py",
    "experiments/rmct_two_bias_eval/deployment.py",
    "experiments/rmct_two_bias_eval/raw_preflight.py",
)


class EvaluationError(ValueError):
    """A campaign input, custody receipt, runtime, or raw cell is unsafe."""


@dataclass(frozen=True)
class Condition:
    name: str
    label: str
    optimizer_step: int | None
    source_kind: str
    artifact_name: str


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        "base",
        "base",
        None,
        "pinned-base-snapshot",
        "rmct-convergence-base-two-bias-hf-v1-r002",
    ),
    Condition(
        "step016",
        "step-016",
        16,
        "raw-training-checkpoint",
        "rmct-convergence-step016-two-bias-hf-v1-r002",
    ),
    Condition(
        "step064",
        "step-064",
        64,
        "raw-training-checkpoint",
        "rmct-convergence-step064-two-bias-hf-v1-r002",
    ),
    Condition(
        "step176",
        "step-176",
        176,
        "raw-training-checkpoint",
        "rmct-convergence-step176-two-bias-hf-v1-r002",
    ),
)
_CONDITION_BY_NAME = {condition.name: condition for condition in CONDITIONS}


@dataclass(frozen=True)
class CampaignPaths:
    root: Path
    input: Path
    deployment_manifest: Path
    contract: Path
    conditions: Path
    completion: Path


@dataclass(frozen=True)
class ConditionPaths:
    root: Path
    runtime_receipt: Path
    evaluation_receipt: Path
    raw: Path
    attempts: Path
    receipts: Path
    clean_gate: Path
    preflight: Path


def _condition(name: str) -> Condition:
    try:
        return _CONDITION_BY_NAME[name]
    except KeyError as exc:
        raise EvaluationError(f"unknown matched-HF condition {name!r}") from exc


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(item in "0123456789abcdef" for item in value)


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
    """Atomically publish a receipt, accepting only byte-identical resume."""

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


def _copy_immutable_file(source: Path, destination: Path, *, label: str) -> None:
    """Copy one EvalLog with write-once publication; never replace evidence."""

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
        deployment_manifest=root / "input" / "stage2-deployment-manifest.json",
        contract=root / "launch-contract.json",
        conditions=root / "conditions",
        completion=root / "completion.json",
    )


def _condition_paths(paths: CampaignPaths, condition: Condition) -> ConditionPaths:
    root = paths.conditions / condition.artifact_name
    return ConditionPaths(
        root=root,
        runtime_receipt=root / "runtime" / "runtime-receipt.json",
        evaluation_receipt=root / "runtime" / "evaluation-receipt.json",
        raw=root / "stage2" / "paired-clean-ready",
        attempts=root / "stage2" / "attempts",
        receipts=root / "stage2" / "receipts",
        clean_gate=root / "stage2" / "paired-clean-ready" / "clean-gate-receipt.json",
        preflight=root / "stage2" / "preflight" / "native-hf-mixed-two-bias.json",
    )


def _critical_source_identities() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in CRITICAL_SOURCES:
        path = _under_root(PROJECT_ROOT / relative, PROJECT_ROOT, label=f"critical source {relative}")
        records[relative] = _identity(path, label=f"critical source {relative}")
    return records


def _snapshot_logical_identity(snapshot: Path, logical: Path, *, label: str) -> dict[str, Any]:
    """Hash one snapshot file while allowing Hugging Face's internal blobs."""

    if not logical.exists():
        raise EvaluationError(f"pinned Qwen3.5 snapshot lacks {label}: {logical}")
    if logical.is_symlink():
        blobs = snapshot.parent.parent / "blobs"
        if blobs.is_symlink() or not blobs.is_dir():
            raise EvaluationError("pinned Qwen3.5 snapshot has no regular Hugging Face blob tree")
        resolved = logical.resolve()
        if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size < 1:
            raise EvaluationError(f"pinned Qwen3.5 {label} resolves to an invalid blob")
        _under_root(resolved, blobs, label=f"pinned Qwen3.5 {label} blob")
    else:
        resolved = logical.resolve()
        if not resolved.is_file() or resolved.stat().st_size < 1:
            raise EvaluationError(f"pinned Qwen3.5 {label} is not a non-empty file")
        _under_root(resolved, snapshot, label=f"pinned Qwen3.5 {label}")
    return {
        "logical_path": str(logical),
        "resolved_path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _snapshot_weight_record(snapshot: Path, logical: Path, *, label: str) -> dict[str, Any]:
    """Bind a large checkpoint shard by its HF content-addressed blob name.

    Re-hashing every 9B-model shard from every one-GPU worker would turn
    custody replay into a multi-terabyte shared-filesystem storm.  Stock HF
    snapshots link each shard to a content-addressed blob, so the immutable
    blob name plus size is the stable weight identity.  A non-symlink local
    snapshot is uncommon; hash it fully rather than accepting an unaddressed
    large file.
    """

    if not logical.exists():
        raise EvaluationError(f"pinned Qwen3.5 snapshot lacks {label}: {logical}")
    if logical.is_symlink():
        blobs = snapshot.parent.parent / "blobs"
        if blobs.is_symlink() or not blobs.is_dir():
            raise EvaluationError("pinned Qwen3.5 snapshot has no regular Hugging Face blob tree")
        resolved = logical.resolve()
        if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size < 1:
            raise EvaluationError(f"pinned Qwen3.5 {label} resolves to an invalid blob")
        _under_root(resolved, blobs, label=f"pinned Qwen3.5 {label} blob")
        if not _is_sha256(resolved.name):
            raise EvaluationError(f"pinned Qwen3.5 {label} blob is not content-addressed: {resolved.name!r}")
        return {
            "logical_path": str(logical),
            "resolved_path": str(resolved),
            "content_address": resolved.name,
            "size_bytes": resolved.stat().st_size,
        }
    resolved = logical.resolve()
    if not resolved.is_file() or resolved.stat().st_size < 1:
        raise EvaluationError(f"pinned Qwen3.5 {label} is not a non-empty regular file")
    _under_root(resolved, snapshot, label=f"pinned Qwen3.5 {label}")
    return {
        "logical_path": str(logical),
        "resolved_path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _snapshot_identity() -> dict[str, Any]:
    """Identity-bind every native-HF model, tokenizer, and weight asset."""

    snapshot = MODEL_SNAPSHOT
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.parent.name != "snapshots":
        raise EvaluationError(f"pinned Qwen3.5 snapshot is absent or linked: {snapshot}")
    files: dict[str, dict[str, Any]] = {
        name: _snapshot_logical_identity(snapshot, snapshot / name, label=name)
        for name in SNAPSHOT_REQUIRED_CONFIGURATION_FILES
    }
    template_assets = {
        candidate.name
        for candidate in snapshot.iterdir()
        if "template" in candidate.name.lower() and (candidate.is_file() or candidate.is_symlink())
    }
    processor_assets = {
        candidate.name
        for candidate in snapshot.iterdir()
        if (
            ("processor" in candidate.name.lower() or "preprocess" in candidate.name.lower())
            and candidate.suffix.lower() == ".json"
            and (candidate.is_file() or candidate.is_symlink())
        )
    }
    for name in sorted(set(SNAPSHOT_OPTIONAL_CONFIGURATION_FILES) | template_assets | processor_assets):
        candidate = snapshot / name
        if candidate.exists() or candidate.is_symlink():
            files[name] = _snapshot_logical_identity(snapshot, candidate, label=name)
    indices = sorted(snapshot.glob("*.safetensors.index.json"))
    if len(indices) > 1:
        raise EvaluationError("pinned Qwen3.5 snapshot has ambiguous safetensors indices")
    indexed_weights: set[str] = set()
    safetensors_index: dict[str, Any] | None = None
    for index in indices:
        relative = index.relative_to(snapshot).as_posix()
        safetensors_index = _snapshot_logical_identity(snapshot, index, label=relative)
        try:
            document = json.loads(Path(safetensors_index["resolved_path"]).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"pinned Qwen3.5 has an invalid safetensors index: {index}") from exc
        if _snapshot_logical_identity(snapshot, index, label=relative) != safetensors_index:
            raise EvaluationError("pinned Qwen3.5 safetensors index changed during custody replay")
        weight_map = document.get("weight_map") if isinstance(document, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise EvaluationError(f"pinned Qwen3.5 safetensors index has no weight map: {index}")
        for filename in weight_map.values():
            if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise EvaluationError(f"pinned Qwen3.5 safetensors index contains an unsafe weight filename: {filename!r}")
            indexed_weights.add(filename)
    weights = indexed_weights | {path.name for path in snapshot.glob("*.safetensors")}
    if not weights:
        raise EvaluationError("pinned Qwen3.5 snapshot has no safetensors weights")
    weight_files = {
        filename: _snapshot_weight_record(snapshot, snapshot / filename, label=filename)
        for filename in sorted(weights)
    }
    for optional in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        candidate = snapshot / optional
        if candidate.exists():
            files[optional] = _snapshot_logical_identity(snapshot, candidate, label=optional)
    return {
        "path": str(snapshot),
        "configuration_files": files,
        "safetensors_index": safetensors_index,
        "indexed_weight_files": sorted(indexed_weights),
        "weight_files": weight_files,
        "weight_file_count": len(weights),
    }


def _installed_version(module: Any, *, distribution: str) -> str | None:
    declared = getattr(module, "__version__", None)
    if isinstance(declared, str) and declared:
        return declared
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_evaluator_environment(*, require_one_gpu: bool) -> dict[str, Any]:
    """Pin the one-GPU native-HF worker and exact Inspect implementation."""

    try:
        import inspect_ai
        import mcq_bias
        import peft
        import safetensors
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - configured Isambard runtime
        raise EvaluationError("native HF two-bias evaluation environment is incomplete") from exc
    installed = {
        "inspect-ai": _installed_version(inspect_ai, distribution="inspect-ai"),
        "torch": _installed_version(torch, distribution="torch"),
        "transformers": _installed_version(transformers, distribution="transformers"),
        "peft": _installed_version(peft, distribution="peft"),
        "safetensors": _installed_version(safetensors, distribution="safetensors"),
    }
    if installed != EVALUATOR_PACKAGE_VERSIONS:
        raise EvaluationError(
            "matched HF campaign evaluator package versions differ from the pinned remote runtime: "
            f"got={installed!r}, expected={EVALUATOR_PACKAGE_VERSIONS!r}"
        )
    if require_one_gpu:
        tokens = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        if len(tokens) != 1 or tokens[0] in {"-1", "NoDevFiles"} or any(any(char.isspace() for char in item) for item in tokens):
            raise EvaluationError("each matched-HF worker requires exactly one Slurm-visible CUDA device")
    attention = os.environ.get("CTM_DISABLE_CUDNN_SDP")
    if attention != "0":
        raise EvaluationError("matched HF campaign requires CTM_DISABLE_CUDNN_SDP to be exactly '0'")
    return {
        "backend": "native-hf-peft",
        "inspect_version": installed["inspect-ai"],
        "torch_version": installed["torch"],
        "transformers_version": installed["transformers"],
        "peft_version": installed["peft"],
        "safetensors_version": installed["safetensors"],
        "mcq_bias_version": _installed_version(mcq_bias, distribution="mcq-bias"),
        "attention_policy": {"ctm_disable_cudnn_sdp": attention},
    }


def _r002_module():
    try:
        from infra.isambard import run_qwen35_rmct_checkpoint_two_bias_evals_16gpu as r002
    except ImportError as exc:  # pragma: no cover - deployment error
        raise EvaluationError("early-checkpoint strict custody verifier is unavailable") from exc
    return r002


def _r005_module():
    try:
        from infra.isambard import run_qwen35_rmct_convergence_r4_two_bias_evals as r005
    except ImportError as exc:  # pragma: no cover - deployment error
        raise EvaluationError("terminal-checkpoint strict custody verifier is unavailable") from exc
    return r005


def _identity_record(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{label} is absent from checkpoint custody")
    path = value.get("path")
    digest = value.get("sha256")
    size = value.get("size_bytes")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or not _is_sha256(digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
    ):
        raise EvaluationError(f"{label} has an incomplete immutable identity")
    if _identity(Path(path), label=label) != dict(value):
        raise EvaluationError(f"{label} changed after strict custody validation")
    return dict(value)


def _revalidate_early_checkpoint_source(*, training_repository: str | Path, step: int) -> dict[str, Any]:
    """Replay raw optimizer-state custody for exactly step 16 or 64."""

    r002 = _r002_module()
    try:
        custody = r002.validate_approved_target(training_repository, step=step)
    except Exception as exc:
        raise EvaluationError(f"raw step-{step} checkpoint failed sealed-custody replay: {exc}") from exc
    if not isinstance(custody, Mapping):
        raise EvaluationError(f"raw step-{step} custody verifier returned no object")
    target = custody.get("target")
    checkpoint = custody.get("checkpoint")
    if (
        not isinstance(target, Mapping)
        or target.get("step") != step
        or not isinstance(checkpoint, Mapping)
        or not isinstance(checkpoint.get("path"), str)
        or not Path(str(checkpoint["path"])).is_absolute()
        or checkpoint.get("optimizer_step") != step
        or checkpoint.get("full_resumability_required") is not True
    ):
        raise EvaluationError(f"raw step-{step} custody lacks the exact resumable checkpoint identity")
    files = checkpoint.get("files")
    required = {
        "adapter_config",
        "adapter_model",
        "optimizer",
        "manifest",
        "replicated_training_manifest",
        "replicated_training_rng",
    }
    if not isinstance(files, Mapping) or set(files) != required:
        raise EvaluationError(f"raw step-{step} custody lacks its complete resumability file set")
    return {
        "source": "raw-training-checkpoint",
        "step": step,
        "condition": target.get("condition"),
        "checkpoint": {
            **dict(checkpoint),
            "files": {name: _identity_record(files[name], label=f"step-{step} checkpoint {name}") for name in sorted(required)},
        },
        "checkpoint_receipt": _identity_record(custody.get("checkpoint_receipt"), label=f"step-{step} checkpoint receipt"),
        "completion_receipt": _identity_record(custody.get("completion_receipt"), label=f"step-{step} completion receipt"),
        "decision_receipt": _identity_record(custody.get("decision_receipt"), label=f"step-{step} continue decision"),
    }


def _revalidate_terminal_checkpoint_source(*, training_repository: str | Path) -> dict[str, Any]:
    """Replay the converged r4 step-176 checkpoint including all six files."""

    r005 = _r005_module()
    try:
        custody = r005.validate_final_checkpoint(training_repository)
    except Exception as exc:
        raise EvaluationError(f"raw terminal step-176 checkpoint failed sealed-custody replay: {exc}") from exc
    if not isinstance(custody, Mapping):
        raise EvaluationError("terminal checkpoint custody verifier returned no object")
    checkpoint_value = custody.get("path")
    strict = custody.get("custody")
    if not isinstance(checkpoint_value, str) or not Path(checkpoint_value).is_absolute() or not isinstance(strict, Mapping):
        raise EvaluationError("terminal custody lacks an absolute checkpoint or strict resumability record")
    checkpoint = Path(checkpoint_value)
    required = {
        "adapter_config": "adapter_config.json",
        "adapter_model": "adapter_model.safetensors",
        "optimizer": "optimizer.pt",
        "manifest": "manifest.json",
        "replicated_training_manifest": "replicated_training_manifest.json",
        "replicated_training_rng": "replicated_training_rng.pt",
    }
    strict_files = strict.get("files")
    if not isinstance(strict_files, Mapping) or set(strict_files) != set(required):
        raise EvaluationError("terminal strict custody lacks adapter/optimizer/manifest/replicated-RNG files")
    files = {name: _identity(checkpoint / filename, label=f"terminal checkpoint {name}") for name, filename in required.items()}
    for name, identity in files.items():
        recorded = strict_files[name]
        if not isinstance(recorded, Mapping) or recorded.get("sha256") != identity["sha256"] or recorded.get("size_bytes") != identity["size_bytes"]:
            raise EvaluationError(f"terminal strict custody conflicts with direct {name} identity")
    return {
        "source": "raw-training-checkpoint",
        "step": 176,
        "condition": "rmct-convergence-r4-s011-two-bias-v1-r005",
        "checkpoint": {
            "path": str(checkpoint),
            "optimizer_step": 176,
            "full_resumability_required": True,
            "files": files,
            "strict_custody": dict(strict),
        },
        "decision_receipt": _identity_record(custody.get("decision_receipt"), label="terminal convergence decision"),
    }


def _base_runtime(*, snapshot: Mapping[str, Any], evaluator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "profile": "hf-base",
        "mode": "native-hf-base",
        "checkpoint_backend": "pinned-base-snapshot",
        "base_model": str(MODEL_SNAPSHOT),
        "model": f"hf/{MODEL_SNAPSHOT}",
        "model_snapshot": dict(snapshot),
        "provider": "hf",
        "model_args": dict(HF_MODEL_ARGS),
        "generation_config": dict(GENERATION_CONFIG),
        "evaluator": dict(evaluator),
    }


def _trained_runtime(source: Mapping[str, Any], *, snapshot: Mapping[str, Any], evaluator: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_record = source.get("checkpoint")
    checkpoint = checkpoint_record.get("path") if isinstance(checkpoint_record, Mapping) else None
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
        raise EvaluationError("checkpoint custody has no absolute raw PEFT adapter path")
    return {
        "profile": "hf-peft",
        "mode": "native-hf-peft",
        "checkpoint_backend": "local",
        "base_model": str(MODEL_SNAPSHOT),
        "model": f"hf/{MODEL_SNAPSHOT}",
        "checkpoint": checkpoint,
        "model_snapshot": dict(snapshot),
        "raw_checkpoint_custody": dict(source),
        "source_checkpoint": {"step": source["step"], "condition": source["condition"]},
        "provider": "hf",
        "model_args": dict(HF_LOCAL_MODEL_ARGS),
        "generation_config": dict(GENERATION_CONFIG),
        "evaluator": dict(evaluator),
    }


def _condition_runtime_records(*, training_repository: str | Path, snapshot: Mapping[str, Any], evaluator: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    step16 = _revalidate_early_checkpoint_source(training_repository=training_repository, step=16)
    step64 = _revalidate_early_checkpoint_source(training_repository=training_repository, step=64)
    step176 = _revalidate_terminal_checkpoint_source(training_repository=training_repository)
    return {
        "base": _base_runtime(snapshot=snapshot, evaluator=evaluator),
        "step016": _trained_runtime(step16, snapshot=snapshot, evaluator=evaluator),
        "step064": _trained_runtime(step64, snapshot=snapshot, evaluator=evaluator),
        "step176": _trained_runtime(step176, snapshot=snapshot, evaluator=evaluator),
    }


def _expected_task_identities() -> tuple[tuple[str, str, str, str, str | None], ...]:
    try:
        from experiments.rmct_tbsr.constants import HLE_DATASET
        from experiments.stage2_ood_hle.prepare import TRAINING_BIAS
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise EvaluationError("frozen Stage-2 constants are unavailable") from exc
    heldout = ("suggested_answer", *HELD_OUT_BIASES)
    identities: list[tuple[str, str, str, str, str | None]] = [
        ("unbiased", "iid", "in_domain", "logiqa", None),
        ("unbiased", "iid", "in_domain", "hellaswag", None),
        ("unbiased", "heldout_dataset", "hle", HLE_DATASET, None),
        ("biased", "iid", "in_domain", "logiqa", TRAINING_BIAS),
        ("biased", "iid", "in_domain", "hellaswag", TRAINING_BIAS),
        ("biased", "heldout_dataset", "hle", HLE_DATASET, TRAINING_BIAS),
    ]
    for bias in heldout:
        identities.extend(("biased", "heldout_bias", "in_domain", dataset, bias) for dataset in ("logiqa", "hellaswag"))
    for bias in heldout:
        identities.append(("biased", "heldout_dataset_and_bias", "hle", HLE_DATASET, bias))
    if len(identities) != TASK_COUNT:  # pragma: no cover - immutable topology guard
        raise EvaluationError("frozen Stage-2 task topology has the wrong size")
    return tuple(identities)


def _sample_count_for_task(task_index: int) -> int:
    if isinstance(task_index, bool) or task_index not in TASK_SAMPLE_COUNTS:
        raise EvaluationError("task index is outside the frozen mixed 21-cell matrix")
    return TASK_SAMPLE_COUNTS[task_index]


def _sampling_contract() -> dict[str, Any]:
    return {
        "task_sample_counts": [
            {"task_index": task_index, "sample_count": _sample_count_for_task(task_index)}
            for task_index in range(1, TASK_COUNT + 1)
        ],
        "task_identities": [
            {
                "task_index": task_index,
                "kind": identity[0],
                "regime": identity[1],
                "population": identity[2],
                "dataset": identity[3],
                "bias_type": identity[4],
            }
            for task_index, identity in enumerate(_expected_task_identities(), start=1)
        ],
        "iid_task_indices": list(IID_TASK_INDICES),
        "hle_task_indices": list(HLE_TASK_INDICES),
        "total_samples_per_condition": TOTAL_SAMPLES_PER_CONDITION,
        "seed_policy": "frozen-stage2-sampling-no-explicit-seed",
    }


def _validate_mixed_sampling_matrix(specs: Sequence[Any]) -> None:
    if len(specs) != TASK_COUNT:
        raise EvaluationError("Stage-2 task factory no longer yields exactly 21 cells")
    expected = _expected_task_identities()
    pools: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    for task_index, spec in enumerate(specs, start=1):
        ids = getattr(spec, "question_ids", None)
        identity = (
            getattr(spec, "kind", None),
            getattr(spec, "regime", None),
            getattr(spec, "population", None),
            getattr(spec, "dataset", None),
            getattr(spec, "bias_type", None),
        )
        expected_population = "in_domain" if task_index in IID_TASK_INDICES else "hle"
        expected_kind = "unbiased" if task_index in CLEAN_TASK_INDICES else "biased"
        if (
            identity != expected[task_index - 1]
            or getattr(spec, "population", None) != expected_population
            or getattr(spec, "kind", None) != expected_kind
            or not isinstance(ids, tuple)
            or len(ids) != 100
            or len(ids) != len(set(ids))
            or any(not isinstance(item, str) or not item for item in ids)
            or len(ids) < _sample_count_for_task(task_index)
        ):
            raise EvaluationError(f"task-{task_index} differs from the frozen mixed-count Stage-2 protocol")
        pools.setdefault((str(spec.population), str(spec.dataset)), []).append(ids)
    if len(pools) != len(CLEAN_TASK_INDICES) or any(len(value) != 7 for value in pools.values()):
        raise EvaluationError("Stage-2 task matrix no longer has seven paired variants per clean population")
    if any(any(candidate != entries[0] for candidate in entries[1:]) for entries in pools.values()):
        raise EvaluationError("Stage-2 variants no longer share their ordered full question-ID pool")


def _subset_spec_for_task(spec: Any, *, task_index: int) -> Any:
    count = _sample_count_for_task(task_index)
    ids = tuple(getattr(spec, "question_ids", ()))
    if len(ids) < count:
        raise EvaluationError(f"task-{task_index} has fewer frozen IDs than its mixed sample cap")
    return replace(spec, question_ids=ids[:count])


def _validate_stage2_substrate(manifest: Path) -> dict[str, Any]:
    try:
        from experiments.rmct_two_bias_eval import contract
        from experiments.stage2_ood_hle.tasks import ood_task_specs
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise EvaluationError("two-bias Stage-2 substrate is unavailable") from exc
    try:
        substrate = contract.validate_stage2_substrate(manifest)
        specs = list(ood_task_specs(manifest))
        _validate_mixed_sampling_matrix(specs)
    except Exception as exc:
        raise EvaluationError(f"Stage-2 substrate failed frozen matrix validation: {exc}") from exc
    if (
        tuple(getattr(contract, "SEEN_BIASES", ())) != SEEN_BIASES
        or tuple(getattr(contract, "HELD_OUT_BIASES", ())) != HELD_OUT_BIASES
        or len(specs) != TASK_COUNT
    ):
        raise EvaluationError("two-bias substrate no longer has the approved seen/held-out bias labels")
    return {
        "path": str(manifest.resolve()),
        "sha256": _sha256_file(manifest),
        "task_count": TASK_COUNT,
        "clean_task_count": len(CLEAN_TASK_INDICES),
        "biased_task_count": len(BIASED_TASK_INDICES),
        "seen_biases": list(SEEN_BIASES),
        "held_out_biases": list(HELD_OUT_BIASES),
        "contract": dict(substrate),
    }


def _deployment_custody(*, source_manifest: Path, artifact_root: Path, output: Path, materialize: bool) -> dict[str, Any]:
    try:
        from experiments.rmct_two_bias_eval import deployment
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise EvaluationError("Stage-2 deployment bridge is unavailable") from exc
    source = _identity(source_manifest, label="source Stage-2 manifest")
    if materialize:
        try:
            record = deployment.materialize_deployment_manifest(source_manifest, artifact_root, output)
        except Exception as exc:
            raise EvaluationError(f"could not materialize immutable Stage-2 deployment manifest: {exc}") from exc
        if not isinstance(record, Mapping) or Path(str(record.get("manifest_path", ""))).resolve() != output.resolve():
            raise EvaluationError("deployment bridge did not bind the requested output manifest")
        result = dict(record)
        result.pop("status", None)
    else:
        try:
            provenance = deployment.validate_deployment_manifest(output)
        except Exception as exc:
            raise EvaluationError(f"could not replay deployed Stage-2 manifest: {exc}") from exc
        if not isinstance(provenance, Mapping):
            raise EvaluationError("deployed Stage-2 manifest lacks immutable deployment provenance")
        if provenance.get("source_manifest") != {"path": source["path"], "sha256": source["sha256"]}:
            raise EvaluationError("deployed Stage-2 manifest binds a different source manifest")
        if provenance.get("artifact_root") != str(artifact_root):
            raise EvaluationError("deployed Stage-2 manifest binds a different artifact root")
        result = {
            "schema": getattr(deployment, "DEPLOYMENT_SCHEMA", "unknown"),
            "manifest_path": str(output),
            "manifest_sha256": _sha256_file(output),
            "source_manifest_sha256": source["sha256"],
            "artifact_root": str(artifact_root),
            "artifacts": dict(provenance.get("copied_artifacts", {})),
        }
    if result.get("source_manifest_sha256") != source["sha256"] or result.get("artifact_root") != str(artifact_root):
        raise EvaluationError("deployment custody differs from the requested source/artifact roots")
    return result


def _topology_contract() -> dict[str, Any]:
    condition_names = {condition.name for condition in CONDITIONS}
    if set(CLEAN_WAVE) != set(range(12)):  # pragma: no cover - fixed constant guard
        raise EvaluationError("clean wave must assign exactly ranks 0 through 11")
    clean_by_condition = {name: [] for name in condition_names}
    for rank, (condition_name, task_index) in sorted(CLEAN_WAVE.items()):
        if condition_name not in condition_names or task_index not in CLEAN_TASK_INDICES:
            raise EvaluationError(f"clean wave rank {rank} has an invalid condition/task assignment")
        clean_by_condition[condition_name].append(task_index)
    if any(sorted(tasks) != list(CLEAN_TASK_INDICES) for tasks in clean_by_condition.values()):
        raise EvaluationError("clean wave must cover every condition's three clean tasks exactly once")

    if list(sorted(BIASED_WAVES)) != [1, 2, 3, 4, 5]:  # pragma: no cover - fixed constant guard
        raise EvaluationError("biased phase must contain exactly five ordered waves")
    expected_wave_sizes = [16, 16, 16, 16, 8]
    biased_by_condition = {name: [] for name in condition_names}
    source_samples_by_rank = {rank: 0 for rank in range(16)}
    biased_waves: list[dict[str, Any]] = []
    for wave, assignments in sorted(BIASED_WAVES.items()):
        if len(assignments) != expected_wave_sizes[wave - 1] or not set(assignments).issubset(set(range(16))):
            raise EvaluationError(f"biased wave {wave} has an invalid rank map")
        wave_assignments: list[dict[str, Any]] = []
        for rank, (condition_name, task_index) in sorted(assignments.items()):
            if condition_name not in condition_names or task_index not in BIASED_TASK_INDICES:
                raise EvaluationError(f"biased wave {wave} rank {rank} has an invalid condition/task assignment")
            biased_by_condition[condition_name].append(task_index)
            sample_count = _sample_count_for_task(task_index)
            source_samples_by_rank[rank] += sample_count
            wave_assignments.append(
                {
                    "rank": rank,
                    "condition": condition_name,
                    "task_index": task_index,
                    "sample_count": sample_count,
                }
            )
        biased_waves.append({"wave": wave, "assignments": wave_assignments})
    if any(sorted(tasks) != list(BIASED_TASK_INDICES) for tasks in biased_by_condition.values()):
        raise EvaluationError("biased waves must cover every condition's 18 biased tasks exactly once")
    if set(source_samples_by_rank.values()) != {300}:
        raise EvaluationError("balanced biased waves must give every rank exactly 300 source samples")
    return {
        "nodes": 4,
        "gpus_per_node": 4,
        "workers": 16,
        "one_gpu_per_worker": True,
        "clean_wave": [
            {"rank": rank, "condition": condition_name, "task_index": task_index, "sample_count": _sample_count_for_task(task_index)}
            for rank, (condition_name, task_index) in sorted(CLEAN_WAVE.items())
        ],
        "clean_task_indices": list(CLEAN_TASK_INDICES),
        "clean_gate_before_biased": True,
        "biased_waves": biased_waves,
        "biased_source_samples_by_rank": [
            {"rank": rank, "sample_count": sample_count}
            for rank, sample_count in sorted(source_samples_by_rank.items())
        ],
    }


def _policy_contract() -> dict[str, bool]:
    return {
        "self_submits": False,
        "scheduler_clients": False,
        "chains_successors": False,
        "cancels_jobs": False,
        "automatic_retries": False,
        "incomplete_attempts_require_operator": True,
        "overwrite_differing_logs": False,
        "partial_attempts_preserved": True,
        "clean_receipt_gate_required": True,
        "strict_checkpoint_and_base_custody": True,
    }


def build_launch_contract(
    *,
    campaign_root: str | Path,
    training_repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
    materialize: bool,
) -> tuple[dict[str, Any], CampaignPaths]:
    """Build/replay the complete all-HF campaign contract without generation."""

    if "seed" in GENERATION_CONFIG:  # pragma: no cover - protects frozen protocol edits
        raise EvaluationError("the frozen Stage-2 sampling protocol must not add a new seed")
    evaluator = _validate_evaluator_environment(require_one_gpu=False)
    paths = _campaign_paths(campaign_root)
    repository = _resolve_unlinked(training_repository, label="sealed training repository")
    source = _resolve_unlinked(source_stage2_manifest, label="source Stage-2 manifest")
    artifacts = _resolve_unlinked(stage2_artifact_root, label="Stage-2 artifact root")
    if not repository.is_dir() or not source.is_file() or not artifacts.is_dir():
        raise EvaluationError("training repository, source manifest, and Stage-2 artifact root must be regular inputs")
    deployment = _deployment_custody(
        source_manifest=source,
        artifact_root=artifacts,
        output=paths.deployment_manifest,
        materialize=materialize,
    )
    substrate = _validate_stage2_substrate(paths.deployment_manifest)
    snapshot = _snapshot_identity()
    runtime = _condition_runtime_records(training_repository=repository, snapshot=snapshot, evaluator=evaluator)
    contract = {
        "schema": LAUNCH_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "training_repository": str(repository),
        "model_snapshot": snapshot,
        "source_stage2_manifest": _identity(source, label="source Stage-2 manifest"),
        "stage2_artifact_root": str(artifacts),
        "deployment_manifest": {"path": str(paths.deployment_manifest), "provenance": deployment, "substrate": substrate},
        "evaluator": evaluator,
        "sampling": _sampling_contract(),
        "conditions": [
            {
                "name": condition.name,
                "label": condition.label,
                "optimizer_step": condition.optimizer_step,
                "source_kind": condition.source_kind,
                "condition": condition.artifact_name,
                "runtime": runtime[condition.name],
            }
            for condition in CONDITIONS
        ],
        "topology": _topology_contract(),
        "critical_sources": _critical_source_identities(),
        "outputs": {
            "root": str(paths.root),
            "input": str(paths.input),
            "deployment_manifest": str(paths.deployment_manifest),
            "conditions": str(paths.conditions),
            "completion": str(paths.completion),
        },
        "policy": _policy_contract(),
    }
    return contract, paths


def _condition_row(contract: Mapping[str, Any], condition: Condition) -> Mapping[str, Any]:
    rows = contract.get("conditions")
    if not isinstance(rows, list):
        raise EvaluationError("campaign contract has no condition records")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("name") == condition.name]
    if len(matches) != 1:
        raise EvaluationError(f"campaign contract has ambiguous record for {condition.name}")
    return matches[0]


def _expected_runtime_receipt(contract: Mapping[str, Any], paths: CampaignPaths, condition: Condition) -> dict[str, Any]:
    row = _condition_row(contract, condition)
    runtime = row.get("runtime")
    if not isinstance(runtime, Mapping):
        raise EvaluationError(f"campaign condition {condition.name} lacks runtime evidence")
    condition_paths = _condition_paths(paths, condition)
    return {
        "schema": RUNTIME_RECEIPT_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "condition": {
            "name": condition.name,
            "label": condition.label,
            "artifact_name": condition.artifact_name,
            "optimizer_step": condition.optimizer_step,
            "source_kind": condition.source_kind,
        },
        "launch_contract": _identity(paths.contract, label="campaign launch contract"),
        "model_snapshot": contract["model_snapshot"],
        "deployment_manifest": contract["deployment_manifest"],
        "sampling": contract["sampling"],
        "runtime": dict(runtime),
        "outputs": {"root": str(condition_paths.root), "raw": str(condition_paths.raw), "attempts": str(condition_paths.attempts)},
    }


def _expected_evaluation_receipt(contract: Mapping[str, Any], paths: CampaignPaths, condition: Condition) -> dict[str, Any]:
    condition_paths = _condition_paths(paths, condition)
    return {
        "schema": EVALUATION_RECEIPT_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "condition": {
            "name": condition.name,
            "label": condition.label,
            "artifact_name": condition.artifact_name,
            "optimizer_step": condition.optimizer_step,
        },
        "launch_contract": _identity(paths.contract, label="campaign launch contract"),
        "runtime_receipt": _identity(condition_paths.runtime_receipt, label=f"{condition.name} runtime receipt"),
        "deployment_manifest": contract["deployment_manifest"],
        "sampling": contract["sampling"],
        "outputs": {
            "root": str(condition_paths.root),
            "raw": str(condition_paths.raw),
            "attempts": str(condition_paths.attempts),
            "receipts": str(condition_paths.receipts),
            "clean_gate": str(condition_paths.clean_gate),
            "preflight": str(condition_paths.preflight),
        },
    }


def prepare(
    *,
    campaign_root: str | Path,
    training_repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
) -> dict[str, Any]:
    """Create/replay contract and per-condition receipts; perform no generation."""

    paths = _campaign_paths(campaign_root)
    if paths.root.exists() and not paths.contract.exists() and paths.conditions.exists():
        raise FileExistsError("refusing to seed a new all-HF contract beside pre-existing condition evidence")
    contract, paths = build_launch_contract(
        campaign_root=campaign_root,
        training_repository=training_repository,
        source_stage2_manifest=source_stage2_manifest,
        stage2_artifact_root=stage2_artifact_root,
        materialize=True,
    )
    launch_status = _write_immutable_json(paths.contract, contract, label="all-HF campaign launch contract")
    runtime_statuses: dict[str, str] = {}
    evaluation_statuses: dict[str, str] = {}
    for condition in CONDITIONS:
        condition_paths = _condition_paths(paths, condition)
        runtime_statuses[condition.name] = _write_immutable_json(
            condition_paths.runtime_receipt,
            _expected_runtime_receipt(contract, paths, condition),
            label=f"{condition.name} native-HF runtime receipt",
        )
        evaluation_statuses[condition.name] = _write_immutable_json(
            condition_paths.evaluation_receipt,
            _expected_evaluation_receipt(contract, paths, condition),
            label=f"{condition.name} evaluation receipt",
        )
    return {
        "campaign": CAMPAIGN_NAME,
        "launch_contract": _identity(paths.contract, label="all-HF campaign launch contract"),
        "launch_status": launch_status,
        "runtime_receipt_statuses": runtime_statuses,
        "evaluation_receipt_statuses": evaluation_statuses,
        "conditions": [condition.name for condition in CONDITIONS],
        "generations_per_condition": TOTAL_SAMPLES_PER_CONDITION,
        "total_generations": TOTAL_SAMPLES_PER_CONDITION * len(CONDITIONS),
    }


def _load_campaign(
    *,
    campaign_root: str | Path,
    training_repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
) -> tuple[dict[str, Any], CampaignPaths]:
    paths = _campaign_paths(campaign_root)
    stored = _read_json(paths.contract, label="all-HF campaign launch contract")
    rebuilt, rebuilt_paths = build_launch_contract(
        campaign_root=campaign_root,
        training_repository=training_repository,
        source_stage2_manifest=source_stage2_manifest,
        stage2_artifact_root=stage2_artifact_root,
        materialize=False,
    )
    if paths != rebuilt_paths or stored != rebuilt:
        raise EvaluationError("all-HF campaign contract differs from current replayed source/checkpoint/runtime custody")
    return rebuilt, paths


def _load_runtime_receipt(contract: Mapping[str, Any], paths: CampaignPaths, condition: Condition) -> tuple[dict[str, Any], str]:
    condition_paths = _condition_paths(paths, condition)
    stored = _read_json(condition_paths.runtime_receipt, label=f"{condition.name} runtime receipt")
    expected = _expected_runtime_receipt(contract, paths, condition)
    if stored != expected:
        raise EvaluationError(f"{condition.name} runtime receipt differs from immutable launch custody")
    return expected, _sha256_file(condition_paths.runtime_receipt)


def _load_evaluation_receipt(contract: Mapping[str, Any], paths: CampaignPaths, condition: Condition) -> tuple[dict[str, Any], str]:
    condition_paths = _condition_paths(paths, condition)
    _load_runtime_receipt(contract, paths, condition)
    stored = _read_json(condition_paths.evaluation_receipt, label=f"{condition.name} evaluation receipt")
    expected = _expected_evaluation_receipt(contract, paths, condition)
    if stored != expected:
        raise EvaluationError(f"{condition.name} evaluation receipt differs from immutable launch custody")
    return expected, _sha256_file(condition_paths.evaluation_receipt)


def _full_question_ids_for_task(*, contract: Mapping[str, Any], task_index: int) -> tuple[str, ...]:
    deployment = contract.get("deployment_manifest")
    deployed = deployment.get("path") if isinstance(deployment, Mapping) else None
    if not isinstance(deployed, str):
        raise EvaluationError("campaign contract lacks a deployed Stage-2 manifest")
    try:
        from experiments.stage2_ood_hle.tasks import ood_task_specs

        specs = list(ood_task_specs(deployed))
        _validate_mixed_sampling_matrix(specs)
    except Exception as exc:
        raise EvaluationError(f"could not replay frozen task IDs for promotion: {exc}") from exc
    if task_index < 1 or task_index > len(specs):
        raise EvaluationError("task index is absent from deployed Stage-2 matrix")
    return tuple(specs[task_index - 1].question_ids)


def _value_field(value: Any, field: str, default: Any = None) -> Any:
    return value.get(field, default) if isinstance(value, Mapping) else getattr(value, field, default)


def _configuration_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            result = method()
            if isinstance(result, Mapping):
                return result
    return {}


def _assert_base_hf_runtime(header_log: Any, *, path: Path) -> tuple[str, dict[str, Any]]:
    """Validate direct base-snapshot headers without pretending they are PEFT."""

    try:
        from experiments.stage2_ood_hle import raw_preflight as mechanical
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise EvaluationError("Stage-2 header validator is unavailable") from exc
    evaluation = mechanical._attribute(header_log, "eval")
    model = str(mechanical._attribute(evaluation, "model", "") or "")
    expected_model = f"hf/{MODEL_SNAPSHOT}"
    if model != expected_model:
        raise EvaluationError(f"base raw log has wrong pinned-snapshot model: {path}: {model!r}")
    metadata = _configuration_mapping(mechanical._attribute(evaluation, "metadata", {}))
    if metadata.get("model") != expected_model or dict(_configuration_mapping(metadata.get("model_args", {}))) != HF_MODEL_ARGS:
        raise EvaluationError(f"base raw log has wrong model/model_args metadata: {path}")
    if dict(_configuration_mapping(metadata.get("generation_config", {}))) != GENERATION_CONFIG:
        raise EvaluationError(f"base raw log has wrong frozen generation metadata: {path}")
    model_args = _configuration_mapping(mechanical._attribute(evaluation, "model_args", {}))
    for field, expected in {"device": "cuda:0", "dtype": "bfloat16"}.items():
        if model_args.get(field) != expected:
            raise EvaluationError(f"base raw log has wrong HF model_args.{field}: {path}")
    generation = _configuration_mapping(mechanical._attribute(evaluation, "model_generate_config", {}))
    for field, expected in GENERATION_CONFIG.items():
        if generation.get(field) != expected:
            raise EvaluationError(f"base raw log has wrong generation.{field}: {path}")
    return model, {
        "profile": "hf-base",
        "provider": "hf",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "max_connections": GENERATION_CONFIG["max_connections"],
        "base_model": str(MODEL_SNAPSHOT),
    }


def _assert_hf_runtime(header_log: Any, *, path: Path, runtime: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    profile = runtime.get("profile")
    if profile == "hf-base":
        return _assert_base_hf_runtime(header_log, path=path)
    if profile != "hf-peft":
        raise EvaluationError("evaluation runtime is not an approved native-HF profile")
    checkpoint = runtime.get("checkpoint")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
        raise EvaluationError("HF/PEFT runtime has no absolute raw checkpoint")
    try:
        from experiments.rmct_two_bias_eval import raw_preflight as snapshot_preflight
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise EvaluationError("snapshot-aware HF header validator is unavailable") from exc
    if getattr(snapshot_preflight, "BASE_MODEL", None) != str(MODEL_SNAPSHOT):
        raise EvaluationError("snapshot-aware HF header validator is pinned to a different base model")
    try:
        model, observed = snapshot_preflight._assert_snapshot_hf_runtime(
            header_log,
            path=path,
            checkpoint=checkpoint,
            max_connections=GENERATION_CONFIG["max_connections"],
        )
    except Exception as exc:
        raise EvaluationError(f"raw log fails native HF/PEFT header validation: {exc}") from exc
    if not isinstance(model, str) or not isinstance(observed, Mapping):
        raise EvaluationError("snapshot-aware HF header validator returned incomplete runtime evidence")
    return model, dict(observed)


def _inspect_success(path: Path, *, task_index: int) -> int:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise EvaluationError("Inspect AI is required to validate task receipts") from exc
    try:
        header = read_eval_log(str(path), header_only=True)
        full = read_eval_log(str(path), header_only=False)
    except Exception as exc:
        raise EvaluationError(f"could not read Inspect EvalLog: {path}") from exc
    evaluation = _value_field(header, "eval")
    metadata = _configuration_mapping(_value_field(evaluation, "metadata", {}))
    expected_count = _sample_count_for_task(task_index)
    samples = _value_field(full, "samples", None)
    if (
        _value_field(header, "status") != "success"
        or metadata.get("task_indices") != [task_index]
        or metadata.get("task_count") != TASK_COUNT
        or not isinstance(samples, (list, tuple))
        or len(samples) != expected_count
    ):
        raise EvaluationError(f"EvalLog is not successful exact mixed task-{task_index}: {path}")
    return expected_count


def _validate_promotable_eval_log(
    *,
    path: Path,
    contract: Mapping[str, Any],
    condition: Condition,
    condition_paths: ConditionPaths,
    task_index: int,
) -> int:
    """Prove exact source IDs, limit, Stage-2 identity, and HF runtime."""

    count = _inspect_success(path, task_index=task_index)
    full_ids = _full_question_ids_for_task(contract=contract, task_index=task_index)
    expected_ids = full_ids[:count]
    row = _condition_row(contract, condition)
    runtime = row.get("runtime")
    if not isinstance(runtime, Mapping):
        raise EvaluationError("campaign condition lacks a runtime record")
    try:
        from experiments.stage2_ood_hle import raw_preflight as mechanical
        from experiments.stage2_ood_hle.tasks import ood_task_specs

        header = mechanical._read_eval_log(path, header_only=True)
        full = mechanical._read_eval_log(path, header_only=False)
        deployed = contract["deployment_manifest"]["path"]
        specs = list(ood_task_specs(deployed))
        _validate_mixed_sampling_matrix(specs)
        source_spec = specs[task_index - 1]
        mechanical._validate_header(header, path=path, spec=source_spec, raw_root=condition_paths.raw)
        _assert_hf_runtime(header, path=path, runtime=runtime)
    except Exception as exc:
        raise EvaluationError(f"task-{task_index} EvalLog fails Stage-2 header/runtime validation: {exc}") from exc
    evaluation = _value_field(full, "eval")
    config = _value_field(evaluation, "config")
    dataset = _value_field(evaluation, "dataset")
    header_ids = mechanical._header_value(evaluation, "question_ids_from")
    dataset_ids = _value_field(dataset, "sample_ids")
    if (
        not isinstance(header_ids, list)
        or tuple(header_ids) != full_ids
        or _value_field(config, "limit") != count
        or _value_field(dataset, "samples") != 100
        or _value_field(dataset, "shuffled") is not False
        or not isinstance(dataset_ids, list)
        or tuple(dataset_ids) != expected_ids
    ):
        raise EvaluationError(f"task-{task_index} EvalLog does not prove the exact mixed limit/full-ID/prefix-ID contract")
    return count


def _task_receipt_path(paths: ConditionPaths, task_index: int) -> Path:
    return paths.receipts / f"task-{task_index:03d}.json"


def _load_task_receipt(
    *,
    contract: Mapping[str, Any],
    paths: CampaignPaths,
    condition: Condition,
    condition_paths: ConditionPaths,
    task_index: int,
    launch_sha256: str,
    runtime_sha256: str,
    evaluation_sha256: str,
) -> dict[str, Any] | None:
    receipt_path = _task_receipt_path(condition_paths, task_index)
    if not receipt_path.exists() and not receipt_path.is_symlink():
        return None
    receipt = _read_json(receipt_path, label=f"{condition.name} task-{task_index} receipt")
    expected_keys = {
        "schema",
        "condition",
        "task_index",
        "sample_count",
        "launch_contract_sha256",
        "runtime_receipt_sha256",
        "evaluation_receipt_sha256",
        "attempt_log",
        "canonical_log",
    }
    if set(receipt) != expected_keys or receipt.get("schema") != TASK_RECEIPT_SCHEMA:
        raise EvaluationError(f"{condition.name} task-{task_index} receipt has unsupported schema")
    if (
        receipt.get("condition") != condition.name
        or receipt.get("task_index") != task_index
        or receipt.get("sample_count") != _sample_count_for_task(task_index)
        or receipt.get("launch_contract_sha256") != launch_sha256
        or receipt.get("runtime_receipt_sha256") != runtime_sha256
        or receipt.get("evaluation_receipt_sha256") != evaluation_sha256
    ):
        raise EvaluationError(f"{condition.name} task-{task_index} receipt binds different launch/runtime/evaluation evidence")
    attempt = receipt.get("attempt_log")
    canonical = receipt.get("canonical_log")
    if not isinstance(attempt, Mapping) or not isinstance(canonical, Mapping):
        raise EvaluationError(f"{condition.name} task-{task_index} receipt lacks attempt/canonical identities")
    attempt_path = Path(str(attempt.get("path", "")))
    canonical_path = Path(str(canonical.get("path", "")))
    if attempt_path.is_symlink() or canonical_path.is_symlink():
        raise EvaluationError(f"{condition.name} task-{task_index} receipt names a symlink")
    attempt_path = attempt_path.resolve()
    canonical_path = canonical_path.resolve()
    _under_root(attempt_path, condition_paths.attempts, label=f"{condition.name} task attempt")
    _under_root(canonical_path, condition_paths.raw, label=f"{condition.name} canonical raw log")
    if dict(attempt) != _identity(attempt_path, label=f"{condition.name} task attempt"):
        raise EvaluationError(f"{condition.name} task-{task_index} attempt changed after receipt publication")
    if dict(canonical) != _identity(canonical_path, label=f"{condition.name} canonical raw log"):
        raise EvaluationError(f"{condition.name} task-{task_index} canonical log changed after receipt publication")
    _validate_promotable_eval_log(
        path=canonical_path,
        contract=contract,
        condition=condition,
        condition_paths=condition_paths,
        task_index=task_index,
    )
    return receipt


def _promote_attempt(
    *,
    attempt: Path,
    contract: Mapping[str, Any],
    paths: CampaignPaths,
    condition: Condition,
    condition_paths: ConditionPaths,
    task_index: int,
    launch_sha256: str,
    runtime_sha256: str,
    evaluation_sha256: str,
) -> bool:
    if attempt.is_symlink() or not attempt.is_dir():
        raise EvaluationError(f"task attempt must be a regular directory: {attempt}")
    selected: Path | None = None
    for candidate in sorted(attempt.rglob("*.eval")):
        if candidate.is_symlink() or not candidate.is_file():
            raise EvaluationError(f"task attempt contains a linked/non-file EvalLog: {candidate}")
        try:
            _validate_promotable_eval_log(
                path=candidate,
                contract=contract,
                condition=condition,
                condition_paths=condition_paths,
                task_index=task_index,
            )
        except EvaluationError:
            continue  # preserve incomplete/wrong evidence without promoting it
        if selected is not None:
            raise EvaluationError(f"one task attempt produced ambiguous valid EvalLogs: {selected} and {candidate}")
        selected = candidate
    if selected is None:
        return False
    existing = _load_task_receipt(
        contract=contract,
        paths=paths,
        condition=condition,
        condition_paths=condition_paths,
        task_index=task_index,
        launch_sha256=launch_sha256,
        runtime_sha256=runtime_sha256,
        evaluation_sha256=evaluation_sha256,
    )
    if existing is not None:
        return False
    digest = _sha256_file(selected)
    canonical = condition_paths.raw / f"task-{task_index:03d}" / f"{digest}.eval"
    count = _validate_promotable_eval_log(
        path=selected,
        contract=contract,
        condition=condition,
        condition_paths=condition_paths,
        task_index=task_index,
    )
    _copy_immutable_file(selected, canonical, label=f"{condition.name} task-{task_index} canonical EvalLog")
    _validate_promotable_eval_log(
        path=canonical,
        contract=contract,
        condition=condition,
        condition_paths=condition_paths,
        task_index=task_index,
    )
    receipt = {
        "schema": TASK_RECEIPT_SCHEMA,
        "condition": condition.name,
        "task_index": task_index,
        "sample_count": count,
        "launch_contract_sha256": launch_sha256,
        "runtime_receipt_sha256": runtime_sha256,
        "evaluation_receipt_sha256": evaluation_sha256,
        "attempt_log": _identity(selected, label=f"{condition.name} task-{task_index} attempt EvalLog"),
        "canonical_log": _identity(canonical, label=f"{condition.name} task-{task_index} canonical EvalLog"),
    }
    _write_immutable_json(_task_receipt_path(condition_paths, task_index), receipt, label=f"{condition.name} task-{task_index} receipt")
    if _load_task_receipt(
        contract=contract,
        paths=paths,
        condition=condition,
        condition_paths=condition_paths,
        task_index=task_index,
        launch_sha256=launch_sha256,
        runtime_sha256=runtime_sha256,
        evaluation_sha256=evaluation_sha256,
    ) is None:
        raise EvaluationError(f"{condition.name} task-{task_index} receipt was unreadable after promotion")
    return True


def _promote_prior_attempts(
    *,
    contract: Mapping[str, Any],
    paths: CampaignPaths,
    condition: Condition,
    condition_paths: ConditionPaths,
    task_index: int,
    launch_sha256: str,
    runtime_sha256: str,
    evaluation_sha256: str,
) -> bool:
    root = condition_paths.attempts / f"task-{task_index:03d}"
    if not root.exists():
        return False
    if root.is_symlink() or not root.is_dir():
        raise EvaluationError(f"task attempt root must be a regular directory: {root}")
    promoted = False
    for attempt in sorted(item for item in root.iterdir() if item.is_dir() and not item.is_symlink()):
        promoted = _promote_attempt(
            attempt=attempt,
            contract=contract,
            paths=paths,
            condition=condition,
            condition_paths=condition_paths,
            task_index=task_index,
            launch_sha256=launch_sha256,
            runtime_sha256=runtime_sha256,
            evaluation_sha256=evaluation_sha256,
        ) or promoted
    return promoted


def _has_preserved_attempts(root: Path) -> bool:
    """Reject a new generation if an earlier incomplete attempt exists.

    Receipt-safe promotion is resumability: an earlier complete EvalLog can be
    promoted without generating again.  Starting another evaluator after an
    incomplete attempt would instead be an automatic retry, which this
    campaign deliberately does not authorize.
    """

    if not root.exists() and not root.is_symlink():
        return False
    if root.is_symlink() or not root.is_dir():
        raise EvaluationError(f"task attempt root must be a regular directory: {root}")
    entries = list(root.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_dir():
            raise EvaluationError(f"task attempt root contains unsafe retained evidence: {entry}")
    return bool(entries)


def _expected_clean_gate(
    *,
    contract: Mapping[str, Any],
    paths: CampaignPaths,
    condition: Condition,
    condition_paths: ConditionPaths,
    launch_sha256: str,
    runtime_sha256: str,
    evaluation_sha256: str,
) -> dict[str, Any]:
    clean: list[dict[str, Any]] = []
    for task_index in CLEAN_TASK_INDICES:
        receipt = _load_task_receipt(
            contract=contract,
            paths=paths,
            condition=condition,
            condition_paths=condition_paths,
            task_index=task_index,
            launch_sha256=launch_sha256,
            runtime_sha256=runtime_sha256,
            evaluation_sha256=evaluation_sha256,
        )
        if receipt is None:
            raise EvaluationError(f"{condition.name} clean gate is missing task-{task_index} receipt")
        canonical = receipt["canonical_log"]
        clean.append(
            {
                "task_index": task_index,
                "task_receipt": _identity(_task_receipt_path(condition_paths, task_index), label=f"clean task-{task_index} receipt"),
                "canonical_log": dict(canonical),
            }
        )
    return {
        "schema": CLEAN_GATE_RECEIPT_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "condition": condition.name,
        "launch_contract_sha256": launch_sha256,
        "runtime_receipt_sha256": runtime_sha256,
        "evaluation_receipt_sha256": evaluation_sha256,
        "clean": clean,
        "publication_order": "canonical_clean_log_then_task_receipt_then_gate",
    }


def _seal_clean_gate(
    *,
    contract: Mapping[str, Any],
    paths: CampaignPaths,
    condition: Condition,
    condition_paths: ConditionPaths,
    launch_sha256: str,
    runtime_sha256: str,
    evaluation_sha256: str,
) -> dict[str, Any]:
    document = _expected_clean_gate(
        contract=contract,
        paths=paths,
        condition=condition,
        condition_paths=condition_paths,
        launch_sha256=launch_sha256,
        runtime_sha256=runtime_sha256,
        evaluation_sha256=evaluation_sha256,
    )
    _write_immutable_json(condition_paths.clean_gate, document, label=f"{condition.name} clean-gate receipt")
    return document


def _require_clean_gate(
    *,
    contract: Mapping[str, Any],
    paths: CampaignPaths,
    condition: Condition,
    condition_paths: ConditionPaths,
    launch_sha256: str,
    runtime_sha256: str,
    evaluation_sha256: str,
) -> dict[str, Any]:
    stored = _read_json(condition_paths.clean_gate, label=f"{condition.name} clean-gate receipt")
    expected = _expected_clean_gate(
        contract=contract,
        paths=paths,
        condition=condition,
        condition_paths=condition_paths,
        launch_sha256=launch_sha256,
        runtime_sha256=runtime_sha256,
        evaluation_sha256=evaluation_sha256,
    )
    if stored != expected:
        raise EvaluationError(f"{condition.name} clean-gate receipt differs from current canonical clean custody")
    return stored


def _task_command(
    *,
    python: str,
    contract: Mapping[str, Any],
    condition: Condition,
    condition_paths: ConditionPaths,
    task_index: int,
    attempt: Path,
) -> list[str]:
    """Build one direct HF or HF/PEFT cell command with no vLLM route."""

    row = _condition_row(contract, condition)
    runtime = row.get("runtime")
    deployment = contract.get("deployment_manifest")
    if not isinstance(runtime, Mapping) or not isinstance(deployment, Mapping):
        raise EvaluationError("campaign contract lacks runtime/deployment for task command")
    task_args = {
        "manifest": str(deployment["path"]),
        "unbiased_log": str(condition_paths.raw),
        "prompt_style": "none",
        "include_bias_acknowledged": False,
    }
    command = [python, str(PROJECT_ROOT / "scripts" / "run_evals.py"), "--task-factory", TASK_FACTORY]
    if runtime.get("profile") == "hf-base":
        model = runtime.get("model")
        if model != f"hf/{MODEL_SNAPSHOT}":
            raise EvaluationError("base condition lacks the exact pinned direct HF model")
        command.extend(["--model", str(model)])
    elif runtime.get("profile") == "hf-peft":
        checkpoint = runtime.get("checkpoint")
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
            raise EvaluationError("trained condition lacks an absolute raw PEFT checkpoint")
        command.extend(["--local-checkpoint", checkpoint, "--base-model", str(MODEL_SNAPSHOT)])
    else:
        raise EvaluationError("condition runtime is not an approved direct-HF profile")
    expected_model_args = HF_MODEL_ARGS if runtime.get("profile") == "hf-base" else HF_LOCAL_MODEL_ARGS
    if dict(runtime.get("model_args", {})) != expected_model_args or dict(runtime.get("generation_config", {})) != GENERATION_CONFIG:
        raise EvaluationError("condition runtime changed native-HF model/decode controls")
    command.extend(
        [
            "--task-args",
            json.dumps(task_args, sort_keys=True, separators=(",", ":")),
            "--model-args",
            json.dumps(expected_model_args, sort_keys=True, separators=(",", ":")),
            "--generation-config",
            json.dumps(GENERATION_CONFIG, sort_keys=True, separators=(",", ":")),
            "--log-dir",
            str(attempt),
            "--limit",
            str(_sample_count_for_task(task_index)),
            "--max-tasks",
            "1",
            "--isolate-tasks",
            "--task-index",
            str(task_index),
            "--yes",
        ]
    )
    return command


def worker(
    *,
    campaign_root: str | Path,
    training_repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
    condition_name: str,
    task_index: int,
    python: str,
) -> dict[str, Any]:
    """Run or resume one cell; never schedule or automatically retry another."""

    condition = _condition(condition_name)
    _sample_count_for_task(task_index)
    contract, paths = _load_campaign(
        campaign_root=campaign_root,
        training_repository=training_repository,
        source_stage2_manifest=source_stage2_manifest,
        stage2_artifact_root=stage2_artifact_root,
    )
    condition_paths = _condition_paths(paths, condition)
    _evaluation, evaluation_sha256 = _load_evaluation_receipt(contract, paths, condition)
    _runtime, runtime_sha256 = _load_runtime_receipt(contract, paths, condition)
    launch_sha256 = _sha256_file(paths.contract)
    if task_index in BIASED_TASK_INDICES:
        _require_clean_gate(
            contract=contract,
            paths=paths,
            condition=condition,
            condition_paths=condition_paths,
            launch_sha256=launch_sha256,
            runtime_sha256=runtime_sha256,
            evaluation_sha256=evaluation_sha256,
        )
    if _load_task_receipt(
        contract=contract,
        paths=paths,
        condition=condition,
        condition_paths=condition_paths,
        task_index=task_index,
        launch_sha256=launch_sha256,
        runtime_sha256=runtime_sha256,
        evaluation_sha256=evaluation_sha256,
    ) is not None:
        return {"condition": condition.name, "task_index": task_index, "status": "resumed", "promoted": False}
    attempt_root = condition_paths.attempts / f"task-{task_index:03d}"
    had_preserved_attempts = _has_preserved_attempts(attempt_root)
    promoted = _promote_prior_attempts(
        contract=contract,
        paths=paths,
        condition=condition,
        condition_paths=condition_paths,
        task_index=task_index,
        launch_sha256=launch_sha256,
        runtime_sha256=runtime_sha256,
        evaluation_sha256=evaluation_sha256,
    )
    if _load_task_receipt(
        contract=contract,
        paths=paths,
        condition=condition,
        condition_paths=condition_paths,
        task_index=task_index,
        launch_sha256=launch_sha256,
        runtime_sha256=runtime_sha256,
        evaluation_sha256=evaluation_sha256,
    ) is not None:
        return {"condition": condition.name, "task_index": task_index, "status": "promoted-prior", "promoted": promoted}
    if had_preserved_attempts:
        raise EvaluationError(
            f"{condition.name} task-{task_index} has a preserved incomplete attempt; "
            "refusing an automatic retry"
        )
    _validate_evaluator_environment(require_one_gpu=True)
    if not Path(python).exists() or not os.access(python, os.X_OK):
        raise EvaluationError("worker Python is unavailable or not executable")
    attempt = _next_attempt(condition_paths.attempts / f"task-{task_index:03d}", label="attempt")
    command = _task_command(
        python=python,
        contract=contract,
        condition=condition,
        condition_paths=condition_paths,
        task_index=task_index,
        attempt=attempt,
    )
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=os.environ.copy(), check=False)
    promoted_now = _promote_attempt(
        attempt=attempt,
        contract=contract,
        paths=paths,
        condition=condition,
        condition_paths=condition_paths,
        task_index=task_index,
        launch_sha256=launch_sha256,
        runtime_sha256=runtime_sha256,
        evaluation_sha256=evaluation_sha256,
    )
    if result.returncode:
        raise EvaluationError(f"{condition.name} task-{task_index} evaluator exited {result.returncode}; preserved attempt: {attempt}")
    if not promoted_now:
        raise EvaluationError(f"{condition.name} task-{task_index} exited successfully without a promotable EvalLog")
    return {"condition": condition.name, "task_index": task_index, "status": "generated", "promoted": True}


def seal_clean_gate(
    *,
    campaign_root: str | Path,
    training_repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
    condition_name: str,
) -> dict[str, Any]:
    """Seal a condition's three clean cells before bias tasks can be admitted."""

    condition = _condition(condition_name)
    contract, paths = _load_campaign(
        campaign_root=campaign_root,
        training_repository=training_repository,
        source_stage2_manifest=source_stage2_manifest,
        stage2_artifact_root=stage2_artifact_root,
    )
    condition_paths = _condition_paths(paths, condition)
    _evaluation, evaluation_sha256 = _load_evaluation_receipt(contract, paths, condition)
    _runtime, runtime_sha256 = _load_runtime_receipt(contract, paths, condition)
    launch_sha256 = _sha256_file(paths.contract)
    document = _seal_clean_gate(
        contract=contract,
        paths=paths,
        condition=condition,
        condition_paths=condition_paths,
        launch_sha256=launch_sha256,
        runtime_sha256=runtime_sha256,
        evaluation_sha256=evaluation_sha256,
    )
    return {
        "condition": condition.name,
        "clean_gate": _identity(condition_paths.clean_gate, label=f"{condition.name} clean-gate receipt"),
        "clean_task_indices": list(CLEAN_TASK_INDICES),
        "status": "sealed" if document else "unreachable",
    }


def _validate_canonical_raw_custody(
    *,
    contract: Mapping[str, Any],
    paths: CampaignPaths,
    condition: Condition,
    condition_paths: ConditionPaths,
    launch_sha256: str,
    runtime_sha256: str,
    evaluation_sha256: str,
) -> list[dict[str, Any]]:
    if condition_paths.raw.is_symlink() or not condition_paths.raw.is_dir():
        raise EvaluationError(f"{condition.name} canonical raw root must be a regular directory")
    receipts: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    for task_index in range(1, TASK_COUNT + 1):
        receipt = _load_task_receipt(
            contract=contract,
            paths=paths,
            condition=condition,
            condition_paths=condition_paths,
            task_index=task_index,
            launch_sha256=launch_sha256,
            runtime_sha256=runtime_sha256,
            evaluation_sha256=evaluation_sha256,
        )
        if receipt is None:
            raise EvaluationError(f"{condition.name} task-{task_index} is not sealed")
        canonical = receipt["canonical_log"]
        canonical_path = Path(str(canonical["path"])).resolve()
        _under_root(canonical_path, condition_paths.raw, label=f"{condition.name} canonical raw log")
        expected_paths.add(canonical_path)
        receipts.append(receipt)
    if len(expected_paths) != TASK_COUNT:
        raise EvaluationError(f"{condition.name} task receipts do not name 21 distinct canonical logs")
    found: set[Path] = set()
    for candidate in condition_paths.raw.rglob("*"):
        if candidate.is_symlink():
            raise EvaluationError(f"{condition.name} canonical raw tree contains a symlink: {candidate}")
        if candidate.is_file() and candidate.suffix == ".eval":
            found.add(_under_root(candidate, condition_paths.raw, label=f"{condition.name} raw EvalLog"))
    if found != expected_paths:
        raise EvaluationError(
            f"{condition.name} canonical raw tree differs from task receipts; "
            f"unexpected={sorted(str(item) for item in found - expected_paths)}, "
            f"missing={sorted(str(item) for item in expected_paths - found)}"
        )
    return receipts


def _validate_ordered_source_and_stored_ids(
    *,
    dataset_sample_ids: Any,
    samples: Sequence[Any],
    expected_ids: tuple[str, ...],
    expected_count: int,
    task_index: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Bind Inspect's two distinct representations of a selected ID subset.

    Inspect 0.3.258 retains the original selected source-prefix order in
    ``eval.dataset.sample_ids`` but writes ``full.samples`` in lexical sample
    ID order.  Treating the latter as source order would reject a correct raw
    log whenever the frozen question IDs are not already lexical.  Both views
    are therefore receipt-relevant: the dataset field proves prefix identity,
    while the stored samples must be its unique lexically sorted set.
    """

    if not isinstance(dataset_sample_ids, list):
        raise EvaluationError(f"task-{task_index} has no list-valued eval.dataset.sample_ids")
    source_ids = tuple(dataset_sample_ids)
    stored_ids = tuple(_value_field(sample, "id", "") for sample in samples)
    if (
        expected_count != len(expected_ids)
        or len(source_ids) != expected_count
        or len(stored_ids) != expected_count
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in source_ids)
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in stored_ids)
        or len(source_ids) != len(set(source_ids))
        or len(stored_ids) != len(set(stored_ids))
        or source_ids != expected_ids
        or set(stored_ids) != set(source_ids)
        or stored_ids != tuple(sorted(source_ids))
    ):
        raise EvaluationError(
            f"task-{task_index} does not retain the exact ordered source prefix and lexical stored-sample identity"
        )
    return source_ids, stored_ids


def _mixed_preflight(
    *,
    contract: Mapping[str, Any],
    paths: CampaignPaths,
    condition: Condition,
    condition_paths: ConditionPaths,
    launch_sha256: str,
    runtime_sha256: str,
    evaluation_sha256: str,
) -> dict[str, Any]:
    """Validate this condition's receipt-selected 50/100 raw matrix in full."""

    try:
        from experiments.stage2_ood_hle import raw_preflight as mechanical
        from experiments.stage2_ood_hle.tasks import ood_task_specs
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise EvaluationError("Stage-2 raw preflight primitives are unavailable") from exc
    _require_clean_gate(
        contract=contract,
        paths=paths,
        condition=condition,
        condition_paths=condition_paths,
        launch_sha256=launch_sha256,
        runtime_sha256=runtime_sha256,
        evaluation_sha256=evaluation_sha256,
    )
    _validate_canonical_raw_custody(
        contract=contract,
        paths=paths,
        condition=condition,
        condition_paths=condition_paths,
        launch_sha256=launch_sha256,
        runtime_sha256=runtime_sha256,
        evaluation_sha256=evaluation_sha256,
    )
    deployment = contract["deployment_manifest"]
    deployed = Path(str(deployment["path"]))
    row = _condition_row(contract, condition)
    runtime = row.get("runtime")
    if not isinstance(runtime, Mapping):
        raise EvaluationError("condition has no runtime record for raw preflight")
    try:
        mechanical.validate_manifest(deployed)
        specs = list(ood_task_specs(deployed))
        _validate_mixed_sampling_matrix(specs)
    except Exception as exc:
        raise EvaluationError(f"could not replay Stage-2 matrix for native-HF preflight: {exc}") from exc
    loaded: dict[int, Any] = {}
    sample_counts: dict[int, int] = {}
    source_prefix_ids: dict[int, tuple[str, ...]] = {}
    stored_sample_ids: dict[int, tuple[str, ...]] = {}
    for task_index, source_spec in enumerate(specs, start=1):
        receipt = _load_task_receipt(
            contract=contract,
            paths=paths,
            condition=condition,
            condition_paths=condition_paths,
            task_index=task_index,
            launch_sha256=launch_sha256,
            runtime_sha256=runtime_sha256,
            evaluation_sha256=evaluation_sha256,
        )
        if receipt is None:  # retained for a direct clear error
            raise EvaluationError(f"preflight has no sealed task-{task_index} receipt")
        path = Path(str(receipt["canonical_log"]["path"])).resolve()
        try:
            header = mechanical._read_eval_log(path, header_only=True)
            created = mechanical._validate_header(header, path=path, spec=source_spec, raw_root=condition_paths.raw)
            model, observed_runtime = _assert_hf_runtime(header, path=path, runtime=runtime)
        except Exception as exc:
            raise EvaluationError(f"could not validate task-{task_index} native-HF header/runtime: {exc}") from exc
        loaded[task_index] = mechanical.LoadedTaskLog(
            _subset_spec_for_task(source_spec, task_index=task_index), path, created, header, model, observed_runtime
        )
    clean_paths = {
        (loaded[task_index].spec.population, loaded[task_index].spec.dataset): loaded[task_index].path
        for task_index in CLEAN_TASK_INDICES
    }
    if len(clean_paths) != len(CLEAN_TASK_INDICES):
        raise EvaluationError("preflight has no distinct canonical clean log for each population")
    for task_index in range(1, TASK_COUNT + 1):
        item = loaded[task_index]
        try:
            full = mechanical._read_eval_log(item.path, header_only=False)
            count = mechanical._validate_samples(full, loaded=item, clean_paths=clean_paths)
        except Exception as exc:
            raise EvaluationError(f"task-{task_index} sample/switch validation failed: {exc}") from exc
        samples = list(mechanical._attribute(full, "samples", []) or [])
        evaluation = mechanical._attribute(full, "eval")
        dataset = mechanical._attribute(evaluation, "dataset")
        source_ids, stored_ids = _validate_ordered_source_and_stored_ids(
            dataset_sample_ids=mechanical._attribute(dataset, "sample_ids", None),
            samples=samples,
            expected_ids=tuple(item.spec.question_ids),
            expected_count=_sample_count_for_task(task_index),
            task_index=task_index,
        )
        if count != _sample_count_for_task(task_index):
            raise EvaluationError(f"task-{task_index} has a different mixed sample count after source/stored ID validation")
        sample_counts[task_index] = count
        source_prefix_ids[task_index] = source_ids
        stored_sample_ids[task_index] = stored_ids
        if item.spec.kind == "biased":
            for sample in samples:
                _switch, metadata = mechanical._switch_score(sample, path=item.path)
                for key, note in metadata.items():
                    if "missing" in str(key).lower() or "unmatched" in str(key).lower():
                        safe_note = note is None or note is False or note == "" or note == 0 or note == [] or note == {}
                        if not safe_note:
                            raise EvaluationError(f"task-{task_index} switch scorer reported missing/unmatched IDs in {key!r}")
    clean_source_prefix_ids = {
        (loaded[task_index].spec.population, loaded[task_index].spec.dataset): source_prefix_ids[task_index]
        for task_index in CLEAN_TASK_INDICES
    }
    clean_stored_sample_ids = {
        (loaded[task_index].spec.population, loaded[task_index].spec.dataset): stored_sample_ids[task_index]
        for task_index in CLEAN_TASK_INDICES
    }
    for task_index in BIASED_TASK_INDICES:
        item = loaded[task_index]
        population_dataset = (item.spec.population, item.spec.dataset)
        if (
            source_prefix_ids[task_index] != clean_source_prefix_ids.get(population_dataset)
            or stored_sample_ids[task_index] != clean_stored_sample_ids.get(population_dataset)
        ):
            raise EvaluationError(
                f"task-{task_index} does not use its matching clean source-prefix and stored-sample ID identities"
            )
    clean_records: dict[tuple[str, str], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for task_index in range(1, TASK_COUNT + 1):
        item = loaded[task_index]
        if item.spec.kind != "unbiased":
            continue
        record = mechanical._source_record(item, sample_count=sample_counts[task_index], clean_records={})
        record.update(
            {
                "task_index": task_index,
                "expected_sample_count": _sample_count_for_task(task_index),
                "task_receipt": _identity(_task_receipt_path(condition_paths, task_index), label=f"task-{task_index} receipt"),
                "evaluation_bias_status": None,
            }
        )
        clean_records[(item.spec.population, item.spec.dataset)] = record
        sources.append(record)
    for task_index in range(1, TASK_COUNT + 1):
        item = loaded[task_index]
        if item.spec.kind != "biased":
            continue
        record = mechanical._source_record(item, sample_count=sample_counts[task_index], clean_records=clean_records)
        record["unbiased_log"] = str(condition_paths.raw)
        record.update(
            {
                "task_index": task_index,
                "expected_sample_count": _sample_count_for_task(task_index),
                "task_receipt": _identity(_task_receipt_path(condition_paths, task_index), label=f"task-{task_index} receipt"),
                "evaluation_bias_status": "seen" if item.spec.bias_type in SEEN_BIASES else "held_out",
            }
        )
        sources.append(record)
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "condition": condition.name,
        "condition_artifact_name": condition.artifact_name,
        "evaluation_receipt": _identity(condition_paths.evaluation_receipt, label=f"{condition.name} evaluation receipt"),
        "runtime_receipt": _identity(condition_paths.runtime_receipt, label=f"{condition.name} runtime receipt"),
        "raw_root": str(condition_paths.raw),
        "clean_gate": _identity(condition_paths.clean_gate, label=f"{condition.name} clean-gate receipt"),
        "sampling": _sampling_contract(),
        "science": {
            "all_biases": list(ALL_BIASES),
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "include_bias_acknowledged": False,
            "grader_model": None,
        },
        "runtime_profile": runtime["profile"],
        "total_samples": sum(sample_counts.values()),
        "sources": sources,
    }
    validate_preflight_report(report)
    _write_immutable_json(condition_paths.preflight, report, label=f"{condition.name} native-HF mixed preflight")
    return report


def validate_preflight_report(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        report = dict(value)
    else:
        report = _read_json(value, label="native-HF mixed preflight")
    required = {
        "schema",
        "campaign",
        "condition",
        "condition_artifact_name",
        "evaluation_receipt",
        "runtime_receipt",
        "raw_root",
        "clean_gate",
        "sampling",
        "science",
        "runtime_profile",
        "total_samples",
        "sources",
    }
    if set(report) != required or report.get("schema") != PREFLIGHT_SCHEMA or report.get("campaign") != CAMPAIGN_NAME:
        raise EvaluationError("native-HF mixed preflight has an unsupported schema")
    condition = _condition(str(report.get("condition", "")))
    if report.get("condition_artifact_name") != condition.artifact_name:
        raise EvaluationError("native-HF mixed preflight condition identity differs from campaign topology")
    if report.get("sampling") != _sampling_contract() or report.get("total_samples") != TOTAL_SAMPLES_PER_CONDITION:
        raise EvaluationError("native-HF mixed preflight does not bind the exact 1,400-sample protocol")
    science = report.get("science")
    if science != {
        "all_biases": list(ALL_BIASES),
        "seen_biases": list(SEEN_BIASES),
        "held_out_biases": list(HELD_OUT_BIASES),
        "include_bias_acknowledged": False,
        "grader_model": None,
    }:
        raise EvaluationError("native-HF mixed preflight has changed two-bias science labels")
    if report.get("runtime_profile") not in {"hf-base", "hf-peft"}:
        raise EvaluationError("native-HF mixed preflight has an unsupported runtime profile")
    sources = report.get("sources")
    if not isinstance(sources, list) or len(sources) != TASK_COUNT:
        raise EvaluationError("native-HF mixed preflight must retain exactly 21 task sources")
    task_indices = [source.get("task_index") for source in sources if isinstance(source, Mapping)]
    if sorted(task_indices) != list(range(1, TASK_COUNT + 1)):
        raise EvaluationError("native-HF mixed preflight sources do not bind every task index exactly once")
    if sum(source.get("sample_count", -1) for source in sources if isinstance(source, Mapping)) != TOTAL_SAMPLES_PER_CONDITION:
        raise EvaluationError("native-HF mixed preflight source counts do not total 1,400")
    for source in sources:
        assert isinstance(source, Mapping)  # list check above leaves clear failure below
        task_index = source.get("task_index")
        if source.get("sample_count") != _sample_count_for_task(task_index) or source.get("expected_sample_count") != _sample_count_for_task(task_index):
            raise EvaluationError("native-HF mixed preflight source count differs from frozen matrix")
    return report


def finalize_condition(
    *,
    campaign_root: str | Path,
    training_repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
    condition_name: str,
) -> dict[str, Any]:
    """Require one condition's complete custody and seal its full raw preflight."""

    condition = _condition(condition_name)
    contract, paths = _load_campaign(
        campaign_root=campaign_root,
        training_repository=training_repository,
        source_stage2_manifest=source_stage2_manifest,
        stage2_artifact_root=stage2_artifact_root,
    )
    condition_paths = _condition_paths(paths, condition)
    _evaluation, evaluation_sha256 = _load_evaluation_receipt(contract, paths, condition)
    _runtime, runtime_sha256 = _load_runtime_receipt(contract, paths, condition)
    launch_sha256 = _sha256_file(paths.contract)
    report = _mixed_preflight(
        contract=contract,
        paths=paths,
        condition=condition,
        condition_paths=condition_paths,
        launch_sha256=launch_sha256,
        runtime_sha256=runtime_sha256,
        evaluation_sha256=evaluation_sha256,
    )
    return {
        "condition": condition.name,
        "preflight": _identity(condition_paths.preflight, label=f"{condition.name} native-HF mixed preflight"),
        "total_samples": report["total_samples"],
    }


def finalize(
    *,
    campaign_root: str | Path,
    training_repository: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
) -> dict[str, Any]:
    """Seal only after all four exact 1,400-sample condition preflights pass."""

    contract, paths = _load_campaign(
        campaign_root=campaign_root,
        training_repository=training_repository,
        source_stage2_manifest=source_stage2_manifest,
        stage2_artifact_root=stage2_artifact_root,
    )
    conditions: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        result = finalize_condition(
            campaign_root=campaign_root,
            training_repository=training_repository,
            source_stage2_manifest=source_stage2_manifest,
            stage2_artifact_root=stage2_artifact_root,
            condition_name=condition.name,
        )
        condition_paths = _condition_paths(paths, condition)
        conditions.append(
            {
                "name": condition.name,
                "optimizer_step": condition.optimizer_step,
                "runtime_receipt": _identity(condition_paths.runtime_receipt, label=f"{condition.name} runtime receipt"),
                "evaluation_receipt": _identity(condition_paths.evaluation_receipt, label=f"{condition.name} evaluation receipt"),
                "preflight": result["preflight"],
                "generations": result["total_samples"],
            }
        )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "launch_contract": _identity(paths.contract, label="all-HF campaign launch contract"),
        "model_snapshot": contract["model_snapshot"],
        "sampling": contract["sampling"],
        "conditions": conditions,
        "total_generations": TOTAL_SAMPLES_PER_CONDITION * len(CONDITIONS),
    }
    status = _write_immutable_json(paths.completion, completion, label="all-HF campaign completion receipt")
    return {"completion": _identity(paths.completion, label="all-HF campaign completion receipt"), "status": status}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def campaign_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--campaign-root", required=True, type=Path)
        command.add_argument("--training-repository", required=True, type=Path)
        command.add_argument("--source-stage2-manifest", required=True, type=Path)
        command.add_argument("--stage2-artifact-root", required=True, type=Path)

    prepare_parser = commands.add_parser("prepare", help="replay strict source custody and write campaign receipts")
    campaign_inputs(prepare_parser)
    prepare_parser.add_argument("--yes", action="store_true")

    worker_parser = commands.add_parser("worker", help="run one task on exactly one Slurm-visible GPU")
    campaign_inputs(worker_parser)
    worker_parser.add_argument("--condition", choices=[condition.name for condition in CONDITIONS], required=True)
    worker_parser.add_argument("--task-index", type=int, required=True)
    worker_parser.add_argument("--python", required=True)

    gate_parser = commands.add_parser("seal-clean-gate", help="require and receipt-seal the three clean cells")
    campaign_inputs(gate_parser)
    gate_parser.add_argument("--condition", choices=[condition.name for condition in CONDITIONS], required=True)

    condition_parser = commands.add_parser("finalize-condition", help="write one condition's full mixed-count preflight")
    campaign_inputs(condition_parser)
    condition_parser.add_argument("--condition", choices=[condition.name for condition in CONDITIONS], required=True)

    final_parser = commands.add_parser("finalize", help="seal all four native-HF condition preflights")
    campaign_inputs(final_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    inputs = {
        "campaign_root": args.campaign_root,
        "training_repository": args.training_repository,
        "source_stage2_manifest": args.source_stage2_manifest,
        "stage2_artifact_root": args.stage2_artifact_root,
    }
    try:
        if args.command == "prepare":
            if not args.yes:
                parser.error("prepare requires --yes after reviewing the four strict source targets")
            result = prepare(**inputs)
        elif args.command == "worker":
            result = worker(**inputs, condition_name=args.condition, task_index=args.task_index, python=args.python)
        elif args.command == "seal-clean-gate":
            result = seal_clean_gate(**inputs, condition_name=args.condition)
        elif args.command == "finalize-condition":
            result = finalize_condition(**inputs, condition_name=args.condition)
        elif args.command == "finalize":
            result = finalize(**inputs)
        else:  # pragma: no cover - argparse enforces choices
            parser.error("unsupported command")
            return 2
    except (EvaluationError, FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "ALL_BIASES",
    "BIASED_WAVES",
    "CAMPAIGN_NAME",
    "CLEAN_WAVE",
    "CLEAN_TASK_INDICES",
    "CONDITIONS",
    "EVALUATOR_PACKAGE_VERSIONS",
    "EvaluationError",
    "GENERATION_CONFIG",
    "HF_LOCAL_MODEL_ARGS",
    "HF_MODEL_ARGS",
    "INSPECT_VERSION",
    "TASK_COUNT",
    "TOTAL_SAMPLES_PER_CONDITION",
    "_sampling_contract",
    "_task_command",
    "_topology_contract",
    "build_launch_contract",
    "finalize",
    "finalize_condition",
    "prepare",
    "seal_clean_gate",
    "validate_preflight_report",
    "worker",
]
