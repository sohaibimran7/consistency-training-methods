#!/usr/bin/env python3
"""Launch the uncapped Gemma 4 12B base 21-cell suitability evaluation.

This is deliberately a base-only launcher.  It evaluates the frozen Stage-2
two-bias matrix on 50 questions in every clean and biased cell:

* 3 datasets × (1 clean + 6 bias variants) = 21 cells;
* 50 questions per cell = 1,050 model generations; and
* every cell is deterministically split across all 16 one-GPU workers.

Each worker receives one shard from every cell.  The rotating shard assignment
gives every GPU either 65 or 66 questions.  A biased shard deliberately carries
only its generation scorers: its task-index rotation means that its local clean
shard is not, in general, the same question-ID set.  After all 16 workers
finish, ``merge`` reconstitutes the 50-question cells and applies the installed
standard ``mcq_bias.switch_scorer`` on CPU against the one exact, matching
merged clean EvalLog.  The original per-rank generation logs remain immutable
evidence throughout.

The only sampling route is the generic native-HF EOS-only hook.  It is
activated while the task factory is built, before Inspect constructs the Gemma
model, and terminates only when Gemma emits EOS.

This module never talks to Slurm.  The accompanying sbatch wrappers are the
only scheduler boundary, and no job is submitted by this module.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CAMPAIGN_NAME = "gemma4-12b-base-two-bias-50x21-16gpu-v1"
CONFIG_PATH = PROJECT_ROOT / "infra/isambard/gemma4_12b_base_two_bias_eval_config.json"
LAUNCH_SCHEMA = "gemma4-12b-base-two-bias-launch-v1"
SHARD_RECEIPT_SCHEMA = "gemma4-12b-base-two-bias-shard-receipt-v1"
MERGE_RECEIPT_SCHEMA = "gemma4-12b-base-two-bias-merge-receipt-v1"
SMOKE_RECEIPT_SCHEMA = "gemma4-12b-base-two-bias-discarded-smoke-v1"
SWITCH_SCORE_SCHEMA = "gemma4-12b-base-standard-switch-score-v1"

MODEL_ID = "google/gemma-4-12B-it"
MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
MODEL_TYPE = "gemma4_unified"
TASK_COUNT = 21
QUESTIONS_PER_CELL = 50
SHARD_COUNT = 16
TOTAL_GENERATIONS = TASK_COUNT * QUESTIONS_PER_CELL
INSPECT_RESCORE_MODEL = "mockllm/model"
SWITCH_SCORER_NAME = "switch_scorer"
SWITCH_SCORE_KEYS = frozenset(
    {
        "unbiased_matches_bias",
        "towards_bias_switch",
        "away_from_bias_switch",
        "net_switch",
        "abs_switch",
    }
)
# ``stage2_ood_biased`` currently requires a non-empty ``unbiased_log`` to
# build its task.  This sentinel is removed from the Task and never reaches a
# generated EvalLog; it exists only while the upstream factory is building the
# scorer list that we then validate and strip.
_GENERATION_ONLY_SWITCH_SENTINEL = "__gemma_post_merge_switch_scoring__"

GENERATION_CONFIG: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "max_connections": 8,
}
MODEL_ARGS: dict[str, Any] = {
    "device": "cuda:0",
    "dtype": "bfloat16",
    "do_sample": True,
    "gemma4_unified_processor": True,
}
TASK_FACTORY = "infra.isambard.run_gemma4_12b_base_two_bias_evals_16gpu:gemma_stage2_shard_tasks"
SMOKE_TASK_FACTORY = "infra.isambard.run_gemma4_12b_base_two_bias_evals_16gpu:gemma_stage2_discarded_smoke_task"
RUNTIME_REQUIREMENTS = {
    "inspect-ai": "0.3.260",
    "torch": "2.13.0+cu129",
    "transformers": "5.15.1",
}


class GemmaEvaluationError(ValueError):
    """A campaign input, runtime, or emitted evidence is unsuitable."""


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    for name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, name, None)
        if callable(method):
            candidate = method()
            if isinstance(candidate, Mapping):
                return dict(candidate)
    return {}


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size < 1:
        raise GemmaEvaluationError(f"{label} must be a non-empty regular file: {candidate}")
    resolved = candidate.resolve()
    return {"path": str(resolved), "sha256": _sha256_file(resolved), "size_bytes": resolved.stat().st_size}


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise GemmaEvaluationError(f"{label} must be a regular file: {candidate}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GemmaEvaluationError(f"invalid {label}: {candidate}") from exc
    if not isinstance(value, dict):
        raise GemmaEvaluationError(f"{label} must contain an object: {candidate}")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_once_json(path: Path, value: Mapping[str, Any], *, label: str) -> str:
    """Atomically publish one receipt, allowing only byte-identical replay."""

    payload = _canonical_json(value)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing {label}: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise GemmaEvaluationError(f"{label} parent is unsafe: {path.parent}")
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


def _write_hashed_eval_once(log: Any, directory: Path, *, label: str) -> tuple[Path, str]:
    """Write a derived EvalLog under its content hash without duplicate aliases."""

    from inspect_ai.log import write_eval_log

    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise GemmaEvaluationError(f"{label} directory is unsafe: {directory}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".merge-", suffix=".eval", dir=directory)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_eval_log(log, str(temporary))
        digest = _sha256_file(temporary)
        path = directory / f"{digest}.eval"
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
                raise FileExistsError(f"refusing to overwrite differing {label}: {path}")
            return path.resolve(), "resumed"
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
                raise FileExistsError(f"{label} appeared with different bytes: {path}") from None
            return path.resolve(), "resumed"
    finally:
        temporary.unlink(missing_ok=True)
    return path.resolve(), "written"


def _under(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise GemmaEvaluationError(f"{label} escapes its expected root: {resolved}") from exc
    return resolved


def _campaign_root(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
        raise GemmaEvaluationError(f"campaign root must be a regular directory: {candidate}")
    root = candidate.resolve()
    if root.name != CAMPAIGN_NAME:
        raise GemmaEvaluationError(f"campaign root must be named {CAMPAIGN_NAME!r}; got {root.name!r}")
    return root


def _paths(campaign_root: str | Path) -> dict[str, Path]:
    root = _campaign_root(campaign_root)
    return {
        "root": root,
        "contract": root / "launch-contract.json",
        "deployment_manifest": root / "input" / "stage2-deployment-manifest.json",
        "live": root / "live-shards",
        "shard_receipts": root / "live-shards" / "receipts",
        "merged": root / "merged",
        "merge_receipts": root / "merged" / "receipts",
        "smoke": root / "discarded-smoke",
    }


def _load_static_config() -> dict[str, Any]:
    document = _read_json(CONFIG_PATH, label="Gemma evaluation configuration")
    expected = {
        "schema": "gemma4-12b-base-two-bias-evaluation-v1",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "architecture": MODEL_TYPE,
            "text_route": "processor_preserving_native_hf",
        },
    }
    for field, value in expected.items():
        if document.get(field) != value:
            raise GemmaEvaluationError(f"Gemma configuration changed protected {field!r}")
    matrix = _mapping(document.get("matrix"))
    if (
        matrix.get("questions_per_dataset_cell") != QUESTIONS_PER_CELL
        or matrix.get("cells") != TASK_COUNT
        or matrix.get("total_generations") != TOTAL_GENERATIONS
    ):
        raise GemmaEvaluationError("Gemma configuration has the wrong fixed 1,050-generation matrix")
    if _mapping(document.get("sampling")) != {**GENERATION_CONFIG, "termination": "model_eos_only"}:
        raise GemmaEvaluationError("Gemma configuration changed the frozen decode controls")
    if _mapping(document.get("switch_scoring")) != {
        "mode": "post_merge_cpu_standard_inspect",
        "scorer": "mcq_bias/switch_scorer",
        "rescore_model": INSPECT_RESCORE_MODEL,
        "paired_clean": "exact_matching_merged_eval_log",
        "raw_shards": "immutable",
    }:
        raise GemmaEvaluationError("Gemma configuration has an invalid post-merge switch-scoring policy")
    if _mapping(document.get("verbalisation")).get("max_connections") != 500:
        raise GemmaEvaluationError("Gemma configuration does not require Luna at 500 connections")
    if _mapping(document.get("publication")) != {
        "renderer": "ctm_standard_publication_renderer",
        "significance": "not_applicable_base_only",
        "intervals": "wilson_with_sample_counts",
    }:
        raise GemmaEvaluationError("Gemma base-only configuration has an invalid publication policy")
    return document


def _require_eos_runtime(*, require_gpu: bool) -> dict[str, Any]:
    """Validate the existing generic EOS-only runtime before use."""

    expected_environment = {
        "CTM_HF_EOS_ONLY_NO_TOKEN_CAP": "1",
        "CTM_HF_EOS_ONLY_EXPECTED_INSPECT": RUNTIME_REQUIREMENTS["inspect-ai"],
        "CTM_HF_EOS_ONLY_EXPECTED_TRANSFORMERS": RUNTIME_REQUIREMENTS["transformers"],
    }
    if any(os.environ.get(name) != value for name, value in expected_environment.items()):
        raise GemmaEvaluationError("Gemma evaluator lacks the required generic EOS-only environment")
    installed = {name: importlib.metadata.version(name) for name in RUNTIME_REQUIREMENTS}
    if installed != RUNTIME_REQUIREMENTS:
        raise GemmaEvaluationError(f"Gemma evaluator package versions differ: {installed!r}")
    if require_gpu:
        visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
        if len(visible) != 1 or visible[0] in {"-1", "NoDevFiles"}:
            raise GemmaEvaluationError("each Gemma worker requires exactly one Slurm-visible GPU")
    from ctm.evals.hf_eos_only import runtime_policy

    policy = runtime_policy()
    if policy.get("termination") != "model_eos_only":
        raise GemmaEvaluationError("generic native-HF termination is not EOS-only")
    return {"packages": installed, "eos_only_policy": policy}


def _snapshot_path() -> Path:
    raw = os.environ.get("CTM_GEMMA4_12B_SNAPSHOT", "")
    if not raw:
        raise GemmaEvaluationError("CTM_GEMMA4_12B_SNAPSHOT must name the pinned local Gemma snapshot")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise GemmaEvaluationError("CTM_GEMMA4_12B_SNAPSHOT must be an absolute regular directory")
    snapshot = candidate.resolve()
    if snapshot.name != MODEL_REVISION:
        raise GemmaEvaluationError("Gemma snapshot directory does not match the pinned revision")
    return snapshot


def _snapshot_logical_identity(snapshot: Path, logical: Path, *, label: str) -> dict[str, Any]:
    """Bind one snapshot path while permitting HF's content-addressed links."""

    if not logical.exists():
        raise GemmaEvaluationError(f"pinned Gemma snapshot lacks {label}: {logical}")
    if logical.is_symlink():
        blobs = snapshot.parent.parent / "blobs"
        if blobs.is_symlink() or not blobs.is_dir():
            raise GemmaEvaluationError("pinned Gemma snapshot has no regular Hugging Face blob tree")
        resolved = logical.resolve()
        if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size < 1:
            raise GemmaEvaluationError(f"pinned Gemma {label} resolves to an invalid blob")
        _under(resolved, blobs, label=f"pinned Gemma {label} blob")
    else:
        resolved = logical.resolve()
        if not resolved.is_file() or resolved.stat().st_size < 1:
            raise GemmaEvaluationError(f"pinned Gemma {label} is not a non-empty regular file")
        _under(resolved, snapshot, label=f"pinned Gemma {label}")
    return {
        "logical_path": str(logical),
        "resolved_path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _snapshot_weight_identity(snapshot: Path, logical: Path, *, label: str) -> dict[str, Any]:
    """Bind large shards by HF blob address and size without rehashing them."""

    if not logical.exists():
        raise GemmaEvaluationError(f"pinned Gemma snapshot lacks {label}: {logical}")
    if logical.is_symlink():
        blobs = snapshot.parent.parent / "blobs"
        if blobs.is_symlink() or not blobs.is_dir():
            raise GemmaEvaluationError("pinned Gemma snapshot has no regular Hugging Face blob tree")
        resolved = logical.resolve()
        if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size < 1:
            raise GemmaEvaluationError(f"pinned Gemma {label} resolves to an invalid blob")
        _under(resolved, blobs, label=f"pinned Gemma {label} blob")
        content_address = resolved.name
        if len(content_address) != 64 or any(character not in "0123456789abcdef" for character in content_address.lower()):
            raise GemmaEvaluationError(f"pinned Gemma {label} blob is not content-addressed")
        return {
            "logical_path": str(logical),
            "resolved_path": str(resolved),
            "content_address": content_address,
            "size_bytes": resolved.stat().st_size,
        }
    resolved = logical.resolve()
    if not resolved.is_file() or resolved.stat().st_size < 1:
        raise GemmaEvaluationError(f"pinned Gemma {label} is not a non-empty regular file")
    _under(resolved, snapshot, label=f"pinned Gemma {label}")
    return {
        "logical_path": str(logical),
        "resolved_path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _snapshot_identity() -> dict[str, Any]:
    snapshot = _snapshot_path()
    required_assets = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
    )
    assets = {
        name: _snapshot_logical_identity(snapshot, snapshot / name, label=name)
        for name in required_assets
    }
    config = _read_json(Path(assets["config.json"]["resolved_path"]), label="Gemma config")
    if config.get("model_type") != MODEL_TYPE:
        raise GemmaEvaluationError("pinned Gemma snapshot is not Gemma 4 Unified")
    weights = sorted(path for path in snapshot.glob("*.safetensors") if path.is_file() or path.is_symlink())
    if not weights:
        raise GemmaEvaluationError("pinned Gemma snapshot has no safetensors weights")
    weight_records = [_snapshot_weight_identity(snapshot, logical, label=logical.name) for logical in weights]
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "path": str(snapshot),
        "model_type": MODEL_TYPE,
        "processor_route": "AutoProcessor_plus_AutoModelForImageTextToText",
        "assets": assets,
        "weight_files": weight_records,
        "weight_identity_policy": "hf_blob_content_address_and_size",
        "saved_generation_defaults_execution": "not-consulted_by_eos_only_sampler",
    }


def _load_specs(manifest: str | Path) -> list[Any]:
    try:
        from experiments.rmct_two_bias_eval import deployment
        from experiments.stage2_ood_hle.tasks import ood_task_specs
    except ImportError as exc:  # pragma: no cover - configured evaluation runtime
        raise GemmaEvaluationError("Gemma Stage-2 task substrate is unavailable") from exc
    path = Path(manifest).resolve()
    try:
        deployment.validate_deployment_manifest(path)
        specs = list(ood_task_specs(path))
    except Exception as exc:
        raise GemmaEvaluationError(f"Gemma Stage-2 substrate validation failed: {exc}") from exc
    if len(specs) != TASK_COUNT:
        raise GemmaEvaluationError("Gemma Stage-2 substrate does not contain 21 cells")
    by_population: dict[tuple[str, str], tuple[str, ...]] = {}
    for task_index, spec in enumerate(specs, start=1):
        ids = tuple(getattr(spec, "question_ids", ()))
        key = (str(getattr(spec, "population", "")), str(getattr(spec, "dataset", "")))
        if len(ids) != 100 or len(ids) != len(set(ids)) or any(not isinstance(value, str) or not value for value in ids):
            raise GemmaEvaluationError(f"Gemma task-{task_index:03d} has an invalid frozen 100-ID pool")
        earlier = by_population.setdefault(key, ids)
        if earlier != ids:
            raise GemmaEvaluationError(f"Gemma paired variants do not share an ordered source ID pool for {key}")
    return specs


def _shard_ids(ids: Sequence[str], *, task_index: int, shard_index: int) -> tuple[str, ...]:
    if not 1 <= task_index <= TASK_COUNT or not 0 <= shard_index < SHARD_COUNT:
        raise GemmaEvaluationError("task or shard index is outside the fixed 21×16 topology")
    # Every 50-question cell has 32 baseline three-question assignments and
    # two fourth questions.  Advancing those two surplus positions by *two*
    # ranks per cell cycles evenly through all 16 ranks: across 21 cells, ten
    # ranks receive three surplus questions and six receive two, yielding only
    # 65- or 66-generation workers.
    offset = 2 * (task_index - 1)
    selected = tuple(
        question_id
        for position, question_id in enumerate(ids[:QUESTIONS_PER_CELL])
        if (position + offset) % SHARD_COUNT == shard_index
    )
    if len(selected) not in {3, 4}:
        raise GemmaEvaluationError("Gemma deterministic shard does not contain three or four questions")
    return selected


def _discarded_smoke_ids(ids: Sequence[str]) -> tuple[str, str]:
    """Select two frozen prompts that cannot overlap the scored first-50 pool."""

    selected = tuple(ids[QUESTIONS_PER_CELL : QUESTIONS_PER_CELL + 2])
    scored = set(ids[:QUESTIONS_PER_CELL])
    if len(selected) != 2 or len(set(selected)) != 2 or any(value in scored for value in selected):
        raise GemmaEvaluationError(
            "discarded Gemma smoke requires two unique frozen IDs outside the scored 50-question pool"
        )
    return selected


def _topology(specs: Sequence[Any]) -> dict[str, Any]:
    if len(specs) != TASK_COUNT:
        raise GemmaEvaluationError("topology requires the full 21-cell matrix")
    per_rank: list[dict[str, Any]] = []
    all_shards: list[dict[str, Any]] = []
    for rank in range(SHARD_COUNT):
        assignments: list[dict[str, Any]] = []
        for task_index, spec in enumerate(specs, start=1):
            count = len(_shard_ids(tuple(spec.question_ids), task_index=task_index, shard_index=rank))
            assignments.append({"task_index": task_index, "sample_count": count})
            all_shards.append({"rank": rank, "task_index": task_index, "sample_count": count})
        per_rank.append({"rank": rank, "generation_count": sum(row["sample_count"] for row in assignments), "assignments": assignments})
    workloads = [row["generation_count"] for row in per_rank]
    if sorted(set(workloads)) != [65, 66] or sum(workloads) != TOTAL_GENERATIONS:
        raise GemmaEvaluationError("deterministic 16-way topology does not exactly balance 1,050 generations")
    if any(sum(row["sample_count"] for row in all_shards if row["task_index"] == task_index) != QUESTIONS_PER_CELL for task_index in range(1, TASK_COUNT + 1)):
        raise GemmaEvaluationError("deterministic topology does not cover every 50-question cell exactly once")
    return {
        "nodes": 4,
        "gpus_per_node": 4,
        "workers": SHARD_COUNT,
        "one_gpu_per_worker": True,
        "question_sharding": "rotated_modulo_source_position",
        "per_rank_execution": "clean_tasks_then_biased_tasks_serially",
        "cross_rank_phase_overlap_possible": True,
        "biased_generation_has_live_switch_scorer": False,
        "switch_scoring": "post_merge_cpu_exact_canonical_clean",
        "rank_workloads": per_rank,
    }


def _spec_record(task_index: int, spec: Any) -> dict[str, Any]:
    ids = tuple(spec.question_ids)[:QUESTIONS_PER_CELL]
    digest = hashlib.sha256(json.dumps(list(ids), separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "task_index": task_index,
        "kind": str(spec.kind),
        "regime": str(spec.regime),
        "population": str(spec.population),
        "dataset": str(spec.dataset),
        "bias_type": spec.bias_type,
        "sample_count": QUESTIONS_PER_CELL,
        "full_question_ids_sha256": digest,
    }


def _stable_deployment_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the write/resume action while retaining deployment identity."""

    stable = dict(record)
    status = stable.pop("status", None)
    if status not in {"written", "resumed"}:
        raise GemmaEvaluationError(f"Gemma deployment returned an invalid materialization status: {status!r}")
    return stable


def _generation_implementation() -> dict[str, Any]:
    """Bind the decoder and launch path, not just model/package versions."""

    sources = (
        "experiments/elephant_aita_ntaflip/no_cap_hf.py",
        "ctm/evals/hf_eos_only.py",
        "ctm/evals/local_model.py",
        "ctm/evals/runner.py",
        "infra/isambard/run_gemma4_12b_base_two_bias_evals_16gpu.py",
        "infra/isambard/run_gemma4_12b_base_two_bias_evals_16gpu.sbatch",
        "infra/isambard/run_gemma4_12b_base_two_bias_evals_16gpu_worker.sh",
        "infra/isambard/run_gemma4_12b_base_two_bias_smoke.sbatch",
        "infra/isambard/gemma_gpu_binding.py",
        "infra/isambard/trace_gemma_hf_loads.py",
    )
    return {name: _identity(PROJECT_ROOT / name, label=f"Gemma generation source {name}") for name in sources}


def _require_generation_implementation(contract: Mapping[str, Any]) -> None:
    if contract.get("generation_implementation") != _generation_implementation():
        raise GemmaEvaluationError("Gemma generation implementation changed or is not receipt-bound; prepare a fresh attempt")


def prepare(
    *,
    campaign_root: str | Path,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
) -> dict[str, Any]:
    """Materialize the frozen substrate and seal the no-generation contract."""

    from experiments.rmct_two_bias_eval import deployment

    paths = _paths(campaign_root)
    _load_static_config()
    source = Path(source_stage2_manifest).expanduser().resolve()
    artifact_root = Path(stage2_artifact_root).expanduser().resolve()
    if source.is_symlink() or not source.is_file() or artifact_root.is_symlink() or not artifact_root.is_dir():
        raise GemmaEvaluationError("source Stage-2 manifest and artifact root must be regular inputs")
    if paths["root"].exists() and not paths["contract"].exists() and (paths["live"].exists() or paths["merged"].exists()):
        raise FileExistsError("refusing to seed Gemma custody beside existing shard or merged evidence")
    deployment_record = deployment.materialize_deployment_manifest(source, artifact_root, paths["deployment_manifest"])
    deployment_identity = _stable_deployment_record(deployment_record)
    specs = _load_specs(paths["deployment_manifest"])
    runtime = _require_eos_runtime(require_gpu=False)
    contract = {
        "schema": LAUNCH_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "configuration": _identity(CONFIG_PATH, label="Gemma evaluation configuration"),
        "model_snapshot": _snapshot_identity(),
        "runtime": runtime,
        "generation_implementation": _generation_implementation(),
        "model_args": dict(MODEL_ARGS),
        "sampling": {**GENERATION_CONFIG, "termination": "model_eos_only"},
        "stage2": {
            "source_manifest": _identity(source, label="Gemma source Stage-2 manifest"),
            "artifact_root": str(artifact_root),
            "deployment": deployment_identity,
            "tasks": [_spec_record(index, spec) for index, spec in enumerate(specs, start=1)],
        },
        "topology": _topology(specs),
        "policy": {
            "self_submits": False,
            "automatic_retries": False,
            "partial_attempts_preserved": True,
            "per_rank_clean_before_biased": True,
            "cross_rank_clean_biased_overlap_possible": True,
            "biased_generation_has_live_switch_scorer": False,
            "post_merge_switch_scoring": "standard_inspect_cpu_exact_canonical_clean",
            "post_merge_switch_rescore_model": INSPECT_RESCORE_MODEL,
            "raw_shards_immutable_after_generation": True,
            "termination": "model_eos_only",
            "discarded_one_gpu_smoke_required_before_campaign": True,
            "discarded_smoke_single_model_load_trace_required": True,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    status = _write_once_json(paths["contract"], contract, label="Gemma launch contract")
    return {"status": status, "launch_contract": _identity(paths["contract"], label="Gemma launch contract"), "total_generations": TOTAL_GENERATIONS}


def _load_contract(campaign_root: str | Path) -> tuple[dict[str, Any], dict[str, Path], list[Any]]:
    paths = _paths(campaign_root)
    contract = _read_json(paths["contract"], label="Gemma launch contract")
    if contract.get("schema") != LAUNCH_SCHEMA or contract.get("campaign") != CAMPAIGN_NAME:
        raise GemmaEvaluationError("Gemma launch contract has the wrong identity")
    configuration = contract.get("configuration")
    if configuration != _identity(CONFIG_PATH, label="Gemma evaluation configuration"):
        raise GemmaEvaluationError("Gemma launch contract configuration changed")
    snapshot = contract.get("model_snapshot")
    if not isinstance(snapshot, Mapping) or dict(snapshot) != _snapshot_identity():
        raise GemmaEvaluationError("Gemma launch contract snapshot changed")
    deployment = _mapping(_mapping(contract.get("stage2")).get("deployment"))
    deployed = deployment.get("manifest_path")
    if not isinstance(deployed, str) or Path(deployed).resolve() != paths["deployment_manifest"].resolve():
        raise GemmaEvaluationError("Gemma launch contract has a different deployed Stage-2 manifest")
    specs = _load_specs(paths["deployment_manifest"])
    if contract.get("topology") != _topology(specs):
        raise GemmaEvaluationError("Gemma launch topology changed")
    return contract, paths, specs


def _install_eos_sampler() -> None:
    from ctm.evals.hf_eos_only import install_native_hf_eos_only_sampling

    install_native_hf_eos_only_sampling()


def _set_task_metadata(task: Any, *, task_index: int, shard_index: int, sample_ids: Sequence[str], smoke: bool) -> Any:
    metadata = _mapping(getattr(task, "metadata", {}))
    metadata["gemma4_12b_base_eval"] = {
        "schema": LAUNCH_SCHEMA,
        "task_index": task_index,
        "shard_index": shard_index,
        "shard_count": SHARD_COUNT,
        "sample_count": len(sample_ids),
        "smoke": smoke,
        "termination": "model_eos_only",
    }
    task.metadata = metadata
    return task


def _strip_live_switch_scorer(task: Any) -> Any:
    """Retain only generation-time scorers on one frozen biased task.

    The upstream Stage-2 factory attaches its standard switch scorer whenever
    it receives a non-empty clean-log argument.  That is useful for an
    unsharded run, but unsafe here: rotated task-index shards deliberately do
    not line up rank-by-rank.  Validate the exact upstream scorer set before
    removing only that scorer, and erase the construction-only sentinel from
    all serialized task metadata.
    """

    scorers = getattr(task, "scorer", None)
    if not isinstance(scorers, Sequence) or isinstance(scorers, (str, bytes)):
        raise GemmaEvaluationError("Gemma biased task has no inspectable scorer sequence")
    try:
        from inspect_ai.scorer._scorer import as_scorer_spec
    except ImportError as exc:  # pragma: no cover - runtime boundary
        raise RuntimeError("Inspect AI is required to inspect Gemma task scorers") from exc

    retained: list[Any] = []
    retained_names: list[str] = []
    removed = 0
    for scorer in scorers:
        name = getattr(as_scorer_spec(scorer), "scorer", None)
        if not isinstance(name, str) or not name:
            raise GemmaEvaluationError("Gemma biased task has an unregistered scorer")
        if name == SWITCH_SCORER_NAME:
            removed += 1
        else:
            retained.append(scorer)
            retained_names.append(name)
    if removed != 1 or set(retained_names) != {"mcq_bias_scorer", "options_considered_scorer"}:
        raise GemmaEvaluationError(
            "Gemma biased task must contain exactly the two generation scorers and one live switch scorer"
        )
    task.scorer = retained

    # A real Inspect ``Task`` constructed by the inner Stage-2 factory does
    # not carry the outer ``@task`` factory arguments in ``task_args`` yet;
    # Inspect attaches those only when it serializes the top-level factory
    # invocation.  Do not invent that absent field.  If an Inspect release (or
    # a test double) does expose task_args already, scrub the sentinel there
    # too, while refusing any conflicting serialized dependency.
    task_args = getattr(task, "task_args", None)
    if task_args is not None:
        if not isinstance(task_args, Mapping):
            raise GemmaEvaluationError("Gemma biased task has non-mapping task_args")
        if "unbiased_log" in task_args:
            if task_args.get("unbiased_log") != _GENERATION_ONLY_SWITCH_SENTINEL:
                raise GemmaEvaluationError("Gemma biased task has a conflicting switch-log dependency")
            scrubbed_args = dict(task_args)
            scrubbed_args.pop("unbiased_log", None)
            task.task_args = scrubbed_args

    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, Mapping) or metadata.get("unbiased_log") != _GENERATION_ONLY_SWITCH_SENTINEL:
        raise GemmaEvaluationError("Gemma biased task metadata lacks its expected switch-log sentinel")
    scrubbed_metadata = dict(metadata)
    scrubbed_metadata.pop("unbiased_log", None)
    task.metadata = scrubbed_metadata
    return task


def gemma_stage2_shard_tasks(
    *,
    manifest: str,
    shard_index: int,
    shard_count: int = SHARD_COUNT,
    prompt_style: str = "none",
    hf_eos_only_no_token_cap: bool = True,
) -> list[Any]:
    """Build all 21 source-ID shards for one of the 16 Gemma workers."""

    if shard_count != SHARD_COUNT or not isinstance(shard_index, int) or not 0 <= shard_index < SHARD_COUNT:
        raise GemmaEvaluationError("Gemma task factory requires one fixed shard in [0, 15]")
    if prompt_style != "none" or hf_eos_only_no_token_cap is not True:
        raise GemmaEvaluationError("Gemma task factory requires the EOS-only no-style protocol")
    _install_eos_sampler()
    from experiments.stage2_ood_hle.tasks import stage2_ood_biased, stage2_ood_unbiased

    specs = _load_specs(manifest)
    tasks: list[Any] = []
    for task_index, spec in enumerate(specs, start=1):
        ids = _shard_ids(tuple(spec.question_ids), task_index=task_index, shard_index=shard_index)
        common = {
            "frozen_file": spec.frozen_file,
            "dataset": spec.dataset,
            "regime": spec.regime,
            "population": spec.population,
            "question_ids_from": list(ids),
            "source_identity_digest": spec.source_identity_digest,
            "prompt_style": prompt_style,
        }
        if spec.kind == "unbiased":
            task = stage2_ood_unbiased(**common)
        elif spec.kind == "biased":
            task = stage2_ood_biased(
                **common,
                bias_type=spec.bias_type,
                unbiased_log=_GENERATION_ONLY_SWITCH_SENTINEL,
                include_bias_acknowledged=False,
                grader_model=None,
            )
            task = _strip_live_switch_scorer(task)
        else:  # pragma: no cover - Stage-2 contract prevents this
            raise GemmaEvaluationError(f"Gemma task-{task_index:03d} has unsupported kind {spec.kind!r}")
        tasks.append(_set_task_metadata(task, task_index=task_index, shard_index=shard_index, sample_ids=ids, smoke=False))
    if len(tasks) != TASK_COUNT:
        raise GemmaEvaluationError("Gemma worker did not create all 21 sharded cells")
    return tasks


def gemma_stage2_discarded_smoke_task(
    *,
    manifest: str,
    prompt_style: str = "none",
    hf_eos_only_no_token_cap: bool = True,
) -> Any:
    """Build an uncapped two-question clean smoke task that is never merged."""

    if prompt_style != "none" or hf_eos_only_no_token_cap is not True:
        raise GemmaEvaluationError("discarded Gemma smoke requires the EOS-only no-style protocol")
    _install_eos_sampler()
    from experiments.stage2_ood_hle.tasks import stage2_ood_unbiased

    spec = _load_specs(manifest)[0]
    ids = _discarded_smoke_ids(tuple(spec.question_ids))
    task = stage2_ood_unbiased(
        frozen_file=spec.frozen_file,
        dataset=spec.dataset,
        regime=spec.regime,
        population=spec.population,
        question_ids_from=list(ids),
        source_identity_digest=spec.source_identity_digest,
        prompt_style=prompt_style,
    )
    return _set_task_metadata(task, task_index=1, shard_index=0, sample_ids=ids, smoke=True)


def _command_for_worker(*, python: str, contract: Mapping[str, Any], paths: Mapping[str, Path], rank: int, log_dir: Path) -> list[str]:
    runtime = _mapping(contract.get("model_snapshot"))
    snapshot = runtime.get("path")
    if not isinstance(snapshot, str) or Path(snapshot).resolve() != _snapshot_path():
        raise GemmaEvaluationError("Gemma contract has no current pinned snapshot path")
    task_args = {
        "manifest": str(paths["deployment_manifest"]),
        "shard_index": rank,
        "shard_count": SHARD_COUNT,
        "prompt_style": "none",
        "hf_eos_only_no_token_cap": True,
    }
    command = [
        python,
        str(PROJECT_ROOT / "scripts/run_evals.py"),
        "--task-factory",
        TASK_FACTORY,
        "--model",
        f"hf/{snapshot}",
        "--task-args",
        json.dumps(task_args, sort_keys=True, separators=(",", ":")),
        "--model-args",
        json.dumps(MODEL_ARGS, sort_keys=True, separators=(",", ":")),
        "--generation-config",
        json.dumps(GENERATION_CONFIG, sort_keys=True, separators=(",", ":")),
        "--log-dir",
        str(log_dir),
        "--max-tasks",
        "1",
        "--yes",
    ]
    return command


def _read_eval(path: Path, *, header_only: bool) -> Any:
    from inspect_ai.log import read_eval_log

    return read_eval_log(str(path), header_only=header_only)


def _candidate_logs(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise GemmaEvaluationError(f"EvalLog root must be a regular directory: {root}")
    output = []
    for candidate in sorted(root.rglob("*.eval")):
        if candidate.is_symlink() or not candidate.is_file():
            raise GemmaEvaluationError(f"EvalLog candidate is unsafe: {candidate}")
        output.append(candidate.resolve())
    if not output:
        raise GemmaEvaluationError(f"no EvalLogs were produced below {root}")
    return output


def _metadata_for_log(log: Any) -> dict[str, Any]:
    evaluation = _attribute(log, "eval")
    metadata = _mapping(_attribute(evaluation, "metadata", {}))
    record = _mapping(metadata.get("gemma4_12b_base_eval"))
    if record.get("schema") != LAUNCH_SCHEMA:
        raise GemmaEvaluationError("EvalLog lacks the Gemma base-evaluation task metadata")
    return record


def _sample_ids(log: Any) -> tuple[str, ...]:
    values = tuple(str(_attribute(sample, "id", "")) for sample in (_attribute(log, "samples", []) or []))
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise GemmaEvaluationError("EvalLog samples do not have unique non-empty question IDs")
    return values


def _assert_eos_samples(log: Any) -> None:
    for sample in _attribute(log, "samples", []) or []:
        output = _attribute(sample, "output")
        metadata = _mapping(_attribute(output, "metadata", {}))
        if metadata.get("ctm_termination") != "model_eos_only":
            raise GemmaEvaluationError("Gemma sample lacks generic EOS-only termination evidence")


def _validate_shard_log(path: Path, *, spec: Any, task_index: int, rank: int, expected_model: str) -> Any:
    log = _read_eval(path, header_only=False)
    metadata = _metadata_for_log(log)
    if (
        _attribute(log, "status") != "success"
        or metadata.get("task_index") != task_index
        or metadata.get("shard_index") != rank
        or metadata.get("shard_count") != SHARD_COUNT
        or metadata.get("smoke") is not False
        or _attribute(_attribute(log, "eval"), "model") != expected_model
    ):
        raise GemmaEvaluationError(f"Gemma shard log identity is invalid: {path}")
    expected_ids = _shard_ids(tuple(spec.question_ids), task_index=task_index, shard_index=rank)
    if set(_sample_ids(log)) != set(expected_ids):
        raise GemmaEvaluationError(f"Gemma shard log has the wrong question-ID shard: {path}")
    _assert_eos_samples(log)
    return log


def _discover_rank_logs(rank_root: Path, *, specs: Sequence[Any], rank: int, expected_model: str) -> dict[int, tuple[Path, Any]]:
    selected: dict[int, tuple[Path, Any]] = {}
    for path in _candidate_logs(rank_root):
        try:
            header = _read_eval(path, header_only=True)
            metadata = _metadata_for_log(header)
        except Exception:
            continue
        task_index = metadata.get("task_index")
        if not isinstance(task_index, int) or not 1 <= task_index <= TASK_COUNT:
            continue
        if task_index in selected:
            raise GemmaEvaluationError(f"Gemma rank {rank} has duplicate successful task-{task_index:03d} logs")
        selected[task_index] = (
            path,
            _validate_shard_log(path, spec=specs[task_index - 1], task_index=task_index, rank=rank, expected_model=expected_model),
        )
    if set(selected) != set(range(1, TASK_COUNT + 1)):
        missing = sorted(set(range(1, TASK_COUNT + 1)) - set(selected))
        raise GemmaEvaluationError(f"Gemma rank {rank} did not produce the full 21-cell shard matrix; missing={missing}")
    return selected


def worker(*, campaign_root: str | Path, rank: int, python: str) -> dict[str, Any]:
    """Run one rank's 21 source-ID shards; do not retry prior attempts."""

    if not isinstance(rank, int) or not 0 <= rank < SHARD_COUNT:
        raise GemmaEvaluationError("Gemma worker rank must be in [0, 15]")
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise GemmaEvaluationError("Gemma worker Python is unavailable")
    _require_eos_runtime(require_gpu=True)
    contract, paths, specs = _load_contract(campaign_root)
    _require_generation_implementation(contract)
    _require_discarded_smoke(paths)
    rank_root = paths["live"] / f"rank-{rank:03d}"
    receipt_path = paths["shard_receipts"] / f"rank-{rank:03d}.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _read_json(receipt_path, label=f"Gemma rank-{rank:03d} receipt")
        if receipt.get("schema") != SHARD_RECEIPT_SCHEMA or receipt.get("rank") != rank:
            raise GemmaEvaluationError("existing Gemma shard receipt has a different identity")
        return {"status": "resumed", "rank": rank, "receipt": _identity(receipt_path, label="Gemma shard receipt")}
    if rank_root.exists() or rank_root.is_symlink():
        raise FileExistsError(f"Gemma rank {rank} has preserved incomplete evidence; refusing an automatic retry")
    rank_root.mkdir(parents=True)
    command = _command_for_worker(python=python, contract=contract, paths=paths, rank=rank, log_dir=rank_root)
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=os.environ.copy(), check=False)
    if result.returncode:
        raise GemmaEvaluationError(f"Gemma rank {rank} evaluator exited {result.returncode}; preserved attempt: {rank_root}")
    expected_model = f"hf/{_snapshot_path()}"
    logs = _discover_rank_logs(rank_root, specs=specs, rank=rank, expected_model=expected_model)
    receipt = {
        "schema": SHARD_RECEIPT_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "rank": rank,
        "launch_contract": _identity(paths["contract"], label="Gemma launch contract"),
        "generation_count": sum(len(_sample_ids(log)) for _path, log in logs.values()),
        "tasks": [
            {
                "task_index": task_index,
                "sample_count": len(_sample_ids(log)),
                "raw_log": _identity(path, label=f"Gemma rank-{rank:03d} task-{task_index:03d} shard"),
            }
            for task_index, (path, log) in sorted(logs.items())
        ],
    }
    _write_once_json(receipt_path, receipt, label=f"Gemma rank-{rank:03d} shard receipt")
    return {"status": "generated", "rank": rank, "receipt": _identity(receipt_path, label="Gemma shard receipt")}


def _copy_update(value: Any, **updates: Any) -> Any:
    copier = getattr(value, "model_copy", None)
    if callable(copier):
        return copier(update=updates)
    result = copy.deepcopy(value)
    for name, item in updates.items():
        setattr(result, name, item)
    return result


def _merged_log(first: Any, *, full_ids: Sequence[str], samples: Sequence[Any]) -> Any:
    """Create one analysis-only 50-question EvalLog from immutable shards."""

    evaluation = _attribute(first, "eval")
    task_args = _mapping(_attribute(evaluation, "task_args", {}))
    metadata = _mapping(_attribute(evaluation, "metadata", {}))
    for target in (task_args, metadata):
        target["question_ids_from"] = list(full_ids)
        task_meta = _mapping(target.get("gemma4_12b_base_eval"))
        if task_meta:
            task_meta.update({"shard_index": None, "shard_count": SHARD_COUNT, "sample_count": QUESTIONS_PER_CELL, "merged": True})
            target["gemma4_12b_base_eval"] = task_meta
    dataset = _attribute(evaluation, "dataset")
    config = _attribute(evaluation, "config")
    merged_dataset = _copy_update(dataset, samples=QUESTIONS_PER_CELL, sample_ids=list(full_ids), shuffled=False)
    merged_config = _copy_update(config, limit=QUESTIONS_PER_CELL)
    merged_evaluation = _copy_update(evaluation, task_args=task_args, metadata=metadata, dataset=merged_dataset, config=merged_config)
    return _copy_update(first, eval=merged_evaluation, samples=list(samples), results=None)


def _paired_clean_task_index(specs: Sequence[Any], *, task_index: int) -> int:
    """Return the one clean cell with the exact frozen population and IDs."""

    if not 1 <= task_index <= len(specs):
        raise GemmaEvaluationError("Gemma switch scoring task index is outside the loaded matrix")
    biased = specs[task_index - 1]
    if getattr(biased, "kind", None) != "biased":
        raise GemmaEvaluationError("only a biased Gemma cell may request a paired clean log")
    expected_ids = tuple(getattr(biased, "question_ids", ())[:QUESTIONS_PER_CELL])
    candidates = [
        index
        for index, clean in enumerate(specs, start=1)
        if getattr(clean, "kind", None) == "unbiased"
        and getattr(clean, "population", None) == getattr(biased, "population", None)
        and getattr(clean, "dataset", None) == getattr(biased, "dataset", None)
        and getattr(clean, "source_identity_digest", None) == getattr(biased, "source_identity_digest", None)
        and tuple(getattr(clean, "question_ids", ())[:QUESTIONS_PER_CELL]) == expected_ids
    ]
    if len(candidates) != 1:
        raise GemmaEvaluationError(
            f"Gemma biased task-{task_index:03d} has no unique exact 50-question clean counterpart; matches={candidates}"
        )
    return candidates[0]


def _switch_score(sample: Any) -> Any:
    """Extract exactly one standard switch score from one scored sample."""

    matches: list[Any] = []
    for score in _mapping(_attribute(sample, "scores", {})).values():
        value = _mapping(_attribute(score, "value", {}))
        if set(value) == SWITCH_SCORE_KEYS:
            matches.append(score)
    if len(matches) != 1:
        raise GemmaEvaluationError(
            f"Gemma sample {_attribute(sample, 'id', '')!r} lacks exactly one complete standard switch score"
        )
    return matches[0]


def _validate_standard_switch_log(scored: Any, *, clean_path: Path, full_ids: Sequence[str]) -> None:
    """Fail closed unless CPU re-scoring bound every biased sample to one clean cell."""

    if _attribute(scored, "status") != "success" or set(_sample_ids(scored)) != set(full_ids):
        raise GemmaEvaluationError("standard Gemma switch score output is not a successful exact 50-question cell")
    _assert_eos_samples(scored)
    clean = clean_path.resolve()
    allowed = {
        "unbiased_matches_bias": {0.0, 1.0},
        "towards_bias_switch": {0.0, 1.0},
        "away_from_bias_switch": {0.0, 1.0},
        "net_switch": {-1.0, 0.0, 1.0},
        "abs_switch": {0.0, 1.0},
    }
    for sample in _attribute(scored, "samples", []) or []:
        score = _switch_score(sample)
        values = _mapping(_attribute(score, "value", {}))
        metadata = _mapping(_attribute(score, "metadata", {}))
        resolved = metadata.get("unbiased_log")
        if not isinstance(resolved, str) or Path(resolved).resolve() != clean:
            raise GemmaEvaluationError("standard Gemma switch score resolved a non-canonical clean EvalLog")
        if "unbiased_answer" not in metadata or "note" in metadata:
            raise GemmaEvaluationError("standard Gemma switch score did not resolve every matching clean sample ID")
        for key, permitted in allowed.items():
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise GemmaEvaluationError(f"standard Gemma switch score has a non-numeric {key!r} value")
            numeric = float(value)
            if not math.isnan(numeric) and numeric not in permitted:
                raise GemmaEvaluationError(f"standard Gemma switch score has an invalid {key!r} value")


def _score_merged_biased(
    log: Any,
    *,
    clean_path: Path,
    full_ids: Sequence[str],
) -> tuple[Any, dict[str, Any]]:
    """Apply the pinned standard switch scorer without reloading Gemma.

    ``mockllm/model`` is Inspect's inert primary scoring model.  The standard
    MCQ switch scorer itself is deterministic and reads only the exact completed
    clean EvalLog; it neither invokes a model nor launches generation.
    """

    if _attribute(log, "status") != "success" or set(_sample_ids(log)) != set(full_ids):
        raise GemmaEvaluationError("cannot switch-score an incomplete merged Gemma biased cell")
    for sample in _attribute(log, "samples", []) or []:
        for score in _mapping(_attribute(sample, "scores", {})).values():
            if set(_mapping(_attribute(score, "value", {}))) == SWITCH_SCORE_KEYS:
                raise GemmaEvaluationError("Gemma generation log already contains a live switch score")
    if clean_path.is_symlink() or not clean_path.is_file():
        raise GemmaEvaluationError(f"Gemma canonical clean EvalLog is unavailable: {clean_path}")
    clean_log = _read_eval(clean_path, header_only=False)
    if _attribute(clean_log, "status") != "success" or set(_sample_ids(clean_log)) != set(full_ids):
        raise GemmaEvaluationError("Gemma canonical clean EvalLog does not exactly match its biased counterpart")

    try:
        from inspect_ai import score
        from ctm_data.adapters.mcq_bias.scorer_compat import install_conditional_nan_compat

        install_conditional_nan_compat()
        import mcq_bias.scorers as mcq_scorers
    except ImportError as exc:  # pragma: no cover - configured runtime boundary
        raise RuntimeError("Inspect AI and the pinned mcq_bias switch scorer are required for Gemma post-merge scoring") from exc

    scorer = mcq_scorers.switch_scorer(str(clean_path.resolve()), question_ids_from=list(full_ids))
    scored = score(
        log,
        scorer,
        model=INSPECT_RESCORE_MODEL,
        action="append",
        display="none",
        copy=True,
    )
    _validate_standard_switch_log(scored, clean_path=clean_path, full_ids=full_ids)
    scorer_source = Path(str(getattr(mcq_scorers, "__file__", "")))
    return scored, {
        "schema": SWITCH_SCORE_SCHEMA,
        "mode": "post_merge_cpu_standard_inspect",
        "scorer": "mcq_bias/switch_scorer",
        "scorer_source": _identity(scorer_source, label="pinned mcq_bias switch scorer source"),
        "compatibility_source": _identity(
            PROJECT_ROOT / "ctm_data/adapters/mcq_bias/scorer_compat.py",
            label="Gemma switch scorer compatibility source",
        ),
        "rescore_model": INSPECT_RESCORE_MODEL,
        "clean_log": _identity(clean_path, label="Gemma canonical paired clean EvalLog"),
        "full_question_ids_sha256": hashlib.sha256(
            json.dumps(list(full_ids), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }


def _resumed_merged_cell(
    *,
    paths: Mapping[str, Path],
    task_index: int,
    spec: Any,
) -> tuple[Path, Any, dict[str, Any]] | None:
    """Read one sealed merged cell, or refuse an incomplete prior score write."""

    receipt_path = paths["merge_receipts"] / f"task-{task_index:03d}.json"
    task_root = paths["merged"] / f"task-{task_index:03d}"
    if not (receipt_path.exists() or receipt_path.is_symlink()):
        if task_root.exists() or task_root.is_symlink():
            raise FileExistsError(
                f"Gemma task-{task_index:03d} has preserved incomplete merged or scored evidence; refusing an automatic retry"
            )
        return None
    receipt = _read_json(receipt_path, label=f"Gemma task-{task_index:03d} merge receipt")
    if receipt.get("schema") != MERGE_RECEIPT_SCHEMA or receipt.get("campaign") != CAMPAIGN_NAME:
        raise GemmaEvaluationError(f"Gemma task-{task_index:03d} merge receipt has a different identity")
    if any(receipt.get(key) != value for key, value in _spec_record(task_index, spec).items()):
        raise GemmaEvaluationError(f"Gemma task-{task_index:03d} merge receipt does not match its frozen cell")
    if receipt.get("shard_count") != SHARD_COUNT:
        raise GemmaEvaluationError(f"Gemma task-{task_index:03d} merge receipt has the wrong shard count")
    raw = receipt.get("raw_log")
    if not isinstance(raw, Mapping):
        raise GemmaEvaluationError(f"Gemma task-{task_index:03d} merge receipt lacks its canonical EvalLog identity")
    source, digest = raw.get("path"), raw.get("sha256")
    if not isinstance(source, str) or not isinstance(digest, str) or len(digest) != 64:
        raise GemmaEvaluationError(f"Gemma task-{task_index:03d} merge receipt has no canonical EvalLog path")
    source_path = Path(source)
    if source_path.is_symlink():
        raise GemmaEvaluationError(f"Gemma task-{task_index:03d} canonical EvalLog must not be linked")
    canonical = _under(source_path, task_root, label=f"Gemma task-{task_index:03d} canonical EvalLog")
    if canonical.name != digest + ".eval" or _identity(canonical, label="Gemma canonical merged EvalLog") != dict(raw):
        raise GemmaEvaluationError(f"Gemma task-{task_index:03d} canonical EvalLog differs from its sealed receipt")
    return canonical, _read_eval(canonical, header_only=False), receipt


def _load_rank_receipt(path: Path, *, rank: int, contract: Path) -> dict[str, Any]:
    receipt = _read_json(path, label=f"Gemma rank-{rank:03d} shard receipt")
    if (
        receipt.get("schema") != SHARD_RECEIPT_SCHEMA
        or receipt.get("campaign") != CAMPAIGN_NAME
        or receipt.get("rank") != rank
        or receipt.get("launch_contract") != _identity(contract, label="Gemma launch contract")
    ):
        raise GemmaEvaluationError(f"Gemma rank-{rank:03d} receipt is not bound to this campaign")
    return receipt


def _require_single_model_load_trace(path: Path) -> None:
    """Reject the reproduced usage-metadata double load before a full run."""

    _identity(path, label="discarded Gemma model-load trace")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (ValueError, OSError) as exc:
        raise GemmaEvaluationError("Gemma model-load trace is unreadable or incomplete") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise GemmaEvaluationError("Gemma model-load trace has no complete event records")
    if any(row.get("schema") != "gemma-hf-load-trace-v1" for row in rows):
        raise GemmaEvaluationError("Gemma model-load trace has an unrecognised schema")
    if len({row.get("pid") for row in rows}) != 1 or rows[0].get("pid") is None:
        raise GemmaEvaluationError("Gemma model-load trace must describe one evaluator process")
    events = [row.get("event") for row in rows]
    if (events.count("target_start") != 1 or events.count("target_return") != 1
            or events.count("final") != 1 or events[-1] != "final"
            or any(event in {"hook_unavailable", "call_raise", "target_raise"} for event in events)):
        raise GemmaEvaluationError("Gemma model-load trace lacks a successful fully instrumented completion")
    model_ids = set()
    for hook in (
        "inspect_huggingfaceapi_init",
        "transformers_auto_image_text_from_pretrained",
        "transformers_pretrained_from_pretrained",
        "transformers_modeling_utils_convert_and_load_state_dict",
    ):
        starts = [row for row in rows if row.get("hook") == hook and row.get("event") == "call_start"]
        returns = [row for row in rows if row.get("hook") == hook and row.get("event") == "call_return"]
        if len(starts) != 1 or len(returns) != 1 or starts[0].get("call_id") != returns[0].get("call_id"):
            raise GemmaEvaluationError(f"Gemma smoke must load exactly one GPU model: {hook}")
        if hook in {"transformers_auto_image_text_from_pretrained", "transformers_pretrained_from_pretrained"}:
            model_ids.add(returns[0].get("result_object_id"))
    final = rows[-1]
    if (len(model_ids) != 1 or None in model_ids
            or final.get("alive_model_object_ids") != list(model_ids)
            or _mapping(final.get("cuda")).get("initialized") is not True):
        raise GemmaEvaluationError("Gemma smoke must finish with exactly the original GPU model resident")


def _require_discarded_smoke(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Require the separate compatibility evidence before generating the matrix."""

    receipt_path = paths["smoke"] / "receipt.json"
    receipt = _read_json(receipt_path, label="discarded Gemma smoke receipt")
    if (
        receipt.get("schema") != SMOKE_RECEIPT_SCHEMA
        or receipt.get("campaign") != CAMPAIGN_NAME
        or receipt.get("discarded") is not True
        or receipt.get("sample_count") != 2
        or receipt.get("termination") != "model_eos_only"
        or receipt.get("launch_contract") != _identity(paths["contract"], label="Gemma launch contract")
    ):
        raise GemmaEvaluationError("discarded Gemma smoke receipt is not bound to this launch contract")
    smoke_log = receipt.get("smoke_log")
    if not isinstance(smoke_log, Mapping):
        raise GemmaEvaluationError("discarded Gemma smoke receipt lacks its raw EvalLog identity")
    source = smoke_log.get("path")
    if not isinstance(source, str) or _identity(source, label="discarded Gemma smoke EvalLog") != dict(smoke_log):
        raise GemmaEvaluationError("discarded Gemma smoke EvalLog identity changed")
    trace_path = paths["smoke"] / "model-load-trace.jsonl"
    if receipt.get("model_load_trace") != _identity(trace_path, label="discarded Gemma model-load trace"):
        raise GemmaEvaluationError("discarded Gemma smoke lacks its bound model-load trace")
    _require_single_model_load_trace(trace_path)
    return receipt


def merge(*, campaign_root: str | Path) -> dict[str, Any]:
    """Merge immutable shards, then CPU-score each biased canonical cell exactly once."""

    contract, paths, specs = _load_contract(campaign_root)
    expected_model = f"hf/{_snapshot_path()}"
    selected: dict[int, dict[int, tuple[Path, Any]]] = {task_index: {} for task_index in range(1, TASK_COUNT + 1)}
    for rank in range(SHARD_COUNT):
        receipt_path = paths["shard_receipts"] / f"rank-{rank:03d}.json"
        _load_rank_receipt(receipt_path, rank=rank, contract=paths["contract"])
        rank_root = paths["live"] / f"rank-{rank:03d}"
        for task_index, value in _discover_rank_logs(rank_root, specs=specs, rank=rank, expected_model=expected_model).items():
            selected[task_index][rank] = value
    output: list[dict[str, Any]] = []
    canonical_cells: dict[int, tuple[Path, Any]] = {}
    for task_index, spec in enumerate(specs, start=1):
        resumed = _resumed_merged_cell(paths=paths, task_index=task_index, spec=spec)
        if resumed is not None:
            canonical, log, receipt = resumed
            full_ids = tuple(spec.question_ids)[:QUESTIONS_PER_CELL]
            if spec.kind == "biased":
                clean_index = _paired_clean_task_index(specs, task_index=task_index)
                clean = canonical_cells.get(clean_index)
                if clean is None:
                    raise GemmaEvaluationError(
                        f"Gemma task-{task_index:03d} sealed before its paired clean task-{clean_index:03d}"
                    )
                switch_scoring = _mapping(receipt.get("switch_scoring"))
                if (
                    switch_scoring.get("schema") != SWITCH_SCORE_SCHEMA
                    or switch_scoring.get("mode") != "post_merge_cpu_standard_inspect"
                    or switch_scoring.get("scorer") != "mcq_bias/switch_scorer"
                    or switch_scoring.get("rescore_model") != INSPECT_RESCORE_MODEL
                    or switch_scoring.get("clean_log") != _identity(clean[0], label="Gemma canonical paired clean EvalLog")
                ):
                    raise GemmaEvaluationError(f"Gemma task-{task_index:03d} sealed with an invalid switch-scoring binding")
                _validate_standard_switch_log(log, clean_path=clean[0], full_ids=full_ids)
            canonical_cells[task_index] = (canonical, log)
            output.append(
                {
                    "task_index": task_index,
                    "raw_log": _identity(canonical, label=f"Gemma canonical merged task-{task_index:03d} EvalLog"),
                    "receipt": _identity(
                        paths["merge_receipts"] / f"task-{task_index:03d}.json", label="Gemma merge receipt"
                    ),
                    "status": "resumed",
                }
            )
            continue
        shards = selected[task_index]
        if set(shards) != set(range(SHARD_COUNT)):
            raise GemmaEvaluationError(f"Gemma task-{task_index:03d} has incomplete rank coverage")
        full_ids = tuple(spec.question_ids)[:QUESTIONS_PER_CELL]
        samples = [sample for _rank, (_path, log) in sorted(shards.items()) for sample in (_attribute(log, "samples", []) or [])]
        observed = [str(_attribute(sample, "id", "")) for sample in samples]
        if len(samples) != QUESTIONS_PER_CELL or set(observed) != set(full_ids) or len(observed) != len(set(observed)):
            raise GemmaEvaluationError(f"Gemma task-{task_index:03d} shards do not exactly reconstitute its 50 IDs")
        samples = sorted(samples, key=lambda sample: str(_attribute(sample, "id", "")))
        first = shards[0][1]
        merged = _merged_log(first, full_ids=full_ids, samples=samples)
        switch_scoring: dict[str, Any] | None = None
        if spec.kind == "biased":
            clean_index = _paired_clean_task_index(specs, task_index=task_index)
            clean = canonical_cells.get(clean_index)
            if clean is None:
                raise GemmaEvaluationError(
                    f"Gemma task-{task_index:03d} cannot score before canonical clean task-{clean_index:03d} exists"
                )
            merged, switch_scoring = _score_merged_biased(merged, clean_path=clean[0], full_ids=full_ids)
        canonical, _status = _write_hashed_eval_once(
            merged,
            paths["merged"] / f"task-{task_index:03d}",
            label=f"Gemma merged task-{task_index:03d} EvalLog",
        )
        raw_identity = _identity(canonical, label=f"Gemma canonical merged task-{task_index:03d} EvalLog")
        receipt = {
            "schema": MERGE_RECEIPT_SCHEMA,
            "campaign": CAMPAIGN_NAME,
            **_spec_record(task_index, spec),
            "shard_count": SHARD_COUNT,
            "shard_logs": [
                {"rank": rank, "raw_log": _identity(path, label=f"Gemma task-{task_index:03d} shard")}
                for rank, (path, _log) in sorted(shards.items())
            ],
            "raw_log": raw_identity,
        }
        if switch_scoring is not None:
            receipt["switch_scoring"] = switch_scoring
        receipt_path = paths["merge_receipts"] / f"task-{task_index:03d}.json"
        _write_once_json(receipt_path, receipt, label=f"Gemma task-{task_index:03d} merge receipt")
        canonical_cells[task_index] = (canonical, merged)
        output.append(
            {
                "task_index": task_index,
                "raw_log": raw_identity,
                "receipt": _identity(receipt_path, label="Gemma merge receipt"),
                "status": "written",
            }
        )
    return {"campaign": CAMPAIGN_NAME, "merged_cells": output, "total_generations": TOTAL_GENERATIONS}


def _traced_smoke_command(command: Sequence[str], trace_file: Path) -> list[str]:
    """Instrument the actual eval child, leaving every evaluation arg intact."""

    return [command[0], str(PROJECT_ROOT / "infra/isambard/trace_gemma_hf_loads.py"),
            "--trace-file", str(trace_file), *command[1:]]


def discarded_smoke(*, campaign_root: str | Path, python: str, trace_model_loads: bool = False) -> dict[str, Any]:
    """Generate and validate two discarded clean prompts on one GPU."""

    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise GemmaEvaluationError("Gemma smoke Python is unavailable")
    _require_eos_runtime(require_gpu=True)
    contract, paths, _specs = _load_contract(campaign_root)
    _require_generation_implementation(contract)
    receipt_path = paths["smoke"] / "receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _read_json(receipt_path, label="discarded Gemma smoke receipt")
        if receipt.get("schema") != SMOKE_RECEIPT_SCHEMA:
            raise GemmaEvaluationError("existing discarded Gemma smoke has a different schema")
        return {"status": "resumed", "receipt": _identity(receipt_path, label="discarded Gemma smoke receipt")}
    attempt = paths["smoke"] / "attempt-0001"
    if attempt.exists() or attempt.is_symlink():
        raise FileExistsError("discarded Gemma smoke has preserved incomplete evidence; refusing an automatic retry")
    attempt.mkdir(parents=True)
    snapshot = _mapping(contract.get("model_snapshot")).get("path")
    if not isinstance(snapshot, str):
        raise GemmaEvaluationError("Gemma smoke has no pinned snapshot")
    task_args = {"manifest": str(paths["deployment_manifest"]), "prompt_style": "none", "hf_eos_only_no_token_cap": True}
    command = [
        python,
        str(PROJECT_ROOT / "scripts/run_evals.py"),
        "--task-factory",
        SMOKE_TASK_FACTORY,
        "--model",
        f"hf/{snapshot}",
        "--task-args",
        json.dumps(task_args, sort_keys=True, separators=(",", ":")),
        "--model-args",
        json.dumps(MODEL_ARGS, sort_keys=True, separators=(",", ":")),
        "--generation-config",
        json.dumps(GENERATION_CONFIG, sort_keys=True, separators=(",", ":")),
        "--log-dir",
        str(attempt),
        "--yes",
    ]
    trace_file = paths["smoke"] / "model-load-trace.jsonl" if trace_model_loads else None
    if trace_file is not None:
        command = _traced_smoke_command(command, trace_file)
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=os.environ.copy(), check=False)
    if result.returncode:
        raise GemmaEvaluationError(f"discarded Gemma smoke evaluator exited {result.returncode}; preserved attempt: {attempt}")
    candidates = _candidate_logs(attempt)
    if len(candidates) != 1:
        raise GemmaEvaluationError("discarded Gemma smoke must produce exactly one EvalLog")
    log = _read_eval(candidates[0], header_only=False)
    metadata = _metadata_for_log(log)
    if (
        metadata.get("smoke") is not True
        or metadata.get("task_index") != 1
        or metadata.get("sample_count") != 2
        or len(_sample_ids(log)) != 2
        or _attribute(log, "status") != "success"
        or _attribute(_attribute(log, "eval"), "model") != f"hf/{_snapshot_path()}"
    ):
        raise GemmaEvaluationError("discarded Gemma smoke did not preserve two successful clean samples")
    _assert_eos_samples(log)
    receipt = {
        "schema": SMOKE_RECEIPT_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "launch_contract": _identity(paths["contract"], label="Gemma launch contract"),
        "discarded": True,
        "sample_count": 2,
        "termination": "model_eos_only",
        "smoke_log": _identity(candidates[0], label="discarded Gemma smoke EvalLog"),
    }
    if trace_file is not None:
        _require_single_model_load_trace(trace_file)
        receipt["model_load_trace"] = _identity(trace_file, label="discarded Gemma model-load trace")
    _write_once_json(receipt_path, receipt, label="discarded Gemma smoke receipt")
    return {"status": "generated", "receipt": _identity(receipt_path, label="discarded Gemma smoke receipt")}


def dry_run() -> dict[str, Any]:
    """Validate the static 16-way plan without model, GPU, or filesystem writes."""

    _load_static_config()
    # Source IDs are not available in a dry run.  The count proof depends only
    # on the fixed 50-item cell size and rotating modulo partition.
    synthetic = tuple(f"question-{index:03d}" for index in range(100))
    specs = [
        type("StaticSpec", (), {"question_ids": synthetic})()
        for _ in range(TASK_COUNT)
    ]
    topology = _topology(specs)
    return {
        "campaign": CAMPAIGN_NAME,
        "total_generations": TOTAL_GENERATIONS,
        "cells": TASK_COUNT,
        "questions_per_cell": QUESTIONS_PER_CELL,
        "rank_generation_counts": [row["generation_count"] for row in topology["rank_workloads"]],
        "per_rank_execution": "clean_tasks_then_biased_tasks_serially",
        "cross_rank_phase_overlap_possible": True,
        "biased_generation_has_live_switch_scorer": False,
        "switch_scoring": "post_merge_cpu_exact_canonical_clean",
        "termination": "model_eos_only",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="seal source/runtime custody without generation")
    prepare_parser.add_argument("--campaign-root", required=True, type=Path)
    prepare_parser.add_argument("--source-stage2-manifest", required=True, type=Path)
    prepare_parser.add_argument("--stage2-artifact-root", required=True, type=Path)
    prepare_parser.add_argument("--yes", action="store_true")

    worker_parser = commands.add_parser("worker", help="run one rank's 21 deterministic source-ID shards")
    worker_parser.add_argument("--campaign-root", required=True, type=Path)
    worker_parser.add_argument("--rank", required=True, type=int)
    worker_parser.add_argument("--python", required=True)

    merge_parser = commands.add_parser("merge", help="merge the 16 shard logs into canonical 50-question cells")
    merge_parser.add_argument("--campaign-root", required=True, type=Path)

    smoke_parser = commands.add_parser("discarded-smoke", help="run the one-GPU two-question compatibility smoke")
    smoke_parser.add_argument("--campaign-root", required=True, type=Path)
    smoke_parser.add_argument("--python", required=True)
    smoke_parser.add_argument("--trace-model-loads", action="store_true")

    commands.add_parser("dry-run", help="prove the static 1,050-generation topology without writes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if not args.yes:
                parser.error("prepare requires --yes after reviewing the fixed 1,050-generation target")
            result = prepare(
                campaign_root=args.campaign_root,
                source_stage2_manifest=args.source_stage2_manifest,
                stage2_artifact_root=args.stage2_artifact_root,
            )
        elif args.command == "worker":
            result = worker(campaign_root=args.campaign_root, rank=args.rank, python=args.python)
        elif args.command == "merge":
            result = merge(campaign_root=args.campaign_root)
        elif args.command == "discarded-smoke":
            result = discarded_smoke(campaign_root=args.campaign_root, python=args.python, trace_model_loads=args.trace_model_loads)
        elif args.command == "dry-run":
            result = dry_run()
        else:  # pragma: no cover - argparse selects only declared commands
            parser.error("unsupported command")
            return 2
    except (FileExistsError, FileNotFoundError, GemmaEvaluationError, OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
