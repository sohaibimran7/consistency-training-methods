"""Execute one fresh RMCT HF/PEFT evaluation contract safely and resumably.

``launch_fresh_onpolicy_evals.py`` deliberately writes a reviewable contract
without starting any work.  This companion is the narrow execution boundary
for the two current RMCT targets only.  It never guesses a checkpoint,
manifest, condition, runtime, or GPU allocation:

* revalidate the write-once launch contract before execution;
* use exactly four explicitly supplied physical GPUs, one visible device per
  worker process;
* retain only successful, header- and payload-validated raw Inspect logs;
* move unreadable, partial, and non-success raw ``.eval`` files to a
  recoverable archive under the contract output root before retrying;
* finish all clean cells and validate their full payloads before a biased
  worker may start; and
* run the contract's existing CPU preflight, staging, Luna, and analysis
  commands in its declared order.

The executor is intentionally restricted to the native Transformers/PEFT
fallback for ``rmct-main`` and ``rmct-control``.  It is not a generic launcher
and it does not provide a route around the parity-attested vLLM policy for any
other target.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import launch_fresh_onpolicy_evals as launch


EXECUTOR_SCHEMA = "qwen35-fresh-onpolicy-hf-peft-executor-v1"
ALLOWED_TARGETS = frozenset({"rmct-main", "rmct-control"})
EXPECTED_GPUS = 4
EXPECTED_MAX_CONNECTIONS = 8
EXPECTED_GENERATION_CONFIG = {
    "max_tokens": 20480,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "max_connections": EXPECTED_MAX_CONNECTIONS,
}
EXPECTED_MODEL_ARGS = {"provider": "hf", "device": "cuda:0", "dtype": "bfloat16"}
EXPECTED_EXECUTION = {"max_tasks": 1, "isolate_tasks": True, "persistent_vllm_server": False}

_COMMAND_ORDER = (
    "revalidate_fresh_contract",
    "stage1_raw_generation",
    "stage1_raw_preflight",
    "stage1_stage_hash_bound_raw_logs",
    "stage1_luna_verbalisation",
    "stage1_tbsr_and_luna_analysis",
    "stage2_raw_generation",
    "stage2_raw_preflight",
    "stage2_stage_hash_bound_raw_logs",
    "stage2_luna_verbalisation",
    "stage2_tbsr_and_luna_analysis",
)
_CPU_COMMANDS = frozenset(_COMMAND_ORDER) - {
    "stage1_raw_generation",
    "stage2_raw_generation",
}
_RAW_ONLY_COMMAND_ORDER = (
    "revalidate_fresh_contract",
    "stage1_raw_generation",
    "stage1_raw_preflight",
    "stage1_stage_hash_bound_raw_logs",
    "stage2_raw_generation",
    "stage2_raw_preflight",
    "stage2_stage_hash_bound_raw_logs",
)
_RAW_ONLY_DEFERRED_COMMANDS = tuple(name for name in _COMMAND_ORDER if name not in _RAW_ONLY_COMMAND_ORDER)
_MISSING = object()


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """The small, validated subset of a launch contract this executor needs."""

    contract_path: Path
    contract_sha256: str
    document: Mapping[str, Any]
    commands: Mapping[str, Mapping[str, Any]]
    output_root: Path
    condition: str
    checkpoint: Path
    stage1_raw: Path
    stage2_raw: Path


@dataclass(frozen=True, slots=True)
class ValidatedCell:
    """One reusable raw cell after semantic Inspect validation."""

    task_index: int
    kind: str
    identity: Mapping[str, Any]
    path: Path
    sha256: str
    created: str
    runtime: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ArchiveCandidate:
    """A retryable raw EvalLog which is safe to move out of the raw root."""

    path: Path
    reason: str
    sha256: str | None


@dataclass(frozen=True, slots=True)
class Audit:
    """The usable cells and retryable artifacts found in one raw stage."""

    selected: Mapping[int, ValidatedCell]
    archived: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _Stage1Spec:
    task_index: int
    task_name: str
    kind: str
    split: str
    dataset: str
    question_ids: tuple[str, ...]
    frozen_file: Path
    source_identity_digest: str


@dataclass(frozen=True, slots=True)
class _Stage1Loaded:
    spec: _Stage1Spec
    cell: ValidatedCell
    header: Any
    full_log: Any


@dataclass(frozen=True, slots=True)
class _Stage2Loaded:
    cell: ValidatedCell
    loaded: Any
    full_log: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_or_none(path: Path) -> str | None:
    try:
        return _sha256_file(path)
    except OSError:
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _under_root(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its expected root: {resolved}") from exc
    return resolved


def _safe_component(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{label} must be one non-empty path component")
    return value


def _absolute_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return path.resolve()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically create a receipt, accepting only a byte-identical resume."""

    encoded = _json_bytes(payload)
    if path.exists():
        if path.is_file() and path.read_bytes() == encoded:
            return "resumed"
        raise FileExistsError(f"refusing to overwrite differing executor artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != encoded:
                raise FileExistsError(f"executor artifact appeared and differs: {path}")
            return "resumed"
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return "written"


def parse_physical_gpus(value: str) -> tuple[int, ...]:
    """Parse exactly four unique physical GPU indices; never infer visibility."""

    if not isinstance(value, str):
        raise ValueError("--gpus must be a comma-separated string of physical GPU indices")
    tokens = [token.strip() for token in value.split(",")]
    if len(tokens) != EXPECTED_GPUS or any(not token or not token.isdecimal() for token in tokens):
        raise ValueError("--gpus must list exactly four non-negative physical GPU indices")
    gpus = tuple(int(token) for token in tokens)
    if len(set(gpus)) != EXPECTED_GPUS:
        raise ValueError("--gpus must not contain duplicate physical GPU indices")
    return gpus


def _option(argv: Sequence[str], name: str) -> str:
    positions = [index for index, token in enumerate(argv) if token == name]
    if len(positions) != 1:
        raise ValueError(f"contract command must contain exactly one {name}")
    position = positions[0]
    if position + 1 >= len(argv):
        raise ValueError(f"contract command has no value after {name}")
    return str(argv[position + 1])


def _require_flag(argv: Sequence[str], name: str) -> None:
    if sum(token == name for token in argv) != 1:
        raise ValueError(f"contract command must contain exactly one {name}")


def _command_mapping(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    commands = document.get("commands")
    if not isinstance(commands, list):
        raise ValueError("fresh evaluation contract has no command list")
    names: list[str] = []
    mapped: dict[str, Mapping[str, Any]] = {}
    for command in commands:
        if not isinstance(command, Mapping):
            raise ValueError("fresh evaluation contract command is not an object")
        name = command.get("name")
        if not isinstance(name, str) or not name or name in mapped:
            raise ValueError("fresh evaluation contract has an invalid or duplicate command name")
        argv = command.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(token, str) or not token for token in argv):
            raise ValueError(f"fresh evaluation contract command {name!r} has an invalid argv")
        cwd = command.get("cwd")
        if not isinstance(cwd, str) or Path(cwd).resolve() != PROJECT_ROOT:
            raise ValueError(f"fresh evaluation contract command {name!r} has an unexpected cwd")
        names.append(name)
        mapped[name] = command
    if tuple(names) != _COMMAND_ORDER:
        raise ValueError(
            "fresh evaluation contract command order differs from the approved raw/preflight/staging/Luna/analysis sequence"
        )
    policy = document.get("execution_policy")
    if not isinstance(policy, Mapping) or policy.get("required_order") != list(_COMMAND_ORDER):
        raise ValueError("fresh evaluation contract has an unexpected execution policy order")
    return mapped


def _validate_raw_command(
    command: Mapping[str, Any],
    *,
    context: ExecutionContext,
    stage: str,
) -> None:
    """Reject a raw command whose task/runtime identity drifted from the contract."""

    argv = command["argv"]
    assert isinstance(argv, list)  # checked by _command_mapping
    if command.get("kind") != "gpu_generation":
        raise ValueError(f"{stage} raw command is not marked gpu_generation")
    if any(token == "--task-index" or token.startswith("--task-index=") for token in argv):
        raise ValueError(f"{stage} raw contract must not preselect task indices")
    _require_flag(argv, "--isolate-tasks")
    _require_flag(argv, "--yes")
    if "--persistent-vllm-server" in argv:
        raise ValueError("the fresh RMCT executor never permits persistent vLLM")
    if _option(argv, "--base-model") != launch.MODEL:
        raise ValueError(f"{stage} raw command has the wrong base model")
    if _absolute_path(_option(argv, "--local-checkpoint"), label=f"{stage} raw checkpoint") != context.checkpoint:
        raise ValueError(f"{stage} raw command has the wrong checkpoint")
    raw_root = context.stage1_raw if stage == "stage1" else context.stage2_raw
    if _absolute_path(_option(argv, "--log-dir"), label=f"{stage} raw log dir") != raw_root:
        raise ValueError(f"{stage} raw command has the wrong log directory")
    if _option(argv, "--max-tasks") != "1":
        raise ValueError(f"{stage} raw command must use --max-tasks 1")
    expected_factory = (
        "experiments.stage1_iid_diagnostic_none.tasks:diagnostic_matrix_tasks"
        if stage == "stage1"
        else "experiments.stage2_ood_hle.tasks:ood_tasks"
    )
    if _option(argv, "--task-factory") != expected_factory:
        raise ValueError(f"{stage} raw command has an unexpected task factory")
    expected_limit = "100" if stage == "stage1" else "200"
    if _option(argv, "--limit") != expected_limit:
        raise ValueError(f"{stage} raw command has an unexpected task limit")
    try:
        task_args = json.loads(_option(argv, "--task-args"))
        model_args = json.loads(_option(argv, "--model-args"))
        generation = json.loads(_option(argv, "--generation-config"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{stage} raw command has invalid JSON arguments") from exc
    if model_args != EXPECTED_MODEL_ARGS or generation != EXPECTED_GENERATION_CONFIG:
        raise ValueError(f"{stage} raw command does not preserve the approved HF/PEFT runtime")
    expected_manifest = str(_absolute_path(context.document["evaluation"][stage]["path"], label=f"{stage} manifest"))
    expected_task_args = {
        "manifest": expected_manifest,
        "unbiased_log": str(raw_root),
        "prompt_style": "none",
        "include_bias_acknowledged": False,
    }
    if task_args != expected_task_args:
        raise ValueError(f"{stage} raw command task arguments differ from the frozen no-CoT contract")


def _context_from_document(contract_path: Path, document: Mapping[str, Any]) -> ExecutionContext:
    """Validate the executor's deliberately small allowed contract surface."""

    if document.get("schema") != launch.LAUNCH_SCHEMA:
        raise ValueError("fresh evaluation contract has an unsupported schema")
    if document.get("contract_path") != str(contract_path):
        raise ValueError("fresh evaluation contract does not bind its own path")
    target = document.get("target")
    if target not in ALLOWED_TARGETS:
        raise ValueError(f"fresh HF/PEFT executor supports only {sorted(ALLOWED_TARGETS)}, got {target!r}")
    condition = _safe_component(document.get("condition"), label="contract condition")

    runtime = document.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("fresh evaluation contract has no runtime")
    if runtime.get("profile") != "hf-peft":
        raise ValueError("fresh RMCT executor requires runtime.profile='hf-peft'")
    if runtime.get("model_args") != EXPECTED_MODEL_ARGS:
        raise ValueError("fresh RMCT executor requires the approved native HF/PEFT model args")
    if runtime.get("generation_config") != EXPECTED_GENERATION_CONFIG:
        raise ValueError("fresh RMCT executor requires max_tokens=20480 and max_connections=8")
    if runtime.get("execution") != EXPECTED_EXECUTION:
        raise ValueError("fresh RMCT executor requires one isolated HF model process per visible GPU")
    checkpoint = _absolute_path(runtime.get("checkpoint"), label="contract runtime checkpoint")

    training = document.get("training")
    if not isinstance(training, Mapping) or not isinstance(training.get("checkpoint"), Mapping):
        raise ValueError("fresh evaluation contract has no training checkpoint identity")
    training_checkpoint = training["checkpoint"]
    if _absolute_path(training_checkpoint.get("path"), label="contract training checkpoint") != checkpoint:
        raise ValueError("fresh evaluation contract runtime checkpoint differs from its training identity")
    required_hashes = {
        "adapter_model_sha256",
        "adapter_config_sha256",
        "checkpoint_manifest_sha256",
    }
    if not required_hashes <= set(training_checkpoint):
        raise ValueError("fresh evaluation contract has incomplete raw PEFT checkpoint hashes")

    evaluation = document.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("fresh evaluation contract has no evaluation identity")
    for stage, expected_count, expected_clean in (("stage1", 8, None), ("stage2", 21, 3)):
        entry = evaluation.get(stage)
        if not isinstance(entry, Mapping):
            raise ValueError(f"fresh evaluation contract has no {stage} identity")
        if entry.get("prompt_style") != "none" or entry.get("raw_task_count") != expected_count:
            raise ValueError(f"fresh evaluation contract has an unexpected {stage} no-CoT task matrix")
        if entry.get("raw_generation_has_luna") is not False:
            raise ValueError(f"fresh evaluation contract {stage} raw generation must not include Luna")
        if expected_clean is not None and entry.get("clean_task_count") != expected_clean:
            raise ValueError("fresh evaluation contract has an unexpected Stage 2 clean matrix")

    outputs = document.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("fresh evaluation contract has no output paths")
    output_root = _absolute_path(outputs.get("root"), label="contract output root")
    if output_root == Path("/"):
        raise ValueError("fresh evaluation output root may not be /")
    expected_outputs = {
        "stage1_raw": output_root / "stage1" / "raw",
        "stage2_raw": output_root / "stage2" / "raw",
    }
    normalized_outputs: dict[str, Path] = {}
    for name, value in outputs.items():
        path = _absolute_path(value, label=f"contract output {name}")
        _under_root(path, output_root, label=f"contract output {name}")
        normalized_outputs[name] = path
    for name, expected in expected_outputs.items():
        if normalized_outputs.get(name) != expected:
            raise ValueError(f"fresh evaluation contract {name} is outside its expected condition layout")

    commands = _command_mapping(document)
    context = ExecutionContext(
        contract_path=contract_path,
        contract_sha256=_sha256_file(contract_path),
        document=document,
        commands=commands,
        output_root=output_root,
        condition=condition,
        checkpoint=checkpoint,
        stage1_raw=expected_outputs["stage1_raw"],
        stage2_raw=expected_outputs["stage2_raw"],
    )
    _validate_raw_command(commands["stage1_raw_generation"], context=context, stage="stage1")
    _validate_raw_command(commands["stage2_raw_generation"], context=context, stage="stage2")
    return context


def load_execution_context(contract: str | Path) -> ExecutionContext:
    """Rebuild a written contract before allowing it to schedule any process."""

    contract_path = Path(contract).expanduser().resolve()
    validated = launch.validate_launch_contract(contract_path)
    document = validated.get("contract") if isinstance(validated, Mapping) else None
    if not isinstance(document, Mapping):
        raise ValueError("fresh evaluation contract revalidation did not return a contract object")
    return _context_from_document(contract_path, document)


def _read_inspect_log(path: Path, *, header_only: bool) -> Any:
    """Read one Inspect log lazily so dry-run and module import stay CPU-only."""

    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - production evaluation environment only
        raise RuntimeError("Inspect AI is required to audit fresh raw EvalLogs") from exc
    return read_eval_log(str(path), header_only=header_only)


def _physical_eval_paths(raw_root: Path, *, stage: str) -> list[Path]:
    """Find every physical EvalLog, including partial files Inspect does not list."""

    raw_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for candidate in raw_root.rglob("*.eval"):
        if candidate.is_symlink():
            raise ValueError(f"{stage} raw EvalLog may not be a symlink: {candidate}")
        if not candidate.is_file():
            raise ValueError(f"{stage} raw EvalLog path is not a regular file: {candidate}")
        paths.append(_under_root(candidate, raw_root, label=f"{stage} raw EvalLog"))
    return sorted(set(paths))


def _archive_candidates(
    candidates: Sequence[ArchiveCandidate],
    *,
    context: ExecutionContext,
    stage: str,
    raw_root: Path,
) -> tuple[Mapping[str, Any], ...]:
    """Recoverably move only retryable raw logs; never delete them."""

    if not candidates:
        return ()
    archive_parent = context.output_root / "_archive" / context.condition / stage
    _under_root(archive_parent, context.output_root, label="executor archive root")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = f"raw-retry-{stamp}-{os.getpid()}"
    archive_root = archive_parent / base
    suffix = 0
    while True:
        candidate_root = archive_root if suffix == 0 else archive_parent / f"{base}-{suffix}"
        try:
            candidate_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        archive_root = candidate_root
        break

    planned: list[tuple[ArchiveCandidate, Path]] = []
    for candidate in candidates:
        source = _under_root(candidate.path, raw_root, label="raw archive source")
        relative = source.relative_to(raw_root)
        destination = archive_root / "partial-or-non-success" / relative
        if destination.exists():  # Defensive; archive_root itself was freshly created.
            raise FileExistsError(f"raw archive destination already exists: {destination}")
        planned.append((candidate, destination))

    records: list[dict[str, Any]] = []
    for candidate, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(candidate.path), str(destination))
        observed = _sha256_or_none(destination)
        if candidate.sha256 is not None and observed != candidate.sha256:
            raise RuntimeError(f"raw EvalLog bytes changed while archiving {candidate.path}")
        records.append(
            {
                "source": str(candidate.path),
                "destination": str(destination),
                "reason": candidate.reason,
                "sha256": observed,
            }
        )
    _write_immutable_json(
        archive_root / "archive-receipt.json",
        {
            "schema": EXECUTOR_SCHEMA,
            "kind": "raw-retry-archive",
            "contract_sha256": context.contract_sha256,
            "target": context.document["target"],
            "condition": context.condition,
            "stage": stage,
            "raw_root": str(raw_root),
            "entries": records,
        },
    )
    return tuple(records)


def _bind_success_receipt(context: ExecutionContext, *, stage: str, cell: ValidatedCell) -> str:
    """Record a successful raw hash without ever mutating an earlier receipt."""

    receipt_root = context.output_root / "_executor-receipts" / context.condition / stage
    _under_root(receipt_root, context.output_root, label="executor receipt root")
    path = receipt_root / f"task-{cell.task_index}-{cell.sha256}.json"
    payload = {
        "schema": EXECUTOR_SCHEMA,
        "kind": "validated-raw-cell",
        "contract_sha256": context.contract_sha256,
        "target": context.document["target"],
        "condition": context.condition,
        "stage": stage,
        "task_index": cell.task_index,
        "task_kind": cell.kind,
        "identity": dict(cell.identity),
        "raw_log": {"path": str(cell.path), "sha256": cell.sha256},
        "created": cell.created,
        "runtime": dict(cell.runtime),
    }
    return _write_immutable_json(path, payload)


def _header_field(
    evaluation: Any,
    *names: str,
    label: str,
    required: bool = True,
    default: Any = _MISSING,
) -> Any:
    """Read duplicated Inspect task args/metadata without accepting disagreement."""

    values: list[Any] = []
    for source in (_mapping(_attribute(evaluation, "task_args", {})), _mapping(_attribute(evaluation, "metadata", {}))):
        for name in names:
            if name in source:
                values.append(source[name])
    if not values:
        if required:
            raise ValueError(f"Inspect header is missing required {label}")
        return default
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"Inspect header has conflicting {label}: {values!r}")
    return first


def _has_luna_score(sample: Any) -> bool:
    for score in _mapping(_attribute(sample, "scores", {})).values():
        value = _attribute(score, "value")
        if isinstance(value, Mapping) and "bias_acknowledged" in value:
            return True
    return False


def _stage1_specs(context: ExecutionContext) -> dict[tuple[str, str, str], _Stage1Spec]:
    """Revalidate frozen no-CoT Stage 1 IDs and return the fixed task order."""

    from experiments.stage1_iid_diagnostic import gate_analysis
    from experiments.stage1_iid_diagnostic_none import raw_preflight

    entry = _mapping(context.document["evaluation"]["stage1"])
    manifest = _absolute_path(entry.get("path"), label="Stage 1 manifest")
    split_entries = _mapping(entry.get("splits"))
    split_files = {
        split: _absolute_path(_mapping(split_entries.get(split)).get("path"), label=f"Stage 1 {split} split")
        for split in gate_analysis.SPLITS
    }
    document, expected_ids = raw_preflight._manifest_and_expected_ids(manifest, split_files)
    source = _mapping(document.get("source")).get("content_sha256")
    if not isinstance(source, str) or len(source) != 64:
        raise ValueError("Stage 1 manifest has no frozen source identity")
    specs: dict[tuple[str, str, str], _Stage1Spec] = {}
    index = 1
    for task_name, kind in (("stage1_iid_unbiased", "clean"), ("stage1_iid_biased", "biased")):
        for split in gate_analysis.SPLITS:
            for dataset in gate_analysis.DATASETS:
                spec = _Stage1Spec(
                    task_index=index,
                    task_name=task_name,
                    kind=kind,
                    split=split,
                    dataset=dataset,
                    question_ids=expected_ids[(split, dataset)],
                    frozen_file=split_files[split],
                    source_identity_digest=source,
                )
                specs[(task_name, split, dataset)] = spec
                index += 1
    if len(specs) != 8:
        raise AssertionError("Stage 1 task layout drifted from its 4-clean/4-biased contract")
    return specs


def _validate_stage1_header(
    header: Any,
    *,
    path: Path,
    spec: _Stage1Spec,
    context: ExecutionContext,
) -> tuple[str, Mapping[str, Any]]:
    """Validate exact Stage 1 task/frozen/runtime provenance before reuse."""

    from experiments.stage1_iid_diagnostic import gate_analysis
    from experiments.stage1_iid_diagnostic_none import raw_preflight

    if _attribute(header, "status") != "success":
        raise ValueError(f"successful Stage 1 header changed status: {path}")
    evaluation = _attribute(header, "eval")
    task = gate_analysis._task_basename(_attribute(evaluation, "task"))
    if task != spec.task_name:
        raise ValueError(f"Stage 1 raw task has wrong type: {path}")
    split = _header_field(evaluation, "split", label="Stage 1 split")
    dataset = _header_field(evaluation, "dataset", "source_dataset", label="Stage 1 dataset")
    if (split, dataset) != (spec.split, spec.dataset):
        raise ValueError(f"Stage 1 raw task has the wrong frozen split/dataset: {path}")
    question_ids = _header_field(evaluation, "question_ids_from", label="Stage 1 question_ids_from")
    if not isinstance(question_ids, list) or tuple(question_ids) != spec.question_ids:
        raise ValueError(f"Stage 1 raw task question IDs differ from the frozen manifest: {path}")
    prompt_style = _header_field(evaluation, "prompt_style", label="Stage 1 prompt_style")
    if prompt_style != "none":
        raise ValueError(f"Stage 1 raw task has a non-none prompt style: {path}")
    source = _header_field(
        evaluation,
        "source_identity_digest",
        "source_sha256",
        label="Stage 1 source identity",
    )
    if source != spec.source_identity_digest:
        raise ValueError(f"Stage 1 raw task source identity differs from the frozen manifest: {path}")
    bias_type = _header_field(evaluation, "bias_type", label="Stage 1 bias_type", required=False, default=None)
    if (spec.kind == "clean" and bias_type is not None) or (spec.kind == "biased" and bias_type != "wrong_argument"):
        raise ValueError(f"Stage 1 raw task has the wrong bias identity: {path}")
    include_luna = _header_field(
        evaluation,
        "include_bias_acknowledged",
        label="Stage 1 acknowledgement flag",
        required=False,
        default=False,
    )
    grader_model = _header_field(
        evaluation,
        "grader_model",
        label="Stage 1 grader model",
        required=False,
        default=None,
    )
    if include_luna is not False or grader_model is not None:
        raise ValueError(f"Stage 1 raw task unexpectedly enables Luna grading: {path}")

    frozen_values: list[Any] = []
    for source_mapping in (_mapping(_attribute(evaluation, "task_args", {})), _mapping(_attribute(evaluation, "metadata", {}))):
        for field in ("frozen_file", "dataset_file"):
            if field in source_mapping:
                frozen_values.append(source_mapping[field])
    if not frozen_values:
        raise ValueError(f"Stage 1 raw task has no frozen-file identity: {path}")
    for value in frozen_values:
        if not isinstance(value, str) or not value or Path(value).resolve() != spec.frozen_file:
            raise ValueError(f"Stage 1 raw task frozen file differs from the manifest: {path}")

    unbiased_log = _header_field(
        evaluation,
        "unbiased_log",
        label="Stage 1 clean-log dependency",
        required=False,
        default=None,
    )
    if spec.kind == "clean":
        if unbiased_log not in {None, ""}:
            raise ValueError(f"Stage 1 clean task unexpectedly references a clean-log root: {path}")
    else:
        if not isinstance(unbiased_log, str) or not unbiased_log or Path(unbiased_log).resolve() != context.stage1_raw:
            raise ValueError(f"Stage 1 biased task does not bind this condition's clean-log root: {path}")
    variant_file = _header_field(
        evaluation,
        "variant_file",
        label="Stage 1 variant file",
        required=False,
        default=None,
    )
    if variant_file is not None:
        raise ValueError(f"Stage 1 raw task has an unexpected alternate prompt file: {path}")
    created = _attribute(evaluation, "created", "")
    if not isinstance(created, str) or not created:
        raise ValueError(f"Stage 1 raw task has no creation timestamp: {path}")
    _model, runtime = raw_preflight._assert_expected_hf_peft_model(
        path,
        base_model=launch.MODEL,
        checkpoint=str(context.checkpoint),
        max_connections=EXPECTED_MAX_CONNECTIONS,
    )
    return created, runtime


def _validate_stage1_clean_samples(log: Any, *, spec: _Stage1Spec, path: Path) -> None:
    if _attribute(log, "status") != "success":
        raise ValueError(f"Stage 1 full clean EvalLog is not successful: {path}")
    samples = list(_attribute(log, "samples", []) or [])
    if not samples:
        raise ValueError(f"Stage 1 clean EvalLog has no samples: {path}")
    seen: set[str] = set()
    for sample in samples:
        sample_id = _attribute(sample, "id", "")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError(f"Stage 1 clean EvalLog has a missing or duplicate question ID: {path}")
        seen.add(sample_id)
        metadata = _mapping(_attribute(sample, "metadata", {}))
        if (
            metadata.get("variant") != "unbiased"
            or metadata.get("source_dataset") != spec.dataset
            or metadata.get("prompt_style") != "none"
            or metadata.get("bias_type") is not None
        ):
            raise ValueError(f"Stage 1 clean sample has the wrong frozen task identity: {path}")
        if _has_luna_score(sample):
            raise ValueError(f"Stage 1 raw clean sample already has a Luna score: {path}")
    if seen != set(spec.question_ids):
        raise ValueError(f"Stage 1 clean EvalLog sample IDs differ from the frozen task: {path}")


def _validate_stage1_biased_samples(log: Any, *, loaded: _Stage1Loaded) -> None:
    """Use the existing paired-switch parser once the clean barrier is complete."""

    from experiments.stage1_iid_diagnostic import gate_analysis

    spec = loaded.spec
    header = gate_analysis.LogHeader(
        prompt_variant="native",
        split=spec.split,
        dataset=spec.dataset,
        question_ids=spec.question_ids,
        variant_file=None,
        unbiased_log=str(loaded.cell.path.parent),  # parser validates samples; header root was checked above.
        prompt_style="none",
        source_identity_digest=spec.source_identity_digest,
        created=loaded.cell.created,
    )
    gate_analysis.observations_from_raw_log(log, header)
    for sample in list(_attribute(log, "samples", []) or []):
        if _has_luna_score(sample):
            raise ValueError(f"Stage 1 raw biased sample already has a Luna score: {loaded.cell.path}")


def _audit_stage1(context: ExecutionContext, *, validate_biased: bool) -> Audit:
    """Validate reusable Stage 1 raw cells and archive only retryable failures."""

    from experiments.stage1_iid_diagnostic import gate_analysis

    specs = _stage1_specs(context)
    raw_root = context.stage1_raw
    candidates: list[ArchiveCandidate] = []
    loaded: dict[int, _Stage1Loaded] = {}
    fatal: list[str] = []
    for path in _physical_eval_paths(raw_root, stage="Stage 1"):
        try:
            header = _read_inspect_log(path, header_only=True)
        except Exception as exc:
            candidates.append(ArchiveCandidate(path, f"unreadable_header:{type(exc).__name__}", _sha256_or_none(path)))
            continue
        if _attribute(header, "status") != "success":
            candidates.append(ArchiveCandidate(path, "non_success_status", _sha256_or_none(path)))
            continue
        try:
            evaluation = _attribute(header, "eval")
            task_name = gate_analysis._task_basename(_attribute(evaluation, "task"))
            split = _header_field(evaluation, "split", label="Stage 1 split")
            dataset = _header_field(evaluation, "dataset", "source_dataset", label="Stage 1 dataset")
            spec = specs.get((task_name, split, dataset))
            if spec is None:
                raise ValueError(f"successful EvalLog is outside the Stage 1 frozen task matrix: {task_name}/{split}/{dataset}")
            created, runtime = _validate_stage1_header(header, path=path, spec=spec, context=context)
        except Exception as exc:
            fatal.append(f"{path}: {exc}")
            continue
        try:
            full_log = _read_inspect_log(path, header_only=False)
        except Exception as exc:
            candidates.append(ArchiveCandidate(path, f"partial_payload:{type(exc).__name__}", _sha256_or_none(path)))
            continue
        cell = ValidatedCell(
            task_index=spec.task_index,
            kind=spec.kind,
            identity={"task": spec.task_name, "split": spec.split, "dataset": spec.dataset},
            path=path,
            sha256=_sha256_file(path),
            created=created,
            runtime=dict(runtime),
        )
        if spec.task_index in loaded:
            fatal.append(f"duplicate successful Stage 1 cell {spec.task_index}: {loaded[spec.task_index].cell.path} and {path}")
            continue
        loaded[spec.task_index] = _Stage1Loaded(spec, cell, header, full_log)

    for item in loaded.values():
        try:
            if item.spec.kind == "clean":
                _validate_stage1_clean_samples(item.full_log, spec=item.spec, path=item.cell.path)
            elif validate_biased:
                _validate_stage1_biased_samples(item.full_log, loaded=item)
        except Exception as exc:
            fatal.append(f"{item.cell.path}: {exc}")
    if fatal:
        raise ValueError("refusing to resume around a successful but invalid Stage 1 EvalLog:\n" + "\n".join(fatal))
    archived = _archive_candidates(candidates, context=context, stage="stage1", raw_root=raw_root)
    selected = {index: item.cell for index, item in loaded.items()}
    for cell in selected.values():
        if cell.kind == "clean" or validate_biased:
            _bind_success_receipt(context, stage="stage1", cell=cell)
    return Audit(selected=selected, archived=archived)


def _audit_stage2(context: ExecutionContext, *, validate_biased: bool) -> Audit:
    """Validate reusable Stage 2 raw cells using the existing strict OOD parser."""

    from experiments.stage2_ood_hle import raw_preflight
    from experiments.stage2_ood_hle.tasks import ood_task_specs

    entry = _mapping(context.document["evaluation"]["stage2"])
    manifest = _absolute_path(entry.get("path"), label="Stage 2 manifest")
    raw_preflight.validate_manifest(manifest)
    specs = ood_task_specs(manifest)
    expected = raw_preflight._expected_cell_specs(specs)
    index_by_identity = {raw_preflight._task_identity(spec): index for index, spec in enumerate(specs, start=1)}
    if len(index_by_identity) != 21:
        raise ValueError("Stage 2 frozen task matrix has an invalid task-index mapping")
    runtime = raw_preflight._validate_runtime_contract(
        runtime_profile="hf-peft",
        expected_base_model=launch.MODEL,
        expected_checkpoint=str(context.checkpoint),
        expected_max_connections=EXPECTED_MAX_CONNECTIONS,
        require_vllm_adapter_attestation=False,
    )
    candidates: list[ArchiveCandidate] = []
    loaded: dict[int, _Stage2Loaded] = {}
    fatal: list[str] = []
    for path in _physical_eval_paths(context.stage2_raw, stage="Stage 2"):
        try:
            header = _read_inspect_log(path, header_only=True)
        except Exception as exc:
            candidates.append(ArchiveCandidate(path, f"unreadable_header:{type(exc).__name__}", _sha256_or_none(path)))
            continue
        if raw_preflight._attribute(header, "status") != "success":
            candidates.append(ArchiveCandidate(path, "non_success_status", _sha256_or_none(path)))
            continue
        try:
            evaluation = raw_preflight._attribute(header, "eval")
            task_name = raw_preflight._task_basename(raw_preflight._attribute(evaluation, "task"))
            if task_name not in {raw_preflight.TASK_UNBIASED, raw_preflight.TASK_BIASED}:
                raise ValueError(f"successful EvalLog has an unexpected Stage 2 task {task_name!r}")
            identity = raw_preflight._parse_candidate_identity(evaluation, task_name=task_name, path=path)
            spec = expected.get(identity)
            if spec is None:
                raise ValueError(f"successful EvalLog is outside the Stage 2 frozen matrix: {identity!r}")
            created = raw_preflight._validate_header(header, path=path, spec=spec, raw_root=context.stage2_raw)
            model, observed_runtime = raw_preflight._assert_runtime(path, runtime=runtime)
        except Exception as exc:
            fatal.append(f"{path}: {exc}")
            continue
        try:
            full_log = _read_inspect_log(path, header_only=False)
        except Exception as exc:
            candidates.append(ArchiveCandidate(path, f"partial_payload:{type(exc).__name__}", _sha256_or_none(path)))
            continue
        index = index_by_identity[identity]
        cell = ValidatedCell(
            task_index=index,
            kind="clean" if spec.kind == "unbiased" else "biased",
            identity={
                "kind": spec.kind,
                "regime": spec.regime,
                "population": spec.population,
                "dataset": spec.dataset,
                "bias_type": spec.bias_type,
            },
            path=path,
            sha256=_sha256_file(path),
            created=created,
            runtime=dict(observed_runtime),
        )
        if index in loaded:
            fatal.append(f"duplicate successful Stage 2 cell {index}: {loaded[index].cell.path} and {path}")
            continue
        loaded_log = raw_preflight.LoadedTaskLog(spec, path, created, header, model, observed_runtime)
        loaded[index] = _Stage2Loaded(cell, loaded_log, full_log)

    clean_paths = {
        (item.loaded.spec.population, item.loaded.spec.dataset): item.loaded.path
        for item in loaded.values()
        if item.loaded.spec.kind == "unbiased"
    }
    for item in loaded.values():
        try:
            if item.loaded.spec.kind == "unbiased" or validate_biased:
                raw_preflight._validate_samples(item.full_log, loaded=item.loaded, clean_paths=clean_paths)
        except Exception as exc:
            fatal.append(f"{item.cell.path}: {exc}")
    if fatal:
        raise ValueError("refusing to resume around a successful but invalid Stage 2 EvalLog:\n" + "\n".join(fatal))
    archived = _archive_candidates(candidates, context=context, stage="stage2", raw_root=context.stage2_raw)
    selected = {index: item.cell for index, item in loaded.items()}
    for cell in selected.values():
        if cell.kind == "clean" or validate_biased:
            _bind_success_receipt(context, stage="stage2", cell=cell)
    return Audit(selected=selected, archived=archived)


def _task_argv(command: Mapping[str, Any], task_indices: Sequence[int]) -> list[str]:
    argv = list(command["argv"])
    if not task_indices:
        raise ValueError("cannot launch an empty raw task group")
    if any(index < 1 for index in task_indices):
        raise ValueError("raw task indices must be positive")
    if any(token == "--task-index" or token.startswith("--task-index=") for token in argv):
        raise ValueError("raw command already contains a task selection")
    for index in task_indices:
        argv.extend(("--task-index", str(index)))
    return argv


def _run_worker_group(
    command: Mapping[str, Any],
    *,
    gpu: int,
    task_indices: Sequence[int],
) -> str | None:
    """Run one serial Inspect worker group with exactly one physical GPU visible."""

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # A launch environment on the shared host may have stale vLLM endpoint
    # variables.  Native HF/PEFT must neither reuse nor publish such a server.
    for name in ("VLLM_BASE_URL", "VLLM_API_KEY", "CTM_PERSISTENT_VLLM_SERVER_METADATA"):
        environment.pop(name, None)
    argv = _task_argv(command, task_indices)
    print(f"raw worker gpu={gpu} task_indices={list(task_indices)}", flush=True)
    completed = subprocess.run(
        argv,
        cwd=str(PROJECT_ROOT),
        env=environment,
        check=False,
    )
    if completed.returncode:
        return f"gpu {gpu}, tasks {list(task_indices)} exited with status {completed.returncode}"
    return None


def _run_missing_tasks(
    command: Mapping[str, Any],
    *,
    task_indices: Sequence[int],
    gpus: Sequence[int],
) -> list[str]:
    """Round-robin missing cells across four serial one-GPU Inspect workers."""

    groups: dict[int, list[int]] = {gpu: [] for gpu in gpus}
    for offset, task_index in enumerate(task_indices):
        groups[gpus[offset % len(gpus)]].append(task_index)
    active = [(gpu, indices) for gpu, indices in groups.items() if indices]
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active)) as executor:
        futures = {
            executor.submit(_run_worker_group, command, gpu=gpu, task_indices=indices): (gpu, indices)
            for gpu, indices in active
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                failure = future.result()
            except Exception as exc:  # Keep other GPU workers running to preserve their valid logs.
                gpu, indices = futures[future]
                failures.append(f"gpu {gpu}, tasks {indices} raised {type(exc).__name__}: {exc}")
            else:
                if failure:
                    failures.append(failure)
    return failures


def _require_cells(audit: Audit, expected: Sequence[int], *, stage: str, phase: str) -> None:
    missing = [index for index in expected if index not in audit.selected]
    if missing:
        raise ValueError(f"{stage} {phase} barrier is incomplete; missing validated task indices {missing}")


def _run_raw_stage(context: ExecutionContext, *, stage: str, gpus: Sequence[int]) -> dict[str, Any]:
    """Run only missing raw cells, with a full clean barrier before biased cells."""

    if stage == "stage1":
        command = context.commands["stage1_raw_generation"]
        audit = _audit_stage1
        clean = (1, 2, 3, 4)
        biased = (5, 6, 7, 8)
    elif stage == "stage2":
        command = context.commands["stage2_raw_generation"]
        audit = _audit_stage2
        clean = (1, 2, 3)
        biased = tuple(range(4, 22))
    else:  # pragma: no cover - internal fixed caller only
        raise AssertionError(f"unknown raw stage {stage}")

    first = audit(context, validate_biased=False)
    missing_clean = [index for index in clean if index not in first.selected]
    clean_failures: list[str] = []
    if missing_clean:
        clean_failures = _run_missing_tasks(command, task_indices=missing_clean, gpus=gpus)
    after_clean = audit(context, validate_biased=False)
    if clean_failures:
        raise RuntimeError(
            f"{stage} clean workers failed; successful raw cells were retained and retryable partials were archived: "
            + "; ".join(clean_failures)
        )
    _require_cells(after_clean, clean, stage=stage, phase="clean")

    # A successful header is not sufficient to unlock paired biased work.  The
    # preceding audit validated every clean full payload against frozen IDs;
    # now re-audit biased cells with the clean paths available to their switch
    # scorer validation.
    before_biased = audit(context, validate_biased=True)
    _require_cells(before_biased, clean, stage=stage, phase="clean")
    missing_biased = [index for index in biased if index not in before_biased.selected]
    biased_failures: list[str] = []
    if missing_biased:
        biased_failures = _run_missing_tasks(command, task_indices=missing_biased, gpus=gpus)
    complete = audit(context, validate_biased=True)
    if biased_failures:
        raise RuntimeError(
            f"{stage} biased workers failed; successful raw cells were retained and retryable partials were archived: "
            + "; ".join(biased_failures)
        )
    _require_cells(complete, (*clean, *biased), stage=stage, phase="full raw matrix")
    return {
        "clean_reused": len(clean) - len(missing_clean),
        "clean_generated": len(missing_clean),
        "biased_reused": len(biased) - len(missing_biased),
        "biased_generated": len(missing_biased),
        "archived": [*first.archived, *after_clean.archived, *before_biased.archived, *complete.archived],
    }


def _run_contract_command(command: Mapping[str, Any]) -> None:
    """Run one exact CPU command from the already revalidated contract."""

    name = str(command["name"])
    if name not in _CPU_COMMANDS:
        raise ValueError(f"not a CPU contract command: {name}")
    print(f"contract command: {name}", flush=True)
    subprocess.run(list(command["argv"]), cwd=str(PROJECT_ROOT), check=True)


def _raw_only_receipt(context: ExecutionContext, *, gpus: Sequence[int]) -> str:
    """Seal a complete raw/preflight/staging handoff without invoking Luna.

    The receipt intentionally records the *current* fully validated raw bytes,
    rather than transient worker logs or archive timestamps.  A later raw-only
    invocation therefore resumes only if the exact 8- and 21-cell matrices are
    still present.  It gives the separate single-budget Luna owner a compact
    proof that both conditions reached the no-network handoff boundary.
    """

    stage1 = _audit_stage1(context, validate_biased=True)
    stage2 = _audit_stage2(context, validate_biased=True)
    _require_cells(stage1, tuple(range(1, 9)), stage="stage1", phase="full raw matrix")
    _require_cells(stage2, tuple(range(1, 22)), stage="stage2", phase="full raw matrix")
    path = context.output_root / "_executor-receipts" / context.condition / "raw-only-complete.json"
    _under_root(path, context.output_root, label="raw-only receipt")
    payload = {
        "schema": EXECUTOR_SCHEMA,
        "kind": "raw-only-complete",
        "contract": {"path": str(context.contract_path), "sha256": context.contract_sha256},
        "target": context.document["target"],
        "condition": context.condition,
        "runtime_profile": "hf-peft",
        "gpus": list(gpus),
        "completed_contract_commands": list(_RAW_ONLY_COMMAND_ORDER),
        "deferred_contract_commands": list(_RAW_ONLY_DEFERRED_COMMANDS),
        "stages": {
            stage: [
                {
                    "task_index": cell.task_index,
                    "task_kind": cell.kind,
                    "identity": dict(cell.identity),
                    "raw_log": {"path": str(cell.path), "sha256": cell.sha256},
                    "runtime": dict(cell.runtime),
                }
                for _, cell in sorted(audit.selected.items())
            ]
            for stage, audit in (("stage1", stage1), ("stage2", stage2))
        },
    }
    return _write_immutable_json(path, payload)


def dry_run_summary(context: ExecutionContext, *, gpus: Sequence[int], raw_only: bool = False) -> dict[str, Any]:
    """Return a no-side-effect preview suitable for the shared production host."""

    return {
        "schema": EXECUTOR_SCHEMA,
        "contract": str(context.contract_path),
        "contract_sha256": context.contract_sha256,
        "target": context.document["target"],
        "condition": context.condition,
        "runtime_profile": "hf-peft",
        "gpus": list(gpus),
        "stage1": {"clean_task_indices": [1, 2, 3, 4], "biased_task_indices": [5, 6, 7, 8]},
        "stage2": {"clean_task_indices": [1, 2, 3], "biased_task_indices": list(range(4, 22))},
        "raw_only": raw_only,
        "command_order": list(_RAW_ONLY_COMMAND_ORDER if raw_only else _COMMAND_ORDER),
        "deferred_commands": list(_RAW_ONLY_DEFERRED_COMMANDS if raw_only else ()),
    }


def execute_contract(contract: str | Path, *, gpus: Sequence[int], raw_only: bool = False) -> dict[str, Any]:
    """Execute the immutable contract in its declared order.

    Callers must explicitly choose four physical GPUs.  This function performs
    no GPU/network work until ``load_execution_context`` has revalidated the
    immutable training, checkpoint, and frozen-manifest identities.
    """

    if len(gpus) != EXPECTED_GPUS or len(set(gpus)) != EXPECTED_GPUS or any(
        isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 for gpu in gpus
    ):
        raise ValueError("execute_contract requires exactly four unique non-negative physical GPU indices")
    context = load_execution_context(contract)
    results: dict[str, Any] = {
        "schema": EXECUTOR_SCHEMA,
        "contract": str(context.contract_path),
        "target": context.document["target"],
        "condition": context.condition,
        "gpus": list(gpus),
        "steps": [],
    }
    command_order = _RAW_ONLY_COMMAND_ORDER if raw_only else _COMMAND_ORDER
    for name in command_order:
        command = context.commands[name]
        if name == "stage1_raw_generation":
            # Rebuild immediately before a potentially long GPU phase too;
            # this catches checkpoint/manifest drift after an interrupted run.
            context = load_execution_context(context.contract_path)
            result = _run_raw_stage(context, stage="stage1", gpus=gpus)
            results["steps"].append({"name": name, "result": result})
        elif name == "stage2_raw_generation":
            context = load_execution_context(context.contract_path)
            result = _run_raw_stage(context, stage="stage2", gpus=gpus)
            results["steps"].append({"name": name, "result": result})
        else:
            _run_contract_command(command)
            results["steps"].append({"name": name, "result": "completed"})
    if raw_only:
        receipt_status = _raw_only_receipt(context, gpus=gpus)
        results["raw_only_receipt"] = {
            "path": str(context.output_root / "_executor-receipts" / context.condition / "raw-only-complete.json"),
            "status": receipt_status,
            "deferred_commands": list(_RAW_ONLY_DEFERRED_COMMANDS),
        }
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path, help="write-once JSON contract from launch_fresh_onpolicy_evals.py")
    parser.add_argument(
        "--gpus",
        required=True,
        help="exactly four comma-separated physical GPU indices (production shared host: 4,5,6,7)",
    )
    parser.add_argument("--yes", action="store_true", help="execute GPU generation and the contract's Luna grading commands")
    parser.add_argument("--dry-run", action="store_true", help="revalidate and print the execution plan without starting any process")
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="run raw generation plus CPU preflight/staging only; defer paid Luna grading and analysis with an immutable receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.yes:
        parser.error("--dry-run and --yes are mutually exclusive")
    if not args.dry_run and not args.yes:
        parser.error("--yes is required to start GPU generation or Luna grading; use --dry-run to inspect safely")
    try:
        gpus = parse_physical_gpus(args.gpus)
        context = load_execution_context(args.contract)
        if args.dry_run:
            print(json.dumps(dry_run_summary(context, gpus=gpus, raw_only=args.raw_only), indent=2, sort_keys=True))
            return
        result = execute_contract(context.contract_path, gpus=gpus, raw_only=args.raw_only)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


__all__ = [
    "ALLOWED_TARGETS",
    "Audit",
    "EXECUTOR_SCHEMA",
    "ExecutionContext",
    "ValidatedCell",
    "dry_run_summary",
    "execute_contract",
    "load_execution_context",
    "main",
    "parse_physical_gpus",
]


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
