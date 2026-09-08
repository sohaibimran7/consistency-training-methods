"""Promote already-graded Stage 2 Luna artifacts without calling a grader.

This is deliberately a local, evidence-preserving operation.  It accepts one
or more separately copied Stage 2 output trees (for example, a completed
matrix's ``luna-derived-v1`` tree and an incremental consumer's
``luna-incremental-derived-v1`` tree), validates every complete derived
trio, then copies the original bytes into one canonical destination.

Promotion never imports the Luna scorer, creates an OpenRouter client, or
rewrites provenance.  In particular, copied provenance keeps the original
remote paths intact.  The caller supplies the local copies of the staged and
raw evidence roots that correspond to each derived root, so the preserved
paths can be checked by their canonical Stage 2 layout rather than silently
rebased.

Every candidate must prove all of the following before *any* destination file
is written:

* a complete ``.eval`` / ``.jsonl`` / ``.provenance.json`` trio whose rows
  reproduce exactly from the derived EvalLog;
* an unchanged staged source EvalLog, the original source SHA-256, and the
  pinned GPT-5.6 Luna 5 x 100 x 256 policy;
* the paired-clean SHA-256, frozen-manifest SHA-256, canonical task identity,
  and native-HF checkpoint identity; and
* an adjacent immutable resume-handoff receipt whenever one exists (it is
  required for receipt-bound incremental outputs).

The destination is immutable.  An existing artifact can only be resumed when
its bytes match exactly; a collision between two non-identical candidates is
ambiguous and fails closed rather than selecting one or overwriting evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.stage2_ood_hle import grade_luna, raw_preflight
from experiments.stage2_ood_hle.hf_peft_resume import HANDOFF_SCHEMA
from experiments.stage2_ood_hle.hf_peft_runner import BASE_MODEL, condition_spec
from experiments.stage2_ood_hle.materialize import (
    HELDOUT_BIAS,
    HELDOUT_BIASES,
    HELDOUT_DATASET,
    HELDOUT_DATASET_AND_BIAS,
    HLE_DATASET,
    IID,
    IN_DOMAIN_DATASETS,
    PROMPT_STYLE,
    TRAINING_BIAS,
)


EXPECTED_GRADER_MODEL = "openrouter/openai/gpt-5.6-luna-20260709"
EXPECTED_WORKERS = 5
EXPECTED_CONNECTIONS_PER_WORKER = 100
EXPECTED_AGGREGATE_CONNECTION_LIMIT = 500
EXPECTED_GRADER_MAX_TOKENS = 256
_SHA256_HEX = frozenset("0123456789abcdef")

Identity = tuple[str, str, str, str, str]


@dataclass(frozen=True, slots=True)
class SourceRoots:
    """One copied evidence bundle used as an immutable promotion input."""

    derived_root: Path
    staged_root: Path
    raw_root: Path
    manifest: Path
    preflight_root: Path


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """One validated derived trio and its immutable destination outcome."""

    source_eval: Path
    destination_eval: Path
    status: str


@dataclass(frozen=True, slots=True)
class _Binding:
    kind: str
    paired_clean: Mapping[str, Any]
    manifest_sha256: str
    report_sha256: str | None = None
    receipt_sha256: str | None = None
    receipt_path: Path | None = None


@dataclass(frozen=True, slots=True)
class _Candidate:
    roots: SourceRoots
    eval_path: Path
    rows_path: Path
    provenance_path: Path
    relative_eval: Path
    relative_rows: Path
    relative_provenance: Path
    hashes: tuple[str, str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _require_sha256(value: Any, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"Stage 2 Luna promotion has invalid {label}")
    return str(value)


def _require_component(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Stage 2 Luna promotion has invalid {label}")
    return value


def _require_absolute_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"Stage 2 Luna promotion has no absolute {label}")
    return Path(value).resolve()


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Stage 2 Luna promotion has invalid {label}")
    return value


def _require_regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Stage 2 Luna promotion {label} is not a regular file: {path}")
    return path.resolve()


def _require_directory(path: str | Path, *, label: str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_dir():
        raise FileNotFoundError(f"Stage 2 Luna promotion {label} is not a real directory: {supplied}")
    return supplied.resolve()


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    _require_regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Stage 2 Luna promotion {label} is not valid JSON: {path}") from exc
    return _require_mapping(value, label=label)


def _under(path: Path, root: Path, *, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Stage 2 Luna promotion {label} escapes its declared root: {path}") from exc
    return path


def _canonical_biased_identities() -> frozenset[Identity]:
    """Return the frozen 18-cell biased topology without relying on ordering."""

    identities: list[Identity] = []
    identities.extend(("unbiased", IID, "in_domain", dataset, "") for dataset in IN_DOMAIN_DATASETS)
    identities.append(("unbiased", HELDOUT_DATASET, "hle", HLE_DATASET, ""))
    identities.extend(("biased", IID, "in_domain", dataset, TRAINING_BIAS) for dataset in IN_DOMAIN_DATASETS)
    identities.append(("biased", HELDOUT_DATASET, "hle", HLE_DATASET, TRAINING_BIAS))
    for bias_type in HELDOUT_BIASES:
        identities.extend(("biased", HELDOUT_BIAS, "in_domain", dataset, bias_type) for dataset in IN_DOMAIN_DATASETS)
        identities.append(("biased", HELDOUT_DATASET_AND_BIAS, "hle", HLE_DATASET, bias_type))
    if len(identities) != 21 or len(set(identities)) != 21:  # pragma: no cover - fixed imported topology
        raise RuntimeError("Stage 2 Luna promotion has an invalid frozen task topology")
    return frozenset(identity for identity in identities if identity[0] == "biased")


def _identity_from_mapping(value: Any, *, label: str) -> Identity:
    record = _require_mapping(value, label=label)
    if set(record) != {"kind", "regime", "population", "dataset", "bias_type"}:
        raise ValueError(f"Stage 2 Luna promotion {label} has unexpected fields")
    kind = _require_component(record.get("kind"), label=f"{label}.kind")
    regime = _require_component(record.get("regime"), label=f"{label}.regime")
    population = _require_component(record.get("population"), label=f"{label}.population")
    dataset = _require_component(record.get("dataset"), label=f"{label}.dataset")
    bias_type = _require_component(record.get("bias_type"), label=f"{label}.bias_type")
    identity = kind, regime, population, dataset, bias_type
    if kind != "biased" or identity not in _canonical_biased_identities():
        raise ValueError(f"Stage 2 Luna promotion {label} is not a canonical biased Stage 2 cell")
    return identity


def _provenance_identity(value: Mapping[str, Any]) -> Identity:
    identity = (
        _require_component(value.get("kind", "biased"), label="provenance.kind"),
        _require_component(value.get("regime"), label="provenance.regime"),
        _require_component(value.get("population"), label="provenance.population"),
        _require_component(value.get("dataset"), label="provenance.dataset"),
        _require_component(value.get("bias_type"), label="provenance.bias_type"),
    )
    # Grade provenance intentionally does not serialize ``kind``.  It only
    # ever represents biased inputs, so retain that convention while making
    # the inferred kind explicit for every downstream comparison.
    if identity[0] != "biased" or identity not in _canonical_biased_identities():
        raise ValueError("Stage 2 Luna promotion provenance is not a canonical biased Stage 2 cell")
    return identity


def _canonical_shard_index(*, condition: str, identity: Identity) -> int:
    _kind, regime, population, dataset, bias_type = identity
    encoded = "\0".join((condition, regime, population, dataset, bias_type)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % EXPECTED_WORKERS


def _validate_paired_clean(value: Any, *, full: bool) -> Mapping[str, Any]:
    paired = _require_mapping(value, label="paired_clean")
    expected = {"raw_log", "raw_log_sha256"}
    if full:
        expected.update({"question_ids_sha256", "source_identity_digest"})
    if set(paired) != expected:
        raise ValueError("Stage 2 Luna promotion paired_clean has unexpected fields")
    _require_absolute_path(paired.get("raw_log"), label="paired_clean.raw_log")
    _require_sha256(paired.get("raw_log_sha256"), label="paired_clean.raw_log_sha256")
    if full:
        _require_sha256(paired.get("question_ids_sha256"), label="paired_clean.question_ids_sha256")
        digest = paired.get("source_identity_digest")
        if not isinstance(digest, str) or not digest.startswith("stage2-ood-hle-2x2:") or not _is_sha256(digest.removeprefix("stage2-ood-hle-2x2:")):
            raise ValueError("Stage 2 Luna promotion paired_clean has an invalid source_identity_digest")
    return dict(paired)


def _validate_policy(provenance: Mapping[str, Any], *, condition: str, identity: Identity) -> None:
    expected = {
        "schema": grade_luna.PROVENANCE_SCHEMA,
        "condition": condition,
        "regime": identity[1],
        "population": identity[2],
        "dataset": identity[3],
        "bias_type": identity[4],
        "grader_model": EXPECTED_GRADER_MODEL,
        "worker_count": EXPECTED_WORKERS,
        "connections_per_worker": EXPECTED_CONNECTIONS_PER_WORKER,
        "aggregate_connection_limit": EXPECTED_AGGREGATE_CONNECTION_LIMIT,
        "grader_max_tokens": EXPECTED_GRADER_MAX_TOKENS,
        "inspect_rescore_model": grade_luna.INSPECT_RESCORE_MODEL,
        "smoke_samples": None,
        "deterministic_shard_index": _canonical_shard_index(condition=condition, identity=identity),
    }
    for field, required in expected.items():
        if provenance.get(field) != required:
            raise ValueError(f"Stage 2 Luna promotion provenance {field} differs from the approved Luna policy")


def _validate_provenance(provenance: Mapping[str, Any]) -> tuple[str, Identity, str, Path, _Binding]:
    condition = _require_component(provenance.get("condition"), label="provenance.condition")
    identity = _provenance_identity(provenance)
    _validate_policy(provenance, condition=condition, identity=identity)
    source_path = _require_absolute_path(provenance.get("source_log"), label="provenance.source_log")
    if source_path.suffix != ".eval":
        raise ValueError("Stage 2 Luna promotion provenance source_log is not an .eval file")
    source_sha256 = _require_sha256(provenance.get("source_sha256"), label="provenance.source_sha256")

    has_raw = "raw_preflight" in provenance
    has_incremental = "incremental_handoff" in provenance
    if has_raw == has_incremental:
        raise ValueError("Stage 2 Luna promotion provenance must have exactly one source binding")
    common = {
        "schema",
        "source_log",
        "source_sha256",
        "condition",
        "regime",
        "population",
        "dataset",
        "bias_type",
        "grader_model",
        "worker_count",
        "connections_per_worker",
        "aggregate_connection_limit",
        "deterministic_shard_index",
        "grader_max_tokens",
        "inspect_rescore_model",
        "smoke_samples",
    }
    if has_raw:
        if set(provenance) != common | {"raw_preflight"}:
            raise ValueError("Stage 2 Luna promotion raw-preflight provenance has unexpected fields")
        binding = _require_mapping(provenance["raw_preflight"], label="raw_preflight")
        if set(binding) != {"schema", "report_sha256", "source_sha256", "manifest_sha256", "paired_clean"}:
            raise ValueError("Stage 2 Luna promotion raw_preflight binding has unexpected fields")
        if binding.get("schema") != raw_preflight.PREFLIGHT_SCHEMA:
            raise ValueError("Stage 2 Luna promotion raw_preflight binding has an unexpected schema")
        if binding.get("source_sha256") != source_sha256:
            raise ValueError("Stage 2 Luna promotion raw_preflight source SHA-256 differs from provenance")
        result = _Binding(
            kind="raw_preflight",
            paired_clean=_validate_paired_clean(binding.get("paired_clean"), full=True),
            manifest_sha256=_require_sha256(binding.get("manifest_sha256"), label="raw_preflight.manifest_sha256"),
            report_sha256=_require_sha256(binding.get("report_sha256"), label="raw_preflight.report_sha256"),
        )
    else:
        if set(provenance) != common | {"incremental_handoff"}:
            raise ValueError("Stage 2 Luna promotion incremental provenance has unexpected fields")
        binding = _require_mapping(provenance["incremental_handoff"], label="incremental_handoff")
        if set(binding) != {"schema", "receipt_path", "receipt_sha256", "source_sha256", "manifest_sha256", "paired_clean"}:
            raise ValueError("Stage 2 Luna promotion incremental binding has unexpected fields")
        if binding.get("schema") != HANDOFF_SCHEMA:
            raise ValueError("Stage 2 Luna promotion incremental binding has an unexpected receipt schema")
        if binding.get("source_sha256") != source_sha256:
            raise ValueError("Stage 2 Luna promotion incremental source SHA-256 differs from provenance")
        result = _Binding(
            kind="incremental_handoff",
            paired_clean=_validate_paired_clean(binding.get("paired_clean"), full=False),
            manifest_sha256=_require_sha256(binding.get("manifest_sha256"), label="incremental_handoff.manifest_sha256"),
            receipt_sha256=_require_sha256(binding.get("receipt_sha256"), label="incremental_handoff.receipt_sha256"),
            receipt_path=_require_absolute_path(binding.get("receipt_path"), label="incremental_handoff.receipt_path"),
        )
    return condition, identity, source_sha256, source_path, result


def _local_staged_path(roots: SourceRoots, *, declared: Path, condition: str, identity: Identity) -> Path:
    _kind, _regime, population, dataset, bias_type = identity
    expected_tail = (condition, population, bias_type, dataset, declared.name)
    if len(declared.parts) < len(expected_tail) or tuple(declared.parts[-len(expected_tail) :]) != expected_tail:
        raise ValueError("Stage 2 Luna promotion provenance source_log does not use the canonical staged layout")
    local = roots.staged_root.joinpath(*expected_tail)
    _under(local, roots.staged_root, label="local staged source")
    _require_regular_file(local, label="staged source EvalLog")
    return local


def _same_declared_layout(declared: Path, local: Path, *, label: str) -> None:
    if declared.name != local.name or len(declared.parts) < 5 or tuple(declared.parts[-5:]) != tuple(local.parts[-5:]):
        raise ValueError(f"Stage 2 Luna promotion {label} does not use the canonical staged layout")


def _path_and_sha(value: Any, *, label: str) -> tuple[Path, str]:
    record = _require_mapping(value, label=label)
    if set(record) != {"path", "sha256"}:
        raise ValueError(f"Stage 2 Luna promotion {label} has unexpected fields")
    return (
        _require_absolute_path(record.get("path"), label=f"{label}.path"),
        _require_sha256(record.get("sha256"), label=f"{label}.sha256"),
    )


def _validate_checkpoint(condition: str, value: Any) -> None:
    record = _require_mapping(value, label="receipt.checkpoint")
    expected = condition_spec(condition)
    required = {
        "path",
        "checkpoint_name",
        "backend",
        "lora",
        "base_model",
        "adapter_model_sha256",
        "adapter_config_sha256",
        "manifest_sha256",
    }
    if set(record) != required:
        raise ValueError("Stage 2 Luna promotion receipt checkpoint has unexpected fields")
    _require_absolute_path(record.get("path"), label="receipt.checkpoint.path")
    if (
        record.get("checkpoint_name") != expected.checkpoint_name
        or record.get("backend") != "local"
        or record.get("lora") is not True
        or record.get("base_model") != BASE_MODEL
        or record.get("adapter_model_sha256") != expected.adapter_model_sha256
        or record.get("adapter_config_sha256") != expected.adapter_config_sha256
        or record.get("manifest_sha256") != expected.manifest_sha256
    ):
        raise ValueError(f"Stage 2 Luna promotion receipt checkpoint identity differs for {condition!r}")


def _validate_receipt(
    roots: SourceRoots,
    *,
    condition: str,
    identity: Identity,
    source_sha256: str,
    staged_path: Path,
    binding: _Binding,
    manifest_sha256: str,
) -> Mapping[str, Any] | None:
    """Check the optional adjacent receipt, requiring it for incremental output."""

    receipt_path = staged_path.with_suffix(".resume-handoff.json")
    exists = receipt_path.exists() or receipt_path.is_symlink()
    if not exists:
        if binding.kind == "incremental_handoff":
            raise FileNotFoundError(f"Stage 2 Luna promotion incremental source has no adjacent handoff receipt: {receipt_path}")
        return None
    _require_regular_file(receipt_path, label="adjacent handoff receipt")
    if binding.receipt_path is not None:
        _same_declared_layout(binding.receipt_path, receipt_path, label="incremental receipt path")
    if binding.receipt_sha256 is not None and _sha256(receipt_path) != binding.receipt_sha256:
        raise ValueError("Stage 2 Luna promotion adjacent receipt SHA-256 differs from provenance")

    receipt = _read_json_object(receipt_path, label="adjacent handoff receipt")
    expected_fields = {
        "schema",
        "condition",
        "task_index",
        "identity",
        "raw_log",
        "staged_log",
        "paired_clean",
        "raw_log_dir",
        "manifest",
        "checkpoint",
        "protocol",
    }
    if set(receipt) != expected_fields or receipt.get("schema") != HANDOFF_SCHEMA:
        raise ValueError("Stage 2 Luna promotion adjacent receipt has an unexpected schema or fields")
    if receipt.get("condition") != condition:
        raise ValueError("Stage 2 Luna promotion adjacent receipt condition differs from provenance")
    receipt_identity = _identity_from_mapping(receipt.get("identity"), label="receipt.identity")
    task_index = receipt.get("task_index")
    # The complete five-field identity is analysis-relevant.  Task indices are
    # only a runner scheduling convention, and historic approved handoffs used
    # a different (but still complete) held-out-bias ordering than the current
    # task factory.  Do not reinterpret that original evidence under today's
    # ordering; require an unambiguous biased-task index instead.
    if (
        receipt_identity != identity
        or isinstance(task_index, bool)
        or not isinstance(task_index, int)
        or not 4 <= task_index <= 21
    ):
        raise ValueError("Stage 2 Luna promotion adjacent receipt task identity differs from provenance")

    raw_path, raw_sha256 = _path_and_sha(receipt.get("raw_log"), label="receipt.raw_log")
    declared_staged, staged_sha256 = _path_and_sha(receipt.get("staged_log"), label="receipt.staged_log")
    clean_path, clean_sha256 = _path_and_sha(receipt.get("paired_clean"), label="receipt.paired_clean")
    declared_raw_root = _require_absolute_path(receipt.get("raw_log_dir"), label="receipt.raw_log_dir")
    _under(raw_path, declared_raw_root, label="receipt raw log")
    _under(clean_path, declared_raw_root, label="receipt paired clean log")
    _same_declared_layout(declared_staged, staged_path, label="receipt staged path")
    if raw_sha256 != source_sha256 or staged_sha256 != source_sha256 or _sha256(staged_path) != source_sha256:
        raise ValueError("Stage 2 Luna promotion receipt/source/staged SHA-256 bindings differ")

    for declared, expected_sha256, label in (
        (raw_path, raw_sha256, "receipt raw EvalLog"),
        (clean_path, clean_sha256, "receipt paired-clean EvalLog"),
    ):
        local = _local_raw_evidence_path(roots, condition=condition, declared=declared, label=label)
        _require_regular_file(local, label=label)
        if _sha256(local) != expected_sha256:
            raise ValueError(f"Stage 2 Luna promotion {label} SHA-256 differs from its receipt")

    manifest = _require_mapping(receipt.get("manifest"), label="receipt.manifest")
    if set(manifest) != {"path", "sha256"}:
        raise ValueError("Stage 2 Luna promotion receipt manifest has unexpected fields")
    _require_absolute_path(manifest.get("path"), label="receipt.manifest.path")
    if _require_sha256(manifest.get("sha256"), label="receipt.manifest.sha256") != manifest_sha256:
        raise ValueError("Stage 2 Luna promotion receipt manifest SHA-256 differs from provenance")
    _validate_checkpoint(condition, receipt.get("checkpoint"))
    expected_protocol = {
        "runtime_profile": "hf-peft",
        "prompt_style": PROMPT_STYLE,
        "include_bias_acknowledged": False,
        "max_tokens": 20480,
        "max_connections": 8,
        "validated_with_full_paired_switch_scores": True,
    }
    if receipt.get("protocol") != expected_protocol:
        raise ValueError("Stage 2 Luna promotion receipt protocol differs from the validated native-HF/PEFT policy")
    paired = binding.paired_clean
    if paired.get("raw_log_sha256") != clean_sha256:
        raise ValueError("Stage 2 Luna promotion receipt paired-clean SHA-256 differs from provenance")
    return receipt


def _local_raw_evidence_path(roots: SourceRoots, *, condition: str, declared: Path, label: str) -> Path:
    """Resolve one copied raw file without silently choosing a root layout.

    A source bundle may retain either the condition directory itself or the
    shared ``raw-no-luna`` parent.  Supporting both makes copied historical
    matrices practical, but accepting both candidates at once would let a
    duplicate filename select evidence ambiguously.
    """

    candidates = (roots.raw_root / condition / declared.name, roots.raw_root / declared.name)
    existing = [path for path in candidates if path.exists() or path.is_symlink()]
    if len(existing) != 1:
        rendered = ", ".join(str(path) for path in existing) or "<none>"
        raise FileNotFoundError(f"Stage 2 Luna promotion {label} has no unambiguous copied raw evidence: {rendered}")
    local = existing[0]
    _under(local, roots.raw_root, label=label)
    return local


def _matching_preflight_report(roots: SourceRoots, *, report_sha256: str) -> Path:
    directory = _require_directory(roots.preflight_root, label="preflight root")
    matches: list[Path] = []
    for path in sorted(directory.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        if _sha256(path) == report_sha256:
            matches.append(path.resolve())
    if len(matches) != 1:
        rendered = ", ".join(str(path) for path in matches) or "<none>"
        raise ValueError(
            "Stage 2 Luna promotion needs exactly one local raw-preflight report matching its provenance SHA-256; "
            f"found={rendered}"
        )
    return matches[0]


def _validate_raw_preflight(
    roots: SourceRoots,
    *,
    condition: str,
    identity: Identity,
    source_sha256: str,
    staged_path: Path,
    binding: _Binding,
    manifest_sha256: str,
    receipt: Mapping[str, Any] | None,
) -> None:
    assert binding.report_sha256 is not None  # constructed by _validate_provenance
    report_path = _matching_preflight_report(roots, report_sha256=binding.report_sha256)
    report = raw_preflight.validate_preflight_report(report_path)
    if report.get("condition") != condition or report.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Stage 2 Luna promotion raw-preflight condition or manifest differs from provenance")
    contract = _require_mapping(report.get("contract"), label="raw_preflight.contract")
    runtime = _require_mapping(contract.get("runtime"), label="raw_preflight.contract.runtime")
    expected_checkpoint = _require_absolute_path(runtime.get("expected_checkpoint"), label="raw_preflight.expected_checkpoint")
    checkpoint = condition_spec(condition)
    if runtime.get("profile") != "hf-peft" or expected_checkpoint.name != checkpoint.checkpoint_name:
        raise ValueError("Stage 2 Luna promotion raw-preflight checkpoint identity differs from the condition")
    sources = report.get("sources")
    if not isinstance(sources, list):  # retained for the benefit of monkeypatched/offline validators
        raise ValueError("Stage 2 Luna promotion raw-preflight report has no source list")
    matches = [
        source
        for source in sources
        if isinstance(source, Mapping)
        and (
            source.get("kind"),
            source.get("regime"),
            source.get("population"),
            source.get("dataset"),
            source.get("bias_type"),
        )
        == identity
    ]
    if len(matches) != 1:
        raise ValueError("Stage 2 Luna promotion raw-preflight report has an ambiguous source cell")
    source = matches[0]
    if source.get("raw_log_sha256") != source_sha256:
        raise ValueError("Stage 2 Luna promotion raw-preflight source SHA-256 differs from provenance")
    source_runtime = _require_mapping(source.get("runtime"), label="raw_preflight.source.runtime")
    if source_runtime.get("checkpoint") != str(expected_checkpoint):
        raise ValueError("Stage 2 Luna promotion raw-preflight source checkpoint differs from its condition contract")
    report_paired = source.get("paired_clean")
    if not isinstance(report_paired, Mapping) or dict(report_paired) != dict(binding.paired_clean):
        raise ValueError("Stage 2 Luna promotion raw-preflight paired-clean binding differs from provenance")
    raw_log = _require_absolute_path(source.get("raw_log"), label="raw_preflight.source.raw_log")
    if raw_log.name != staged_path.name:
        raise ValueError("Stage 2 Luna promotion raw-preflight source filename differs from staged evidence")
    if receipt is not None:
        receipt_manifest = _require_mapping(receipt.get("manifest"), label="receipt.manifest")
        if receipt_manifest.get("sha256") != manifest_sha256:
            raise ValueError("Stage 2 Luna promotion receipt manifest differs from raw-preflight provenance")
        receipt_clean = _require_mapping(receipt.get("paired_clean"), label="receipt.paired_clean")
        if receipt_clean.get("sha256") != binding.paired_clean.get("raw_log_sha256"):
            raise ValueError("Stage 2 Luna promotion receipt paired-clean SHA-256 differs from raw preflight")


def _read_eval_log(path: Path) -> Any:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - configured evaluation environment only
        raise RuntimeError("Inspect AI is required to verify a Stage 2 Luna derived EvalLog") from exc
    return read_eval_log(str(path))


def _grade_input(
    *,
    staged_path: Path,
    condition: str,
    identity: Identity,
    source_sha256: str,
    binding: _Binding,
) -> grade_luna.GradeInput:
    _kind, regime, population, dataset, bias_type = identity
    if binding.kind == "raw_preflight":
        assert binding.report_sha256 is not None
        return grade_luna.GradeInput(
            path=staged_path,
            condition=condition,
            regime=regime,
            population=population,
            dataset=dataset,
            bias_type=bias_type,
            created="",
            expected_sha256=source_sha256,
            preflight_report_sha256=binding.report_sha256,
            manifest_sha256=binding.manifest_sha256,
            paired_clean=dict(binding.paired_clean),
        )
    assert binding.receipt_sha256 is not None and binding.receipt_path is not None
    return grade_luna.GradeInput(
        path=staged_path,
        condition=condition,
        regime=regime,
        population=population,
        dataset=dataset,
        bias_type=bias_type,
        created="",
        expected_sha256=source_sha256,
        preflight_report_sha256=binding.receipt_sha256,
        manifest_sha256=binding.manifest_sha256,
        paired_clean=dict(binding.paired_clean),
        binding_kind="incremental_handoff",
        binding_sha256=binding.receipt_sha256,
        binding_path=str(binding.receipt_path),
        binding_schema=HANDOFF_SCHEMA,
    )


def _validate_rows(eval_path: Path, rows_path: Path, *, source: grade_luna.GradeInput) -> None:
    log = _read_eval_log(eval_path)
    if grade_luna._attribute(log, "status") != "success":
        raise ValueError(f"Stage 2 Luna promotion derived EvalLog is not successful: {eval_path}")
    samples = list(grade_luna._attribute(log, "samples", []) or [])
    if not samples:
        raise ValueError(f"Stage 2 Luna promotion derived EvalLog has no samples: {eval_path}")
    for sample in samples:
        _name, score = grade_luna._score_mapping(sample)
        metadata = grade_luna._mapping(grade_luna._attribute(score, "metadata", {}))
        if metadata.get("grader_model") != EXPECTED_GRADER_MODEL or metadata.get("grader_max_tokens") != EXPECTED_GRADER_MAX_TOKENS:
            raise ValueError(f"Stage 2 Luna promotion derived EvalLog has a different embedded grader policy: {eval_path}")
    rows = grade_luna._export_rows(log, source)
    # A cap-hit scorer response is preserved as null rather than fabricated as
    # YES or NO.  It is valid evidence (the rows retain the cap-hit metadata),
    # while any other numeric value would be a different Luna protocol.
    if any(row.get("bias_acknowledged") not in {None, 0.0, 1.0} for row in rows):
        raise ValueError(f"Stage 2 Luna promotion derived EvalLog has an invalid Luna acknowledgement: {eval_path}")
    expected = b"".join((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") for row in rows)
    if rows_path.read_bytes() != expected:
        raise ValueError(f"Stage 2 Luna promotion JSONL rows do not reproduce from their derived EvalLog: {rows_path}")


def _expected_trio_paths(provenance_path: Path) -> tuple[Path, Path, Path]:
    if not provenance_path.name.endswith("-luna.provenance.json"):
        raise ValueError(f"Stage 2 Luna promotion provenance filename is not canonical: {provenance_path}")
    stem = provenance_path.name[: -len(".provenance.json")]
    eval_path = provenance_path.with_name(f"{stem}.eval")
    return eval_path, eval_path.with_suffix(".jsonl"), provenance_path


def _validate_trio(roots: SourceRoots, provenance_path: Path) -> _Candidate:
    eval_path, rows_path, provenance_path = _expected_trio_paths(provenance_path)
    for path, label in (
        (eval_path, "derived EvalLog"),
        (rows_path, "derived JSONL"),
        (provenance_path, "derived provenance"),
    ):
        _require_regular_file(path, label=label)
        _under(path.resolve(), roots.derived_root, label=label)
    provenance = _read_json_object(provenance_path, label="derived provenance")
    condition, identity, source_sha256, declared_source, binding = _validate_provenance(provenance)
    manifest_path = _require_regular_file(roots.manifest, label="frozen manifest")
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != binding.manifest_sha256:
        raise ValueError("Stage 2 Luna promotion local frozen manifest SHA-256 differs from provenance")
    # It is sufficient to hash-bind the copied immutable manifest here.  Full
    # materializer validation needs every original frozen source path to remain
    # mounted, which a legitimate evidence copy need not preserve.
    _read_json_object(manifest_path, label="frozen manifest")
    staged_path = _local_staged_path(roots, declared=declared_source, condition=condition, identity=identity)
    if _sha256(staged_path) != source_sha256:
        raise ValueError("Stage 2 Luna promotion staged source SHA-256 differs from provenance")
    receipt = _validate_receipt(
        roots,
        condition=condition,
        identity=identity,
        source_sha256=source_sha256,
        staged_path=staged_path,
        binding=binding,
        manifest_sha256=manifest_sha256,
    )
    if binding.kind == "raw_preflight":
        _validate_raw_preflight(
            roots,
            condition=condition,
            identity=identity,
            source_sha256=source_sha256,
            staged_path=staged_path,
            binding=binding,
            manifest_sha256=manifest_sha256,
            receipt=receipt,
        )
    source = _grade_input(
        staged_path=staged_path,
        condition=condition,
        identity=identity,
        source_sha256=source_sha256,
        binding=binding,
    )
    expected_eval, expected_rows, expected_provenance = grade_luna.output_paths(roots.derived_root, source)
    if (eval_path, rows_path, provenance_path) != (expected_eval, expected_rows, expected_provenance):
        raise ValueError("Stage 2 Luna promotion derived trio does not use the canonical output layout")
    _validate_rows(eval_path, rows_path, source=source)
    return _Candidate(
        roots=roots,
        eval_path=eval_path,
        rows_path=rows_path,
        provenance_path=provenance_path,
        relative_eval=eval_path.relative_to(roots.derived_root),
        relative_rows=rows_path.relative_to(roots.derived_root),
        relative_provenance=provenance_path.relative_to(roots.derived_root),
        hashes=(_sha256(eval_path), _sha256(rows_path), _sha256(provenance_path)),
    )


def _source_provenance_paths(roots: SourceRoots) -> list[Path]:
    provenance = sorted(roots.derived_root.rglob("*-luna.provenance.json"))
    all_eval = sorted(roots.derived_root.rglob("*.eval"))
    all_rows = sorted(roots.derived_root.rglob("*.jsonl"))
    all_provenance = sorted(roots.derived_root.rglob("*.provenance.json"))
    if not provenance:
        raise FileNotFoundError(f"Stage 2 Luna promotion source root has no derived provenance files: {roots.derived_root}")
    if any(not path.name.endswith("-luna.eval") for path in all_eval):
        raise ValueError(f"Stage 2 Luna promotion source root contains a non-Luna EvalLog: {roots.derived_root}")
    if any(not path.name.endswith("-luna.jsonl") for path in all_rows):
        raise ValueError(f"Stage 2 Luna promotion source root contains a non-Luna JSONL file: {roots.derived_root}")
    if any(not path.name.endswith("-luna.provenance.json") for path in all_provenance):
        raise ValueError(f"Stage 2 Luna promotion source root contains a non-Luna provenance file: {roots.derived_root}")
    expected = {_expected_trio_paths(path) for path in provenance}
    observed = {(path, path.with_suffix(".jsonl"), path.with_suffix(".provenance.json")) for path in all_eval}
    if expected != observed:
        raise ValueError(f"Stage 2 Luna promotion source root has an incomplete or ambiguous derived artifact trio: {roots.derived_root}")
    return provenance


def _coerce_roots(value: SourceRoots) -> SourceRoots:
    return SourceRoots(
        derived_root=_require_directory(value.derived_root, label="derived root"),
        staged_root=_require_directory(value.staged_root, label="staged root"),
        raw_root=_require_directory(value.raw_root, label="raw root"),
        manifest=_require_regular_file(Path(value.manifest), label="frozen manifest"),
        preflight_root=Path(value.preflight_root),
    )


def _validate_roots(roots: Sequence[SourceRoots], output_root: Path) -> tuple[SourceRoots, ...]:
    if not roots:
        raise ValueError("at least one Stage 2 Luna promotion source is required")
    normalized = tuple(_coerce_roots(item) for item in roots)
    if len({item.derived_root for item in normalized}) != len(normalized):
        raise ValueError("Stage 2 Luna promotion derived roots must be distinct")
    for item in normalized:
        for input_root in (item.derived_root, item.staged_root, item.raw_root):
            if output_root == input_root or output_root.is_relative_to(input_root) or input_root.is_relative_to(output_root):
                raise ValueError("Stage 2 Luna promotion destination must be separate and non-nested from every evidence root")
    return normalized


def _files_match(path: Path, expected_sha256: str) -> bool:
    return not path.is_symlink() and path.is_file() and _sha256(path) == expected_sha256


def _ensure_destination_parent(destination: Path, output_root: Path) -> None:
    relative = destination.relative_to(output_root)
    parent = output_root
    for component in relative.parts[:-1]:
        parent = parent / component
        if parent.exists():
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError(f"Stage 2 Luna promotion destination parent is not a real directory: {parent}")
        else:
            parent.mkdir()


def _copy_new(source: Path, destination: Path, *, expected_sha256: str, output_root: Path) -> str:
    if _sha256(source) != expected_sha256:
        raise ValueError(f"Stage 2 Luna promotion source changed after validation: {source}")
    if destination.exists() or destination.is_symlink():
        if _files_match(destination, expected_sha256):
            return "resumed"
        raise FileExistsError(f"Stage 2 Luna promotion refuses to overwrite a differing artifact: {destination}")
    _ensure_destination_parent(destination, output_root)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if _sha256(temporary) != expected_sha256:
            raise ValueError(f"Stage 2 Luna promotion temporary copy has the wrong SHA-256: {source}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not _files_match(destination, expected_sha256):
                raise FileExistsError(f"Stage 2 Luna promotion artifact appeared and differs: {destination}")
            return "resumed"
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return "promoted"


def _equivalent(left: _Candidate, right: _Candidate) -> bool:
    return left.hashes == right.hashes


def promote(
    sources: Sequence[SourceRoots],
    output_root: str | Path,
    *,
    dry_run: bool = False,
) -> list[PromotionResult]:
    """Validate and consolidate existing Luna trios without scoring anything.

    All input validation, including collisions, completes before the first
    destination write.  A stopped copy may be resumed only if every already
    present destination member still has the exact validated bytes.
    """

    destination = Path(output_root)
    if destination.exists() and (destination.is_symlink() or not destination.is_dir()):
        raise ValueError(f"Stage 2 Luna promotion destination is not a real directory: {destination}")
    destination = destination.resolve()
    normalized = _validate_roots(sources, destination)
    candidates = [
        _validate_trio(roots, provenance_path)
        for roots in normalized
        for provenance_path in _source_provenance_paths(roots)
    ]
    by_relative: dict[Path, list[_Candidate]] = {}
    for candidate in candidates:
        by_relative.setdefault(candidate.relative_eval, []).append(candidate)
    chosen: list[_Candidate] = []
    duplicate_status: dict[Path, str] = {}
    for relative, group in sorted(by_relative.items(), key=lambda item: str(item[0])):
        first = group[0]
        if any(not _equivalent(first, other) for other in group[1:]):
            paths = ", ".join(str(item.eval_path) for item in group)
            raise FileExistsError(
                "Stage 2 Luna promotion found non-equivalent source collisions for one destination; "
                f"manual curation is required: {relative} <- {paths}"
            )
        chosen.append(first)
        for duplicate in group[1:]:
            duplicate_status[duplicate.eval_path] = "equivalent"

    existing_status: dict[Path, str] = {}
    for candidate in chosen:
        triples = (
            (candidate.eval_path, destination / candidate.relative_eval, candidate.hashes[0]),
            (candidate.rows_path, destination / candidate.relative_rows, candidate.hashes[1]),
            (candidate.provenance_path, destination / candidate.relative_provenance, candidate.hashes[2]),
        )
        existing = [target.exists() or target.is_symlink() for _source, target, _hash in triples]
        if any(existing):
            for _source, target, expected_sha256 in triples:
                if target.exists() or target.is_symlink():
                    if not _files_match(target, expected_sha256):
                        raise FileExistsError(f"Stage 2 Luna promotion destination collision differs: {target}")
            existing_status[candidate.eval_path] = "resumed" if all(existing) else "completed"
        else:
            existing_status[candidate.eval_path] = "ready" if dry_run else "promoted"

    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        for candidate in chosen:
            triples = (
                (candidate.eval_path, destination / candidate.relative_eval, candidate.hashes[0]),
                (candidate.rows_path, destination / candidate.relative_rows, candidate.hashes[1]),
                (candidate.provenance_path, destination / candidate.relative_provenance, candidate.hashes[2]),
            )
            statuses = [_copy_new(source, target, expected_sha256=digest, output_root=destination) for source, target, digest in triples]
            if existing_status[candidate.eval_path] == "promoted" and all(status == "resumed" for status in statuses):
                existing_status[candidate.eval_path] = "resumed"
            elif existing_status[candidate.eval_path] == "completed" and all(status == "resumed" for status in statuses):
                existing_status[candidate.eval_path] = "resumed"

    results = [
        PromotionResult(
            source_eval=candidate.eval_path,
            destination_eval=destination / candidate.relative_eval,
            status=(
                duplicate_status[candidate.eval_path]
                if candidate.eval_path in duplicate_status
                else existing_status[candidate.eval_path]
            ),
        )
        for candidate in candidates
    ]
    return sorted(results, key=lambda item: (str(item.destination_eval), str(item.source_eval)))


def _roots_from_args(args: argparse.Namespace) -> tuple[SourceRoots, ...]:
    fields = {
        "--source-root": args.source_root,
        "--staged-root": args.staged_root,
        "--raw-root": args.raw_root,
        "--manifest": args.manifest,
        "--preflight-root": args.preflight_root,
    }
    lengths = {len(value) for value in fields.values()}
    if len(lengths) != 1:
        rendered = ", ".join(f"{name}={len(value)}" for name, value in fields.items())
        raise ValueError(f"repeat every per-source promotion argument equally and in matching order ({rendered})")
    return tuple(
        SourceRoots(
            derived_root=derived,
            staged_root=staged,
            raw_root=raw,
            manifest=manifest,
            preflight_root=preflight,
        )
        for derived, staged, raw, manifest, preflight in zip(
            args.source_root,
            args.staged_root,
            args.raw_root,
            args.manifest,
            args.preflight_root,
            strict=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", required=True, type=Path, help="repeat once per derived Luna source tree")
    parser.add_argument("--staged-root", action="append", required=True, type=Path, help="matching copied staged-raw tree")
    parser.add_argument(
        "--raw-root",
        action="append",
        required=True,
        type=Path,
        help="matching copied raw/no-Luna condition tree or its shared parent",
    )
    parser.add_argument("--manifest", action="append", required=True, type=Path, help="matching frozen Stage 2 manifest")
    parser.add_argument("--preflight-root", action="append", required=True, type=Path, help="matching raw-preflight report directory")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate and resolve collisions without writing destination files")
    args = parser.parse_args(argv)
    try:
        results = promote(_roots_from_args(args), args.output_root, dry_run=args.dry_run)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    for result in results:
        print(f"{result.status}: {result.source_eval} -> {result.destination_eval}")


__all__ = [
    "EXPECTED_AGGREGATE_CONNECTION_LIMIT",
    "EXPECTED_CONNECTIONS_PER_WORKER",
    "EXPECTED_GRADER_MAX_TOKENS",
    "EXPECTED_GRADER_MODEL",
    "EXPECTED_WORKERS",
    "PromotionResult",
    "SourceRoots",
    "promote",
]


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
