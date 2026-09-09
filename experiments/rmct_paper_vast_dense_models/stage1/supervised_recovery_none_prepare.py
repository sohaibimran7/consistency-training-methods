"""Prepare and verify immutable inputs for the Qwen3.5 no-CoT SFT recovery.

This module is deliberately narrow.  It accepts only the attested conversion
of the original 3,000-row legacy-G4 wrong-argument source to ``prompt_style:
none``.  It then derives every prompt-dependent SFT input from that recovered
source, while treating the already-sampled cleaned-Alpaca instruction targets
as a separately verified immutable input.

It makes no model or network call.  BCT target sampling remains a separate
four-GPU command in the experiment plan.  The two CLI operations are designed
to be run before any training target:

``prepare``
    validate the recovered source and immutable instruction targets, derive
    canonical consistency pairs and repaired-ACT splits, and write a single
    completion manifest last;

``verify-bct-targets``
    prove that fresh BCT/main-control targets were generated from that same
    recovered source, rather than from the historical CoT prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm.artifacts import plain_file_identity, write_atomic_bytes
from ctm.training.bct_targets import BCT_TARGET_SCHEMA_VERSION
from ctm_data.adapters.mcq_bias.consistency_pairs import materialize_consistency_pairs
from ctm_data.adapters.mcq_bias.data import _validate_frozen_row
from ctm_data.adapters.mcq_bias.recover_legacy_cot_to_none import (
    ARTIFACT_KIND as RECOVERY_ARTIFACT_KIND,
    LEGACY_G4_WRONG_ARGUMENT_TEMPLATE,
    LEGACY_COT_TERMINAL,
    NONE_ANSWER_FORMAT_TERMINAL,
    SCHEMA_VERSION as RECOVERY_SCHEMA_VERSION,
    TRANSFORM_VERSION as RECOVERY_TRANSFORM_VERSION,
)

SCHEMA_VERSION = 1
MODEL = "Qwen/Qwen3.5-9B"

# These values identify the exact frozen source recovery approved for the
# Qwen3.5 Stage-1 replacement.  The source data are deterministic and the
# conversion serializes sorted JSON, so pinning the output digest is stronger
# than merely trusting a ``prompt_style`` metadata label.
LEGACY_COT_SOURCE_SHA256 = "dfc10d55e51de48566488d107cdc90a0965ac5a73a80f3af68e605e40eff24ad"
RECOVERED_NONE_SOURCE_SHA256 = "7d113ee1858426721d09b23a78f4bbb0e9b16e7576b3ee5ab7da4924c3a0ef3b"
RECOVERED_NONE_ROWS = 3000
DATASETS = ("logiqa", "hellaswag")
RECOVERED_COUNTS = {"logiqa": 1500, "hellaswag": 1500}

# The old Stage-1 order alternates datasets, so this exact prefix remains
# balanced: 1,024 LogiQA + 1,024 HellaSwag.  Do not substitute a shuffled or
# relabelled source here: these offsets are part of the frozen recovery plan.
STANDARD_TRAINING_ROWS = 2048
STANDARD_TRAINING_COUNTS = {"logiqa": 1024, "hellaswag": 1024}
REPAIRED_ACT_ROWS = 200
REPAIRED_ACT_COUNTS = {"logiqa": 100, "hellaswag": 100}
REPAIRED_ACT_HELDOUT_START = STANDARD_TRAINING_ROWS
REPAIRED_ACT_HELDOUT_STOP = REPAIRED_ACT_HELDOUT_START + REPAIRED_ACT_ROWS

# The cleaned-Alpaca targets were generated once from the frozen base.  Their
# source prompts contain no MCQ wrapper and are invariant to this no-CoT
# recovery.  Both main/control JSONLs are intentionally byte-identical.
FROZEN_INSTRUCTION_SOURCE_SHA256 = "e264c3e20505c36cf4c30a9f6fa883080bb8a0815f9bc3dce05237ea6d2701b3"
FROZEN_INSTRUCTION_TARGET_SHA256 = "dbd27bb3eb4ce56bd149626c7e8b18b8cfce903473242e246133b24e41723163"
FROZEN_INSTRUCTION_ROWS = 2048
# The frozen instruction targets retain their independently attested 32-way
# generation. Fresh recovered-prompt BCT targets were deliberately generated
# with 96-way concurrency; keep their provenance contract separate.
TARGET_GENERATION = {"max_tokens": 20480, "temperature": 1.0, "max_concurrency": 32}
FRESH_BCT_TARGET_GENERATION = {"max_tokens": 20480, "temperature": 1.0, "max_concurrency": 96}

RECOVERED_MANIFEST_KIND = RECOVERY_ARTIFACT_KIND
PREPARED_INPUTS_KIND = "qwen35_supervised_none_recovery_inputs"
SPLITS_KIND = "qwen35_supervised_none_recovery_splits"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _expect_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _identity(path: Path) -> dict[str, Any]:
    return plain_file_identity(path)


def _read_jsonl(path: Path, *, label: str) -> tuple[list[dict[str, Any]], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{label} has a blank line at {path}:{line_number}")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON in {label} at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} row must be an object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} has no rows: {path}")
    return rows, payload


def _require_counts(
    rows: Sequence[Mapping[str, Any]], *, expected_rows: int, expected_counts: Mapping[str, int], label: str
) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"{label}: expected {expected_rows} rows, got {len(rows)}")
    counts = Counter(str(row.get("source_dataset", "")) for row in rows)
    actual = {dataset: counts.get(dataset, 0) for dataset in DATASETS}
    if actual != dict(expected_counts):
        raise ValueError(f"{label}: expected dataset counts {dict(expected_counts)}, got {actual}")


def verify_recovered_none_pairs(source: str | Path, source_manifest: str | Path) -> list[dict[str, Any]]:
    """Validate the exact content-level CoT-to-none source recovery.

    This is intentionally stronger than a manifest/hash check: every row is
    checked against the historical legacy-G4 body and the canonical ``none``
    terminal.  A source with a changed wrapper, a metadata-only relabel, or a
    retained terminal reasoning instruction is rejected before SFT data are
    derived from it.
    """

    source_path = Path(source)
    manifest_path = Path(source_manifest)
    manifest = _json_object(manifest_path, label="recovered-none manifest")
    _expect_equal(manifest.get("kind"), RECOVERED_MANIFEST_KIND, label="recovered-none manifest.kind")
    _expect_equal(
        manifest.get("schema_version"), RECOVERY_SCHEMA_VERSION, label="recovered-none manifest.schema_version"
    )

    transform = manifest.get("transform")
    if not isinstance(transform, Mapping):
        raise ValueError("recovered-none manifest.transform must be an object")
    _expect_equal(transform.get("version"), RECOVERY_TRANSFORM_VERSION, label="recovered-none transform.version")
    _expect_equal(transform.get("source_prompt_style"), "encourage_cot", label="recovered-none source prompt style")
    _expect_equal(transform.get("target_prompt_style"), "none", label="recovered-none target prompt style")
    _expect_equal(transform.get("bias_type"), "wrong_argument", label="recovered-none bias type")

    source_identity = manifest.get("source")
    output_identity = manifest.get("output")
    selection = manifest.get("selection")
    if (
        not isinstance(source_identity, Mapping)
        or not isinstance(output_identity, Mapping)
        or not isinstance(selection, Mapping)
    ):
        raise ValueError("recovered-none manifest must contain source, output, and selection objects")
    _expect_equal(source_identity.get("content_sha256"), LEGACY_COT_SOURCE_SHA256, label="legacy CoT source digest")
    _expect_equal(source_identity.get("row_count"), RECOVERED_NONE_ROWS, label="legacy CoT source row count")
    _expect_equal(
        output_identity.get("content_sha256"), RECOVERED_NONE_SOURCE_SHA256, label="recovered none source digest"
    )
    _expect_equal(output_identity.get("row_count"), RECOVERED_NONE_ROWS, label="recovered none source row count")
    _expect_equal(selection.get("row_count"), RECOVERED_NONE_ROWS, label="recovered none selection row count")
    _expect_equal(selection.get("source_dataset_counts"), RECOVERED_COUNTS, label="recovered none dataset counts")

    rows, payload = _read_jsonl(source_path, label="recovered-none source")
    _expect_equal(_sha256(payload), RECOVERED_NONE_SOURCE_SHA256, label="recovered-none source bytes")
    _require_counts(
        rows, expected_rows=RECOVERED_NONE_ROWS, expected_counts=RECOVERED_COUNTS, label="recovered-none source"
    )

    question_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        validated = _validate_frozen_row(row, path=source_path, line_number=line_number)
        location = f"{source_path}:{line_number}"
        if validated["prompt_style"] != "none" or validated["bias_type"] != "wrong_argument":
            raise ValueError(f"{location}: expected a none-style wrong_argument row")
        question_id = validated["question_id"]
        if question_id in question_ids:
            raise ValueError(f"{location}: duplicate question_id {question_id!r}")
        question_ids.add(question_id)
        unbiased = validated["unbiased_messages"]
        biased = validated["biased_messages"]
        if (
            len(unbiased) != 1
            or len(biased) != 1
            or unbiased[0].get("role") != "user"
            or biased[0].get("role") != "user"
        ):
            raise ValueError(f"{location}: recovered legacy-G4 rows must contain one user message per view")
        expected_unbiased = validated["question"] + NONE_ANSWER_FORMAT_TERMINAL
        expected_biased = (
            LEGACY_G4_WRONG_ARGUMENT_TEMPLATE.format(argument=validated["biasing_text"], question=validated["question"])
            + NONE_ANSWER_FORMAT_TERMINAL
        )
        if unbiased[0].get("content") != expected_unbiased or biased[0].get("content") != expected_biased:
            raise ValueError(f"{location}: recovered none messages do not match the attested legacy-G4 conversion")
        if unbiased[0]["content"].endswith(LEGACY_COT_TERMINAL) or biased[0]["content"].endswith(LEGACY_COT_TERMINAL):
            raise ValueError(f"{location}: retained the terminal CoT instruction")
    return rows


def verify_frozen_instruction_targets(
    main: str | Path,
    control: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify the only reusable SFT input: cleaned-Alpaca base targets."""

    main_path, control_path, target_manifest_path = Path(main), Path(control), Path(manifest_path)
    manifest = _json_object(target_manifest_path, label="frozen instruction-target manifest")
    _expect_equal(manifest.get("kind"), "ctm_bct_targets", label="instruction target manifest.kind")
    _expect_equal(
        manifest.get("schema_version"), BCT_TARGET_SCHEMA_VERSION, label="instruction target manifest.schema_version"
    )
    _expect_equal(manifest.get("model"), MODEL, label="instruction target model")
    _expect_equal(manifest.get("backend"), "FrozenBaseVLLMBackend", label="instruction target backend")
    _expect_equal(manifest.get("row_count"), FROZEN_INSTRUCTION_ROWS, label="instruction target row count")
    _expect_equal(manifest.get("generation"), TARGET_GENERATION, label="instruction target generation")
    _expect_equal(
        manifest.get("fields"),
        {
            "source_messages": "reference_messages",
            "main_messages": "variant_messages",
            "control_messages": "reference_messages",
        },
        label="instruction target fields",
    )

    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or len(source_files) != 1 or not isinstance(source_files[0], Mapping):
        raise ValueError("instruction target manifest.source_files must contain exactly one object")
    _expect_equal(
        source_files[0].get("content_sha256"),
        FROZEN_INSTRUCTION_SOURCE_SHA256,
        label="instruction prompt source digest",
    )
    _expect_equal(source_files[0].get("row_count"), FROZEN_INSTRUCTION_ROWS, label="instruction prompt source rows")

    main_rows, main_payload = _read_jsonl(main_path, label="frozen instruction main targets")
    control_rows, control_payload = _read_jsonl(control_path, label="frozen instruction control targets")
    _expect_equal(_sha256(main_payload), FROZEN_INSTRUCTION_TARGET_SHA256, label="instruction main target bytes")
    _expect_equal(_sha256(control_payload), FROZEN_INSTRUCTION_TARGET_SHA256, label="instruction control target bytes")
    if main_payload != control_payload:
        raise ValueError("frozen instruction main/control targets must remain byte-identical")
    if len(main_rows) != FROZEN_INSTRUCTION_ROWS or len(control_rows) != FROZEN_INSTRUCTION_ROWS:
        raise ValueError("frozen instruction target row count does not match its manifest")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("instruction target manifest.outputs must be an object")
    for name in ("main", "control"):
        entry = outputs.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"instruction target manifest.outputs.{name} must be an object")
        _expect_equal(
            entry.get("content_sha256"), FROZEN_INSTRUCTION_TARGET_SHA256, label=f"instruction {name} target digest"
        )

    for index, row in enumerate(main_rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"frozen instruction target row {index} lacks a prompt and completion")
        assistant = messages[-1]
        if (
            not isinstance(assistant, Mapping)
            or assistant.get("role") != "assistant"
            or not str(assistant.get("content", "")).strip()
        ):
            raise ValueError(f"frozen instruction target row {index} lacks a non-empty assistant target")

    return {
        "main": _identity(main_path),
        "control": _identity(control_path),
        "manifest": _identity(target_manifest_path),
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)


def _publish_exact(path: Path, payload: bytes) -> str:
    """Write once, or accept only a byte-identical prior artifact."""

    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing recovery artifact: {path}")
        return "resumed"
    write_atomic_bytes(path, payload)
    return "written"


def _split_entry(
    path: Path, rows: Sequence[Mapping[str, Any]], payload: bytes, *, source_rows: tuple[int, int]
) -> dict[str, Any]:
    counts = Counter(str(row["source_dataset"]) for row in rows)
    ids = [str(row["question_id"]) for row in rows]
    return {
        "path": str(path.resolve()),
        "content_sha256": _sha256(payload),
        "row_count": len(rows),
        "counts_by_dataset": {dataset: counts.get(dataset, 0) for dataset in DATASETS},
        "question_ids_sha256": _sha256("".join(f"{question_id}\n" for question_id in ids).encode("utf-8")),
        "source_rows_1_based_inclusive": list(source_rows),
    }


def _prepare_repaired_act_splits(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, source: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    """Publish immutable n=200 train/held-out selections for repaired ACT."""

    train_rows = list(rows[:REPAIRED_ACT_ROWS])
    heldout_rows = list(rows[REPAIRED_ACT_HELDOUT_START:REPAIRED_ACT_HELDOUT_STOP])
    _require_counts(
        train_rows,
        expected_rows=REPAIRED_ACT_ROWS,
        expected_counts=REPAIRED_ACT_COUNTS,
        label="repaired ACT train split",
    )
    _require_counts(
        heldout_rows,
        expected_rows=REPAIRED_ACT_ROWS,
        expected_counts=REPAIRED_ACT_COUNTS,
        label="repaired ACT held-out split",
    )
    if {str(row["question_id"]) for row in train_rows} & {str(row["question_id"]) for row in heldout_rows}:
        raise ValueError("repaired ACT train and held-out question IDs overlap")

    train_path = output_dir / "repaired-act-train-n200.jsonl"
    heldout_path = output_dir / "repaired-act-heldout-n200.jsonl"
    manifest_path = output_dir / "repaired-act-splits.manifest.json"
    train_payload, heldout_payload = _jsonl_bytes(train_rows), _jsonl_bytes(heldout_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": SPLITS_KIND,
        "source": _identity(source),
        "selection": {
            "method": "fixed_source_offsets_without_shuffle",
            "train_source_rows_1_based_inclusive": [1, REPAIRED_ACT_ROWS],
            "heldout_source_rows_1_based_inclusive": [REPAIRED_ACT_HELDOUT_START + 1, REPAIRED_ACT_HELDOUT_STOP],
            "assertions": {"disjoint_question_ids": True, "prompt_style": "none"},
        },
        "splits": {
            "train": _split_entry(train_path, train_rows, train_payload, source_rows=(1, REPAIRED_ACT_ROWS)),
            "heldout": _split_entry(
                heldout_path,
                heldout_rows,
                heldout_payload,
                source_rows=(REPAIRED_ACT_HELDOUT_START + 1, REPAIRED_ACT_HELDOUT_STOP),
            ),
        },
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    statuses = {
        "repaired_act_train": _publish_exact(train_path, train_payload),
        "repaired_act_heldout": _publish_exact(heldout_path, heldout_payload),
        "repaired_act_splits_manifest": _publish_exact(manifest_path, manifest_payload),
    }
    return manifest, statuses


def _canonical_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "standard": output_dir / "canonical-consistency-pairs-n2048.jsonl",
        "standard_manifest": output_dir / "canonical-consistency-pairs-n2048.manifest.json",
        "repaired_act_train": output_dir / "canonical-repaired-act-train-n200.jsonl",
        "repaired_act_train_manifest": output_dir / "canonical-repaired-act-train-n200.manifest.json",
        "repaired_act_heldout": output_dir / "canonical-repaired-act-heldout-n200.jsonl",
        "repaired_act_heldout_manifest": output_dir / "canonical-repaired-act-heldout-n200.manifest.json",
    }


def prepare_supervised_recovery_inputs(
    *,
    source: str | Path,
    source_manifest: str | Path,
    instruction_main: str | Path,
    instruction_control: str | Path,
    instruction_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare every CPU-only input consumed by the no-CoT SFT recovery."""

    source_path = Path(source)
    output_path = Path(output_dir)
    rows = verify_recovered_none_pairs(source_path, source_manifest)
    _require_counts(
        rows[:STANDARD_TRAINING_ROWS],
        expected_rows=STANDARD_TRAINING_ROWS,
        expected_counts=STANDARD_TRAINING_COUNTS,
        label="standard supervised training prefix",
    )
    instruction_identity = verify_frozen_instruction_targets(
        instruction_main, instruction_control, instruction_manifest
    )

    split_manifest, statuses = _prepare_repaired_act_splits(rows, output_path, source_path)
    paths = _canonical_paths(output_path)
    canonical_statuses = {
        "standard_canonical": materialize_consistency_pairs(
            source_path,
            paths["standard"],
            paths["standard_manifest"],
            limit=STANDARD_TRAINING_ROWS,
        ),
        "repaired_act_train_canonical": materialize_consistency_pairs(
            output_path / "repaired-act-train-n200.jsonl",
            paths["repaired_act_train"],
            paths["repaired_act_train_manifest"],
        ),
        "repaired_act_heldout_canonical": materialize_consistency_pairs(
            output_path / "repaired-act-heldout-n200.jsonl",
            paths["repaired_act_heldout"],
            paths["repaired_act_heldout_manifest"],
        ),
    }
    statuses.update(canonical_statuses)

    prepared_manifest_path = output_path / "supervised-recovery-inputs.manifest.json"
    prepared_manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": PREPARED_INPUTS_KIND,
        "model": MODEL,
        "recovered_none_source": {
            "source": _identity(source_path),
            "recovery_manifest": _identity(Path(source_manifest)),
        },
        "standard_training": {
            "rows": STANDARD_TRAINING_ROWS,
            "counts_by_dataset": STANDARD_TRAINING_COUNTS,
            "canonical": {
                "data": _identity(paths["standard"]),
                "manifest": _identity(paths["standard_manifest"]),
            },
        },
        "repaired_act": {
            "rows_per_split": REPAIRED_ACT_ROWS,
            "counts_by_dataset": REPAIRED_ACT_COUNTS,
            "splits_manifest": _identity(output_path / "repaired-act-splits.manifest.json"),
            "train_canonical": {
                "data": _identity(paths["repaired_act_train"]),
                "manifest": _identity(paths["repaired_act_train_manifest"]),
            },
            "heldout_canonical": {
                "data": _identity(paths["repaired_act_heldout"]),
                "manifest": _identity(paths["repaired_act_heldout_manifest"]),
            },
        },
        "reused_instruction_targets": instruction_identity,
        "assertions": {
            "prompt_style": "none",
            "prompt_dependent_artifacts_derived_from_recovered_none_source": True,
            "instruction_targets_reused_only_after_hash_verification": True,
        },
    }
    prepared_payload = (json.dumps(prepared_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    statuses["prepared_inputs_manifest"] = _publish_exact(prepared_manifest_path, prepared_payload)
    return {"statuses": statuses, "manifest": prepared_manifest}


def _validate_bct_target_rows(
    *,
    main_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> None:
    if len(main_rows) != STANDARD_TRAINING_ROWS or len(control_rows) != STANDARD_TRAINING_ROWS:
        raise ValueError("fresh BCT target files must each contain exactly 2,048 rows")
    for index, (main, control, source) in enumerate(zip(main_rows, control_rows, source_rows, strict=True), start=1):
        source_id = source["question_id"]
        if main.get("source_id") != source_id or control.get("source_id") != source_id:
            raise ValueError(f"fresh BCT target row {index} does not retain recovered source_id")
        main_messages = main.get("messages")
        control_messages = control.get("messages")
        if not isinstance(main_messages, list) or not isinstance(control_messages, list):
            raise ValueError(f"fresh BCT target row {index} has no messages")
        if main_messages[:-1] != source["biased_messages"]:
            raise ValueError(f"fresh BCT main target row {index} does not use the recovered none biased prompt")
        if control_messages[:-1] != source["unbiased_messages"]:
            raise ValueError(f"fresh BCT control target row {index} does not use the recovered none clean prompt")
        if not main_messages or not control_messages or main_messages[-1] != control_messages[-1]:
            raise ValueError(f"fresh BCT target row {index} lacks a matched main/control completion")
        assistant = main_messages[-1]
        if (
            not isinstance(assistant, Mapping)
            or assistant.get("role") != "assistant"
            or not str(assistant.get("content", "")).strip()
        ):
            raise ValueError(f"fresh BCT target row {index} lacks a non-empty assistant completion")


def verify_fresh_bct_targets(
    *,
    source: str | Path,
    source_manifest: str | Path,
    main: str | Path,
    control: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Require BCT main/control targets to be derived from recovered none rows."""

    source_path = Path(source)
    source_rows = verify_recovered_none_pairs(source_path, source_manifest)
    main_path, control_path, target_manifest_path = Path(main), Path(control), Path(manifest_path)
    manifest = _json_object(target_manifest_path, label="fresh BCT target manifest")
    _expect_equal(manifest.get("kind"), "ctm_bct_targets", label="fresh BCT target manifest.kind")
    _expect_equal(
        manifest.get("schema_version"), BCT_TARGET_SCHEMA_VERSION, label="fresh BCT target manifest.schema_version"
    )
    _expect_equal(manifest.get("model"), MODEL, label="fresh BCT target model")
    _expect_equal(manifest.get("backend"), "FrozenBaseVLLMBackend", label="fresh BCT target backend")
    _expect_equal(manifest.get("row_count"), STANDARD_TRAINING_ROWS, label="fresh BCT target row count")
    _expect_equal(
        manifest.get("generation"),
        FRESH_BCT_TARGET_GENERATION,
        label="fresh BCT target generation",
    )
    _expect_equal(
        manifest.get("fields"),
        {
            "source_messages": "unbiased_messages",
            "main_messages": "biased_messages",
            "control_messages": "unbiased_messages",
        },
        label="fresh BCT target fields",
    )
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or len(source_files) != 1 or not isinstance(source_files[0], Mapping):
        raise ValueError("fresh BCT target manifest.source_files must contain exactly one object")
    source_identity = source_files[0]
    _expect_equal(
        source_identity.get("content_sha256"), RECOVERED_NONE_SOURCE_SHA256, label="fresh BCT target source digest"
    )
    _expect_equal(source_identity.get("row_count"), RECOVERED_NONE_ROWS, label="fresh BCT target source rows")

    main_rows, main_payload = _read_jsonl(main_path, label="fresh BCT main targets")
    control_rows, control_payload = _read_jsonl(control_path, label="fresh BCT control targets")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("fresh BCT target manifest.outputs must be an object")
    for name, payload in (("main", main_payload), ("control", control_payload)):
        entry = outputs.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"fresh BCT target manifest.outputs.{name} must be an object")
        _expect_equal(entry.get("content_sha256"), _sha256(payload), label=f"fresh BCT {name} target bytes")
    _validate_bct_target_rows(
        main_rows=main_rows, control_rows=control_rows, source_rows=source_rows[:STANDARD_TRAINING_ROWS]
    )
    return {
        "main": _identity(main_path),
        "control": _identity(control_path),
        "manifest": _identity(target_manifest_path),
    }


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True, help="attested 3,000-row recovered none-style pair JSONL")
    parser.add_argument("--source-manifest", type=Path, required=True, help="CoT-to-none recovery manifest")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="verify inputs and materialize CPU-only recovery artifacts")
    _add_source_args(prepare)
    prepare.add_argument("--instruction-main", type=Path, required=True)
    prepare.add_argument("--instruction-control", type=Path, required=True)
    prepare.add_argument("--instruction-manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    verify_bct = subparsers.add_parser(
        "verify-bct-targets", help="verify fresh BCT targets bind to the recovered none source"
    )
    _add_source_args(verify_bct)
    verify_bct.add_argument("--main", type=Path, required=True)
    verify_bct.add_argument("--control", type=Path, required=True)
    verify_bct.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_supervised_recovery_inputs(
                source=args.source,
                source_manifest=args.source_manifest,
                instruction_main=args.instruction_main,
                instruction_control=args.instruction_control,
                instruction_manifest=args.instruction_manifest,
                output_dir=args.output_dir,
            )
            print("prepared no-CoT supervised recovery inputs: " + json.dumps(result["statuses"], sort_keys=True))
        else:
            result = verify_fresh_bct_targets(
                source=args.source,
                source_manifest=args.source_manifest,
                main=args.main,
                control=args.control,
                manifest_path=args.manifest,
            )
            print("verified fresh no-CoT BCT targets: " + json.dumps(result, sort_keys=True))
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()


__all__ = [
    "FROZEN_INSTRUCTION_TARGET_SHA256",
    "LEGACY_COT_SOURCE_SHA256",
    "MODEL",
    "RECOVERED_NONE_SOURCE_SHA256",
    "STANDARD_TRAINING_ROWS",
    "prepare_supervised_recovery_inputs",
    "verify_fresh_bct_targets",
    "verify_frozen_instruction_targets",
    "verify_recovered_none_pairs",
]
