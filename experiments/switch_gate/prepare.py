"""Freeze the deterministic screen/confirmation split for the switch gate.

This module only reads an existing matched-pair JSONL and writes filtered
JSONL files plus a manifest.  It never imports a model backend or calls a
model service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SELECTION_SEED = "20260729"
AUTHORITATIVE_ROWS = 2048
SCREEN_COUNTS = {"logiqa": 23, "hellaswag": 77}
AUTHORITATIVE_COUNTS = {"logiqa": 472, "hellaswag": 1576}
CONFIRMATION_COUNTS = {"logiqa": 449, "hellaswag": 1499}
CONFIRMATION_SIZES = (600, 800, 1000, 1200, 1600, 1948)

_REQUIRED_FIELDS = frozenset(
    {
        "question",
        "question_id",
        "source_dataset",
        "prompt_style",
        "unbiased_messages",
        "biased_messages",
        "bias_type",
        "ground_truth",
        "biased_option",
        "biasing_text",
    }
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rank(row: Mapping[str, Any]) -> tuple[bytes, str]:
    question_id = row["question_id"]
    digest = hashlib.sha256(f"{SELECTION_SEED}{question_id}".encode("utf-8")).digest()
    return digest, question_id


def _validate_messages(value: object, *, location: str, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}: {field} must be a non-empty message list")
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"{location}: {field}[{index}] must be an object")
        for key in ("role", "content"):
            item = message.get(key)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{location}: {field}[{index}].{key} must be a non-empty string")


def _validate_row(row: object, *, path: Path, line_number: int) -> dict[str, Any]:
    location = f"{path}:{line_number}"
    if not isinstance(row, dict):
        raise ValueError(f"{location}: row must be a JSON object")
    missing = sorted(_REQUIRED_FIELDS - row.keys())
    if missing:
        raise ValueError(f"{location}: missing frozen field(s): {', '.join(missing)}")

    for field in (
        "question",
        "question_id",
        "source_dataset",
        "prompt_style",
        "bias_type",
        "ground_truth",
        "biased_option",
        "biasing_text",
    ):
        if not isinstance(row[field], str):
            raise ValueError(f"{location}: {field} must be a string")
    if not row["question_id"].strip():
        raise ValueError(f"{location}: question_id must not be empty")
    if row["source_dataset"] not in AUTHORITATIVE_COUNTS:
        raise ValueError(
            f"{location}: source_dataset must be one of {sorted(AUTHORITATIVE_COUNTS)}, "
            f"got {row['source_dataset']!r}"
        )
    if row["prompt_style"] != "none":
        raise ValueError(f"{location}: prompt_style must be 'none', got {row['prompt_style']!r}")
    if row["bias_type"] != "wrong_argument":
        raise ValueError(f"{location}: bias_type must be 'wrong_argument', got {row['bias_type']!r}")
    if not row["biased_option"].strip():
        raise ValueError(f"{location}: biased_option must not be empty")
    if row["biased_option"] == row["ground_truth"]:
        raise ValueError(f"{location}: biased_option must be a distractor, not the ground truth")
    _validate_messages(row["unbiased_messages"], location=location, field="unbiased_messages")
    _validate_messages(row["biased_messages"], location=location, field="biased_messages")
    return row


def _authoritative_prefix(path: Path, payload: bytes) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    nonempty_rows = 0
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        nonempty_rows += 1
        if len(rows) >= AUTHORITATIVE_ROWS:
            continue
        try:
            decoded = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        rows.append(_validate_row(decoded, path=path, line_number=line_number))

    if len(rows) != AUTHORITATIVE_ROWS:
        raise ValueError(f"{path}: expected at least {AUTHORITATIVE_ROWS} non-empty rows, found {len(rows)}")
    question_ids = [row["question_id"] for row in rows]
    if len(question_ids) != len(set(question_ids)):
        duplicates = sorted(question_id for question_id, count in Counter(question_ids).items() if count > 1)
        raise ValueError(f"{path}: duplicate question_id(s) in first {AUTHORITATIVE_ROWS} rows: {duplicates[:5]}")
    counts = Counter(row["source_dataset"] for row in rows)
    if dict(counts) != AUTHORITATIVE_COUNTS:
        raise ValueError(
            f"{path}: first {AUTHORITATIVE_ROWS} rows have dataset counts {dict(counts)}, "
            f"expected {AUTHORITATIVE_COUNTS}"
        )
    return rows, nonempty_rows


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _artifact_entry(path: Path, rows: list[dict[str, Any]], payload: bytes) -> dict[str, Any]:
    counts = Counter(row["source_dataset"] for row in rows)
    return {
        "path": str(path.resolve()),
        "row_count": len(rows),
        "counts_by_dataset": {dataset: counts.get(dataset, 0) for dataset in AUTHORITATIVE_COUNTS},
        "question_ids": [row["question_id"] for row in rows],
        "content_sha256": _sha256(payload),
    }


def _confirmation_logiqa_count(size: int) -> int:
    if size == sum(CONFIRMATION_COUNTS.values()):
        return CONFIRMATION_COUNTS["logiqa"]
    return round(size * AUTHORITATIVE_COUNTS["logiqa"] / AUTHORITATIVE_ROWS)


def _write_new_atomic(path: Path, payload: bytes) -> None:
    """Publish a completed temporary file after a no-overwrite check."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        archive = path.parent / "_archive"
        archive.mkdir(exist_ok=True)
        temporary_path.replace(archive / temporary_path.name)
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    temporary_path.replace(path)


def prepare_switch_gate(
    source_pairs: str | Path,
    output_dir: str | Path,
    manifest_output: str | Path,
) -> dict[str, Any]:
    """Create the immutable switch-gate split artifacts and return the manifest."""

    source_path = Path(source_pairs)
    output_path = Path(output_dir)
    manifest_path = Path(manifest_output)
    if not source_path.is_file():
        raise FileNotFoundError(f"source matched-pair JSONL does not exist: {source_path}")

    screen_path = output_path / "screen.jsonl"
    confirmation_paths = {
        f"confirmation-n{size}": output_path / f"confirmation-n{size}.jsonl" for size in CONFIRMATION_SIZES
    }
    targets = [screen_path, *confirmation_paths.values(), manifest_path]
    resolved_targets = [target.resolve() for target in targets]
    if len(resolved_targets) != len(set(resolved_targets)):
        raise ValueError("output JSONLs and --manifest-output must be distinct paths")
    existing = [str(target) for target in targets if target.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output(s): {existing}")

    source_payload = source_path.read_bytes()
    authoritative, source_row_count = _authoritative_prefix(source_path, source_payload)
    strata = {
        dataset: sorted(
            (row for row in authoritative if row["source_dataset"] == dataset),
            key=_rank,
        )
        for dataset in AUTHORITATIVE_COUNTS
    }

    screen_rows = sorted(
        [row for dataset, count in SCREEN_COUNTS.items() for row in strata[dataset][:count]],
        key=_rank,
    )
    remaining = {dataset: strata[dataset][SCREEN_COUNTS[dataset] :] for dataset in AUTHORITATIVE_COUNTS}

    confirmation_rows: dict[str, list[dict[str, Any]]] = {}
    for size in CONFIRMATION_SIZES:
        logiqa_count = _confirmation_logiqa_count(size)
        counts = {"logiqa": logiqa_count, "hellaswag": size - logiqa_count}
        name = f"confirmation-n{size}"
        confirmation_rows[name] = sorted(
            [row for dataset, count in counts.items() for row in remaining[dataset][:count]],
            key=_rank,
        )

    screen_ids = {row["question_id"] for row in screen_rows}
    full_confirmation = confirmation_rows[f"confirmation-n{CONFIRMATION_SIZES[-1]}"]
    full_confirmation_ids = {row["question_id"] for row in full_confirmation}
    authoritative_ids = {row["question_id"] for row in authoritative}
    confirmation_id_sets = [
        {row["question_id"] for row in confirmation_rows[f"confirmation-n{size}"]} for size in CONFIRMATION_SIZES
    ]
    assertions = {
        "authoritative_ids_unique": len(authoritative_ids) == AUTHORITATIVE_ROWS,
        "screen_disjoint_from_confirmation": screen_ids.isdisjoint(full_confirmation_ids),
        "confirmation_prefixes_nested": all(
            earlier < later for earlier, later in zip(confirmation_id_sets, confirmation_id_sets[1:])
        ),
        "full_confirmation_is_authoritative_remainder": full_confirmation_ids == authoritative_ids - screen_ids,
        "screen_union_full_confirmation_is_authoritative_prefix": (
            screen_ids | full_confirmation_ids == authoritative_ids
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(f"internal split invariant failed: {assertions}")

    screen_payload = _jsonl_payload(screen_rows)
    confirmation_payloads = {name: _jsonl_payload(rows) for name, rows in confirmation_rows.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "switch_gate_split_manifest",
        "source": {
            "path": str(source_path.resolve()),
            "content_sha256": _sha256(source_payload),
            "row_count": source_row_count,
            "authoritative_prefix_row_count": AUTHORITATIVE_ROWS,
            "authoritative_prefix_counts_by_dataset": AUTHORITATIVE_COUNTS,
        },
        "selection": {
            "seed": SELECTION_SEED,
            "rank": "sha256(utf8(seed + question_id)) within source_dataset",
            "screen_counts_by_dataset": SCREEN_COUNTS,
            "confirmation_sizes": list(CONFIRMATION_SIZES),
            "confirmation_allocation": "round(n * 472 / 2048) LogiQA; remainder HellaSwag",
        },
        "screen": _artifact_entry(screen_path, screen_rows, screen_payload),
        "confirmation": {
            name: _artifact_entry(confirmation_paths[name], rows, confirmation_payloads[name])
            for name, rows in confirmation_rows.items()
        },
        "assertions": assertions,
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    _write_new_atomic(screen_path, screen_payload)
    for name in confirmation_paths:
        _write_new_atomic(confirmation_paths[name], confirmation_payloads[name])
    _write_new_atomic(manifest_path, manifest_payload)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Freeze deterministic local-only inputs for the switch-gate experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_switch_gate(args.source_pairs, args.output_dir, args.manifest_output)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Prepared screen n={manifest['screen']['row_count']} and "
        f"{len(manifest['confirmation'])} confirmation files; no model or Tinker call was made."
    )


if __name__ == "__main__":
    main()
