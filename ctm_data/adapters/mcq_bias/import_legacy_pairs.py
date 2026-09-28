"""Import legacy cot-transparency prompt pairs as native ``mcq_bias`` rows.

The released ``distractor_argument_g4`` files already contain matched biased
and unbiased prompts.  Reusing those prompts avoids regenerating wrong
arguments, but they use the legacy field names and include rows whose biased
prompt is empty.  This module is the translation boundary: it filters invalid
legacy rows, selects a deterministic per-source prefix, and round-robins the
sources into the native frozen-row schema.

For normal legacy rows, ``biasing_text`` is the single non-empty body enclosed
by ``<argument>`` and ``</argument>`` in ``biased_question``.  If that exact
structure cannot be isolated safely, the biased prompt is still usable and a
deliberately conspicuous, provenance-accurate marker is stored instead.  The
manifest reports both cases so downstream users never mistake the marker for
recovered argument text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ctm.artifacts import write_atomic_bytes
from ctm_data.adapters.mcq_bias.data import _validate_frozen_row

LEGACY_BIAS_NAME = "distractor_argument"
NATIVE_BIAS_TYPE = "wrong_argument"
FALLBACK_BIASING_TEXT = (
    "[legacy distractor_argument: exact argument text could not be isolated; "
    "the original bias is preserved in biased_messages]"
)

_ARGUMENT_OPEN = "<argument>"
_ARGUMENT_CLOSE = "</argument>"
_MAX_INVALID_EXAMPLES = 10


class LegacyRowError(ValueError):
    """One legacy row cannot be translated without guessing."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _ConvertedRow:
    row: dict[str, Any]
    biasing_text_mode: str


def _jsonl_line(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _require_nonempty_string(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise LegacyRowError(f"invalid_{field}", f"{field} must be a non-empty string")
    return value


def _convert_messages(row: dict[str, Any], legacy_field: str) -> list[dict[str, str]]:
    messages = row.get(legacy_field)
    if not isinstance(messages, list) or not messages:
        raise LegacyRowError(
            f"empty_{legacy_field}",
            f"{legacy_field} must be a non-empty message list",
        )
    converted = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise LegacyRowError(
                f"invalid_{legacy_field}",
                f"{legacy_field}[{index}] must be an object",
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip() or not isinstance(content, str) or not content.strip():
            raise LegacyRowError(
                f"invalid_{legacy_field}",
                f"{legacy_field}[{index}] must contain non-empty string role/content fields",
            )
        converted.append({"role": role, "content": content})
    return converted


def _extract_biasing_text(messages: Sequence[dict[str, str]]) -> tuple[str, str]:
    """Return an unambiguous argument body, or the documented fallback marker."""

    contents = [message["content"] for message in messages]
    if sum(content.count(_ARGUMENT_OPEN) for content in contents) != 1:
        return FALLBACK_BIASING_TEXT, "fallback_marker"
    if sum(content.count(_ARGUMENT_CLOSE) for content in contents) != 1:
        return FALLBACK_BIASING_TEXT, "fallback_marker"

    for content in contents:
        start = content.find(_ARGUMENT_OPEN)
        if start < 0:
            continue
        end = content.find(_ARGUMENT_CLOSE, start + len(_ARGUMENT_OPEN))
        if end < 0:
            return FALLBACK_BIASING_TEXT, "fallback_marker"
        argument = content[start + len(_ARGUMENT_OPEN) : end].strip()
        if argument and _ARGUMENT_OPEN not in argument and _ARGUMENT_CLOSE not in argument:
            return argument, "argument_block"
        return FALLBACK_BIASING_TEXT, "fallback_marker"
    return FALLBACK_BIASING_TEXT, "fallback_marker"


def convert_legacy_row(
    row: object,
    *,
    path: str | Path,
    line_number: int,
    prompt_style: str = "encourage_cot",
) -> tuple[dict[str, Any], str]:
    """Translate and validate one legacy row.

    Returns ``(native_row, biasing_text_mode)``.  The second item is either
    ``"argument_block"`` or ``"fallback_marker"`` for manifest accounting.
    """

    source_path = Path(path)
    if not isinstance(row, dict):
        raise LegacyRowError("row_not_object", "row must be a JSON object")

    bias_name = _require_nonempty_string(row, "bias_name")
    if bias_name != LEGACY_BIAS_NAME:
        raise LegacyRowError(
            "unexpected_bias_name",
            f"bias_name must be {LEGACY_BIAS_NAME!r}, got {bias_name!r}",
        )

    unbiased_messages = _convert_messages(row, "unbiased_question")
    biased_messages = _convert_messages(row, "biased_question")
    biasing_text, biasing_text_mode = _extract_biasing_text(biased_messages)
    native = {
        "question": _require_nonempty_string(row, "original_question"),
        "question_id": _require_nonempty_string(row, "original_question_hash"),
        "source_dataset": _require_nonempty_string(row, "original_dataset"),
        "prompt_style": prompt_style,
        "unbiased_messages": unbiased_messages,
        "biased_messages": biased_messages,
        "bias_type": NATIVE_BIAS_TYPE,
        "ground_truth": _require_nonempty_string(row, "ground_truth"),
        "biased_option": _require_nonempty_string(row, "biased_option"),
        "biasing_text": biasing_text,
    }
    try:
        validated = _validate_frozen_row(native, path=source_path, line_number=line_number)
    except (ValueError, NotImplementedError) as exc:
        raise LegacyRowError("native_schema_invalid", str(exc)) from exc
    return validated, biasing_text_mode


def interleave_sources(per_source_rows: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Round-robin rows while preserving the order within every source."""

    rows: list[dict[str, Any]] = []
    width = len(per_source_rows)
    if width == 0:
        return rows
    max_length = max((len(source) for source in per_source_rows), default=0)
    for row_index in range(max_length):
        for source_index in range(width):
            source = per_source_rows[source_index]
            if row_index < len(source):
                rows.append(source[row_index])
    return rows


def _read_source(
    path: Path,
    *,
    per_source_limit: int,
    prompt_style: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        source_payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read legacy source {path}: {exc}") from exc
    try:
        source_text = source_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"legacy source {path} is not UTF-8: {exc}") from exc

    valid_rows: list[_ConvertedRow] = []
    invalid_reasons: Counter[str] = Counter()
    invalid_examples: list[dict[str, Any]] = []
    source_datasets: Counter[str] = Counter()
    valid_hasher = hashlib.sha256()
    invalid_hasher = hashlib.sha256()
    nonempty_line_count = 0
    blank_line_count = 0

    for line_number, raw_line in enumerate(source_text.splitlines(keepends=True), start=1):
        if not raw_line.strip():
            blank_line_count += 1
            continue
        nonempty_line_count += 1
        raw_bytes = raw_line.encode("utf-8")
        try:
            legacy_row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            error = LegacyRowError("invalid_json", f"invalid JSON: {exc.msg}")
        else:
            try:
                row, mode = convert_legacy_row(
                    legacy_row,
                    path=path,
                    line_number=line_number,
                    prompt_style=prompt_style,
                )
            except LegacyRowError as exc:
                error = exc
            else:
                valid_rows.append(_ConvertedRow(row=row, biasing_text_mode=mode))
                source_datasets[row["source_dataset"]] += 1
                valid_hasher.update(raw_bytes)
                continue

        invalid_reasons[error.code] += 1
        invalid_hasher.update(raw_bytes)
        if len(invalid_examples) < _MAX_INVALID_EXAMPLES:
            invalid_examples.append(
                {
                    "line_number": line_number,
                    "reason": error.code,
                    "detail": str(error),
                }
            )

    if len(valid_rows) < per_source_limit:
        raise ValueError(
            f"{path} contains only {len(valid_rows)}/{per_source_limit} valid legacy prompt pairs "
            f"({sum(invalid_reasons.values())} invalid)"
        )

    selected = valid_rows[:per_source_limit]
    selected_rows = [converted.row for converted in selected]
    selected_payload = b"".join(_jsonl_line(row) for row in selected_rows)
    valid_modes = Counter(converted.biasing_text_mode for converted in valid_rows)
    selected_modes = Counter(converted.biasing_text_mode for converted in selected)
    valid_question_ids = "\n".join(converted.row["question_id"] for converted in valid_rows).encode("utf-8")
    selected_question_ids = "\n".join(row["question_id"] for row in selected_rows).encode("utf-8")
    source_manifest = {
        "path": str(path.resolve()),
        "byte_count": len(source_payload),
        "content_sha256": hashlib.sha256(source_payload).hexdigest(),
        "nonempty_line_count": nonempty_line_count,
        "blank_line_count": blank_line_count,
        "valid_row_count": len(valid_rows),
        "invalid_row_count": sum(invalid_reasons.values()),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "invalid_examples": invalid_examples,
        "valid_source_rows_sha256": valid_hasher.hexdigest(),
        "invalid_source_rows_sha256": invalid_hasher.hexdigest(),
        "valid_question_ids_sha256": hashlib.sha256(valid_question_ids).hexdigest(),
        "source_datasets": dict(sorted(source_datasets.items())),
        "biasing_text_modes": dict(sorted(valid_modes.items())),
        "selected_row_count": len(selected_rows),
        "selected_rows_sha256": hashlib.sha256(selected_payload).hexdigest(),
        "selected_question_ids_sha256": hashlib.sha256(selected_question_ids).hexdigest(),
        "selected_biasing_text_modes": dict(sorted(selected_modes.items())),
    }
    return selected_rows, source_manifest


def _refuse_existing(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output(s): {existing}")


def import_legacy_pairs(
    inputs: Sequence[str | Path],
    *,
    output: str | Path,
    manifest_output: str | Path,
    per_source_limit: int = 1500,
    prompt_style: str = "encourage_cot",
) -> dict[str, Any]:
    """Import deterministic prefixes from legacy files and publish JSONL + manifest."""

    if not isinstance(per_source_limit, int) or isinstance(per_source_limit, bool) or per_source_limit < 1:
        raise ValueError("per_source_limit must be a positive integer")
    source_paths = [Path(path) for path in inputs]
    if not source_paths:
        raise ValueError("at least one legacy input is required")
    resolved_sources = [path.resolve() for path in source_paths]
    if len(resolved_sources) != len(set(resolved_sources)):
        raise ValueError("legacy inputs must name distinct files")
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise ValueError(f"legacy input is not a file: {missing}")

    output_path = Path(output)
    manifest_path = Path(manifest_output)
    if output_path.resolve() == manifest_path.resolve():
        raise ValueError("output and manifest_output must be different paths")
    targets = (output_path, manifest_path)
    _refuse_existing(targets)

    per_source_rows = []
    sources = []
    for path in source_paths:
        rows, source_manifest = _read_source(
            path,
            per_source_limit=per_source_limit,
            prompt_style=prompt_style,
        )
        per_source_rows.append(rows)
        sources.append(source_manifest)

    rows = interleave_sources(per_source_rows)
    payload = b"".join(_jsonl_line(row) for row in rows)
    manifest = {
        "schema_version": 1,
        "kind": "mcq_bias_legacy_pair_import",
        "written_at": datetime.now(UTC).isoformat(),
        "source_format": "cot-transparency distractor_argument_g4 paired JSONL",
        "field_mapping": {
            "original_question": "question",
            "original_question_hash": "question_id",
            "original_dataset": "source_dataset",
            "unbiased_question": "unbiased_messages",
            "biased_question": "biased_messages",
            "bias_name=distractor_argument": "bias_type=wrong_argument",
        },
        "selection": {
            "per_source_limit": per_source_limit,
            "within_source": "first valid rows in file order",
            "merge": "round_robin in CLI input order",
            "prompt_style": prompt_style,
            "shuffle": False,
        },
        "biasing_text": {
            "extraction": "single non-empty <argument>...</argument> block across biased_messages",
            "fallback_marker": FALLBACK_BIASING_TEXT,
        },
        "sources": sources,
        "row_count": len(rows),
        "output": {
            "path": str(output_path.resolve()),
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    # Repeat the overwrite check after source scanning so another process cannot
    # quietly create either named target during a long import.
    _refuse_existing(targets)
    write_atomic_bytes(output_path, payload)
    write_atomic_bytes(manifest_path, manifest_payload)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Import legacy distractor_argument_g4 prompt pairs as native mcq_bias JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "--inputs",
        dest="inputs",
        action="extend",
        nargs="+",
        type=Path,
        required=True,
        help="Legacy paired JSONL files, in desired round-robin order",
    )
    parser.add_argument("--per-source-limit", type=int, default=1500)
    parser.add_argument("--prompt-style", choices=("none", "encourage_cot"), default="encourage_cot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.per_source_limit < 1:
        parser.error("--per-source-limit must be >= 1")
    try:
        manifest = import_legacy_pairs(
            args.inputs,
            output=args.output,
            manifest_output=args.manifest_output,
            per_source_limit=args.per_source_limit,
            prompt_style=args.prompt_style,
        )
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    invalid = sum(source["invalid_row_count"] for source in manifest["sources"])
    print(
        f"Imported {manifest['row_count']} native mcq_bias rows from {len(args.inputs)} sources "
        f"({invalid} invalid legacy rows filtered) to {args.output}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FALLBACK_BIASING_TEXT",
    "LegacyRowError",
    "convert_legacy_row",
    "import_legacy_pairs",
    "interleave_sources",
    "main",
]
