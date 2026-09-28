"""Export the RMCT reproduction's pinned text-only multiple-choice HLE subset.

The source dataset stores answer choices inline in the question text. This
module performs the one schema conversion that ``mcq_bias`` deliberately does
not guess: filter ``answer_type == 'multipleChoice'``, exclude image-dependent
rows, split the inline choices, and write canonical local MCQ JSONL.

HLE is gated and its maintainers ask users not to redistribute the benchmark.
The generated JSONL is therefore an experiment artifact, not repository data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from string import ascii_uppercase
from typing import Any

from ctm.artifacts import write_atomic_bytes

DATASET_ID = "cais/hle"
DATASET_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
EXPECTED_TEXT_MC_COUNT = 513

_CHOICES_MARKER = "Answer Choices:"
_CHOICE_START = re.compile(r"(?m)^([A-Z])\.\s+")


def parse_multiple_choice_question(value: str) -> tuple[str, list[str]]:
    """Split one HLE inline multiple-choice question into stem and choices."""

    if not isinstance(value, str) or _CHOICES_MARKER not in value:
        raise ValueError("HLE multiple-choice question has no 'Answer Choices:' marker")
    stem, raw_choices = value.rsplit(_CHOICES_MARKER, 1)
    matches = list(_CHOICE_START.finditer(raw_choices.strip()))
    if len(matches) < 2:
        raise ValueError("HLE multiple-choice question has fewer than two labelled choices")
    labels = [match.group(1) for match in matches]
    expected = list(ascii_uppercase[: len(labels)])
    if labels != expected:
        raise ValueError(f"HLE choice labels must be consecutive from A; got {labels}")
    choices = []
    text = raw_choices.strip()
    matches = list(_CHOICE_START.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        choice = text[match.end() : end].strip()
        if not choice:
            raise ValueError(f"HLE choice {match.group(1)} is empty")
        choices.append(choice)
    if not stem.strip():
        raise ValueError("HLE question stem is empty")
    return stem.strip(), choices


def canonical_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return canonical local-MCQ rows for text-only multiple-choice inputs."""

    output = []
    for row in rows:
        if row.get("answer_type") != "multipleChoice" or row.get("image"):
            continue
        question, choices = parse_multiple_choice_question(row.get("question"))
        answer = row.get("answer")
        if not isinstance(answer, str) or len(answer.strip()) != 1:
            raise ValueError(f"HLE row {row.get('id')!r} has invalid multiple-choice answer {answer!r}")
        answer = answer.strip().upper()
        if answer not in ascii_uppercase[: len(choices)]:
            raise ValueError(f"HLE row {row.get('id')!r} answer {answer!r} is outside {len(choices)} choices")
        output.append(
            {
                "source_id": str(row.get("id", "")),
                "question": question,
                "options": choices,
                "answer": answer,
            }
        )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export pinned HLE text-only multiple-choice rows to canonical local MCQ JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TEXT_MC_COUNT)
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args(argv)

    if args.expected_count < 1:
        parser.error("--expected-count must be >= 1")
    if args.output.resolve() == args.manifest_output.resolve():
        parser.error("--output and --manifest-output must differ")
    existing = [str(path) for path in (args.output, args.manifest_output) if path.exists()]
    if existing:
        parser.error(f"refusing to overwrite existing output(s): {existing}")

    print("\nHLE export plan:")
    print(f"  source={DATASET_ID}@{DATASET_REVISION}")
    print("  filter=answer_type:multipleChoice and no image")
    print(f"  expected_count={args.expected_count}")
    print(f"  output={args.output}")
    print(f"  manifest={args.manifest_output}")
    print("  The generated benchmark data must not be committed or redistributed.")
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split="test", revision=DATASET_REVISION)
    rows = canonical_rows(dataset)
    if len(rows) != args.expected_count:
        raise ValueError(
            f"pinned HLE source produced {len(rows)} text-only multiple-choice rows; "
            f"expected exactly {args.expected_count}"
        )
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    manifest = {
        "schema_version": 1,
        "kind": "hle_text_multiple_choice_export",
        "written_at": datetime.now(UTC).isoformat(),
        "source": {"dataset": DATASET_ID, "revision": DATASET_REVISION, "split": "test"},
        "selection": {"answer_type": "multipleChoice", "image": False},
        "row_count": len(rows),
        "output": {
            "path": str(args.output.resolve()),
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    write_atomic_bytes(args.output, payload)
    write_atomic_bytes(
        args.manifest_output,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"Exported {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
