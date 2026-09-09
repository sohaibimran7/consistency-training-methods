"""Create or verify the remote-local immutable inputs for the ACT repair gate.

The Stage 1 IID manifest intentionally records absolute paths.  Copying a
manifest from a different machine therefore makes an otherwise valid frozen
split non-portable.  This small gate-local wrapper recreates the split manifest
from the one immutable 3,000-row source on the execution host, then creates or
verifies the two split-specific canonical consistency artifacts.

Every artifact is immutable.  A complete prior artifact is recomputed and
byte-verified before being resumed; a partial artifact set fails closed so it
cannot be mistaken for a completed experimental input.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ctm.artifacts import read_verified_artifact_manifest
from ctm_data.adapters.mcq_bias.consistency_pairs import (
    ARTIFACT_SCHEMA,
    SCHEMA_VERSION as CANONICAL_SCHEMA_VERSION,
    TRANSFORM_VERSION,
    materialize_consistency_pairs,
)
from experiments.stage1_iid_diagnostic import prepare as iid_prepare

_SPLITS = {
    "train_eval": {
        "source_filename": iid_prepare.TRAIN_EVAL_FILENAME,
        "canonical_filename": "canonical-train-eval-n200.jsonl",
        "canonical_manifest_filename": "canonical-train-eval-n200.manifest.json",
        "counts": iid_prepare.TRAIN_EVAL_COUNTS,
    },
    "heldout_in_domain": {
        "source_filename": iid_prepare.HELDOUT_FILENAME,
        "canonical_filename": "canonical-heldout-in-domain-n200.jsonl",
        "canonical_manifest_filename": "canonical-heldout-in-domain-n200.manifest.json",
        "counts": iid_prepare.HELDOUT_COUNTS,
    },
}


def _split_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / iid_prepare.TRAIN_EVAL_FILENAME,
        output_dir / iid_prepare.HELDOUT_FILENAME,
        output_dir / iid_prepare.DEFAULT_MANIFEST_FILENAME,
    )


def _require_remote_local_manifest(document: dict[str, Any], *, source: Path, output_dir: Path) -> None:
    """Reject a copied manifest whose absolute artifact paths name another host."""

    manifest_source = Path(str(document["source"]["path"])).resolve()
    if manifest_source != source.resolve():
        raise ValueError(
            "IID manifest source.path is not this host's staged frozen source: "
            f"{manifest_source} != {source.resolve()}"
        )
    for split, spec in _SPLITS.items():
        actual = Path(str(document["splits"][split]["path"])).resolve()
        expected = (output_dir / spec["source_filename"]).resolve()
        if actual != expected:
            raise ValueError(
                f"IID manifest {split!r} path is not this gate's remote-local split: "
                f"{actual} != {expected}"
            )


def _ensure_iid_splits(source: Path, output_dir: Path) -> tuple[str, dict[str, Any]]:
    """Create all three split artifacts once, or verify all three on resume."""

    train, heldout, manifest = _split_paths(output_dir)
    state = [path.exists() for path in (train, heldout, manifest)]
    if any(state) and not all(state):
        present = [str(path) for path, exists in zip((train, heldout, manifest), state, strict=True) if exists]
        raise FileExistsError(
            "incomplete remote-local IID split artifact set; archive the partial "
            f"files before retrying: {present}"
        )
    if all(state):
        document = iid_prepare.validate_manifest(manifest, verify_source=True)
        _require_remote_local_manifest(document, source=source, output_dir=output_dir)
        return "resumed", document

    document = iid_prepare.prepare_iid_diagnostic(source, output_dir, manifest)
    _require_remote_local_manifest(document, source=source, output_dir=output_dir)
    return "written", document


def _last_user_content(row: dict[str, Any], *, location: str) -> str:
    messages = row.get("unbiased_messages")
    if not isinstance(messages, list):
        raise ValueError(f"{location}: unbiased_messages is not a list")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content:
                return content
    raise ValueError(f"{location}: unbiased_messages has no non-empty user message")


def _verify_canonical_rows(path: Path, *, expected_counts: dict[str, int]) -> None:
    """Check split population and the exact suffix invariant without model imports."""

    counts: Counter[str] = Counter()
    ids: set[str] = set()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid canonical JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: canonical row is not an object")
            source_dataset = row.get("source_dataset")
            question_id = row.get("question_id")
            if not isinstance(source_dataset, str) or not isinstance(question_id, str) or not question_id:
                raise ValueError(f"{path}:{line_number}: canonical row lacks source_dataset/question_id")
            if question_id in ids:
                raise ValueError(f"{path}:{line_number}: duplicate canonical question_id {question_id!r}")
            ids.add(question_id)
            counts[source_dataset] += 1
            if row.get("consistency_pair_transform") != TRANSFORM_VERSION:
                raise ValueError(f"{path}:{line_number}: wrong canonical transform marker")
            clean = _last_user_content(row, location=f"{path}:{line_number}")
            biased_messages = row.get("biased_messages")
            if not isinstance(biased_messages, list):
                raise ValueError(f"{path}:{line_number}: biased_messages is not a list")
            try:
                biased = next(
                    message["content"]
                    for message in reversed(biased_messages)
                    if isinstance(message, dict) and message.get("role") == "user"
                )
            except StopIteration as exc:
                raise ValueError(f"{path}:{line_number}: biased_messages has no user message") from exc
            if not isinstance(biased, str) or not biased.endswith(clean):
                raise ValueError(f"{path}:{line_number}: canonical biased prompt lacks the exact clean suffix")
            rows += 1
    if counts != Counter(expected_counts) or rows != sum(expected_counts.values()):
        raise ValueError(
            f"{path}: canonical split population must be {expected_counts} "
            f"({sum(expected_counts.values())} rows), got {dict(counts)} ({rows} rows)"
        )


def _ensure_canonical_split(output_dir: Path, *, split: str) -> str:
    spec = _SPLITS[split]
    source = output_dir / spec["source_filename"]
    output = output_dir / spec["canonical_filename"]
    manifest = output_dir / spec["canonical_manifest_filename"]
    state = [path.exists() for path in (output, manifest)]
    if any(state) and not all(state):
        present = [str(path) for path, exists in zip((output, manifest), state, strict=True) if exists]
        raise FileExistsError(
            f"incomplete canonical {split!r} artifact set; archive the partial files before retrying: {present}"
        )

    status = materialize_consistency_pairs(source, output, manifest)
    document = read_verified_artifact_manifest(
        output,
        manifest_path=manifest,
        expected_schema=ARTIFACT_SCHEMA,
        expected_schema_version=CANONICAL_SCHEMA_VERSION,
    )
    if document["row_count"] != sum(spec["counts"].values()):
        raise ValueError(f"{manifest}: unexpected canonical row count")
    provenance = document["provenance"]
    if provenance.get("transform", {}).get("name") != TRANSFORM_VERSION:
        raise ValueError(f"{manifest}: unexpected canonical transform")
    source_path = Path(str(provenance.get("source", {}).get("path", ""))).resolve()
    if source_path != source.resolve():
        raise ValueError(f"{manifest}: canonical source path is not this gate's {split!r} split")
    _verify_canonical_rows(output, expected_counts=dict(spec["counts"]))
    return status


def prepare_act_repair_gate(source: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create or verify every gate-local data artifact without overwriting one."""

    source_path = Path(source).resolve()
    root = Path(output_dir).resolve()
    split_status, document = _ensure_iid_splits(source_path, root)
    statuses = {
        split: _ensure_canonical_split(root, split=split)
        for split in _SPLITS
    }
    return {
        "iid_splits": split_status,
        "canonical_splits": statuses,
        "manifest": str((root / iid_prepare.DEFAULT_MANIFEST_FILENAME).resolve()),
        "source": str(source_path),
        "split_rows": {
            split: document["splits"][split]["row_count"] for split in _SPLITS
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = prepare_act_repair_gate(args.source, args.output_dir)
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        "ACT gate data "
        f"IID={result['iid_splits']}; "
        "canonical="
        + ", ".join(f"{split}:{status}" for split, status in result["canonical_splits"].items())
        + f"; manifest={result['manifest']}"
    )


if __name__ == "__main__":
    main()


__all__ = ["prepare_act_repair_gate"]
