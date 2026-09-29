"""Stage the two pinned Hugging Face Arrow train splits as audited JSONL.

This command is deliberately local-only.  It reads a supplied Arrow IPC file,
checks its pinned schema and row count, converts it into the three-field input
consumed by :mod:`experiments.act_expanded.selection`, and publishes both the
JSONL and its manifest under content-addressed names.  It never downloads data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "act_expanded_local_arrow_stage_v1"
SCHEMA_VERSION = 1

PINNED_SOURCES: dict[str, dict[str, Any]] = {
    "logiqa": {
        "repository": "lucasmccabe/logiqa",
        "revision": "fa9f9918fa81eca088805c1395d7f592f7755ae0",
        "split": "train",
        "row_count": 7376,
        "arrow_fields": ("context", "query", "options", "correct_option"),
    },
    "hellaswag": {
        "repository": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "split": "train",
        "row_count": 39905,
        "arrow_fields": (
            "ind",
            "activity_label",
            "ctx_a",
            "ctx_b",
            "ctx",
            "endings",
            "source_id",
            "split",
            "split_type",
            "label",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PublishedStage:
    data_path: Path
    manifest_path: Path
    content_sha256: str
    manifest_sha256: str
    status: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _regular_file(path: str | Path) -> Path:
    supplied = Path(path)
    resolved = supplied.resolve()
    if supplied.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"Arrow source must be a regular non-symlink file: {supplied}")
    return resolved


def _publish_immutable(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace differing immutable artifact: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep exceptional temporaries recoverable under ``_archive``.  On
    # success, atomic replace moves the temporary out of the archive and into
    # its content-addressed final name without deleting anything.
    archive = path.parent / "_archive"
    archive.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=archive, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return "written"


def _read_arrow(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - dependency error is actionable at runtime
        raise RuntimeError("pyarrow is required to stage the pinned local Arrow files") from exc

    with pa.memory_map(str(path), "r") as source:
        try:
            reader = pa.ipc.open_stream(source)
        except pa.ArrowInvalid:
            source.seek(0)
            reader = pa.ipc.open_file(source)
        table = reader.read_all()
    return list(table.column_names), table.to_pylist()


def _nonempty_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _options(value: object, *, location: str) -> list[str]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{location} must contain at least two options")
    return [_nonempty_string(item, location=f"{location}[{index}]") for index, item in enumerate(value)]


def _convert_row(dataset: str, row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    location = f"{dataset} Arrow row {index}"
    if dataset == "logiqa":
        context = _nonempty_string(row.get("context"), location=f"{location}.context")
        query = _nonempty_string(row.get("query"), location=f"{location}.query")
        options = _options(row.get("options"), location=f"{location}.options")
        answer = row.get("correct_option")
        question = context + "\n" + query
    elif dataset == "hellaswag":
        question = _nonempty_string(row.get("ctx"), location=f"{location}.ctx")
        options = _options(row.get("endings"), location=f"{location}.endings")
        label = row.get("label")
        if not isinstance(label, str) or not label.isdigit():
            raise ValueError(f"{location}.label must be a decimal answer index")
        answer = int(label)
        if row.get("split") != "train":
            raise ValueError(f"{location}.split must be 'train'")
    else:  # pragma: no cover - caller validates this first
        raise ValueError(f"unsupported dataset {dataset!r}")
    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(options):
        raise ValueError(f"{location} has an answer index outside its option range")
    return {"ground_truth_idx": answer, "options": options, "question": question}


def stage_pinned_arrow(*, dataset: str, arrow_source: str | Path, output_dir: str | Path) -> PublishedStage:
    """Convert one exact pinned local Arrow split into content-addressed JSONL."""

    if dataset not in PINNED_SOURCES:
        raise ValueError(f"unsupported dataset {dataset!r}; choose one of {tuple(PINNED_SOURCES)}")
    source_path = _regular_file(arrow_source)
    source_payload = source_path.read_bytes()
    fields, raw_rows = _read_arrow(source_path)
    spec = PINNED_SOURCES[dataset]
    if tuple(fields) != tuple(spec["arrow_fields"]):
        raise ValueError(
            f"{dataset} Arrow fields differ from the pinned schema: "
            f"expected {list(spec['arrow_fields'])!r}, got {fields!r}"
        )
    if len(raw_rows) != spec["row_count"]:
        raise ValueError(
            f"{dataset} Arrow row count differs from the pinned split: "
            f"expected {spec['row_count']}, got {len(raw_rows)}"
        )
    rows = [_convert_row(dataset, row, index=index) for index, row in enumerate(raw_rows, start=1)]
    payload = _canonical_jsonl(rows)
    digest = _sha256(payload)
    output = Path(output_dir).resolve()
    data_path = output / f"act-expanded-staged-{dataset}-{digest}.jsonl"
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": "act_expanded_local_pinned_arrow_stage",
        "source_dataset": dataset,
        "repository": spec["repository"],
        "revision": spec["revision"],
        "split": spec["split"],
        "arrow_source": {
            "filename": source_path.name,
            "content_sha256": _sha256(source_payload),
            "byte_count": len(source_payload),
            "row_count": len(raw_rows),
            "fields": fields,
        },
        "conversion": (
            "question=context+'\\n'+query; options=options; ground_truth_idx=correct_option"
            if dataset == "logiqa"
            else "question=ctx; options=endings; ground_truth_idx=int(label)"
        ),
        "output": {
            "filename": data_path.name,
            "content_sha256": digest,
            "byte_count": len(payload),
            "row_count": len(rows),
            "fields": ["ground_truth_idx", "options", "question"],
        },
        "assertions": {"local_only": True, "source_order_preserved": True, "no_network_or_model_call": True},
    }
    manifest_payload = _canonical_json(manifest)
    manifest_sha = _sha256(manifest_payload)
    manifest_path = output / f"act-expanded-staged-{dataset}-manifest-{manifest_sha}.json"
    data_status = _publish_immutable(data_path, payload)
    manifest_status = _publish_immutable(manifest_path, manifest_payload)
    status = "resumed" if data_status == manifest_status == "resumed" else "written"
    return PublishedStage(data_path, manifest_path, digest, manifest_sha, status)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(PINNED_SOURCES), required=True)
    parser.add_argument("--arrow-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = stage_pinned_arrow(
            dataset=args.dataset,
            arrow_source=args.arrow_source,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "data": str(result.data_path),
                "manifest": str(result.manifest_path),
                "sha256": result.content_sha256,
                "status": result.status,
                "verified": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["PINNED_SOURCES", "PublishedStage", "stage_pinned_arrow"]
