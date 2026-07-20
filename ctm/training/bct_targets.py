"""Generate shared BCT and control targets from paired prompts.

The source dataset owns only the prompt pair. CTM owns the model-derived
training target: it samples the frozen base model on the reference prompt once,
then attaches that same completion to the main and control prompts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ctm.artifacts import plain_file_identity, write_atomic_bytes
from ctm.backends.base import SamplerHandle
from ctm.backends.renderers import decode_response
from ctm.cli_safety import redact_secrets

BCT_TARGET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PairedPrompt:
    """One validated source row and the three prompt views used by BCT."""

    source_id: str
    source_messages: list[dict[str, str]]
    main_messages: list[dict[str, str]]
    control_messages: list[dict[str, str]]


def _validate_messages(value: Any, *, location: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty message list")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ValueError(f"{location}[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role or not isinstance(content, str):
            raise ValueError(f"{location}[{index}] must contain non-empty role and string content fields")
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "user":
        raise ValueError(f"{location} must end with a user message so CTM can sample an assistant target")
    return messages


def prepare_paired_prompts(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_messages_field: str,
    main_messages_field: str,
    control_messages_field: str,
) -> list[PairedPrompt]:
    """Validate every row before backend setup and select its prompt views."""

    if not rows:
        raise ValueError("BCT target preparation needs at least one source row")
    fields = (source_messages_field, main_messages_field, control_messages_field)
    if any(not isinstance(field, str) or not field for field in fields):
        raise ValueError("message field names must be non-empty strings")

    prepared: list[PairedPrompt] = []
    source_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index + 1} must be a JSON object")
        source_id_value = row.get("source_id", row.get("question_id", row.get("id", f"row-{index + 1}")))
        if not isinstance(source_id_value, (str, int)) or isinstance(source_id_value, bool):
            raise ValueError(f"row {index + 1} source_id/question_id/id must be a string or integer")
        source_id = str(source_id_value)
        if source_id in source_ids:
            raise ValueError(f"duplicate source id {source_id!r} at row {index + 1}")
        source_ids.add(source_id)

        for field in set(fields):
            if field not in row:
                raise ValueError(f"row {index + 1} is missing message field {field!r}")
        prepared.append(
            PairedPrompt(
                source_id=source_id,
                source_messages=_validate_messages(
                    row[source_messages_field], location=f"row {index + 1}.{source_messages_field}"
                ),
                main_messages=_validate_messages(
                    row[main_messages_field], location=f"row {index + 1}.{main_messages_field}"
                ),
                control_messages=_validate_messages(
                    row[control_messages_field], location=f"row {index + 1}.{control_messages_field}"
                ),
            )
        )
    return prepared


async def generate_bct_rows(
    prompts: Sequence[PairedPrompt],
    *,
    sampler: SamplerHandle,
    renderer: Any,
    tokenizer: Any,
    max_tokens: int = 32768,
    temperature: float = 0.0,
    max_concurrency: int = 32,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample one shared base completion per prompt and construct both datasets."""

    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1:
        raise ValueError("max_concurrency must be a positive integer")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature < 0:
        raise ValueError("temperature must be a non-negative number")

    semaphore = asyncio.Semaphore(max_concurrency)
    stop = renderer.get_stop_sequences()

    async def sample_one(index: int, item: PairedPrompt) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = renderer.build_generation_prompt(item.source_messages)
        async with semaphore:
            sequences = await sampler.sample(
                prompt,
                max_tokens=max_tokens,
                temperature=float(temperature),
                stop=stop,
                num_samples=1,
            )
        if len(sequences) != 1:
            raise RuntimeError(
                f"base sampler returned {len(sequences)} sequences for row {index + 1} "
                f"({item.source_id!r}); expected exactly one"
            )
        completion = decode_response(renderer, tokenizer, sequences[0].tokens)
        if not completion.strip():
            raise RuntimeError(f"base sampler returned an empty completion for row {index + 1} ({item.source_id!r})")
        assistant = {"role": "assistant", "content": completion}
        metadata = {"source_id": item.source_id}
        return (
            {"messages": [*item.main_messages, assistant], **metadata},
            {"messages": [*item.control_messages, assistant], **metadata},
        )

    generated = await asyncio.gather(*(sample_one(index, item) for index, item in enumerate(prompts)))
    return [pair[0] for pair in generated], [pair[1] for pair in generated]


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)


def write_bct_target_artifacts(
    *,
    main_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    main_output: str | Path,
    control_output: str | Path,
    manifest_output: str | Path,
    source_files: Sequence[str | Path],
    model: str,
    backend_name: str,
    source_messages_field: str,
    main_messages_field: str,
    control_messages_field: str,
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish matched BCT datasets and their shared provenance."""

    if len(main_rows) != len(control_rows):
        raise ValueError(f"main/control row count mismatch: {len(main_rows)} != {len(control_rows)}")
    main_path = Path(main_output)
    control_path = Path(control_output)
    manifest_path = Path(manifest_output)
    paths = [main_path, control_path, manifest_path]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("main_output, control_output, and manifest_output must be different paths")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing BCT target artifact(s): {existing}")

    main_payload = _jsonl_bytes(main_rows)
    control_payload = _jsonl_bytes(control_rows)
    manifest = redact_secrets(
        {
            "schema_version": BCT_TARGET_SCHEMA_VERSION,
            "kind": "ctm_bct_targets",
            "written_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "backend": backend_name,
            "source_files": [plain_file_identity(path) for path in source_files],
            "fields": {
                "source_messages": source_messages_field,
                "main_messages": main_messages_field,
                "control_messages": control_messages_field,
            },
            "generation": dict(generation_config),
            "row_count": len(main_rows),
            "outputs": {
                "main": {
                    "path": str(main_path.resolve()),
                    "content_sha256": hashlib.sha256(main_payload).hexdigest(),
                },
                "control": {
                    "path": str(control_path.resolve()),
                    "content_sha256": hashlib.sha256(control_payload).hexdigest(),
                },
            },
        }
    )

    # Outputs are constructed completely in memory before any path is published.
    # The manifest is written last, so its presence is the completion marker.
    write_atomic_bytes(main_path, main_payload)
    write_atomic_bytes(control_path, control_payload)
    write_atomic_bytes(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


__all__ = [
    "BCT_TARGET_SCHEMA_VERSION",
    "PairedPrompt",
    "generate_bct_rows",
    "prepare_paired_prompts",
    "write_bct_target_artifacts",
]
