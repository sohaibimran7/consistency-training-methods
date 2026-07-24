"""Generate shared BCT and control targets from paired prompts.

The source dataset owns only the prompt pair. CTM owns the model-derived
training target: it samples the frozen base model on the reference prompt once,
then attaches that same completion to the main and control prompts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ctm.artifacts import plain_file_identity, write_atomic_bytes
from ctm.backends.base import SamplerHandle
from ctm.backends.renderers import decode_response
from ctm.cli_safety import redact_secrets

BCT_TARGET_SCHEMA_VERSION = 1
BCT_PROGRESS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PairedPrompt:
    """One validated source row and the three prompt views used by BCT."""

    source_id: str
    source_messages: list[dict[str, str]]
    main_messages: list[dict[str, str]]
    control_messages: list[dict[str, str]]


def _validate_generated_pair(
    index: int,
    prompt: PairedPrompt,
    main: Any,
    control: Any,
    *,
    location: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for name, value, expected_prompt in (
        ("main", main, prompt.main_messages),
        ("control", control, prompt.control_messages),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"{location} {name} output must be an object")
        if value.get("source_id") != prompt.source_id:
            raise ValueError(f"{location} {name} output source_id does not match input row {index + 1}")
        messages = value.get("messages")
        if not isinstance(messages, list) or len(messages) != len(expected_prompt) + 1:
            raise ValueError(f"{location} {name} output must contain its prompt plus one assistant message")
        if messages[:-1] != expected_prompt:
            raise ValueError(f"{location} {name} output prompt does not match input row {index + 1}")
        assistant = messages[-1]
        if (
            not isinstance(assistant, dict)
            or assistant.get("role") != "assistant"
            or not isinstance(assistant.get("content"), str)
            or not assistant["content"].strip()
        ):
            raise ValueError(f"{location} {name} output must end with a non-empty assistant message")
        outputs.append(value)
    if outputs[0]["messages"][-1] != outputs[1]["messages"][-1]:
        raise ValueError(f"{location} main/control assistant targets differ")
    return outputs[0], outputs[1]


def build_bct_progress_identity(
    prompts: Sequence[PairedPrompt],
    *,
    source_files: Sequence[str | Path],
    model: str,
    backend: str,
    source_messages_field: str,
    main_messages_field: str,
    control_messages_field: str,
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the exact target-generation job accepted by a progress store."""

    prompt_payload = json.dumps(
        [
            {
                "source_id": prompt.source_id,
                "source_messages": prompt.source_messages,
                "main_messages": prompt.main_messages,
                "control_messages": prompt.control_messages,
            }
            for prompt in prompts
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return redact_secrets(
        {
            "model": model,
            "backend": backend,
            "source_files": [plain_file_identity(path) for path in source_files],
            "fields": {
                "source_messages": source_messages_field,
                "main_messages": main_messages_field,
                "control_messages": control_messages_field,
            },
            "generation": dict(generation_config),
            "prompt_count": len(prompts),
            "prompts_sha256": hashlib.sha256(prompt_payload).hexdigest(),
        }
    )


class BCTProgressStore:
    """Per-row atomic checkpoints for an otherwise immutable BCT target build."""

    MANIFEST_NAME = "progress-manifest.json"

    def __init__(self, directory: str | Path, identity: Mapping[str, Any]):
        self.directory = Path(directory)
        self.manifest_path = self.directory / self.MANIFEST_NAME
        self.manifest = {
            "schema_version": BCT_PROGRESS_SCHEMA_VERSION,
            "kind": "ctm_bct_target_progress",
            "identity": redact_secrets(dict(identity)),
        }

        if self.directory.exists() and not self.directory.is_dir():
            raise ValueError(f"BCT progress path is not a directory: {self.directory}")
        if self.directory.exists():
            if not self.manifest_path.is_file():
                raise ValueError(f"BCT progress directory has no manifest: {self.directory}")
            try:
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid BCT progress manifest {self.manifest_path}: {exc}") from exc
            if existing != self.manifest:
                raise ValueError(
                    f"BCT progress identity differs from the requested generation job: {self.manifest_path}"
                )
        else:
            self.directory.mkdir(parents=True)
            write_atomic_bytes(
                self.manifest_path,
                (json.dumps(self.manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )

    def _row_path(self, index: int) -> Path:
        return self.directory / f"row-{index:08d}.json"

    @staticmethod
    def _validate_record(value: Any, prompts: Sequence[PairedPrompt], *, path: Path) -> tuple[int, dict, dict]:
        if not isinstance(value, Mapping):
            raise ValueError(f"BCT progress row must be an object: {path}")
        index = value.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(prompts):
            raise ValueError(f"BCT progress row has invalid index: {path}")
        prompt = prompts[index]
        if value.get("source_id") != prompt.source_id:
            raise ValueError(f"BCT progress row source_id does not match input row {index + 1}: {path}")
        main, control = _validate_generated_pair(
            index,
            prompt,
            value.get("main"),
            value.get("control"),
            location=f"BCT progress row {path}",
        )
        return index, main, control

    def load(self, prompts: Sequence[PairedPrompt]) -> dict[int, tuple[dict, dict]]:
        """Load and validate every completed row in this progress directory."""

        completed: dict[int, tuple[dict, dict]] = {}
        for path in sorted(self.directory.glob("row-*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid BCT progress row {path}: {exc}") from exc
            index, main, control = self._validate_record(value, prompts, path=path)
            if path != self._row_path(index):
                raise ValueError(f"BCT progress row filename does not match its index: {path}")
            if index in completed:
                raise ValueError(f"duplicate BCT progress row index {index}: {path}")
            completed[index] = (main, control)
        return completed

    def record(self, index: int, prompt: PairedPrompt, main: dict, control: dict) -> None:
        """Atomically checkpoint one successful completion."""

        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"BCT progress row index must be a non-negative integer, got {index!r}")
        main, control = _validate_generated_pair(
            index,
            prompt,
            main,
            control,
            location=f"BCT progress row {index + 1}",
        )
        path = self._row_path(index)
        record = {
            "index": index,
            "source_id": prompt.source_id,
            "main": main,
            "control": control,
        }
        payload = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"refusing to replace conflicting BCT progress row: {path}")
            return
        write_atomic_bytes(path, payload)

    def archive(self) -> Path | None:
        """Move completed progress aside after immutable outputs are published."""

        if not self.directory.exists():
            return None
        archive_dir = self.directory.parent / "_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = archive_dir / f"{self.directory.name}.{timestamp}.complete"
        try:
            self.directory.replace(destination)
        except OSError as exc:
            warnings.warn(f"could not archive completed BCT progress {self.directory}: {exc}")
            return None
        return destination


def _validate_messages(value: Any, *, location: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location} must be a non-empty message list")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ValueError(f"{location}[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip() or not isinstance(content, str) or not content.strip():
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
    completed: Mapping[int, tuple[dict[str, Any], dict[str, Any]]] | None = None,
    on_completed: Callable[[int, PairedPrompt, dict[str, Any], dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample one shared base completion per prompt and construct both datasets.

    ``completed`` supplies validated rows restored from a progress store.
    ``on_completed`` is called after each new row succeeds, allowing callers to
    checkpoint work before another concurrent sample can fail.
    """

    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1:
        raise ValueError("max_concurrency must be a positive integer")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature < 0:
        raise ValueError("temperature must be a non-negative number")

    semaphore = asyncio.Semaphore(max_concurrency)
    stop = renderer.get_stop_sequences()

    restored = dict(completed or {})
    invalid_indices = sorted(
        (
            index
            for index in restored
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(prompts)
        ),
        key=repr,
    )
    if invalid_indices:
        raise ValueError(f"completed BCT rows contain invalid indices: {invalid_indices}")
    for index, (main, control) in restored.items():
        restored[index] = _validate_generated_pair(
            index,
            prompts[index],
            main,
            control,
            location=f"completed BCT row {index + 1}",
        )

    async def sample_one(index: int, item: PairedPrompt) -> tuple[int, dict[str, Any], dict[str, Any]]:
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
        main = {"messages": [*item.main_messages, assistant], **metadata}
        control = {"messages": [*item.control_messages, assistant], **metadata}
        main, control = _validate_generated_pair(
            index,
            item,
            main,
            control,
            location=f"generated BCT row {index + 1}",
        )
        if on_completed is not None:
            on_completed(index, item, main, control)
        return index, main, control

    generated = await asyncio.gather(
        *(sample_one(index, item) for index, item in enumerate(prompts) if index not in restored)
    )
    for index, main, control in generated:
        restored[index] = (main, control)
    if len(restored) != len(prompts):
        missing = sorted(set(range(len(prompts))) - set(restored))
        raise RuntimeError(f"BCT target generation finished with missing row indices: {missing}")
    return (
        [restored[index][0] for index in range(len(prompts))],
        [restored[index][1] for index in range(len(prompts))],
    )


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
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite completed BCT target artifact: {manifest_path}")

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

    def publish_dataset(path: Path, payload: bytes) -> None:
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(f"refusing to overwrite conflicting partial BCT target artifact: {path}")
            return
        write_atomic_bytes(path, payload)

    # Outputs are constructed completely in memory before any path is published.
    # The manifest is written last, so its presence is the completion marker. An
    # interrupted retry may reuse a byte-identical dataset already published
    # before the interruption; conflicting bytes still fail closed.
    publish_dataset(main_path, main_payload)
    publish_dataset(control_path, control_payload)
    write_atomic_bytes(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


__all__ = [
    "BCTProgressStore",
    "BCT_PROGRESS_SCHEMA_VERSION",
    "BCT_TARGET_SCHEMA_VERSION",
    "PairedPrompt",
    "build_bct_progress_identity",
    "generate_bct_rows",
    "prepare_paired_prompts",
    "write_bct_target_artifacts",
]
