#!/usr/bin/env python3
"""Generate pinned EvalAwareBench Figure 6 responses through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ctm_data.adapters.eval_awareness.figure6_materialize import load_figure6_artifact
from ctm_data.adapters.eval_awareness.figure6_spec import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_SPECS,
    ModelSpec,
    PromptSpec,
    UPSTREAM_CODE_REVISION,
    get_model_spec,
    load_verified_model_prompt,
)

GENERATION_SCHEMA = "ctm.eval_awareness.figure6_generation"
GENERATION_SCHEMA_VERSION = 1
DEFAULT_REPLICATES = 3
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_MAX_RETRIES = 3

_TAGGED_REASONING_RE = re.compile(
    r"<(?P<tag>think|SCRATCHPAD_REASONING)>\s*(?P<body>.*?)\s*</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[^\s,;]+")
_API_KEY_RE = re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+")
_MISSING = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc_timestamp(value: Any, *, field: str, row_number: int) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"generation log row {row_number} {field} must be an ISO 8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"generation log row {row_number} {field} must be an ISO 8601 UTC timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):  # pragma: no cover - Z construction guarantees this
        raise ValueError(f"generation log row {row_number} {field} must be UTC")
    return parsed


def make_generation_key(model_key: str, condition_id: str, replicate: int) -> str:
    """Build the stable logical key used for resume and deduplication."""

    if model_key not in MODEL_SPECS:
        get_model_spec(model_key)  # raises the canonical useful error
    if not isinstance(condition_id, str) or not condition_id:
        raise ValueError("condition_id must be a non-empty string")
    if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate < 1:
        raise ValueError("replicate must be an integer >= 1")
    return f"{model_key}|{condition_id}|{replicate}"


generation_key = make_generation_key


def extract_reasoning(response: str, reasoning_content: str | None = None) -> dict[str, Any]:
    """Separate an answer from native or tagged reasoning while preserving raw text."""

    if not isinstance(response, str):
        raise TypeError("response must be a string")
    if reasoning_content is not None:
        if not isinstance(reasoning_content, str):
            reasoning_content = str(reasoning_content)
        return {
            "response": response,
            "reasoning": reasoning_content,
            "answer": response.strip(),
            "trace_present": bool(reasoning_content.strip()),
            "trace_source": "reasoning_content",
        }

    matches = list(_TAGGED_REASONING_RE.finditer(response))
    if not matches:
        return {
            "response": response,
            "reasoning": "",
            "answer": response.strip(),
            "trace_present": False,
            "trace_source": "none",
        }
    traces = [match.group("body") for match in matches]
    tags = {match.group("tag").lower() for match in matches}
    if tags == {"think"}:
        source = "think_tags"
    elif tags == {"scratchpad_reasoning"}:
        source = "scratchpad_reasoning_tags"
    else:
        source = "mixed_reasoning_tags"
    answer = _TAGGED_REASONING_RE.sub("", response).strip()
    reasoning = "\n\n".join(trace for trace in traces if trace)
    return {
        "response": response,
        "reasoning": reasoning,
        "answer": answer,
        "trace_present": bool(reasoning.strip()),
        "trace_source": source,
    }


extract_reasoning_and_answer = extract_reasoning


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    result = getattr(value, name, _MISSING)
    if result is not _MISSING:
        return result
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, Mapping):
        return model_extra.get(name, default)
    return default


def _message_content(message: Any) -> str:
    content = _field(message, "content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts: list[str] = []
        for part in content:
            text = _field(part, "text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)


def _parse_completion(completion: Any) -> dict[str, Any]:
    choices = _field(completion, "choices")
    if not isinstance(choices, Sequence) or not choices:
        raise ValueError("endpoint response has no completion choice")
    message = _field(choices[0], "message")
    if message is None:
        raise ValueError("endpoint response choice has no message")
    response = _message_content(message)
    reasoning = _field(message, "reasoning", _MISSING)
    reasoning_content = _field(message, "reasoning_content", _MISSING)
    native_reasoning = reasoning
    native_trace_source = "reasoning"
    if native_reasoning is _MISSING or native_reasoning is None:
        native_reasoning = reasoning_content
        native_trace_source = "reasoning_content"
    elif reasoning_content is not _MISSING and reasoning_content is not None and reasoning_content != native_reasoning:
        raise ValueError("endpoint response has conflicting reasoning and reasoning_content fields")
    extracted = extract_reasoning(
        response,
        None if native_reasoning is _MISSING or native_reasoning is None else native_reasoning,
    )
    if extracted["trace_present"] and native_trace_source == "reasoning":
        extracted["trace_source"] = native_trace_source
    usage = _field(completion, "usage")
    extracted.update(
        {
            "prompt_tokens": _field(usage, "prompt_tokens"),
            "completion_tokens": _field(usage, "completion_tokens"),
            "total_tokens": _field(usage, "total_tokens"),
            "response_model": _field(completion, "model"),
            "response_id": _field(completion, "id"),
            "finish_reason": _field(choices[0], "finish_reason"),
        }
    )
    return extracted


def _safe_error(error: BaseException, secrets: Sequence[str | None] = ()) -> str:
    text = f"{type(error).__name__}: {error}"
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    return _API_KEY_RE.sub(r"\1[REDACTED]", text)


def _record_id(logical_key: str, resume_attempt: int) -> str:
    digest = hashlib.sha256(f"{logical_key}|resume={resume_attempt}".encode()).hexdigest()[:40]
    return f"f6gen_{digest}"


def generation_provenance_path(output_path: str | Path) -> Path:
    """Return the immutable run-provenance sidecar for an append-only log."""

    target = Path(output_path)
    return target.with_suffix(target.suffix + ".provenance.json")


def selection_provenance_path(output_path: str | Path) -> Path:
    """Return the append-only pilot/full selection history for a generation log."""

    target = Path(output_path)
    return target.with_suffix(target.suffix + ".selections.jsonl")


def _condition_ids_sha256(condition_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(condition_ids).encode()).hexdigest()


def _full_generation_provenance(
    *,
    model: ModelSpec,
    prompt: PromptSpec,
    artifact_manifest: Mapping[str, Any],
    temperature: float,
    max_tokens: int,
    replicates: int,
) -> dict[str, Any]:
    return {
        "provenance_schema": "ctm.eval_awareness.figure6_generation_run",
        "schema_version": GENERATION_SCHEMA_VERSION,
        "artifact_schema": artifact_manifest["artifact_schema"],
        "artifact_schema_version": artifact_manifest["schema_version"],
        "artifact_sha256": artifact_manifest["content_sha256"],
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "model_id": model.model_id,
        "model_key": model.key,
        "model_revision": model.revision,
        "prompt_key": prompt.key,
        "prompt_revision": UPSTREAM_CODE_REVISION,
        "prompt_sha256": prompt.sha256,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "replicates": replicates,
    }


def _compact_generation_provenance(full_provenance: Mapping[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(dict(full_provenance), sort_keys=True, separators=(",", ":")).encode()
    compact = dict(full_provenance)
    compact["provenance_sha256"] = hashlib.sha256(canonical).hexdigest()
    return compact


def _selection_provenance(
    *,
    artifact_manifest: Mapping[str, Any],
    limit_conditions: int,
    selected_condition_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "provenance_schema": "ctm.eval_awareness.figure6_generation_selection",
        "schema_version": GENERATION_SCHEMA_VERSION,
        "artifact_sha256": artifact_manifest["content_sha256"],
        "limit_conditions": limit_conditions,
        "selected_condition_count": len(selected_condition_ids),
        "selected_condition_ids": list(selected_condition_ids),
        "selected_condition_ids_sha256": _condition_ids_sha256(selected_condition_ids),
        "selection_rule": "first N rows in the verified deterministic artifact order; 0 means all rows",
    }


def _ensure_generation_provenance(
    output_path: Path,
    provenance: Mapping[str, Any],
    *,
    write_if_missing: bool,
) -> Path:
    sidecar = generation_provenance_path(output_path)
    if sidecar.exists():
        try:
            existing = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid generation provenance sidecar {sidecar}: {exc.msg}") from exc
        if existing != dict(provenance):
            raise ValueError(f"generation provenance mismatch for existing output: {sidecar}")
        return sidecar
    if output_path.exists() and output_path.stat().st_size:
        raise ValueError(f"existing generation output has no provenance sidecar: {sidecar}")
    if write_if_missing:
        payload = (json.dumps(dict(provenance), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sidecar.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:  # concurrent writers must not silently mix runs
            raise FileExistsError(f"refusing to overwrite generation provenance: {sidecar}") from exc
    return sidecar


def _ensure_selection_provenance(
    output_path: Path,
    selection: Mapping[str, Any],
    *,
    all_condition_ids: Sequence[str],
    write_if_missing: bool,
) -> Path:
    sidecar = selection_provenance_path(output_path)
    existing = read_generation_records(sidecar)
    current_ids = selection["selected_condition_ids"]
    for index, prior in enumerate(existing, start=1):
        if prior.get("provenance_schema") != "ctm.eval_awareness.figure6_generation_selection":
            raise ValueError(f"selection provenance row {index} has an invalid schema")
        if prior.get("schema_version") != GENERATION_SCHEMA_VERSION:
            raise ValueError(f"selection provenance row {index} has an invalid schema_version")
        if prior.get("artifact_sha256") != selection["artifact_sha256"]:
            raise ValueError(f"selection provenance row {index} has a different artifact")
        prior_ids = prior.get("selected_condition_ids")
        if not isinstance(prior_ids, list) or any(not isinstance(value, str) for value in prior_ids):
            raise ValueError(f"selection provenance row {index} has invalid selected_condition_ids")
        if prior.get("selected_condition_count") != len(prior_ids):
            raise ValueError(f"selection provenance row {index} has an invalid selected_condition_count")
        if prior.get("selected_condition_ids_sha256") != _condition_ids_sha256(prior_ids):
            raise ValueError(f"selection provenance row {index} has an invalid selected-condition digest")
        if list(all_condition_ids[: len(prior_ids)]) != prior_ids:
            raise ValueError(f"selection provenance row {index} is not a deterministic artifact prefix")
        if list(current_ids[: len(prior_ids)]) != prior_ids:
            raise ValueError("current condition selection cannot shrink or diverge from prior pilot provenance")
    if output_path.exists() and output_path.stat().st_size and not existing:
        raise ValueError(f"existing generation output has no selection provenance sidecar: {sidecar}")
    if write_if_missing and dict(selection) not in existing:
        _append_durable(sidecar, selection)
    return sidecar


def _record_provenance(
    *,
    row: Mapping[str, Any],
    model: ModelSpec,
    prompt: PromptSpec,
    artifact_manifest: Mapping[str, Any],
    replicate: int,
    temperature: float,
    max_tokens: int,
    generation_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    logical_key = make_generation_key(model.key, row["condition_id"], replicate)
    return {
        "schema": GENERATION_SCHEMA,
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_key": logical_key,
        "model_key": model.key,
        "model": model.model_id,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "model_display": model.display_name,
        "model_display_name": model.display_name,
        "comparison_family": model.comparison_family,
        "comparison_stage": model.comparison_stage,
        "condition_id": row["condition_id"],
        "pair_id": row["pair_id"],
        "task_id": row["task_id"],
        "task_name": row["task_name"],
        "valence": row["valence"],
        "condition": row["condition"],
        "config_name": row["condition"],
        "replicate": replicate,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_key": prompt.key,
        "prompt_revision": UPSTREAM_CODE_REVISION,
        "prompt_sha256": prompt.sha256,
        "system_prompt_sha256": prompt.sha256,
        "artifact_schema": artifact_manifest["artifact_schema"],
        "artifact_schema_version": artifact_manifest["schema_version"],
        "artifact_sha256": artifact_manifest["content_sha256"],
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_config": row["source_config"],
        "prompt": row["prompt"],
        "generation_provenance": dict(generation_provenance),
        "system_prompt_provenance": {
            "prompt_key": prompt.key,
            "prompt_revision": UPSTREAM_CODE_REVISION,
            "prompt_sha256": prompt.sha256,
        },
    }


def read_generation_records(path: str | Path) -> list[dict[str, Any]]:
    """Read an append-only generation log with line-numbered JSON errors."""

    target = Path(path)
    if not target.exists():
        return []
    if not target.is_file():
        raise ValueError(f"generation output is not a file: {target}")
    records: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{target}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{target}:{line_number}: generation record must be an object")
            records.append(record)
    return records


def validate_generation_records(
    records: Sequence[Mapping[str, Any]],
    *,
    artifact_rows: Sequence[Mapping[str, Any]],
    artifact_manifest: Mapping[str, Any],
    model: ModelSpec,
    prompt: PromptSpec,
    temperature: float,
    max_tokens: int,
    generation_provenance: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Validate log provenance and return histories keyed by logical generation key."""

    rows_by_condition = {row["condition_id"]: row for row in artifact_rows}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_record_ids: set[str] = set()
    expected_fixed = {
        "schema": GENERATION_SCHEMA,
        "schema_version": GENERATION_SCHEMA_VERSION,
        "model_key": model.key,
        "model": model.model_id,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "model_display": model.display_name,
        "model_display_name": model.display_name,
        "prompt_key": prompt.key,
        "prompt_revision": UPSTREAM_CODE_REVISION,
        "prompt_sha256": prompt.sha256,
        "system_prompt_sha256": prompt.sha256,
        "artifact_schema": artifact_manifest["artifact_schema"],
        "artifact_schema_version": artifact_manifest["schema_version"],
        "artifact_sha256": artifact_manifest["content_sha256"],
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "generation_provenance": dict(generation_provenance),
        "system_prompt_provenance": {
            "prompt_key": prompt.key,
            "prompt_revision": UPSTREAM_CODE_REVISION,
            "prompt_sha256": prompt.sha256,
        },
    }
    for line_number, record_like in enumerate(records, start=1):
        record = dict(record_like)
        for field, expected in expected_fixed.items():
            if record.get(field) != expected:
                raise ValueError(
                    f"generation log row {line_number} has incompatible {field}: "
                    f"{record.get(field)!r} != {expected!r}"
                )
        condition_id = record.get("condition_id")
        if condition_id not in rows_by_condition:
            raise ValueError(f"generation log row {line_number} names unknown condition_id {condition_id!r}")
        source_row = rows_by_condition[condition_id]
        for field in ("pair_id", "task_id", "task_name", "valence", "condition"):
            if record.get(field) != source_row[field]:
                raise ValueError(f"generation log row {line_number} {field} does not match the prompt artifact")
        if record.get("config_name") != source_row["condition"]:
            raise ValueError(f"generation log row {line_number} config_name does not match the prompt artifact")
        if record.get("dataset_config") != source_row["source_config"]:
            raise ValueError(f"generation log row {line_number} dataset_config does not match the prompt artifact")
        if record.get("prompt") != source_row["prompt"]:
            raise ValueError(f"generation log row {line_number} prompt does not match the prompt artifact")
        replicate = record.get("replicate")
        if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate < 1:
            raise ValueError(f"generation log row {line_number} has invalid replicate")
        logical_key = make_generation_key(model.key, condition_id, replicate)
        if record.get("generation_key") != logical_key:
            raise ValueError(f"generation log row {line_number} has an invalid generation_key")
        resume_attempt = record.get("resume_attempt")
        if not isinstance(resume_attempt, int) or isinstance(resume_attempt, bool) or resume_attempt < 1:
            raise ValueError(f"generation log row {line_number} has invalid resume_attempt")
        record_id = record.get("record_id")
        if record_id != _record_id(logical_key, resume_attempt):
            raise ValueError(f"generation log row {line_number} has an invalid record_id")
        if record_id in seen_record_ids:
            raise ValueError(f"duplicate generation record_id at row {line_number}: {record_id}")
        seen_record_ids.add(record_id)
        status = record.get("status")
        if status not in {"success", "error"}:
            raise ValueError(f"generation log row {line_number} has invalid status {status!r}")
        attempts = record.get("attempts")
        if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
            raise ValueError(f"generation log row {line_number} has invalid attempts")
        started_at = _parse_utc_timestamp(record.get("started_at"), field="started_at", row_number=line_number)
        completed_at = _parse_utc_timestamp(record.get("completed_at"), field="completed_at", row_number=line_number)
        if completed_at < started_at:
            raise ValueError(f"generation log row {line_number} completed_at precedes started_at")
        elapsed_seconds = record.get("elapsed_seconds")
        if (
            not isinstance(elapsed_seconds, (int, float))
            or isinstance(elapsed_seconds, bool)
            or not math.isfinite(elapsed_seconds)
            or elapsed_seconds < 0
        ):
            raise ValueError(f"generation log row {line_number} has invalid elapsed_seconds")
        if status == "success":
            if not isinstance(record.get("response"), str) or not isinstance(record.get("answer"), str):
                raise ValueError(f"successful generation log row {line_number} lacks response text")
            if not isinstance(record.get("reasoning"), str) or not record["reasoning"].strip():
                raise ValueError(f"successful generation log row {line_number} lacks reasoning text")
            if record.get("trace_present") is not True:
                raise ValueError(f"successful generation log row {line_number} has no reasoning trace")
            if record.get("trace_source") in {None, "", "none"}:
                raise ValueError(f"successful generation log row {line_number} has no reasoning trace source")
            if record.get("error") is not None:
                raise ValueError(f"successful generation log row {line_number} has a non-null error")
        elif not isinstance(record.get("error"), str) or not record["error"]:
            raise ValueError(f"failed generation log row {line_number} lacks an error")
        if not isinstance(record.get("trace_present"), bool):
            raise ValueError(f"generation log row {line_number} has invalid trace_present")
        if not isinstance(record.get("trace_source"), str) or not record["trace_source"]:
            raise ValueError(f"generation log row {line_number} has invalid trace_source")
        for token_field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            token_count = record.get(token_field)
            if token_count is not None and (
                not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 0
            ):
                raise ValueError(f"generation log row {line_number} has invalid {token_field}")
        histories[logical_key].append(record)

    for logical_key, history in histories.items():
        attempts = [record["resume_attempt"] for record in history]
        if attempts != list(range(1, len(history) + 1)):
            raise ValueError(f"generation key {logical_key!r} has duplicate or non-contiguous resume attempts")
        success_positions = [index for index, record in enumerate(history) if record["status"] == "success"]
        if len(success_positions) > 1:
            raise ValueError(f"generation key {logical_key!r} has duplicate successful records")
        if success_positions and success_positions[0] != len(history) - 1:
            raise ValueError(f"generation key {logical_key!r} has records appended after success")
    return dict(histories)


async def _endpoint_model_check(client: Any, model: ModelSpec) -> dict[str, Any]:
    models_resource = getattr(client, "models", None)
    list_method = getattr(models_resource, "list", None)
    if not callable(list_method):
        return {"status": "unsupported", "served_model_ids": []}
    listing = await list_method()
    data = _field(listing, "data", [])
    served_ids = sorted(
        {served_id for entry in data or [] if isinstance((served_id := _field(entry, "id")), str) and served_id}
    )
    accepted_ids = {model.model_id, model.key}
    if not accepted_ids.intersection(served_ids):
        raise ValueError(
            f"endpoint model identity mismatch: expected {model.model_id!r} (or served alias {model.key!r}), "
            f"found {served_ids}"
        )
    return {"status": "verified", "served_model_ids": served_ids}


def _create_openai_client(*, base_url: str, api_key: str) -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - dependency is intentionally lazy
        raise RuntimeError("generation requires the optional openai package") from exc
    return AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)


async def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


def _append_durable(path: Path, record: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


async def _generate_one(
    *,
    client: Any,
    row: Mapping[str, Any],
    model: ModelSpec,
    prompt: PromptSpec,
    system_prompt: str,
    artifact_manifest: Mapping[str, Any],
    replicate: int,
    resume_attempt: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_base_seconds: float,
    api_key: str | None,
    generation_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = _record_provenance(
        row=row,
        model=model,
        prompt=prompt,
        artifact_manifest=artifact_manifest,
        replicate=replicate,
        temperature=temperature,
        max_tokens=max_tokens,
        generation_provenance=generation_provenance,
    )
    logical_key = provenance["generation_key"]
    last_error: str | None = None
    started_at = _utc_now()
    started_monotonic = time.monotonic()

    def timing_fields() -> dict[str, Any]:
        # Elapsed time intentionally spans all endpoint attempts and retry backoff.
        return {
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_seconds": max(0.0, time.monotonic() - started_monotonic),
        }

    for attempt in range(1, max_retries + 1):
        try:
            completion = await client.chat.completions.create(
                model=model.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["prompt"]},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed = _parse_completion(completion)
            if not parsed["trace_present"]:
                return {
                    **provenance,
                    "record_id": _record_id(logical_key, resume_attempt),
                    "resume_attempt": resume_attempt,
                    "status": "error",
                    "error": "missing reasoning trace: response had neither reasoning_content nor supported tags",
                    "attempts": attempt,
                    **timing_fields(),
                    **parsed,
                }
            return {
                **provenance,
                "record_id": _record_id(logical_key, resume_attempt),
                "resume_attempt": resume_attempt,
                "status": "success",
                "error": None,
                "attempts": attempt,
                **timing_fields(),
                **parsed,
            }
        except Exception as exc:  # every terminal request outcome is durably recorded
            last_error = _safe_error(exc, [api_key])
            if attempt < max_retries and retry_base_seconds:
                await asyncio.sleep(retry_base_seconds * (2 ** (attempt - 1)))
    return {
        **provenance,
        "record_id": _record_id(logical_key, resume_attempt),
        "resume_attempt": resume_attempt,
        "status": "error",
        "error": last_error or "unknown generation error",
        "attempts": max_retries,
        **timing_fields(),
        "response": None,
        "reasoning": "",
        "answer": "",
        "trace_present": False,
        "trace_source": "none",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "response_model": None,
        "response_id": None,
        "finish_reason": None,
    }


async def generate_figure6(
    artifact_path: str | Path,
    output_path: str | Path,
    *,
    model_key: str,
    prompt_path: str | Path,
    client: Any | None = None,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    replicates: int = DEFAULT_REPLICATES,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = 1.0,
    check_endpoint_model: bool = True,
    dry_run: bool = False,
    limit_conditions: int = 0,
) -> dict[str, Any]:
    """Resume an append-only generation run and return a non-secret summary.

    Existing failures form an audit history and are retried with a monotonically
    increasing ``resume_attempt``.  Existing successes are never called again.
    """

    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
        raise ValueError("replicates must be an integer >= 1")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be an integer >= 1")
    if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1:
        raise ValueError("max_concurrency must be an integer >= 1")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 1:
        raise ValueError("max_retries must be an integer >= 1")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature < 0:
        raise ValueError("temperature must be a non-negative number")
    if retry_base_seconds < 0:
        raise ValueError("retry_base_seconds must be >= 0")
    if not isinstance(limit_conditions, int) or isinstance(limit_conditions, bool) or limit_conditions < 0:
        raise ValueError("limit_conditions must be an integer >= 0")
    if client is None and base_url.rstrip("/").split("/")[-1] != "v1":
        raise ValueError("OpenAI-compatible base_url must end in /v1")

    model = get_model_spec(model_key)
    system_prompt, prompt = load_verified_model_prompt(model_key, prompt_path)
    artifact_rows, artifact_manifest = load_figure6_artifact(artifact_path)
    if limit_conditions > len(artifact_rows):
        raise ValueError(
            f"limit_conditions={limit_conditions} exceeds the verified artifact row count {len(artifact_rows)}"
        )
    selected_rows = artifact_rows[:limit_conditions] if limit_conditions else artifact_rows
    selected_condition_ids = [row["condition_id"] for row in selected_rows]
    base_provenance = _full_generation_provenance(
        model=model,
        prompt=prompt,
        artifact_manifest=artifact_manifest,
        temperature=float(temperature),
        max_tokens=max_tokens,
        replicates=replicates,
    )
    stable_provenance = _compact_generation_provenance(base_provenance)
    selection_provenance = _selection_provenance(
        artifact_manifest=artifact_manifest,
        limit_conditions=limit_conditions,
        selected_condition_ids=selected_condition_ids,
    )
    output = Path(output_path)
    provenance_sidecar = _ensure_generation_provenance(output, stable_provenance, write_if_missing=False)
    selection_sidecar = _ensure_selection_provenance(
        output,
        selection_provenance,
        all_condition_ids=[row["condition_id"] for row in artifact_rows],
        write_if_missing=False,
    )
    existing_records = read_generation_records(output)
    histories = validate_generation_records(
        existing_records,
        artifact_rows=selected_rows,
        artifact_manifest=artifact_manifest,
        model=model,
        prompt=prompt,
        temperature=float(temperature),
        max_tokens=max_tokens,
        generation_provenance=stable_provenance,
    )

    required: list[tuple[dict[str, Any], int, str]] = []
    successful_keys = {
        logical_key for logical_key, history in histories.items() if history and history[-1]["status"] == "success"
    }
    for row in selected_rows:
        for replicate in range(1, replicates + 1):
            logical_key = make_generation_key(model.key, row["condition_id"], replicate)
            required.append((row, replicate, logical_key))
    required_keys = {logical_key for _, _, logical_key in required}
    pending = [item for item in required if item[2] not in successful_keys]
    summary: dict[str, Any] = {
        "model_key": model.key,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "prompt_key": prompt.key,
        "prompt_sha256": prompt.sha256,
        "artifact_sha256": artifact_manifest["content_sha256"],
        "artifact_rows": len(artifact_rows),
        "selected_condition_count": len(selected_rows),
        "selected_condition_ids": selected_condition_ids,
        "selected_condition_ids_sha256": selection_provenance["selected_condition_ids_sha256"],
        "limit_conditions": limit_conditions,
        "generation_provenance": stable_provenance,
        "selection_provenance": selection_provenance,
        "provenance_path": str(provenance_sidecar),
        "selection_provenance_path": str(selection_sidecar),
        "replicates": replicates,
        "required_keys": len(required_keys),
        "existing_records": len(existing_records),
        "existing_successes": len(required_keys.intersection(successful_keys)),
        "pending_keys": len(pending),
        "dry_run": dry_run,
        "api_calls_made": 0,
    }
    if dry_run:
        summary.update(
            {
                "complete": not pending,
                "planned_new_calls": len(pending),
                "pending_key_examples": [item[2] for item in pending[:5]],
                "endpoint_check": {"status": "not_run_dry_run", "served_model_ids": []},
            }
        )
        return summary
    if not pending:
        summary.update(
            {
                "complete": True,
                "planned_new_calls": 0,
                "new_successes": 0,
                "new_errors": 0,
                "failed_keys": 0,
                "endpoint_check": {"status": "not_needed_complete", "served_model_ids": []},
            }
        )
        return summary

    provenance_sidecar = _ensure_generation_provenance(output, stable_provenance, write_if_missing=True)
    selection_sidecar = _ensure_selection_provenance(
        output,
        selection_provenance,
        all_condition_ids=[row["condition_id"] for row in artifact_rows],
        write_if_missing=True,
    )
    summary["provenance_path"] = str(provenance_sidecar)
    summary["selection_provenance_path"] = str(selection_sidecar)

    created_client = client is None
    resolved_api_key = api_key
    if created_client:
        resolved_api_key = api_key if api_key is not None else os.environ.get(api_key_env, "EMPTY")
        client = _create_openai_client(base_url=base_url, api_key=resolved_api_key)
    endpoint_check: dict[str, Any] = {"status": "skipped", "served_model_ids": []}
    new_records: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(max_concurrency)
    append_lock = asyncio.Lock()

    async def process(item: tuple[dict[str, Any], int, str]) -> dict[str, Any]:
        row, replicate, logical_key = item
        async with semaphore:
            record = await _generate_one(
                client=client,
                row=row,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                artifact_manifest=artifact_manifest,
                replicate=replicate,
                resume_attempt=len(histories.get(logical_key, [])) + 1,
                temperature=float(temperature),
                max_tokens=max_tokens,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                api_key=resolved_api_key,
                generation_provenance=stable_provenance,
            )
        async with append_lock:
            _append_durable(output, record)
        return record

    try:
        if check_endpoint_model:
            endpoint_check = await _endpoint_model_check(client, model)
        new_records = list(await asyncio.gather(*(process(item) for item in pending)))
    finally:
        if created_client:
            await _close_client(client)

    new_successful = {record["generation_key"] for record in new_records if record["status"] == "success"}
    final_successful = successful_keys | new_successful
    failed_keys = sorted(required_keys - final_successful)
    summary.update(
        {
            "complete": not failed_keys,
            "planned_new_calls": len(pending),
            "api_calls_made": sum(record["attempts"] for record in new_records),
            "new_successes": len(new_successful),
            "new_errors": sum(record["status"] == "error" for record in new_records),
            "failed_keys": len(failed_keys),
            "failed_key_examples": failed_keys[:5],
            "endpoint_check": endpoint_check,
        }
    )
    return summary


run_generation = generate_figure6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-key", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--skip-endpoint-model-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit-conditions",
        type=int,
        default=0,
        help="Pilot limiter applied before replication; 0 selects all verified artifact rows",
    )
    args = parser.parse_args(argv)
    try:
        summary = asyncio.run(
            generate_figure6(
                args.artifact,
                args.output,
                model_key=args.model_key,
                prompt_path=args.prompt_path,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                replicates=args.replicates,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_concurrency=args.max_concurrency,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                check_endpoint_model=not args.skip_endpoint_model_check,
                dry_run=args.dry_run,
                limit_conditions=args.limit_conditions,
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {_safe_error(exc)}")
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run or summary["complete"]:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_REPLICATES",
    "DEFAULT_TEMPERATURE",
    "GENERATION_SCHEMA",
    "GENERATION_SCHEMA_VERSION",
    "extract_reasoning",
    "extract_reasoning_and_answer",
    "generate_figure6",
    "generation_key",
    "generation_provenance_path",
    "main",
    "make_generation_key",
    "read_generation_records",
    "run_generation",
    "selection_provenance_path",
    "validate_generation_records",
]
