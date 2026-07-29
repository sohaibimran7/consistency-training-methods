#!/usr/bin/env python3
"""Create and collect offline OpenAI Batch files for Figure 6 judging.

This module intentionally has no batch-submission command.  It turns an
already-materialized generation JSONL into deterministic request JSONL and
normalizes a downloaded Batch result JSONL.  The paper's judge prompt remains
an external, hash-pinned input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from string import Formatter
from typing import Any

from ctm.artifacts import write_atomic_bytes

PAPER_JUDGE_TEMPLATE_SHA256 = "e6158c9dba2466519450f4234e5dc0f9b4c97717b759ba6a133e2233f6dc3870"
DEFAULT_JUDGE_MODEL = "gpt-5"
MAX_JUDGE_TOKENS = 4_096
BATCH_ENDPOINT = "/v1/chat/completions"
# Conservative safety margins, deliberately below the documented OpenAI Batch
# maxima (50,000 requests and 200 MB input JSONL as of July 2026).  Queued-token
# limits remain model-specific and must be checked separately before submission.
DEFAULT_MAX_REQUESTS_PER_SHARD = 5_000
DEFAULT_MAX_BYTES_PER_SHARD = 150 * 1024 * 1024
GENERATION_KEY_FIELDS = (
    "model_key",
    "condition_id",
    "task_id",
    "valence",
    "config_name",
    "replicate",
)
REQUIRED_GENERATION_FIELDS = frozenset(
    {
        *GENERATION_KEY_FIELDS,
        "pair_id",
        "model_display",
        "model_id",
        "model_revision",
        "prompt",
        "response",
        "reasoning",
        "answer",
        "trace_present",
        "trace_source",
        "status",
        "generation_provenance",
        "system_prompt_provenance",
    }
)
VALID_VALENCES = frozenset({"safety", "capability"})
VALID_CONFIGS = frozenset({"baseline", *(f"F{index}" for index in range(1, 9))})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FENCED_JSON_RE = re.compile(r"\A\s*```(?:json)?\s*\n?(.*?)\n?```\s*\Z", re.DOTALL | re.IGNORECASE)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON: {exc}") from exc


def _read_jsonl(paths: Sequence[str | Path], *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_like in paths:
        path = Path(path_like)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: each {label} row must be an object")
                rows.append(row)
    if not rows:
        raise ValueError(f"{label} JSONL contained no records")
    return rows


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join((_canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows)


def _require_new_path(path: str | Path) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    return target


def load_judge_template(path: str | Path, *, expected_sha256: str) -> str:
    """Load the external paper prompt only when its exact bytes match the pin."""

    if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256.lower()) is None:
        raise ValueError("expected judge template SHA256 must be 64 lowercase hexadecimal characters")
    if expected_sha256.lower() != PAPER_JUDGE_TEMPLATE_SHA256:
        raise ValueError(
            "the expected judge template SHA256 must equal the pinned paper prompt hash "
            f"{PAPER_JUDGE_TEMPLATE_SHA256}"
        )
    target = Path(path)
    payload = target.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"judge template digest mismatch for {target}: expected {expected_sha256.lower()}, got {actual_sha256}"
        )
    try:
        template = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"judge template is not UTF-8: {target}") from exc
    if not template.strip():
        raise ValueError("judge template must not be empty")
    return template


def render_judge_prompt(template: str, *, task: str, reasoning: str, answer: str) -> str:
    """Render with the exact ``str.format`` semantics used by the paper code."""

    values = {"task": task, "reasoning": reasoning, "answer": answer}
    if any(not isinstance(value, str) for value in values.values()):
        raise TypeError("task, reasoning, and answer must be strings")
    try:
        fields = [field_name for _, field_name, _, _ in Formatter().parse(template) if field_name is not None]
    except ValueError as exc:
        raise ValueError(f"judge template has invalid format braces: {exc}") from exc
    missing = sorted(set(values) - set(fields))
    if missing:
        raise ValueError(f"judge template is missing placeholders: {missing}")
    unknown = sorted(set(fields) - set(values))
    if unknown:
        raise ValueError(f"judge template has unexpected placeholders: {unknown}")
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"judge template could not be rendered: {exc}") from exc


def generation_key(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicit, stable identity of one target-model generation."""

    source = _canonical_generation_aliases(record)
    missing = [field for field in GENERATION_KEY_FIELDS if field not in source]
    if missing:
        raise ValueError(f"generation is missing key fields: {missing}")
    return {field: source[field] for field in GENERATION_KEY_FIELDS}


def custom_id_for_generation(record: Mapping[str, Any]) -> str:
    """Derive a stable Batch custom ID from the complete generation key."""

    digest = hashlib.sha256(_canonical_json(generation_key(record)).encode("utf-8")).hexdigest()
    return f"figure6-{digest[:48]}"


def _canonical_generation_aliases(record: Mapping[str, Any]) -> dict[str, Any]:
    """Accept the materializer's names while exposing one judge-side schema."""

    output = dict(record)
    if "model_display" not in output and "model_display_name" in output:
        output["model_display"] = output["model_display_name"]
    if "config_name" not in output and "condition" in output:
        output["config_name"] = output["condition"]
    if "generation_provenance" not in output:
        keys = (
            "schema",
            "schema_version",
            "temperature",
            "max_tokens",
            "artifact_schema",
            "artifact_schema_version",
            "artifact_sha256",
            "dataset_id",
            "dataset_revision",
        )
        provenance = {key: output[key] for key in keys if key in output}
        if provenance:
            output["generation_provenance"] = provenance
    if "system_prompt_provenance" not in output:
        keys = ("prompt_key", "prompt_revision", "prompt_sha256", "system_prompt_sha256")
        provenance = {key: output[key] for key in keys if key in output}
        if provenance:
            output["system_prompt_provenance"] = provenance
    return output


def validate_generation(record: Mapping[str, Any], *, index: int | None = None) -> dict[str, Any]:
    prefix = f"generation {index}" if index is not None else "generation"
    output = _canonical_generation_aliases(record)
    missing = sorted(REQUIRED_GENERATION_FIELDS - output.keys())
    if missing:
        raise ValueError(f"{prefix} is missing required fields: {missing}")
    for field in (
        "model_key",
        "model_display",
        "model_id",
        "model_revision",
        "condition_id",
        "pair_id",
        "task_id",
        "prompt",
        "response",
        "reasoning",
        "answer",
        "trace_source",
        "status",
    ):
        if not isinstance(output[field], str) or (
            field not in {"reasoning", "answer", "response"} and not output[field]
        ):
            raise ValueError(f"{prefix}.{field} must be a non-empty string")
    if output["valence"] not in VALID_VALENCES:
        raise ValueError(f"{prefix}.valence must be safety or capability")
    if output["config_name"] not in VALID_CONFIGS:
        raise ValueError(f"{prefix}.config_name must be baseline or F1..F8")
    replicate = output["replicate"]
    if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate not in {1, 2, 3}:
        raise ValueError(f"{prefix}.replicate must be one of 1, 2, 3")
    if not isinstance(output["trace_present"], bool):
        raise ValueError(f"{prefix}.trace_present must be a boolean")
    if not output["trace_present"]:
        raise ValueError(f"{prefix} has no reasoning trace and must not be sent for paid judging")
    if not output["reasoning"].strip():
        raise ValueError(f"{prefix}.reasoning must be non-blank for judging")
    for field in ("generation_provenance", "system_prompt_provenance"):
        if not isinstance(output[field], Mapping) or not output[field]:
            raise ValueError(f"{prefix}.{field} must be a non-empty object")
        _canonical_json(output[field])
    return output


def select_successful_generations(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse append-only resume histories to one terminal success per key."""

    if not records:
        raise ValueError("generation records must not be empty")
    histories: dict[str, list[dict[str, Any]]] = {}
    for index, record_like in enumerate(records, start=1):
        if not isinstance(record_like, Mapping):
            raise TypeError(f"generation {index} must be an object")
        record = _canonical_generation_aliases(record_like)
        status = record.get("status")
        if status not in {"success", "error"}:
            raise ValueError(f"generation {index}.status must be success or error")
        logical_key_value = record.get("generation_key")
        logical_key = (
            logical_key_value
            if isinstance(logical_key_value, str) and logical_key_value
            else _canonical_json(generation_key(record))
        )
        histories.setdefault(logical_key, []).append(record)
    selected = []
    for logical_key, history in histories.items():
        successes = [record for record in history if record["status"] == "success"]
        if not successes:
            raise ValueError(f"generation key {logical_key!r} has no successful terminal record")
        if len(successes) > 1:
            raise ValueError(f"generation key {logical_key!r} has duplicate successful records")
        if history[-1] is not successes[0]:
            raise ValueError(f"generation key {logical_key!r} has records appended after success")
        selected.append(successes[0])
    selected.sort(key=custom_id_for_generation)
    return selected


def create_batch_requests(
    generations: Sequence[Mapping[str, Any]],
    template: str,
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_completion_tokens: int = MAX_JUDGE_TOKENS,
) -> list[dict[str, Any]]:
    """Build deterministic Chat Completions Batch request records."""

    if not isinstance(judge_model, str) or not judge_model:
        raise ValueError("judge_model must be a non-empty string")
    if (
        not isinstance(max_completion_tokens, int)
        or isinstance(max_completion_tokens, bool)
        or max_completion_tokens < 1
    ):
        raise ValueError("max_completion_tokens must be an integer >= 1")
    selected = select_successful_generations(generations)
    validated = [validate_generation(record, index=index) for index, record in enumerate(selected, start=1)]
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in validated:
        custom_id = custom_id_for_generation(record)
        if custom_id in seen:
            raise ValueError(f"duplicate generation key produces custom_id {custom_id}")
        seen.add(custom_id)
        judge_prompt = render_judge_prompt(
            template,
            task=record["prompt"],
            reasoning=record["reasoning"],
            answer=record["answer"],
        )
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": {
                    "model": judge_model,
                    "max_completion_tokens": max_completion_tokens,
                    "messages": [{"role": "system", "content": judge_prompt}],
                },
            }
        )
    requests.sort(key=lambda row: row["custom_id"])
    return requests


def shard_batch_requests(
    requests: Sequence[Mapping[str, Any]],
    *,
    max_requests_per_shard: int = DEFAULT_MAX_REQUESTS_PER_SHARD,
    max_bytes_per_shard: int = DEFAULT_MAX_BYTES_PER_SHARD,
) -> list[list[dict[str, Any]]]:
    """Greedily shard canonical request lines by count and exact UTF-8 bytes."""

    if (
        not isinstance(max_requests_per_shard, int)
        or isinstance(max_requests_per_shard, bool)
        or max_requests_per_shard < 1
    ):
        raise ValueError("max_requests_per_shard must be an integer >= 1")
    if not isinstance(max_bytes_per_shard, int) or isinstance(max_bytes_per_shard, bool) or max_bytes_per_shard < 1:
        raise ValueError("max_bytes_per_shard must be an integer >= 1")
    if not requests:
        raise ValueError("batch requests must not be empty")
    ordered = [dict(request) for request in sorted(requests, key=lambda row: str(row.get("custom_id", "")))]
    shards: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for request in ordered:
        line_bytes = len((_canonical_json(request) + "\n").encode("utf-8"))
        if line_bytes > max_bytes_per_shard:
            raise ValueError(
                f"request {request.get('custom_id')!r} is {line_bytes} bytes and exceeds the "
                f"{max_bytes_per_shard}-byte shard safety limit"
            )
        would_exceed_count = len(current) >= max_requests_per_shard
        would_exceed_bytes = current and current_bytes + line_bytes > max_bytes_per_shard
        if would_exceed_count or would_exceed_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(request)
        current_bytes += line_bytes
    if current:
        shards.append(current)
    return shards


def _batch_output_paths(output_path: str | Path, shard_count: int) -> tuple[list[Path], Path]:
    base = Path(output_path)
    if base.suffix.lower() != ".jsonl":
        raise ValueError("batch request output path must end in .jsonl")
    if shard_count == 1:
        shard_paths = [base]
    else:
        shard_paths = [
            base.with_name(f"{base.stem}.part-{index:05d}-of-{shard_count:05d}{base.suffix}")
            for index in range(1, shard_count + 1)
        ]
    manifest_path = base.with_suffix(base.suffix + ".manifest.json")
    return shard_paths, manifest_path


def create_batch_jsonl(
    generations: Sequence[Mapping[str, Any]],
    *,
    template_path: str | Path,
    expected_template_sha256: str,
    output_path: str | Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_completion_tokens: int = MAX_JUDGE_TOKENS,
    max_requests_per_shard: int = DEFAULT_MAX_REQUESTS_PER_SHARD,
    max_bytes_per_shard: int = DEFAULT_MAX_BYTES_PER_SHARD,
) -> dict[str, Any]:
    """Write deterministic request shards and a manifest, but never submit."""

    template = load_judge_template(template_path, expected_sha256=expected_template_sha256)
    requests = create_batch_requests(
        generations,
        template,
        judge_model=judge_model,
        max_completion_tokens=max_completion_tokens,
    )
    shards = shard_batch_requests(
        requests,
        max_requests_per_shard=max_requests_per_shard,
        max_bytes_per_shard=max_bytes_per_shard,
    )
    shard_paths, manifest_path = _batch_output_paths(output_path, len(shards))
    payloads = [_jsonl_bytes(shard) for shard in shards]
    reserved_paths = [Path(output_path), *shard_paths, manifest_path]
    for target in dict.fromkeys(reserved_paths):
        _require_new_path(target)
    shard_metadata = []
    for index, (path, shard, payload) in enumerate(zip(shard_paths, shards, payloads, strict=True), start=1):
        custom_ids = [str(request["custom_id"]) for request in shard]
        shard_metadata.append(
            {
                "index": index,
                "file_name": path.name,
                "row_count": len(shard),
                "utf8_bytes": len(payload),
                "content_sha256": hashlib.sha256(payload).hexdigest(),
                "custom_ids": custom_ids,
            }
        )
    manifest = {
        "schema": "ctm.eval_awareness.figure6_judge_batch_manifest.v1",
        "row_count": len(requests),
        "shard_count": len(shards),
        "total_utf8_bytes": sum(len(payload) for payload in payloads),
        "ordered_custom_ids_sha256": hashlib.sha256(
            "\n".join(request["custom_id"] for request in requests).encode("utf-8")
        ).hexdigest(),
        "judge_model": judge_model,
        "max_completion_tokens": max_completion_tokens,
        "judge_template_sha256": expected_template_sha256.lower(),
        "endpoint": BATCH_ENDPOINT,
        "safety_limits": {
            "max_requests_per_shard": max_requests_per_shard,
            "max_utf8_bytes_per_shard": max_bytes_per_shard,
            "note": "Configurable safety margins below API maxima; queued-token limits are model-specific.",
        },
        "shards": shard_metadata,
        "submitted": False,
    }
    for path, payload in zip(shard_paths, payloads, strict=True):
        write_atomic_bytes(path, payload)
    write_atomic_bytes(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return {**manifest, "manifest_path": str(manifest_path), "shard_paths": [str(path) for path in shard_paths]}


def parse_judge_json(content: str) -> dict[str, Any]:
    """Parse a plain JSON object or one object enclosed in a JSON code fence."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("judge message content must be a non-empty string")
    match = _FENCED_JSON_RE.fullmatch(content)
    candidate = match.group(1) if match is not None else content.strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge content is not plain or fenced JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("judge content JSON must be an object")
    return value


def _casefold_key(mapping: Mapping[str, Any], *names: str) -> Any:
    by_name = {str(key).casefold(): value for key, value in mapping.items()}
    for name in names:
        if name.casefold() in by_name:
            return by_name[name.casefold()]
    return None


def _conclusion(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or value.strip().casefold() not in {"yes", "no"}:
        raise ValueError(f"judge {field} must be Yes or No")
    return value.strip().casefold()


def normalize_judge_object(value: Mapping[str, Any]) -> dict[str, str]:
    """Validate the paper judge's conclusions and normalize their casing."""

    awareness_block = _casefold_key(value, "model_awareness", "awareness", "evaluation_awareness")
    performance_block = _casefold_key(value, "task_performance", "performance")
    if not isinstance(awareness_block, Mapping):
        awareness_block = {}
    if not isinstance(performance_block, Mapping):
        performance_block = {}

    awareness_value = _casefold_key(value, "awareness_conclusion")
    if awareness_value is None:
        awareness_value = _casefold_key(awareness_block, "conclusion", "awareness_conclusion")
    type_value = _casefold_key(value, "awareness_type")
    if type_value is None:
        type_value = _casefold_key(awareness_block, "awareness_type", "type")
    performance_value = _casefold_key(value, "performance_conclusion")
    if performance_value is None:
        performance_value = _casefold_key(performance_block, "conclusion", "performance_conclusion")

    awareness = _conclusion(awareness_value, field="awareness_conclusion")
    performance = _conclusion(performance_value, field="performance_conclusion")
    if not isinstance(type_value, str):
        raise ValueError("judge awareness_type must be Safety, Capability, or N/A")
    normalized_type = type_value.strip().casefold().replace(" ", "")
    if normalized_type in {"n/a", "na", "notapplicable"}:
        awareness_type = "N/A"
    elif normalized_type in {"safety", "capability"}:
        awareness_type = normalized_type
    else:
        raise ValueError("judge awareness_type must be Safety, Capability, or N/A")
    if awareness == "no" and awareness_type != "N/A":
        raise ValueError("judge awareness=no requires awareness_type=N/A")
    if awareness == "yes" and awareness_type == "N/A":
        raise ValueError("judge awareness=yes requires awareness_type=safety or capability")
    return {
        "awareness_conclusion": awareness,
        "awareness_type": awareness_type,
        "performance_conclusion": performance,
    }


def _message_content(batch_record: Mapping[str, Any], *, custom_id: str) -> tuple[str, dict[str, Any]]:
    error = batch_record.get("error")
    if error not in (None, {}):
        raise ValueError(f"batch output {custom_id} contains an error record: {error!r}")
    response = batch_record.get("response")
    if not isinstance(response, Mapping):
        raise ValueError(f"batch output {custom_id}.response must be an object")
    status_code = response.get("status_code")
    if status_code != 200:
        raise ValueError(f"batch output {custom_id} has HTTP status {status_code!r}, expected 200")
    body = response.get("body")
    if not isinstance(body, Mapping):
        raise ValueError(f"batch output {custom_id}.response.body must be an object")
    if body.get("error") not in (None, {}):
        raise ValueError(f"batch output {custom_id} body contains an error: {body['error']!r}")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ValueError(f"batch output {custom_id} must contain exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError(f"batch output {custom_id} choice must contain string message.content")
    return message["content"], dict(body)


def _normalized_generation_fields(generation: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        **generation_key(generation),
        "model_display": generation["model_display"],
        "model_id": generation["model_id"],
        "model_revision": generation["model_revision"],
        "pair_id": generation["pair_id"],
        "generation_status": generation["status"],
        "trace_present": generation["trace_present"],
        "trace_source": generation["trace_source"],
        "generation_provenance": dict(generation["generation_provenance"]),
        "system_prompt_provenance": dict(generation["system_prompt_provenance"]),
    }
    fields["generation_key"] = generation_key(generation)
    fields["generation_record_sha256"] = hashlib.sha256(_canonical_json(dict(generation)).encode("utf-8")).hexdigest()
    return fields


def collect_batch_outputs(
    generations: Sequence[Mapping[str, Any]],
    batch_outputs: Sequence[Mapping[str, Any]],
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_template_sha256: str = PAPER_JUDGE_TEMPLATE_SHA256,
    max_completion_tokens: int = MAX_JUDGE_TOKENS,
) -> list[dict[str, Any]]:
    """Match downloaded Batch rows to generations and normalize all judgments."""

    if not batch_outputs:
        raise ValueError("batch output records must not be empty")
    if judge_template_sha256 != PAPER_JUDGE_TEMPLATE_SHA256:
        raise ValueError("judge template hash does not match the pinned paper prompt")
    if (
        not isinstance(max_completion_tokens, int)
        or isinstance(max_completion_tokens, bool)
        or max_completion_tokens < 1
    ):
        raise ValueError("max_completion_tokens must be an integer >= 1")
    selected = select_successful_generations(generations)
    validated = [validate_generation(record, index=index) for index, record in enumerate(selected, start=1)]
    generations_by_id: dict[str, dict[str, Any]] = {}
    for generation in validated:
        custom_id = custom_id_for_generation(generation)
        if custom_id in generations_by_id:
            raise ValueError(f"duplicate generation key produces custom_id {custom_id}")
        generations_by_id[custom_id] = generation

    outputs_by_id: dict[str, Mapping[str, Any]] = {}
    for index, output in enumerate(batch_outputs, start=1):
        if not isinstance(output, Mapping):
            raise TypeError(f"batch output {index} must be an object")
        custom_id = output.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"batch output {index}.custom_id must be a non-empty string")
        if custom_id in outputs_by_id:
            raise ValueError(f"duplicate batch output custom_id {custom_id}")
        if custom_id not in generations_by_id:
            raise ValueError(f"unexpected batch output custom_id {custom_id}")
        outputs_by_id[custom_id] = output
    missing = sorted(set(generations_by_id) - set(outputs_by_id))
    if missing:
        raise ValueError(f"missing {len(missing)} batch output record(s), including {missing[:3]}")

    normalized: list[dict[str, Any]] = []
    for custom_id in sorted(generations_by_id):
        content, raw_body = _message_content(outputs_by_id[custom_id], custom_id=custom_id)
        raw_object = parse_judge_json(content)
        conclusions = normalize_judge_object(raw_object)
        normalized.append(
            {
                **_normalized_generation_fields(generations_by_id[custom_id]),
                "custom_id": custom_id,
                "judge_model": judge_model,
                "judge_template_sha256": judge_template_sha256,
                "judge_max_completion_tokens": max_completion_tokens,
                **conclusions,
                "judge_status": "ok",
                "raw_judge_object": raw_object,
                "raw_judge_body": raw_body,
            }
        )
    return normalized


def collect_batch_jsonl(
    generations: Sequence[Mapping[str, Any]],
    batch_outputs: Sequence[Mapping[str, Any]],
    *,
    template_path: str | Path,
    expected_template_sha256: str,
    output_path: str | Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_completion_tokens: int = MAX_JUDGE_TOKENS,
) -> dict[str, Any]:
    """Verify the external prompt and write normalized judgment JSONL."""

    load_judge_template(template_path, expected_sha256=expected_template_sha256)
    normalized = collect_batch_outputs(
        generations,
        batch_outputs,
        judge_model=judge_model,
        judge_template_sha256=expected_template_sha256.lower(),
        max_completion_tokens=max_completion_tokens,
    )
    target = _require_new_path(output_path)
    payload = _jsonl_bytes(normalized)
    write_atomic_bytes(target, payload)
    return {
        "row_count": len(normalized),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "judge_model": judge_model,
        "judge_template_sha256": expected_template_sha256.lower(),
        "max_completion_tokens": max_completion_tokens,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "collect"):
        child = subparsers.add_parser(command)
        child.add_argument("--generations", type=Path, nargs="+", required=True)
        child.add_argument("--judge-template", type=Path, required=True)
        child.add_argument(
            "--expected-template-sha256",
            required=True,
            help=f"Must be the paper prompt pin: {PAPER_JUDGE_TEMPLATE_SHA256}",
        )
        child.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
        child.add_argument("--max-completion-tokens", type=int, default=MAX_JUDGE_TOKENS)
        child.add_argument("--output", type=Path, required=True)
    subparsers.choices["create"].add_argument(
        "--max-requests-per-shard",
        type=int,
        default=DEFAULT_MAX_REQUESTS_PER_SHARD,
        help="Configurable safety margin; not the API maximum.",
    )
    subparsers.choices["create"].add_argument(
        "--max-bytes-per-shard",
        type=int,
        default=DEFAULT_MAX_BYTES_PER_SHARD,
        help="Exact canonical JSONL UTF-8 byte safety margin; not the API maximum.",
    )
    subparsers.choices["collect"].add_argument("--batch-output", type=Path, nargs="+", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    generations = _read_jsonl(args.generations, label="generation")
    if args.command == "create":
        summary = create_batch_jsonl(
            generations,
            template_path=args.judge_template,
            expected_template_sha256=args.expected_template_sha256,
            output_path=args.output,
            judge_model=args.judge_model,
            max_completion_tokens=args.max_completion_tokens,
            max_requests_per_shard=args.max_requests_per_shard,
            max_bytes_per_shard=args.max_bytes_per_shard,
        )
    else:
        summary = collect_batch_jsonl(
            generations,
            _read_jsonl(args.batch_output, label="batch output"),
            template_path=args.judge_template,
            expected_template_sha256=args.expected_template_sha256,
            output_path=args.output,
            judge_model=args.judge_model,
            max_completion_tokens=args.max_completion_tokens,
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
