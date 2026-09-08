#!/usr/bin/env python3
"""Submit, monitor, and download OpenAI Batch jobs for Figure 6 judging.

Request creation and result collection remain in :mod:`figure6_judge`.  This
module is the deliberately separate paid-operation boundary: only ``submit``
can create provider resources, and it requires an explicit ``--yes`` flag.
``status`` and ``download`` perform read-only provider operations.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from ctm.artifacts import write_atomic_bytes
from ctm_data.adapters.eval_awareness.figure6_judge import (
    BATCH_ENDPOINT,
    DEFAULT_JUDGE_MODEL,
    MAX_JUDGE_TOKENS,
    PAPER_JUDGE_TEMPLATE_SHA256,
)

REQUEST_MANIFEST_SCHEMA = "ctm.eval_awareness.figure6_judge_batch_manifest.v1"
LIFECYCLE_MANIFEST_SCHEMA = "ctm.eval_awareness.figure6_openai_batch_lifecycle.v1"
JUDGE_PROTOCOL = "ctm.eval_awareness.figure6.paper_judge.v1"
COMPLETION_WINDOW = "24h"

ACTIVE_BATCH_STATUSES = frozenset({"validating", "in_progress", "finalizing", "cancelling"})
TERMINAL_BATCH_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})
FAILED_BATCH_STATUSES = frozenset({"failed", "expired", "cancelled"})
KNOWN_BATCH_STATUSES = ACTIVE_BATCH_STATUSES | TERMINAL_BATCH_STATUSES
PROVIDER_TIMESTAMP_FIELDS = (
    "created_at",
    "in_progress_at",
    "expires_at",
    "finalizing_at",
    "completed_at",
    "failed_at",
    "expired_at",
    "cancelling_at",
    "cancelled_at",
)


class BatchLifecycleError(RuntimeError):
    """The local audit state or provider lifecycle is unsafe to continue."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BatchLifecycleError(f"value is not canonical JSON: {exc}") from exc


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _json_safe(value: object) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump(mode="json"))
        except TypeError:
            return _json_safe(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return repr(value)


def _required_string(value: object, field: str) -> str:
    result = _field(value, field)
    if not isinstance(result, str) or not result:
        raise BatchLifecycleError(f"provider response has no non-empty {field}")
    return result


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise BatchLifecycleError(f"missing {label}: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BatchLifecycleError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchLifecycleError(f"{label} {path} must contain a JSON object")
    return value, payload


def _read_request_rows(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BatchLifecycleError(f"invalid request JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise BatchLifecycleError(f"request at {path}:{line_number} must be an object")
        rows.append(row)
    if not rows:
        raise BatchLifecycleError(f"request shard contains no records: {path}")
    return rows, payload


def _validate_request_protocol(row: Mapping[str, Any], *, path: Path, line_number: int) -> str:
    prefix = f"{path}:{line_number}"
    custom_id = row.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        raise BatchLifecycleError(f"{prefix}: custom_id must be a non-empty string")
    if row.get("method") != "POST":
        raise BatchLifecycleError(f"{prefix}: method must be POST")
    if row.get("url") != BATCH_ENDPOINT:
        raise BatchLifecycleError(f"{prefix}: url must be {BATCH_ENDPOINT}")
    body = row.get("body")
    if not isinstance(body, Mapping):
        raise BatchLifecycleError(f"{prefix}: body must be an object")
    if body.get("model") != DEFAULT_JUDGE_MODEL:
        raise BatchLifecycleError(f"{prefix}: judge model must be {DEFAULT_JUDGE_MODEL}")
    if body.get("max_completion_tokens") != MAX_JUDGE_TOKENS:
        raise BatchLifecycleError(f"{prefix}: max_completion_tokens must be {MAX_JUDGE_TOKENS}")
    messages = body.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], Mapping)
        or messages[0].get("role") != "system"
        or not isinstance(messages[0].get("content"), str)
        or not messages[0]["content"]
    ):
        raise BatchLifecycleError(f"{prefix}: messages must contain the one non-empty paper judge system prompt")
    return custom_id


def _as_nonnegative_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BatchLifecycleError(f"{label} must be a non-negative integer")
    return value


def load_request_plan(request_manifest_path: str | Path) -> dict[str, Any]:
    """Load and fully verify deterministic request shards before submission."""

    manifest_path = Path(request_manifest_path).expanduser().resolve()
    manifest, manifest_payload = _read_json_object(manifest_path, label="request manifest")
    if manifest.get("schema") != REQUEST_MANIFEST_SCHEMA:
        raise BatchLifecycleError(
            f"request manifest schema must be {REQUEST_MANIFEST_SCHEMA}, got {manifest.get('schema')!r}"
        )

    expected_protocol = {
        "judge_model": DEFAULT_JUDGE_MODEL,
        "max_completion_tokens": MAX_JUDGE_TOKENS,
        "judge_template_sha256": PAPER_JUDGE_TEMPLATE_SHA256,
        "endpoint": BATCH_ENDPOINT,
    }
    for field, expected in expected_protocol.items():
        if manifest.get(field) != expected:
            raise BatchLifecycleError(
                f"request manifest {field} must be the pinned Figure 6 value {expected!r}, "
                f"got {manifest.get(field)!r}"
            )

    declared_row_count = _as_nonnegative_int(manifest.get("row_count"), label="request manifest row_count")
    declared_shard_count = _as_nonnegative_int(manifest.get("shard_count"), label="request manifest shard_count")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise BatchLifecycleError("request manifest shards must be a non-empty list")
    if declared_shard_count != len(shards):
        raise BatchLifecycleError(
            f"request manifest shard_count is {declared_shard_count}, but it lists {len(shards)} shards"
        )

    verified_shards: list[dict[str, Any]] = []
    all_custom_ids: list[str] = []
    seen_custom_ids: set[str] = set()
    for expected_index, shard in enumerate(shards, start=1):
        if not isinstance(shard, Mapping):
            raise BatchLifecycleError(f"request manifest shard {expected_index} must be an object")
        if shard.get("index") != expected_index:
            raise BatchLifecycleError(
                f"request manifest shard {expected_index} has unexpected index {shard.get('index')!r}"
            )
        file_name = shard.get("file_name")
        if not isinstance(file_name, str) or not file_name or Path(file_name).name != file_name:
            raise BatchLifecycleError(f"request manifest shard {expected_index}.file_name must be a plain file name")
        shard_path = (manifest_path.parent / file_name).resolve()
        if not shard_path.is_file():
            raise BatchLifecycleError(f"missing request shard {expected_index}: {shard_path}")
        rows, payload = _read_request_rows(shard_path)
        actual_sha256 = _sha256(payload)
        declared_sha256 = shard.get("content_sha256")
        if actual_sha256 != declared_sha256:
            raise BatchLifecycleError(
                f"request shard {shard_path} hash mismatch: manifest has {declared_sha256!r}, "
                f"bytes hash to {actual_sha256}"
            )
        declared_bytes = _as_nonnegative_int(
            shard.get("utf8_bytes"), label=f"request manifest shard {expected_index}.utf8_bytes"
        )
        if declared_bytes != len(payload):
            raise BatchLifecycleError(
                f"request shard {shard_path} byte count mismatch: manifest has {declared_bytes}, "
                f"file has {len(payload)}"
            )
        declared_rows = _as_nonnegative_int(
            shard.get("row_count"), label=f"request manifest shard {expected_index}.row_count"
        )
        if declared_rows != len(rows):
            raise BatchLifecycleError(
                f"request shard {shard_path} row count mismatch: manifest has {declared_rows}, " f"file has {len(rows)}"
            )
        custom_ids = [
            _validate_request_protocol(row, path=shard_path, line_number=line_number)
            for line_number, row in enumerate(rows, start=1)
        ]
        declared_custom_ids = shard.get("custom_ids")
        if declared_custom_ids != custom_ids:
            raise BatchLifecycleError(f"request shard {shard_path} custom_ids do not match its manifest entry")
        duplicates = sorted(custom_id for custom_id in custom_ids if custom_id in seen_custom_ids)
        if duplicates:
            raise BatchLifecycleError(f"duplicate request custom_id across shards: {duplicates[:3]}")
        seen_custom_ids.update(custom_ids)
        all_custom_ids.extend(custom_ids)
        verified_shards.append(
            {
                "index": expected_index,
                "request_path": str(shard_path),
                "request_file_name": file_name,
                "request_count": len(rows),
                "request_utf8_bytes": len(payload),
                "request_content_sha256": actual_sha256,
                "custom_ids_sha256": _sha256("\n".join(custom_ids).encode("utf-8")),
                "remote_file_name": f"ctm-figure6-{expected_index:05d}-{actual_sha256}.jsonl",
            }
        )

    if declared_row_count != len(all_custom_ids):
        raise BatchLifecycleError(
            f"request manifest row_count is {declared_row_count}, but shards contain {len(all_custom_ids)} rows"
        )
    ordered_hash = _sha256("\n".join(all_custom_ids).encode("utf-8"))
    if manifest.get("ordered_custom_ids_sha256") != ordered_hash:
        raise BatchLifecycleError("request manifest ordered_custom_ids_sha256 does not match the shard contents")

    return {
        "request_manifest_path": str(manifest_path),
        "request_manifest_content_sha256": _sha256(manifest_payload),
        "request_count": len(all_custom_ids),
        "shard_count": len(verified_shards),
        "ordered_custom_ids_sha256": ordered_hash,
        "protocol": {
            "name": JUDGE_PROTOCOL,
            **expected_protocol,
            "completion_window": COMPLETION_WINDOW,
        },
        "shards": verified_shards,
    }


def _operation_id(plan: Mapping[str, Any], shard: Mapping[str, Any]) -> str:
    payload = {
        "request_manifest_content_sha256": plan["request_manifest_content_sha256"],
        "request_content_sha256": shard["request_content_sha256"],
        "shard_index": shard["index"],
        "protocol": plan["protocol"],
    }
    return _sha256(_canonical_json(payload).encode("utf-8"))


def _new_lifecycle_manifest(plan: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    created_at = _utc_now()
    shards = []
    for planned in plan["shards"]:
        shards.append(
            {
                **dict(planned),
                "operation_id": _operation_id(plan, planned),
                "input_file_id": None,
                "uploaded_at": None,
                "batch_id": None,
                "batch_created_at": None,
                "status": "not_submitted",
                "status_checked_at": None,
                "provider_timestamps": {},
                "request_counts": None,
                "provider_errors": None,
                "output_file_id": None,
                "error_file_id": None,
                "downloads": {},
            }
        )
    return {
        "schema": LIFECYCLE_MANIFEST_SCHEMA,
        "manifest_path": str(manifest_path),
        "created_at": created_at,
        "updated_at": created_at,
        "request_manifest": {
            "path": plan["request_manifest_path"],
            "content_sha256": plan["request_manifest_content_sha256"],
            "ordered_custom_ids_sha256": plan["ordered_custom_ids_sha256"],
        },
        "protocol": dict(plan["protocol"]),
        "request_count": plan["request_count"],
        "shard_count": plan["shard_count"],
        "approval": None,
        "shards": shards,
        "events": [
            {
                "at": created_at,
                "event": "lifecycle_initialized",
                "request_manifest_content_sha256": plan["request_manifest_content_sha256"],
            }
        ],
    }


def _validate_lifecycle_manifest(manifest: Mapping[str, Any], *, manifest_path: Path) -> None:
    if manifest.get("schema") != LIFECYCLE_MANIFEST_SCHEMA:
        raise BatchLifecycleError(
            f"lifecycle manifest schema must be {LIFECYCLE_MANIFEST_SCHEMA}, got {manifest.get('schema')!r}"
        )
    if manifest.get("manifest_path") != str(manifest_path):
        raise BatchLifecycleError(
            f"lifecycle manifest path binding is {manifest.get('manifest_path')!r}, expected {str(manifest_path)!r}"
        )
    protocol = manifest.get("protocol")
    expected_protocol = {
        "name": JUDGE_PROTOCOL,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "max_completion_tokens": MAX_JUDGE_TOKENS,
        "judge_template_sha256": PAPER_JUDGE_TEMPLATE_SHA256,
        "endpoint": BATCH_ENDPOINT,
        "completion_window": COMPLETION_WINDOW,
    }
    if protocol != expected_protocol:
        raise BatchLifecycleError("lifecycle manifest judge protocol does not match the pinned Figure 6 protocol")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise BatchLifecycleError("lifecycle manifest shards must be a non-empty list")
    if manifest.get("shard_count") != len(shards):
        raise BatchLifecycleError("lifecycle manifest shard_count does not match its shards")
    request_count = sum(
        _as_nonnegative_int(shard.get("request_count"), label=f"lifecycle shard {index}.request_count")
        for index, shard in enumerate(shards, start=1)
        if isinstance(shard, Mapping)
    )
    if len(shards) != sum(isinstance(shard, Mapping) for shard in shards):
        raise BatchLifecycleError("every lifecycle manifest shard must be an object")
    if manifest.get("request_count") != request_count:
        raise BatchLifecycleError("lifecycle manifest request_count does not match its shards")


def _assert_lifecycle_matches_plan(manifest: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    request_manifest = manifest.get("request_manifest")
    expected_request_manifest = {
        "path": plan["request_manifest_path"],
        "content_sha256": plan["request_manifest_content_sha256"],
        "ordered_custom_ids_sha256": plan["ordered_custom_ids_sha256"],
    }
    if request_manifest != expected_request_manifest:
        raise BatchLifecycleError("existing lifecycle manifest is bound to a different request manifest")
    if manifest.get("protocol") != plan.get("protocol"):
        raise BatchLifecycleError("existing lifecycle manifest is bound to a different judge protocol")
    if manifest.get("request_count") != plan.get("request_count"):
        raise BatchLifecycleError("existing lifecycle manifest is bound to a different request count")
    current_shards = manifest.get("shards")
    planned_shards = plan.get("shards")
    if not isinstance(current_shards, list) or not isinstance(planned_shards, list):
        raise BatchLifecycleError("lifecycle/request plan shards are malformed")
    if len(current_shards) != len(planned_shards):
        raise BatchLifecycleError("existing lifecycle manifest is bound to a different shard count")
    binding_fields = (
        "index",
        "request_path",
        "request_file_name",
        "request_count",
        "request_utf8_bytes",
        "request_content_sha256",
        "custom_ids_sha256",
        "remote_file_name",
    )
    for current, planned in zip(current_shards, planned_shards, strict=True):
        for field in binding_fields:
            if current.get(field) != planned.get(field):
                raise BatchLifecycleError(
                    f"existing lifecycle shard {planned.get('index')} has a different {field} binding"
                )
        if current.get("operation_id") != _operation_id(plan, planned):
            raise BatchLifecycleError(f"existing lifecycle shard {planned.get('index')} has an invalid operation_id")


def _load_lifecycle(manifest_path: Path) -> dict[str, Any]:
    manifest, _ = _read_json_object(manifest_path, label="lifecycle manifest")
    _validate_lifecycle_manifest(manifest, manifest_path=manifest_path)
    return manifest


def _save_lifecycle(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _validate_lifecycle_manifest(manifest, manifest_path=manifest_path)
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_atomic_bytes(manifest_path, payload)
    saved, saved_payload = _read_json_object(manifest_path, label="saved lifecycle manifest")
    if _sha256(saved_payload) != _sha256(payload) or saved != manifest:
        raise BatchLifecycleError(f"atomic lifecycle manifest verification failed: {manifest_path}")


@contextmanager
def _manifest_lock(manifest_path: Path) -> Iterable[None]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = manifest_path.with_name(manifest_path.name + ".lock")
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BatchLifecycleError(f"another process is using lifecycle manifest {manifest_path}") from exc
        yield


def _append_event(manifest: dict[str, Any], event: str, *, shard: Mapping[str, Any], **details: Any) -> None:
    manifest["events"].append(
        {
            "at": _utc_now(),
            "event": event,
            "shard_index": shard["index"],
            **{key: _json_safe(value) for key, value in details.items()},
        }
    )


def _create_openai_client() -> Any:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise BatchLifecycleError("OPENAI_API_KEY must be set in the environment")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise BatchLifecycleError("OpenAI Batch lifecycle commands require openai>=2.45") from exc
    return OpenAI(api_key=api_key)


def _submission_plan_lines(plan: Mapping[str, Any], manifest_path: Path) -> list[str]:
    protocol = plan["protocol"]
    lines = [
        "PAID OPENAI BATCH SUBMISSION PLAN",
        f"request manifest: {plan['request_manifest_path']}",
        f"lifecycle manifest: {manifest_path}",
        f"protocol: {protocol['name']}",
        f"judge model: {protocol['judge_model']}",
        f"endpoint: {protocol['endpoint']}",
        f"max_completion_tokens: {protocol['max_completion_tokens']}",
        f"judge template SHA-256: {protocol['judge_template_sha256']}",
        f"completion_window: {protocol['completion_window']}",
        f"requests: {plan['request_count']}",
        f"shards: {plan['shard_count']}",
    ]
    for shard in plan["shards"]:
        lines.append(
            f"  [{shard['index']}] {shard['request_path']} "
            f"(requests={shard['request_count']}, bytes={shard['request_utf8_bytes']}, "
            f"sha256={shard['request_content_sha256']})"
        )
    return lines


def _batch_metadata(manifest: Mapping[str, Any], shard: Mapping[str, Any]) -> dict[str, str]:
    return {
        "ctm_protocol": JUDGE_PROTOCOL,
        "ctm_operation_id": str(shard["operation_id"]),
        "ctm_request_sha256": str(shard["request_content_sha256"]),
        "ctm_manifest_sha256": str(manifest["request_manifest"]["content_sha256"]),
        "ctm_shard_index": str(shard["index"]),
    }


def _content_bytes(response: object) -> bytes:
    if isinstance(response, bytes):
        return response
    if isinstance(response, (bytearray, memoryview)):
        return bytes(response)
    read = getattr(response, "read", None)
    if callable(read):
        value = read()
        if isinstance(value, bytes):
            return value
        if isinstance(value, (bytearray, memoryview)):
            return bytes(value)
        if isinstance(value, str):
            return value.encode("utf-8")
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content
    if isinstance(content, (bytearray, memoryview)):
        return bytes(content)
    text_value = getattr(response, "text", None)
    if isinstance(text_value, str):
        return text_value.encode("utf-8")
    if isinstance(response, str):
        return response.encode("utf-8")
    raise BatchLifecycleError(
        f"unsupported OpenAI file-content response type: {type(response).__module__}.{type(response).__name__}"
    )


def _iter_provider_items(page: object) -> Iterable[object]:
    if isinstance(page, Mapping):
        data = page.get("data")
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            return data
    if isinstance(page, Iterable) and not isinstance(page, (str, bytes, bytearray, Mapping)):
        return page
    data = getattr(page, "data", None)
    if isinstance(data, Sequence):
        return data
    raise BatchLifecycleError(f"unsupported provider list response type: {type(page).__name__}")


def _find_remote_file(client: Any, shard: Mapping[str, Any]) -> object | None:
    page = client.files.list(purpose="batch", limit=10_000)
    candidates = []
    for item in _iter_provider_items(page):
        if _field(item, "filename") != shard["remote_file_name"]:
            continue
        if _field(item, "purpose") not in (None, "batch"):
            continue
        if _field(item, "bytes") not in (None, shard["request_utf8_bytes"]):
            continue
        file_id = _required_string(item, "id")
        payload = _content_bytes(client.files.content(file_id))
        if len(payload) == shard["request_utf8_bytes"] and _sha256(payload) == shard["request_content_sha256"]:
            candidates.append(item)
    if len(candidates) > 1:
        ids = sorted(_required_string(candidate, "id") for candidate in candidates)
        raise BatchLifecycleError(
            f"multiple exact provider files match shard {shard['index']}: {ids}; refusing an ambiguous adoption"
        )
    return candidates[0] if candidates else None


def _find_remote_batch(client: Any, manifest: Mapping[str, Any], shard: Mapping[str, Any]) -> object | None:
    expected_metadata = _batch_metadata(manifest, shard)
    matches = []
    page = client.batches.list(limit=100)
    for item in _iter_provider_items(page):
        metadata = _field(item, "metadata")
        if not isinstance(metadata, Mapping):
            metadata = _json_safe(metadata)
        if not isinstance(metadata, Mapping) or metadata.get("ctm_operation_id") != shard["operation_id"]:
            continue
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise BatchLifecycleError(
                    f"provider batch {_field(item, 'id')!r} has operation_id for shard {shard['index']} "
                    f"but mismatched metadata field {key}"
                )
        if _field(item, "endpoint") != BATCH_ENDPOINT:
            raise BatchLifecycleError(f"provider batch {_field(item, 'id')!r} has the wrong endpoint")
        if _field(item, "completion_window") != COMPLETION_WINDOW:
            raise BatchLifecycleError(f"provider batch {_field(item, 'id')!r} has the wrong completion_window")
        matches.append(item)
    if len(matches) > 1:
        ids = sorted(_required_string(candidate, "id") for candidate in matches)
        raise BatchLifecycleError(
            f"multiple provider batches match shard {shard['index']} operation_id: {ids}; refusing duplication"
        )
    return matches[0] if matches else None


def _request_counts(value: object) -> dict[str, int] | None:
    if value is None:
        return None
    counts = {}
    for field in ("total", "completed", "failed"):
        item = _field(value, field)
        if item is None:
            continue
        counts[field] = _as_nonnegative_int(item, label=f"provider request_counts.{field}")
    return counts or None


def _apply_batch_snapshot(
    manifest: dict[str, Any],
    shard: dict[str, Any],
    batch: object,
    *,
    event: str,
) -> None:
    batch_id = _required_string(batch, "id")
    input_file_id = _required_string(batch, "input_file_id")
    status = _required_string(batch, "status")
    if status not in KNOWN_BATCH_STATUSES:
        raise BatchLifecycleError(f"provider batch {batch_id} returned unknown status {status!r}")
    if shard.get("batch_id") not in (None, batch_id):
        raise BatchLifecycleError(
            f"lifecycle shard {shard['index']} is bound to batch {shard.get('batch_id')}, not {batch_id}"
        )
    if shard.get("input_file_id") not in (None, input_file_id):
        raise BatchLifecycleError(
            f"provider batch {batch_id} input file {input_file_id} does not match lifecycle "
            f"file {shard.get('input_file_id')}"
        )
    if _field(batch, "endpoint") != BATCH_ENDPOINT:
        raise BatchLifecycleError(f"provider batch {batch_id} endpoint does not match {BATCH_ENDPOINT}")
    if _field(batch, "completion_window") != COMPLETION_WINDOW:
        raise BatchLifecycleError(f"provider batch {batch_id} completion_window does not match {COMPLETION_WINDOW}")

    old_status = shard.get("status")
    shard["batch_id"] = batch_id
    shard["input_file_id"] = input_file_id
    shard["status"] = status
    shard["status_checked_at"] = _utc_now()
    shard["provider_timestamps"] = {
        field: _field(batch, field) for field in PROVIDER_TIMESTAMP_FIELDS if _field(batch, field) is not None
    }
    shard["request_counts"] = _request_counts(_field(batch, "request_counts"))
    shard["provider_errors"] = _json_safe(_field(batch, "errors"))
    for field in ("output_file_id", "error_file_id"):
        file_id = _field(batch, field)
        if file_id is not None and (not isinstance(file_id, str) or not file_id):
            raise BatchLifecycleError(f"provider batch {batch_id} has invalid {field}")
        existing = shard.get(field)
        if existing not in (None, file_id) and file_id is not None:
            raise BatchLifecycleError(f"provider batch {batch_id} changed {field} from {existing!r} to {file_id!r}")
        if file_id is not None:
            shard[field] = file_id
    if event != "status_checked" or status != old_status:
        _append_event(
            manifest,
            event if event != "status_checked" else "batch_status_changed",
            shard=shard,
            batch_id=batch_id,
            previous_status=old_status,
            status=status,
            request_counts=shard["request_counts"],
        )


def _verify_remote_file_for_shard(client: Any, shard: Mapping[str, Any], file_id: str) -> None:
    payload = _content_bytes(client.files.content(file_id))
    actual_sha256 = _sha256(payload)
    if len(payload) != shard["request_utf8_bytes"] or actual_sha256 != shard["request_content_sha256"]:
        raise BatchLifecycleError(
            f"provider input file {file_id} does not match request shard {shard['index']}: "
            f"expected {shard['request_utf8_bytes']} bytes/{shard['request_content_sha256']}, "
            f"got {len(payload)} bytes/{actual_sha256}"
        )


def _adopt_batch_if_present(
    client: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
    shard: dict[str, Any],
) -> bool:
    remote_batch = _find_remote_batch(client, manifest, shard)
    if remote_batch is None:
        return False
    input_file_id = _required_string(remote_batch, "input_file_id")
    _verify_remote_file_for_shard(client, shard, input_file_id)
    _apply_batch_snapshot(manifest, shard, remote_batch, event="provider_batch_recovered")
    if shard.get("uploaded_at") is None:
        shard["uploaded_at"] = _utc_now()
    if shard.get("batch_created_at") is None:
        shard["batch_created_at"] = _utc_now()
    _save_lifecycle(manifest_path, manifest)
    return True


def _upload_shard(
    client: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
    shard: dict[str, Any],
) -> None:
    recovered = _find_remote_file(client, shard)
    if recovered is not None:
        file_id = _required_string(recovered, "id")
        shard["input_file_id"] = file_id
        shard["uploaded_at"] = _utc_now()
        shard["status"] = "uploaded"
        _append_event(manifest, "provider_file_recovered", shard=shard, input_file_id=file_id)
        _save_lifecycle(manifest_path, manifest)
        return

    shard["upload_started_at"] = _utc_now()
    _append_event(manifest, "file_upload_started", shard=shard)
    _save_lifecycle(manifest_path, manifest)
    request_path = Path(shard["request_path"])
    with request_path.open("rb") as handle:
        response = client.files.create(
            file=(shard["remote_file_name"], handle, "application/jsonl"),
            purpose="batch",
        )
    file_id = _required_string(response, "id")
    provider_bytes = _field(response, "bytes")
    if provider_bytes is not None and provider_bytes != shard["request_utf8_bytes"]:
        raise BatchLifecycleError(
            f"uploaded file {file_id} reports {provider_bytes} bytes, expected {shard['request_utf8_bytes']}"
        )
    purpose = _field(response, "purpose")
    if purpose not in (None, "batch"):
        raise BatchLifecycleError(f"uploaded file {file_id} reports unexpected purpose {purpose!r}")
    shard["input_file_id"] = file_id
    shard["uploaded_at"] = _utc_now()
    shard["status"] = "uploaded"
    _append_event(manifest, "file_uploaded", shard=shard, input_file_id=file_id)
    _save_lifecycle(manifest_path, manifest)


def _create_batch(
    client: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
    shard: dict[str, Any],
) -> None:
    input_file_id = shard.get("input_file_id")
    if not isinstance(input_file_id, str) or not input_file_id:
        raise BatchLifecycleError(f"shard {shard['index']} has no uploaded input file")
    shard["batch_create_started_at"] = _utc_now()
    _append_event(manifest, "batch_create_started", shard=shard, input_file_id=input_file_id)
    _save_lifecycle(manifest_path, manifest)
    response = client.batches.create(
        input_file_id=input_file_id,
        endpoint=BATCH_ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        metadata=_batch_metadata(manifest, shard),
    )
    _apply_batch_snapshot(manifest, shard, response, event="batch_created")
    shard["batch_created_at"] = _utc_now()
    _save_lifecycle(manifest_path, manifest)


def _failed_status_message(manifest: Mapping[str, Any]) -> str | None:
    failed = [
        f"shard {shard['index']} batch {shard.get('batch_id')}={shard.get('status')}"
        for shard in manifest["shards"]
        if shard.get("status") in FAILED_BATCH_STATUSES
    ]
    if not failed:
        return None
    return "terminal Batch failure(s): " + "; ".join(failed)


def _unsubmitted_status_message(manifest: Mapping[str, Any]) -> str | None:
    unsubmitted = [str(shard["index"]) for shard in manifest["shards"] if not shard.get("batch_id")]
    if not unsubmitted:
        return None
    return "lifecycle has unsubmitted shard(s): " + ", ".join(unsubmitted)


def _manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    uploaded = 0
    submitted = 0
    downloaded_outputs = 0
    downloaded_errors = 0
    for shard in manifest["shards"]:
        status = str(shard.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
        uploaded += int(bool(shard.get("input_file_id")))
        submitted += int(bool(shard.get("batch_id")))
        downloads = shard.get("downloads")
        if isinstance(downloads, Mapping):
            downloaded_outputs += int("output" in downloads)
            downloaded_errors += int("error" in downloads)
    return {
        "manifest_path": manifest["manifest_path"],
        "request_count": manifest["request_count"],
        "shard_count": manifest["shard_count"],
        "uploaded_shards": uploaded,
        "submitted_batches": submitted,
        "statuses": statuses,
        "downloaded_output_files": downloaded_outputs,
        "downloaded_error_files": downloaded_errors,
    }


def submit_batches(
    *,
    request_manifest_path: str | Path,
    lifecycle_manifest_path: str | Path,
    yes: bool,
    client: Any | None = None,
    stdout: TextIO | None = None,
) -> dict[str, Any]:
    """Submit one paid Batch per verified shard, resuming recorded state."""

    output = stdout if stdout is not None else sys.stdout
    plan = load_request_plan(request_manifest_path)
    manifest_path = Path(lifecycle_manifest_path).expanduser().resolve()
    print("\n".join(_submission_plan_lines(plan, manifest_path)), file=output, flush=True)
    if not yes:
        raise BatchLifecycleError("paid submission requires --yes after reviewing the complete plan above")

    with _manifest_lock(manifest_path):
        if manifest_path.exists():
            manifest = _load_lifecycle(manifest_path)
            _assert_lifecycle_matches_plan(manifest, plan)
        else:
            manifest = _new_lifecycle_manifest(plan, manifest_path)
            manifest["approval"] = {
                "confirmed": True,
                "confirmed_at": _utc_now(),
                "confirmation_flag": "--yes",
            }
            _save_lifecycle(manifest_path, manifest)
        if manifest.get("approval") is None:
            manifest["approval"] = {
                "confirmed": True,
                "confirmed_at": _utc_now(),
                "confirmation_flag": "--yes",
            }
            _save_lifecycle(manifest_path, manifest)

        if all(shard.get("batch_id") for shard in manifest["shards"]):
            return _manifest_summary(manifest)
        resolved_client = client if client is not None else _create_openai_client()
        for shard in manifest["shards"]:
            if shard.get("batch_id"):
                continue
            if _adopt_batch_if_present(resolved_client, manifest_path, manifest, shard):
                failure = _failed_status_message(manifest)
                if failure:
                    raise BatchLifecycleError(failure)
                continue
            if not shard.get("input_file_id"):
                _upload_shard(resolved_client, manifest_path, manifest, shard)
            if _adopt_batch_if_present(resolved_client, manifest_path, manifest, shard):
                failure = _failed_status_message(manifest)
                if failure:
                    raise BatchLifecycleError(failure)
                continue
            _create_batch(resolved_client, manifest_path, manifest, shard)
            failure = _failed_status_message(manifest)
            if failure:
                raise BatchLifecycleError(failure)
        return _manifest_summary(manifest)


def _refresh_status_once(
    client: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    for shard in manifest["shards"]:
        batch_id = shard.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            continue
        response = client.batches.retrieve(batch_id)
        _apply_batch_snapshot(manifest, shard, response, event="status_checked")
        _save_lifecycle(manifest_path, manifest)
    return _manifest_summary(manifest)


def status_batches(
    *,
    lifecycle_manifest_path: str | Path,
    wait: bool = False,
    poll_interval_seconds: float = 60.0,
    client: Any | None = None,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Refresh provider status once, or poll until every submitted Batch is terminal."""

    if poll_interval_seconds <= 0:
        raise BatchLifecycleError("poll_interval_seconds must be greater than zero")
    manifest_path = Path(lifecycle_manifest_path).expanduser().resolve()
    with _manifest_lock(manifest_path):
        manifest = _load_lifecycle(manifest_path)
        resolved_client = client if client is not None else _create_openai_client()
        while True:
            summary = _refresh_status_once(resolved_client, manifest_path, manifest)
            failure = _failed_status_message(manifest)
            if failure:
                raise BatchLifecycleError(failure)
            unsubmitted = _unsubmitted_status_message(manifest)
            if wait and unsubmitted:
                raise BatchLifecycleError(f"cannot wait for completion: {unsubmitted}")
            all_terminal = all(
                bool(shard.get("batch_id")) and shard.get("status") in TERMINAL_BATCH_STATUSES
                for shard in manifest["shards"]
            )
            if not wait or all_terminal:
                return summary
            sleep(poll_interval_seconds)


def _verify_recorded_download(record: Mapping[str, Any], *, target: Path, file_id: str) -> bool:
    if record.get("file_id") != file_id:
        raise BatchLifecycleError(
            f"download record for {target} is bound to file {record.get('file_id')!r}, not {file_id!r}"
        )
    if record.get("path") != str(target):
        raise BatchLifecycleError(
            f"download was previously bound to {record.get('path')!r}; refusing a new destination {str(target)!r}"
        )
    if not target.is_file():
        return False
    payload = target.read_bytes()
    actual_sha256 = _sha256(payload)
    if record.get("utf8_bytes") != len(payload) or record.get("content_sha256") != actual_sha256:
        raise BatchLifecycleError(
            f"existing downloaded file {target} no longer matches its manifest; refusing to overwrite it"
        )
    return True


def _download_provider_file(
    client: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
    shard: dict[str, Any],
    *,
    kind: str,
    file_id: str,
    target: Path,
) -> None:
    downloads = shard["downloads"]
    existing_record = downloads.get(kind)
    if isinstance(existing_record, Mapping) and _verify_recorded_download(
        existing_record, target=target, file_id=file_id
    ):
        return

    payload = _content_bytes(client.files.content(file_id))
    payload_sha256 = _sha256(payload)
    if isinstance(existing_record, Mapping):
        if existing_record.get("utf8_bytes") != len(payload) or existing_record.get("content_sha256") != payload_sha256:
            raise BatchLifecycleError(
                f"provider file {file_id} content changed since its recorded download; refusing to write {target}"
            )
    if target.exists():
        existing_payload = target.read_bytes()
        if existing_payload != payload:
            raise BatchLifecycleError(f"refusing to overwrite mismatched existing download: {target}")
    else:
        write_atomic_bytes(target, payload)
    verified_payload = target.read_bytes()
    if len(verified_payload) != len(payload) or _sha256(verified_payload) != payload_sha256:
        raise BatchLifecycleError(f"download verification failed after writing {target}")
    downloads[kind] = {
        "file_id": file_id,
        "path": str(target),
        "utf8_bytes": len(payload),
        "content_sha256": payload_sha256,
        "downloaded_at": _utc_now(),
    }
    _append_event(
        manifest,
        f"{kind}_file_downloaded",
        shard=shard,
        file_id=file_id,
        path=str(target),
        utf8_bytes=len(payload),
        content_sha256=payload_sha256,
    )
    _save_lifecycle(manifest_path, manifest)


def download_batches(
    *,
    lifecycle_manifest_path: str | Path,
    output_dir: str | Path,
    client: Any | None = None,
) -> dict[str, Any]:
    """Refresh statuses and download all available terminal output/error files."""

    manifest_path = Path(lifecycle_manifest_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    with _manifest_lock(manifest_path):
        manifest = _load_lifecycle(manifest_path)
        resolved_client = client if client is not None else _create_openai_client()
        _refresh_status_once(resolved_client, manifest_path, manifest)
        destination.mkdir(parents=True, exist_ok=True)
        for shard in manifest["shards"]:
            if shard.get("status") not in TERMINAL_BATCH_STATUSES:
                continue
            for kind, field in (("output", "output_file_id"), ("error", "error_file_id")):
                file_id = shard.get(field)
                if not isinstance(file_id, str) or not file_id:
                    continue
                target = destination / f"batch-{kind}.part-{shard['index']:05d}.jsonl"
                _download_provider_file(
                    resolved_client,
                    manifest_path,
                    manifest,
                    shard,
                    kind=kind,
                    file_id=file_id,
                    target=target,
                )
        failure = _failed_status_message(manifest)
        if failure:
            raise BatchLifecycleError(failure)
        unsubmitted = _unsubmitted_status_message(manifest)
        if unsubmitted:
            raise BatchLifecycleError(f"download incomplete: {unsubmitted}")
        return _manifest_summary(manifest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="Explicitly upload shards and create paid OpenAI Batches.")
    submit.add_argument("--request-manifest", type=Path, required=True)
    submit.add_argument("--manifest", type=Path, required=True, help="Atomic lifecycle manifest to create/resume.")
    submit.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the printed exact shard/protocol/request plan and authorize paid submission.",
    )

    status = subparsers.add_parser("status", help="Read provider Batch status and update the manifest.")
    status.add_argument("--manifest", type=Path, required=True)
    status.add_argument("--wait", action="store_true", help="Poll until every shard has a terminal Batch.")
    status.add_argument("--poll-interval-seconds", type=float, default=60.0)

    download = subparsers.add_parser(
        "download", help="Refresh terminal status and download verified output/error files."
    )
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "submit":
            summary = submit_batches(
                request_manifest_path=args.request_manifest,
                lifecycle_manifest_path=args.manifest,
                yes=args.yes,
            )
        elif args.command == "status":
            summary = status_batches(
                lifecycle_manifest_path=args.manifest,
                wait=args.wait,
                poll_interval_seconds=args.poll_interval_seconds,
            )
        else:
            summary = download_batches(
                lifecycle_manifest_path=args.manifest,
                output_dir=args.output_dir,
            )
    except (BatchLifecycleError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
