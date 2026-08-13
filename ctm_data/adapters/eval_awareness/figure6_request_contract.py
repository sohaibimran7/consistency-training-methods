#!/usr/bin/env python3
"""Run the minimal Qwen MO midtrained request-contract diagnostic.

The diagnostic deliberately changes only the target request contract.  It uses
the pinned Figure 6 artifact, checkpoint, and natural system prompt, selects
replicate 1 of the 100 safety-baseline tasks, and sends the target endpoint the
same small OpenAI-compatible body shape recorded in Igor's Inspect run:
``model``, ``messages``, and ``reasoning_effort='medium'``.  Temperature,
top-p, output-token ceiling, and seed are intentionally absent.

Every real run is bound to a separately reviewed dry-run hash.  Results are an
append-only generation log, suitable for the existing post-hoc judge adapter.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import stat
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator
from urllib.parse import urlparse

from ctm.artifacts import write_atomic_bytes
from ctm_data.adapters.eval_awareness.figure6_generate import (
    _append_durable,
    _close_client,
    _create_openai_client,
    _endpoint_model_check,
    _parse_completion,
    _record_id,
    _record_provenance,
    _safe_error,
    make_generation_key,
    read_generation_records,
)
from ctm_data.adapters.eval_awareness.figure6_judge import validate_generation
from ctm_data.adapters.eval_awareness.figure6_materialize import load_figure6_artifact
from ctm_data.adapters.eval_awareness.figure6_spec import (
    DATASET_ID,
    DATASET_REVISION,
    UPSTREAM_CODE_REVISION,
    ModelSpec,
    PromptSpec,
    get_model_spec,
    load_verified_model_prompt,
)

MODEL_KEY = "qwen_mo_mid"
REASONING_EFFORT = "medium"
WIRE_SMOKE_COUNT = 20
BASELINE_COUNT = 100
SCOPES = {"wire-smoke": WIRE_SMOKE_COUNT, "baseline": BASELINE_COUNT}
DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_BASE_SECONDS = 0.0
EXPECTED_VLLM_VERSION = "0.26.0"
EXPECTED_MAX_MODEL_LEN = 8192
EXPECTED_GENERATION_CONFIG = "auto"
RUN_PROVENANCE_SCHEMA = "ctm.eval_awareness.figure6_request_contract_run.v1"
MANIFEST_SCHEMA = "ctm.eval_awareness.figure6_request_contract_manifest.v1"
SERVER_ATTESTATION_SCHEMA = "ctm.eval_awareness.figure6_request_contract_server.v1"
DISPATCH_INTENT_SCHEMA = "ctm.eval_awareness.figure6_request_contract_dispatch_intent.v1"
PROTOCOL_ID = "igor-target-openai-chat-medium-omitted-sampling-v1"
OMITTED_REQUEST_FIELDS = ("temperature", "top_p", "max_tokens", "seed")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = (REPOSITORY_ROOT / "artifacts").resolve()


@dataclass(frozen=True, slots=True)
class ServerRuntimeProfile:
    """One reviewed, exact local-serving runtime for a diagnostic round."""

    key: str
    vllm_version: str
    tensor_parallel_size: int
    dtype: str
    max_model_len: int
    reasoning_parser: str
    generation_config: str

    def attestation_fields(self) -> dict[str, Any]:
        return {
            "vllm_version": self.vllm_version,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "reasoning_parser": self.reasoning_parser,
            "generation_config": self.generation_config,
        }


# The request-only diagnostic remains fixed to this profile.  The second
# profile is opt-in and exists solely for the separate runtime ablation.
REQUEST_ONLY_RUNTIME_PROFILE = ServerRuntimeProfile(
    key="vllm-0.26.0-tp1-bf16-ctx8192-qwen3-auto",
    vllm_version=EXPECTED_VLLM_VERSION,
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_model_len=EXPECTED_MAX_MODEL_LEN,
    reasoning_parser="qwen3",
    generation_config=EXPECTED_GENERATION_CONFIG,
)
RUNTIME_ABLATION_V023_TP4_PROFILE = ServerRuntimeProfile(
    key="vllm-0.23.0-tp4-bf16-ctx8192-qwen3-auto",
    vllm_version="0.23.0",
    tensor_parallel_size=4,
    dtype="bfloat16",
    max_model_len=8192,
    reasoning_parser="qwen3",
    generation_config="auto",
)
SERVER_RUNTIME_PROFILES: dict[str, ServerRuntimeProfile] = {
    profile.key: profile
    for profile in (
        REQUEST_ONLY_RUNTIME_PROFILE,
        RUNTIME_ABLATION_V023_TP4_PROFILE,
    )
}


def get_server_runtime_profile(key: str) -> ServerRuntimeProfile:
    """Return a registered exact runtime profile, rejecting unreviewed stacks."""

    try:
        return SERVER_RUNTIME_PROFILES[key]
    except KeyError as exc:
        raise RequestContractError(f"unknown server runtime profile {key!r}") from exc


class RequestContractError(ValueError):
    """Raised when a diagnostic lifecycle is inconsistent or unaudited."""


class _PreDispatchIntentError(RequestContractError):
    """Raised when no network request was sent because its intent was not durable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def request_contract_manifest_path(output_path: str | Path) -> Path:
    """Return the paid-approval and plan-history sidecar for ``output_path``."""

    target = Path(output_path)
    return target.with_suffix(target.suffix + ".request-contract-manifest.json")


def request_contract_dispatch_path(output_path: str | Path) -> Path:
    """Return the write-ahead dispatch-intent journal for ``output_path``."""

    target = Path(output_path)
    return target.with_suffix(target.suffix + ".request-contract-dispatch.jsonl")


def require_artifact_path(path: str | Path) -> Path:
    """Confine raw target data and lifecycle sidecars to the ignored artifact root."""

    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(ARTIFACT_ROOT)
    except ValueError as exc:
        raise RequestContractError(f"request-contract outputs must stay under {ARTIFACT_ROOT}") from exc
    if not relative.parts:
        raise RequestContractError("request-contract output must be a file below the artifact root")
    return resolved


def load_server_attestation(
    path: str | Path,
    model: ModelSpec,
    *,
    runtime_profile: ServerRuntimeProfile | None = None,
) -> tuple[dict[str, Any], str]:
    """Load a sanitized attestation for the selected exact serving stack.

    Leaving ``runtime_profile`` unset deliberately preserves the v0.26/TP1
    request-only gate.  Runtime experiments must pass a registered profile
    explicitly, making a stack change visible in their reviewed plan.
    """

    target = require_artifact_path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RequestContractError(f"missing server attestation: {target}") from exc
    except json.JSONDecodeError as exc:
        raise RequestContractError(f"invalid server attestation {target}: {exc.msg}") from exc
    if not isinstance(value, dict) or value.get("schema") != SERVER_ATTESTATION_SCHEMA:
        raise RequestContractError("server attestation has an unsupported schema")
    expected = {"model_id": model.model_id, "model_revision": model.revision}
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise RequestContractError(f"server attestation {field} does not match the pinned model")
    required_fields = {
        "vllm_version": str,
        "tensor_parallel_size": int,
        "dtype": str,
        "max_model_len": int,
        "reasoning_parser": str,
        "generation_config": str,
        "launch_command_sha256": str,
    }
    allowed_fields = {"schema", *expected, *required_fields}
    unexpected_fields = sorted(set(value) - allowed_fields)
    if unexpected_fields:
        raise RequestContractError(
            "server attestation contains unsupported fields; keep it sanitized and content-free: "
            + ", ".join(unexpected_fields)
        )
    for field, field_type in required_fields.items():
        item = value.get(field)
        if not isinstance(item, field_type) or isinstance(item, bool) or item in {"", 0}:
            raise RequestContractError(f"server attestation has invalid {field}")
    command_hash = value["launch_command_sha256"]
    if len(command_hash) != 64 or any(character not in "0123456789abcdef" for character in command_hash):
        raise RequestContractError("server attestation launch_command_sha256 must be lowercase SHA-256")
    fixed_stack = (
        runtime_profile.attestation_fields()
        if runtime_profile is not None
        else {
            "vllm_version": EXPECTED_VLLM_VERSION,
            "tensor_parallel_size": model.tensor_parallel_size,
            "dtype": model.dtype,
            "max_model_len": EXPECTED_MAX_MODEL_LEN,
            "reasoning_parser": model.reasoning_parser,
            "generation_config": EXPECTED_GENERATION_CONFIG,
        }
    )
    profile_label = runtime_profile.key if runtime_profile is not None else "the request-only gate"
    for field, expected_value in fixed_stack.items():
        if value[field] != expected_value:
            raise RequestContractError(
                f"server attestation {field} must remain fixed at {expected_value!r} for {profile_label}"
            )
    canonical = dict(value)
    return canonical, _sha256_json(canonical)


def _sanitized_endpoint(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("base_url must be a non-empty string")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("request-contract target endpoint must be loopback-only")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain credentials, a query, or a fragment")
    if parsed.path.rstrip("/").split("/")[-1] != "v1":
        raise ValueError("OpenAI-compatible base_url must end in /v1")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url has an invalid port") from exc
    authority = parsed.hostname if port is None else f"{parsed.hostname}:{port}"
    return f"{parsed.scheme}://{authority}{parsed.path.rstrip('/')}/chat/completions"


def _baseline_rows(artifact_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in artifact_rows
        if row.get("valence") == "safety" and row.get("condition") == "baseline"
    ]
    rows.sort(key=lambda row: (str(row.get("pair_id")), str(row.get("condition_id"))))
    if len(rows) != BASELINE_COUNT:
        raise RequestContractError(
            f"verified Figure 6 artifact must contain exactly {BASELINE_COUNT} safety-baseline rows; got {len(rows)}"
        )
    if len({row.get("pair_id") for row in rows}) != BASELINE_COUNT:
        raise RequestContractError("safety-baseline cohort must contain exactly one row per task pair")
    if len({row.get("condition_id") for row in rows}) != BASELINE_COUNT:
        raise RequestContractError("safety-baseline cohort contains duplicate condition IDs")
    return rows


def select_request_contract_rows(
    artifact_rows: Sequence[Mapping[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    """Select the deterministic 20- or 100-row safety-baseline cohort."""

    try:
        count = SCOPES[scope]
    except KeyError as exc:
        raise ValueError(f"scope must be one of {sorted(SCOPES)}") from exc
    return _baseline_rows(artifact_rows)[:count]


def build_target_request(*, model_id: str, system_prompt: str, task_prompt: str) -> dict[str, Any]:
    """Return the exact Igor-shaped target request body.

    Keep this as a literal allow-list: adding a convenience default here would
    change the experiment being tested.
    """

    if not all(isinstance(value, str) and value for value in (model_id, system_prompt, task_prompt)):
        raise TypeError("model_id, system_prompt, and task_prompt must be non-empty strings")
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ],
        "reasoning_effort": REASONING_EFFORT,
    }


def _run_provenance(
    *,
    model: ModelSpec,
    prompt: PromptSpec,
    artifact_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "provenance_schema": RUN_PROVENANCE_SCHEMA,
        "schema_version": 1,
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
        "replicates": 1,
        "request_protocol_id": PROTOCOL_ID,
        "reasoning_effort": REASONING_EFFORT,
        "omitted_request_fields": list(OMITTED_REQUEST_FIELDS),
    }
    return {**core, "provenance_sha256": _sha256_json(core)}


def _plan_document(
    *,
    model: ModelSpec,
    prompt: PromptSpec,
    artifact_manifest: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    scope: str,
    endpoint: str,
    max_concurrency: int,
    max_retries: int,
    retry_base_seconds: float,
    server_attestation: Mapping[str, Any],
    server_attestation_sha256: str,
) -> dict[str, Any]:
    condition_ids = [str(row["condition_id"]) for row in selected_rows]
    return {
        "schema": "ctm.eval_awareness.figure6_request_contract_plan.v1",
        "scope": scope,
        "selected_count": len(selected_rows),
        "selected_condition_ids_sha256": _ids_sha256(condition_ids),
        "artifact_sha256": artifact_manifest["content_sha256"],
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "model_key": model.key,
        "model_id": model.model_id,
        "model_revision": model.revision,
        "system_prompt_key": prompt.key,
        "system_prompt_sha256": prompt.sha256,
        "endpoint": endpoint,
        "server_attestation": dict(server_attestation),
        "server_attestation_sha256": server_attestation_sha256,
        "request": {
            "keys": ["messages", "model", "reasoning_effort"],
            "message_roles": ["system", "user"],
            "reasoning_effort": REASONING_EFFORT,
            "omitted_fields": list(OMITTED_REQUEST_FIELDS),
        },
        "replicates": 1,
        "max_concurrency": max_concurrency,
        "max_retries_per_generation": max_retries,
        "retry_base_seconds": retry_base_seconds,
        "resume_policy": "append-only; one request per generation; any uncertainty quarantines the lifecycle",
        "scheduler": "rolling bounded; stop refill on first uncertainty; drain in-flight work",
        "dispatch_journal_schema": DISPATCH_INTENT_SCHEMA,
        "dispatch_policy": "fsync intent before network send; unresolved intent blocks every resume",
    }


def _core_plan_document(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key not in {"scope", "selected_count", "selected_condition_ids_sha256"}}


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RequestContractError(f"invalid request-contract manifest {path}: {exc.msg}") from exc
    if not isinstance(value, dict) or value.get("schema") != MANIFEST_SCHEMA:
        raise RequestContractError(f"request-contract manifest {path} has an unsupported schema")
    plans = value.get("plan_history")
    approvals = value.get("approvals")
    if not isinstance(plans, list) or not plans or not isinstance(approvals, list):
        raise RequestContractError(f"request-contract manifest {path} has an invalid history")
    for index, plan_entry in enumerate(plans, start=1):
        if not isinstance(plan_entry, Mapping) or not isinstance(plan_entry.get("document"), Mapping):
            raise RequestContractError(f"request-contract manifest plan {index} is invalid")
        document = dict(plan_entry["document"])
        scope = document.get("scope")
        if (
            document.get("schema") != "ctm.eval_awareness.figure6_request_contract_plan.v1"
            or not isinstance(scope, str)
            or scope not in SCOPES
            or document.get("selected_count") != SCOPES[scope]
        ):
            raise RequestContractError(f"request-contract manifest plan {index} has an invalid scope")
        if plan_entry.get("plan_sha256") != _sha256_json(document):
            raise RequestContractError(f"request-contract manifest plan {index} has a bad hash")
        if plan_entry.get("core_plan_sha256") != _sha256_json(_core_plan_document(document)):
            raise RequestContractError(f"request-contract manifest plan {index} has a bad core hash")
    known_hashes = {entry["plan_sha256"] for entry in plans}
    for index, approval in enumerate(approvals, start=1):
        if (
            not isinstance(approval, Mapping)
            or approval.get("confirmation") != "--yes"
            or approval.get("plan_sha256") not in known_hashes
            or approval.get("reviewed_plan_sha256") != approval.get("plan_sha256")
        ):
            raise RequestContractError(f"request-contract manifest approval {index} is invalid")
    return value


def _approved_plan_hashes(manifest: Mapping[str, Any] | None) -> set[str]:
    if manifest is None:
        return set()
    return {str(approval["plan_sha256"]) for approval in manifest["approvals"]}


def _save_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    value = {**dict(manifest), "updated_at": _utc_now()}
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_atomic_bytes(path, payload)


def _dispatch_intent(*, generation_key: str, plan_sha256: str) -> dict[str, Any]:
    identity = {
        "generation_key": generation_key,
        "plan_sha256": plan_sha256,
        "protocol_id": PROTOCOL_ID,
    }
    return {
        "schema": DISPATCH_INTENT_SCHEMA,
        **identity,
        "intent_id": _sha256_json(identity),
        "created_at": _utc_now(),
    }


def _read_dispatch_intents(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if not path.is_file():
        raise RequestContractError(f"dispatch-intent journal is not a file: {path}")
    intents: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RequestContractError(
                    f"dispatch-intent journal {path}:{line_number} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise RequestContractError(
                    f"dispatch-intent journal {path}:{line_number} must contain an object"
                )
            intents.append(value)
    return intents


def _ensure_dispatch_journal(path: Path) -> None:
    """Create and directory-fsync the empty write-ahead journal before use."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory = path.parent
        while True:
            directory_descriptor = os.open(directory, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            if directory == ARTIFACT_ROOT:
                break
            directory = directory.parent
        metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RequestContractError(f"dispatch-intent journal must be a regular file: {path}")


def _validate_dispatch_intents(
    intents: Sequence[Mapping[str, Any]],
    *,
    required_keys: set[str],
    allowed_plan_keys: Mapping[str, set[str]],
) -> dict[str, dict[str, Any]]:
    expected_fields = {
        "schema",
        "generation_key",
        "plan_sha256",
        "protocol_id",
        "intent_id",
        "created_at",
    }
    by_key: dict[str, dict[str, Any]] = {}
    for index, value_like in enumerate(intents, start=1):
        value = dict(value_like)
        if set(value) != expected_fields or value.get("schema") != DISPATCH_INTENT_SCHEMA:
            raise RequestContractError(f"dispatch intent {index} has an invalid schema or fields")
        generation_key = value.get("generation_key")
        plan_sha256 = value.get("plan_sha256")
        if not isinstance(generation_key, str) or generation_key not in required_keys:
            raise RequestContractError(f"dispatch intent {index} is outside the selected scope")
        if not isinstance(plan_sha256, str) or plan_sha256 not in allowed_plan_keys:
            raise RequestContractError(f"dispatch intent {index} has an unapproved plan hash")
        if generation_key not in allowed_plan_keys[plan_sha256]:
            raise RequestContractError(f"dispatch intent {index} is outside its reviewed plan scope")
        identity = {
            "generation_key": generation_key,
            "plan_sha256": plan_sha256,
            "protocol_id": PROTOCOL_ID,
        }
        if value.get("protocol_id") != PROTOCOL_ID or value.get("intent_id") != _sha256_json(identity):
            raise RequestContractError(f"dispatch intent {index} has an invalid identity")
        if not isinstance(value.get("created_at"), str) or not value["created_at"]:
            raise RequestContractError(f"dispatch intent {index} has an invalid timestamp")
        if generation_key in by_key:
            raise RequestContractError(f"dispatch intent {index} duplicates {generation_key}")
        by_key[generation_key] = value
    return by_key


@contextmanager
def _run_lock(output_path: Path) -> Iterator[BinaryIO]:
    lock_path = output_path.with_suffix(output_path.suffix + ".request-contract.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RequestContractError(f"another process is using request-contract output {output_path}") from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _expected_record_base(
    *,
    row: Mapping[str, Any],
    model: ModelSpec,
    prompt: PromptSpec,
    artifact_manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return _record_provenance(
        row=row,
        model=model,
        prompt=prompt,
        artifact_manifest=artifact_manifest,
        replicate=1,
        temperature=None,  # honest record of an omitted request field
        max_tokens=None,  # honest record of an omitted request field
        generation_provenance=provenance,
    )


def _validate_history(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    model: ModelSpec,
    prompt: PromptSpec,
    artifact_manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    allowed_plan_keys: Mapping[str, set[str]],
) -> dict[str, list[dict[str, Any]]]:
    by_condition = {str(row["condition_id"]): row for row in selected_rows}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    record_ids: set[str] = set()
    for index, row_like in enumerate(records, start=1):
        row = dict(row_like)
        condition_id = row.get("condition_id")
        if not isinstance(condition_id, str) or condition_id not in by_condition:
            raise RequestContractError(f"request-contract output row {index} is outside the selected scope")
        expected = _expected_record_base(
            row=by_condition[condition_id],
            model=model,
            prompt=prompt,
            artifact_manifest=artifact_manifest,
            provenance=provenance,
        )
        for field, value in expected.items():
            if row.get(field) != value:
                raise RequestContractError(f"request-contract output row {index} has incompatible {field}")
        if row.get("diagnostic_protocol_id") != PROTOCOL_ID:
            raise RequestContractError(f"request-contract output row {index} has the wrong protocol")
        logical_key = expected["generation_key"]
        diagnostic_plan_sha256 = row.get("diagnostic_plan_sha256")
        if (
            not isinstance(diagnostic_plan_sha256, str)
            or diagnostic_plan_sha256 not in allowed_plan_keys
        ):
            raise RequestContractError(f"request-contract output row {index} has an unapproved plan hash")
        if logical_key not in allowed_plan_keys[diagnostic_plan_sha256]:
            raise RequestContractError(
                f"request-contract output row {index} is outside its reviewed plan scope"
            )
        resume_attempt = row.get("resume_attempt")
        if not isinstance(resume_attempt, int) or isinstance(resume_attempt, bool) or resume_attempt < 1:
            raise RequestContractError(f"request-contract output row {index} has an invalid resume attempt")
        if row.get("record_id") != _record_id(logical_key, resume_attempt):
            raise RequestContractError(f"request-contract output row {index} has an invalid record ID")
        if row["record_id"] in record_ids:
            raise RequestContractError(f"request-contract output row {index} duplicates a record ID")
        record_ids.add(row["record_id"])
        if row.get("status") not in {"success", "uncertain"}:
            raise RequestContractError(f"request-contract output row {index} has an invalid status")
        if row["status"] == "success":
            validate_generation(row, index=index)
            if row.get("error") is not None:
                raise RequestContractError(f"successful request-contract output row {index} has an error")
        elif not isinstance(row.get("error"), str) or not row["error"]:
            raise RequestContractError(f"uncertain request-contract output row {index} lacks an error")
        histories[logical_key].append(row)
    for logical_key, history in histories.items():
        if [row["resume_attempt"] for row in history] != list(range(1, len(history) + 1)):
            raise RequestContractError(f"request-contract history is non-contiguous for {logical_key}")
        if len(history) != 1:
            raise RequestContractError(f"request-contract history must have exactly one terminal row for {logical_key}")
    return dict(histories)


async def _generate_one(
    *,
    client: Any,
    row: Mapping[str, Any],
    model: ModelSpec,
    prompt: PromptSpec,
    system_prompt: str,
    artifact_manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    plan_sha256: str,
    resume_attempt: int,
    api_key: str | None,
) -> dict[str, Any]:
    base = _expected_record_base(
        row=row,
        model=model,
        prompt=prompt,
        artifact_manifest=artifact_manifest,
        provenance=provenance,
    )
    logical_key = base["generation_key"]
    started_at = _utc_now()
    started_monotonic = time.monotonic()

    def timing() -> dict[str, Any]:
        return {
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_seconds": max(0.0, time.monotonic() - started_monotonic),
        }

    request = build_target_request(
        model_id=model.model_id,
        system_prompt=system_prompt,
        task_prompt=str(row["prompt"]),
    )
    try:
        completion = await client.chat.completions.create(**request)
        parsed = _parse_completion(completion)
        if parsed.get("response_model") not in {model.model_id, model.key}:
            raise ValueError(f"unexpected target response model: {parsed.get('response_model')!r}")
        if not parsed["trace_present"]:
            raise ValueError("missing reasoning trace: response had neither native reasoning nor supported tags")
        return {
            **base,
            "record_id": _record_id(logical_key, resume_attempt),
            "resume_attempt": resume_attempt,
            "diagnostic_protocol_id": PROTOCOL_ID,
            "diagnostic_plan_sha256": plan_sha256,
            "status": "success",
            "error": None,
            "attempts": 1,
            **timing(),
            **parsed,
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {
            **base,
            "record_id": _record_id(logical_key, resume_attempt),
            "resume_attempt": resume_attempt,
            "diagnostic_protocol_id": PROTOCOL_ID,
            "diagnostic_plan_sha256": plan_sha256,
            "status": "uncertain",
            "error": _safe_error(exc, [api_key]),
            "attempts": 1,
            **timing(),
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


async def run_request_contract_gate(
    artifact_path: str | Path,
    output_path: str | Path,
    *,
    prompt_path: str | Path,
    server_attestation_path: str | Path,
    scope: str = "wire-smoke",
    base_url: str = "http://127.0.0.1:8000/v1",
    client: Any | None = None,
    api_key: str | None = None,
    api_key_env: str = "FIGURE6_LOCAL_ENDPOINT_TOKEN",
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
    expected_plan_sha256: str | None = None,
    confirm_requests: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Plan or execute the append-only request-contract diagnostic."""

    if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1:
        raise ValueError("max_concurrency must be an integer >= 1")
    if max_retries != 1:
        raise ValueError("request-contract generations permit exactly one attempt; retries are prohibited")
    if retry_base_seconds != 0:
        raise ValueError("request-contract generations prohibit retry backoff because retries are prohibited")
    if api_key_env == "OPENAI_API_KEY":
        raise ValueError("request-contract target generation must not use an OpenAI API credential")
    if dry_run and confirm_requests:
        raise ValueError("dry_run cannot include request confirmation")

    endpoint = _sanitized_endpoint(base_url)
    model = get_model_spec(MODEL_KEY)
    system_prompt, prompt = load_verified_model_prompt(MODEL_KEY, prompt_path)
    server_attestation, server_attestation_sha256 = load_server_attestation(server_attestation_path, model)
    artifact_rows, artifact_manifest = load_figure6_artifact(artifact_path)
    selected_rows = select_request_contract_rows(artifact_rows, scope)
    smoke_rows = select_request_contract_rows(artifact_rows, "wire-smoke")
    baseline_rows = select_request_contract_rows(artifact_rows, "baseline")
    provenance = _run_provenance(model=model, prompt=prompt, artifact_manifest=artifact_manifest)
    plan = _plan_document(
        model=model,
        prompt=prompt,
        artifact_manifest=artifact_manifest,
        selected_rows=selected_rows,
        scope=scope,
        endpoint=endpoint,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        retry_base_seconds=float(retry_base_seconds),
        server_attestation=server_attestation,
        server_attestation_sha256=server_attestation_sha256,
    )
    plan_sha256 = _sha256_json(plan)
    core_plan_sha256 = _sha256_json(_core_plan_document(plan))
    smoke_plan = _plan_document(
        model=model,
        prompt=prompt,
        artifact_manifest=artifact_manifest,
        selected_rows=smoke_rows,
        scope="wire-smoke",
        endpoint=endpoint,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        retry_base_seconds=float(retry_base_seconds),
        server_attestation=server_attestation,
        server_attestation_sha256=server_attestation_sha256,
    )
    smoke_plan_sha256 = _sha256_json(smoke_plan)
    baseline_plan = _plan_document(
        model=model,
        prompt=prompt,
        artifact_manifest=artifact_manifest,
        selected_rows=baseline_rows,
        scope="baseline",
        endpoint=endpoint,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        retry_base_seconds=float(retry_base_seconds),
        server_attestation=server_attestation,
        server_attestation_sha256=server_attestation_sha256,
    )
    expected_plan_documents = {"wire-smoke": smoke_plan, "baseline": baseline_plan}
    scope_rows = {"wire-smoke": smoke_rows, "baseline": baseline_rows}
    if expected_plan_sha256 is not None and expected_plan_sha256 != plan_sha256:
        raise RequestContractError(
            f"deterministic plan hash mismatch: expected {expected_plan_sha256}, calculated {plan_sha256}"
        )

    output = require_artifact_path(output_path)
    attestation_target = require_artifact_path(server_attestation_path)
    dispatch_path = require_artifact_path(request_contract_dispatch_path(output))
    if output in {attestation_target, Path(artifact_path).resolve(), Path(prompt_path).resolve()}:
        raise RequestContractError("request-contract output must not overwrite an input or server attestation")
    manifest_path = request_contract_manifest_path(output)
    with _run_lock(output):
        _ensure_dispatch_journal(dispatch_path)
        manifest = _read_manifest(manifest_path)
        existing_records = read_generation_records(output)
        if existing_records and manifest is None:
            raise RequestContractError(f"existing output has no request-contract manifest: {manifest_path}")
        if manifest is None:
            manifest: dict[str, Any] = {
                "schema": MANIFEST_SCHEMA,
                "created_at": _utc_now(),
                "plan_history": [],
                "approvals": [],
                "blocked_generation_keys": [],
                "dispatched_generation_keys": [],
                "dispatched_generation_keys_sha256": _ids_sha256([]),
                "status": "planned",
            }

        blocked = manifest.get("blocked_generation_keys", [])
        if (
            not isinstance(blocked, list)
            or any(not isinstance(key, str) or not key for key in blocked)
            or len(blocked) != len(set(blocked))
        ):
            raise RequestContractError("request-contract manifest has invalid blocked generation keys")
        if blocked:
            raise RequestContractError(
                "request-contract lifecycle has uncertain target outcomes; manual reconciliation is required"
            )

        plan_history = manifest["plan_history"]
        prior_core_hashes = {entry["core_plan_sha256"] for entry in plan_history}
        if prior_core_hashes and prior_core_hashes != {core_plan_sha256}:
            raise RequestContractError("existing request-contract output is bound to a different core plan")
        for entry in plan_history:
            entry_scope = entry["document"]["scope"]
            if entry["document"] != expected_plan_documents[entry_scope]:
                raise RequestContractError(
                    "existing request-contract plan history does not match the immutable cohort"
                )
        prior_counts = [entry["document"]["selected_count"] for entry in plan_history]
        if prior_counts and len(selected_rows) < max(prior_counts):
            raise RequestContractError("request-contract scope cannot shrink an existing lifecycle")

        known_plan_hashes = {entry["plan_sha256"] for entry in plan_history}
        reviewed_before_call = plan_sha256 in known_plan_hashes

        approved_hashes = _approved_plan_hashes(manifest)
        approved_plan_keys = {
            entry["plan_sha256"]: {
                make_generation_key(MODEL_KEY, str(row["condition_id"]), 1)
                for row in scope_rows[entry["document"]["scope"]]
            }
            for entry in plan_history
            if entry["plan_sha256"] in approved_hashes
        }
        histories = _validate_history(
            existing_records,
            selected_rows=selected_rows,
            model=model,
            prompt=prompt,
            artifact_manifest=artifact_manifest,
            provenance=provenance,
            allowed_plan_keys=approved_plan_keys,
        )
        required = [make_generation_key(MODEL_KEY, str(row["condition_id"]), 1) for row in selected_rows]
        required_set = set(required)
        dispatch_intents = _validate_dispatch_intents(
            _read_dispatch_intents(dispatch_path),
            required_keys=required_set,
            allowed_plan_keys=approved_plan_keys,
        )
        dispatch_commitment_fields = {
            "dispatched_generation_keys",
            "dispatched_generation_keys_sha256",
        }
        present_dispatch_commitment_fields = dispatch_commitment_fields.intersection(manifest)
        if present_dispatch_commitment_fields != dispatch_commitment_fields:
            raise RequestContractError("request-contract manifest lacks its dispatch commitment")
        dispatched_generation_keys = manifest["dispatched_generation_keys"]
        if (
            not isinstance(dispatched_generation_keys, list)
            or any(
                not isinstance(logical_key, str) or logical_key not in required_set
                for logical_key in dispatched_generation_keys
            )
            or len(dispatched_generation_keys) != len(set(dispatched_generation_keys))
            or manifest["dispatched_generation_keys_sha256"]
            != _ids_sha256(sorted(dispatched_generation_keys))
        ):
            raise RequestContractError("request-contract manifest has an invalid dispatch commitment")
        if set(dispatched_generation_keys) != set(dispatch_intents):
            raise RequestContractError(
                "request-contract dispatch journal does not match the manifest commitment"
            )
        for logical_key, history in histories.items():
            intent = dispatch_intents.get(logical_key)
            if intent is None or intent["plan_sha256"] != history[-1]["diagnostic_plan_sha256"]:
                raise RequestContractError(
                    f"request-contract output for {logical_key} lacks its matching pre-dispatch intent"
                )
        completion_fields = {
            "completed_plan_sha256",
            "completed_generation_keys_sha256",
            "completed_record_bindings_sha256",
            "completed_successes",
        }
        present_completion_fields = completion_fields.intersection(manifest)
        if manifest.get("status") == "completed" and present_completion_fields != completion_fields:
            raise RequestContractError("completed request-contract manifest lacks its key commitment")
        if present_completion_fields:
            if present_completion_fields != completion_fields:
                raise RequestContractError("request-contract manifest has a partial completion commitment")
            completed_plan_sha256 = manifest["completed_plan_sha256"]
            completed_entries = [
                entry for entry in manifest["plan_history"] if entry["plan_sha256"] == completed_plan_sha256
            ]
            if len(completed_entries) != 1 or completed_plan_sha256 not in approved_hashes:
                raise RequestContractError("request-contract completion references an invalid plan")
            completed_scope = completed_entries[0]["document"]["scope"]
            committed_keys = sorted(
                make_generation_key(MODEL_KEY, str(row["condition_id"]), 1)
                for row in select_request_contract_rows(artifact_rows, completed_scope)
            )
            if (
                manifest["completed_successes"] != len(committed_keys)
                or manifest["completed_generation_keys_sha256"] != _ids_sha256(committed_keys)
            ):
                raise RequestContractError("request-contract completion key commitment is invalid")
            committed_bindings: list[str] = []
            for logical_key in committed_keys:
                history = histories.get(logical_key)
                intent = dispatch_intents.get(logical_key)
                if (
                    history is None
                    or history[-1]["status"] != "success"
                    or intent is None
                    or intent["plan_sha256"] != history[-1]["diagnostic_plan_sha256"]
                ):
                    raise RequestContractError(
                        "request-contract completed artifacts do not match the manifest commitment"
                    )
                committed_bindings.append(
                    f"{logical_key}|{history[-1]['diagnostic_plan_sha256']}"
                )
            if manifest["completed_record_bindings_sha256"] != _ids_sha256(committed_bindings):
                raise RequestContractError(
                    "request-contract completed record bindings do not match the manifest commitment"
                )
        successes = {
            logical_key
            for logical_key, history in histories.items()
            if history and history[-1]["status"] == "success"
        }
        uncertain = {
            logical_key
            for logical_key, history in histories.items()
            if history and history[-1]["status"] == "uncertain"
        }
        unresolved_dispatches = set(dispatch_intents) - set(histories)
        blocked_outcomes = uncertain | unresolved_dispatches
        if blocked_outcomes:
            manifest["blocked_generation_keys"] = sorted(blocked_outcomes)
            manifest["status"] = "failed"
            _save_manifest(manifest_path, manifest)
            raise RequestContractError(
                "request-contract lifecycle has uncertain or unresolved target dispatches; "
                "manual reconciliation is required"
            )

        smoke_required = {
            make_generation_key(MODEL_KEY, str(row["condition_id"]), 1) for row in smoke_rows
        }
        if scope == "baseline":
            approved_smoke_plans = {
                entry["plan_sha256"]
                for entry in manifest["plan_history"]
                if entry["plan_sha256"] == smoke_plan_sha256
                and entry["document"] == smoke_plan
                and entry["plan_sha256"] in approved_hashes
            }
            if not approved_smoke_plans or not smoke_required.issubset(successes):
                raise RequestContractError(
                    "baseline scope requires an approved, completed 20-generation wire-smoke lifecycle"
                )

        if dry_run and not reviewed_before_call:
            manifest["plan_history"].append(
                {
                    "plan_sha256": plan_sha256,
                    "core_plan_sha256": core_plan_sha256,
                    "document": plan,
                }
            )
            manifest["status"] = "planned"
            _save_manifest(manifest_path, manifest)

        pending_rows = [
            row
            for row, logical_key in zip(selected_rows, required, strict=True)
            if logical_key not in successes
        ]
        summary: dict[str, Any] = {
            "schema": "ctm.eval_awareness.figure6_request_contract_summary.v1",
            "model_key": MODEL_KEY,
            "scope": scope,
            "selected_count": len(selected_rows),
            "selected_condition_ids_sha256": plan["selected_condition_ids_sha256"],
            "artifact_sha256": artifact_manifest["content_sha256"],
            "system_prompt_sha256": prompt.sha256,
            "server_attestation_sha256": server_attestation_sha256,
            "request_protocol_id": PROTOCOL_ID,
            "request_body_keys": list(plan["request"]["keys"]),
            "omitted_request_fields": list(OMITTED_REQUEST_FIELDS),
            "plan_sha256": plan_sha256,
            "core_plan_sha256": core_plan_sha256,
            "required": len(required),
            "existing_successes": len(successes.intersection(required)),
            "pending": len(pending_rows),
            "existing_records": len(existing_records),
            "planned_api_calls_minimum": len(pending_rows),
            "planned_api_calls_ceiling": len(pending_rows),
            "api_calls_made": 0,
            "dry_run": dry_run,
            "complete": not pending_rows,
            "manifest": str(manifest_path),
            "dispatch_journal": str(dispatch_path),
            "output": str(output),
            "plan": plan,
        }
        if dry_run:
            return summary

        if not confirm_requests:
            raise RequestContractError("real target requests require explicit --yes after reviewing a dry-run plan")
        if expected_plan_sha256 is None:
            raise RequestContractError(
                "real target requests require --expected-plan-sha256 from a reviewed dry run"
            )
        if not reviewed_before_call:
            raise RequestContractError("real target requests require a pre-existing reviewed dry-run plan")

        if plan_sha256 not in approved_hashes:
            manifest["approvals"].append(
                {
                    "approved_at": _utc_now(),
                    "confirmation": "--yes",
                    "plan_sha256": plan_sha256,
                    "reviewed_plan_sha256": expected_plan_sha256,
                }
            )
            approved_hashes.add(plan_sha256)
        approved_plan_keys[plan_sha256] = required_set
        manifest["active_plan_sha256"] = plan_sha256
        manifest["status"] = "running" if pending_rows else "completed"
        _save_manifest(manifest_path, manifest)

        if not pending_rows:
            return summary

        created_client = client is None
        resolved_api_key = api_key
        if created_client:
            resolved_api_key = api_key if api_key is not None else os.environ.get(api_key_env, "EMPTY")
            client = _create_openai_client(base_url=base_url, api_key=resolved_api_key)

        async def process(row: Mapping[str, Any]) -> dict[str, Any]:
            logical_key = make_generation_key(MODEL_KEY, str(row["condition_id"]), 1)
            try:
                _append_durable(
                    dispatch_path,
                    _dispatch_intent(generation_key=logical_key, plan_sha256=plan_sha256),
                )
                dispatched_generation_keys.append(logical_key)
                dispatched_generation_keys.sort()
                manifest["dispatched_generation_keys"] = list(dispatched_generation_keys)
                manifest["dispatched_generation_keys_sha256"] = _ids_sha256(
                    dispatched_generation_keys
                )
                _save_manifest(manifest_path, manifest)
            except Exception as exc:
                raise _PreDispatchIntentError(
                    f"could not durably record dispatch intent for {logical_key}; no request was sent"
                ) from exc
            record = await _generate_one(
                client=client,
                row=row,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                artifact_manifest=artifact_manifest,
                provenance=provenance,
                plan_sha256=plan_sha256,
                resume_attempt=1,
                api_key=resolved_api_key,
            )
            _append_durable(output, record)
            return record

        new_records: list[dict[str, Any]] = []
        active: set[asyncio.Task[dict[str, Any]]] = set()
        stopped = False
        pre_dispatch_errors: list[BaseException] = []
        dispatched_errors: list[BaseException] = []
        endpoint_check: dict[str, Any] = {"status": "not_run", "served_model_ids": []}

        def refill(pending_iter: Iterator[Mapping[str, Any]]) -> None:
            while not stopped and len(active) < max_concurrency:
                try:
                    row = next(pending_iter)
                except StopIteration:
                    return
                active.add(asyncio.create_task(process(row)))

        try:
            endpoint_check = await _endpoint_model_check(client, model)
            if endpoint_check.get("status") != "verified":
                raise RequestContractError("target endpoint must support a verified /models identity check")
            manifest["endpoint_check"] = endpoint_check
            _save_manifest(manifest_path, manifest)

            pending_iter = iter(pending_rows)
            refill(pending_iter)
            while active:
                done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        record = task.result()
                    except _PreDispatchIntentError as exc:
                        stopped = True
                        pre_dispatch_errors.append(exc)
                    except BaseException as exc:
                        stopped = True
                        dispatched_errors.append(exc)
                    else:
                        new_records.append(record)
                        if record["status"] != "success":
                            stopped = True
                    finally:
                        active.discard(task)
                refill(pending_iter)

        except BaseException:
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            manifest["status"] = "failed"
            try:
                observed_keys = {
                    str(record["generation_key"]) for record in read_generation_records(output)
                }
                intent_keys = set(
                    _validate_dispatch_intents(
                        _read_dispatch_intents(dispatch_path),
                        required_keys=required_set,
                        allowed_plan_keys=approved_plan_keys,
                    )
                )
                manifest["blocked_generation_keys"] = sorted(intent_keys - observed_keys)
                _save_manifest(manifest_path, manifest)
            except Exception:
                # The fsynced intent journal remains the fail-closed authority
                # even when best-effort cleanup metadata cannot be written.
                pass
            raise
        finally:
            if created_client:
                await _close_client(client)

        final_records = read_generation_records(output)
        final_histories = _validate_history(
            final_records,
            selected_rows=selected_rows,
            model=model,
            prompt=prompt,
            artifact_manifest=artifact_manifest,
            provenance=provenance,
            allowed_plan_keys=approved_plan_keys,
        )
        final_successes = {
            logical_key
            for logical_key, history in final_histories.items()
            if history[-1]["status"] == "success"
        }
        final_uncertain = sorted(
            logical_key
            for logical_key, history in final_histories.items()
            if history[-1]["status"] == "uncertain"
        )
        final_intents = _validate_dispatch_intents(
            _read_dispatch_intents(dispatch_path),
            required_keys=required_set,
            allowed_plan_keys=approved_plan_keys,
        )
        for logical_key, history in final_histories.items():
            intent = final_intents.get(logical_key)
            if intent is None or intent["plan_sha256"] != history[-1]["diagnostic_plan_sha256"]:
                raise RequestContractError(
                    f"request-contract output for {logical_key} lacks its matching pre-dispatch intent"
                )
        unresolved_dispatches = set(final_intents) - set(final_histories)
        final_blocked = sorted(set(final_uncertain) | unresolved_dispatches)
        if final_blocked:
            manifest["blocked_generation_keys"] = final_blocked
            manifest["status"] = "failed"
            _save_manifest(manifest_path, manifest)
            raise RequestContractError(
                "one or more target requests have uncertain or unresolved outcomes; "
                "manual reconciliation is required"
            )
        if pre_dispatch_errors:
            manifest["status"] = "failed"
            _save_manifest(manifest_path, manifest)
            raise RequestContractError(
                "a pre-dispatch intent could not be persisted; no request was sent for that generation"
            )
        if dispatched_errors:
            manifest["status"] = "failed"
            _save_manifest(manifest_path, manifest)
            raise RequestContractError(
                "a dispatched target request did not complete normally; manual reconciliation is required"
            )

        complete = set(required).issubset(final_successes)
        manifest["status"] = "completed" if complete else "failed"
        manifest["completed_successes"] = len(set(required).intersection(final_successes))
        if complete:
            manifest["completed_plan_sha256"] = plan_sha256
            manifest["completed_generation_keys_sha256"] = _ids_sha256(sorted(required))
            manifest["completed_record_bindings_sha256"] = _ids_sha256(
                [
                    f"{logical_key}|{final_histories[logical_key][-1]['diagnostic_plan_sha256']}"
                    for logical_key in sorted(required)
                ]
            )
        _save_manifest(manifest_path, manifest)
        if not complete:
            raise RequestContractError("request-contract lifecycle stopped before all required generations completed")

        new_successes = sum(row["status"] == "success" for row in new_records)
        summary.update(
            {
                "api_calls_made": len(new_records),
                "new_successes": new_successes,
                "new_errors": len(new_records) - new_successes,
                "pending": 0,
                "complete": True,
                "endpoint_check": endpoint_check,
            }
        )
        return summary


__all__ = [
    "BASELINE_COUNT",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BASE_SECONDS",
    "DISPATCH_INTENT_SCHEMA",
    "EXPECTED_GENERATION_CONFIG",
    "EXPECTED_MAX_MODEL_LEN",
    "EXPECTED_VLLM_VERSION",
    "REQUEST_ONLY_RUNTIME_PROFILE",
    "RUNTIME_ABLATION_V023_TP4_PROFILE",
    "SERVER_RUNTIME_PROFILES",
    "ServerRuntimeProfile",
    "MODEL_KEY",
    "OMITTED_REQUEST_FIELDS",
    "PROTOCOL_ID",
    "REASONING_EFFORT",
    "SCOPES",
    "SERVER_ATTESTATION_SCHEMA",
    "WIRE_SMOKE_COUNT",
    "RequestContractError",
    "build_target_request",
    "get_server_runtime_profile",
    "load_server_attestation",
    "require_artifact_path",
    "request_contract_manifest_path",
    "request_contract_dispatch_path",
    "run_request_contract_gate",
    "select_request_contract_rows",
]
