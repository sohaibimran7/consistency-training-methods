#!/usr/bin/env python3
"""Judge Figure 6 generations with a resumable OpenRouter request stream.

The append-only attempt log is the recovery boundary.  Once every generation
has exactly one successful attempt, a compact normalized judgment JSONL is
written for :mod:`figure6_analysis`.
"""

from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import ipaddress
import json
import math
import os
import re
from contextlib import contextmanager
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterator
from urllib.parse import urlparse

import httpx

from ctm.artifacts import write_atomic_bytes
from ctm_data.adapters.eval_awareness.figure6_judge import (
    MAX_JUDGE_TOKENS,
    PAPER_JUDGE_TEMPLATE_SHA256,
    _canonical_json,
    _normalized_generation_fields,
    _read_jsonl,
    custom_id_for_generation,
    normalize_judge_object,
    parse_judge_json,
    render_judge_prompt,
    select_successful_generations,
    validate_generation,
)
from ctm_data.adapters.eval_awareness.figure6_spec import (
    DATASET_ID,
    DATASET_REVISION,
    FIGURE6_CONDITIONS,
    FIGURE6_TASK_COUNT,
    FIGURE6_VALENCES,
    MODEL_SPECS,
    UPSTREAM_CODE_REVISION,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_ENDPOINT = "/chat/completions"
GPT_OSS_120B_NITRO_DIRECT_PROFILE = "gpt-oss-120b-nitro-direct"
DEEPSEEK_V32_DIRECT_PROFILE = "deepseek-v3.2-direct"
CLAUDE_SONNET_46_DIRECT_PROFILE = "claude-sonnet-4.6-direct"
MUSE_US_PROXY_PROFILE = "muse-spark-1.1-us-proxy"
DEFAULT_JUDGE_PROFILE = GPT_OSS_120B_NITRO_DIRECT_PROFILE
GPT_OSS_120B_NITRO_OPENROUTER_JUDGE_MODEL = "openai/gpt-oss-120b:nitro"
GPT_OSS_120B_OPENROUTER_RESPONSE_MODEL = "openai/gpt-oss-120b"
DEEPSEEK_V32_OPENROUTER_JUDGE_MODEL = "deepseek/deepseek-v3.2"
DEEPSEEK_V32_OPENROUTER_RESPONSE_MODEL = "deepseek/deepseek-v3.2-20251201"
MUSE_OPENROUTER_JUDGE_MODEL = "meta/muse-spark-1.1"
MUSE_OPENROUTER_RESPONSE_MODEL = "meta/muse-spark-1.1-20260709"
JUDGE_PROFILES: dict[str, dict[str, Any]] = {
    GPT_OSS_120B_NITRO_DIRECT_PROFILE: {
        "label": "OpenRouter GPT-OSS 120B Nitro alternative judge",
        "model": GPT_OSS_120B_NITRO_OPENROUTER_JUDGE_MODEL,
        "allowed_response_models": (
            GPT_OSS_120B_NITRO_OPENROUTER_JUDGE_MODEL,
            GPT_OSS_120B_OPENROUTER_RESPONSE_MODEL,
        ),
        "route_mode": "direct",
        "temperature": 0.0,
        "max_tokens": MAX_JUDGE_TOKENS,
        "concurrency": 24,
        "max_retry_after": 300.0,
        "provider_routing": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "require_parameters": True,
            "sort": "throughput",
        },
        "reasoning": {"effort": "high", "exclude": True},
        "response_format": {"type": "json_object"},
    },
    DEEPSEEK_V32_DIRECT_PROFILE: {
        "label": "OpenRouter DeepSeek V3.2 alternative judge",
        "model": DEEPSEEK_V32_OPENROUTER_JUDGE_MODEL,
        "allowed_response_models": (
            DEEPSEEK_V32_OPENROUTER_JUDGE_MODEL,
            DEEPSEEK_V32_OPENROUTER_RESPONSE_MODEL,
        ),
        "route_mode": "direct",
        "temperature": 0.0,
        "max_tokens": MAX_JUDGE_TOKENS,
        "concurrency": 24,
        "max_retry_after": 300.0,
        "provider_routing": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "require_parameters": True,
            "sort": "price",
        },
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_object"},
    },
    CLAUDE_SONNET_46_DIRECT_PROFILE: {
        "label": "OpenRouter Claude Sonnet 4.6 alternative judge",
        "model": "anthropic/claude-sonnet-4.6",
        "allowed_response_models": ("anthropic/claude-sonnet-4.6",),
        "route_mode": "direct",
        "temperature": 0.0,
        "max_tokens": MAX_JUDGE_TOKENS,
        "concurrency": 24,
        "max_retry_after": 300.0,
        "provider_routing": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "order": ["Anthropic"],
            "require_parameters": True,
        },
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_object"},
    },
    MUSE_US_PROXY_PROFILE: {
        "label": "OpenRouter Muse alternative judge",
        "model": MUSE_OPENROUTER_JUDGE_MODEL,
        "allowed_response_models": (
            MUSE_OPENROUTER_JUDGE_MODEL,
            MUSE_OPENROUTER_RESPONSE_MODEL,
        ),
        "route_mode": "muse_us_proxy",
        "temperature": 0.0,
        "max_tokens": MAX_JUDGE_TOKENS,
        "concurrency": 24,
        "max_retry_after": 300.0,
        "provider_routing": {
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_object"},
    },
}
DEFAULT_OPENROUTER_JUDGE_MODEL = JUDGE_PROFILES[DEFAULT_JUDGE_PROFILE]["model"]
DEFAULT_ALLOWED_RESPONSE_MODELS = JUDGE_PROFILES[DEFAULT_JUDGE_PROFILE]["allowed_response_models"]
DEFAULT_TEMPERATURE = JUDGE_PROFILES[DEFAULT_JUDGE_PROFILE]["temperature"]
DEFAULT_CONCURRENCY = 24
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_MAX_RETRY_AFTER = JUDGE_PROFILES[DEFAULT_JUDGE_PROFILE]["max_retry_after"]
CURRENT_QWEN_MODEL_KEYS = ("qwen32", "qwen_mo_mid", "qwen_mo_post")
EXPECTED_GENERATIONS_PER_MODEL = 5_400
PERMANENT_ACCOUNT_HTTP_STATUSES = frozenset({401, 402, 403})
ATTEMPT_SCHEMA = "ctm.eval_awareness.figure6.openrouter_attempt.v2"
PLAN_SCHEMA = "ctm.eval_awareness.figure6.openrouter_plan.v3"
MANIFEST_SCHEMA = "ctm.eval_awareness.figure6.openrouter_manifest.v3"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_MARKERS = (
    "example",
    "placeholder",
    "replace",
    "changeme",
    "todo",
    "your-",
    "your_",
    "unit",
    "test",
)


class OpenRouterJudgeError(RuntimeError):
    """The OpenRouter judge run could not complete safely."""


class PermanentOpenRouterError(OpenRouterJudgeError):
    """An account-level error for which retrying would incur delay or risk."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((_canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _judge_profile(profile_id: str) -> dict[str, Any]:
    if profile_id not in JUDGE_PROFILES:
        raise ValueError(f"judge_profile must be one of the explicitly registered profiles: {sorted(JUDGE_PROFILES)}")
    return {**JUDGE_PROFILES[profile_id], "id": profile_id}


def _proxy_provenance(proxy: str | None, *, route_mode: str) -> dict[str, Any]:
    if route_mode not in {"direct", "muse_us_proxy"}:
        raise ValueError(f"unsupported registered route mode: {route_mode!r}")
    if proxy is None:
        if route_mode == "muse_us_proxy":
            raise ValueError("the Muse US-proxy profile requires an explicit loopback socks5h proxy")
        return {"enabled": False}
    if route_mode == "direct":
        raise ValueError("direct judge profiles prohibit a proxy")
    parsed = urlparse(proxy)
    if parsed.scheme != "socks5h":
        raise ValueError("the Muse US-proxy profile requires exactly the socks5h scheme")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("proxy credentials are not accepted on the command line")
    if parsed.query or parsed.fragment or parsed.path:
        raise ValueError("proxy must not include a path, query, or fragment")
    loopback = parsed.hostname == "127.0.0.1"
    if not loopback:
        raise ValueError("the Muse US-proxy profile requires exact proxy host 127.0.0.1")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("proxy has an invalid port") from exc
    if parsed.hostname is None or port is None:
        raise ValueError("proxy must include an explicit host and port")
    if port != 1080:
        raise ValueError("the Muse US-proxy profile requires exact proxy port 1080")
    if proxy != "socks5h://127.0.0.1:1080":
        raise ValueError("the Muse US-proxy profile requires exactly socks5h://127.0.0.1:1080")
    return {
        "enabled": True,
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": port,
        "loopback": loopback,
    }


def _plain_attestation_value(value: Any, *, name: str, max_length: int = 200) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > max_length:
        raise ValueError(f"{name} must be a non-empty trimmed string of at most {max_length} characters")
    folded = value.casefold()
    if (
        not value.isprintable()
        or any(marker in folded for marker in ("://", "@", "bearer ", "api_key", "private key"))
        or any(marker in folded for marker in _PLACEHOLDER_MARKERS)
    ):
        raise ValueError(f"{name} must be a concrete, plain, non-secret value rather than a placeholder")
    return value


def _utc_attestation_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("route_attested_at must be an ISO 8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("route_attested_at must be an ISO 8601 UTC timestamp ending in Z") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):  # pragma: no cover - the Z conversion guarantees UTC
        raise ValueError("route_attested_at must be UTC")
    return value


def _route_provenance(
    *,
    expected_exit_instance_id: str | None,
    expected_exit_ssh_host: str | None,
    expected_exit_ssh_port: int | None,
    route_country_code: str | None,
    route_attested_at: str | None,
    route_attested_by: str | None,
    route_attestation_sha256: str | None,
    required: bool,
) -> dict[str, Any] | None:
    values = (
        expected_exit_instance_id,
        expected_exit_ssh_host,
        expected_exit_ssh_port,
        route_country_code,
        route_attested_at,
        route_attested_by,
        route_attestation_sha256,
    )
    if all(value is None for value in values):
        if required:
            raise ValueError("the Muse judge requires a complete U.S. Vast route attestation")
        return None
    if any(value is None for value in values):
        raise ValueError("route attestation fields must be supplied together")
    instance_id = _plain_attestation_value(
        expected_exit_instance_id,
        name="expected_exit_instance_id",
    )
    if not instance_id.isdecimal() or int(instance_id) < 1:
        raise ValueError("expected_exit_instance_id must be a positive numeric Vast instance ID")
    ssh_host = _plain_attestation_value(expected_exit_ssh_host, name="expected_exit_ssh_host", max_length=253)
    parsed_host = urlparse(f"ssh://{ssh_host}")
    if parsed_host.hostname != ssh_host or ssh_host in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("expected_exit_ssh_host must be a concrete non-loopback hostname or address")
    if (
        not isinstance(expected_exit_ssh_port, int)
        or isinstance(expected_exit_ssh_port, bool)
        or not 1 <= expected_exit_ssh_port <= 65_535
    ):
        raise ValueError("expected_exit_ssh_port must be an integer from 1 through 65535")
    if route_country_code != "US":
        raise ValueError("route_country_code must be exactly US")
    attested_by = _plain_attestation_value(route_attested_by, name="route_attested_by")
    if not isinstance(route_attestation_sha256, str) or _SHA256_RE.fullmatch(route_attestation_sha256) is None:
        raise ValueError("route_attestation_sha256 must be a 64-character lowercase SHA-256 digest")
    return {
        "schema": "ctm.eval_awareness.figure6.vast_route_attestation.v1",
        "vast_instance_id": instance_id,
        "ssh_host": ssh_host,
        "ssh_port": expected_exit_ssh_port,
        "country_code": "US",
        "attested_at": _utc_attestation_timestamp(route_attested_at),
        "attested_by": attested_by,
        "evidence_sha256": route_attestation_sha256,
    }


def _validated_route_evidence(
    evidence_bytes: bytes | None,
    *,
    evidence_sha256: str | None,
    proxy: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(evidence_bytes, bytes) or not evidence_bytes:
        raise ValueError("the Muse US-proxy profile requires archived route-attestation evidence bytes")
    observed_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    if evidence_sha256 != observed_sha256:
        raise ValueError("route-attestation evidence bytes do not match route_attestation_sha256")
    try:
        evidence = json.loads(evidence_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("route-attestation evidence must be UTF-8 JSON") from exc
    if not isinstance(evidence, dict):
        raise ValueError("route-attestation evidence must be a JSON object")
    return _validate_route_evidence_object(evidence, proxy=proxy, route=route)


def _validate_route_evidence_object(
    evidence: Mapping[str, Any],
    *,
    proxy: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = dict(evidence)
    expected_proxy = {
        "enabled": True,
        "scheme": "socks5h",
        "host": "127.0.0.1",
        "port": 1080,
        "loopback": True,
    }
    if dict(proxy) != expected_proxy:
        raise ValueError("Muse route evidence requires the exact registered socks5h://127.0.0.1:1080 proxy")
    required_keys = {
        "schema",
        "vast_instance_id",
        "ssh_host",
        "ssh_port",
        "country_code",
        "attested_at",
        "attested_by",
        "socks_proxy",
        "ssh_control_socket",
        "exit_ip",
        "exit_country_code",
        "vast_console_evidence_sha256",
    }
    if set(evidence) != required_keys:
        raise ValueError("route-attestation evidence must contain exactly the registered credential-free schema fields")
    expected = {
        "schema": "ctm.eval_awareness.figure6.vast_route_evidence.v1",
        "vast_instance_id": route["vast_instance_id"],
        "ssh_host": route["ssh_host"],
        "ssh_port": route["ssh_port"],
        "country_code": "US",
        "attested_at": route["attested_at"],
        "attested_by": route["attested_by"],
        "socks_proxy": f"{proxy['scheme']}://{proxy['host']}:{proxy['port']}",
        "exit_country_code": "US",
    }
    if any(evidence.get(field) != value for field, value in expected.items()):
        raise ValueError("route-attestation evidence content differs from the exact Muse route and socks5h proxy")
    control_socket = evidence["ssh_control_socket"]
    if (
        not isinstance(control_socket, str)
        or not control_socket.startswith("/private/tmp/")
        or control_socket != control_socket.strip()
    ):
        raise ValueError("route-attestation evidence must record a dedicated /private/tmp SSH control socket")
    try:
        ipaddress.ip_address(evidence["exit_ip"])
    except (TypeError, ValueError) as exc:
        raise ValueError("route-attestation evidence exit_ip must be a concrete IP address") from exc
    if not _sha256_digest(evidence["vast_console_evidence_sha256"]):
        raise ValueError("route-attestation evidence must include a lowercase Vast console evidence SHA-256")
    return evidence


def _sha256_digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _expected_qwen_model_keys(values: Sequence[str] | None, *, required: bool) -> tuple[str, ...] | None:
    if values is None:
        if required:
            raise ValueError("paid judging requires one or more explicit expected_model_keys")
        return None
    normalized = tuple(values)
    if (
        not normalized
        or any(not isinstance(value, str) or not value for value in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("expected_model_keys must contain one or more unique non-empty model keys")
    extra = sorted(set(normalized) - set(CURRENT_QWEN_MODEL_KEYS))
    if extra:
        raise ValueError(f"only the current three-Qwen scope is authorized; disallowed model keys: {extra}")
    canonical = tuple(model_key for model_key in CURRENT_QWEN_MODEL_KEYS if model_key in normalized)
    if normalized != canonical:
        raise ValueError(f"expected_model_keys must follow current three-Qwen order {list(CURRENT_QWEN_MODEL_KEYS)}")
    return normalized


def _validate_generation_provenance(record: Mapping[str, Any], *, index: int) -> None:
    model_key = record["model_key"]
    model = MODEL_SPECS[model_key]
    provenance = record["generation_provenance"]
    expected = {
        "provenance_schema": "ctm.eval_awareness.figure6_generation_run",
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "model_id": model.model_id,
        "model_key": model.key,
        "model_revision": model.revision,
        "prompt_key": model.prompt.key,
        "prompt_revision": UPSTREAM_CODE_REVISION,
        "prompt_sha256": model.prompt.sha256,
        "temperature": 0.3,
        "max_tokens": 4_096,
        "replicates": 3,
    }
    observed = {field: provenance.get(field) for field in expected}
    if observed != expected:
        raise ValueError(f"generation {index}.generation_provenance does not match the pinned {model_key} protocol")
    for field in ("artifact_schema", "artifact_schema_version", "artifact_sha256"):
        if field not in provenance:
            raise ValueError(f"generation {index}.generation_provenance is missing {field}")
    if not isinstance(provenance["artifact_schema"], str) or not provenance["artifact_schema"]:
        raise ValueError(f"generation {index}.generation_provenance.artifact_schema must be non-empty")
    if not isinstance(provenance["artifact_schema_version"], int) or isinstance(
        provenance["artifact_schema_version"], bool
    ):
        raise ValueError(f"generation {index}.generation_provenance.artifact_schema_version must be an integer")
    if (
        not isinstance(provenance["artifact_sha256"], str)
        or _SHA256_RE.fullmatch(provenance["artifact_sha256"]) is None
    ):
        raise ValueError(f"generation {index}.generation_provenance.artifact_sha256 must be a SHA-256 digest")
    provenance_sha256 = provenance.get("provenance_sha256")
    unhashed = {field: value for field, value in provenance.items() if field != "provenance_sha256"}
    if provenance_sha256 != _sha256_json(unhashed):
        raise ValueError(f"generation {index}.generation_provenance.provenance_sha256 is invalid")
    expected_system = {
        "prompt_key": model.prompt.key,
        "prompt_revision": UPSTREAM_CODE_REVISION,
        "prompt_sha256": model.prompt.sha256,
    }
    if record["system_prompt_provenance"] != expected_system:
        raise ValueError(f"generation {index}.system_prompt_provenance does not match the pinned {model_key} prompt")


def _validate_exact_qwen_matrix(
    generations: Sequence[Mapping[str, Any]],
    *,
    expected_model_keys: Sequence[str],
) -> dict[str, Any]:
    """Fail closed unless generations are exact complete registered-Qwen matrices."""

    model_keys = _expected_qwen_model_keys(expected_model_keys, required=True)
    assert model_keys is not None
    expected_total = EXPECTED_GENERATIONS_PER_MODEL * len(model_keys)
    if len(generations) != expected_total:
        raise ValueError(
            f"exact paid matrix requires {EXPECTED_GENERATIONS_PER_MODEL:,} generations per model "
            f"({expected_total:,} total for {list(model_keys)}), observed {len(generations):,}"
        )
    observed_model_keys = {record["model_key"] for record in generations}
    if observed_model_keys != set(model_keys):
        raise ValueError(
            "generation model keys do not match the explicitly expected paid scope: "
            f"missing={sorted(set(model_keys) - observed_model_keys)}, "
            f"extra={sorted(observed_model_keys - set(model_keys))}"
        )

    per_model: dict[str, int] = {model_key: 0 for model_key in model_keys}
    pair_tasks: dict[tuple[str, str], str] = {}
    condition_identity: dict[tuple[str, str, str], tuple[str, str]] = {}
    slot_counts: dict[tuple[str, str, str, str, int], int] = {}
    protocol_values: set[str] = set()
    prompt_values: dict[tuple[str, str, str], set[str]] = {}
    for index, record in enumerate(generations, start=1):
        model_key = record["model_key"]
        model = MODEL_SPECS[model_key]
        identity = (record["model_display"], record["model_id"], record["model_revision"])
        expected_identity = (model.display_name, model.model_id, model.revision)
        if identity != expected_identity:
            raise ValueError(f"generation {index} does not match the pinned identity for {model_key}")
        _validate_generation_provenance(record, index=index)
        provenance = record["generation_provenance"]
        protocol_values.add(
            _canonical_json(
                {
                    "artifact_schema": provenance["artifact_schema"],
                    "artifact_schema_version": provenance["artifact_schema_version"],
                    "artifact_sha256": provenance["artifact_sha256"],
                    "dataset_id": provenance["dataset_id"],
                    "dataset_revision": provenance["dataset_revision"],
                    "temperature": provenance["temperature"],
                    "max_tokens": provenance["max_tokens"],
                    "replicates": provenance["replicates"],
                }
            )
        )
        per_model[model_key] += 1
        pair_key = (record["pair_id"], record["valence"])
        prior_task = pair_tasks.setdefault(pair_key, record["task_id"])
        if prior_task != record["task_id"]:
            raise ValueError(f"pair/valence {pair_key!r} maps to multiple task IDs")
        condition_key = (record["pair_id"], record["valence"], record["config_name"])
        prior_condition = condition_identity.setdefault(
            condition_key,
            (record["condition_id"], record["task_id"]),
        )
        if prior_condition != (record["condition_id"], record["task_id"]):
            raise ValueError(f"condition slot {condition_key!r} is inconsistent across models or replicates")
        slot = (
            model_key,
            record["pair_id"],
            record["valence"],
            record["config_name"],
            record["replicate"],
        )
        slot_counts[slot] = slot_counts.get(slot, 0) + 1
        prompt_values.setdefault(condition_key, set()).add(hashlib.sha256(record["prompt"].encode()).hexdigest())

    if set(per_model.values()) != {EXPECTED_GENERATIONS_PER_MODEL}:
        raise ValueError(f"paid matrix must contain exactly 5,400 generations per model; observed {per_model}")
    pairs = sorted({pair_id for pair_id, _ in pair_tasks})
    if len(pairs) != FIGURE6_TASK_COUNT:
        raise ValueError(f"paid matrix requires exactly {FIGURE6_TASK_COUNT} task pairs; observed {len(pairs)}")
    expected_pair_tasks = {(pair_id, valence) for pair_id in pairs for valence in FIGURE6_VALENCES}
    if set(pair_tasks) != expected_pair_tasks:
        raise ValueError("each task pair must have exactly one safety and one capability task")
    expected_slots = {
        (model_key, pair_id, valence, config_name, replicate)
        for model_key in model_keys
        for pair_id in pairs
        for valence in FIGURE6_VALENCES
        for config_name in FIGURE6_CONDITIONS
        for replicate in (1, 2, 3)
    }
    if set(slot_counts) != expected_slots or set(slot_counts.values()) != {1}:
        missing = sorted(expected_slots - set(slot_counts))[:5]
        extra = sorted(set(slot_counts) - expected_slots)[:5]
        duplicates = sorted(slot for slot, count in slot_counts.items() if count != 1)[:5]
        raise ValueError(
            "paid matrix is not exactly 100 task pairs x 2 valences x 9 configurations x 3 replicates "
            f"per model; missing={missing}, extra={extra}, duplicates={duplicates}"
        )
    if len(protocol_values) != 1:
        raise ValueError("paid matrix contains mixed artifact, dataset, or generation protocol provenance")
    if any(len(values) != 1 for values in prompt_values.values()):
        raise ValueError("paid matrix contains inconsistent task prompts across models or replicates")
    return {
        "model_keys": list(model_keys),
        "generation_count": expected_total,
        "generations_per_model": EXPECTED_GENERATIONS_PER_MODEL,
        "task_pair_count": FIGURE6_TASK_COUNT,
        "valences": list(FIGURE6_VALENCES),
        "configurations": list(FIGURE6_CONDITIONS),
        "replicates": [1, 2, 3],
        "pair_ids_sha256": _sha256_json(pairs),
        "generation_protocol_sha256": hashlib.sha256(next(iter(protocol_values)).encode()).hexdigest(),
    }


def _sanitized_endpoint(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("base_url must be a non-empty string")
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or parsed.hostname is None:
        raise ValueError("base_url must be an absolute HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url has an invalid port") from exc
    authority = parsed.hostname if port is None else f"{parsed.hostname}:{port}"
    base_path = parsed.path.rstrip("/")
    return f"https://{authority}{base_path}{OPENROUTER_CHAT_ENDPOINT}"


def _allowed_response_models(profile: Mapping[str, Any], values: Sequence[str] | None) -> tuple[str, ...]:
    registered = tuple(profile["allowed_response_models"])
    if values is None:
        values = registered
    normalized = tuple(values)
    if not normalized or any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError("allowed_response_models must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("allowed_response_models must be unique")
    if profile["model"] not in normalized:
        raise ValueError("allowed_response_models must include the requested judge_model")
    if set(normalized) != set(registered):
        raise ValueError(
            f"judge profile {profile['id']!r} permits exactly the registered response identities {list(registered)}"
        )
    return tuple(sorted(normalized))


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 429} or 500 <= status_code <= 599


@contextmanager
def _run_lock(attempt_log_path: Path) -> Iterator[BinaryIO]:
    """Hold a non-blocking OS lock across history, requests, and final output."""

    attempt_log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = attempt_log_path.with_name(attempt_log_path.name + ".lock")
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OpenRouterJudgeError(f"another process is using OpenRouter attempt log {attempt_log_path}") from exc
        yield handle


def _retry_after_seconds(headers: Mapping[str, str], *, now: datetime | None = None) -> float | None:
    raw = next((value for key, value in headers.items() if key.casefold() == "retry-after"), None)
    if raw is None:
        return None
    try:
        seconds = float(raw)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except ValueError:
        try:
            target = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        return max(0.0, (target - (now or datetime.now(UTC))).total_seconds())


def _safe_error_body(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except (json.JSONDecodeError, ValueError):
        return {"text": response.text[:2_000]}
    return {"body": _audit_json_value(dict(value) if isinstance(value, Mapping) else value)}


def _message_content(body: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    if body.get("error") not in (None, {}):
        raise ValueError(f"OpenRouter response contains an error: {body['error']!r}")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ValueError("OpenRouter response must contain exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("OpenRouter response choice must contain string message.content")
    return message["content"], choices[0]


def _audit_json_value(value: Any) -> Any:
    try:
        _canonical_json(value)
    except ValueError:
        return {"noncanonical_json_repr": repr(value)}
    return value


def _raw_http_200_response(response: httpx.Response) -> dict[str, Any]:
    """Capture the paid provider response before any parsing or validation."""

    raw_body_text = response.text
    try:
        parsed_body = response.json()
    except (json.JSONDecodeError, ValueError):
        parsed_body = None
    body = parsed_body if isinstance(parsed_body, Mapping) else None
    content: Any = None
    if body is not None:
        choices = body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            message = choices[0].get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
    return {
        "status_code": 200,
        "header_request_id": response.headers.get("x-request-id"),
        "id": _audit_json_value(body.get("id") if body is not None else None),
        "model": _audit_json_value(body.get("model") if body is not None else None),
        "usage": _audit_json_value(body.get("usage") if body is not None else None),
        "content": _audit_json_value(content),
        "raw_body": _audit_json_value(parsed_body),
        "raw_body_text": raw_body_text,
        "raw_body_base64": base64.b64encode(response.content).decode("ascii"),
        "raw_body_sha256": hashlib.sha256(response.content).hexdigest(),
    }


def _load_attempt_history(
    path: Path,
    *,
    expected_requests: Mapping[str, Mapping[str, Any]],
    judge_template_sha256: str,
    allowed_plan_sha256s: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int], dict[str, tuple[int, ...]]]:
    if not path.exists():
        return [], {}, {}, {}
    rows = _read_jsonl([path], label="OpenRouter attempt")
    histories: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows, start=1):
        if row.get("schema") != ATTEMPT_SCHEMA:
            raise OpenRouterJudgeError(f"attempt {index} has an unsupported schema")
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or custom_id not in expected_requests:
            raise OpenRouterJudgeError(f"attempt {index} has an unexpected custom_id")
        if row.get("status") not in {"success", "error"}:
            raise OpenRouterJudgeError(f"attempt {index} has an invalid status")
        request = row.get("request")
        if not isinstance(request, Mapping) or _canonical_json(request) != _canonical_json(
            expected_requests[custom_id]
        ):
            raise OpenRouterJudgeError(f"attempt {index} does not match the current judge request protocol")
        approval = row.get("approval")
        plan_sha256 = row.get("plan_sha256")
        if (
            plan_sha256 not in allowed_plan_sha256s
            or not isinstance(approval, Mapping)
            or approval.get("confirmation") != "--yes"
            or approval.get("plan_sha256") != plan_sha256
            or approval.get("reviewed_plan_sha256") != plan_sha256
            or not isinstance(approval.get("approval_index"), int)
            or isinstance(approval.get("approval_index"), bool)
            or approval["approval_index"] < 1
        ):
            raise OpenRouterJudgeError(f"attempt {index} has invalid paid approval metadata")
        histories.setdefault(custom_id, []).append(row)
    successes: dict[str, dict[str, Any]] = {}
    attempt_counts: dict[str, int] = {}
    paid_error_attempts: dict[str, tuple[int, ...]] = {}
    for custom_id, history in histories.items():
        successful = [row for row in history if row["status"] == "success"]
        if len(successful) > 1:
            raise OpenRouterJudgeError(f"attempt log contains duplicate successes for {custom_id}")
        if successful and history[-1] is not successful[0]:
            raise OpenRouterJudgeError(f"attempt log contains records after success for {custom_id}")
        for expected_attempt, row in enumerate(history, start=1):
            if row.get("attempt") != expected_attempt:
                raise OpenRouterJudgeError(f"attempt numbering is not contiguous for {custom_id}")
        attempt_counts[custom_id] = len(history)
        paid_error_attempts[custom_id] = tuple(
            row["attempt"]
            for row in history
            if row["status"] == "error"
            and isinstance(row.get("response"), Mapping)
            and row["response"].get("status_code") == 200
        )
        if successful:
            judgment = successful[0].get("judgment")
            if not isinstance(judgment, Mapping):
                raise OpenRouterJudgeError(f"successful attempt lacks a judgment for {custom_id}")
            expected_judgment_protocol = {
                "judge_profile": expected_requests[custom_id]["judge_profile"],
                "judge_model": expected_requests[custom_id]["model"],
                "judge_template_sha256": judge_template_sha256,
                "judge_max_completion_tokens": expected_requests[custom_id]["max_tokens"],
                "judge_temperature": expected_requests[custom_id]["temperature"],
                "judge_provider_routing": expected_requests[custom_id]["provider_routing"],
                "judge_reasoning": expected_requests[custom_id]["reasoning"],
                "judge_response_format": expected_requests[custom_id]["response_format"],
                "judge_route_mode": expected_requests[custom_id]["route_mode"],
            }
            observed_judgment_protocol = {field: judgment.get(field) for field in expected_judgment_protocol}
            if observed_judgment_protocol != expected_judgment_protocol:
                raise OpenRouterJudgeError(f"successful attempt has mismatched judge provenance for {custom_id}")
            response_model = judgment.get("judge_response_model")
            if response_model not in expected_requests[custom_id]["allowed_response_models"]:
                raise OpenRouterJudgeError(f"successful attempt has an unapproved response model for {custom_id}")
            if judgment.get("generation_record_sha256") != expected_requests[custom_id]["generation_record_sha256"]:
                raise OpenRouterJudgeError(f"successful attempt has mismatched generation identity for {custom_id}")
            successes[custom_id] = dict(judgment)
    return rows, successes, attempt_counts, paid_error_attempts


def _append_attempt(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (_canonical_json(dict(row)) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_or_verify_output(path: Path, judgments: Sequence[Mapping[str, Any]]) -> str:
    payload = _jsonl_bytes(judgments)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.read_bytes() != payload:
            raise OpenRouterJudgeError(f"existing normalized output does not match completed attempt log: {path}")
        return digest
    write_atomic_bytes(path, payload)
    return digest


def _manifest_plan(
    *,
    plan_sha256: str,
    request_protocols: Mapping[str, Mapping[str, Any]],
    judge_profile: str,
    judge_model: str,
    allowed_response_models: Sequence[str],
    judge_template_sha256: str,
    temperature: float,
    max_tokens: int,
    max_attempts: int,
    concurrency: int,
    max_retry_after: float | None,
    endpoint: str,
    proxy: Mapping[str, Any],
    route: Mapping[str, Any] | None,
    route_mode: str,
    provider_routing: Mapping[str, Any],
    reasoning: Mapping[str, Any],
    response_format: Mapping[str, Any],
    matrix: Mapping[str, Any],
    plan_document: Mapping[str, Any],
    core_plan_sha256: str,
) -> dict[str, Any]:
    ordered_ids = sorted(request_protocols)
    return {
        "plan_sha256": plan_sha256,
        "core_plan_sha256": core_plan_sha256,
        "document": dict(plan_document),
        "request_count": len(ordered_ids),
        "ordered_custom_ids_sha256": _sha256_json(ordered_ids),
        "request_protocols_sha256": _sha256_json(
            [{"custom_id": custom_id, "request": request_protocols[custom_id]} for custom_id in ordered_ids]
        ),
        "provider": "OpenRouter",
        "endpoint": endpoint,
        "judge_profile": judge_profile,
        "judge_model": judge_model,
        "allowed_response_models": list(allowed_response_models),
        "judge_template_sha256": judge_template_sha256,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_attempts_per_generation": max_attempts,
        "concurrency": concurrency,
        "max_retry_after": max_retry_after,
        "proxy": dict(proxy),
        "route": dict(route) if route is not None else None,
        "route_mode": route_mode,
        "provider_routing": dict(provider_routing),
        "reasoning": dict(reasoning),
        "response_format": dict(response_format),
        "matrix": dict(matrix),
    }


def _validate_manifest_shape(path: Path, manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise OpenRouterJudgeError(f"OpenRouter manifest {path} has an unsupported schema")
    for name in ("plan_history", "approvals", "amendments", "events"):
        if not isinstance(manifest.get(name), list):
            raise OpenRouterJudgeError(f"OpenRouter manifest {path} has invalid {name}")
    history = manifest["plan_history"]
    if not history or manifest.get("plan") != history[-1]:
        raise OpenRouterJudgeError(f"OpenRouter manifest {path} has an invalid current plan/history chain")
    hashes: list[str] = []
    for index, plan in enumerate(history, start=1):
        if not isinstance(plan, Mapping) or not isinstance(plan.get("document"), Mapping):
            raise OpenRouterJudgeError(f"OpenRouter manifest {path} plan history entry {index} is invalid")
        document = dict(plan["document"])
        if plan.get("plan_sha256") != _sha256_json(document):
            raise OpenRouterJudgeError(f"OpenRouter manifest {path} plan history entry {index} has a bad hash")
        core_document = {key: value for key, value in document.items() if key != "max_attempts_per_generation"}
        if plan.get("core_plan_sha256") != _sha256_json(core_document):
            raise OpenRouterJudgeError(f"OpenRouter manifest {path} plan history entry {index} has a bad core hash")
        hashes.append(plan["plan_sha256"])
    if len(set(hashes)) != len(hashes):
        raise OpenRouterJudgeError(f"OpenRouter manifest {path} repeats a plan hash")
    if len(manifest["amendments"]) != len(history) - 1:
        raise OpenRouterJudgeError(f"OpenRouter manifest {path} has an incomplete amendment chain")
    for index, amendment in enumerate(manifest["amendments"], start=1):
        old_plan = history[index - 1]
        new_plan = history[index]
        if not isinstance(amendment, Mapping) or {
            "amendment_index": amendment.get("amendment_index"),
            "confirmation": amendment.get("confirmation"),
            "old_plan_sha256": amendment.get("old_plan_sha256"),
            "new_plan_sha256": amendment.get("new_plan_sha256"),
            "core_plan_sha256": amendment.get("core_plan_sha256"),
            "old_max_attempts_per_generation": amendment.get("old_max_attempts_per_generation"),
            "new_max_attempts_per_generation": amendment.get("new_max_attempts_per_generation"),
            "reviewed_plan_sha256": amendment.get("reviewed_plan_sha256"),
        } != {
            "amendment_index": index,
            "confirmation": "--yes",
            "old_plan_sha256": old_plan["plan_sha256"],
            "new_plan_sha256": new_plan["plan_sha256"],
            "core_plan_sha256": old_plan["core_plan_sha256"],
            "old_max_attempts_per_generation": old_plan["max_attempts_per_generation"],
            "new_max_attempts_per_generation": new_plan["max_attempts_per_generation"],
            "reviewed_plan_sha256": new_plan["plan_sha256"],
        }:
            raise OpenRouterJudgeError(f"OpenRouter manifest {path} amendment {index} is invalid")
        if (
            old_plan["core_plan_sha256"] != new_plan["core_plan_sha256"]
            or new_plan["max_attempts_per_generation"] <= old_plan["max_attempts_per_generation"]
        ):
            raise OpenRouterJudgeError(f"OpenRouter manifest {path} amendment {index} is not a ceiling-only increase")
    return manifest


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OpenRouterJudgeError(f"invalid OpenRouter manifest {path}: {exc}") from exc
    return _validate_manifest_shape(path, manifest)


def _new_manifest(expected_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at": _utc_now(),
        "updated_at": None,
        "plan": dict(expected_plan),
        "plan_history": [dict(expected_plan)],
        "approvals": [],
        "amendments": [],
        "events": [],
    }


def _load_or_create_manifest(path: Path, expected_plan: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        return _read_manifest(path)
    return _new_manifest(expected_plan)


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_atomic_bytes(path, payload)
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != payload:
        raise OpenRouterJudgeError(f"atomic OpenRouter manifest verification failed: {path}")


async def _judge_generations(
    generations: Sequence[Mapping[str, Any]],
    *,
    template: str,
    attempt_log_path: str | Path,
    output_path: str | Path,
    api_key: str | None,
    manifest_path: str | Path | None = None,
    judge_template_sha256: str = PAPER_JUDGE_TEMPLATE_SHA256,
    judge_profile: str = DEFAULT_JUDGE_PROFILE,
    judge_model: str | None = None,
    allowed_response_models: Sequence[str] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    concurrency: int | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_url: str = OPENROUTER_BASE_URL,
    proxy: str | None = None,
    expected_exit_instance_id: str | None = None,
    expected_exit_ssh_host: str | None = None,
    expected_exit_ssh_port: int | None = None,
    route_country_code: str | None = None,
    route_attested_at: str | None = None,
    route_attested_by: str | None = None,
    route_attestation_sha256: str | None = None,
    route_attestation_evidence: bytes | None = None,
    expected_model_keys: Sequence[str] | None = None,
    confirm_paid: bool = False,
    expected_plan_sha256: str | None = None,
    amend_attempt_ceiling: bool = False,
    rescore_paid_errors: bool = False,
    max_retry_after: float | None = DEFAULT_MAX_RETRY_AFTER,
    dry_run: bool = False,
    client: httpx.AsyncClient | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    _enforce_exact_paid_matrix: bool = True,
    _enforce_registered_profile: bool = True,
) -> dict[str, Any]:
    """Judge all terminal generation successes under one locked audit lifecycle."""

    profile = _judge_profile(judge_profile)
    judge_model = profile["model"] if judge_model is None else judge_model
    temperature = profile["temperature"] if temperature is None else temperature
    max_tokens = profile["max_tokens"] if max_tokens is None else max_tokens
    concurrency = profile["concurrency"] if concurrency is None else concurrency
    if not isinstance(judge_model, str) or not judge_model:
        raise ValueError("judge_model must be a non-empty string")
    if _enforce_registered_profile:
        drift = {
            "judge_model": (judge_model, profile["model"]),
            "temperature": (temperature, profile["temperature"]),
            "max_tokens": (max_tokens, profile["max_tokens"]),
            "concurrency": (concurrency, profile["concurrency"]),
            "max_retry_after": (max_retry_after, profile["max_retry_after"]),
        }
        changed = {name: values for name, values in drift.items() if values[0] != values[1]}
        if changed:
            raise ValueError(f"paid judge profile {judge_profile!r} has immutable registered settings; drift={changed}")
    if (
        not isinstance(judge_template_sha256, str)
        or len(judge_template_sha256) != 64
        or any(character not in "0123456789abcdef" for character in judge_template_sha256)
    ):
        raise ValueError("judge_template_sha256 must be a 64-character SHA-256 digest")
    actual_template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if actual_template_sha256 != judge_template_sha256:
        raise ValueError(
            f"judge template digest mismatch: expected {judge_template_sha256}, got {actual_template_sha256}"
        )
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be a finite number >= 0")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be an integer >= 1")
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError("concurrency must be an integer >= 1")
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
        raise ValueError("max_attempts must be an integer >= 1")
    if max_retry_after is not None and (
        isinstance(max_retry_after, bool)
        or not isinstance(max_retry_after, (int, float))
        or not math.isfinite(max_retry_after)
        or max_retry_after < 0
    ):
        raise ValueError("max_retry_after must be null or a number >= 0")
    if (
        not isinstance(confirm_paid, bool)
        or not isinstance(amend_attempt_ceiling, bool)
        or not isinstance(rescore_paid_errors, bool)
        or not isinstance(dry_run, bool)
    ):
        raise TypeError("paid-control flags must be booleans")
    if dry_run and confirm_paid:
        raise ValueError("dry_run must not include paid confirmation")
    if dry_run and amend_attempt_ceiling:
        raise ValueError("dry_run reports an amendment requirement but must not authorize one")
    allowed_models = _allowed_response_models(profile, allowed_response_models)
    route_mode = profile["route_mode"]
    proxy_metadata = _proxy_provenance(proxy, route_mode=route_mode)
    route_values = (
        expected_exit_instance_id,
        expected_exit_ssh_host,
        expected_exit_ssh_port,
        route_country_code,
        route_attested_at,
        route_attested_by,
        route_attestation_sha256,
        route_attestation_evidence,
    )
    if route_mode == "direct" and any(value is not None for value in route_values):
        raise ValueError("direct judge profiles prohibit proxy and route-attestation fields or evidence")
    route_metadata = _route_provenance(
        expected_exit_instance_id=expected_exit_instance_id,
        expected_exit_ssh_host=expected_exit_ssh_host,
        expected_exit_ssh_port=expected_exit_ssh_port,
        route_country_code=route_country_code,
        route_attested_at=route_attested_at,
        route_attested_by=route_attested_by,
        route_attestation_sha256=route_attestation_sha256,
        required=route_mode == "muse_us_proxy",
    )
    if route_mode == "muse_us_proxy":
        assert route_metadata is not None
        _validated_route_evidence(
            route_attestation_evidence,
            evidence_sha256=route_attestation_sha256,
            proxy=proxy_metadata,
            route=route_metadata,
        )
    endpoint = _sanitized_endpoint(base_url)
    if endpoint != f"{OPENROUTER_BASE_URL}{OPENROUTER_CHAT_ENDPOINT}":
        raise ValueError("registered judge profiles require the exact pinned OpenRouter chat-completions endpoint")
    selected = select_successful_generations(generations)
    validated = [validate_generation(row, index=index) for index, row in enumerate(selected, start=1)]
    scope_model_keys = _expected_qwen_model_keys(
        expected_model_keys,
        required=_enforce_exact_paid_matrix,
    )
    if _enforce_exact_paid_matrix:
        assert scope_model_keys is not None
        matrix_metadata = _validate_exact_qwen_matrix(validated, expected_model_keys=scope_model_keys)
    else:
        observed_model_keys = tuple(sorted({row["model_key"] for row in validated}))
        scope_model_keys = tuple(expected_model_keys) if expected_model_keys is not None else observed_model_keys
        matrix_metadata = {
            "model_keys": list(scope_model_keys),
            "generation_count": len(validated),
            "generations_per_model": None,
            "diagnostic_test_scope": True,
        }
    by_id = {custom_id_for_generation(row): row for row in validated}
    if len(by_id) != len(validated):
        raise ValueError("duplicate generation keys")
    attempt_path = Path(attempt_log_path)
    normalized_path = Path(output_path)
    lifecycle_path = (
        Path(manifest_path)
        if manifest_path is not None
        else attempt_path.with_name(attempt_path.name + ".manifest.json")
    )
    if len({attempt_path.resolve(), normalized_path.resolve(), lifecycle_path.resolve()}) != 3:
        raise ValueError("attempt_log_path, output_path, and manifest_path must be different paths")
    request_protocols: dict[str, dict[str, Any]] = {}
    for custom_id, generation in by_id.items():
        judge_prompt = render_judge_prompt(
            template,
            task=generation["prompt"],
            reasoning=generation["reasoning"],
            answer=generation["answer"],
        )
        generation_record_sha256 = _normalized_generation_fields(generation)["generation_record_sha256"]
        request_protocols[custom_id] = {
            "provider": "OpenRouter",
            "endpoint": endpoint,
            "judge_profile": judge_profile,
            "model": judge_model,
            "allowed_response_models": list(allowed_models),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "provider_routing": dict(profile["provider_routing"]),
            "reasoning": dict(profile["reasoning"]),
            "response_format": dict(profile["response_format"]),
            "route_mode": route_mode,
            "prompt_sha256": hashlib.sha256(judge_prompt.encode("utf-8")).hexdigest(),
            "generation_record_sha256": generation_record_sha256,
            "proxy": proxy_metadata,
            "route": route_metadata,
        }
    plan_document = {
        "schema": PLAN_SCHEMA,
        "provider": "OpenRouter",
        "endpoint": endpoint,
        "judge_profile": judge_profile,
        "judge_model": judge_model,
        "allowed_response_models": list(allowed_models),
        "judge_template_sha256": judge_template_sha256,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_attempts_per_generation": max_attempts,
        "concurrency": concurrency,
        "max_retry_after": max_retry_after,
        "proxy": proxy_metadata,
        "route": route_metadata,
        "route_mode": route_mode,
        "provider_routing": dict(profile["provider_routing"]),
        "reasoning": dict(profile["reasoning"]),
        "response_format": dict(profile["response_format"]),
        "matrix": matrix_metadata,
        "requests": [
            {"custom_id": custom_id, "request": request_protocols[custom_id]} for custom_id in sorted(request_protocols)
        ],
    }
    plan_sha256 = _sha256_json(plan_document)
    core_plan_document = {key: value for key, value in plan_document.items() if key != "max_attempts_per_generation"}
    core_plan_sha256 = _sha256_json(core_plan_document)
    if expected_plan_sha256 is not None and expected_plan_sha256 != plan_sha256:
        raise OpenRouterJudgeError(
            f"deterministic plan hash mismatch: expected {expected_plan_sha256}, calculated {plan_sha256}"
        )
    manifest_plan = _manifest_plan(
        plan_sha256=plan_sha256,
        request_protocols=request_protocols,
        judge_profile=judge_profile,
        judge_model=judge_model,
        allowed_response_models=allowed_models,
        judge_template_sha256=judge_template_sha256,
        temperature=temperature,
        max_tokens=max_tokens,
        max_attempts=max_attempts,
        concurrency=concurrency,
        max_retry_after=max_retry_after,
        endpoint=endpoint,
        proxy=proxy_metadata,
        route=route_metadata,
        route_mode=route_mode,
        provider_routing=profile["provider_routing"],
        reasoning=profile["reasoning"],
        response_format=profile["response_format"],
        matrix=matrix_metadata,
        plan_document=plan_document,
        core_plan_sha256=core_plan_sha256,
    )

    with _run_lock(attempt_path):
        existing_manifest = _read_manifest(lifecycle_path) if lifecycle_path.exists() else None
        amendment_required = False
        previous_plan: Mapping[str, Any] | None = None
        if existing_manifest is not None:
            previous_plan = existing_manifest["plan"]
            if previous_plan != manifest_plan:
                if (
                    previous_plan.get("core_plan_sha256") != core_plan_sha256
                    or not isinstance(previous_plan.get("max_attempts_per_generation"), int)
                    or max_attempts <= previous_plan["max_attempts_per_generation"]
                ):
                    raise OpenRouterJudgeError(
                        f"OpenRouter manifest {lifecycle_path} is bound to a different deterministic core plan; "
                        "only a reviewed attempt-ceiling increase can amend a paid lifecycle"
                    )
                amendment_required = True
            allowed_attempt_plan_hashes = [plan["plan_sha256"] for plan in existing_manifest["plan_history"]]
        else:
            allowed_attempt_plan_hashes = [plan_sha256]
        prior_attempts, successes, attempt_counts, paid_error_attempts = _load_attempt_history(
            attempt_path,
            expected_requests=request_protocols,
            judge_template_sha256=judge_template_sha256,
            allowed_plan_sha256s=allowed_attempt_plan_hashes,
        )
        if prior_attempts and not lifecycle_path.exists():
            raise OpenRouterJudgeError(
                f"attempt history exists without its required paid approval manifest: {lifecycle_path}"
            )
        if existing_manifest is not None:
            approvals = existing_manifest["approvals"]
            for index, row in enumerate(prior_attempts, start=1):
                approval_index = row["approval"]["approval_index"]
                if approval_index > len(approvals):
                    raise OpenRouterJudgeError(f"attempt {index} references a missing manifest approval")
                manifest_approval = approvals[approval_index - 1]
                if (
                    not isinstance(manifest_approval, Mapping)
                    or manifest_approval.get("confirmation") != "--yes"
                    or manifest_approval.get("plan_sha256") != row["plan_sha256"]
                    or manifest_approval.get("reviewed_plan_sha256") != row["plan_sha256"]
                    or manifest_approval.get("reviewed_plan_hash_verified") is not True
                ):
                    raise OpenRouterJudgeError(f"attempt {index} does not match its manifest approval")
        pending_ids = sorted(set(by_id) - set(successes))
        blocked_paid_ids = sorted(custom_id for custom_id in pending_ids if paid_error_attempts.get(custom_id))
        exhausted_ids = sorted(
            custom_id for custom_id in pending_ids if attempt_counts.get(custom_id, 0) >= max_attempts
        )
        summary = {
            "schema": "ctm.eval_awareness.figure6.openrouter_run.v3",
            "provider": "OpenRouter",
            "endpoint": endpoint,
            "judge_profile": judge_profile,
            "judge_profile_label": profile["label"],
            "judge_model": judge_model,
            "allowed_response_models": list(allowed_models),
            "judge_template_sha256": judge_template_sha256,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "concurrency": concurrency,
            "max_attempts": max_attempts,
            "max_retry_after": max_retry_after,
            "proxy": proxy_metadata,
            "route": route_metadata,
            "route_mode": route_mode,
            "provider_routing": dict(profile["provider_routing"]),
            "reasoning": dict(profile["reasoning"]),
            "response_format": dict(profile["response_format"]),
            "matrix": matrix_metadata,
            "generation_count": len(by_id),
            "resumed_successes": len(successes),
            "pending": len(pending_ids),
            "paid_error_generations": len(blocked_paid_ids),
            "rescore_paid_errors": rescore_paid_errors,
            "dry_run": dry_run,
            "paid_confirmation_required": True,
            "plan_sha256": plan_sha256,
            "core_plan_sha256": core_plan_sha256,
            "amendment_required": amendment_required,
            "previous_plan_sha256": previous_plan.get("plan_sha256") if previous_plan is not None else None,
            "previous_max_attempts": (
                previous_plan.get("max_attempts_per_generation") if previous_plan is not None else None
            ),
            "exhausted_generations": len(exhausted_ids),
            "manifest": str(lifecycle_path),
        }
        if dry_run:
            return summary
        if not pending_ids and existing_manifest is not None:
            completed_judgments = [successes[custom_id] for custom_id in sorted(successes)]
            completed_digest = _write_or_verify_output(normalized_path, completed_judgments)
            completion_events = [
                event
                for event in existing_manifest["events"]
                if isinstance(event, Mapping)
                and event.get("event") == "run_completed"
                and event.get("normalized_output_sha256") == completed_digest
                and event.get("judgment_count") == len(completed_judgments)
            ]
            if len(completion_events) > 1:
                raise OpenRouterJudgeError("manifest contains duplicate completion events for the normalized output")
            if len(completion_events) == 1:
                event = completion_events[0]
                return {
                    **summary,
                    "pending": 0,
                    "completed": len(completed_judgments),
                    "normalized_output": str(normalized_path),
                    "normalized_output_sha256": completed_digest,
                    "approval_index": event["approval_index"],
                    "idempotent_resume": True,
                }
        if not confirm_paid:
            raise OpenRouterJudgeError("paid OpenRouter judging requires explicit --yes confirmation")
        if expected_plan_sha256 is None:
            raise OpenRouterJudgeError(
                "paid OpenRouter judging requires expected_plan_sha256 from a separately reviewed dry run"
            )
        if not isinstance(api_key, str) or not api_key:
            raise OpenRouterJudgeError("OPENROUTER_API_KEY must be set for a paid judge run")
        if amendment_required and not amend_attempt_ceiling:
            raise OpenRouterJudgeError(
                "the reviewed plan raises an exhausted retry ceiling; inspect the dry-run plan and pass "
                "--amend-attempt-ceiling with --yes to record the explicit amendment"
            )
        if amend_attempt_ceiling and not amendment_required:
            raise OpenRouterJudgeError("--amend-attempt-ceiling is valid only for a strict ceiling-only plan increase")
        if exhausted_ids and not amendment_required:
            raise OpenRouterJudgeError(
                f"judge attempts were exhausted for {len(exhausted_ids)} generations; rerun a dry plan with a "
                "strictly higher --max-attempts, review its new hash, then authorize --amend-attempt-ceiling"
            )
        if blocked_paid_ids and not rescore_paid_errors:
            raise OpenRouterJudgeError(
                "paid HTTP-200 judge errors are present; inspect the raw attempts and pass "
                "--rescore-paid-errors with --yes to authorize another paid request"
            )

        manifest = existing_manifest or _new_manifest(manifest_plan)
        if amendment_required:
            assert previous_plan is not None
            amendment = {
                "amendment_index": len(manifest["amendments"]) + 1,
                "amended_at": _utc_now(),
                "confirmation": "--yes",
                "old_plan_sha256": previous_plan["plan_sha256"],
                "new_plan_sha256": plan_sha256,
                "core_plan_sha256": core_plan_sha256,
                "old_max_attempts_per_generation": previous_plan["max_attempts_per_generation"],
                "new_max_attempts_per_generation": max_attempts,
                "reviewed_plan_sha256": expected_plan_sha256,
                "reviewed_plan_hash_verified": expected_plan_sha256 == plan_sha256,
                "preserved_attempt_count": len(prior_attempts),
                "preserved_success_count": len(successes),
            }
            manifest["plan_history"].append(dict(manifest_plan))
            manifest["plan"] = dict(manifest_plan)
            manifest["amendments"].append(amendment)
            manifest["events"].append(
                {
                    "at": _utc_now(),
                    "event": "attempt_ceiling_amended",
                    "amendment_index": amendment["amendment_index"],
                    "old_plan_sha256": previous_plan["plan_sha256"],
                    "new_plan_sha256": plan_sha256,
                }
            )
        approval = {
            "approved_at": _utc_now(),
            "confirmation": "--yes",
            "plan_sha256": plan_sha256,
            "reviewed_plan_sha256": expected_plan_sha256,
            "reviewed_plan_hash_verified": expected_plan_sha256 == plan_sha256,
            "core_plan_sha256": core_plan_sha256,
            "expected_model_keys": list(scope_model_keys),
            "pending_before_run": len(pending_ids),
            "resumed_successes": len(successes),
            "rescore_paid_errors": rescore_paid_errors,
            "authorized_paid_error_attempts": {
                custom_id: list(paid_error_attempts[custom_id]) for custom_id in blocked_paid_ids
            },
            "authorized_attempt_ceiling": {custom_id: max_attempts for custom_id in blocked_paid_ids},
            "amendment_index": len(manifest["amendments"]) if amendment_required else None,
        }
        manifest["approvals"].append(approval)
        approval_index = len(manifest["approvals"])
        manifest["events"].append(
            {
                "at": _utc_now(),
                "event": "paid_run_approved",
                "approval_index": approval_index,
                "plan_sha256": plan_sha256,
            }
        )
        _save_manifest(lifecycle_path, manifest)
        attempt_approval = {
            "confirmation": "--yes",
            "approval_index": approval_index,
            "manifest": str(lifecycle_path),
            "plan_sha256": plan_sha256,
            "reviewed_plan_sha256": expected_plan_sha256,
        }

        created_client = False
        if pending_ids and client is None:
            try:
                client = httpx.AsyncClient(
                    timeout=httpx.Timeout(300.0, connect=30.0),
                    proxy=proxy,
                    trust_env=False,
                )
            except ImportError as exc:
                manifest["events"].append(
                    {
                        "at": _utc_now(),
                        "event": "run_failed",
                        "approval_index": approval_index,
                        "error_type": type(exc).__name__,
                        "message": "SOCKS proxy support is unavailable",
                    }
                )
                _save_manifest(lifecycle_path, manifest)
                raise OpenRouterJudgeError("SOCKS proxy support requires the 'socksio' package") from exc
            created_client = True
        request_client = client
        semaphore = asyncio.Semaphore(concurrency)
        append_lock = asyncio.Lock()
        permanent_failure = asyncio.Event()

        async def append(row: Mapping[str, Any]) -> None:
            async with append_lock:
                _append_attempt(attempt_path, row)

        async def score(custom_id: str) -> tuple[str, dict[str, Any]]:
            generation = by_id[custom_id]
            first_attempt = 1 + attempt_counts.get(custom_id, 0)
            attempt_ceiling = max_attempts
            if first_attempt > attempt_ceiling:
                raise OpenRouterJudgeError(
                    f"judge attempts were already exhausted for {custom_id}; a reviewed attempt-ceiling "
                    "amendment is required before another paid request"
                )
            judge_prompt = render_judge_prompt(
                template,
                task=generation["prompt"],
                reasoning=generation["reasoning"],
                answer=generation["answer"],
            )
            request_protocol = request_protocols[custom_id]
            request_body = {
                "model": judge_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "provider": dict(profile["provider_routing"]),
                "reasoning": dict(profile["reasoning"]),
                "response_format": dict(profile["response_format"]),
                "messages": [{"role": "system", "content": judge_prompt}],
            }
            paid_attempt_numbers = list(paid_error_attempts.get(custom_id, ()))
            for attempt in range(first_attempt, attempt_ceiling + 1):
                if permanent_failure.is_set():
                    raise PermanentOpenRouterError("another request encountered a permanent account error")
                started_at = _utc_now()
                response: httpx.Response | None = None
                response_audit: dict[str, Any] = {"status_code": None}
                retryable = False
                retry_after_delay: float | None = None
                error: dict[str, Any] | None = None
                choice: Mapping[str, Any] | None = None
                body: Mapping[str, Any] | None = None
                try:
                    assert request_client is not None
                    async with semaphore:
                        response = await request_client.post(
                            endpoint,
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json=request_body,
                        )
                    if response.status_code != 200:
                        retryable = _is_retryable_http_status(response.status_code)
                        retry_after_delay = _retry_after_seconds(response.headers) if retryable else None
                        error = {
                            "type": "http",
                            "status_code": response.status_code,
                            "body": _safe_error_body(response),
                            "retry_after_seconds": retry_after_delay,
                        }
                        response_audit = {"status_code": response.status_code}
                        if response.status_code in PERMANENT_ACCOUNT_HTTP_STATUSES:
                            permanent_failure.set()
                    else:
                        response_audit = _raw_http_200_response(response)
                        parsed_body = response.json()
                        if not isinstance(parsed_body, Mapping):
                            raise ValueError("OpenRouter response body must be an object")
                        body = parsed_body
                        response_model = body.get("model")
                        if not isinstance(response_model, str) or response_model not in allowed_models:
                            raise ValueError(
                                f"OpenRouter response model {response_model!r} is not in the allowed identities "
                                f"{list(allowed_models)!r}"
                            )
                        response_id = body.get("id")
                        if not isinstance(response_id, str) or not response_id:
                            raise ValueError("OpenRouter response id must be a non-empty string")
                        request_id = response.headers.get("x-request-id") or response_id
                        if not isinstance(request_id, str) or not request_id:
                            raise ValueError("OpenRouter request identity must be a non-empty string")
                        content, choice = _message_content(body)
                        raw_object = parse_judge_json(content)
                        conclusions = normalize_judge_object(raw_object)
                        judgment = {
                            **_normalized_generation_fields(generation),
                            "custom_id": custom_id,
                            "judge_profile": judge_profile,
                            "judge_profile_label": profile["label"],
                            "judge_model": judge_model,
                            "judge_template_sha256": judge_template_sha256,
                            "judge_max_completion_tokens": max_tokens,
                            "judge_temperature": temperature,
                            **conclusions,
                            "judge_status": "ok",
                            "judge_provider": "OpenRouter",
                            "judge_requested_model": judge_model,
                            "judge_allowed_response_models": list(allowed_models),
                            "judge_response_model": response_model,
                            "judge_response_id": response_id,
                            "judge_request_id": request_id,
                            "judge_usage": body.get("usage"),
                            "judge_endpoint": endpoint,
                            "judge_provider_routing": dict(profile["provider_routing"]),
                            "judge_reasoning": dict(profile["reasoning"]),
                            "judge_response_format": dict(profile["response_format"]),
                            "judge_route_mode": route_mode,
                            "judge_proxy": proxy_metadata,
                            "judge_route": route_metadata,
                            "judge_plan_sha256": plan_sha256,
                            "raw_judge_object": raw_object,
                            "raw_judge_body": dict(body),
                        }
                        success_row: dict[str, Any] = {
                            "schema": ATTEMPT_SCHEMA,
                            "custom_id": custom_id,
                            "attempt": attempt,
                            "started_at": started_at,
                            "completed_at": _utc_now(),
                            "status": "success",
                            "retryable": False,
                            "plan_sha256": plan_sha256,
                            "approval": attempt_approval,
                            "request": request_protocol,
                            "response": {
                                "status_code": response.status_code,
                                "id": body.get("id"),
                                "model": response_model,
                                "provider": body.get("provider"),
                                "finish_reason": choice.get("finish_reason"),
                                "usage": body.get("usage"),
                            },
                            "judgment": judgment,
                        }
                        if paid_attempt_numbers and rescore_paid_errors:
                            success_row["rescore_authorization"] = {
                                "flag": "--rescore-paid-errors",
                                "approval_index": approval_index,
                                "paid_error_attempts": list(paid_attempt_numbers),
                            }
                        await append(success_row)
                        return custom_id, judgment
                except httpx.TransportError as exc:
                    retryable = True
                    error = {"type": type(exc).__name__, "message": str(exc)}
                except (json.JSONDecodeError, ValueError) as exc:
                    if response is None or response.status_code != 200:
                        raise
                    retryable = rescore_paid_errors
                    paid_attempt_numbers.append(attempt)
                    error = {
                        "type": "paid_response_validation",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                assert error is not None
                row: dict[str, Any] = {
                    "schema": ATTEMPT_SCHEMA,
                    "custom_id": custom_id,
                    "attempt": attempt,
                    "started_at": started_at,
                    "completed_at": _utc_now(),
                    "status": "error",
                    "retryable": retryable,
                    "plan_sha256": plan_sha256,
                    "approval": attempt_approval,
                    "request": request_protocol,
                    "response": response_audit,
                    "error": error,
                }
                if paid_attempt_numbers and rescore_paid_errors:
                    row["rescore_authorization"] = {
                        "flag": "--rescore-paid-errors",
                        "approval_index": approval_index,
                        "paid_error_attempts": list(paid_attempt_numbers),
                    }
                await append(row)
                if response is not None and response.status_code in PERMANENT_ACCOUNT_HTTP_STATUSES:
                    raise PermanentOpenRouterError(
                        f"OpenRouter account request failed with HTTP {response.status_code}"
                    )
                if response is not None and response.status_code == 200 and not rescore_paid_errors:
                    raise OpenRouterJudgeError(
                        f"paid response validation failed for {custom_id}; raw response preserved in {attempt_path}; "
                        "inspect it and rerun with --rescore-paid-errors plus --yes to authorize another paid request"
                    )
                if not retryable:
                    raise OpenRouterJudgeError(f"non-retryable judge failure for {custom_id}: {error}")
                if attempt >= attempt_ceiling:
                    raise OpenRouterJudgeError(f"judge attempts exhausted for {custom_id}: {error}")
                if (
                    retry_after_delay is not None
                    and max_retry_after is not None
                    and retry_after_delay > max_retry_after
                ):
                    raise OpenRouterJudgeError(
                        f"OpenRouter Retry-After {retry_after_delay} seconds exceeds configured maximum "
                        f"{max_retry_after}; refusing to truncate it"
                    )
                delay = retry_after_delay if retry_after_delay is not None else float(2 ** (attempt - 1))
                await sleep(delay)
            raise AssertionError("unreachable")

        results: list[tuple[str, dict[str, Any]]] = []
        try:
            for offset in range(0, len(pending_ids), concurrency):
                chunk = pending_ids[offset : offset + concurrency]
                outcomes = await asyncio.gather(
                    *(score(custom_id) for custom_id in chunk),
                    return_exceptions=True,
                )
                failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
                results.extend(outcome for outcome in outcomes if not isinstance(outcome, BaseException))
                if failures:
                    raise failures[0]
            successes.update(dict(results))
            if set(successes) != set(by_id):
                raise OpenRouterJudgeError("judge run ended without one success per generation")
            judgments = [successes[custom_id] for custom_id in sorted(successes)]
            digest = _write_or_verify_output(normalized_path, judgments)
        except BaseException as exc:
            manifest["events"].append(
                {
                    "at": _utc_now(),
                    "event": "run_failed",
                    "approval_index": approval_index,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            _save_manifest(lifecycle_path, manifest)
            raise
        finally:
            if created_client:
                assert request_client is not None
                await request_client.aclose()
        manifest["events"].append(
            {
                "at": _utc_now(),
                "event": "run_completed",
                "approval_index": approval_index,
                "plan_sha256": plan_sha256,
                "core_plan_sha256": core_plan_sha256,
                "endpoint": endpoint,
                "judge_profile": judge_profile,
                "proxy": proxy_metadata,
                "route": route_metadata,
                "route_mode": route_mode,
                "temperature": temperature,
                "provider_routing": dict(profile["provider_routing"]),
                "reasoning": dict(profile["reasoning"]),
                "response_format": dict(profile["response_format"]),
                "judge_model": judge_model,
                "allowed_response_models": list(allowed_models),
                "observed_response_models": sorted({judgment["judge_response_model"] for judgment in judgments}),
                "response_request_ids_sha256": _sha256_json(
                    sorted(judgment["judge_request_id"] for judgment in judgments)
                ),
                "judgment_count": len(judgments),
                "normalized_output": str(normalized_path),
                "normalized_output_sha256": digest,
            }
        )
        _save_manifest(lifecycle_path, manifest)
        return {
            **summary,
            "pending": 0,
            "completed": len(judgments),
            "normalized_output": str(normalized_path),
            "normalized_output_sha256": digest,
            "approval_index": approval_index,
        }


async def judge_generations(
    generations: Sequence[Mapping[str, Any]],
    *,
    template: str,
    attempt_log_path: str | Path,
    output_path: str | Path,
    api_key: str | None,
    manifest_path: str | Path | None = None,
    judge_template_sha256: str = PAPER_JUDGE_TEMPLATE_SHA256,
    judge_profile: str = DEFAULT_JUDGE_PROFILE,
    judge_model: str | None = None,
    allowed_response_models: Sequence[str] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    concurrency: int | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_url: str = OPENROUTER_BASE_URL,
    proxy: str | None = None,
    expected_exit_instance_id: str | None = None,
    expected_exit_ssh_host: str | None = None,
    expected_exit_ssh_port: int | None = None,
    route_country_code: str | None = None,
    route_attested_at: str | None = None,
    route_attested_by: str | None = None,
    route_attestation_sha256: str | None = None,
    route_attestation_evidence: bytes | None = None,
    expected_model_keys: Sequence[str] | None = None,
    confirm_paid: bool = False,
    expected_plan_sha256: str | None = None,
    amend_attempt_ceiling: bool = False,
    rescore_paid_errors: bool = False,
    max_retry_after: float | None = DEFAULT_MAX_RETRY_AFTER,
    dry_run: bool = False,
    client: httpx.AsyncClient | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, Any]:
    """Validate an exact registered-Qwen matrix, then execute its reviewed paid plan."""

    return await _judge_generations(
        generations,
        template=template,
        attempt_log_path=attempt_log_path,
        output_path=output_path,
        api_key=api_key,
        manifest_path=manifest_path,
        judge_template_sha256=judge_template_sha256,
        judge_profile=judge_profile,
        judge_model=judge_model,
        allowed_response_models=allowed_response_models,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=concurrency,
        max_attempts=max_attempts,
        base_url=base_url,
        proxy=proxy,
        expected_exit_instance_id=expected_exit_instance_id,
        expected_exit_ssh_host=expected_exit_ssh_host,
        expected_exit_ssh_port=expected_exit_ssh_port,
        route_country_code=route_country_code,
        route_attested_at=route_attested_at,
        route_attested_by=route_attested_by,
        route_attestation_sha256=route_attestation_sha256,
        route_attestation_evidence=route_attestation_evidence,
        expected_model_keys=expected_model_keys,
        confirm_paid=confirm_paid,
        expected_plan_sha256=expected_plan_sha256,
        amend_attempt_ceiling=amend_attempt_ceiling,
        rescore_paid_errors=rescore_paid_errors,
        max_retry_after=max_retry_after,
        dry_run=dry_run,
        client=client,
        sleep=sleep,
        _enforce_exact_paid_matrix=True,
        _enforce_registered_profile=True,
    )
