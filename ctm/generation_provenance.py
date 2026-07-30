"""Fail-closed provenance for fresh and stale externally generated targets.

The contract binds inputs, example identities, generator identity, prompts,
decoding settings, response order, and parent artifacts.  It is deliberately
generic so other jailbreak adapters can record the same external-generation
boundary without depending on a provider SDK.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _CanonicalJsonError(ValueError):
    """A value cannot be represented in the canonical JSON identity form."""


def _normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise _CanonicalJsonError("text values must be strings")
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_json(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise _CanonicalJsonError(f"unsupported JSON value for stable identity: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _CanonicalJsonError(f"{field} must be a lowercase SHA-256 digest")
    return value


PROVENANCE_SCHEMA_VERSION = 1
FRESH_SELF_GENERATED = "fresh_self_generated"
STALE_EXTERNAL = "stale_external"
EXTERNAL_MODEL = "external_model"
OLDER_MODEL_REVISION = "older_model_revision"

_FRESHNESS_VALUES = {FRESH_SELF_GENERATED, STALE_EXTERNAL}
_STALE_REASONS = {EXTERNAL_MODEL, OLDER_MODEL_REVISION}
_MUTABLE_REVISIONS = {"latest", "main", "master", "head", "default", "current"}
_GENERATOR_KEYS = {
    "generator_id",
    "provider",
    "model",
    "model_revision",
    "model_immutable_date",
    "identity_sha256",
}
_PROVENANCE_KEYS = {
    "provenance_schema_version",
    "input_artifact_sha256",
    "input_manifest_sha256",
    "example_manifest",
    "example_manifest_sha256",
    "example_count",
    "generator_identity",
    "reference_generator_identity",
    "prompt_template_sha256",
    "decoding_parameters",
    "decoding_parameters_sha256",
    "ordered_response_manifest",
    "ordered_response_manifest_sha256",
    "response_count",
    "parent_artifact_sha256",
    "generated_at_utc",
    "target_freshness",
    "stale_reason",
    "metadata",
}


class ProvenanceError(ValueError):
    """Generation provenance is incomplete, inconsistent, or not the expected identity."""


def make_generator_identity(
    *,
    generator_id: str,
    provider: str,
    model: str,
    model_revision: str | None = None,
    model_immutable_date: str | date | None = None,
) -> dict[str, Any]:
    """Build a content-bound generator identity with an immutable revision/date."""

    base = {
        "generator_id": _exact_text(generator_id, field="generator_id"),
        "provider": _exact_text(provider, field="provider"),
        "model": _exact_text(model, field="model"),
        "model_revision": _revision(model_revision),
        "model_immutable_date": _immutable_date(model_immutable_date),
    }
    if base["model_revision"] is None and base["model_immutable_date"] is None:
        raise ProvenanceError("generator identity requires model_revision and/or model_immutable_date")
    return {**base, "identity_sha256": _sha256_json(base)}


def validate_generator_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a generator identity and recompute its identity hash."""

    if not isinstance(identity, Mapping):
        raise ProvenanceError("generator identity must be a mapping")
    missing = sorted(_GENERATOR_KEYS - set(identity))
    extra = sorted(set(identity) - _GENERATOR_KEYS)
    if missing or extra:
        raise ProvenanceError(f"generator identity keys mismatch: missing={missing}, extra={extra}")
    rebuilt = make_generator_identity(
        generator_id=identity["generator_id"],
        provider=identity["provider"],
        model=identity["model"],
        model_revision=identity["model_revision"],
        model_immutable_date=identity["model_immutable_date"],
    )
    recorded = _sha256(identity["identity_sha256"], field="generator_identity.identity_sha256")
    if recorded != rebuilt["identity_sha256"]:
        raise ProvenanceError(
            f"generator identity hash mismatch: recorded {recorded}, computed {rebuilt['identity_sha256']}"
        )
    return rebuilt


def make_response_manifest_entry(*, example_id: str, response: str | bytes) -> dict[str, str]:
    """Hash the exact response bytes for one ordered response-manifest entry."""

    stable_id = _stable_example_id(example_id)
    if isinstance(response, str):
        payload = response.encode("utf-8")
    elif isinstance(response, bytes):
        payload = response
    else:
        raise ProvenanceError("response must be str or bytes")
    return {"example_id": stable_id, "response_sha256": _sha256_bytes(payload)}


def stable_example_manifest_sha256(example_ids: Sequence[str]) -> str:
    """Hash the sorted, duplicate-free stable example-ID manifest."""

    return _sha256_json(_stable_example_manifest(example_ids))


def ordered_response_manifest_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    """Hash response entries in their supplied order without an example-set constraint."""

    return _sha256_json(_ordered_response_manifest(entries, expected_example_ids=None))


def build_generation_provenance(
    *,
    input_artifact_sha256: str,
    input_manifest_sha256: str,
    example_ids: Sequence[str],
    generator_identity: Mapping[str, Any],
    reference_generator_identity: Mapping[str, Any],
    prompt_template_sha256: str,
    decoding_parameters: Mapping[str, Any],
    ordered_response_manifest: Sequence[Mapping[str, Any]],
    parent_artifact_sha256: str,
    generated_at_utc: str | datetime,
    target_freshness: str,
    stale_reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and self-validate the complete external-generation contract.

    ``reference_generator_identity`` is the current/self identity against which
    freshness is asserted.  Fresh targets require exact identity equality; stale
    targets require a distinct old or external identity and an explicit reason.
    """

    examples = _stable_example_manifest(example_ids)
    responses = _ordered_response_manifest(ordered_response_manifest, expected_example_ids=set(examples))
    generator = validate_generator_identity(generator_identity)
    reference = validate_generator_identity(reference_generator_identity)
    freshness, reason = _freshness(target_freshness, stale_reason, generator=generator, reference=reference)
    decoding = _canonical_mapping(decoding_parameters, field="decoding_parameters")
    extra = _canonical_mapping(metadata or {}, field="metadata")
    generated = _timestamp(generated_at_utc)
    _ensure_identity_available_at(generator, generated, field="generator_identity")
    _ensure_identity_available_at(reference, generated, field="reference_generator_identity")
    payload = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "input_artifact_sha256": _sha256(input_artifact_sha256, field="input_artifact_sha256"),
        "input_manifest_sha256": _sha256(input_manifest_sha256, field="input_manifest_sha256"),
        "example_manifest": examples,
        "example_manifest_sha256": _sha256_json(examples),
        "example_count": len(examples),
        "generator_identity": generator,
        "reference_generator_identity": reference,
        "prompt_template_sha256": _sha256(prompt_template_sha256, field="prompt_template_sha256"),
        "decoding_parameters": decoding,
        "decoding_parameters_sha256": _sha256_json(decoding),
        "ordered_response_manifest": responses,
        "ordered_response_manifest_sha256": _sha256_json(responses),
        "response_count": len(responses),
        "parent_artifact_sha256": _sha256(parent_artifact_sha256, field="parent_artifact_sha256"),
        "generated_at_utc": generated,
        "target_freshness": freshness,
        "stale_reason": reason,
        "metadata": extra,
    }
    return validate_generation_provenance(payload)


def build_fresh_target_provenance(
    *,
    input_artifact_sha256: str,
    input_manifest_sha256: str,
    example_ids: Sequence[str],
    generator_identity: Mapping[str, Any],
    prompt_template_sha256: str,
    decoding_parameters: Mapping[str, Any],
    ordered_response_manifest: Sequence[Mapping[str, Any]],
    parent_artifact_sha256: str,
    generated_at_utc: str | datetime,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build provenance explicitly asserting same-identity self-generation."""

    return build_generation_provenance(
        input_artifact_sha256=input_artifact_sha256,
        input_manifest_sha256=input_manifest_sha256,
        example_ids=example_ids,
        generator_identity=generator_identity,
        reference_generator_identity=generator_identity,
        prompt_template_sha256=prompt_template_sha256,
        decoding_parameters=decoding_parameters,
        ordered_response_manifest=ordered_response_manifest,
        parent_artifact_sha256=parent_artifact_sha256,
        generated_at_utc=generated_at_utc,
        target_freshness=FRESH_SELF_GENERATED,
        metadata=metadata,
    )


def build_stale_target_provenance(
    *,
    input_artifact_sha256: str,
    input_manifest_sha256: str,
    example_ids: Sequence[str],
    stale_generator_identity: Mapping[str, Any],
    current_generator_identity: Mapping[str, Any],
    stale_reason: str,
    prompt_template_sha256: str,
    decoding_parameters: Mapping[str, Any],
    ordered_response_manifest: Sequence[Mapping[str, Any]],
    parent_artifact_sha256: str,
    generated_at_utc: str | datetime,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build provenance that can only be labeled ``stale_external``.

    Both the source (old/external) identity and the current/self reference are
    required.  Identical identities fail rather than being silently relabeled as
    fresh.
    """

    return build_generation_provenance(
        input_artifact_sha256=input_artifact_sha256,
        input_manifest_sha256=input_manifest_sha256,
        example_ids=example_ids,
        generator_identity=stale_generator_identity,
        reference_generator_identity=current_generator_identity,
        prompt_template_sha256=prompt_template_sha256,
        decoding_parameters=decoding_parameters,
        ordered_response_manifest=ordered_response_manifest,
        parent_artifact_sha256=parent_artifact_sha256,
        generated_at_utc=generated_at_utc,
        target_freshness=STALE_EXTERNAL,
        stale_reason=stale_reason,
        metadata=metadata,
    )


def validate_generation_provenance(
    provenance: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate all internal bindings and any caller-supplied expected identity.

    ``expected`` is a partial, recursively matched mapping.  It can pin any hash
    or identity field, for example ``{"generator_identity": {"model": ...}}``.
    A valid-looking but unexpected SHA-256 therefore fails closed when pinned.
    """

    if not isinstance(provenance, Mapping):
        raise ProvenanceError("generation provenance must be a mapping")
    missing = sorted(_PROVENANCE_KEYS - set(provenance))
    extra = sorted(set(provenance) - _PROVENANCE_KEYS)
    if missing or extra:
        raise ProvenanceError(f"provenance keys mismatch: missing={missing}, extra={extra}")
    if provenance["provenance_schema_version"] != PROVENANCE_SCHEMA_VERSION:
        raise ProvenanceError(f"provenance_schema_version must be {PROVENANCE_SCHEMA_VERSION}")

    input_artifact = _sha256(provenance["input_artifact_sha256"], field="input_artifact_sha256")
    input_manifest = _sha256(provenance["input_manifest_sha256"], field="input_manifest_sha256")
    examples = _stable_example_manifest(provenance["example_manifest"])
    if list(provenance["example_manifest"]) != examples:
        raise ProvenanceError("example_manifest must be sorted canonically by stable example_id")
    example_hash = _sha256(provenance["example_manifest_sha256"], field="example_manifest_sha256")
    computed_example_hash = _sha256_json(examples)
    if example_hash != computed_example_hash:
        raise ProvenanceError(
            f"example manifest hash mismatch: recorded {example_hash}, computed {computed_example_hash}"
        )
    example_count = _count(provenance["example_count"], field="example_count")
    if example_count != len(examples):
        raise ProvenanceError(f"example_count is {example_count}, expected {len(examples)} from example_manifest")

    generator = validate_generator_identity(provenance["generator_identity"])
    reference = validate_generator_identity(provenance["reference_generator_identity"])
    prompt_hash = _sha256(provenance["prompt_template_sha256"], field="prompt_template_sha256")
    decoding = _canonical_mapping(provenance["decoding_parameters"], field="decoding_parameters")
    if dict(provenance["decoding_parameters"]) != decoding:
        raise ProvenanceError("decoding_parameters are not in normalized canonical form")
    decoding_hash = _sha256(
        provenance["decoding_parameters_sha256"],
        field="decoding_parameters_sha256",
    )
    computed_decoding_hash = _sha256_json(decoding)
    if decoding_hash != computed_decoding_hash:
        raise ProvenanceError(
            f"decoding parameters hash mismatch: recorded {decoding_hash}, computed {computed_decoding_hash}"
        )

    responses = _ordered_response_manifest(
        provenance["ordered_response_manifest"],
        expected_example_ids=set(examples),
    )
    if list(provenance["ordered_response_manifest"]) != responses:
        raise ProvenanceError("ordered_response_manifest is not in canonical entry form")
    response_hash = _sha256(
        provenance["ordered_response_manifest_sha256"],
        field="ordered_response_manifest_sha256",
    )
    computed_response_hash = _sha256_json(responses)
    if response_hash != computed_response_hash:
        raise ProvenanceError(
            f"ordered response manifest hash mismatch: recorded {response_hash}, computed {computed_response_hash}"
        )
    response_count = _count(provenance["response_count"], field="response_count")
    if response_count != len(responses):
        raise ProvenanceError(
            f"response_count is {response_count}, expected {len(responses)} from ordered_response_manifest"
        )

    parent_hash = _sha256(provenance["parent_artifact_sha256"], field="parent_artifact_sha256")
    generated = _timestamp(provenance["generated_at_utc"])
    if provenance["generated_at_utc"] != generated:
        raise ProvenanceError("generated_at_utc must use canonical UTC ISO-8601 form ending in 'Z'")
    _ensure_identity_available_at(generator, generated, field="generator_identity")
    _ensure_identity_available_at(reference, generated, field="reference_generator_identity")
    freshness, stale_reason = _freshness(
        provenance["target_freshness"],
        provenance["stale_reason"],
        generator=generator,
        reference=reference,
    )
    metadata = _canonical_mapping(provenance["metadata"], field="metadata")
    if dict(provenance["metadata"]) != metadata:
        raise ProvenanceError("metadata are not in normalized canonical form")

    validated = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "input_artifact_sha256": input_artifact,
        "input_manifest_sha256": input_manifest,
        "example_manifest": examples,
        "example_manifest_sha256": example_hash,
        "example_count": example_count,
        "generator_identity": generator,
        "reference_generator_identity": reference,
        "prompt_template_sha256": prompt_hash,
        "decoding_parameters": decoding,
        "decoding_parameters_sha256": decoding_hash,
        "ordered_response_manifest": responses,
        "ordered_response_manifest_sha256": response_hash,
        "response_count": response_count,
        "parent_artifact_sha256": parent_hash,
        "generated_at_utc": generated,
        "target_freshness": freshness,
        "stale_reason": stale_reason,
        "metadata": metadata,
    }
    if expected is not None:
        if not isinstance(expected, Mapping):
            raise ProvenanceError("expected provenance identity must be a mapping")
        _assert_expected_subset(validated, expected, path="provenance")
    return copy.deepcopy(validated)


def require_fresh_self_generated(
    provenance: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate provenance and fail unless it is fresh same-identity generation."""

    combined = _merge_expected_freshness(expected, FRESH_SELF_GENERATED)
    return validate_generation_provenance(provenance, expected=combined)


def require_stale_external(
    provenance: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate provenance and fail unless it carries a distinct stale identity."""

    combined = _merge_expected_freshness(expected, STALE_EXTERNAL)
    return validate_generation_provenance(provenance, expected=combined)


def _stable_example_manifest(example_ids: Sequence[str]) -> list[str]:
    if isinstance(example_ids, (str, bytes)) or not isinstance(example_ids, Sequence):
        raise ProvenanceError("example_manifest must be a sequence of stable example_id strings")
    values = [_stable_example_id(value) for value in example_ids]
    if not values:
        raise ProvenanceError("example_manifest must not be empty")
    if len(values) != len(set(values)):
        raise ProvenanceError("example_manifest contains duplicate example_id values")
    return sorted(values)


def _ordered_response_manifest(
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_example_ids: set[str] | None,
) -> list[dict[str, str]]:
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise ProvenanceError("ordered_response_manifest must be a sequence")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ProvenanceError(f"ordered_response_manifest[{index}] must be a mapping")
        required = {"example_id", "response_sha256"}
        missing = sorted(required - set(entry))
        extra = sorted(set(entry) - required)
        if missing or extra:
            raise ProvenanceError(f"ordered_response_manifest[{index}] keys mismatch: missing={missing}, extra={extra}")
        example_id = _stable_example_id(entry["example_id"])
        if example_id in seen:
            raise ProvenanceError(f"ordered_response_manifest has duplicate example_id {example_id!r}")
        seen.add(example_id)
        normalized.append(
            {
                "example_id": example_id,
                "response_sha256": _sha256(
                    entry["response_sha256"],
                    field=f"ordered_response_manifest[{index}].response_sha256",
                ),
            }
        )
    if not normalized:
        raise ProvenanceError("ordered_response_manifest must not be empty")
    if expected_example_ids is not None and seen != expected_example_ids:
        missing_ids = sorted(expected_example_ids - seen)
        extra_ids = sorted(seen - expected_example_ids)
        raise ProvenanceError(f"response/example manifest identity mismatch: missing={missing_ids}, extra={extra_ids}")
    return normalized


def _freshness(
    target_freshness: Any,
    stale_reason: Any,
    *,
    generator: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[str, str | None]:
    if not isinstance(target_freshness, str) or target_freshness not in _FRESHNESS_VALUES:
        raise ProvenanceError(f"target_freshness must be exactly one of {sorted(_FRESHNESS_VALUES)}")
    same_identity = generator == reference
    if target_freshness == FRESH_SELF_GENERATED:
        if stale_reason is not None:
            raise ProvenanceError("fresh_self_generated provenance cannot carry stale_reason")
        if not same_identity:
            raise ProvenanceError(
                "fresh_self_generated requires generator_identity to equal reference_generator_identity"
            )
        return target_freshness, None

    if not isinstance(stale_reason, str) or stale_reason not in _STALE_REASONS:
        raise ProvenanceError(f"stale_external requires stale_reason in {sorted(_STALE_REASONS)}")
    if same_identity:
        raise ProvenanceError("stale_external requires a distinct old/external generator identity")
    if stale_reason == EXTERNAL_MODEL:
        external_fields = ("generator_id", "provider", "model")
        if all(generator[field] == reference[field] for field in external_fields):
            raise ProvenanceError(
                "external_model stale_reason requires a different generator_id, provider, or model; "
                "use older_model_revision for the same model"
            )
    else:
        if generator["provider"] != reference["provider"] or generator["model"] != reference["model"]:
            raise ProvenanceError("older_model_revision requires the same provider and model as the current reference")
        source_date = generator["model_immutable_date"]
        reference_date = reference["model_immutable_date"]
        revisions_differ = generator["model_revision"] != reference["model_revision"]
        dates_establish_older = source_date is not None and reference_date is not None and source_date < reference_date
        if source_date is not None and reference_date is not None and source_date >= reference_date:
            raise ProvenanceError("older_model_revision source immutable date must precede the current model date")
        if not revisions_differ and not dates_establish_older:
            raise ProvenanceError("older_model_revision requires a different revision or an earlier immutable date")
    return target_freshness, stale_reason


def _revision(value: Any) -> str | None:
    if value is None:
        return None
    revision = _exact_text(value, field="model_revision")
    if revision.casefold() in _MUTABLE_REVISIONS:
        raise ProvenanceError(f"model_revision {revision!r} is mutable; pin an immutable revision or date")
    return revision


def _immutable_date(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        raise ProvenanceError("model_immutable_date must be a date, not a date/time")
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or value != value.strip():
        raise ProvenanceError("model_immutable_date must be an ISO YYYY-MM-DD date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ProvenanceError("model_immutable_date must be an ISO YYYY-MM-DD date") from exc
    if parsed.isoformat() != value:
        raise ProvenanceError("model_immutable_date must use canonical ISO YYYY-MM-DD form")
    return value


def _timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value == value.strip():
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ProvenanceError("generated_at_utc must be a timezone-aware ISO date/time") from exc
    else:
        raise ProvenanceError("generated_at_utc must be a timezone-aware datetime or ISO string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProvenanceError("generated_at_utc must include a timezone")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _ensure_identity_available_at(identity: Mapping[str, Any], generated_at: str, *, field: str) -> None:
    immutable_date = identity["model_immutable_date"]
    if immutable_date is None:
        return
    generated_date = datetime.fromisoformat(generated_at).date()
    if date.fromisoformat(immutable_date) > generated_date:
        raise ProvenanceError(f"{field}.model_immutable_date cannot be later than generated_at_utc")


def _canonical_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceError(f"{field} must be a mapping")
    try:
        normalized = _normalize_json(value)
        _sha256_json(normalized)
    except (_CanonicalJsonError, TypeError, ValueError) as exc:
        raise ProvenanceError(f"{field} must contain canonical finite JSON values: {exc}") from exc
    if not isinstance(normalized, dict):
        raise ProvenanceError(f"{field} must normalize to a JSON object")
    return normalized


def _sha256(value: Any, *, field: str) -> str:
    try:
        return _require_sha256(value, field=field)
    except _CanonicalJsonError as exc:
        raise ProvenanceError(str(exc)) from exc


def _stable_example_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProvenanceError("example_id must be a non-empty, exactly formatted stable string")
    return value


def _exact_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProvenanceError(f"{field} must be a non-empty string without surrounding whitespace")
    return value


def _count(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProvenanceError(f"{field} must be a positive integer")
    return value


def _assert_expected_subset(actual: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ProvenanceError(f"expected identity mismatch at {path}: actual value is not a mapping")
        for key, expected_value in expected.items():
            if key not in actual:
                raise ProvenanceError(f"expected identity path {path}.{key} is absent")
            _assert_expected_subset(actual[key], expected_value, path=f"{path}.{key}")
        return
    if actual != expected:
        raise ProvenanceError(f"expected identity mismatch at {path}: expected {expected!r}, got {actual!r}")


def _merge_expected_freshness(expected: Mapping[str, Any] | None, status: str) -> dict[str, Any]:
    if expected is None:
        return {"target_freshness": status}
    if not isinstance(expected, Mapping):
        raise ProvenanceError("expected provenance identity must be a mapping")
    combined = copy.deepcopy(dict(expected))
    supplied = combined.get("target_freshness")
    if supplied is not None and supplied != status:
        raise ProvenanceError(f"expected target_freshness {supplied!r} conflicts with required status {status!r}")
    combined["target_freshness"] = status
    return combined


__all__ = [
    "EXTERNAL_MODEL",
    "FRESH_SELF_GENERATED",
    "OLDER_MODEL_REVISION",
    "PROVENANCE_SCHEMA_VERSION",
    "STALE_EXTERNAL",
    "ProvenanceError",
    "build_fresh_target_provenance",
    "build_generation_provenance",
    "build_stale_target_provenance",
    "make_generator_identity",
    "make_response_manifest_entry",
    "ordered_response_manifest_sha256",
    "require_fresh_self_generated",
    "require_stale_external",
    "stable_example_manifest_sha256",
    "validate_generation_provenance",
    "validate_generator_identity",
]
