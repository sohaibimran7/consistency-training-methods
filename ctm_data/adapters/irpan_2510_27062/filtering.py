"""Pure, auditable vulnerability filtering for paired jailbreak judgments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm_data.adapters.irpan_2510_27062.artifacts import producer_identity, read_artifact, write_artifact
from ctm_data.adapters.irpan_2510_27062.partitions import TRAINING
from ctm_data.adapters.irpan_2510_27062.schema import (
    RecordSchemaError,
    make_derived_record,
    normalize_text,
    require_sha256,
    sha256_json,
    validate_record,
)

FILTER_VERSION = "reconstruction_v1"
RETAIN_REASON = "retained_clean_refused_wrapped_fulfilled"
REJECT_CLEAN_REASON = "rejected_clean_not_refused"
REJECT_WRAPPED_REASON = "rejected_wrapped_not_fulfilled"
REJECT_BOTH_REASON = "rejected_clean_not_refused_and_wrapped_not_fulfilled"


class VulnerabilityFilterError(ValueError):
    """Candidates and judgments cannot be joined into an exact paired filter."""


def vulnerability_decision(clean_verdict: str, wrapped_verdict: str) -> tuple[bool, str]:
    """Apply the paper-reported clean-refused/wrapped-answered rule."""

    allowed = {"fulfilled", "refused", "other"}
    if clean_verdict not in allowed:
        raise VulnerabilityFilterError(f"invalid clean verdict {clean_verdict!r}")
    if wrapped_verdict not in allowed:
        raise VulnerabilityFilterError(f"invalid wrapped verdict {wrapped_verdict!r}")
    clean_refused = clean_verdict == "refused"
    wrapped_fulfilled = wrapped_verdict == "fulfilled"
    if clean_refused and wrapped_fulfilled:
        return True, RETAIN_REASON
    if not clean_refused and not wrapped_fulfilled:
        return False, REJECT_BOTH_REASON
    if not clean_refused:
        return False, REJECT_CLEAN_REASON
    return False, REJECT_WRAPPED_REASON


def build_vulnerability_filter(
    candidate_rows: Sequence[Mapping[str, Any]],
    judgment_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return one audit row per candidate and retained candidate rows.

    The function is deliberately pure: it performs validation, strict joins,
    and row construction only.  There is no fallback classifier and no model
    or network boundary.
    """

    candidates = _unique_records(candidate_rows, expected_type="wrapper_candidate")
    if not candidates:
        raise VulnerabilityFilterError("vulnerability filtering requires wrapper candidates")
    judgments = _unique_records(judgment_rows, expected_type="external_judgment")
    candidate_by_id = {row["example_id"]: row for row in candidates}
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for judgment in judgments:
        payload = judgment["payload"]
        candidate_id = _text(payload, "candidate_id", context=judgment["example_id"])
        condition = _condition(payload, context=judgment["example_id"])
        if candidate_id not in candidate_by_id:
            raise VulnerabilityFilterError(f"judgment references extra candidate {candidate_id!r}")
        per_candidate = grouped.setdefault(candidate_id, {})
        if condition in per_candidate:
            raise VulnerabilityFilterError(f"duplicate {condition} judgment for candidate {candidate_id!r}")
        candidate_digest = _sha256(payload.get("candidate_content_sha256"), field="candidate_content_sha256")
        if candidate_digest != candidate_by_id[candidate_id]["content_sha256"]:
            raise VulnerabilityFilterError(f"stale candidate hash in judgment for {candidate_id!r}")
        _sha256(payload.get("completion_content_sha256"), field="completion_content_sha256")
        verdict = _text(payload, "verdict", context=judgment["example_id"])
        if verdict not in {"fulfilled", "refused", "other"}:
            raise VulnerabilityFilterError(f"invalid verdict {verdict!r} for candidate {candidate_id!r}")
        if payload.get("is_refusal") is not (verdict == "refused"):
            raise VulnerabilityFilterError(f"inconsistent is_refusal flag for candidate {candidate_id!r}")
        if payload.get("is_fulfillment") is not (verdict == "fulfilled"):
            raise VulnerabilityFilterError(f"inconsistent is_fulfillment flag for candidate {candidate_id!r}")
        per_candidate[condition] = judgment

    missing_candidates = sorted(set(candidate_by_id) - set(grouped))
    if missing_candidates:
        raise VulnerabilityFilterError(f"judgments missing candidates: {missing_candidates}")
    for candidate_id, conditions in grouped.items():
        if set(conditions) != {"clean", "wrapped"}:
            raise VulnerabilityFilterError(
                f"candidate {candidate_id!r} requires exactly clean/wrapped judgments; got {sorted(conditions)}"
            )

    audits: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: row["example_id"]):
        candidate_id = candidate["example_id"]
        payload = candidate["payload"]
        clean = grouped[candidate_id]["clean"]
        wrapped = grouped[candidate_id]["wrapped"]
        clean_verdict = clean["payload"]["verdict"]
        wrapped_verdict = wrapped["payload"]["verdict"]
        keep, reason_code = vulnerability_decision(clean_verdict, wrapped_verdict)
        judgment_hashes = {
            "clean": clean["content_sha256"],
            "wrapped": wrapped["content_sha256"],
        }
        completion_hashes = {
            "clean": clean["payload"]["completion_content_sha256"],
            "wrapped": wrapped["payload"]["completion_content_sha256"],
        }
        audit_id = f"{candidate_id}:filter:{FILTER_VERSION}"
        audit = make_derived_record(
            record_type="vulnerability_audit",
            example_id=audit_id,
            source="harmbench",
            source_key=f"{candidate['source_key']}::filter",
            payload={
                "audit_id": audit_id,
                "candidate_id": candidate_id,
                "candidate_content_sha256": candidate["content_sha256"],
                "source_example_id": payload["source_example_id"],
                "clean_verdict": clean_verdict,
                "wrapped_verdict": wrapped_verdict,
                "retained": keep,
                "reason_code": reason_code,
                "filter_version": FILTER_VERSION,
                "rule": {
                    "clean_required": "refused",
                    "wrapped_required": "fulfilled",
                },
                "judgment_content_sha256": judgment_hashes,
                "completion_content_sha256": completion_hashes,
            },
            parent_hashes=[candidate["content_sha256"], clean["content_sha256"], wrapped["content_sha256"]],
            metadata={"pure_filter": True, "model_calls_performed": 0},
        )
        audits.append(audit)
        if keep:
            retained.append(
                make_derived_record(
                    record_type="vulnerable_candidate",
                    example_id=candidate_id,
                    source="harmbench",
                    source_key=candidate["source_key"],
                    payload={
                        "candidate_id": candidate_id,
                        "candidate_content_sha256": candidate["content_sha256"],
                        "source_example_id": payload["source_example_id"],
                        "source_id": payload["source_id"],
                        "wrapper_id": payload["wrapper_id"],
                        "wrapper_family": payload["wrapper_family"],
                        "catalog_version": payload["catalog_version"],
                        "core_request": payload["core_request"],
                        "core_request_sha256": payload["core_request_sha256"],
                        "clean_prompt": payload["clean_prompt"],
                        "clean_prompt_sha256": payload["clean_prompt_sha256"],
                        "wrapped_prompt": payload["wrapped_prompt"],
                        "wrapped_prompt_sha256": payload["wrapped_prompt_sha256"],
                        "shared_suffix": payload["shared_suffix"],
                        "shared_suffix_sha256": payload["shared_suffix_sha256"],
                        "alignment": payload["alignment"],
                        "audit_content_sha256": audit["content_sha256"],
                        "judgment_content_sha256": judgment_hashes,
                        "completion_content_sha256": completion_hashes,
                        "filter_version": FILTER_VERSION,
                        "filter_reason_code": reason_code,
                    },
                    parent_hashes=[
                        candidate["content_sha256"],
                        audit["content_sha256"],
                        clean["content_sha256"],
                        wrapped["content_sha256"],
                    ],
                    metadata={
                        "retained_by_pure_filter": True,
                        "paper_rule": "clean refused and wrapped fulfilled",
                    },
                )
            )
    return audits, retained


def materialize_vulnerability_filter(
    candidate_path: str | Path,
    judgment_path: str | Path,
    audit_output_path: str | Path,
    retained_output_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish the complete audit and the retained subset as immutable artifacts."""

    targets = {Path(audit_output_path).resolve(), Path(retained_output_path).resolve()}
    if len(targets) != 2:
        raise VulnerabilityFilterError("audit and retained outputs must be different paths")
    candidates, candidate_manifest = read_artifact(
        candidate_path,
        expected_kind="wrapper_candidates",
        expected_role=TRAINING,
    )
    judgments, judgment_manifest = read_artifact(
        judgment_path,
        expected_kind="external_judgments",
        expected_role=TRAINING,
    )
    audits, retained = build_vulnerability_filter(candidates, judgments)
    common_config = {
        "filter_version": FILTER_VERSION,
        "clean_required": "refused",
        "wrapped_required": "fulfilled",
        "candidate_manifest_sha256": sha256_json(candidate_manifest),
        "judgment_manifest_sha256": sha256_json(judgment_manifest),
    }
    producer = producer_identity("filter_jailbreak_vulnerabilities", __file__)
    audit_manifest = write_artifact(
        audit_output_path,
        audits,
        artifact_kind="vulnerability_filter_audit",
        role=TRAINING,
        producer=producer,
        config=common_config,
        parent_artifacts=[candidate_path, judgment_path],
        provenance={
            "retained_count": len(retained),
            "rejected_count": len(audits) - len(retained),
            "model_calls_performed": 0,
        },
    )
    retained_manifest = write_artifact(
        retained_output_path,
        retained,
        artifact_kind="retained_vulnerabilities",
        role=TRAINING,
        producer=producer,
        config={**common_config, "audit_manifest_sha256": sha256_json(audit_manifest)},
        parent_artifacts=[candidate_path, judgment_path, audit_output_path],
        provenance={"retained_count": len(retained), "model_calls_performed": 0},
    )
    return audit_manifest, retained_manifest


def _unique_records(rows: Sequence[Mapping[str, Any]], *, expected_type: str) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        try:
            plain = validate_record(row, expected_type=expected_type)
        except RecordSchemaError as exc:
            raise VulnerabilityFilterError(str(exc)) from exc
        identity = plain["example_id"]
        if identity in seen:
            raise VulnerabilityFilterError(f"duplicate {expected_type} example_id {identity!r}")
        seen.add(identity)
        validated.append(plain)
    return validated


def _condition(payload: Mapping[str, Any], *, context: str) -> str:
    condition = _text(payload, "condition", context=context)
    if condition not in {"clean", "wrapped"}:
        raise VulnerabilityFilterError(f"{context} has invalid condition {condition!r}")
    return condition


def _text(payload: Mapping[str, Any], field: str, *, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not normalize_text(value):
        raise VulnerabilityFilterError(f"{context} needs non-empty string {field!r}")
    return normalize_text(value)


def _sha256(value: Any, *, field: str) -> str:
    try:
        return require_sha256(value, field=field)
    except RecordSchemaError as exc:
        raise VulnerabilityFilterError(str(exc)) from exc


__all__ = [
    "FILTER_VERSION",
    "REJECT_BOTH_REASON",
    "REJECT_CLEAN_REASON",
    "REJECT_WRAPPED_REASON",
    "RETAIN_REASON",
    "VulnerabilityFilterError",
    "build_vulnerability_filter",
    "materialize_vulnerability_filter",
    "vulnerability_decision",
]
