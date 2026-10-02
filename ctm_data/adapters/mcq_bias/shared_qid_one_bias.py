"""Shared one-QID-once, one-bias-per-QID exposure protocol.

Every method (BCT, ACT, AttCT, MLPCT, OPCT, RMCT) and model family that accepts
the same frozen shared two-bias pool sees the *same* ordered QIDs, and each QID
is assigned exactly one cue (``wrong_argument`` or ``suggested_answer``) by a
method- and family-independent hash of ``(protocol, seed, question_id)``.

A run is one finite traversal: no epochs, cycling, reassignment or substitution.
The encounter cursor is advanced by *consumed* QID/bias examples, including
batches that yield no learning signal, never by optimizer updates.  Optimizer
updates and rollout multiplicities are recorded separately at each checkpoint.

The absent second cue is never fabricated: :func:`project_datum` returns a datum
carrying only the clean prompt and the assigned cue.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm_data.adapters.mcq_bias.shared_qid_two_bias import (
    BIAS_TYPES,
    _publish_immutable,
    _validate_shared_datum,
)

PROTOCOL = "ctm.mcq_bias.shared_qid_one_bias"
PROTOCOL_VERSION = 1
DATUM_SCHEMA = "ctm.mcq_bias.shared_qid_one_bias_datum"
STATE_SCHEMA = "ctm.mcq_bias.shared_qid_one_bias_cursor"
CHECKPOINT_SCHEMA = "ctm.mcq_bias.shared_qid_one_bias_checkpoint"
PRIMARY_BUDGET = "encountered_qid_bias_examples"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def assign_bias(question_id: str, *, seed: int) -> str:
    """Deterministic, method/family-independent cue assignment for one QID."""

    if type(seed) is not int:
        raise ValueError("assignment seed must be an explicit int")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("question_id must be a nonempty string")
    value = int(digest([PROTOCOL, PROTOCOL_VERSION, seed, question_id]), 16)
    return BIAS_TYPES[value % len(BIAS_TYPES)]


def build_manifest(
    ordered_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    source: Mapping[str, str],
    eligible_qids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Freeze ordering and assignments over shared two-bias datums.

    ``ordered_rows`` must already be in the frozen pool order. ``eligible_qids``
    restricts the population without reordering; every listed QID must exist.
    ``source`` records the identity of the frozen pool (e.g. path/sha256 and
    manifest sha256) so consumers can verify they read the same bytes.
    """

    if not source or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in source.items()):
        raise ValueError("source identity must be a nonempty mapping of strings")
    rows = [_validate_shared_datum(row, location=f"pool row {index}") for index, row in enumerate(ordered_rows)]
    qids = [row["question_id"] for row in rows]
    if not rows or len(set(qids)) != len(qids):
        raise ValueError("pool must contain exactly one datum per question_id")
    if eligible_qids is not None:
        eligible = list(eligible_qids)
        if len(set(eligible)) != len(eligible) or not set(eligible) <= set(qids):
            raise ValueError("eligible_qids must be unique and present in the pool")
        keep = set(eligible)
        rows = [row for row in rows if row["question_id"] in keep]
        if not rows:
            raise ValueError("eligible population is empty")
    assignments = []
    for row in rows:
        bias = assign_bias(row["question_id"], seed=seed)
        assignments.append({
            "question_id": row["question_id"],
            "source_dataset": row["source_dataset"],
            "bias": bias,
            "datum_sha256": digest(row),
            "prompt_sha256": digest(row["variants"][bias]["messages"]),
        })
    counts = {bias: sum(a["bias"] == bias for a in assignments) for bias in BIAS_TYPES}
    manifest = {
        "protocol": PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "source": dict(source),
        "eligible_qids_sha256": None if eligible_qids is None else digest(list(eligible_qids)),
        "population_sha256": digest([a["datum_sha256"] for a in assignments]),
        "assignments": assignments,
        "assignment_sha256": digest(assignments),
        "bias_counts": counts,
        "max_encounters": len(assignments),
        "primary_budget": PRIMARY_BUDGET,
        "cycling_allowed": False,
    }
    return manifest


def manifest_identity(manifest: Mapping[str, Any]) -> str:
    return digest(manifest)


def freeze_manifest(manifest: Mapping[str, Any], directory: str | Path) -> Path:
    """Publish the manifest content-addressed; refuses to overwrite different bytes."""

    validate_manifest(manifest)
    identity = manifest_identity(manifest)
    path = Path(directory) / f"shared-qid-one-bias-n{manifest['max_encounters']}-{identity}.json"
    _publish_immutable(path, canonical_bytes(manifest))
    return path


def load_manifest(path: str | Path, *, expected_sha256: str) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_bytes())
    if manifest_identity(manifest) != expected_sha256:
        raise ValueError("one-bias manifest identity differs from the expected sha256")
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol") != PROTOCOL or manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported one-bias protocol manifest")
    assignments = manifest.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("manifest has no assignments")
    if manifest.get("max_encounters") != len(assignments) or manifest.get("cycling_allowed") is not False:
        raise ValueError("manifest encounter bound is inconsistent")
    if digest(assignments) != manifest.get("assignment_sha256"):
        raise ValueError("assignment hash mismatch")
    qids = [a["question_id"] for a in assignments]
    if len(set(qids)) != len(qids):
        raise ValueError("duplicate question_id in manifest")
    for a in assignments:
        if a["bias"] != assign_bias(a["question_id"], seed=manifest["seed"]):
            raise ValueError(f"assignment for {a['question_id']!r} is not the protocol hash assignment")


def project_datum(datum: Mapping[str, Any], assignment: Mapping[str, Any]) -> dict[str, Any]:
    """Return the clean prompt plus only the assigned cue; never a second arm."""

    row = _validate_shared_datum(datum, location=f"datum {datum.get('question_id')!r}")
    if row["question_id"] != assignment["question_id"] or digest(row) != assignment["datum_sha256"]:
        raise ValueError("datum does not match its frozen assignment")
    bias = assignment["bias"]
    return {
        "datum_schema": DATUM_SCHEMA,
        "schema_version": PROTOCOL_VERSION,
        "question_id": row["question_id"],
        "source_dataset": row["source_dataset"],
        "question": row["question"],
        "ground_truth": row["ground_truth"],
        "prompt_style": row["prompt_style"],
        "clean_messages": copy.deepcopy(row["clean_messages"]),
        "bias": bias,
        "variant": copy.deepcopy(row["variants"][bias]),
        "biased_option": row["variants"][bias]["biased_option"],
    }


def initial_state(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_manifest(manifest)
    return {"schema": STATE_SCHEMA, "manifest_sha256": manifest_identity(manifest), "encounters": 0,
            "pending_claim": None, "ledger": []}


def claim(manifest: Mapping[str, Any], state: Mapping[str, Any], count: int, *, attempt: int) -> tuple[list[dict], dict]:
    """Claim the next ``count`` encounters for one sampled batch attempt.

    A claim is recorded before any sampling happens; it must be committed with
    :func:`commit` (even if the batch produced no learning signal).  An open
    claim blocks further claims: an interrupted, uncertain attempt is never
    silently replayed or reused.
    """

    _check_state(manifest, state)
    if state["pending_claim"] is not None:
        raise ValueError("unresolved claimed encounters; resolve before claiming again")
    cursor = state["encounters"]
    if type(count) is not int or count <= 0:
        raise ValueError("claim count must be a positive int")
    if cursor + count > manifest["max_encounters"]:
        raise ValueError("finite population exhausted; QIDs are never cycled or reused")
    rows = copy.deepcopy(manifest["assignments"][cursor:cursor + count])
    new_state = copy.deepcopy(dict(state))
    new_state["pending_claim"] = {"attempt": attempt, "start": cursor, "end": cursor + count}
    return rows, new_state


def commit(state: Mapping[str, Any], *, attempt: int, optimizer_updates_after: int, rollouts: int | None,
           learning_signal: bool) -> dict[str, Any]:
    """Mark the open claim consumed; encounters advance regardless of signal."""

    pending = state.get("pending_claim")
    if pending is None or pending["attempt"] != attempt:
        raise ValueError("no matching open claim to commit")
    if type(optimizer_updates_after) is not int or optimizer_updates_after < 0:
        raise ValueError("optimizer_updates_after must be a nonnegative int")
    new_state = copy.deepcopy(dict(state))
    new_state["encounters"] = pending["end"]
    new_state["pending_claim"] = None
    new_state["ledger"].append({**pending, "optimizer_updates_after": optimizer_updates_after,
                                "rollouts": rollouts, "learning_signal": bool(learning_signal)})
    return new_state


def checkpoint_record(manifest: Mapping[str, Any], state: Mapping[str, Any], *, method: str,
                      optimizer_updates: int) -> dict[str, Any]:
    """Seal a checkpoint at a committed encounter prefix; both counters recorded."""

    _check_state(manifest, state)
    if state["pending_claim"] is not None:
        raise ValueError("checkpoint requires a committed encounter prefix")
    ledger = state["ledger"]
    if ledger and ledger[-1]["optimizer_updates_after"] != optimizer_updates:
        raise ValueError("optimizer update count differs from the ledger")
    consumed = manifest["assignments"][:state["encounters"]]
    return {
        "schema": CHECKPOINT_SCHEMA,
        "method": method,
        "manifest_sha256": state["manifest_sha256"],
        "encountered_qid_bias_examples": state["encounters"],
        "encountered_prefix_sha256": digest([a["question_id"] for a in consumed]),
        "encountered_bias_counts": {b: sum(a["bias"] == b for a in consumed) for b in BIAS_TYPES},
        "optimizer_updates": optimizer_updates,
        "batches_without_learning_signal": sum(not e["learning_signal"] for e in ledger),
        "rollouts": None if any(e["rollouts"] is None for e in ledger) else sum(e["rollouts"] for e in ledger),
        "ledger_sha256": digest(ledger),
        "exhausted": state["encounters"] == manifest["max_encounters"],
    }


def _check_state(manifest: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    if state.get("schema") != STATE_SCHEMA or state.get("manifest_sha256") != manifest_identity(manifest):
        raise ValueError("cursor state belongs to a different manifest")
    cursor = state.get("encounters")
    if type(cursor) is not int or not 0 <= cursor <= manifest["max_encounters"]:
        raise ValueError("invalid encounter cursor")
    expected = 0
    for entry in state.get("ledger", []):
        if entry["start"] != expected or entry["end"] <= entry["start"]:
            raise ValueError("ledger is not a contiguous encounter prefix")
        expected = entry["end"]
    if expected != cursor:
        raise ValueError("ledger does not end at the cursor")


__all__ = [
    "DATUM_SCHEMA", "PRIMARY_BUDGET", "PROTOCOL", "PROTOCOL_VERSION", "assign_bias", "build_manifest",
    "checkpoint_record", "claim", "commit", "digest", "freeze_manifest", "initial_state", "load_manifest",
    "manifest_identity", "project_datum", "validate_manifest",
]
