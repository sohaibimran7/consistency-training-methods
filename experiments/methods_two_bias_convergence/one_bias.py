"""Qwen fresh-campaign exposure protocol: one QID once, one assigned bias.

User-approved (2026-10-02), identical to Gemma: all six methods share the
frozen 7,680-QID pool in its frozen interleaved order; each update consumes four
distinct QIDs, each with exactly one hash-assigned cue (four biased examples).
No QID repeats. TBSR validation every 256 encountered QID/bias examples.
"""
from __future__ import annotations

from pathlib import Path

from ctm_data.adapters.mcq_bias import shared_qid_one_bias as protocol

from . import plan

ASSIGNMENT_SEED = 42
POOL_SCOPE = "shared_frozen_7680_all_six_methods"
QIDS_PER_UPDATE = 4
BIASES_PER_QID = 1
VALIDATION_INTERVAL_ENCOUNTERS = 256
# Same order identity as experiments.gemma4_methods.train.ORDER_SHA.
ORDER_SHA = "83898ab455412dd929e30e6865b25eb81af4f9b370e4e1a057964daaf028df3f"


def manifest_for(pool: list[dict]) -> dict:
    """Deterministic manifest over the frozen ordered pool (identical for every method/family)."""
    if len(pool) != 7680:
        raise ValueError("one-bias campaign uses exactly the frozen 7,680-QID pool")
    order = plan.hashlib.sha256(plan.canonical([row["question_id"] for row in pool])).hexdigest()
    if order != ORDER_SHA:
        raise ValueError("frozen QID order differs from the shared cross-family order")
    return protocol.build_manifest(pool, seed=ASSIGNMENT_SEED, source={
        "pool_sha256": plan.POOL_SHA, "pool_manifest_sha256": plan.MANIFEST_SHA, "order_sha256": ORDER_SHA})


def freeze(root: Path, pool: list[dict]) -> tuple[Path, dict]:
    manifest = manifest_for(pool)
    return protocol.freeze_manifest(manifest, Path(root) / "one-bias"), manifest


def max_updates(manifest: dict) -> int:
    return manifest["max_encounters"] // QIDS_PER_UPDATE


def update_rows(manifest: dict, by_id: dict, attempt: int) -> tuple[list[dict], list[str]]:
    """The four (datum, bias) inputs for a zero-based sampled-batch attempt."""
    assignments = protocol.encounter_slice(manifest, attempt, QIDS_PER_UPDATE)
    rows = [by_id[a["question_id"]] for a in assignments]
    for row, assignment in zip(rows, assignments):
        if protocol.digest(protocol._validate_shared_datum(row, location=row["question_id"])) != assignment["datum_sha256"]:
            raise ValueError("pool datum differs from frozen one-bias assignment")
    return rows, [a["bias"] for a in assignments]


def pairs(rows: list[dict], biases: list[str], *, method: str | None) -> list[dict]:
    """One training pair per QID using only its assigned cue (no absent second arm)."""
    if method is not None and method not in plan.METHODS:
        raise ValueError(f"unknown method: {method}")
    if len(rows) != QIDS_PER_UPDATE or len(biases) != len(rows) or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("one-bias update requires four distinct QIDs each with one assigned bias")
    if any(bias not in plan.BIASES for bias in biases):
        raise ValueError("unknown assigned bias")
    return [plan.training_pair(row, bias, method) for row, bias in zip(rows, biases)]


def observe(state: dict, *, step: int, loss: float, limit: int) -> dict:
    """Loss is diagnostic only; TBSR validation owns stopping. Exhaustion is explicit."""
    if step != state.get("step", 0) + 1 or not plan.math.isfinite(loss):
        raise ValueError("non-contiguous optimizer step or non-finite diagnostic loss")
    return {**state, "step": step, "attempts": step, "pending": [], "last_loss": loss,
            "decision": "exhausted" if step >= limit else "continue"}


def exposure(manifest: dict, step: int) -> dict:
    """Checkpoint exposure: every attempt is an update here (no skip path)."""
    consumed = manifest["assignments"][:QIDS_PER_UPDATE * step]
    return {"one_bias_manifest_sha256": protocol.manifest_identity(manifest),
            "encounter_attempt": step, "actual_optimizer_step": step,
            "encountered_qid_bias_examples": QIDS_PER_UPDATE * step,
            "encountered_prefix_sha256": protocol.digest([a["question_id"] for a in consumed]),
            "encountered_bias_counts": {b: sum(a["bias"] == b for a in consumed) for b in plan.BIASES},
            "trailing_skip_files": [], "validation_due": (QIDS_PER_UPDATE * step) % VALIDATION_INTERVAL_ENCOUNTERS == 0}


def contract_block(manifest: dict) -> dict:
    return {"protocol": protocol.PROTOCOL, "protocol_version": protocol.PROTOCOL_VERSION,
            "assignment_seed": ASSIGNMENT_SEED, "pool_scope": POOL_SCOPE,
            "manifest_sha256": protocol.manifest_identity(manifest),
            "assignment_sha256": manifest["assignment_sha256"], "bias_counts": manifest["bias_counts"],
            "qids_per_update": QIDS_PER_UPDATE, "biases_per_qid": BIASES_PER_QID,
            "paired_rows_per_update": QIDS_PER_UPDATE * BIASES_PER_QID,
            "max_encounters": manifest["max_encounters"], "max_attempts": max_updates(manifest),
            "primary_budget": protocol.PRIMARY_BUDGET, "cycling_allowed": False,
            "skipped_batches_consume_encounters": True,
            "validation_every_encountered_qid_bias_examples": VALIDATION_INTERVAL_ENCOUNTERS,
            "user_approval": "2026-10-02: shared 7,680 pool; 4 distinct QIDs x 1 sampled bias per update; "
                             "TBSR validation every 256 encountered QIDs incl. no-update batches"}
