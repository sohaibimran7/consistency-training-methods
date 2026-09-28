"""Build the immutable, plot-ready final Stage 1 IID result matrix.

This is intentionally an offline *adapter*, not another analyzer.  The final
figure needs one canonical condition name for each of the nine methods, while
the attested runs were published in several immutable analysis reports (for
example ``base-vllm`` and ``repaired-act-vllm-compat``).  Re-running analysis
or copying a source report would make that handoff ambiguous.

The merger therefore reads the already-published analysis and raw-preflight
reports, verifies their hashes and the frozen no-CoT population contract, and
writes a new 18-cell report suitable for ``stage1_iid_diagnostic.plot``.  It
never writes an input report, preflight, EvalLog, or manifest.

The preflight reports are part of the input boundary on purpose.  An analysis
report records the hash of the raw log it graded, whereas the preflight binds
that raw log to the frozen source, split bytes, and per-dataset question IDs.
Together they make the cross-host population comparison auditable even though
the host-local manifests have different absolute paths (and therefore
different whole-file hashes).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.stage1_iid_diagnostic import analyze


SCHEMA = "stage1-iid-final-matrix-adapter-v1"
PREFLIGHT_SCHEMA = "stage1-iid-raw-preflight-v1"
CANONICAL_CONDITIONS = (
    "untrained",
    "bct",
    "bct-control",
    "rmct",
    "rmct-control",
    "act",
    "attct",
    "mlpct",
    "opct",
)
# BCT's control is a causal comparison, not merely another row in the
# method matrix.  Its raw Qwen3.5 adapter cannot be represented faithfully by
# the vLLM compatibility path, so both members of this pair must be evaluated
# through the same native Transformers/PEFT transport before the final matrix
# may be published.
BACKEND_MATCHED_BCT_CONDITIONS = ("bct", "bct-control")
BCT_RUNTIME_PROFILE = "hf-peft"
CHECKPOINT_ARTIFACT_IDENTITY_KEYS = frozenset(
    {"adapter_model_sha256", "adapter_config_sha256", "checkpoint_manifest_sha256"}
)
VLLM_COMPATIBILITY_ADAPTER_IDENTITY_KEYS = frozenset(
    {
        "adapter_model_sha256",
        "adapter_config_sha256",
        "compatibility_manifest_sha256",
        "parity_attestation_sha256",
        "source_adapter_model_sha256",
    }
)
SPLITS = ("train_eval", "heldout_in_domain")
DATASETS = ("logiqa", "hellaswag")
EXPECTED_METRIC_DEFINITIONS = {
    "tbsr": "P(biased answer = bias answer | clean answer != bias answer, jointly parsed)",
    "away_from_bias": "P(biased answer != bias answer | clean answer = bias answer, jointly parsed)",
    "total_switch": "P(biased answer differs from clean answer | jointly parsed)",
    "luna_yes": "P(Luna YES | Luna verdict parsed)",
}
RATE_KEYS = frozenset(
    {
        "tbsr",
        "away_from_bias",
        "total_switch",
        "luna_yes",
        "luna_yes_given_towards_bias_switch",
    }
)
COUNT_KEYS = frozenset(
    {
        "samples",
        "joint_parsed",
        "joint_parse_failures",
        "clean_answer_not_bias_answer",
        "clean_answer_equals_bias_answer",
        "luna_parsed",
        "luna_parse_failures",
        "generation_max_token_cap_hits",
        "grader_max_token_cap_hits",
    }
)

# These aliases are operational directory/report names from the attested Stage
# 1 recovery.  The final chart deliberately uses the stable method names in
# CANONICAL_CONDITIONS instead.  Keep this table narrow: an unknown condition
# must stop publication rather than accidentally appear as a known method.
CONDITION_ALIASES = {
    "untrained": "untrained",
    "base": "untrained",
    "base-vllm": "untrained",
    "act": "act",
    "repaired-act": "act",
    "repaired-act-vllm-compat": "act",
    "attct": "attct",
    "attct-vllm-compat": "attct",
    "mlpct": "mlpct",
    "mlpct-vllm-compat": "mlpct",
    "bct": "bct",
    "bct-main": "bct",
    # The native-HF BCT-main rerun is used with the native-HF control for the
    # backend-matched causal comparison.  The separately archived vLLM result
    # remains the fast-serving attestation evidence, not a duplicate matrix
    # cell.
    "bct-main-hf-peft": "bct",
    "bct-main-vllm": "bct",
    "bct-main-vllm-compat": "bct",
    "bct-control": "bct-control",
    # The control's raw Qwen3.5 adapter has a demonstrated HF↔vLLM delta
    # mismatch, so the authoritative final control is evaluated through the
    # native Transformers/PEFT path rather than a translated vLLM adapter.
    "bct-control-hf-peft": "bct-control",
    "bct-control-vllm": "bct-control",
    "bct-control-vllm-compat": "bct-control",
    "opct": "opct",
    "opct-vllm": "opct",
    "opct-vllm-compat": "opct",
    "rmct": "rmct",
    "rmct-control": "rmct-control",
    # This alias is admissible only in the raw-preflight/source-provenance
    # chain.  A publication report under this name must first pass through the
    # dedicated canonicalizer, which recomputes rmct_first64.
    "rmct-control-b8-accelerated": "rmct-control",
}

# The no-CoT Stage-1 suite is frozen.  The whole manifest bytes can legitimately
# differ after host-local absolute paths are substituted, so the merge boundary
# compares the semantic population fingerprints instead of requiring a single
# manifest byte hash across hosts.
FROZEN_SOURCE_SHA256 = "7d113ee1858426721d09b23a78f4bbb0e9b16e7576b3ee5ab7da4924c3a0ef3b"
FROZEN_SPLITS = {
    "train_eval": {
        "content_sha256": "fa5dabe9f7b8a53958833b9d0ed7692437979a5a4ff142d2ba9555dc35dade63",
        "question_ids_sha256": "df76a000e7a60bfdbb44ed077c37ed21828eec5a12c8da53742e5ae9c7924391",
    },
    "heldout_in_domain": {
        "content_sha256": "375158a369e3f040c562136c280ba1cb3f6cf456ffc0874cdb387de0ea44c9c4",
        "question_ids_sha256": "499106c1c45cc422d8b231d17a0b87d6cd0636a843fc0c222cdb04bed0198ae1",
    },
}
FROZEN_CELL_QUESTION_IDS_SHA256 = {
    ("train_eval", "logiqa"): "a582a4212cdefd961fb1d243b5b98c0f2158af26ee1b562eaf140c7d17f76dbb",
    ("train_eval", "hellaswag"): "8ba6fa9423904e9b7759a02d666f34f0b837cafc65074440a168897bbcafd9ec",
    ("heldout_in_domain", "logiqa"): "a0c725943040a510fb43d8af28f13c395a28e232fcdbe25b7790a70d6f23b95b",
    ("heldout_in_domain", "hellaswag"): "d2cec72250a5cc6ca15164b2108966f3ee50a3f18de0789830a984b572c93700",
}
FROZEN_RMCT_FIRST64_IDS_SHA256 = "bab1ad6f5608139d08c017ade3be772b0b18115d68226b1d7c3ee4c8c72f1e2b"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_payload(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_object(path: str | Path, *, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    payload = resolved.read_bytes()
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {resolved}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object: {resolved}")
    return resolved, payload, document


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _canonical_condition(raw: Any, *, origin: str, allow_accelerated_rmct_control: bool = False) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{origin} has no condition")
    if raw == "rmct-control-b8-accelerated" and not allow_accelerated_rmct_control:
        raise ValueError(f"{origin} is the operational RMCT-control alias; publish a canonical rmct-control report first")
    canonical = CONDITION_ALIASES.get(raw)
    if canonical is None:
        raise ValueError(f"{origin} has unsupported condition {raw!r}")
    return canonical


def _ids_sha256(ids: Sequence[str]) -> str:
    return _sha256_bytes("".join(f"{question_id}\n" for question_id in ids).encode("utf-8"))


def load_population_fingerprint(manifest: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    """Read a no-CoT manifest and reduce it to its cross-host invariants.

    This deliberately does not dereference the host-specific absolute split
    paths.  The split content hashes and ordered ID hashes are the invariant
    evidence we need here; raw-preflight reports bind every source log to those
    same hashes.  It makes the final merger usable after a safe artifact copy.
    """

    path, payload, document = _read_object(manifest, label="population manifest")
    if document.get("schema_version") != 1 or document.get("kind") != "stage1_iid_diagnostic_none_manifest":
        raise ValueError("population manifest has an unsupported no-CoT Stage 1 schema")
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("population manifest source must be an object")
    if source.get("content_sha256") != FROZEN_SOURCE_SHA256:
        raise ValueError("population manifest has the wrong recovered no-CoT source SHA-256")
    if source.get("row_count") != 3000 or source.get("counts_by_dataset") != {"hellaswag": 1500, "logiqa": 1500}:
        raise ValueError("population manifest source population is not the frozen 3,000-row suite")
    if source.get("bias_type") != "wrong_argument" or source.get("prompt_style") != "none":
        raise ValueError("population manifest is not the final no-CoT wrong-argument suite")

    split_entries = document.get("splits")
    if not isinstance(split_entries, Mapping) or set(split_entries) != set(SPLITS):
        raise ValueError("population manifest must contain exactly the two final IID splits")
    fingerprint_splits: dict[str, dict[str, Any]] = {}
    split_ids: dict[str, list[str]] = {}
    for split in SPLITS:
        entry = split_entries[split]
        if not isinstance(entry, Mapping):
            raise ValueError(f"population manifest split {split!r} must be an object")
        expected = FROZEN_SPLITS[split]
        if entry.get("content_sha256") != expected["content_sha256"]:
            raise ValueError(f"population manifest split {split!r} has the wrong content SHA-256")
        ids = entry.get("question_ids")
        if not isinstance(ids, list) or len(ids) != 200 or any(not isinstance(item, str) or not item for item in ids):
            raise ValueError(f"population manifest split {split!r} has invalid ordered question IDs")
        ids_digest = _ids_sha256(ids)
        if entry.get("question_ids_sha256") != ids_digest or ids_digest != expected["question_ids_sha256"]:
            raise ValueError(f"population manifest split {split!r} has the wrong ordered-ID SHA-256")
        if entry.get("row_count") != 200 or entry.get("counts_by_dataset") != {"hellaswag": 100, "logiqa": 100}:
            raise ValueError(f"population manifest split {split!r} has the wrong frozen population counts")
        fingerprint_splits[split] = {
            "content_sha256": expected["content_sha256"],
            "question_ids_sha256": expected["question_ids_sha256"],
            "row_count": 200,
            "counts_by_dataset": {"hellaswag": 100, "logiqa": 100},
        }
        split_ids[split] = ids
    if set(split_ids["train_eval"]) & set(split_ids["heldout_in_domain"]):
        raise ValueError("population manifest train and held-out IDs overlap")

    first64 = document.get("rmct_first64")
    if not isinstance(first64, Mapping):
        raise ValueError("population manifest has no rmct_first64 section")
    first64_ids = first64.get("question_ids")
    if first64_ids != split_ids["train_eval"][:64]:
        raise ValueError("population manifest rmct_first64 is not the train split prefix")
    if first64.get("question_ids_sha256") != FROZEN_RMCT_FIRST64_IDS_SHA256 or _ids_sha256(first64_ids) != FROZEN_RMCT_FIRST64_IDS_SHA256 or first64.get("row_count") != 64 or first64.get("counts_by_dataset") != {"hellaswag": 32, "logiqa": 32}:
        raise ValueError("population manifest rmct_first64 fingerprint is invalid")
    fingerprint = {
        "source_sha256": FROZEN_SOURCE_SHA256,
        "source_rows": 3000,
        "source_counts_by_dataset": {"hellaswag": 1500, "logiqa": 1500},
        "bias_type": "wrong_argument",
        "prompt_style": "none",
        "splits": fingerprint_splits,
        "cell_question_ids_sha256": {f"{split}/{dataset}": FROZEN_CELL_QUESTION_IDS_SHA256[(split, dataset)] for split in SPLITS for dataset in DATASETS},
        "rmct_first64": {
            "question_ids_sha256": FROZEN_RMCT_FIRST64_IDS_SHA256,
            "row_count": 64,
            "counts_by_dataset": {"hellaswag": 32, "logiqa": 32},
        },
    }
    return path, payload, fingerprint


def _validate_rate(value: Any, *, label: str, require_denominator: bool) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a rate object")
    numerator = _positive_int(value.get("numerator"), label=f"{label}.numerator", allow_zero=True)
    denominator = _positive_int(value.get("denominator"), label=f"{label}.denominator", allow_zero=not require_denominator)
    if numerator > denominator:
        raise ValueError(f"{label} numerator exceeds denominator")
    rate = value.get("rate")
    if denominator == 0:
        if rate is not None:
            raise ValueError(f"{label} must have null rate with a zero denominator")
    else:
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(float(rate)):
            raise ValueError(f"{label}.rate must be finite")
        if not math.isclose(float(rate), numerator / denominator, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label}.rate is inconsistent with its exact count")
    return numerator, denominator


def _validate_summary(summary: Any, *, label: str, expected_samples: int) -> None:
    if not isinstance(summary, Mapping):
        raise ValueError(f"{label} summary must be an object")
    counts = summary.get("counts")
    rates = summary.get("rates")
    if not isinstance(counts, Mapping) or set(counts) != COUNT_KEYS:
        raise ValueError(f"{label} counts do not have the canonical Stage 1 fields")
    if not isinstance(rates, Mapping) or set(rates) != RATE_KEYS:
        raise ValueError(f"{label} rates do not have the canonical Stage 1 fields")
    numeric_counts = {key: _positive_int(value, label=f"{label}.counts.{key}", allow_zero=True) for key, value in counts.items()}
    if numeric_counts["samples"] != expected_samples:
        raise ValueError(f"{label} has {numeric_counts['samples']} samples, expected {expected_samples}")
    if numeric_counts["joint_parsed"] + numeric_counts["joint_parse_failures"] != expected_samples:
        raise ValueError(f"{label} joint-parse counts do not sum to samples")
    if numeric_counts["clean_answer_not_bias_answer"] + numeric_counts["clean_answer_equals_bias_answer"] != numeric_counts["joint_parsed"]:
        raise ValueError(f"{label} clean-answer eligibility counts do not sum to jointly parsed")
    if numeric_counts["luna_parsed"] + numeric_counts["luna_parse_failures"] != expected_samples:
        raise ValueError(f"{label} Luna counts do not sum to samples")
    if numeric_counts["generation_max_token_cap_hits"] > expected_samples or numeric_counts["grader_max_token_cap_hits"] > numeric_counts["luna_parsed"]:
        raise ValueError(f"{label} cap-hit count is impossible")

    rate_values = {
        key: _validate_rate(
            rates[key],
            label=f"{label}.rates.{key}",
            require_denominator=key in {"tbsr", "luna_yes"},
        )
        for key in RATE_KEYS
    }
    if rate_values["tbsr"][1] != numeric_counts["clean_answer_not_bias_answer"]:
        raise ValueError(f"{label} TBSR denominator does not match clean-answer eligibility")
    if rate_values["away_from_bias"][1] != numeric_counts["clean_answer_equals_bias_answer"]:
        raise ValueError(f"{label} away-from-bias denominator does not match clean-answer eligibility")
    if rate_values["total_switch"][1] != numeric_counts["joint_parsed"]:
        raise ValueError(f"{label} total-switch denominator does not match jointly parsed")
    if rate_values["luna_yes"][1] != numeric_counts["luna_parsed"]:
        raise ValueError(f"{label} Luna denominator does not match Luna-parsed")
    if rate_values["luna_yes_given_towards_bias_switch"][1] != rate_values["tbsr"][0]:
        raise ValueError(f"{label} Luna-given-switch denominator does not match TBSR numerator")


def _validate_cell(cell: Any, *, raw_condition: str, split: str, canonical: str) -> None:
    if not isinstance(cell, Mapping):
        raise ValueError(f"{raw_condition}/{split} analysis cell must be an object")
    allowed = {"condition", "split", "pooled", "per_dataset"}
    if canonical in {"rmct", "rmct-control"} and split == "train_eval":
        allowed.add("rmct_first64")
    if set(cell) != allowed:
        raise ValueError(f"{raw_condition}/{split} has unexpected or missing analysis-cell fields")
    if cell.get("condition") != raw_condition or cell.get("split") != split:
        raise ValueError(f"{raw_condition}/{split} analysis cell has conflicting identity")
    _validate_summary(cell.get("pooled"), label=f"{raw_condition}/{split}/pooled", expected_samples=200)
    per_dataset = cell.get("per_dataset")
    if not isinstance(per_dataset, Mapping) or set(per_dataset) != set(DATASETS):
        raise ValueError(f"{raw_condition}/{split} has incomplete per-dataset analysis")
    for dataset in DATASETS:
        _validate_summary(
            per_dataset[dataset],
            label=f"{raw_condition}/{split}/{dataset}",
            expected_samples=100,
        )
    _validate_pooled_sum(cell, label=f"{raw_condition}/{split}")
    if "rmct_first64" in cell:
        subset = cell["rmct_first64"]
        if not isinstance(subset, Mapping) or set(subset) != {"pooled", "per_dataset"}:
            raise ValueError(f"{raw_condition}/{split} rmct_first64 has an invalid shape")
        _validate_summary(subset.get("pooled"), label=f"{raw_condition}/{split}/rmct_first64/pooled", expected_samples=64)
        per_dataset_subset = subset.get("per_dataset")
        if not isinstance(per_dataset_subset, Mapping) or set(per_dataset_subset) != set(DATASETS):
            raise ValueError(f"{raw_condition}/{split} rmct_first64 has incomplete per-dataset analysis")
        for dataset in DATASETS:
            _validate_summary(
                per_dataset_subset[dataset],
                label=f"{raw_condition}/{split}/rmct_first64/{dataset}",
                expected_samples=32,
            )
        _validate_pooled_sum(subset, label=f"{raw_condition}/{split}/rmct_first64")


def _validate_pooled_sum(cell: Mapping[str, Any], *, label: str) -> None:
    pooled = cell["pooled"]
    per_dataset = cell["per_dataset"]
    for count in COUNT_KEYS:
        observed = pooled["counts"][count]
        expected = sum(per_dataset[dataset]["counts"][count] for dataset in DATASETS)
        if observed != expected:
            raise ValueError(f"{label} pooled count {count!r} is not the exact per-dataset sum")
    for metric in RATE_KEYS:
        pooled_rate = pooled["rates"][metric]
        numerator = sum(per_dataset[dataset]["rates"][metric]["numerator"] for dataset in DATASETS)
        denominator = sum(per_dataset[dataset]["rates"][metric]["denominator"] for dataset in DATASETS)
        if pooled_rate["numerator"] != numerator or pooled_rate["denominator"] != denominator:
            raise ValueError(f"{label} pooled rate {metric!r} is not the exact per-dataset sum")


def _validate_analysis_report(
    document: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, str], dict[tuple[str, str, str], Mapping[str, Any]]]:
    """Validate a complete immutable report and return its condition/source indexes."""

    if document.get("schema") != analyze.ANALYSIS_SCHEMA:
        raise ValueError(f"{label} has unsupported analysis schema")
    if document.get("grader_model") != analyze.DEFAULT_LUNA_GRADER_MODEL:
        raise ValueError(f"{label} uses an unexpected Luna grader")
    if document.get("inspect_rescore_model") != analyze.INSPECT_RESCORE_MODEL:
        raise ValueError(f"{label} uses an unexpected Inspect rescore model")
    if document.get("grader_max_tokens") != 1024:
        raise ValueError(f"{label} must use the final 1,024-token Luna cap")
    if document.get("metric_definitions") != EXPECTED_METRIC_DEFINITIONS:
        raise ValueError(f"{label} metric definitions do not match the paired final analysis")
    if not _valid_sha256(document.get("diagnostic_manifest_sha256")):
        raise ValueError(f"{label} has no valid diagnostic manifest SHA-256")
    if not isinstance(document.get("diagnostic_manifest"), str) or not document["diagnostic_manifest"]:
        raise ValueError(f"{label} has no diagnostic manifest path")

    cells = document.get("cells")
    if not isinstance(cells, Mapping) or not cells:
        raise ValueError(f"{label} has no analysis cells")
    raw_conditions: set[str] = set()
    for key, cell in cells.items():
        if not isinstance(key, str) or "/" not in key or not isinstance(cell, Mapping):
            raise ValueError(f"{label} has malformed analysis-cell key")
        raw_condition, split = key.rsplit("/", 1)
        canonical = _canonical_condition(raw_condition, origin=f"{label} report cell")
        if split not in SPLITS:
            raise ValueError(f"{label} has unsupported analysis split {split!r}")
        _validate_cell(cell, raw_condition=raw_condition, split=split, canonical=canonical)
        raw_conditions.add(raw_condition)
    expected_keys = {f"{condition}/{split}" for condition in raw_conditions for split in SPLITS}
    if set(cells) != expected_keys:
        raise ValueError(f"{label} has incomplete or mixed analysis cells")

    canonical_by_raw: dict[str, str] = {}
    for raw_condition in raw_conditions:
        canonical = _canonical_condition(raw_condition, origin=f"{label} report")
        if canonical in canonical_by_raw.values():
            raise ValueError(f"{label} contains duplicate aliases for canonical condition {canonical!r}")
        canonical_by_raw[raw_condition] = canonical

    sources = document.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"{label} sources must be an array")
    source_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    expected_sources = {(raw, split, dataset) for raw in raw_conditions for split in SPLITS for dataset in DATASETS}
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError(f"{label} contains a non-object source")
        raw_condition = source.get("condition")
        split = source.get("split")
        dataset = source.get("dataset")
        if raw_condition not in raw_conditions or split not in SPLITS or dataset not in DATASETS:
            raise ValueError(f"{label} source has an unexpected condition/split/dataset identity")
        key = (raw_condition, split, dataset)
        if key in source_index:
            raise ValueError(f"{label} has duplicate source evidence for {key!r}")
        if source.get("samples") != 100:
            raise ValueError(f"{label} source {key!r} must contain exactly 100 samples")
        if not _valid_sha256(source.get("source_sha256")) or not _valid_sha256(source.get("graded_log_sha256")):
            raise ValueError(f"{label} source {key!r} has invalid raw or graded SHA-256")
        if not isinstance(source.get("source_log"), str) or not source["source_log"]:
            raise ValueError(f"{label} source {key!r} has no raw source-log path")
        if not isinstance(source.get("graded_log"), str) or not source["graded_log"]:
            raise ValueError(f"{label} source {key!r} has no graded-log path")
        source_index[key] = source
    if set(source_index) != expected_sources:
        raise ValueError(f"{label} source evidence is incomplete or has unexpected cells")
    return canonical_by_raw, source_index


def _validate_preflight(
    document: Mapping[str, Any],
    *,
    label: str,
    population: Mapping[str, Any],
) -> tuple[
    str,
    str,
    dict[tuple[str, str], Mapping[str, Any]],
    dict[str, str] | None,
    dict[str, str] | None,
]:
    """Validate one raw-preflight report and return canonical/source condition indexes."""

    if document.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError(f"{label} has unsupported raw-preflight schema")
    raw_condition = document.get("condition")
    canonical = _canonical_condition(
        raw_condition,
        origin=label,
        allow_accelerated_rmct_control=True,
    )
    contract = document.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{label} has no preflight contract")
    expected_contract = {
        "source_sha256": population["source_sha256"],
        "bias_type": population["bias_type"],
        "prompt_style": population["prompt_style"],
        "expected_base_model": "Qwen/Qwen3.5-9B",
        "include_bias_acknowledged": False,
        "grader_model": None,
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            raise ValueError(f"{label} contract {field!r} is not pinned to the final no-CoT diagnostic")
    runtime_profile = contract.get("runtime_profile")
    if runtime_profile not in {"vllm", "hf-peft"}:
        raise ValueError(f"{label} contract has no runtime profile")
    expected_checkpoint = contract.get("expected_checkpoint")
    if runtime_profile == "hf-peft":
        if not isinstance(expected_checkpoint, str) or not expected_checkpoint:
            raise ValueError(f"{label} HF/PEFT contract has no expected checkpoint")
        max_connections = contract.get("expected_max_connections")
        if isinstance(max_connections, bool) or not isinstance(max_connections, int) or max_connections < 1:
            raise ValueError(f"{label} HF/PEFT contract has no positive max-connections bound")
    elif canonical == "untrained":
        if expected_checkpoint is not None:
            raise ValueError(f"{label} untrained vLLM baseline must not name a local adapter checkpoint")
    elif not isinstance(expected_checkpoint, str) or not expected_checkpoint:
        raise ValueError(f"{label} adapted vLLM contract has no expected compatibility-adapter checkpoint")

    artifact_identity_value = contract.get("checkpoint_artifact_identity")
    artifact_identity: dict[str, str] | None = None
    if artifact_identity_value is not None:
        if runtime_profile != "hf-peft":
            raise ValueError(f"{label} has checkpoint artifact identity outside HF/PEFT")
        if not isinstance(artifact_identity_value, Mapping) or set(artifact_identity_value) != CHECKPOINT_ARTIFACT_IDENTITY_KEYS:
            raise ValueError(f"{label} has an invalid checkpoint artifact identity")
        artifact_identity = {}
        for field in sorted(CHECKPOINT_ARTIFACT_IDENTITY_KEYS):
            digest = artifact_identity_value.get(field)
            if not _valid_sha256(digest):
                raise ValueError(f"{label} checkpoint artifact {field!r} is not a SHA-256")
            artifact_identity[field] = str(digest)

    vllm_identity_value = contract.get("vllm_compatibility_adapter_identity")
    vllm_compatibility_adapter_identity: dict[str, str] | None = None
    if vllm_identity_value is not None:
        if runtime_profile != "vllm":
            raise ValueError(f"{label} has vLLM compatibility-adapter identity outside vLLM")
        if canonical == "untrained":
            raise ValueError(f"{label} untrained vLLM baseline must not bind a compatibility adapter")
        if (
            not isinstance(vllm_identity_value, Mapping)
            or set(vllm_identity_value) != VLLM_COMPATIBILITY_ADAPTER_IDENTITY_KEYS
        ):
            raise ValueError(f"{label} has an invalid vLLM compatibility-adapter identity")
        vllm_compatibility_adapter_identity = {}
        for field in sorted(VLLM_COMPATIBILITY_ADAPTER_IDENTITY_KEYS):
            digest = vllm_identity_value.get(field)
            if not _valid_sha256(digest):
                raise ValueError(f"{label} vLLM compatibility adapter artifact {field!r} is not a SHA-256")
            vllm_compatibility_adapter_identity[field] = str(digest)
    elif runtime_profile == "vllm" and canonical != "untrained":
        raise ValueError(
            f"{label} adapted vLLM contract must bind compatibility-adapter, translation-manifest, and parity-attestation SHA-256s"
        )

    frozen_splits = document.get("frozen_splits")
    if not isinstance(frozen_splits, Mapping) or set(frozen_splits) != set(SPLITS):
        raise ValueError(f"{label} frozen-split evidence is incomplete")
    for split in SPLITS:
        entry = frozen_splits[split]
        if not isinstance(entry, Mapping) or entry.get("sha256") != population["splits"][split]["content_sha256"]:
            raise ValueError(f"{label} frozen split {split!r} does not match the final population fingerprint")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError(f"{label} frozen split {split!r} has no path")
    if not _valid_sha256(document.get("manifest_sha256")) or not isinstance(document.get("manifest"), str):
        raise ValueError(f"{label} has no valid host-local manifest provenance")

    sources = document.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"{label} preflight sources must be an array")
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError(f"{label} contains a non-object preflight source")
        split = source.get("split")
        dataset = source.get("dataset")
        if split not in SPLITS or dataset not in DATASETS:
            raise ValueError(f"{label} has a preflight source with invalid split/dataset")
        key = (split, dataset)
        if key in index:
            raise ValueError(f"{label} has duplicate preflight source evidence for {key!r}")
        if source.get("sample_count") != 100:
            raise ValueError(f"{label} preflight source {key!r} must contain exactly 100 samples")
        if source.get("source_identity_digest") != population["source_sha256"]:
            raise ValueError(f"{label} preflight source {key!r} has the wrong recovered-source SHA-256")
        if source.get("prompt_style") != population["prompt_style"]:
            raise ValueError(f"{label} preflight source {key!r} has the wrong prompt style")
        if not _valid_sha256(source.get("raw_log_sha256")):
            raise ValueError(f"{label} preflight source {key!r} has no valid raw-log SHA-256")
        if source.get("question_ids_sha256") != population["cell_question_ids_sha256"][f"{split}/{dataset}"]:
            raise ValueError(f"{label} preflight source {key!r} has the wrong frozen question-ID fingerprint")
        if not isinstance(source.get("raw_log"), str) or not source["raw_log"]:
            raise ValueError(f"{label} preflight source {key!r} has no raw-log path")
        index[key] = source
    expected = {(split, dataset) for split in SPLITS for dataset in DATASETS}
    if set(index) != expected:
        raise ValueError(f"{label} preflight source evidence is incomplete")
    return canonical, raw_condition, index, artifact_identity, vllm_compatibility_adapter_identity


def _validate_rmct_control_canonicalization(
    path: str | Path,
    *,
    canonical_report_path: Path,
    canonical_report_payload: bytes,
    canonical_report: Mapping[str, Any],
) -> dict[str, Any]:
    proof_path, proof_payload, proof = _read_object(path, label="RMCT-control canonicalization provenance")
    if proof.get("schema") != "stage1-iid-rmct-control-canonicalization-v1":
        raise ValueError("RMCT-control canonicalization provenance has an unsupported schema")
    if proof.get("alias_condition") != "rmct-control-b8-accelerated" or proof.get("canonical_condition") != "rmct-control":
        raise ValueError("RMCT-control canonicalization provenance has the wrong condition mapping")
    if proof.get("canonical_analysis_sha256") != _sha256_bytes(canonical_report_payload):
        raise ValueError("RMCT-control canonicalization provenance does not attest this canonical report's bytes")
    if proof.get("diagnostic_manifest_sha256") != canonical_report.get("diagnostic_manifest_sha256"):
        raise ValueError("RMCT-control canonicalization provenance disagrees on its diagnostic manifest")
    checks = proof.get("checks")
    if not isinstance(checks, Mapping) or checks.get("raw_or_graded_logs_modified") is not False:
        raise ValueError("RMCT-control canonicalization provenance does not attest immutable source logs")
    for field in (
        "frozen_manifest_validated",
        "graded_eval_log_ids_exactly_match_manifest",
        "alias_analysis_sources_exactly_match_graded_eval_logs",
        "pooled_and_per_dataset_cells_equivalent_except_condition_label",
    ):
        if checks.get(field) is not True:
            raise ValueError(f"RMCT-control canonicalization provenance has not passed {field!r}")
    return {
        "path": str(proof_path),
        "sha256": _sha256_bytes(proof_payload),
        "canonical_report": str(canonical_report_path),
    }


def build_final_matrix(
    reports: Sequence[str | Path],
    preflights: Sequence[str | Path],
    population_manifest: str | Path,
    *,
    rmct_control_canonicalization: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and merge immutable evidence into the exact nine-by-two matrix.

    No model, grader, Inspect EvalLog, or network service is touched.  The
    returned objects are not written until :func:`write_final_matrix` is
    called, making this suitable for a pre-publication dry-run.
    """

    if not reports or not preflights:
        raise ValueError("the final merge requires analysis reports and raw-preflight reports")
    population_path, population_payload, population = load_population_fingerprint(population_manifest)

    report_records: list[dict[str, Any]] = []
    canonical_cells: dict[str, Any] = {}
    canonical_sources: list[dict[str, Any]] = []
    cell_sources: dict[str, dict[str, Any]] = {}
    condition_evidence: dict[str, dict[str, Any]] = {}
    shared_metadata: dict[str, Any] | None = None
    canonical_control_report_context: tuple[Path, bytes, Mapping[str, Any]] | None = None

    for index, report_path in enumerate(reports, start=1):
        path, payload, report = _read_object(report_path, label=f"analysis report {index}")
        raw_to_canonical, source_index = _validate_analysis_report(report, label=f"analysis report {path}")
        metadata = {field: report[field] for field in ("grader_model", "grader_max_tokens", "inspect_rescore_model", "metric_definitions")}
        if shared_metadata is None:
            shared_metadata = metadata
        elif metadata != shared_metadata:
            raise ValueError("analysis reports disagree on final scorer/metric metadata")
        record_conditions: list[str] = []
        for raw_condition, canonical in raw_to_canonical.items():
            if canonical in condition_evidence:
                previous = condition_evidence[canonical]["report_path"]
                raise ValueError(f"duplicate evidence for canonical condition {canonical!r}: {previous} and {path}")
            record_conditions.append(canonical)
            evidence = {
                "report_path": str(path),
                "report_sha256": _sha256_bytes(payload),
                "report_manifest": report["diagnostic_manifest"],
                "report_manifest_sha256": report["diagnostic_manifest_sha256"],
                "raw_condition": raw_condition,
                "report_sources": source_index,
            }
            condition_evidence[canonical] = evidence
            for split in SPLITS:
                source_cell_key = f"{raw_condition}/{split}"
                canonical_key = f"{canonical}/{split}"
                if canonical_key in canonical_cells:
                    raise AssertionError("duplicate canonical cell passed condition-level uniqueness check")
                copied = copy.deepcopy(report["cells"][source_cell_key])
                copied["condition"] = canonical
                canonical_cells[canonical_key] = copied
                cell_sources[canonical_key] = {
                    "source_report": str(path),
                    "source_report_sha256": _sha256_bytes(payload),
                    "source_analysis_cell": source_cell_key,
                    "source_analysis_cell_sha256": _sha256_bytes(_json_payload({"cell": report["cells"][source_cell_key]})),
                    "source_condition": raw_condition,
                }
            for split in SPLITS:
                for dataset in DATASETS:
                    source = source_index[(raw_condition, split, dataset)]
                    source_condition = source.get("source_condition", raw_condition)
                    if not isinstance(source_condition, str) or not source_condition:
                        raise ValueError(f"analysis report {path} has invalid source_condition provenance")
                    if canonical != "rmct-control" and source_condition != raw_condition:
                        raise ValueError(f"analysis report {path} has non-canonical source-condition provenance outside RMCT-control")
                    if canonical == "rmct-control" and source_condition not in {
                        "rmct-control",
                        "rmct-control-b8-accelerated",
                    }:
                        raise ValueError(f"analysis report {path} has invalid RMCT-control source-condition provenance")
                    canonical_sources.append(
                        {
                            **copy.deepcopy(dict(source)),
                            "condition": canonical,
                            "source_condition": source_condition,
                            "source_report": str(path),
                            "source_report_sha256": _sha256_bytes(payload),
                            "source_diagnostic_manifest_sha256": report["diagnostic_manifest_sha256"],
                        }
                    )
            if canonical == "rmct-control":
                canonical_control_report_context = (path, payload, report)
        report_records.append(
            {
                "path": str(path),
                "sha256": _sha256_bytes(payload),
                "reported_diagnostic_manifest": report["diagnostic_manifest"],
                "reported_diagnostic_manifest_sha256": report["diagnostic_manifest_sha256"],
                "canonical_conditions": sorted(record_conditions),
            }
        )

    expected_conditions = set(CANONICAL_CONDITIONS)
    actual_conditions = set(condition_evidence)
    if actual_conditions != expected_conditions:
        raise ValueError(f"analysis reports must supply exactly the final nine canonical conditions; missing={sorted(expected_conditions - actual_conditions)}, unexpected={sorted(actual_conditions - expected_conditions)}")
    expected_cells = {f"{condition}/{split}" for condition in CANONICAL_CONDITIONS for split in SPLITS}
    if set(canonical_cells) != expected_cells:
        raise AssertionError("canonical condition validation did not produce exactly 18 cells")

    preflight_records: list[dict[str, Any]] = []
    preflight_evidence: dict[str, dict[str, Any]] = {}
    for index, preflight_path in enumerate(preflights, start=1):
        path, payload, preflight = _read_object(preflight_path, label=f"raw-preflight report {index}")
        (
            canonical,
            raw_condition,
            source_index,
            checkpoint_artifact_identity,
            vllm_compatibility_adapter_identity,
        ) = _validate_preflight(
            preflight,
            label=f"raw-preflight report {path}",
            population=population,
        )
        if canonical in preflight_evidence:
            raise ValueError(f"duplicate raw-preflight evidence for canonical condition {canonical!r}")
        preflight_evidence[canonical] = {
            "path": str(path),
            "sha256": _sha256_bytes(payload),
            "raw_condition": raw_condition,
            "manifest": preflight["manifest"],
            "manifest_sha256": preflight["manifest_sha256"],
            "runtime_profile": preflight["contract"]["runtime_profile"],
            "checkpoint_artifact_identity": checkpoint_artifact_identity,
            "vllm_compatibility_adapter_identity": vllm_compatibility_adapter_identity,
            "sources": source_index,
        }
        preflight_records.append(
            {
                "path": str(path),
                "sha256": _sha256_bytes(payload),
                "raw_condition": raw_condition,
                "canonical_condition": canonical,
                "manifest": preflight["manifest"],
                "manifest_sha256": preflight["manifest_sha256"],
                "runtime_profile": preflight["contract"]["runtime_profile"],
                "checkpoint_artifact_identity": checkpoint_artifact_identity,
                "vllm_compatibility_adapter_identity": vllm_compatibility_adapter_identity,
            }
        )
    if set(preflight_evidence) != expected_conditions:
        raise ValueError(f"raw-preflight reports must supply exactly the final nine canonical conditions; missing={sorted(expected_conditions - set(preflight_evidence))}, unexpected={sorted(set(preflight_evidence) - expected_conditions)}")

    bct_runtime_profiles = {
        condition: preflight_evidence[condition]["runtime_profile"] for condition in BACKEND_MATCHED_BCT_CONDITIONS
    }
    if set(bct_runtime_profiles.values()) != {BCT_RUNTIME_PROFILE}:
        raise ValueError(
            "BCT main/control evidence must both use the backend-matched native HF/PEFT runtime; "
            f"got {bct_runtime_profiles}"
        )
    if any(preflight_evidence[condition]["checkpoint_artifact_identity"] is None for condition in BACKEND_MATCHED_BCT_CONDITIONS):
        raise ValueError(
            "BCT main/control evidence must bind the raw adapter, PEFT config, and checkpoint manifest SHA-256s"
        )
    vllm_lora_conditions = tuple(
        condition
        for condition in CANONICAL_CONDITIONS
        if condition != "untrained" and preflight_evidence[condition]["runtime_profile"] == "vllm"
    )
    if any(preflight_evidence[condition]["vllm_compatibility_adapter_identity"] is None for condition in vllm_lora_conditions):
        raise AssertionError("validated adapted vLLM preflight lost its compatibility-adapter identity")

    # Match the analysis' graded raw-log hashes to their original preflight
    # source evidence.  That is the important report-to-population binding.
    for canonical in CANONICAL_CONDITIONS:
        report_evidence = condition_evidence[canonical]
        preflight_evidence_for_condition = preflight_evidence[canonical]
        source_conditions = {source.get("source_condition", report_evidence["raw_condition"]) for source in report_evidence["report_sources"].values()}
        if len(source_conditions) != 1:
            raise ValueError(f"{canonical} analysis report has mixed raw source-condition provenance")
        (source_condition,) = tuple(source_conditions)
        if source_condition != preflight_evidence_for_condition["raw_condition"]:
            raise ValueError(f"{canonical} analysis/preflight raw conditions disagree: {source_condition!r} != {preflight_evidence_for_condition['raw_condition']!r}")
        for split in SPLITS:
            for dataset in DATASETS:
                analysis_source = report_evidence["report_sources"][(report_evidence["raw_condition"], split, dataset)]
                preflight_source = preflight_evidence_for_condition["sources"][(split, dataset)]
                if analysis_source["source_sha256"] != preflight_source["raw_log_sha256"]:
                    raise ValueError(f"{canonical}/{split}/{dataset} graded analysis does not bind to its preflight raw-log SHA-256")
                if analysis_source["samples"] != preflight_source["sample_count"]:
                    raise ValueError(f"{canonical}/{split}/{dataset} analysis/preflight sample counts disagree")
        if report_evidence["report_manifest_sha256"] != preflight_evidence_for_condition["manifest_sha256"]:
            raise ValueError(f"{canonical} analysis/preflight diagnostic manifest SHA-256 values disagree")

    canonicalization_record: dict[str, Any] | None = None
    rmct_control_sources = {source["source_condition"] for source in canonical_sources if source["condition"] == "rmct-control"}
    if rmct_control_sources == {"rmct-control-b8-accelerated"}:
        if rmct_control_canonicalization is None or canonical_control_report_context is None:
            raise ValueError("accelerated RMCT-control evidence requires its canonicalization provenance alongside the canonical report")
        canonicalization_record = _validate_rmct_control_canonicalization(
            rmct_control_canonicalization,
            canonical_report_path=canonical_control_report_context[0],
            canonical_report_payload=canonical_control_report_context[1],
            canonical_report=canonical_control_report_context[2],
        )
    elif rmct_control_canonicalization is not None:
        raise ValueError("RMCT-control canonicalization provenance was supplied but is not required by the source evidence")

    if shared_metadata is None:
        raise AssertionError("non-empty report input did not produce shared metadata")
    canonical_sources.sort(key=lambda source: (source["condition"], source["split"], source["dataset"]))
    adapter = {
        # Keep the ordinary analysis schema so the existing publication plotter
        # accepts this adapter without a special plotting branch.
        "schema": analyze.ANALYSIS_SCHEMA,
        **shared_metadata,
        "diagnostic_manifest": str(population_path),
        "diagnostic_manifest_sha256": _sha256_bytes(population_payload),
        "sources": canonical_sources,
        "cells": {key: canonical_cells[key] for key in sorted(canonical_cells)},
        "final_matrix_adapter": {
            "schema": SCHEMA,
            "population_fingerprint": population,
            "cell_sources": {key: cell_sources[key] for key in sorted(cell_sources)},
            "source_reports": report_records,
            "raw_preflight_reports": preflight_records,
            "rmct_control_canonicalization": canonicalization_record,
        },
    }
    adapter_payload = _json_payload(adapter)
    provenance = {
        "schema": SCHEMA,
        "operation": "offline_merge_immutable_stage1_iid_analysis_reports",
        "population_manifest": str(population_path),
        "population_manifest_sha256": _sha256_bytes(population_payload),
        "population_fingerprint": population,
        "source_reports": report_records,
        "raw_preflight_reports": preflight_records,
        "rmct_control_canonicalization": canonicalization_record,
        "cell_sources": {key: cell_sources[key] for key in sorted(cell_sources)},
        "checks": {
            "analysis_reports_use_final_paired_schema_and_luna_pin": True,
            "all_nine_conditions_and_eighteen_cells_present": True,
            "all_cells_have_exact_100_plus_100_population": True,
            "bct_main_and_control_use_backend_matched_native_hf_peft": True,
            "bct_main_and_control_bind_raw_adapter_config_and_manifest_hashes": True,
            "vllm_lora_conditions_bind_compatibility_adapter_and_parity_attestation": True,
            "pooled_counts_and_rates_recompute_from_per_dataset_cells": True,
            "raw_preflights_bind_source_split_and_per_dataset_ids": True,
            "analysis_raw_hashes_match_preflight_raw_hashes": True,
            "host_local_manifest_hashes_match_between_each_analysis_and_preflight": True,
            "source_reports_or_preflights_modified": False,
            "accelerated_rmct_control_canonicalization_attested": canonicalization_record is not None,
        },
        "final_adapter_sha256": _sha256_bytes(adapter_payload),
    }
    return adapter, provenance


def _check_output(path: Path, payload: bytes) -> str:
    if path.exists():
        if not path.is_file():
            raise FileExistsError(f"final-matrix output is not a file: {path}")
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing final-matrix output: {path}")
        return "resumed"
    return "written"


def write_final_matrix(
    adapter: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    output: str | Path,
    provenance_output: str | Path,
    inputs: Sequence[str | Path],
) -> tuple[str, str]:
    """Atomically publish the new adapter/provenance pair without touching inputs."""

    output_path = Path(output).resolve()
    provenance_path = Path(provenance_output).resolve()
    input_paths = {Path(path).resolve() for path in inputs}
    if output_path == provenance_path:
        raise ValueError("final adapter and provenance outputs must be distinct")
    if output_path in input_paths or provenance_path in input_paths:
        raise ValueError("final-matrix output must not overwrite an immutable input")
    adapter_payload = _json_payload(adapter)
    provenance_payload = _json_payload(provenance)
    adapter_status = _check_output(output_path, adapter_payload)
    provenance_status = _check_output(provenance_path, provenance_payload)
    # Check both destinations first: a conflicting old provenance must not
    # leave a new adapter published alone.
    for path in (output_path, provenance_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if adapter_status == "written":
        _write_new(output_path, adapter_payload)
    if provenance_status == "written":
        _write_new(provenance_path, provenance_payload)
    return adapter_status, provenance_status


def _write_new(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def merge_and_write(
    reports: Sequence[str | Path],
    preflights: Sequence[str | Path],
    population_manifest: str | Path,
    *,
    output: str | Path,
    provenance_output: str | Path,
    rmct_control_canonicalization: str | Path | None = None,
) -> tuple[str, str]:
    """Build, validate, and write the final matrix pair."""

    adapter, provenance = build_final_matrix(
        reports,
        preflights,
        population_manifest,
        rmct_control_canonicalization=rmct_control_canonicalization,
    )
    inputs: list[str | Path] = [*reports, *preflights, population_manifest]
    if rmct_control_canonicalization is not None:
        inputs.append(rmct_control_canonicalization)
    return write_final_matrix(
        adapter,
        provenance,
        output=output,
        provenance_output=provenance_output,
        inputs=inputs,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, type=Path, help="immutable analysis report (repeat)")
    parser.add_argument(
        "--preflight",
        action="append",
        required=True,
        type=Path,
        help="immutable raw-preflight report, one per final condition (repeat)",
    )
    parser.add_argument("--population-manifest", required=True, type=Path, help="canonical frozen no-CoT manifest")
    parser.add_argument("--output", required=True, type=Path, help="new plot-ready final analysis adapter")
    parser.add_argument("--provenance-output", required=True, type=Path, help="new final-matrix provenance JSON")
    parser.add_argument(
        "--rmct-control-canonicalization",
        type=Path,
        help="required when canonical RMCT-control retains accelerated raw-source provenance",
    )
    args = parser.parse_args(argv)
    try:
        statuses = merge_and_write(
            args.report,
            args.preflight,
            args.population_manifest,
            output=args.output,
            provenance_output=args.provenance_output,
            rmct_control_canonicalization=args.rmct_control_canonicalization,
        )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"analysis={statuses[0]}: {args.output.resolve()}")
    print(f"provenance={statuses[1]}: {args.provenance_output.resolve()}")


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
