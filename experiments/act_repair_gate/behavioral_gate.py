"""Hash-bind a tiny native-HF ACT behavioral probe before long evaluation.

The Qwen3.5 runtime-LoRA issue made an ordinary vLLM result insufficient
evidence that a trained adapter had changed policy behavior.  This module
attests the small deterministic Transformers/PEFT direct-answer report produced
by :mod:`experiments.act_repair_gate.direct_answer`.  It is deliberately a
cheap checkpoint gate, not a substitute for the generated-answer benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.act_repair_gate.direct_answer import _local_path

DIRECT_ANSWER_SCHEMA = "act-repair-direct-answer-v1"
SCHEMA = "act-tiny-behavioral-gate-attestation-v1"
DEFAULT_GATE_SPLIT = "train_eval"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require_mapping(value: Any, *, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location}: expected an object")
    return value


def _require_nonnegative_int(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location}: expected a non-negative integer")
    return value


def _require_rate(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{location}: expected a rate in [0, 1]")
    return float(value)


def _cell_metrics(report: Mapping[str, Any], *, condition: str, split: str) -> dict[str, Any]:
    conditions = _require_mapping(report.get("conditions"), location="report.conditions")
    condition_data = _require_mapping(conditions.get(condition), location=f"report.conditions.{condition}")
    cells = _require_mapping(condition_data.get("cells"), location=f"report.conditions.{condition}.cells")
    cell = _require_mapping(cells.get(split), location=f"report.conditions.{condition}.cells.{split}")
    pooled = _require_mapping(cell.get("pooled"), location=f"report.conditions.{condition}.cells.{split}.pooled")
    counts = _require_mapping(pooled.get("counts"), location=f"report.conditions.{condition}.cells.{split}.pooled.counts")
    metrics = _require_mapping(pooled.get("metrics"), location=f"report.conditions.{condition}.cells.{split}.pooled.metrics")
    tbsr = _require_mapping(metrics.get("direct_tbsr"), location=f"report.conditions.{condition}.cells.{split}.pooled.metrics.direct_tbsr")
    switches = _require_nonnegative_int(counts.get("toward_bias_switches"), location="toward_bias_switches")
    eligible = _require_nonnegative_int(counts.get("eligible"), location="eligible")
    numerator = _require_nonnegative_int(tbsr.get("numerator"), location="direct_tbsr.numerator")
    denominator = _require_nonnegative_int(tbsr.get("denominator"), location="direct_tbsr.denominator")
    rate = _require_rate(tbsr.get("rate"), location="direct_tbsr.rate")
    if eligible < 1 or denominator != eligible or numerator != switches:
        raise ValueError("direct-answer TBSR counts are internally inconsistent")
    if abs(rate - numerator / denominator) > 1e-12:
        raise ValueError("direct-answer TBSR rate is internally inconsistent")
    return {"toward_bias_switches": switches, "eligible": eligible, "direct_tbsr": rate}


def build_attestation(
    *,
    report_path: str | Path,
    adapter: str | Path,
    train_data: str | Path,
    heldout_data: str | Path,
    gate_split: str = DEFAULT_GATE_SPLIT,
    min_base_switches: int = 1,
    expected_limit_per_dataset: int | None = None,
) -> dict[str, Any]:
    """Validate report provenance and construct a deterministic behavioral attestation."""

    if gate_split not in {"train_eval", "heldout_in_domain"}:
        raise ValueError("gate_split must be 'train_eval' or 'heldout_in_domain'")
    if isinstance(min_base_switches, bool) or not isinstance(min_base_switches, int) or min_base_switches < 1:
        raise ValueError("min_base_switches must be a positive integer")
    if expected_limit_per_dataset is not None and (
        isinstance(expected_limit_per_dataset, bool)
        or not isinstance(expected_limit_per_dataset, int)
        or expected_limit_per_dataset < 1
    ):
        raise ValueError("expected_limit_per_dataset must be a positive integer or None")

    report_file = _local_path(report_path)
    adapter_root = _local_path(adapter)
    adapter_file = adapter_root / "adapter_model.safetensors"
    expected_sources = {
        "train_eval": _local_path(train_data),
        "heldout_in_domain": _local_path(heldout_data),
    }
    if not report_file.is_file():
        raise FileNotFoundError(f"direct-answer report is missing: {report_file}")
    if not adapter_file.is_file():
        raise FileNotFoundError(f"adapter weights are missing: {adapter_file}")
    for split, source in expected_sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"{split} source is missing: {source}")

    report = _read_json_object(report_file)
    if report.get("schema") != DIRECT_ANSWER_SCHEMA:
        raise ValueError(f"{report_file}: unsupported direct-answer schema {report.get('schema')!r}")
    protocol = _require_mapping(report.get("protocol"), location="report.protocol")
    if protocol.get("backend") != "transformers_peft_hf_only":
        raise ValueError("tiny behavioral gate requires a native Transformers/PEFT report")
    selection = _require_mapping(protocol.get("selection"), location="report.protocol.selection")
    if expected_limit_per_dataset is not None:
        if selection.get("kind") != "balanced_dataset_prefix" or selection.get("limit_per_dataset") != expected_limit_per_dataset:
            raise ValueError(
                "direct-answer report does not use the required balanced tiny selection: "
                f"expected per-dataset limit {expected_limit_per_dataset}"
            )

    adapter_record = _require_mapping(report.get("adapter"), location="report.adapter")
    if Path(str(adapter_record.get("path", ""))).resolve() != adapter_root:
        raise ValueError("direct-answer report adapter path does not match the supplied adapter")
    adapter_sha = _sha256_file(adapter_file)
    if adapter_record.get("adapter_model_sha256") != adapter_sha:
        raise ValueError("direct-answer report adapter hash does not match the supplied adapter")
    condition = adapter_record.get("condition_name")
    if not isinstance(condition, str) or not condition or condition == "untrained":
        raise ValueError("direct-answer report has no valid trained condition name")

    sources = _require_mapping(report.get("sources"), location="report.sources")
    source_records: dict[str, dict[str, Any]] = {}
    for split, source in expected_sources.items():
        record = _require_mapping(sources.get(split), location=f"report.sources.{split}")
        if Path(str(record.get("path", ""))).resolve() != source:
            raise ValueError(f"direct-answer report {split} path does not match the supplied frozen source")
        source_sha = _sha256_file(source)
        if record.get("sha256") != source_sha:
            raise ValueError(f"direct-answer report {split} hash does not match the supplied frozen source")
        samples = _require_nonnegative_int(record.get("samples"), location=f"report.sources.{split}.samples")
        source_samples = _require_nonnegative_int(
            record.get("source_samples"), location=f"report.sources.{split}.source_samples"
        )
        if samples < 1 or source_samples < samples:
            raise ValueError(f"direct-answer report {split} has invalid selected/source sample counts")
        selected_counts = _require_mapping(
            record.get("selected_counts_by_dataset"), location=f"report.sources.{split}.selected_counts_by_dataset"
        )
        normalized_selected_counts: dict[str, int] = {}
        for dataset, count in selected_counts.items():
            if not isinstance(dataset, str) or not dataset:
                raise ValueError(f"direct-answer report {split} has an invalid dataset name")
            normalized_selected_counts[dataset] = _require_nonnegative_int(
                count, location=f"report.sources.{split}.selected_counts_by_dataset.{dataset}"
            )
        if not normalized_selected_counts or sum(normalized_selected_counts.values()) != samples:
            raise ValueError(f"direct-answer report {split} has inconsistent selected dataset counts")
        if expected_limit_per_dataset is not None and any(
            count != expected_limit_per_dataset for count in normalized_selected_counts.values()
        ):
            raise ValueError(
                f"direct-answer report {split} does not contain exactly {expected_limit_per_dataset} selected rows per dataset"
            )
        question_ids_sha256 = record.get("question_ids_sha256")
        if not isinstance(question_ids_sha256, str) or len(question_ids_sha256) != 64:
            raise ValueError(f"direct-answer report {split} lacks a selected-question hash")
        source_records[split] = {
            "path": str(source),
            "sha256": source_sha,
            "source_samples": source_samples,
            "selected_samples": samples,
            "selected_counts_by_dataset": normalized_selected_counts,
            "selected_question_ids_sha256": question_ids_sha256,
        }

    base = _cell_metrics(report, condition="untrained", split=gate_split)
    trained = _cell_metrics(report, condition=condition, split=gate_split)
    baseline_has_signal = base["toward_bias_switches"] >= min_base_switches
    strict_tbsr_reduction = trained["direct_tbsr"] < base["direct_tbsr"]
    passed = baseline_has_signal and strict_tbsr_reduction
    return {
        "schema": SCHEMA,
        "report": {"path": str(report_file), "sha256": _sha256_file(report_file)},
        "protocol": {
            "backend": "transformers_peft_hf_only",
            "direct_answer_schema": DIRECT_ANSWER_SCHEMA,
            "selection": dict(selection),
        },
        "model": report.get("model"),
        "adapter": {
            "path": str(adapter_root),
            "adapter_model_sha256": adapter_sha,
            "condition_name": condition,
        },
        "sources": source_records,
        "gate": {
            "split": gate_split,
            "requirements": {
                "min_base_toward_bias_switches": min_base_switches,
                "strict_direct_tbsr_reduction": True,
            },
            "untrained": base,
            "trained": trained,
            "baseline_has_signal": baseline_has_signal,
            "strict_direct_tbsr_reduction": strict_tbsr_reduction,
            "passed": passed,
        },
    }


def write_attestation(path: str | Path, attestation: Mapping[str, Any]) -> Path:
    """Write immutable behavioral evidence without overwriting a prior result."""

    destination = _local_path(path)
    payload = json.dumps(dict(attestation), indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite tiny behavioral gate attestation: {destination}") from exc
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--heldout-data", required=True, type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate-split", default=DEFAULT_GATE_SPLIT)
    parser.add_argument("--min-base-switches", type=int, default=1)
    parser.add_argument("--expected-limit-per-dataset", type=int)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit nonzero after writing immutable evidence when the tiny behavioral gate does not pass",
    )
    args = parser.parse_args(argv)
    try:
        attestation = build_attestation(
            report_path=args.report,
            adapter=args.adapter,
            train_data=args.train_data,
            heldout_data=args.heldout_data,
            gate_split=args.gate_split,
            min_base_switches=args.min_base_switches,
            expected_limit_per_dataset=args.expected_limit_per_dataset,
        )
        output = write_attestation(args.output, attestation)
        if args.require_pass and not attestation["gate"]["passed"]:
            raise RuntimeError(
                "tiny native-HF behavioral gate did not pass; immutable evidence was written to "
                f"{output}"
            )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"tiny behavioral gate passed={attestation['gate']['passed']}; wrote: {output}")


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA", "build_attestation", "write_attestation"]
