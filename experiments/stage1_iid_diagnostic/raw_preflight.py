"""Fail-closed provenance check before paid Stage 1 Luna grading.

The model-free paired-switch gate already validates the local score schema.
This module adds the missing staging boundary: it proves that the four native
biased logs for one condition were generated from the exact frozen train/IID
splits before their responses are sent to a model-based acknowledgement grader.

It is deliberately CPU-only. Run it on the generation host, where the frozen
split JSONLs named by the immutable manifest are still available, then copy its
small JSON report alongside the raw logs. A later grading command must only use
the files and SHA-256 values recorded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.stage1_iid_diagnostic import gate_analysis
from experiments.stage1_iid_diagnostic.prepare import (
    BIAS_TYPE,
    DATASETS,
    HELDOUT_COUNTS,
    MANIFEST_KIND,
    PROMPT_STYLE,
    SCHEMA_VERSION,
    SOURCE_COUNTS,
    SOURCE_ROWS,
    SOURCE_SHA256,
    TRAIN_EVAL_COUNTS,
)


PREFLIGHT_SCHEMA = "stage1-iid-raw-preflight-v1"
_SPLITS = gate_analysis.SPLITS
_COUNTS_BY_SPLIT = {
    "train_eval": TRAIN_EVAL_COUNTS,
    "heldout_in_domain": HELDOUT_COUNTS,
}
_RUNTIME_PROFILES = frozenset({"vllm", "hf-peft"})
_HF_PEFT_BASE_MODEL = "Qwen/Qwen3.5-9B"
_HF_PEFT_MODEL = f"hf/{_HF_PEFT_BASE_MODEL}"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _read_frozen_split(path: Path, *, split: str, entry: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Verify one locally available split against its immutable manifest entry."""

    if not path.is_file():
        raise FileNotFoundError(f"supplied frozen {split} split does not exist: {path}")
    payload = path.read_bytes()
    if entry.get("content_sha256") != _sha256_bytes(payload):
        raise ValueError(f"supplied frozen {split} split has a different SHA-256 than its manifest")
    if entry.get("byte_count") != len(payload):
        raise ValueError(f"supplied frozen {split} split has a different byte count than its manifest")
    ids: list[str] = []
    by_dataset: dict[str, list[str]] = {dataset: [] for dataset in DATASETS}
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid frozen JSONL row") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_number}: frozen row must be an object")
        question_id = row.get("question_id")
        dataset = row.get("source_dataset")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{path}:{line_number}: frozen row has no non-empty question_id")
        if dataset not in DATASETS:
            raise ValueError(f"{path}:{line_number}: frozen row has unexpected source_dataset {dataset!r}")
        if row.get("bias_type") != BIAS_TYPE or row.get("prompt_style") != PROMPT_STYLE:
            raise ValueError(f"{path}:{line_number}: frozen row conflicts with the Stage 1 bias/prompt contract")
        ids.append(question_id)
        by_dataset[dataset].append(question_id)
    expected_counts = _COUNTS_BY_SPLIT[split]
    if len(ids) != entry.get("row_count") or len(ids) != sum(expected_counts.values()):
        raise ValueError(f"supplied frozen {split} split has an unexpected row count")
    if len(ids) != len(set(ids)):
        raise ValueError(f"supplied frozen {split} split has duplicate question IDs")
    if tuple(ids) != tuple(entry.get("question_ids", ())):
        raise ValueError(f"supplied frozen {split} split IDs do not match its manifest")
    counts = {dataset: len(by_dataset[dataset]) for dataset in DATASETS}
    if counts != dict(expected_counts) or entry.get("counts_by_dataset") != dict(expected_counts):
        raise ValueError(f"supplied frozen {split} split has unexpected per-dataset counts")
    return tuple(ids), {dataset: tuple(by_dataset[dataset]) for dataset in DATASETS}


def _manifest_and_expected_ids(
    manifest_path: Path,
    split_files: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[tuple[str, str], tuple[str, ...]]]:
    """Validate the portable manifest contract and its supplied local split copies."""

    document = _read_json(manifest_path)
    if document.get("schema_version") != SCHEMA_VERSION or document.get("kind") != MANIFEST_KIND:
        raise ValueError("unsupported Stage 1 IID diagnostic manifest schema")
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Stage 1 manifest has no source object")
    expected_source = {
        "content_sha256": SOURCE_SHA256,
        "row_count": SOURCE_ROWS,
        "counts_by_dataset": SOURCE_COUNTS,
        "bias_type": BIAS_TYPE,
        "prompt_style": PROMPT_STYLE,
    }
    for field, expected in expected_source.items():
        if source.get(field) != expected:
            raise ValueError(f"Stage 1 manifest source.{field} does not match the pinned diagnostic contract")
    splits = document.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(_SPLITS):
        raise ValueError("Stage 1 manifest must contain exactly the two diagnostic splits")
    if set(split_files) != set(_SPLITS):
        raise ValueError("supply exactly train_eval and heldout_in_domain frozen split files")

    expected: dict[tuple[str, str], tuple[str, ...]] = {}
    all_ids: dict[str, tuple[str, ...]] = {}
    for split in _SPLITS:
        entry = splits[split]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Stage 1 manifest {split} entry must be an object")
        ids, by_dataset = _read_frozen_split(Path(split_files[split]), split=split, entry=entry)
        all_ids[split] = ids
        for dataset in DATASETS:
            expected[(split, dataset)] = by_dataset[dataset]
    if not set(all_ids["train_eval"]).isdisjoint(all_ids["heldout_in_domain"]):
        raise ValueError("Stage 1 manifest diagnostic split IDs overlap")
    return document, expected


def _configuration_mapping(value: Any) -> Mapping[str, Any]:
    """Read an Inspect configuration whether it is a dict or Pydantic model.

    ``EvalLog.eval.model_generate_config`` is an Inspect ``GenerateConfig``
    object in current Inspect releases, whereas ``model_args`` is an ordinary
    dict.  Keep the preflight's decode checks equally strict for both
    representations rather than accidentally treating a populated
    ``GenerateConfig`` as an empty mapping.
    """

    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        dumped = legacy_dict()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _assert_expected_model(path: Path, *, base_model: str | None, checkpoint: str | None) -> str:
    """Optionally bind the preflight to checkpoint and Stage-1 decode identity."""

    if base_model is None and checkpoint is None:
        return ""
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - requires Inspect only at runtime
        raise RuntimeError("Inspect AI is required to inspect raw Stage 1 logs") from exc
    log = read_eval_log(str(path), header_only=True)
    evaluation = gate_analysis._attribute(log, "eval")
    model = str(gate_analysis._attribute(evaluation, "model", "") or "")
    if base_model is not None:
        # A vLLM adapter is recorded as ``vllm/<base>:<adapter>``, while an
        # unadapted native vLLM run has no trailing colon at all.  Accept both
        # exact representations; the latter is the required base-model
        # baseline for the final comparison.
        expected_bare_model = f"vllm/{base_model}"
        if model != expected_bare_model and f"/{base_model}:" not in model:
            raise ValueError(f"raw log has unexpected base model: {path}: {model!r}")
    if checkpoint is not None and not model.endswith(f":{checkpoint}"):
        raise ValueError(f"raw log has unexpected checkpoint: {path}: {model!r}")
    generation = _configuration_mapping(gate_analysis._attribute(evaluation, "model_generate_config", {}))
    expected_generation = {"max_tokens": 20480, "temperature": 1.0, "top_p": 0.95, "top_k": 20}
    for field, expected in expected_generation.items():
        if generation.get(field) != expected:
            raise ValueError(f"raw log has unexpected {field} decode setting: {path}")
    model_args = _configuration_mapping(gate_analysis._attribute(evaluation, "model_args", {}))
    expected_model_args = {
        "gpu_memory_utilization": 0.9,
        "max_model_len": 32768,
        "language_model_only": True,
        "max_num_seqs": 256,
    }
    for field, expected in expected_model_args.items():
        if model_args.get(field) != expected:
            raise ValueError(f"raw log has unexpected {field} model setting: {path}")
    return model


def _assert_expected_hf_peft_model(
    path: Path,
    *,
    base_model: str,
    checkpoint: str,
    max_connections: int,
) -> tuple[str, dict[str, Any]]:
    """Bind one log to the explicit local HF/PEFT fallback runtime.

    This is intentionally not a relaxed form of the native-vLLM check above.
    The Qwen3.5 control adapter is evaluated through Transformers/PEFT only
    after its weak effect could not satisfy the independent vLLM parity gate.
    Every runtime-affecting field used by that fallback is therefore checked
    from the Inspect evaluation header before a model-based grader can consume
    any generated response.
    """

    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - requires Inspect only at runtime
        raise RuntimeError("Inspect AI is required to inspect raw Stage 1 logs") from exc
    log = read_eval_log(str(path), header_only=True)
    evaluation = gate_analysis._attribute(log, "eval")
    model = str(gate_analysis._attribute(evaluation, "model", "") or "")
    if model != _HF_PEFT_MODEL:
        raise ValueError(f"raw log has unexpected HF/PEFT evaluator model: {path}: {model!r}")

    metadata = _configuration_mapping(gate_analysis._attribute(evaluation, "metadata", {}))
    expected_metadata = {
        "checkpoint": checkpoint,
        "checkpoint_backend": "local",
        "base_model": base_model,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"raw log has unexpected HF/PEFT metadata.{field}: {path}")

    # ``local_checkpoint_model`` consumes ``provider`` before it creates the
    # Inspect HF model.  Consequently the EvalLog's actual provider arguments
    # contain device/dtype, while the runner's immutable metadata preserves the
    # full requested local-checkpoint contract including ``provider='hf'``.
    # Validate both representations rather than requiring an impossible field
    # from the former or trusting the latter in isolation.
    model_args = _configuration_mapping(gate_analysis._attribute(evaluation, "model_args", {}))
    expected_model_args = {
        "device": "cuda:0",
        "dtype": "bfloat16",
    }
    for field, expected in expected_model_args.items():
        if model_args.get(field) != expected:
            raise ValueError(f"raw log has unexpected HF/PEFT model_args.{field}: {path}")

    requested_model_args = _configuration_mapping(metadata.get("model_args", {}))
    expected_requested_model_args = {"provider": "hf", **expected_model_args}
    for field, expected in expected_requested_model_args.items():
        if requested_model_args.get(field) != expected:
            raise ValueError(f"raw log has unexpected HF/PEFT metadata.model_args.{field}: {path}")

    generation = _configuration_mapping(gate_analysis._attribute(evaluation, "model_generate_config", {}))
    expected_generation = {
        "max_tokens": 20480,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_connections": max_connections,
    }
    for field, expected in expected_generation.items():
        if generation.get(field) != expected:
            raise ValueError(f"raw log has unexpected HF/PEFT {field} decode setting: {path}")

    return model, {
        "profile": "hf-peft",
        "provider": requested_model_args["provider"],
        "device": model_args["device"],
        "dtype": model_args["dtype"],
        "max_connections": generation["max_connections"],
        "checkpoint": metadata["checkpoint"],
        "checkpoint_backend": metadata["checkpoint_backend"],
        "base_model": metadata["base_model"],
    }


def _validate_runtime_profile(
    *,
    runtime_profile: str | None,
    expected_base_model: str | None,
    expected_checkpoint: str | None,
    expected_max_connections: int | None,
) -> None:
    """Validate optional profile-specific preflight arguments before discovery."""

    if runtime_profile is None:
        if expected_max_connections is not None:
            raise ValueError("--expected-max-connections requires --runtime-profile hf-peft")
        return
    if runtime_profile not in _RUNTIME_PROFILES:
        raise ValueError(f"unsupported runtime profile {runtime_profile!r}; choose from {sorted(_RUNTIME_PROFILES)}")
    if runtime_profile != "hf-peft":
        if expected_max_connections is not None:
            raise ValueError("--expected-max-connections is valid only with --runtime-profile hf-peft")
        return
    if expected_base_model != _HF_PEFT_BASE_MODEL:
        raise ValueError(f"--runtime-profile hf-peft requires --expected-base-model {_HF_PEFT_BASE_MODEL}")
    if not isinstance(expected_checkpoint, str) or not expected_checkpoint:
        raise ValueError("--runtime-profile hf-peft requires a non-empty --expected-checkpoint")
    if isinstance(expected_max_connections, bool) or not isinstance(expected_max_connections, int) or expected_max_connections < 1:
        raise ValueError("--runtime-profile hf-peft requires --expected-max-connections >= 1")


def preflight_raw_logs(
    raw_root: str | Path,
    manifest: str | Path,
    *,
    split_files: Mapping[str, str | Path],
    condition: str,
    expected_base_model: str | None = None,
    expected_checkpoint: str | None = None,
    runtime_profile: str | None = None,
    expected_max_connections: int | None = None,
) -> dict[str, Any]:
    """Return a hash-bound native-log report or reject the raw staging set.

    ``raw_root`` must contain exactly one successful native biased log per
    ``split × dataset`` cell. ``gate_analysis`` additionally checks the
    paired-switch representation and every sample's biased metadata.
    """

    if not isinstance(condition, str) or not condition.strip():
        raise ValueError("condition must be a non-empty string")
    _validate_runtime_profile(
        runtime_profile=runtime_profile,
        expected_base_model=expected_base_model,
        expected_checkpoint=expected_checkpoint,
        expected_max_connections=expected_max_connections,
    )
    raw_directory = Path(raw_root).resolve()
    manifest_path = Path(manifest).resolve()
    normalized_files = {str(split): Path(path).resolve() for split, path in split_files.items()}
    document, expected_ids = _manifest_and_expected_ids(manifest_path, normalized_files)
    loaded = gate_analysis.scan_variant_logs(raw_directory, prompt_variant="native")
    expected_cells = {(split, dataset) for split in _SPLITS for dataset in DATASETS}
    if set(loaded) != expected_cells:
        missing = sorted(expected_cells - set(loaded))
        unexpected = sorted(set(loaded) - expected_cells)
        raise ValueError(f"native raw logs are incomplete or unexpected; missing={missing}, unexpected={unexpected}")

    sources: list[dict[str, Any]] = []
    model_identity: str | None = None
    for split, dataset in sorted(expected_cells):
        item = loaded[(split, dataset)]
        header = item.header
        if header.question_ids != expected_ids[(split, dataset)]:
            raise ValueError(f"raw {condition}/{split}/{dataset} question IDs do not match the frozen split")
        if header.prompt_style != PROMPT_STYLE:
            raise ValueError(f"raw {condition}/{split}/{dataset} has unexpected prompt style {header.prompt_style!r}")
        if header.source_identity_digest != SOURCE_SHA256:
            raise ValueError(f"raw {condition}/{split}/{dataset} has unexpected frozen-source digest")
        if runtime_profile == "hf-peft":
            observed_model, observed_runtime = _assert_expected_hf_peft_model(
                item.path,
                base_model=expected_base_model,
                checkpoint=expected_checkpoint,
                max_connections=expected_max_connections,
            )
        else:
            observed_model = _assert_expected_model(
                item.path,
                base_model=expected_base_model,
                checkpoint=expected_checkpoint,
            )
            observed_runtime = {"profile": "vllm"} if runtime_profile == "vllm" and observed_model else None
        if observed_model:
            if model_identity is None:
                model_identity = observed_model
            elif model_identity != observed_model:
                raise ValueError("native raw cells do not use one identical checkpoint/model identity")
        source = {
            "split": split,
            "dataset": dataset,
            "raw_log": str(item.path),
            "raw_log_sha256": item.sha256,
            "created": header.created,
            "sample_count": len(item.rows),
            "question_ids_sha256": gate_analysis._ids_sha256(header.question_ids),
            "unbiased_log": header.unbiased_log,
            "prompt_style": header.prompt_style,
            "source_identity_digest": header.source_identity_digest,
            "variant_file": header.variant_file,
            "model": observed_model or None,
        }
        if observed_runtime is not None:
            source["runtime"] = observed_runtime
        sources.append(source)
    contract = {
        "source_sha256": document["source"]["content_sha256"],
        "bias_type": BIAS_TYPE,
        "prompt_style": PROMPT_STYLE,
        "native_variant_file": None,
        "include_bias_acknowledged": False,
        "grader_model": None,
        "expected_base_model": expected_base_model,
        "expected_checkpoint": expected_checkpoint,
    }
    if runtime_profile is not None:
        contract["runtime_profile"] = runtime_profile
    if runtime_profile == "hf-peft":
        contract["expected_max_connections"] = expected_max_connections
    return {
        "schema": PREFLIGHT_SCHEMA,
        "condition": condition,
        "raw_root": str(raw_directory),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "frozen_splits": {
            split: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for split, path in sorted(normalized_files.items())
        },
        "contract": contract,
        "sources": sources,
    }


def _parse_split_files(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        split, separator, raw_path = value.partition("=")
        if not separator or not split or not raw_path:
            raise ValueError("--split-file values must use SPLIT=PATH")
        if split in parsed:
            raise ValueError(f"duplicate --split-file for {split!r}")
        parsed[split] = Path(raw_path)
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-log-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-file", required=True, action="append", default=[], metavar="SPLIT=PATH")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--expected-base-model")
    parser.add_argument("--expected-checkpoint")
    parser.add_argument(
        "--runtime-profile",
        choices=sorted(_RUNTIME_PROFILES),
        help="Optional evaluator runtime contract; hf-peft validates the explicit local Transformers/PEFT fallback",
    )
    parser.add_argument(
        "--expected-max-connections",
        type=int,
        help="Required with --runtime-profile hf-peft; exact safe HF generation batch size",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = preflight_raw_logs(
            args.raw_log_root,
            args.manifest,
            split_files=_parse_split_files(args.split_file),
            condition=args.condition,
            expected_base_model=args.expected_base_model,
            expected_checkpoint=args.expected_checkpoint,
            runtime_profile=args.runtime_profile,
            expected_max_connections=args.expected_max_connections,
        )
        status = gate_analysis.write_report(args.output, report)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{status}: {args.output.resolve()}")


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
