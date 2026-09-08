"""Fail-closed raw-log preflight for the no-CoT Stage 1 IID diagnostic.

This is the no-CoT counterpart of the historical raw-log gate.  It shares the
generic paired-switch/header parser and evaluator-runtime checks, but binds
every accepted cell to the recovered source digest and ``prompt_style: none``
before any response may be sent to a model-based acknowledgement grader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm.evals.qwen35_vllm_attestation import (
    COMPATIBILITY_MANIFEST_SCHEMA,
    is_verified_qwen35_vllm_compat_adapter,
)
from experiments.stage1_iid_diagnostic import gate_analysis
from experiments.stage1_iid_diagnostic.raw_preflight import (
    PREFLIGHT_SCHEMA,
    _assert_expected_hf_peft_model,
    _assert_expected_model,
    _sha256_file,
    _validate_runtime_profile,
)
from experiments.stage1_iid_diagnostic_none.prepare import (
    BIAS_TYPE,
    DATASETS,
    HELDOUT_COUNTS,
    PROMPT_STYLE,
    SOURCE_SHA256,
    TRAIN_EVAL_COUNTS,
    validate_manifest,
)
from experiments.rmct_paper_vast_dense_models.stage1.repaired_act_evaluation_guard import (
    validate_repaired_act_evaluation_chain_attestation,
)

_SPLITS = gate_analysis.SPLITS
_COUNTS_BY_SPLIT = {
    "train_eval": TRAIN_EVAL_COUNTS,
    "heldout_in_domain": HELDOUT_COUNTS,
}
_QWEN35_BASE_MODEL = "Qwen/Qwen3.5-9B"
_VLLM_COMPATIBILITY_MANIFEST = "compatibility-manifest.json"
_VLLM_PARITY_ATTESTATION = "vllm-parity-attestation.json"
_VLLM_COMPATIBILITY_ADAPTER_IDENTITY_KEYS = frozenset(
    {
        "adapter_model_sha256",
        "adapter_config_sha256",
        "compatibility_manifest_sha256",
        "parity_attestation_sha256",
        "source_adapter_model_sha256",
    }
)
_VLLM_SOURCE_PREFIX = "base_model.model.model.layers."
_VLLM_DESTINATION_PREFIX = "base_model.model.model.language_model.layers."
# These names are reserved for the *fresh* corrected repaired-ACT trajectory.
# Historical/raw conditions retain their existing optional provenance paths,
# but a result published under either corrected name cannot bypass the tiny-HF
# and adapter-parity chain merely by omitting a CLI flag.
_REPAIRED_ACT_CHAIN_REQUIRED_CONDITIONS = frozenset(
    {"repaired-act-vllm-compat", "repaired-act-hf-peft"}
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    """Read one required immutable provenance object without guessing its shape."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return document


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_vllm_compatibility_adapter_identity(
    *,
    runtime_profile: str | None,
    expected_base_model: str | None,
    expected_checkpoint: str | None,
) -> dict[str, str] | None:
    """Bind a served Qwen3.5 vLLM LoRA copy to its immutable parity evidence.

    Qwen3.5's vLLM wrapper uses a translated adapter-key namespace.  A path in
    the EvalLog is not enough to prove that vLLM received that translated copy,
    nor that the copy still has the HF/vLLM parity evidence which prevents the
    historical silent-base failure.  For an adapted vLLM invocation, require
    the exact compatibility-copy bytes, its translation manifest, and a live
    immutable parity attestation before any raw response can be staged.

    The untrained vLLM baseline has no local checkpoint and intentionally has
    no such identity record.  Native HF/PEFT uses the separate raw-adapter
    identity contract above.
    """

    if runtime_profile != "vllm" or expected_checkpoint is None:
        return None
    if expected_base_model != _QWEN35_BASE_MODEL:
        raise ValueError(
            "a vLLM local checkpoint in the final no-CoT diagnostic requires "
            f"--expected-base-model {_QWEN35_BASE_MODEL}"
        )
    if not isinstance(expected_checkpoint, str) or not expected_checkpoint:
        raise ValueError("vLLM compatibility adapter identity requires a non-empty expected checkpoint")

    checkpoint = Path(expected_checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"vLLM compatibility adapter directory does not exist: {checkpoint}")
    paths = {
        "adapter_model_sha256": checkpoint / "adapter_model.safetensors",
        "adapter_config_sha256": checkpoint / "adapter_config.json",
        "compatibility_manifest_sha256": checkpoint / _VLLM_COMPATIBILITY_MANIFEST,
        "parity_attestation_sha256": checkpoint / _VLLM_PARITY_ATTESTATION,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"vLLM compatibility adapter artifact does not exist: {path}")

    # This validator follows every attestation reference (including the
    # composite ACT exception) and confirms that its parity report still names
    # this exact compatibility-copy adapter hash.
    if not is_verified_qwen35_vllm_compat_adapter(checkpoint):
        raise ValueError(
            "vLLM compatibility adapter lacks a valid immutable Qwen3.5 HF/vLLM parity attestation: "
            f"{checkpoint}"
        )

    compatibility_manifest = _read_json_object(
        paths["compatibility_manifest_sha256"],
        label="vLLM compatibility manifest",
    )
    if compatibility_manifest.get("schema") != COMPATIBILITY_MANIFEST_SCHEMA:
        raise ValueError("vLLM compatibility manifest has an unsupported schema")
    source = compatibility_manifest.get("source")
    destination = compatibility_manifest.get("destination")
    translation = compatibility_manifest.get("translation")
    if not isinstance(source, Mapping) or not isinstance(destination, Mapping) or not isinstance(translation, Mapping):
        raise ValueError("vLLM compatibility manifest has incomplete source/destination/translation evidence")
    source_path = source.get("path")
    if not isinstance(source_path, str) or not source_path or not Path(source_path).is_absolute():
        raise ValueError("vLLM compatibility manifest has no absolute raw-adapter source path")
    source_digest = source.get("adapter_model_sha256")
    if not _is_sha256(source_digest):
        raise ValueError("vLLM compatibility manifest has no valid raw-adapter SHA-256")
    if destination.get("path") != str(checkpoint):
        raise ValueError("vLLM compatibility manifest destination does not name the served compatibility adapter")
    adapter_digest = _sha256_file(paths["adapter_model_sha256"])
    if destination.get("adapter_model_sha256") != adapter_digest:
        raise ValueError("vLLM compatibility manifest destination adapter SHA-256 does not match adapter bytes")
    if translation.get("source_prefix") != _VLLM_SOURCE_PREFIX:
        raise ValueError("vLLM compatibility manifest has an unexpected source key prefix")
    if translation.get("destination_prefix") != _VLLM_DESTINATION_PREFIX:
        raise ValueError("vLLM compatibility manifest has an unexpected destination key prefix")
    tensor_count = _positive_int(translation.get("tensor_count"), label="vLLM compatibility manifest tensor_count")
    translated_tensor_count = _positive_int(
        translation.get("translated_tensor_count"),
        label="vLLM compatibility manifest translated_tensor_count",
    )
    if tensor_count != translated_tensor_count:
        raise ValueError("vLLM compatibility manifest translated tensor count does not match tensor count")
    if not _is_sha256(translation.get("translated_tensor_names_sha256")):
        raise ValueError("vLLM compatibility manifest has no valid translated tensor-name SHA-256")

    observed = {name: _sha256_file(path) for name, path in paths.items()}
    observed["source_adapter_model_sha256"] = str(source_digest)
    if set(observed) != _VLLM_COMPATIBILITY_ADAPTER_IDENTITY_KEYS:
        raise AssertionError("vLLM compatibility identity keys drifted from the final-matrix contract")
    return observed


def _validate_checkpoint_artifact_identity(
    *,
    runtime_profile: str | None,
    expected_checkpoint: str | None,
    expected_adapter_model_sha256: str | None,
    expected_adapter_config_sha256: str | None,
    expected_checkpoint_manifest_sha256: str | None,
) -> dict[str, str] | None:
    """Verify optional immutable PEFT artifact bytes before grading is allowed.

    An Inspect HF header identifies the checkpoint path used for generation,
    but a path alone is not an immutable checkpoint identity.  Native-HF BCT
    main/control recovery passes all three expected hashes, making the raw
    logs attributable to the exact adapter tensor, PEFT config, and CTM
    checkpoint manifest.
    """

    expected = {
        "adapter_model_sha256": expected_adapter_model_sha256,
        "adapter_config_sha256": expected_adapter_config_sha256,
        "checkpoint_manifest_sha256": expected_checkpoint_manifest_sha256,
    }
    if all(value is None for value in expected.values()):
        return None
    if runtime_profile != "hf-peft":
        raise ValueError("checkpoint artifact hashes require --runtime-profile hf-peft")
    if any(value is None for value in expected.values()):
        raise ValueError(
            "supply all of --expected-adapter-model-sha256, --expected-adapter-config-sha256, "
            "and --expected-checkpoint-manifest-sha256 together"
        )
    for name, digest in expected.items():
        if not _is_sha256(digest):
            raise ValueError(f"{name} must be a SHA-256 hex digest")
    if not isinstance(expected_checkpoint, str) or not expected_checkpoint:
        # The runtime-profile validator normally catches this first. Keep the
        # helper self-contained for programmatic callers.
        raise ValueError("checkpoint artifact hashes require an expected checkpoint")

    checkpoint = Path(expected_checkpoint)
    paths = {
        "adapter_model_sha256": checkpoint / "adapter_model.safetensors",
        "adapter_config_sha256": checkpoint / "adapter_config.json",
        "checkpoint_manifest_sha256": checkpoint / "manifest.json",
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"expected checkpoint artifact does not exist: {path}")
        digest = _sha256_file(path)
        if digest != expected[name]:
            raise ValueError(f"expected checkpoint {path.name} SHA-256 does not match {name}")
        observed[name] = digest
    return observed


def _read_frozen_split(
    path: Path,
    *,
    split: str,
    entry: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Prove a staged local split is the exact no-CoT frozen split."""

    if not path.is_file():
        raise FileNotFoundError(f"supplied no-CoT {split} split does not exist: {path}")
    payload = path.read_bytes()
    if entry.get("content_sha256") != _sha256_bytes(payload):
        raise ValueError(f"supplied no-CoT {split} split has a different SHA-256 than its manifest")
    if entry.get("byte_count") != len(payload):
        raise ValueError(f"supplied no-CoT {split} split has a different byte count than its manifest")
    ids: list[str] = []
    by_dataset: dict[str, list[str]] = {dataset: [] for dataset in DATASETS}
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"{path}:{line_number}: blank frozen no-CoT row")
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid frozen no-CoT JSONL row") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:{line_number}: frozen no-CoT row must be an object")
        question_id = row.get("question_id")
        dataset = row.get("source_dataset")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{path}:{line_number}: frozen no-CoT row has no non-empty question_id")
        if dataset not in DATASETS:
            raise ValueError(f"{path}:{line_number}: frozen no-CoT row has unexpected source_dataset {dataset!r}")
        if row.get("bias_type") != BIAS_TYPE or row.get("prompt_style") != PROMPT_STYLE:
            raise ValueError(f"{path}:{line_number}: frozen row conflicts with the no-CoT bias/prompt contract")
        ids.append(question_id)
        by_dataset[dataset].append(question_id)
    expected_counts = _COUNTS_BY_SPLIT[split]
    if len(ids) != entry.get("row_count") or len(ids) != sum(expected_counts.values()):
        raise ValueError(f"supplied no-CoT {split} split has an unexpected row count")
    if len(ids) != len(set(ids)):
        raise ValueError(f"supplied no-CoT {split} split has duplicate question IDs")
    if ids != entry.get("question_ids"):
        raise ValueError(f"supplied no-CoT {split} split IDs do not match its manifest")
    counts = {dataset: len(by_dataset[dataset]) for dataset in DATASETS}
    if counts != dict(expected_counts) or entry.get("counts_by_dataset") != dict(expected_counts):
        raise ValueError(f"supplied no-CoT {split} split has unexpected per-dataset counts")
    return {dataset: tuple(by_dataset[dataset]) for dataset in DATASETS}


def _manifest_and_expected_ids(
    manifest_path: Path,
    split_files: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[tuple[str, str], tuple[str, ...]]]:
    """Validate the source-attested manifest and staged split copies."""

    document = validate_manifest(manifest_path, verify_source=True)
    if set(split_files) != set(_SPLITS):
        raise ValueError("supply exactly train_eval and heldout_in_domain frozen no-CoT split files")
    expected: dict[tuple[str, str], tuple[str, ...]] = {}
    all_ids: dict[str, tuple[str, ...]] = {}
    for split in _SPLITS:
        entry = document["splits"][split]
        by_dataset = _read_frozen_split(Path(split_files[split]), split=split, entry=entry)
        all_ids[split] = tuple(entry["question_ids"])
        for dataset in DATASETS:
            expected[(split, dataset)] = by_dataset[dataset]
    if not set(all_ids["train_eval"]).isdisjoint(all_ids["heldout_in_domain"]):
        raise ValueError("no-CoT manifest diagnostic split IDs overlap")
    return document, expected


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
    expected_adapter_model_sha256: str | None = None,
    expected_adapter_config_sha256: str | None = None,
    expected_checkpoint_manifest_sha256: str | None = None,
    repaired_act_chain_attestation: str | Path | None = None,
) -> dict[str, Any]:
    """Return a hash-bound no-CoT raw-log report, or reject the staging set."""

    if not isinstance(condition, str) or not condition.strip():
        raise ValueError("condition must be a non-empty string")
    if condition in _REPAIRED_ACT_CHAIN_REQUIRED_CONDITIONS and repaired_act_chain_attestation is None:
        raise ValueError(
            f"fresh repaired-ACT condition {condition!r} requires --repaired-act-chain-attestation"
        )
    _validate_runtime_profile(
        runtime_profile=runtime_profile,
        expected_base_model=expected_base_model,
        expected_checkpoint=expected_checkpoint,
        expected_max_connections=expected_max_connections,
    )
    checkpoint_artifact_identity = _validate_checkpoint_artifact_identity(
        runtime_profile=runtime_profile,
        expected_checkpoint=expected_checkpoint,
        expected_adapter_model_sha256=expected_adapter_model_sha256,
        expected_adapter_config_sha256=expected_adapter_config_sha256,
        expected_checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256,
    )
    vllm_compatibility_adapter_identity = _validate_vllm_compatibility_adapter_identity(
        runtime_profile=runtime_profile,
        expected_base_model=expected_base_model,
        expected_checkpoint=expected_checkpoint,
    )
    repaired_act_evaluation_chain_identity: dict[str, str] | None = None
    if repaired_act_chain_attestation is not None:
        if runtime_profile not in {"vllm", "hf-peft"}:
            raise ValueError("repaired-ACT evaluation chain requires --runtime-profile vllm or hf-peft")
        if not expected_checkpoint:
            raise ValueError("repaired-ACT evaluation chain requires --expected-checkpoint")
        chain = validate_repaired_act_evaluation_chain_attestation(
            repaired_act_chain_attestation,
            expected_runtime_profile=runtime_profile,
            expected_checkpoint=expected_checkpoint if runtime_profile == "hf-peft" else None,
            expected_vllm_compat_adapter=expected_checkpoint if runtime_profile == "vllm" else None,
        )
        chain_attestation = chain["attestation"]
        raw_identity = chain["raw_checkpoint"]
        repaired_act_evaluation_chain_identity = {
            "chain_attestation_path": str(chain_attestation["path"]),
            "chain_attestation_sha256": str(chain_attestation["sha256"]),
            "raw_adapter_model_sha256": str(raw_identity["adapter_model_sha256"]),
        }
        if runtime_profile == "vllm":
            chain_compat = chain["runtime"]["vllm_compatibility_adapter"]
            if vllm_compatibility_adapter_identity is None:
                raise AssertionError("vLLM repaired-ACT chain lost compatibility adapter identity")
            for name in (
                "adapter_model_sha256",
                "adapter_config_sha256",
                "compatibility_manifest_sha256",
                "parity_attestation_sha256",
                "source_adapter_model_sha256",
            ):
                if chain_compat[name] != vllm_compatibility_adapter_identity[name]:
                    raise ValueError("repaired-ACT evaluation chain compatibility identity disagrees with raw preflight")
            repaired_act_evaluation_chain_identity["vllm_compatibility_adapter_sha256"] = str(
                chain_compat["adapter_model_sha256"]
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
        raise ValueError(f"native no-CoT raw logs are incomplete or unexpected; missing={missing}, unexpected={unexpected}")

    sources: list[dict[str, Any]] = []
    model_identity: str | None = None
    for split, dataset in sorted(expected_cells):
        item = loaded[(split, dataset)]
        header = item.header
        if header.question_ids != expected_ids[(split, dataset)]:
            raise ValueError(f"raw {condition}/{split}/{dataset} question IDs do not match the frozen no-CoT split")
        if header.prompt_style != PROMPT_STYLE:
            raise ValueError(f"raw {condition}/{split}/{dataset} has unexpected prompt style {header.prompt_style!r}")
        if header.source_identity_digest != SOURCE_SHA256:
            raise ValueError(f"raw {condition}/{split}/{dataset} has unexpected recovered-source digest")
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
                raise ValueError("native no-CoT raw cells do not use one identical checkpoint/model identity")
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
    if checkpoint_artifact_identity is not None:
        contract["checkpoint_artifact_identity"] = checkpoint_artifact_identity
    if vllm_compatibility_adapter_identity is not None:
        contract["vllm_compatibility_adapter_identity"] = vllm_compatibility_adapter_identity
    if repaired_act_evaluation_chain_identity is not None:
        contract["repaired_act_evaluation_chain_identity"] = repaired_act_evaluation_chain_identity
    return {
        "schema": PREFLIGHT_SCHEMA,
        "condition": condition,
        "raw_root": str(raw_directory),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "frozen_splits": {
            split: {"path": str(path), "sha256": _sha256_file(path)}
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
        choices=("vllm", "hf-peft"),
        help=(
            "Optional evaluator runtime contract; vllm binds an adapted checkpoint to its verified "
            "Qwen3.5 compatibility/parity artifacts, while hf-peft validates the explicit local "
            "Transformers/PEFT fallback"
        ),
    )
    parser.add_argument(
        "--expected-max-connections",
        type=int,
        help="Required with --runtime-profile hf-peft; exact safe HF generation batch size",
    )
    parser.add_argument(
        "--expected-adapter-model-sha256",
        help="Optional immutable adapter_model.safetensors SHA-256; requires all three checkpoint artifact hashes",
    )
    parser.add_argument(
        "--expected-adapter-config-sha256",
        help="Optional immutable adapter_config.json SHA-256; requires all three checkpoint artifact hashes",
    )
    parser.add_argument(
        "--expected-checkpoint-manifest-sha256",
        help="Optional immutable checkpoint manifest.json SHA-256; requires all three checkpoint artifact hashes",
    )
    parser.add_argument(
        "--repaired-act-chain-attestation",
        type=Path,
        help=(
            "Required handoff from the fresh repaired-ACT pre-launch wrapper when this condition uses that "
            "trajectory; revalidates the passed tiny native-HF gate and, for vLLM, the exact compatibility/parity copy"
        ),
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
            expected_adapter_model_sha256=args.expected_adapter_model_sha256,
            expected_adapter_config_sha256=args.expected_adapter_config_sha256,
            expected_checkpoint_manifest_sha256=args.expected_checkpoint_manifest_sha256,
            repaired_act_chain_attestation=args.repaired_act_chain_attestation,
        )
        status = gate_analysis.write_report(args.output, report)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{status}: {args.output.resolve()}")


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
