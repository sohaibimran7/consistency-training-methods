"""Raw-generation preflight for the converged two-bias RMCT evaluation.

The frozen 21-task factory and sample-level switch validation are reused from
the Stage 2 OOD substrate.  This module owns the changed science/runtime
boundary: two seen biases, four held-out biases, and the exact offline Qwen
snapshot recorded by r4 rather than the historical model alias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.rmct_two_bias_eval.contract import (
    ALL_BIASES,
    BASE_MODEL,
    EXPECTED_BIASED_TASKS,
    EXPECTED_CLEAN_TASKS,
    EXPECTED_TASKS,
    HELD_OUT_BIASES,
    PROMPT_STYLE,
    SEEN_BIASES,
    VLLM_GENERATION_CONFIG,
    VLLM_MODEL_ARGS,
    VLLM_SAMPLER_ENVIRONMENT,
    VLLM_SAMPLER_RUNTIME,
    bias_status,
    verify_evaluation_receipt,
)

PREFLIGHT_SCHEMA = "rmct-two-bias-eval-raw-preflight-v1"
_HEX = frozenset("0123456789abcdef")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _configuration_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    for name in ("model_dump", "dict"):
        method = getattr(value, name, None)
        if callable(method):
            mapped = method()
            if isinstance(mapped, Mapping):
                return mapped
    return {}


def _assert_snapshot_hf_runtime(
    header_log: Any,
    *,
    path: Path,
    checkpoint: str,
    max_connections: int,
) -> tuple[str, dict[str, Any]]:
    """Require the exact native-HF/PEFT evaluator header for the r4 adapter."""

    from experiments.stage2_ood_hle import raw_preflight as legacy

    evaluation = legacy._attribute(header_log, "eval")
    model = str(legacy._attribute(evaluation, "model", "") or "")
    if model != f"hf/{BASE_MODEL}":
        raise ValueError(f"raw log has wrong snapshot-backed HF model: {path}: {model!r}")
    metadata = _configuration_mapping(legacy._attribute(evaluation, "metadata", {}))
    expected_metadata = {
        "checkpoint": checkpoint,
        "checkpoint_backend": "local",
        "base_model": BASE_MODEL,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"raw log has wrong HF/PEFT metadata.{field}: {path}")
    model_args = _configuration_mapping(legacy._attribute(evaluation, "model_args", {}))
    for field, expected in {"device": "cuda:0", "dtype": "bfloat16"}.items():
        if model_args.get(field) != expected:
            raise ValueError(f"raw log has wrong HF/PEFT model_args.{field}: {path}")
    requested = _configuration_mapping(metadata.get("model_args", {}))
    for field, expected in {"provider": "hf", "device": "cuda:0", "dtype": "bfloat16"}.items():
        if requested.get(field) != expected:
            raise ValueError(f"raw log has wrong HF/PEFT metadata.model_args.{field}: {path}")
    generation = _configuration_mapping(legacy._attribute(evaluation, "model_generate_config", {}))
    expected_generation = {
        "max_tokens": 20480,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_connections": max_connections,
    }
    for field, expected in expected_generation.items():
        if generation.get(field) != expected:
            raise ValueError(f"raw log has wrong HF/PEFT generation {field}: {path}")
    return model, {
        "profile": "hf-peft",
        "provider": "hf",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "max_connections": max_connections,
        "checkpoint": checkpoint,
        "checkpoint_backend": "local",
        "base_model": BASE_MODEL,
    }


def _assert_snapshot_vllm_runtime(
    header_log: Any,
    *,
    path: Path,
    runtime: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Require the parity-attested native-vLLM header for the r4 adapter."""

    from experiments.stage2_ood_hle import raw_preflight as legacy

    checkpoint = runtime["checkpoint"]
    evaluation = legacy._attribute(header_log, "eval")
    model = str(legacy._attribute(evaluation, "model", "") or "")
    if model != f"vllm/{BASE_MODEL}:{checkpoint}":
        raise ValueError(f"raw log has wrong parity-attested vLLM model: {path}: {model!r}")
    metadata = _configuration_mapping(legacy._attribute(evaluation, "metadata", {}))
    for field, expected in {"checkpoint": checkpoint, "checkpoint_backend": "local"}.items():
        if metadata.get(field) != expected:
            raise ValueError(f"raw log has wrong vLLM metadata.{field}: {path}")
    # The runner records the original command arguments in metadata, including
    # the provider it removes before calling Inspect's native vLLM constructor.
    requested = _configuration_mapping(metadata.get("model_args", {}))
    if dict(requested) != VLLM_MODEL_ARGS:
        raise ValueError(f"raw log has wrong vLLM metadata.model_args: {path}")
    requested_generation = _configuration_mapping(metadata.get("generation_config", {}))
    if dict(requested_generation) != VLLM_GENERATION_CONFIG:
        raise ValueError(f"raw log has wrong vLLM metadata.generation_config: {path}")
    model_args = _configuration_mapping(legacy._attribute(evaluation, "model_args", {}))
    expected_model_args = {key: value for key, value in VLLM_MODEL_ARGS.items() if key != "provider"}
    for field, expected in expected_model_args.items():
        if model_args.get(field) != expected:
            raise ValueError(f"raw log has wrong native-vLLM model_args.{field}: {path}")
    generation = _configuration_mapping(legacy._attribute(evaluation, "model_generate_config", {}))
    for field, expected in VLLM_GENERATION_CONFIG.items():
        if generation.get(field) != expected:
            raise ValueError(f"raw log has wrong native-vLLM generation {field}: {path}")
    return model, dict(runtime)


def _receipt_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"evaluation receipt must be a regular file: {path}")
    return path


def _select_logs(raw_root: Path, *, manifest: Path, runtime: Mapping[str, Any]) -> dict[Any, Any]:
    """Reuse strict Stage 2 task/sample guards with r4 snapshot runtime checks."""

    from experiments.stage2_ood_hle import raw_preflight as legacy
    from experiments.stage2_ood_hle.tasks import ood_task_specs

    expected = legacy._expected_cell_specs(ood_task_specs(manifest))
    selected: dict[Any, Any] = {}
    for path in legacy._discover_eval_log_paths(raw_root):
        try:
            header = legacy._read_eval_log(path, header_only=True)
        except Exception as exc:
            raise ValueError(f"could not read raw EvalLog header: {path}") from exc
        if legacy._attribute(header, "status") != "success":
            continue
        evaluation = legacy._attribute(header, "eval")
        task_name = legacy._task_basename(legacy._attribute(evaluation, "task"))
        if task_name not in {legacy.TASK_UNBIASED, legacy.TASK_BIASED}:
            continue
        identity = legacy._parse_candidate_identity(evaluation, task_name=task_name, path=path)
        spec = expected.get(identity)
        if spec is None:
            raise ValueError(f"unexpected successful raw Stage 2 task cell: {identity!r} at {path}")
        created = legacy._validate_header(header, path=path, spec=spec, raw_root=raw_root)
        if runtime.get("profile") == "hf-peft":
            model, observed_runtime = _assert_snapshot_hf_runtime(
                header,
                path=path,
                checkpoint=str(runtime["checkpoint"]),
                max_connections=int(runtime["max_connections"]),
            )
        elif runtime.get("profile") == "vllm":
            model, observed_runtime = _assert_snapshot_vllm_runtime(header, path=path, runtime=runtime)
        else:  # receipt validation normally prevents this; keep direct callers fail-closed.
            raise ValueError("raw-log selection received an unsupported runtime profile")
        current = legacy.LoadedTaskLog(spec, path, created, header, model, observed_runtime)
        previous = selected.get(identity)
        if previous is not None and previous.created == current.created:
            raise ValueError(f"ambiguous successful raw retries for {identity!r}: {previous.path} and {path}")
        if previous is None or current.created > previous.created:
            selected[identity] = current
    if set(selected) != set(expected):
        missing = sorted(set(expected) - set(selected))
        unexpected = sorted(set(selected) - set(expected))
        raise ValueError(f"raw task matrix is incomplete; missing={missing}, unexpected={unexpected}")
    if len({loaded.path for loaded in selected.values()}) != EXPECTED_TASKS:
        raise ValueError("one raw EvalLog was selected for multiple Stage 2 task cells")
    expected_model = (
        f"hf/{BASE_MODEL}"
        if runtime.get("profile") == "hf-peft"
        else f"vllm/{BASE_MODEL}:{runtime['checkpoint']}"
    )
    if {loaded.model for loaded in selected.values()} != {expected_model}:
        raise ValueError("raw task matrix does not share the exact sealed snapshot model identity")
    return selected


def _validate_full_logs(selected: Mapping[Any, Any]) -> dict[Any, int]:
    from experiments.stage2_ood_hle import raw_preflight as legacy

    clean_paths = {
        (loaded.spec.population, loaded.spec.dataset): loaded.path
        for loaded in selected.values()
        if loaded.spec.kind == "unbiased"
    }
    if len(clean_paths) != EXPECTED_CLEAN_TASKS:
        raise ValueError("raw task matrix is missing a shared clean reference")
    counts: dict[Any, int] = {}
    for identity, loaded in selected.items():
        try:
            full_log = legacy._read_eval_log(loaded.path, header_only=False)
        except Exception as exc:
            raise ValueError(f"could not read full raw EvalLog: {loaded.path}") from exc
        counts[identity] = legacy._validate_samples(full_log, loaded=loaded, clean_paths=clean_paths)
    return counts


def preflight_raw_logs(raw_root: str | Path, evaluation_receipt: str | Path) -> dict[str, Any]:
    """Hash-bind a complete raw 21-task matrix to a verified r4 receipt.

    The function is read-only: it neither stages nor grades a response.  It
    rejects any attempted alias-model run, any missing clean task, and any
    biased score whose resolved clean log is not the matching shared-ID task.
    """

    receipt_path = _receipt_path(evaluation_receipt)
    receipt = verify_evaluation_receipt(receipt_path)
    if receipt["runtime"]["profile"] == "vllm":
        for name, expected in VLLM_SAMPLER_ENVIRONMENT.items():
            if os.environ.get(name) != expected:
                raise ValueError(
                    f"raw vLLM preflight requires {name}={expected!r} for the pinned native sampler runtime"
                )
    raw_directory = Path(raw_root).expanduser().resolve()
    if raw_directory.is_symlink() or not raw_directory.is_dir():
        raise FileNotFoundError(f"raw_log_root must be a regular directory: {raw_directory}")
    declared_root = receipt["raw_generation"]["raw_log_root"]
    if declared_root is not None and Path(declared_root).resolve() != raw_directory:
        raise ValueError("raw_log_root differs from the immutable evaluation receipt")

    manifest = Path(receipt["stage2_substrate"]["manifest"]["path"])
    selected = _select_logs(
        raw_directory,
        manifest=manifest,
        runtime=receipt["runtime"],
    )
    counts = _validate_full_logs(selected)
    from experiments.stage2_ood_hle import raw_preflight as legacy

    clean_records: dict[tuple[str, str], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for identity in sorted(selected):
        loaded = selected[identity]
        if loaded.spec.kind != "unbiased":
            continue
        record = legacy._source_record(loaded, sample_count=counts[identity], clean_records={})
        record["evaluation_bias_status"] = None
        clean_records[(loaded.spec.population, loaded.spec.dataset)] = record
        sources.append(record)
    for identity in sorted(selected):
        loaded = selected[identity]
        if loaded.spec.kind != "biased":
            continue
        record = legacy._source_record(loaded, sample_count=counts[identity], clean_records=clean_records)
        record["unbiased_log"] = str(raw_directory)
        record["evaluation_bias_status"] = bias_status(str(loaded.spec.bias_type))
        sources.append(record)

    report = {
        "schema": PREFLIGHT_SCHEMA,
        "evaluation_receipt": {"path": str(receipt_path), "sha256": _sha256_file(receipt_path)},
        "condition": receipt["condition"],
        "raw_root": str(raw_directory),
        "contract": {
            "task_count": EXPECTED_TASKS,
            "clean_task_count": EXPECTED_CLEAN_TASKS,
            "biased_task_count": EXPECTED_BIASED_TASKS,
            "all_biases": list(ALL_BIASES),
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "prompt_style": PROMPT_STYLE,
            "include_bias_acknowledged": False,
            "grader_model": None,
            "runtime": dict(receipt["runtime"]),
        },
        "sources": sources,
    }
    validate_preflight_report(report)
    return report


def _load_report(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"raw preflight report must be a regular file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid raw preflight report: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError("raw preflight report must be a JSON object")
    return document


def _validate_report_runtime(runtime: Any) -> tuple[str, Mapping[str, Any]]:
    """Validate the persisted runtime without silently reducing vLLM evidence."""

    if not isinstance(runtime, Mapping):
        raise ValueError("raw preflight report has no runtime")
    if runtime.get("profile") == "hf-peft":
        if (
            dict(runtime)
            != {
                "profile": "hf-peft",
                "base_model": BASE_MODEL,
                "checkpoint": runtime.get("checkpoint"),
                "max_connections": runtime.get("max_connections"),
                "provider": "hf",
                "device": "cuda:0",
                "dtype": "bfloat16",
            }
            or not isinstance(runtime.get("checkpoint"), str)
            or not Path(runtime["checkpoint"]).is_absolute()
            or isinstance(runtime.get("max_connections"), bool)
            or not isinstance(runtime.get("max_connections"), int)
            or runtime["max_connections"] < 1
        ):
            raise ValueError("raw preflight report has invalid snapshot HF runtime")
        return f"hf/{BASE_MODEL}", runtime
    if runtime.get("profile") != "vllm":
        raise ValueError("raw preflight report has unsupported runtime profile")
    compatibility = runtime.get("compatibility_adapter")
    expected_runtime_keys = {
        "profile",
        "base_model",
        "checkpoint",
        "raw_checkpoint",
        "provider",
        "generation",
        "model_args",
        "sampler",
        "compatibility_adapter",
    }
    if (
        set(runtime) != expected_runtime_keys
        or runtime.get("base_model") != BASE_MODEL
        or runtime.get("provider") != "vllm"
        or not isinstance(runtime.get("checkpoint"), str)
        or not Path(runtime["checkpoint"]).is_absolute()
        or not isinstance(runtime.get("raw_checkpoint"), str)
        or not Path(runtime["raw_checkpoint"]).is_absolute()
        or dict(runtime.get("generation", {})) != VLLM_GENERATION_CONFIG
        or dict(runtime.get("model_args", {})) != VLLM_MODEL_ARGS
        or dict(runtime.get("sampler", {})) != VLLM_SAMPLER_RUNTIME
        or not isinstance(compatibility, Mapping)
    ):
        raise ValueError("raw preflight report has invalid vLLM runtime")
    compatibility_keys = {
        "path",
        "adapter_model_sha256",
        "adapter_config_sha256",
        "compatibility_manifest",
        "parity_attestation",
        "parity_reports",
        "source_raw_checkpoint",
        "source_raw_adapter_model_sha256",
        "base_model",
    }
    manifest = compatibility.get("compatibility_manifest")
    attestation = compatibility.get("parity_attestation")
    reports = compatibility.get("parity_reports")
    if (
        set(compatibility) != compatibility_keys
        or compatibility.get("path") != runtime["checkpoint"]
        or compatibility.get("source_raw_checkpoint") != runtime["raw_checkpoint"]
        or compatibility.get("base_model") != BASE_MODEL
        or any(not _is_sha256(compatibility.get(field)) for field in ("adapter_model_sha256", "adapter_config_sha256", "source_raw_adapter_model_sha256"))
        or not isinstance(manifest, Mapping)
        or set(manifest) != {"path", "sha256"}
        or not isinstance(manifest.get("path"), str)
        or not Path(manifest["path"]).is_absolute()
        or not _is_sha256(manifest.get("sha256"))
        or not isinstance(attestation, Mapping)
        or set(attestation) != {"path", "sha256", "schema"}
        or not isinstance(attestation.get("path"), str)
        or not Path(attestation["path"]).is_absolute()
        or not _is_sha256(attestation.get("sha256"))
        or not isinstance(attestation.get("schema"), str)
        or not isinstance(reports, list)
        or not reports
        or any(
            not isinstance(report, Mapping)
            or set(report) != {"path", "sha256"}
            or not isinstance(report.get("path"), str)
            or not Path(report["path"]).is_absolute()
            or not _is_sha256(report.get("sha256"))
            for report in reports
        )
    ):
        raise ValueError("raw preflight report has incomplete vLLM compatibility/parity evidence")
    return f"vllm/{BASE_MODEL}:{runtime['checkpoint']}", runtime


def validate_preflight_report(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Validate report structure and the explicit two-seen/four-held-out labels."""

    report = _load_report(value)
    if set(report) != {"schema", "evaluation_receipt", "condition", "raw_root", "contract", "sources"}:
        raise ValueError("raw preflight report has unexpected or missing top-level fields")
    if report.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError("unsupported two-bias raw preflight schema")
    if not isinstance(report.get("condition"), str) or Path(report["condition"]).name != report["condition"]:
        raise ValueError("raw preflight report has invalid condition")
    if not isinstance(report.get("raw_root"), str) or not Path(report["raw_root"]).is_absolute():
        raise ValueError("raw preflight report has invalid raw root")
    receipt = report.get("evaluation_receipt")
    if not isinstance(receipt, Mapping) or set(receipt) != {"path", "sha256"} or not isinstance(receipt.get("path"), str) or not Path(receipt["path"]).is_absolute() or not _is_sha256(receipt.get("sha256")):
        raise ValueError("raw preflight report has invalid evaluation receipt identity")
    contract = report.get("contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "task_count",
        "clean_task_count",
        "biased_task_count",
        "all_biases",
        "seen_biases",
        "held_out_biases",
        "prompt_style",
        "include_bias_acknowledged",
        "grader_model",
        "runtime",
    }:
        raise ValueError("raw preflight report has invalid contract")
    if (
        (contract.get("task_count"), contract.get("clean_task_count"), contract.get("biased_task_count"))
        != (EXPECTED_TASKS, EXPECTED_CLEAN_TASKS, EXPECTED_BIASED_TASKS)
        or contract.get("all_biases") != list(ALL_BIASES)
        or contract.get("seen_biases") != list(SEEN_BIASES)
        or contract.get("held_out_biases") != list(HELD_OUT_BIASES)
        or contract.get("prompt_style") != PROMPT_STYLE
        or contract.get("include_bias_acknowledged") is not False
        or contract.get("grader_model") is not None
        or not isinstance(contract.get("runtime"), Mapping)
    ):
        raise ValueError("raw preflight report contract differs from two-bias evaluation")
    expected_model, runtime = _validate_report_runtime(contract["runtime"])

    sources = report.get("sources")
    if not isinstance(sources, list) or len(sources) != EXPECTED_TASKS:
        raise ValueError("raw preflight report must retain all 21 task sources")
    clean = [source for source in sources if isinstance(source, Mapping) and source.get("kind") == "unbiased"]
    biased = [source for source in sources if isinstance(source, Mapping) and source.get("kind") == "biased"]
    if (len(clean), len(biased)) != (EXPECTED_CLEAN_TASKS, EXPECTED_BIASED_TASKS):
        raise ValueError("raw preflight report has wrong clean/biased task counts")
    seen_by_bias: dict[str, int] = {bias: 0 for bias in ALL_BIASES}
    clean_by_population: dict[tuple[str, str], Mapping[str, Any]] = {}
    raw_logs: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("raw preflight source must be an object")
        common = {
            "kind", "regime", "population", "dataset", "bias_type", "raw_log", "raw_log_sha256", "created", "sample_count",
            "question_ids_sha256", "frozen_file", "frozen_file_sha256", "source_identity_digest", "prompt_style", "model", "runtime",
            "evaluation_bias_status",
        }
        expected = set(common)
        if source.get("kind") == "biased":
            expected.update({"unbiased_log", "paired_clean"})
        if set(source) != expected:
            raise ValueError("raw preflight source has incomplete fields")
        raw_log = source.get("raw_log")
        if not isinstance(raw_log, str) or not Path(raw_log).is_absolute() or raw_log in raw_logs:
            raise ValueError("raw preflight source has invalid or duplicate raw log")
        raw_logs.add(raw_log)
        if any(not _is_sha256(source.get(field)) for field in ("raw_log_sha256", "question_ids_sha256", "frozen_file_sha256")):
            raise ValueError("raw preflight source has invalid content identity")
        if source.get("prompt_style") != PROMPT_STYLE or source.get("model") != expected_model or dict(source.get("runtime", {})) != dict(runtime):
            raise ValueError("raw preflight source does not bind the sealed snapshot runtime")
        if isinstance(source.get("sample_count"), bool) or not isinstance(source.get("sample_count"), int) or source["sample_count"] < 1:
            raise ValueError("raw preflight source has invalid sample count")
        population_key = (source.get("population"), source.get("dataset"))
        if source.get("kind") == "unbiased":
            if source.get("bias_type") is not None or source.get("evaluation_bias_status") is not None:
                raise ValueError("raw preflight clean source has a bias label")
            clean_by_population[population_key] = source
            continue
        bias = source.get("bias_type")
        if bias not in ALL_BIASES or source.get("evaluation_bias_status") != bias_status(str(bias)):
            raise ValueError("raw preflight source has a misclassified bias")
        seen_by_bias[str(bias)] += 1
        if source.get("unbiased_log") != report["raw_root"]:
            raise ValueError("raw preflight biased source does not declare shared clean root")
    if set(clean_by_population) != {(source.get("population"), source.get("dataset")) for source in clean}:
        raise ValueError("raw preflight clean references are ambiguous")
    if seen_by_bias != {bias: 3 for bias in ALL_BIASES}:
        raise ValueError("raw preflight report does not preserve three task cells per named bias")
    for source in biased:
        paired = source.get("paired_clean")
        clean_source = clean_by_population.get((source.get("population"), source.get("dataset")))
        if not isinstance(paired, Mapping) or clean_source is None or set(paired) != {
            "raw_log", "raw_log_sha256", "question_ids_sha256", "source_identity_digest"
        }:
            raise ValueError("raw preflight biased source has invalid paired-clean provenance")
        for field in paired:
            if paired[field] != clean_source.get(field):
                raise ValueError("raw preflight biased source is paired to the wrong clean task")
    return report


def write_preflight_report(path: str | Path, report: Mapping[str, Any]) -> str:
    document = validate_preflight_report(report)
    destination = Path(path).expanduser().resolve()
    payload = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing raw preflight report: {destination}")
        return "resumed"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
                raise FileExistsError(f"raw preflight report appeared and differs: {destination}") from None
            return "resumed"
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-log-root", required=True, type=Path)
    parser.add_argument("--evaluation-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = preflight_raw_logs(args.raw_log_root, args.evaluation_receipt)
        status = write_preflight_report(args.output, report)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{status}: {args.output.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = ["PREFLIGHT_SCHEMA", "preflight_raw_logs", "validate_preflight_report", "write_preflight_report"]
