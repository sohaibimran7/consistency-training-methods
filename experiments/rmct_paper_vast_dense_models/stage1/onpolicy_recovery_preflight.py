"""Fail-closed no-CoT input attestation for Qwen3.5 on-policy runs.

Legacy RMCT and OPCT targets consume the recovered wrong-argument pair store
directly. RMCT-256 consumes a separately attested ordered prefix of its
canonical reconstruction. In both cases, a ``prompt_style: none`` field in an
arbitrary JSONL is not enough evidence: the historical Stage-1 rows were
genuinely encourage-CoT prompts, and merely relabelling them would recreate
the original scientific failure.

This module deliberately delegates content validation to
``verify_recovered_none_pairs`` in :mod:`supervised_recovery_none_prepare`.
That verifier checks the complete legacy-G4 conversion, exact source/output
hashes, row ordering/counts, question IDs, and removal of the terminal CoT
instruction. This module adds the immutable *consumer-independent* recovered
source attestation, then binds an RMCT-256 selection proof into the
target-specific sidecar before worker/model initialization.

Run it before any model/backend initialization.  The Vast launcher writes one
such sidecar beside each target's logs, but the attestation makes no claim
about a particular target and can also be used by future on-policy methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ctm.artifacts import plain_file_identity, write_atomic_bytes
from experiments.rmct_paper_vast_dense_models.stage1.supervised_recovery_none_prepare import (
    MODEL,
    RECOVERED_NONE_ROWS,
    RECOVERED_NONE_SOURCE_SHA256,
    verify_recovered_none_pairs,
)


ATTESTATION_SCHEMA = "qwen35-no-cot-recovered-source-attestation-v1"
TARGET_CONTRACT_SCHEMA = "qwen35-onpolicy-target-contract-v1"
TARGET_ATTESTATION_SCHEMA = "qwen35-onpolicy-target-attestation-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# RMCT-256 is not another mutable ``prompt_style: none`` input.  Its selected
# 256 rows are an immutable ordered prefix whose provenance reaches through
# the recovered no-CoT source, the canonical n=2048 reconstruction, the
# historical original64 reference, and the Stage-2 in-domain exclusion.  Keep
# the input names here so launcher, target sidecar, and training child use the
# same closed proof bundle.
_RMCT256_SELECTION_INPUTS = (
    "selection",
    "selection_manifest",
    "canonical_source",
    "canonical_source_manifest",
    "original64_reference_manifest",
    "stage2_manifest",
)


def _serialized_attestation(*, source: Path, source_manifest: Path) -> tuple[bytes, dict[str, Any]]:
    """Run the complete recovered-source proof and construct its identity record.

    ``verify_recovered_none_pairs`` is intentionally the first operation.  Do
    not replace it with a check of prompt-style metadata or of a source-side
    manifest alone: both are weaker than the content proof used for the
    no-CoT SFT recovery.
    """

    if source.resolve() == source_manifest.resolve():
        raise ValueError("recovered source and recovery manifest must be distinct files")
    rows = verify_recovered_none_pairs(source, source_manifest)
    if len(rows) != RECOVERED_NONE_ROWS:
        # The imported verifier already guarantees this.  Retain an explicit
        # invariant here so future changes cannot quietly weaken the emitted
        # attestation contract.
        raise AssertionError(f"exact recovered-source verifier returned {len(rows)} rows, expected {RECOVERED_NONE_ROWS}")
    document = {
        "schema": ATTESTATION_SCHEMA,
        "model": MODEL,
        "verification": {
            "kind": "exact_legacy_g4_cot_to_none_content_proof",
            "verifier": "supervised_recovery_none_prepare.verify_recovered_none_pairs",
            "expected_recovered_source_sha256": RECOVERED_NONE_SOURCE_SHA256,
            "expected_row_count": RECOVERED_NONE_ROWS,
        },
        "source": plain_file_identity(source),
        "source_manifest": plain_file_identity(source_manifest),
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"), document


def attest_recovered_none_source(
    *,
    source: str | Path,
    source_manifest: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Write or resume one immutable exact-proof source attestation.

    The output cannot collide with either input.  Re-running with the exact
    same inputs is safe and returns ``status='resumed'``; a changed source or
    manifest generates different bytes and is refused rather than overwritten.
    """

    source_path = Path(source).resolve()
    manifest_path = Path(source_manifest).resolve()
    output_path = Path(output).resolve()
    if output_path in {source_path, manifest_path}:
        raise ValueError("recovered-source attestation output must be distinct from its source and manifest")
    payload, document = _serialized_attestation(source=source_path, source_manifest=manifest_path)
    if output_path.exists():
        if not output_path.is_file() or output_path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different recovered-source attestation: {output_path}")
        return {"status": "resumed", "path": str(output_path), "attestation": document}
    write_atomic_bytes(output_path, payload)
    return {"status": "written", "path": str(output_path), "attestation": document}


def verify_recovered_none_source(
    *,
    source: str | Path,
    source_manifest: str | Path,
) -> dict[str, Any]:
    """Validate the exact proof without writing an output sidecar."""

    source_path = Path(source).resolve()
    manifest_path = Path(source_manifest).resolve()
    _, document = _serialized_attestation(source=source_path, source_manifest=manifest_path)
    return document


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read one sidecar as an object, with an actionable fail-closed error."""

    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return document


def _identity_equal(path: Path, expected: Any, *, label: str) -> dict[str, Any]:
    """Recompute and compare an immutable plain-file identity."""

    if not isinstance(expected, dict):
        raise ValueError(f"{label} has no valid recorded identity")
    observed = plain_file_identity(path)
    if observed != expected:
        raise ValueError(
            f"{label} changed after attestation: expected {expected!r}, got {observed!r}"
        )
    return observed


def validate_recovered_none_source_attestation(
    *,
    attestation: str | Path,
    source: str | Path,
    source_manifest: str | Path,
) -> dict[str, Any]:
    """Validate the immutable source-proof sidecar against its two inputs.

    The expensive, content-level legacy conversion proof is performed when this
    sidecar is created.  Consumers rebind its recorded input identities here;
    a source or manifest mutation after that proof is therefore a hard error,
    not stale provenance.
    """

    attestation_path = Path(attestation).resolve()
    source_path = Path(source).resolve()
    manifest_path = Path(source_manifest).resolve()
    document = _read_json_object(attestation_path, label="recovered no-CoT source attestation")
    _assert_equal(document.get("schema"), ATTESTATION_SCHEMA, label="recovered source attestation schema")
    _assert_equal(document.get("model"), MODEL, label="recovered source attestation model")
    verification = document.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("recovered source attestation has no verification object")
    _assert_equal(
        verification.get("kind"),
        "exact_legacy_g4_cot_to_none_content_proof",
        label="recovered source verification kind",
    )
    _assert_equal(
        verification.get("verifier"),
        "supervised_recovery_none_prepare.verify_recovered_none_pairs",
        label="recovered source verifier",
    )
    _assert_equal(
        verification.get("expected_recovered_source_sha256"),
        RECOVERED_NONE_SOURCE_SHA256,
        label="recovered source expected hash",
    )
    _assert_equal(
        verification.get("expected_row_count"),
        RECOVERED_NONE_ROWS,
        label="recovered source expected row count",
    )
    _identity_equal(source_path, document.get("source"), label="recovered no-CoT source")
    _identity_equal(manifest_path, document.get("source_manifest"), label="recovered no-CoT source manifest")
    return document


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _assert_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _rmct256_selection_paths(
    *,
    selection: str | Path | None,
    selection_manifest: str | Path | None,
    canonical_source: str | Path | None,
    canonical_source_manifest: str | Path | None,
    original64_reference_manifest: str | Path | None,
    stage2_manifest: str | Path | None,
    required: bool,
) -> dict[str, Path] | None:
    """Resolve the all-or-nothing RMCT-256 source-proof bundle.

    A target that declares a selection manifest must receive the complete
    source-proof bundle.  Conversely, allowing a legacy target to carry an
    unrelated partial bundle would make the target contract ambiguous.  Keep
    this check separate from the selection verifier so callers get a clear
    error before any path is dereferenced.
    """

    values: dict[str, str | Path | None] = {
        "selection": selection,
        "selection_manifest": selection_manifest,
        "canonical_source": canonical_source,
        "canonical_source_manifest": canonical_source_manifest,
        "original64_reference_manifest": original64_reference_manifest,
        "stage2_manifest": stage2_manifest,
    }
    supplied = [name for name, value in values.items() if value is not None]
    if required and len(supplied) != len(_RMCT256_SELECTION_INPUTS):
        missing = [name for name in _RMCT256_SELECTION_INPUTS if values[name] is None]
        raise ValueError(
            "RMCT-256 target requires the complete selection proof bundle; "
            f"missing {', '.join(missing)}"
        )
    if not required and supplied:
        raise ValueError("only an RMCT-256 target declaring selection_manifest may receive a selection proof bundle")
    if not required:
        return None
    return {name: Path(values[name]).resolve() for name in _RMCT256_SELECTION_INPUTS}  # type: ignore[arg-type]


def _verify_rmct256_selection_bundle(paths: dict[str, Path], *, source: Path, source_manifest: Path) -> dict[str, Any]:
    """Perform the complete RMCT-256 content proof and bind every input.

    The selection verifier intentionally repeats the source-chain checks on
    every protected target attestation and child startup.  A filename or a
    manifest hash alone would not prove that the selected JSONL remains the
    exact canonical prefix with the required no-overlap constraint.
    """

    from experiments.rmct_256.selection import verify_selection_manifest

    document = verify_selection_manifest(
        paths["selection_manifest"],
        selection_path=paths["selection"],
        canonical_source=paths["canonical_source"],
        canonical_source_manifest=paths["canonical_source_manifest"],
        recovered_none_source=source,
        recovered_none_manifest=source_manifest,
        original64_reference_manifest=paths["original64_reference_manifest"],
        stage2_manifest=paths["stage2_manifest"],
        verify_sources=True,
    )
    return {
        **{name: plain_file_identity(paths[name]) for name in _RMCT256_SELECTION_INPUTS},
        "manifest_document_sha256": _canonical_json_sha256(document),
    }


def _rmct256_selection_paths_from_attestation(record: Any) -> dict[str, Path] | None:
    """Recover and validate the optional selection bundle from a sidecar."""

    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError("RMCT-256 selection proof in target attestation must be an object")
    unknown = sorted(set(record) - {*_RMCT256_SELECTION_INPUTS, "manifest_document_sha256"})
    missing = sorted({*_RMCT256_SELECTION_INPUTS, "manifest_document_sha256"} - set(record))
    if unknown or missing:
        details = [
            *(f"missing {missing}" for _ in [None] if missing),
            *(f"unknown {unknown}" for _ in [None] if unknown),
        ]
        raise ValueError(f"RMCT-256 selection proof in target attestation: {', '.join(details)}")
    digest = record.get("manifest_document_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("RMCT-256 selection proof has no valid manifest document hash")
    return {name: _attested_path(record[name], label=f"attested RMCT-256 {name}") for name in _RMCT256_SELECTION_INPUTS}


def _canonical_json_sha256(value: Any) -> str:
    """Hash one JSON-only value in a stable, unambiguous representation."""

    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("on-policy target contract contains non-canonical JSON") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _binary_file_identity(path: str | Path) -> dict[str, Any]:
    """Hash an executable without loading its whole binary into memory."""

    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"attested executable is missing: {target}")
    digest = hashlib.sha256()
    size = 0
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"path": str(target), "content_sha256": digest.hexdigest(), "size_bytes": size}


def _binary_identity_equal(path: str | Path, expected: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(expected, dict):
        raise ValueError(f"{label} has no valid recorded identity")
    observed = _binary_file_identity(path)
    if observed != expected:
        raise ValueError(f"{label} changed after attestation: expected {expected!r}, got {observed!r}")
    return observed


def _normalized_cuda_visible_devices(value: str | None) -> list[str]:
    """Canonicalize the complete inherited allocation, including coordinator."""

    raw = (value or "").strip()
    if not raw or raw in {"-1", "NoDevFiles"}:
        raise ValueError("on-policy target attestation requires an explicit CUDA_VISIBLE_DEVICES allocation")
    tokens = [token.strip() for token in raw.split(",")]
    if any(not token for token in tokens) or len(tokens) != len(set(tokens)):
        raise ValueError(f"invalid CUDA_VISIBLE_DEVICES allocation for on-policy target attestation: {value!r}")
    return tokens


def _training_script_path(script: str) -> Path:
    if script not in {"scripts/train_opct.py", "scripts/train_rlct.py"}:
        raise ValueError(f"unsupported on-policy training script identity: {script!r}")
    return (PROJECT_ROOT / script).resolve()


def _compiled_target_entry_and_argv(
    *,
    plan: Path,
    target: str,
    topology_profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Compile exactly one selected target and render its pre-injection argv.

    This intentionally goes through the same public compiler and argv renderer
    as ``scripts/run_experiment.py``.  A source-YAML hash alone would not catch
    a changed experiment factory, while an argv hash alone would not show which
    complete compiled entry produced it.  The target sidecar records both.
    """

    from scripts import run_experiment

    compiled = run_experiment.load_experiment(plan, topology_profile=topology_profile)
    entries = [entry for entry in compiled.get("training", []) if entry.get("target") == target]
    if len(entries) != 1:
        raise ValueError(f"plan must resolve exactly one training command for target {target!r}, got {len(entries)}")
    entry = dict(entries[0])
    context = run_experiment.initial_context(compiled)
    planned = run_experiment.planned_commands(
        compiled,
        ["training"],
        context,
        strict=True,
        target=target,
    )
    if len(planned) != 1:
        raise ValueError(f"plan must render exactly one training argv for target {target!r}, got {len(planned)}")
    stage, _name, argv = planned[0]
    _assert_equal(stage, "training", label="on-policy target stage")
    if len(argv) < 2:
        raise ValueError(f"plan target {target!r} rendered no training script")
    return compiled, entry, list(argv)


def _worker_engine_kwargs(worker: dict[str, Any]) -> dict[str, Any]:
    """Mirror the generic worker-preflight engine contract exactly."""

    return {
        "gpu_memory_utilization": float(worker["gpu_memory_utilization"]),
        "max_model_len": int(worker["max_model_len"]),
        "max_num_seqs": int(worker["max_num_seqs"]),
        "max_num_batched_tokens": int(worker["max_num_batched_tokens"]),
        "language_model_only": True,
        "logprobs_mode": "processed_logprobs",
        "tensor_parallel_size": 1,
        "seed": int(worker["seed_base"]),
        **(
            {"gdn_prefill_backend": str(worker["gdn_prefill_backend"])}
            if worker.get("gdn_prefill_backend") is not None
            else {}
        ),
    }


def _validate_worker_parity_attestation(
    *,
    worker_parity_attestation: str | Path,
    worker: dict[str, Any],
    cuda_visible_devices: str | None,
) -> dict[str, Any]:
    """Rebuild the Qwen worker proof against the live allocated GPU mapping.

    This is deliberately an input-only check: ``resolve_rollout_gpus`` uses
    the inherited allocation string and never queries or initializes CUDA.
    ``validate_qwen35_rollout_worker_parity_attestation`` in turn rebinds the
    fixed nonzero adapter evidence to its raw/translated files.
    """

    from ctm.backends.local.qwen35_vllm_compat import validate_qwen35_rollout_worker_parity_attestation
    from ctm.backends.local.rollout_workers import resolve_rollout_gpus

    workers = resolve_rollout_gpus(
        str(worker["rollout_gpus"]),
        cuda_visible_devices=cuda_visible_devices,
        coordinator_device=str(worker["coordinator_device"]),
    )
    return validate_qwen35_rollout_worker_parity_attestation(
        worker_parity_attestation,
        expected_model=MODEL,
        expected_worker_gpus=workers,
        expected_worker_engine_kwargs=_worker_engine_kwargs(worker),
    )


def verify_onpolicy_target_contract(
    *,
    plan: str | Path,
    target: str,
    source: str | Path,
    source_manifest: str | Path,
    experiment_name: str,
    run_name: str,
    worker_gpus: str,
    worker_gpu_mem_util: float,
    worker_max_model_len: int,
    worker_max_num_seqs: int,
    worker_max_num_batched_tokens: int,
    target_logprob_chunk_size: int,
    worker_gdn_prefill_backend: str | None = None,
    worker_seed_base: int = 42,
    topology_profile: str | None = None,
    rmct256_selection: str | Path | None = None,
    rmct256_selection_manifest: str | Path | None = None,
    rmct256_canonical_source: str | Path | None = None,
    rmct256_canonical_source_manifest: str | Path | None = None,
    rmct256_original64_reference_manifest: str | Path | None = None,
    rmct256_stage2_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Bind a named on-policy target to its recovered source and optional selection proof.

    This no-model check prevents a hand-edited YAML from making the launcher
    attest one source/options set while its target command consumes another.
    When the compiled target declares an RMCT-256 selection manifest, the
    complete canonical-prefix/original64/Stage-2 proof is mandatory here.
    """

    plan_path = Path(plan).resolve()
    source_path = Path(source).resolve()
    manifest_path = Path(source_manifest).resolve()
    compiled, entry, argv = _compiled_target_entry_and_argv(
        plan=plan_path,
        target=target,
        topology_profile=topology_profile,
    )
    _assert_equal(compiled.get("name"), experiment_name, label="on-policy experiment namespace")
    if topology_profile is not None:
        _assert_equal(
            compiled.get("onpolicy_topology_profile"),
            topology_profile,
            label="on-policy topology profile",
        )
        try:
            worker_indices = [int(token) for token in worker_gpus.split(",")]
        except ValueError as exc:
            raise ValueError("on-policy topology worker_gpus must contain comma-separated logical integers") from exc
        if not worker_indices or worker_gpus != ",".join(str(index) for index in worker_indices):
            raise ValueError("on-policy topology worker_gpus must use canonical comma-separated logical integers")
        expected_topology = {
            "gpu_count": len(worker_indices) + 1,
            "coordinator_device": "cuda:0",
            "rollout_gpus": worker_indices,
        }
        _assert_equal(compiled.get("onpolicy_topology"), expected_topology, label="on-policy selected topology")
        _assert_equal(entry.get("gpu_count"), expected_topology["gpu_count"], label=f"{target} selected gpu_count")
    args = entry.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"plan target {target!r} has no argument object")

    expected_script = "scripts/train_opct.py" if target == "opct" else "scripts/train_rlct.py"
    _assert_equal(entry.get("command"), ["${python}", expected_script], label=f"{target} training command")
    _assert_equal(args.get("model"), MODEL, label=f"{target} model")
    _assert_equal(args.get("experiment_name"), "${experiment}", label=f"{target} experiment placeholder")
    _assert_equal(args.get("run_name"), run_name, label=f"{target} run name")
    _assert_equal(
        args.get("require_onpolicy_target_attestation"),
        True,
        label=f"{target} protected child requirement",
    )
    if worker_gdn_prefill_backend not in {None, "flashinfer", "triton"}:
        raise ValueError("worker_gdn_prefill_backend must be None, 'flashinfer', or 'triton'")
    if (
        isinstance(worker_seed_base, bool)
        or not isinstance(worker_seed_base, int)
        or not 0 <= worker_seed_base <= 2**31 - 1
    ):
        raise ValueError("worker_seed_base must be an integer in [0, 2147483647]")
    local_contract = {
        "backend": "local",
        "local_sampler": "vllm",
        "local_device": "cuda:0",
        "local_rollout_gpus": worker_gpus,
        "local_rollout_gpu_mem_util": worker_gpu_mem_util,
        "local_rollout_seed_base": worker_seed_base,
        "local_vllm_max_model_len": worker_max_model_len,
        "local_vllm_max_num_seqs": worker_max_num_seqs,
        "local_vllm_max_num_batched_tokens": worker_max_num_batched_tokens,
        "local_vllm_gdn_prefill_backend": worker_gdn_prefill_backend,
        "local_target_logprob_chunk_size": target_logprob_chunk_size,
    }
    for key, expected in local_contract.items():
        _assert_equal(args.get(key), expected, label=f"{target} {key}")

    selection_paths: dict[str, Path] | None = None
    if target == "opct":
        data = args.get("data")
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], str):
            raise ValueError("opct target must have exactly one frozen pair data specification")
        data_path_text, separator, limit = data[0].rpartition(":")
        if not separator or not limit.isdigit() or int(limit) < 1:
            raise ValueError("opct frozen pair data must use PATH:N syntax")
        _assert_equal(_project_path(data_path_text), source_path, label="opct frozen pair source")
        manifests = args.get("data_manifest")
        if not isinstance(manifests, list) or len(manifests) != 1:
            raise ValueError("opct target must declare exactly one frozen pair manifest")
        _assert_equal(_project_path(str(manifests[0])), manifest_path, label="opct frozen pair manifest")
        selection_paths = _rmct256_selection_paths(
            selection=rmct256_selection,
            selection_manifest=rmct256_selection_manifest,
            canonical_source=rmct256_canonical_source,
            canonical_source_manifest=rmct256_canonical_source_manifest,
            original64_reference_manifest=rmct256_original64_reference_manifest,
            stage2_manifest=rmct256_stage2_manifest,
            required=False,
        )
    else:
        setting_config = args.get("setting_config")
        if not isinstance(setting_config, dict):
            raise ValueError(f"{target} target must have a setting_config object")
        data_paths = setting_config.get("data_paths")
        if not isinstance(data_paths, list) or len(data_paths) != 1:
            raise ValueError(f"{target} target must have exactly one frozen pair source")
        load_config = args.get("load_config")
        if not isinstance(load_config, dict):
            raise ValueError(f"{target} target must have a load_config object")
        command_selection_manifest = load_config.get("selection_manifest")
        selection_paths = _rmct256_selection_paths(
            selection=rmct256_selection,
            selection_manifest=rmct256_selection_manifest,
            canonical_source=rmct256_canonical_source,
            canonical_source_manifest=rmct256_canonical_source_manifest,
            original64_reference_manifest=rmct256_original64_reference_manifest,
            stage2_manifest=rmct256_stage2_manifest,
            required=command_selection_manifest is not None,
        )
        training_source_path = selection_paths["selection"] if selection_paths is not None else source_path
        _assert_equal(_project_path(str(data_paths[0])), training_source_path, label=f"{target} frozen pair source")
        if selection_paths is not None:
            if not isinstance(command_selection_manifest, str) or not command_selection_manifest:
                raise ValueError(f"{target} selection_manifest must be a non-empty path")
            _assert_equal(
                _project_path(command_selection_manifest),
                selection_paths["selection_manifest"],
                label=f"{target} frozen selection manifest",
            )

    rmct256_selection_proof = (
        _verify_rmct256_selection_bundle(selection_paths, source=source_path, source_manifest=manifest_path)
        if selection_paths is not None
        else None
    )

    return {
        "schema": TARGET_CONTRACT_SCHEMA,
        "plan": plain_file_identity(plan_path),
        "target": target,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "topology_profile": topology_profile,
        "training_script": expected_script,
        "compiled_entry": entry,
        "compiled_entry_sha256": _canonical_json_sha256(entry),
        # ``sys.argv`` in the training child starts at the script, so retain
        # that exact hand-off shape here.  The interpreter is sealed as its
        # own argv[0] plus binary identity below rather than being confused
        # with the child-visible argument vector.
        "interpreter_argv0": argv[0],
        "child_argv": argv[1:],
        "child_argv_sha256": _canonical_json_sha256(argv[1:]),
        "source_path": str(source_path),
        "source_manifest_path": str(manifest_path),
        **({"rmct256_selection": rmct256_selection_proof} if rmct256_selection_proof is not None else {}),
        "worker": {
            "coordinator_device": "cuda:0",
            "rollout_gpus": worker_gpus,
            "gpu_memory_utilization": worker_gpu_mem_util,
            "max_model_len": worker_max_model_len,
            "max_num_seqs": worker_max_num_seqs,
            "max_num_batched_tokens": worker_max_num_batched_tokens,
            "target_logprob_chunk_size": target_logprob_chunk_size,
            "seed_base": worker_seed_base,
            **(
                {"gdn_prefill_backend": worker_gdn_prefill_backend}
                if worker_gdn_prefill_backend is not None
                else {}
            ),
        },
    }


def _target_attestation_document(
    *,
    plan: Path,
    target: str,
    source: Path,
    source_manifest: Path,
    source_attestation: Path,
    worker_parity_attestation: Path,
    output: Path,
    experiment_name: str,
    run_name: str,
    worker_gpus: str,
    worker_gpu_mem_util: float,
    worker_max_model_len: int,
    worker_max_num_seqs: int,
    worker_max_num_batched_tokens: int,
    target_logprob_chunk_size: int,
    cuda_visible_devices: str | None,
    worker_gdn_prefill_backend: str | None = None,
    worker_seed_base: int = 42,
    topology_profile: str | None = None,
    rmct256_selection: Path | None = None,
    rmct256_selection_manifest: Path | None = None,
    rmct256_canonical_source: Path | None = None,
    rmct256_canonical_source_manifest: Path | None = None,
    rmct256_original64_reference_manifest: Path | None = None,
    rmct256_stage2_manifest: Path | None = None,
) -> dict[str, Any]:
    """Build a complete immutable hand-off contract for one training child."""

    source_document = validate_recovered_none_source_attestation(
        attestation=source_attestation,
        source=source,
        source_manifest=source_manifest,
    )
    # The source sidecar is useful provenance, but is not a substitute for the
    # actual content verifier.  Re-run it here so the target sidecar is only
    # ever minted for the exact canonical recovery.
    _assert_equal(
        verify_recovered_none_source(source=source, source_manifest=source_manifest),
        source_document,
        label="target source proof",
    )
    contract = verify_onpolicy_target_contract(
        plan=plan,
        target=target,
        source=source,
        source_manifest=source_manifest,
        experiment_name=experiment_name,
        run_name=run_name,
        worker_gpus=worker_gpus,
        worker_gpu_mem_util=worker_gpu_mem_util,
        worker_max_model_len=worker_max_model_len,
        worker_max_num_seqs=worker_max_num_seqs,
        worker_max_num_batched_tokens=worker_max_num_batched_tokens,
        target_logprob_chunk_size=target_logprob_chunk_size,
        worker_gdn_prefill_backend=worker_gdn_prefill_backend,
        worker_seed_base=worker_seed_base,
        topology_profile=topology_profile,
        rmct256_selection=rmct256_selection,
        rmct256_selection_manifest=rmct256_selection_manifest,
        rmct256_canonical_source=rmct256_canonical_source,
        rmct256_canonical_source_manifest=rmct256_canonical_source_manifest,
        rmct256_original64_reference_manifest=rmct256_original64_reference_manifest,
        rmct256_stage2_manifest=rmct256_stage2_manifest,
    )
    worker = contract["worker"]
    full_cuda_visible_devices = _normalized_cuda_visible_devices(cuda_visible_devices)
    expected_worker_sidecar = (
        PROJECT_ROOT
        / "logs"
        / experiment_name
        / run_name
        / "rollout_workers"
        / "qwen35-rollout-worker-parity-attestation.json"
    ).resolve()
    _assert_equal(
        worker_parity_attestation,
        expected_worker_sidecar,
        label="on-policy worker parity sidecar path",
    )
    _validate_worker_parity_attestation(
        worker_parity_attestation=worker_parity_attestation,
        worker=worker,
        cuda_visible_devices=cuda_visible_devices,
    )

    # The runner adds this terminal argument after it has re-read the YAML.
    # Record the final child-visible argv in the same shape Python exposes in
    # sys.argv. The interpreter's argv[0] and binary identity are recorded
    # separately below, so both halves of the real command are sealed.
    final_child_argv = [
        *contract["child_argv"],
        "--onpolicy-target-attestation",
        str(output),
    ]
    if not final_child_argv:
        raise ValueError("on-policy target command has no training script")
    _assert_equal(final_child_argv[0], contract["training_script"], label="on-policy target training script argv")
    return {
        "schema": TARGET_ATTESTATION_SCHEMA,
        "authored_plan": contract["plan"],
        "target": target,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "topology_profile": topology_profile,
        "training_script": contract["training_script"],
        "compiled_entry": contract["compiled_entry"],
        "compiled_entry_sha256": contract["compiled_entry_sha256"],
        "child_argv": final_child_argv,
        "child_argv_sha256": _canonical_json_sha256(final_child_argv),
        # Path equality alone does not prove that a host's virtualenv or the
        # training script was unchanged between launcher, runner, and child.
        # Record both the compiler's interpreter argv[0] spelling and the
        # resolved executable bytes, while preserving child_argv as Python
        # sees it (script + arguments).
        "interpreter_argv0": contract["interpreter_argv0"],
        "interpreter": _binary_file_identity(contract["interpreter_argv0"]),
        "training_script_identity": plain_file_identity(_training_script_path(contract["training_script"])),
        "cuda_visible_devices": full_cuda_visible_devices,
        "source": plain_file_identity(source),
        "source_manifest": plain_file_identity(source_manifest),
        "source_attestation": plain_file_identity(source_attestation),
        **(
            {"rmct256_selection": contract["rmct256_selection"]}
            if "rmct256_selection" in contract
            else {}
        ),
        "worker": worker,
        "worker_parity_attestation": plain_file_identity(worker_parity_attestation),
    }


def attest_onpolicy_target(
    *,
    plan: str | Path,
    target: str,
    source: str | Path,
    source_manifest: str | Path,
    source_attestation: str | Path,
    worker_parity_attestation: str | Path,
    output: str | Path,
    experiment_name: str,
    run_name: str,
    worker_gpus: str,
    worker_gpu_mem_util: float,
    worker_max_model_len: int,
    worker_max_num_seqs: int,
    worker_max_num_batched_tokens: int,
    target_logprob_chunk_size: int,
    cuda_visible_devices: str | None,
    worker_gdn_prefill_backend: str | None = None,
    worker_seed_base: int = 42,
    topology_profile: str | None = None,
    rmct256_selection: str | Path | None = None,
    rmct256_selection_manifest: str | Path | None = None,
    rmct256_canonical_source: str | Path | None = None,
    rmct256_canonical_source_manifest: str | Path | None = None,
    rmct256_original64_reference_manifest: str | Path | None = None,
    rmct256_stage2_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Write or resume the immutable target-to-child hand-off sidecar.

    It binds the authored plan bytes, the entire compiled selected entry, the
    exact final child argv, recovered source proof, optional full RMCT-256
    selection proof, and topology-bound worker proof. A rerun may only reuse
    byte-identical evidence; it cannot silently replace a sidecar after a
    plan/source/selection/worker change.
    """

    plan_path = Path(plan).resolve()
    source_path = Path(source).resolve()
    manifest_path = Path(source_manifest).resolve()
    source_attestation_path = Path(source_attestation).resolve()
    worker_attestation_path = Path(worker_parity_attestation).resolve()
    output_path = Path(output).resolve()
    selection_paths = _rmct256_selection_paths(
        selection=rmct256_selection,
        selection_manifest=rmct256_selection_manifest,
        canonical_source=rmct256_canonical_source,
        canonical_source_manifest=rmct256_canonical_source_manifest,
        original64_reference_manifest=rmct256_original64_reference_manifest,
        stage2_manifest=rmct256_stage2_manifest,
        # The compiler determines whether this bundle is required; this early
        # pass only rejects a malformed partial optional bundle before output
        # collision checks.  ``verify_onpolicy_target_contract`` below applies
        # the target-specific requirement after expanding the plan.
        required=any(
            value is not None
            for value in (
                rmct256_selection,
                rmct256_selection_manifest,
                rmct256_canonical_source,
                rmct256_canonical_source_manifest,
                rmct256_original64_reference_manifest,
                rmct256_stage2_manifest,
            )
        ),
    )
    protected = {
        plan_path,
        source_path,
        manifest_path,
        source_attestation_path,
        worker_attestation_path,
        *(selection_paths.values() if selection_paths is not None else ()),
    }
    if output_path in protected:
        raise ValueError("on-policy target attestation output must be distinct from every attested input")
    document = _target_attestation_document(
        plan=plan_path,
        target=target,
        source=source_path,
        source_manifest=manifest_path,
        source_attestation=source_attestation_path,
        worker_parity_attestation=worker_attestation_path,
        output=output_path,
        experiment_name=experiment_name,
        run_name=run_name,
        worker_gpus=worker_gpus,
        worker_gpu_mem_util=worker_gpu_mem_util,
        worker_max_model_len=worker_max_model_len,
        worker_max_num_seqs=worker_max_num_seqs,
        worker_max_num_batched_tokens=worker_max_num_batched_tokens,
        target_logprob_chunk_size=target_logprob_chunk_size,
        cuda_visible_devices=cuda_visible_devices,
        worker_gdn_prefill_backend=worker_gdn_prefill_backend,
        worker_seed_base=worker_seed_base,
        topology_profile=topology_profile,
        rmct256_selection=selection_paths["selection"] if selection_paths is not None else None,
        rmct256_selection_manifest=selection_paths["selection_manifest"] if selection_paths is not None else None,
        rmct256_canonical_source=selection_paths["canonical_source"] if selection_paths is not None else None,
        rmct256_canonical_source_manifest=(
            selection_paths["canonical_source_manifest"] if selection_paths is not None else None
        ),
        rmct256_original64_reference_manifest=(
            selection_paths["original64_reference_manifest"] if selection_paths is not None else None
        ),
        rmct256_stage2_manifest=selection_paths["stage2_manifest"] if selection_paths is not None else None,
    )
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # Use exclusive creation rather than a check-then-replace: two concurrent
    # launch attempts must race to one immutable result, never let a late
    # writer replace an already-published contract with its own view.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as handle:
            handle.write(payload)
        status = "written"
    except FileExistsError:
        if not output_path.is_file() or output_path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different on-policy target attestation: {output_path}")
        status = "resumed"
    return {
        "status": status,
        "path": str(output_path),
        "attestation": document,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_target_attestation(attestation: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(attestation).resolve()
    document = _read_json_object(path, label="on-policy target attestation")
    _assert_equal(document.get("schema"), TARGET_ATTESTATION_SCHEMA, label="on-policy target attestation schema")
    return path, document


def _expected_final_child_argv(
    *,
    plan: Path,
    target: str,
    attestation: Path,
    topology_profile: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str, list[str]]:
    compiled, entry, argv = _compiled_target_entry_and_argv(
        plan=plan,
        target=target,
        topology_profile=topology_profile,
    )
    return compiled, entry, argv[0], [*argv[1:], "--onpolicy-target-attestation", str(attestation)]


def _validate_target_sidecar_shape(
    *,
    document: dict[str, Any],
    plan: Path,
    target: str,
    attestation: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Recompile the target and validate plan/entry/final-argv bindings."""

    _assert_equal(document.get("target"), target, label="on-policy target")
    _identity_equal(plan, document.get("authored_plan"), label="authored on-policy YAML")
    topology_profile = document.get("topology_profile")
    if topology_profile is not None and not isinstance(topology_profile, str):
        raise ValueError("on-policy target topology profile must be a string or null")
    compiled, entry, expected_interpreter, expected_child_argv = _expected_final_child_argv(
        plan=plan,
        target=target,
        attestation=attestation,
        topology_profile=topology_profile,
    )
    _assert_equal(document.get("compiled_entry"), entry, label="compiled selected on-policy entry")
    _assert_equal(
        document.get("compiled_entry_sha256"),
        _canonical_json_sha256(entry),
        label="compiled selected on-policy entry hash",
    )
    _assert_equal(
        document.get("child_argv"),
        expected_child_argv,
        label="final on-policy child argv",
    )
    _assert_equal(
        document.get("child_argv_sha256"),
        _canonical_json_sha256(expected_child_argv),
        label="final on-policy child argv hash",
    )
    if not expected_interpreter or not expected_child_argv:
        raise ValueError("recompiled on-policy target has no interpreter and training script")
    _assert_equal(document.get("training_script"), expected_child_argv[0], label="on-policy training script")
    _assert_equal(document.get("interpreter_argv0"), expected_interpreter, label="on-policy interpreter argv[0]")
    _binary_identity_equal(
        expected_interpreter,
        document.get("interpreter"),
        label="on-policy child interpreter",
    )
    _identity_equal(
        _training_script_path(expected_child_argv[0]),
        document.get("training_script_identity"),
        label="on-policy training script",
    )
    return compiled, entry, expected_child_argv


def validate_onpolicy_target_attestation_for_runner(
    *,
    attestation: str | Path,
    plan: str | Path,
    target: str,
    child_argv: Sequence[str],
    interpreter: str | Path,
    cuda_visible_devices: str | None,
    topology_profile: str | None = None,
) -> dict[str, Any]:
    """Validate the runner's second YAML read before it starts a child.

    The runner sees the YAML after the launcher's first proof.  This gate makes
    that second read fail closed if either the raw plan, factory-expanded entry,
    or final injected command differs from the immutable target sidecar.
    """

    attestation_path, document = _read_target_attestation(attestation)
    _assert_equal(
        document.get("topology_profile"),
        topology_profile,
        label="runner on-policy topology profile",
    )
    plan_path = Path(plan).resolve()
    _compiled, _entry, expected_child_argv = _validate_target_sidecar_shape(
        document=document,
        plan=plan_path,
        target=target,
        attestation=attestation_path,
    )
    _assert_equal(list(child_argv), expected_child_argv, label="runner on-policy child argv")
    _assert_equal(
        str(interpreter),
        document.get("interpreter_argv0"),
        label="runner on-policy interpreter argv[0]",
    )
    _binary_identity_equal(interpreter, document.get("interpreter"), label="runner on-policy interpreter")
    _assert_equal(
        document.get("cuda_visible_devices"),
        _normalized_cuda_visible_devices(cuda_visible_devices),
        label="runner on-policy CUDA_VISIBLE_DEVICES",
    )
    return document


def _attested_path(identity: Any, *, label: str) -> Path:
    if not isinstance(identity, dict) or not isinstance(identity.get("path"), str) or not identity["path"]:
        raise ValueError(f"{label} has no valid path identity")
    return Path(identity["path"]).resolve()


def validate_onpolicy_target_attestation_for_child(
    *,
    attestation: str | Path,
    expected_training_script: str,
    child_argv: Sequence[str],
    interpreter: str | Path,
    training_script_path: str | Path,
    cuda_visible_devices: str | None,
) -> dict[str, Any]:
    """Run the child-side no-model gate before it loads data or a backend.

    This independently repeats the entire cheap portion of the launch proof:
    recompile the selected entry, bind the actual argv, rerun the exact source
    recovery proof, and rebuild the Qwen worker-parity proof against the live
    allocation.  The training CLIs call this before ``prepare_setting``,
    ``load_and_combine``, ``build_backend``, or trainer construction.
    """

    attestation_path, document = _read_target_attestation(attestation)
    _assert_equal(document.get("training_script"), expected_training_script, label="child training script")
    plan_path = _attested_path(document.get("authored_plan"), label="authored on-policy YAML")
    target = document.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError("on-policy target attestation has no target name")
    _compiled, entry, expected_child_argv = _validate_target_sidecar_shape(
        document=document,
        plan=plan_path,
        target=target,
        attestation=attestation_path,
    )
    # Python exposes the script and its arguments in sys.argv, not the
    # interpreter. Verify the true executable separately while retaining the
    # exact child-visible argv shape sealed by the runner.
    _assert_equal(list(child_argv), expected_child_argv, label="actual on-policy child argv")
    _assert_equal(
        str(Path(interpreter).resolve()),
        str(Path(str(document.get("interpreter_argv0", ""))).resolve()),
        label="actual on-policy interpreter argv[0]",
    )
    _binary_identity_equal(interpreter, document.get("interpreter"), label="actual on-policy child interpreter")
    expected_script_path = _training_script_path(expected_training_script)
    _assert_equal(
        Path(training_script_path).resolve(),
        expected_script_path,
        label="actual on-policy training script path",
    )
    _identity_equal(
        expected_script_path,
        document.get("training_script_identity"),
        label="actual on-policy training script",
    )
    _assert_equal(
        document.get("cuda_visible_devices"),
        _normalized_cuda_visible_devices(cuda_visible_devices),
        label="actual on-policy CUDA_VISIBLE_DEVICES",
    )
    args = entry.get("args")
    if not isinstance(args, dict):
        raise ValueError("compiled selected on-policy entry has no argument object")
    _assert_equal(args.get("model"), MODEL, label="child on-policy model")
    _assert_equal(args.get("experiment_name"), "${experiment}", label="child experiment placeholder")
    _assert_equal(args.get("run_name"), document.get("run_name"), label="child run namespace")

    source_path = _attested_path(document.get("source"), label="attested recovered source")
    manifest_path = _attested_path(document.get("source_manifest"), label="attested recovered source manifest")
    source_attestation_path = _attested_path(document.get("source_attestation"), label="source proof sidecar")
    _identity_equal(source_path, document.get("source"), label="attested recovered source")
    _identity_equal(manifest_path, document.get("source_manifest"), label="attested recovered source manifest")
    source_document = validate_recovered_none_source_attestation(
        attestation=source_attestation_path,
        source=source_path,
        source_manifest=manifest_path,
    )
    _identity_equal(source_attestation_path, document.get("source_attestation"), label="source proof sidecar")
    _assert_equal(
        verify_recovered_none_source(source=source_path, source_manifest=manifest_path),
        source_document,
        label="child exact recovered-source proof",
    )

    selection_record = document.get("rmct256_selection")
    selection_paths = _rmct256_selection_paths_from_attestation(selection_record)
    if selection_paths is not None:
        for name in _RMCT256_SELECTION_INPUTS:
            _identity_equal(
                selection_paths[name],
                selection_record[name],
                label=f"attested RMCT-256 {name}",
            )
        _assert_equal(
            _verify_rmct256_selection_bundle(
                selection_paths,
                source=source_path,
                source_manifest=manifest_path,
            ),
            selection_record,
            label="child exact RMCT-256 selection proof",
        )

    worker = document.get("worker")
    if not isinstance(worker, dict):
        raise ValueError("on-policy target attestation has no worker contract")
    worker_attestation_path = _attested_path(
        document.get("worker_parity_attestation"),
        label="worker-parity sidecar",
    )
    expected_worker_sidecar = (
        PROJECT_ROOT
        / "logs"
        / str(document.get("experiment_name"))
        / str(document.get("run_name"))
        / "rollout_workers"
        / "qwen35-rollout-worker-parity-attestation.json"
    ).resolve()
    _assert_equal(worker_attestation_path, expected_worker_sidecar, label="child worker parity sidecar path")
    _identity_equal(worker_attestation_path, document.get("worker_parity_attestation"), label="worker-parity sidecar")
    _validate_worker_parity_attestation(
        worker_parity_attestation=worker_attestation_path,
        worker=worker,
        cuda_visible_devices=cuda_visible_devices,
    )

    # Reapply the semantic source/worker contract, not just byte equality.  It
    # catches a sidecar that is structurally self-consistent but names a source
    # or execution option that the compiled target cannot actually consume.
    contract = verify_onpolicy_target_contract(
        plan=plan_path,
        target=target,
        source=source_path,
        source_manifest=manifest_path,
        experiment_name=str(document.get("experiment_name")),
        run_name=str(document.get("run_name")),
        worker_gpus=str(worker.get("rollout_gpus")),
        worker_gpu_mem_util=float(worker.get("gpu_memory_utilization")),
        worker_max_model_len=int(worker.get("max_model_len")),
        worker_max_num_seqs=int(worker.get("max_num_seqs")),
        worker_max_num_batched_tokens=int(worker.get("max_num_batched_tokens")),
        target_logprob_chunk_size=int(worker.get("target_logprob_chunk_size")),
        worker_gdn_prefill_backend=(
            str(worker["gdn_prefill_backend"])
            if worker.get("gdn_prefill_backend") is not None
            else None
        ),
        worker_seed_base=int(worker.get("seed_base")),
        topology_profile=document.get("topology_profile"),
        rmct256_selection=selection_paths["selection"] if selection_paths is not None else None,
        rmct256_selection_manifest=selection_paths["selection_manifest"] if selection_paths is not None else None,
        rmct256_canonical_source=selection_paths["canonical_source"] if selection_paths is not None else None,
        rmct256_canonical_source_manifest=(
            selection_paths["canonical_source_manifest"] if selection_paths is not None else None
        ),
        rmct256_original64_reference_manifest=(
            selection_paths["original64_reference_manifest"] if selection_paths is not None else None
        ),
        rmct256_stage2_manifest=selection_paths["stage2_manifest"] if selection_paths is not None else None,
    )
    _assert_equal(contract["compiled_entry"], entry, label="child recompiled selected entry")
    _assert_equal(contract["worker"], worker, label="child worker contract")
    identity = plain_file_identity(attestation_path)
    return {
        "schema": TARGET_ATTESTATION_SCHEMA,
        "path": str(attestation_path),
        "sha256": identity["content_sha256"],
        "target": target,
        "source_sha256": document["source"]["content_sha256"],
        **(
            {"rmct256_selection_sha256": selection_record["selection"]["content_sha256"]}
            if selection_record is not None
            else {}
        ),
        "worker_parity_sha256": document["worker_parity_attestation"]["content_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="attested recovered none-style pair JSONL")
    parser.add_argument("--source-manifest", required=True, type=Path, help="exact CoT-to-none recovery manifest")
    parser.add_argument("--output", type=Path, help="immutable identity sidecar to create or resume")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the exact content proof and print identities without writing a sidecar",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.dry_run == (args.output is not None):
        # Exactly one mode: a regular run records the proof; dry-run has no
        # write and therefore must not accept an accidental output path.
        raise SystemExit("error: pass --output for a recording run, or --dry-run without --output")
    try:
        if args.dry_run:
            document = verify_recovered_none_source(source=args.source, source_manifest=args.source_manifest)
            print("QWEN35_RECOVERED_NONE_SOURCE_PREFLIGHT_DRY_RUN=" + json.dumps(document, sort_keys=True))
        else:
            result = attest_recovered_none_source(
                source=args.source,
                source_manifest=args.source_manifest,
                output=args.output,
            )
            print(
                "QWEN35_RECOVERED_NONE_SOURCE_ATTESTATION="
                + json.dumps(
                    {
                        "status": result["status"],
                        "path": result["path"],
                        "source": result["attestation"]["source"],
                        "source_manifest": result["attestation"]["source_manifest"],
                    },
                    sort_keys=True,
                )
            )
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()


__all__ = [
    "ATTESTATION_SCHEMA",
    "TARGET_CONTRACT_SCHEMA",
    "TARGET_ATTESTATION_SCHEMA",
    "attest_onpolicy_target",
    "attest_recovered_none_source",
    "validate_onpolicy_target_attestation_for_child",
    "validate_onpolicy_target_attestation_for_runner",
    "validate_recovered_none_source_attestation",
    "verify_onpolicy_target_contract",
    "verify_recovered_none_source",
]
