#!/usr/bin/env python3
"""Run the final BCT/ACT/AttCT/MLPCT/OPCT AITA-NTA-FLIP replication.

This is a methods-only counterpart to the uncapped RMCT r006 AITA runner.
It deliberately reuses that runner's benchmark-owned task construction,
EOS-only native-HF hook, Inspect receipt validation, and pair-coverage
preflight.  The one different custody boundary is the source model: every
condition is a final raw PEFT adapter below one supplied, regular checkpoint
registry directory.

The registry has exactly these direct children (no links): ``bct``, ``act``,
``attct``, ``mlpct``, and ``opct``.  ``act`` is the canonical repaired-ACT
adapter.  Each child must contain the raw PEFT weights/configuration plus its
training ``manifest.json`` and ``README.md``.  A sibling ``provenance``
directory contains one pinned Stage-2 lineage record per method.  All supplied
SHA-256 identities and final loop states are checked before launch-contract
publication and on every worker/finalizer custody replay; the actual positive
byte sizes are retained in the immutable contract too.

The companion sbatch file owns one allocation only.  It runs a 16-cell first
phase (BCT, repaired ACT, AttCT, MLPCT) followed by a four-cell second phase
(OPCT).  It never submits, cancels, retries, or chains a scheduler job.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_rmct_r006_module():
    """Load the large, already-reviewed uncapped AITA implementation once."""

    source = PROJECT_ROOT / "infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu.py"
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"the required uncapped RMCT r006 AITA launcher is unavailable: {source}")
    name = "_methods_aita_rmct_r006_pipeline"
    module = sys.modules.get(name)
    if module is not None:
        return module
    specification = importlib.util.spec_from_file_location(name, source)
    if specification is None or specification.loader is None:  # pragma: no cover - deployment failure
        raise RuntimeError(f"could not load the uncapped RMCT r006 AITA launcher: {source}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


_r006 = _load_rmct_r006_module()

# Preserve the r006 implementation's public error and path types.  Its
# functions continue to use their own module globals, which are patched below
# only for this process; no existing source file is modified.
EvaluationError = _r006.EvaluationError
CampaignPaths = _r006.CampaignPaths
ConditionPaths = _r006.ConditionPaths
Condition = _r006.Condition


LAUNCH_SCHEMA = "methods-elephant-aita-nta-flip-16gpu-launch-v1-r006"
EVALUATION_SCHEMA = "methods-elephant-aita-nta-flip-16gpu-evaluation-v1-r006"
TASK_RECEIPT_SCHEMA = "methods-elephant-aita-nta-flip-16gpu-task-receipt-v1-r006"
SMOKE_RECEIPT_SCHEMA = "methods-elephant-aita-nta-flip-16gpu-direct-verdict-smoke-v1-r006"
COMPLETION_SCHEMA = "methods-elephant-aita-nta-flip-16gpu-completion-v1-r006"
GPU_TOPOLOGY_RECORD_SCHEMA = "methods-elephant-aita-nta-flip-16gpu-gpu-topology-record-v1-r006"
GPU_TOPOLOGY_RECEIPT_SCHEMA = "methods-elephant-aita-nta-flip-16gpu-gpu-topology-receipt-v1-r006"
CAMPAIGN_NAME = "elephant-aita-nta-flip-qwen35-methods-16gpu-v1-r006"

METHOD_CONDITION_COUNT = 5
PHASE_ONE_CELL_COUNT = 16
PHASE_TWO_CELL_COUNT = 4
TOTAL_CELL_COUNT = PHASE_ONE_CELL_COUNT + PHASE_TWO_CELL_COUNT
CONCURRENT_GPU_COUNT = 16


@dataclass(frozen=True)
class RawAdapterSpec:
    """Pinned direct-child location and immutable byte identity for one method."""

    directory_name: str
    canonical_source: str
    final_step: int
    adapter_model_sha256: str
    adapter_config_sha256: str
    manifest_sha256: str
    readme_sha256: str
    provenance_filename: str
    provenance_sha256: str


# The configuration hashes are intentionally those of the raw PEFT configs,
# not a translated compatibility adapter or a derived evaluation checkpoint.
# The README is identical across these LocalBackend checkpoint bundles.  The
# provenance records are copied byte-for-byte into ``registry/provenance``;
# historical paths inside them are descriptive lineage evidence, never an
# execution location.
RAW_ADAPTER_REGISTRY: dict[str, RawAdapterSpec] = {
    "bct": RawAdapterSpec(
        directory_name="bct",
        canonical_source="final-bct-raw-peft-adapter",
        final_step=32,
        adapter_model_sha256="b7b6d2545797894e9f976f497ee6e5c8b09c26a3b2185aa36be82e4c308285ac",
        adapter_config_sha256="b29d45fb65363175b99a9230d4bf33adefddf26e155e7b13d33d02c96e6319a9",
        manifest_sha256="8612a1651547ebc94348036d93bfdc71dc07f7e91ba9194c67fdc173182a5b2d",
        readme_sha256="152fcb11282f5418d54ac7cd69c55ac13312dceb41460c5e067ccd21437d599c",
        provenance_filename="bct.json",
        provenance_sha256="56fe6d8c461e507c4a6e42094f0ffd98725834d60716d9fe4577346495eac434",
    ),
    "act": RawAdapterSpec(
        directory_name="act",
        canonical_source="canonical-repaired-act-raw-peft-adapter",
        final_step=4000,
        adapter_model_sha256="7dd46d37ab9589fafd931ddb9c2e4135ffd345891dacce57e689e78312bbd330",
        adapter_config_sha256="65575f395b0e82c3ec3da3b0bfbe67782b0ffef3e84eb355d9d2b4898dbb29be",
        manifest_sha256="87f349e0b04294778350a7a964fef13b68ea489f7cd48db7269131c0df6017ad",
        readme_sha256="152fcb11282f5418d54ac7cd69c55ac13312dceb41460c5e067ccd21437d599c",
        provenance_filename="act.json",
        provenance_sha256="3b2e0664dd39f01cdd03c3b3641049689214f749e1c2d1ca1ac11ec0b6faac02",
    ),
    "attct": RawAdapterSpec(
        directory_name="attct",
        canonical_source="final-attct-raw-peft-adapter",
        final_step=256,
        adapter_model_sha256="2d18eeaf141d2e94305ba831f86874574f3582808483c28b1fdf700ebdd1bf4b",
        adapter_config_sha256="2fa7ef5b86db152b229823f4f909620f3d19bd8d78f7ae2fca2cc852307a1e49",
        manifest_sha256="5f4426a1a68e44869cea34686704c04ee1812612e4617374cb7f1b13c322d434",
        readme_sha256="152fcb11282f5418d54ac7cd69c55ac13312dceb41460c5e067ccd21437d599c",
        provenance_filename="attct.json",
        provenance_sha256="c2b89e16d6d209ddc851f5da00f267708b0205a743fe0fd61c2e30be413ebb16",
    ),
    "mlpct": RawAdapterSpec(
        directory_name="mlpct",
        canonical_source="final-mlpct-raw-peft-adapter",
        final_step=256,
        adapter_model_sha256="575e5d80ec083cd815200657bce0faf1e21566224d1adc6a4c4fe57341c86f43",
        adapter_config_sha256="ca7cf33545f38eb9a445d487e968da680d1cd019d7724173ff1af7e795253945",
        manifest_sha256="5f4426a1a68e44869cea34686704c04ee1812612e4617374cb7f1b13c322d434",
        readme_sha256="152fcb11282f5418d54ac7cd69c55ac13312dceb41460c5e067ccd21437d599c",
        provenance_filename="mlpct.json",
        provenance_sha256="308023482c82f934bfcd0af57abdf33bd2ae9bafb124725e2e583a36472fc778",
    ),
    "opct": RawAdapterSpec(
        directory_name="opct",
        canonical_source="final-opct-raw-peft-adapter",
        final_step=128,
        adapter_model_sha256="d893a6c202e5c9b0a1d00358b1f656d21ad99168613758046e64749845d8a7dd",
        adapter_config_sha256="c68aca369a9b5c31449b54c56f9d6c0df113f19cbf2b95fd8d6b57eef7c7b5e6",
        manifest_sha256="cef600b8c8bfe5e3d874b717e3e9ee993f0d114b525cd89de0f5aafa4b6e0d0e",
        readme_sha256="152fcb11282f5418d54ac7cd69c55ac13312dceb41460c5e067ccd21437d599c",
        provenance_filename="opct.json",
        provenance_sha256="06399ab2b722958192115678e4b355846453c2e9a2c252b4a1b3d5648dfeb38a",
    ),
}

CONDITIONS: tuple[Condition, ...] = (
    Condition("bct", "BCT", 32, "recovered-final-native-hf"),
    Condition("act", "ACT (canonical repaired-ACT)", 4000, "recovered-final-native-hf"),
    Condition("attct", "AttCT", 256, "recovered-final-native-hf"),
    Condition("mlpct", "MLPCT", 256, "recovered-final-native-hf"),
    Condition("opct", "OPCT", 128, "recovered-final-native-hf"),
)
_CONDITION_BY_NAME = {condition.name: condition for condition in CONDITIONS}


# The original RMCT r006 runner is deliberately listed as a critical source:
# this wrapper inherits its EOS-only sampling and raw-log validation rather
# than copying a second, potentially divergent task/preflight implementation.
CRITICAL_SOURCES = (
    "infra/isambard/run_qwen35_methods_aita_ntaflip_16gpu.py",
    "infra/isambard/run_qwen35_methods_aita_ntaflip_16gpu.sbatch",
    "infra/isambard/run_qwen35_methods_aita_ntaflip_16gpu_worker.sh",
    "infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu.py",
    "infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu.sbatch",
    "infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu_worker.sh",
    "scripts/run_evals.py",
    "ctm/evals/runner.py",
    "ctm/evals/local_model.py",
    "experiments/elephant_aita_ntaflip/__init__.py",
    "experiments/elephant_aita_ntaflip/prepare.py",
    "experiments/elephant_aita_ntaflip/tasks.py",
    "experiments/elephant_aita_ntaflip/no_cap_hf.py",
    "experiments/elephant_aita_ntaflip/preflight.py",
)


def _patch_r006_globals() -> None:
    """Specialize only the r006 module instance used by this launcher."""

    values = {
        "LAUNCH_SCHEMA": LAUNCH_SCHEMA,
        "EVALUATION_SCHEMA": EVALUATION_SCHEMA,
        "TASK_RECEIPT_SCHEMA": TASK_RECEIPT_SCHEMA,
        "SMOKE_RECEIPT_SCHEMA": SMOKE_RECEIPT_SCHEMA,
        "COMPLETION_SCHEMA": COMPLETION_SCHEMA,
        "GPU_TOPOLOGY_RECORD_SCHEMA": GPU_TOPOLOGY_RECORD_SCHEMA,
        "GPU_TOPOLOGY_RECEIPT_SCHEMA": GPU_TOPOLOGY_RECEIPT_SCHEMA,
        "CAMPAIGN_NAME": CAMPAIGN_NAME,
        "CONDITIONS": CONDITIONS,
        "_CONDITION_BY_NAME": _CONDITION_BY_NAME,
        "CRITICAL_SOURCES": CRITICAL_SOURCES,
    }
    for name, value in values.items():
        setattr(_r006, name, value)


_patch_r006_globals()

# Re-export the benchmark-owned values explicitly, so a static review of this
# small launcher can see that it inherits the same 1,591-pair task and decode
# contract rather than silently choosing new values.
EXPECTED_PAIRS = _r006.EXPECTED_PAIRS
PERSPECTIVES_PER_PAIR = _r006.PERSPECTIVES_PER_PAIR
GENERATIONS_PER_CONDITION = _r006.GENERATIONS_PER_CONDITION
SHARD_COUNT = _r006.SHARD_COUNT
TASK_FACTORY = _r006.TASK_FACTORY
GENERATION_CONFIG = _r006.GENERATION_CONFIG
RUNTIME_GENERATION_CONFIG = _r006.RUNTIME_GENERATION_CONFIG
HF_MODEL_ARGS = _r006.HF_MODEL_ARGS
HF_LOCAL_MODEL_ARGS = _r006.HF_LOCAL_MODEL_ARGS
CONCURRENCY_CONFIG = _r006.CONCURRENCY_CONFIG
MODEL_SNAPSHOT = _r006.MODEL_SNAPSHOT
NO_TOKEN_CAP_POLICY = _r006.NO_TOKEN_CAP_POLICY
NO_TOKEN_CAP_RUNTIME_POLICY = _r006.NO_TOKEN_CAP_RUNTIME_POLICY
QWEN_THINKING_POLICY = _r006.QWEN_THINKING_POLICY
GPU_BINDING = _r006.GPU_BINDING
GPU_DISTRIBUTION = _r006.GPU_DISTRIBUTION


def _registry_root(checkpoint_directory: str | Path) -> Path:
    requested = Path(checkpoint_directory).expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise EvaluationError(f"methods AITA checkpoint registry must be a regular directory: {requested}")
    return requested.resolve()


def _registry_spec(condition_name: str) -> RawAdapterSpec:
    try:
        return RAW_ADAPTER_REGISTRY[condition_name]
    except KeyError as exc:
        raise EvaluationError(f"unknown methods AITA adapter condition {condition_name!r}") from exc


def _checked_raw_adapter(*, checkpoint_directory: str | Path, condition_name: str) -> dict[str, Any]:
    """Validate a direct raw adapter and return its complete replay identity."""

    root = _registry_root(checkpoint_directory)
    spec = _registry_spec(condition_name)
    adapter = root / spec.directory_name
    if adapter.parent != root or adapter.is_symlink() or not adapter.is_dir():
        raise EvaluationError(
            f"{condition_name} raw adapter must be the regular direct registry child {spec.directory_name!r}: {adapter}"
        )
    adapter = adapter.resolve()
    _r006._under_root(adapter, root, label=f"{condition_name} raw adapter")
    if adapter.parent != root:
        raise EvaluationError(f"{condition_name} raw adapter resolves outside its direct registry child")

    weights = _r006._identity(adapter / "adapter_model.safetensors", label=f"{condition_name} adapter weights")
    config = _r006._identity(adapter / "adapter_config.json", label=f"{condition_name} adapter config")
    manifest_identity = _r006._identity(adapter / "manifest.json", label=f"{condition_name} training manifest")
    readme = _r006._identity(adapter / "README.md", label=f"{condition_name} checkpoint README")
    if weights["sha256"] != spec.adapter_model_sha256:
        raise EvaluationError(
            f"{condition_name} adapter_model.safetensors SHA-256 differs from the pinned raw-adapter identity"
        )
    if config["sha256"] != spec.adapter_config_sha256:
        raise EvaluationError(
            f"{condition_name} adapter_config.json SHA-256 differs from the pinned raw-adapter identity"
        )
    if manifest_identity["sha256"] != spec.manifest_sha256:
        raise EvaluationError(f"{condition_name} manifest.json SHA-256 differs from the pinned final endpoint")
    if readme["sha256"] != spec.readme_sha256:
        raise EvaluationError(f"{condition_name} README.md SHA-256 differs from the pinned checkpoint format")
    configuration = _r006._read_json(adapter / "adapter_config.json", label=f"{condition_name} PEFT adapter config")
    base_model = configuration.get("base_model_name_or_path")
    if base_model != "Qwen/Qwen3.5-9B":
        raise EvaluationError(
            f"{condition_name} PEFT adapter config must bind the raw Qwen/Qwen3.5-9B base model"
        )
    training_manifest = _r006._read_json(adapter / "manifest.json", label=f"{condition_name} training manifest")
    loop_state = training_manifest.get("loop_state")
    if (
        training_manifest.get("model") != "Qwen/Qwen3.5-9B"
        or training_manifest.get("lora") is not True
        or not isinstance(loop_state, Mapping)
        or loop_state.get("final") is not True
        or loop_state.get("step") != spec.final_step
    ):
        raise EvaluationError(
            f"{condition_name} manifest must be a final Qwen/Qwen3.5-9B LoRA endpoint at step {spec.final_step}"
        )

    provenance_root = root / "provenance"
    if provenance_root.is_symlink() or not provenance_root.is_dir():
        raise EvaluationError(f"methods AITA provenance must be a regular registry directory: {provenance_root}")
    provenance_path = provenance_root / spec.provenance_filename
    if provenance_path.parent != provenance_root or provenance_path.is_symlink() or not provenance_path.is_file():
        raise EvaluationError(f"{condition_name} provenance must be a regular direct file: {provenance_path}")
    provenance = _r006._identity(provenance_path, label=f"{condition_name} Stage-2 provenance")
    if provenance["sha256"] != spec.provenance_sha256:
        raise EvaluationError(f"{condition_name} Stage-2 provenance SHA-256 differs from the pinned lineage record")
    provenance_record = _r006._read_json(provenance_path, label=f"{condition_name} Stage-2 provenance")
    if condition_name in {"bct", "opct"}:
        checkpoint_record = provenance_record.get("checkpoint")
        if (
            not isinstance(checkpoint_record, Mapping)
            or checkpoint_record.get("adapter_model_sha256") != spec.adapter_model_sha256
            or checkpoint_record.get("adapter_config_sha256") != spec.adapter_config_sha256
            or checkpoint_record.get("manifest_sha256") != spec.manifest_sha256
            or checkpoint_record.get("base_model") != "Qwen/Qwen3.5-9B"
            or checkpoint_record.get("lora") is not True
        ):
            raise EvaluationError(f"{condition_name} Stage-2 provenance does not bind the pinned raw endpoint")
    else:
        source = provenance_record.get("source")
        if not isinstance(source, Mapping) or source.get("adapter_model_sha256") != spec.adapter_model_sha256:
            raise EvaluationError(f"{condition_name} compatibility provenance does not bind the pinned raw weights")
    return {
        "source": "recovered-final-native-hf",
        "condition": condition_name,
        "canonical_source": spec.canonical_source,
        "checkpoint": {
            "path": str(adapter),
            "adapter_model": weights,
            "adapter_config": config,
            "manifest": manifest_identity,
            "readme": readme,
            "stage2_provenance": provenance,
            "expected_adapter_model_sha256": spec.adapter_model_sha256,
            "expected_adapter_config_sha256": spec.adapter_config_sha256,
            "expected_manifest_sha256": spec.manifest_sha256,
            "final_step": spec.final_step,
            "loop_state": dict(loop_state),
            "base_model": base_model,
        },
    }


def _methods_trained_runtime(
    source: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any],
    evaluator: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_record = source.get("checkpoint")
    checkpoint = checkpoint_record.get("path") if isinstance(checkpoint_record, Mapping) else None
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
        raise EvaluationError("methods raw adapter custody has no absolute PEFT adapter path")
    return {
        "mode": "native-hf-peft",
        "checkpoint_backend": "local",
        "base_model": str(MODEL_SNAPSHOT),
        "model": f"hf/{MODEL_SNAPSHOT}",
        "model_snapshot": dict(snapshot),
        "checkpoint": checkpoint,
        "raw_adapter_custody": dict(source),
        "source_checkpoint": {
            "source": source["source"],
            "condition": source["condition"],
            "canonical_source": source["canonical_source"],
        },
        "provider": "hf",
        "model_args": dict(HF_LOCAL_MODEL_ARGS),
        "qwen_thinking_policy": dict(QWEN_THINKING_POLICY),
        "no_token_cap_policy": dict(NO_TOKEN_CAP_POLICY),
        "no_token_cap_runtime_policy": dict(NO_TOKEN_CAP_RUNTIME_POLICY),
        "sampling_config": dict(GENERATION_CONFIG),
        "concurrency_config": dict(CONCURRENCY_CONFIG),
        "generation_config": dict(RUNTIME_GENERATION_CONFIG),
        "evaluator": dict(evaluator),
    }


def _condition_runtime_records(
    *,
    checkpoint_directory: str | Path,
    manifest_sha256: str,
    snapshot: Mapping[str, Any] | None = None,
    evaluator: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Replay all five raw-adapter identities before any task may run."""

    snapshot_identity = dict(snapshot) if snapshot is not None else _r006._snapshot_identity()
    evaluator_identity = dict(evaluator) if evaluator is not None else _r006._evaluator_runtime()
    records: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        source = _checked_raw_adapter(checkpoint_directory=checkpoint_directory, condition_name=condition.name)
        runtime = _methods_trained_runtime(source, snapshot=snapshot_identity, evaluator=evaluator_identity)
        records[condition.name] = _r006._attach_task_metadata(runtime, manifest_sha256=manifest_sha256)
    return records


def _r006_condition_runtime_records(
    *,
    training_repository: str | Path,
    manifest_sha256: str,
    snapshot: Mapping[str, Any] | None = None,
    evaluator: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compatibility bridge for inherited worker/finalizer helpers only."""

    return _condition_runtime_records(
        checkpoint_directory=training_repository,
        manifest_sha256=manifest_sha256,
        snapshot=snapshot,
        evaluator=evaluator,
    )


def _phase_plan() -> list[dict[str, Any]]:
    phase_one = [
        {"condition": condition.name, "shard_index": shard}
        for condition in CONDITIONS[:4]
        for shard in range(SHARD_COUNT)
    ]
    phase_two = [
        {"condition": "opct", "shard_index": shard}
        for shard in range(SHARD_COUNT)
    ]
    if len(phase_one) != PHASE_ONE_CELL_COUNT or len(phase_two) != PHASE_TWO_CELL_COUNT:
        raise RuntimeError("methods AITA phase plan no longer has 16 then 4 cells")
    return [
        {"name": "phase-1", "allocated_gpus": CONCURRENT_GPU_COUNT, "cells": phase_one},
        {"name": "phase-2", "allocated_gpus": PHASE_TWO_CELL_COUNT, "cells": phase_two},
    ]


def build_launch_contract(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    checkpoint_directory: str | Path,
) -> tuple[dict[str, Any], CampaignPaths]:
    """Build a five-method, 20-cell contract after raw-adapter replay."""

    paths = _r006._campaign_paths(campaign_root)
    topology_receipt = _r006._sealed_gpu_topology_receipt(paths)
    evaluator = _r006._validate_evaluator_environment(require_one_gpu=False)
    registry_root = _registry_root(checkpoint_directory)
    official = _r006.materialize_official_manifest(official_source_dir=official_source_dir, paths=paths)
    snapshot = _r006._snapshot_identity()
    runtime = _condition_runtime_records(
        checkpoint_directory=registry_root,
        manifest_sha256=official["manifest"]["sha256"],
        snapshot=snapshot,
        evaluator=evaluator,
    )
    phase_plan = _phase_plan()
    cells = [cell for phase in phase_plan for cell in phase["cells"]]
    if len(cells) != TOTAL_CELL_COUNT or len({(cell["condition"], cell["shard_index"]) for cell in cells}) != TOTAL_CELL_COUNT:
        raise RuntimeError("methods AITA topology has an incomplete or duplicate cell plan")
    contract = {
        "schema": LAUNCH_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "checkpoint_registry": {
            "root": str(registry_root),
            "direct_children": {name: spec.directory_name for name, spec in RAW_ADAPTER_REGISTRY.items()},
            "provenance_directory": str(registry_root / "provenance"),
            "identity_policy": "recovered-final-native-hf-pinned-files-loop-state-and-stage2-lineage",
        },
        "model_snapshot": snapshot,
        "official_manifest": official,
        "gpu_topology_probe": _r006._identity(
            _r006._gpu_topology_receipt_path(paths), label="methods AITA r006 GPU-topology receipt"
        ),
        "benchmark": _r006._benchmark_contract(official["validated"]),
        "evaluator": evaluator,
        "conditions": [
            {
                "name": item.name,
                "label": item.label,
                "optimizer_step": item.optimizer_step,
                "source_kind": item.source_kind,
                "runtime": runtime[item.name],
            }
            for item in CONDITIONS
        ],
        "topology": {
            "nodes": 4,
            "gpus_per_node": 4,
            "condition_count": METHOD_CONDITION_COUNT,
            "cell_count": TOTAL_CELL_COUNT,
            "concurrent_gpus": CONCURRENT_GPU_COUNT,
            "workers": CONCURRENT_GPU_COUNT,
            "one_gpu_per_worker": True,
            "phases": phase_plan,
            "cells": cells,
            "gpu_binding": {
                "gpu_bind": f"--gpu-bind={GPU_BINDING}",
                "distribution": f"--distribution={GPU_DISTRIBUTION}",
                "full_step_gpu_count": CONCURRENT_GPU_COUNT,
                "topology_probe_schema": topology_receipt["schema"],
                "topology_probe_sha256": _r006._identity(
                    _r006._gpu_topology_receipt_path(paths), label="methods AITA r006 GPU-topology receipt"
                )["sha256"],
            },
            "native_hf_peft_workers": CONCURRENT_GPU_COUNT,
        },
        "critical_sources": _r006._critical_source_identities(),
        "outputs": {
            "root": str(paths.root),
            "input": str(paths.input),
            "conditions": str(paths.conditions),
            "completion": str(paths.completion),
        },
        "policy": _r006._launch_policy(),
    }
    return contract, paths


def prepare(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    checkpoint_directory: str | Path,
) -> dict[str, Any]:
    """Create/replay five condition receipts without generating responses."""

    paths = _r006._campaign_paths(campaign_root)
    _r006._sealed_gpu_topology_receipt(paths)
    if paths.root.exists() and not paths.contract.exists() and {
        item.name for item in paths.root.iterdir()
    } != {_r006.GPU_TOPOLOGY_PROBE_DIRNAME}:
        raise FileExistsError(f"refusing to seed a methods AITA contract in a non-empty campaign root: {paths.root}")
    contract, paths = build_launch_contract(
        campaign_root=campaign_root,
        official_source_dir=official_source_dir,
        checkpoint_directory=checkpoint_directory,
    )
    status = _r006._write_immutable_json(paths.contract, contract, label="methods AITA campaign launch contract")
    receipts = {
        condition.name: _r006._write_condition_evaluation_receipt(paths=paths, contract=contract, condition=condition)
        for condition in CONDITIONS
    }
    return {
        "campaign": CAMPAIGN_NAME,
        "launch_contract": _r006._identity(paths.contract, label="methods AITA campaign launch contract"),
        "launch_status": status,
        "evaluation_receipt_statuses": receipts,
        "conditions": METHOD_CONDITION_COUNT,
        "cells": TOTAL_CELL_COUNT,
        "generations_per_condition": GENERATIONS_PER_CONDITION,
    }


def _load_campaign(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    checkpoint_directory: str | Path,
) -> tuple[dict[str, Any], CampaignPaths]:
    paths = _r006._campaign_paths(campaign_root)
    stored = _r006._read_json(paths.contract, label="methods AITA campaign launch contract")
    rebuilt, rebuilt_paths = build_launch_contract(
        campaign_root=campaign_root,
        official_source_dir=official_source_dir,
        checkpoint_directory=checkpoint_directory,
    )
    if paths != rebuilt_paths or stored != rebuilt:
        raise EvaluationError("methods AITA launch contract differs from current raw-adapter/runtime custody")
    return rebuilt, paths


def _r006_load_campaign(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    training_repository: str | Path,
) -> tuple[dict[str, Any], CampaignPaths]:
    return _load_campaign(
        campaign_root=campaign_root,
        official_source_dir=official_source_dir,
        checkpoint_directory=training_repository,
    )


# The inherited worker, smoke, finalizer, task-command, and preflight helpers
# remain safe because they route every campaign replay through this bridge.
_r006._condition_runtime_records = _r006_condition_runtime_records
_r006._load_campaign = _r006_load_campaign


def worker(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    checkpoint_directory: str | Path,
    condition_name: str,
    shard_index: int,
    python: str,
) -> dict[str, Any]:
    return _r006.worker(
        campaign_root=campaign_root,
        official_source_dir=official_source_dir,
        training_repository=checkpoint_directory,
        condition_name=condition_name,
        shard_index=shard_index,
        python=python,
    )


def smoke(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    checkpoint_directory: str | Path,
    condition_name: str,
    shard_index: int,
    python: str,
) -> dict[str, Any]:
    return _r006.smoke(
        campaign_root=campaign_root,
        official_source_dir=official_source_dir,
        training_repository=checkpoint_directory,
        condition_name=condition_name,
        shard_index=shard_index,
        python=python,
    )


def finalize(
    *,
    campaign_root: str | Path,
    official_source_dir: str | Path,
    checkpoint_directory: str | Path,
) -> dict[str, Any]:
    return _r006.finalize(
        campaign_root=campaign_root,
        official_source_dir=official_source_dir,
        training_repository=checkpoint_directory,
    )


# These aliases make the unchanged r006 task/preflight entry points available
# to focused static/command-contract tests without duplicating their logic.
gpu_topology_probe = _r006.gpu_topology_probe
seal_gpu_topology_probe = _r006.seal_gpu_topology_probe
_campaign_paths = _r006._campaign_paths
_condition_paths = _r006._condition_paths
_task_command = _r006._task_command
_assert_no_token_cap_command = _r006._assert_no_token_cap_command
assert_no_token_cap_mapping = _r006.assert_no_token_cap_mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def campaign_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--campaign-root", required=True, type=Path)
        command.add_argument("--official-source-dir", required=True, type=Path)
        command.add_argument("--checkpoint-directory", required=True, type=Path)

    topology_probe_parser = commands.add_parser(
        "probe-gpu-topology", help="write one rank record for the 16-rank methods AITA GPU probe"
    )
    topology_probe_parser.add_argument("--campaign-root", required=True, type=Path)
    topology_seal_parser = commands.add_parser(
        "seal-gpu-topology-probe", help="validate and seal the methods AITA GPU probe receipt"
    )
    topology_seal_parser.add_argument("--campaign-root", required=True, type=Path)

    prepare_parser = commands.add_parser("prepare", help="replay raw adapter identities and write campaign receipts")
    campaign_inputs(prepare_parser)
    prepare_parser.add_argument("--yes", action="store_true")

    worker_parser = commands.add_parser("worker", help="run one condition/shard on one Slurm-visible GPU")
    campaign_inputs(worker_parser)
    worker_parser.add_argument("--condition", choices=[condition.name for condition in CONDITIONS], required=True)
    worker_parser.add_argument("--shard-index", type=int, required=True)
    worker_parser.add_argument("--python", required=True)

    smoke_parser = commands.add_parser("smoke", help="run a small direct-verdict check before full evaluation")
    campaign_inputs(smoke_parser)
    smoke_parser.add_argument("--condition", choices=[condition.name for condition in CONDITIONS], required=True)
    smoke_parser.add_argument("--shard-index", type=int, required=True)
    smoke_parser.add_argument("--python", required=True)

    finalize_parser = commands.add_parser("finalize", help="preflight and seal all 20 completed cells")
    campaign_inputs(finalize_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    values: dict[str, Any] = {}
    if args.command not in {"probe-gpu-topology", "seal-gpu-topology-probe"}:
        values = {
            "campaign_root": args.campaign_root,
            "official_source_dir": args.official_source_dir,
            "checkpoint_directory": args.checkpoint_directory,
        }
    try:
        if args.command == "probe-gpu-topology":
            result = gpu_topology_probe(campaign_root=args.campaign_root)
        elif args.command == "seal-gpu-topology-probe":
            result = seal_gpu_topology_probe(campaign_root=args.campaign_root)
        elif args.command == "prepare":
            if not args.yes:
                parser.error("prepare requires --yes after reviewing the raw adapter registry")
            result = prepare(**values)
        elif args.command == "worker":
            result = worker(
                **values,
                condition_name=args.condition,
                shard_index=args.shard_index,
                python=args.python,
            )
        elif args.command == "smoke":
            result = smoke(
                **values,
                condition_name=args.condition,
                shard_index=args.shard_index,
                python=args.python,
            )
        elif args.command == "finalize":
            result = finalize(**values)
        else:  # pragma: no cover - argparse guarantees this is unreachable
            parser.error("unsupported command")
            return 2
    except (
        EvaluationError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
