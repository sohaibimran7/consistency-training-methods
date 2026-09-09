"""Fail-closed launch contract for native-HF/PEFT Stage 2 OOD raw runs.

The Stage 2 OOD matrix is deliberately separate from the Stage 1 IID
diagnostic.  In particular, it has three shared clean tasks and eighteen
biased tasks across two populations and six bias variants.  Do not reuse the
Stage 1 raw-preflight schema for this suite.

This module is CPU-only: it validates immutable local inputs and writes a
hash-bound launch contract.  The companion shell launcher subsequently calls
``scripts/run_evals.py`` with a raw LocalBackend checkpoint through Inspect's
native Hugging Face provider, where ``PeftModel.from_pretrained`` applies the
adapter.  It never initializes a model or contacts a remote service itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ctm.evals.local_model import read_local_checkpoint
from experiments.stage2_ood_hle.materialize import PROMPT_STYLE, validate_manifest
from experiments.stage2_ood_hle.tasks import ood_task_specs


LAUNCH_SCHEMA = "stage2-ood-hf-peft-raw-launch-v1"
BASE_MODEL = "Qwen/Qwen3.5-9B"
TASK_FACTORY = "experiments.stage2_ood_hle.tasks:ood_tasks"
EXPECTED_TASKS = 21
EXPECTED_UNBIASED_TASKS = 3
EXPECTED_BIASED_TASKS = 18


@dataclass(frozen=True, slots=True)
class HFPEFTCondition:
    """Immutable identity of one original, raw LoRA checkpoint."""

    checkpoint_name: str
    adapter_model_sha256: str
    adapter_config_sha256: str
    manifest_sha256: str


# These are the byte identities of the original final adapters.  They are
# intentionally *not* the translated vLLM compatibility adapters: native HF
# evaluation applies the exact PEFT tensors trained by LocalBackend.
CONDITIONS: dict[str, HFPEFTCondition] = {
    "bct-hf-peft": HFPEFTCondition(
        checkpoint_name="bct",
        adapter_model_sha256="b7b6d2545797894e9f976f497ee6e5c8b09c26a3b2185aa36be82e4c308285ac",
        adapter_config_sha256="b29d45fb65363175b99a9230d4bf33adefddf26e155e7b13d33d02c96e6319a9",
        manifest_sha256="8612a1651547ebc94348036d93bfdc71dc07f7e91ba9194c67fdc173182a5b2d",
    ),
    "bct-control-hf-peft": HFPEFTCondition(
        checkpoint_name="bct-control",
        adapter_model_sha256="84efdc5b38f488ad49ef3659518408ec4f93cf624bf6f30b535cbcf897ad19f5",
        adapter_config_sha256="a3814737831811f7b46f7df475cb220bec20c80f803f1af6c0ce9cd7d98f88ac",
        manifest_sha256="8612a1651547ebc94348036d93bfdc71dc07f7e91ba9194c67fdc173182a5b2d",
    ),
    # Evaluate RMCT's raw adapter natively as well as through its separately
    # attested vLLM compatibility copy.  This is the backend-matched main
    # condition for the native-HF RMCT control below.
    "rmct-hf-peft": HFPEFTCondition(
        checkpoint_name="rmct_paper_isambard_phase2_qwen3_5_9b_rng_repair_4gpu_20260803_rate-matching-lr-1e-4",
        adapter_model_sha256="0ea90421fa2f81fee390288aca21b57d942945ebea10afc28b2a5267d244d707",
        adapter_config_sha256="0ddae2df7fd16cc50e03a7bdaa4118175c805e08c1d1a714d03107c8f943c603",
        manifest_sha256="d846cd872e392eafd21be095e4fcab1694a5b8cf9b798f4c7a7e4ed118ddadb5",
    ),
    "rmct-control-hf-peft": HFPEFTCondition(
        checkpoint_name="rmct_paper_isambard_phase2_qwen3_5_9b_rng_repair_4gpu_20260803_rate-matching-control-lr-1e-4",
        adapter_model_sha256="bded56fd53c606d8b589667d2afcda8213d6c9ef692d2101ed42598f53663f14",
        adapter_config_sha256="384cbfb571704f03e308a224d4659e22ab68f392a7d66c3cd9c67d547a3cc4b2",
        manifest_sha256="d846cd872e392eafd21be095e4fcab1694a5b8cf9b798f4c7a7e4ed118ddadb5",
    ),
    # Phase 2 OPCT is evaluated through the untouched PEFT adapter.  This is
    # deliberately distinct from the legacy OPCT vLLM target, whose serving
    # policy requires a separately translated and parity-attested copy.
    "opct-phase2-hf-peft": HFPEFTCondition(
        checkpoint_name="rmct_paper_isambard_phase2_qwen3_5_9b_opct_rng_repair_4gpu_20260803_opct-lr-1e-4",
        adapter_model_sha256="d893a6c202e5c9b0a1d00358b1f656d21ad99168613758046e64749845d8a7dd",
        adapter_config_sha256="c68aca369a9b5c31449b54c56f9d6c0df113f19cbf2b95fd8d6b57eef7c7b5e6",
        manifest_sha256="cef600b8c8bfe5e3d874b717e3e9ee993f0d114b525cd89de0f5aafa4b6e0d0e",
    ),
}

# The completed Stage 2 pipeline publishes these two conditions under their
# paper labels.  These aliases bind the exact same immutable PEFT artifacts,
# allowing incremental grading to land in the eventual canonical output tree
# and be reused without another paid grader call.
CONDITIONS["rmct"] = CONDITIONS["rmct-hf-peft"]
CONDITIONS["rmct-control"] = CONDITIONS["rmct-control-hf-peft"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str, *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 does not match the immutable {label} identity: {path}")
    return observed


def _strict_positive_integer(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def condition_spec(condition: str) -> HFPEFTCondition:
    """Return one supported raw-checkpoint condition or reject it."""

    try:
        return CONDITIONS[condition]
    except KeyError as exc:
        raise ValueError(f"unsupported Stage 2 native-HF/PEFT condition {condition!r}; choose from {sorted(CONDITIONS)}") from exc


def _read_adapter_config(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid PEFT adapter configuration: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"PEFT adapter configuration must be an object: {path}")
    return value


def validate_raw_hf_peft_checkpoint(condition: str, checkpoint: str | Path) -> dict[str, Any]:
    """Prove that ``checkpoint`` is the condition's original PEFT adapter."""

    expected = condition_spec(condition)
    directory, manifest = read_local_checkpoint(checkpoint)
    if directory.name != expected.checkpoint_name:
        raise ValueError(f"{condition}: raw checkpoint directory must be named {expected.checkpoint_name!r}, got {directory.name!r}")
    if manifest.get("model") != BASE_MODEL or manifest.get("lora") is not True:
        raise ValueError(f"{condition}: checkpoint is not a {BASE_MODEL} LoRA checkpoint")

    adapter_model = directory / "adapter_model.safetensors"
    adapter_config = directory / "adapter_config.json"
    checkpoint_manifest = directory / "manifest.json"
    observed = {
        "adapter_model_sha256": _require_sha256(
            adapter_model,
            expected.adapter_model_sha256,
            label="raw PEFT adapter_model.safetensors",
        ),
        "adapter_config_sha256": _require_sha256(
            adapter_config,
            expected.adapter_config_sha256,
            label="raw PEFT adapter_config.json",
        ),
        "manifest_sha256": _require_sha256(
            checkpoint_manifest,
            expected.manifest_sha256,
            label="raw LocalBackend manifest.json",
        ),
    }
    config = _read_adapter_config(adapter_config)
    if config.get("base_model_name_or_path") != BASE_MODEL:
        raise ValueError(f"{condition}: PEFT adapter config does not bind {BASE_MODEL}")
    return {
        "path": str(directory),
        "checkpoint_name": expected.checkpoint_name,
        "backend": "local",
        "lora": True,
        "base_model": BASE_MODEL,
        **observed,
    }


def stage2_task_args(manifest: str | Path, raw_log_dir: str | Path) -> dict[str, Any]:
    """Build the exact task arguments for the raw/no-Luna Stage 2 matrix."""

    log_path = Path(raw_log_dir).resolve()
    if not str(log_path):  # pragma: no cover - Path always renders non-empty
        raise ValueError("raw Stage 2 OOD log directory must be non-empty")
    return {
        "manifest": str(Path(manifest).resolve()),
        "unbiased_log": str(log_path),
        "prompt_style": PROMPT_STYLE,
        "include_bias_acknowledged": False,
    }


def build_launch_contract(
    *,
    condition: str,
    checkpoint: str | Path,
    manifest: str | Path,
    raw_log_dir: str | Path,
    max_connections: int,
) -> dict[str, Any]:
    """Validate the complete CPU-only Stage 2 native-HF/PEFT launch contract."""

    max_connections = _strict_positive_integer(max_connections, label="max_connections")
    manifest_path = Path(manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage 2 OOD manifest does not exist: {manifest_path}")
    validate_manifest(manifest_path)
    specs = ood_task_specs(manifest_path)
    unbiased_tasks = sum(spec.kind == "unbiased" for spec in specs)
    biased_tasks = sum(spec.kind == "biased" for spec in specs)
    if (len(specs), unbiased_tasks, biased_tasks) != (
        EXPECTED_TASKS,
        EXPECTED_UNBIASED_TASKS,
        EXPECTED_BIASED_TASKS,
    ):
        raise ValueError("Stage 2 OOD task factory does not resolve to the required 3 clean + 18 biased matrix")

    generation_config = {
        "max_tokens": 20480,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_connections": max_connections,
    }
    model_args = {
        "provider": "hf",
        "device": "cuda:0",
        "dtype": "bfloat16",
    }
    task_args = stage2_task_args(manifest_path, raw_log_dir)
    return {
        "schema": LAUNCH_SCHEMA,
        "condition": condition,
        "mode": "raw-no-luna",
        "checkpoint": validate_raw_hf_peft_checkpoint(condition, checkpoint),
        "stage2": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "task_factory": TASK_FACTORY,
            "task_count": len(specs),
            "unbiased_task_count": unbiased_tasks,
            "biased_task_count": biased_tasks,
            "prompt_style": PROMPT_STYLE,
            "include_bias_acknowledged": False,
            "grader_model": None,
        },
        "task_args": task_args,
        "model_args": model_args,
        "generation_config": generation_config,
        "execution": {
            "max_tasks": 1,
            # Every production native-HF launcher uses the parent as a
            # task-isolation supervisor: each selected task executes in a
            # fresh child, so its model and async lifecycle cannot leak into
            # another task.  Keep this in the immutable launch contract as
            # well as in the shell argv; otherwise a receipt would falsely
            # describe the execution that generated its EvalLogs.
            "isolate_tasks": True,
            "persistent_vllm_server": False,
        },
    }


def write_launch_contract(path: str | Path, contract: Mapping[str, Any]) -> str:
    """Write an immutable launch contract, or resume only if it is byte-identical."""

    destination = Path(path)
    payload = (json.dumps(dict(contract), indent=2, sort_keys=True) + "\n").encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing Stage 2 native-HF/PEFT launch contract: {destination}")
        return "resumed"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return "written"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-log-dir", required=True, type=Path)
    parser.add_argument("--max-connections", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        contract = build_launch_contract(
            condition=args.condition,
            checkpoint=args.checkpoint,
            manifest=args.manifest,
            raw_log_dir=args.raw_log_dir,
            max_connections=args.max_connections,
        )
        status = write_launch_contract(args.output, contract)
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{status}: {args.output.resolve()}")


__all__ = [
    "BASE_MODEL",
    "CONDITIONS",
    "EXPECTED_BIASED_TASKS",
    "EXPECTED_TASKS",
    "EXPECTED_UNBIASED_TASKS",
    "HFPEFTCondition",
    "LAUNCH_SCHEMA",
    "TASK_FACTORY",
    "build_launch_contract",
    "condition_spec",
    "stage2_task_args",
    "validate_raw_hf_peft_checkpoint",
    "write_launch_contract",
]


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
