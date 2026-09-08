#!/usr/bin/env python3
"""Seal the exact PEFT LoRA parameter inventory for RMCT256 Isambard runs.

This is deliberately a metadata-only construction: it loads the already
attested pinned snapshot's ``config.json`` and creates the causal-LM module
graph on PyTorch's ``meta`` device.  The same local-backend target resolver and
PEFT ``get_peft_model`` call used by training then produce the exact target
module names and LoRA A/B parameter shapes without materializing nine billion
weight values or starting a worker.  The pinned base-snapshot sidecar has
already hashed the real snapshot files, so config architecture plus that
sidecar gives a reproducible, fail-closed inventory without a redundant full
model load.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from infra.isambard import rmct256_convergence_segment_contract as contract


SCHEMA = contract.LORA_FINGERPRINT_SCHEMA
EXPECTED_LORA_CONFIG = contract.FROZEN_LORA_CONFIG


def fingerprint_path(root: Path, segment: contract.Segment) -> Path:
    return contract.run_root(root, segment) / "preflight" / "qwen35-lora-fingerprint-attestation.json"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_offline() -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise contract.ContractError("HF_HUB_OFFLINE=1 is required for the LoRA fingerprint gate")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise contract.ContractError("TRANSFORMERS_OFFLINE=1 is required for the LoRA fingerprint gate")


def _read_base_snapshot_attestation(path: Path, segment: contract.Segment) -> tuple[dict[str, Any], Path]:
    document = contract._json_object(path, label="LoRA fingerprint base-snapshot attestation")
    contract._expect(document.get("schema"), contract.BASE_SNAPSHOT_SCHEMA, label="LoRA base-snapshot schema")
    contract._expect(document.get("condition"), contract.CONDITION, label="LoRA base-snapshot condition")
    contract._expect(document.get("segment"), segment.global_index, label="LoRA base-snapshot segment")
    contract._expect(document.get("repo_id"), contract.MODEL_REPO, label="LoRA base-snapshot repository")
    contract._expect(document.get("revision"), contract.MODEL_REVISION, label="LoRA base-snapshot revision")
    contract._expect(document.get("hf_hub_offline"), True, label="LoRA base-snapshot offline policy")
    contract._expect(document.get("transformers_offline"), True, label="LoRA Transformers offline policy")
    snapshot = document.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise contract.ContractError("LoRA base-snapshot attestation has no snapshot record")
    raw_path = snapshot.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise contract.ContractError("LoRA base-snapshot attestation has no resolved snapshot path")
    resolved = Path(raw_path).resolve()
    contract._assert_regular_directory(resolved, label="LoRA pinned snapshot directory")
    if resolved.name != contract.MODEL_REVISION:
        raise contract.ContractError("LoRA pinned snapshot directory does not name the exact required revision")
    return document, resolved


def _lora_config(root: Path, plan: Path, segment: contract.Segment) -> tuple[dict[str, Any], dict[str, Any]]:
    validated = contract.validate_segment_plan(root, plan, segment)
    raw = validated["args"].get("lora_config")
    contract._expect(raw, EXPECTED_LORA_CONFIG, label="LoRA fingerprint frozen configuration")
    if not isinstance(raw, Mapping):  # defensive: the exact check above has already narrowed this
        raise contract.ContractError("LoRA fingerprint configuration is not an object")
    return dict(raw), validated


def _inventory_from_model(
    *,
    config: Any,
    model: Any,
    lora: Any,
    peft_module: Any,
    peft_lora_config: Any,
    get_peft_model: Any,
    resolve_target_modules: Any,
    resolve_target_parameters: Any,
) -> dict[str, Any]:
    """Resolve and record the exact PEFT inventory on an already-built model.

    This is deliberately factored from the snapshot constructor so the test
    suite can exercise the same target resolver and PEFT wrapping logic on a
    tiny meta-device causal-LM fixture.  Production always supplies the Qwen
    graph constructed from the attested pinned snapshot config.
    """

    target_modules = sorted(resolve_target_modules(model, lora))
    target_parameters = sorted(resolve_target_parameters(model, lora))
    if not target_modules and not target_parameters:
        raise contract.ContractError("resolved RMCT256 LoRA configuration selected no target modules or parameters")
    config_for_peft = peft_lora_config(
        r=lora.rank,
        lora_alpha=lora.resolved_alpha,
        lora_dropout=lora.dropout,
        target_modules=target_modules or [],
        target_parameters=target_parameters or None,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, config_for_peft)
    actual_config = peft_model.peft_config.get("default")
    if actual_config is None:
        raise contract.ContractError("PEFT did not expose the required default LoRA adapter")
    actual_targets = sorted(str(item) for item in (actual_config.target_modules or ()))
    actual_target_parameters = sorted(str(item) for item in (getattr(actual_config, "target_parameters", None) or ()))
    contract._expect(actual_targets, target_modules, label="resolved PEFT target modules")
    contract._expect(actual_target_parameters, target_parameters, label="resolved PEFT target parameters")
    trainable = []
    for name, parameter in sorted(peft_model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        shape = [int(dim) for dim in parameter.shape]
        trainable.append(
            {
                "name": name,
                "shape": shape,
                "numel": int(parameter.numel()),
                "dtype": str(parameter.dtype),
            }
        )
    if not trainable:
        raise contract.ContractError("resolved PEFT model exposes no trainable LoRA parameters")
    if len({item["name"] for item in trainable}) != len(trainable):
        raise contract.ContractError("resolved PEFT model exposes duplicate trainable parameter names")
    return {
        "derivation_mode": "meta_model_from_pinned_snapshot_config",
        "config_class": f"{type(config).__module__}.{type(config).__qualname__}",
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_dtype": str(next(model.parameters()).dtype),
        "peft_version": str(getattr(peft_module, "__version__", "unknown")),
        "adapter_name": "default",
        "peft": {
            "r": int(actual_config.r),
            "lora_alpha": int(actual_config.lora_alpha),
            "lora_dropout": float(actual_config.lora_dropout),
            "bias": str(actual_config.bias),
            "task_type": str(actual_config.task_type),
        },
        "resolved_target_modules": target_modules,
        "resolved_target_parameters": target_parameters,
        "trainable_parameters": trainable,
        "trainable_parameter_count": len(trainable),
        "trainable_parameter_numel": sum(item["numel"] for item in trainable),
    }


def _meta_peft_inventory(snapshot: Path, raw_lora: Mapping[str, Any]) -> dict[str, Any]:
    """Return the actual PEFT inventory for a pinned snapshot's meta model."""

    try:
        import peft
        import torch
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM

        from ctm.backends.local.engine import _lora_target_module_names, _lora_target_parameter_names
        from ctm.core.config import resolve_lora_config
    except ImportError as exc:  # pragma: no cover - deployment/bootstrap error
        raise contract.ContractError("torch, transformers, and peft are required for the LoRA fingerprint gate") from exc

    try:
        lora = resolve_lora_config(dict(raw_lora))
    except (TypeError, ValueError) as exc:
        raise contract.ContractError(f"cannot resolve frozen LoRA configuration: {exc}") from exc
    try:
        config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
        return _inventory_from_model(
            config=config,
            model=model,
            lora=lora,
            peft_module=peft,
            peft_lora_config=PeftLoraConfig,
            get_peft_model=get_peft_model,
            resolve_target_modules=_lora_target_module_names,
            resolve_target_parameters=_lora_target_parameter_names,
        )
    except (OSError, RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
        if isinstance(exc, contract.ContractError):
            raise
        raise contract.ContractError(f"cannot construct the pinned-snapshot PEFT LoRA fingerprint: {exc}") from exc
    finally:
        # Meta tensors have no weight allocation, but release the module graph
        # promptly before the three-worker probe begins on the same allocation.
        gc.collect()


def _document(
    *,
    root: Path,
    plan: Path,
    segment: contract.Segment,
    base_snapshot_attestation: Path,
) -> dict[str, Any]:
    _require_offline()
    raw_lora, _validated = _lora_config(root, plan, segment)
    base_document, snapshot = _read_base_snapshot_attestation(base_snapshot_attestation, segment)
    inventory = _meta_peft_inventory(snapshot, raw_lora)
    semantic = {
        "lora_config": raw_lora,
        "inventory": inventory,
    }
    return {
        "schema": SCHEMA,
        "condition": contract.CONDITION,
        "segment": {
            "global_segment_index": segment.global_index,
            "target": segment.target,
            "run_name": segment.run_name,
        },
        "plan": contract.file_identity(plan, label="LoRA fingerprint plan"),
        "base_snapshot_attestation": contract.file_identity(
            base_snapshot_attestation, label="LoRA fingerprint base-snapshot attestation"
        ),
        "base_snapshot": {
            "repo_id": contract.MODEL_REPO,
            "revision": contract.MODEL_REVISION,
            "resolved_snapshot_path": str(snapshot),
            "main_ref": base_document.get("main_ref"),
        },
        "lora_config": raw_lora,
        "inventory": inventory,
        "fingerprint_sha256": _canonical_sha256(semantic),
    }


def capture(
    root: Path,
    plan: Path,
    segment: contract.Segment,
    *,
    base_snapshot_attestation: Path,
    output: Path,
) -> dict[str, Any]:
    """Write an immutable inventory sidecar, or prove an existing one remains exact."""

    document = _document(
        root=root,
        plan=plan,
        segment=segment,
        base_snapshot_attestation=base_snapshot_attestation,
    )
    output = output.resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise contract.ContractError(f"LoRA fingerprint output escapes repository root: {output}") from exc
    status = contract._write_immutable_json(output, document, label="RMCT256 LoRA fingerprint attestation")
    return {"status": status, "path": str(output), "fingerprint_sha256": document["fingerprint_sha256"]}


def validate(
    root: Path,
    plan: Path,
    segment: contract.Segment,
    *,
    base_snapshot_attestation: Path,
    receipt: Path,
) -> dict[str, Any]:
    """Recompute and compare every effective PEFT name/shape before training."""

    receipt = receipt.resolve()
    recorded = contract._json_object(receipt, label="RMCT256 LoRA fingerprint attestation")
    expected = _document(
        root=root,
        plan=plan,
        segment=segment,
        base_snapshot_attestation=base_snapshot_attestation,
    )
    contract._expect(recorded, expected, label="RMCT256 LoRA fingerprint attestation")
    return {
        "status": "validated",
        "path": str(receipt),
        "fingerprint_sha256": str(recorded["fingerprint_sha256"]),
        "trainable_parameter_count": int(recorded["inventory"]["trainable_parameter_count"]),
        "trainable_parameter_numel": int(recorded["inventory"]["trainable_parameter_numel"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--repo-root", type=Path, required=True)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--segment-index", type=int, required=True)
        command.add_argument("--base-snapshot-attestation", type=Path, required=True)
        if name == "capture":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = contract._absolute_root(args.repo_root)
        plan = contract.plan_path(root, args.plan)
        segment = contract.segment_for_index(args.segment_index)
        base = args.base_snapshot_attestation.resolve()
        if args.command == "capture":
            result = capture(root, plan, segment, base_snapshot_attestation=base, output=args.output)
        else:
            result = validate(root, plan, segment, base_snapshot_attestation=base, receipt=args.receipt)
    except (contract.ContractError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
