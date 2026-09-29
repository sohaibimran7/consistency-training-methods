"""Step-352 run recipe using the existing frozen tasks, evaluator and scorers.

Generation omits scoring so clean and biased cells can share a GPU wave.
After all generation receipts exist, the ordinary Inspect scorers run on
copies, clean first. Neither previous checkpoints nor reference logs change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from infra.isambard.run_qwen35_rmct_checkpoint_two_bias_evals_16gpu import (
    GENERATION_CONFIG, MODEL_SNAPSHOT, TASK_SAMPLE_COUNTS, VLLM_MODEL_ARGS,
)

PROJECT = Path(__file__).resolve().parents[2]
CONDITION = "rmct_step352"
CHECKPOINT_SHA256 = "214c97e0b1b4af027495797a3cddd7f3e1fcb1a600a0968904aa7d125429f921"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_once(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"Refusing to change existing receipt: {path}")
        return
    with path.open("x") as handle:
        handle.write(payload)


def tasks(manifest: str, unbiased_log: str, generation_only: bool = True):
    from experiments.stage2_ood_hle.tasks import ood_tasks
    result = ood_tasks(manifest=manifest, unbiased_log=unbiased_log,
                       prompt_style="none", include_bias_acknowledged=False)
    for task in result:
        if generation_only:
            task.scorer = None
    return result


def prepare(root: Path, training: Path, substrate: Path, parity_data: Path) -> None:
    from experiments.rmct_convergence import patience
    from experiments.rmct_convergence_r5_patience import plan
    from infra.isambard.verify_rmct_convergence_r4_recovery_production_ready import _strict_checkpoint_identity
    from experiments.rmct_two_bias_eval.deployment import materialize_deployment_manifest
    from experiments.stage2_ood_hle.tasks import ood_task_specs
    amendment = patience.verify_amendment(training, training / patience.AMENDMENT_RELATIVE)
    checkpoint = plan.final_checkpoint_path(training, 21)
    identity = _strict_checkpoint_identity(training, checkpoint, expected_segment_index=21, label="step352 eval")
    if identity["optimizer_step"] != 352 or digest(checkpoint / "adapter_model.safetensors") != CHECKPOINT_SHA256:
        raise ValueError("Unexpected step-352 checkpoint")
    receipt_dir = checkpoint.parent.parent / "patience-decisions"
    receipts = list(receipt_dir.glob("patience-decision-*.json"))
    if len(receipts) != 1:
        raise ValueError("Ambiguous final patience receipt")
    decision = patience.verify_decision_receipt(training, receipts[0])
    if decision["decision"] != "converged":
        raise ValueError("Source has not met the agreed convergence rule")
    root.mkdir(parents=True, exist_ok=True)
    materialize_deployment_manifest(substrate / "manifest.json", substrate, root / "manifest.json")
    specs = ood_task_specs(root / "manifest.json")
    assert len(specs) == 21 and sum(TASK_SAMPLE_COUNTS.values()) == 1400
    write_once(root / "plan.json", {
        "schema": "rmct-step352-comparison-v1", "condition": CONDITION,
        "checkpoint": str(checkpoint), "checkpoint_identity": identity,
        "generation_config": GENERATION_CONFIG, "model_args": VLLM_MODEL_ARGS,
        "parity_data": str(parity_data), "parity_data_sha256": digest(parity_data),
        "manifest_sha256": digest(root / "manifest.json"),
        "source_sha256": {str(p.relative_to(PROJECT)): digest(p) for d in ("ctm", "ctm_data", "experiments", "infra", "scripts") for p in sorted((PROJECT / d).rglob("*.py"))},
        "training_policy": amendment["policy"], "decision_receipt": str(receipts[0]),
        "decision_sha256": digest(receipts[0]),
        "cells": [{"task_index": i, "kind": s.kind, "dataset": s.dataset,
                   "bias_type": s.bias_type, "n": TASK_SAMPLE_COUNTS[i],
                   "question_ids": list(s.question_ids[:TASK_SAMPLE_COUNTS[i]])}
                  for i, s in enumerate(specs, 1)],
        "reference_policy": "Reuse step16 50/50/100 and step176 100/100/100; paired tests use comparison-specific QID overlap; no reference downsampling.",
    })


def checked_plan(root: Path) -> dict:
    plan = json.loads((root / "plan.json").read_text())
    if digest(Path(plan["checkpoint"]) / "adapter_model.safetensors") != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint weights changed")
    if digest(root / "manifest.json") != plan["manifest_sha256"]:
        raise ValueError("Deployment manifest changed")
    for relative, expected in plan["source_sha256"].items():
        if digest(PROJECT / relative) != expected:
            raise ValueError(f"Evaluation source changed: {relative}")
    return plan


def parity(root: Path) -> None:
    from experiments.act_repair_gate.vllm_compat_adapter import make_compat_adapter, attest_compat_adapter
    from infra.isambard.run_qwen35_rmct_checkpoint_two_bias_evals_16gpu import _parity_command, _validate_strict_parity_report
    from experiments.rmct_two_bias_eval.contract import VLLM_SAMPLER_RUNTIME
    plan = checked_plan(root)
    adapter = root / "adapter"
    tokens = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(tokens) != 4 or len(set(tokens)) != 4:
        raise ValueError("Parity needs exactly four Slurm-visible GPUs")
    if not adapter.exists():
        make_compat_adapter(plan["checkpoint"], adapter)
    sampler = {**VLLM_SAMPLER_RUNTIME, "vllm_device_tokens": tokens}
    report = root / "parity" / "report.json"
    if not report.exists():
        command = _parity_command(python=sys.executable, adapter=adapter,
                                  raw={"path": plan["checkpoint"]}, data=plan["parity_data"],
                                  attempt=root / "parity", sampler=sampler)
        subprocess.run(command, cwd=PROJECT, check=True)
    _validate_strict_parity_report(report, adapter=adapter, sampler=sampler)
    if not (adapter / "vllm-parity-attestation.json").exists():
        attest_compat_adapter(adapter, report)
    write_once(root / "parity-complete.json", {"report": str(report), "sha256": digest(report), "adapter_sha256": digest(adapter / "adapter_model.safetensors")})


def worker(root: Path, index: int, smoke: bool = False) -> None:
    from inspect_ai.log import read_eval_log
    plan = checked_plan(root)
    if not (root / "parity-complete.json").is_file():
        raise ValueError("Missing completed strict parity proof")
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
        raise ValueError("Worker must have exactly one Slurm-visible GPU")
    cell = plan["cells"][index - 1]
    count = 1 if smoke else cell["n"]
    destination = root / ("smoke" if smoke else "raw") / f"task-{index:03d}"
    receipt = destination / "receipt.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if digest(Path(record["path"])) != record["sha256"]:
            raise ValueError("Completed generation changed")
        return
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"task-{index:03d}-", dir=attempts))
    command = [sys.executable, str(PROJECT / "scripts/run_evals.py"),
        "--task-factory", "experiments.rmct_two_bias_eval.step352:tasks",
        "--local-checkpoint", str(root / "adapter"), "--base-model", str(MODEL_SNAPSHOT),
        "--task-args", json.dumps({"manifest": str(root / "manifest.json"), "unbiased_log": str(root / "scored"), "generation_only": True}),
        "--model-args", json.dumps(VLLM_MODEL_ARGS), "--generation-config", json.dumps(GENERATION_CONFIG),
        "--log-dir", str(attempt), "--limit", str(count), "--max-tasks", "1",
        "--isolate-tasks", "--persistent-vllm-server", "--task-index", str(index), "--yes"]
    write_once(attempt / "command.json", command)
    subprocess.run(command, cwd=PROJECT, check=True)
    logs = list(attempt.rglob("*.eval"))
    if len(logs) != 1:
        raise ValueError(f"Expected one generation log, got {logs}")
    log = read_eval_log(str(logs[0]))
    validate_generation(log, cell["question_ids"][:count])
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / logs[0].name
    if target.exists():
        raise FileExistsError(target)
    shutil.copyfile(logs[0], target)
    write_once(receipt, {"task_index": index, "n": count, "path": str(target), "sha256": digest(target), "generation_only": True})


def validate_generation(log, expected_ids: list[str]) -> None:
    """Inspect stores concurrently completed samples in completion order."""
    actual = [str(sample.id) for sample in log.samples or []]
    if log.status != "success" or len(actual) != len(expected_ids):
        raise ValueError("Incomplete generation log")
    if len(set(actual)) != len(actual) or set(actual) != set(expected_ids):
        raise ValueError("Generation QIDs differ from frozen selection")


def score_all(root: Path) -> None:
    from inspect_ai import score
    from inspect_ai.log import read_eval_log, write_eval_log
    plan = checked_plan(root)
    scoring_tasks = tasks(str(root / "manifest.json"), str(root / "scored"), generation_only=False)
    sources = []
    for cell in plan["cells"]:
        receipt = root / "raw" / f"task-{cell['task_index']:03d}" / "receipt.json"
        record = json.loads(receipt.read_text())
        if digest(Path(record["path"])) != record["sha256"] or record["n"] != cell["n"]:
            raise ValueError("Incomplete/changed source generation")
        sources.append(record)
    outputs = []
    # Frozen task order is the three clean references, then all biased cells.
    for record, task in zip(sources, scoring_tasks, strict=True):
        target = root / "scored" / f"task-{record['task_index']:03d}.eval"
        receipt = target.with_suffix(".receipt.json")
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            if prior["source"] != record or digest(target) != prior["sha256"]:
                raise ValueError("Scored output changed")
        else:
            if target.exists():
                raise FileExistsError(target)
            raw = read_eval_log(record["path"])
            scored = score(raw, task.scorer, model="mockllm/model", action="replace", display="none", copy=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_eval_log(scored, str(target))
            write_once(receipt, {"source": record, "path": str(target), "sha256": digest(target)})
        outputs.append(json.loads(receipt.read_text()))
    write_once(root / "generation-and-switch-complete.json", {"condition": CONDITION, "n": 1400, "sources": outputs})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "parity", "worker", "score"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--substrate", type=Path)
    parser.add_argument("--parity-data", type=Path)
    parser.add_argument("--task-index", type=int, choices=range(1, 22))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root, args.training, args.substrate, args.parity_data)
    elif args.mode == "parity":
        parity(args.root)
    elif args.mode == "worker":
        worker(args.root, args.task_index, args.smoke)
    else:
        score_all(args.root)


if __name__ == "__main__":
    main()
