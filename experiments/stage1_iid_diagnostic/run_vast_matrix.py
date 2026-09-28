"""Run the nine-checkpoint Stage 1 IID diagnostic across assigned local GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.5-9B"
TASK_FACTORY = "experiments.stage1_iid_diagnostic.tasks:diagnostic_tasks"
TASK_COUNT = 4
SPLITS = ("train_eval", "heldout_in_domain")

# The order is operational: fill eight GPUs with trained conditions first, then
# put the untrained reference on the first GPU that becomes free.
CHECKPOINTS: dict[str, str | None] = {
    "rmct": "rmct_paper_vast_dense_qwen3_5_9b_stage1_rate-matching-lr-1e-4",
    "rmct-control": "rmct_paper_vast_dense_qwen3_5_9b_stage1_rate-matching-control-lr-1e-4",
    "bct": "rmct_paper_vast_dense_qwen3_5_9b_stage1_bias-augmented-consistency-lr-1e-4",
    "bct-control": "rmct_paper_vast_dense_qwen3_5_9b_stage1_bias-augmented-consistency-control-lr-1e-4",
    "act": "rmct_paper_vast_dense_qwen3_5_9b_stage1_act-lr-1e-4",
    "attct": "rmct_paper_vast_dense_qwen3_5_9b_stage1_attct-lr-1e-4",
    "mlpct": "rmct_paper_vast_dense_qwen3_5_9b_stage1_mlpct-lr-1e-4",
    "opct": "rmct_paper_vast_dense_qwen3_5_9b_stage1_opct-lr-1e-4",
    "untrained": None,
}

MODEL_ARGS = {
    "provider": "vllm",
    "gpu_memory_utilization": 0.9,
    "language_model_only": True,
    "max_model_len": 32768,
    "max_num_seqs": 256,
}
NATIVE_MODEL_ARGS = {name: value for name, value in MODEL_ARGS.items() if name != "provider"}
GENERATION_CONFIG = {
    "extra_body": {"top_k": 20},
    "max_tokens": 20480,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _successful_task_indices(
    log_dir: Path,
    *,
    task_args: dict[str, object],
    checkpoint: Path | None,
) -> set[int]:
    """Return task positions already proven successful in this condition dir."""

    if not log_dir.is_dir():
        return set()
    from inspect_ai.log import list_eval_logs, read_eval_log

    completed: set[int] = set()
    for info in list_eval_logs(str(log_dir), formats=["eval"], recursive=False):
        try:
            log = read_eval_log(info, header_only=True)
        except Exception:
            continue
        metadata = log.eval.metadata or {}
        indices = metadata.get("task_indices")
        expected_model = f"vllm/{BASE_MODEL}" if checkpoint is None else None
        identity_matches = metadata.get("model") == expected_model if checkpoint is None else metadata.get("checkpoint") == str(checkpoint)
        if (
            log.status == "success"
            and metadata.get("task_factory") == TASK_FACTORY
            and metadata.get("task_args") == task_args
            and metadata.get("generation_config") == GENERATION_CONFIG
            and metadata.get("task_count") == TASK_COUNT
            and identity_matches
            and isinstance(indices, list)
            and len(indices) == 1
            and isinstance(indices[0], int)
            and 1 <= indices[0] <= TASK_COUNT
        ):
            completed.add(indices[0])
    return completed


def _checkpoint_path(checkpoint_root: Path, condition: str) -> Path | None:
    name = CHECKPOINTS[condition]
    if name is None:
        return None
    path = checkpoint_root / name
    manifest = path / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"{condition}: missing checkpoint manifest: {manifest}")
    adapter_config_path = path / "adapter_config.json"
    adapter_weights = path / "adapter_model.safetensors"
    if not adapter_config_path.is_file() or not adapter_weights.is_file():
        raise FileNotFoundError(f"{condition}: incomplete LoRA checkpoint: {path}")
    adapter_config = json.loads(adapter_config_path.read_text())
    if adapter_config.get("base_model_name_or_path") != BASE_MODEL:
        raise ValueError(f"{condition}: checkpoint base model is not {BASE_MODEL}: {path}")
    return path


def _split_command(
    *,
    repo_root: Path,
    manifest: Path,
    checkpoint_root: Path,
    log_root: Path,
    condition: str,
    split: str,
) -> list[str] | None:
    if split not in SPLITS:
        raise ValueError(f"unsupported diagnostic split: {split}")
    log_dir = log_root / condition / split
    log_dir.mkdir(parents=True, exist_ok=True)
    task_args = {
        "include_bias_acknowledged": False,
        "manifest": str(manifest),
        "split": split,
        "unbiased_log": str(log_dir),
    }
    checkpoint = _checkpoint_path(checkpoint_root, condition)
    missing = sorted(
        set(range(1, TASK_COUNT + 1))
        - _successful_task_indices(
            log_dir,
            task_args=task_args,
            checkpoint=checkpoint,
        )
    )
    if not missing:
        return None

    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_evals.py"),
        "--task-factory",
        TASK_FACTORY,
        "--task-args",
        _json(task_args),
        "--generation-config",
        _json(GENERATION_CONFIG),
        "--model-args",
        _json(NATIVE_MODEL_ARGS if checkpoint is None else MODEL_ARGS),
        "--log-dir",
        str(log_dir),
        "--max-tasks",
        "1",
        "--isolate-tasks",
        "--persistent-vllm-server",
        "--yes",
    ]
    if checkpoint is None:
        command.extend(("--model", f"vllm/{BASE_MODEL}"))
    else:
        command.extend(("--local-checkpoint", str(checkpoint), "--base-model", BASE_MODEL))
    for task_index in missing:
        command.extend(("--task-index", str(task_index)))
    return command


def _worker_command(
    *,
    manifest: Path,
    checkpoint_root: Path,
    log_root: Path,
    condition: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--manifest",
        str(manifest),
        "--checkpoint-root",
        str(checkpoint_root),
        "--log-root",
        str(log_root),
        "--worker-condition",
        condition,
    ]


def _open_runner_log(log_root: Path, condition: str):
    """Open a condition runner log after creating its isolated directory."""
    runner_log = log_root / condition / "runner.log"
    runner_log.parent.mkdir(parents=True, exist_ok=True)
    return runner_log, runner_log.open("ab")


def _write_provenance(
    *,
    output: Path,
    manifest: Path,
    checkpoint_root: Path,
    conditions: list[str],
    gpus: list[int],
) -> None:
    document = {
        "schema_version": 1,
        "base_model": BASE_MODEL,
        "task_factory": TASK_FACTORY,
        "task_count_per_condition": TASK_COUNT,
        "splits": list(SPLITS),
        "conditions": conditions,
        "gpus": gpus,
        "diagnostic_manifest": str(manifest),
        "diagnostic_manifest_sha256": _sha256(manifest),
        "generation_config": GENERATION_CONFIG,
        "model_args": MODEL_ARGS,
        "checkpoints": {
            condition: (
                None
                if CHECKPOINTS[condition] is None
                else {
                    "path": str(_checkpoint_path(checkpoint_root, condition)),
                    "manifest_sha256": _sha256(_checkpoint_path(checkpoint_root, condition) / "manifest.json"),
                    "adapter_config_sha256": _sha256(_checkpoint_path(checkpoint_root, condition) / "adapter_config.json"),
                    "adapter_model_sha256": _sha256(_checkpoint_path(checkpoint_root, condition) / "adapter_model.safetensors"),
                }
            )
            for condition in conditions
        },
    }
    payload = (_json(document) + "\n").encode()
    if output.exists():
        if output.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace differing provenance: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)


def _parse_csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("GPU list must contain unique non-negative integers")
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", type=_parse_csv_ints)
    parser.add_argument("--conditions", nargs="+", choices=list(CHECKPOINTS), default=list(CHECKPOINTS))
    parser.add_argument("--worker-condition", choices=list(CHECKPOINTS), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    manifest = args.manifest.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    log_root = args.log_root.resolve()
    if not manifest.is_file():
        parser.error(f"manifest does not exist: {manifest}")
    if not checkpoint_root.is_dir():
        parser.error(f"checkpoint root does not exist: {checkpoint_root}")
    if len(args.conditions) != len(set(args.conditions)):
        parser.error("conditions must be unique")

    if args.worker_condition is not None:
        for split in SPLITS:
            command = _split_command(
                repo_root=repo_root,
                manifest=manifest,
                checkpoint_root=checkpoint_root,
                log_root=log_root,
                condition=args.worker_condition,
                split=split,
            )
            if command is None:
                print(
                    f"{args.worker_condition}/{split}: all {TASK_COUNT} tasks already successful; skipping",
                    flush=True,
                )
                continue
            print(f"{args.worker_condition}/{split}: starting", flush=True)
            completed = subprocess.run(command, cwd=repo_root, check=False)
            if completed.returncode:
                raise SystemExit(f"{args.worker_condition}/{split}: failed with status {completed.returncode}")
        return

    _write_provenance(
        output=log_root / "run-provenance.json",
        manifest=manifest,
        checkpoint_root=checkpoint_root,
        conditions=args.conditions,
        gpus=args.gpus,
    )

    available = deque(args.gpus)
    pending = deque(args.conditions)
    running: dict[subprocess.Popen[bytes], tuple[str, int, object]] = {}
    failures: list[tuple[str, int]] = []
    while pending or running:
        while pending and available:
            condition = pending.popleft()
            gpu = available.popleft()
            command = _worker_command(
                manifest=manifest,
                checkpoint_root=checkpoint_root,
                log_root=log_root,
                condition=condition,
            )
            runner_log, handle = _open_runner_log(log_root, condition)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment.pop("VLLM_BASE_URL", None)
            environment.pop("VLLM_API_KEY", None)
            print(f"{condition}: launching on physical GPU {gpu}; log={runner_log}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[process] = (condition, gpu, handle)

        if not running:
            continue
        time.sleep(5.0)
        for process, (condition, gpu, handle) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            del running[process]
            available.append(gpu)
            if return_code:
                failures.append((condition, return_code))
                print(f"{condition}: FAILED with status {return_code} on GPU {gpu}", flush=True)
            else:
                print(f"{condition}: complete on GPU {gpu}", flush=True)

    if failures:
        details = ", ".join(f"{condition}={status}" for condition, status in failures)
        raise SystemExit(f"{len(failures)} condition(s) failed: {details}")
    print("All requested Stage 1 IID diagnostic conditions completed.", flush=True)


if __name__ == "__main__":
    main()
