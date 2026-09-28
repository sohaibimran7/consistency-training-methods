"""Fail closed before Gemma weights load unless all 16 GPU bindings are unique.

CUDA_VISIBLE_DEVICES alone is not proof: each process can see one logical
device while multiple processes address the same GPU. Each worker records the
UUID seen through CUDA, then validates the whole job-step's mapping. The
timeout bounds startup coordination only; it is never a generation limit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import tempfile
import time


SCHEMA = "ctm-gemma-16gpu-binding-v1"
WORLD_SIZE = 16


def validate_bindings(records: list[dict]) -> dict:
    if len(records) != WORLD_SIZE:
        raise ValueError("expected exactly 16 GPU binding records")
    if {record.get("rank") for record in records} != set(range(WORLD_SIZE)):
        raise ValueError("GPU binding ranks must be exactly 0 through 15")
    nodes: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("schema") != SCHEMA or record.get("visible_device_count") != 1:
            raise ValueError("every worker must attest exactly one CUDA-visible device")
        if not record.get("gpu_uuid") or not record.get("hostname"):
            raise ValueError("GPU UUID and hostname must be present")
        nodes[record["hostname"]].append(record)
    if len({(r.get("job_id"), r.get("step_id")) for r in records}) != 1:
        raise ValueError("GPU binding records must belong to one Slurm job step")
    if len(nodes) != 4 or any(len(items) != 4 for items in nodes.values()):
        raise ValueError("expected four workers on each of four nodes")
    for items in nodes.values():
        if {item.get("local_rank") for item in items} != set(range(4)):
            raise ValueError("each node must contain local ranks 0 through 3")
    duplicates = [uuid for uuid, count in Counter(r["gpu_uuid"] for r in records).items() if count != 1]
    if duplicates:
        raise ValueError(f"multiple workers share a GPU UUID: {duplicates}")
    return {"schema": SCHEMA, "status": "passed", "gpu_count": WORLD_SIZE,
            "node_count": 4, "bindings": sorted(records, key=lambda item: item["rank"])}


def _write_exclusive_json(path: Path, value: dict) -> None:
    """Publish a complete record atomically, refusing any existing target."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".binding-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()  # Only this process's just-created temporary file.


def local_cuda_identity() -> dict:
    """Ask CUDA, rather than NVML's potentially node-wide device listing."""
    import torch

    count = torch.cuda.device_count()
    if count != 1:
        raise ValueError(f"worker sees {count} CUDA devices instead of one")
    properties = torch.cuda.get_device_properties(0)
    uuid = str(getattr(properties, "uuid", ""))
    if not uuid or uuid == "None":
        raise ValueError("pinned Torch runtime did not report a CUDA GPU UUID")
    return {"gpu_uuid": uuid, "gpu_name": properties.name, "visible_device_count": count}


def attest(campaign_root: Path, *, timeout_seconds: float = 180.0) -> dict:
    if timeout_seconds <= 0:
        raise ValueError("startup timeout must be positive")
    if not campaign_root.is_absolute() or campaign_root.is_symlink() or not campaign_root.is_dir():
        raise ValueError("campaign root must be an existing regular absolute directory")
    env = os.environ
    job_id, step_id = env.get("SLURM_JOB_ID", ""), env.get("SLURM_STEP_ID", "")
    if not re.fullmatch(r"\d+", job_id) or not re.fullmatch(r"\d+", step_id):
        raise ValueError("binding attestation requires a numeric Slurm job and step ID")
    rank, local_rank = int(env["SLURM_PROCID"]), int(env["SLURM_LOCALID"])
    if rank not in range(WORLD_SIZE) or local_rank not in range(4):
        raise ValueError("unexpected Slurm global/local rank")
    if int(env.get("SLURM_NTASKS", "0")) != WORLD_SIZE:
        raise ValueError("binding attestation requires exactly 16 Slurm tasks")

    record = {"schema": SCHEMA, "job_id": job_id, "step_id": step_id,
              "rank": rank, "local_rank": local_rank, "hostname": os.uname().nodename,
              **local_cuda_identity(),
              "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", "")}
    parent = campaign_root / "gpu-bindings"
    parent.mkdir(exist_ok=True)
    if parent.is_symlink():
        raise ValueError("GPU binding directory must not be linked")
    directory = parent / f"job-{job_id}-step-{step_id}"
    directory.mkdir(exist_ok=True)
    if directory.is_symlink():
        raise ValueError("GPU binding job-step directory must not be linked")
    _write_exclusive_json(directory / f"rank-{rank:02d}.json", record)
    deadline = time.monotonic() + timeout_seconds
    paths = [directory / f"rank-{index:02d}.json" for index in range(WORLD_SIZE)]
    while not all(path.is_file() and not path.is_symlink() for path in paths):
        if time.monotonic() >= deadline:
            missing = [path.name for path in paths if not path.is_file() or path.is_symlink()]
            raise TimeoutError(f"GPU binding startup barrier incomplete: {missing}")
        time.sleep(0.5)
    result = validate_bindings([json.loads(path.read_text()) for path in paths])
    if rank == 0:
        _write_exclusive_json(directory / "attestation.json", result)
    print(json.dumps({"rank": rank, "gpu_binding_attestation": "passed", "receipt_dir": str(directory)}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--campaign-root", type=Path)
    mode.add_argument("--probe-single", action="store_true", help="Verify pinned-runtime CUDA UUID support on one GPU, without generation")
    parser.add_argument("--startup-timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if args.probe_single:
        print(json.dumps({"schema": SCHEMA, "single_gpu_probe": local_cuda_identity()}), flush=True)
    else:
        attest(args.campaign_root, timeout_seconds=args.startup_timeout_seconds)


if __name__ == "__main__":
    main()
