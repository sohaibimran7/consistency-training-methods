"""Opt-in, verified four-GPU Grace CPU/memory placement for Gemma RMCT.

Discover within the full Slurm CPU allocation before narrowing the trainer.
Workers inherit the manifest, not the trainer's CPU mask. No GPU ordinals are
assumed to match NVML ordinals; identity is resolved through CUDA PCI bus IDs.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil


def cuda_pci_ids():
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuInit(0) != 0:
        raise RuntimeError("CUDA driver initialization failed")
    count = ctypes.c_int()
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
        raise RuntimeError("CUDA device enumeration failed")
    result = []
    for ordinal in range(count.value):
        device, bus = ctypes.c_int(), ctypes.create_string_buffer(32)
        if cuda.cuDeviceGet(ctypes.byref(device), ordinal) != 0 or cuda.cuDeviceGetPCIBusId(bus, len(bus), device) != 0:
            raise RuntimeError("CUDA PCI identity lookup failed")
        result.append(bus.value.decode())
    return result


def validate_bindings(records, allowed):
    if len(records) != 4 or [r["ordinal"] for r in records] != list(range(4)):
        raise ValueError("expected four ordered GPU bindings")
    for key in ("uuid", "pci", "node", "device_token"):
        if len({r[key] for r in records}) != 4:
            raise ValueError(f"GPU bindings must have unique {key}")
    used = set()
    for record in records:
        cpus = set(record["cpus"])
        if len(cpus) != 16 or not cpus <= set(allowed) or cpus & used:
            raise ValueError("each GPU requires 16 distinct allocated local CPUs")
        used.update(cpus)


def discover_bindings():
    import pynvml as nvml
    buses = cuda_pci_ids()
    tokens = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(buses) != 4 or len(tokens) != 4:
        raise RuntimeError("placement requires exactly four allocated visible GPUs")
    nvml.nvmlInit()
    try:
        allowed = os.sched_getaffinity(0)
        records = []
        for ordinal, bus in enumerate(buses):
            handle = nvml.nvmlDeviceGetHandleByPciBusId(bus)
            mask = nvml.nvmlDeviceGetCpuAffinity(handle, (os.cpu_count()+63)//64)
            cpus = sorted(c for c in allowed if mask[c//64] & (1 << (c % 64)))[:16]
            if len(cpus) != 16:
                raise RuntimeError(f"GPU {ordinal} lacks 16 allocated local CPUs")
            node_sets = [set(p.name for p in Path(f"/sys/devices/system/cpu/cpu{c}").glob("node[0-9]*")) for c in cpus]
            if len(node_sets[0]) != 1 or any(n != node_sets[0] for n in node_sets):
                raise RuntimeError("local CPU set spans unknown/multiple NUMA nodes")
            records.append(dict(ordinal=ordinal, pci=bus, uuid=nvml.nvmlDeviceGetUUID(handle),
                device_token=tokens[ordinal], cpus=cpus, node=int(next(iter(node_sets[0]))[4:])))
        validate_bindings(records, allowed)
        return dict(bindings=records, allocated_cpus=sorted(allowed))
    finally:
        nvml.nvmlShutdown()


def bind_memory(node):
    # Use syscall wrappers with return codes; numa_set_membind merely warns on
    # failure. Read back the policy so placement never silently degrades.
    lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
    maxnode = max(int(p.name[4:]) for p in Path("/sys/devices/system/node").glob("node[0-9]*")) + 1
    bits = ctypes.sizeof(ctypes.c_ulong)*8
    mask_type = ctypes.c_ulong * ((maxnode+bits-1)//bits)
    requested, actual = mask_type(), mask_type()
    requested[node//bits] = 1 << (node % bits)
    lib.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
    lib.get_mempolicy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
    if lib.set_mempolicy(2, requested, maxnode) != 0:
        raise OSError(ctypes.get_errno(), "set_mempolicy failed")
    mode = ctypes.c_int()
    if lib.get_mempolicy(ctypes.byref(mode), actual, maxnode, None, 0) != 0:
        raise OSError(ctypes.get_errno(), "get_mempolicy failed")
    if mode.value != 2 or list(actual) != list(requested):
        raise RuntimeError("NUMA memory binding verification failed")


def bind_worker(ordinal, device_token):
    document = json.loads(Path(os.environ["CTM_LOCAL_GPU_BINDINGS"]).read_text())
    records = document["bindings"]
    validate_bindings(records, document["allocated_cpus"])
    record = records[ordinal]
    if record["device_token"] != device_token or cuda_pci_ids() != [record["pci"]]:
        raise RuntimeError("rollout GPU identity differs from placement manifest")
    os.sched_setaffinity(0, record["cpus"])
    if set(os.sched_getaffinity(0)) != set(record["cpus"]):
        raise RuntimeError("worker CPU binding verification failed")
    bind_memory(record["node"])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a coordinator command is required")
    document = discover_bindings()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
    record = document["bindings"][0]
    os.environ["CTM_LOCAL_GPU_BINDINGS"] = str(args.output.resolve())
    numactl = shutil.which("numactl")
    if not numactl:
        raise RuntimeError("numactl is required")
    print(json.dumps({"gpu_local_placement": document}), flush=True)
    os.execv(numactl, [numactl, "--physcpubind="+",".join(map(str,record["cpus"])),
                      f"--membind={record['node']}", *command])


if __name__ == "__main__":
    main()
