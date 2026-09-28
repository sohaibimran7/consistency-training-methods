"""Run step 352 with the existing uncapped AITA protocol on 16 GPUs.

Four execution partitions subdivide each original logical shard. Both
perspectives stay together. Immutable partition logs are retained; four
explicitly derived assembly logs feed the existing full-corpus validator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from experiments.rmct_two_bias_eval.step352 import CHECKPOINT_SHA256, digest, write_once
from experiments.elephant_aita_ntaflip import prepare as data
from infra.isambard import run_qwen35_rmct_aita_ntaflip_16gpu as prior

PROJECT = Path(__file__).resolve().parents[2]


def partition_ids(pair_ids, partition):
    if isinstance(partition, bool) or partition not in range(4):
        raise ValueError("Execution partition must be 0..3")
    return list(pair_ids[partition::4])


def partition_task(manifest: str, shard_index: int, partition_index: int):
    from experiments.elephant_aita_ntaflip.tasks import aita_nta_flip_shard, task_specs
    task = aita_nta_flip_shard(manifest=manifest, shard_index=shard_index)
    ids = partition_ids(task_specs(manifest)[shard_index].pair_ids, partition_index)
    task.dataset = task.dataset.filter(lambda sample: sample.metadata["pair_id"] in set(ids))
    task.metadata = {**(task.metadata or {}), "execution_partition": {
        "index": partition_index, "count": 4, "pair_ids": ids,
        "note": "Execution-only subset of the unchanged four-shard benchmark; full logical-shard counts remain in benchmark metadata.",
    }}
    return task


def prepare(root: Path, source: Path, mcq_plan: Path):
    root.mkdir(parents=True, exist_ok=True)
    source_manifest = source / data.MANIFEST_FILENAME
    document, _, _ = data.load_frozen_pairs(source_manifest)
    if document["schema"] != data.MANIFEST_SCHEMA:
        raise ValueError("Only the uncapped historical manifest is permitted")
    mcq = json.loads(mcq_plan.read_text())
    checkpoint = Path(mcq["checkpoint"])
    if mcq["checkpoint_identity"]["optimizer_step"] != 352 or digest(checkpoint / "adapter_model.safetensors") != CHECKPOINT_SHA256:
        raise ValueError("Unexpected source checkpoint")
    for name in (data.MANIFEST_FILENAME, data.PAIR_ARTIFACT_FILENAME):
        prior._copy_immutable_file(source / name, root / "input" / name, label="AITA source")
    files = set(prior.CRITICAL_SOURCES) | {
        "experiments/elephant_aita_ntaflip/step352.py",
        "experiments/rmct_two_bias_eval/step352.py",
        "infra/isambard/run_rmct_step352_aita.sbatch",
        "infra/isambard/run_rmct_step352_aita_worker.sh",
    }
    write_once(root / "plan.json", {
        "condition": "step352", "checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256,
        "adapter_config_sha256": digest(checkpoint / "adapter_config.json"),
        "mcq_checkpoint_plan": {"path": str(mcq_plan), "sha256": digest(mcq_plan)},
        "manifest_sha256": digest(source_manifest), "model_snapshot": prior._snapshot_identity(),
        "model_args": prior.HF_LOCAL_MODEL_ARGS, "generation_config": prior.RUNTIME_GENERATION_CONFIG,
        "no_token_cap_policy": data.NO_TOKEN_CAP_POLICY, "thinking_policy": prior.QWEN_THINKING_POLICY,
        "expected_packages": prior.EVALUATOR_PACKAGE_VERSIONS,
        "source_sha256": {name: digest(PROJECT / name) for name in sorted(files)},
        "workers": [{"rank": shard * 4 + part, "shard_index": shard, "partition_index": part,
                    "pair_ids": partition_ids(entry["pair_ids"], part)}
                    for shard, entry in enumerate(document["shards"]) for part in range(4)],
        "topology_note": "16 execution partitions versus historical 4; seed and sampling config unchanged, random draw sequence need not match historical runs.",
    })


def checked(root: Path):
    plan = json.loads((root / "plan.json").read_text())
    if digest(Path(plan["checkpoint"]) / "adapter_model.safetensors") != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint weights changed")
    if digest(Path(plan["checkpoint"]) / "adapter_config.json") != plan["adapter_config_sha256"]:
        raise ValueError("Adapter config changed")
    if digest(root / "input" / data.MANIFEST_FILENAME) != plan["manifest_sha256"]:
        raise ValueError("AITA manifest changed")
    data.load_frozen_pairs(root / "input" / data.MANIFEST_FILENAME)
    for name, expected in plan["source_sha256"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"Source changed: {name}")
    data.assert_no_token_cap_mapping(plan["generation_config"], label="step352 AITA config")
    return plan


def probe(root: Path, rank: int):
    prior._validate_evaluator_environment(require_one_gpu=True)
    write_once(root / "gpu-probe" / f"rank-{rank:02d}.json", {
        "rank": rank, "host": os.uname().nodename, "uuid": prior._visible_gpu_uuid(),
        "visible": os.environ["CUDA_VISIBLE_DEVICES"], "job": os.environ["SLURM_JOB_ID"],
    })


def verify_topology(root: Path):
    records = [json.loads((root / "gpu-probe" / f"rank-{i:02d}.json").read_text()) for i in range(16)]
    if len({r["uuid"] for r in records}) != 16 or len({r["host"] for r in records}) != 4 or len({r["job"] for r in records}) != 1:
        raise ValueError("Expected 16 distinct GPUs across four nodes of one job")
    write_once(root / "gpu-topology.json", {"records": records})


def worker(root: Path, rank: int, smoke: bool = False):
    from inspect_ai.log import read_eval_log
    from experiments.elephant_aita_ntaflip.preflight import parse_final_answer_only
    plan = checked(root)
    prior._validate_evaluator_environment(require_one_gpu=True)
    if not smoke:
        verify_topology(root)
    entry = plan["workers"][rank]
    destination = root / ("smoke" if smoke else "partitions") / f"rank-{rank:02d}"
    receipt = destination / "receipt.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if digest(Path(saved["path"])) != saved["sha256"]:
            raise ValueError("Completed partition changed")
        return
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"rank-{rank:02d}-", dir=attempts))
    command = [sys.executable, str(PROJECT / "scripts/run_evals.py"),
               "--task-factory", "experiments.elephant_aita_ntaflip.step352:partition_task",
               "--local-checkpoint", plan["checkpoint"], "--base-model", str(prior.MODEL_SNAPSHOT),
               "--task-args", json.dumps({"manifest": str(root / "input" / data.MANIFEST_FILENAME),
                    "shard_index": entry["shard_index"], "partition_index": entry["partition_index"]}),
               "--model-args", json.dumps(plan["model_args"]),
               "--generation-config", json.dumps(plan["generation_config"]),
               "--log-dir", str(attempt), "--max-tasks", "1", "--yes"]
    if smoke:
        command += ["--limit", "2"]
    prior._assert_no_token_cap_command(command, label="step352 AITA command")
    write_once(attempt / "command.json", {"argv": command})
    subprocess.run(command, cwd=PROJECT, check=True)
    candidates = list(attempt.glob("*.eval"))
    if len(candidates) != 1:
        raise ValueError("Expected one partition EvalLog")
    log = read_eval_log(str(candidates[0]))
    ids = [f"{pair}::{perspective}" for pair in entry["pair_ids"] for perspective in ("flipped", "original")]
    if smoke:
        ids = ids[:2]
    actual = [sample.id for sample in log.samples or []]
    if log.status != "success" or len(actual) != len(ids) or set(actual) != set(ids):
        raise ValueError("Incomplete or incorrect AITA partition")
    data.assert_no_token_cap_mapping(log.eval.model_generate_config.model_dump(exclude_none=True), label="logged config")
    if smoke and any(parse_final_answer_only(sample.output.completion).label not in {"NTA", "YTA"} for sample in log.samples):
        raise ValueError("Direct verdict smoke did not yield two parsed finals")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / candidates[0].name
    prior._copy_immutable_file(candidates[0], target, label="AITA partition")
    write_once(receipt, {"path": str(target), "sha256": digest(target), "worker": entry, "n": len(ids)})


def finalize(root: Path):
    from inspect_ai.log import read_eval_log, write_eval_log
    from experiments.elephant_aita_ntaflip.preflight import preflight_raw_logs, write_preflight_report
    checked(root)
    verify_topology(root)
    sources = []
    for shard in range(4):
        logs, identities = [], []
        for rank in range(shard * 4, shard * 4 + 4):
            record = json.loads((root / "partitions" / f"rank-{rank:02d}" / "receipt.json").read_text())
            if digest(Path(record["path"])) != record["sha256"]:
                raise ValueError("Partition changed before assembly")
            logs.append(read_eval_log(record["path"]))
            identities.append(record)
        target = root / "assembled" / f"shard-{shard}.eval"
        receipt = target.with_suffix(".receipt.json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["sources"] != identities or digest(target) != saved["sha256"]:
                raise ValueError("Derived assembly changed")
        else:
            if target.exists():
                raise FileExistsError(target)
            joined = deepcopy(logs[0])
            joined.samples = [sample for log in logs for sample in log.samples]
            joined.eval.metadata = {key: value for key, value in joined.eval.metadata.items() if key != "execution_partition"}
            joined.eval.metadata["derived_partition_assembly"] = {"sources": identities, "raw_generations_unchanged": True}
            # The provenance explicitly identifies a derived complete shard,
            # not a claim that one original worker produced all its samples.
            joined.eval.dataset.samples = len(joined.samples)
            joined.results = None
            joined.reductions = None
            joined.stats.model_usage = {}
            joined.stats.started_at = min(log.stats.started_at for log in logs)
            joined.stats.completed_at = max(log.stats.completed_at for log in logs)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_eval_log(joined, str(target))
            write_once(receipt, {"sources": identities, "sha256": digest(target), "path": str(target), "derived": True})
        sources.extend(identities)
    manifest = root / "input" / data.MANIFEST_FILENAME
    expected = {"model": f"hf/{prior.MODEL_SNAPSHOT}", "model_args": prior.HF_LOCAL_MODEL_ARGS,
                "generation_config": prior.RUNTIME_GENERATION_CONFIG}
    report = preflight_raw_logs(root / "assembled", manifest, expected_runtime=expected)
    write_preflight_report(report, root / "step352-preflight.json", manifest=manifest,
                           raw_root=root / "assembled", expected_runtime=expected)
    write_once(root / "complete.json", {"condition": "step352", "sources": sources,
               "pairs": data.EXPECTED_PAIRS, "generations": 2 * data.EXPECTED_PAIRS,
               "preflight_sha256": digest(root / "step352-preflight.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "probe", "verify-topology", "worker", "smoke", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--mcq-plan", type=Path)
    parser.add_argument("--rank", type=int, choices=range(16), default=0)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root, args.source, args.mcq_plan)
    elif args.mode == "probe":
        probe(args.root, args.rank)
    elif args.mode == "verify-topology":
        verify_topology(args.root)
    elif args.mode == "finalize":
        finalize(args.root)
    else:
        worker(args.root, args.rank, args.mode == "smoke")


if __name__ == "__main__":
    main()
