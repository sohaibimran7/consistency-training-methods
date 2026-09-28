"""Score-only recovery of omitted generic-runner selection metadata.

The frozen partition factory returns one Task, but the generic runner emits
task_count/task_indices only when explicit --task-index selection is used.
Validate that single-task execution from every raw partition, then annotate
new derived assemblies. Never change a raw or previously assembled log.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from experiments.elephant_aita_ntaflip import step352 as run


def recover(root: Path):
    from inspect_ai.log import read_eval_log, write_eval_log
    from experiments.elephant_aita_ntaflip.preflight import preflight_raw_logs, write_preflight_report
    plan = run.checked(root)
    run.verify_topology(root)
    all_sources = []
    output = root / "score-recovery-r1"
    for shard in range(4):
        source = root / "assembled" / f"shard-{shard}.eval"
        assembly_receipt = json.loads(source.with_suffix(".receipt.json").read_text())
        if run.digest(source) != assembly_receipt["sha256"]:
            raise ValueError("Original assembly hash mismatch")
        partition_samples = {}
        for record in assembly_receipt["sources"]:
            if run.digest(Path(record["path"])) != record["sha256"]:
                raise ValueError("Original partition hash mismatch")
            raw = read_eval_log(record["path"])
            entry = plan["workers"][record["worker"]["rank"]]
            if entry != record["worker"] or entry["shard_index"] != shard:
                raise ValueError("Partition receipt differs from frozen execution plan")
            metadata = raw.eval.metadata
            if metadata.get("task_factory") != "experiments.elephant_aita_ntaflip.step352:partition_task":
                raise ValueError("Wrong single-Task factory")
            if metadata.get("task_count") is not None or metadata.get("task_indices") is not None:
                raise ValueError("Recovery is restricted to the exact omitted-field case")
            if metadata.get("checkpoint") != plan["checkpoint"] or raw.status != "success":
                raise ValueError("Wrong checkpoint or incomplete partition")
            if metadata["execution_partition"]["pair_ids"] != entry["pair_ids"]:
                raise ValueError("Wrong raw partition membership")
            expected_ids = {f"{pair}::{perspective}" for pair in entry["pair_ids"] for perspective in ("original", "flipped")}
            if len(raw.samples or []) != len(expected_ids) or {s.id for s in raw.samples} != expected_ids:
                raise ValueError("Raw partition does not exactly cover its planned pairs")
            for sample in raw.samples:
                if sample.id in partition_samples:
                    raise ValueError("Duplicate sample across partitions")
                partition_samples[sample.id] = sample
        old = read_eval_log(str(source))
        if len(old.samples) != len(partition_samples) or any(sample != partition_samples[sample.id] for sample in old.samples):
            raise ValueError("Assembly changed original generation data")
        derived = deepcopy(old)
        derived.eval.metadata.update({"task_count": 1, "task_indices": [1], "derived_selection_metadata": {
            "original_assembly": {"path": str(source), "sha256": run.digest(source)},
            "basis": "All four source commands use the hash-bound factory that returns exactly one Task; original fields were omitted, not a multi-task run.",
            "raw_samples_unchanged": True,
        }})
        target = output / "assemblies" / f"shard-{shard}.eval"
        receipt = target.with_suffix(".receipt.json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["sources"] != assembly_receipt["sources"] or run.digest(target) != saved["sha256"]:
                raise ValueError("Recovery output changed")
        else:
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_eval_log(derived, str(target))
            run.write_once(receipt, {"sources": assembly_receipt["sources"], "sha256": run.digest(target), "path": str(target)})
        all_sources.extend(assembly_receipt["sources"])
    manifest = root / "input" / run.data.MANIFEST_FILENAME
    expected = {"model": f"hf/{run.prior.MODEL_SNAPSHOT}", "model_args": run.prior.HF_LOCAL_MODEL_ARGS,
                "generation_config": run.prior.RUNTIME_GENERATION_CONFIG}
    report = preflight_raw_logs(output / "assemblies", manifest, expected_runtime=expected)
    write_preflight_report(report, output / "step352-preflight.json", manifest=manifest,
                           raw_root=output / "assemblies", expected_runtime=expected)
    run.write_once(output / "complete.json", {"condition": "step352", "pairs": 1591, "generations": 3182,
                   "source_partitions": all_sources, "preflight_sha256": run.digest(output / "step352-preflight.json"),
                   "score_only_recovery_source_sha256": run.digest(Path(__file__))})
    print(json.dumps(report["metrics"]["primary_metric"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    recover(parser.parse_args().root)
