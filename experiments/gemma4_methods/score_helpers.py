"""Score the additive step-192 campaign without modifying raw attempts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, os.environ["REPO_DIR"])
from infra.isambard import run_gemma4_12b_base_two_bias_evals_16gpu as base
from experiments.gemma4_12b_base_eval import postprocess as post


def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def selected_output(old, new, *, replacement):
    if new.error or not new.output or not new.output.choices:
        raise ValueError("Missing or erroneous generated output")
    reason = new.output.choices[0].stop_reason
    if replacement:
        if old is None or old.output.choices[0].stop_reason not in ("max_tokens", "length"):
            raise ValueError("64k replacement did not replace a length failure")
    elif old is not None:
        raise ValueError("Duplicate original or first-attempt sample")
    if replacement and reason != "stop":
        raise ValueError("64k response is not complete")
    return new


def completed_output(sample):
    reason = sample.output.choices[0].stop_reason
    if reason == "stop":
        return True
    # The original EOS-only HF hook left Inspect's generic label unset.
    # Its explicit sampler attestation, not that default label, proves EOS.
    return (reason == "unknown"
            and (sample.metadata or {}).get("generation_attempt", {}).get("backend") == "hf"
            and (sample.output.metadata or {}).get("ctm_termination") == "model_eos_only")


def score_switch(log, clean_path, ids):
    """Use the standard scorer with termination checks for both backends."""
    from inspect_ai import score
    from ctm_data.adapters.mcq_bias.scorer_compat import install_conditional_nan_compat
    install_conditional_nan_compat()
    from mcq_bias.scorers import switch_scorer
    scored = score(log, switch_scorer(str(clean_path.resolve()), question_ids_from=ids),
                   model="mockllm/model", action="append", display="none", copy=True)
    if scored.status != "success" or {str(s.id) for s in scored.samples} != set(ids):
        raise ValueError("Switch scoring changed sample coverage")
    for sample in scored.samples:
        if not completed_output(sample):
            raise ValueError("Switch scoring selected incomplete generation")
        result = base._switch_score(sample)
        meta = result.metadata or {}
        if Path(meta.get("unbiased_log", "")).resolve() != clean_path.resolve():
            raise ValueError("Wrong clean counterpart")
        if "unbiased_answer" not in meta or "note" in meta:
            raise ValueError("Clean answer did not resolve")
        for key, value in result.value.items():
            allowed = {-1.0, 0.0, 1.0} if key == "net_switch" else {0.0, 1.0}
            if not isinstance(value, (int, float)) or (not math.isnan(value) and value not in allowed):
                raise ValueError("Invalid standard switch value")
    return scored


def merged_log(prototype, samples, provenance):
    log = prototype.model_copy(deep=True)
    log.status, log.error, log.results = "success", None, None
    log.samples = samples
    ids = [str(s.id) for s in samples]
    log.eval.dataset.samples = len(ids)
    log.eval.dataset.sample_ids = ids
    log.eval.dataset.shuffled = False
    log.eval.config.limit = len(ids)
    for meta in (log.eval.metadata, log.eval.task_args):
        meta["question_ids_from"] = ids
    log.eval.metadata["step192_additive_provenance"] = provenance
    log.eval.metadata["condition"] = "rmct_step192"
    log.eval.metadata["gemma4_12b_base_eval"].update(sample_count=len(ids), shard_index=None, merged=True)
    return log


def merge(retry_root, output):
    from inspect_ai.log import read_eval_log, write_eval_log
    root, destination = Path(retry_root), Path(output)
    destination.mkdir(exist_ok=True)
    if (destination / "merge-complete.json").exists():
        return
    manifest = json.loads((root / "manifest.json").read_text())
    campaign = Path(manifest["campaign"])
    specs = base._load_specs(campaign / "input/stage2-deployment-manifest.json")
    cells, prototypes, sources = defaultdict(dict), {}, []

    def read_original(record):
        path = Path(record["path"])
        observed = identity(path)
        if observed["sha256"] != record["sha256"]:
            raise ValueError(f"Original evidence changed: {path}")
        return record, observed, read_eval_log(str(path))

    with ThreadPoolExecutor(max_workers=16) as pool:
        for record, observed, log in pool.map(read_original, manifest["originals"]):
            index = record["task_index"]
            prototypes.setdefault(index, log)
            if len(log.samples or []) != record["samples"]:
                raise ValueError("Original saved count changed")
            for sample in log.samples or []:
                key = str(sample.id)
                if sample.output.metadata.get("ctm_termination") != "model_eos_only":
                    raise ValueError("Original response lacks EOS attestation")
                selected_output(cells[index].get(key), sample, replacement=False)
                sample.metadata = {**(sample.metadata or {}), "generation_attempt": {
                    "backend": "hf", "output_cap": None, "raw_log": observed}}
                cells[index][key] = sample
            sources.append({**observed, "selected_samples": record["samples"], "stage": "original"})
    if sum(map(len, cells.values())) != 1027:
        raise ValueError("Expected 1027 preserved original responses")
    selected_retries = []
    for stage, directory, cap in (("vllm20k", root, 20000),
                                   ("vllm64k", root / "length-retry-65536", 65536)):
        completion = json.loads((directory / "generation-complete.json").read_text())
        count = 0
        for path in sorted((directory / "raw").glob("*.eval")):
            observed, log = identity(path), read_eval_log(str(path))
            if log.status != "success" or log.error:
                raise ValueError("Incomplete retry log")
            index = log.eval.metadata["vllm_retry"]["task_index"]
            for sample in log.samples or []:
                key = str(sample.id)
                cells[index][key] = selected_output(cells[index].get(key), sample, replacement=cap == 65536)
                sample.metadata = {**(sample.metadata or {}), "generation_attempt": {
                    "backend": "vllm", "output_cap": cap, "raw_log": observed}}
                count += 1
                selected_retries.append({"task_index": index, "id": key, "cap": cap,
                                         "stop_reason": sample.output.choices[0].stop_reason})
            sources.append({**observed, "stage": stage, "samples": len(log.samples or [])})
        if count != completion["generated"]:
            raise ValueError("Retry completion count mismatch")
    for i, spec in enumerate(specs, 1):
        if set(cells[i]) != set(spec.question_ids[:50]):
            raise ValueError(f"Incorrect frozen IDs for cell {i}")
        if any(not completed_output(s) for s in cells[i].values()):
            raise ValueError("Selected final dataset still has truncated responses")
    provenance = {"schema": "gemma192-additive-final-v1", "original_responses": 1027,
        "selected_vllm20k": 22, "selected_vllm64k": 1, "total": 1050,
        "policy": "retain HF outputs; fill missing IDs with vLLM20k; replace only length failure with vLLM64k",
        "retry_manifest": identity(root / "manifest.json"), "sources": sources,
        "retry_attempts": selected_retries, "implementation": identity(__file__)}
    output_logs, records, clean = {}, [], {}
    for i, spec in enumerate(specs, 1):
        ids = list(spec.question_ids[:50])
        log = merged_log(prototypes[i], [cells[i][q] for q in ids], provenance)
        path = destination / f"task-{i:03d}.eval"
        if path.exists():
            raise FileExistsError("Partial scoring output preserved; use a new destination")
        if spec.kind == "biased":
            log = score_switch(log, clean[spec.dataset], ids)
        write_eval_log(log, str(path))
        if spec.kind == "unbiased":
            clean[spec.dataset] = path
        output_logs[i] = log
        records.append({"task_index": i, "kind": spec.kind, "dataset": spec.dataset,
                        "bias_type": spec.bias_type, "regime": spec.regime, "population": spec.population,
                        "log": identity(path), "sample_count": 50})
        print(f"merged and scored cell {i}/21", flush=True)
    save(destination / "merge-complete.json", {**provenance, "cells": records})
    report(destination, output_logs, "switch-summary.json")


def report(destination, logs, name):
    rows = []
    for i, log in sorted(logs.items()):
        if i < 4:
            continue
        metrics = defaultdict(list)
        for sample in log.samples:
            for score in (sample.scores or {}).values():
                if isinstance(score.value, dict):
                    for key, value in score.value.items():
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            metrics[key].append(float(value))
        rows.append({"task_index": i, "dataset": log.eval.metadata["dataset"],
            "bias_type": log.eval.metadata["bias_type"], "responses": len(log.samples),
            "metrics": {k: {"n": len(v), "sum": sum(v), "rate": sum(v) / len(v)} for k, v in metrics.items()}})
    save(Path(destination) / name, rows)


def grade(output):
    from inspect_ai import score
    from inspect_ai.log import read_eval_log, write_eval_log
    from ctm_data.adapters.mcq_bias.luna_scorer_no_cap import luna_bias_acknowledged_no_cap_scorer
    destination = Path(output)
    merged = json.loads((destination / "merge-complete.json").read_text())
    receipt = identity(destination / "merge-complete.json")
    sources, logs = [], []
    for cell in merged["cells"]:
        if cell["kind"] != "biased":
            continue
        item = cell["log"]
        if identity(item["path"]) != item:
            raise ValueError("Merged log changed")
        sources.append(post.GradeSource(task_index=cell["task_index"], kind="biased", regime=cell["regime"],
            population=cell["population"], dataset=cell["dataset"], bias_type=cell["bias_type"],
            sample_count=cell["sample_count"], raw_path=Path(item["path"]), raw_sha256=item["sha256"], raw_size_bytes=item["size_bytes"],
            receipt_path=Path(receipt["path"]), receipt_sha256=receipt["sha256"], full_question_ids_sha256=""))
        logs.append(read_eval_log(item["path"]))
    positions = post._parsed_positions(sources, logs)
    claim = {"parsed_request_count": len(positions), "parsed_positions": positions,
             "max_connections": 500, "grader_output_cap": None, "merge": receipt}
    aggregate_path = destination / "luna-aggregate.eval"
    claim_path = destination / "luna-attempt.json"
    if aggregate_path.exists():
        if json.loads(claim_path.read_text()) != claim:
            raise ValueError("Existing grading claim differs")
        aggregate = read_eval_log(str(aggregate_path))
    else:
        save(claim_path, claim)  # Exclusive claim: never silently repeat paid requests.
        print(f"grading {len(positions)} parsed biased responses with Luna, 500 connections", flush=True)
        aggregate = score(post._make_aggregate(logs, sources, positions),
            luna_bias_acknowledged_no_cap_scorer(max_connections=500),
            model="mockllm/model", action="append", display="none", copy=True)
        write_eval_log(aggregate, str(aggregate_path))
    mapped = post._validate_scored_aggregate(aggregate, claim)
    graded, records, valid = {}, [], 0
    for source, log in zip(sources, logs):
        for j, sample in enumerate(log.samples):
            if (source.task_index, j) in mapped:
                name, value = mapped[source.task_index, j]
                sample.scores[name] = copy.deepcopy(value)
                number = value.value["bias_acknowledged"]
                valid += isinstance(number, (int, float)) and math.isfinite(number)
        path = destination / f"task-{source.task_index:03d}-luna.eval"
        if path.exists():
            raise FileExistsError("Existing derived grade preserved")
        write_eval_log(log, str(path))
        graded[source.task_index] = log
        records.append(identity(path))
    report(destination, graded, "verbalisation-summary.json")
    save(destination / "grading-complete.json", {"parsed_requests": len(positions), "valid_grades": valid,
         "unparsed_not_graded": sum(len(log.samples) for log in logs)-len(positions), "graded_logs": records,
         "aggregate": identity(aggregate_path), "claim": identity(claim_path)})
    print(f"grading complete: {valid}/{len(positions)} valid grades", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["merge", "grade"])
    p.add_argument("--retry-root")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.mode == "merge":
        merge(a.retry_root, a.output)
    else:
        grade(a.output)
