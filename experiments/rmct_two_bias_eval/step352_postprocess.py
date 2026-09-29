"""Grade and publish the authorized 352-versus-16/176 comparison.

Use the existing postprocess runtime for `grade` (mcq_bias installed), and
the worktree's plotting runtime for `plot`. Source EvalLogs are never edited.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import tempfile

from experiments.rmct_two_bias_eval.step352 import digest, write_once

PROJECT = Path(__file__).resolve().parents[2]
NEW = "rmct_step352"
REFERENCES = ("rmct_step16", "rmct_step176")


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": digest(path), "size_bytes": path.stat().st_size}


def verified(path: Path, record: dict) -> Path:
    if digest(path) != record["sha256"]:
        raise ValueError(f"Source hash mismatch: {path}")
    return path


def new_sources(bundle: Path) -> list[dict]:
    complete = json.loads((bundle / "generation-and-switch-complete.json").read_text())
    if complete["condition"] != NEW or complete["n"] != 1400 or len(complete["sources"]) != 21:
        raise ValueError("Missing complete 21-cell generation and switch-scoring receipt")
    sources = []
    for index, record in enumerate(complete["sources"], 1):
        if record["source"]["task_index"] != index:
            raise ValueError("Completion receipt task order changed")
        path = verified(bundle / "scored" / f"task-{index:03d}.eval", record)
        if index > 3:
            sources.append({**identity(path), "task_index": index})
    return sources


async def grade(bundle: Path, output: Path) -> None:
    from dotenv import load_dotenv
    from inspect_ai import score_async
    from inspect_ai.log import read_eval_log, write_eval_log
    from inspect_ai.model import GenerateConfig, get_model
    from ctm_data.adapters.mcq_bias.luna_scorer_no_cap import (
        GRADER_MODEL, _provider_attestation, luna_bias_acknowledged_no_cap_scorer,
    )
    # Existing configured research workflow; do not copy or expose the key.
    load_dotenv("/Users/work/consistency-training-methods/.env")
    config = GenerateConfig(reasoning_effort="low", max_connections=500)
    attestation = _provider_attestation(get_model(GRADER_MODEL, config=config), config)
    sources = new_sources(bundle)
    scorer = luna_bias_acknowledged_no_cap_scorer(max_connections=500)
    write_once(output / "luna-policy.json", attestation)

    async def one(source: dict) -> dict:
        target = output / "luna" / f"task-{source['task_index']:03d}.eval"
        receipt = target.with_suffix(".receipt.json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["source"] != source:
                raise ValueError("Luna receipt source changed")
            verified(target, saved)
            return saved
        if target.exists():
            raise FileExistsError(f"Unreceipted Luna output: {target}")
        original = read_eval_log(source["path"])
        scored = await score_async(original, [scorer], model="mockllm/model", action="append", display="none", copy=True)
        if len(scored.samples or []) != len(original.samples or []):
            raise ValueError("Grading changed sample count")
        for before, after in zip(original.samples, scored.samples, strict=True):
            if before.id != after.id or before.output != after.output:
                raise ValueError("Grading changed a generation")
            additions = set(after.scores or {}) - set(before.scores or {})
            if len(additions) != 1:
                raise ValueError("Expected exactly one appended Luna score")
            for key, value in (before.scores or {}).items():
                if after.scores[key] != value:
                    raise ValueError("Grading changed a prior score")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(scored, str(target))
        saved = {**identity(target), "source": source, "policy": attestation}
        write_once(receipt, saved)
        print(f"Luna complete task {source['task_index']}: {len(scored.samples)} samples", flush=True)
        return saved

    # All 18 cells share one model/provider concurrency pool, capped at 500
    # simultaneous requests in total, not 500 per cell. No response token cap.
    results = await asyncio.gather(*(one(source) for source in sources), return_exceptions=True)
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise RuntimeError(f"{len(errors)} Luna cells failed; successful receipts preserved") from errors[0]
    write_once(output / "luna-complete.json", {"sources": results, "policy": attestation})


def reference_sources(metric: str) -> tuple[dict, dict]:
    name = "switch-rate" if metric == "towards_bias_switch" else "bias-verbalisation"
    path = PROJECT / f"artifacts/rmct-step16-step176-standard-{name}-by-dataset-significance-key-r003-20260821/manifest.json"
    document = json.loads(path.read_text())
    records = document["source_inputs"]
    key = "receipt_selected_biased_logs" if metric == "towards_bias_switch" else "receipt_selected_biased_luna_logs"
    field = "local_input" if metric == "towards_bias_switch" else "luna_eval_log"
    selected = {REFERENCES[0]: [record[field] for record in records[REFERENCES[0]][key]],
                REFERENCES[1]: records[REFERENCES[1]]}
    for entries in selected.values():
        if len(entries) != 18:
            raise ValueError("Reference matrix is incomplete")
        for record in entries:
            verified(Path(record["path"]), record)
    return selected, identity(path)


def plot(bundle: Path, output: Path, metric: str) -> None:
    from inspect_ai.log import read_eval_log
    from ctm_data.adapters.mcq_bias.analysis import aggregate_logs, append_bias_group_summaries, append_binomial_wilson_intervals
    from ctm_data.adapters.mcq_bias.plot import render_publication_plot
    from experiments.rmct_two_bias_eval import checkpoint_publication as standard

    records, reference_manifest = reference_sources(metric)
    if metric == "bias_acknowledged":
        records[NEW] = json.loads((output / "luna-complete.json").read_text())["sources"]
    else:
        records[NEW] = new_sources(bundle)
    logs = {condition: [read_eval_log(str(verified(Path(record["path"]), record))) for record in entries]
            for condition, entries in records.items()}
    counts = {"logiqa": 50, "hellaswag": 50, "hle-text-mc": 100}
    selections = {condition: standard._condition_question_ids(values, condition=condition,
                  expected_counts={key: 100 for key in counts} if condition == REFERENCES[1] else counts)
                  for condition, values in logs.items()}
    for dataset in counts:
        if selections[NEW][dataset] != selections[REFERENCES[0]][dataset]:
            raise ValueError("Step352 and the receipt-selected step16 pool differ")
    # Normalize only the analysis task identity, retaining each full source pool.
    bars = {condition: [standard._filtered_log_view(log, ids=selections[condition][standard._dataset_from_log(log)],
                      condition=condition, selection_label="full own pool") for log in values]
            for condition, values in logs.items()}
    metadata = {condition: {"condition_label": f"RMCT step {condition.removeprefix('rmct_step')}", "method": "rate_matching"}
                for condition in bars}
    rows = []
    for population, datasets in standard.POPULATION_DATASETS.items():
        population_logs = {condition: [log for log in values if standard._dataset_from_log(log) in datasets]
                           for condition, values in bars.items()}
        aggregated = aggregate_logs(population_logs, metric=metric, stderr="binomial", variant="biased",
                     metadata={"model": "qwen3.5-9b", "population": population, "population_datasets": list(datasets)},
                     condition_metadata=metadata, expected_biases=standard.ALL_BIASES, expected_datasets=datasets)
        rows.extend(append_binomial_wilson_intervals(append_bias_group_summaries(aggregated, groups=standard.BIAS_GROUPS)))
    for row in rows:
        row["bias_status"] = ("seen" if row["bias_type"] in standard.SEEN_BIASES else
                              "held_out" if row["bias_type"] in standard.HELD_OUT_BIASES else "aggregate")
    tests = []
    for reference in REFERENCES:
        overlap = {dataset: selections[NEW][dataset] & selections[reference][dataset] for dataset in counts}
        pair = {condition: [standard._filtered_log_view(log, ids=overlap[standard._dataset_from_log(log)],
                           condition=condition, selection_label="paired QID intersection") for log in logs[condition]]
                for condition in (reference, NEW)}
        for row in rows:
            if row["condition"] != NEW:
                continue
            datasets = standard.POPULATION_DATASETS[row["population"]]
            biases = row.get("component_biases") or [row["bias_type"]]
            clustered = {condition: standard._cluster_counts(values, datasets=datasets, biases=biases, metric=metric)
                         for condition, values in pair.items()}
            result = standard._paired_label_swap(clustered[NEW], clustered[reference],
                     treatment_name=f"{NEW}_vs_{reference}", metric=metric, population=row["population"],
                     bias_type=row["bias_type"], permutations=10_000)
            result.update(reference=reference, population=row["population"], bias_type=row["bias_type"],
                          significance_baseline=reference, paired_ids={dataset: sorted(overlap[dataset]) for dataset in datasets})
            tests.append(result)
    if len(tests) != 36:
        raise ValueError("Expected 36 hypothesis tests per metric")
    running = 0.0
    for rank, test in enumerate(sorted(tests, key=lambda test: test["p_value_raw"])):
        running = min(1.0, max(running, (len(tests) - rank) * test["p_value_raw"]))
        test.update(p_value=running, p_value_holm=running, significance=standard._significance_marker(running), holm_family_size=36)
    for reference in REFERENCES:
        target = output / "plots" / f"{metric}-vs-{reference}"
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        spec = standard.publication_spec(metric=metric)
        spec.update(condition_order=[*REFERENCES, NEW], condition_labels={key: value["condition_label"] for key, value in metadata.items()},
                    condition_styles={REFERENCES[0]: {"color": "#6fa8dc"}, REFERENCES[1]: {"color": "#8cc39a"}, NEW: {"color": "#d99858"}})
        spec["significance_note"] = (
            f"Stars compare step 352 with {metadata[reference]['condition_label']}. "
            "Bars retain full available pools (step 16/352: 50+50 IID, 100 HLE; step 176: 100+100 IID, 100 HLE). "
            "Tests use exact paired question-ID overlap; two-sided whole-question label swaps, 10,000 permutations. "
            "Holm correction covers 36 cells across both checkpoint comparisons per metric. "
            "95% Wilson intervals; n denotes scored/eligible responses. * p<0.05; ** p<0.01; *** p<0.001."
        )
        if metric == "bias_acknowledged":
            spec["significance_note"] += " Historical Luna grades are reused; new grades use the same model pin/rubrics without the historical output cap."
        figure_rows = []
        for row in rows:
            value = {**row, "significance": ""}
            if row["condition"] == NEW:
                value.update(next(test for test in tests if test["reference"] == reference and
                             test["population"] == row["population"] and test["bias_type"] == row["bias_type"]))
            figure_rows.append(value)
        with tempfile.TemporaryDirectory(prefix=".plot-", dir=target.parent) as temporary:
            staged = Path(temporary) / "figure"
            staged.mkdir()
            write_once(staged / "chart-rows.json", figure_rows)
            write_once(staged / "chart-spec.json", spec)
            for extension in ("png", "svg"):
                render_publication_plot(figure_rows, spec, staged / f"{standard.OUTPUT_STEMS[metric]}.{extension}")
            write_once(staged / "manifest.json", {"metric": metric, "reference": reference, "source_inputs": records,
                       "reference_publication": reference_manifest, "all_36_tests": tests,
                       "full_pool_ids": {c: {d: sorted(ids) for d, ids in selection.items()} for c, selection in selections.items()},
                       "postprocess_source": identity(Path(__file__))})
            os.replace(staged, target)
        print(target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("grade", "plot"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("towards_bias_switch", "bias_acknowledged"), default="towards_bias_switch")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "postprocess.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "grade":
            asyncio.run(grade(args.bundle.resolve(), args.output.resolve()))
        else:
            plot(args.bundle.resolve(), args.output.resolve(), args.metric)


if __name__ == "__main__":
    main()
