#!/usr/bin/env python3
"""Export an audited Vast-vs-Tinker training metrics report from W&B.

The run list is deliberately explicit.  This prevents failed predecessors or
later proposed configs from being silently mixed with the checkpoints used by
the current evaluation figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


@dataclass(frozen=True)
class RunSpec:
    platform: str
    model: str
    method: str
    control: bool
    learning_rate: str
    project: str
    run_id: str
    forward_microbatch: str
    note: str = ""


ENTITY = "sohaibimran"


VAST_RUNS = [
    RunSpec("Vast", "Qwen3.5-9B", "ACT", False, "1e-4", "rmct_paper_vast_dense_models_stage1_supervised_recovery_none_20260801", "791a00cr", "<=1 datum; <=20,480 tokens"),
    RunSpec("Vast", "Qwen3.5-9B", "AttCT", False, "1e-4", "rmct_paper_vast_dense_models_stage1_supervised_recovery_none_20260801", "ovhfsbd5", "<=8 datums; <=20,480 tokens", "Executed checkpoint differs from the later 4,096-update proposed YAML."),
    RunSpec("Vast", "Qwen3.5-9B", "MLPCT", False, "1e-4", "rmct_paper_vast_dense_models_stage1_supervised_recovery_none_20260801", "9luq24r0", "<=8 datums; <=20,480 tokens", "Executed checkpoint differs from the later 512-update proposed YAML."),
    RunSpec("Vast", "Qwen3.5-9B", "BCT", False, "1e-4", "rmct_paper_vast_dense_models_stage1_supervised_recovery_none_20260801", "rswyqrjx", "<=8 datums; <=20,480 tokens"),
    RunSpec("Vast", "Qwen3.5-9B", "BCT", True, "1e-4", "rmct_paper_vast_dense_models_stage1_supervised_recovery_none_20260801", "nv6sfiov", "<=8 datums; <=20,480 tokens"),
    RunSpec("Vast", "Qwen3.5-9B", "RMCT", False, "1e-4", "rmct_paper_isambard_phase2_rng_repair_4gpu_20260803", "d5wj3a3i", "dynamic <=8 datums; <=20,480 tokens", "Project name says Isambard because that topology config was reused; execution/evaluation lineage is the Vast run."),
    RunSpec("Vast", "Qwen3.5-9B", "RMCT", True, "1e-4", "rmct_paper_isambard_phase2_rng_repair_4gpu_20260803", "ectjp0kh", "dynamic <=8 datums; <=20,480 tokens", "One of 16 rollout batches was skipped as empty, so 15 optimizer updates were applied."),
    RunSpec("Vast", "Qwen3.5-9B", "OPCT", False, "1e-4", "rmct_paper_isambard_phase2_opct_rng_repair_4gpu_20260803", "44tne7qw", "dynamic <=8 datums; <=49,152 tokens", "Project name says Isambard because that topology config was reused; execution/evaluation lineage is the Vast run."),
]


TINKER_RUNS = [
    # Llama BCT
    RunSpec("Tinker", "Llama-3.1-8B-Instruct", "BCT", False, "1e-4", "bct_da_llama_lravg_lr1e4", "shzspmbh", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "Llama-3.1-8B-Instruct", "BCT", True, "1e-4", "bct_da_llama_lravg_lr1e4", "kk2hnfkc", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "Llama-3.1-8B-Instruct", "BCT", False, "2.86e-4", "bct_da_llama_lravg_lr2_86e4", "33nrs1mk", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "Llama-3.1-8B-Instruct", "BCT", True, "2.86e-4", "bct_da_llama_lravg_lr2_86e4", "h2sdc4jh", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "Llama-3.1-8B-Instruct", "BCT", False, "5e-4", "bct_da_llama_lravg_lr5e4", "xx43bkl1", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "Llama-3.1-8B-Instruct", "BCT", True, "5e-4", "bct_da_llama_lravg_lr5e4", "l8rgz1ln", "Tinker-managed; not exposed"),
    # Llama RMCT
    RunSpec("Tinker", "Llama-3.1-8B-Instruct", "RMCT", False, "1e-4", "rlct_da_aw0_llama_r128b4_lr1e4", "86uf52fe", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "Llama-3.1-8B-Instruct", "RMCT", True, "1e-4", "rlct_da_aw0_llama_r128b4_lr1e4", "y3i32atx", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "Llama-3.1-8B-Instruct", "RMCT", False, "2.86e-4", "rlct_da_aw0_llama_r128b4_lr2_86e4", "eihc4l4e", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "Llama-3.1-8B-Instruct", "RMCT", True, "2.86e-4", "rlct_da_aw0_llama_r128b4_lr2_86e4", "kscwc23k", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "Llama-3.1-8B-Instruct", "RMCT", False, "5e-4", "rlct_da_aw0_llama_r128b4_lr5e4", "bllzbxei", "Tinker-managed; not exposed", "Canonical completed retry; crashed predecessor 371n128v is excluded."),
    RunSpec("Tinker", "Llama-3.1-8B-Instruct", "RMCT", True, "5e-4", "rlct_da_aw0_llama_r128b4_lr5e4", "41f1fyth", "Tinker-managed; not exposed"),
    # GPT-OSS BCT
    RunSpec("Tinker", "GPT-OSS-20B", "BCT", False, "1e-4", "bct_da_20b_lravg_lr1e4", "e8zdrkds", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "GPT-OSS-20B", "BCT", True, "1e-4", "bct_da_20b_lravg_lr1e4", "rczc06j3", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "GPT-OSS-20B", "BCT", False, "2.86e-4", "bct_da_20b_lravg_lr2_86e4", "u3uewbb7", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "GPT-OSS-20B", "BCT", True, "2.86e-4", "bct_da_20b_lravg_lr2_86e4", "luy24fzs", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "GPT-OSS-20B", "BCT", False, "5e-4", "bct_da_20b_lravg_lr5e4", "trfy3bcc", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "GPT-OSS-20B", "BCT", True, "5e-4", "bct_da_20b_lravg_lr5e4", "2di1z0xb", "Tinker-managed; not exposed"),
    # GPT-OSS RMCT
    RunSpec("Tinker", "GPT-OSS-20B", "RMCT", False, "1e-4", "rlct_da_aw0_20b_r128b4_lr1e4", "tn0lqf7w", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "GPT-OSS-20B", "RMCT", True, "1e-4", "rlct_da_aw0_20b_r128b4_lr1e4", "s86q5410", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "GPT-OSS-20B", "RMCT", False, "2.86e-4", "rlct_da_aw0_20b_r128b4_lr2_86e4", "49kq86bb", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "GPT-OSS-20B", "RMCT", True, "2.86e-4", "rlct_da_aw0_20b_r128b4_lr2_86e4", "8rsaef9g", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "GPT-OSS-20B", "RMCT", False, "5e-4", "rlct_da_aw0_20b_r128b4_lr5e4", "nlzxwscy", "Tinker-managed; not exposed"),
    RunSpec("Tinker", "GPT-OSS-20B", "RMCT", True, "5e-4", "rlct_da_aw0_20b_r128b4_lr5e4", "vbq883ag", "Tinker-managed; not exposed"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vast-analysis", type=Path)
    parser.add_argument("--tinker-analysis", type=Path)
    return parser.parse_args()


def load_wandb_env(path: Path) -> None:
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"}:
            os.environ[key] = value.strip().strip('"').strip("'")


def nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def first_last(values: Iterable[float]) -> tuple[float | None, float | None]:
    seq = list(values)
    return (seq[0], seq[-1]) if seq else (None, None)


def metric_group(metric: str) -> str:
    lower = metric.lower()
    if "loss" in lower or "nll" in lower or "cross_entropy" in lower:
        return "loss"
    if "reward" in lower or "advantage" in lower or "consistency_gap" in lower:
        return "reward"
    if "kl" in lower or "ratio" in lower or "clipped" in lower:
        return "policy_optimization"
    if "entropy" in lower:
        return "entropy"
    if any(part in lower for part in ("p_hat", "p_ref", "rate_var")):
        return "rate_estimation"
    if any(part in lower for part in ("parse", "failure", "skipped")):
        return "reliability"
    if any(part in lower for part in ("response_length", "scored_tokens", "n_rollouts", "count")):
        return "sampling"
    if lower.endswith("/lr") or "optimizer_step" in lower or lower.endswith("/epoch"):
        return "optimizer"
    return "other"


def system_metric_group(metric: str) -> str:
    lower = metric.lower()
    if ".gpu" in lower or "gpu." in lower or "cuda" in lower:
        return "gpu"
    if "cpu" in lower:
        return "cpu"
    if "memory" in lower or "swap" in lower:
        return "memory"
    if "disk" in lower:
        return "disk"
    if "network" in lower:
        return "network"
    if "power" in lower or "temp" in lower or "fan" in lower:
        return "power_thermal"
    if "proc" in lower:
        return "process"
    return "other"


def run_batch_fields(spec: RunSpec, config: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    batch = config.get("batch_size", nested(config, "loop", "batch_size", default=1))
    grad_acc = config.get(
        "gradient_accumulation_steps",
        nested(config, "loop", "gradient_accumulation_steps", default=1),
    )
    batch = int(batch or 1)
    grad_acc = int(grad_acc or 1)
    n_units = config.get("n_samples", config.get("n_datapoints"))
    epochs = config.get("n_epochs", nested(config, "loop", "n_epochs", default=1))
    scheduled = int(config.get("total_steps", math.ceil((n_units or 0) / batch / grad_acc) * (epochs or 1)))
    completed = config.get("completed_optimizer_steps")
    if completed is None:
        logged_steps = [row.get("train/optimizer_step") for row in history if finite_number(row.get("train/optimizer_step"))]
        if logged_steps and max(logged_steps) > 0:
            completed = int(max(logged_steps))
        elif spec.method == "RMCT":
            skipped = sum(int(row.get("train/skipped_empty_batch", 0) or 0) for row in history)
            completed = max(0, len(history) - skipped)
        else:
            completed = len(history)
    if spec.method == "RMCT":
        ref = int(nested(config, "reference_rate", "n_rollouts", default=0) or 0)
        cued = int(nested(config, "training", "n_rollouts_for_rate", default=0) or 0)
        consistency = int(nested(config, "training", "n_rollouts_for_consistency", default=0) or 0)
        anchor = int(nested(config, "training", "n_rollouts_for_anchor", default=0) or 0)
        group = f"ref={ref}; cued/rate={cued}; consistency subset={consistency}; anchor subset={anchor}"
        generated_per_update = batch * (ref + cued)
    elif spec.method == "OPCT":
        rollouts = int(config.get("rollouts_per_prompt", nested(config, "generation", "rollouts_per_prompt", default=0)) or 0)
        group = f"{rollouts} policy responses/prompt"
        generated_per_update = batch * grad_acc * rollouts
    else:
        group = "n/a"
        generated_per_update = 0
    presentations = int(n_units * epochs) if n_units is not None and epochs is not None else None
    return {
        "configured_batch": batch,
        "gradient_accumulation": grad_acc,
        "effective_optimizer_batch": batch * grad_acc,
        "forward_microbatch_cap": spec.forward_microbatch,
        "unique_training_units": n_units,
        "epochs": epochs,
        "training_unit_presentations": presentations,
        "scheduled_loop_batches": scheduled,
        "applied_optimizer_updates": int(completed),
        "rollout_group": group,
        "generated_rollouts_per_update": generated_per_update,
        "scheduled_generated_rollouts": generated_per_update * scheduled,
        "max_new_tokens": nested(config, "generation", "max_new_tokens"),
    }


def optimizer_bin(history: list[dict[str, Any]], grad_acc: int) -> list[dict[str, Any]]:
    numeric_by_update: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ordinal, row in enumerate(history, start=1):
        update = math.ceil(ordinal / max(1, grad_acc))
        for key, value in row.items():
            if key.startswith("_") or not finite_number(value):
                continue
            numeric_by_update[update][key].append(float(value))
    output: list[dict[str, Any]] = []
    for update in sorted(numeric_by_update):
        out = {"optimizer_update_bin": update}
        for key, values in numeric_by_update[update].items():
            out[key] = fmean(values)
        output.append(out)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        if abs(value) >= 1000 or (value and abs(value) < 1e-3):
            return f"{value:.3g}"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(fmt(cell).replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def key_metric_summary(binned: list[dict[str, Any]], metric: str) -> tuple[Any, Any, Any, Any]:
    vals = [float(row[metric]) for row in binned if finite_number(row.get(metric))]
    if not vals:
        return None, None, None, None
    return vals[0], vals[-1], min(vals), max(vals)


def make_figures(out_dir: Path, records: list[dict[str, Any]], binned_by_run: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib.pyplot as plt

    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)

    def plot_panel(ax: Any, selected: list[dict[str, Any]], metric_for: Any, title: str, ylabel: str) -> None:
        for rec in selected:
            metric = metric_for(rec)
            rows = binned_by_run[rec["run_id"]]
            points = [(row["optimizer_update_bin"], row[metric]) for row in rows if finite_number(row.get(metric))]
            if not points:
                continue
            label = rec["method"] + (" control" if rec["control"] else "")
            ax.plot([p[0] for p in points], [p[1] for p in points], label=label, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("optimizer update")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    vast = [r for r in records if r["platform"] == "Vast"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    plot_panel(axes[0], [r for r in vast if r["method"] in {"ACT", "AttCT", "MLPCT"}], lambda _: "train/loss", "Vast consistency objectives", "logged loss")
    plot_panel(axes[1], [r for r in vast if r["method"] == "BCT"], lambda _: "train/nll", "Vast BCT", "token NLL")
    fig.tight_layout()
    fig.savefig(figures / "vast-supervised-losses.png", dpi=180)
    plt.close(fig)

    rmct = [r for r in vast if r["method"] == "RMCT"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title in zip(
        axes.flat,
        ["train/consistency_gap_1", "train/kl_policy_base", "train/p_hat_1", "train/train/loss"],
        ["Consistency gap", "Policy-to-base KL", "Biased rate estimate", "PPO loss (Vast mean schema)"],
    ):
        plot_panel(ax, rmct, lambda _r, m=metric: m, title, metric)
    fig.tight_layout()
    fig.savefig(figures / "vast-rmct-loss-reward-kl.png", dpi=180)
    plt.close(fig)

    opct = [r for r in vast if r["method"] == "OPCT"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title in zip(
        axes.flat,
        ["train/loss", "train/teacher_kl", "train/student_entropy", "train/avg_response_length"],
        ["OPCT loss", "Teacher KL", "Student entropy", "Response length"],
    ):
        plot_panel(ax, opct, lambda _r, m=metric: m, title, metric)
    fig.tight_layout()
    fig.savefig(figures / "vast-opct-training.png", dpi=180)
    plt.close(fig)

    tinker_bct = [r for r in records if r["platform"] == "Tinker" and r["method"] == "BCT"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, model in zip(axes, ["Llama-3.1-8B-Instruct", "GPT-OSS-20B"]):
        for rec in [r for r in tinker_bct if r["model"] == model]:
            rows = binned_by_run[rec["run_id"]]
            pts = [(row["optimizer_update_bin"], row["train/nll"]) for row in rows if finite_number(row.get("train/nll"))]
            style = "--" if rec["control"] else "-"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=f"{rec['learning_rate']}" + (" ctrl" if rec["control"] else ""))
        ax.set_title(model)
        ax.set_xlabel("optimizer update")
        ax.set_ylabel("token NLL")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "tinker-bct-nll-by-learning-rate.png", dpi=180)
    plt.close(fig)

    tinker_rmct = [r for r in records if r["platform"] == "Tinker" and r["method"] == "RMCT"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, model, metric in [
        (axes[0, 0], "Llama-3.1-8B-Instruct", "train/consistency_gap_1"),
        (axes[0, 1], "GPT-OSS-20B", "train/consistency_gap_1"),
        (axes[1, 0], "Llama-3.1-8B-Instruct", "train/kl_policy_base"),
        (axes[1, 1], "GPT-OSS-20B", "train/kl_policy_base"),
    ]:
        for rec in [r for r in tinker_rmct if r["model"] == model]:
            pts = [(row["optimizer_update_bin"], row[metric]) for row in binned_by_run[rec["run_id"]] if finite_number(row.get(metric))]
            style = "--" if rec["control"] else "-"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=f"{rec['learning_rate']}" + (" ctrl" if rec["control"] else ""))
        ax.set_title(f"{model}: {metric.split('/')[-1]}")
        ax.set_xlabel("optimizer update")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "tinker-rmct-reward-kl-by-learning-rate.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    load_wandb_env(args.env_file)
    import wandb

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=90)

    records: list[dict[str, Any]] = []
    metric_history: list[dict[str, Any]] = []
    metric_summaries: list[dict[str, Any]] = []
    optimizer_history: list[dict[str, Any]] = []
    system_history: list[dict[str, Any]] = []
    system_metric_summaries: list[dict[str, Any]] = []
    binned_by_run: dict[str, list[dict[str, Any]]] = {}
    run_payloads: list[dict[str, Any]] = []

    for spec in [*VAST_RUNS, *TINKER_RUNS]:
        run = api.run(f"{ENTITY}/{spec.project}/{spec.run_id}")
        history = list(run.scan_history(page_size=10_000))
        # Machine telemetry lives in W&B's separate system stream. All audited
        # runs are below this 10k-row request cap; retain the returned numeric
        # samples in wide form so hundreds of GPU/CPU keys do not explode into
        # an unnecessarily large long-form export.
        system_rows = list(run.history(samples=10_000, pandas=False, stream="system") or [])
        config = dict(run.config)
        batch = run_batch_fields(spec, config, history)
        record = {
            **asdict(spec),
            **batch,
            "run_name": run.name,
            "state": run.state,
            "run_url": run.url,
            "project_url": f"https://wandb.ai/{ENTITY}/{spec.project}/workspace",
            "history_rows": len(history),
            "system_history_rows": len(system_rows),
            "runtime_seconds": max((float(row.get("_runtime", 0) or 0) for row in history), default=0),
        }
        records.append(record)
        binned = optimizer_bin(history, batch["gradient_accumulation"] if spec.platform == "Vast" else 1)
        binned_by_run[spec.run_id] = binned
        for row in binned:
            optimizer_history.append({**{k: record[k] for k in ("platform", "model", "method", "control", "learning_rate", "project", "run_id", "run_name", "run_url")}, **row})

        by_metric: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row_index, row in enumerate(history, start=1):
            context = {
                "platform": spec.platform,
                "model": spec.model,
                "method": spec.method,
                "control": spec.control,
                "learning_rate": spec.learning_rate,
                "project": spec.project,
                "run_id": spec.run_id,
                "run_name": run.name,
                "run_url": run.url,
                "history_row": row_index,
                "wandb_step": row.get("_step"),
                "logged_optimizer_step": row.get("train/optimizer_step"),
                "timestamp": row.get("_timestamp"),
                "runtime_seconds": row.get("_runtime"),
            }
            for key, value in row.items():
                if key.startswith("_") or not finite_number(value):
                    continue
                metric_history.append({**context, "metric_group": metric_group(key), "metric": key, "value": float(value)})
                by_metric[key].append((row_index, float(value)))
        for key, pairs in sorted(by_metric.items()):
            values = [value for _, value in pairs]
            metric_summaries.append({
                "platform": spec.platform,
                "model": spec.model,
                "method": spec.method,
                "control": spec.control,
                "learning_rate": spec.learning_rate,
                "project": spec.project,
                "run_id": spec.run_id,
                "run_name": run.name,
                "run_url": run.url,
                "metric_group": metric_group(key),
                "metric": key,
                "count": len(values),
                "first": values[0],
                "last": values[-1],
                "min": min(values),
                "max": max(values),
                "mean": fmean(values),
            })

        system_by_metric: dict[str, list[float]] = defaultdict(list)
        for row_index, row in enumerate(system_rows, start=1):
            context = {
                "platform": spec.platform,
                "model": spec.model,
                "method": spec.method,
                "control": spec.control,
                "learning_rate": spec.learning_rate,
                "project": spec.project,
                "run_id": spec.run_id,
                "run_name": run.name,
                "run_url": run.url,
                "system_history_row": row_index,
                "timestamp": row.get("_timestamp"),
                "runtime_seconds": row.get("_runtime"),
            }
            numeric = {
                key: float(value)
                for key, value in row.items()
                if not key.startswith("_") and finite_number(value)
            }
            if numeric:
                system_history.append({**context, **numeric})
            for key, value in numeric.items():
                system_by_metric[key].append(value)
        for key, values in sorted(system_by_metric.items()):
            system_metric_summaries.append({
                "platform": spec.platform,
                "model": spec.model,
                "method": spec.method,
                "control": spec.control,
                "learning_rate": spec.learning_rate,
                "project": spec.project,
                "run_id": spec.run_id,
                "run_name": run.name,
                "run_url": run.url,
                "metric_group": system_metric_group(key),
                "metric": key,
                "count": len(values),
                "first": values[0],
                "last": values[-1],
                "min": min(values),
                "max": max(values),
                "mean": fmean(values),
            })
        run_payloads.append({"spec": asdict(spec), "url": run.url, "state": run.state, "name": run.name, "config": config, "summary": dict(run.summary)})

    write_csv(out / "runs.csv", records)
    write_csv(out / "metric-history-long.csv", metric_history)
    write_csv(out / "metric-summary.csv", metric_summaries)
    write_csv(out / "optimizer-binned-history.csv", optimizer_history)
    write_csv(out / "system-metric-history-wide.csv", system_history)
    write_csv(out / "system-metric-summary.csv", system_metric_summaries)
    (out / "wandb-run-snapshots.json").write_text(json.dumps(run_payloads, indent=2, sort_keys=True, default=str) + "\n")
    make_figures(out, records, binned_by_run)

    # Behavioral-result extracts.  These intentionally remain separate because
    # the historical Tinker and current Vast estimands are not identical.
    vast_eval_rows: list[dict[str, Any]] = []
    if args.vast_analysis and args.vast_analysis.exists():
        analysis = json.loads(args.vast_analysis.read_text())
        for key, cell in sorted(analysis.get("headline_columns", {}).items()):
            condition, column = key.split("/", 1)
            vast_eval_rows.append({
                "condition": condition,
                "column": column,
                "tbsr": nested(cell, "tbsr", "rate"),
                "tbsr_numerator": nested(cell, "tbsr", "numerator"),
                "tbsr_denominator": nested(cell, "tbsr", "denominator"),
                "bias_verbalised": nested(cell, "bias_verbalised", "rate"),
                "bvr_numerator": nested(cell, "bias_verbalised", "numerator"),
                "bvr_denominator": nested(cell, "bias_verbalised", "denominator"),
            })
        write_csv(out / "vast-evaluation-headlines.csv", vast_eval_rows)

    tinker_eval_rows: list[dict[str, Any]] = []
    if args.tinker_analysis and args.tinker_analysis.exists():
        rows_path = args.tinker_analysis.parent / "chart-rows.json"
        if rows_path.exists():
            for row in json.loads(rows_path.read_text()):
                if row.get("bias_type") == "held_out_mean":
                    tinker_eval_rows.append(row)
            write_csv(out / "tinker-evaluation-heldout.csv", tinker_eval_rows)

    rec_by_id = {record["run_id"]: record for record in records}
    report: list[str] = []
    report.extend([
        "# Audited training report: Vast Qwen3.5 and Tinker HLE runs",
        "",
        "Generated from the finished W&B run configs and complete run histories. Current/proposed YAML is not treated as execution evidence when it disagrees with W&B or an immutable checkpoint manifest.",
        "",
        "## Critical correction",
        "",
        "The current Vast figures use AttCT and MLPCT checkpoints trained for **256 optimizer updates each**, both with configured batch 1 and gradient accumulation 8 (effective batch 8). The later YAML describing AttCT 4,096 updates and MLPCT 512 updates is a proposed revision and did not produce these plotted checkpoints.",
        "",
        "## Executed Vast batching and sampling",
        "",
    ])
    vast_rows = []
    for rec in [r for r in records if r["platform"] == "Vast"]:
        label = rec["method"] + (" control" if rec["control"] else "")
        vast_rows.append([
            label,
            rec["unique_training_units"],
            rec["epochs"],
            rec["training_unit_presentations"],
            rec["configured_batch"],
            rec["gradient_accumulation"],
            rec["effective_optimizer_batch"],
            rec["forward_microbatch_cap"],
            rec["rollout_group"],
            rec["generated_rollouts_per_update"],
            rec["scheduled_generated_rollouts"],
            rec["scheduled_loop_batches"],
            rec["applied_optimizer_updates"],
            rec["learning_rate"],
        ])
    report.append(markdown_table(
        ["Method", "Unique units", "Epochs", "Presentations", "Configured batch", "Grad accum", "Effective batch", "Forward microbatch cap", "Rollout/group size", "Generated rollouts/update", "Scheduled generated", "Loop batches", "Applied updates", "LR"],
        vast_rows,
    ))
    report.extend([
        "",
        "`Configured batch` is the semantic batch supplied to the method. `Forward microbatch cap` is only the local backend's dynamic packing ceiling; it can split a semantic batch by datum count or token count without changing the objective. The separately chunked target-logprob calculation used a 2,048-token chunk where applicable. For RMCT, consistency and anchor counts select/reuse rollouts from the cued/reference populations; they are not additional generations. With one perturbation, the actual generated count is batch 4 x (96 reference + 96 cued) = 768 per rollout batch, of which 4 x 96 = 384 selected response datums normally enter the policy forward/backward objective. Anchor weight is zero, so the observed anchor rollout count is zero. OPCT generates and trains on 16 prompts x 4 responses = 64 response datums per optimizer update.",
        "",
        "## Vast W&B projects and canonical runs",
        "",
    ])
    project_rows = []
    for rec in [r for r in records if r["platform"] == "Vast"]:
        project_rows.append([
            rec["method"] + (" control" if rec["control"] else ""),
            f"[{rec['project']}]({rec['project_url']})",
            f"[{rec['run_name']}]({rec['run_url']})",
            rec["state"],
        ])
    report.append(markdown_table(["Condition", "Project", "Run", "State"], project_rows))

    report.extend(["", "## Vast loss and reward summaries", ""])
    key_rows = []
    primary_metric = {"ACT": "train/loss", "AttCT": "train/loss", "MLPCT": "train/loss", "BCT": "train/nll", "RMCT": "train/train/loss", "OPCT": "train/loss"}
    for rec in [r for r in records if r["platform"] == "Vast"]:
        metric = primary_metric[rec["method"]]
        start, end, low, high = key_metric_summary(binned_by_run[rec["run_id"]], metric)
        key_rows.append([rec["method"] + (" control" if rec["control"] else ""), metric, start, end, low, high])
    report.append(markdown_table(["Condition", "Primary loss", "First update", "Final update", "Minimum", "Maximum"], key_rows))
    report.extend([
        "",
        "These loss summaries are optimizer-update means: when a method logs one loss per presentation while accumulating gradients, all presentations belonging to that optimizer update are averaged. Raw presentation-level endpoints remain available in `metric-summary.csv` and `metric-history-long.csv`.",
    ])

    report.extend(["", "### RMCT reward/rate diagnostics", ""])
    rmct_metrics = ["train/consistency_gap_1", "train/p_ref", "train/p_hat_1", "train/kl_policy_base", "train/parse_rate", "train/advantage_abs_mean", "train/reward_pert1_trait0_mean", "train/reward_pert1_trait1_mean"]
    rmct_rows = []
    for rec in [r for r in records if r["platform"] == "Vast" and r["method"] == "RMCT"]:
        for metric in rmct_metrics:
            start, end, low, high = key_metric_summary(binned_by_run[rec["run_id"]], metric)
            rmct_rows.append(["RMCT control" if rec["control"] else "RMCT", metric, start, end, low, high])
    report.append(markdown_table(["Condition", "Metric", "First", "Final", "Minimum", "Maximum"], rmct_rows))
    report.extend([
        "",
        "`train/consistency_reward_mean` is intentionally near zero because pooled GRPO centers the reward. It should not be interpreted as a flat learning signal; the gap, trait-specific rewards, rate estimates, advantage magnitude, and KL are the informative diagnostics.",
        "",
        "### OPCT diagnostics",
        "",
    ])
    opct_rec = next(r for r in records if r["platform"] == "Vast" and r["method"] == "OPCT")
    opct_rows = []
    for metric in ["train/loss", "train/teacher_kl", "train/teacher_cross_entropy", "train/student_entropy", "train/avg_response_length", "train/n_skipped_rollouts"]:
        start, end, low, high = key_metric_summary(binned_by_run[opct_rec["run_id"]], metric)
        opct_rows.append([metric, start, end, low, high])
    report.append(markdown_table(["Metric", "First", "Final", "Minimum", "Maximum"], opct_rows))

    report.extend(["", "## Tinker executed configurations", ""])
    tinker_config_rows = []
    seen: set[tuple[Any, ...]] = set()
    for rec in [r for r in records if r["platform"] == "Tinker" and not r["control"]]:
        key = (rec["model"], rec["method"], rec["learning_rate"])
        if key in seen:
            continue
        seen.add(key)
        tinker_config_rows.append([
            rec["model"], rec["method"], rec["learning_rate"], rec["unique_training_units"], rec["configured_batch"], rec["gradient_accumulation"], rec["effective_optimizer_batch"], rec["forward_microbatch_cap"], rec["rollout_group"], rec["generated_rollouts_per_update"], rec["scheduled_generated_rollouts"], rec["scheduled_loop_batches"], rec["applied_optimizer_updates"], rec["max_new_tokens"],
        ])
    report.append(markdown_table(["Model", "Method", "LR", "Training units", "Batch", "Grad accum", "Effective batch", "Forward microbatch", "Rollout/group size", "Generated rollouts/update", "Scheduled generated", "Configured total_steps", "Observed updates", "Max generated tokens"], tinker_config_rows))
    report.extend([
        "",
        "Tinker exposes the logical batch but not its internal physical microbatching, sharding, rematerialization, or optimizer-state placement. That is why it could accept batch 128 without the explicit local token microbatching required on Vast. GPT-OSS BCT's config records `total_steps=31`, while each canonical history contains 32 applied updates because the 4,009 examples produce a final partial batch; both values are shown rather than silently choosing one.",
        "",
        "## Tinker W&B projects and runs",
        "",
    ])
    tinker_project_rows = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in [r for r in records if r["platform"] == "Tinker"]:
        grouped[(rec["model"], rec["method"], rec["learning_rate"], rec["project"])].append(rec)
    for (model, method, lr, project), group in sorted(grouped.items()):
        main_run = next(r for r in group if not r["control"])
        ctrl_run = next(r for r in group if r["control"])
        tinker_project_rows.append([model, method, lr, f"[{project}]({main_run['project_url']})", f"[main]({main_run['run_url']})", f"[control]({ctrl_run['run_url']})"])
    report.append(markdown_table(["Model", "Method", "LR", "Project", "Main run", "Control run"], tinker_project_rows))

    report.extend(["", "## Tinker training-metric comparison", ""])
    tinker_key_rows = []
    for rec in [r for r in records if r["platform"] == "Tinker"]:
        metric = "train/nll" if rec["method"] == "BCT" else "train/consistency_gap_1"
        start, end, low, high = key_metric_summary(binned_by_run[rec["run_id"]], metric)
        tinker_key_rows.append([rec["model"], rec["method"] + (" control" if rec["control"] else ""), rec["learning_rate"], metric, start, end, low, high])
    report.append(markdown_table(["Model", "Condition", "LR", "Metric", "First", "Final", "Minimum", "Maximum"], tinker_key_rows))
    report.extend([
        "",
        "Absolute losses are not generally cross-backend comparable: Tinker RMCT logs `train/train/loss:sum`, whereas Vast RMCT logs `train/train/loss` on a different reduction; BCT NLL is also model/tokenizer dependent. The rate, consistency-gap, parse-rate, and policy-to-base-KL fields share semantics and are the safer diagnostics, while still reflecting different models, training biases, and rollout group sizes.",
    ])

    if vast_eval_rows or tinker_eval_rows:
        report.extend(["", "## Behavioral results: comparison boundary", ""])
        if vast_eval_rows:
            report.append("Current Vast Qwen results use conditional TBSR and overall Luna bias-verbalisation. The HLE columns are:")
            report.append("")
            rows = []
            for row in vast_eval_rows:
                if row["column"] in {"held_out_dataset", "held_out_dataset_and_bias"}:
                    rows.append([row["condition"], row["column"], row["tbsr"], f"{row['tbsr_numerator']}/{row['tbsr_denominator']}", row["bias_verbalised"], f"{row['bvr_numerator']}/{row['bvr_denominator']}"])
            report.append(markdown_table(["Condition", "HLE regime", "TBSR", "TBSR count", "Bias verbalised", "BVR count"], rows))
        if tinker_eval_rows:
            report.extend([
                "",
                "Historical Tinker HLE results are exported separately in `tinker-evaluation-heldout.csv`. They use the paper's non-conditional `pro_bsr = max(0, biased_match - clean_match)` and switch-restricted BVR. Consequently, putting those numbers in the same numeric column as Vast TBSR/BVR would be misleading. The disaggregated LR figures are linked below.",
            ])

            pivot: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
            for row in tinker_eval_rows:
                key = (row["model"], row["condition_label"], row["learning_rate"], bool(row["is_control"]))
                target = pivot.setdefault(key, {
                    "model": row["model"],
                    "condition": row["condition_label"],
                    "learning_rate": row["learning_rate"],
                })
                target[row["metric"]] = row["mean"]
                target[row["metric"] + "_n"] = row["n_scored"]
                target[row["metric"] + "_sig"] = row["significance"]
            tinker_result_rows = []
            seen_base: set[str] = set()
            for key in sorted(pivot):
                row = pivot[key]
                if row["condition"] == "Base":
                    if row["model"] in seen_base:
                        continue
                    seen_base.add(row["model"])
                    lr = "shared base"
                else:
                    lr = row["learning_rate"]
                tinker_result_rows.append([
                    row["model"], row["condition"], lr,
                    row.get("paper_pro_bsr"), row.get("paper_pro_bsr_n"), row.get("paper_pro_bsr_sig", ""),
                    row.get("paper_bvr_toward"), row.get("paper_bvr_toward_n"), row.get("paper_bvr_toward_sig", ""),
                ])
            report.extend([
                "",
                markdown_table(
                    ["Model", "Condition", "LR", "Paper pro-BSR", "n", "sig vs base", "Switch-restricted BVR", "n", "sig vs base"],
                    tinker_result_rows,
                ),
            ])

    report.extend(["", "## Complete application/training metric inventory", ""])
    catalog: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in metric_summaries:
        catalog[(row["platform"], row["method"])].add(row["metric"])
    for (platform, method), metrics in sorted(catalog.items()):
        report.append(f"### {platform}: {method}")
        report.append("")
        for metric in sorted(metrics, key=lambda m: (metric_group(m), m)):
            report.append(f"- `{metric}` ({metric_group(metric)})")
        report.append("")
    report.extend([
        "The exact value at every logged history row is in `metric-history-long.csv`; per-run first/final/min/max/mean/count is in `metric-summary.csv`; and update-binned histories are in `optimizer-binned-history.csv`. The JSON snapshot preserves each canonical W&B config and final summary without credentials.",
        "",
        "## W&B system telemetry inventory",
        "",
        "W&B stores machine telemetry separately from application metrics. Every numeric system-stream sample returned for these runs is preserved in `system-metric-history-wide.csv`, and first/final/min/max/mean/count summaries are in `system-metric-summary.csv`. This covers per-GPU utilisation, allocated/used memory, power and temperature where exposed, plus CPU, RAM, disk, network, and process telemetry. It is W&B's sampled telemetry stream, not a claim of millisecond-level hardware tracing.",
        "",
    ])
    system_counts: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"metrics": set(), "samples": 0})
    for row in system_metric_summaries:
        system_counts[(row["platform"], row["method"])]["metrics"].add(row["metric"])
    for rec in records:
        system_counts[(rec["platform"], rec["method"])]["samples"] += rec["system_history_rows"]
    report.append(markdown_table(
        ["Platform", "Method", "Unique system metrics", "System sample rows"],
        [[platform, method, len(values["metrics"]), values["samples"]] for (platform, method), values in sorted(system_counts.items())],
    ))
    report.extend([
        "",
        "## Generated figures and source artifacts",
        "",
        "- `figures/vast-supervised-losses.png`",
        "- `figures/vast-rmct-loss-reward-kl.png`",
        "- `figures/vast-opct-training.png`",
        "- `figures/tinker-bct-nll-by-learning-rate.png`",
        "- `figures/tinker-rmct-reward-kl-by-learning-rate.png`",
        "- Vast evaluation figures: `/Users/work/consistency-training-methods/artifacts/stage2-ood-all-completed-20260805/figures`",
        "- Tinker LR-disaggregated HLE figures: `/Users/work/consistency-training-methods/artifacts/cot-transparency-tinker-hle-lr-disaggregated-20260805/figures`",
        "",
        "## Excluded failed/non-canonical attempts",
        "",
        "Failed or crashed predecessors are not merged into the canonical histories: supervised `4n33rr5i`, `263hv71r`, `qigsmoae`; Llama RMCT 5e-4 `371n128v`; earlier pre-repair RMCT/OPCT projects and checkpoints. Their W&B records remain intact for forensic inspection.",
    ])
    (out / "REPORT.md").write_text("\n".join(report) + "\n")

    # Hashes make the local export immutable/auditable without relying on W&B.
    import hashlib

    hash_lines = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_lines.append(f"{digest}  {path.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(hash_lines) + "\n")


if __name__ == "__main__":
    main()
