"""Expand a concise mcq-bias comparison into CTM's explicit command plan."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def _section(value: Any, label: str, fields: set[str], optional: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    value = dict(value)
    optional = optional or set()
    missing = sorted(fields - value.keys())
    unknown = sorted(value.keys() - fields - optional)
    if missing or unknown:
        details = [*(f"missing {missing}" for _ in [0] if missing), *(f"unknown {unknown}" for _ in [0] if unknown)]
        raise ValueError(f"{label}: {', '.join(details)}")
    return value


def _named_items(value: Any, label: str, fields: set[str], optional: set[str] | None = None) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    items = [_section(item, f"{label}[{index}]", fields, optional) for index, item in enumerate(value)]
    names = [item["name"] for item in items]
    if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"{label} names must be non-empty and unique")
    return items


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]", "_", value.replace("-", "_"))
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_-]*", value):
        raise ValueError(f"cannot use {value!r} as a command name")
    return value


def _metric(value: str) -> str:
    # This is the current mcq-bias field name for total bias switch.
    return "abs_switch" if value == "total_bias_switch" else value


def compile_experiment(*, name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Derive repeated runs and paths while keeping scientific choices explicit."""

    spec = _section(
        spec,
        "spec",
        {
            "model", "backend", "seed", "artifact_root", "figure_root", "learning_rates", "lora", "data",
            "local", "conditions", "supervised_consistency", "rate_matching", "evaluation", "tracking", "reports",
        },
    )
    if spec["backend"] != "local":
        raise ValueError("the five-method comparison requires backend: local")
    rates = _named_items(spec["learning_rates"], "learning_rates", {"name", "value"})
    conditions = _named_items(spec["conditions"], "conditions", {"name", "method"}, {"control"})
    if sum(condition["method"] == "none" for condition in conditions) != 1:
        raise ValueError("conditions must contain exactly one method: none")
    methods = {"none", "rate_matching", "bias_augmented_consistency", "act", "attct", "mlpct"}
    if any(condition["method"] not in methods for condition in conditions):
        raise ValueError(f"condition methods must be one of {sorted(methods)}")
    if any(not isinstance(condition.get("control", False), bool) for condition in conditions):
        raise ValueError("condition control values must be booleans")
    if any(condition["method"] == "none" and condition.get("control", False) for condition in conditions):
        raise ValueError("an untrained condition cannot be a control")

    data = _section(spec["data"], "data", {"training", "instruction", "evaluation"})
    train_data = _section(
        data["training"], "data.training",
        {"bias_type", "datasets", "prompt_style", "pool_per_dataset", "minimum_per_dataset", "examples", "argument_model"},
    )
    instruction = _section(data["instruction"], "data.instruction", {"source", "examples"})
    eval_data = _section(
        data["evaluation"], "data.evaluation",
        {"source", "expected_source_count", "questions", "biases", "argument_model", "grader_model"},
    )
    if instruction["source"] != "cleaned_alpaca" or eval_data["source"] != "hle":
        raise ValueError("this compiler requires cleaned_alpaca instruction data and hle evaluation data")
    lora = _section(spec["lora"], "lora", {"rank", "alpha", "dropout", "train_mlp", "train_attn", "train_unembed"})
    local = _section(spec["local"], "local", {"dtype", "sampler", "gpu_memory_utilization"})
    sft = _section(
        spec["supervised_consistency"], "supervised_consistency",
        {
            "batch_size", "gradient_accumulation_steps", "epochs", "save_every", "learning_rate_schedule",
            "bias_augmented_targets", "method_config",
        },
    )
    target_generation = _section(
        sft["bias_augmented_targets"], "supervised_consistency.bias_augmented_targets",
        {"max_tokens", "temperature", "max_concurrency"},
    )
    method_configs = _section(sft["method_config"], "supervised_consistency.method_config", {"act", "attct", "mlpct"})
    rm = _section(
        spec["rate_matching"], "rate_matching",
        {
            "datapoints", "rollouts", "batch_size", "epochs", "temperature", "max_new_tokens",
            "learning_rate_schedule", "kl_coefficient", "anchor_weight", "anchor_model", "loss",
            "advantage_estimator", "normalization", "gradient_accumulation_steps", "refresh_every",
            "checkpoint_every",
        },
    )
    rollouts = _section(rm["rollouts"], "rate_matching.rollouts", {"reference", "training", "consistency", "anchor"})
    evaluation = _section(spec["evaluation"], "evaluation", {"max_tokens", "temperature"})
    tracking = _section(spec["tracking"], "tracking", {"wandb_project"})
    reports = _section(spec["reports"], "reports", {"held_out_exclude", "standard_error", "charts", "items"})
    charts = reports["charts"]
    if not isinstance(charts, Mapping):
        raise ValueError("reports.charts must be an object")
    report_items = _named_items(reports["items"], "reports.items", {"name", "metric", "chart"}, {"given"})

    model, backend, seed = spec["model"], spec["backend"], spec["seed"]
    root, figure_root = str(spec["artifact_root"]).rstrip("/"), str(spec["figure_root"]).rstrip("/")
    data_root, eval_root, log_root = f"{root}/data", f"{root}/mcq-bias-evaluation", "logs/evals/${experiment}"
    paths = {
        "hle": f"{data_root}/hle-text-mc.jsonl",
        "hle_manifest": f"{data_root}/hle-text-mc.manifest.json",
        "pairs": f"{data_root}/distractor-argument-pairs.jsonl",
        "pairs_manifest": f"{data_root}/distractor-argument-pairs.manifest.json",
        "prompts": f"{data_root}/cleaned-alpaca-prompts.jsonl",
        "prompts_manifest": f"{data_root}/cleaned-alpaca-prompts.manifest.json",
        "bct": f"{data_root}/bias-augmented-consistency.jsonl",
        "bct_control": f"{data_root}/bias-augmented-consistency-control.jsonl",
        "bct_manifest": f"{data_root}/bias-augmented-consistency.manifest.json",
        "instruction": f"{data_root}/instruction-targets.jsonl",
        "instruction_control": f"{data_root}/instruction-targets-control.jsonl",
        "instruction_manifest": f"{data_root}/instruction-targets.manifest.json",
    }
    lora_config = {**lora, "seed": seed}
    local_args = {
        "backend": backend, "local_dtype": local["dtype"], "local_sampler": local["sampler"],
        "local_gpu_mem_util": local["gpu_memory_utilization"],
    }
    yes = {"yes": True}

    data_generation = [
        {
            "name": "hle-source", "command": ["${python}", "-m", "ctm_data.sources.hle"],
            "args": {"output": paths["hle"], "manifest_output": paths["hle_manifest"],
                     "expected_count": eval_data["expected_source_count"], **yes},
        },
        {
            "name": "distractor-argument-pairs",
            "command": ["${python}", "-m", "ctm_data.adapters.mcq_bias.materialize"],
            "args": {
                "bias_type": train_data["bias_type"], "datasets": train_data["datasets"],
                "prompt_style": train_data["prompt_style"], "n_questions": train_data["pool_per_dataset"],
                "min_n_questions": train_data["minimum_per_dataset"], "seed": str(seed),
                "argument_model": train_data["argument_model"], "generate_missing_arguments": True,
                "dataset_dir": f"{data_root}/mcq-bias-train", "output": paths["pairs"],
                "manifest_output": paths["pairs_manifest"], **yes,
            },
        },
        {
            "name": "cleaned-alpaca-prompts", "command": ["${python}", "-m", "ctm_data.sources.cleaned_alpaca"],
            "args": {"output": paths["prompts"], "manifest_output": paths["prompts_manifest"],
                     "count": instruction["examples"], "seed": str(seed), **yes},
        },
    ]
    target_common = {
        **local_args, "model": model, "max_tokens": target_generation["max_tokens"],
        "temperature": target_generation["temperature"], "max_concurrency": target_generation["max_concurrency"], **yes,
    }
    data_preparation = [
        {
            "name": "bias-augmented-consistency-targets", "command": ["${python}", "scripts/prepare_bct_targets.py"],
            "args": {
                **target_common, "data": [paths["pairs"]], "limit": train_data["examples"],
                "source_messages_field": "unbiased_messages", "main_messages_field": "biased_messages",
                "control_messages_field": "unbiased_messages", "main_output": paths["bct"],
                "control_output": paths["bct_control"], "manifest_output": paths["bct_manifest"],
            },
        },
        {
            "name": "instruction-targets", "command": ["${python}", "scripts/prepare_bct_targets.py"],
            "args": {
                **target_common, "data": [paths["prompts"]], "limit": instruction["examples"],
                "source_messages_field": "reference_messages", "main_messages_field": "variant_messages",
                "control_messages_field": "reference_messages", "main_output": paths["instruction"],
                "control_output": paths["instruction_control"], "manifest_output": paths["instruction_manifest"],
            },
        },
    ]

    task_args = {
        "bias_types": eval_data["biases"], "datasets": [paths["hle"]], "prompt_style": "none",
        "n_questions": eval_data["questions"], "seed": str(seed), "argument_model": eval_data["argument_model"],
        "generate_missing_arguments": True, "dataset_dir": eval_root, "grader_model": eval_data["grader_model"],
        "include_bias_acknowledged": True,
    }
    eval_common = {
        "task_factory": "mcq_bias.tasks:suite_tasks", "base_model": model,
        "generation_config": {"max_tokens": evaluation["max_tokens"], "temperature": evaluation["temperature"]},
        **yes,
    }
    training: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    analysis_runs: list[str] = []
    for condition in conditions:
        condition_name, method, control = condition["name"], condition["method"], condition.get("control", False)
        if method == "none":
            log_dir = f"{log_root}/{condition_name}"
            evals.append({
                "name": condition_name, "command": ["${python}", "scripts/run_evals.py"],
                "args": {**{key: value for key, value in eval_common.items() if key != "base_model"},
                         "model": f"hf/{model}",
                         "task_args": {**task_args, "unbiased_log": log_dir}, "log_dir": log_dir},
            })
            analysis_runs.append(f"{condition_name}={log_dir}")
            continue

        for rate_index, rate in enumerate(rates, start=1):
            command_name = f"{_slug(condition_name)}_lr{rate_index}"
            if method in {"bias_augmented_consistency", "act", "attct", "mlpct"}:
                bct_path = paths["bct_control"] if control else paths["bct"]
                instruction_path = paths["instruction_control"] if control else paths["instruction"]
                args = {
                    **local_args, "model": model,
                    "batch_size": sft["batch_size"],
                    "gradient_accumulation_steps": sft["gradient_accumulation_steps"],
                    "epochs": sft["epochs"],
                    "lora_config": lora_config,
                    "optimizer_config": {"learning_rate": rate["value"], "lr_schedule": sft["learning_rate_schedule"]},
                    "save_every": sft["save_every"], "experiment_name": "${experiment}",
                    "wandb_project": tracking["wandb_project"], **yes,
                }
                if method == "bias_augmented_consistency":
                    args.update({
                        "method": "bct",
                        "data": [
                            f"{bct_path}:{train_data['examples']}",
                            f"{instruction_path}:{instruction['examples']}",
                        ],
                        "data_manifest": [paths["bct_manifest"], paths["instruction_manifest"]],
                        "interleave": True,
                    })
                else:
                    args.update({
                        "method": method,
                        "method_config": method_configs[method],
                        "data": [f"{paths['pairs']}:{train_data['examples']}"],
                        "data_manifest": [paths["pairs_manifest"]],
                        "reference_messages_field": "unbiased_messages",
                        "variant_messages_field": "unbiased_messages" if control else "biased_messages",
                    })
                command = ["${python}", "scripts/train_bct.py"]
            else:
                setting_config = {"data_paths": [paths["pairs"]], **({"control": True} if control else {})}
                args = {
                    **local_args, "model": model, "setting_factory": "ctm_data.adapters.mcq_bias:create_setting",
                    "load_config": {"n_datapoints": rm["datapoints"]}, "setting_config": setting_config,
                    "experiment_name": "${experiment}", "seed": seed, "lora_config": lora_config,
                    "lr": rate["value"], "lr_schedule": rm["learning_rate_schedule"], "kl_coef": rm["kl_coefficient"],
                    "anchor_weight": rm["anchor_weight"], "anchor_model": rm["anchor_model"], "loss_fn": rm["loss"],
                    "advantage_estimator": rm["advantage_estimator"], "normalization": rm["normalization"],
                    "n_ref_rollouts": rollouts["reference"], "n_train_rollouts": rollouts["training"],
                    "n_consistency_rollouts": rollouts["consistency"], "n_anchor_rollouts": rollouts["anchor"],
                    "temperature": rm["temperature"], "max_new_tokens": rm["max_new_tokens"],
                    "batch_size": rm["batch_size"], "gradient_accumulation_steps": rm["gradient_accumulation_steps"],
                    "refresh_every": rm["refresh_every"], "n_epochs": rm["epochs"],
                    "checkpoint_every": rm["checkpoint_every"], "wandb_project": tracking["wandb_project"], **yes,
                }
                command = ["${python}", "scripts/train_rlct.py"]
            args["run_name"] = f"{condition_name}-lr-{rate['name']}"
            training.append({"name": command_name, "command": command, "args": args})
            log_dir = f"{log_root}/{condition_name}/lr{rate_index}"
            evals.append({
                "name": f"{condition_name}-lr{rate_index}", "command": ["${python}", "scripts/run_evals.py"],
                "args": {**eval_common, "local_checkpoint": f"${{training.{command_name}.checkpoint}}",
                         "task_args": {**task_args, "unbiased_log": log_dir}, "log_dir": log_dir},
            })
            analysis_runs.append(f"{condition_name}={log_dir}")

    analysis: list[dict[str, Any]] = []
    for report in report_items:
        if report["chart"] not in charts:
            raise ValueError(f"report {report['name']!r} refers to unknown chart {report['chart']!r}")
        result, figure = f"{root}/results/{report['name']}.json", f"{figure_root}/{report['name']}.svg"
        args = {
            "run": analysis_runs, "metric": _metric(report["metric"]), "stderr": reports["standard_error"],
            "held_out_exclude": reports["held_out_exclude"], "output": result, **yes,
        }
        if "given" in report:
            if report["given"] not in {"towards_bias_switch", "total_bias_switch"}:
                raise ValueError("report given must be towards_bias_switch or total_bias_switch")
            args.update({"where_metric": _metric(report["given"]), "where_value": 1.0})
        analysis.extend([
            {"name": f"aggregate-{report['name']}", "command": ["${python}", "-m", "ctm_data.adapters.mcq_bias.analysis"], "args": args},
            {"name": f"render-{report['name']}", "command": ["node", "scripts/render_flint.mjs"],
             "args": {"data": result, "spec": charts[report["chart"]], "output": figure}},
        ])

    return {
        "name": name, "data_generation": data_generation, "data_preparation": data_preparation,
        "training": training, "evaluation": evals, "analysis": analysis,
    }


__all__ = ["compile_experiment"]
