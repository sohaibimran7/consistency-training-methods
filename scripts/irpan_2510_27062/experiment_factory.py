"""Compile the Irpan paper suite and repository-method extensions.

The authored YAML keeps scientific inputs explicit.  This compiler expands the
two phenomena across an evaluation-only base model plus BCT, RMCT, ACT, AttCT,
MLPCT, and OPCT.  Only validation jobs consume training checkpoints; final jobs
take an explicit, already-selected model locator from the authored spec.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ctm.generation_provenance import make_generator_identity
from ctm.training.consistency_losses import create_consistency_loss
from ctm.training.sft import METHOD_LOSS_FNS
from scripts.irpan_2510_27062.analysis import SELECTION_OBSERVATION_SCHEMA

DOMAINS = ("sycophancy", "jailbreak")
METHODS = ("base", "bct", "rmct", "act", "attct", "mlpct", "opct")
PAPER_METHODS = frozenset({"base", "bct", "act"})
EXTENSION_METHODS = frozenset({"rmct", "attct", "mlpct", "opct"})

_VALIDATION_KEYS = {
    "sycophancy": {"mmlu"},
    "jailbreak": {"harmbench", "or_bench"},
}
_FINAL_KEYS = {
    "sycophancy": {"mmlu"},
    "jailbreak": {"clearharm", "wildguardtest", "xstest", "wildjailbreak"},
}


def _section(
    value: Any,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    plain = dict(value)
    missing = sorted(required - set(plain))
    unknown = sorted(set(plain) - required - set(optional or ()))
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ValueError(f"{label}: {', '.join(details)}")
    return plain


def _positive_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nonnegative_number(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return normalized


def _positive_number(value: Any, *, label: str) -> float:
    normalized = _nonnegative_number(value, label=label)
    if normalized == 0:
        raise ValueError(f"{label} must be a finite positive number")
    return normalized


def _unit_interval(value: Any, *, label: str) -> float:
    normalized = _nonnegative_number(value, label=label)
    if normalized > 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return normalized


def _path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    return value.strip()


def _manifest(path: str) -> str:
    return f"{path}.manifest.json"


def _method_status(method: str) -> str:
    return "paper_condition" if method in PAPER_METHODS else "repository_extension"


def _base_eval_args(model: str) -> dict[str, Any]:
    return {"model": f"hf/{model}"}


def _trained_eval_args(model: str, training_name: str) -> dict[str, Any]:
    return {
        "local_checkpoint": f"${{training.{training_name}.checkpoint}}",
        "base_model": model,
    }


def _selected_eval_args(
    selection: Mapping[str, Any],
    *,
    model: str,
    domain: str,
    method: str,
) -> dict[str, Any]:
    selected = _section(selection, f"selected_final_models.{domain}.{method}", {"kind", "value"})
    kind = selected["kind"]
    value = _path(selected["value"], label=f"selected_final_models.{domain}.{method}.value")
    if method == "base":
        if kind != "model":
            raise ValueError(f"selected_final_models.{domain}.base must use kind: model")
        return {"model": value}
    if kind != "local_checkpoint":
        raise ValueError(f"selected_final_models.{domain}.{method}: trained conditions must use kind: local_checkpoint")
    if "${training." in value:
        raise ValueError(
            f"selected_final_models.{domain}.{method}.value must be an explicit post-selection checkpoint, "
            "not a training placeholder"
        )
    return {"local_checkpoint": value, "base_model": model}


def _eval_entry(
    *,
    name: str,
    target: str,
    task_factory: str,
    task_args: Mapping[str, Any],
    model_args: Mapping[str, Any],
    generation: Mapping[str, Any],
    log_dir: str,
    smoke: bool,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    args = {
        "task_factory": task_factory,
        **dict(model_args),
        "task_args": dict(task_args),
        "generation_config": dict(generation),
        "log_dir": log_dir,
        "dry_run": smoke,
        "yes": not smoke,
    }
    if metadata is not None:
        args["metadata"] = dict(metadata)
    return {
        "name": name,
        "target": target,
        "command": ["${python}", "scripts/run_evals.py"],
        "args": args,
    }


def _selection_metadata(
    *,
    domain: str,
    method: str,
    model_args: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind validation logs to one explicit per-method candidate identity."""

    if "model" in model_args:
        locator = {"kind": "model", "value": model_args["model"]}
    elif "local_checkpoint" in model_args:
        locator = {
            "kind": "local_checkpoint",
            "value": model_args["local_checkpoint"],
            "base_model": model_args["base_model"],
        }
    else:  # pragma: no cover - every factory model path is validated above
        raise ValueError("validation selection metadata requires a model locator")
    return {
        "selection_candidate": {
            "domain": domain,
            "method": method,
            "candidate_id": f"{domain}:{method}:configured",
            "candidate_locator": locator,
            "candidate_details": {
                "method": method,
                "paper_status": _method_status(method),
            },
        }
    }


def compile_experiment(*, name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Expand one strict full/smoke spec into the executable job graph."""

    spec = _section(
        spec,
        "spec",
        {
            "mode",
            "model",
            "backend",
            "seed",
            "artifact_root",
            "data",
            "selected_final_models",
            "training",
            "evaluation",
            "tracking",
        },
    )
    if spec["mode"] not in {"full", "smoke"}:
        raise ValueError("spec.mode must be full or smoke")
    smoke = spec["mode"] == "smoke"
    if spec["backend"] != "local":
        raise ValueError("ACT, AttCT, MLPCT, and OPCT require backend: local in this comparison")
    model = _path(spec["model"], label="spec.model")
    seed = _positive_int(spec["seed"], label="spec.seed")
    root = _path(spec["artifact_root"], label="spec.artifact_root").rstrip("/")

    raw_data = _section(spec["data"], "data", set(DOMAINS))
    data: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        required = {"training_artifact", "bct_result_export", "validation", "final"}
        section = _section(
            raw_data[domain],
            f"data.{domain}",
            required,
        )
        validation = _section(
            section["validation"],
            f"data.{domain}.validation",
            _VALIDATION_KEYS[domain],
        )
        final = _section(section["final"], f"data.{domain}.final", _FINAL_KEYS[domain])
        data[domain] = {
            "training_artifact": _path(section["training_artifact"], label=f"data.{domain}.training_artifact"),
            "bct_result_export": _path(section["bct_result_export"], label=f"data.{domain}.bct_result_export"),
            "validation": {
                key: _path(value, label=f"data.{domain}.validation.{key}") for key, value in validation.items()
            },
            "final": {key: _path(value, label=f"data.{domain}.final.{key}") for key, value in final.items()},
        }
        role_paths = [
            data[domain]["training_artifact"],
            *data[domain]["validation"].values(),
            *data[domain]["final"].values(),
        ]
        if len(role_paths) != len(set(role_paths)):
            raise ValueError(f"data.{domain} training, validation, and final artifact paths must be disjoint")

    training = _section(
        spec["training"],
        "training",
        {
            "examples",
            "batch_size",
            "gradient_accumulation_steps",
            "epochs",
            "learning_rate",
            "learning_rate_schedule",
            "checkpoint_every",
            "local",
            "lora",
            "method_config",
            "target_generation",
            "rmct",
            "opct",
        },
    )
    examples_raw = _section(training["examples"], "training.examples", set(DOMAINS))
    examples = {domain: _positive_int(examples_raw[domain], label=f"training.examples.{domain}") for domain in DOMAINS}
    batch_size = _positive_int(training["batch_size"], label="training.batch_size")
    grad_accum = _positive_int(training["gradient_accumulation_steps"], label="training.gradient_accumulation_steps")
    epochs = _positive_int(training["epochs"], label="training.epochs")
    checkpoint_every = _positive_int(training["checkpoint_every"], label="training.checkpoint_every")
    learning_rate = _nonnegative_number(training["learning_rate"], label="training.learning_rate")
    if learning_rate == 0:
        raise ValueError("training.learning_rate must be positive")
    if training["learning_rate_schedule"] not in {"constant", "linear", "cosine"}:
        raise ValueError("training.learning_rate_schedule must be constant, linear, or cosine")

    local = _section(training["local"], "training.local", {"dtype", "sampler", "gpu_memory_utilization"})
    lora = _section(
        training["lora"],
        "training.lora",
        {"rank", "alpha", "dropout", "train_mlp", "train_attn", "train_unembed"},
    )
    lora = {**lora, "seed": seed}
    method_config = _section(training["method_config"], "training.method_config", {"act", "attct", "mlpct"})
    for method in ("act", "attct", "mlpct"):
        if not isinstance(method_config[method], Mapping):
            raise TypeError(f"training.method_config.{method} must be an object")
        create_consistency_loss(METHOD_LOSS_FNS[method], dict(method_config[method]))
    target_generation = _section(
        training["target_generation"],
        "training.target_generation",
        {"generator_identity", "decoding_parameters"},
    )
    if not isinstance(target_generation["generator_identity"], Mapping):
        raise TypeError("training.target_generation.generator_identity must be an object")
    if not isinstance(target_generation["decoding_parameters"], Mapping):
        raise TypeError("training.target_generation.decoding_parameters must be an object")
    raw_generator = _section(
        target_generation["generator_identity"],
        "training.target_generation.generator_identity",
        {"generator_id", "provider", "model", "model_revision", "model_immutable_date"},
    )
    generator_identity = make_generator_identity(**raw_generator)
    if generator_identity["model"] != model:
        raise ValueError(
            "training.target_generation.generator_identity.model must exactly equal spec.model "
            f"({generator_identity['model']!r} != {model!r})"
        )
    decoding_parameters = dict(target_generation["decoding_parameters"])

    rmct = _section(
        training["rmct"],
        "training.rmct",
        {
            "reference_rollouts",
            "training_rollouts",
            "consistency_rollouts",
            "anchor_rollouts",
            "temperature",
            "max_new_tokens",
            "kl_coefficient",
            "anchor_weight",
        },
    )
    rmct = {
        "reference_rollouts": _positive_int(rmct["reference_rollouts"], label="training.rmct.reference_rollouts"),
        "training_rollouts": _positive_int(rmct["training_rollouts"], label="training.rmct.training_rollouts"),
        "consistency_rollouts": _nonnegative_int(
            rmct["consistency_rollouts"], label="training.rmct.consistency_rollouts"
        ),
        "anchor_rollouts": _nonnegative_int(rmct["anchor_rollouts"], label="training.rmct.anchor_rollouts"),
        "temperature": _nonnegative_number(rmct["temperature"], label="training.rmct.temperature"),
        "max_new_tokens": _positive_int(rmct["max_new_tokens"], label="training.rmct.max_new_tokens"),
        "kl_coefficient": _nonnegative_number(rmct["kl_coefficient"], label="training.rmct.kl_coefficient"),
        "anchor_weight": _unit_interval(rmct["anchor_weight"], label="training.rmct.anchor_weight"),
    }
    if rmct["consistency_rollouts"] > rmct["training_rollouts"]:
        raise ValueError("training.rmct.consistency_rollouts cannot exceed training_rollouts")
    if rmct["anchor_rollouts"] > rmct["reference_rollouts"]:
        raise ValueError("training.rmct.anchor_rollouts cannot exceed reference_rollouts")
    if not (
        (rmct["anchor_weight"] < 1 and rmct["consistency_rollouts"] > 0)
        or (rmct["anchor_weight"] > 0 and rmct["anchor_rollouts"] > 0)
    ):
        raise ValueError("training.rmct rollout/weight configuration has no active gradient term")
    opct = _section(
        training["opct"],
        "training.opct",
        {
            "rollouts_per_prompt",
            "temperature",
            "max_new_tokens",
            "kl_coefficient",
            "kl_discount_factor",
            "loss",
        },
    )
    opct = {
        "rollouts_per_prompt": _positive_int(opct["rollouts_per_prompt"], label="training.opct.rollouts_per_prompt"),
        "temperature": _nonnegative_number(opct["temperature"], label="training.opct.temperature"),
        "max_new_tokens": _positive_int(opct["max_new_tokens"], label="training.opct.max_new_tokens"),
        "kl_coefficient": _positive_number(opct["kl_coefficient"], label="training.opct.kl_coefficient"),
        "kl_discount_factor": _unit_interval(opct["kl_discount_factor"], label="training.opct.kl_discount_factor"),
        "loss": opct["loss"],
    }
    if opct["loss"] not in {"importance_sampling", "ppo"}:
        raise ValueError("training.opct.loss must be importance_sampling or ppo")
    evaluation = _section(spec["evaluation"], "evaluation", {"max_tokens", "temperature", "judge_model"})
    generation = {
        "max_tokens": _positive_int(evaluation["max_tokens"], label="evaluation.max_tokens"),
        "temperature": _nonnegative_number(evaluation["temperature"], label="evaluation.temperature"),
    }
    judge_model = _path(evaluation["judge_model"], label="evaluation.judge_model")
    tracking = _section(spec["tracking"], "tracking", set(), {"wandb_project"})
    wandb_project = tracking.get("wandb_project")
    if wandb_project is not None:
        wandb_project = _path(wandb_project, label="tracking.wandb_project")

    selected_raw = _section(spec["selected_final_models"], "selected_final_models", set(DOMAINS))
    selected: dict[str, dict[str, dict[str, Any]]] = {}
    for domain in DOMAINS:
        by_method = _section(selected_raw[domain], f"selected_final_models.{domain}", set(METHODS))
        selected[domain] = {
            method: _selected_eval_args(
                by_method[method],
                model=model,
                domain=domain,
                method=method,
            )
            for method in METHODS
        }

    paths: dict[str, dict[str, str]] = {}
    preparation: list[dict[str, Any]] = []
    for domain in DOMAINS:
        domain_root = f"{root}/data/{domain}"
        paths[domain] = {
            "pairs": f"{domain_root}/training-pairs.jsonl",
            "requests": f"{domain_root}/bct-target-requests.jsonl",
            "targets": f"{domain_root}/bct-targets.jsonl",
            "bct": f"{domain_root}/bct-training.jsonl",
        }
        domain_preparation = [
            {
                "name": f"{domain}-training-view",
                "target": "data",
                "resource": "cpu",
                "command": [
                    "${python}",
                    "-m",
                    "scripts.irpan_2510_27062",
                    "export-training-view",
                ],
                "args": {
                    "domain": domain,
                    "input": data[domain]["training_artifact"],
                    "output": paths[domain]["pairs"],
                },
            },
            {
                "name": f"{domain}-bct-target-requests",
                "target": "data",
                "resource": "cpu",
                "command": [
                    "${python}",
                    "-m",
                    "scripts.irpan_2510_27062",
                    "build-bct-requests",
                ],
                "args": {
                    "training_view": paths[domain]["pairs"],
                    "output": paths[domain]["requests"],
                    "generator_identity": generator_identity,
                },
            },
            {
                "name": f"{domain}-bct-target-import",
                "target": "data",
                "resource": "cpu",
                "command": [
                    "${python}",
                    "-m",
                    "scripts.irpan_2510_27062",
                    "import-bct-targets",
                ],
                "args": {
                    "requests": paths[domain]["requests"],
                    "results": data[domain]["bct_result_export"],
                    "output": paths[domain]["targets"],
                    "generator_identity": generator_identity,
                    "decoding_parameters": decoding_parameters,
                },
            },
            {
                "name": f"{domain}-bct-training-export",
                "target": "data",
                "resource": "cpu",
                "command": [
                    "${python}",
                    "-m",
                    "scripts.irpan_2510_27062",
                    "export-bct-training",
                ],
                "args": {
                    "training_view": paths[domain]["pairs"],
                    "targets": paths[domain]["targets"],
                    "output": paths[domain]["bct"],
                },
            },
        ]
        if smoke:
            domain_preparation.insert(
                2,
                {
                    "name": f"{domain}-smoke-bct-results",
                    "target": "data",
                    "resource": "cpu",
                    "command": [
                        "${python}",
                        "-m",
                        "scripts.irpan_2510_27062",
                        "build-smoke-bct-results",
                    ],
                    "args": {
                        "requests": paths[domain]["requests"],
                        "output": data[domain]["bct_result_export"],
                    },
                },
            )
        preparation.extend(domain_preparation)

    local_args = {
        "backend": "local",
        "local_dtype": local["dtype"],
        "local_sampler": local["sampler"],
        "local_gpu_mem_util": local["gpu_memory_utilization"],
    }
    optimizer = {
        "learning_rate": learning_rate,
        "lr_schedule": training["learning_rate_schedule"],
    }
    common = {
        **local_args,
        "model": model,
        "experiment_name": name,
        "lora_config": lora,
        "seed": seed,
        "wandb_project": wandb_project,
        "dry_run": smoke,
        "yes": not smoke,
    }

    training_jobs: list[dict[str, Any]] = []
    for domain in DOMAINS:
        limit = examples[domain]
        bct_name = f"{domain}_bct"
        training_jobs.append(
            {
                "name": bct_name,
                "target": domain,
                "command": ["${python}", "scripts/train_bct.py"],
                "args": {
                    **common,
                    "method": "bct",
                    "run_name": bct_name,
                    "data": [f"{paths[domain]['bct']}:{limit}"],
                    "data_manifest": [_manifest(paths[domain]["bct"])],
                    "optimizer_config": optimizer,
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": grad_accum,
                    "epochs": epochs,
                    "save_every": checkpoint_every,
                },
            }
        )
        for method in ("act", "attct", "mlpct"):
            run_name = f"{domain}_{method}"
            training_jobs.append(
                {
                    "name": run_name,
                    "target": domain,
                    "command": ["${python}", "scripts/train_bct.py"],
                    "args": {
                        **common,
                        "method": method,
                        "method_config": dict(method_config[method]),
                        "run_name": run_name,
                        "data": [f"{paths[domain]['pairs']}:{limit}"],
                        "data_manifest": [_manifest(paths[domain]["pairs"])],
                        "reference_messages_field": "reference_messages",
                        "variant_messages_field": "variant_messages",
                        "alignment_text_field": "alignment_text",
                        "optimizer_config": optimizer,
                        "batch_size": batch_size,
                        "gradient_accumulation_steps": grad_accum,
                        "epochs": epochs,
                        "save_every": checkpoint_every,
                    },
                }
            )
        rmct_name = f"{domain}_rmct"
        training_jobs.append(
            {
                "name": rmct_name,
                "target": domain,
                "command": ["${python}", "scripts/train_rlct.py"],
                "args": {
                    **common,
                    "setting_factory": ("scripts.irpan_2510_27062.rmct_setting:" f"{domain}_rmct_setting"),
                    "setting_config": {
                        "training_view_path": paths[domain]["pairs"],
                        **({"grader_model": judge_model} if domain == "jailbreak" else {}),
                    },
                    "n_datapoints": limit,
                    "run_name": rmct_name,
                    "lr": learning_rate,
                    "lr_schedule": training["learning_rate_schedule"],
                    "n_ref_rollouts": rmct["reference_rollouts"],
                    "n_train_rollouts": rmct["training_rollouts"],
                    "n_consistency_rollouts": rmct["consistency_rollouts"],
                    "n_anchor_rollouts": rmct["anchor_rollouts"],
                    "temperature": rmct["temperature"],
                    "max_new_tokens": rmct["max_new_tokens"],
                    "kl_coef": rmct["kl_coefficient"],
                    "anchor_weight": rmct["anchor_weight"],
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": grad_accum,
                    "n_epochs": epochs,
                    "checkpoint_every": checkpoint_every,
                },
            }
        )
        opct_name = f"{domain}_opct"
        training_jobs.append(
            {
                "name": opct_name,
                "target": domain,
                "command": ["${python}", "scripts/train_opct.py"],
                "args": {
                    **common,
                    "run_name": opct_name,
                    "data": [f"{paths[domain]['pairs']}:{limit}"],
                    "data_manifest": [_manifest(paths[domain]["pairs"])],
                    "reference_messages_field": "reference_messages",
                    "variant_messages_field": "variant_messages",
                    "optimizer_config": optimizer,
                    "rollouts_per_prompt": opct["rollouts_per_prompt"],
                    "temperature": opct["temperature"],
                    "max_new_tokens": opct["max_new_tokens"],
                    "kl_coef": opct["kl_coefficient"],
                    "kl_discount_factor": opct["kl_discount_factor"],
                    "loss_fn": opct["loss"],
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": grad_accum,
                    "epochs": epochs,
                    "checkpoint_every": checkpoint_every,
                },
            }
        )

    log_root = f"logs/evals/{name}"
    evaluations: list[dict[str, Any]] = []
    for method in METHODS:
        training_name = f"sycophancy_{method}"
        model_args = _base_eval_args(model) if method == "base" or smoke else _trained_eval_args(model, training_name)
        metadata_suffix = f"{_method_status(method)}/sycophancy/{method}"
        for condition, factory in (
            (
                "clean",
                "scripts.irpan_2510_27062.mmlu_tasks:mmlu_clean_validation_task",
            ),
            (
                "wrong_suggestion",
                "scripts.irpan_2510_27062.mmlu_tasks:mmlu_wrong_suggestion_validation_task",
            ),
        ):
            evaluations.append(
                _eval_entry(
                    name=f"validation_sycophancy_{method}_{condition}",
                    target="validation",
                    task_factory=factory,
                    task_args={"artifact_path": data["sycophancy"]["validation"]["mmlu"]},
                    model_args=model_args,
                    generation=generation,
                    log_dir=f"{log_root}/validation/{metadata_suffix}/{condition}",
                    smoke=smoke,
                    metadata=_selection_metadata(
                        domain="sycophancy",
                        method=method,
                        model_args=model_args,
                    ),
                )
            )

    for method in METHODS:
        training_name = f"jailbreak_{method}"
        model_args = _base_eval_args(model) if method == "base" or smoke else _trained_eval_args(model, training_name)
        metadata_suffix = f"{_method_status(method)}/jailbreak/{method}"
        evaluations.append(
            _eval_entry(
                name=f"validation_jailbreak_{method}",
                target="validation",
                task_factory="scripts.irpan_2510_27062.safety_tasks:paper_safety_tasks",
                task_args={
                    "datasets": [
                        data["jailbreak"]["validation"]["harmbench"],
                        data["jailbreak"]["validation"]["or_bench"],
                    ],
                    "judge_model": judge_model,
                },
                model_args=model_args,
                generation=generation,
                log_dir=f"{log_root}/validation/{metadata_suffix}",
                smoke=smoke,
                metadata=_selection_metadata(
                    domain="jailbreak",
                    method=method,
                    model_args=model_args,
                ),
            )
        )

    for method in METHODS:
        metadata_suffix = f"{_method_status(method)}/sycophancy/{method}"
        for condition, factory in (
            ("clean", "scripts.irpan_2510_27062.mmlu_tasks:mmlu_clean_task"),
            (
                "wrong_suggestion",
                "scripts.irpan_2510_27062.mmlu_tasks:mmlu_wrong_suggestion_task",
            ),
        ):
            evaluations.append(
                _eval_entry(
                    name=f"final_sycophancy_{method}_{condition}",
                    target="final",
                    task_factory=factory,
                    task_args={"artifact_path": data["sycophancy"]["final"]["mmlu"]},
                    model_args=selected["sycophancy"][method],
                    generation=generation,
                    log_dir=f"{log_root}/final/{metadata_suffix}/{condition}",
                    smoke=smoke,
                )
            )
        metadata_suffix = f"{_method_status(method)}/jailbreak/{method}"
        evaluations.append(
            _eval_entry(
                name=f"final_jailbreak_{method}",
                target="final",
                task_factory="scripts.irpan_2510_27062.safety_tasks:paper_safety_tasks",
                task_args={
                    "datasets": [data["jailbreak"]["final"][key] for key in sorted(_FINAL_KEYS["jailbreak"])],
                    "judge_model": judge_model,
                },
                model_args=selected["jailbreak"][method],
                generation=generation,
                log_dir=f"{log_root}/final/{metadata_suffix}",
                smoke=smoke,
            )
        )

    plan: dict[str, Any] = {
        "name": name,
        "variables": {
            "paper_id": "irpan_2510_27062",
            "mode": spec["mode"],
            "opct_source_commit": "79347b6dad38074436a6a739c3b246c49ddcb83f",
            "selection_observation_schema": SELECTION_OBSERVATION_SCHEMA,
        },
        "data_preparation": preparation,
        "training": training_jobs,
        "evaluation": evaluations,
    }
    if smoke:
        plan["data_generation"] = {
            "name": "synthetic-smoke-fixtures",
            "target": "data",
            "resource": "cpu",
            "command": [
                "${python}",
                "-m",
                "scripts.irpan_2510_27062",
                "materialize-smoke-fixtures",
            ],
            "args": {"output_dir": f"{root}/fixtures"},
        }
    else:
        selection_root = f"{root}/selection"
        analysis: list[dict[str, Any]] = []
        for domain in DOMAINS:
            for method in METHODS:
                observations = f"{selection_root}/{domain}/{method}-validation-observations.jsonl"
                analysis.extend(
                    [
                        {
                            "name": f"collect-{domain}-{method}-validation-observations",
                            "target": "selection",
                            "resource": "cpu",
                            "command": [
                                "${python}",
                                "-m",
                                "scripts.irpan_2510_27062",
                                "collect-validation-observations",
                            ],
                            "args": {
                                "domain": domain,
                                "method": method,
                                "log_dir": f"{log_root}/validation",
                                "schema": SELECTION_OBSERVATION_SCHEMA,
                                "output": observations,
                            },
                        },
                        {
                            "name": f"select-{domain}-{method}-validation",
                            "target": "selection",
                            "resource": "cpu",
                            "command": [
                                "${python}",
                                "-m",
                                "scripts.irpan_2510_27062",
                                "select-validation",
                            ],
                            "args": {
                                "domain": domain,
                                "method": method,
                                "input": observations,
                                "output": f"{selection_root}/{domain}/{method}-selected-candidate.json",
                            },
                        },
                    ]
                )
        plan["analysis"] = analysis
    return plan


__all__ = [
    "DOMAINS",
    "EXTENSION_METHODS",
    "METHODS",
    "PAPER_METHODS",
    "compile_experiment",
]
