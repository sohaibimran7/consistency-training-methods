"""Expand a concise mcq-bias comparison into CTM's explicit command plan."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any


def _section(value: Any, label: str, fields: set[str], optional: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
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


def _execution_target(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value):
        raise ValueError(f"{label} must start with a letter or digit and contain only letters, digits, dots, underscores, and hyphens")
    return value


def _apply_execution_allocations(
    stages: Mapping[str, list[dict[str, Any]]],
    value: Any,
) -> dict[str, Any] | None:
    """Attach operational node/GPU metadata without changing scientific arguments."""

    if value is None:
        return None
    execution = _section(value, "execution", {"publication_owner", "allocations"})
    publication_owner = _execution_target(execution["publication_owner"], "execution.publication_owner")
    raw_allocations = execution["allocations"]
    if not isinstance(raw_allocations, list) or not raw_allocations:
        raise ValueError("execution.allocations must be a non-empty list")

    allocated: set[tuple[str, str]] = set()
    training_targets: list[str] = []
    for index, raw_allocation in enumerate(raw_allocations):
        label = f"execution.allocations[{index}]"
        allocation = _section(
            raw_allocation,
            label,
            {"target", "stage", "commands"},
            {"gpu_count", "local_device", "rollout_gpus", "gradient_checkpointing_layers"},
        )
        target = _execution_target(allocation["target"], f"{label}.target")
        stage = allocation["stage"]
        if stage not in {"data_preparation", "training", "evaluation"}:
            raise ValueError(f"{label}.stage must be data_preparation, training, or evaluation")
        commands = allocation["commands"]
        if not isinstance(commands, list) or not commands or any(not isinstance(command, str) or not command for command in commands) or len(commands) != len(set(commands)):
            raise ValueError(f"{label}.commands must be a non-empty list of unique command names")
        entries_by_name = {entry["name"]: entry for entry in stages.get(stage, [])}
        unknown = sorted(set(commands) - entries_by_name.keys())
        if unknown:
            raise ValueError(f"{label}.commands names are absent from {stage}: {unknown}")

        # An allocation is one schedulable resource bundle.  Keep CPU-only
        # transforms distinct from GPU work so a cheap local JSONL transform
        # never reserves an otherwise idle accelerator.  ``resource`` mirrors
        # scripts.run_experiment.command_resource's stage default.
        resources = {
            entry.get("resource", "gpu" if stage in {"data_preparation", "training", "evaluation"} else "cpu")
            for entry in (entries_by_name[command_name] for command_name in commands)
        }
        if resources - {"cpu", "gpu"}:
            raise ValueError(f"{label}.commands have invalid resource declaration(s): {sorted(resources)}")
        if len(resources) != 1:
            raise ValueError(f"{label}.commands must all use the same resource; split CPU and GPU commands into separate allocations")
        resource = resources.pop()
        gpu_count = allocation.get("gpu_count")
        if resource == "cpu":
            if gpu_count is not None:
                raise ValueError(f"{label}.gpu_count applies only to GPU allocations")
            cpu_gpu_fields = [
                field
                for field in ("local_device", "rollout_gpus", "gradient_checkpointing_layers")
                if allocation.get(field) is not None
            ]
            if cpu_gpu_fields:
                raise ValueError(f"{label} CPU allocation cannot set GPU field(s): {cpu_gpu_fields}")
        elif isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
            raise ValueError(f"{label}.gpu_count must be a positive integer")

        checkpoint_layers = allocation.get("gradient_checkpointing_layers")
        if checkpoint_layers is not None and checkpoint_layers != "all" and (isinstance(checkpoint_layers, bool) or not isinstance(checkpoint_layers, int) or checkpoint_layers < 1):
            raise ValueError(f"{label}.gradient_checkpointing_layers must be a positive integer or 'all'")

        rollout_gpus = allocation.get("rollout_gpus")
        local_device = allocation.get("local_device")
        if resource == "cpu":
            # The validations above make all of these absent.  A CPU command
            # gets only its target; the runner consequently schedules it with
            # no CUDA_VISIBLE_DEVICES reservation.
            pass
        elif rollout_gpus is None:
            if local_device is not None:
                raise ValueError(f"{label}.local_device requires rollout_gpus")
            if gpu_count != 1:
                raise ValueError(f"{label} needs rollout_gpus when gpu_count is greater than one")
        else:
            if not isinstance(rollout_gpus, list) or not rollout_gpus or any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 for gpu in rollout_gpus) or len(rollout_gpus) != len(set(rollout_gpus)):
                raise ValueError(f"{label}.rollout_gpus must be a non-empty list of unique non-negative indices")
            if any(gpu >= gpu_count for gpu in rollout_gpus):
                raise ValueError(f"{label}.rollout_gpus indices must be inside the allocated GPU bundle")
            if stage == "training":
                match = re.fullmatch(r"cuda:(\d+)", str(local_device or ""))
                if match is None:
                    raise ValueError(f"{label}.local_device must be an explicit cuda:N coordinator")
                coordinator_gpu = int(match.group(1))
                if len(rollout_gpus) + 1 != gpu_count:
                    raise ValueError(f"{label} must allocate exactly one coordinator plus one GPU per rollout worker")
                if coordinator_gpu >= gpu_count or coordinator_gpu in rollout_gpus:
                    raise ValueError(f"{label} coordinator must be inside the bundle and absent from rollout_gpus")
            elif stage == "data_preparation":
                if local_device is not None:
                    raise ValueError(f"{label}.local_device is invalid for coordinator-free target generation")
                if set(rollout_gpus) != set(range(gpu_count)):
                    raise ValueError(f"{label} base-only target generation must assign one worker to every logical GPU")
            else:
                raise ValueError(f"{label}.rollout_gpus applies only to training or data_preparation")

        for command_name in commands:
            key = (stage, command_name)
            if key in allocated:
                raise ValueError(f"execution allocates {stage} command {command_name!r} more than once")
            allocated.add(key)
            entry = entries_by_name[command_name]
            entry["target"] = target
            if resource == "gpu":
                entry["gpu_count"] = gpu_count
            if checkpoint_layers is not None:
                if stage != "training":
                    raise ValueError(f"{label}.gradient_checkpointing_layers applies only to training")
                if checkpoint_layers == "all":
                    entry["args"].pop("local_gradient_checkpointing_layers", None)
                else:
                    entry["args"]["local_gradient_checkpointing_layers"] = checkpoint_layers
            if rollout_gpus is not None:
                supported = {"scripts/train_rlct.py", "scripts/train_opct.py"} if stage == "training" else {"scripts/prepare_bct_targets.py"}
                if entry["command"][-1] not in supported:
                    raise ValueError(f"{label}.rollout_gpus is unsupported for command {entry['command'][-1]!r}")
                args = entry["args"]
                if "local_device" in args or "local_rollout_gpus" in args:
                    raise ValueError(f"{label} conflicts with existing local rollout arguments")
                if stage == "training":
                    args["local_device"] = local_device
                args["local_rollout_gpus"] = ",".join(str(gpu) for gpu in rollout_gpus)
        if stage == "training" and target not in training_targets:
            training_targets.append(target)

    expected = {(stage, entry["name"]) for stage in ("data_preparation", "training", "evaluation") for entry in stages.get(stage, [])}
    missing = sorted(expected - allocated)
    if missing:
        raise ValueError(f"execution.allocations must cover every GPU-stage command; missing {missing}")
    return {"owner": publication_owner, "targets": training_targets}


def compile_experiment(
    *,
    name: str,
    spec: Mapping[str, Any],
    topology_profile: str | None = None,
) -> dict[str, Any]:
    """Derive repeated runs and paths while keeping scientific choices explicit."""

    if topology_profile is not None:
        raise ValueError(
            "this mcq-bias plan has no selectable topology profile; use a plan-specific factory that declares one"
        )

    spec = _section(
        spec,
        "spec",
        {
            "model",
            "backend",
            "seed",
            "artifact_root",
            "figure_root",
            "learning_rates",
            "lora",
            "data",
            "local",
            "conditions",
            "supervised_consistency",
            "rate_matching",
            "evaluation",
            "tracking",
            "reports",
        },
        {"opct", "execution", "training_only", "onpolicy_target_attestation"},
    )
    if spec["backend"] != "local":
        raise ValueError("the mcq-bias comparison requires backend: local")
    training_only = spec.get("training_only", False)
    if not isinstance(training_only, bool):
        raise TypeError("training_only must be a boolean")
    require_onpolicy_target_attestation = spec.get("onpolicy_target_attestation", False)
    if not isinstance(require_onpolicy_target_attestation, bool):
        raise TypeError("onpolicy_target_attestation must be a boolean")
    rates = _named_items(spec["learning_rates"], "learning_rates", {"name", "value"})
    conditions = _named_items(spec["conditions"], "conditions", {"name", "method"}, {"control"})
    if sum(condition["method"] == "none" for condition in conditions) != 1:
        raise ValueError("conditions must contain exactly one method: none")
    methods = {"none", "rate_matching", "bias_augmented_consistency", "opct", "act", "attct", "mlpct"}
    if any(condition["method"] not in methods for condition in conditions):
        raise ValueError(f"condition methods must be one of {sorted(methods)}")
    if any(not isinstance(condition.get("control", False), bool) for condition in conditions):
        raise ValueError("condition control values must be booleans")
    if any(condition["method"] == "none" and condition.get("control", False) for condition in conditions):
        raise ValueError("an untrained condition cannot be a control")
    if any(condition["method"] in {"act", "attct", "mlpct"} and condition.get("control", False) for condition in conditions):
        raise ValueError("act, attct, and mlpct cannot have control conditions: pairing unbiased_messages with itself makes the consistency loss identically zero, so the run duplicates 'none'")

    data = _section(
        spec["data"],
        "data",
        {"training", "instruction", "evaluation"},
        {"shared_root", "prepare_shared"},
    )
    prepare_shared = data.get("prepare_shared", True)
    if not isinstance(prepare_shared, bool):
        raise TypeError("data.prepare_shared must be a boolean")
    shared_root = data.get("shared_root")
    if shared_root is not None and (not isinstance(shared_root, str) or not shared_root.strip()):
        raise ValueError("data.shared_root must be a non-empty path")
    if not prepare_shared and shared_root is None:
        raise ValueError("data.prepare_shared: false requires data.shared_root")
    if training_only and prepare_shared:
        raise ValueError("training_only requires data.prepare_shared: false so it cannot create mutable shared inputs")
    if training_only:
        unsupported = sorted(
            condition["method"]
            for condition in conditions
            if condition["method"] not in {"none", "rate_matching", "opct"}
        )
        if unsupported:
            raise ValueError(
                "training_only supports only none, rate_matching, and opct conditions; "
                f"these methods require omitted target-generation stages: {unsupported}"
            )
    train_data = _section(
        data["training"],
        "data.training",
        {
            "bias_type",
            "datasets",
            "prompt_style",
            "pool_per_dataset",
            "minimum_per_dataset",
            "examples",
            "argument_model",
        },
        {"pairs_path", "selection_manifest", "continuation_manifest"},
    )
    training_pairs_path = train_data.get("pairs_path")
    training_selection_manifest = train_data.get("selection_manifest")
    training_continuation_manifest = train_data.get("continuation_manifest")
    if training_selection_manifest is not None and training_continuation_manifest is not None:
        raise ValueError("data.training may declare selection_manifest or continuation_manifest, not both")
    training_manifest = training_selection_manifest or training_continuation_manifest
    if (training_pairs_path is None) != (training_manifest is None):
        raise ValueError(
            "data.training.pairs_path and exactly one immutable training manifest must be supplied together"
        )
    if training_pairs_path is not None:
        if not isinstance(training_pairs_path, str) or not training_pairs_path.strip():
            raise ValueError("data.training.pairs_path must be a non-empty path")
        if training_selection_manifest is not None and (
            not isinstance(training_selection_manifest, str) or not training_selection_manifest.strip()
        ):
            raise ValueError("data.training.selection_manifest must be a non-empty path")
        if training_continuation_manifest is not None and (
            not isinstance(training_continuation_manifest, str) or not training_continuation_manifest.strip()
        ):
            raise ValueError("data.training.continuation_manifest must be a non-empty path")
        if prepare_shared:
            raise ValueError(
                "data.training.pairs_path is a frozen-input override and requires data.prepare_shared: false"
            )
    instruction = _section(data["instruction"], "data.instruction", {"source", "examples"})
    eval_data = _section(
        data["evaluation"],
        "data.evaluation",
        {"source", "expected_source_count", "questions", "biases", "argument_model", "grader_model"},
        {"minimum_questions_by_bias"},
    )
    if instruction["source"] != "cleaned_alpaca" or eval_data["source"] != "hle":
        raise ValueError("this compiler requires cleaned_alpaca instruction data and hle evaluation data")
    eval_questions = eval_data["questions"]
    if isinstance(eval_questions, bool) or not isinstance(eval_questions, int) or eval_questions < 1:
        raise TypeError("data.evaluation.questions must be a positive integer")
    eval_biases = eval_data["biases"]
    if not isinstance(eval_biases, list) or not eval_biases or any(not isinstance(bias, str) or not bias for bias in eval_biases) or len(eval_biases) != len(set(eval_biases)):
        raise ValueError("data.evaluation.biases must be a non-empty list of unique bias names")
    raw_eval_minimums = eval_data.get("minimum_questions_by_bias", {})
    if not isinstance(raw_eval_minimums, Mapping):
        raise TypeError("data.evaluation.minimum_questions_by_bias must be an object")
    eval_minimums = dict(raw_eval_minimums)
    unknown_minimums = sorted(set(eval_minimums) - set(eval_biases))
    if unknown_minimums:
        raise ValueError(f"data.evaluation.minimum_questions_by_bias contains biases absent from data.evaluation.biases: {unknown_minimums}")
    for bias, minimum in eval_minimums.items():
        if isinstance(minimum, bool) or not isinstance(minimum, int) or not 1 <= minimum <= eval_questions:
            raise ValueError(f"data.evaluation.minimum_questions_by_bias[{bias!r}] must be an integer between 1 and data.evaluation.questions")
        if minimum < eval_questions and bias != "wrong_argument":
            raise ValueError("only wrong_argument may use fewer than data.evaluation.questions; the other bias transforms must retain the full deterministic question pool")
    eval_min_questions = min([eval_questions, *eval_minimums.values()])
    lora = _section(
        spec["lora"],
        "lora",
        {"rank", "alpha", "dropout", "train_mlp", "train_attn", "train_unembed"},
        {"target_modules"},
    )
    local = _section(
        spec["local"],
        "local",
        {"dtype", "sampler", "gpu_memory_utilization"},
        {
            "rollout_gpu_memory_utilization",
            "rollout_seed_base",
            "vllm_language_model_only",
            "vllm_max_num_seqs",
            "vllm_max_num_batched_tokens",
            "vllm_max_model_len",
            "vllm_gdn_prefill_backend",
            "gradient_checkpointing",
            "gradient_checkpointing_layers",
            "forward_microbatch_max_datums",
            "forward_microbatch_max_tokens",
            "target_logprob_chunk_size",
        },
    )
    for field in ("gpu_memory_utilization", "rollout_gpu_memory_utilization"):
        value = local.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError(f"local.{field} must be a finite number in (0, 1]")
    rollout_seed_base = local.get("rollout_seed_base")
    if rollout_seed_base is not None and (
        isinstance(rollout_seed_base, bool)
        or not isinstance(rollout_seed_base, int)
        or not 0 <= rollout_seed_base <= 2**31 - 1
    ):
        raise ValueError("local.rollout_seed_base must be an integer in [0, 2147483647]")
    if not isinstance(local.get("vllm_language_model_only", False), bool):
        raise TypeError("local.vllm_language_model_only must be a boolean")
    gradient_checkpointing = local.get("gradient_checkpointing", False)
    if not isinstance(gradient_checkpointing, bool):
        raise TypeError("local.gradient_checkpointing must be a boolean")
    gradient_checkpointing_layers = local.get("gradient_checkpointing_layers")
    if gradient_checkpointing_layers is not None and (not isinstance(gradient_checkpointing_layers, int) or isinstance(gradient_checkpointing_layers, bool) or gradient_checkpointing_layers < 1):
        raise TypeError("local.gradient_checkpointing_layers must be a positive integer")
    if gradient_checkpointing_layers is not None and not gradient_checkpointing:
        raise ValueError("local.gradient_checkpointing_layers requires local.gradient_checkpointing=true")
    vllm_max_num_seqs = local.get("vllm_max_num_seqs")
    if vllm_max_num_seqs is not None and (not isinstance(vllm_max_num_seqs, int) or isinstance(vllm_max_num_seqs, bool) or vllm_max_num_seqs < 1):
        raise TypeError("local.vllm_max_num_seqs must be a positive integer")
    vllm_max_num_batched_tokens = local.get("vllm_max_num_batched_tokens")
    if vllm_max_num_batched_tokens is not None and (not isinstance(vllm_max_num_batched_tokens, int) or isinstance(vllm_max_num_batched_tokens, bool) or vllm_max_num_batched_tokens < 1):
        raise TypeError("local.vllm_max_num_batched_tokens must be a positive integer")
    vllm_max_model_len = local.get("vllm_max_model_len")
    if vllm_max_model_len is not None and (not isinstance(vllm_max_model_len, int) or isinstance(vllm_max_model_len, bool) or vllm_max_model_len < 1):
        raise TypeError("local.vllm_max_model_len must be a positive integer")
    vllm_gdn_prefill_backend = local.get("vllm_gdn_prefill_backend")
    if vllm_gdn_prefill_backend is not None and vllm_gdn_prefill_backend not in {"flashinfer", "triton"}:
        raise ValueError("local.vllm_gdn_prefill_backend must be 'flashinfer' or 'triton'")
    forward_microbatch_max_datums = local.get("forward_microbatch_max_datums", 8)
    forward_microbatch_max_tokens = local.get("forward_microbatch_max_tokens", 2048)
    target_logprob_chunk_size = local.get("target_logprob_chunk_size", 32)
    for field, value in (
        ("forward_microbatch_max_datums", forward_microbatch_max_datums),
        ("forward_microbatch_max_tokens", forward_microbatch_max_tokens),
        ("target_logprob_chunk_size", target_logprob_chunk_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise TypeError(f"local.{field} must be a positive integer")
    sft = _section(
        spec["supervised_consistency"],
        "supervised_consistency",
        {
            "batch_size",
            "gradient_accumulation_steps",
            "epochs",
            "save_every",
            "learning_rate_schedule",
            "bias_augmented_targets",
            "method_config",
        },
        {"minimum_optimizer_steps"},
    )
    target_generation = _section(
        sft["bias_augmented_targets"],
        "supervised_consistency.bias_augmented_targets",
        {"max_tokens", "temperature", "max_concurrency"},
    )
    method_configs = _section(sft["method_config"], "supervised_consistency.method_config", {"act", "attct", "mlpct"})
    if "minimum_optimizer_steps" in sft:
        minimum_optimizer_steps = sft["minimum_optimizer_steps"]
        if (
            isinstance(minimum_optimizer_steps, bool)
            or not isinstance(minimum_optimizer_steps, int)
            or minimum_optimizer_steps < 1
        ):
            raise ValueError("supervised_consistency.minimum_optimizer_steps must be a positive integer")
    opct: dict[str, Any] | None = None
    opct_rates: list[dict[str, Any]] = []
    if "opct" in spec:
        opct = _section(
            spec["opct"],
            "opct",
            {
                "learning_rates",
                "batch_size",
                "gradient_accumulation_steps",
                "epochs",
                "rollouts_per_prompt",
                "temperature",
                "max_new_tokens",
                "learning_rate_schedule",
                "kl_coefficient",
                "kl_discount_factor",
                "loss",
                "checkpoint_every",
            },
            {"examples"},
        )
        opct_rates = _named_items(opct["learning_rates"], "opct.learning_rates", {"name", "value"})
        for index, rate in enumerate(opct_rates):
            value = rate["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"opct.learning_rates[{index}].value must be a finite positive number")
        for field in (
            "batch_size",
            "gradient_accumulation_steps",
            "epochs",
            "rollouts_per_prompt",
            "max_new_tokens",
            "checkpoint_every",
        ):
            value = opct[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"opct.{field} must be a positive integer")
        if "examples" in opct:
            value = opct["examples"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("opct.examples must be a positive integer")
        temperature = opct["temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("opct.temperature must be a finite non-negative number")
        kl_coefficient = opct["kl_coefficient"]
        if isinstance(kl_coefficient, bool) or not isinstance(kl_coefficient, (int, float)) or not math.isfinite(kl_coefficient) or kl_coefficient <= 0:
            raise ValueError("opct.kl_coefficient must be a finite positive number")
        kl_discount_factor = opct["kl_discount_factor"]
        if isinstance(kl_discount_factor, bool) or not isinstance(kl_discount_factor, (int, float)) or not math.isfinite(kl_discount_factor) or not 0 <= kl_discount_factor <= 1:
            raise ValueError("opct.kl_discount_factor must be a finite number in [0, 1]")
        if opct["learning_rate_schedule"] not in {"constant", "linear", "cosine"}:
            raise ValueError("opct.learning_rate_schedule must be constant, linear, or cosine")
        if opct["loss"] not in {"importance_sampling", "ppo"}:
            raise ValueError("opct.loss must be importance_sampling or ppo")
    elif any(condition["method"] == "opct" for condition in conditions):
        raise ValueError("an opct condition requires an opct configuration")
    rm = _section(
        spec["rate_matching"],
        "rate_matching",
        {
            "datapoints",
            "rollouts",
            "batch_size",
            "epochs",
            "temperature",
            "max_new_tokens",
            "learning_rate_schedule",
            "kl_coefficient",
            "anchor_weight",
            "anchor_model",
            "loss",
            "advantage_estimator",
            "normalization",
            "gradient_accumulation_steps",
            "refresh_every",
            "checkpoint_every",
        },
        {"save_state", "resume_from", "resume_with_optimizer", "resume_state_required"},
    )
    rollouts = _section(rm["rollouts"], "rate_matching.rollouts", {"reference", "training", "consistency", "anchor"})
    save_state = rm.get("save_state", False)
    if not isinstance(save_state, bool):
        raise TypeError("rate_matching.save_state must be a boolean")
    resume_from = rm.get("resume_from")
    resume_with_optimizer = rm.get("resume_with_optimizer", False)
    resume_state_required = rm.get("resume_state_required", False)
    if resume_from is None:
        if "resume_with_optimizer" in rm or "resume_state_required" in rm:
            raise ValueError(
                "rate_matching.resume_with_optimizer and resume_state_required require rate_matching.resume_from"
            )
    else:
        if not isinstance(resume_from, str) or not resume_from.strip():
            raise TypeError("rate_matching.resume_from must be a non-empty checkpoint URI/path string")
        if not isinstance(resume_with_optimizer, bool) or not resume_with_optimizer:
            raise ValueError("rate_matching.resume_from requires resume_with_optimizer: true")
        if not isinstance(resume_state_required, bool) or not resume_state_required:
            raise ValueError(
                "rate_matching.resume_from requires resume_state_required: true; "
                "optimizer-only on-policy continuation is intentionally not compiled"
            )
        if not save_state:
            raise ValueError("rate_matching.resume_from requires save_state: true")
    evaluation = _section(
        spec["evaluation"],
        "evaluation",
        {"max_tokens", "temperature"},
        {
            "top_p",
            "top_k",
            "provider",
            "model_args",
            "max_tasks",
            "isolate_tasks",
            "persistent_vllm_server",
        },
    )
    evaluation_provider = evaluation.get("provider", "hf")
    if evaluation_provider not in {"hf", "vllm"}:
        raise ValueError("evaluation.provider must be hf or vllm")
    evaluation_model_args = evaluation.get("model_args", {})
    if not isinstance(evaluation_model_args, Mapping):
        raise TypeError("evaluation.model_args must be an object")
    evaluation_model_args = dict(evaluation_model_args)
    if "provider" in evaluation_model_args:
        raise ValueError("evaluation.model_args.provider is reserved; use evaluation.provider")
    evaluation_max_tasks = evaluation.get("max_tasks")
    if evaluation_max_tasks is not None and (not isinstance(evaluation_max_tasks, int) or isinstance(evaluation_max_tasks, bool) or evaluation_max_tasks < 1):
        raise TypeError("evaluation.max_tasks must be a positive integer")
    evaluation_isolate_tasks = evaluation.get("isolate_tasks")
    if evaluation_isolate_tasks is not None and not isinstance(evaluation_isolate_tasks, bool):
        raise TypeError("evaluation.isolate_tasks must be a boolean")
    evaluation_persistent_vllm_server = evaluation.get("persistent_vllm_server")
    if evaluation_persistent_vllm_server is not None and not isinstance(evaluation_persistent_vllm_server, bool):
        raise TypeError("evaluation.persistent_vllm_server must be a boolean")
    if evaluation_persistent_vllm_server:
        if not evaluation_isolate_tasks:
            raise ValueError("evaluation.persistent_vllm_server requires evaluation.isolate_tasks: true")
        if evaluation_provider != "vllm":
            raise ValueError("evaluation.persistent_vllm_server requires evaluation.provider: vllm")
    if "top_p" in evaluation:
        top_p = evaluation["top_p"]
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("evaluation.top_p must be a finite number in (0, 1]")
    if "top_k" in evaluation:
        top_k = evaluation["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("evaluation.top_k must be a positive integer")
    tracking = _section(spec["tracking"], "tracking", {"wandb_project"})
    reports = _section(
        spec["reports"],
        "reports",
        {"held_out_exclude", "standard_error", "charts", "items"},
        {"significance_baseline"},
    )
    if reports["standard_error"] not in {"inspect", "sample", "binomial"}:
        raise ValueError("reports.standard_error must be inspect, sample, or binomial")
    charts = reports["charts"]
    if not isinstance(charts, Mapping):
        raise TypeError("reports.charts must be an object")
    report_items = _named_items(
        reports["items"],
        "reports.items",
        {"name", "metric", "chart"},
        {"given", "where", "variant", "held_out_summary", "ratio", "ratio_baseline"},
    )

    model, backend, seed = spec["model"], spec["backend"], spec["seed"]
    root, figure_root = str(spec["artifact_root"]).rstrip("/"), str(spec["figure_root"]).rstrip("/")
    shared_artifact_root = str(shared_root).rstrip("/") if shared_root is not None else root
    shared_data_root = f"{shared_artifact_root}/data"
    model_data_root = f"{root}/data"
    eval_root, log_root = f"{shared_artifact_root}/mcq-bias-evaluation", "logs/evals/${experiment}"
    paths = {
        "hle": f"{shared_data_root}/hle-text-mc.jsonl",
        "hle_manifest": f"{shared_data_root}/hle-text-mc.manifest.json",
        "pairs": training_pairs_path or f"{shared_data_root}/distractor-argument-pairs.jsonl",
        "pairs_manifest": f"{shared_data_root}/distractor-argument-pairs.manifest.json",
        "consistency_pairs": f"{model_data_root}/canonical-consistency-pairs.jsonl",
        "consistency_pairs_manifest": f"{model_data_root}/canonical-consistency-pairs.manifest.json",
        "prompts": f"{shared_data_root}/cleaned-alpaca-prompts.jsonl",
        "prompts_manifest": f"{shared_data_root}/cleaned-alpaca-prompts.manifest.json",
        "bct": f"{model_data_root}/bias-augmented-consistency.jsonl",
        "bct_control": f"{model_data_root}/bias-augmented-consistency-control.jsonl",
        "bct_manifest": f"{model_data_root}/bias-augmented-consistency.manifest.json",
        "instruction": f"{model_data_root}/instruction-targets.jsonl",
        "instruction_control": f"{model_data_root}/instruction-targets-control.jsonl",
        "instruction_manifest": f"{model_data_root}/instruction-targets.manifest.json",
    }
    lora_config = {**lora, "seed": seed}
    local_args = {
        "backend": backend,
        "local_dtype": local["dtype"],
        "local_sampler": local["sampler"],
        "local_gpu_mem_util": local["gpu_memory_utilization"],
        **({"local_rollout_gpu_mem_util": local["rollout_gpu_memory_utilization"]} if "rollout_gpu_memory_utilization" in local else {}),
        **({"local_rollout_seed_base": rollout_seed_base} if rollout_seed_base is not None else {}),
        "local_vllm_language_model_only": local.get("vllm_language_model_only", False),
        "local_gradient_checkpointing": gradient_checkpointing,
        **({"local_gradient_checkpointing_layers": gradient_checkpointing_layers} if gradient_checkpointing_layers is not None else {}),
        **({"local_vllm_max_num_seqs": vllm_max_num_seqs} if vllm_max_num_seqs is not None else {}),
        **({"local_vllm_max_num_batched_tokens": vllm_max_num_batched_tokens} if vllm_max_num_batched_tokens is not None else {}),
        **({"local_vllm_max_model_len": vllm_max_model_len} if vllm_max_model_len is not None else {}),
        **(
            {"local_vllm_gdn_prefill_backend": vllm_gdn_prefill_backend}
            if vllm_gdn_prefill_backend is not None
            else {}
        ),
        "local_forward_microbatch_max_datums": forward_microbatch_max_datums,
        "local_forward_microbatch_max_tokens": forward_microbatch_max_tokens,
        "local_target_logprob_chunk_size": target_logprob_chunk_size,
    }
    yes = {"yes": True}

    data_generation = (
        [
            {
                "name": "hle-source",
                "command": ["${python}", "-m", "scripts.rmct_paper_vast_more_methods.hle_source"],
                "args": {
                    "output": paths["hle"],
                    "manifest_output": paths["hle_manifest"],
                    "expected_count": eval_data["expected_source_count"],
                    **yes,
                },
            },
            {
                "name": "distractor-argument-pairs",
                "command": ["${python}", "-m", "ctm_data.adapters.mcq_bias.materialize"],
                "args": {
                    "bias_type": train_data["bias_type"],
                    "datasets": train_data["datasets"],
                    "prompt_style": train_data["prompt_style"],
                    "n_questions": train_data["pool_per_dataset"],
                    "min_n_questions": train_data["minimum_per_dataset"],
                    "seed": str(seed),
                    "argument_model": train_data["argument_model"],
                    "generate_missing_arguments": True,
                    "dataset_dir": f"{shared_data_root}/mcq-bias-train",
                    "output": paths["pairs"],
                    "manifest_output": paths["pairs_manifest"],
                    **yes,
                },
            },
            {
                "name": "cleaned-alpaca-prompts",
                "command": ["${python}", "-m", "scripts.rmct_paper_vast_more_methods.cleaned_alpaca_source"],
                "args": {
                    "output": paths["prompts"],
                    "manifest_output": paths["prompts_manifest"],
                    "count": instruction["examples"],
                    "seed": str(seed),
                    **yes,
                },
            },
        ]
        if prepare_shared and not training_only
        else []
    )
    target_common = {
        **local_args,
        "model": model,
        "max_tokens": target_generation["max_tokens"],
        "temperature": target_generation["temperature"],
        "max_concurrency": target_generation["max_concurrency"],
        **yes,
    }
    shared_data_preparation = (
        [
            {
                "name": "evaluation-suite",
                "resource": "cpu",
                "command": ["${python}", "-m", "ctm_data.adapters.mcq_bias.materialize_eval"],
                "args": {
                    "bias_types": eval_data["biases"],
                    "datasets": [paths["hle"]],
                    "prompt_style": "none",
                    "n_questions": eval_data["questions"],
                    **({"min_n_questions": eval_min_questions} if eval_min_questions < eval_questions else {}),
                    "seed": str(seed),
                    "argument_model": eval_data["argument_model"],
                    "generate_missing_arguments": True,
                    "dataset_dir": eval_root,
                    **yes,
                },
            }
        ]
        if prepare_shared
        else []
    )
    data_preparation = [
        *shared_data_preparation,
        {
            "name": "canonical-consistency-pairs",
            "resource": "cpu",
            "command": ["${python}", "-m", "ctm_data.adapters.mcq_bias.consistency_pairs"],
            "args": {
                "source": paths["pairs"],
                "output": paths["consistency_pairs"],
                "manifest_output": paths["consistency_pairs_manifest"],
                **yes,
            },
        },
        {
            "name": "bias-augmented-consistency-targets",
            "command": ["${python}", "scripts/prepare_bct_targets.py"],
            "args": {
                **target_common,
                "data": [paths["pairs"]],
                "limit": train_data["examples"],
                "source_messages_field": "unbiased_messages",
                "main_messages_field": "biased_messages",
                "control_messages_field": "unbiased_messages",
                "main_output": paths["bct"],
                "control_output": paths["bct_control"],
                "manifest_output": paths["bct_manifest"],
            },
        },
        {
            "name": "instruction-targets",
            "command": ["${python}", "scripts/prepare_bct_targets.py"],
            "args": {
                **target_common,
                "data": [paths["prompts"]],
                "limit": instruction["examples"],
                "source_messages_field": "reference_messages",
                "main_messages_field": "variant_messages",
                "control_messages_field": "reference_messages",
                "main_output": paths["instruction"],
                "control_output": paths["instruction_control"],
                "manifest_output": paths["instruction_manifest"],
            },
        },
    ]
    if training_only:
        # RMCT and OPCT consume the immutable shared pair store directly.
        # Do not leave unrelated target generation runnable in a recovery plan.
        data_preparation = []

    task_args = {
        "bias_types": eval_data["biases"],
        "datasets": [paths["hle"]],
        "prompt_style": "none",
        "n_questions": eval_data["questions"],
        **({"min_n_questions": eval_min_questions} if eval_min_questions < eval_questions else {}),
        "seed": str(seed),
        "argument_model": eval_data["argument_model"],
        "generate_missing_arguments": False,
        "dataset_dir": eval_root,
        "grader_model": eval_data["grader_model"],
        "include_bias_acknowledged": True,
    }
    generation_config = {
        "max_tokens": evaluation["max_tokens"],
        "temperature": evaluation["temperature"],
        **({"top_p": evaluation["top_p"]} if "top_p" in evaluation else {}),
        **({"top_k": evaluation["top_k"]} if "top_k" in evaluation else {}),
        **(
            {"extra_body": {"top_k": evaluation["top_k"]}}
            if evaluation_provider == "vllm" and "top_k" in evaluation
            else {}
        ),
    }
    eval_common = {
        "task_factory": "ctm_data.adapters.mcq_bias.tasks:suite_tasks",
        "base_model": model,
        "generation_config": generation_config,
        "model_args": {"provider": evaluation_provider, **evaluation_model_args},
        "max_tasks": evaluation_max_tasks,
        **({"isolate_tasks": evaluation_isolate_tasks} if evaluation_isolate_tasks is not None else {}),
        **({"persistent_vllm_server": evaluation_persistent_vllm_server} if evaluation_persistent_vllm_server is not None else {}),
        **yes,
    }
    training: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    analysis_runs: list[str] = []
    for condition in conditions:
        condition_name, method, control = condition["name"], condition["method"], condition.get("control", False)
        if method == "none":
            if not training_only:
                log_dir = f"{log_root}/{condition_name}"
                evals.append(
                    {
                        "name": condition_name,
                        "command": ["${python}", "scripts/run_evals.py"],
                        "args": {
                            **{key: value for key, value in eval_common.items() if key not in {"base_model", "model_args"}},
                            "model": f"{evaluation_provider}/{model}",
                            "model_args": evaluation_model_args,
                            "task_args": {**task_args, "unbiased_log": log_dir},
                            "log_dir": log_dir,
                        },
                    }
                )
                analysis_runs.append(f"{condition_name}={log_dir}")
            continue

        condition_rates = opct_rates if method == "opct" else rates
        for rate_index, rate in enumerate(condition_rates, start=1):
            command_name = f"{_slug(condition_name)}_lr{rate_index}"
            if method in {"bias_augmented_consistency", "act", "attct", "mlpct"}:
                bct_path = paths["bct_control"] if control else paths["bct"]
                instruction_path = paths["instruction_control"] if control else paths["instruction"]
                args = {
                    **local_args,
                    "model": model,
                    "batch_size": sft["batch_size"],
                    "gradient_accumulation_steps": sft["gradient_accumulation_steps"],
                    "epochs": sft["epochs"],
                    "lora_config": lora_config,
                    "optimizer_config": {"learning_rate": rate["value"], "lr_schedule": sft["learning_rate_schedule"]},
                    "save_every": sft["save_every"],
                    **({"minimum_optimizer_steps": sft["minimum_optimizer_steps"]} if "minimum_optimizer_steps" in sft else {}),
                    "experiment_name": "${experiment}",
                    "wandb_project": tracking["wandb_project"],
                    **yes,
                }
                if method == "bias_augmented_consistency":
                    args.update(
                        {
                            "method": "bct",
                            "data": [
                                f"{bct_path}:{train_data['examples']}",
                                f"{instruction_path}:{instruction['examples']}",
                            ],
                            "data_manifest": [paths["bct_manifest"], paths["instruction_manifest"]],
                            "interleave": True,
                        }
                    )
                else:
                    args.update(
                        {
                            "method": method,
                            "method_config": method_configs[method],
                            "data": [f"{paths['consistency_pairs']}:{train_data['examples']}"],
                            "data_manifest": [paths["consistency_pairs_manifest"]],
                            "reference_messages_field": "unbiased_messages",
                            "variant_messages_field": "biased_messages",
                        }
                    )
                command = ["${python}", "scripts/train_bct.py"]
            elif method == "opct":
                assert opct is not None  # Validated before expanding any commands.
                args = {
                    **local_args,
                    "model": model,
                    "data": [f"{paths['pairs']}:{opct.get('examples', train_data['examples'])}"],
                    "data_manifest": [paths["pairs_manifest"]],
                    "reference_messages_field": "unbiased_messages",
                    "variant_messages_field": "unbiased_messages" if control else "biased_messages",
                    "experiment_name": "${experiment}",
                    "seed": seed,
                    "lora_config": lora_config,
                    "lr": rate["value"],
                    "lr_schedule": opct["learning_rate_schedule"],
                    "kl_coef": opct["kl_coefficient"],
                    "kl_discount_factor": opct["kl_discount_factor"],
                    "loss_fn": opct["loss"],
                    "rollouts_per_prompt": opct["rollouts_per_prompt"],
                    "temperature": opct["temperature"],
                    "max_new_tokens": opct["max_new_tokens"],
                    "batch_size": opct["batch_size"],
                    "gradient_accumulation_steps": opct["gradient_accumulation_steps"],
                    "epochs": opct["epochs"],
                    "checkpoint_every": opct["checkpoint_every"],
                    "wandb_project": tracking["wandb_project"],
                    **(
                        {"require_onpolicy_target_attestation": True}
                        if require_onpolicy_target_attestation
                        else {}
                    ),
                    **yes,
                }
                command = ["${python}", "scripts/train_opct.py"]
            else:
                setting_config = {"data_paths": [paths["pairs"]], **({"control": True} if control else {})}
                args = {
                    **local_args,
                    "model": model,
                    "setting_factory": "ctm_data.adapters.mcq_bias:create_setting",
                    "load_config": {
                        "n_datapoints": rm["datapoints"],
                        **(
                            {"selection_manifest": training_selection_manifest}
                            if training_selection_manifest is not None
                            else {}
                        ),
                        **(
                            {"continuation_manifest": training_continuation_manifest}
                            if training_continuation_manifest is not None
                            else {}
                        ),
                    },
                    "setting_config": setting_config,
                    "experiment_name": "${experiment}",
                    "seed": seed,
                    "lora_config": lora_config,
                    "lr": rate["value"],
                    "lr_schedule": rm["learning_rate_schedule"],
                    "kl_coef": rm["kl_coefficient"],
                    "anchor_weight": rm["anchor_weight"],
                    "anchor_model": rm["anchor_model"],
                    "loss_fn": rm["loss"],
                    "advantage_estimator": rm["advantage_estimator"],
                    "normalization": rm["normalization"],
                    "n_ref_rollouts": rollouts["reference"],
                    "n_train_rollouts": rollouts["training"],
                    "n_consistency_rollouts": rollouts["consistency"],
                    "n_anchor_rollouts": rollouts["anchor"],
                    "temperature": rm["temperature"],
                    "max_new_tokens": rm["max_new_tokens"],
                    "batch_size": rm["batch_size"],
                    "gradient_accumulation_steps": rm["gradient_accumulation_steps"],
                    "refresh_every": rm["refresh_every"],
                    "n_epochs": rm["epochs"],
                    "checkpoint_every": rm["checkpoint_every"],
                    # Saving optimizer state is useful custody/recovery evidence,
                    # but intentionally does not imply an exact on-policy resume:
                    # rollout-worker/RNG state is separately attested and is not
                    # restored by this flag alone.
                    **({"save_state": True} if save_state else {}),
                    **({"resume_from": resume_from} if resume_from is not None else {}),
                    **({"resume_with_optimizer": True, "resume_state_required": True} if resume_from is not None else {}),
                    "wandb_project": tracking["wandb_project"],
                    **(
                        {"require_onpolicy_target_attestation": True}
                        if require_onpolicy_target_attestation
                        else {}
                    ),
                    **yes,
                }
                command = ["${python}", "scripts/train_rlct.py"]
            args["run_name"] = f"{condition_name}-lr-{rate['name']}"
            training.append({"name": command_name, "command": command, "args": args})
            if not training_only:
                log_dir = f"{log_root}/{condition_name}/lr{rate_index}"
                evals.append(
                    {
                        "name": f"{condition_name}-lr{rate_index}",
                        "command": ["${python}", "scripts/run_evals.py"],
                        "args": {
                            **eval_common,
                            "local_checkpoint": f"${{training.{command_name}.checkpoint}}",
                            "task_args": {**task_args, "unbiased_log": log_dir},
                            "log_dir": log_dir,
                        },
                    }
                )
                analysis_runs.append(f"{condition_name}={log_dir}")

    analysis: list[dict[str, Any]] = []
    rendering: list[dict[str, Any]] = []
    condition_metadata = {
        condition["name"]: {
            "method": condition["method"],
            "is_control": condition.get("control", False),
            **({"control_for": condition["method"]} if condition.get("control", False) else {}),
        }
        for condition in conditions
    }
    report_metadata = {
        "schema_version": 2,
        "model": model,
        "prompt_style": "none",
        "training_biases": [train_data["bias_type"]],
        "training_regime": train_data["bias_type"],
    }
    for report in report_items:
        if report["chart"] not in charts:
            raise ValueError(f"report {report['name']!r} refers to unknown chart {report['chart']!r}")
        result, figure = f"{root}/results/{report['name']}.json", f"{figure_root}/{report['name']}.svg"
        variant = report.get("variant", "biased")
        if variant not in {"biased", "unbiased"}:
            raise ValueError("report variant must be biased or unbiased")
        held_out_summary = report.get("held_out_summary", variant == "biased")
        if not isinstance(held_out_summary, bool):
            raise TypeError("report held_out_summary must be a boolean")
        ratio = report.get("ratio", False)
        if not isinstance(ratio, bool):
            raise TypeError("report ratio must be a boolean")
        ratio_baseline = report.get("ratio_baseline", reports.get("significance_baseline")) if ratio else None
        if ratio and not isinstance(ratio_baseline, str):
            raise ValueError("a ratio report requires ratio_baseline or reports.significance_baseline")
        args = {
            "run": analysis_runs,
            "metric": _metric(report["metric"]),
            "stderr": reports["standard_error"],
            "variant": variant,
            "metadata": {**report_metadata, "variant": variant},
            "condition_metadata": condition_metadata,
            "held_out_exclude": reports["held_out_exclude"] if held_out_summary else None,
            "significance_baseline": reports.get("significance_baseline") if not ratio else None,
            "ratio_baseline": ratio_baseline,
            # Unbiased logs have no bias facet. Omitting the constraint is both
            # semantically correct and avoids emitting an invalid empty CLI value.
            "expected_biases": eval_data["biases"] if variant == "biased" else None,
            "expected_datasets": [paths["hle"]],
            "output": result,
            **yes,
        }
        if "given" in report:
            if "where" in report:
                raise ValueError("report may use given or where, not both")
            if report["given"] not in {"towards_bias_switch", "total_bias_switch"}:
                raise ValueError("report given must be towards_bias_switch or total_bias_switch")
            args.update({"where_metric": _metric(report["given"]), "where_value": 1.0})
        elif "where" in report:
            if not isinstance(report["where"], Mapping):
                raise ValueError("report where must be an object")
            args["where"] = report["where"]
        analysis.append(
            {
                "name": f"aggregate-{report['name']}",
                "command": ["${python}", "-m", "ctm_data.adapters.mcq_bias.analysis"],
                "args": args,
            }
        )
        rendering.append(
            {
                "name": f"render-{report['name']}",
                "command": ["${python}", "-m", "ctm_data.adapters.mcq_bias.plot"],
                "args": {"data": result, "spec": charts[report["chart"]], "output": figure},
            }
        )

    if training_only:
        # A raw Qwen3.5 checkpoint must never be accidentally sent through
        # the vLLM evaluator. A separate post-training compatibility/attestation
        # handoff owns evaluation and report construction.
        analysis = []
        rendering = []

    stages = {
        "data_generation": data_generation,
        "data_preparation": data_preparation,
        "training": training,
        "evaluation": evals,
        "analysis": analysis,
        "rendering": rendering,
    }
    publication = _apply_execution_allocations(stages, spec.get("execution"))
    return {
        "name": name,
        **({"training_output_publication": publication} if publication is not None else {}),
        **{stage: commands for stage, commands in stages.items() if commands},
    }


__all__ = ["compile_experiment"]
