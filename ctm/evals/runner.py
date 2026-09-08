"""Run an upstream Inspect task factory against a model or checkpoint."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

from ctm.cli_safety import parse_json_object as parse_json_object
from ctm.cli_safety import redact_secrets
from ctm.importing import load_callable

DEFAULT_GENERATION_CONFIG: dict[str, Any] = {"temperature": 0.0}
TINKER_SUPPORTED_GENERATION_FIELDS = frozenset(
    {
        # Consumed by Inspect's generic Model wrapper.
        "adaptive_connections",
        "cache",
        "max_connections",
        "max_retries",
        "max_tool_output",
        "reasoning_history",
        "timeout",
        "attempt_timeout",
        # Consumed by tinker-cookbook's InspectAPIFromTinkerSampling.
        "max_tokens",
        "num_choices",
        "seed",
        "system_message",
        "temperature",
        "top_k",
        "top_p",
    }
)


def normalize_generation_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return validated provider-independent generation parameters."""

    from inspect_ai.model import GenerateConfig

    supplied = dict(value or {})
    unknown = sorted(set(supplied) - set(GenerateConfig.model_fields))
    if unknown:
        raise ValueError(f"unknown Inspect GenerateConfig field(s): {unknown}")
    normalized = {**DEFAULT_GENERATION_CONFIG, **supplied}
    GenerateConfig(**normalized)
    return normalized


def effective_provider_generation_config(
    value: Mapping[str, Any] | None,
    *,
    model: str | None = None,
    local_checkpoint: str | None = None,
    model_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact generation config sent to the selected provider.

    Inspect exposes ``top_k`` on :class:`GenerateConfig`, but its native vLLM
    provider is OpenAI-compatible and does not copy that field into the request.
    vLLM accepts it through ``extra_body``. Keep the provider-independent field
    for provenance while also routing it through the provider-supported channel.
    Other providers (notably the Tinker adapter) retain their native handling.
    """

    normalized = normalize_generation_config(value)
    options = dict(model_args or {})
    native_vllm = bool(model and model.startswith("vllm/")) or bool(local_checkpoint and options.get("provider", "hf") == "vllm")
    if not native_vllm or normalized.get("top_k") is None:
        return normalized

    top_k = normalized["top_k"]
    extra_body = dict(normalized.get("extra_body") or {})
    if "top_k" in extra_body and extra_body["top_k"] != top_k:
        raise ValueError(f"native vLLM generation config has conflicting top_k values: top_k={top_k!r}, extra_body.top_k={extra_body['top_k']!r}")
    extra_body["top_k"] = top_k
    normalized["extra_body"] = extra_body
    return normalized


def validate_tinker_generation_config(value: Mapping[str, Any] | None) -> None:
    """Reject Inspect options that the cookbook Tinker adapter would ignore."""

    unsupported = sorted(set(value or {}) - TINKER_SUPPORTED_GENERATION_FIELDS)
    if unsupported:
        raise ValueError(f"Tinker models do not support these generation_config field(s): {unsupported}; use only {sorted(TINKER_SUPPORTED_GENERATION_FIELDS)}")


def resolve_eval_model(
    *,
    model: str | None = None,
    tinker_base_model_name: str | None = None,
    tinker_checkpoint: str | None = None,
    local_checkpoint: str | None = None,
    base_model: str | None = None,
    renderer_name: str | None = None,
    model_args: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
    include_reasoning: bool = False,
):
    """Resolve a provider model, Tinker base/checkpoint, or local checkpoint."""

    if sum(value is not None for value in (model, tinker_base_model_name, tinker_checkpoint, local_checkpoint)) != 1:
        raise ValueError("pass exactly one of model, tinker_base_model_name, tinker_checkpoint, or local_checkpoint")
    if model and base_model is not None:
        raise ValueError("base_model applies only to saved checkpoints")
    if model and include_reasoning:
        raise ValueError("include_reasoning applies only to Tinker models")
    if "config" in dict(model_args or {}):
        raise ValueError("put generation parameters in generation_config, not model_args.config")
    effective_generation_config = effective_provider_generation_config(
        generation_config,
        model=model,
        local_checkpoint=local_checkpoint,
        model_args=model_args,
    )
    if tinker_base_model_name or tinker_checkpoint:
        if model_args:
            raise ValueError("model_args apply only to ordinary Inspect providers and local checkpoints")
        validate_tinker_generation_config(effective_generation_config)
        from inspect_ai.model import GenerateConfig

        from ctm.evals.tinker_model import tinker_base_model, tinker_checkpoint_model

        if tinker_base_model_name:
            if base_model is not None:
                raise ValueError("base_model is redundant with tinker_base_model_name")
            return tinker_base_model(
                tinker_base_model_name,
                renderer_name=renderer_name,
                config=GenerateConfig(**effective_generation_config),
                include_reasoning=include_reasoning,
            )
        return tinker_checkpoint_model(
            tinker_checkpoint,
            base_model=base_model,
            renderer_name=renderer_name,
            config=GenerateConfig(**effective_generation_config),
            include_reasoning=include_reasoning,
        )
    if local_checkpoint:
        if renderer_name is not None:
            raise ValueError("renderer_name applies only to Tinker models")
        if include_reasoning:
            raise ValueError("include_reasoning applies only to Tinker models")
        from ctm.evals.local_model import local_checkpoint_model

        return local_checkpoint_model(
            local_checkpoint,
            base_model=base_model,
            model_args=model_args,
            generation_config=effective_generation_config,
        )
    if renderer_name is not None:
        raise ValueError("renderer_name applies only to Tinker models")
    options = dict(model_args or {})
    hf_language_model_only = options.get("hf_language_model_only", False)
    if not isinstance(hf_language_model_only, bool):
        raise ValueError("hf_language_model_only must be boolean")
    if hf_language_model_only and not str(model).startswith("hf/"):
        raise ValueError("hf_language_model_only requires an explicit hf/... model")
    gemma4_unified_processor = options.get("gemma4_unified_processor", False)
    if not isinstance(gemma4_unified_processor, bool):
        raise TypeError("gemma4_unified_processor must be boolean")
    if gemma4_unified_processor:
        if hf_language_model_only:
            raise ValueError(
                "Gemma 4 unified evaluation must retain its full conditional-generation wrapper; "
                "do not set hf_language_model_only"
            )
        from ctm.evals.local_model import gemma4_unified_hf_model

        return gemma4_unified_hf_model(
            model,
            model_args=options,
            generation_config=effective_generation_config,
        )

    options.pop("hf_language_model_only", None)
    options.pop("gemma4_unified_processor", None)
    from inspect_ai.model import GenerateConfig, get_model

    resolved = get_model(
        model,
        config=GenerateConfig(**effective_generation_config),
        **options,
    )
    if hf_language_model_only:
        from ctm.evals.local_model import detach_muse_glimmer_vision_modules

        api_model = getattr(getattr(resolved, "api", None), "model", None)
        detach_muse_glimmer_vision_modules(api_model)
    return resolved


def load_task_factory(spec: str):
    """Load ``module:callable`` without introducing a benchmark registry."""

    return load_callable(spec, label="task_factory")


def build_tasks(task_factory: str, *, task_args: Mapping[str, Any] | None = None) -> list[Any]:
    """Call an upstream task factory and normalize one task or a task list."""

    result = load_task_factory(task_factory)(**dict(task_args or {}))
    tasks = list(result) if isinstance(result, (list, tuple)) else [result]
    if not tasks or any(task is None for task in tasks):
        raise ValueError(f"task factory {task_factory!r} produced no tasks")
    return tasks


def normalize_task_indices(task_indices: Sequence[int] | None, *, task_count: int) -> list[int]:
    """Validate optional 1-based task positions and preserve their requested order."""

    if task_count < 1:
        raise ValueError("task_count must be positive")
    if task_indices is None:
        return list(range(1, task_count + 1))
    if not task_indices:
        raise ValueError("task_indices must not be empty")

    normalized: list[int] = []
    for index in task_indices:
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise ValueError("task indices are 1-based positive integers")
        if index > task_count:
            raise ValueError(f"task index {index} is out of range for a {task_count}-task suite")
        if index in normalized:
            raise ValueError(f"task index {index} was requested more than once")
        normalized.append(index)
    return normalized


def select_tasks(tasks: Sequence[Any], task_indices: Sequence[int] | None) -> tuple[list[Any], list[int]]:
    """Select original task objects by stable 1-based factory position."""

    indices = normalize_task_indices(task_indices, task_count=len(tasks))
    return [tasks[index - 1] for index in indices], indices


def run_task_evals(
    task_factory: str,
    *,
    model: str | None = None,
    tinker_base_model_name: str | None = None,
    tinker_checkpoint: str | None = None,
    local_checkpoint: str | None = None,
    base_model: str | None = None,
    renderer_name: str | None = None,
    task_args: Mapping[str, Any] | None = None,
    model_args: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
    include_reasoning: bool = False,
    log_dir: str | None = None,
    limit: int | None = None,
    epochs: int | None = None,
    max_tasks: int | None = None,
    task_indices: Sequence[int] | None = None,
    metadata: Mapping[str, Any] | None = None,
):
    """Run exactly the requested upstream task factory."""

    import inspect_ai

    effective_generation_config = effective_provider_generation_config(
        generation_config,
        model=model,
        local_checkpoint=local_checkpoint,
        model_args=model_args,
    )
    all_tasks = build_tasks(task_factory, task_args=task_args)
    tasks, selected_task_indices = select_tasks(all_tasks, task_indices)
    resolved_model = resolve_eval_model(
        model=model,
        tinker_base_model_name=tinker_base_model_name,
        tinker_checkpoint=tinker_checkpoint,
        local_checkpoint=local_checkpoint,
        base_model=base_model,
        renderer_name=renderer_name,
        model_args=model_args,
        generation_config=effective_generation_config,
        include_reasoning=include_reasoning,
    )
    run_metadata = {
        **redact_secrets(dict(metadata or {})),
        "task_factory": task_factory,
        "task_args": redact_secrets(dict(task_args or {})),
        "model_args": redact_secrets(dict(model_args or {})),
        "generation_config": redact_secrets(effective_generation_config),
        "include_reasoning": include_reasoning,
        "max_tasks": max_tasks,
        **(
            {
                "task_indices": selected_task_indices,
                "task_count": len(all_tasks),
            }
            if task_indices is not None
            else {}
        ),
    }
    if tinker_checkpoint:
        run_metadata.update(
            {
                "checkpoint": tinker_checkpoint,
                "base_model": resolved_model.api.model_name,
                "renderer_name": getattr(resolved_model.api, "renderer_name", None),
            }
        )
    elif tinker_base_model_name:
        run_metadata.update(
            {
                "model": tinker_base_model_name,
                "model_backend": "tinker",
                "renderer_name": getattr(resolved_model.api, "renderer_name", None),
            }
        )
    elif local_checkpoint:
        run_metadata.update(
            {
                "checkpoint": local_checkpoint,
                "base_model": resolved_model.api.model_name,
                "checkpoint_backend": "local",
            }
        )
    else:
        run_metadata["model"] = model
    return inspect_ai.eval(
        tasks=tasks,
        model=resolved_model,
        log_dir=log_dir,
        limit=limit,
        epochs=epochs,
        max_tasks=max_tasks,
        metadata=run_metadata,
    )
