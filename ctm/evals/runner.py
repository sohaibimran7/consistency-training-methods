"""Run an upstream Inspect task factory against a model or checkpoint."""

from __future__ import annotations

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


def validate_tinker_generation_config(value: Mapping[str, Any] | None) -> None:
    """Reject Inspect options that the cookbook Tinker adapter would ignore."""

    unsupported = sorted(set(value or {}) - TINKER_SUPPORTED_GENERATION_FIELDS)
    if unsupported:
        raise ValueError(
            "Tinker models do not support these generation_config field(s): "
            f"{unsupported}; use only {sorted(TINKER_SUPPORTED_GENERATION_FIELDS)}"
        )


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
    effective_generation_config = normalize_generation_config(generation_config)
    if tinker_base_model_name or tinker_checkpoint:
        if model_args:
            raise ValueError("model_args apply only to ordinary Inspect providers and local checkpoints")
        validate_tinker_generation_config(effective_generation_config)
        from ctm.evals.tinker_model import tinker_base_model, tinker_checkpoint_model
        from inspect_ai.model import GenerateConfig

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
    from inspect_ai.model import GenerateConfig, get_model

    return get_model(
        model,
        config=GenerateConfig(**effective_generation_config),
        **dict(model_args or {}),
    )


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
    metadata: Mapping[str, Any] | None = None,
):
    """Run exactly the requested upstream task factory."""

    import inspect_ai

    tasks = build_tasks(task_factory, task_args=task_args)
    resolved_model = resolve_eval_model(
        model=model,
        tinker_base_model_name=tinker_base_model_name,
        tinker_checkpoint=tinker_checkpoint,
        local_checkpoint=local_checkpoint,
        base_model=base_model,
        renderer_name=renderer_name,
        model_args=model_args,
        generation_config=generation_config,
        include_reasoning=include_reasoning,
    )
    run_metadata = {
        **redact_secrets(dict(metadata or {})),
        "task_factory": task_factory,
        "task_args": redact_secrets(dict(task_args or {})),
        "model_args": redact_secrets(dict(model_args or {})),
        "generation_config": redact_secrets(normalize_generation_config(generation_config)),
        "include_reasoning": include_reasoning,
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
        metadata=run_metadata,
    )
