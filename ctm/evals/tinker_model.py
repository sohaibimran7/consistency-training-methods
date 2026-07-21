"""Validated Tinker-checkpoint resolution for Inspect evaluation.

The actual ModelAPI adapter is maintained by ``tinker-cookbook``. This module
only resolves checkpoint-owned base-model/renderer metadata and refuses a
caller-supplied mismatch before constructing that adapter.
"""

from __future__ import annotations

from typing import Optional

import tinker
from inspect_ai.model import GenerateConfig, Model
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling


def tinker_checkpoint_model(
    checkpoint: str,
    *,
    base_model: str | None = None,
    renderer_name: str | None = None,
    config: GenerateConfig | None = None,
    include_reasoning: bool = False,
    service_client: Optional[tinker.ServiceClient] = None,
) -> Model:
    """Resolve checkpoint metadata and return the cookbook's Inspect adapter."""

    if not checkpoint.startswith("tinker://"):
        raise ValueError("Tinker checkpoint must start with 'tinker://'")
    client = service_client or tinker.ServiceClient()
    try:
        training_run = client.create_rest_client().get_training_run_by_tinker_path(checkpoint).result()
    except Exception as exc:  # noqa: BLE001 - SDK versions expose several provider error types
        raise RuntimeError(f"could not resolve Tinker checkpoint metadata for {checkpoint}: {exc}") from exc

    checkpoint_base_model = getattr(training_run, "base_model", None)
    if not isinstance(checkpoint_base_model, str) or not checkpoint_base_model:
        raise ValueError(f"Tinker training run for {checkpoint} has no base_model metadata")
    if base_model is not None and base_model != checkpoint_base_model:
        raise ValueError(f"base_model {base_model!r} does not match checkpoint metadata {checkpoint_base_model!r}")
    resolved_base_model = checkpoint_base_model

    user_metadata = getattr(training_run, "user_metadata", None) or {}
    checkpoint_renderer = user_metadata.get(checkpoint_utils.RENDERER_NAME_METADATA_KEY)
    if renderer_name is not None and checkpoint_renderer is not None and renderer_name != checkpoint_renderer:
        raise ValueError(f"renderer_name {renderer_name!r} does not match checkpoint metadata {checkpoint_renderer!r}")
    if renderer_name is None and checkpoint_renderer is None:
        raise ValueError(
            "checkpoint has no renderer metadata; pass renderer_name explicitly instead of relying on a "
            "potentially drifted current default"
        )
    resolved_renderer = renderer_name or checkpoint_renderer
    assert resolved_renderer is not None

    generate_config = config or GenerateConfig()
    sampling_client = client.create_sampling_client(model_path=checkpoint, base_model=resolved_base_model)
    api = InspectAPIFromTinkerSampling(
        renderer_name=resolved_renderer,
        model_name=resolved_base_model,
        sampling_client=sampling_client,
        config=generate_config,
        include_reasoning=include_reasoning,
    )
    # Expose the resolved checkpoint identity to the generic runner's metadata.
    api.checkpoint = checkpoint
    api.renderer_name = resolved_renderer
    return Model(api=api, config=generate_config)


def tinker_base_model(
    base_model: str,
    *,
    renderer_name: str | None = None,
    config: GenerateConfig | None = None,
    include_reasoning: bool = False,
    service_client: Optional[tinker.ServiceClient] = None,
) -> Model:
    """Return the cookbook Inspect adapter for an untrained Tinker base model."""

    if not isinstance(base_model, str) or not base_model:
        raise ValueError("Tinker base model must be a non-empty string")
    resolved_renderer = renderer_name or model_info.get_recommended_renderer_name(base_model)
    generate_config = config or GenerateConfig()
    client = service_client or tinker.ServiceClient()
    sampling_client = client.create_sampling_client(base_model=base_model)
    api = InspectAPIFromTinkerSampling(
        renderer_name=resolved_renderer,
        model_name=base_model,
        sampling_client=sampling_client,
        config=generate_config,
        include_reasoning=include_reasoning,
    )
    api.renderer_name = resolved_renderer
    return Model(api=api, config=generate_config)


__all__ = ["tinker_base_model", "tinker_checkpoint_model"]
