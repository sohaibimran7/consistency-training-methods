"""Inspect model bridge for LocalBackend LoRA and full-weight checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _checkpoint_directory(value: str | Path) -> Path:
    raw = str(value)
    path = Path(raw.removeprefix("file://")).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"local checkpoint directory does not exist: {path}")
    return path


def read_local_checkpoint(value: str | Path) -> tuple[Path, dict[str, Any]]:
    """Read and validate a LocalBackend checkpoint manifest."""

    directory = _checkpoint_directory(value)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"local checkpoint has no manifest.json: {directory}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid local checkpoint manifest: {manifest_path}") from exc
    if manifest.get("backend") != "local":
        raise ValueError(f"checkpoint manifest backend must be 'local': {manifest_path}")
    if not isinstance(manifest.get("lora"), bool):
        raise ValueError(f"checkpoint manifest must record boolean lora mode: {manifest_path}")
    if not isinstance(manifest.get("model"), str) or not manifest["model"].strip():
        raise ValueError(f"checkpoint manifest has no base model: {manifest_path}")
    if manifest["lora"]:
        if not (directory / "adapter_config.json").is_file():
            raise ValueError(f"local LoRA checkpoint has no adapter_config.json: {directory}")
    elif not (directory / "weights.pt").is_file():
        raise ValueError(f"local full-weight checkpoint has no weights.pt: {directory}")
    return directory, manifest


def local_checkpoint_model(
    checkpoint: str | Path,
    *,
    base_model: str | None = None,
    model_args: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
):
    """Load a LocalBackend checkpoint as an ordinary Inspect HF model."""

    from inspect_ai.model import GenerateConfig, get_model

    directory, manifest = read_local_checkpoint(checkpoint)
    recorded_model = manifest["model"]
    if base_model is not None and base_model != recorded_model:
        raise ValueError(
            f"local checkpoint base model mismatch: manifest records {recorded_model!r}, "
            f"but {base_model!r} was requested"
        )
    options = dict(model_args or {})
    forbidden = sorted(set(options) & {"model_path", "tokenizer", "tokenizer_path"})
    if forbidden:
        raise ValueError(
            f"local checkpoint model_args cannot override {forbidden}; "
            "the checkpoint manifest owns the base model and tokenizer"
        )
    model = get_model(
        f"hf/{recorded_model}",
        config=GenerateConfig(**dict(generation_config or {})),
        **options,
    )
    if not hasattr(model.api, "model"):
        raise TypeError("Inspect's Hugging Face provider did not expose a model instance")
    if manifest["lora"]:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("local LoRA checkpoint evaluation requires peft") from exc
        model.api.model = PeftModel.from_pretrained(model.api.model, str(directory))
    else:
        import torch

        state = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
        model.api.model.load_state_dict(state)
    return model


__all__ = ["local_checkpoint_model", "read_local_checkpoint"]
