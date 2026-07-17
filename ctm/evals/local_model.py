"""Inspect model bridge for LocalBackend LoRA checkpoints."""

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
    """Read and validate a LocalBackend LoRA checkpoint manifest."""

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
    if manifest.get("lora") is not True:
        raise ValueError("local evaluation currently supports LoRA checkpoints only")
    if not isinstance(manifest.get("model"), str) or not manifest["model"].strip():
        raise ValueError(f"checkpoint manifest has no base model: {manifest_path}")
    if not (directory / "adapter_config.json").is_file():
        raise ValueError(f"local LoRA checkpoint has no adapter_config.json: {directory}")
    return directory, manifest


def local_checkpoint_model(
    checkpoint: str | Path,
    *,
    base_model: str | None = None,
    model_args: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
):
    """Load a LocalBackend PEFT adapter as an ordinary Inspect HF model."""

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
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("local checkpoint evaluation requires peft") from exc
    if not hasattr(model.api, "model"):
        raise TypeError("Inspect's Hugging Face provider did not expose a model instance")
    model.api.model = PeftModel.from_pretrained(model.api.model, str(directory))
    return model


__all__ = ["local_checkpoint_model", "read_local_checkpoint"]
