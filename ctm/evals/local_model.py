"""Inspect model bridge for LocalBackend LoRA and full-weight checkpoints."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ctm.cli_safety import redact_secrets
from ctm.evals.qwen35_vllm_attestation import is_verified_qwen35_vllm_compat_adapter

PERSISTENT_VLLM_CHILD_METADATA_ENV = "CTM_PERSISTENT_VLLM_SERVER_METADATA"
_DISABLE_CUDNN_SDP_ENV = "CTM_DISABLE_CUDNN_SDP"
_PERSISTENT_VLLM_LOGGERS = (
    "inspect_ai._util.local_server",
    "inspect_ai.model._providers.vllm",
    "inspect_ai.model._providers._vllm_lora",
)

# Gemma 4 is a unified multimodal architecture even for a text-only request.
# Its processor owns the canonical chat template and produces the text inputs
# expected by the conditional-generation wrapper.  Do not substitute its
# nested tokenizer or its text-only language submodel: either substitution can
# change prompt serialization or bypass wrapper-side input preparation.
GEMMA4_UNIFIED_PROCESSOR_MODEL_ARG = "gemma4_unified_processor"
# ``AutoModelForImageTextToText`` is the canonical Transformers route for the
# Gemma4Unified conditional-generation wrapper.  It avoids depending on the
# broader MultimodalLM alias, whose availability has varied across releases.
GEMMA4_UNIFIED_AUTO_MODEL_CLASS = "AutoModelForImageTextToText"
_GEMMA4_UNIFIED_MODEL_TYPES = frozenset({"gemma4_unified", "gemma4"})
_TOKEN_CAP_FIELD_NAMES = frozenset(
    {
        "max_tokens",
        "max_new_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "generation_token_cap",
        "output_token_cap",
        "completion_token_cap",
    }
)


def _normalized_option_name(value: object) -> str:
    """Normalize a config spelling without accepting a concrete cap alias."""

    spelling = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value))
    return re.sub(r"[^a-zA-Z0-9]+", "_", spelling).strip("_").lower()


def _require_no_token_cap(value: Mapping[str, Any] | None, *, label: str) -> None:
    """Fail closed if a Gemma evaluator is asked to use an output cap."""

    for key, nested in dict(value or {}).items():
        if _normalized_option_name(key) in _TOKEN_CAP_FIELD_NAMES and nested is not None:
            raise ValueError(f"{label} contains forbidden output-token cap {key}={nested!r}")


def _is_gemma4_unified_model(model: Any) -> bool:
    """Identify the public Gemma 4 unified wrapper by its configuration."""

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return isinstance(model_type, str) and model_type.lower() in _GEMMA4_UNIFIED_MODEL_TYPES


def _load_gemma4_unified_processor(model_source: str, *, token: str | None = None) -> Any:
    """Load the official processor from the exact model/snapshot source."""

    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_source, token=token)


class Gemma4UnifiedTextProcessor:
    """Expose Gemma 4's canonical processor through Inspect's tokenizer surface.

    Inspect's native HF provider calls its tokenizer positionally, whereas
    ``Gemma4Processor`` reserves the first positional parameter for images.
    This small adapter maps that positional text argument to ``text=`` and
    delegates all chat-template and decoding behavior to the official
    processor.  The text-only evaluator deliberately requests no multimodal
    token-type IDs: all prompts are text-only, and Inspect's provider accepts
    only ``input_ids`` and ``attention_mask`` at its batching boundary.
    """

    def __init__(self, processor: Any) -> None:
        object.__setattr__(self, "processor", processor)

    @property
    def tokenizer(self) -> Any:
        return self.processor.tokenizer

    @property
    def chat_template(self) -> Any:
        return getattr(self.processor, "chat_template", None)

    @chat_template.setter
    def chat_template(self, value: Any) -> None:
        self.processor.chat_template = value

    @property
    def eos_token(self) -> Any:
        return getattr(self.tokenizer, "eos_token", None)

    @property
    def eos_token_id(self) -> Any:
        return getattr(self.tokenizer, "eos_token_id", None)

    @property
    def pad_token(self) -> Any:
        return getattr(self.tokenizer, "pad_token", None)

    @pad_token.setter
    def pad_token(self, value: Any) -> None:
        self.tokenizer.pad_token = value

    @property
    def pad_token_id(self) -> Any:
        return getattr(self.tokenizer, "pad_token_id", None)

    @property
    def padding_side(self) -> Any:
        return getattr(self.tokenizer, "padding_side", None)

    @padding_side.setter
    def padding_side(self, value: Any) -> None:
        self.tokenizer.padding_side = value

    @property
    def name_or_path(self) -> Any:
        return getattr(self.processor, "name_or_path", getattr(self.tokenizer, "name_or_path", None))

    def __call__(self, text: Any = None, /, **kwargs: Any) -> Any:
        requested_mm_types = kwargs.pop("return_mm_token_type_ids", False)
        if requested_mm_types is not False:
            raise ValueError(
                "Gemma 4 unified text evaluation accepts text-only processor inputs; "
                "return_mm_token_type_ids must be false"
            )
        return self.processor(text=text, return_mm_token_type_ids=False, **kwargs)

    @staticmethod
    def _chat_message_payload(message: Any) -> dict[str, Any]:
        """Convert Inspect chat models to the mapping shape Gemma expects."""

        if isinstance(message, Mapping):
            payload = dict(message)
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if role is None or content is None:
                raise TypeError(
                    "Gemma 4 unified chat templates require messages with concrete role and content fields"
                )
            payload = {"role": role, "content": content}
        if not isinstance(payload.get("role"), str) or not isinstance(payload.get("content"), str):
            raise TypeError(
                "Gemma 4 unified text evaluation requires string role/content chat messages"
            )
        return payload

    @classmethod
    def _chat_conversation_payload(cls, conversation: Any) -> Any:
        """Normalize one text-only conversation, or a batch of conversations."""

        if not isinstance(conversation, (list, tuple)):
            raise TypeError("Gemma 4 unified chat template requires a list or tuple of messages")
        if not conversation:
            return []
        if all(isinstance(item, (list, tuple)) for item in conversation):
            return [cls._chat_conversation_payload(item) for item in conversation]
        return [cls._chat_message_payload(item) for item in conversation]

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        # Inspect AI's HF provider passes its ChatMessage pydantic models
        # directly.  ProcessorMixin expects plain mappings and otherwise
        # iterates a pydantic model into ``(field, value)`` tuples before
        # calling ``message.get``.  Convert only this structural boundary;
        # the official Gemma processor still owns template rendering.
        positional = list(args)
        if positional:
            positional[0] = self._chat_conversation_payload(positional[0])
        elif "conversation" in kwargs:
            kwargs = dict(kwargs)
            kwargs["conversation"] = self._chat_conversation_payload(kwargs["conversation"])
        else:
            raise TypeError("Gemma 4 unified chat template requires a conversation")
        return self.processor.apply_chat_template(*positional, **kwargs)

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.processor.batch_decode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.processor.decode(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Prefer processor-owned behavior, then tokenizer compatibility APIs."""

        try:
            return getattr(self.processor, name)
        except AttributeError:
            return getattr(self.tokenizer, name)


def configure_gemma4_unified_hf_api(api: Any, *, model_source: str | None = None) -> Gemma4UnifiedTextProcessor:
    """Attach Gemma 4's canonical text processor to an Inspect HF API.

    The model remains the full unified conditional-generation wrapper.  This
    matters even in a text-only evaluation: it retains the model's official
    generation preparation path while the processor preserves its chat
    template.  The function is idempotent so local-checkpoint loading can add
    PEFT weights after setup without reloading processor assets.
    """

    existing = getattr(api, "_ctm_gemma4_unified_text_processor", None)
    if isinstance(existing, Gemma4UnifiedTextProcessor):
        return existing
    model = getattr(api, "model", None)
    if not _is_gemma4_unified_model(model):
        observed = getattr(getattr(model, "config", None), "model_type", None)
        raise ValueError(
            "gemma4_unified_processor requires a Gemma 4 unified conditional-generation model; "
            f"got config.model_type={observed!r}"
        )
    source = model_source or getattr(api, "model_name", None)
    if not isinstance(source, str) or not source:
        raise ValueError("Gemma 4 unified processor setup requires the exact model or snapshot path")
    token = getattr(api, "api_key", None)
    if not isinstance(token, str):
        token = None
    processor = _load_gemma4_unified_processor(source, token=token)
    if not hasattr(processor, "tokenizer") or not callable(getattr(processor, "apply_chat_template", None)):
        raise TypeError("AutoProcessor did not produce a Gemma 4 processor with a tokenizer and chat template")
    adapter = Gemma4UnifiedTextProcessor(processor)
    if adapter.eos_token is None:
        raise ValueError("Gemma 4 unified processor has no concrete EOS token")
    adapter.pad_token = adapter.eos_token
    adapter.padding_side = "left"

    current_call_args = getattr(api, "tokenizer_call_args", None)
    if current_call_args is not None and not isinstance(current_call_args, Mapping):
        raise TypeError("Inspect Hugging Face tokenizer_call_args must be a mapping")
    call_args = dict(current_call_args or {})
    if call_args.get("return_mm_token_type_ids", False) is not False:
        raise ValueError(
            "Gemma 4 unified text evaluation cannot request multimodal token-type IDs; "
            "all evaluation prompts must be text-only"
        )
    call_args["return_mm_token_type_ids"] = False
    api.processor = processor
    api.tokenizer = adapter
    api.tokenizer_call_args = call_args
    api._ctm_gemma4_unified_text_processor = adapter
    return adapter


def _gemma4_unified_context_length(api: Any) -> int:
    """Read the true context window from the already-loaded Gemma wrapper.

    Inspect's ACP usage mapper receives only the rendered ``hf/...`` model
    string.  If that string is absent from its model-info registry, Inspect
    resolves it by constructing another provider instance.  For a local Gemma
    snapshot that would load the full model a second time.  The first wrapper
    already exposes its authoritative configuration, so use that factual
    context window rather than guessing a model-family default.
    """

    config = getattr(getattr(api, "model", None), "config", None)
    for candidate in (config, getattr(config, "text_config", None)):
        context_length = getattr(candidate, "max_position_embeddings", None)
        if (
            isinstance(context_length, int)
            and not isinstance(context_length, bool)
            and context_length > 0
        ):
            return context_length
    raise ValueError(
        "Gemma 4 unified model configuration has no positive max_position_embeddings context length"
    )


def _register_gemma4_unified_model_info(resolved: Any) -> None:
    """Register exact local-model metadata before Inspect maps usage events.

    ``set_model_info`` is Inspect's supported custom-model metadata API.  The
    key is deliberately ``str(resolved)`` because that is the exact value put
    on each ``ModelEvent``; registering a source path or a canonical alias
    would leave ACP's later string lookup free to reconstruct a second HF
    provider.  This records only the context window read from the model that
    is already resident; it does not alter generation or its EOS-only policy.
    """

    model_name = str(resolved)
    if not model_name:
        raise ValueError("Gemma 4 unified model has no exact resolved Inspect model name")
    from inspect_ai.model import ModelInfo, set_model_info

    set_model_info(
        model_name,
        ModelInfo(context_length=_gemma4_unified_context_length(getattr(resolved, "api", None))),
    )


def gemma4_unified_hf_model(
    model: str,
    *,
    model_args: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
) -> Any:
    """Resolve a direct ``hf/...`` Gemma 4 model with its canonical processor.

    This is intentionally a narrow route rather than a generic multimodal
    abstraction.  It handles text-only benchmark prompts for the pinned Gemma
    4 12B model, preserves the official processor, and leaves output
    termination to the separately installed EOS-only native-HF runtime.
    """

    if not isinstance(model, str) or not model.startswith("hf/"):
        raise ValueError("gemma4_unified_processor requires an explicit hf/... model")
    _require_no_token_cap(generation_config, label="Gemma 4 unified generation_config")
    options = dict(model_args or {})
    marker = options.pop(GEMMA4_UNIFIED_PROCESSOR_MODEL_ARG, None)
    if marker is not True:
        raise ValueError(f"Gemma 4 unified loading requires {GEMMA4_UNIFIED_PROCESSOR_MODEL_ARG}=true")
    if options.pop("hf_language_model_only", False):
        raise ValueError(
            "Gemma 4 unified evaluation must retain its full conditional-generation wrapper; "
            "do not set hf_language_model_only"
        )
    requested_loader = options.get("auto_model_class")
    if requested_loader is not None and requested_loader != GEMMA4_UNIFIED_AUTO_MODEL_CLASS:
        raise ValueError(
            "Gemma 4 unified evaluation requires "
            f"auto_model_class={GEMMA4_UNIFIED_AUTO_MODEL_CLASS!r}, got {requested_loader!r}"
        )
    options["auto_model_class"] = GEMMA4_UNIFIED_AUTO_MODEL_CLASS

    from inspect_ai.model import GenerateConfig, get_model

    resolved = get_model(
        model,
        config=GenerateConfig(**dict(generation_config or {})),
        **options,
    )
    api = getattr(resolved, "api", None)
    configure_gemma4_unified_hf_api(api, model_source=model.removeprefix("hf/"))
    _register_gemma4_unified_model_info(resolved)
    return resolved


def _qwen35_vllm_lora_is_unsafe(model_name: str) -> bool:
    """Identify the currently unsupported Qwen3.5 PEFT/vLLM bridge.

    vLLM 0.26 serves Qwen3.5 through ``Qwen3_5ForConditionalGeneration``.
    Its adapter mapper omits the ``model.layers.`` ->
    ``language_model.model.layers.`` bridge required by PEFT adapters trained
    through Transformers' text-only Qwen3.5 model.  vLLM accepts and exposes
    such an adapter while applying zero tensors.  Do not allow a silent
    base-model evaluation until a compatibility adapter has passed the fixed
    token HF/vLLM parity gate.
    """

    normalized = model_name.lower().replace("_", ".")
    return "qwen3.5" in normalized


def _unsafe_qwen35_vllm_lora_error(model_name: str) -> ValueError:
    return ValueError(
        f"vLLM evaluation of the Qwen3.5 LoRA checkpoint for {model_name!r} is disabled: "
        "vLLM 0.26 accepts the PEFT adapter but maps no `model.layers.*` tensors to "
        "its `language_model.model.layers.*` runtime modules. Use provider='hf', or a "
        "separate verified compatibility adapter that has passed HF/vLLM fixed-token parity."
    )


def _is_verified_qwen35_vllm_compat_adapter(directory: Path) -> bool:
    """Accept only a byte-bound compatibility copy with sufficient parity evidence.

    v1 remains the original all-variants direct gate.  The imported validator
    additionally recognizes one fixed composite schema; it cannot turn a
    generic Qwen3.5 adapter into an allowed vLLM checkpoint.
    """

    return is_verified_qwen35_vllm_compat_adapter(directory)


def _require_safe_qwen35_vllm_lora(model_name: str, directory: Path) -> None:
    if _qwen35_vllm_lora_is_unsafe(model_name) and not _is_verified_qwen35_vllm_compat_adapter(directory):
        raise _unsafe_qwen35_vllm_lora_error(model_name)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _configure_native_hf_attention_backends() -> None:
    """Apply the explicit native-HF SDPA recovery switch, if requested.

    Qwen3.5 normally uses PyTorch's automatic SDPA dispatch.  A small number
    of long, variable-length decode batches on some CUDA/cuDNN combinations
    have failed inside cuDNN's fused MHA graph rather than falling back.  The
    opt-in switch leaves Flash and math SDPA available while excluding only
    cuDNN SDPA.  It is deliberately an environment switch so ordinary runs
    retain the default backend and a recovery launch can attest the deviation.
    """

    value = os.environ.get(_DISABLE_CUDNN_SDP_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{_DISABLE_CUDNN_SDP_ENV} must be '0' or '1', got {value!r}")
    if value == "0":
        return
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)
    logging.getLogger(__name__).warning(
        "Disabled cuDNN SDPA for this native-HF worker via %s; Flash/math SDPA remain enabled.",
        _DISABLE_CUDNN_SDP_ENV,
    )


class _SecretRedactingFormatter(logging.Formatter):
    def __init__(self, secrets: tuple[str, ...]) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        for secret in self._secrets:
            rendered = rendered.replace(secret, "<redacted>")
        return rendered


def _persistent_log_directory(log_dir: str | Path) -> Path:
    raw = str(log_dir)
    if raw.startswith("file://"):
        raw = raw.removeprefix("file://")
    elif "://" in raw:
        raise ValueError("persistent vLLM server artifacts require a local --log-dir")
    return Path(raw).expanduser().resolve()


def _require_native_vllm_api(model: Any):
    from inspect_ai.model._providers.vllm import VLLMAPI

    api = getattr(model, "api", None)
    if not isinstance(api, VLLMAPI):
        raise TypeError("persistent evaluation requires Inspect's native vLLM provider")
    return api


def native_vllm_request_identity(
    *,
    model: str | None = None,
    local_checkpoint: str | Path | None = None,
    model_args: Mapping[str, Any] | None = None,
) -> tuple[str, str | None]:
    """Return the exact base and optional adapter identity for a native vLLM request."""

    if (model is None) == (local_checkpoint is None):
        raise ValueError("native vLLM identity requires exactly one model or local checkpoint")
    if local_checkpoint is not None:
        if dict(model_args or {}).get("provider", "hf") != "vllm":
            raise ValueError("local checkpoint is not configured for the native vLLM provider")
        directory, manifest = read_local_checkpoint(local_checkpoint)
        if not manifest["lora"]:
            raise ValueError("vLLM local-checkpoint evaluation currently requires a LoRA checkpoint")
        _require_safe_qwen35_vllm_lora(manifest["model"], directory)
        return manifest["model"], str(directory)

    if not model or not model.startswith("vllm/"):
        raise ValueError("model is not an Inspect native vllm/... model")
    from inspect_ai.model._providers._vllm_lora import parse_vllm_model

    base_model, adapter = parse_vllm_model(model.removeprefix("vllm/"))
    if adapter is not None:
        _require_safe_qwen35_vllm_lora(base_model, Path(adapter.name).expanduser().resolve())
    return base_model, adapter.name if adapter is not None else None


@dataclass
class PersistentNativeVLLMServer:
    """Parent-owned native Inspect vLLM server shared by isolated task children."""

    model: Any
    api: Any
    log_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    process: Any = None
    base_url: str | None = None
    api_key: str | None = None
    base_model: str | None = None
    adapter_name: str | None = None
    served_model: str | None = None
    _handler: logging.Handler | None = None
    _logger_levels: dict[str, int] = field(default_factory=dict)
    _closed: bool = False

    @classmethod
    def start(
        cls,
        model: Any,
        *,
        log_dir: str | Path,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> "PersistentNativeVLLMServer":
        """Start exactly one native vLLM server and record its condition artifacts."""

        api = _require_native_vllm_api(model)
        if getattr(api, "_init_base_url", None) is not None or os.environ.get("VLLM_BASE_URL"):
            raise ValueError(
                "persistent vLLM mode must launch and own its server; unset VLLM_BASE_URL "
                "and do not pass base_url or port"
            )

        directory = _persistent_log_directory(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "vllm-server.log"
        metadata_path = directory / "vllm-server.json"
        adapter = getattr(api, "adapter", None)
        base_model = getattr(api, "base_model", None)
        adapter_name = getattr(adapter, "name", None) if adapter is not None else None
        served_model = adapter_name or base_model
        if not isinstance(base_model, str) or not base_model or not isinstance(served_model, str):
            raise TypeError("Inspect's native vLLM provider did not expose an exact model identity")

        instance = cls(
            model=model,
            api=api,
            log_path=log_path,
            metadata_path=metadata_path,
            base_model=base_model,
            adapter_name=adapter_name,
            served_model=served_model,
            api_key=getattr(api, "api_key", None),
            metadata={
                "schema_version": 1,
                "status": "starting",
                "owner_pid": os.getpid(),
                "started_at": _utc_now(),
                "base_model": base_model,
                "adapter": adapter_name,
                "served_model": served_model,
                "server_args": redact_secrets(dict(getattr(api, "server_args", {}))),
                "source": redact_secrets(dict(source_metadata or {})),
                "server_log": str(log_path),
            },
        )
        instance._install_log_handler()
        instance._write_metadata()
        logging.getLogger("inspect_ai.model._providers.vllm").info(
            "Starting parent-owned persistent vLLM server for %s", served_model
        )

        try:
            resolve_server = getattr(api, "_resolve_server", None)
            if not callable(resolve_server):
                raise RuntimeError("installed Inspect vLLM provider has no native server startup hook")
            resolve_server()
            server = getattr(api, "_server", None)
            instance.process = getattr(server, "process", None)
            instance.base_url = getattr(server, "base_url", None)
            instance.api_key = getattr(server, "api_key", None) or instance.api_key
            if instance.process is None or not isinstance(instance.base_url, str) or not instance.base_url:
                raise RuntimeError("Inspect connected to an external vLLM endpoint instead of launching one")
            if not isinstance(instance.api_key, str) or not instance.api_key:
                raise RuntimeError("Inspect's native vLLM server did not expose its API key")
            instance.assert_healthy()
        except BaseException as exc:
            instance.metadata.update(
                {
                    "status": "startup_failed",
                    "failed_at": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            try:
                api.close()
            except Exception as cleanup_exc:  # noqa: BLE001 -- retain the startup failure
                instance.metadata["cleanup_error"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            instance._write_metadata()
            instance._remove_log_handler()
            raise

        instance.metadata.update(
            {
                "status": "ready",
                "ready_at": _utc_now(),
                "base_url": instance.base_url,
                "server_pid": instance.process.pid,
            }
        )
        instance._write_metadata()
        return instance

    def _install_log_handler(self) -> None:
        handler = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
        secrets = (self.api_key,) if isinstance(self.api_key, str) and self.api_key else ()
        handler.setFormatter(_SecretRedactingFormatter(secrets))
        handler.setLevel(logging.DEBUG)
        self._handler = handler
        for name in _PERSISTENT_VLLM_LOGGERS:
            logger = logging.getLogger(name)
            self._logger_levels[name] = logger.level
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)

    def _remove_log_handler(self) -> None:
        handler = self._handler
        if handler is None:
            return
        for name in _PERSISTENT_VLLM_LOGGERS:
            logger = logging.getLogger(name)
            logger.removeHandler(handler)
            if name in self._logger_levels:
                logger.setLevel(self._logger_levels[name])
        handler.close()
        self._handler = None

    def _write_metadata(self) -> None:
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def health_error(self) -> str | None:
        """Return a diagnostic if the owned process or exact served model is unavailable."""

        if self.process is None:
            return "server process was never recorded"
        returncode = self.process.poll()
        if returncode is not None:
            return f"server process {self.process.pid} exited with status {returncode}"
        if not self.base_url or not self.api_key or not self.served_model:
            return "server connection identity is incomplete"

        import httpx

        try:
            response = httpx.get(
                f"{self.base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5.0,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 -- health diagnostics must include protocol failures
            return f"model endpoint check failed: {type(exc).__name__}: {exc}"
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
            return "model endpoint returned an invalid model-list payload"
        model_ids = {
            item.get("id")
            for item in payload.get("data", [])
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        }
        required_ids = {self.base_model, self.served_model}
        missing = sorted(model_id for model_id in required_ids if model_id not in model_ids)
        if missing:
            return f"model endpoint is missing exact model identity {missing}"
        return None

    def assert_healthy(self) -> None:
        error = self.health_error()
        if error is None:
            return
        self.metadata.update(
            {
                "status": "unhealthy",
                "unhealthy_at": _utc_now(),
                "health_error": error,
            }
        )
        self._write_metadata()
        raise RuntimeError(f"parent-owned vLLM server is unhealthy: {error}; see {self.log_path}")

    def child_environment(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return an external-server environment for one isolated child."""

        if self._closed or not self.base_url or not self.api_key:
            raise RuntimeError("parent-owned vLLM server is not available to children")
        child_metadata = {
            "mode": "parent_owned_external",
            "owner_pid": self.metadata["owner_pid"],
            "base_model": self.base_model,
            "adapter": self.adapter_name,
            "served_model": self.served_model,
            "base_url": self.base_url,
            "server_log": str(self.log_path),
            "server_metadata": str(self.metadata_path),
        }
        return {
            **dict(os.environ if environ is None else environ),
            "VLLM_BASE_URL": self.base_url,
            "VLLM_API_KEY": self.api_key,
            PERSISTENT_VLLM_CHILD_METADATA_ENV: json.dumps(child_metadata, sort_keys=True),
        }

    def close(self, *, status: str = "completed") -> None:
        """Terminate the parent-owned process and finalize condition metadata."""

        if self._closed:
            return
        self._closed = True
        cleanup_error: Exception | None = None
        try:
            self.api.close()
        except Exception as exc:  # noqa: BLE001 -- record cleanup failure before surfacing it
            cleanup_error = exc
        self.metadata.update(
            {
                "status": "cleanup_failed" if cleanup_error is not None else status,
                "stopped_at": _utc_now(),
                "server_returncode": self.process.poll() if self.process is not None else None,
            }
        )
        if cleanup_error is not None:
            self.metadata["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        self._write_metadata()
        self._remove_log_handler()
        if cleanup_error is not None:
            raise RuntimeError(f"failed to clean up parent-owned vLLM server: {cleanup_error}") from cleanup_error

    def __enter__(self) -> "PersistentNativeVLLMServer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.close(status="failed" if exc_type is not None else "completed")
        except Exception as cleanup_exc:
            if exc is None:
                raise
            exc.add_note(str(cleanup_exc))
        return False


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
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    """Load a LocalBackend checkpoint through Inspect's HF or vLLM provider."""

    from inspect_ai.model import GenerateConfig, get_model

    directory, manifest = read_local_checkpoint(checkpoint)
    recorded_model = manifest["model"]
    if base_model is not None and base_model != recorded_model:
        raise ValueError(
            f"local checkpoint base model mismatch: manifest records {recorded_model!r}, "
            f"but {base_model!r} was requested"
        )
    options = dict(model_args or {})
    provider = options.pop("provider", "hf")
    hf_language_model_only = options.pop("hf_language_model_only", False)
    gemma4_unified_processor = options.pop(GEMMA4_UNIFIED_PROCESSOR_MODEL_ARG, False)
    if not isinstance(hf_language_model_only, bool):
        raise ValueError("hf_language_model_only must be boolean")
    if not isinstance(gemma4_unified_processor, bool):
        raise ValueError(f"{GEMMA4_UNIFIED_PROCESSOR_MODEL_ARG} must be boolean")
    if provider not in {"hf", "vllm"}:
        raise ValueError(f"local checkpoint provider must be 'hf' or 'vllm', got {provider!r}")
    forbidden = sorted(set(options) & {"model_path", "tokenizer", "tokenizer_path"})
    if forbidden:
        raise ValueError(
            f"local checkpoint model_args cannot override {forbidden}; "
            "the checkpoint manifest owns the base model and tokenizer"
        )
    config = GenerateConfig(**dict(generation_config or {}))
    if provider == "vllm":
        if hf_language_model_only:
            raise ValueError("hf_language_model_only applies only to provider='hf'")
        if gemma4_unified_processor:
            raise ValueError("gemma4_unified_processor applies only to provider='hf'")
        if not manifest["lora"]:
            raise ValueError("vLLM local-checkpoint evaluation currently requires a LoRA checkpoint")
        _require_safe_qwen35_vllm_lora(recorded_model, directory)
        return get_model(f"vllm/{recorded_model}:{directory}", config=config, **options)

    _configure_native_hf_attention_backends()
    if gemma4_unified_processor:
        if hf_language_model_only:
            raise ValueError(
                "Gemma 4 unified evaluation must retain its full conditional-generation wrapper; "
                "do not set hf_language_model_only"
            )
        model = gemma4_unified_hf_model(
            f"hf/{recorded_model}",
            model_args={**options, GEMMA4_UNIFIED_PROCESSOR_MODEL_ARG: True},
            generation_config=generation_config,
        )
    else:
        model = get_model(f"hf/{recorded_model}", config=config, **options)
    if not hasattr(model.api, "model"):
        raise TypeError("Inspect's Hugging Face provider did not expose a model instance")
    if hf_language_model_only:
        detach_muse_glimmer_vision_modules(model.api.model)
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


def detach_muse_glimmer_vision_modules(model: Any) -> tuple[str, ...]:
    """Detach Muse's unused vision modules while preserving its text-module paths."""

    import torch

    if not isinstance(model, torch.nn.Module) or getattr(getattr(model, "config", None), "model_type", None) != "muse_glimmer":
        raise ValueError("hf_language_model_only currently supports only a Muse Glimmer torch model")
    multimodal = getattr(model, "model", None)
    language_model = getattr(multimodal, "language_model", None)
    if not isinstance(language_model, torch.nn.Module):
        raise ValueError("Muse Glimmer HF evaluation model has no model.language_model")
    detached: list[str] = []
    for name in ("vision_tower", "vision_adapter", "vision_projection", "perception_emb_norm"):
        if not hasattr(multimodal, name):
            raise ValueError(f"Muse Glimmer HF evaluation model has no model.{name}")
        if getattr(multimodal, name) is not None:
            setattr(multimodal, name, None)
            detached.append(name)
    if not detached:
        raise ValueError("Muse Glimmer HF evaluation vision modules were already absent")
    return tuple(detached)


__all__ = [
    "GEMMA4_UNIFIED_AUTO_MODEL_CLASS",
    "GEMMA4_UNIFIED_PROCESSOR_MODEL_ARG",
    "PERSISTENT_VLLM_CHILD_METADATA_ENV",
    "Gemma4UnifiedTextProcessor",
    "PersistentNativeVLLMServer",
    "configure_gemma4_unified_hf_api",
    "detach_muse_glimmer_vision_modules",
    "gemma4_unified_hf_model",
    "local_checkpoint_model",
    "native_vllm_request_identity",
    "read_local_checkpoint",
]
