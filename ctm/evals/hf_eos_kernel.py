"""Fail-closed primitives for uncapped, EOS-only native-HF decoding.

The helpers here deliberately implement a small, inspectable subset of
``GenerationMixin.generate``.  They have no experiment, task, launcher, or
runtime-version policy: callers supply those contracts around this kernel.
The loop has no token counter or length stopping condition; it terminates only
when every sequence emits one of the model's concrete EOS tokens.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any


# These names cover the common aliases exposed by Inspect, Transformers,
# OpenAI-compatible APIs, and saved generation configurations. ``max_length``
# is an output-generation bound in Transformers. Connection/task/sample limits
# are intentionally absent: they are not token-generation limits.
TOKEN_CAP_FIELD_NAMES = frozenset(
    {
        "max_tokens",
        "max_new_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "max_generation_tokens",
        "generation_max_tokens",
        "completion_max_tokens",
        "output_max_tokens",
        "max_length",
        "generation_max_length",
    }
)
_TOKEN_CAP_COMPACT_NAMES = frozenset(name.replace("_", "") for name in TOKEN_CAP_FIELD_NAMES)


class HFEOSError(RuntimeError):
    """The native-HF EOS-only contract is capped, unsafe, or unsupported."""


def assert_no_token_cap_mapping(value: Any, *, label: str) -> None:
    """Reject an explicit or nested output-token limit.

    A missing key is meaningful: an upstream provider can otherwise inject a
    mutable default. Optional object-dump fields set to ``None`` are absent,
    while every concrete token cap fails closed.
    """

    def normalize_key(raw_key: object) -> str:
        value = str(raw_key).replace("-", "_")
        normalized: list[str] = []
        for index, character in enumerate(value):
            if character.isupper() and index > 0 and value[index - 1] != "_":
                normalized.append("_")
            normalized.append(character.lower())
        return "".join(normalized)

    def walk(candidate: Any, path: str) -> None:
        if isinstance(candidate, Mapping):
            for raw_key, nested in candidate.items():
                key = normalize_key(raw_key)
                compact_key = "".join(character.lower() for character in str(raw_key) if character.isalnum())
                nested_path = f"{path}.{raw_key}"
                if (key in TOKEN_CAP_FIELD_NAMES or compact_key in _TOKEN_CAP_COMPACT_NAMES) and nested is not None:
                    raise ValueError(f"{label} must not contain output-token cap field {nested_path!r}")
                walk(nested, nested_path)
        elif isinstance(candidate, (list, tuple)):
            for index, nested in enumerate(candidate):
                walk(nested, f"{path}[{index}]")

    walk(value, label)


def configuration_mapping(config: Any) -> dict[str, Any]:
    """Return an inspectable generation-config mapping when one is available."""

    if isinstance(config, Mapping):
        return dict(config)
    for name in ("model_dump", "to_dict", "dict"):
        method = getattr(config, name, None)
        if callable(method):
            candidate = method()
            if isinstance(candidate, Mapping):
                return dict(candidate)
    return {}


def assert_no_runtime_token_cap(config: Any, *, label: str) -> None:
    """Validate an Inspect-like configuration and translate failures to the kernel error."""

    mapping = configuration_mapping(config)
    if not mapping:
        raise HFEOSError(f"{label} is not an inspectable generation configuration")
    try:
        assert_no_token_cap_mapping(mapping, label=label)
    except ValueError as exc:
        raise HFEOSError(str(exc)) from exc


def config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read a generation option without bypassing mapping validation."""

    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def concrete_eos_ids(value: Any) -> set[int]:
    """Return concrete EOS IDs from supported scalar/list configuration forms."""

    if isinstance(value, int) and not isinstance(value, bool):
        return {value}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {
            item
            for item in value
            if isinstance(item, int) and not isinstance(item, bool)
        }
    return set()


def eos_ids(tokenizer: Any, model: Any) -> set[int]:
    """Collect every concrete EOS ID exposed by the active text model.

    A tokenizer EOS must not hide a wrapper/model end-of-turn marker. Each ID
    is an EOS condition only; none adds a generation-length limit.
    """

    model_config = getattr(model, "config", None)
    candidates = (
        getattr(tokenizer, "eos_token_id", None),
        getattr(model_config, "eos_token_id", None),
        getattr(getattr(model_config, "text_config", None), "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    )
    values = set().union(*(concrete_eos_ids(candidate) for candidate in candidates))
    if values:
        return values
    raise HFEOSError("EOS-only sampling requires a concrete model EOS token")


def pad_token_id(tokenizer: Any, eos_token_ids: set[int], saved: Mapping[str, Any] | None = None) -> int:
    """Resolve padding without changing the model's EOS-only termination policy."""

    candidate = (saved or {}).get("pad_token_id")
    if isinstance(candidate, int) and not isinstance(candidate, bool):
        return candidate
    candidate = getattr(tokenizer, "pad_token_id", None)
    if isinstance(candidate, int) and not isinstance(candidate, bool):
        return candidate
    return min(eos_token_ids)


# The loop deliberately does not claim generic GenerationConfig parity. These
# reviewed fields cover basic sampling plus Gemma's saved multimodal-token
# suppression. An unknown saved processor fails before model execution.
SUPPORTED_SAVED_GENERATION_FIELDS = frozenset(
    {
        "bos_token_id",
        "do_sample",
        "eos_token_id",
        "pad_token_id",
        "suppress_tokens",
        "temperature",
        "top_k",
        "top_p",
        "transformers_version",
        "use_cache",
    }
)


def saved_generation_config(model: Any) -> dict[str, Any]:
    """Return the reviewed saved GenerationConfig *diff*, or fail closed.

    ``GenerationConfig.to_dict()`` includes library defaults such as length
    fields; those are not a saved decoding contract and must not be treated as
    an EOS-loop limit. Real GenerationConfig objects must therefore expose
    ``to_diff_dict()``. Small test/custom objects may provide an explicit
    mapping or attributes, subject to the same strict field check.
    """

    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return {}
    to_diff_dict = getattr(generation_config, "to_diff_dict", None)
    if callable(to_diff_dict):
        try:
            candidate = to_diff_dict()
        except Exception as exc:  # pragma: no cover - third-party model boundary
            raise HFEOSError("could not inspect the model's saved GenerationConfig diff") from exc
        if not isinstance(candidate, Mapping):
            raise HFEOSError("model GenerationConfig.to_diff_dict() did not return a mapping")
        saved = dict(candidate)
    elif isinstance(generation_config, Mapping):
        saved = dict(generation_config)
    else:
        attributes = getattr(generation_config, "__dict__", None)
        if not isinstance(attributes, Mapping):
            raise HFEOSError("model GenerationConfig has no inspectable saved configuration")
        saved = dict(attributes)

    unsupported = sorted(set(saved) - SUPPORTED_SAVED_GENERATION_FIELDS)
    if unsupported:
        raise HFEOSError(
            "EOS-only sampler cannot faithfully apply saved GenerationConfig field(s): "
            + ", ".join(unsupported)
        )
    if "use_cache" in saved and saved["use_cache"] is not None and saved["use_cache"] is not True:
        raise HFEOSError("EOS-only sampler requires the saved GenerationConfig to permit caching")
    return saved


def saved_suppress_tokens(saved: Mapping[str, Any]) -> tuple[int, ...]:
    """Validate Gemma's reviewed saved logits-processor surface."""

    raw = saved.get("suppress_tokens")
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise HFEOSError("saved suppress_tokens must be a sequence of token IDs")
    token_ids: list[int] = []
    for token_id in raw:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise HFEOSError("saved suppress_tokens contains an invalid token ID")
        token_ids.append(token_id)
    if len(set(token_ids)) != len(token_ids):
        raise HFEOSError("saved suppress_tokens contains duplicate token IDs")
    return tuple(token_ids)


def saved_logits_processor(saved: Mapping[str, Any], *, device: Any) -> Any | None:
    """Build the reviewed saved processor before temperature/top-k/top-p warpers.

    Gemma's saved ``suppress_tokens`` excludes multimodal sentinel IDs. The
    standard processor changes logits only: it adds no stopping, max-length,
    or forced-EOS behavior, and must run before sampling warpers to match
    ``GenerationMixin``.
    """

    suppress_tokens = saved_suppress_tokens(saved)
    if not suppress_tokens:
        return None
    try:
        from transformers.generation import SuppressTokensLogitsProcessor
    except ImportError as exc:  # pragma: no cover - pinned runtime boundary
        raise HFEOSError("Transformers SuppressTokensLogitsProcessor is required for saved decoding") from exc
    return SuppressTokensLogitsProcessor(list(suppress_tokens), device=str(device))


def effective_sampling_value(config: Any, saved: Mapping[str, Any], name: str) -> Any:
    """Mirror normal GenerationConfig override precedence for basic samplers."""

    value = config_value(config, name)
    return saved.get(name) if value is None else value


def supports_logits_to_keep(model: Any) -> bool:
    """Whether the model's forward accepts the normal generation prefill hint."""

    forward = getattr(model, "forward", None)
    if not callable(forward):
        return False
    try:
        return "logits_to_keep" in inspect.signature(forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - opaque extension boundary
        return False


def sample_next_tokens(
    *,
    input_ids: Any,
    logits: Any,
    temperature: Any,
    top_p: Any,
    top_k: Any,
    do_sample: bool,
    logits_processor: Any | None = None,
) -> Any:
    """Sample one token without calling Transformers ``generate``.

    The direct loop supports only reviewed sampling warpers and the reviewed
    saved logits processor. It intentionally does not add a length condition;
    EOS handling belongs to :func:`eos_only_model_generate`.
    """

    import torch

    if do_sample is not True:
        raise HFEOSError("EOS-only sampler requires stochastic native-HF decoding")
    scores = logits.to(dtype=torch.float32)
    if logits_processor is not None:
        scores = logits_processor(input_ids, scores)
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or float(temperature) <= 0:
            raise HFEOSError("EOS-only sampler received an invalid temperature")
        from transformers.generation import TemperatureLogitsWarper

        if float(temperature) != 1.0:
            scores = TemperatureLogitsWarper(float(temperature))(input_ids, scores)
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise HFEOSError("EOS-only sampler received an invalid top_k")
        from transformers.generation import TopKLogitsWarper

        if top_k != 0:
            scores = TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1)(input_ids, scores)
    if top_p is not None:
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < float(top_p) <= 1:
            raise HFEOSError("EOS-only sampler received an invalid top_p")
        if float(top_p) < 1:
            from transformers.generation import TopPLogitsWarper

            scores = TopPLogitsWarper(top_p=float(top_p), min_tokens_to_keep=1)(input_ids, scores)
    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1).squeeze(1)


def eos_only_model_generate(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    tokenizer: Any,
    config: Any,
    do_sample: bool,
    return_dict_in_generate: bool,
    output_logits: bool,
    output_hidden_states: bool,
) -> Any:
    """Run uncapped autoregressive native-HF sampling until every row emits EOS."""

    import torch

    assert_no_runtime_token_cap(config, label="effective Inspect generation config")
    if not return_dict_in_generate:
        raise HFEOSError("EOS-only sampler requires Inspect return_dict_in_generate")
    # Retaining every step's tensors would turn uncapped generation into
    # unbounded host-memory retention. Reject rather than change the result.
    if output_logits or output_hidden_states or config_value(config, "logprobs"):
        raise HFEOSError("EOS-only sampler does not support retained logits or hidden states")
    if do_sample is not True:
        raise HFEOSError("EOS-only sampler requires stochastic native-HF decoding")
    if config_value(config, "stop_seqs") is not None:
        raise HFEOSError("EOS-only sampler permits only model-EOS termination")

    saved_generation = saved_generation_config(model)
    saved_processor = saved_logits_processor(saved_generation, device=input_ids.device)
    eos_token_ids = eos_ids(tokenizer, model)
    padding_token_id = pad_token_id(tokenizer, eos_token_ids, saved_generation)
    temperature = effective_sampling_value(config, saved_generation, "temperature")
    top_p = effective_sampling_value(config, saved_generation, "top_p")
    top_k = effective_sampling_value(config, saved_generation, "top_k")
    batch_size = input_ids.shape[0]
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
    generated = input_ids
    running_attention = attention_mask
    past_key_values = None

    # No counter or maximum-length stopping criterion belongs in this loop.
    # The only exit is all rows emitting an EOS token.
    while bool(torch.any(unfinished).item()):
        model_input_ids = generated if past_key_values is None else generated[:, -1:]
        model_kwargs = {
            "input_ids": model_input_ids,
            "attention_mask": running_attention,
            "past_key_values": past_key_values,
            "use_cache": True,
            "return_dict": True,
        }
        # Gemma's native generate path requests final-token logits only. This
        # is a forward-memory hint, never a generation length setting.
        if supports_logits_to_keep(model):
            model_kwargs["logits_to_keep"] = 1
        outputs = model(**model_kwargs)
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise HFEOSError("native-HF model did not return a cache for EOS-only sampling")
        next_tokens = sample_next_tokens(
            input_ids=generated,
            logits=outputs.logits[:, -1, :],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            logits_processor=saved_processor,
        )
        next_tokens = torch.where(unfinished, next_tokens, torch.full_like(next_tokens, padding_token_id))
        generated = torch.cat((generated, next_tokens[:, None]), dim=-1)
        running_attention = torch.cat(
            (
                running_attention,
                torch.ones((batch_size, 1), dtype=running_attention.dtype, device=running_attention.device),
            ),
            dim=-1,
        )
        finished_now = torch.zeros_like(unfinished)
        for eos_token_id in eos_token_ids:
            finished_now |= next_tokens.eq(eos_token_id)
        unfinished &= ~finished_now

    return type(
        "EOSOnlyModelGenerateOutput",
        (),
        {"sequences": generated, "logits": None, "hidden_states": None},
    )()


__all__ = [
    "HFEOSError",
    "SUPPORTED_SAVED_GENERATION_FIELDS",
    "TOKEN_CAP_FIELD_NAMES",
    "assert_no_runtime_token_cap",
    "assert_no_token_cap_mapping",
    "concrete_eos_ids",
    "config_value",
    "configuration_mapping",
    "effective_sampling_value",
    "eos_ids",
    "eos_only_model_generate",
    "pad_token_id",
    "sample_next_tokens",
    "saved_generation_config",
    "saved_logits_processor",
    "saved_suppress_tokens",
    "supports_logits_to_keep",
]
