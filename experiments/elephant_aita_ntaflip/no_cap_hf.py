"""EOS-only native-HF sampling for the r005 AITA rerun.

Inspect 0.3.258 resolves an omitted ``GenerateConfig.max_tokens`` through
``HuggingFaceAPI.max_tokens()``, and the ordinary Transformers ``generate``
route then has a separate default ``max_length``.  Neither is acceptable for
this campaign.  This narrowly scoped runtime hook leaves Inspect's task and
logging interfaces intact while replacing only its native-HF sampling call
with an autoregressive loop that terminates solely on the model's EOS token.

It is intentionally activated only by the r005 environment marker.  The hook
is not a generic project default and must not leak into other evaluations.
"""

from __future__ import annotations

import functools
import importlib.metadata
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .prepare import TOKEN_CAP_FIELD_NAMES, assert_no_token_cap_mapping

RUNTIME_ENV = "CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP"
RUNTIME_ENV_VALUE = "1"
RUNTIME_SCHEMA = "elephant-aita-nta-flip-native-hf-eos-only-v1-r005"
INSPECT_VERSION = "0.3.258"
TRANSFORMERS_VERSION = "5.5.4"

# This is the exact Inspect 0.3.258 native-HF model argument that is passed
# through to Qwen's ``apply_chat_template`` call.  Keep the attestation
# separate from the no-cap manifest policy: one proves prompt serialization,
# the other proves the termination mechanism.
QWEN_THINKING_POLICY: dict[str, Any] = {
    "interface": "inspect-ai-0.3.258-hf-model-arg.enable_thinking",
    "enable_thinking": False,
    "expected_output_mode": "direct-verdict-without-template-thinking-block",
}

RUNTIME_POLICY: dict[str, Any] = {
    "schema": RUNTIME_SCHEMA,
    "activation_environment": {RUNTIME_ENV: RUNTIME_ENV_VALUE},
    "sampling": "native-hf-autoregressive-eos-only",
    "batching": "one-sequence-per-native-hf-generate-call",
    "standard_transformers_generate": "not-invoked",
    "inspect_hf_default_max_tokens": "overridden-to-none",
    "qwen_chat_template": deepcopy(QWEN_THINKING_POLICY),
    "forbidden_generation_fields": sorted(TOKEN_CAP_FIELD_NAMES),
}


class NoTokenCapRuntimeError(RuntimeError):
    """The requested r005 execution path would use a token-generation cap."""


def runtime_policy() -> dict[str, Any]:
    """Return the immutable policy copied into every r005 runtime receipt."""

    return deepcopy(RUNTIME_POLICY)


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_exact_runtime() -> None:
    if os.environ.get(RUNTIME_ENV) != RUNTIME_ENV_VALUE:
        raise NoTokenCapRuntimeError(
            f"r005 EOS-only sampling requires {RUNTIME_ENV}={RUNTIME_ENV_VALUE!r}"
        )
    if _installed_version("inspect-ai") != INSPECT_VERSION:
        raise NoTokenCapRuntimeError("r005 EOS-only sampling requires Inspect AI 0.3.258")
    if _installed_version("transformers") != TRANSFORMERS_VERSION:
        raise NoTokenCapRuntimeError("r005 EOS-only sampling requires Transformers 5.5.4")


def _configuration_mapping(config: Any) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    for name in ("model_dump", "to_dict", "dict"):
        method = getattr(config, name, None)
        if callable(method):
            candidate = method()
            if isinstance(candidate, Mapping):
                return dict(candidate)
    return {}


def _assert_no_runtime_token_cap(config: Any, *, label: str) -> None:
    mapping = _configuration_mapping(config)
    if not mapping:
        raise NoTokenCapRuntimeError(f"{label} is not an inspectable generation configuration")
    try:
        assert_no_token_cap_mapping(mapping, label=label)
    except ValueError as exc:
        raise NoTokenCapRuntimeError(str(exc)) from exc


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read an Inspect config defensively without bypassing mapping guards."""

    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _concrete_eos_ids(value: Any) -> set[int]:
    """Return a configuration value's concrete EOS IDs, if any.

    A multimodal wrapper, its nested text model, and its generation config can
    each contribute an end marker.  In particular, a tokenizer's usual EOS is
    not necessarily the chat-template end-of-turn marker.  Treat every
    concrete integer in a supported scalar/list form as an EOS condition.
    """

    if isinstance(value, int) and not isinstance(value, bool):
        return {value}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {
            item
            for item in value
            if isinstance(item, int) and not isinstance(item, bool)
        }
    return set()


def _eos_ids(tokenizer: Any, model: Any) -> set[int]:
    """Collect every concrete EOS ID exposed by the active text model.

    Do not let a tokenizer-level EOS hide end-of-turn IDs declared by the
    wrapper/model generation configuration.  The sampling loop still has no
    length condition: each of these IDs is solely a model-provided stopping
    condition.
    """

    model_config = getattr(model, "config", None)
    candidates = (
        getattr(tokenizer, "eos_token_id", None),
        getattr(model_config, "eos_token_id", None),
        getattr(getattr(model_config, "text_config", None), "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    )
    values = set().union(*(_concrete_eos_ids(candidate) for candidate in candidates))
    if values:
        return values
    raise NoTokenCapRuntimeError("r005 EOS-only sampling requires a concrete model EOS token")


def _pad_token_id(tokenizer: Any, eos_ids: set[int]) -> int:
    candidate = getattr(tokenizer, "pad_token_id", None)
    if isinstance(candidate, int) and not isinstance(candidate, bool):
        return candidate
    return min(eos_ids)


def _sample_next_tokens(
    *, input_ids: Any, logits: Any, temperature: Any, top_p: Any, top_k: Any, do_sample: bool
) -> Any:
    """Sample one token without delegating to Transformers ``generate``.

    This intentionally implements only the frozen AITA decode contract
    (temperature/top-p/top-k).  Adding a length limit here would violate the
    campaign policy; termination is handled by EOS below.
    """

    import torch

    if do_sample is not True:
        raise NoTokenCapRuntimeError("r005 EOS-only sampler requires stochastic native-HF decoding")
    scores = logits.to(dtype=torch.float32)
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or float(temperature) <= 0:
            raise NoTokenCapRuntimeError("r005 EOS-only sampler received an invalid temperature")
        from transformers.generation import TemperatureLogitsWarper

        scores = TemperatureLogitsWarper(float(temperature))(input_ids, scores)
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise NoTokenCapRuntimeError("r005 EOS-only sampler received an invalid top_k")
        from transformers.generation import TopKLogitsWarper

        scores = TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1)(input_ids, scores)
    if top_p is not None:
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < float(top_p) <= 1:
            raise NoTokenCapRuntimeError("r005 EOS-only sampler received an invalid top_p")
        if float(top_p) < 1:
            from transformers.generation import TopPLogitsWarper

            scores = TopPLogitsWarper(top_p=float(top_p), min_tokens_to_keep=1)(input_ids, scores)
    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1).squeeze(1)


def _eos_only_model_generate(
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
    """Run uncapped autoregressive HF sampling until every row emits EOS."""

    import torch

    _assert_no_runtime_token_cap(config, label="effective Inspect generation config")
    if not return_dict_in_generate:
        raise NoTokenCapRuntimeError("r005 EOS-only sampler requires Inspect return_dict_in_generate")
    # r005 does not request logits/hidden states, and retaining every step's
    # tensor would turn a deliberately uncapped generation into unbounded
    # host memory retention.  Reject rather than silently changing the result.
    if output_logits or output_hidden_states or _config_value(config, "logprobs"):
        raise NoTokenCapRuntimeError("r005 EOS-only sampler does not support retained logits or hidden states")
    if do_sample is not True:
        # Inspect keeps sampling on the HF provider as a model argument.  It
        # is intentionally not inferred from a generation-config default.
        raise NoTokenCapRuntimeError("r005 EOS-only sampler requires stochastic native-HF decoding")
    if _config_value(config, "stop_seqs") is not None:
        raise NoTokenCapRuntimeError("r005 EOS-only sampler permits only model-EOS termination")

    eos_ids = _eos_ids(tokenizer, model)
    pad_token_id = _pad_token_id(tokenizer, eos_ids)
    batch_size = input_ids.shape[0]
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
    generated = input_ids
    running_attention = attention_mask
    past_key_values = None

    # No counter or maximum-length stopping criterion belongs in this loop.
    # The only exit is all rows emitting an EOS token.
    while bool(torch.any(unfinished).item()):
        if past_key_values is None:
            model_input_ids = generated
        else:
            model_input_ids = generated[:, -1:]
        outputs = model(
            input_ids=model_input_ids,
            attention_mask=running_attention,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise NoTokenCapRuntimeError("native-HF model did not return a cache for EOS-only sampling")
        next_tokens = _sample_next_tokens(
            input_ids=generated,
            logits=outputs.logits[:, -1, :],
            temperature=_config_value(config, "temperature"),
            top_p=_config_value(config, "top_p"),
            top_k=_config_value(config, "top_k"),
            do_sample=do_sample,
        )
        next_tokens = torch.where(unfinished, next_tokens, torch.full_like(next_tokens, pad_token_id))
        generated = torch.cat((generated, next_tokens[:, None]), dim=-1)
        running_attention = torch.cat(
            (running_attention, torch.ones((batch_size, 1), dtype=running_attention.dtype, device=running_attention.device)),
            dim=-1,
        )
        finished_now = torch.zeros_like(unfinished)
        for eos_token_id in eos_ids:
            finished_now |= next_tokens.eq(eos_token_id)
        unfinished &= ~finished_now

    # Match the minimal surface consumed by Inspect's batched HF helper.
    return type(
        "EOSOnlyModelGenerateOutput",
        (),
        {"sequences": generated, "logits": None, "hidden_states": None},
    )()


async def _eos_only_hf_generate(self: Any, input: Any, tools: Any, tool_choice: Any, config: Any) -> Any:
    """Inspect 0.3.258 ``HuggingFaceAPI.generate`` replacement for r005."""

    from inspect_ai.model._providers import hf

    _assert_no_runtime_token_cap(config, label="effective Inspect generation config")
    if self.enable_thinking is not False:
        raise NoTokenCapRuntimeError("r005 native-HF runtime did not set enable_thinking=False")
    if self.do_sample is not True:
        raise NoTokenCapRuntimeError("r005 native-HF runtime did not set do_sample=True")
    if _config_value(config, "stop_seqs") is not None:
        raise NoTokenCapRuntimeError("r005 native-HF runtime permits only model-EOS termination")
    handler = hf.HFHandler(self.model_name, self.model_family()) if len(tools) > 0 else None
    chat = self.hf_chat(input, tools)
    tokenizer = functools.partial(
        self.tokenizer,
        return_tensors="pt",
        padding=True,
        **self.tokenizer_call_args,
    )
    generator = functools.partial(
        _eos_only_model_generate,
        self.model,
        tokenizer=self.tokenizer,
        config=config,
        do_sample=self.do_sample,
        return_dict_in_generate=True,
        output_logits=False,
        output_hidden_states=False,
    )
    decoder = functools.partial(
        self.tokenizer.batch_decode,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = await hf.batched_generate(
        hf.GenerateInput(
            input=chat,
            device=self.model.device,
            tokenizer=tokenizer,
            generator=generator,
            decoder=decoder,
            # Qwen3.5's cache includes state for both its attention types.
            # One sequence per native call avoids mixing finished padded rows
            # into a still-running row's cache.  Inspect still schedules up to
            # the frozen connection count; this is only the model batch size.
            batch_size=1,
        )
    )
    choice = hf.ChatCompletionChoice(
        message=hf.chat_completion_assistant_message(response, tools, handler, self.model_name),
        logprobs=None,
    )
    return hf.ModelOutput(
        model=self.model_name,
        choices=[choice],
        usage=hf.ModelUsage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
        ),
        time=response.time,
        metadata={
            "hidden_states": None,
            "r005_no_token_cap": True,
            "r005_termination": "model_eos_only",
            "qwen_enable_thinking": False,
        },
    )


def install_native_hf_eos_only_sampling() -> dict[str, Any]:
    """Install the exact r005 no-cap hook once in the current interpreter.

    The task factory is called before ``scripts.run_evals`` resolves a model,
    so this installs before either the base or the local-PEFT HF provider is
    constructed.  It is deliberately process-local and only active with the
    explicit r005 environment marker.
    """

    _require_exact_runtime()
    from inspect_ai.model._providers.hf import HuggingFaceAPI

    if getattr(HuggingFaceAPI, "_ctm_aita_r005_eos_only_installed", False):
        return runtime_policy()

    original_generate = HuggingFaceAPI.generate
    original_max_tokens = HuggingFaceAPI.max_tokens

    async def patched_generate(self: Any, input: Any, tools: Any, tool_choice: Any, config: Any) -> Any:
        return await _eos_only_hf_generate(self, input, tools, tool_choice, config)

    def patched_max_tokens(self: Any) -> None:
        # ``Model.generate`` asks this method only when its config has no
        # explicit max_tokens.  Returning None preserves that absence for our
        # EOS-only provider implementation.
        return None

    # Keep original methods attached for forensic inspection; no production
    # code may call them while the r005 marker is active.
    HuggingFaceAPI._ctm_aita_r005_original_generate = original_generate
    HuggingFaceAPI._ctm_aita_r005_original_max_tokens = original_max_tokens
    HuggingFaceAPI.generate = patched_generate
    HuggingFaceAPI.max_tokens = patched_max_tokens
    HuggingFaceAPI._ctm_aita_r005_eos_only_installed = True
    return runtime_policy()


__all__ = [
    "INSPECT_VERSION",
    "QWEN_THINKING_POLICY",
    "RUNTIME_ENV",
    "RUNTIME_ENV_VALUE",
    "RUNTIME_POLICY",
    "RUNTIME_SCHEMA",
    "TRANSFORMERS_VERSION",
    "NoTokenCapRuntimeError",
    "install_native_hf_eos_only_sampling",
    "runtime_policy",
]
