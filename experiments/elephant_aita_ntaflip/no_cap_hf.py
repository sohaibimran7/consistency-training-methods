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
from copy import deepcopy
from typing import Any

from ctm.evals.hf_eos_kernel import (
    HFEOSError as NoTokenCapRuntimeError,
    TOKEN_CAP_FIELD_NAMES,
    assert_no_runtime_token_cap as _assert_no_runtime_token_cap,
    assert_no_token_cap_mapping,
    concrete_eos_ids as _concrete_eos_ids,
    config_value as _config_value,
    configuration_mapping as _configuration_mapping,
    effective_sampling_value as _effective_sampling_value,
    eos_ids as _eos_ids,
    eos_only_model_generate as _eos_only_model_generate,
    pad_token_id as _pad_token_id,
    sample_next_tokens as _sample_next_tokens,
    saved_generation_config as _saved_generation_config,
    saved_logits_processor as _saved_logits_processor,
    saved_suppress_tokens as _saved_suppress_tokens,
    supports_logits_to_keep as _supports_logits_to_keep,
)

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


# Existing diagnostic callers may still use these names; all implementation
# lives in the generic kernel above.
__all__ = [
    "assert_no_token_cap_mapping",
    "_concrete_eos_ids",
    "_configuration_mapping",
    "_effective_sampling_value",
    "_eos_ids",
    "_pad_token_id",
    "_sample_next_tokens",
    "_saved_generation_config",
    "_saved_logits_processor",
    "_saved_suppress_tokens",
    "_supports_logits_to_keep",
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
