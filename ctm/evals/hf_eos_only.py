"""Explicit uncapped, EOS-only sampling for pinned native-HF evaluations.

Inspect and Transformers both have mutable/default length limits.  A run that
sets the activation marker below replaces Inspect's native-HF generation call
with the already-tested one-sequence autoregressive loop used by the AITA
replication.  That loop has no token counter or length stopping criterion; it
returns only after the model emits EOS.
"""

from __future__ import annotations

import functools
import importlib.metadata
import os
from typing import Any

from .hf_eos_kernel import (
    HFEOSError,
    assert_no_runtime_token_cap as _assert_no_runtime_token_cap,
    config_value as _config_value,
    eos_only_model_generate as _eos_only_model_generate,
)


RUNTIME_ENV = "CTM_HF_EOS_ONLY_NO_TOKEN_CAP"
RUNTIME_ENV_VALUE = "1"
EXPECTED_INSPECT_ENV = "CTM_HF_EOS_ONLY_EXPECTED_INSPECT"
EXPECTED_TRANSFORMERS_ENV = "CTM_HF_EOS_ONLY_EXPECTED_TRANSFORMERS"
RUNTIME_SCHEMA = "ctm-native-hf-eos-only-no-token-cap-v1"


def _required_environment() -> tuple[str, str]:
    if os.environ.get(RUNTIME_ENV) != RUNTIME_ENV_VALUE:
        raise HFEOSError(f"EOS-only native HF requires {RUNTIME_ENV}={RUNTIME_ENV_VALUE!r}")
    inspect_version = os.environ.get(EXPECTED_INSPECT_ENV, "")
    transformers_version = os.environ.get(EXPECTED_TRANSFORMERS_ENV, "")
    if not inspect_version or not transformers_version:
        raise HFEOSError("EOS-only native HF requires explicit expected package versions")
    if importlib.metadata.version("inspect-ai") != inspect_version:
        raise HFEOSError("installed Inspect version differs from the EOS-only runtime contract")
    if importlib.metadata.version("transformers") != transformers_version:
        raise HFEOSError("installed Transformers version differs from the EOS-only runtime contract")
    return inspect_version, transformers_version


def runtime_policy() -> dict[str, Any]:
    inspect_version, transformers_version = _required_environment()
    return {
        "schema": RUNTIME_SCHEMA,
        "activation_environment": {
            RUNTIME_ENV: RUNTIME_ENV_VALUE,
            EXPECTED_INSPECT_ENV: inspect_version,
            EXPECTED_TRANSFORMERS_ENV: transformers_version,
        },
        "sampling": "native-hf-autoregressive-eos-only",
        "batching": "one-sequence-per-native-hf-generate-call",
        "standard_transformers_generate": "not-invoked",
        "inspect_hf_default_max_tokens": "overridden-to-none",
        "termination": "model_eos_only",
        "output_token_cap": None,
    }


async def _generate(self: Any, input: Any, tools: Any, tool_choice: Any, config: Any) -> Any:
    _required_environment()
    _assert_no_runtime_token_cap(config, label="effective Inspect generation config")
    if self.do_sample is not True:
        raise HFEOSError("EOS-only native HF requires explicit stochastic sampling")
    if _config_value(config, "stop_seqs") is not None:
        raise HFEOSError("EOS-only native HF permits only model-EOS termination")
    from inspect_ai.model._providers import hf

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
            "ctm_no_output_token_cap": True,
            "ctm_termination": "model_eos_only",
        },
    )


def install_native_hf_eos_only_sampling() -> dict[str, Any]:
    """Install the fail-closed runtime hook once in this interpreter."""

    policy = runtime_policy()
    from inspect_ai.model._providers.hf import HuggingFaceAPI

    if getattr(HuggingFaceAPI, "_ctm_generic_eos_only_installed", False):
        return policy
    if getattr(HuggingFaceAPI, "_ctm_aita_r005_eos_only_installed", False):
        raise HFEOSError("a different native-HF EOS-only hook is already installed")
    HuggingFaceAPI._ctm_generic_original_generate = HuggingFaceAPI.generate
    HuggingFaceAPI._ctm_generic_original_max_tokens = HuggingFaceAPI.max_tokens
    HuggingFaceAPI.generate = _generate
    HuggingFaceAPI.max_tokens = lambda self: None
    HuggingFaceAPI._ctm_generic_eos_only_installed = True
    return policy


__all__ = [
    "EXPECTED_INSPECT_ENV",
    "EXPECTED_TRANSFORMERS_ENV",
    "HFEOSError",
    "RUNTIME_ENV",
    "RUNTIME_ENV_VALUE",
    "RUNTIME_SCHEMA",
    "install_native_hf_eos_only_sampling",
    "runtime_policy",
]
