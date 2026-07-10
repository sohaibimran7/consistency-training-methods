"""Fast rollout sampling via in-process vLLM with LoRA hot-reload.

The training model (transformers+PEFT) and the vLLM engine coexist in one
process: on every policy refresh the backend snapshots the current adapter to a
scratch directory and calls ``advance_policy``; subsequent policy samples attach
a ``LoRARequest`` with a fresh id (vLLM caches adapters by id, so a new id per
snapshot forces a reload). Base-model sampling simply omits the LoRA request —
the engine's weights ARE the frozen base.

Memory note: vLLM holds its own copy of the base weights, so budget
``gpu_memory_utilization`` (default 0.45 here) alongside the training model, or
run vLLM on a second GPU via ``vllm_options={"tensor_parallel_size": ...}`` /
CUDA_VISIBLE_DEVICES splits.

vLLM is intentionally NOT a hard dependency (no macOS wheels; aarch64/GH200
needs a platform build) — imports are lazy, and the ``engine``/``api`` hooks
exist so the request/extraction logic is unit-testable without vLLM installed.
Written against the vllm>=0.6 API (TokensPrompt, LoRARequest, SamplingParams).
"""

from types import SimpleNamespace
from typing import Any, Optional

from ctm.backends.base import SampledSequence


def _load_vllm_api() -> SimpleNamespace:
    try:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.lora.request import LoRARequest
    except ImportError as e:
        raise ImportError(
            "vLLM sampling requires the vllm package, which ships platform-specific wheels "
            "(CUDA x86 / aarch64). Install it on the GPU box (`uv pip install vllm`), or use "
            "sampler='hf' (correct but slow) for debugging."
        ) from e
    return SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams, TokensPrompt=TokensPrompt, LoRARequest=LoRARequest)


class VLLMSampler:
    """Owns the vLLM engine and the current policy-adapter snapshot."""

    def __init__(
        self,
        model: str,
        *,
        enable_lora: bool = True,
        engine: Optional[Any] = None,
        api: Optional[SimpleNamespace] = None,
        **engine_kwargs,
    ):
        """
        Args:
            model: HF model id / path for the frozen base weights.
            enable_lora: allow per-request LoRA (required for policy sampling).
            engine: pre-built engine (tests); skips LLM construction.
            api: vllm API namespace override (tests).
            engine_kwargs: forwarded to ``vllm.LLM`` (gpu_memory_utilization,
                max_model_len, tensor_parallel_size, max_lora_rank, ...).
        """
        self._api = api if api is not None else _load_vllm_api()
        self.enable_lora = enable_lora
        if engine is not None:
            self.engine = engine
        else:
            kwargs = {"gpu_memory_utilization": 0.45, "max_lora_rank": 64, **engine_kwargs}
            if not enable_lora:
                kwargs.pop("max_lora_rank", None)
            self.engine = self._api.LLM(model=model, enable_lora=enable_lora, **kwargs)
        self.adapter_dir: Optional[str] = None
        self.adapter_version: int = 0

    def advance_policy(self, adapter_dir: str) -> None:
        """Point policy sampling at a freshly saved adapter snapshot.

        The version bump makes the LoRARequest id unique, defeating vLLM's
        adapter cache so the new weights actually load.
        """
        self.adapter_dir = adapter_dir
        self.adapter_version += 1

    def _policy_lora_request(self):
        if not self.enable_lora or self.adapter_dir is None:
            return None
        return self._api.LoRARequest(f"policy_v{self.adapter_version}", self.adapter_version, self.adapter_dir)

    def sample(
        self,
        prompt_tokens: list[int],
        *,
        max_tokens: int,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
    ) -> list[SampledSequence]:
        stop_ids = [t for t in (stop or []) if isinstance(t, int)] or None
        params = self._api.SamplingParams(
            n=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_token_ids=stop_ids,
            logprobs=0,  # 0 extra top-k → still returns the sampled token's own logprob
        )
        outputs = self.engine.generate(
            [self._api.TokensPrompt(prompt_token_ids=list(prompt_tokens))],
            params,
            lora_request=None if use_base else self._policy_lora_request(),
            use_tqdm=False,
        )
        sequences: list[SampledSequence] = []
        for completion in outputs[0].outputs:
            tokens = list(completion.token_ids)
            logprobs: Optional[list[float]] = None
            if completion.logprobs is not None:
                logprobs = []
                for token, entry_dict in zip(tokens, completion.logprobs):
                    entry = entry_dict.get(token)
                    if entry is None:
                        # A sampled token without its own logprob poisons the IS
                        # ratio downstream — mark the whole sequence logprob-less
                        # so the loop excludes it (same contract as Tinker).
                        logprobs = None
                        break
                    logprobs.append(float(entry.logprob))
            sequences.append(SampledSequence(tokens=tokens, logprobs=logprobs))
        return sequences
