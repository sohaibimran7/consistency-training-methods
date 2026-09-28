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

import gc
import math
from types import SimpleNamespace
from typing import Any, Optional

from ctm.backends.base import SampledSequence
from ctm.backends.local.qwen35_vllm_compat import (
    is_qwen35_model_name,
    validate_qwen35_vllm_rollout_compat_adapter,
)


def _qwen35_runtime_lora_is_unsafe(model_name: str) -> bool:
    """Return whether vLLM 0.26 would silently drop this model's PEFT LoRA.

    Qwen3.5 PEFT checkpoints trained through Transformers name their text
    blocks ``model.layers.*``.  vLLM 0.26 serves Qwen3.5 through a conditional
    wrapper whose runtime blocks are ``language_model.model.layers.*`` and its
    adapter mapper lacks that bridge.  The server accepts the adapter but uses
    zero of its tensors.  This matters for policy rollouts as much as for
    evaluation: a no-op rollout policy invalidates OPCT's on-policy data.
    """

    return is_qwen35_model_name(model_name)


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
    """Owns the vLLM engine and the current policy-adapter snapshot.

    ``enable_sleep_mode=True`` opts into vLLM's RLHF-oriented sleep lifecycle.
    In that mode :meth:`sleep` moves the engine's weights off the GPU (and
    discards its KV cache), while :meth:`wake_up` restores the engine before an
    operation that needs it.  The methods are deliberately idempotent: a
    higher-level worker pool can use them as phase barriers without having to
    infer whether a preceding training operation actually slept the engine.
    """

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
        # Keep this option in ``engine_kwargs``: vLLM must be constructed with
        # sleep support enabled before its EngineCore is launched.  Treat it as
        # an explicit opt-in so existing local and worker-pool deployments keep
        # their historical resident-engine behavior.
        self.sleep_enabled = bool(engine_kwargs.get("enable_sleep_mode", False))
        self.sleeping = False
        self._unsafe_qwen35_lora = enable_lora and _qwen35_runtime_lora_is_unsafe(model)
        requested_logprobs_mode = engine_kwargs.get("logprobs_mode", "processed_logprobs")
        if requested_logprobs_mode != "processed_logprobs":
            raise ValueError(
                "VLLMSampler requires logprobs_mode='processed_logprobs' so generated-token "
                "scores describe the actual sampling distribution; "
                f"got {requested_logprobs_mode!r}"
            )
        # Generated-token scores must include temperature and any other
        # sampling processors because they are the off-policy denominator.
        # vLLM prompt_logprobs remain raw in either mode: prompt tokens never
        # pass through sampling processors, so completion rescoring below still
        # returns raw student/teacher policy scores.
        engine_kwargs["logprobs_mode"] = "processed_logprobs"
        if engine is not None:
            self.engine = engine
        else:
            kwargs = {"gpu_memory_utilization": 0.45, "max_lora_rank": 64, **engine_kwargs}
            if not enable_lora:
                kwargs.pop("max_lora_rank", None)
            self.engine = self._api.LLM(model=model, enable_lora=enable_lora, **kwargs)
        self.adapter_dir: Optional[str] = None
        self.adapter_version: int = 0

    def sleep(self, *, level: int = 1) -> bool:
        """Release vLLM GPU allocations while a colocated trainer is active.

        Level 1 is the intended production mode: it releases GPU weights and
        KV-cache allocations while retaining weights in host memory for a fast
        wake.  ``False`` means sleep mode was not enabled (or the sampler had
        already been shut down); ``True`` means the sampler is now, or already
        was, asleep.
        """

        if not self.sleep_enabled or self.engine is None:
            return False
        if self.sleeping:
            return True
        if level not in {1, 2}:
            raise ValueError(f"vLLM sleep level must be 1 or 2, got {level}")
        sleep = getattr(self.engine, "sleep", None)
        if not callable(sleep):
            raise RuntimeError(
                "vLLM sleep mode was requested but this engine does not expose sleep(); "
                "install a vLLM release with enable_sleep_mode support"
            )
        sleep(level=level)
        self.sleeping = True
        return True

    def wake_up(self) -> bool:
        """Restore a sleeping vLLM engine before sampling or scoring.

        Returns ``True`` if the engine is now, or already was, awake under an
        enabled lifecycle; returns ``False`` when sleep mode is disabled or the
        sampler has been shut down.
        """

        if not self.sleep_enabled or self.engine is None:
            return False
        if not self.sleeping:
            return True
        wake_up = getattr(self.engine, "wake_up", None)
        if not callable(wake_up):
            raise RuntimeError(
                "vLLM sleep mode was requested but this engine does not expose wake_up(); "
                "install a vLLM release with enable_sleep_mode support"
            )
        wake_up()
        self.sleeping = False
        return True

    def shutdown(self) -> None:
        """Release the vLLM engine and its worker process, if it was started."""
        if getattr(self, "engine", None) is None:
            return
        # vLLM has no stable public shutdown method across supported versions;
        # dropping the final LLM reference tears down its EngineCore worker.
        self.engine = None
        self.sleeping = False
        gc.collect()

    def advance_policy(self, adapter_dir: str, *, version: Optional[int] = None) -> None:
        """Point policy sampling at a freshly saved adapter snapshot.

        The version bump makes the LoRARequest id unique, defeating vLLM's
        adapter cache so the new weights actually load.  Distributed rollout
        workers pass the coordinator's explicit ``version`` so every worker can
        acknowledge the same policy barrier before policy sampling resumes.
        The in-process backend keeps the historical implicit increment.
        """
        # A sleeping engine cannot safely accept a new LoRA request.  Waking
        # here makes direct users safe; the worker pool also wakes explicitly
        # as part of its version-publish barrier, so ordinary distributed runs
        # retain a visible, measured lifecycle transition.
        self.wake_up()
        next_version = self.adapter_version + 1 if version is None else version
        if self._unsafe_qwen35_lora:
            # Raw Transformers/PEFT keys are a silent vLLM no-op.  The only
            # permitted exception is the versioned compatibility copy made by
            # the local rollout publisher.  Its sidecar binds this path to the
            # untouched raw snapshot and catches a missing/stale/corrupt copy
            # before any on-policy sample can be drawn from it.
            try:
                validate_qwen35_vllm_rollout_compat_adapter(
                    adapter_dir,
                    expected_version=next_version,
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "vLLM LoRA policy sampling for Qwen3.5 is disabled for raw PEFT adapters: "
                    "vLLM 0.26 maps no `model.layers.*` tensors to "
                    "`language_model.model.layers.*`. Publish the per-version, hash-bound "
                    "Qwen3.5 compatibility snapshot before sampling. "
                    f"Validation failed for {adapter_dir!r}: {exc}"
                ) from exc
        if next_version <= self.adapter_version:
            raise ValueError(
                f"adapter version must increase (current={self.adapter_version}, requested={next_version})"
            )
        self.adapter_dir = adapter_dir
        self.adapter_version = next_version

    def _policy_lora_request(self):
        if not self.enable_lora:
            return None
        if self.adapter_dir is None:
            if self._unsafe_qwen35_lora:
                raise RuntimeError(
                    "Qwen3.5 policy sampling requires an acknowledged per-version compatibility adapter; "
                    "refusing to silently sample the frozen base"
                )
            return None
        return self._api.LoRARequest(f"policy_v{self.adapter_version}", self.adapter_version, self.adapter_dir)

    def sample(
        self,
        prompt_tokens: list[int],
        *,
        max_tokens: int | None,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
        ignore_eos: bool = False,
    ) -> list[SampledSequence]:
        return self.sample_batch(
            [prompt_tokens],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            num_samples=num_samples,
            use_base=use_base,
            ignore_eos=ignore_eos,
        )[0]

    def sample_batch(
        self,
        prompt_tokens_batch: list[list[int]],
        *,
        max_tokens: int | None,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
        ignore_eos: bool = False,
    ) -> list[list[SampledSequence]]:
        """Generate a scheduler-visible batch in one vLLM engine call."""
        if not prompt_tokens_batch:
            return []
        self.wake_up()
        if self.engine is None:
            raise RuntimeError("vLLM sampler has been shut down")
        stop_ids = [t for t in (stop or []) if isinstance(t, int)] or None
        params = self._api.SamplingParams(
            n=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_token_ids=stop_ids,
            ignore_eos=ignore_eos,
            logprobs=0,  # 0 extra top-k → still returns the sampled token's own logprob
        )
        outputs = self.engine.generate(
            [self._api.TokensPrompt(prompt_token_ids=list(tokens)) for tokens in prompt_tokens_batch],
            params,
            lora_request=None if use_base else self._policy_lora_request(),
            use_tqdm=False,
        )
        if len(outputs) != len(prompt_tokens_batch):
            raise RuntimeError(f"vLLM returned {len(outputs)} prompt results for a batch of {len(prompt_tokens_batch)}")
        return [
            self._extract_sequences(output, require_eos_only=max_tokens is None)
            for output in outputs
        ]

    def score_completions(
        self,
        prompt_tokens_batch: list[list[int]],
        completion_tokens_batch: list[list[int]],
        *,
        use_base: bool,
    ) -> list[list[float]]:
        """Return exact raw-policy logprobs for supplied continuations.

        vLLM exposes teacher-forced token scores through ``prompt_logprobs``.
        Each request therefore uses ``prompt + completion`` as its prompt. The
        API still requires generation, so ``max_tokens=1`` adds one unused
        token after the scored context.
        """

        return self._score_completions_with_tail(
            prompt_tokens_batch,
            completion_tokens_batch,
            use_base=use_base,
            tail_max_tokens=1,
            require_eos_only=False,
        )

    def score_completions_uncapped_eos_tail(
        self,
        prompt_tokens_batch: list[list[int]],
        completion_tokens_batch: list[list[int]],
        *,
        use_base: bool,
    ) -> list[list[float]]:
        """Prompt-score continuations while the API's unused tail is uncapped.

        vLLM exposes the requested teacher-forced values as prompt logprobs but
        still starts a continuation after that prompt.  Muse parity probes use
        this method so that continuation has ``max_tokens=None`` and must stop
        through EOS.  It is deliberately separate from the historical scorer:
        production Muse training never calls either prompt-scoring path.
        """

        return self._score_completions_with_tail(
            prompt_tokens_batch,
            completion_tokens_batch,
            use_base=use_base,
            tail_max_tokens=None,
            require_eos_only=True,
        )

    def _score_completions_with_tail(
        self,
        prompt_tokens_batch: list[list[int]],
        completion_tokens_batch: list[list[int]],
        *,
        use_base: bool,
        tail_max_tokens: int | None,
        require_eos_only: bool,
    ) -> list[list[float]]:
        if len(prompt_tokens_batch) != len(completion_tokens_batch):
            raise ValueError(
                "prompt_tokens_batch and completion_tokens_batch must have the same length, "
                f"got {len(prompt_tokens_batch)} and {len(completion_tokens_batch)}"
            )
        if not prompt_tokens_batch:
            return []
        self.wake_up()
        if self.engine is None:
            raise RuntimeError("vLLM sampler has been shut down")

        prompts = [list(tokens) for tokens in prompt_tokens_batch]
        completions = [list(tokens) for tokens in completion_tokens_batch]
        for index, (prompt, completion) in enumerate(zip(prompts, completions)):
            if not prompt:
                raise ValueError(f"prompt {index} is empty")
            if not completion:
                raise ValueError(f"completion {index} is empty")

        combined = [prompt + completion for prompt, completion in zip(prompts, completions)]
        params = self._api.SamplingParams(
            n=1,
            max_tokens=tail_max_tokens,
            temperature=0.0,
            prompt_logprobs=0,
        )
        outputs = self.engine.generate(
            [self._api.TokensPrompt(prompt_token_ids=tokens) for tokens in combined],
            params,
            lora_request=None if use_base else self._policy_lora_request(),
            use_tqdm=False,
        )
        if len(outputs) != len(combined):
            raise RuntimeError(f"vLLM returned {len(outputs)} score results for a batch of {len(combined)}")
        if require_eos_only:
            for output in outputs:
                self._extract_sequences(output, require_eos_only=True)

        scores: list[list[float]] = []
        for request_index, (output, expected_tokens, prompt, completion) in enumerate(
            zip(outputs, combined, prompts, completions)
        ):
            returned_tokens = getattr(output, "prompt_token_ids", None)
            if returned_tokens is None:
                raise RuntimeError(f"vLLM score result {request_index} omitted prompt_token_ids")
            returned_tokens = list(returned_tokens)
            if returned_tokens != expected_tokens:
                raise RuntimeError(
                    f"vLLM score result {request_index} prompt tokens are misaligned: "
                    f"expected {len(expected_tokens)} exact token(s), got {len(returned_tokens)}"
                )

            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            if prompt_logprobs is None:
                raise RuntimeError(f"vLLM score result {request_index} omitted prompt_logprobs")
            if len(prompt_logprobs) != len(expected_tokens):
                raise RuntimeError(
                    f"vLLM score result {request_index} returned {len(prompt_logprobs)} prompt-logprob "
                    f"position(s) for {len(expected_tokens)} token(s)"
                )

            completion_scores: list[float] = []
            for completion_index, (position, token) in enumerate(
                zip(range(len(prompt), len(expected_tokens)), completion)
            ):
                entry_dict = prompt_logprobs[position]
                if entry_dict is None or not callable(getattr(entry_dict, "get", None)):
                    raise RuntimeError(
                        f"vLLM score result {request_index} has no logprob mapping for completion token "
                        f"{completion_index} at prompt position {position}"
                    )
                entry = entry_dict.get(token)
                if entry is None:
                    raise RuntimeError(
                        f"vLLM score result {request_index} omitted token {token} at completion index "
                        f"{completion_index} (prompt position {position})"
                    )
                try:
                    value = float(entry.logprob)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"vLLM score result {request_index} has an invalid logprob for completion token "
                        f"{completion_index}"
                    ) from exc
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"vLLM score result {request_index} has non-finite logprob {value} for completion "
                        f"token {completion_index}"
                    )
                completion_scores.append(value)
            if len(completion_scores) != len(completion):
                raise RuntimeError(
                    f"vLLM score result {request_index} produced {len(completion_scores)} score(s) "
                    f"for {len(completion)} completion token(s)"
                )
            scores.append(completion_scores)
        return scores

    @staticmethod
    def _extract_sequences(
        output: Any,
        *,
        require_eos_only: bool = False,
    ) -> list[SampledSequence]:
        sequences: list[SampledSequence] = []
        for completion in output.outputs:
            if require_eos_only:
                finish_reason = getattr(completion, "finish_reason", None)
                if finish_reason != "stop":
                    raise RuntimeError(
                        "uncapped vLLM generation did not terminate through an EOS/stop token: "
                        f"finish_reason={finish_reason!r}, stop_reason={getattr(completion, 'stop_reason', None)!r}"
                    )
            tokens = list(completion.token_ids)
            finish_reason = getattr(completion, "finish_reason", None) or "unknown"
            logprobs: Optional[list[float]] = None
            if completion.logprobs is not None:
                if len(completion.logprobs) != len(tokens):
                    raise ValueError("vLLM returned mismatched token/logprob lengths")
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
            sequence = SampledSequence(tokens=tokens, logprobs=logprobs, finish_reason=finish_reason)
            sequence.validate()
            sequences.append(sequence)
        return sequences
