"""Tinker service adapter — the original backend, now behind the protocol.

All Tinker SDK service calls the training loops used to make directly live here:
client construction, sampling, forward/backward + optim futures, KL-to-base via
the cookbook helper, and checkpoint save. Construction is lazy (no ServiceClient
until ``setup()``), so importing/instantiating never needs credentials.
"""

from pathlib import Path
from typing import Any, Optional, Sequence

import tinker
from tinker import types
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl.metrics import incorporate_kl_penalty as _cookbook_incorporate_kl_penalty
from tinker_cookbook.rl.train import _remove_mask as remove_mask

from ctm.backends.base import ForwardBackwardOutput, SampledSequence
from ctm.core.config import AdamConfig, LoRAConfig


class TinkerSamplerHandle:
    """Wraps a ``tinker.SamplingClient`` behind the SamplerHandle protocol."""

    def __init__(self, client: tinker.SamplingClient):
        self.client = client

    async def sample(self, prompt: Any, *, max_tokens: int, temperature: float,
                     stop: Any, num_samples: int) -> list[SampledSequence]:
        result = await self.client.sample_async(
            prompt=prompt,
            sampling_params=types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            ),
            num_samples=num_samples,
        )
        return [
            SampledSequence(
                tokens=list(seq.tokens),
                logprobs=list(seq.logprobs) if seq.logprobs else None,
            )
            for seq in result.sequences
        ]


class _TinkerPendingForwardBackward:
    def __init__(self, future):
        self._future = future

    async def result(self) -> ForwardBackwardOutput:
        res = await self._future.result_async()
        logprobs = [output["logprobs"].to_torch() for output in res.loss_fn_outputs]
        metrics = dict(res.metrics) if getattr(res, "metrics", None) else {}
        return ForwardBackwardOutput(logprobs=logprobs, metrics=metrics)


class _TinkerPendingOptimStep:
    def __init__(self, future):
        self._future = future

    async def result(self) -> None:
        await self._future.result_async()


class TinkerBackend:
    """TrainingBackend implementation on the Tinker service (LoRA training)."""

    def __init__(self, service_client: Optional[tinker.ServiceClient] = None):
        self._service_client = service_client
        self.training_client: Optional[tinker.TrainingClient] = None
        self._base_sampling_client: Optional[tinker.SamplingClient] = None
        self.model: Optional[str] = None

    # ── lifecycle ────────────────────────────────────────────────────────

    @property
    def service_client(self) -> tinker.ServiceClient:
        if self._service_client is None:
            self._service_client = tinker.ServiceClient()
        return self._service_client

    def setup(self, *, model: str, lora: LoRAConfig,
              resume_from: Optional[str] = None, resume_with_optimizer: bool = False) -> None:
        self.model = model
        self.training_client = self.service_client.create_lora_training_client(
            base_model=model,
            **lora.model_dump(),
        )
        if resume_from:
            if resume_with_optimizer:
                print(f"Loading weights + optimizer from: {resume_from}")
                self.training_client.load_state_with_optimizer(resume_from).result()
            else:
                print(f"Loading weights from: {resume_from}")
                self.training_client.load_state(resume_from).result()
            print("Checkpoint loaded successfully")

    def _require_training_client(self) -> tinker.TrainingClient:
        if self.training_client is None:
            raise RuntimeError("TinkerBackend.setup() must be called before use")
        return self.training_client

    # ── samplers ─────────────────────────────────────────────────────────

    def policy_sampler(self, name: str) -> TinkerSamplerHandle:
        client = self._require_training_client().save_weights_and_get_sampling_client(name=name)
        return TinkerSamplerHandle(client)

    async def refresh_policy_sampler(self, name: str) -> TinkerSamplerHandle:
        client = await self._require_training_client().save_weights_and_get_sampling_client_async(name=name)
        return TinkerSamplerHandle(client)

    def base_sampler(self) -> TinkerSamplerHandle:
        if self._base_sampling_client is None:
            if self.model is None:
                raise RuntimeError("TinkerBackend.setup() must be called before base_sampler()")
            self._base_sampling_client = self.service_client.create_sampling_client(base_model=self.model)
        return TinkerSamplerHandle(self._base_sampling_client)

    # ── training ─────────────────────────────────────────────────────────

    async def submit_forward_backward(self, datums: Sequence[Any], loss_fn: str) -> _TinkerPendingForwardBackward:
        future = await self._require_training_client().forward_backward_async(
            [remove_mask(d) for d in datums], loss_fn=loss_fn
        )
        return _TinkerPendingForwardBackward(future)

    async def submit_optim_step(self, *, learning_rate: float, adam: AdamConfig) -> _TinkerPendingOptimStep:
        future = await self._require_training_client().optim_step_async(
            types.AdamParams(
                learning_rate=learning_rate,
                beta1=adam.beta1,
                beta2=adam.beta2,
                eps=adam.eps,
                weight_decay=adam.weight_decay,
                grad_clip_norm=adam.grad_clip_norm,
            )
        )
        return _TinkerPendingOptimStep(future)

    async def incorporate_kl_penalty(self, datums: Sequence[Any], *, kl_coef: float,
                                     kl_discount_factor: float) -> dict[str, float]:
        # The cookbook helper scores the sampled tokens under the frozen base model
        # via a raw sampling client, then folds -kl_coef * KL into the advantages.
        self.base_sampler()  # ensure the raw base client exists
        assert self._base_sampling_client is not None
        return await _cookbook_incorporate_kl_penalty(
            data_D=list(datums),
            base_sampling_client=self._base_sampling_client,
            kl_penalty_coef=kl_coef,
            kl_discount_factor=kl_discount_factor,
        )

    # ── checkpoints ──────────────────────────────────────────────────────

    async def save_checkpoint(self, *, name: str, log_dir: str | Path, loop_state: dict, kind: str) -> dict:
        return await checkpoint_utils.save_checkpoint_async(
            self._require_training_client(),
            name=name,
            log_path=str(log_dir),
            loop_state=loop_state,
            kind=kind,
        )
