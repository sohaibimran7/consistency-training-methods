"""LocalBackend — in-process torch training + HF sampling for self-hosted GPUs.

Implements the ``TrainingBackend`` protocol on a single node (workstation, a
Vast.ai box, or one Isambard GH200 node): training via ``transformers`` (+ PEFT
LoRA when installed), rollouts via ``model.generate``. The same tinker Datums the
loops already build are consumed directly (they're offline containers), so the
loops don't know which backend they're on.

Notes / current limits (phase 1):
- LoRA requires ``peft`` (``uv pip install peft``); without it use
  ``use_lora=False`` (full fine-tune — no KL-to-base, which needs the frozen
  base via ``disable_adapter``).
- ``submit_*`` executes eagerly and returns an already-resolved pending object —
  the two-phase protocol shape is preserved, the prefetch overlap just buys
  nothing extra in-process.
- Sampling is HF ``generate`` (correct, not fast). A vLLM sampler with LoRA
  hot-reload is the planned fast path for real runs; see
  ``ctm/backends/local/__init__.py``.
- Checkpoints are directories: ``<log_dir>/checkpoints/<name>/`` with adapter or
  full weights, optional ``optimizer.pt``, and a ``manifest.json`` (returned
  paths use the ``file://`` scheme so eval runners can dispatch on it).
"""

import json
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized

from ctm.backends.base import ForwardBackwardOutput, SampledSequence
from ctm.backends.local import losses
from ctm.core.config import AdamConfig, LoRAConfig

try:  # optional: LoRA support
    import peft
    from peft import LoraConfig as PeftLoraConfig, get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


def _strip_file_scheme(path: str) -> Path:
    return Path(path[len("file://") :] if path.startswith("file://") else path)


class _ResolvedPending:
    """Already-computed result behind the two-phase pending interface."""

    def __init__(self, value):
        self._value = value

    async def result(self):
        return self._value


class LocalSamplerHandle:
    """Samples from the backend's live model (policy) or its frozen base."""

    def __init__(self, backend: "LocalBackend", use_base: bool):
        self._backend = backend
        self._use_base = use_base

    async def sample(
        self, prompt: Any, *, max_tokens: int, temperature: float, stop: Any, num_samples: int
    ) -> list[SampledSequence]:
        return self._backend._sample(
            prompt_tokens=list(prompt.to_ints()),
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            num_samples=num_samples,
            use_base=self._use_base,
        )


class LocalBackend:
    """TrainingBackend on local hardware (torch + transformers [+ peft])."""

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        use_lora: bool = True,
        model_instance: Optional[torch.nn.Module] = None,
        ppo_clip_epsilon: float = losses.PPO_CLIP_EPSILON,
        sampler: str = "hf",
        vllm_options: Optional[dict] = None,
    ):
        """
        Args:
            device: "cuda" / "cpu" / "mps"; auto-detects cuda when None.
            dtype: model dtype (bfloat16 recommended on GPU).
            use_lora: wrap the model with a PEFT LoRA adapter (requires peft).
            model_instance: pre-built model (e.g. a tiny random model in tests);
                skips ``from_pretrained`` in setup().
            ppo_clip_epsilon: clip range for the "ppo" loss_fn.
            sampler: rollout engine — "hf" (model.generate; correct, slow) or
                "vllm" (in-process vLLM with LoRA hot-reload; production path).
                The vLLM engine boots lazily on the first sampler request, so
                runs that never sample (SFT) pay nothing for it.
            vllm_options: forwarded to VLLMSampler / vllm.LLM
                (gpu_memory_utilization, tensor_parallel_size, max_model_len, ...).
        """
        if sampler not in ("hf", "vllm"):
            raise ValueError(f"Unknown sampler: {sampler!r} (expected 'hf' or 'vllm')")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.use_lora = use_lora
        self.model: Optional[torch.nn.Module] = model_instance
        self.model_name: Optional[str] = None
        self.ppo_clip_epsilon = ppo_clip_epsilon
        self.sampler = sampler
        self.vllm_options = vllm_options or {}
        self._vllm = None  # VLLMSampler, booted lazily by _ensure_vllm() when sampler == "vllm"
        self._adapter_scratch: Optional[Path] = None
        self._optimizer: Optional[torch.optim.AdamW] = None
        self._pending_optimizer_state: Optional[dict] = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def setup(
        self, *, model: str, lora: LoRAConfig, resume_from: Optional[str] = None, resume_with_optimizer: bool = False
    ) -> None:
        self.model_name = model
        if self.model is None:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=self.dtype)
        if self.use_lora:
            if not HAS_PEFT:
                raise ImportError(
                    "LocalBackend(use_lora=True) requires peft: `uv pip install peft`, "
                    "or pass use_lora=False for full fine-tuning."
                )
            if not (lora.train_mlp and lora.train_attn):
                raise NotImplementedError(
                    "LocalBackend maps LoRA onto all linear layers; per-module selection "
                    "(train_mlp/train_attn=False) is not implemented yet."
                )
            if lora.seed is not None:
                torch.manual_seed(lora.seed)
            peft_cfg = PeftLoraConfig(
                r=lora.rank,
                lora_alpha=2 * lora.rank,  # cookbook-style alpha/r = 2
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_cfg)
        self.model.to(self.device)
        if resume_from:
            self._load_checkpoint(resume_from, with_optimizer=resume_with_optimizer)
        if self.sampler == "vllm" and not self.use_lora:
            raise NotImplementedError(
                "sampler='vllm' requires use_lora=True: policy refresh works by "
                "hot-reloading the adapter; full-finetune weights cannot be swapped "
                "into a running vLLM engine."
            )
        # The vLLM engine itself boots lazily (_ensure_vllm) on the first sampler
        # request: runs that never sample (SFT) must not pay its GPU memory or
        # require the vllm package at all.

    def _require_model(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("LocalBackend.setup() must be called before use")
        return self.model

    # ── samplers ─────────────────────────────────────────────────────────

    def _ensure_vllm(self) -> None:
        """Boot the vLLM engine on first use and publish the current adapter."""
        if self.sampler != "vllm" or self._vllm is not None:
            return
        self._require_model()
        import tempfile

        from ctm.backends.local.vllm_sampler import VLLMSampler

        self._adapter_scratch = Path(tempfile.mkdtemp(prefix="ctm-policy-adapter-"))
        self._vllm = VLLMSampler(model=self.model_name, enable_lora=True, **self.vllm_options)
        self._publish_adapter()  # initial policy (fresh or resumed adapter)

    def policy_sampler(self, name: str) -> LocalSamplerHandle:
        self._require_model()
        self._ensure_vllm()
        return LocalSamplerHandle(self, use_base=False)

    async def refresh_policy_sampler(self, name: str) -> LocalSamplerHandle:
        # HF sampler: the in-process model IS the live policy — refresh is free.
        # vLLM sampler: snapshot the adapter and hot-reload it into the engine.
        # (Cold engine: policy_sampler's lazy boot does the initial publish.)
        if self._vllm is not None:
            self._publish_adapter()
        return self.policy_sampler(name)

    def _publish_adapter(self) -> None:
        """Snapshot the current LoRA adapter and point the vLLM engine at it."""
        assert self._vllm is not None and self._adapter_scratch is not None
        version_dir = self._adapter_scratch / f"v{self._vllm.adapter_version + 1}"
        self._require_model().save_pretrained(str(version_dir))
        self._vllm.advance_policy(str(version_dir))

    def base_sampler(self) -> LocalSamplerHandle:
        self._require_base()
        self._ensure_vllm()
        return LocalSamplerHandle(self, use_base=True)

    def _require_base(self):
        if not (self.use_lora and HAS_PEFT):
            raise NotImplementedError(
                "Base-model access (anchor sampling / KL-to-base / distill) on LocalBackend "
                "requires LoRA (the frozen base is the model with the adapter disabled). "
                "For full fine-tuning, run with kl_coef=0 and anchor_model='initial_policy'."
            )

    def _base_ctx(self):
        return self._require_model().disable_adapter()  # peft context manager

    def _sample(
        self,
        *,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
    ) -> list[SampledSequence]:
        """Route sampling to the configured engine (vLLM if set, else HF generate)."""
        if self._vllm is not None:
            return self._vllm.sample(
                prompt_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                num_samples=num_samples,
                use_base=use_base,
            )
        return self._generate(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            num_samples=num_samples,
            use_base=use_base,
        )

    def _generate(
        self,
        *,
        prompt_tokens: list[int],
        max_tokens: int,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
    ) -> list[SampledSequence]:
        model = self._require_model()
        was_training = model.training
        model.eval()
        eos_ids = [t for t in (stop or []) if isinstance(t, int)] or None
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        try:
            with torch.no_grad():
                ctx = self._base_ctx() if use_base else _nullcontext()
                with ctx:
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        temperature=max(temperature, 1e-4),
                        max_new_tokens=max_tokens,
                        num_return_sequences=num_samples,
                        eos_token_id=eos_ids,
                        pad_token_id=eos_ids[0] if eos_ids else 0,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
        finally:
            if was_training:
                model.train()

        prompt_len = input_ids.shape[1]
        sequences = []
        for s in range(out.sequences.shape[0]):
            tokens: list[int] = []
            logprobs: list[float] = []
            for t, step_scores in enumerate(out.scores):
                pos = prompt_len + t
                if pos >= out.sequences.shape[1]:
                    break
                token = int(out.sequences[s, pos])
                lp = torch.log_softmax(step_scores[s].float(), dim=-1)[token]
                tokens.append(token)
                logprobs.append(float(lp))
                if eos_ids and token in eos_ids:
                    break
            sequences.append(SampledSequence(tokens=tokens, logprobs=logprobs))
        return sequences

    # ── training ─────────────────────────────────────────────────────────

    def _target_logprobs(self, datums: Sequence[Any], use_base: bool = False) -> list[torch.Tensor]:
        """Forward the batch and gather per-token logprobs of each datum's target_tokens."""
        model = self._require_model()
        token_lists = [d.model_input.to_ints() for d in datums]
        max_len = max(len(t) for t in token_lists)
        input_ids = torch.zeros((len(datums), max_len), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(input_ids)
        for i, toks in enumerate(token_lists):
            input_ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
            attention_mask[i, : len(toks)] = 1
        ctx = self._base_ctx() if use_base else _nullcontext()
        with ctx:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        out = []
        for i, d in enumerate(datums):
            targets = d.loss_fn_inputs["target_tokens"].to_torch().long().to(self.device)
            n = len(token_lists[i])
            out.append(logprobs[i, :n].gather(1, targets.unsqueeze(1)).squeeze(1))
        return out

    async def submit_forward_backward(self, datums: Sequence[Any], loss_fn: str) -> _ResolvedPending:
        model = self._require_model()
        model.train()
        target_logprobs = self._target_logprobs(datums)

        if loss_fn == "cross_entropy":
            weights = [d.loss_fn_inputs["weights"].to_torch().to(self.device) for d in datums]
            loss = losses.cross_entropy_loss(target_logprobs, weights)
        elif loss_fn in ("ppo", "importance_sampling"):
            sampled = [d.loss_fn_inputs["logprobs"].to_torch().to(self.device) for d in datums]
            advs = [d.loss_fn_inputs["advantages"].to_torch().to(self.device) for d in datums]
            masks = [d.loss_fn_inputs["mask"].to_torch().float().to(self.device) for d in datums]
            if loss_fn == "ppo":
                loss = losses.ppo_loss(target_logprobs, sampled, advs, masks, clip_epsilon=self.ppo_clip_epsilon)
            else:
                loss = losses.importance_sampling_loss(target_logprobs, sampled, advs, masks)
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        loss.backward()  # accumulate; optim_step applies + zeroes
        return _ResolvedPending(
            ForwardBackwardOutput(
                logprobs=[lp.detach().cpu() for lp in target_logprobs],
                metrics={"loss": float(loss.detach())},
            )
        )

    async def submit_optim_step(self, *, learning_rate: float, adam: AdamConfig) -> _ResolvedPending:
        model = self._require_model()
        params = [p for p in model.parameters() if p.requires_grad]
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                params,
                lr=learning_rate,
                betas=(adam.beta1, adam.beta2),
                eps=adam.eps,
                weight_decay=adam.weight_decay,
            )
            if self._pending_optimizer_state is not None:
                self._optimizer.load_state_dict(self._pending_optimizer_state)
                self._pending_optimizer_state = None
        for group in self._optimizer.param_groups:
            group["lr"] = learning_rate
        if adam.grad_clip_norm and adam.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, adam.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        return _ResolvedPending(None)

    async def incorporate_kl_penalty(
        self, datums: Sequence[Any], *, kl_coef: float, kl_discount_factor: float
    ) -> dict[str, float]:
        """Same math as tinker_cookbook.rl.metrics.incorporate_kl_penalty, with base
        logprobs from a local forward pass under the disabled adapter."""
        import tinker

        self._require_base()
        with torch.no_grad():
            base_logprobs = self._target_logprobs(datums, use_base=True)

        sampled = [d.loss_fn_inputs["logprobs"].to_torch() for d in datums]
        masks = [d.loss_fn_inputs["mask"].to_torch().float() for d in datums]
        diffs = [(s - b.cpu()) * m for s, b, m in zip(sampled, base_logprobs, masks)]
        total_mask = sum(m.sum() for m in masks)
        avg_diff = sum(d.sum() for d in diffs) / total_mask.clamp(min=1e-8)
        for i, datum in enumerate(datums):
            kl_advantages = kl_coef * masks[i] * (avg_diff - diffs[i])
            if kl_discount_factor > 0:
                kl_advantages = discounted_future_sum_vectorized(kl_advantages, kl_discount_factor)
            datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
                datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
            )
        return {"kl_policy_base": float(avg_diff)}

    # ── checkpoints ──────────────────────────────────────────────────────

    async def save_checkpoint(self, *, name: str, log_dir: str | Path, loop_state: dict, kind: str) -> dict:
        model = self._require_model()
        ckpt_dir = Path(log_dir) / "checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if self.use_lora and HAS_PEFT:
            model.save_pretrained(str(ckpt_dir))  # adapter weights only
        else:
            torch.save(model.state_dict(), ckpt_dir / "weights.pt")
        state_saved = False
        if kind in ("state", "both") and self._optimizer is not None:
            torch.save(self._optimizer.state_dict(), ckpt_dir / "optimizer.pt")
            state_saved = True
        (ckpt_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "backend": "local",
                    "model": self.model_name,
                    "lora": self.use_lora,
                    "kind": kind,
                    "loop_state": loop_state,
                },
                indent=1,
            )
        )
        uri = f"file://{ckpt_dir.resolve()}"
        return {"sampler_path": uri, "state_path": uri if state_saved else None}

    def _load_checkpoint(self, resume_from: str, with_optimizer: bool) -> None:
        ckpt_dir = _strip_file_scheme(resume_from)
        if self.use_lora and HAS_PEFT:
            state = peft.utils.load_peft_weights(str(ckpt_dir))
            peft.set_peft_model_state_dict(self.model, state)
        else:
            weights = ckpt_dir / "weights.pt"
            self._require_model().load_state_dict(torch.load(weights, map_location=self.device))
        opt_path = ckpt_dir / "optimizer.pt"
        if with_optimizer and opt_path.exists():
            # Optimizer is created lazily at the first optim_step; stage the state.
            self._pending_optimizer_state = torch.load(opt_path, map_location=self.device)
        print(f"LocalBackend: loaded checkpoint from {ckpt_dir} (optimizer: {with_optimizer and opt_path.exists()})")
