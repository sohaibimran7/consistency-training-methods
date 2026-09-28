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

import asyncio
import copy
import fnmatch
import hashlib
import json
import math
import random
import re
import warnings
from collections.abc import Sequence
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from torch.utils.checkpoint import checkpoint

from ctm.backends.base import ForwardBackwardOutput, SampledSequence
from ctm.backends.local import losses
from ctm.backends.local.mlp_hooks import MLPHookManager
from ctm.backends.local.qwen35_vllm_compat import (
    is_qwen35_model_name,
    materialize_qwen35_vllm_rollout_compat_adapter,
)
from ctm.core.config import AdamConfig, LoRAConfig
from ctm.training import consistency_losses

# Internal-consistency loss_fns (ACT / AttCT / MLPCT) — LocalBackend only: they
# need paired forward passes with attentions / hidden states / MLP hooks, which
# the Tinker service API doesn't expose.
CONSISTENCY_LOSS_CLASSES = consistency_losses.CONSISTENCY_LOSS_CLASSES

# Casting every token-by-vocabulary logit to float32 at once is prohibitively
# expensive for large vocabularies (Qwen3.5 has roughly 250k tokens).  Keep the
# temporary cross-entropy workspace bounded and recompute each chunk during
# backward instead of retaining all of the float32 softmax intermediates.
_TARGET_LOGPROB_CHUNK_SIZE = 32

# A single RMCT logical batch can contain one datum per consistency rollout
# (currently up to 128).  Materializing that as one padded
# [datum, sequence, vocabulary] tensor is unsafe for large-vocabulary models.
# These limits bound each internal model forward while leaving the public
# forward_backward/gradient-accumulation boundary unchanged.  The token limit
# is the padded input footprint (number of datums times longest sequence), not
# the sum of unpadded lengths.
_DEFAULT_FORWARD_MICROBATCH_MAX_DATUMS = 8
_DEFAULT_FORWARD_MICROBATCH_MAX_TOKENS = 2048

# These objectives are normalized by a sum over selected tokens (or SFT
# weights), rather than by the number of datums.  Replicated training ranks
# must therefore use the *same logical-batch denominator* before their local
# gradients are summed.  Keeping the definition here makes the shard boundary
# explicit and avoids accidentally turning the objective into a mean of
# per-rank means.
_GLOBAL_NORMALIZED_LOSS_FNS = frozenset({"cross_entropy", "ppo", "importance_sampling"})

# These text CausalLM implementations apply their output embedding directly to
# the decoder's final hidden state. Keep this deliberately conservative: an
# unknown architecture falls back to its public CausalLM forward rather than
# risking omission of architecture-specific logit processing.
_DIRECT_LM_HEAD_CLASSES = {
    "gemma": frozenset({"GemmaForCausalLM"}),
    "gpt2": frozenset({"GPT2LMHeadModel"}),
    "llama": frozenset({"LlamaForCausalLM"}),
    "mistral": frozenset({"MistralForCausalLM"}),
    "mixtral": frozenset({"MixtralForCausalLM"}),
    "qwen2": frozenset({"Qwen2ForCausalLM"}),
    "qwen2_moe": frozenset({"Qwen2MoeForCausalLM"}),
    "qwen3": frozenset({"Qwen3ForCausalLM"}),
    "qwen3_5": frozenset({"Qwen3_5ForCausalLM"}),
    "qwen3_5_text": frozenset({"Qwen3_5ForCausalLM"}),
    "qwen3_5_moe": frozenset({"Qwen3_5MoeForCausalLM"}),
    "qwen3_moe": frozenset({"Qwen3MoeForCausalLM"}),
}
_SOFTCAPPED_LM_HEAD_CLASSES = {"gemma2": frozenset({"Gemma2ForCausalLM"})}

# Gemma 4 12B is a unified conditional-generation model, not a text-only
# CausalLM.  Its name is intentionally narrow here: LocalBackend supports the
# exact candidate under evaluation while direct benchmark evaluation uses the
# stronger config-based check in ``ctm.evals.local_model``.
_GEMMA4_UNIFIED_MODEL_TYPES = frozenset({"gemma4_unified", "gemma4"})


def _is_gemma4_unified_model_reference(model: str) -> bool:
    normalized = model.lower().replace("_", "-")
    return "gemma-4-12b-it" in normalized or "gemma4-12b-it" in normalized

# The dense Qwen3.5-9B hybrid backbone has 32 physical decoder layers.  Only
# every fourth layer is conventional softmax attention; the other 24 are
# DeltaNet layers and do not return the probability matrices AttCT optimizes.
# Keep the preflight deliberately specific to this topology rather than
# silently treating a superficially similar hybrid model as equivalent.
_QWEN35_CONSISTENCY_DECODER_LAYERS = 32
_QWEN35_CONSISTENCY_FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
_QWEN35_CONSISTENCY_TARGET_PROJECTIONS = ("q_proj", "v_proj")
_QWEN35_ACT_FULL_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_QWEN35_ACT_LINEAR_ATTENTION_PROJECTIONS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)
_QWEN35_ACT_TARGET_PROJECTIONS = (
    *_QWEN35_ACT_FULL_ATTENTION_PROJECTIONS,
    *_QWEN35_ACT_LINEAR_ATTENTION_PROJECTIONS,
)
_QWEN35_LORA_PARAMETER_RE = re.compile(
    r"(?:^|.*\.)model\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<projection>q_proj|v_proj)\.lora_(?P<kind>A|B)\."
    r"(?P<adapter>[^.]+)\.weight$"
)
_QWEN35_ACT_LORA_PARAMETER_RE = re.compile(
    r"(?:^|.*\.)model\.layers\.(?P<layer>\d+)\."
    r"(?P<family>self_attn|linear_attn)\."
    r"(?P<projection>[a-z_]+)\.lora_(?P<kind>A|B)\."
    r"(?P<adapter>[^.]+)\.weight$"
)

try:  # optional: LoRA support
    import peft
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class FusedExpertLoRAWarning(UserWarning):
    """PEFT is targeting fused 3-D expert tensors instead of Linear modules."""


@dataclass(frozen=True)
class _SelectedTokenComponents:
    """Verified components needed to bypass full sequence-vocabulary logits."""

    backbone: torch.nn.Module
    lm_head: torch.nn.Module
    output_multiplier: float = 1.0
    final_logit_softcapping: float | None = None


def _strip_file_scheme(path: str) -> Path:
    return Path(path.removeprefix("file://"))


def logical_loss_denominator(
    datums: Sequence[Any],
    loss_fn: str,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return one shard's contribution to a globally normalized objective.

    For ``cross_entropy`` this is the sum of datum ``weights``; for PPO and
    importance sampling it is the sum of their token ``mask`` values.  A
    phase-shared replicated trainer can either call this on the unsharded
    logical batch, or SUM-reduce the returned scalar across ranks, then pass
    that result to :meth:`LocalBackend.submit_forward_backward` via
    ``global_loss_denominator``.

    The result is deliberately *not* clamped here.  The consuming loss clamps
    exactly as the historical single-rank path did, after the global value is
    known.  This preserves the empty/zero-weight behavior while making an
    accidental per-rank normalization observable.
    """

    if loss_fn not in _GLOBAL_NORMALIZED_LOSS_FNS:
        raise ValueError(
            "logical_loss_denominator supports only cross_entropy, ppo, or "
            f"importance_sampling; got {loss_fn!r}"
        )
    field = "weights" if loss_fn == "cross_entropy" else "mask"
    terms: list[torch.Tensor] = []
    for datum in datums:
        value = datum.loss_fn_inputs[field].to_torch()
        if device is not None:
            value = value.to(device)
        # This mirrors the existing PPO/IS path, whose masks are explicitly
        # float-valued before summing.  Cross-entropy retains its supplied
        # weight dtype for backward compatibility.
        if loss_fn != "cross_entropy":
            value = value.float()
        terms.append(value.sum())
    if not terms:
        return torch.zeros((), device=device)
    return sum(terms)


def sum_trainable_gradients_torch_distributed(
    parameters: Sequence[torch.nn.Parameter],
    *,
    process_group: Any = None,
) -> None:
    """SUM-reduce trainable gradients once at an optimizer boundary.

    This helper intentionally does not divide by world size: each rank has
    already backpropagated its local numerator divided by the common global
    denominator, so summing yields the original logical-batch gradient.  It
    is called after all public gradient accumulations and before clipping.

    Gradients are flattened by dtype/device to avoid one collective per LoRA
    tensor.  A tiny activity all-reduce lets an empty shard contribute zeros
    for parameters used on another rank without changing the result.  Every
    rank in the process group must expose the same trainable parameter order,
    as is standard for replicated-data-parallel training.
    """

    try:
        import torch.distributed as dist
    except ImportError as exc:  # pragma: no cover - torch builds normally expose this module
        raise RuntimeError("torch.distributed is unavailable for gradient reduction") from exc
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "gradient_reducer='torch.distributed' requires an initialized torch.distributed process group"
        )
    if dist.get_world_size(group=process_group) <= 1:
        return

    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        return
    devices = {parameter.device for parameter in trainable}
    if len(devices) != 1:
        raise ValueError(
            "torch.distributed gradient reduction requires one local device per LocalBackend; "
            f"got trainable parameters on {sorted(map(str, devices))}"
        )

    # A rank with no sequences in this logical shard has no autograd graph and
    # therefore no .grad tensors.  Determine the global active set first, then
    # materialize only the required local zeros so the flattened collective is
    # well-defined and still exactly represents a sum of local gradients.
    device = trainable[0].device
    active = torch.tensor(
        [parameter.grad is not None for parameter in trainable],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(active, op=dist.ReduceOp.SUM, group=process_group)
    if not bool((active > 0).any()):
        return

    grouped: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    with torch.no_grad():
        for parameter, is_active in zip(trainable, active.tolist()):
            if not is_active:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            gradient = parameter.grad
            assert gradient is not None
            grouped.setdefault((gradient.device, gradient.dtype), []).append(gradient)

        for gradients in grouped.values():
            flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=process_group)
            offset = 0
            for gradient in gradients:
                size = gradient.numel()
                gradient.copy_(flat[offset : offset + size].reshape_as(gradient))
                offset += size


def _selected_target_logprobs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = _TARGET_LOGPROB_CHUNK_SIZE,
) -> torch.Tensor:
    """Return selected-token logprobs with bounded float32 workspace.

    ``cross_entropy(..., reduction="none")`` is exactly the negative selected
    log-softmax.  Checkpointing prevents autograd from retaining one full
    float32 softmax buffer per chunk, which would otherwise recover the same
    peak memory as an unchunked ``log_softmax`` by the end of the forward pass.
    """

    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [tokens, vocabulary], got {tuple(logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(f"targets must have shape [tokens] matching logits; got logits={tuple(logits.shape)}, targets={tuple(targets.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if logits.shape[0] == 0:
        return logits.new_empty((0,), dtype=torch.float32)

    def selected(chunk_logits: torch.Tensor, chunk_targets: torch.Tensor) -> torch.Tensor:
        return -F.cross_entropy(chunk_logits.float(), chunk_targets, reduction="none")

    pieces = []
    for start in range(0, logits.shape[0], chunk_size):
        end = min(start + chunk_size, logits.shape[0])
        chunk_logits = logits[start:end]
        chunk_targets = targets[start:end]
        if torch.is_grad_enabled() and chunk_logits.requires_grad:
            values = checkpoint(selected, chunk_logits, chunk_targets, use_reentrant=False)
        else:
            values = selected(chunk_logits, chunk_targets)
        pieces.append(values)
    return torch.cat(pieces)


def _selected_hidden_logprobs(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    chunk_size: int = _TARGET_LOGPROB_CHUNK_SIZE,
    output_multiplier: float = 1.0,
    final_logit_softcapping: float | None = None,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """Apply ``lm_head`` only to selected hidden states in bounded chunks.

    The head and float32 cross-entropy live inside one checkpointed function,
    so neither a full ``[batch, sequence, vocabulary]`` tensor nor one retained
    ``[all selected tokens, vocabulary]`` tensor is created. PEFT/LoRA modules
    remain in the ordinary module call graph, including an adapted LM head.
    """

    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must have shape [tokens, hidden], got {tuple(hidden_states.shape)}")
    if targets.ndim != 1 or targets.shape[0] != hidden_states.shape[0]:
        raise ValueError(f"targets must have shape [tokens] matching hidden_states; got hidden_states={tuple(hidden_states.shape)}, targets={tuple(targets.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty((0,), dtype=torch.float32)

    def selected(chunk_hidden: torch.Tensor, chunk_targets: torch.Tensor) -> torch.Tensor:
        chunk_logits = lm_head(chunk_hidden)
        if chunk_logits.ndim != 2 or chunk_logits.shape[0] != chunk_hidden.shape[0]:
            raise RuntimeError(f"selected-token LM head must return [tokens, vocabulary], got {tuple(chunk_logits.shape)} from {type(lm_head).__name__}")
        if output_multiplier != 1.0:
            chunk_logits = chunk_logits * output_multiplier
        if final_logit_softcapping is not None:
            chunk_logits = torch.tanh(chunk_logits / final_logit_softcapping) * final_logit_softcapping
        return -F.cross_entropy(
            chunk_logits.float(),
            chunk_targets.to(chunk_logits.device),
            reduction="none",
        )

    pieces = []
    for start in range(0, hidden_states.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_states.shape[0])
        chunk_hidden = hidden_states[start:end]
        chunk_targets = targets[start:end]
        if checkpoint_chunks and torch.is_grad_enabled() and (chunk_hidden.requires_grad or any(parameter.requires_grad for parameter in lm_head.parameters())):
            values = checkpoint(selected, chunk_hidden, chunk_targets, use_reentrant=False)
        else:
            values = selected(chunk_hidden, chunk_targets)
        pieces.append(values)
    return torch.cat(pieces)


def _selected_hidden_raw_and_temperature_logprobs(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    temperature: float,
    chunk_size: int = _TARGET_LOGPROB_CHUNK_SIZE,
    output_multiplier: float = 1.0,
    final_logit_softcapping: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw-policy and temperature-adjusted selected-token scores.

    OPCT samples from ``softmax(logits / temperature)`` but optimizes the raw
    policy. Computing both scores from one selected-token head call avoids a
    second backbone pass while keeping the importance-sampling denominator
    distinct from the raw-policy score used by the reverse-KL signal.
    """

    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must have shape [tokens, hidden], got {tuple(hidden_states.shape)}")
    if targets.ndim != 1 or targets.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            "targets must have shape [tokens] matching hidden_states; "
            f"got hidden_states={tuple(hidden_states.shape)}, targets={tuple(targets.shape)}"
        )
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if hidden_states.shape[0] == 0:
        empty = hidden_states.new_empty((0,), dtype=torch.float32)
        return empty, empty.clone()

    def selected(
        chunk_hidden: torch.Tensor,
        chunk_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_logits = lm_head(chunk_hidden)
        if chunk_logits.ndim != 2 or chunk_logits.shape[0] != chunk_hidden.shape[0]:
            raise RuntimeError(
                "selected-token LM head must return [tokens, vocabulary], "
                f"got {tuple(chunk_logits.shape)} from {type(lm_head).__name__}"
            )
        if output_multiplier != 1.0:
            chunk_logits = chunk_logits * output_multiplier
        if final_logit_softcapping is not None:
            chunk_logits = torch.tanh(chunk_logits / final_logit_softcapping) * final_logit_softcapping
        float_logits = chunk_logits.float()
        chunk_targets = chunk_targets.to(chunk_logits.device)
        raw = -F.cross_entropy(float_logits, chunk_targets, reduction="none")
        behavior = -F.cross_entropy(float_logits / temperature, chunk_targets, reduction="none")
        return raw, behavior

    raw_pieces = []
    behavior_pieces = []
    for start in range(0, hidden_states.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_states.shape[0])
        chunk_hidden = hidden_states[start:end]
        chunk_targets = targets[start:end]
        if torch.is_grad_enabled() and (
            chunk_hidden.requires_grad or any(parameter.requires_grad for parameter in lm_head.parameters())
        ):
            raw, behavior = checkpoint(selected, chunk_hidden, chunk_targets, use_reentrant=False)
        else:
            raw, behavior = selected(chunk_hidden, chunk_targets)
        raw_pieces.append(raw)
        behavior_pieces.append(behavior)
    return torch.cat(raw_pieces), torch.cat(behavior_pieces)


def _selected_token_components(model: torch.nn.Module) -> _SelectedTokenComponents | None:
    """Resolve an exact backbone/head path, or ``None`` for safe dense fallback.

    PEFT's public ``get_base_model`` retains all injected LoRA modules while
    exposing the underlying Hugging Face CausalLM shape. Architecture support
    is allowlisted because some CausalLMs transform logits after their LM head;
    bypassing an unknown forward would silently change the training objective.
    """

    causal_lm = model
    get_base_model = getattr(model, "get_base_model", None)
    if HAS_PEFT and isinstance(model, peft.PeftModel) and callable(get_base_model):
        causal_lm = get_base_model()

    config = getattr(causal_lm, "config", None)
    model_type = getattr(config, "model_type", None)
    if (
        model_type == "muse_glimmer"
        and type(causal_lm).__name__ == "MuseGlimmerForConditionalGeneration"
        and type(causal_lm).__module__.startswith("transformers.models.")
    ):
        multimodal_backbone = getattr(causal_lm, "model", None)
        backbone = getattr(multimodal_backbone, "language_model", None)
        get_output_embeddings = getattr(causal_lm, "get_output_embeddings", None)
        lm_head = get_output_embeddings() if callable(get_output_embeddings) else None
        text_config = getattr(config, "text_config", None)
        multiplier = getattr(text_config, "output_multiplier", None)
        softcap = getattr(text_config, "final_logit_softcapping", None)
        if (
            not isinstance(backbone, torch.nn.Module)
            or not isinstance(lm_head, torch.nn.Module)
            or isinstance(multiplier, bool)
            or not isinstance(multiplier, (int, float))
            or not math.isfinite(float(multiplier))
            or float(multiplier) <= 0
            or isinstance(softcap, bool)
            or not isinstance(softcap, (int, float))
            or not math.isfinite(float(softcap))
            or float(softcap) <= 0
        ):
            return None
        return _SelectedTokenComponents(
            backbone=backbone,
            lm_head=lm_head,
            output_multiplier=float(multiplier),
            final_logit_softcapping=float(softcap),
        )
    supported_classes = _DIRECT_LM_HEAD_CLASSES.get(model_type) or _SOFTCAPPED_LM_HEAD_CLASSES.get(model_type)
    causal_lm_type = type(causal_lm)
    if supported_classes is None or causal_lm_type.__name__ not in supported_classes or not causal_lm_type.__module__.startswith("transformers.models."):
        return None

    prefix = getattr(causal_lm, "base_model_prefix", None)
    backbone = getattr(causal_lm, prefix, None) if isinstance(prefix, str) and prefix else None
    get_output_embeddings = getattr(causal_lm, "get_output_embeddings", None)
    lm_head = get_output_embeddings() if callable(get_output_embeddings) else None
    if not isinstance(backbone, torch.nn.Module) or backbone is causal_lm or not isinstance(lm_head, torch.nn.Module):
        return None

    softcap = None
    if model_type in _SOFTCAPPED_LM_HEAD_CLASSES:
        softcap = getattr(config, "final_logit_softcapping", None)
        if softcap is not None and (not isinstance(softcap, (int, float)) or isinstance(softcap, bool) or softcap <= 0):
            return None
    elif getattr(config, "final_logit_softcapping", None) is not None:
        return None
    return _SelectedTokenComponents(
        backbone=backbone,
        lm_head=lm_head,
        final_logit_softcapping=float(softcap) if softcap is not None else None,
    )


def _gradient_checkpointing_backbone_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Resolve the decoder layer list used for selective checkpointing.

    Qwen3.5 exposes its decoder as ``causal_lm.<base_model_prefix>.layers``.
    Keep this structural check strict: silently mutating an unrelated module
    list would make the requested memory/performance policy inaccurate.
    """

    causal_lm = model
    get_base_model = getattr(model, "get_base_model", None)
    if HAS_PEFT and isinstance(model, peft.PeftModel) and callable(get_base_model):
        causal_lm = get_base_model()

    prefix = getattr(causal_lm, "base_model_prefix", None)
    backbone = getattr(causal_lm, prefix, None) if isinstance(prefix, str) and prefix else None
    if getattr(getattr(causal_lm, "config", None), "model_type", None) == "muse_glimmer":
        backbone = getattr(backbone, "language_model", None)
    layers = (
        getattr(backbone, "layers", None)
        if isinstance(backbone, torch.nn.Module) and backbone is not causal_lm
        else None
    )
    if not isinstance(layers, torch.nn.ModuleList) or not layers:
        raise RuntimeError(
            "selective gradient checkpointing could not identify "
            f"{type(causal_lm).__name__} backbone layers at <base_model_prefix>.layers"
        )
    resolved = list(layers)
    if any(not isinstance(layer, torch.nn.Module) for layer in resolved):
        raise RuntimeError("selective gradient checkpointing resolved a non-module backbone layer")
    return resolved


def _underlying_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap PEFT once for architecture-level consistency checks."""

    get_base_model = getattr(model, "get_base_model", None)
    if HAS_PEFT and isinstance(model, peft.PeftModel) and callable(get_base_model):
        candidate = get_base_model()
        if isinstance(candidate, torch.nn.Module):
            return candidate
    return model


def _inspect_qwen35_consistency_topology(model: torch.nn.Module) -> tuple[dict[str, Any], list[str]]:
    """Validate the exact Qwen3.5-9B hybrid layout used by AttCT/MLPCT.

    AttCT's attention tuple is compact: it contains only the eight physical
    ``full_attention`` layers.  A loose "Qwen-like" check would make a future
    architecture change look like an ordinary experimental result, so this
    reports and rejects any deviation before a paid training trajectory starts.
    """

    causal_lm = _underlying_causal_lm(model)
    config = getattr(causal_lm, "config", None)
    model_type = getattr(config, "model_type", None)
    prefix = getattr(causal_lm, "base_model_prefix", None)
    backbone = getattr(causal_lm, prefix, None) if isinstance(prefix, str) and prefix else None
    layers = getattr(backbone, "layers", None) if isinstance(backbone, torch.nn.Module) else None
    layer_types = getattr(config, "layer_types", None)
    observed_layer_types = list(layer_types) if isinstance(layer_types, (list, tuple)) else None
    observed_full_attention_layers = (
        [index for index, layer_type in enumerate(observed_layer_types) if layer_type == "full_attention"]
        if observed_layer_types is not None
        else []
    )
    report: dict[str, Any] = {
        "causal_lm_class": type(causal_lm).__name__,
        "model_type": model_type,
        "base_model_prefix": prefix,
        "decoder_layer_count": len(layers) if isinstance(layers, torch.nn.ModuleList) else None,
        "layer_types": observed_layer_types,
        "full_attention_layers": observed_full_attention_layers,
        "expected_decoder_layer_count": _QWEN35_CONSISTENCY_DECODER_LAYERS,
        "expected_full_attention_layers": list(_QWEN35_CONSISTENCY_FULL_ATTENTION_LAYERS),
    }
    errors: list[str] = []
    if model_type not in {"qwen3_5", "qwen3_5_text"}:
        errors.append(
            "Qwen3.5 consistency preflight requires config.model_type to be "
            f"'qwen3_5' or 'qwen3_5_text', got {model_type!r}"
        )
        return report, errors
    if not isinstance(layers, torch.nn.ModuleList):
        errors.append(
            "Qwen3.5 consistency preflight could not resolve decoder layers at "
            "<base_model_prefix>.layers"
        )
        return report, errors
    if len(layers) != _QWEN35_CONSISTENCY_DECODER_LAYERS:
        errors.append(
            "Qwen3.5 consistency preflight requires exactly "
            f"{_QWEN35_CONSISTENCY_DECODER_LAYERS} decoder layers, got {len(layers)}"
        )
    expected_layer_types = [
        "full_attention" if index in _QWEN35_CONSISTENCY_FULL_ATTENTION_LAYERS else "linear_attention"
        for index in range(_QWEN35_CONSISTENCY_DECODER_LAYERS)
    ]
    if observed_layer_types != expected_layer_types:
        errors.append(
            "Qwen3.5 consistency preflight requires the canonical 24 DeltaNet / 8 full-attention "
            f"layer pattern; observed full-attention layers {observed_full_attention_layers}"
        )
    for layer_index in _QWEN35_CONSISTENCY_FULL_ATTENTION_LAYERS:
        if layer_index >= len(layers):
            continue
        layer = layers[layer_index]
        if getattr(layer, "layer_type", None) != "full_attention":
            errors.append(f"decoder layer {layer_index} is not marked full_attention")
            continue
        attention = getattr(layer, "self_attn", None)
        if not isinstance(attention, torch.nn.Module):
            errors.append(f"full-attention decoder layer {layer_index} has no self_attn module")
            continue
        for projection in _QWEN35_CONSISTENCY_TARGET_PROJECTIONS:
            if not isinstance(getattr(attention, projection, None), torch.nn.Module):
                errors.append(f"full-attention decoder layer {layer_index} has no {projection} module")
    return report, errors


def _inspect_qwen35_strict_qv_lora(
    model: torch.nn.Module,
    configured_lora: LoRAConfig | None,
) -> tuple[dict[str, Any], list[tuple[int, str, str, str, str, torch.nn.Parameter]], list[str]]:
    """Require the paper-style Q/V adapter set and return its LoRA-B tensors."""

    errors: list[str] = []
    configured_targets = None if configured_lora is None else configured_lora.target_modules
    report: dict[str, Any] = {
        "configured_target_modules": configured_targets,
        "configured_train_mlp": None if configured_lora is None else configured_lora.train_mlp,
        "configured_train_attn": None if configured_lora is None else configured_lora.train_attn,
        "configured_train_unembed": None if configured_lora is None else configured_lora.train_unembed,
        "expected_target_modules": list(_QWEN35_CONSISTENCY_TARGET_PROJECTIONS),
        "expected_adapter_name": "default",
    }
    if configured_lora is None:
        errors.append("Qwen3.5 consistency preflight has no recorded LoRA configuration")
    else:
        targets = configured_lora.target_modules
        if not isinstance(targets, list) or len(targets) != 2 or set(targets) != set(_QWEN35_CONSISTENCY_TARGET_PROJECTIONS):
            errors.append(
                "Qwen3.5 consistency preflight requires exact target_modules=['q_proj', 'v_proj']"
            )
        if configured_lora.train_mlp or configured_lora.train_attn or configured_lora.train_unembed:
            errors.append(
                "Qwen3.5 consistency preflight requires all portable LoRA component flags "
                "(train_mlp/train_attn/train_unembed) to be false when explicit Q/V targets are used"
            )
    if not HAS_PEFT or not isinstance(model, peft.PeftModel):
        errors.append("Qwen3.5 consistency preflight requires a PEFT LoRA model")

    trainable: list[tuple[str, torch.nn.Parameter]] = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    parsed: dict[tuple[int, str, str, str], tuple[str, torch.nn.Parameter]] = {}
    unexpected_trainable: list[str] = []
    for name, parameter in trainable:
        match = _QWEN35_LORA_PARAMETER_RE.fullmatch(name)
        if match is None:
            unexpected_trainable.append(name)
            continue
        key = (
            int(match.group("layer")),
            match.group("projection"),
            match.group("kind"),
            match.group("adapter"),
        )
        if key in parsed:
            errors.append(f"duplicate Qwen3.5 LoRA parameter for {key}")
        parsed[key] = (name, parameter)

    expected_keys = {
        (layer, projection, kind, "default")
        for layer in _QWEN35_CONSISTENCY_FULL_ATTENTION_LAYERS
        for projection in _QWEN35_CONSISTENCY_TARGET_PROJECTIONS
        for kind in ("A", "B")
    }
    observed_keys = set(parsed)
    missing = sorted(expected_keys - observed_keys)
    unexpected_keys = sorted(observed_keys - expected_keys)
    if unexpected_trainable:
        errors.append(
            "Qwen3.5 consistency preflight found trainable parameters outside the expected Q/V LoRA set: "
            + ", ".join(unexpected_trainable[:4])
            + (" ..." if len(unexpected_trainable) > 4 else "")
        )
    if missing:
        errors.append(f"Qwen3.5 consistency preflight is missing expected Q/V LoRA tensors: {missing}")
    if unexpected_keys:
        errors.append(f"Qwen3.5 consistency preflight found unexpected Q/V LoRA tensors: {unexpected_keys}")

    report.update(
        {
            "trainable_parameter_count": len(trainable),
            "expected_trainable_parameter_count": len(expected_keys),
            "unexpected_trainable_parameters": unexpected_trainable,
            "missing_expected_parameters": [list(key) for key in missing],
            "unexpected_lora_parameters": [list(key) for key in unexpected_keys],
        }
    )
    b_parameters = [
        (layer, "self_attn", projection, adapter, name, parameter)
        for (layer, projection, kind, adapter), (name, parameter) in parsed.items()
        if kind == "B"
    ]
    b_parameters.sort(key=lambda item: (item[0], _QWEN35_CONSISTENCY_TARGET_PROJECTIONS.index(item[2]), item[3]))
    report["lora_b_parameter_count"] = len(b_parameters)
    return report, b_parameters, errors


def _inspect_qwen35_act_attention_lora(
    model: torch.nn.Module,
    configured_lora: LoRAConfig | None,
    *,
    qv_fused_qkv: bool = False,
) -> tuple[dict[str, Any], list[tuple[int, str, str, str, str, torch.nn.Parameter]], list[str]]:
    """Require ACT's explicit hybrid-attention LoRA set and return LoRA-B tensors.

    ACT is intentionally broader than the Q/V-only AttCT/MLPCT protocol: it
    compares residual states across every Qwen3.5 layer, so the repaired ACT
    configuration adapts both conventional self-attention projections and the
    DeltaNet linear-attention projections.  This validator proves that exact
    contract before the one-pair HF/PEFT backward probe runs.
    """

    # Explicit opt-in for the expanded-pool ACT comparison. Qwen3.5 exposes
    # DeltaNet Q/K/V as one projection, so this is an attested fused-QKV
    # approximation to paper Q/V LoRA, not a claim of slice-exact Q/V tuning.
    full_projections = ("q_proj", "v_proj") if qv_fused_qkv else _QWEN35_ACT_FULL_ATTENTION_PROJECTIONS
    linear_projections = ("in_proj_qkv",) if qv_fused_qkv else _QWEN35_ACT_LINEAR_ATTENTION_PROJECTIONS
    target_projections = (*full_projections, *linear_projections)
    errors: list[str] = []
    configured_targets = None if configured_lora is None else configured_lora.target_modules
    report: dict[str, Any] = {
        "configured_target_modules": configured_targets,
        "configured_train_mlp": None if configured_lora is None else configured_lora.train_mlp,
        "configured_train_attn": None if configured_lora is None else configured_lora.train_attn,
        "configured_train_unembed": None if configured_lora is None else configured_lora.train_unembed,
        "expected_target_modules": list(target_projections),
        "expected_adapter_name": "default",
    }
    if configured_lora is None:
        errors.append("Qwen3.5 ACT preflight has no recorded LoRA configuration")
    else:
        targets = configured_lora.target_modules
        if (
            not isinstance(targets, list)
            or len(targets) != len(target_projections)
            or set(targets) != set(target_projections)
        ):
            errors.append(
                "Qwen3.5 ACT preflight requires exact target_modules="
                f"{list(target_projections)!r}"
            )
        if configured_lora.train_mlp or configured_lora.train_attn or configured_lora.train_unembed:
            errors.append(
                "Qwen3.5 ACT preflight requires all portable LoRA component flags "
                "(train_mlp/train_attn/train_unembed) to be false when explicit attention targets are used"
            )
    if not HAS_PEFT or not isinstance(model, peft.PeftModel):
        errors.append("Qwen3.5 ACT preflight requires a PEFT LoRA model")

    trainable: list[tuple[str, torch.nn.Parameter]] = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    parsed: dict[tuple[int, str, str, str, str], tuple[str, torch.nn.Parameter]] = {}
    unexpected_trainable: list[str] = []
    for name, parameter in trainable:
        match = _QWEN35_ACT_LORA_PARAMETER_RE.fullmatch(name)
        if match is None:
            unexpected_trainable.append(name)
            continue
        layer = int(match.group("layer"))
        family = match.group("family")
        projection = match.group("projection")
        expected_projections = (
            full_projections
            if family == "self_attn" and layer in _QWEN35_CONSISTENCY_FULL_ATTENTION_LAYERS
            else linear_projections
            if family == "linear_attn" and layer not in _QWEN35_CONSISTENCY_FULL_ATTENTION_LAYERS
            else ()
        )
        if projection not in expected_projections:
            unexpected_trainable.append(name)
            continue
        key = (layer, family, projection, match.group("kind"), match.group("adapter"))
        if key in parsed:
            errors.append(f"duplicate Qwen3.5 ACT LoRA parameter for {key}")
        parsed[key] = (name, parameter)

    expected_keys = {
        (layer, "self_attn", projection, kind, "default")
        for layer in _QWEN35_CONSISTENCY_FULL_ATTENTION_LAYERS
        for projection in full_projections
        for kind in ("A", "B")
    }
    expected_keys.update(
        (layer, "linear_attn", projection, kind, "default")
        for layer in range(_QWEN35_CONSISTENCY_DECODER_LAYERS)
        if layer not in _QWEN35_CONSISTENCY_FULL_ATTENTION_LAYERS
        for projection in linear_projections
        for kind in ("A", "B")
    )
    observed_keys = set(parsed)
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    if unexpected_trainable:
        errors.append(
            "Qwen3.5 ACT preflight found trainable parameters outside the expected hybrid-attention LoRA set: "
            + ", ".join(unexpected_trainable[:4])
            + (" ..." if len(unexpected_trainable) > 4 else "")
        )
    if missing:
        errors.append(f"Qwen3.5 ACT preflight is missing expected attention LoRA tensors: {missing}")
    if unexpected:
        errors.append(f"Qwen3.5 ACT preflight found unexpected attention LoRA tensors: {unexpected}")

    report.update(
        {
            "trainable_parameter_count": len(trainable),
            "expected_trainable_parameter_count": len(expected_keys),
            "unexpected_trainable_parameters": unexpected_trainable,
            "missing_expected_parameters": [list(key) for key in missing],
            "unexpected_lora_parameters": [list(key) for key in unexpected],
        }
    )
    projection_order = {
        "self_attn": {name: index for index, name in enumerate(full_projections)},
        "linear_attn": {name: index for index, name in enumerate(linear_projections)},
    }
    b_parameters = [
        (layer, family, projection, adapter, name, parameter)
        for (layer, family, projection, kind, adapter), (name, parameter) in parsed.items()
        if kind == "B"
    ]
    b_parameters.sort(key=lambda item: (item[0], item[1], projection_order[item[1]][item[2]], item[3]))
    report["lora_b_parameter_count"] = len(b_parameters)
    return report, b_parameters, errors


def _adapter_parameter_digest(model: torch.nn.Module) -> tuple[str, int]:
    """Hash all LoRA tensors without changing adapter state or dtype."""

    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if ".lora_" not in name:
            continue
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(raw)
        count += 1
    return digest.hexdigest(), count


class _ResolvedPending:
    """Already-computed result behind the two-phase pending interface."""

    def __init__(self, value):
        self._value = value

    async def result(self):
        return self._value


def _name_matches(name: str, selector: str) -> bool:
    """Match a full name by glob, or a plain selector by dotted component."""

    if any(character in selector for character in "*?["):
        return fnmatch.fnmatchcase(name, selector)
    return name == selector or name.endswith(f".{selector}") or selector in name.split(".")


def _lora_target_module_names(model: torch.nn.Module, config: LoRAConfig) -> list[str]:
    """Resolve exact targets or portable component flags to PEFT module names."""

    output = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    components: dict[str, list[str]] = {"mlp": [], "attn": [], "unembed": []}
    for name, module in model.named_modules():
        if not name or not (isinstance(module, torch.nn.Linear) or type(module).__name__ == "Conv1D"):
            continue
        if module is output:
            component = "unembed"
        elif "attn" in name.lower() or "attention" in name.lower():
            component = "attn"
        else:
            component = "mlp"
        components[component].append(name)

    if config.target_modules is not None:
        names = [name for component_names in components.values() for name in component_names]
        selected = [name for name in names if any(_name_matches(name, target) for target in config.target_modules)]
        unmatched = [
            target for target in config.target_modules if not any(_name_matches(name, target) for name in names)
        ]
        if unmatched:
            raise ValueError(f"model {type(model).__name__} has no linear modules matching target_modules={unmatched}")
        return selected

    enabled = {
        "mlp": config.train_mlp,
        "attn": config.train_attn,
        "unembed": config.train_unembed,
    }
    raw_mlp_parameters = _lora_target_parameter_names(model, config)
    missing = [component for component, selected in enabled.items() if selected and not components[component] and not (component == "mlp" and raw_mlp_parameters)]
    if missing:
        raise NotImplementedError(f"model {type(model).__name__} exposes no local LoRA modules for selected component(s): {missing}")
    return [name for component, names in components.items() if enabled[component] for name in names]


def _lora_target_parameter_names(model: torch.nn.Module, config: LoRAConfig) -> list[str]:
    """Return fused MoE expert matrices that cannot be targeted as modules.

    GPT-OSS represents its expert projections as three-dimensional Parameters
    rather than ``nn.Linear`` modules. Recent PEFT versions support these
    through ``target_parameters``.
    """

    if config.target_modules is not None or not config.train_mlp:
        return []
    suffixes = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
    return [name for name, _ in model.named_parameters() if name.endswith(suffixes)]


def _configure_full_finetune_parameters(model: torch.nn.Module, selectors: Sequence[str] | None) -> list[str]:
    """Enable either every parameter or the explicitly selected parameter groups."""

    if selectors is None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return [name for name, _ in model.named_parameters()]
    if not selectors or any(not isinstance(selector, str) or not selector.strip() for selector in selectors):
        raise ValueError("full_finetune_modules must contain non-empty module selectors")

    selected = []
    for name, parameter in model.named_parameters():
        train = any(_name_matches(name, selector) for selector in selectors)
        parameter.requires_grad_(train)
        if train:
            selected.append(name)
    unmatched = [selector for selector in selectors if not any(_name_matches(name, selector) for name in selected)]
    if unmatched:
        raise ValueError(f"model {type(model).__name__} has no parameters matching full_finetune_modules={unmatched}")
    if not selected:
        raise ValueError("full_finetune_modules selected no parameters")
    return selected


def local_hf_eos_token_ids(model: Any, stop: Any) -> list[int]:
    """Resolve the existing HF sampler's model and renderer terminators.

    A chat tokenizer can use an end-of-turn token distinct from the model's
    generation-config EOS. Consumers validating sampled tokens must use this
    same set, not just generation_config.eos_token_id.
    """
    eos_ids = [t for t in (stop or []) if isinstance(t, int)]
    configured_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(configured_eos, int):
        configured_eos = [configured_eos]
    if isinstance(configured_eos, (list, tuple)):
        eos_ids.extend(t for t in configured_eos if isinstance(t, int))
    return list(dict.fromkeys(eos_ids))


class LocalSamplerHandle:
    """Samples from the backend's live model (policy) or its frozen base.

    Concurrent coroutine calls are coalesced into one backend batch. This keeps
    synchronous HF/vLLM generation off the event loop and lets vLLM schedule all
    prompts together without making unsafe concurrent ``LLM.generate`` calls.
    """

    def __init__(self, backend: "LocalBackend", use_base: bool):
        self._backend = backend
        self._use_base = use_base
        self._pending: list[tuple[dict[str, Any], asyncio.Future]] = []
        self._flush_task: asyncio.Task | None = None

    async def sample(self, prompt: Any, *, max_tokens: int | None, temperature: float, stop: Any, num_samples: int) -> list[SampledSequence]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending.append(
            (
                {
                    "prompt_tokens": list(prompt.to_ints()),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                    "num_samples": num_samples,
                },
                future,
            )
        )
        if self._flush_task is None:
            self._flush_task = loop.create_task(self._flush_pending())
        return await future

    async def score_completions(
        self,
        prompts: Sequence[Any],
        completion_tokens: Sequence[Sequence[int]],
    ) -> list[list[float]]:
        """Score supplied tokens with this handle's raw policy distribution."""

        return await asyncio.to_thread(
            self._backend._score_completions,
            prompts,
            completion_tokens,
            use_base=self._use_base,
        )

    async def _flush_pending(self) -> None:
        # Give sibling tasks created by gather() one event-loop turn to enqueue.
        await asyncio.sleep(0)
        pending, self._pending = self._pending, []
        try:
            groups: dict[tuple[Any, ...], list[tuple[dict[str, Any], asyncio.Future]]] = {}
            for request, future in pending:
                stop_ids = tuple(token for token in (request["stop"] or []) if isinstance(token, int))
                key = (
                    request["max_tokens"],
                    float(request["temperature"]),
                    stop_ids,
                    request["num_samples"],
                )
                groups.setdefault(key, []).append((request, future))

            for group in groups.values():
                first = group[0][0]
                try:
                    results = await asyncio.to_thread(
                        self._backend._sample_batch,
                        prompt_tokens_batch=[request["prompt_tokens"] for request, _ in group],
                        max_tokens=first["max_tokens"],
                        temperature=first["temperature"],
                        stop=first["stop"],
                        num_samples=first["num_samples"],
                        use_base=self._use_base,
                    )
                    if len(results) != len(group):
                        raise RuntimeError(f"local sampler returned {len(results)} prompt results for {len(group)} requests")
                except BaseException as exc:  # noqa: BLE001 -- propagate cancellation/failure to every queued future
                    for _, future in group:
                        if not future.done():
                            future.set_exception(exc)
                else:
                    for result, (_, future) in zip(results, group):
                        if not future.done():
                            future.set_result(result)
        finally:
            self._flush_task = None
            if self._pending:
                self._flush_task = asyncio.get_running_loop().create_task(self._flush_pending())


class LocalBackend:
    """TrainingBackend on local hardware (torch + transformers [+ peft])."""

    supports_fused_opct_scoring = True

    renderer_source = "hf"
    # Local sampler handles route to the backend's current in-process model or
    # current vLLM adapter; retaining a handle does not freeze its weights.
    policy_samplers_are_snapshots = False

    @property
    def sampling_training_overlap_supported(self) -> bool:
        """Whether rollout sampling may overlap coordinator training work.

        Sleep mode deliberately reclaims the in-process sampler's GPU
        allocations for training, so a caller must treat sampling and training
        as distinct phases.  With the explicit option absent, preserve the
        historical resident-engine overlap contract.
        """

        return not bool(self.vllm_options.get("enable_sleep_mode", False))

    def __init__(
        self,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        use_lora: bool = True,
        model_instance: torch.nn.Module | None = None,
        ppo_clip_epsilon: float = losses.PPO_CLIP_EPSILON,
        sampler: str = "hf",
        vllm_options: dict | None = None,
        consistency_loss_options: dict | None = None,
        full_finetune_modules: Sequence[str] | None = None,
        keep_frozen_base: bool = False,
        hf_language_model_only: bool = False,
        hf_streaming_sampling: bool = False,
        device_map: Optional[Any] = None,
        max_memory: Optional[dict] = None,
        gradient_checkpointing: bool = False,
        gradient_checkpointing_layers: int | None = None,
        forward_microbatch_max_datums: int | None = _DEFAULT_FORWARD_MICROBATCH_MAX_DATUMS,
        forward_microbatch_max_tokens: int | None = _DEFAULT_FORWARD_MICROBATCH_MAX_TOKENS,
        target_logprob_chunk_size: int = _TARGET_LOGPROB_CHUNK_SIZE,
        gradient_reducer: Callable[[Sequence[torch.nn.Parameter]], Any] | str | None = None,
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
            consistency_loss_options: constructor kwargs for the consistency
                loss_fns (weight, layer_selection, ...); defaults are the
                AttCT-paper settings.
            full_finetune_modules: parameter-name globs or dotted components to
                train when ``use_lora=False``. ``None`` trains every parameter.
            keep_frozen_base: retain an immutable copy of the initial model for
                clean/reference forwards during selective full fine-tuning.
            hf_language_model_only: after loading a supported multimodal model,
                detach its unused vision/connector modules while retaining the
                canonical outer text-model key path needed by PEFT and vLLM.
            hf_streaming_sampling: opt into the cached temperature-only sampler
                for finite output limits too. Retains selected-token logprobs,
                not a vocabulary-sized score tensor for every generated token.
                False preserves the historical capped model.generate path.
            device_map: Transformers/Accelerate device map (for example,
                ``"auto"``) used to place one training model across the GPUs in
                this process. This is placement only; the training methods do
                not depend on it.
            max_memory: per-device limits forwarded to ``from_pretrained``.
                Requires ``device_map``. Leave activation headroom because the
                limits only govern model placement.
            gradient_checkpointing: recompute backbone activations during
                backward to make long individual sequences fit in memory.
            gradient_checkpointing_layers: when set, checkpoint only the first
                this-many backbone layers. Requires ``gradient_checkpointing``;
                omitting it preserves full-model checkpointing.
            forward_microbatch_max_datums: maximum datums in one internal
                training/scoring model forward. ``None`` disables this limit.
            forward_microbatch_max_tokens: maximum padded input-token slots in
                one internal model forward. ``None`` disables this limit. A
                single indivisible datum longer than the budget is forwarded
                alone.
            target_logprob_chunk_size: maximum selected predictive positions
                passed through the LM head at once. This bounds the temporary
                token-by-vocabulary logits and float32 cross-entropy workspace.
            gradient_reducer: optional optimizer-boundary gradient SUM reducer
                for replicated trainers. Pass a callable accepting the ordered
                trainable parameters, or ``"torch.distributed"`` to use the
                initialized default process group. ``None`` preserves the
                historical single-rank no-op behavior. Reducers run once after
                public gradient accumulation and before clipping.
        """
        if sampler not in ("hf", "vllm"):
            raise ValueError(f"Unknown sampler: {sampler!r} (expected 'hf' or 'vllm')")
        if (
            isinstance(ppo_clip_epsilon, bool)
            or not isinstance(ppo_clip_epsilon, (int, float))
            or not math.isfinite(float(ppo_clip_epsilon))
            or not 0 < ppo_clip_epsilon < 1
        ):
            raise ValueError("ppo_clip_epsilon must be a finite number in (0, 1)")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.use_lora = use_lora
        self.model: Optional[torch.nn.Module] = model_instance
        self.model_name: Optional[str] = None
        self.ppo_clip_epsilon = float(ppo_clip_epsilon)
        self.sampler = sampler
        self.vllm_options = vllm_options or {}
        self.consistency_loss_options = consistency_loss_options or {}
        self.full_finetune_modules = list(full_finetune_modules) if full_finetune_modules is not None else None
        self.keep_frozen_base = keep_frozen_base
        self.hf_language_model_only = hf_language_model_only
        if not isinstance(hf_streaming_sampling, bool):
            raise TypeError("hf_streaming_sampling must be a boolean")
        self.hf_streaming_sampling = hf_streaming_sampling
        if max_memory is not None and device_map is None:
            raise ValueError("max_memory requires device_map")
        if model_instance is not None and device_map is not None:
            raise ValueError("device_map applies only when LocalBackend loads the model")
        if not isinstance(gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a boolean")
        if gradient_checkpointing_layers is not None and (
            not isinstance(gradient_checkpointing_layers, int)
            or isinstance(gradient_checkpointing_layers, bool)
            or gradient_checkpointing_layers < 1
        ):
            raise ValueError(
                "gradient_checkpointing_layers must be a positive integer or None, "
                f"got {gradient_checkpointing_layers!r}"
            )
        if gradient_checkpointing_layers is not None and not gradient_checkpointing:
            raise ValueError("gradient_checkpointing_layers requires gradient_checkpointing=True")
        for name, value in (
            ("forward_microbatch_max_datums", forward_microbatch_max_datums),
            ("forward_microbatch_max_tokens", forward_microbatch_max_tokens),
            ("target_logprob_chunk_size", target_logprob_chunk_size),
        ):
            allows_none = name != "target_logprob_chunk_size"
            if (value is None and not allows_none) or (value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1)):
                suffix = " or None" if allows_none else ""
                raise ValueError(f"{name} must be a positive integer{suffix}, got {value!r}")
        self.device_map = device_map
        self.max_memory = max_memory
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_layers = gradient_checkpointing_layers
        self.forward_microbatch_max_datums = forward_microbatch_max_datums
        self.forward_microbatch_max_tokens = forward_microbatch_max_tokens
        self.target_logprob_chunk_size = target_logprob_chunk_size
        self._gradient_reducer = self._resolve_gradient_reducer(gradient_reducer)
        self._vllm = None  # VLLMSampler, booted lazily by _ensure_vllm() when sampler == "vllm"
        self._adapter_scratch: Path | None = None
        self._optimizer: torch.optim.AdamW | None = None
        self._pending_optimizer_state: dict | None = None
        self._consistency_loss_modules: dict[str, consistency_losses.ConsistencyLoss] = {}
        self._mlp_hooks: MLPHookManager | None = None
        self._base_mlp_hooks: MLPHookManager | None = None
        self._frozen_base_model: torch.nn.Module | None = None
        self._gradient_accumulations = 0
        self._trainable_parameter_names: list[str] = []
        self._configured_lora: LoRAConfig | None = None

    @staticmethod
    def _resolve_gradient_reducer(
        gradient_reducer: Callable[[Sequence[torch.nn.Parameter]], Any] | str | None,
    ) -> Callable[[Sequence[torch.nn.Parameter]], Any] | None:
        """Validate the explicit, optimizer-boundary gradient reducer hook."""

        if gradient_reducer is None:
            return None
        if isinstance(gradient_reducer, str):
            if gradient_reducer in {"torch.distributed", "torch_distributed"}:
                return sum_trainable_gradients_torch_distributed
            raise ValueError(
                "gradient_reducer must be None, a callable, or 'torch.distributed'; "
                f"got {gradient_reducer!r}"
            )
        if not callable(gradient_reducer):
            raise TypeError(
                "gradient_reducer must be None, a callable, or 'torch.distributed'; "
                f"got {type(gradient_reducer).__name__}"
            )
        return gradient_reducer

    def set_gradient_reducer(
        self,
        gradient_reducer: Callable[[Sequence[torch.nn.Parameter]], Any] | str | None,
    ) -> None:
        """Configure replicated-gradient synchronization before accumulation.

        Reconfiguring midway through an accumulated logical update would mix
        synchronized and unsynchronized terms, so it is rejected.  This is a
        lightweight hook for orchestration layers; it does not create or wrap
        a distributed process group itself.
        """

        if self._gradient_accumulations:
            raise RuntimeError("cannot change gradient_reducer while gradients are accumulated")
        self._gradient_reducer = self._resolve_gradient_reducer(gradient_reducer)

    # ── lifecycle ────────────────────────────────────────────────────────

    def setup(self, *, model: str, lora: LoRAConfig, resume_from: Optional[str] = None, resume_with_optimizer: bool = False) -> None:
        self.model_name = model
        self._configured_lora = lora.model_copy(deep=True)
        if self.model is None:
            import transformers

            load_kwargs: dict[str, Any] = {"torch_dtype": self.dtype}
            if self.device_map is not None:
                load_kwargs["device_map"] = self.device_map
                if self.max_memory is not None:
                    load_kwargs["max_memory"] = self.max_memory
            requires_unified_gemma_loader = _is_gemma4_unified_model_reference(model)
            if self.hf_language_model_only or requires_unified_gemma_loader:
                from transformers import AutoConfig

                model_config = AutoConfig.from_pretrained(model)
                load_kwargs["config"] = model_config
            else:
                model_config = None
            model_type = getattr(model_config, "model_type", None)
            if model_type in _GEMMA4_UNIFIED_MODEL_TYPES:
                if self.hf_language_model_only:
                    raise ValueError(
                        "Gemma 4 unified loading must retain the full conditional-generation wrapper; "
                        "do not set hf_language_model_only"
                    )
                model_loader = getattr(transformers, "AutoModelForImageTextToText", None)
                if model_loader is None:
                    raise ImportError(
                        "Gemma 4 unified loading requires a Transformers runtime exposing "
                        "AutoModelForImageTextToText"
                    )
                self.model = model_loader.from_pretrained(model, **load_kwargs)
            elif model_type == "muse_glimmer":
                from transformers import AutoModelForMultimodalLM

                self.model = AutoModelForMultimodalLM.from_pretrained(model, **load_kwargs)
            else:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        if self.hf_language_model_only:
            self._detach_unused_multimodal_modules()
        if self.use_lora:
            if self.full_finetune_modules is not None:
                raise ValueError("full_finetune_modules applies only when use_lora=False")
            if self.keep_frozen_base:
                raise ValueError("keep_frozen_base is unnecessary with LoRA; disabling the adapter is the frozen base")
            if not HAS_PEFT:
                raise ImportError("LocalBackend(use_lora=True) requires peft: `uv pip install peft`, or pass use_lora=False for full fine-tuning.")
            if lora.seed is not None:
                torch.manual_seed(lora.seed)
            target_modules = _lora_target_module_names(self.model, lora)
            target_parameters = _lora_target_parameter_names(self.model, lora)
            parameters = dict(self.model.named_parameters())
            fused_experts = [(name, tuple(parameters[name].shape)) for name in target_parameters if name in parameters and parameters[name].ndim == 3]
            if fused_experts:
                examples = ", ".join(f"{name} {shape}" for name, shape in fused_experts[:2])
                warnings.warn(
                    f"LoRA selected {len(fused_experts)} fused 3-D expert parameter(s) on "
                    f"{type(self.model).__name__} ({examples}). PEFT target_parameters may materialize "
                    "dense expert-sized deltas and substantially increase training memory. Prefer a dense model "
                    "or a backend with native MoE-aware adapters for routine experiments.",
                    FusedExpertLoRAWarning,
                    stacklevel=2,
                )
            if not target_modules and not target_parameters:
                raise ValueError("LoRA must train at least one of MLP, attention, or unembedding modules")
            peft_cfg = PeftLoraConfig(
                r=lora.rank,
                lora_alpha=lora.resolved_alpha,
                lora_dropout=lora.dropout,
                target_modules=target_modules or [],
                target_parameters=target_parameters or None,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_cfg)
            self._trainable_parameter_names = [
                name for name, parameter in self.model.named_parameters() if parameter.requires_grad
            ]
        else:
            if self.keep_frozen_base:
                self._frozen_base_model = copy.deepcopy(self.model)
                self._frozen_base_model.eval()
                for parameter in self._frozen_base_model.parameters():
                    parameter.requires_grad_(False)
            self._trainable_parameter_names = _configure_full_finetune_parameters(self.model, self.full_finetune_modules)
        if self.device_map is None:
            self.model.to(self.device)
            if self._frozen_base_model is not None:
                self._frozen_base_model.to(self.device)
        else:
            # Accelerate already placed the layers and installed the hooks that
            # transfer activations between them. Calling .to() here would undo
            # that placement by collapsing the model back onto one device.
            placement = getattr(self.model, "hf_device_map", None)
            if not placement:
                raise RuntimeError("device_map was requested but the loaded model has no hf_device_map")
            unsupported = sorted({str(value) for value in placement.values() if str(value) in {"cpu", "disk", "meta"}})
            if unsupported:
                raise ValueError(f"LocalBackend device_map training requires GPU-only placement; found offload target(s): {unsupported}")
            embeddings = self.model.get_input_embeddings() if hasattr(self.model, "get_input_embeddings") else None
            embedding_parameter = next(embeddings.parameters(), None) if embeddings is not None else None
            if embedding_parameter is not None and embedding_parameter.device.type != "meta":
                self.device = str(embedding_parameter.device)
            else:
                first = next(iter(placement.values()))
                self.device = f"cuda:{first}" if isinstance(first, int) else str(first)
            devices = sorted({str(value) for value in placement.values()})
            print(f"LocalBackend: model sharded across {len(devices)} device(s): {devices}", flush=True)
        if self.gradient_checkpointing:
            model = self._require_model()
            if not hasattr(model, "gradient_checkpointing_enable"):
                raise NotImplementedError(f"{type(model).__name__} does not support requested gradient checkpointing")
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "config"):
                model.config.use_cache = False
            if not getattr(model, "is_gradient_checkpointing", False):
                raise RuntimeError("gradient checkpointing was requested but did not activate")
            if self.gradient_checkpointing_layers is not None:
                layers = _gradient_checkpointing_backbone_layers(model)
                if self.gradient_checkpointing_layers > len(layers):
                    raise ValueError(
                        f"gradient_checkpointing_layers={self.gradient_checkpointing_layers} "
                        f"exceeds the model's {len(layers)} backbone layers"
                    )
                inactive = [
                    index
                    for index, layer in enumerate(layers)
                    if getattr(layer, "gradient_checkpointing", None) is not True
                ]
                if inactive:
                    raise RuntimeError(
                        "selective gradient checkpointing requires the model-wide enable call to activate "
                        f"a gradient_checkpointing flag on every backbone layer; inactive indices: {inactive}"
                    )
                for layer in layers[self.gradient_checkpointing_layers :]:
                    layer.gradient_checkpointing = False
        if resume_from:
            self._load_checkpoint(resume_from, with_optimizer=resume_with_optimizer)
        if self.sampler == "vllm" and not self.use_lora:
            raise NotImplementedError("sampler='vllm' requires use_lora=True: policy refresh works by hot-reloading the adapter; full-finetune weights cannot be swapped into a running vLLM engine.")
        # The vLLM engine itself boots lazily (_ensure_vllm) on the first sampler
        # request: runs that never sample (SFT) must not pay its GPU memory or
        # require the vllm package at all.

    def _require_model(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("LocalBackend.setup() must be called before use")
        return self.model

    def _detach_unused_multimodal_modules(self) -> None:
        """Drop Muse's unused vision tower without changing text adapter keys."""

        model = self._require_model()
        if getattr(getattr(model, "config", None), "model_type", None) != "muse_glimmer":
            raise NotImplementedError(
                "hf_language_model_only currently supports only Muse Glimmer"
            )
        multimodal = getattr(model, "model", None)
        language_model = getattr(multimodal, "language_model", None)
        if not isinstance(language_model, torch.nn.Module):
            raise RuntimeError(
                "Muse Glimmer text-only loading requires model.language_model"
            )
        detached = []
        for name in (
            "vision_tower",
            "vision_adapter",
            "vision_projection",
            "perception_emb_norm",
        ):
            if not hasattr(multimodal, name):
                raise RuntimeError(
                    f"Muse Glimmer text-only loading could not find model.{name}"
                )
            if getattr(multimodal, name) is not None:
                setattr(multimodal, name, None)
                detached.append(name)
        if not detached:
            raise RuntimeError("Muse Glimmer vision modules were already absent")

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

    def _sleep_vllm_for_training(self) -> bool:
        """Idempotently release a started colocated sampler before HF work.

        This never boots vLLM merely to sleep it.  ``VLLMSampler`` owns the
        enablement check and automatically wakes itself before sampling or
        adapter publication, so disabled mode is a strict no-op.
        """

        if self._vllm is None:
            return False
        return self._vllm.sleep()

    async def enter_training_phase(self) -> None:
        """Release a started in-process vLLM engine for coordinator work.

        This is intentionally a no-op when vLLM has not yet been lazily
        created, and is idempotent when the engine is already sleeping.  The
        generic RL loop uses it only when sleep mode disables sampling/training
        overlap, but direct callers may safely use it as the same phase barrier.
        """

        self._sleep_vllm_for_training()

    async def enter_rollout_phase(self) -> None:
        """Restore a previously started in-process vLLM engine for rollout.

        Do not boot an engine here: a training-only run must retain lazy vLLM
        startup.  VLLMSampler also auto-wakes immediately before a direct
        sampling call, so this lifecycle method is a phase barrier rather than
        a prerequisite for correctness.
        """

        self.release_training_memory_for_rollout()
        if self._vllm is not None:
            self._vllm.wake_up()

    def release_training_memory_for_rollout(self) -> None:
        """Return unused trainer allocations before a colocated vLLM wake.

        A rollout-worker topology keeps this backend's in-process sampler cold,
        so the release cannot be conditional on ``self._vllm`` existing.  CUDA
        kernels are synchronized first; otherwise another process could reclaim
        allocator blocks while rank-local work was still in flight.
        """

        if not str(self.device).startswith("cuda"):
            return
        device = torch.device(self.device)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    def shutdown(self) -> None:
        """Release the lazily started vLLM sampler, if any."""
        if self._vllm is None:
            return
        try:
            self._vllm.shutdown()
        finally:
            self._vllm = None

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
        """Snapshot the current LoRA adapter and point vLLM at that exact policy.

        Qwen3.5 requires a sibling vLLM-only key translation.  The raw PEFT
        snapshot stays untouched for coordinator forward/backward and is
        hash-bound to the translated copy before the latter is ever attached
        to a vLLM request.
        """

        assert self._vllm is not None and self._adapter_scratch is not None
        if self._vllm.sleeping and str(self.device).startswith("cuda"):
            # VLLMSampler wakes before accepting a new LoRA version.  Free
            # trainer-cache blocks first so that publication is a safe phase
            # transition even when refresh_policy_sampler() runs before the
            # loop's explicit enter_rollout_phase() call.
            torch.cuda.empty_cache()
        version = self._vllm.adapter_version + 1
        version_dir = self._adapter_scratch / f"v{version}"
        if version_dir.exists():
            raise RuntimeError(f"refusing to overwrite vLLM policy snapshot: {version_dir}")
        model = self._require_model()
        if is_qwen35_model_name(self.model_name):
            raw_dir = version_dir / "raw"
            compatibility_dir = version_dir / "vllm_compat"
            model.save_pretrained(str(raw_dir))
            materialize_qwen35_vllm_rollout_compat_adapter(
                raw_dir,
                compatibility_dir,
                model=str(self.model_name),
                adapter_version=version,
            )
            self._vllm.advance_policy(str(compatibility_dir), version=version)
            return
        model.save_pretrained(str(version_dir))
        self._vllm.advance_policy(str(version_dir), version=version)

    def base_sampler(self) -> LocalSamplerHandle:
        self._require_base()
        self._ensure_vllm()
        return LocalSamplerHandle(self, use_base=True)

    def _require_base(self):
        if not ((self.use_lora and HAS_PEFT) or self._frozen_base_model is not None):
            raise NotImplementedError("Base-model access (anchor sampling / KL-to-base / distill) on LocalBackend requires LoRA or keep_frozen_base=True for full fine-tuning.")

    def _base_ctx(self):
        self._require_base()
        if self.use_lora:
            return self._require_model().disable_adapter()  # peft context manager
        return _nullcontext()

    def _model_for(self, *, use_base: bool) -> torch.nn.Module:
        if not use_base:
            return self._require_model()
        self._require_base()
        if self.use_lora:
            return self._require_model()
        assert self._frozen_base_model is not None
        return self._frozen_base_model

    def run_qwen35_consistency_preflight(
        self,
        datums: Sequence[Any],
        *,
        method: str,
        expected_group_size: int | None = None,
        lora_scope: str = "historical",
    ) -> dict[str, Any]:
        """Run a real, no-step ACT/AttCT/MLPCT backward probe and restore state.

        This is intentionally a LocalBackend capability rather than a generic
        training-backend protocol operation.  It verifies the exact Qwen3.5-9B
        hybrid topology and the method's declared LoRA set, then exercises the
        same Transformers/PEFT paired forward/backward path as the training
        loop on the caller's deterministic probe group. The caller owns
        immutable evidence logging and fails closed when ``passed`` is false.
        """

        loss_by_method = {
            "act": "activation_consistency",
            "attct": "attention_consistency",
            "mlpct": "mlp_consistency",
        }
        report: dict[str, Any] = {
            "schema_version": 1,
            "kind": "qwen35-consistency-preflight",
            "method": method,
            "model": self.model_name,
            "passed": False,
            "topology": {},
            "lora": {},
            "probe": {"datums_used": 0, "expected_group_size": expected_group_size},
            "state_restoration": {},
            "errors": [],
        }
        errors: list[str] = report["errors"]
        report["lora_scope"] = lora_scope
        if lora_scope not in {"historical", "act_qv_fused_qkv"} or (
            lora_scope == "act_qv_fused_qkv" and method != "act"
        ):
            errors.append("unsupported explicit Qwen3.5 consistency LoRA scope")
            return report
        loss_fn = loss_by_method.get(method)
        if loss_fn is None:
            errors.append("Qwen3.5 consistency preflight supports only method='act', 'attct', or 'mlpct'")
            return report

        try:
            model = self._require_model()
        except RuntimeError as exc:
            errors.append(str(exc))
            return report

        # The probe must be observational: preserve stochastic state so the
        # first real training microbatch receives exactly the same dropout and
        # shuffle stream it would have received without this attestation.
        python_rng_state = random.getstate()
        torch_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        trainable_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        gradients_before = [
            (parameter, None if parameter.grad is None else parameter.grad.detach().clone())
            for _, parameter in trainable_parameters
        ]
        adapter_parameter_snapshots = [
            (parameter, parameter.detach().clone())
            for name, parameter in model.named_parameters()
            if ".lora_" in name
        ]
        module_training_before = [(module, module.training) for module in model.modules()]
        attention_implementation_before: list[tuple[torch.nn.Module, Any, Any]] = []
        seen_attention_configs: set[int] = set()
        for candidate in (_underlying_causal_lm(model), model):
            config = getattr(candidate, "config", None)
            if config is None or not hasattr(config, "_attn_implementation") or id(config) in seen_attention_configs:
                continue
            seen_attention_configs.add(id(config))
            attention_implementation_before.append(
                (candidate, config, getattr(config, "_attn_implementation"))
            )
        consistency_loss_modules_before = dict(self._consistency_loss_modules)
        mlp_hooks_before = self._mlp_hooks
        base_mlp_hooks_before = self._base_mlp_hooks
        gradient_accumulations_before = self._gradient_accumulations
        optimizer_before = self._optimizer
        try:
            adapter_digest_before, adapter_parameter_count = _adapter_parameter_digest(model)
        except Exception as exc:  # pragma: no cover - only pathological device/module states
            adapter_digest_before, adapter_parameter_count = None, 0
            errors.append(f"could not fingerprint LoRA adapter before probe: {type(exc).__name__}: {exc}")

        report["state_restoration"] = {
            "gradient_accumulations_before": gradient_accumulations_before,
            "optimizer_present_before": optimizer_before is not None,
            "adapter_parameter_count": adapter_parameter_count,
            "adapter_digest_before": adapter_digest_before,
            "attention_implementations_before": [
                {
                    "model_class": type(candidate).__name__,
                    "value": implementation,
                }
                for candidate, _config, implementation in attention_implementation_before
            ],
        }
        probe_observation: dict[str, Any] = {}
        lora_b_parameters: list[tuple[int, str, str, str, str, torch.nn.Parameter]] = []

        try:
            topology, topology_errors = _inspect_qwen35_consistency_topology(model)
            report["topology"] = topology
            errors.extend(topology_errors)
            if method == "act":
                lora_report, lora_b_parameters, lora_errors = _inspect_qwen35_act_attention_lora(
                    model,
                    self._configured_lora,
                    qv_fused_qkv=lora_scope == "act_qv_fused_qkv",
                )
            else:
                lora_report, lora_b_parameters, lora_errors = _inspect_qwen35_strict_qv_lora(
                    model,
                    self._configured_lora,
                )
            report["lora"] = lora_report
            errors.extend(lora_errors)

            probe_datums = list(datums)
            if not probe_datums:
                errors.append("Qwen3.5 consistency preflight needs at least one aligned paired datum")
            if expected_group_size is not None and len(probe_datums) != expected_group_size:
                errors.append(
                    "Qwen3.5 consistency preflight received "
                    f"{len(probe_datums)} datum(s), but its deterministic first accumulation group requires "
                    f"{expected_group_size}"
                )
            if not errors:
                # Isolate the probe's gradients from any pre-existing state;
                # they are restored below even when the probe rejects the run.
                for _, parameter in trainable_parameters:
                    parameter.grad = None
                pending = self._consistency_forward_backward(
                    probe_datums,
                    loss_fn,
                    preflight_observation=probe_observation,
                )
                raw_loss = pending._value.metrics.get("loss")
                loss = float(raw_loss) if raw_loss is not None else math.nan
                report["probe"].update(probe_observation)
                report["probe"]["datums_used"] = len(probe_datums)
                report["probe"]["loss"] = loss if math.isfinite(loss) else None
                if not math.isfinite(loss):
                    errors.append("Qwen3.5 consistency preflight produced a non-finite loss")
                elif loss <= 0:
                    errors.append("Qwen3.5 consistency preflight produced a non-positive loss")

                if method == "attct":
                    expected_count = len(_QWEN35_CONSISTENCY_FULL_ATTENTION_LAYERS)
                    for side in ("variant_attention_counts", "reference_attention_counts"):
                        observed = probe_observation.get(side, [])
                        if len(observed) != len(probe_datums) or any(count != expected_count for count in observed):
                            errors.append(
                                "Qwen3.5 AttCT preflight requires "
                                f"{expected_count} attention tensors on every {side.removesuffix('_counts')}, "
                                f"got {observed!r}"
                            )
                elif method == "mlpct":
                    expected_count = _QWEN35_CONSISTENCY_DECODER_LAYERS
                    for side in ("variant_mlp_hook_counts", "reference_mlp_hook_counts"):
                        observed = probe_observation.get(side, [])
                        if len(observed) != len(probe_datums) or any(count != expected_count for count in observed):
                            errors.append(
                                "Qwen3.5 MLPCT preflight requires "
                                f"{expected_count} captured MLP states on every {side.removesuffix('_counts')}, "
                                f"got {observed!r}"
                            )
                else:
                    expected_count = _QWEN35_CONSISTENCY_DECODER_LAYERS + 1
                    for side in ("variant_hidden_state_counts", "reference_hidden_state_counts"):
                        observed = probe_observation.get(side, [])
                        if len(observed) != len(probe_datums) or any(count != expected_count for count in observed):
                            errors.append(
                                "Qwen3.5 ACT preflight requires "
                                f"{expected_count} hidden-state tensors on every {side.removesuffix('_counts')}, "
                                f"got {observed!r}"
                            )
                    matched = probe_observation.get("act_matching_suffix_lengths", [])
                    if len(matched) != len(probe_datums) or any(
                        not isinstance(length, int) or length <= 0 for length in matched
                    ):
                        errors.append(
                            "Qwen3.5 ACT preflight requires a positive matching suffix on every paired datum; "
                            f"got {matched!r}"
                        )

                gradient_report: list[dict[str, Any]] = []
                positive_by_family: dict[str, int] = {}
                for layer, family, projection, adapter, name, parameter in lora_b_parameters:
                    gradient = parameter.grad
                    exempt_terminal_attct_v = method == "attct" and layer == 31 and projection == "v_proj"
                    finite: bool | None
                    norm: float | None
                    positive = False
                    if gradient is None:
                        finite, norm = None, None
                    else:
                        finite = bool(torch.isfinite(gradient).all().item())
                        norm = float(torch.linalg.vector_norm(gradient.detach().float())) if finite else None
                        positive = bool(finite and norm is not None and norm > 0.0)
                    # AttCT/MLPCT retain their stricter every-tensor check. ACT
                    # has a broader hybrid projection set; the direct one-pair
                    # gate proves a real gradient reaches each attention family
                    # while still rejecting a non-finite individual tensor.
                    valid = (
                        bool(finite is not False and (positive or exempt_terminal_attct_v))
                        if method != "act"
                        else finite is not False
                    )
                    gradient_report.append(
                        {
                            "name": name,
                            "physical_layer": layer,
                            "family": family,
                            "projection": projection,
                            "adapter": adapter,
                            "gradient_present": gradient is not None,
                            "finite": finite,
                            "l2_norm": norm,
                            "positive": positive,
                            "exempt_terminal_attct_v": exempt_terminal_attct_v,
                            "valid": valid,
                        }
                    )
                    if positive:
                        positive_by_family[family] = positive_by_family.get(family, 0) + 1
                    if not valid:
                        requirement = "a finite LoRA-B gradient" if method == "act" else "a finite positive LoRA-B gradient"
                        errors.append(
                            f"Qwen3.5 consistency preflight requires {requirement} for "
                            f"physical layer {layer} {projection}; got gradient_present={gradient is not None}, "
                            f"finite={finite}, l2_norm={norm}"
                        )
                report["lora"]["lora_b_gradients"] = gradient_report
                if method == "act":
                    report["lora"]["positive_lora_b_gradients_by_family"] = positive_by_family
                    for family in ("self_attn", "linear_attn"):
                        if positive_by_family.get(family, 0) < 1:
                            errors.append(
                                "Qwen3.5 ACT preflight requires at least one finite positive LoRA-B gradient "
                                f"in the {family} family; got {positive_by_family.get(family, 0)}"
                            )
        except Exception as exc:
            errors.append(f"Qwen3.5 consistency preflight probe failed: {type(exc).__name__}: {exc}")
        finally:
            # Fingerprint before restoring snapshots so an unexpected in-place
            # adapter mutation is both recorded and fail-closed, then restore
            # the original tensor bytes before returning control to training.
            adapter_digest_after_probe = None
            try:
                adapter_digest_after_probe, _ = _adapter_parameter_digest(model)
            except Exception as exc:  # pragma: no cover - only pathological device/module states
                errors.append(f"could not fingerprint LoRA adapter after probe: {type(exc).__name__}: {exc}")
            adapter_mutated = (
                adapter_digest_before is not None
                and adapter_digest_after_probe is not None
                and adapter_digest_before != adapter_digest_after_probe
            )
            if adapter_mutated:
                errors.append("Qwen3.5 consistency preflight mutated LoRA adapter weights")
            adapter_restored = False
            try:
                with torch.no_grad():
                    for parameter, snapshot in adapter_parameter_snapshots:
                        parameter.copy_(snapshot)
                adapter_restored = True
            except Exception as exc:  # pragma: no cover - only pathological device/module states
                errors.append(f"could not restore LoRA adapter after probe: {type(exc).__name__}: {exc}")

            gradients_restored = False
            try:
                for parameter, gradient in gradients_before:
                    parameter.grad = gradient
                gradients_restored = True
            except Exception as exc:  # pragma: no cover - only pathological device/module states
                errors.append(f"could not restore gradients after probe: {type(exc).__name__}: {exc}")
            self._gradient_accumulations = gradient_accumulations_before
            gradient_restoration_verified = gradients_restored and all(
                parameter.grad is gradient for parameter, gradient in gradients_before
            )

            for module, was_training in module_training_before:
                module.training = was_training
            module_training_modes_restored = all(
                module.training is was_training for module, was_training in module_training_before
            )

            attention_implementations_after_probe = [
                {
                    "model_class": type(candidate).__name__,
                    "value": getattr(config, "_attn_implementation"),
                }
                for candidate, config, _implementation in attention_implementation_before
            ]
            attention_implementation_restored = False
            try:
                for candidate, config, implementation in attention_implementation_before:
                    if getattr(config, "_attn_implementation") == implementation:
                        continue
                    setter = getattr(candidate, "set_attn_implementation", None)
                    if implementation is not None and callable(setter):
                        setter(implementation)
                    else:
                        setattr(config, "_attn_implementation", implementation)
                attention_implementation_restored = all(
                    getattr(config, "_attn_implementation") == implementation
                    for _candidate, config, implementation in attention_implementation_before
                )
                if not attention_implementation_restored:
                    errors.append("could not restore attention implementation after consistency preflight")
            except Exception as exc:  # pragma: no cover - architecture-specific setter failure
                errors.append(
                    "could not restore attention implementation after consistency preflight: "
                    f"{type(exc).__name__}: {exc}"
                )

            backend_caches_restored = False
            try:
                self._consistency_loss_modules.clear()
                self._consistency_loss_modules.update(consistency_loss_modules_before)
                self._mlp_hooks = mlp_hooks_before
                self._base_mlp_hooks = base_mlp_hooks_before
                backend_caches_restored = (
                    self._consistency_loss_modules == consistency_loss_modules_before
                    and self._mlp_hooks is mlp_hooks_before
                    and self._base_mlp_hooks is base_mlp_hooks_before
                )
                if not backend_caches_restored:
                    errors.append("could not restore LocalBackend consistency caches after preflight")
            except Exception as exc:  # pragma: no cover - defensive cache recovery
                errors.append(f"could not restore LocalBackend consistency caches after preflight: {type(exc).__name__}: {exc}")

            # Do this after every model/config/cache restoration in case an
            # architecture-specific attention setter touches a generator.
            rng_restored = False
            try:
                random.setstate(python_rng_state)
                torch.set_rng_state(torch_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
                rng_restored = True
            except Exception as exc:  # pragma: no cover - hardware-specific CUDA failure
                errors.append(f"could not restore RNG state after probe: {type(exc).__name__}: {exc}")
            rng_restoration_verified = False
            try:
                current_cuda_rng_states = torch.cuda.get_rng_state_all() if cuda_rng_states is not None else None
                cuda_rng_matches = cuda_rng_states is None or (
                    current_cuda_rng_states is not None
                    and len(current_cuda_rng_states) == len(cuda_rng_states)
                    and all(
                        torch.equal(current, expected)
                        for current, expected in zip(current_cuda_rng_states, cuda_rng_states)
                    )
                )
                rng_restoration_verified = bool(
                    rng_restored
                    and random.getstate() == python_rng_state
                    and torch.equal(torch.get_rng_state(), torch_rng_state)
                    and cuda_rng_matches
                )
            except Exception as exc:  # pragma: no cover - hardware-specific CUDA failure
                errors.append(f"could not verify RNG restoration after probe: {type(exc).__name__}: {exc}")

            adapter_digest_after_restore = None
            try:
                adapter_digest_after_restore, _ = _adapter_parameter_digest(model)
            except Exception as exc:  # pragma: no cover - only pathological device/module states
                errors.append(f"could not fingerprint restored LoRA adapter: {type(exc).__name__}: {exc}")
            optimizer_identity_preserved = self._optimizer is optimizer_before
            if not optimizer_identity_preserved:
                errors.append("Qwen3.5 consistency preflight unexpectedly created or replaced an optimizer")
            report["state_restoration"].update(
                {
                    "gradient_accumulations_after": self._gradient_accumulations,
                    "gradient_accumulations_restored": self._gradient_accumulations == gradient_accumulations_before,
                    "gradients_restored": gradients_restored,
                    "gradient_restoration_verified": gradient_restoration_verified,
                    "rng_restored": rng_restored,
                    "rng_restoration_verified": rng_restoration_verified,
                    "module_training_modes_restored": module_training_modes_restored,
                    "attention_implementations_after_probe": attention_implementations_after_probe,
                    "attention_implementation_restored": attention_implementation_restored,
                    "backend_caches_restored": backend_caches_restored,
                    "optimizer_present_after": self._optimizer is not None,
                    "optimizer_identity_preserved": optimizer_identity_preserved,
                    "adapter_digest_after_probe": adapter_digest_after_probe,
                    "adapter_mutated_during_probe": adapter_mutated,
                    "adapter_restored": adapter_restored,
                    "adapter_digest_after_restore": adapter_digest_after_restore,
                    "adapter_restoration_verified": (
                        adapter_digest_before is not None
                        and adapter_digest_after_restore is not None
                        and adapter_digest_before == adapter_digest_after_restore
                    ),
                }
            )
            # A preflight is not evidence unless every restoration invariant is
            # true. Keep this centralized so a future diagnostic cannot be
            # reported as false while the run still proceeds.
            restoration_invariants = {
                "gradient_accumulations_restored": report["state_restoration"]["gradient_accumulations_restored"],
                "gradients_restored": report["state_restoration"]["gradients_restored"],
                "gradient_restoration_verified": report["state_restoration"]["gradient_restoration_verified"],
                "rng_restored": report["state_restoration"]["rng_restored"],
                "rng_restoration_verified": report["state_restoration"]["rng_restoration_verified"],
                "module_training_modes_restored": report["state_restoration"]["module_training_modes_restored"],
                "attention_implementation_restored": report["state_restoration"]["attention_implementation_restored"],
                "backend_caches_restored": report["state_restoration"]["backend_caches_restored"],
                "optimizer_identity_preserved": report["state_restoration"]["optimizer_identity_preserved"],
                "adapter_restored": report["state_restoration"]["adapter_restored"],
                "adapter_restoration_verified": report["state_restoration"]["adapter_restoration_verified"],
            }
            for name, valid in restoration_invariants.items():
                if valid is not True:
                    errors.append(f"Qwen3.5 consistency preflight state-restoration invariant failed: {name}")

        report["passed"] = not errors
        return report

    def _sample(
        self,
        *,
        prompt_tokens: list[int],
        max_tokens: int | None,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
    ) -> list[SampledSequence]:
        """Route sampling to the configured engine (vLLM if set, else HF generate)."""
        return self._sample_batch(
            prompt_tokens_batch=[prompt_tokens],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            num_samples=num_samples,
            use_base=use_base,
        )[0]

    def _sample_batch(
        self,
        *,
        prompt_tokens_batch: Sequence[Sequence[int]],
        max_tokens: int | None,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
    ) -> list[list[SampledSequence]]:
        """Route a prompt batch to one vLLM/HF generation call."""
        if self._vllm is not None:
            return self._vllm.sample_batch(
                [list(tokens) for tokens in prompt_tokens_batch],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                num_samples=num_samples,
                use_base=use_base,
            )
        return self._generate_batch(
            prompt_tokens_batch=prompt_tokens_batch,
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
        max_tokens: int | None,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
    ) -> list[SampledSequence]:
        return self._generate_batch(
            prompt_tokens_batch=[prompt_tokens],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            num_samples=num_samples,
            use_base=use_base,
        )[0]

    def _generate_batch(
        self,
        *,
        prompt_tokens_batch: Sequence[Sequence[int]],
        max_tokens: int | None,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
    ) -> list[list[SampledSequence]]:
        if not prompt_tokens_batch:
            return []
        model = self._model_for(use_base=use_base)
        was_training = model.training
        model.eval()
        eos_ids = local_hf_eos_token_ids(model, stop) or None
        max_prompt_len = max(len(tokens) for tokens in prompt_tokens_batch)
        if max_prompt_len == 0:
            raise ValueError("sampling prompts must contain at least one token")
        pad_token_id = eos_ids[0] if eos_ids else 0
        input_ids = torch.full(
            (len(prompt_tokens_batch), max_prompt_len),
            pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, tokens in enumerate(prompt_tokens_batch):
            if not tokens:
                raise ValueError("sampling prompts must contain at least one token")
            input_ids[index, -len(tokens) :] = torch.tensor(tokens, dtype=torch.long, device=self.device)
            attention_mask[index, -len(tokens) :] = 1
        if max_tokens is None or self.hf_streaming_sampling:
            if max_tokens is not None and (
                isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
            ):
                raise ValueError("streaming HF max_tokens must be a positive integer or None")
            if not eos_ids:
                raise RuntimeError("streaming local HF generation requires at least one concrete EOS/stop token")
            if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 1:
                raise ValueError("num_samples must be a positive integer")

            # Do not call Transformers.generate here: its defaults introduce a
            # hidden length bound when max_new_tokens is absent, and retains
            # vocabulary-sized output_scores for every generated token. Keep
            # the existing temperature-only sampling distribution. None still
            # means EOS-only; a finite limit is an explicit output-token cap.
            expanded_input_ids = input_ids.repeat_interleave(num_samples, dim=0)
            running_attention = attention_mask.repeat_interleave(num_samples, dim=0)
            generated = expanded_input_ids
            sequence_count = generated.shape[0]
            unfinished = torch.ones(sequence_count, dtype=torch.bool, device=self.device)
            sampled_tokens: list[list[int]] = [[] for _ in range(sequence_count)]
            sampled_logprobs: list[list[float]] = [[] for _ in range(sequence_count)]
            past_key_values = None
            generated_steps = 0
            try:
                with torch.no_grad():
                    ctx = self._base_ctx() if use_base else _nullcontext()
                    with ctx:
                        while bool(torch.any(unfinished).item()):
                            if max_tokens is not None and generated_steps >= max_tokens:
                                break
                            model_input_ids = generated if past_key_values is None else generated[:, -1:]
                            outputs = model(
                                input_ids=model_input_ids,
                                attention_mask=running_attention,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True,
                            )
                            past_key_values = getattr(outputs, "past_key_values", None)
                            if past_key_values is None:
                                raise RuntimeError("EOS-only local HF generation requires a model KV cache")
                            logits = outputs.logits[:, -1, :].float()
                            if temperature > 0:
                                behavior_logprobs = torch.log_softmax(logits / float(temperature), dim=-1)
                                next_tokens = torch.multinomial(behavior_logprobs.exp(), num_samples=1).squeeze(-1)
                            else:
                                behavior_logprobs = torch.log_softmax(logits, dim=-1)
                                next_tokens = torch.argmax(logits, dim=-1)
                            selected_logprobs = behavior_logprobs.gather(-1, next_tokens[:, None]).squeeze(-1)

                            active_before_step = unfinished.clone()
                            emitted = torch.where(
                                active_before_step,
                                next_tokens,
                                torch.full_like(next_tokens, pad_token_id),
                            )
                            for sequence_index in torch.nonzero(active_before_step, as_tuple=False).flatten().tolist():
                                sampled_tokens[sequence_index].append(int(emitted[sequence_index]))
                                sampled_logprobs[sequence_index].append(float(selected_logprobs[sequence_index]))

                            generated = torch.cat((generated, emitted[:, None]), dim=-1)
                            running_attention = torch.cat(
                                (
                                    running_attention,
                                    active_before_step.to(dtype=running_attention.dtype)[:, None],
                                ),
                                dim=-1,
                            )
                            finished_now = torch.zeros_like(unfinished)
                            for eos_token_id in eos_ids:
                                finished_now |= emitted.eq(eos_token_id)
                            unfinished &= ~finished_now
                            generated_steps += 1
            finally:
                if was_training:
                    model.train()

            return [
                [
                    SampledSequence(
                        tokens=sampled_tokens[prompt_index * num_samples + sample_index],
                        logprobs=sampled_logprobs[prompt_index * num_samples + sample_index],
                    )
                    for sample_index in range(num_samples)
                ]
                for prompt_index in range(len(prompt_tokens_batch))
            ]
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
                        pad_token_id=pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
        finally:
            if was_training:
                model.train()

        batches: list[list[SampledSequence]] = []
        for prompt_index in range(len(prompt_tokens_batch)):
            sequences: list[SampledSequence] = []
            for sample_index in range(num_samples):
                sequence_index = prompt_index * num_samples + sample_index
                tokens: list[int] = []
                logprobs: list[float] = []
                for step, step_scores in enumerate(out.scores):
                    position = max_prompt_len + step
                    if position >= out.sequences.shape[1]:
                        break
                    token = int(out.sequences[sequence_index, position])
                    logprob = torch.log_softmax(step_scores[sequence_index].float(), dim=-1)[token]
                    tokens.append(token)
                    logprobs.append(float(logprob))
                    if eos_ids and token in eos_ids:
                        break
                finish_reason = "stop" if tokens and eos_ids and tokens[-1] in eos_ids else "length"
                sequence = SampledSequence(tokens=tokens, logprobs=logprobs, finish_reason=finish_reason)
                sequence.validate()
                sequences.append(sequence)
            batches.append(sequences)
        return batches

    # ── training ─────────────────────────────────────────────────────────

    def _forward_microbatches(self, token_counts: Sequence[int]) -> list[list[int]]:
        """Bucket datum indices under both count and padded-token limits.

        A datum is indivisible: one whose sequence alone exceeds the token
        budget is emitted as a singleton. Longest-first bucketing avoids
        pathological padding for mixed-length rollouts; callers restore results
        to the original datum order.
        """

        for index, token_count in enumerate(token_counts):
            if token_count < 0:
                raise ValueError(f"token counts must be non-negative, got {token_count} at index {index}")
        if not token_counts:
            return []
        all_fit = (self.forward_microbatch_max_datums is None or len(token_counts) <= self.forward_microbatch_max_datums) and (self.forward_microbatch_max_tokens is None or len(token_counts) * max(token_counts) <= self.forward_microbatch_max_tokens)
        if all_fit:
            return [list(range(len(token_counts)))]

        chunks: list[list[int]] = []
        chunk: list[int] = []
        max_tokens = 0
        ordered_indices = sorted(range(len(token_counts)), key=lambda index: (-token_counts[index], index))
        for index in ordered_indices:
            token_count = token_counts[index]
            next_count = len(chunk) + 1
            next_max_tokens = max(max_tokens, token_count)
            exceeds_datums = self.forward_microbatch_max_datums is not None and next_count > self.forward_microbatch_max_datums
            exceeds_tokens = self.forward_microbatch_max_tokens is not None and next_count * next_max_tokens > self.forward_microbatch_max_tokens
            if chunk and (exceeds_datums or exceeds_tokens):
                chunks.append(chunk)
                chunk = []
                max_tokens = 0
            chunk.append(index)
            max_tokens = max(max_tokens, token_count)
        if chunk:
            chunks.append(chunk)
        return chunks

    def _predictive_logprobs_batch(
        self,
        token_lists: Sequence[Sequence[int]],
        prediction_positions: Sequence[Sequence[int]],
        targets: Sequence[torch.Tensor],
        *,
        use_base: bool,
        behavior_temperature: float | None = None,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Score selected predictive positions in one already-bounded batch."""

        if not (len(token_lists) == len(prediction_positions) == len(targets)):
            raise ValueError("token_lists, prediction_positions, and targets must have the same length")
        if not token_lists:
            return ([], []) if behavior_temperature is not None else []
        if behavior_temperature is not None and (
            not math.isfinite(behavior_temperature) or behavior_temperature <= 0
        ):
            raise ValueError("behavior_temperature must be a finite positive number")

        normalized_tokens = [list(tokens) for tokens in token_lists]
        normalized_positions = [list(positions) for positions in prediction_positions]
        normalized_targets = [target.long().to(self.device) for target in targets]
        for index, (tokens, positions, target) in enumerate(zip(normalized_tokens, normalized_positions, normalized_targets)):
            if len(positions) != target.numel():
                raise ValueError(f"prediction positions and targets differ at index {index}: {len(positions)} and {target.numel()}")
            if any(position < 0 or position >= len(tokens) for position in positions):
                raise ValueError(f"prediction positions at index {index} must be within a {len(tokens)}-token input, got {positions}")

        max_len = max(len(tokens) for tokens in normalized_tokens)
        if max_len == 0:
            output = [torch.empty(0, dtype=torch.float32, device=self.device) for _ in normalized_tokens]
            return (output, [values.clone() for values in output]) if behavior_temperature is not None else output

        input_ids = torch.zeros((len(normalized_tokens), max_len), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(input_ids)
        for index, tokens in enumerate(normalized_tokens):
            if tokens:
                input_ids[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=self.device)
                attention_mask[index, : len(tokens)] = 1

        model = self._model_for(use_base=use_base)
        components = _selected_token_components(model)
        ctx = self._base_ctx() if use_base else _nullcontext()
        with ctx:
            if components is None:
                # Exact objective-preserving fallback for architectures whose
                # post-head logit semantics have not been explicitly verified.
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                raw_output = []
                behavior_output = []
                for index, (positions, target) in enumerate(zip(normalized_positions, normalized_targets)):
                    position_tensor = torch.tensor(positions, dtype=torch.long, device=logits.device)
                    selected_logits = logits[index].index_select(0, position_tensor)
                    raw_output.append(
                        _selected_target_logprobs(
                            selected_logits,
                            target.to(logits.device),
                            chunk_size=self.target_logprob_chunk_size,
                        ).to(self.device)
                    )
                    if behavior_temperature is not None:
                        behavior_output.append(
                            _selected_target_logprobs(
                                selected_logits / behavior_temperature,
                                target.to(logits.device),
                                chunk_size=self.target_logprob_chunk_size,
                            ).to(self.device)
                        )
                return (raw_output, behavior_output) if behavior_temperature is not None else raw_output

            backbone_output = components.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden_states = getattr(backbone_output, "last_hidden_state", None)
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
                raise RuntimeError(f"{type(components.backbone).__name__} did not return [batch, sequence, hidden] last_hidden_state")
            if hidden_states.shape[:2] != input_ids.shape:
                raise RuntimeError(f"selected-token backbone changed the batch/sequence shape: input={tuple(input_ids.shape)}, hidden={tuple(hidden_states.shape)}")
            del backbone_output

            selected_hidden = []
            selected_targets = []
            lengths = []
            for index, (positions, target) in enumerate(zip(normalized_positions, normalized_targets)):
                lengths.append(len(positions))
                if positions:
                    position_tensor = torch.tensor(positions, dtype=torch.long, device=hidden_states.device)
                    selected_hidden.append(hidden_states[index].index_select(0, position_tensor))
                    selected_targets.append(target)
            if not selected_hidden:
                output = [hidden_states.new_empty((0,), dtype=torch.float32) for _ in normalized_tokens]
                return (output, [values.clone() for values in output]) if behavior_temperature is not None else output

            flat_hidden = torch.cat(selected_hidden)
            flat_targets = torch.cat(selected_targets)
            if behavior_temperature is None:
                flat_logprobs = _selected_hidden_logprobs(
                    flat_hidden,
                    flat_targets,
                    components.lm_head,
                    chunk_size=self.target_logprob_chunk_size,
                    output_multiplier=components.output_multiplier,
                    final_logit_softcapping=components.final_logit_softcapping,
                    # Disabled-adapter base scoring is no-grad in supported callers.
                    # Avoid a future backward recomputation after that context exits.
                    checkpoint_chunks=not use_base,
                )
                return [values.to(self.device) for values in flat_logprobs.split(lengths)]

            flat_raw, flat_behavior = _selected_hidden_raw_and_temperature_logprobs(
                flat_hidden,
                flat_targets,
                components.lm_head,
                temperature=behavior_temperature,
                chunk_size=self.target_logprob_chunk_size,
                output_multiplier=components.output_multiplier,
                final_logit_softcapping=components.final_logit_softcapping,
            )
            return (
                [values.to(self.device) for values in flat_raw.split(lengths)],
                [values.to(self.device) for values in flat_behavior.split(lengths)],
            )

    def _target_logprobs_batch(self, datums: Sequence[Any], use_base: bool = False) -> list[torch.Tensor]:
        """Forward one already-bounded batch and gather its target logprobs."""

        if not datums:
            return []
        token_lists = [d.model_input.to_ints() for d in datums]
        targets = [d.loss_fn_inputs["target_tokens"].to_torch() for d in datums]
        for index, (tokens, target) in enumerate(zip(token_lists, targets)):
            if len(tokens) != target.numel():
                raise ValueError(f"model_input and target_tokens must align at index {index}, got {len(tokens)} and {target.numel()} tokens")
        output = self._predictive_logprobs_batch(
            token_lists,
            [range(len(tokens)) for tokens in token_lists],
            targets,
            use_base=use_base,
        )
        if not isinstance(output, list):
            raise TypeError("ordinary target scoring unexpectedly returned temperature-adjusted scores")
        return output

    def _opct_logprobs_batch(
        self,
        datums: Sequence[Any],
        *,
        behavior_temperature: float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return raw and sampling-distribution scores from one model pass."""

        if not datums:
            return [], []
        token_lists = [datum.model_input.to_ints() for datum in datums]
        targets = [datum.loss_fn_inputs["target_tokens"].to_torch() for datum in datums]
        for index, (tokens, target) in enumerate(zip(token_lists, targets)):
            if len(tokens) != target.numel():
                raise ValueError(
                    "model_input and target_tokens must align at index "
                    f"{index}, got {len(tokens)} and {target.numel()} tokens"
                )
        output = self._predictive_logprobs_batch(
            token_lists,
            [range(len(tokens)) for tokens in token_lists],
            targets,
            use_base=False,
            behavior_temperature=behavior_temperature,
        )
        if not isinstance(output, tuple):
            raise TypeError("OPCT target scoring did not return temperature-adjusted scores")
        return output

    def _target_logprobs(self, datums: Sequence[Any], use_base: bool = False) -> list[torch.Tensor]:
        """Gather target logprobs using bounded internal model forwards.

        Training uses :meth:`_target_logprobs_batch` directly so each chunk's
        graph can be backpropagated and released immediately. This wrapper is
        primarily for no-grad policy/base scoring (including the KL penalty).
        """

        datums = list(datums)
        token_counts = [len(d.model_input.to_ints()) for d in datums]
        out: list[torch.Tensor | None] = [None] * len(datums)
        for indices in self._forward_microbatches(token_counts):
            chunk = self._target_logprobs_batch([datums[index] for index in indices], use_base=use_base)
            for index, logprobs in zip(indices, chunk):
                out[index] = logprobs
        assert all(logprobs is not None for logprobs in out)
        return [logprobs for logprobs in out if logprobs is not None]

    def logical_loss_denominator(self, datums: Sequence[Any], loss_fn: str) -> torch.Tensor:
        """Return this backend's contribution to an RL/SFT logical denominator.

        See :func:`logical_loss_denominator` for the distributed contract.  A
        coordinator can use this method on each shard and SUM the scalar, or
        calculate it once before sharding.  The returned tensor lives on this
        backend's training device so it can be passed directly to a collective.
        """

        return logical_loss_denominator(datums, loss_fn, device=self.device)

    @staticmethod
    def _coerce_global_loss_denominator(
        global_loss_denominator: torch.Tensor | float | int,
        *,
        like: torch.Tensor,
    ) -> torch.Tensor:
        """Move a caller-provided global normalizer onto the local device."""

        if isinstance(global_loss_denominator, torch.Tensor):
            if global_loss_denominator.numel() != 1:
                raise ValueError(
                    "global_loss_denominator must be scalar, got "
                    f"shape {tuple(global_loss_denominator.shape)}"
                )
            denominator = global_loss_denominator.detach().to(device=like.device, dtype=like.dtype)
        else:
            denominator = torch.as_tensor(
                global_loss_denominator,
                device=like.device,
                dtype=like.dtype,
            )
            if denominator.numel() != 1:
                raise ValueError("global_loss_denominator must be scalar")
        if not bool(torch.isfinite(denominator).all()):
            raise ValueError("global_loss_denominator must be finite")
        return denominator

    async def submit_forward_backward(
        self,
        datums: Sequence[Any],
        loss_fn: str,
        *,
        global_loss_denominator: torch.Tensor | float | int | None = None,
    ) -> _ResolvedPending:
        self._sleep_vllm_for_training()
        if loss_fn in CONSISTENCY_LOSS_CLASSES:
            if global_loss_denominator is not None:
                raise ValueError(
                    "global_loss_denominator is currently defined only for "
                    "cross_entropy, ppo, and importance_sampling"
                )
            return self._consistency_forward_backward(datums, loss_fn)
        if loss_fn not in ("cross_entropy", "ppo", "importance_sampling"):
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        datums = list(datums)
        model = self._require_model()
        model.train()

        if loss_fn == "cross_entropy":
            weights = [d.loss_fn_inputs["weights"].to_torch().to(self.device) for d in datums]
            local_denominator = (
                sum(weight.sum() for weight in weights)
                if weights
                else torch.zeros((), device=self.device)
            )
        else:
            sampled = [d.loss_fn_inputs["logprobs"].to_torch().to(self.device) for d in datums]
            advs = [d.loss_fn_inputs["advantages"].to_torch().to(self.device) for d in datums]
            masks = [d.loss_fn_inputs["mask"].to_torch().float().to(self.device) for d in datums]
            local_denominator = (
                sum(mask.sum() for mask in masks)
                if masks
                else torch.zeros((), device=self.device)
            )
        denominator = (
            local_denominator
            if global_loss_denominator is None
            else self._coerce_global_loss_denominator(global_loss_denominator, like=local_denominator)
        )
        denominator = denominator.clamp(min=1e-8)

        # Backpropagate each independent graph immediately. Every term uses the
        # original logical batch's one global denominator, so this is the same
        # objective as losses.{cross_entropy,ppo,importance_sampling}_loss, not
        # a mean of per-microbatch means.
        detached_logprobs: list[torch.Tensor | None] = [None] * len(datums)
        total_numerator: torch.Tensor | None = None
        token_counts = [len(d.model_input.to_ints()) for d in datums]
        for indices in self._forward_microbatches(token_counts):
            chunk_logprobs = self._target_logprobs_batch([datums[index] for index in indices])
            if loss_fn == "cross_entropy":
                chunk_numerator = sum((weights[index] * logprob).sum() for index, logprob in zip(indices, chunk_logprobs))
            else:
                chunk_terms = []
                for index, logprob in zip(indices, chunk_logprobs):
                    sampled_logprob = sampled[index]
                    advantage = advs[index]
                    mask = masks[index]
                    ratio = torch.exp(logprob - sampled_logprob)
                    if loss_fn == "ppo":
                        clipped = torch.clamp(
                            ratio,
                            1.0 - self.ppo_clip_epsilon,
                            1.0 + self.ppo_clip_epsilon,
                        )
                        surrogate = torch.minimum(ratio * advantage, clipped * advantage)
                    else:
                        surrogate = ratio * advantage
                    chunk_terms.append((mask * surrogate).sum())
                chunk_numerator = sum(chunk_terms)

            chunk_loss = -chunk_numerator / denominator
            if chunk_loss.requires_grad:
                chunk_loss.backward()
            for index, logprob in zip(indices, chunk_logprobs):
                detached_logprobs[index] = logprob.detach().cpu()
            detached_numerator = chunk_numerator.detach()
            total_numerator = detached_numerator if total_numerator is None else total_numerator + detached_numerator
            del chunk_logprobs, chunk_numerator, chunk_loss

        if total_numerator is None:
            total_numerator = torch.zeros((), device=self.device)
        loss = -total_numerator / denominator

        # Internal chunks are one public/logical forward_backward. The later
        # optimizer step must average this submission once, not once per model
        # forward.
        self._gradient_accumulations += 1
        assert all(logprobs is not None for logprobs in detached_logprobs)
        return _ResolvedPending(
            ForwardBackwardOutput(
                logprobs=[logprobs for logprobs in detached_logprobs if logprobs is not None],
                metrics={"loss": float(loss.detach())},
            )
        )

    async def submit_opct_forward_backward(
        self,
        datums: Sequence[Any],
        *,
        behavior_temperature: float,
        kl_coef: float,
        kl_discount_factor: float,
        loss_fn: str,
        global_loss_denominator: torch.Tensor | float | int | None = None,
    ) -> _ResolvedPending:
        """Run OPCT raw current-policy scoring and backward in one pass.

        ``behavior_temperature`` is retained as a compatibility/validation
        argument, but it is not used to reconstruct the behavior distribution.
        The authoritative processed behavior log-probabilities are the values
        returned by generation and stored in each datum's ``logprobs`` tensor.
        They are detached in the importance ratio. The reverse-KL signal uses
        a detached raw student score and worker-supplied frozen-teacher scores.
        """

        if loss_fn not in ("importance_sampling", "ppo"):
            raise ValueError(f"OPCT supports importance_sampling or ppo, got {loss_fn!r}")
        if not math.isfinite(behavior_temperature) or behavior_temperature <= 0:
            raise ValueError("behavior_temperature must be a finite positive number")
        if not math.isfinite(kl_coef) or kl_coef <= 0:
            raise ValueError("kl_coef must be a finite positive number")
        if not math.isfinite(kl_discount_factor) or not 0 <= kl_discount_factor <= 1:
            raise ValueError("kl_discount_factor must be in [0, 1]")

        self._sleep_vllm_for_training()
        datums = list(datums)
        if not datums:
            # Replicated phase sharing can legitimately assign an empty shard
            # when the logical batch has fewer datums than training ranks.  It
            # still has to count as one public accumulation so every rank
            # reaches the same optimizer-boundary gradient collective.  The
            # historical single-rank call remains an error because an empty
            # OPCT batch without a coordinator-supplied global denominator is
            # almost certainly a caller bug.
            if global_loss_denominator is None:
                raise ValueError("OPCT needs at least one training datum")
            local_denominator = torch.zeros((), dtype=torch.float32, device=self.device)
            denominator = self._coerce_global_loss_denominator(
                global_loss_denominator,
                like=local_denominator,
            ).clamp(min=1)
            self._gradient_accumulations += 1
            return _ResolvedPending(
                ForwardBackwardOutput(
                    logprobs=[],
                    metrics={
                        "loss": 0.0,
                        "teacher_kl": 0.0,
                        "student_entropy": 0.0,
                        "teacher_cross_entropy": 0.0,
                        "teacher_scored_tokens": float(denominator),
                    },
                )
            )
        model = self._require_model()
        model.train()

        masks = [datum.loss_fn_inputs["mask"].to_torch().bool().to(self.device) for datum in datums]
        behaviors = []
        teachers = []
        for index, (datum, mask) in enumerate(zip(datums, masks)):
            behavior_data = datum.loss_fn_inputs.get("logprobs")
            if behavior_data is None:
                raise ValueError(f"OPCT datum {index} is missing sampled behavior logprobs")
            behavior = behavior_data.to_torch().float().to(self.device)
            if behavior.shape != mask.shape:
                raise ValueError(
                    f"OPCT datum {index} behavior/mask shape mismatch: "
                    f"behavior={tuple(behavior.shape)}, mask={tuple(mask.shape)}"
                )
            if not torch.isfinite(behavior[mask]).all():
                raise ValueError(f"OPCT datum {index} contains non-finite sampled behavior logprobs")

            teacher_data = datum.loss_fn_inputs.get("opct_teacher_logprobs")
            if teacher_data is None:
                raise ValueError(f"OPCT datum {index} is missing opct_teacher_logprobs")
            teacher = teacher_data.to_torch().float().to(self.device)
            if teacher.ndim != 1 or teacher.numel() != int(mask.sum()):
                raise ValueError(
                    f"OPCT datum {index} has {int(mask.sum())} action tokens but "
                    f"{teacher.numel()} teacher logprobs"
                )
            if not torch.isfinite(teacher).all():
                raise ValueError(f"OPCT datum {index} contains non-finite teacher logprobs")
            behaviors.append(behavior)
            teachers.append(teacher)

        # Retain the established OPCT denominator path by default.  The
        # explicit global-normalizer branch is for a replicated shard only.
        if global_loss_denominator is None:
            denominator = sum(mask.sum() for mask in masks).clamp(min=1)
        else:
            local_denominator = self.logical_loss_denominator(datums, loss_fn)
            denominator = self._coerce_global_loss_denominator(
                global_loss_denominator,
                like=local_denominator,
            ).clamp(min=1)
        detached_logprobs: list[torch.Tensor | None] = [None] * len(datums)
        total_numerator: torch.Tensor | None = None
        reverse_kl_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        student_logprob_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        teacher_logprob_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        token_counts = [len(datum.model_input.to_ints()) for datum in datums]

        for indices in self._forward_microbatches(token_counts):
            raw_scores = self._target_logprobs_batch([datums[index] for index in indices])
            chunk_terms = []
            for index, raw in zip(indices, raw_scores):
                mask = masks[index]
                if raw.shape != mask.shape:
                    raise ValueError(
                        f"OPCT datum {index} raw-score/mask shape mismatch: "
                        f"raw={tuple(raw.shape)}, mask={tuple(mask.shape)}"
                    )
                raw_action = raw[mask]
                behavior_action = behaviors[index][mask]
                teacher = teachers[index]
                reverse_kl = raw_action.detach() - teacher
                action_signal = -kl_coef * reverse_kl
                if kl_discount_factor > 0:
                    action_signal = discounted_future_sum_vectorized(action_signal, kl_discount_factor)

                # Only the raw current-policy score is recomputed. The behavior
                # denominator is the exact processed score returned for the
                # sampled token (temperature, filters, and all), not a logits /
                # temperature approximation reconstructed on the coordinator.
                ratio = torch.exp(raw_action - behavior_action.detach())
                if loss_fn == "ppo":
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.ppo_clip_epsilon,
                        1.0 + self.ppo_clip_epsilon,
                    )
                    surrogate = torch.minimum(ratio * action_signal, clipped * action_signal)
                else:
                    surrogate = ratio * action_signal
                chunk_terms.append(surrogate.sum())
                detached_logprobs[index] = raw.detach().cpu()
                reverse_kl_sum += reverse_kl.sum()
                student_logprob_sum += raw_action.detach().sum()
                teacher_logprob_sum += teacher.sum()

            chunk_numerator = sum(chunk_terms)
            chunk_loss = -chunk_numerator / denominator
            if chunk_loss.requires_grad:
                chunk_loss.backward()
            detached_numerator = chunk_numerator.detach()
            total_numerator = (
                detached_numerator if total_numerator is None else total_numerator + detached_numerator
            )
            del raw_scores, chunk_numerator, chunk_loss

        if total_numerator is None:
            total_numerator = torch.zeros((), device=self.device)
        loss = -total_numerator / denominator
        scored_tokens = float(denominator)
        self._gradient_accumulations += 1
        assert all(logprobs is not None for logprobs in detached_logprobs)
        return _ResolvedPending(
            ForwardBackwardOutput(
                logprobs=[logprobs for logprobs in detached_logprobs if logprobs is not None],
                metrics={
                    "loss": float(loss.detach()),
                    "teacher_kl": float(reverse_kl_sum / denominator),
                    "student_entropy": float(-student_logprob_sum / denominator),
                    "teacher_cross_entropy": float(-teacher_logprob_sum / denominator),
                    "teacher_scored_tokens": scored_tokens,
                },
            )
        )

    def _consistency_loss(self, loss_fn: str) -> consistency_losses.ConsistencyLoss:
        if loss_fn not in self._consistency_loss_modules:
            self._consistency_loss_modules[loss_fn] = consistency_losses.create_consistency_loss(loss_fn, self.consistency_loss_options)
        return self._consistency_loss_modules[loss_fn]

    def _consistency_forward_backward(
        self,
        datums: Sequence[Any],
        loss_fn: str,
        *,
        preflight_observation: dict[str, Any] | None = None,
    ) -> _ResolvedPending:
        """Paired-pass gradient accumulation for the consistency loss_fns (ACT/AttCT/MLPCT).

        Per datum (batch of 1, mirroring the upstream AttCT pipeline): a
        differentiable pass on the biased prompt, a no-grad reference pass on
        the clean prompt under the disabled adapter (the frozen base), then the
        loss over the aligned window from the datum's loss_fn_inputs. Backward
        runs per datum (mean over the batch), so peak memory holds one graph.
        """
        self._sleep_vllm_for_training()
        model = self._require_model()
        self._require_base()  # clean pass needs the frozen base (LoRA adapter disabled)
        model.train()
        # The preflight must not leave a cached loss object or hook manager
        # behind. Its local loss has no parameters; the ordinary training path
        # continues to cache exactly as before.
        is_preflight = preflight_observation is not None
        loss_module = (
            consistency_losses.create_consistency_loss(loss_fn, self.consistency_loss_options)
            if is_preflight
            else self._consistency_loss(loss_fn)
        )
        needs_attentions = loss_fn == "attention_consistency"
        needs_hidden = loss_fn == "activation_consistency"

        # sdpa/flash kernels don't materialize attention weights (transformers ≥5
        # returns empty ``attentions`` instead of falling back) — switch to eager.
        if needs_attentions and model.config._attn_implementation != "eager":
            print(f"LocalBackend: switching attention from {model.config._attn_implementation!r} to 'eager' for {loss_fn}")
            model.set_attn_implementation("eager")
        base_model = self._model_for(use_base=True)
        if needs_attentions and base_model is not model and base_model.config._attn_implementation != "eager":
            base_model.set_attn_implementation("eager")

        hooks = None
        base_hooks = None
        if loss_module.needs_mlp_hooks:
            if is_preflight:
                hooks = MLPHookManager(model, variant=getattr(loss_module, "variant", "hidden")).install()
            else:
                if self._mlp_hooks is None:
                    self._mlp_hooks = MLPHookManager(model, variant=getattr(loss_module, "variant", "hidden"))
                hooks = self._mlp_hooks.install()
            if base_model is model:
                base_hooks = hooks
            else:
                if is_preflight:
                    base_hooks = MLPHookManager(base_model, variant=getattr(loss_module, "variant", "hidden")).install()
                else:
                    if self._base_mlp_hooks is None:
                        self._base_mlp_hooks = MLPHookManager(base_model, variant=getattr(loss_module, "variant", "hidden"))
                    base_hooks = self._base_mlp_hooks.install()

        def forward(tokens: list[int], use_base: bool):
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            ctx = self._base_ctx() if use_base else _nullcontext()
            active_model = self._model_for(use_base=use_base)
            components = _selected_token_components(active_model)
            forward_model = components.backbone if components is not None else active_model
            active_hooks = base_hooks if use_base else hooks
            with ctx:
                outputs = forward_model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    output_attentions=needs_attentions,
                    output_hidden_states=needs_hidden,
                    use_cache=False,
                    return_dict=True,
                )
            mlp_states = None
            if active_hooks is not None:
                mlp_states = active_hooks.get_states()
                active_hooks.clear()
            return outputs, mlp_states

        total_loss = 0.0
        try:
            for d in datums:
                idx = {k: int(d.loss_fn_inputs[k].to_torch()[0]) for k in ("start_index", "clean_start_index", "clean_len", "match_len")}
                adv_outputs, adv_mlp_states = forward(d.model_input.to_ints(), use_base=False)
                with torch.no_grad():
                    clean_outputs, clean_mlp_states = forward(d.loss_fn_inputs["clean_tokens"].to_torch().long().tolist(), use_base=True)
                if preflight_observation is not None:
                    if needs_attentions:
                        preflight_observation.setdefault("variant_attention_counts", []).append(
                            len(getattr(adv_outputs, "attentions", ()) or ())
                        )
                        preflight_observation.setdefault("reference_attention_counts", []).append(
                            len(getattr(clean_outputs, "attentions", ()) or ())
                        )
                    if needs_hidden:
                        preflight_observation.setdefault("variant_hidden_state_counts", []).append(
                            len(getattr(adv_outputs, "hidden_states", ()) or ())
                        )
                        preflight_observation.setdefault("reference_hidden_state_counts", []).append(
                            len(getattr(clean_outputs, "hidden_states", ()) or ())
                        )
                    if loss_module.needs_mlp_hooks:
                        preflight_observation.setdefault("variant_mlp_hook_counts", []).append(len(adv_mlp_states or ()))
                        preflight_observation.setdefault("reference_mlp_hook_counts", []).append(len(clean_mlp_states or ()))
                out = loss_module(
                    clean_outputs,
                    adv_outputs,
                    **idx,
                    clean_mlp_states=clean_mlp_states,
                    adv_mlp_states=adv_mlp_states,
                )
                if preflight_observation is not None:
                    preflight_observation["loss_layers_used"] = len(out.get("layer_losses", []))
                    if loss_fn == "activation_consistency":
                        preflight_observation.setdefault("act_matching_suffix_lengths", []).append(
                            int(out.get("match_len", 0))
                        )
                (out["loss"] / len(datums)).backward()  # accumulate; frees this datum's graph
                total_loss += float(out["loss"].detach())
        finally:
            if hooks is not None:
                hooks.remove()
            if base_hooks is not None and base_hooks is not hooks:
                base_hooks.remove()

        self._gradient_accumulations += 1
        return _ResolvedPending(ForwardBackwardOutput(logprobs=[], metrics={"loss": total_loss / max(len(datums), 1)}))

    async def submit_optim_step(self, *, learning_rate: float, adam: AdamConfig) -> _ResolvedPending:
        self._sleep_vllm_for_training()
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
        if self._gradient_accumulations > 1:
            scale = 1.0 / self._gradient_accumulations
            for parameter in params:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
        if self._gradient_accumulations and self._gradient_reducer is not None:
            # One reducer call per optimizer step, after all local public
            # accumulations but before clipping.  The built-in distributed
            # reducer performs SUM (not mean), because each shard has already
            # used the common logical-batch denominator.
            self._gradient_reducer(params)
        if adam.grad_clip_norm and adam.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, adam.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self._gradient_accumulations = 0
        return _ResolvedPending(None)

    async def incorporate_kl_penalty(self, datums: Sequence[Any], *, kl_coef: float, kl_discount_factor: float) -> dict[str, float]:
        """Same math as tinker_cookbook.rl.metrics.incorporate_kl_penalty, with base
        logprobs from a local forward pass under the disabled adapter."""
        import tinker

        self._sleep_vllm_for_training()
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
            datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages)
        return {"kl_policy_base": float(avg_diff)}

    def _score_completions(
        self,
        prompts: Sequence[Any],
        completion_tokens: Sequence[Sequence[int]],
        *,
        use_base: bool,
    ) -> list[list[float]]:
        """Score continuations under the selected raw policy on supplied prompts.

        Logits at prompt position ``R - 1 + t`` predict completion token ``t``.
        This is intentionally independent of the generation temperature used to
        obtain the tokens.
        """

        self._sleep_vllm_for_training()
        if len(prompts) != len(completion_tokens):
            raise ValueError(f"prompts and completion_tokens must have the same length, got {len(prompts)} and {len(completion_tokens)}")
        if use_base:
            self._require_base()
        else:
            self._require_model()
        if not prompts:
            return []

        prompt_tokens = [list(prompt.to_ints()) for prompt in prompts]
        continuations = [list(tokens) for tokens in completion_tokens]
        for index, (prompt, completion) in enumerate(zip(prompt_tokens, continuations)):
            if not prompt:
                raise ValueError(f"prompt {index} is empty")
            if not completion:
                raise ValueError(f"completion {index} is empty")

        # The final completion token is a target, never an input needed to score
        # this continuation. Position R - 1 + t predicts completion token t.
        model_inputs = [prompt + completion[:-1] for prompt, completion in zip(prompt_tokens, continuations)]
        prediction_positions = [range(len(prompt) - 1, len(model_input)) for prompt, model_input in zip(prompt_tokens, model_inputs)]
        targets = [torch.tensor(completion, dtype=torch.long) for completion in continuations]

        model = self._model_for(use_base=use_base)
        was_training = model.training
        model.eval()
        try:
            output: list[list[float] | None] = [None] * len(model_inputs)
            with torch.no_grad():
                for indices in self._forward_microbatches([len(model_input) for model_input in model_inputs]):
                    chunk = self._predictive_logprobs_batch(
                        [model_inputs[index] for index in indices],
                        [prediction_positions[index] for index in indices],
                        [targets[index] for index in indices],
                        use_base=use_base,
                    )
                    for index, values in zip(indices, chunk):
                        output[index] = values.detach().cpu().tolist()
        finally:
            model.train(was_training)

        assert all(values is not None for values in output)
        return [values for values in output if values is not None]

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
                    "full_finetune_modules": self.full_finetune_modules,
                    "keep_frozen_base": self.keep_frozen_base,
                    "trainable_parameter_names": self._trainable_parameter_names,
                    "kind": kind,
                    "loop_state": loop_state,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        uri = f"file://{ckpt_dir.resolve()}"
        return {"sampler_path": uri, "state_path": uri if state_saved else None}

    def _load_checkpoint(self, resume_from: str, with_optimizer: bool) -> None:
        ckpt_dir = _strip_file_scheme(resume_from)
        if ckpt_dir.is_symlink() or not ckpt_dir.is_dir():
            raise FileNotFoundError(f"checkpoint must be a regular directory: {ckpt_dir}")
        if self.use_lora and HAS_PEFT:
            state = peft.utils.load_peft_weights(str(ckpt_dir))
            peft.set_peft_model_state_dict(self.model, state)
        else:
            weights = ckpt_dir / "weights.pt"
            if weights.is_symlink() or not weights.is_file():
                raise FileNotFoundError(f"checkpoint weights are missing: {weights}")
            self._require_model().load_state_dict(torch.load(weights, map_location=self.device))
        opt_path = ckpt_dir / "optimizer.pt"
        if with_optimizer:
            if opt_path.is_symlink() or not opt_path.is_file():
                raise FileNotFoundError(
                    f"optimizer-state resume was requested but optimizer.pt is missing: {opt_path}"
                )
            # Optimizer is created lazily at the first optim_step; stage the state.
            self._pending_optimizer_state = torch.load(opt_path, map_location=self.device)
        print(f"LocalBackend: loaded checkpoint from {ckpt_dir} (optimizer: {with_optimizer})")
