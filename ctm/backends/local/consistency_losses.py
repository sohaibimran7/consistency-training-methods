"""Internal-consistency losses: ACT, AttCT, and MLPCT (torch/HF-native, LocalBackend only).

Ported from https://github.com/c-wei/AttCT ``losses/losses.py`` @ 79527cf
(2026-07-10). Only the paper methods are ported — JSD attention consistency
(AttCT), residual-stream activation consistency (ACT, Irpan et al. 2025 Eq. 1),
and SwiGLU post-activation MLP consistency (MLPCT). Upstream's six ablated
attention variants (its README: they "diverge or grow exponentially"), its
SFTLoss (BCT here is the ``cross_entropy`` loss_fn), and its legacy knobs
(``loss_formulation="mse"``, ``layer_selection="all_with_embedding"``) stay behind.

All losses compare a differentiable forward pass on the biased/wrapped prompt
against a no-grad reference pass on the clean prompt (the frozen base model —
LoRA adapter disabled), matching representations on the shared content region:

- ``start_index`` / ``clean_start_index`` / ``clean_len``: token boundary of the
  clean content inside the wrapped / clean sequence (AttCT + MLPCT window).
- ``match_len``: longest matching token suffix of the two sequences (ACT window).

Each ``forward`` returns a dict with a differentiable ``"loss"`` plus float
diagnostics (``layer_losses``, ``mean_layer_loss``, ...).
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module, ABC):
    """Base class for paired clean/biased consistency losses.

    Args:
        weight: Global scalar multiplier applied to the final loss.
    """

    needs_mlp_hooks: bool = False

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    @abstractmethod
    def forward(
        self,
        clean_outputs,
        adv_outputs,
        *,
        start_index: int,
        clean_start_index: int,
        clean_len: int,
        match_len: Optional[int] = None,
        **kwargs,
    ) -> dict: ...


def _jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Jensen-Shannon Divergence between two batched probability distributions.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),  M = 0.5 * (P + Q)

    Bounded in [0, log 2], symmetric, always finite.
    """
    m = 0.5 * (p + q)
    log_m = torch.clamp(m, min=eps).log()
    log_p = torch.clamp(p, min=eps).log()
    log_q = torch.clamp(q, min=eps).log()

    kl_p_m = (p * (log_p - log_m)).sum(dim=-1)
    kl_q_m = (q * (log_q - log_m)).sum(dim=-1)

    return (0.5 * (kl_p_m + kl_q_m)).mean()


def _get_layer_weight(layer_weights_type: str, layer_idx: int, total_layers: int) -> float:
    if layer_weights_type == "uniform":
        return 1.0
    elif layer_weights_type == "linear_decay":
        return (layer_idx + 1) / total_layers
    elif layer_weights_type == "exponential_decay":
        return 2 ** (layer_idx / total_layers) - 1
    return 1.0


def _resolve_layer_indices(layer_selection, num_layers: int, first: int = 0) -> list[int]:
    """Turn a layer_selection spec into concrete indices over [first, num_layers)."""
    if layer_selection == "all":
        return list(range(first, num_layers))
    elif layer_selection == "last":
        return [num_layers - 1]
    elif layer_selection == "middle":
        return [num_layers // 2]
    elif layer_selection == "last_half":
        return list(range(num_layers // 2, num_layers))
    elif layer_selection == "last_quarter":
        return list(range((3 * num_layers) // 4, num_layers))
    elif isinstance(layer_selection, (list, tuple)):
        return [int(i) for i in layer_selection]
    raise ValueError(
        f"Unknown layer_selection: {layer_selection!r}. Choose 'all', 'last', 'middle', "
        "'last_half', 'last_quarter', or a list of indices."
    )


class JSDAttentionConsistencyLoss(ConsistencyLoss):
    """AttCT: Jensen-Shannon Divergence on per-head attention weights.

    JSD preferred over KL because it is symmetric (no arbitrary direction choice),
    bounded in [0, log 2] (no gradient spikes early in training), and always finite
    even when one distribution has zero mass where the other doesn't (common with
    causal masking).

    Args:
        weight:          Global scalar multiplier.
        layer_weights:   "uniform", "linear_decay", or "exponential_decay".
        layer_selection: "all", "last_half", "last_quarter", or a list of layer indices.
    """

    def __init__(self, weight: float = 1.0, layer_weights: str = "uniform", layer_selection="all", **kwargs):
        super().__init__(weight)
        self.layer_weights_type = layer_weights
        self.layer_selection = layer_selection

    def forward(
        self,
        clean_outputs,
        adv_outputs,
        *,
        start_index: int,
        clean_start_index: int,
        clean_len: int,
        match_len: Optional[int] = None,
        **kwargs,
    ) -> dict:
        if not getattr(clean_outputs, "attentions", None) or not getattr(adv_outputs, "attentions", None):
            raise ValueError(
                "Model outputs must include attentions (output_attentions=True, eager attention implementation)."
            )

        total_loss = torch.tensor(0.0, device=clean_outputs.attentions[0].device)
        layer_losses = []
        num_layers = len(clean_outputs.attentions)
        end_index = start_index + clean_len
        clean_end_index = clean_start_index + clean_len
        layer_indices = _resolve_layer_indices(self.layer_selection, num_layers)

        for layer_idx, (clean_att, adv_att) in enumerate(zip(clean_outputs.attentions, adv_outputs.attentions)):
            if layer_idx not in layer_indices:
                continue
            # Full-matrix slice — both query and key dims restricted to the content region.
            sliced_adv = adv_att[:, :, start_index:end_index, start_index:end_index]
            sliced_clean = clean_att[:, :, clean_start_index:clean_end_index, clean_start_index:clean_end_index]

            if sliced_clean.shape != sliced_adv.shape:
                import warnings

                warnings.warn(
                    f"JSDAttentionConsistencyLoss: skipping layer {layer_idx} — "
                    f"shape mismatch between clean {sliced_clean.shape} and adv {sliced_adv.shape}."
                )
                continue

            layer_loss = _jsd(sliced_clean, sliced_adv)
            layer_weight = _get_layer_weight(self.layer_weights_type, layer_idx, num_layers)
            total_loss = total_loss + layer_weight * layer_loss
            layer_losses.append(layer_loss.item())

        if not layer_losses:
            import warnings

            warnings.warn(
                "JSDAttentionConsistencyLoss: all layers skipped due to shape mismatches. "
                "Returning zero loss for this batch."
            )
            # Connect to the computation graph so loss.backward() succeeds.
            # Using adv attention (goes through LoRA layers) ensures grad_fn is present.
            zero_loss = adv_outputs.attentions[-1].sum() * 0.0
            return {"loss": zero_loss, "layer_losses": [], "mean_layer_loss": 0.0}

        avg_loss = total_loss / len(layer_losses)
        return {
            "loss": self.weight * avg_loss,
            "layer_losses": layer_losses,
            "mean_layer_loss": sum(layer_losses) / len(layer_losses),
        }


class ActivationConsistencyLoss(ConsistencyLoss):
    """ACT: MSE on the residual stream, from Irpan et al. 2025 (Eq. 1).

        ℓ = E_{t,l} [ || h_θ,t,l(p_wrapped) − sg(h_θ_init,t,l(p_clean)) ||² ]

    ||·||² is summed over hidden_dim, then averaged over token positions and layers
    (skipping the input-embedding layer, which is a pure function of token IDs and
    trivially matches at content positions).

    Matching positions: prefers ``match_len`` (the longest matching token suffix —
    the window the paper trains); falls back to the content-body window indexed by
    ``start_index`` / ``clean_start_index`` / ``clean_len`` when absent.

    Args:
        weight:          Global scalar multiplier.
        layer_selection: "all" (transformer layers only), "last", "middle", or a
                         list of hidden_states indices.
        normalize:       If True, L2-normalize activations before comparison.
    """

    def __init__(self, weight: float = 1.0, layer_selection="all", normalize: bool = False, **kwargs):
        super().__init__(weight)
        self.layer_selection = layer_selection
        self.normalize = normalize

    def forward(
        self,
        clean_outputs,
        adv_outputs,
        *,
        start_index: int = 0,
        clean_start_index: int = 0,
        clean_len: int = 0,
        match_len: Optional[int] = None,
        **kwargs,
    ) -> dict:
        if not getattr(clean_outputs, "hidden_states", None) or not getattr(adv_outputs, "hidden_states", None):
            raise ValueError("Model outputs must include hidden_states (output_hidden_states=True).")

        # hidden_states is len(transformer_layers) + 1 — index 0 is the input
        # embedding, indices 1..L are the residual stream after each block.
        num_hs = len(clean_outputs.hidden_states)
        layer_indices = _resolve_layer_indices(self.layer_selection, num_hs, first=1)

        # Pick the matching window: longest matching suffix (paper-correct) when
        # provided, else the content-body window.
        if match_len is not None and match_len > 0:
            clean_start = clean_outputs.hidden_states[0].shape[1] - match_len
            adv_start = adv_outputs.hidden_states[0].shape[1] - match_len
            window_len = match_len
        else:
            clean_start = clean_start_index
            adv_start = start_index
            window_len = clean_len

        if window_len <= 0:
            # No positions to compare — return a zero loss connected to the graph.
            zero_loss = adv_outputs.hidden_states[-1].sum() * 0.0
            return {"loss": zero_loss, "layer_losses": [], "mean_layer_loss": 0.0, "num_layers_used": 0, "match_len": 0}

        total_loss = torch.tensor(0.0, device=clean_outputs.hidden_states[0].device)
        layer_losses = []

        for layer_idx in layer_indices:
            aligned_clean = clean_outputs.hidden_states[layer_idx][
                :, clean_start : clean_start + window_len, :
            ].detach()
            aligned_adv = adv_outputs.hidden_states[layer_idx][:, adv_start : adv_start + window_len, :]

            if self.normalize:
                aligned_clean = F.normalize(aligned_clean, p=2, dim=-1)
                aligned_adv = F.normalize(aligned_adv, p=2, dim=-1)

            # E_{t,l}[||·||²]: sum over hidden_dim, then mean over (batch, tokens).
            layer_loss = ((aligned_adv - aligned_clean) ** 2).sum(dim=-1).mean()
            total_loss = total_loss + layer_loss
            layer_losses.append(layer_loss.item())

        avg_loss = total_loss / len(layer_indices)
        return {
            "loss": self.weight * avg_loss,
            "layer_losses": layer_losses,
            "mean_layer_loss": sum(layer_losses) / len(layer_losses),
            "num_layers_used": len(layer_indices),
            "match_len": int(window_len),
        }


class MLPConsistencyLoss(ConsistencyLoss):
    """MLPCT: cosine distance on SwiGLU post-activation MLP hidden states.

    Matches the input to each layer's down-projection (σ(W_gate·x) ⊙ W_up·x)
    between the clean and biased passes. States are captured by
    ``ctm.backends.local.mlp_hooks.MLPHookManager`` — the backend detects
    ``needs_mlp_hooks`` and installs hooks around both forward passes.

    Args:
        weight:          Global scalar multiplier.
        variant:         "hidden" (input to down_proj) or "output" (down_proj output).
        layer_selection: "all", "last", "middle", "last_half", "last_quarter", or indices.
        layer_weights:   "uniform", "linear_decay", or "exponential_decay".
        distance_metric: "cosine", "mse", or "smooth_l1".
        normalize:       If True, L2-normalize states before comparison.
    """

    needs_mlp_hooks: bool = True

    def __init__(
        self,
        weight: float = 1.0,
        variant: str = "hidden",
        layer_selection="all",
        layer_weights: str = "uniform",
        distance_metric: str = "cosine",
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__(weight)
        if variant not in ("hidden", "output"):
            raise ValueError(f"variant must be 'hidden' or 'output', got '{variant}'")
        self.variant = variant
        self.layer_selection = layer_selection
        self.layer_weights_type = layer_weights
        self.distance_metric = distance_metric
        self.normalize = normalize

    def forward(
        self,
        clean_outputs,
        adv_outputs,
        *,
        start_index: int,
        clean_start_index: int,
        clean_len: int,
        match_len: Optional[int] = None,
        clean_mlp_states: Optional[list[torch.Tensor]] = None,
        adv_mlp_states: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ) -> dict:
        if clean_mlp_states is None or adv_mlp_states is None:
            raise ValueError("MLPConsistencyLoss requires clean_mlp_states and adv_mlp_states.")

        num_layers = len(clean_mlp_states)
        layer_indices = _resolve_layer_indices(self.layer_selection, num_layers)
        total_loss = torch.tensor(0.0, device=clean_mlp_states[0].device)
        layer_losses = []

        for layer_idx in layer_indices:
            clean_neurons = clean_mlp_states[layer_idx]
            adv_neurons = adv_mlp_states[layer_idx]
            avail_clean = clean_neurons.shape[1] - clean_start_index
            avail_adv = adv_neurons.shape[1] - start_index
            actual_len = min(clean_len, avail_clean, avail_adv)
            if actual_len <= 0:
                continue
            aligned_clean = clean_neurons[:, clean_start_index : clean_start_index + actual_len, :].detach()
            aligned_adv = adv_neurons[:, start_index : start_index + actual_len, :]
            if self.normalize:
                aligned_clean = F.normalize(aligned_clean, p=2, dim=-1)
                aligned_adv = F.normalize(aligned_adv, p=2, dim=-1)
            layer_loss = self._compute_distance(aligned_adv, aligned_clean)
            layer_weight = _get_layer_weight(self.layer_weights_type, layer_idx, num_layers)
            total_loss = total_loss + layer_weight * layer_loss
            layer_losses.append(layer_loss.item())

        avg_loss = total_loss / max(len(layer_indices), 1)
        return {
            "loss": self.weight * avg_loss,
            "layer_losses": layer_losses,
            "mean_layer_loss": sum(layer_losses) / max(len(layer_losses), 1),
            "num_layers_used": len(layer_indices),
        }

    def _compute_distance(self, adv: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == "cosine":
            return (1 - F.cosine_similarity(adv, clean, dim=-1)).mean()
        elif self.distance_metric == "mse":
            return F.mse_loss(adv, clean)
        elif self.distance_metric == "smooth_l1":
            return F.smooth_l1_loss(adv, clean)
        raise ValueError(f"Unknown distance_metric: '{self.distance_metric}'")
