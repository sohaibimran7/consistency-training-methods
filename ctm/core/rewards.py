"""Consistency reward — pure math, no backend coupling.

Canonical home of ``ConsistencyReward`` (previously defined inside
``cot_transparency.apis.tinker.rl_training``, which now re-exports from here).
"""

from typing import Optional

from ctm.core.types import Rollout


class ConsistencyReward:
    """Consistency reward for training perturbation rollouts.

    Pushes p_hat toward p_ref (privileged) using variance-optimal baseline p_hat.
    r = -(p_hat - p_ref) * (trait - p_hat)

    Anchor reward is computed separately on reference perturbation rollouts.
    """

    def compute_rewards(
        self,
        rollouts: list[Rollout],
        p_hat: dict[int, float],
        p_ref: float,
        gaps: Optional[dict[int, float]] = None,
        baseline: Optional[float] = None,
    ) -> list[float]:
        """Consistency-only rewards for training perturbation rollouts.

        r = -gap[pert] * (trait - baseline)

        ``gaps`` lets the caller substitute a transformed gap (e.g. SNR-scaled)
        per perturbation. When None, the raw gap (p_hat[pert] - p_ref) is used.
        ``baseline`` overrides the per-perturbation mean p_hat[pert] used to centre the
        score term; the matched-pair estimator passes p_ref (the neutral control) so every
        cued rollout is judged against the reference, not its own per-cue mean. Any constant
        baseline leaves E[grad] = -gap * ∇p_hat unchanged (it only affects variance).
        """
        if gaps is None:
            gaps = {pert: rate - p_ref for pert, rate in p_hat.items()}
        return [
            -gaps[r.perturbation_idx] * (r.trait_value - (p_hat[r.perturbation_idx] if baseline is None else baseline))
            for r in rollouts
        ]

    def compute_anchor_rewards(
        self, ref_rollouts: list[Rollout], p_ref: float, p_ref_initial: Optional[float], gap: Optional[float] = None
    ) -> list[float]:
        """Anchor rewards for reference perturbation rollouts.

        r = -gap * (trait - p_ref), gap = (p_ref - p_ref_initial) by default.

        ``gap`` lets the caller substitute a transformed gap. Returns zeros when no
        anchor target is available (p_ref_initial is None and no gap supplied).
        """
        if gap is None:
            if p_ref_initial is None:
                return [0.0] * len(ref_rollouts)
            gap = p_ref - p_ref_initial
        return [-gap * (r.trait_value - p_ref) for r in ref_rollouts]
