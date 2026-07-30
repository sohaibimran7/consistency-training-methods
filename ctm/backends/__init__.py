"""Training/sampling backends behind one protocol (see ctm.backends.base).

- ``ctm.backends.tinker`` — Tinker service (LoRA training + managed sampling).
- ``ctm.backends.local``  — in-process torch/PEFT engine for self-hosted GPUs
  (Isambard, Vast.ai, workstations). Heavy imports live in the submodule; import
  it explicitly.
"""

from ctm.backends.base import (
    ForwardBackwardOutput,
    PendingForwardBackward,
    PendingOptimStep,
    PolicyScorerHandle,
    SampledSequence,
    SamplerHandle,
    TrainingBackend,
)

__all__ = [
    "ForwardBackwardOutput",
    "PendingForwardBackward",
    "PendingOptimStep",
    "PolicyScorerHandle",
    "SampledSequence",
    "SamplerHandle",
    "TrainingBackend",
]
