"""Fail-closed convergence accounting for the segmented RMCT-256 run."""

from .plateau import (
    DEFAULT_HARD_CAP_PASSES,
    DEFAULT_MIN_DELTA,
    DEFAULT_PATIENCE,
    PlateauConfig,
    SegmentPosition,
    guard_from_directory,
    publish_metrics_from_source_receipt,
    verify_decision_receipt,
)

__all__ = [
    "DEFAULT_HARD_CAP_PASSES",
    "DEFAULT_MIN_DELTA",
    "DEFAULT_PATIENCE",
    "PlateauConfig",
    "SegmentPosition",
    "guard_from_directory",
    "publish_metrics_from_source_receipt",
    "verify_decision_receipt",
]
