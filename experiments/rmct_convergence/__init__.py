"""Fail-closed checkpoint-window convergence accounting for RMCT.

This package is intentionally separate from the immutable RMCT-256 plateau
controller.  It controls a simple sequence of sealed 16-optimizer-step
segments and can therefore be used by a fresh RMCT experiment without
changing the older experiment's accounting contract.
"""

from .controller import (
    CHECKPOINT_RECEIPT_SCHEMA,
    COMPLETION_RECEIPT_SCHEMA,
    DECISION_RECEIPT_SCHEMA,
    SOURCE_METRICS_SCHEMA,
    CheckpointWindowConfig,
    ConvergenceError,
    PublishedDecision,
    WindowThresholds,
    extract_source_metrics,
    guard_from_paths,
    require_continue,
    successor_action,
    verify_decision_receipt,
)

__all__ = [
    "CHECKPOINT_RECEIPT_SCHEMA",
    "COMPLETION_RECEIPT_SCHEMA",
    "DECISION_RECEIPT_SCHEMA",
    "SOURCE_METRICS_SCHEMA",
    "CheckpointWindowConfig",
    "ConvergenceError",
    "PublishedDecision",
    "WindowThresholds",
    "extract_source_metrics",
    "guard_from_paths",
    "require_continue",
    "successor_action",
    "verify_decision_receipt",
]
