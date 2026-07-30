"""Compatibility exports for the generic prompt-pair schema.

New code should import these primitives from :mod:`ctm.pairs`.  This module is
kept so existing dataset adapters and callers do not break during migration.
"""

from ctm.pairs import (
    PAIR_MESSAGE_FIELDS,
    REFERENCE_MESSAGES_FIELD,
    VARIANT_MESSAGES_FIELD,
    PairRowError,
    canonical_pair_row,
    canonical_pair_rows,
    make_pair_row,
)

__all__ = [
    "PAIR_MESSAGE_FIELDS",
    "REFERENCE_MESSAGES_FIELD",
    "VARIANT_MESSAGES_FIELD",
    "PairRowError",
    "canonical_pair_row",
    "canonical_pair_rows",
    "make_pair_row",
]
